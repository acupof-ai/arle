use std::path::PathBuf;
use std::sync::{
    Arc, Barrier, Mutex,
    atomic::{AtomicBool, AtomicU32, Ordering},
};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use infer::backend::cuda::bootstrap::{
    DeepseekParallelConfig, InferenceEngineOptions, ModelType, SchedulerRuntimeGuard,
    ServerRuntimeConfig, detect_model_type, spawn_scheduler_handle_from_path,
};
use infer::backend::cuda::tensor::{DeviceContext, with_device_ordinal_override};
use infer::hf_hub;
use infer::http_server::{HttpServerConfig, TrainControlTarget, build_app_with_config};
use infer::kv_tier::ClusterSharedBackendConfig;
use infer::logging;
use infer::model::{GenerationState, KVCacheDtype, KVFormat, ModelForward};
use infer::request_handle::{DistributedSchedulerGroup, NumaSchedulerRouter, NumaSchedulerWorker};
use infer::runtime_topology::{
    AffinityApplyResult, RuntimeTopology, WorkerPlacement, bind_process_to_placement,
    configured_cuda_worker_ordinals, sample_process_numa_maps,
};
use infer::scheduler::{
    DraftMode, SchedulePolicy, SchedulerAdmissionPolicy, SchedulerConfig, SchedulerHandle,
    SchedulerMixedPolicy,
};
use infer::server_engine::EnginePoolModelSpec;
use infer::trace_reporter::{TraceStartupConfig, configure_global_tracing};
use log::info;
use rand::{SeedableRng, rngs::StdRng};

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
const DEFAULT_SEQ_LEN: usize = 4096;
const VALID_KV_CACHE_MODES: &str = "'auto', 'bf16', 'fp8', 'int8', 'tq2', 'tq3', or 'tq4'";
const VALID_QUANT_FORMATS: &str = "'auto' or 'marlin_w4a8'";
const CONTIGUOUS_KV_TOKENS: usize = 512;

#[derive(Parser)]
#[command(name = "infer", about = "Qwen3/3.5 GPU inference server")]
struct Args {
    /// Model directory containing config, tokenizer, and safetensor shards
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,

    /// Port to listen on
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Enable CUDA Graph capture/replay on decode path (`--cuda-graph=false` to disable)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cuda_graph: bool,

    /// SGLang-compatible alias for `--cuda-graph=false`.
    #[arg(long)]
    disable_cuda_graph: bool,

    /// Enable request tracing and write trace JSON files to this directory
    #[arg(long)]
    trace_output_path: Option<PathBuf>,

    /// Request tracing level: `off`, `basic`, or `verbose`
    #[arg(long)]
    trace_level: Option<String>,

    /// Fraction of requests to trace, between 0.0 and 1.0
    #[arg(long)]
    trace_sample_rate: Option<f64>,

    /// Flush interval for trace exporters in milliseconds
    #[arg(long)]
    trace_report_interval_ms: Option<u64>,

    /// Promote slow requests above this latency threshold (ms) when the caller opts in
    #[arg(long)]
    trace_slow_request_ms: Option<u64>,

    /// OTLP traces endpoint, e.g. `http://127.0.0.1:4318`
    #[arg(long)]
    otlp_traces_endpoint: Option<String>,

    /// Service name to attach to exported traces
    #[arg(long)]
    trace_service_name: Option<String>,

    /// Additional OTLP headers as `k=v,k2=v2`
    #[arg(long)]
    trace_otlp_headers: Option<String>,

    /// OTLP export timeout in milliseconds
    #[arg(long)]
    trace_otlp_timeout_ms: Option<u64>,

    /// Number of concurrent request slots (each gets its own KV cache).
    /// If unset, auto-computed from available GPU memory.
    #[arg(long)]
    num_slots: Option<usize>,

    /// Maximum sequence length (tokens) per KV cache slot. If unset, auto-computed
    /// from available GPU memory to fit all slots without OOM.
    #[arg(long)]
    max_seq_len: Option<usize>,

    /// Maximum number of tokens in a single prefill chunk.
    /// If unset, auto-picked from total GPU HBM (SGLang-style tiering:
    /// <35 GiB → 2048, <60 → 4096, <90 → 8192, ≥90 → 16384).
    #[arg(long)]
    chunked_prefill_size: Option<usize>,

    /// Maximum total tokens to advance in one scheduler step.
    /// Decode rows consume one token each; prefill rows consume their admitted chunk.
    #[arg(long, default_value_t = 16384)]
    max_num_batched_tokens: usize,

    /// Maximum total prefill tokens to queue in one scheduler step.
    /// If unset, defaults to `chunked_prefill_size` so the prefill activation
    /// buffer stays sized for one chunk rather than the whole-step budget.
    #[arg(long)]
    max_prefill_tokens: Option<usize>,

    /// Maximum number of prefilling requests to advance in one scheduler step.
    /// If omitted, the scheduler only enforces the token budget.
    #[arg(long)]
    prefill_max_requests: Option<usize>,

    /// Request scheduling policy. ARLE CUDA currently implements SGLang's
    /// default `fcfs`; other policy names are rejected instead of accepted as
    /// no-ops.
    #[arg(long, default_value = "fcfs")]
    schedule_policy: String,

    /// Admission policy: `queue-bound` preserves legacy queue-cap behavior;
    /// `prefix-aware` reserves queue headroom for warm prefix-cache hits.
    #[arg(long, default_value = "queue-bound")]
    admission_policy: String,

    /// Cold-request headroom reserved for warm prefix-cache hits when
    /// `--admission-policy=prefix-aware`. Defaults to max_waiting / 4.
    #[arg(long)]
    cold_headroom: Option<usize>,

    /// Decode-active prefill policy: `split` keeps production prefill+decode
    /// launches separate; `mixed` opts into the experimental single mixed launch.
    #[arg(long, default_value = "split")]
    scheduler_mixed_policy: String,

    /// SGLang-compatible streaming interval in generated tokens.
    #[arg(long, default_value_t = 1)]
    stream_interval: usize,

    /// Enable Phase 2 speculative decode plumbing. Defaults off.
    #[arg(long, default_value_t = false)]
    spec_enabled: bool,

    /// Maximum draft tokens proposed per speculative decode step.
    #[arg(long, default_value_t = 5)]
    spec_draft_k: usize,

    /// Minimum rolling acceptance rate required to keep speculation active.
    #[arg(long, default_value_t = 0.6)]
    spec_acceptance_threshold: f32,

    /// Draft mode: "none", "self"/"self-spec", or "external:<path>".
    #[arg(long, visible_alias = "spec-draft-mode", default_value = "none")]
    spec_draft_model: String,

    /// Enable MagicDec-style sparse-KV self-spec draft views.
    #[arg(long, default_value_t = false)]
    spec_sparse_kv_enabled: bool,

    /// Recent-token window included in each sparse-KV draft view.
    #[arg(long, default_value_t = 512)]
    spec_sparse_recent_tokens: usize,

    /// LRU-hot page budget included in each sparse-KV draft view.
    #[arg(long, default_value_t = 32)]
    spec_sparse_top_k_pages: usize,

    /// Disable RadixAttention-style prefix cache lookup and publish.
    #[arg(long)]
    disable_radix_cache: bool,

    /// Disable short-prompt bypass for prefix prefetch and split scheduling.
    #[arg(long)]
    disable_short_prompt_bypass: bool,

    /// Prompt length at or below which ARLE skips staged prefix prefetch and
    /// avoids decode+prefill split launches.
    #[arg(long, default_value_t = 256)]
    short_prompt_bypass_tokens: usize,

    /// Fraction of total GPU memory for weights + KV cache (SGLang-compatible).
    /// The remaining (1 - fraction) is headroom for activations, CUDA graphs,
    /// TileLang/native CUDA workspaces, and OS. Default 0.85 matches SGLang's
    /// `mem_fraction_static` default in `server_args.py`. K3 follow-up
    /// 2026-04-29 — bumped from 0.88 → 0.85 so the workspace estimate at
    /// the new `max_prefill_tokens=16384` default fits headroom without
    /// the OOM warn firing. Increase to 0.92 on dedicated inference
    /// boxes; decrease to 0.80 if sharing GPU.
    #[arg(long, default_value_t = 0.85)]
    mem_fraction_static: f64,

    /// Minimum sequence length per slot when auto-sizing KV cache.
    #[arg(long, default_value_t = 256)]
    min_seq_len: usize,

    /// Fallback KV pool budget (MB) when GPU memory query fails.
    #[arg(long, default_value_t = 4096)]
    kv_pool_fallback_mb: usize,

    /// KV cache mode: "auto" (default), "bf16", "fp8", "int8", or TurboQuant pool
    /// modes "tq2"/"tq3"/"tq4". FP8 and TurboQuant keep the contiguous prefill
    /// cache in BF16 and quantize when migrating into the paged token pool.
    ///
    /// `auto` defaults to FP8 paged pool (BF16 contiguous), halving per-token
    /// KV bytes vs full BF16 with negligible quality impact on Qwen3-family
    /// models. Falls back to BF16 paged pool if the FP8 dispatch is
    /// unavailable for the model arch.
    #[arg(long, default_value = "auto")]
    kv_cache_dtype: String,

    /// Weight quantization override: "auto" (checkpoint metadata) or "marlin_w4a8".
    #[arg(long, default_value = "auto")]
    quant_format: String,

    /// Optional upstream train control-plane URL to expose under `/v1/train/*`.
    #[arg(long)]
    train_control_url: Option<String>,

    /// Additional engine-pool model metadata to expose from `/v1/models`.
    #[arg(long = "pool-model", value_name = "SPEC")]
    pool_models: Vec<String>,

    /// Host-pinned T1 high-water mark as a fraction of host-pool capacity.
    #[arg(long)]
    t1_host_pinned_high_water: Option<f64>,

    /// Host-pinned T1 low-water mark as a fraction of host-pool capacity.
    #[arg(long)]
    t1_host_pinned_low_water: Option<f64>,

    /// Anti-thrash keepalive in radix logical ticks for freshly demoted T1 blocks.
    #[arg(long)]
    t1_host_pinned_keepalive_ticks: Option<u64>,

    /// Explicit host-pinned T1 pool capacity in MiB.
    #[arg(long)]
    t1_host_pinned_capacity_mb: Option<usize>,

    /// Minimum prompt length before session prefixes are eligible for T1 swap.
    #[arg(long)]
    t1_host_pinned_min_prompt_tokens: Option<usize>,

    /// Root directory for the node-local T2 disk store.
    #[arg(long)]
    disk_store_root: Option<PathBuf>,

    /// Root directory for the cluster-shared T3 shared-fs backend.
    #[arg(long)]
    cluster_shared_root: Option<PathBuf>,

    /// Run the F0 two-rank NCCL all-reduce smoke and exit.
    #[arg(long)]
    nccl_smoke: bool,

    /// Run a direct DeepSeek V4 multi-rank generation smoke and exit.
    ///
    /// This bypasses the HTTP scheduler and drives ARLE's CUDA model forward
    /// path directly across one rank thread per configured CUDA ordinal.
    #[arg(long)]
    deepseek_distributed_generate: bool,

    /// Prompt used by `--deepseek-distributed-generate`.
    #[arg(long, default_value = "Hello")]
    deepseek_distributed_prompt: String,

    /// New tokens to generate with `--deepseek-distributed-generate`.
    #[arg(long, default_value_t = 1)]
    deepseek_distributed_max_new_tokens: usize,

    /// Number of DeepSeek transformer layers to run on the GPU direct path.
    /// Leave unset to run one layer for bring-up; set to the model layer count
    /// for a full-model correctness run.
    #[arg(long)]
    deepseek_distributed_layers: Option<usize>,
}

fn main() {
    let args = Args::parse();
    apply_quant_format_override(&args);
    logging::init_default();
    if args.deepseek_distributed_generate {
        run_deepseek_distributed_generate(&args)
            .unwrap_or_else(|err| panic!("DeepSeek distributed generate failed: {err:#}"));
        return;
    }
    configure_deepseek_serving_env_if_needed(&args)
        .unwrap_or_else(|err| panic!("DeepSeek serving env setup failed: {err:#}"));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    runtime.block_on(async_main(args));
}

fn apply_quant_format_override(args: &Args) {
    match args.quant_format.trim().to_ascii_lowercase().as_str() {
        "auto" => {}
        "marlin_w4a8" | "w4a8_marlin" => {
            // SAFETY: called from synchronous main before constructing the
            // Tokio runtime or installing tracing/background workers.
            unsafe {
                std::env::set_var("INFER_QUANT_FORMAT_OVERRIDE", "marlin_w4a8");
            }
        }
        other => panic!("Invalid --quant-format '{other}': expected {VALID_QUANT_FORMATS}"),
    }
}

unsafe fn set_env_default(key: &str, value: impl AsRef<str>) {
    if std::env::var_os(key).is_none() {
        // SAFETY: callers invoke this only from startup paths before the
        // Tokio runtime or CUDA scheduler worker threads are spawned.
        unsafe {
            std::env::set_var(key, value.as_ref());
        }
    }
}

fn configure_deepseek_serving_env_if_needed(args: &Args) -> Result<()> {
    let model_path = args
        .model_path
        .to_str()
        .context("Model path must be valid UTF-8")?;
    let resolved_model_path = hf_hub::resolve_model_path(model_path)
        .with_context(|| format!("failed to resolve model path {model_path}"))?;
    let resolved_model_path = resolved_model_path
        .to_str()
        .context("Resolved model path must be valid UTF-8")?
        .to_string();
    let model_type = detect_model_type(&resolved_model_path)?;
    if !matches!(model_type, ModelType::DeepSeekV4) {
        return Ok(());
    }

    let runtime =
        infer::model::deepseek::DeepseekRuntimeConfig::from_model_dir(&resolved_model_path)?;
    let layers = args
        .deepseek_distributed_layers
        .unwrap_or(runtime.num_hidden_layers);
    if layers == 0 || layers > runtime.num_hidden_layers {
        bail!(
            "--deepseek-distributed-layers must be in 1..={}, got {}",
            runtime.num_hidden_layers,
            layers
        );
    }
    let world_size = configured_cuda_worker_ordinals()
        .map_err(|err| anyhow::anyhow!("invalid CUDA workers: {err}"))?
        .len();
    if world_size == 0 {
        bail!("at least one CUDA ordinal is required");
    }
    if world_size > 1 && runtime.o_groups != world_size {
        bail!(
            "DeepSeek V4 HTTP distributed path maps TP ranks to O-LoRA groups; configured ranks={} but o_groups={}",
            world_size,
            runtime.o_groups
        );
    }
    if world_size > 1 && !runtime.n_routed_experts.is_multiple_of(world_size) {
        bail!(
            "DeepSeek V4 n_routed_experts={} must be divisible by ranks={}",
            runtime.n_routed_experts,
            world_size
        );
    }

    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").context("reserve NCCL rendezvous TCP port")?;
    let port = listener
        .local_addr()
        .context("read NCCL rendezvous TCP port")?
        .port();
    drop(listener);

    // SAFETY: this runs before the Tokio runtime and before CUDA scheduler
    // worker threads are spawned. NCCL EnvBootstrap reads MASTER_* during
    // model construction on each rank.
    unsafe {
        if world_size > 1 {
            std::env::set_var("MASTER_ADDR", "127.0.0.1");
            std::env::set_var("MASTER_PORT", port.to_string());
            std::env::set_var("WORLD_SIZE", world_size.to_string());
        }
        set_env_default("ARLE_DSV4_LOAD_LAYER_WEIGHTS", "1");
        set_env_default("ARLE_DSV4_GPU_FULL_LAYERS", layers.to_string());
        set_env_default("ARLE_DSV4_GPU_CONTEXT_TOKENS", "1");
        set_env_default("ARLE_DSV4_INFER_REAL_REFERENCE", "0");
    }
    info!(
        "DeepSeek V4 serving distributed env: ranks={} gpu_layers={} model={}",
        world_size, layers, resolved_model_path
    );
    Ok(())
}

fn run_deepseek_distributed_generate(args: &Args) -> Result<()> {
    let tracing = configure_global_tracing(TraceStartupConfig {
        level: args.trace_level.clone(),
        sample_rate: args.trace_sample_rate,
        report_interval_ms: args.trace_report_interval_ms,
        slow_request_ms: args.trace_slow_request_ms,
        file_output: args.trace_output_path.clone(),
        otlp_endpoint: args.otlp_traces_endpoint.clone(),
        otlp_headers: args.trace_otlp_headers.clone(),
        otlp_timeout_ms: args.trace_otlp_timeout_ms,
        service_name: args.trace_service_name.clone(),
    })
    .context("invalid tracing config")?;
    if tracing.reporter_installed() {
        info!("Tracing configured: {}", tracing.config().summary());
    }

    #[cfg(not(all(feature = "cuda", feature = "nccl")))]
    {
        let _ = args;
        bail!("--deepseek-distributed-generate requires building infer with --features cuda,nccl");
    }

    #[cfg(all(feature = "cuda", feature = "nccl"))]
    {
        use infer::distributed::expert_state::ExpertGroup;
        use infer::model::deepseek::{DeepseekModel, DeepseekRuntimeConfig};
        use infer::sampler::SamplingParams;
        use infer::tensor_parallel::TpConfig;
        use infer::tokenizer::Tokenizer;
        use serde::Serialize;

        #[derive(Serialize)]
        struct DirectRankPerf {
            event: &'static str,
            rank: usize,
            world_size: usize,
            tp_rank: usize,
            ep_rank: usize,
            layers: usize,
            prompt_tokens: usize,
            completion_tokens: usize,
            load_ms: f64,
            prefill_ms: f64,
            decode_ms: f64,
            total_ms: f64,
        }

        let model_path = args
            .model_path
            .to_str()
            .context("Model path must be valid UTF-8")?;
        let resolved_model_path = hf_hub::resolve_model_path(model_path)
            .with_context(|| format!("failed to resolve model path {model_path}"))?;
        let resolved_model_path = resolved_model_path
            .to_str()
            .context("Resolved model path must be valid UTF-8")?
            .to_string();
        let model_type = detect_model_type(&resolved_model_path)?;
        if !matches!(model_type, ModelType::DeepSeekV4) {
            bail!("--deepseek-distributed-generate only supports DeepSeek V4, got {model_type}");
        }

        let spec_runtime = DeepseekRuntimeConfig::from_model_dir(&resolved_model_path)?;
        let layers = args.deepseek_distributed_layers.unwrap_or(1);
        if layers == 0 || layers > spec_runtime.num_hidden_layers {
            bail!(
                "--deepseek-distributed-layers must be in 1..={}, got {}",
                spec_runtime.num_hidden_layers,
                layers
            );
        }

        let cuda_ordinals = configured_cuda_worker_ordinals()
            .map_err(|err| anyhow::anyhow!("invalid CUDA workers: {err}"))?;
        if cuda_ordinals.is_empty() {
            bail!("at least one CUDA ordinal is required");
        }
        let world_size = cuda_ordinals.len();
        if spec_runtime.o_groups != world_size {
            bail!(
                "DeepSeek V4 direct distributed path maps TP ranks to O-LoRA groups; configured ranks={} but o_groups={}",
                world_size,
                spec_runtime.o_groups
            );
        }
        if !spec_runtime.n_routed_experts.is_multiple_of(world_size) {
            bail!(
                "DeepSeek V4 n_routed_experts={} must be divisible by ranks={}",
                spec_runtime.n_routed_experts,
                world_size
            );
        }

        let tokenizer = Tokenizer::from_file(&resolved_model_path)?;
        let prompt_tokens = tokenizer.encode(&args.deepseek_distributed_prompt)?;
        if prompt_tokens.is_empty() {
            bail!("distributed generate prompt encoded to zero tokens");
        }
        let prompt_tokens = Arc::new(prompt_tokens);
        let max_seq_len = args
            .max_seq_len
            .unwrap_or_else(|| prompt_tokens.len() + args.deepseek_distributed_max_new_tokens + 1);
        let kv_mode = parse_kv_cache_mode(&args.kv_cache_dtype)
            .map_err(|err| anyhow::anyhow!("invalid KV cache mode: {err}"))?;
        let kv_cache_dtype = match kv_mode {
            RequestedKvCacheMode::Auto => KVCacheDtype::BF16,
            RequestedKvCacheMode::Explicit { kv_cache_dtype, .. } => kv_cache_dtype,
        };

        configure_deepseek_distributed_env(layers, world_size)?;

        let selected_token = Arc::new(AtomicU32::new(0));
        let should_stop = Arc::new(AtomicBool::new(false));
        let generated = Arc::new(Mutex::new(Vec::new()));
        let perf_summaries = Arc::new(Mutex::new(Vec::<DirectRankPerf>::new()));
        let prefill_barrier = Arc::new(Barrier::new(world_size));
        let token_barrier = Arc::new(Barrier::new(world_size));
        let decode_barrier = Arc::new(Barrier::new(world_size));
        let trace_root = fastrace::Span::root(
            "deepseek_distributed_generate",
            fastrace::collector::SpanContext::random(),
        )
        .with_properties(|| {
            [
                ("model", resolved_model_path.clone()),
                ("world_size", world_size.to_string()),
                ("prompt_tokens", prompt_tokens.len().to_string()),
                (
                    "max_new_tokens",
                    args.deepseek_distributed_max_new_tokens.to_string(),
                ),
                ("layers", layers.to_string()),
            ]
        });
        let trace_root_context = fastrace::collector::SpanContext::from_span(&trace_root)
            .unwrap_or_else(fastrace::collector::SpanContext::random);

        info!(
            "DeepSeek distributed direct generate: model={} ranks={} cuda_ordinals={:?} prompt_tokens={} max_new_tokens={} gpu_layers={}",
            resolved_model_path,
            world_size,
            cuda_ordinals,
            prompt_tokens.len(),
            args.deepseek_distributed_max_new_tokens,
            layers,
        );

        let mut handles = Vec::with_capacity(world_size);
        for (rank, cuda_ordinal) in cuda_ordinals.into_iter().enumerate() {
            let model_path = resolved_model_path.clone();
            let prompt_tokens = Arc::clone(&prompt_tokens);
            let selected_token = Arc::clone(&selected_token);
            let should_stop = Arc::clone(&should_stop);
            let generated = Arc::clone(&generated);
            let perf_summaries = Arc::clone(&perf_summaries);
            let prefill_barrier = Arc::clone(&prefill_barrier);
            let token_barrier = Arc::clone(&token_barrier);
            let decode_barrier = Arc::clone(&decode_barrier);
            let max_new_tokens = args.deepseek_distributed_max_new_tokens;
            let trace_parent = trace_root_context;
            handles.push(
                std::thread::Builder::new()
                    .name(format!("dsv4-direct-rank-{rank}"))
                    .spawn(move || -> Result<()> {
                        let total_start = Instant::now();
                        let rank_span = fastrace::Span::root("deepseek_direct_rank", trace_parent)
                            .with_properties(|| {
                                [
                                    ("rank", rank.to_string()),
                                    ("world_size", world_size.to_string()),
                                    ("tp_rank", rank.to_string()),
                                    ("ep_rank", rank.to_string()),
                                    ("layers", layers.to_string()),
                                ]
                            });
                        with_device_ordinal_override(cuda_ordinal as u32, || -> Result<()> {
                            let load_start = Instant::now();
                            let mut runtime = DeepseekRuntimeConfig::from_model_dir(&model_path)?;
                            runtime.enable_cuda_graph = false;
                            runtime.tp = TpConfig::new(world_size, rank)?;
                            runtime.ep =
                                ExpertGroup::new(rank, world_size, runtime.spec.n_routed_experts)?;
                            let model = {
                                let _load_span = fastrace::Span::enter_with_parent(
                                    "deepseek_direct_load",
                                    &rank_span,
                                )
                                .with_properties(|| {
                                    [
                                        ("rank", rank.to_string()),
                                        ("cuda_ordinal", cuda_ordinal.to_string()),
                                    ]
                                });
                                DeepseekModel::from_safetensors(&model_path, runtime).with_context(
                                    || format!("rank {rank}: load DeepSeek V4 weights"),
                                )?
                            };
                            let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
                            let mut state = model.create_state()?;
                            state.set_max_seq_len(max_seq_len);
                            state.set_kv_dtype(kv_cache_dtype);
                            let prefill_start = Instant::now();
                            {
                                let _prefill_span = fastrace::Span::enter_with_parent(
                                    "deepseek_direct_prefill",
                                    &rank_span,
                                )
                                .with_properties(|| {
                                    [
                                        ("rank", rank.to_string()),
                                        ("tokens", prompt_tokens.len().to_string()),
                                    ]
                                });
                                model
                                    .forward_prefill(&prompt_tokens, &mut state)
                                    .with_context(|| format!("rank {rank}: prefill"))?;
                            }
                            let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
                            prefill_barrier.wait();

                            let mut rng = StdRng::seed_from_u64(42);
                            let sampling = SamplingParams::default();
                            let decode_start = Instant::now();
                            let _decode_span = fastrace::Span::enter_with_parent(
                                "deepseek_direct_decode",
                                &rank_span,
                            )
                            .with_properties(|| {
                                [
                                    ("rank", rank.to_string()),
                                    ("max_new_tokens", max_new_tokens.to_string()),
                                ]
                            });
                            let mut completion_tokens = 0usize;
                            for step_idx in 0..max_new_tokens {
                                if rank == 0 {
                                    let token = model
                                        .select_token(&mut state, &sampling, &mut rng)
                                        .with_context(|| {
                                            format!("rank 0: select token step {step_idx}")
                                        })?;
                                    selected_token.store(token, Ordering::SeqCst);
                                    generated
                                        .lock()
                                        .expect("generated lock poisoned")
                                        .push(token);
                                    if model.is_stop_token(token) {
                                        should_stop.store(true, Ordering::SeqCst);
                                    }
                                }
                                token_barrier.wait();
                                completion_tokens = completion_tokens.saturating_add(1);

                                let token = selected_token.load(Ordering::SeqCst);
                                let stop = should_stop.load(Ordering::SeqCst);
                                if step_idx + 1 < max_new_tokens && !stop {
                                    model.forward_decode(token, &mut state).with_context(|| {
                                        format!("rank {rank}: decode step {step_idx} token {token}")
                                    })?;
                                }
                                decode_barrier.wait();
                                if stop {
                                    break;
                                }
                            }
                            let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
                            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
                            rank_span.add_properties(|| {
                                [
                                    ("load_ms", format!("{load_ms:.3}")),
                                    ("prefill_ms", format!("{prefill_ms:.3}")),
                                    ("decode_ms", format!("{decode_ms:.3}")),
                                    ("total_ms", format!("{total_ms:.3}")),
                                    ("completion_tokens", completion_tokens.to_string()),
                                ]
                            });
                            perf_summaries
                                .lock()
                                .expect("perf summary lock poisoned")
                                .push(DirectRankPerf {
                                    event: "deepseek_distributed_perf",
                                    rank,
                                    world_size,
                                    tp_rank: rank,
                                    ep_rank: rank,
                                    layers,
                                    prompt_tokens: prompt_tokens.len(),
                                    completion_tokens,
                                    load_ms,
                                    prefill_ms,
                                    decode_ms,
                                    total_ms,
                                });
                            Ok(())
                        })
                    })?,
            );
        }

        for (rank, handle) in handles.into_iter().enumerate() {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("DeepSeek direct rank {rank} thread panicked"))?
                .with_context(|| format!("DeepSeek direct rank {rank} failed"))?;
        }

        let generated = generated.lock().expect("generated lock poisoned").clone();
        let mut perf_summaries = perf_summaries
            .lock()
            .expect("perf summary lock poisoned")
            .drain(..)
            .collect::<Vec<_>>();
        perf_summaries.sort_by_key(|summary| summary.rank);
        for summary in perf_summaries {
            println!("{}", serde_json::to_string(&summary)?);
        }
        let generated_text = tokenizer
            .decode(&generated)
            .unwrap_or_else(|err| format!("<decode failed for {:?}: {err:#}>", generated));
        println!("deepseek_distributed_generated_token_ids={generated:?}");
        println!("deepseek_distributed_generated_text={generated_text:?}");
        drop(trace_root);
        if tracing.reporter_installed() {
            fastrace::flush();
        }
        Ok(())
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn configure_deepseek_distributed_env(layers: usize, world_size: usize) -> Result<()> {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").context("reserve NCCL rendezvous TCP port")?;
    let port = listener
        .local_addr()
        .context("read NCCL rendezvous TCP port")?
        .port();
    drop(listener);

    // SAFETY: direct distributed generate runs before the Tokio runtime starts
    // and before any background threads are spawned.
    unsafe {
        std::env::set_var("MASTER_ADDR", "127.0.0.1");
        std::env::set_var("MASTER_PORT", port.to_string());
        std::env::set_var("WORLD_SIZE", world_size.to_string());
        set_env_default("ARLE_DSV4_LOAD_LAYER_WEIGHTS", "1");
        set_env_default("ARLE_DSV4_GPU_FULL_LAYERS", layers.to_string());
        set_env_default("ARLE_DSV4_GPU_CONTEXT_TOKENS", "1");
        set_env_default("ARLE_DSV4_INFER_REAL_REFERENCE", "0");
    }
    Ok(())
}

#[derive(Clone)]
struct CudaWorkerBootstrap {
    cuda_ordinal: usize,
    placement: WorkerPlacement,
}

struct StartedCudaWorker {
    handle: SchedulerHandle,
    guard: SchedulerRuntimeGuard,
    placement: WorkerPlacement,
}

fn format_nics(nics: &[String]) -> String {
    if nics.is_empty() {
        "none".to_string()
    } else {
        nics.join(",")
    }
}

fn multi_worker_affinity_placeholder() -> AffinityApplyResult {
    AffinityApplyResult {
        label: "main-multi-worker".to_string(),
        applied: false,
        requested_cpus: Vec::new(),
        applied_threads: 0,
        failed_threads: 0,
        reason: "multi-worker bootstrap applies affinity per worker before CUDA context"
            .to_string(),
    }
}

fn build_cuda_worker_bootstrap(topology: &RuntimeTopology) -> Vec<CudaWorkerBootstrap> {
    let cuda_ordinals = configured_cuda_worker_ordinals()
        .unwrap_or_else(|err| panic!("invalid CUDA workers: {err}"));
    cuda_ordinals
        .into_iter()
        .enumerate()
        .map(|(worker_id, cuda_ordinal)| CudaWorkerBootstrap {
            cuda_ordinal,
            placement: topology.placement_for_cuda_device_ordinal(cuda_ordinal, worker_id),
        })
        .collect()
}

fn log_final_worker_topology(worker: &CudaWorkerBootstrap, affinity: &AffinityApplyResult) {
    info!(
        "Final runtime worker placement: worker={} cuda_ordinal={} gpu={} numa={:?} cpus={} nics={} affinity_applied={} reason={}",
        worker.placement.worker_id,
        worker.cuda_ordinal,
        worker.placement.gpu_ordinal,
        worker.placement.numa_node,
        worker.placement.cpus.len(),
        format_nics(&worker.placement.nics),
        affinity.applied,
        affinity.reason,
    );
}

fn early_pre_model_free_bytes(worker: &CudaWorkerBootstrap) -> Option<usize> {
    with_device_ordinal_override(worker.cuda_ordinal as u32, || match DeviceContext::new() {
        Ok(_ctx) => match DeviceContext::gpu_memory_info() {
            Ok((free, total)) => {
                info!(
                    "GPU memory @ post_cuda_ctx (early): cuda_ordinal={} gpu={} free={:.2} GB / total={:.2} GB \
                     (driver+ctx+cuBLAS overhead = {:.0} MB)",
                    worker.cuda_ordinal,
                    worker.placement.gpu_ordinal,
                    free as f64 / 1e9,
                    total as f64 / 1e9,
                    (total - free) as f64 / 1e6,
                );
                Some(free)
            }
            Err(err) => {
                log::warn!("post_cuda_ctx GPU memory query failed: {err}");
                None
            }
        },
        Err(err) => {
            log::warn!(
                "Early DeviceContext::new() failed on cuda_ordinal={}: {err} — pre_model_free snapshot disabled",
                worker.cuda_ordinal,
            );
            None
        }
    })
}

fn shutdown_started_workers(workers: Vec<StartedCudaWorker>) {
    for StartedCudaWorker { handle, guard, .. } in workers {
        drop(handle);
        guard.wait();
    }
}

fn spawn_cuda_worker_group(
    model_path: &str,
    args: &Args,
    model_type: ModelType,
    num_slots: usize,
    kv_cache_dtype: KVCacheDtype,
    kv_pool_format: KVFormat,
    workers: &[CudaWorkerBootstrap],
    single_worker_pre_model_free_bytes: Option<usize>,
    metrics: &infer::metrics::ServerMetrics,
) -> anyhow::Result<Vec<StartedCudaWorker>> {
    let world_size = workers.len();
    let mut planned = Vec::with_capacity(workers.len());
    for (rank, worker) in workers.iter().enumerate() {
        let runtime = ServerRuntimeConfig {
            engine: InferenceEngineOptions {
                enable_cuda_graph: args.cuda_graph && !args.disable_cuda_graph,
            },
            scheduler: scheduler_config_from_args(args, num_slots),
            runtime_envelope: infer::scheduler::RuntimeEnvelopeOverrides {
                chunked_prefill_size: args.chunked_prefill_size,
                max_prefill_tokens: args.max_prefill_tokens,
            },
            seed: 42,
            max_seq_len: args.max_seq_len,
            kv_cache_dtype,
            kv_pool_format,
            pre_model_free_bytes: (workers.len() == 1)
                .then_some(single_worker_pre_model_free_bytes)
                .flatten(),
            worker_placement: Some(worker.placement.clone()),
            cuda_device_ordinal: Some(worker.cuda_ordinal as u32),
            deepseek_parallel: (matches!(model_type, ModelType::DeepSeekV4) && world_size > 1)
                .then_some(DeepseekParallelConfig { rank, world_size }),
        };
        planned.push((rank, worker.clone(), runtime));
    }

    if world_size <= 1 {
        let mut started = Vec::with_capacity(planned.len());
        for (_, worker, runtime) in planned {
            match spawn_scheduler_handle_from_path(model_path, runtime, metrics.clone()) {
                Ok((handle, guard)) => {
                    started.push(StartedCudaWorker {
                        handle,
                        guard,
                        placement: worker.placement.clone(),
                    });
                }
                Err(err) => {
                    shutdown_started_workers(started);
                    return Err(err).with_context(|| {
                        format!(
                            "worker={} cuda_ordinal={} gpu={}",
                            worker.placement.worker_id,
                            worker.cuda_ordinal,
                            worker.placement.gpu_ordinal
                        )
                    });
                }
            }
        }
        return Ok(started);
    }

    let mut load_threads = Vec::with_capacity(planned.len());
    for (rank, worker, runtime) in planned {
        let model_path = model_path.to_string();
        let metrics = metrics.clone();
        load_threads.push(std::thread::spawn(
            move || -> (usize, anyhow::Result<StartedCudaWorker>) {
                let result = spawn_scheduler_handle_from_path(&model_path, runtime, metrics)
                    .map(|(handle, guard)| StartedCudaWorker {
                        handle,
                        guard,
                        placement: worker.placement.clone(),
                    })
                    .with_context(|| {
                        format!(
                            "worker={} cuda_ordinal={} gpu={}",
                            worker.placement.worker_id,
                            worker.cuda_ordinal,
                            worker.placement.gpu_ordinal
                        )
                    });
                (rank, result)
            },
        ));
    }

    let mut started_by_rank = (0..world_size).map(|_| None).collect::<Vec<_>>();
    let mut first_err = None;
    for thread in load_threads {
        let (rank, result) = thread
            .join()
            .map_err(|_| anyhow::anyhow!("CUDA worker loader thread panicked"))?;
        match result {
            Ok(worker) => started_by_rank[rank] = Some(worker),
            Err(err) if first_err.is_none() => first_err = Some(err),
            Err(_) => {}
        }
    }
    if let Some(err) = first_err {
        shutdown_started_workers(started_by_rank.into_iter().flatten().collect());
        return Err(err);
    }
    Ok(started_by_rank
        .into_iter()
        .map(|worker| worker.expect("all ranks loaded successfully"))
        .collect())
}

async fn async_main(args: Args) {
    if args.nccl_smoke {
        #[cfg(feature = "nccl")]
        {
            infer::distributed::smoke_2_thread_all_reduce()
                .unwrap_or_else(|err| panic!("NCCL smoke failed: {err:#}"));
            info!("NCCL smoke passed");
            return;
        }
        #[cfg(not(feature = "nccl"))]
        {
            panic!("--nccl-smoke requires building infer with --features nccl");
        }
    }

    let tracing = configure_global_tracing(TraceStartupConfig {
        level: args.trace_level.clone(),
        sample_rate: args.trace_sample_rate,
        report_interval_ms: args.trace_report_interval_ms,
        slow_request_ms: args.trace_slow_request_ms,
        file_output: args.trace_output_path.clone(),
        otlp_endpoint: args.otlp_traces_endpoint.clone(),
        otlp_headers: args.trace_otlp_headers.clone(),
        otlp_timeout_ms: args.trace_otlp_timeout_ms,
        service_name: args.trace_service_name.clone(),
    })
    .unwrap_or_else(|err| panic!("invalid tracing config: {err}"));

    if tracing.reporter_installed() {
        info!("Tracing configured: {}", tracing.config().summary());
    }

    // Install CUDA Profiler API signal handlers (SIGUSR1=start,
    // SIGUSR2=stop) so `nsys profile --capture-range=cudaProfilerApi
    // --capture-range-end=stop` can delimit trace windows reliably.
    // Per docs/plans/M_nsys-cuda-profiler-api-integration.md.
    #[cfg(feature = "cuda")]
    if let Err(e) = install_cuda_profiler_signal_handlers() {
        log::warn!("install_cuda_profiler_signal_handlers failed: {e}");
    }

    let model_path = args
        .model_path
        .to_str()
        .expect("Model path must be valid UTF-8");
    let resolved_model_path =
        hf_hub::resolve_model_path(model_path).expect("Failed to resolve model path");
    let resolved_model_path = resolved_model_path
        .to_str()
        .expect("Resolved model path must be valid UTF-8");
    let model_type = detect_model_type(resolved_model_path).expect("Failed to detect model type");
    info!("=== Infer Server - {} (GPU) ===", model_type);
    let metrics = infer::metrics::ServerMetrics::new(model_path);
    let runtime_topology = RuntimeTopology::discover();
    runtime_topology.log_summary();
    let worker_bootstrap = build_cuda_worker_bootstrap(&runtime_topology);
    let primary_worker = worker_bootstrap
        .first()
        .expect("configured CUDA workers must not be empty");
    let affinity = if worker_bootstrap.len() == 1 {
        bind_process_to_placement(&primary_worker.placement, "main-before-cuda")
    } else {
        info!(
            "Configured {} CUDA scheduler workers; per-worker bootstrap will bind CPU affinity before initializing CUDA",
            worker_bootstrap.len()
        );
        multi_worker_affinity_placeholder()
    };
    for worker in &worker_bootstrap {
        log_final_worker_topology(worker, &affinity);
    }
    metrics.set_runtime_topology(&runtime_topology, &primary_worker.placement, &affinity);
    if let Some(numastat) = sample_process_numa_maps() {
        let local_nodes = worker_bootstrap
            .iter()
            .map(|worker| worker.placement.numa_node)
            .collect::<Vec<_>>();
        metrics.set_runtime_numastat_for_nodes(&numastat, &local_nodes);
    }

    // Earliest possible CUDA snapshot: initialize the primary context (and
    // cuBLAS handle) here, BEFORE any cuda-kernels lazy-static cubin loaders
    // fire on first kernel use. The free-memory delta between this and the
    // pre-model-load snapshot in `bootstrap.rs:spawn_scheduler_handle_from_path`
    // tells us the AOT cubin + workspace overhead our boot path pays that
    // SGLang's lazy PyTorch boot does not. The captured value is fed into
    // `ServerRuntimeConfig.pre_model_free_bytes`, matching SGLang's
    // `pre_model_load_memory` semantics in `profile_max_num_token`.
    let pre_model_free_bytes = if worker_bootstrap.len() == 1 {
        early_pre_model_free_bytes(primary_worker)
    } else {
        info!(
            "Skipping process-wide early GPU memory snapshot for {} CUDA workers; each worker snapshots after NUMA binding before model load",
            worker_bootstrap.len()
        );
        None
    };

    info!("Loading model...");
    let start = Instant::now();
    let requested_kv_mode =
        parse_kv_cache_mode(&args.kv_cache_dtype).unwrap_or_else(|err| panic!("{err}"));

    let num_slots = args.num_slots.unwrap_or_else(|| {
        auto_num_slots(
            resolved_model_path,
            args.max_seq_len,
            requested_kv_mode.slot_sizing_format(),
            args.mem_fraction_static,
            Some(primary_worker.cuda_ordinal),
        )
    });
    let kv_candidates = kv_mode_candidates(requested_kv_mode, args.max_seq_len.is_some());
    let mut last_err = None;
    let mut selected_mode = None;
    let mut scheduler_workers = None;

    for (candidate_idx, (kv_cache_dtype, kv_pool_format, kv_mode_label)) in
        kv_candidates.iter().copied().enumerate()
    {
        match spawn_cuda_worker_group(
            model_path,
            &args,
            model_type,
            num_slots,
            kv_cache_dtype,
            kv_pool_format,
            &worker_bootstrap,
            pre_model_free_bytes,
            &metrics,
        ) {
            Ok(workers) => {
                selected_mode = Some((kv_cache_dtype, kv_pool_format, kv_mode_label));
                scheduler_workers = Some(workers);
                break;
            }
            Err(err) => {
                let err_chain = format!("{err:#}");
                let can_retry = candidate_idx + 1 < kv_candidates.len()
                    && err_chain.contains("requested scheduler envelope needs at least");
                if can_retry {
                    info!(
                        "KV auto fallback: {} failed to satisfy the requested envelope ({}); retrying denser layout",
                        kv_mode_label, err_chain
                    );
                    last_err = Some(err);
                    continue;
                }
                panic!("Failed to create scheduler: {err_chain}");
            }
        }
    }

    let (kv_cache_dtype, kv_pool_format, kv_mode_label) = selected_mode.unwrap_or_else(|| {
        panic!(
            "Failed to create scheduler{}",
            last_err
                .as_ref()
                .map(|err| format!(": {err:#}"))
                .unwrap_or_default()
        )
    });
    let mut scheduler_workers = scheduler_workers.expect("scheduler workers must exist");
    metrics.set_detokenizer_topology(scheduler_workers.len(), scheduler_workers.len());
    let primary_handle = scheduler_workers
        .first()
        .expect("scheduler workers must not be empty")
        .handle
        .clone();

    info!(
        "Config: model_path={}, cuda_graph={}, num_slots={} ({}), kv_cache_mode={} ({}), cuda_workers={}",
        args.model_path.display(),
        args.cuda_graph && !args.disable_cuda_graph,
        num_slots,
        if args.num_slots.is_some() {
            "explicit"
        } else {
            "auto"
        },
        args.kv_cache_dtype,
        kv_mode_label,
        scheduler_workers.len(),
    );
    info!("KV cache layout: contiguous={kv_cache_dtype:?}, paged_pool={kv_pool_format:?}");
    log_tier_config_overrides(&args);

    info!(
        "Model loaded: elapsed_ms={}, model_id={}",
        start.elapsed().as_millis(),
        primary_handle.model_id()
    );
    for worker in &mut scheduler_workers {
        worker.guard.wait_ready().unwrap_or_else(|err| {
            panic!(
                "scheduler warmup failed for worker={} gpu={}: {err}",
                worker.placement.worker_id, worker.placement.gpu_ordinal
            )
        });
    }
    if let Some(numastat) = sample_process_numa_maps() {
        let local_nodes = scheduler_workers
            .iter()
            .map(|worker| worker.placement.numa_node)
            .collect::<Vec<_>>();
        metrics.set_runtime_numastat_for_nodes(&numastat, &local_nodes);
    }

    let train_control_target = args
        .train_control_url
        .as_deref()
        .map(TrainControlTarget::parse)
        .transpose()
        .unwrap_or_else(|err| panic!("invalid --train-control-url: {err}"));
    let router_workers = scheduler_workers
        .iter()
        .map(|worker| NumaSchedulerWorker {
            handle: worker.handle.clone(),
            placement: worker.placement.clone(),
        })
        .collect::<Vec<_>>();
    let request_handle: Arc<dyn infer::request_handle::RequestHandle> =
        if matches!(model_type, ModelType::DeepSeekV4) && scheduler_workers.len() > 1 {
            info!(
                "Using DeepSeek distributed scheduler group: ranks={}",
                scheduler_workers.len()
            );
            Arc::new(DistributedSchedulerGroup::new(
                router_workers,
                metrics.clone(),
            ))
        } else {
            Arc::new(NumaSchedulerRouter::new(
                runtime_topology.clone(),
                router_workers,
                metrics.clone(),
            ))
        };
    let app = build_app_with_config(
        request_handle.clone(),
        metrics,
        HttpServerConfig {
            train_control_target,
            pool_models: parse_pool_models(&args.pool_models),
            runtime_topology: Some(runtime_topology.clone()),
            ..Default::default()
        },
    );

    let addr = format!("0.0.0.0:{}", args.port);
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {addr}: {e}"));
    axum::serve(
        axum::serve::ListenerExt::tap_io(listener, |tcp_stream| {
            let _ = tcp_stream.set_nodelay(true);
        }),
        app,
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .expect("Server error");

    // Drop the last submission handle before joining the scheduler thread so
    // request_rx disconnects and the scheduler can unwind its CUDA resources.
    drop(request_handle);
    drop(primary_handle);
    shutdown_started_workers(scheduler_workers);

    if tracing.reporter_installed() {
        info!("Flushing pending traces...");
        fastrace::flush();
    }
}

fn parse_pool_models(raw: &[String]) -> Vec<EnginePoolModelSpec> {
    raw.iter()
        .map(|spec| {
            EnginePoolModelSpec::parse_cli(spec)
                .unwrap_or_else(|err| panic!("invalid --pool-model `{spec}`: {err}"))
        })
        .collect()
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    info!("Shutdown signal received");
}

/// Install SIGUSR1/SIGUSR2 handlers that drive the CUDA Profiler API
/// (`cuProfilerStart` / `cuProfilerStop`). Used by `nsys profile
/// --capture-range=cudaProfilerApi --capture-range-end=stop` to
/// delimit trace capture exactly to a benchmark window — works
/// reliably across all workload shapes, where the legacy
/// `--delay`/`--duration` timing fails on low-density CUDA-Graph
/// long-context workloads.
///
/// Usage:
/// ```bash
/// nsys profile --output trace --trace cuda,nvtx,osrt \
///   --capture-range=cudaProfilerApi --capture-range-end=stop \
///   target/release/infer ...
///
/// # In another shell, after server is ready:
/// kill -USR1 $(pgrep -f 'target/release/infer')   # start capture
/// # ... run bench ...
/// kill -USR2 $(pgrep -f 'target/release/infer')   # stop capture
/// ```
///
/// Per `docs/plans/M_nsys-cuda-profiler-api-integration.md`. The
/// signal handler is dormant until SIGUSR1 fires, so production
/// deployments pay zero runtime cost.
#[cfg(feature = "cuda")]
fn install_cuda_profiler_signal_handlers() -> Result<(), Box<dyn std::error::Error>> {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigusr1 = signal(SignalKind::user_defined1())?;
    let mut sigusr2 = signal(SignalKind::user_defined2())?;

    tokio::spawn(async move {
        // The signal handler runs on a tokio worker thread that does NOT
        // have a CUDA context bound by default. cuProfilerStart/Stop
        // require a current context, so acquire the device-0 primary
        // context handle and bind it to this thread before calling.
        // This handle increments the primary context refcount which
        // CUDA already maintains for the main scheduler thread, so it
        // is safe to bind concurrently.
        let ctx_for_handler = match cudarc::driver::CudaContext::new(0) {
            Ok(c) => Some(c),
            Err(e) => {
                log::warn!(
                    "CUDA profiler signal handler could not acquire context: {e} \
                     — SIGUSR1/SIGUSR2 disabled this run"
                );
                return;
            }
        };

        loop {
            tokio::select! {
                _ = sigusr1.recv() => {
                    if let Some(ref ctx) = ctx_for_handler
                        && let Err(e) = ctx.bind_to_thread()
                    {
                        log::warn!("CUDA profiler bind_to_thread (start) failed: {e}");
                        continue;
                    }
                    match cudarc::driver::profiler_start() {
                        Ok(()) => info!("cuProfilerStart fired (nsys capture begin)"),
                        Err(e) => log::warn!("cuProfilerStart failed: {e}"),
                    }
                }
                _ = sigusr2.recv() => {
                    if let Some(ref ctx) = ctx_for_handler
                        && let Err(e) = ctx.bind_to_thread()
                    {
                        log::warn!("CUDA profiler bind_to_thread (stop) failed: {e}");
                        continue;
                    }
                    match cudarc::driver::profiler_stop() {
                        Ok(()) => info!("cuProfilerStop fired (nsys capture end)"),
                        Err(e) => log::warn!("cuProfilerStop failed: {e}"),
                    }
                }
            }
        }
    });

    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestedKvCacheMode {
    Auto,
    Explicit {
        kv_cache_dtype: KVCacheDtype,
        kv_pool_format: KVFormat,
    },
}

impl RequestedKvCacheMode {
    /// Per-slot KV bytes are sized against this format. For `auto`, mirror
    /// the first candidate in `kv_mode_candidates` (FP8) so the auto slot
    /// count matches the format the runtime will actually pick. If the FP8
    /// candidate fails the envelope check and the loop falls back to BF16,
    /// the caller may end up with more slots than the BF16 pool can fit —
    /// but that path also retries the whole runtime construction with the
    /// BF16 format, and the BF16 envelope check rejects oversized slot
    /// counts there.
    fn slot_sizing_format(self) -> KVFormat {
        match self {
            Self::Auto => KVFormat::FP8E4M3,
            Self::Explicit { kv_pool_format, .. } => kv_pool_format,
        }
    }
}

fn parse_kv_cache_mode(mode: &str) -> std::result::Result<RequestedKvCacheMode, String> {
    let normalized = mode.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "auto" => Ok(RequestedKvCacheMode::Auto),
        "bf16" => Ok(RequestedKvCacheMode::Explicit {
            kv_cache_dtype: KVCacheDtype::BF16,
            kv_pool_format: KVFormat::BF16,
        }),
        "fp8" => Ok(RequestedKvCacheMode::Explicit {
            kv_cache_dtype: KVCacheDtype::BF16,
            kv_pool_format: KVFormat::FP8E4M3,
        }),
        "int8" => Ok(RequestedKvCacheMode::Explicit {
            kv_cache_dtype: KVCacheDtype::INT8,
            kv_pool_format: KVFormat::INT8,
        }),
        "tq2" => Ok(RequestedKvCacheMode::Explicit {
            kv_cache_dtype: KVCacheDtype::BF16,
            kv_pool_format: KVFormat::TurboQuant {
                key_bits: 2,
                val_bits: 2,
            },
        }),
        "tq3" => Ok(RequestedKvCacheMode::Explicit {
            kv_cache_dtype: KVCacheDtype::BF16,
            kv_pool_format: KVFormat::TurboQuant {
                key_bits: 3,
                val_bits: 3,
            },
        }),
        "tq4" => Ok(RequestedKvCacheMode::Explicit {
            kv_cache_dtype: KVCacheDtype::BF16,
            kv_pool_format: KVFormat::TurboQuant {
                key_bits: 4,
                val_bits: 4,
            },
        }),
        _ => Err(format!(
            "Invalid --kv-cache-dtype '{mode}': expected {VALID_KV_CACHE_MODES}"
        )),
    }
}

fn kv_mode_candidates(
    requested_mode: RequestedKvCacheMode,
    _has_explicit_max_seq_len: bool,
) -> Vec<(KVCacheDtype, KVFormat, &'static str)> {
    match requested_mode {
        // Pure-inference auto: FP8 paged pool by default — halves the per-token
        // KV bytes vs BF16 with negligible quality regression on Qwen3 family,
        // matching SGLang/vLLM v1's default for L4-class GPUs. The contiguous
        // single-request cache stays BF16; quantization happens on migration
        // into the paged pool. Fall back to BF16 if the FP8 path can't satisfy
        // the requested envelope (e.g. no FP8 kernel for the model arch).
        RequestedKvCacheMode::Auto => {
            vec![
                (KVCacheDtype::BF16, KVFormat::FP8E4M3, "auto-fp8"),
                (KVCacheDtype::BF16, KVFormat::BF16, "auto-bf16"),
            ]
        }
        RequestedKvCacheMode::Explicit {
            kv_cache_dtype,
            kv_pool_format,
        } => vec![(kv_cache_dtype, kv_pool_format, "explicit")],
    }
}

fn scheduler_config_from_args(args: &Args, num_slots: usize) -> SchedulerConfig {
    let schedule_policy =
        SchedulePolicy::parse(&args.schedule_policy).unwrap_or_else(|err| panic!("{err}"));
    let admission_policy = SchedulerAdmissionPolicy::parse(&args.admission_policy)
        .unwrap_or_else(|err| panic!("{err}"));
    let mixed_policy = SchedulerMixedPolicy::parse(&args.scheduler_mixed_policy)
        .unwrap_or_else(|err| panic!("{err}"));
    let spec_draft_model =
        parse_draft_mode(&args.spec_draft_model).unwrap_or_else(|err| panic!("{err}"));
    // `chunked_prefill_size` / `max_prefill_tokens` are not plugged into the
    // `SchedulerConfig` here — when the operator did not supply a value, the
    // CUDA bootstrap resolves them against HBM via `RuntimeEnvelopeOverrides`.
    // Anything we set on the config now would be silently overwritten there.
    let mut config = SchedulerConfig {
        max_num_batched_tokens: args.max_num_batched_tokens,
        prefill_max_requests: args.prefill_max_requests,
        short_prompt_bypass_tokens: if args.disable_short_prompt_bypass {
            0
        } else {
            args.short_prompt_bypass_tokens
        },
        prefix_cache_enabled: !args.disable_radix_cache,
        admission_policy,
        cold_headroom: args.cold_headroom,
        schedule_policy,
        mixed_policy,
        stream_interval: args.stream_interval,
        spec_enabled: args.spec_enabled,
        spec_draft_k: args.spec_draft_k,
        spec_acceptance_threshold: args.spec_acceptance_threshold,
        spec_draft_model,
        spec_sparse_kv_enabled: args.spec_sparse_kv_enabled,
        spec_sparse_recent_tokens: args.spec_sparse_recent_tokens,
        spec_sparse_top_k_pages: args.spec_sparse_top_k_pages,
        mem_fraction_static: args.mem_fraction_static,
        min_seq_len: args.min_seq_len,
        kv_pool_fallback_bytes: args.kv_pool_fallback_mb.saturating_mul(1024 * 1024),
        ..SchedulerConfig::runtime_defaults(num_slots)
    };
    if let Some(high_water) = args.t1_host_pinned_high_water {
        config.t1_host_pinned_high_water = high_water;
    }
    if let Some(low_water) = args.t1_host_pinned_low_water {
        config.t1_host_pinned_low_water = low_water;
    }
    if let Some(keepalive_ticks) = args.t1_host_pinned_keepalive_ticks {
        config.t1_host_pinned_keepalive_ticks = keepalive_ticks;
    }
    if let Some(capacity_mb) = args.t1_host_pinned_capacity_mb {
        config.t1_host_pinned_capacity_bytes = Some(capacity_mb.saturating_mul(1024 * 1024));
    }
    if let Some(min_prompt_tokens) = args.t1_host_pinned_min_prompt_tokens {
        config.t1_host_pinned_min_prompt_tokens = min_prompt_tokens;
    }
    if let Some(root) = args.disk_store_root.as_ref() {
        config.disk_store_root = root.clone();
    }
    config.cluster_shared_backend = args
        .cluster_shared_root
        .as_ref()
        .map(|root| ClusterSharedBackendConfig::SharedFilesystem { root: root.clone() });
    config
}

fn parse_draft_mode(raw: &str) -> anyhow::Result<DraftMode> {
    let trimmed = raw.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "none" => Ok(DraftMode::None),
        "self" | "self-spec" | "selfspec" => Ok(DraftMode::SelfSpec),
        _ if trimmed.to_ascii_lowercase().starts_with("external:") => {
            let path = trimmed
                .split_once(':')
                .map(|(_, path)| path.trim())
                .unwrap_or_default();
            if path.is_empty() {
                anyhow::bail!("--spec-draft-model external:<path> requires a non-empty path");
            }
            Ok(DraftMode::External(PathBuf::from(path)))
        }
        other => anyhow::bail!(
            "unsupported --spec-draft-model '{other}': expected none, self, self-spec, or external:<path>"
        ),
    }
}

fn log_tier_config_overrides(args: &Args) {
    if args.t1_host_pinned_high_water.is_none()
        && args.t1_host_pinned_low_water.is_none()
        && args.t1_host_pinned_keepalive_ticks.is_none()
        && args.t1_host_pinned_capacity_mb.is_none()
        && args.t1_host_pinned_min_prompt_tokens.is_none()
        && args.disk_store_root.is_none()
        && args.cluster_shared_root.is_none()
    {
        return;
    }

    info!(
        "Tier config: t1_high_water={}, t1_low_water={}, t1_keepalive_ticks={}, t1_capacity_mb={}, t1_min_prompt_tokens={}, disk_store_root={}, cluster_shared_root={}",
        args.t1_host_pinned_high_water
            .map(|value| value.to_string())
            .unwrap_or_else(|| "default".to_string()),
        args.t1_host_pinned_low_water
            .map(|value| value.to_string())
            .unwrap_or_else(|| "default".to_string()),
        args.t1_host_pinned_keepalive_ticks
            .map(|value| value.to_string())
            .unwrap_or_else(|| "default".to_string()),
        args.t1_host_pinned_capacity_mb
            .map(|value| value.to_string())
            .unwrap_or_else(|| "default".to_string()),
        args.t1_host_pinned_min_prompt_tokens
            .map(|value| value.to_string())
            .unwrap_or_else(|| "default".to_string()),
        args.disk_store_root
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "default".to_string()),
        args.cluster_shared_root
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "disabled".to_string()),
    );
}

/// Auto-calculate num_slots from GPU memory and model config.
///
/// Strategy: estimate model weight size from safetensor files, subtract from GPU free
/// memory, take out the pool-side reserves the user passed via CLI flags, then
/// divide the remainder by the per-slot KV-cache cost at the requested dtype.
/// Clamp to [4, 128].
///
/// **Dtype awareness** (2026-04-15): the per-slot estimate now respects
/// `kv_pool_format`, so INT8 / FP8 quant pools auto-size to roughly twice the
/// number of slots BF16 picks at the same `max_seq_len`. Without this, the
/// auto-sizer was bf16-blind and quant KV silently lost its capacity benefit
/// at default flags. See
/// `docs/experience/wins/2026-04-15-bench-hbm-peak-throughput.md` for the
/// HBM inventory that surfaced this.
///
/// SGLang-compatible memory budget: `total_budget = gpu_total × mem_fraction_static`.
/// KV budget = total_budget − weight_size. Single knob, no multi-parameter tuning.
fn auto_num_slots(
    model_path: &str,
    max_seq_len: Option<usize>,
    kv_pool_format: KVFormat,
    mem_fraction_static: f64,
    cuda_ordinal: Option<usize>,
) -> usize {
    use infer::backend::cuda::tensor::DeviceContext;
    use std::path::Path;

    const MIN_SLOTS: usize = 4;
    const MAX_SLOTS: usize = 128;

    let seq_len = max_seq_len.unwrap_or(DEFAULT_SEQ_LEN);

    let weight_bytes: u64 = std::fs::read_dir(Path::new(model_path))
        .ok()
        .map_or(0, |entries| {
            entries
                .filter_map(std::result::Result::ok)
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
                .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                .sum()
        });

    let cuda_ctx = match cuda_ordinal {
        Some(cuda_ordinal) => with_device_ordinal_override(cuda_ordinal as u32, DeviceContext::new),
        None => DeviceContext::new(),
    };
    let Ok(_ctx) = cuda_ctx else {
        info!("auto_num_slots: CUDA init failed, using default 8 slots");
        return 8;
    };

    let Ok((free_bytes, total_bytes)) = DeviceContext::gpu_memory_info() else {
        info!("auto_num_slots: GPU memory query failed, using default 8 slots");
        return 8;
    };

    // SGLang formula: total_budget = gpu_total × fraction, kv_budget = total_budget − weights.
    // Cap by free_bytes so we don't over-admit on shared GPUs.
    let total_budget = (total_bytes as f64 * mem_fraction_static) as usize;
    let kv_budget = total_budget
        .min(free_bytes)
        .saturating_sub(weight_bytes as usize);

    let per_slot_bytes =
        estimate_per_slot_bytes(model_path, seq_len, CONTIGUOUS_KV_TOKENS, kv_pool_format);

    let slots = if per_slot_bytes > 0 {
        (kv_budget / per_slot_bytes).clamp(MIN_SLOTS, MAX_SLOTS)
    } else {
        8
    };

    let headroom_gb = (total_bytes as f64 * (1.0 - mem_fraction_static)) / 1e9;
    info!(
        "auto_num_slots: gpu_total={:.1}GB, weights={:.1}GB, fraction={:.0}%, \
         headroom={:.1}GB, kv_budget={:.1}GB, per_slot={:.1}MB, slots={}",
        total_bytes as f64 / 1e9,
        weight_bytes as f64 / 1e9,
        mem_fraction_static * 100.0,
        headroom_gb,
        kv_budget as f64 / 1e9,
        per_slot_bytes as f64 / 1e6,
        slots,
    );

    slots
}

/// Estimate per-slot memory cost from model config.json.
///
/// `kv_pool_format` is consulted for the contiguous KV byte width so INT8 and
/// FP8 quant pools auto-size to the smaller per-token footprint instead of
/// being charged as bf16. The recurrent state (Qwen3.5 hybrid models) is
/// always f32 regardless of the KV format choice.
fn estimate_per_slot_bytes(
    model_path: &str,
    seq_len: usize,
    chunk_size: usize,
    kv_pool_format: KVFormat,
) -> usize {
    use std::path::Path;

    let config_path = Path::new(model_path).join("config.json");
    let Ok(config_str) = std::fs::read_to_string(&config_path) else {
        return 0;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) else {
        return 0;
    };

    let num_layers = config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
    let num_kv_heads = config["num_key_value_heads"].as_u64().unwrap_or(4) as usize;
    let head_dim = config["head_dim"].as_u64().unwrap_or(128) as usize;

    // Check if hybrid model (Qwen3.5): only full-attention layers use KV cache
    let num_full_attn = config["num_full_attention_layers"]
        .as_u64()
        .unwrap_or(num_layers as u64) as usize;
    let kv_layers = num_full_attn.min(num_layers);

    // Per-slot contiguous KV bytes, dtype-aware via
    // KVFormat::pool_bytes_per_kv_head (BF16=2*head_dim, FP8/INT8=head_dim+4
    // including per-token f32 scale, TurboQuant=packed+norms).
    let bytes_per_kv_head_side = kv_pool_format.pool_bytes_per_kv_head(head_dim);
    // Per-slot cost = contiguous working buffer (chunk_size) + paged pool share (full seq_len).
    // Contiguous is the small prefill chunk; paged covers the full sequence.
    let bytes_per_token_kv = 2 * kv_layers * num_kv_heads * bytes_per_kv_head_side;
    let kv_bytes = bytes_per_token_kv * chunk_size + bytes_per_token_kv * seq_len;

    // Recurrent state (if hybrid): per linear layer, fixed size independent of seq_len
    let num_linear_layers = num_layers.saturating_sub(kv_layers);
    let linear_key_dim = config["linear_key_head_dim"].as_u64().unwrap_or(128) as usize;
    let linear_val_dim = config["linear_value_head_dim"].as_u64().unwrap_or(128) as usize;
    let linear_val_heads = config["linear_num_value_heads"].as_u64().unwrap_or(32) as usize;
    let recurrent_bytes =
        num_linear_layers * linear_val_heads * linear_key_dim * linear_val_dim * 4; // f32

    kv_bytes + recurrent_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_kv_cache_mode_supports_auto() {
        assert_eq!(
            parse_kv_cache_mode("auto").unwrap(),
            RequestedKvCacheMode::Auto
        );
    }

    #[test]
    fn parse_kv_cache_mode_supports_all_quantized_pool_modes() {
        assert_eq!(
            parse_kv_cache_mode("bf16").unwrap(),
            RequestedKvCacheMode::Explicit {
                kv_cache_dtype: KVCacheDtype::BF16,
                kv_pool_format: KVFormat::BF16,
            }
        );
        assert_eq!(
            parse_kv_cache_mode("fp8").unwrap(),
            RequestedKvCacheMode::Explicit {
                kv_cache_dtype: KVCacheDtype::BF16,
                kv_pool_format: KVFormat::FP8E4M3,
            }
        );
        assert_eq!(
            parse_kv_cache_mode("int8").unwrap(),
            RequestedKvCacheMode::Explicit {
                kv_cache_dtype: KVCacheDtype::INT8,
                kv_pool_format: KVFormat::INT8,
            }
        );
        assert_eq!(
            parse_kv_cache_mode("tq2").unwrap(),
            RequestedKvCacheMode::Explicit {
                kv_cache_dtype: KVCacheDtype::BF16,
                kv_pool_format: KVFormat::TurboQuant {
                    key_bits: 2,
                    val_bits: 2
                }
            }
        );
        assert_eq!(
            parse_kv_cache_mode("tq3").unwrap(),
            RequestedKvCacheMode::Explicit {
                kv_cache_dtype: KVCacheDtype::BF16,
                kv_pool_format: KVFormat::TurboQuant {
                    key_bits: 3,
                    val_bits: 3
                }
            }
        );
        assert_eq!(
            parse_kv_cache_mode("tq4").unwrap(),
            RequestedKvCacheMode::Explicit {
                kv_cache_dtype: KVCacheDtype::BF16,
                kv_pool_format: KVFormat::TurboQuant {
                    key_bits: 4,
                    val_bits: 4
                }
            }
        );
    }

    #[test]
    fn parse_kv_cache_mode_is_case_insensitive() {
        assert_eq!(
            parse_kv_cache_mode("FP8").unwrap(),
            RequestedKvCacheMode::Explicit {
                kv_cache_dtype: KVCacheDtype::BF16,
                kv_pool_format: KVFormat::FP8E4M3,
            }
        );
        assert_eq!(
            parse_kv_cache_mode("INT8").unwrap(),
            RequestedKvCacheMode::Explicit {
                kv_cache_dtype: KVCacheDtype::INT8,
                kv_pool_format: KVFormat::INT8,
            }
        );
    }

    #[test]
    fn parse_kv_cache_mode_rejects_unknown_values() {
        let err = parse_kv_cache_mode("fp4").unwrap_err();
        assert!(err.contains("fp4"));
        assert!(err.contains(VALID_KV_CACHE_MODES));
    }
}

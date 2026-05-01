use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use infer::backend::cuda::bootstrap::{
    InferenceEngineOptions, ServerRuntimeConfig, detect_model_type,
    spawn_scheduler_handle_from_path,
};
use infer::backend::cuda::tensor::DeviceContext;
use infer::hf_hub;
use infer::http_server::{HttpServerConfig, TrainControlTarget, build_app_with_config};
use infer::kv_tier::ClusterSharedBackendConfig;
use infer::logging;
use infer::model::{KVCacheDtype, KVFormat};
use infer::scheduler::{DraftMode, SchedulePolicy, SchedulerConfig};
use infer::trace_reporter::{TraceStartupConfig, configure_global_tracing};
use log::info;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
const DEFAULT_SEQ_LEN: usize = 4096;
const VALID_KV_CACHE_MODES: &str = "'auto', 'bf16', 'fp8', 'int8', 'tq2', 'tq3', or 'tq4'";
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
    #[arg(long, default_value = "none")]
    spec_draft_model: String,

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
    /// FlashInfer workspace, and OS. Default 0.85 matches SGLang's
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

    /// Optional upstream train control-plane URL to expose under `/v1/train/*`.
    #[arg(long)]
    train_control_url: Option<String>,

    /// Host-pinned T1 high-water mark as a fraction of host-pool capacity.
    #[arg(long)]
    t1_host_pinned_high_water: Option<f64>,

    /// Host-pinned T1 low-water mark as a fraction of host-pool capacity.
    #[arg(long)]
    t1_host_pinned_low_water: Option<f64>,

    /// Anti-thrash keepalive in radix logical ticks for freshly demoted T1 blocks.
    #[arg(long)]
    t1_host_pinned_keepalive_ticks: Option<u64>,

    /// Root directory for the node-local T2 disk store.
    #[arg(long)]
    disk_store_root: Option<PathBuf>,

    /// Root directory for the cluster-shared T3 shared-fs backend.
    #[arg(long)]
    cluster_shared_root: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    logging::init_default();

    let args = Args::parse();
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

    // Earliest possible CUDA snapshot: initialize the primary context (and
    // cuBLAS handle) here, BEFORE any cuda-kernels lazy-static cubin loaders
    // fire on first kernel use. The free-memory delta between this and the
    // pre-model-load snapshot in `bootstrap.rs:spawn_scheduler_handle_from_path`
    // tells us the AOT cubin + workspace overhead our boot path pays that
    // SGLang's lazy PyTorch boot does not. The captured value is fed into
    // `ServerRuntimeConfig.pre_model_free_bytes`, matching SGLang's
    // `pre_model_load_memory` semantics in `profile_max_num_token`.
    let pre_model_free_bytes = match DeviceContext::new() {
        Ok(_ctx) => match DeviceContext::gpu_memory_info() {
            Ok((free, total)) => {
                info!(
                    "GPU memory @ post_cuda_ctx (early): free={:.2} GB / total={:.2} GB \
                     (driver+ctx+cuBLAS overhead = {:.0} MB)",
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
                "Early DeviceContext::new() failed: {err} — pre_model_free snapshot disabled"
            );
            None
        }
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
        )
    });
    let metrics = infer::metrics::ServerMetrics::new(model_path);
    let kv_candidates = kv_mode_candidates(requested_kv_mode, args.max_seq_len.is_some());
    let mut last_err = None;
    let mut selected_mode = None;
    let mut scheduler = None;

    for (candidate_idx, (kv_cache_dtype, kv_pool_format, kv_mode_label)) in
        kv_candidates.iter().copied().enumerate()
    {
        let runtime = ServerRuntimeConfig {
            engine: InferenceEngineOptions {
                enable_cuda_graph: args.cuda_graph && !args.disable_cuda_graph,
            },
            scheduler: scheduler_config_from_args(&args, num_slots),
            runtime_envelope: infer::scheduler::RuntimeEnvelopeOverrides {
                chunked_prefill_size: args.chunked_prefill_size,
                max_prefill_tokens: args.max_prefill_tokens,
            },
            seed: 42,
            max_seq_len: args.max_seq_len,
            kv_cache_dtype,
            kv_pool_format,
            pre_model_free_bytes,
        };

        match spawn_scheduler_handle_from_path(model_path, runtime, metrics.clone()) {
            Ok(pair) => {
                selected_mode = Some((kv_cache_dtype, kv_pool_format, kv_mode_label));
                scheduler = Some(pair);
                break;
            }
            Err(err) => {
                let can_retry = candidate_idx + 1 < kv_candidates.len()
                    && err
                        .to_string()
                        .contains("requested scheduler envelope needs at least");
                if can_retry {
                    info!(
                        "KV auto fallback: {} failed to satisfy the requested envelope ({}); retrying denser layout",
                        kv_mode_label, err
                    );
                    last_err = Some(err);
                    continue;
                }
                panic!("Failed to create scheduler: {err}");
            }
        }
    }

    let (kv_cache_dtype, kv_pool_format, kv_mode_label) = selected_mode.unwrap_or_else(|| {
        panic!(
            "Failed to create scheduler{}",
            last_err
                .as_ref()
                .map(|err| format!(": {err}"))
                .unwrap_or_default()
        )
    });
    let (handle, mut scheduler_runtime) = scheduler.expect("scheduler pair must exist");

    info!(
        "Config: model_path={}, cuda_graph={}, num_slots={} ({}), kv_cache_mode={} ({})",
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
    );
    info!("KV cache layout: contiguous={kv_cache_dtype:?}, paged_pool={kv_pool_format:?}");
    log_tier_config_overrides(&args);

    info!(
        "Model loaded: elapsed_ms={}, model_id={}",
        start.elapsed().as_millis(),
        handle.model_id()
    );
    scheduler_runtime
        .wait_ready()
        .unwrap_or_else(|err| panic!("scheduler warmup failed: {err}"));

    let train_control_target = args
        .train_control_url
        .as_deref()
        .map(TrainControlTarget::parse)
        .transpose()
        .unwrap_or_else(|err| panic!("invalid --train-control-url: {err}"));
    let app = build_app_with_config(
        handle.clone(),
        metrics,
        HttpServerConfig {
            train_control_target,
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
    drop(handle);
    scheduler_runtime.wait();

    if tracing.reporter_installed() {
        info!("Flushing pending traces...");
        fastrace::flush();
    }
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    info!("Shutdown signal received");
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
        schedule_policy,
        stream_interval: args.stream_interval,
        spec_enabled: args.spec_enabled,
        spec_draft_k: args.spec_draft_k,
        spec_acceptance_threshold: args.spec_acceptance_threshold,
        spec_draft_model,
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
        && args.disk_store_root.is_none()
        && args.cluster_shared_root.is_none()
    {
        return;
    }

    info!(
        "Tier config: t1_high_water={}, t1_low_water={}, t1_keepalive_ticks={}, disk_store_root={}, cluster_shared_root={}",
        args.t1_host_pinned_high_water
            .map(|value| value.to_string())
            .unwrap_or_else(|| "default".to_string()),
        args.t1_host_pinned_low_water
            .map(|value| value.to_string())
            .unwrap_or_else(|| "default".to_string()),
        args.t1_host_pinned_keepalive_ticks
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

    let Ok(_ctx) = DeviceContext::new() else {
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

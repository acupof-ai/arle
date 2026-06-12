//! `arle serve` — in-process OpenAI v1 serving entry.
//!
//! Builds the backend router and runs `axum::serve` inside this process via
//! [`infer_api::serve_http`]. There is no standalone serve binary to exec: the
//! rewrite ships only the `arle` binary, so serving is in-process. The requested
//! backend must match the one compiled into this binary
//! ([`CompiledBackend::detect`]); a mismatch is rejected up front rather than
//! silently serving the compiled backend.

use std::{env, process::ExitCode};

use infer_api::{
    EngineLoadConfig, KvCacheDtype, ServeHttpOptions, ServeKvSsdOptions, ServeSpecOptions,
    ServeSpecType, serve_http,
};

use crate::{
    args::{Args, ServeArgs, ServeBackendArg, ServeKvCacheDtypeArg, ServeSpecTypeArg},
    hardware::CompiledBackend,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ServeBackend {
    Cuda,
    Metal,
    Hip,
    Vulkan,
    Cpu,
}

impl ServeBackend {
    fn label(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Hip => "hip",
            Self::Vulkan => "vulkan",
            Self::Cpu => "cpu",
        }
    }
}

/// Resolved, validated serve configuration ready to hand to
/// [`infer_api::serve_http`]. Holds the in-process options plus the diagnostics
/// the CLI prints before binding.
#[derive(Debug)]
struct ServeConfig {
    backend: ServeBackend,
    options: ServeHttpOptions,
    bind_warning: Option<String>,
}

pub(crate) fn run_serve(args: &Args, serve_args: ServeArgs) -> ExitCode {
    // CLI-fronted comm-backend knob, exported via env BEFORE the multiproc
    // coordinator spawns workers (children inherit it) and before any engine
    // build reads it (`TpRuntime::init_oneshot_comm`).
    // SAFETY: single CLI thread, pre-spawn, pre-tokio.
    unsafe {
        std::env::set_var(
            "ARLE_COMM_BACKEND",
            match serve_args.comm_backend {
                crate::args::ServeCommBackendArg::Auto => "auto",
                crate::args::ServeCommBackendArg::Nccl => "nccl",
            },
        );
        // Same pre-spawn env channel for the batched DSv4 decode opt-in
        // (`dsv4_batched_decode_enabled` in infer-cuda reads it; workers
        // inherit). Setting the env var directly remains a harness shim.
        if serve_args.dsv4_batched_decode {
            std::env::set_var("INFER_DSV4_BATCHED_DECODE", "1");
        }
    }
    match resolve_config(args, &serve_args) {
        Ok(config) => run_config(config),
        Err(err) => {
            eprintln!("[ARLE serve] error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run_config(config: ServeConfig) -> ExitCode {
    if let Some(warning) = config.bind_warning.as_deref() {
        eprintln!("[ARLE serve] warning: {warning}");
    }

    // Multi-rank TP CUDA models (DSv4, Qwen3.5/3.6 MoE): become the
    // multiproc-serve COORDINATOR (rank 0). Bind the relay, spawn N-1 worker
    // processes (one per GPU via INFER_CUDA_DEVICES / INFER_TP_SIZE), accept
    // their relay connects, boot-ping, install the admission broadcaster — THEN
    // fall through to the normal in-process serve below (rank 0 owns HTTP). The
    // broadcaster fans out each request the rank-0 engine admits to the workers
    // so they step their executors' NCCL `forward` in lockstep. The guard holds
    // the relay + worker pipes for the serve loop's lifetime; dropping it on
    // return EOFs the workers so they exit. On a single GPU it is a no-op and
    // serve proceeds single-process. Dense Qwen3 skips this entirely.
    //
    // `// STAGE 3:` chunked-prefill scratch chunk-bounding + the decode path
    // beyond a single short prompt are later stages (see Stage-2 markers in
    // `serve_multiproc.rs` / `multiproc_relay.rs` / `execution.rs`).
    #[cfg(all(unix, feature = "cuda"))]
    let _coordinator_guard = if config.backend == ServeBackend::Cuda
        && infer_api::cuda_model_takes_multiproc_serve(&config.options.model_path)
    {
        match crate::serve_multiproc::bind_relay_and_spawn_workers(
            &config.options.model_path,
            &config.options.engine_config,
        ) {
            Ok(guard) => guard,
            Err(err) => {
                eprintln!("[ARLE serve] multiproc coordinator setup failed: {err:#}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    eprintln!(
        "[ARLE serve] starting {} backend in-process on {}:{}",
        config.backend.label(),
        config.options.bind,
        config.options.port,
    );
    match serve_http(config.options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[ARLE serve] error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn resolve_config(args: &Args, serve_args: &ServeArgs) -> Result<ServeConfig, String> {
    let backend = resolve_backend(serve_args.backend)?;

    let model_path = serve_args
        .model_path
        .as_deref()
        .or(args.model_path.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(model_from_env)
        .ok_or_else(|| {
            "no model selected; pass `arle serve --model-path ...`, top-level `--model-path`, or set ARLE_MODEL".to_string()
        })?;

    // Metal is the only backend whose router honors a custom bind address today;
    // CUDA/CPU still bind, but flag a non-default `--bind` as Metal-only so the
    // surface matches what the legacy `metal_serve` bin exposed.
    let bind_warning = if backend == ServeBackend::Metal {
        None
    } else if serve_args.bind != "127.0.0.1" {
        Some(format!(
            "--bind={} was historically Metal-only; the {} backend now honors it in-process",
            serve_args.bind,
            backend.label()
        ))
    } else {
        None
    };

    // Speculative / MTP routing is checkpoint-native CUDA-only in the rewrite
    // serve stack. DSv4's depth-K MTP head lowers through `mtp_draft_tokens`;
    // Metal's monolith-era external draft route has not been re-ported, so the
    // CLI fails closed before startup rather than letting infer-api fail later.
    if serve_args.spec_type == ServeSpecTypeArg::Auto {
        return Err("--spec-type auto is not implemented; use mtp".to_string());
    }
    if serve_args.spec_type != ServeSpecTypeArg::None && backend != ServeBackend::Cuda {
        return Err("--spec-type is currently only supported by the CUDA backend".to_string());
    }
    if serve_args.mtp_draft_model.is_some() {
        return Err("--mtp-draft-model is not supported by the rewrite serve stack".to_string());
    }
    if serve_args.mtp_draft_tokens.is_some() && backend != ServeBackend::Cuda {
        return Err(
            "--mtp-draft-tokens is currently only supported by the CUDA backend".to_string(),
        );
    }

    // Surfaces the rewrite serve router does not expose yet. Reject rather than
    // silently ignore so the user is not misled into thinking they took effect.
    if serve_args.train_control_url.is_some() {
        return Err(
            "--train-control-url is not yet supported by the in-process serve stack (the rewrite router has no /v1/train/* routes)".to_string(),
        );
    }
    if !serve_args.pool_models.is_empty() {
        return Err(
            "--pool-model is not yet supported by the in-process serve stack (the rewrite router has no engine-pool /v1/models metadata)".to_string(),
        );
    }
    if !serve_args.extra_args.is_empty() {
        return Err(format!(
            "unrecognized backend flags after `--`: {}; the in-process serve stack does not forward to a standalone binary",
            serve_args.extra_args.join(" ")
        ));
    }

    let mut engine_config = resolve_engine_config(backend, serve_args)?;
    let spec = resolve_spec_options(backend, serve_args);
    let kv_ssd = resolve_kv_ssd_options(serve_args)?;
    // Lower MTP spec into the engine config at the CLI level so BOTH paths carry the
    // draft depth: the multiproc coordinator serializes `config.options.engine_config`
    // into `ARLE_WORKER_ENGINE_CONFIG` before spawning workers and NEVER runs
    // `serve_http`'s lowering — without this every rank builds with
    // `mtp_draft_tokens=None` and skips the MTP-head load. (serve_http re-applies the
    // same lowering idempotently for the single-proc path.)
    if spec.spec_type == ServeSpecType::Mtp {
        engine_config.mtp_draft_tokens = Some(spec.mtp_draft_tokens.unwrap_or(1));
    }

    let options = ServeHttpOptions {
        model_path,
        bind: serve_args.bind.clone(),
        port: serve_args.port,
        // `--no-cuda-graph` flips the CUDA decode-graph default off; honored by
        // the CUDA backend only (Metal/CPU ignore it).
        enable_cuda_graph: !args.no_cuda_graph,
        engine_config,
        spec,
        kv_ssd,
    };

    Ok(ServeConfig {
        backend,
        options,
        bind_warning,
    })
}

fn resolve_spec_options(backend: ServeBackend, serve_args: &ServeArgs) -> ServeSpecOptions {
    if !matches!(backend, ServeBackend::Metal | ServeBackend::Cuda) {
        return ServeSpecOptions::default();
    }
    let mut spec_type = match serve_args.spec_type {
        ServeSpecTypeArg::None => ServeSpecType::None,
        ServeSpecTypeArg::Auto => ServeSpecType::Auto,
        ServeSpecTypeArg::Mtp => ServeSpecType::Mtp,
    };
    if spec_type == ServeSpecType::None
        && (serve_args.mtp_draft_model.is_some() || serve_args.mtp_draft_tokens.is_some())
    {
        spec_type = ServeSpecType::Mtp;
    }
    ServeSpecOptions {
        spec_type,
        mtp_draft_model: serve_args.mtp_draft_model.clone(),
        mtp_draft_tokens: serve_args.mtp_draft_tokens,
    }
}

fn resolve_kv_ssd_options(serve_args: &ServeArgs) -> Result<ServeKvSsdOptions, String> {
    if serve_args.kv_ssd_max_bytes.is_some() && serve_args.kv_ssd_path.is_none() {
        return Err("--kv-ssd-max-bytes requires --kv-ssd-path".to_string());
    }

    Ok(ServeKvSsdOptions {
        root: serve_args.kv_ssd_path.clone(),
        max_bytes: serve_args.kv_ssd_max_bytes,
        high_performance_non_preemptive: serve_args.kv_ssd_path.is_some(),
    })
}

fn resolve_backend(arg: ServeBackendArg) -> Result<ServeBackend, String> {
    let requested = match arg {
        ServeBackendArg::Cuda => Some(ServeBackend::Cuda),
        ServeBackendArg::Metal => Some(ServeBackend::Metal),
        ServeBackendArg::Hip => Some(ServeBackend::Hip),
        ServeBackendArg::Vulkan => Some(ServeBackend::Vulkan),
        ServeBackendArg::Cpu => Some(ServeBackend::Cpu),
        ServeBackendArg::Auto => None,
    };

    let compiled = match CompiledBackend::detect() {
        CompiledBackend::Cuda => Some(ServeBackend::Cuda),
        CompiledBackend::Metal => Some(ServeBackend::Metal),
        #[cfg(feature = "hip")]
        CompiledBackend::Hip => Some(ServeBackend::Hip),
        #[cfg(feature = "vulkan")]
        CompiledBackend::Vulkan => Some(ServeBackend::Vulkan),
        CompiledBackend::Cpu => Some(ServeBackend::Cpu),
        #[cfg(not(any(
            feature = "cuda",
            feature = "metal",
            feature = "hip",
            feature = "vulkan",
            feature = "cpu"
        )))]
        CompiledBackend::None => None,
    };

    let Some(compiled) = compiled else {
        return Err(
            "serve requires a backend build; rebuild with cuda, metal/no-cuda, vulkan/no-cuda, or cpu/no-cuda"
                .to_string(),
        );
    };

    match requested {
        // `auto` always serves the compiled backend.
        None => Ok(compiled),
        // An explicit backend must match the one compiled in: serving is
        // in-process, so a mismatch cannot be satisfied.
        Some(requested) if requested == compiled => Ok(requested),
        Some(requested) => Err(format!(
            "requested --backend {} but this binary was built with the {} backend; rebuild with the matching feature or use --backend {}/auto",
            requested.label(),
            compiled.label(),
            compiled.label(),
        )),
    }
}

fn model_from_env() -> Option<String> {
    env::var("ARLE_MODEL")
        .ok()
        .or_else(|| env::var("AGENT_INFER_MODEL").ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn resolve_engine_config(
    backend: ServeBackend,
    serve_args: &ServeArgs,
) -> Result<EngineLoadConfig, String> {
    let kv_cache_dtype = match serve_args.kv_cache_dtype {
        ServeKvCacheDtypeArg::Auto => KvCacheDtype::Auto,
        ServeKvCacheDtypeArg::Bf16 => KvCacheDtype::Bf16,
        ServeKvCacheDtypeArg::Int8 => KvCacheDtype::Int8,
        ServeKvCacheDtypeArg::Fp8 => KvCacheDtype::Fp8,
        ServeKvCacheDtypeArg::Tq4 => KvCacheDtype::Tq4,
    };
    let mut config = EngineLoadConfig {
        kv_cache_dtype,
        ..EngineLoadConfig::default()
    };

    // Quant KV dtypes route to the backend that implements them. Reject the
    // mismatched combos at the CLI boundary; the per-backend `*KvCacheDtype::
    // resolve` is the fine authority for "supported vs pending #68 T3" on the
    // matching backend (CUDA fp8/tq4 flow through to it and fail loud there
    // until their paged path lands).
    match config.kv_cache_dtype {
        // INT8 is the Metal int8 path today; CUDA int8 lands with #68 T3 (this
        // arm then widens to also admit the CUDA backend).
        KvCacheDtype::Int8 if backend != ServeBackend::Metal => {
            return Err(format!(
                "--kv-cache-dtype int8 is currently implemented for the Metal backend; active backend is {}",
                backend.label()
            ));
        }
        // FP8 / TQ4 are CUDA-only paged quant modes.
        KvCacheDtype::Fp8 | KvCacheDtype::Tq4 if backend != ServeBackend::Cuda => {
            return Err(format!(
                "--kv-cache-dtype {} is a CUDA-only paged quant mode; active backend is {}",
                config.kv_cache_dtype.label(),
                backend.label()
            ));
        }
        _ => {}
    }

    if serve_args.low_impact {
        config.low_impact = true;
        config.num_slots = 1;
        config.chunked_prefill_size = config.chunked_prefill_size.min(32);
    }

    if let Some(value) = serve_args.num_slots {
        config.num_slots = value;
    }
    if let Some(value) = serve_args.total_pages {
        config.total_pages = value;
    }
    if let Some(value) = serve_args.page_size {
        config.page_size = value;
    }
    if let Some(value) = serve_args.max_prompt_tokens {
        config.max_prompt_tokens = value;
    }
    if let Some(value) = serve_args.max_total_tokens {
        config.max_total_tokens = value;
    }
    if let Some(value) = serve_args.chunked_prefill_size {
        config.chunked_prefill_size = value;
    }
    if serve_args.kv_t1_budget_bytes.is_some() {
        config.kv_t1_budget_bytes = serve_args.kv_t1_budget_bytes;
    }
    if serve_args.memory_budget_bytes.is_some() {
        config.memory_budget_bytes = serve_args.memory_budget_bytes;
    }
    if serve_args.system_reserve_bytes.is_some() {
        config.system_reserve_bytes = serve_args.system_reserve_bytes;
    }
    config.allow_swap = serve_args.allow_swap;

    if config.max_prompt_tokens > config.max_total_tokens {
        return Err(format!(
            "--max-prompt-tokens ({}) must be <= --max-total-tokens ({})",
            config.max_prompt_tokens, config.max_total_tokens
        ));
    }

    let capacity_tokens = config
        .total_pages
        .checked_mul(config.page_size)
        .ok_or_else(|| "--total-pages * --page-size overflows usize".to_string())?;
    if capacity_tokens < config.max_total_tokens {
        return Err(format!(
            "KV capacity is too small for one max-length request: total_pages({}) * page_size({}) = {} tokens, max_total_tokens={}",
            config.total_pages, config.page_size, capacity_tokens, config.max_total_tokens
        ));
    }

    Ok(config)
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    fn parse_serve(argv: &[&str]) -> (Args, ServeArgs) {
        let mut args = Args::parse_from(argv);
        let serve = match args.command.take().expect("serve command") {
            crate::args::CliCommand::Serve(serve) => *serve,
            _ => panic!("expected serve"),
        };
        (args, serve)
    }

    /// The compiled backend (what `--backend auto` / a matching explicit backend
    /// resolves to). Tests pick the backend the current binary was built with so
    /// the in-process backend-match check passes.
    fn compiled_backend_flag() -> &'static str {
        match CompiledBackend::detect() {
            CompiledBackend::Cuda => "cuda",
            CompiledBackend::Metal => "metal",
            #[cfg(feature = "hip")]
            CompiledBackend::Hip => "hip",
            #[cfg(feature = "vulkan")]
            CompiledBackend::Vulkan => "vulkan",
            CompiledBackend::Cpu => "cpu",
            #[cfg(not(any(
                feature = "cuda",
                feature = "metal",
                feature = "hip",
                feature = "vulkan",
                feature = "cpu"
            )))]
            CompiledBackend::None => "auto",
        }
    }

    fn skip_if_no_backend() -> bool {
        !CompiledBackend::detect().supports_inference()
    }

    #[test]
    fn serve_uses_subcommand_model_path() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "from-sub",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(config.options.model_path, "from-sub");
    }

    #[test]
    fn serve_uses_top_level_model_path() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "--model-path",
            "from-root",
            "serve",
            "--backend",
            compiled_backend_flag(),
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(config.options.model_path, "from-root");
    }

    #[test]
    fn no_cuda_graph_flag_disables_decode_graph_default() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "--no-cuda-graph",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert!(!config.options.enable_cuda_graph);
    }

    #[test]
    fn default_enables_decode_graph_default() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert!(config.options.enable_cuda_graph);
    }

    #[test]
    fn port_and_bind_flow_into_options() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--port",
            "8123",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(config.options.port, 8123);
        assert_eq!(config.options.bind, "127.0.0.1");
    }

    #[test]
    fn low_impact_keeps_context_capacity() {
        if skip_if_no_backend() {
            return;
        }
        let defaults = EngineLoadConfig::default();
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--low-impact",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(config.options.engine_config.num_slots, 1);
        assert_eq!(
            config.options.engine_config.total_pages,
            defaults.total_pages
        );
        assert_eq!(
            config.options.engine_config.max_prompt_tokens,
            defaults.max_prompt_tokens
        );
        assert_eq!(
            config.options.engine_config.max_total_tokens,
            defaults.max_total_tokens
        );
        assert_eq!(config.options.engine_config.chunked_prefill_size, 32);
        assert!(config.options.engine_config.low_impact);
    }

    #[test]
    fn kv_ssd_options_are_forwarded_to_service_layer() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--kv-ssd-path",
            "/tmp/arle-kv",
            "--kv-ssd-max-bytes",
            "1073741824",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(
            config.options.kv_ssd.root.as_deref(),
            Some(std::path::Path::new("/tmp/arle-kv"))
        );
        assert_eq!(config.options.kv_ssd.max_bytes, Some(1_073_741_824));
        assert!(config.options.kv_ssd.high_performance_non_preemptive);
    }

    #[test]
    fn kv_cache_dtype_int8_flows_to_metal_engine_config() {
        if skip_if_no_backend() || compiled_backend_flag() != "metal" {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            "metal",
            "--model-path",
            "model",
            "--kv-cache-dtype",
            "int8",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(
            config.options.engine_config.kv_cache_dtype,
            KvCacheDtype::Int8
        );
    }

    #[test]
    fn kv_cache_dtype_int8_rejects_non_metal_backend() {
        if skip_if_no_backend() || compiled_backend_flag() == "metal" {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--kv-cache-dtype",
            "int8",
        ]);
        let err = resolve_config(&args, &serve).expect_err("non-metal int8 rejected");
        assert!(err.contains("--kv-cache-dtype int8"), "got: {err}");
    }

    #[test]
    fn kv_cache_dtype_fp8_rejects_non_cuda_backend() {
        // FP8/TQ4 are CUDA-only paged quant modes; on any non-CUDA backend the
        // CLI guard must reject early (symmetric to the int8/Metal guard). Runs
        // on the metal/cpu lanes; skips on CUDA (where fp8 flows through to the
        // CUDA resolve's "pending #68 T3" bail instead).
        if skip_if_no_backend() || compiled_backend_flag() == "cuda" {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--kv-cache-dtype",
            "fp8",
        ]);
        let err = resolve_config(&args, &serve).expect_err("non-cuda fp8 rejected");
        assert!(err.contains("CUDA-only paged quant mode"), "got: {err}");
    }

    #[test]
    fn memory_budget_flags_flow_to_engine_config() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--memory-budget-bytes",
            "25769803776",
            "--system-reserve-bytes",
            "17179869184",
            "--allow-swap",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(
            config.options.engine_config.memory_budget_bytes,
            Some(25_769_803_776)
        );
        assert_eq!(
            config.options.engine_config.system_reserve_bytes,
            Some(17_179_869_184)
        );
        assert!(config.options.engine_config.allow_swap);
    }

    #[test]
    fn explicit_engine_budget_overrides_low_impact() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--low-impact",
            "--num-slots",
            "2",
            "--total-pages",
            "2048",
            "--max-prompt-tokens",
            "8192",
            "--max-total-tokens",
            "16384",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(config.options.engine_config.num_slots, 2);
        assert_eq!(config.options.engine_config.total_pages, 2048);
        assert_eq!(config.options.engine_config.max_prompt_tokens, 8192);
        assert_eq!(config.options.engine_config.max_total_tokens, 16_384);
    }

    #[test]
    fn engine_budget_rejects_capacity_below_max_total() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--total-pages",
            "1",
            "--page-size",
            "16",
            "--max-prompt-tokens",
            "16",
            "--max-total-tokens",
            "32",
        ]);
        let err = resolve_config(&args, &serve).expect_err("capacity rejected");
        assert!(err.contains("KV capacity is too small"), "got: {err}");
    }

    #[test]
    fn explicit_backend_mismatch_is_rejected() {
        // Pick a backend that is NOT the compiled one (or skip if no backend is
        // compiled in, where every explicit backend is rejected anyway).
        let other = match CompiledBackend::detect() {
            CompiledBackend::Metal => "cuda",
            #[cfg(feature = "hip")]
            CompiledBackend::Hip => "metal",
            #[cfg(feature = "vulkan")]
            CompiledBackend::Vulkan => "metal",
            CompiledBackend::Cuda | CompiledBackend::Cpu => "metal",
            #[cfg(not(any(
                feature = "cuda",
                feature = "metal",
                feature = "hip",
                feature = "vulkan",
                feature = "cpu"
            )))]
            CompiledBackend::None => return,
        };
        let (args, serve) =
            parse_serve(&["arle", "serve", "--backend", other, "--model-path", "model"]);
        let err = resolve_config(&args, &serve).expect_err("backend mismatch rejected");
        assert!(err.contains("but this binary was built with"), "got: {err}");
    }

    #[test]
    fn train_control_url_is_rejected() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--train-control-url",
            "http://localhost:9000",
        ]);
        let err = resolve_config(&args, &serve).expect_err("train control url rejected");
        assert!(err.contains("--train-control-url"), "got: {err}");
    }

    #[test]
    fn pool_model_is_rejected() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--pool-model",
            "embed=/models/embed,type=embedding",
        ]);
        let err = resolve_config(&args, &serve).expect_err("pool model rejected");
        assert!(err.contains("--pool-model"), "got: {err}");
    }

    #[test]
    fn extra_args_after_dashes_are_rejected() {
        if skip_if_no_backend() {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--",
            "--num-slots",
            "8",
        ]);
        let err = resolve_config(&args, &serve).expect_err("extra args rejected");
        assert!(err.contains("unrecognized backend flags"), "got: {err}");
    }

    #[test]
    fn non_cuda_spec_type_errors_when_compiled_non_cuda() {
        if skip_if_no_backend() {
            return;
        }
        // Only meaningful when the compiled backend is not CUDA; on a CUDA
        // build `--spec-type mtp` is accepted for checkpoint-native DSv4 MTP.
        if CompiledBackend::detect() == CompiledBackend::Cuda {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--spec-type",
            "mtp",
        ]);
        let err = resolve_config(&args, &serve).expect_err("reject non-CUDA spec type");
        assert_eq!(
            err,
            "--spec-type is currently only supported by the CUDA backend"
        );
    }

    #[test]
    fn cuda_spec_type_accepted_when_compiled_cuda() {
        if CompiledBackend::detect() != CompiledBackend::Cuda {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            "cuda",
            "--model-path",
            "model",
            "--spec-type",
            "mtp",
        ]);
        let config = resolve_config(&args, &serve).expect("CUDA accepts checkpoint-native MTP");
        assert_eq!(config.backend, ServeBackend::Cuda);
        assert_eq!(config.options.spec.spec_type, ServeSpecType::Mtp);
        assert_eq!(config.options.engine_config.mtp_draft_tokens, Some(1));
    }

    #[test]
    fn metal_spec_type_rejected_when_compiled_metal() {
        if CompiledBackend::detect() != CompiledBackend::Metal {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            "metal",
            "--model-path",
            "model",
            "--spec-type",
            "mtp",
        ]);
        let err = resolve_config(&args, &serve).expect_err("Metal MTP is fail-closed");
        assert_eq!(
            err,
            "--spec-type is currently only supported by the CUDA backend"
        );
    }

    #[test]
    fn mtp_draft_model_is_rejected() {
        if CompiledBackend::detect() != CompiledBackend::Metal {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            "metal",
            "--model-path",
            "model",
            "--mtp-draft-model",
            "draft-model",
            "--mtp-draft-tokens",
            "2",
        ]);
        let err = resolve_config(&args, &serve).expect_err("external draft model is not re-ported");
        assert_eq!(
            err,
            "--mtp-draft-model is not supported by the rewrite serve stack"
        );
    }

    #[test]
    fn serve_backend_arle_alias_selects_compiled_backend() {
        let (_args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            "arle",
            "--model-path",
            "/models/main",
        ]);
        assert_eq!(serve.backend, ServeBackendArg::Auto);
    }
}

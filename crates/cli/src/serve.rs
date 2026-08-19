//! `arle serve` — in-process OpenAI v1 serving entry.
//!
//! Builds the backend router and runs `axum::serve` inside this process via
//! [`infer_api::serve_http`]. There is no standalone serve binary to exec: the
//! rewrite ships only the `arle` binary, so serving is in-process. The requested
//! backend must match the one compiled into this binary
//! ([`CompiledBackend::detect`]); a mismatch is rejected up front rather than
//! silently serving the compiled backend.

use std::{env, process::ExitCode};

#[cfg(feature = "cuda")]
use infer_api::LoadedInferenceEngine;
use infer_api::{
    DEFAULT_MTP_DRAFT_TOKENS, DEFAULT_MTP_DRAFT_TOPK, EngineLoadConfig, KvCacheDtype,
    ServeHttpOptions, ServeSpecOptions, ServeSpecType, serve_http,
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
}

/// Exits when the supervising process is gone — the only cleanup that survives
/// the supervisor being SIGKILLed, since no handler of its own gets to run.
///
/// Two signals, because neither alone is sufficient: reparenting is exact but
/// only applies when the engine is a direct child (a shell wrapper breaks it),
/// while `kill(pid, 0)` works for any ancestor but can be fooled by a recycled
/// pid.
fn watch_parent(pid: i32) {
    // SAFETY: getppid/kill with signal 0 only read process state.
    let direct_child = unsafe { libc::getppid() } == pid;
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(2));
            let reparented = direct_child && unsafe { libc::getppid() } != pid;
            let gone = unsafe { libc::kill(pid, 0) } != 0
                && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
            if reparented || gone {
                eprintln!("[ARLE serve] parent {pid} exited; shutting down");
                // `_exit`, not `exit`: during a multi-GiB weight load another
                // thread holds the allocator lock, and atexit handlers would
                // block there — the orphan this watchdog exists to prevent.
                // SAFETY: immediate termination; the kernel reclaims everything.
                unsafe { libc::_exit(0) };
            }
        }
    });
}

pub(crate) fn run_serve(args: &Args, serve_args: ServeArgs) -> ExitCode {
    if let Some(pid) = serve_args.parent_pid {
        watch_parent(pid);
    }
    // Lower the probe flags into the env the CUDA executor reads at first use.
    // MUST run pre-spawn: multiproc TP rank children inherit the parent env
    // (env is the transport; flags are the only public interface).
    // CC-trajectory capture: install the /v1/messages dump sink before the
    // router (single- and multiproc coordinator both handle HTTP in-process).
    if let Some(dir) = serve_args.dump_messages_dir.as_deref() {
        if let Err(err) = infer_api::set_messages_dump_dir(dir) {
            eprintln!(
                "[ARLE serve] error: create --dump-messages-dir {}: {err}",
                dir.display()
            );
            return ExitCode::FAILURE;
        }
        eprintln!(
            "[ARLE serve] dumping raw /v1/messages bodies to {}",
            dir.display()
        );
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
    // The DSpark train sidecar + `--dspark-markov-init` are wired only on the
    // single-process serve path (`on_engine_loaded` below); the multiproc
    // coordinator returns before that hook exists. Fail fast rather than serve
    // with the sidecar silently inert — a no-op flag must reject, not no-op.
    // Gate on the RESOLVED world size, not just model kind: a multiproc-capable
    // model (Qwen3.5/3.6, DSv4) at --tensor-parallel-size 1 serves single-process,
    // where the sidecar DOES run — rejecting that would block DSpark test-time
    // training on the one config that supports it.
    #[cfg(all(unix, feature = "cuda"))]
    if config.backend == ServeBackend::Cuda
        && config.options.spec.dspark_markov_init.is_some()
        && config.options.engine_config.world_size.unwrap_or(1) > 1
        && infer_api::cuda_model_takes_multiproc_serve(&config.options.model_path)
    {
        eprintln!(
            "[ARLE serve] --dspark-markov-init requires a single-process \
             serve, but this model runs multiproc TP; the train sidecar is not wired into \
             the coordinator. Serve a single-GPU model (e.g. Qwen3.6-27B-FP8 TP=1) for \
             DSpark test-time training."
        );
        return ExitCode::FAILURE;
    }

    // Multi-rank TP CUDA models (DSv4, Qwen3.5/3.6 MoE): SPMD (B) split. The parent
    // becomes the engine-less coordinator — binds the relay, spawns all N workers,
    // and runs the thin coordinator HTTP loop. Single GPU returns an empty vec and
    // falls through to the byte-identical in-process path below; dense Qwen3 skips it.
    #[cfg(all(unix, feature = "cuda"))]
    if config.backend == ServeBackend::Cuda
        && infer_api::cuda_model_takes_multiproc_serve(&config.options.model_path)
    {
        match crate::serve_multiproc::bind_relay_and_spawn_workers(
            &config.options.model_path,
            &config.options.engine_config,
        ) {
            Ok(groups) if !groups.is_empty() => {
                let (relays, guards): (Vec<_>, Vec<_>) =
                    groups.into_iter().map(|c| (c.relay, c.guard)).unzip();
                eprintln!(
                    "[ARLE serve] starting cuda multiproc coordinator on {}:{}",
                    config.options.bind, config.options.port,
                );
                let result = infer_api::serve_coordinator_http_dp(
                    &config.options.model_path,
                    &config.options.bind,
                    config.options.port,
                    config.options.engine_config.max_thinking_tokens,
                    relays,
                );
                // Keep the worker children alive until the serve loop returns.
                drop(guards);
                return match result {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(err) => {
                        eprintln!("[ARLE serve] coordinator error: {err:#}");
                        ExitCode::FAILURE
                    }
                };
            }
            // Empty: single GPU — fall through to the in-process serve below.
            Ok(_) => {}
            Err(err) => {
                eprintln!("[ARLE serve] multiproc coordinator setup failed: {err:#}");
                return ExitCode::FAILURE;
            }
        }
    }

    eprintln!(
        "[ARLE serve] starting {} backend in-process on {}:{}",
        config.backend.label(),
        config.options.bind,
        config.options.port,
    );

    // `--dspark-markov-init` installs a saved Markov head over the draft
    // checkpoint's once the engine is loaded. The head slot itself is
    // materialized by the executor (`markov_head_rank`).
    #[cfg(feature = "cuda")]
    #[allow(clippy::type_complexity)]
    let on_engine_loaded: Option<
        Box<dyn Fn(&std::sync::Arc<LoadedInferenceEngine>) -> anyhow::Result<()> + Send + Sync>,
    > = {
        let init = config.options.spec.dspark_markov_init.clone();
        let is_dspark = config.options.spec.spec_type == ServeSpecType::Dspark;
        if init.is_some() && !is_dspark {
            eprintln!(
                "[ARLE serve] warning: --dspark-markov-init requires --spec-type dspark; ignoring"
            );
        }
        match init.filter(|_| is_dspark) {
            None => None,
            Some(path) => Some(Box::new(
                move |engine: &std::sync::Arc<LoadedInferenceEngine>| {
                    let (w1, w2) = spec_train::markov_head::load(&path)?;
                    engine.update_dspark_markov_weights(&w1, &w2)?;
                    eprintln!(
                        "[ARLE serve] DSpark Markov head loaded from {}",
                        path.display()
                    );
                    Ok(())
                },
            )),
        }
    };
    #[cfg(not(feature = "cuda"))]
    let on_engine_loaded = None;

    match serve_http(config.options, on_engine_loaded) {
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

    // Speculative / MTP routing is checkpoint-native CUDA-only in the rewrite
    // serve stack. DSv4's depth-K MTP head lowers through `mtp_draft_tokens`;
    // Metal's monolith-era external draft route has not been re-ported, so the
    // CLI fails closed before startup rather than letting infer-api fail later.
    if serve_args.spec_type == ServeSpecTypeArg::Auto {
        return Err("--spec-type auto is not implemented; use mtp or dspark".to_string());
    }
    if serve_args.spec_type != ServeSpecTypeArg::None && backend != ServeBackend::Cuda {
        return Err("--spec-type is currently only supported by the CUDA backend".to_string());
    }
    if serve_args.spec_type == ServeSpecTypeArg::Dspark && serve_args.mtp_draft_model.is_none() {
        return Err(
            "--spec-type dspark requires --mtp-draft-model <DSpark/DFlash checkpoint dir>"
                .to_string(),
        );
    }
    if serve_args.mtp_draft_model.is_some() && serve_args.spec_type != ServeSpecTypeArg::Dspark {
        return Err(
            "--mtp-draft-model is only consumed by --spec-type dspark on this serve stack"
                .to_string(),
        );
    }
    if serve_args.mtp_draft_tokens.is_some() && backend != ServeBackend::Cuda {
        return Err(
            "--mtp-draft-tokens is currently only supported by the CUDA backend".to_string(),
        );
    }
    if serve_args.mtp_draft_topk.is_some() && backend != ServeBackend::Cuda {
        return Err("--mtp-draft-topk is currently only supported by the CUDA backend".to_string());
    }

    // Surfaces the rewrite serve router does not expose yet. Reject rather than
    // silently ignore so the user is not misled into thinking they took effect.
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
    if serve_args.lora_adapters.is_some() && backend != ServeBackend::Cuda {
        return Err("--lora-adapters is currently only supported by the CUDA backend".to_string());
    }

    let mut engine_config = resolve_engine_config(backend, serve_args)?;
    // The student-LoRA re-merge rides the engine config so multiproc worker
    // ranks (which see only ARLE_WORKER_ENGINE_CONFIG) apply it too.
    engine_config.student_lora_adapters = serve_args.lora_adapters.clone();
    engine_config.student_lora_alpha = serve_args.lora_alpha;
    // DSv4 multiproc auto-context: resolve max_total_tokens from the checkpoint
    // when unset. CUDA-only (the gate fns are CUDA-gated) — no non-CUDA path.
    #[cfg(feature = "cuda")]
    if backend == ServeBackend::Cuda
        && serve_args.max_prompt_tokens.is_none()
        && serve_args.max_total_tokens.is_none()
        && infer_api::cuda_model_takes_multiproc_serve(&model_path)
        && let Some(max_ctx) = crate::read_model_max_context(&model_path)
    {
        // DSv4's FlashMLA pool sizing crashes the coordinator instead of
        // shrinking num_slots when max_seq_len is very large (pod-verified
        // 2026-07-06: DeepSeek-V4-Flash-FP8's native 1,048,576-token context
        // OOMs the fixed per-slot band on boot). Cap the auto-resolved
        // default there; an explicit `--max-total-tokens` still bypasses
        // this untouched. Qwen35 profiles its pool from measured free VRAM
        // and has no such ceiling.
        let resolved = if infer_api::cuda_model_is_dsv4(&model_path) {
            max_ctx.min(infer_api::DSV4_AUTO_CONTEXT_CEILING)
        } else {
            max_ctx
        };
        log::info!(
            "DSv4 max context: auto-resolved to {resolved} from {model_path}/config.json (max_position_embeddings={max_ctx})"
        );
        engine_config.max_prompt_tokens = resolved;
        engine_config.max_total_tokens = resolved;
    }
    let spec = resolve_spec_options(backend, serve_args);
    // The L3 spill request rides the engine config so BOTH CUDA paths carry
    // it: the multiproc coordinator serializes ONLY engine_config into
    // ARLE_WORKER_ENGINE_CONFIG, so serve-layer-only options never reach
    // worker ranks (same reason as the MTP lowering below). Validated here,
    // pre-spawn, for the same reason. A bare `--kv-disk` parses as the empty
    // path (clap `default_missing_value`) and resolves to the default root.
    engine_config.kv_ssd_root = serve_args.kv_disk.clone().map(|dir| {
        if dir.as_os_str().is_empty() {
            infer_api::default_kv_ssd_root()
        } else {
            dir
        }
    });
    engine_config.kv_disk_limit = serve_args.kv_disk_limit;
    infer_api::validate_kv_ssd_config(&engine_config).map_err(|err| format!("{err:#}"))?;
    // Lower MTP spec into the engine config at the CLI level so BOTH paths carry the
    // draft depth: the multiproc coordinator serializes `config.options.engine_config`
    // into `ARLE_WORKER_ENGINE_CONFIG` before spawning workers and NEVER runs
    // `serve_http`'s lowering — without this every rank builds with
    // `mtp_draft_tokens=None` and skips the MTP-head load. (serve_http re-applies the
    // same lowering idempotently for the single-proc path.)
    if spec.spec_type == ServeSpecType::Mtp {
        engine_config.mtp_draft_tokens =
            Some(spec.mtp_draft_tokens.unwrap_or(DEFAULT_MTP_DRAFT_TOKENS));
        engine_config.mtp_draft_topk = Some(spec.mtp_draft_topk.unwrap_or(DEFAULT_MTP_DRAFT_TOPK));
    }
    if spec.spec_type == ServeSpecType::Dspark {
        // Validated non-None above; lowered here so multiproc worker ranks
        // (which see only ARLE_WORKER_ENGINE_CONFIG) load the drafter too.
        engine_config.dspark_draft_model =
            spec.mtp_draft_model.clone().map(std::path::PathBuf::from);
        engine_config.dspark_sps_bias_ms = spec.dspark_sps_bias_ms;
        engine_config.dspark_sps_row_ms = spec.dspark_sps_row_ms;
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
    };

    Ok(ServeConfig { backend, options })
}

fn resolve_spec_options(backend: ServeBackend, serve_args: &ServeArgs) -> ServeSpecOptions {
    if !matches!(backend, ServeBackend::Metal | ServeBackend::Cuda) {
        return ServeSpecOptions::default();
    }
    let mut spec_type = match serve_args.spec_type {
        ServeSpecTypeArg::None => ServeSpecType::None,
        ServeSpecTypeArg::Auto => ServeSpecType::Auto,
        ServeSpecTypeArg::Mtp => ServeSpecType::Mtp,
        ServeSpecTypeArg::Dspark => ServeSpecType::Dspark,
    };
    if spec_type == ServeSpecType::None
        && (serve_args.mtp_draft_model.is_some()
            || serve_args.mtp_draft_tokens.is_some()
            || serve_args.mtp_draft_topk.is_some())
    {
        spec_type = ServeSpecType::Mtp;
    }
    ServeSpecOptions {
        spec_type,
        mtp_draft_model: serve_args.mtp_draft_model.clone(),
        dspark_sps_bias_ms: serve_args.dspark_sps_bias_ms,
        dspark_sps_row_ms: serve_args.dspark_sps_row_ms,
        mtp_draft_tokens: serve_args.mtp_draft_tokens,
        mtp_draft_topk: serve_args.mtp_draft_topk,
        dspark_block_size: serve_args.dspark_block_size,
        dspark_markov_init: serve_args.dspark_markov_init.clone(),
    }
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
        world_size: Some(serve_args.tensor_parallel_size * serve_args.context_parallel_size),
        context_parallel_size: Some(serve_args.context_parallel_size),
        ..EngineLoadConfig::default()
    };

    // Quant KV dtypes route to the backend that implements them. Reject the
    // mismatched combos at the CLI boundary; the per-backend `*KvCacheDtype::
    // resolve` is the fine authority for "supported vs deferred" on the
    // matching backend (CUDA tq4 flows through to it and fails loud there
    // until a TQ paged-prefill path exists).
    match config.kv_cache_dtype {
        // INT8 runs on Metal (int8 cache) and CUDA (paged quant pool, #68 T3).
        KvCacheDtype::Int8 if backend != ServeBackend::Metal && backend != ServeBackend::Cuda => {
            return Err(format!(
                "--kv-cache-dtype int8 is currently implemented for the Metal and CUDA backends; active backend is {}",
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
        config.max_running_requests = Some(1);
        config.chunked_prefill_size = Some(config.chunked_prefill_size.map_or(32, |v| v.min(32)));
    }

    if let Some(value) = serve_args.max_running_requests {
        config.max_running_requests = Some(value);
    }
    if let Some(value) = serve_args.max_prompt_tokens {
        config.max_prompt_tokens = value;
    }
    if let Some(value) = serve_args.max_total_tokens {
        config.max_total_tokens = value;
    }
    config.max_thinking_tokens = serve_args.max_thinking_tokens;
    config.mem_fraction_static = serve_args.mem_fraction_static;
    config.kv_dram = serve_args.kv_dram;
    if let Some(value) = serve_args.chunked_prefill_size {
        config.chunked_prefill_size = Some(value);
    }
    if let Some(value) = serve_args.max_num_batched_tokens {
        config.max_num_batched_tokens = Some(value);
    }
    config.slot_oversubscription = serve_args.kv_oversubscription;
    if let Some(value) = serve_args.kv_oversubscription_min_slice {
        config.oversubscription_min_slice = value;
    }
    config.memory_budget_bytes = serve_args.memory_budget_bytes;
    config.system_reserve_bytes = serve_args.system_reserve_bytes;
    config.allow_swap = serve_args.allow_swap;
    // Backend runtime toggles ride the engine config (multiproc worker ranks
    // see only ARLE_WORKER_ENGINE_CONFIG, so flags must live here, not env).
    config.cuda = serve_args.cuda_runtime_flags();
    config.metal = serve_args.metal_runtime_flags();
    config.diffusion_max_denoising_steps = serve_args.diffusion_max_denoising_steps;
    config.vulkan_submit_cap = serve_args.vulkan_submit_cap;

    // A user-supplied --max-prompt-tokens above the total is a genuine
    // contradiction and stays a hard error. The built-in default cap, however,
    // must never trip on its own: `--max-total-tokens N` alone (with the prompt
    // cap left at its default) clamps the prompt cap down to N instead of
    // erroring — the API scheduler clamps identically (infer-api loaded.rs).
    if config.max_prompt_tokens > config.max_total_tokens {
        if serve_args.max_prompt_tokens.is_some() {
            return Err(format!(
                "--max-prompt-tokens ({}) must be <= --max-total-tokens ({})",
                config.max_prompt_tokens, config.max_total_tokens
            ));
        }
        config.max_prompt_tokens = config.max_total_tokens;
    }

    Ok(config)
}

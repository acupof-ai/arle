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

pub(crate) fn run_serve(args: &Args, serve_args: ServeArgs) -> ExitCode {
    // Lower the probe flags into the env the CUDA executor reads at first use.
    // MUST run pre-spawn: multiproc TP rank children inherit the parent env
    // (env is the transport; flags are the only public interface).
    apply_probe_env(&serve_args);
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

/// Lower `--probe-out` / `--probe-lens-layers` / `--probe-token-entropy` into
/// the `ARLE_PROBE_*` env the `infer-cuda` probe reads once (`OnceLock`). The
/// path is absolutized so multiproc rank children resolve the same file.
///
/// SAFETY: single CLI thread, pre-spawn, pre-tokio (same contract as the
/// env pre-spawn).
fn apply_probe_env(serve_args: &ServeArgs) {
    let Some(path) = serve_args.probe_out.as_ref() else {
        return;
    };
    let abs = if path.is_absolute() {
        path.clone()
    } else {
        env::current_dir()
            .map(|dir| dir.join(path))
            .unwrap_or_else(|_| path.clone())
    };
    // SAFETY: single CLI thread, pre-spawn, pre-tokio (children inherit the
    // env pre-spawn).
    unsafe {
        std::env::set_var("ARLE_PROBE_JSONL", &abs);
        std::env::set_var(
            "ARLE_PROBE_LENS_LAYERS",
            serve_args.probe_lens_layers.to_string(),
        );
        std::env::set_var(
            "ARLE_PROBE_TOKEN_ENTROPY",
            if serve_args.probe_token_entropy {
                "1"
            } else {
                "0"
            },
        );
    }
    eprintln!(
        "[probe] lens_layers={} token_entropy={} → {}",
        serve_args.probe_lens_layers,
        serve_args.probe_token_entropy,
        abs.display()
    );
}

fn run_config(config: ServeConfig) -> ExitCode {
    // The DSpark train sidecar + `--dspark-markov-init` are wired only on the
    // single-process serve path (`on_engine_loaded` below); the multiproc
    // coordinator returns before that hook exists. Fail fast rather than serve
    // with the sidecar silently inert — a no-op flag must reject, not no-op.
    // Gate on the RESOLVED world size, not just model kind: a multiproc-capable
    // model (Qwen3.5/3.6, DSv4) at INFER_TP_SIZE=1 serves single-process, where
    // the sidecar DOES run — rejecting that would block DSpark test-time training
    // on the one config that supports it.
    #[cfg(all(unix, feature = "cuda"))]
    if config.backend == ServeBackend::Cuda
        && config.options.spec.dspark_markov_init.is_some()
        && crate::serve_multiproc::world_size_from_env() > 1
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
    // and runs the thin coordinator HTTP loop. Single GPU returns `None` and falls
    // through to the byte-identical in-process path below; dense Qwen3 skips it.
    #[cfg(all(unix, feature = "cuda"))]
    if config.backend == ServeBackend::Cuda
        && infer_api::cuda_model_takes_multiproc_serve(&config.options.model_path)
    {
        match crate::serve_multiproc::bind_relay_and_spawn_workers(
            &config.options.model_path,
            &config.options.engine_config,
        ) {
            Ok(Some(coordinator)) => {
                let crate::serve_multiproc::MultiprocCoordinator { relay, guard } = coordinator;
                eprintln!(
                    "[ARLE serve] starting cuda multiproc coordinator on {}:{}",
                    config.options.bind, config.options.port,
                );
                let result = infer_api::serve_coordinator_http(
                    &config.options.model_path,
                    &config.options.bind,
                    config.options.port,
                    config.options.engine_config.max_thinking_tokens,
                    relay,
                );
                // Keep the worker children alive until the serve loop returns.
                drop(guard);
                return match result {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(err) => {
                        eprintln!("[ARLE serve] coordinator error: {err:#}");
                        ExitCode::FAILURE
                    }
                };
            }
            // None: single GPU — fall through to the in-process serve below.
            Ok(None) => {}
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

    config.kv_recall = serve_args.kv_recall;

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
    config.slot_oversubscription = serve_args.kv_oversubscription;
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
    fn max_total_tokens_alone_clamps_default_prompt_cap() {
        if skip_if_no_backend() {
            return;
        }
        // Regression: `--max-total-tokens N` below the built-in prompt-cap
        // default (e.g. 32768) used to trip `--max-prompt-tokens must be <=
        // --max-total-tokens`. The default cap must clamp down silently instead.
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--max-total-tokens",
            "4096",
        ]);
        let config =
            resolve_config(&args, &serve).expect("must not error on lone --max-total-tokens");
        assert_eq!(config.options.engine_config.max_total_tokens, 4096);
        assert!(
            config.options.engine_config.max_prompt_tokens
                <= config.options.engine_config.max_total_tokens,
            "prompt cap {} must be clamped to <= total cap {}",
            config.options.engine_config.max_prompt_tokens,
            config.options.engine_config.max_total_tokens,
        );
    }

    #[test]
    fn explicit_prompt_cap_above_total_still_errors() {
        if skip_if_no_backend() {
            return;
        }
        // A user-supplied prompt cap above the total is a genuine contradiction
        // and must stay a hard error (only the default cap is auto-clamped).
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--max-prompt-tokens",
            "8000",
            "--max-total-tokens",
            "4096",
        ]);
        let err = resolve_config(&args, &serve).expect_err("explicit contradiction must error");
        assert!(
            err.contains("must be <= --max-total-tokens"),
            "unexpected error: {err}"
        );
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
        assert_eq!(
            config.options.engine_config.num_slots, defaults.num_slots,
            "low-impact caps via max_running_requests, not num_slots"
        );
        assert_eq!(config.options.engine_config.max_running_requests, Some(1));
        assert_eq!(
            config.options.engine_config.total_pages,
            defaults.total_pages
        );
        // The unbounded default prompt cap is clamped down to the default
        // total-token budget (see resolve_engine_config); low-impact leaves
        // that baseline clamp untouched, it does not shrink it further.
        assert_eq!(
            config.options.engine_config.max_prompt_tokens,
            defaults.max_total_tokens
        );
        assert_eq!(
            config.options.engine_config.max_total_tokens,
            defaults.max_total_tokens
        );
        assert_eq!(config.options.engine_config.chunked_prefill_size, Some(32));
        assert!(config.options.engine_config.low_impact);
    }

    #[test]
    fn kv_disk_flags_lower_into_engine_config() {
        if skip_if_no_backend() {
            return;
        }
        // Multiproc contract: ONLY engine_config reaches worker ranks
        // (ARLE_WORKER_ENGINE_CONFIG), so the L3 spill request must ride it.
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            compiled_backend_flag(),
            "--model-path",
            "model",
            "--kv-disk",
            "/tmp/arle-kv",
            "--kv-disk-limit",
            "1GiB",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(
            config.options.engine_config.kv_ssd_root.as_deref(),
            Some(std::path::Path::new("/tmp/arle-kv"))
        );
        assert_eq!(
            config.options.engine_config.kv_disk_limit,
            Some(infer_api::KvTierBudget::Bytes(1 << 30))
        );
    }

    #[test]
    fn bare_kv_disk_flag_uses_default_root() {
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
            "--kv-disk",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(
            config.options.engine_config.kv_ssd_root,
            Some(infer_api::default_kv_ssd_root())
        );
        assert_eq!(config.options.engine_config.kv_disk_limit, None);
    }

    #[test]
    fn kv_disk_limit_without_kv_disk_errors() {
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
            "--kv-disk-limit",
            "1GiB",
        ]);
        let err = resolve_config(&args, &serve).expect_err("limit without root must error");
        assert!(
            err.contains("requires --kv-disk"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn kv_dram_zero_lowers_to_off() {
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
            "--kv-dram",
            "0",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(
            config.options.engine_config.kv_dram,
            infer_api::KvTierBudget::Off
        );
    }

    #[test]
    fn kv_oversubscription_lowers_into_engine_config() {
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
            "--kv-oversubscription",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert!(config.options.engine_config.slot_oversubscription);
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
    fn kv_cache_dtype_int8_flows_to_cuda_engine_config() {
        if skip_if_no_backend() || compiled_backend_flag() != "cuda" {
            return;
        }
        let (args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            "cuda",
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
    fn kv_cache_dtype_int8_rejects_unsupported_backend() {
        // INT8 runs on Metal and CUDA; every other backend is rejected at the
        // CLI boundary. Skips on the metal/cuda lanes (where int8 flows
        // through to the per-backend resolve instead).
        if skip_if_no_backend()
            || compiled_backend_flag() == "metal"
            || compiled_backend_flag() == "cuda"
        {
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
        let err = resolve_config(&args, &serve).expect_err("unsupported-backend int8 rejected");
        assert!(err.contains("--kv-cache-dtype int8"), "got: {err}");
    }

    #[test]
    fn kv_cache_dtype_fp8_rejects_non_cuda_backend() {
        // FP8/TQ4 are CUDA-only paged quant modes; on any non-CUDA backend the
        // CLI guard must reject early (symmetric to the int8 guard). Runs on
        // the metal/cpu lanes; skips on CUDA (where fp8 flows through to the
        // CUDA paged quant path, #68 T3).
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

    // The CLI leaves `num_slots` at its auto-ceiling default and carries the
    // cap in `max_running_requests`; infer-api's `hot_workspace_slots` then
    // makes a set cap the executor slot budget at load.
    #[test]
    fn max_running_requests_is_the_concurrency_knob() {
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
            "--max-running-requests",
            "2",
            "--max-prompt-tokens",
            "8192",
            "--max-total-tokens",
            "16384",
        ]);
        let config = resolve_config(&args, &serve).expect("resolve");
        assert_eq!(config.options.engine_config.num_slots, defaults.num_slots);
        assert_eq!(config.options.engine_config.max_running_requests, Some(2));
        assert_eq!(
            config.options.engine_config.max_running_requests.unwrap(),
            2,
            "serve exposes one concurrency knob"
        );
        assert_eq!(config.options.engine_config.max_prompt_tokens, 8192);
        assert_eq!(config.options.engine_config.max_total_tokens, 16_384);
    }

    #[test]
    fn hip_engine_budget_keeps_internal_paged_kv_defaults() {
        let defaults = EngineLoadConfig::default();
        let (_args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            "hip",
            "--model-path",
            "model.gguf",
            "--max-prompt-tokens",
            "16",
            "--max-total-tokens",
            "32",
        ]);
        let config = resolve_engine_config(ServeBackend::Hip, &serve).expect("HIP config resolves");
        // total_pages/page_size are no longer user-facing; the internal defaults
        // (the executor's profiling floor) are preserved untouched.
        assert_eq!(config.total_pages, defaults.total_pages);
        assert_eq!(config.page_size, defaults.page_size);
        assert_eq!(config.max_total_tokens, 32);
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
            "--unknown-backend-flag",
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
        // build the CLI accepts the flag, then infer-api fail-closes it for
        // non-DSv4 model kinds once the checkpoint type is known.
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
        assert_eq!(
            config.options.engine_config.mtp_draft_tokens,
            Some(DEFAULT_MTP_DRAFT_TOKENS)
        );
        assert_eq!(
            config.options.engine_config.mtp_draft_topk,
            Some(DEFAULT_MTP_DRAFT_TOPK)
        );
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
        let err = resolve_config(&args, &serve).expect_err("draft model needs --spec-type dspark");
        assert_eq!(
            err,
            "--mtp-draft-model is only consumed by --spec-type dspark on this serve stack"
        );
    }

    #[test]
    fn metal_speculative_flags_ride_engine_config() {
        // Explicit draft + depth + width map into MetalRuntimeFlags verbatim.
        let (_args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            "metal",
            "--model-path",
            "models/qwen36-27b",
            "--draft-model",
            "mlx-community/Qwen3.6-27B-MTP-4bit",
            "--speculative-tokens",
            "3",
            "--spec-accept-topk",
            "2",
        ]);
        let flags = serve.metal_runtime_flags();
        assert!(flags.speculative);
        assert_eq!(
            flags.draft_model.as_deref(),
            Some("mlx-community/Qwen3.6-27B-MTP-4bit")
        );
        assert_eq!(flags.speculative_tokens, Some(3));
        assert_eq!(flags.spec_accept_topk, 2);

        // --no-speculative clears the master switch (auto-resolve suppressed).
        let (_args, serve) = parse_serve(&[
            "arle",
            "serve",
            "--backend",
            "metal",
            "--model-path",
            "models/qwen36-27b",
            "--no-speculative",
        ]);
        let flags = serve.metal_runtime_flags();
        assert!(!flags.speculative);
        assert!(flags.draft_model.is_none());

        // No spec flags: auto-resolve path — no explicit draft, no forced
        // depth (the resolver's default applies), exact-greedy width 1.
        let (_args, serve) = parse_serve(&["arle", "serve", "--model-path", "models/qwen36-27b"]);
        let flags = serve.metal_runtime_flags();
        assert!(flags.speculative);
        assert!(flags.draft_model.is_none());
        assert_eq!(
            flags.speculative_tokens, None,
            "depth is not forced on the auto-resolve-only path"
        );
        assert_eq!(flags.spec_accept_topk, 1);
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

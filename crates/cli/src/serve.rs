//! `arle serve` — in-process OpenAI v1 serving entry.
//!
//! Builds the backend router and runs `axum::serve` inside this process via
//! [`infer_api::serve_http`]. There is no standalone serve binary to exec: the
//! rewrite ships only the `arle` binary, so serving is in-process. The requested
//! backend must match the one compiled into this binary
//! ([`CompiledBackend::detect`]); a mismatch is rejected up front rather than
//! silently serving the compiled backend.

use std::{env, process::ExitCode};

use infer_api::{EngineLoadConfig, ServeHttpOptions, serve_http};

use crate::{
    args::{Args, ServeArgs, ServeBackendArg, ServeSpecTypeArg},
    hardware::CompiledBackend,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ServeBackend {
    Cuda,
    Metal,
    Cpu,
}

impl ServeBackend {
    fn label(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
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

    // DSv4 + CUDA: become the multiproc-serve COORDINATOR (rank 0). Bind the
    // relay, spawn N-1 worker processes (one per GPU via INFER_CUDA_DEVICES /
    // INFER_TP_SIZE), accept their relay connects, boot-ping, install the
    // admission broadcaster — THEN fall through to the normal in-process serve
    // below (rank 0 owns HTTP). The broadcaster fans out each request the rank-0
    // engine admits to the workers so they step their executors' NCCL `forward`
    // in lockstep. The guard holds the relay + worker pipes for the serve loop's
    // lifetime; dropping it on return EOFs the workers so they exit. On a single
    // GPU it is a no-op and serve proceeds single-process. Non-DSv4 models skip
    // this entirely.
    //
    // `// STAGE 3:` chunked-prefill scratch chunk-bounding + the decode path
    // beyond a single short prompt are later stages (see Stage-2 markers in
    // `serve_multiproc.rs` / `multiproc_relay.rs` / `execution.rs`).
    #[cfg(all(unix, feature = "cuda"))]
    let _coordinator_guard = if config.backend == ServeBackend::Cuda
        && crate::serve_multiproc::is_dsv4_model(&config.options.model_path)
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

    // Speculative / MTP routing is Metal-only on the rewrite serve stack today.
    // The CLI rejects it elsewhere so the error surface matches the old front
    // door; the router itself does not yet thread these through (follow-up).
    if serve_args.spec_type != ServeSpecTypeArg::None && backend != ServeBackend::Metal {
        return Err("--spec-type is currently only supported by the Metal backend".to_string());
    }
    if serve_args.mtp_draft_model.is_some() && backend != ServeBackend::Metal {
        return Err(
            "--mtp-draft-model is currently only supported by the Metal backend".to_string(),
        );
    }
    if serve_args.mtp_draft_tokens.is_some() && backend != ServeBackend::Metal {
        return Err(
            "--mtp-draft-tokens is currently only supported by the Metal backend".to_string(),
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

    let options = ServeHttpOptions {
        model_path,
        bind: serve_args.bind.clone(),
        port: serve_args.port,
        // `--no-cuda-graph` flips the CUDA decode-graph default off; honored by
        // the CUDA backend only (Metal/CPU ignore it).
        enable_cuda_graph: !args.no_cuda_graph,
        engine_config: EngineLoadConfig::default(),
    };

    Ok(ServeConfig {
        backend,
        options,
        bind_warning,
    })
}

fn resolve_backend(arg: ServeBackendArg) -> Result<ServeBackend, String> {
    let requested = match arg {
        ServeBackendArg::Cuda => Some(ServeBackend::Cuda),
        ServeBackendArg::Metal => Some(ServeBackend::Metal),
        ServeBackendArg::Cpu => Some(ServeBackend::Cpu),
        ServeBackendArg::Auto => None,
    };

    let compiled = match CompiledBackend::detect() {
        CompiledBackend::Cuda => Some(ServeBackend::Cuda),
        CompiledBackend::Metal => Some(ServeBackend::Metal),
        CompiledBackend::Cpu => Some(ServeBackend::Cpu),
        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
        CompiledBackend::None => None,
    };

    let Some(compiled) = compiled else {
        return Err(
            "serve requires a backend build; rebuild with cuda, metal/no-cuda, or cpu/no-cuda"
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
            CompiledBackend::Cpu => "cpu",
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
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
    fn explicit_backend_mismatch_is_rejected() {
        // Pick a backend that is NOT the compiled one (or skip if no backend is
        // compiled in, where every explicit backend is rejected anyway).
        let other = match CompiledBackend::detect() {
            CompiledBackend::Metal => "cuda",
            CompiledBackend::Cuda | CompiledBackend::Cpu => "metal",
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
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
    fn non_metal_spec_type_errors_when_compiled_non_metal() {
        if skip_if_no_backend() {
            return;
        }
        // Only meaningful when the compiled backend is not Metal; on a Metal
        // build `--spec-type mtp` is accepted, so skip there.
        if CompiledBackend::detect() == CompiledBackend::Metal {
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
        let err = resolve_config(&args, &serve).expect_err("reject non-Metal spec type");
        assert_eq!(
            err,
            "--spec-type is currently only supported by the Metal backend"
        );
    }

    #[test]
    fn metal_spec_type_accepted_when_compiled_metal() {
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
        let config = resolve_config(&args, &serve).expect("metal accepts spec type");
        assert_eq!(config.backend, ServeBackend::Metal);
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

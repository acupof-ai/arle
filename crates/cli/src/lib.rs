// CLI crate: unsafe blocks are FFI calls (signal handlers, process spawning,
// ioctl) with invariants documented at the call site; crate-level allow avoids
// repetitive SAFETY comments on well-understood libc invocations.
#![allow(clippy::undocumented_unsafe_blocks)]

mod args;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod banner;
mod doctor;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod download;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod eli;
mod hardware;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod hf_search;
mod hub_discovery;
mod model_catalog;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod model_picker;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod modelscope;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod ocr;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod repl;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod runtime_report;
mod serve;
#[cfg(all(unix, feature = "cuda"))]
mod serve_multiproc;
#[cfg(feature = "cuda")]
mod spec_train_target;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod startup;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod tps;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod trace;
mod train_cli;
#[cfg(all(unix, feature = "cuda"))]
mod train_multiproc;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod welcome;

use std::process::ExitCode;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::time::Instant;

use anyhow::Result;
use args::{Args, CliCommand, RunArgs};
use clap::Parser;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer_api::{InferenceEngine, LoadedInferenceEngine};

/// Serializes unit tests that mutate process-global state (env vars, override
/// registries): the default test harness runs threads in ONE process, so an
/// unguarded `set_var`/`remove_var` races every other test touching the same
/// global. Every such test takes this lock first. Poison-tolerant so one
/// failing test doesn't cascade into every later env test.
#[cfg(all(test, any(feature = "cuda", feature = "metal", feature = "cpu")))]
#[allow(dead_code)]
pub(crate) fn test_env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub fn run() -> ExitCode {
    // Pre-CUDA sandbox-spawner helper entry. When `ARLE_SPAWNER_LISTEN` is set
    // (only by `SpawnerHandle::launch`, which re-exec's THIS binary before any
    // CUDA/thread init), this process is the non-CUDA spawn helper: run the
    // request/response loop and exit. MUST be the very first thing in `run` —
    // before the logger, clap, or any thread spawns — so the helper stays a plain
    // single-threaded non-CUDA process that ELKEID never SIGABRTs on `fork()`.
    if std::env::var(train::spawner::LISTEN_ENV).is_ok() {
        return ExitCode::from(train::spawner::serve_loop() as u8);
    }

    // DSv4 multiproc-serve worker entry. When ARLE_WORKER_RANK>0 is set (by the
    // serve coordinator before spawning this process), this process is a worker
    // rank: short-circuit BEFORE clap parsing — it joins the NCCL group as its
    // rank and runs the relay-receiver loop instead of the normal CLI. Mirrors
    // the deleted `infer/src/main.rs` fn main worker-entry. No-op on non-CUDA /
    // non-Unix builds (the module isn't compiled there).
    // MUST run before the generic logger init below: workers install their own
    // `[rankN]`-prefixed logger, and `init` is call_once — a generic init first
    // silently drops the prefix.
    #[cfg(all(unix, feature = "cuda"))]
    if let Some(code) = serve_multiproc::worker_entry() {
        return code;
    }

    // Install the logger before any subcommand dispatch. `serve`, `train`,
    // and `model` early-return from the match below and never reach `run_impl`'s
    // init, so without this they would run with NO logger — every `log::error!`
    // in the serve/engine path (e.g. an engine-thread `step()` failure) became a
    // silent no-op, masking real failures. RUST_LOG overrides. Serve defaults
    // to `info`: the operator-facing startup truth (resolved KV tiers, the
    // checkpoint-scaled engine-ready barrier) is logged at info, and the
    // multiproc coordinator was silently dropping it at `warn`.
    // Pre-clap sniff — this must run before `Args::parse` (workers/subcommands
    // early-return) and `init` is call_once, so parse-then-init is not an option.
    // Mesh worker's prefixed logger must beat the call_once init below, which
    // would otherwise win and drop the rank prefix.
    #[cfg(all(unix, feature = "cuda"))]
    let mesh_worker_logged = train_multiproc::install_mesh_worker_logger();
    #[cfg(not(all(unix, feature = "cuda")))]
    let mesh_worker_logged = false;

    let level = if std::env::args().any(|arg| arg == "serve") {
        "info"
    } else {
        "warn"
    };
    if !mesh_worker_logged {
        infer_util::logging::init_stderr(level);
    }

    let mut args = Args::parse();
    if args.kernel_build_id {
        println!("{}", infer_api::kernel_build_id());
        println!("capabilities:{}", infer_api::kernel_capabilities());
        return ExitCode::SUCCESS;
    }
    let command = args.command.take();
    #[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
    let _exit_report = if should_print_exit_report(&args, command.as_ref()) {
        Some(runtime_report::ExitResourceReport::enabled())
    } else {
        None
    };

    match command {
        Some(CliCommand::Train(command)) => return train_cli::run_train(*command),
        #[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
        Some(CliCommand::Model(command)) => return train_cli::run_model(*command),
        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
        Some(CliCommand::Model(_)) => {
            eprintln!("[ARLE] error: model download requires cuda/metal/cpu feature build");
            return ExitCode::FAILURE;
        }
        Some(CliCommand::Serve(command)) => return serve::run_serve(&args, *command),
        #[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
        Some(CliCommand::Ocr(ocr_args)) => match ocr::run(&ocr_args) {
            Ok(()) => return ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("[ARLE] error: {err:#}");
                return ExitCode::FAILURE;
            }
        },
        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
        Some(CliCommand::Ocr(_)) => {
            eprintln!("[ARLE] error: `arle ocr` requires a metal/cuda/cpu backend build");
            return ExitCode::FAILURE;
        }
        Some(CliCommand::Run(run_args)) => match run_impl(args, Some(*run_args)) {
            Ok(()) => return ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("[ARLE] error: {err:#}");
                return ExitCode::FAILURE;
            }
        },
        None => {}
    }

    match run_impl(args, None) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[ARLE] error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn should_print_exit_report(args: &Args, command: Option<&CliCommand>) -> bool {
    !args.doctor && !args.list_models && matches!(command, None | Some(CliCommand::Run(_)))
}

fn run_impl(args: Args, run_args: Option<RunArgs>) -> Result<()> {
    #[cfg(all(not(feature = "cuda"), not(feature = "metal"), not(feature = "cpu")))]
    let _ = &run_args;

    if args.doctor {
        doctor::run(&args)?;
        return Ok(());
    }

    if args.list_models {
        doctor::list_models(&args)?;
        return Ok(());
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "metal"), not(feature = "cpu")))]
    {
        anyhow::bail!(
            "ARLE requires a local inference backend. Rebuild with either \
             the default `cuda` feature, `--no-default-features --features metal,no-cuda`, \
             or `--no-default-features --features cpu,no-cuda`."
        );
    }

    #[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
    {
        use std::io::IsTerminal;

        // Logger already installed at the top of `run()` (quiet "warn" default;
        // RUST_LOG overrides). No re-init here.

        let one_shot = run_args
            .as_ref()
            .is_some_and(|r| r.prompt.is_some() || r.stdin || !r.image.is_empty());
        let interactive_tty = !args.non_interactive
            && std::io::stdin().is_terminal()
            && std::io::stderr().is_terminal();
        if !one_shot
            && interactive_tty
            && let Some(action) = decide_eli_launch(&args)
        {
            return run_eli_frontend(&args, action);
        }

        let model_source = match startup::resolve_model_interactive(&args) {
            Ok(src) => src,
            Err(err) => {
                let can_wizard = !args.non_interactive
                    && std::io::stdin().is_terminal()
                    && std::io::stderr().is_terminal();
                if can_wizard {
                    match startup::run_hub_wizard()? {
                        Some(path) => path,
                        None => {
                            eprintln!(
                                "No model selected. Pass --model-path <dir-or-hf-id> \
                                 or rerun to use the model picker."
                            );
                            return Err(err);
                        }
                    }
                } else {
                    return Err(err);
                }
            }
        };

        log::info!("Loading model from: {}", model_source);
        let load_start = Instant::now();
        let mut engine = match LoadedInferenceEngine::load(&model_source) {
            Ok(e) => e,
            Err(err) => {
                // Detect the specific case where a user pointed at a DFlash
                // *draft* model — these have no tokenizer and load fails with
                // "tokenizer.json not found", which is opaque. The picker
                // filters them out as of 0.1.5, but `--model-path` can still
                // hit one directly.
                if let Some(arch) = peek_model_architecture(&model_source)
                    && arch == "DFlashDraftModel"
                {
                    return Err(anyhow::anyhow!(
                        "`{model_source}` is a DFlash *draft* model (architecture `DFlashDraftModel`), \
                         not a standalone target. Drafts ship without a tokenizer and only assist \
                         speculative decoding for a paired target.\n\
                         Hint: load the matching target instead — e.g. `mlx-community/Qwen3.6-35B-A3B-4bit` \
                         for the `z-lab/Qwen3.6-35B-A3B-DFlash` draft."
                    ));
                }
                return Err(anyhow::anyhow!(
                    "failed to load model from `{model_source}`: {err:#}\n\
                     Hint: verify --model-path points to a model directory with config.json.\n\
                     Hint: direct Metal smoke: `arle serve --backend metal --model-path <path>`."
                ));
            }
        };
        let backend_name = engine.backend_name().to_string();

        let load_secs = load_start.elapsed().as_secs_f64();
        banner::print_model_loaded(engine.model_id(), &backend_name, load_secs);

        if !args.non_interactive
            && std::io::stdin().is_terminal()
            && std::io::stderr().is_terminal()
        {
            welcome::print_welcome_banner(engine.model_id());
        }

        let max_tokens = resolve_max_tokens(&model_source, args.max_tokens);

        // Failures here ARE surfaced to the user — we want them to know the
        // path was unwritable before the agent loop quietly drops every record.
        let trace_writer = match args.trace.as_ref() {
            Some(path) => match trace::TraceWriter::open(path, args.trace_prompts.keep_prompts()) {
                Ok(writer) => {
                    log::info!(
                        "trajectory: writing JSONL to {} (trace_prompts={})",
                        writer.path().display(),
                        if args.trace_prompts.keep_prompts() {
                            "on"
                        } else {
                            "off"
                        }
                    );
                    Some(writer)
                }
                Err(err) => {
                    return Err(anyhow::anyhow!(
                        "failed to open --trace path `{}`: {err:#}",
                        path.display()
                    ));
                }
            },
            None => None,
        };

        match run_args {
            Some(run_args) if run_args.prompt.is_some() || run_args.stdin => {
                let tools_enabled = !(args.no_tools || run_args.no_tools);
                repl::run_one_shot(
                    &mut engine,
                    &backend_name,
                    args.max_turns,
                    max_tokens,
                    args.temperature,
                    &run_args,
                    tools_enabled,
                    trace_writer.as_ref(),
                )?
            }
            Some(run_args) if !run_args.image.is_empty() => {
                anyhow::bail!(
                    "run --image requires --prompt or --stdin; in REPL use /image <path-or-url>"
                )
            }
            Some(run_args) => repl::run_repl(
                &mut engine,
                &backend_name,
                args.max_turns,
                max_tokens,
                args.temperature,
                !(args.no_tools || run_args.no_tools),
                trace_writer.as_ref(),
            )?,
            None => repl::run_repl(
                &mut engine,
                &backend_name,
                args.max_turns,
                max_tokens,
                args.temperature,
                !args.no_tools,
                trace_writer.as_ref(),
            )?,
        }

        Ok(())
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
enum EliLaunch {
    /// A saved Eli default exists: serve this model and launch Eli directly,
    /// skipping the model picker.
    Saved { model: String, mode: eli::EliMode },
    /// Launch Eli but resolve the model interactively first (picker), then
    /// persist the choice so subsequent runs take the `Saved` path.
    Pick { mode: eli::EliMode },
}

/// Decide whether (and how) to hand this interactive start to Eli.
///
/// Precedence:
/// 1. `--agent arle` → `None` (built-in REPL, explicit opt-out).
/// 2. `--agent eli` → `Pick` (force Eli; picker selects the model).
/// 3. no flag + saved `agent.toml` selects Eli → `Saved` (launch directly).
/// 4. no flag + no saved Eli default → `Pick` (Eli is the default front-end;
///    the first run picks a model then persists Eli as the default).
///
/// Non-explicit paths (3, 4) fall back to the built-in REPL (`None`) when the
/// Eli binary isn't installed — arle stays usable standalone (Eli is an
/// optional runtime dependency, discovered, never a build dep).
/// `--gateway` overrides the persisted/derived mode when present.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn decide_eli_launch(args: &Args) -> Option<EliLaunch> {
    use args::AgentFrontendArg;

    if matches!(args.agent, Some(AgentFrontendArg::Arle)) {
        return None;
    }

    let saved = eli::AgentConfig::load();
    let flag_mode = if args.gateway {
        Some(eli::EliMode::Gateway)
    } else {
        None
    };

    // Forced Eli via flag → always pick a model fresh (errors later if Eli is
    // not installed — the user asked for it explicitly).
    if matches!(args.agent, Some(AgentFrontendArg::Eli)) {
        return Some(EliLaunch::Pick {
            mode: flag_mode.unwrap_or(eli::EliMode::Chat),
        });
    }

    // No explicit `--agent` below this point. Eli is the default front-end, but
    // ONLY when it's actually installed: a missing Eli falls back to the
    // built-in REPL (graceful, no error). This is the optional-runtime-dependency
    // contract — arle works standalone; Eli is an opt-in enhancement.
    if eli::find_eli_binary().is_err() {
        return None;
    }

    if saved.selects_eli() {
        let model = saved.model.clone().unwrap_or_default();
        let mode = flag_mode.unwrap_or_else(|| saved.launch_mode());
        return Some(EliLaunch::Saved { model, mode });
    }

    Some(EliLaunch::Pick {
        mode: flag_mode.unwrap_or(eli::EliMode::Chat),
    })
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_eli_frontend(args: &Args, action: EliLaunch) -> Result<()> {
    // `auto` resolves to the single compiled backend, which matches this
    // interactive build.
    let backend = "auto";

    let (model, mode, persist) = match action {
        EliLaunch::Saved { model, mode } => (model, mode, false),
        EliLaunch::Pick { mode } => {
            let model = match startup::resolve_model_interactive(args) {
                Ok(src) => src,
                Err(err) => match startup::run_hub_wizard()? {
                    Some(path) => path,
                    None => {
                        eprintln!(
                            "No model selected. Pass --model-path <dir-or-hf-id> \
                             or rerun to use the model picker."
                        );
                        return Err(err);
                    }
                },
            };
            (model, mode, true)
        }
    };

    if persist {
        let cfg = eli::AgentConfig {
            agent: Some("eli".to_string()),
            model: Some(model.clone()),
            mode: Some(mode.config_value().to_string()),
        };
        // A persistence failure must not block the launch; warn and continue.
        if let Err(err) = cfg.save() {
            log::warn!("could not persist arle agent config: {err:#}");
        }
    }

    eli::launch_eli(&model, mode, backend)
}

/// Resolve the per-turn token cap. `requested == 0` means "auto" — read
/// `max_position_embeddings` (or `context_length` for GGUF) from the
/// model's config.json and use that. Falls back to 256K (262144) only
/// when the config truly can't be read, and logs which path won so
/// users can verify.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn resolve_max_tokens(model_path: &str, requested: usize) -> usize {
    const FALLBACK: usize = 262_144;
    if requested > 0 {
        return requested;
    }
    if let Some(n) = read_model_generation_max_tokens(model_path) {
        log::info!(
            "max-tokens: auto-resolved to {n} from {model_path}/config.json (generation_config.max_new_tokens)"
        );
        return n;
    }
    match read_model_max_context(model_path) {
        Some(n) => {
            log::info!(
                "max-tokens: auto-resolved to {n} from {model_path}/config.json (max_position_embeddings)"
            );
            n
        }
        None => {
            log::info!(
                "max-tokens: auto-resolution failed for {model_path}; falling back to {FALLBACK}"
            );
            FALLBACK
        }
    }
}

/// Accepts a local directory or an HF repo id (`org/repo`); for the latter
/// tries every cached snapshot until one yields an answer. `None` on any
/// failure — the caller falls back to the generic error path.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn peek_model_architecture(model_source: &str) -> Option<String> {
    if let Some(arch) = read_arch_from_dir(std::path::Path::new(model_source)) {
        return Some(arch);
    }

    let (org, repo) = model_source.split_once('/')?;
    let cache_root = hub_discovery::hub_cache_root()?;
    let repo_dir = cache_root.join(format!("models--{org}--{repo}"));
    let snapshots = std::fs::read_dir(repo_dir.join("snapshots")).ok()?;
    for entry in snapshots.flatten() {
        if let Some(arch) = read_arch_from_dir(&entry.path()) {
            return Some(arch);
        }
    }
    None
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn read_arch_from_dir(dir: &std::path::Path) -> Option<String> {
    let cfg = read_model_config_from_dir(dir)?;
    cfg.get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|s| s.as_str())
        .map(str::to_string)
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn read_model_config(model_path: &str) -> Option<serde_json::Value> {
    if let Some(cfg) = read_model_config_from_dir(std::path::Path::new(model_path)) {
        return Some(cfg);
    }

    let (org, repo) = model_path.split_once('/')?;
    let cache_root = hub_discovery::hub_cache_root()?;
    let repo_dir = cache_root.join(format!("models--{org}--{repo}"));
    let snapshots = std::fs::read_dir(repo_dir.join("snapshots")).ok()?;
    for entry in snapshots.flatten() {
        if let Some(cfg) = read_model_config_from_dir(&entry.path()) {
            return Some(cfg);
        }
    }
    None
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn read_model_config_from_dir(dir: &std::path::Path) -> Option<serde_json::Value> {
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    serde_json::from_str(&raw).ok()
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn read_model_generation_max_tokens(model_path: &str) -> Option<usize> {
    let cfg = read_model_config(model_path)?;
    let model_type = cfg.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    let arch_is_diffusion = cfg
        .get("architectures")
        .and_then(|a| a.as_array())
        .is_some_and(|archs| {
            archs
                .iter()
                .filter_map(|v| v.as_str())
                .any(|name| name.contains("DiffusionGemma") || name.contains("Gemma4"))
        });
    if !(matches!(model_type, "diffusion_gemma" | "gemma4") || arch_is_diffusion) {
        return None;
    }
    cfg.get("generation_config")
        .and_then(|v| v.get("max_new_tokens"))
        .and_then(|v| v.as_u64())
        .or_else(|| cfg.get("canvas_length").and_then(|v| v.as_u64()))
        .map(|n| n as usize)
        .filter(|&n| n > 0)
}

/// Tries `max_position_embeddings` (HF transformers convention) then
/// `context_length` (GGUF / llama.cpp convention); `None` on any failure.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn read_model_max_context(model_path: &str) -> Option<usize> {
    let cfg = read_model_config(model_path)?;
    cfg.get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .or_else(|| cfg.get("context_length").and_then(|v| v.as_u64()))
        .map(|n| n as usize)
        .filter(|&n| n > 0)
}

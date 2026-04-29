mod args;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod banner;
mod doctor;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod download;
mod hardware;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod hf_search;
mod hub_discovery;
mod model_catalog;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod model_picker;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod repl;
mod serve;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod startup;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod tps;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod trace;
mod train_cli;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod welcome;

use std::process::ExitCode;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::time::Instant;

use anyhow::Result;
use args::{Args, CliCommand, RunArgs};
use clap::Parser;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer::server_engine::{InferenceEngine, LoadedInferenceEngine};

pub fn run() -> ExitCode {
    let mut args = Args::parse();
    let command = args.command.take();

    match command {
        Some(CliCommand::Train(command)) => return train_cli::run_train(*command),
        Some(CliCommand::Data(command)) => return train_cli::run_data(*command),
        Some(CliCommand::Serve(command)) => return serve::run_serve(&args, *command),
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

        // Keep the interactive CLI quiet by default. Users can opt into
        // verbose internals by setting RUST_LOG explicitly.
        infer::logging::init_stderr("warn");

        // Interactive startup: hardware detection + model picker + download.
        // Falls back to resolve_model_source() when non-interactive.
        let model_source = match startup::resolve_model_interactive(&args) {
            Ok(src) => src,
            Err(err) => {
                // Main resolve path failed — if this is an interactive
                // terminal, offer the HF-cache discovery wizard before
                // giving up.
                let can_wizard = !args.non_interactive
                    && std::io::stdin().is_terminal()
                    && std::io::stderr().is_terminal();
                if can_wizard {
                    match startup::run_hub_wizard()? {
                        Some(path) => path,
                        None => {
                            eprintln!(
                                "No model selected. Pass --model-path or try \
                                 ./scripts/run_dflash.sh serve."
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
        let mut engine = match LoadedInferenceEngine::load(&model_source, !args.no_cuda_graph) {
            Ok(e) => e,
            Err(err) => {
                // Detect the specific case where a user pointed at a DFlash
                // *draft* model — these have no tokenizer and load fails with
                // "tokenizer.json not found", which is opaque. The picker
                // filters them out as of 0.1.5, but `--model-path` can still
                // hit one directly.
                if let Some(arch) = peek_model_architecture(&model_source) {
                    if arch == "DFlashDraftModel" {
                        return Err(anyhow::anyhow!(
                            "`{model_source}` is a DFlash *draft* model (architecture `DFlashDraftModel`), \
                             not a standalone target. Drafts ship without a tokenizer and only assist \
                             speculative decoding for a paired target.\n\
                             Hint: load the matching target instead — e.g. `mlx-community/Qwen3.6-35B-A3B-4bit` \
                             for the `z-lab/Qwen3.6-35B-A3B-DFlash` draft.\n\
                             Hint: for Apple Silicon DFlash speculative serving, see `./scripts/run_dflash.sh serve`."
                        ));
                    }
                }
                return Err(anyhow::anyhow!(
                    "failed to load model from `{model_source}`: {err:#}\n\
                     Hint: verify --model-path points to a model directory with config.json.\n\
                     Hint: for Apple Silicon, try `./scripts/run_dflash.sh serve`.\n\
                     Hint: direct Metal smoke: `cargo run --release -p infer --bin metal_bench -- --model <path>`."
                ));
            }
        };
        let backend_name = engine.backend_name().to_string();

        let load_secs = load_start.elapsed().as_secs_f64();
        banner::print_model_loaded(engine.model_id(), &backend_name, load_secs);

        // First-run welcome banner (interactive only). On subsequent runs
        // this degrades to a 1-line model reminder.
        if !args.non_interactive
            && std::io::stdin().is_terminal()
            && std::io::stderr().is_terminal()
        {
            welcome::print_welcome_banner(engine.model_id());
        }

        let max_tokens = resolve_max_tokens(&model_source, args.max_tokens);

        // Open the trajectory writer, if requested. Failures here ARE
        // surfaced to the user — we want them to know the path was
        // unwritable before the agent loop quietly drops every record.
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

/// Best-effort peek at the model's first declared `architectures` entry from
/// its `config.json`. Accepts either a local directory path or a HuggingFace
/// repo id (`org/repo`); for the latter we resolve through the HF hub cache
/// (`~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<hash>/`) and
/// try every snapshot dir until one yields an answer. Returns `None` on any
/// failure — the caller falls back to the generic error path.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn peek_model_architecture(model_source: &str) -> Option<String> {
    // 1. Local-path case: try `<source>/config.json` first.
    if let Some(arch) = read_arch_from_dir(std::path::Path::new(model_source)) {
        return Some(arch);
    }

    // 2. HuggingFace repo-id case: walk the hub cache for matching snapshots.
    let (org, repo) = model_source.split_once('/')?;
    let cache_root = hub_discovery::hub_cache_root()?;
    let repo_dir = cache_root.join(format!(
        "models--{}--{}",
        org,
        repo.replace('-', "--").replace('-', "-") // safe no-op; preserve as-is
    ));
    let repo_dir = if repo_dir.exists() {
        repo_dir
    } else {
        // Fall back to a literal join — HF caches names verbatim with `--`
        // for path separators only, hyphens are preserved as-is.
        cache_root.join(format!("models--{org}--{repo}"))
    };
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
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&raw).ok()?;
    cfg.get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|s| s.as_str())
        .map(str::to_string)
}

/// Best-effort lookup of the model's context length. Tries, in order,
/// `max_position_embeddings` (HF transformers convention) and
/// `context_length` (GGUF / llama.cpp convention) from `<model_path>/config.json`.
/// Returns `None` for any failure (missing path, bad JSON, missing field).
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn read_model_max_context(model_path: &str) -> Option<usize> {
    let cfg_path = std::path::Path::new(model_path).join("config.json");
    let raw = std::fs::read_to_string(&cfg_path).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&raw).ok()?;
    cfg.get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .or_else(|| cfg.get("context_length").and_then(|v| v.as_u64()))
        .map(|n| n as usize)
        .filter(|&n| n > 0)
}

#[cfg(test)]
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod resolve_tests {
    use super::{read_model_max_context, resolve_max_tokens};
    use std::io::Write;

    fn write_config(json: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut f = std::fs::File::create(dir.path().join("config.json")).expect("create");
        f.write_all(json.as_bytes()).expect("write");
        dir
    }

    #[test]
    fn explicit_max_tokens_wins_over_auto() {
        let dir = write_config(r#"{"max_position_embeddings": 32768}"#);
        // requested == 4096 (non-zero) → ignore config, use 4096.
        assert_eq!(resolve_max_tokens(dir.path().to_str().unwrap(), 4096), 4096);
    }

    #[test]
    fn auto_pulls_max_position_embeddings_from_config() {
        let dir = write_config(r#"{"max_position_embeddings": 262144, "other": "x"}"#);
        assert_eq!(resolve_max_tokens(dir.path().to_str().unwrap(), 0), 262_144);
    }

    #[test]
    fn auto_falls_back_to_context_length_for_gguf_style_config() {
        // Some configs (often GGUF-derived) only carry `context_length`.
        let dir = write_config(r#"{"context_length": 32768}"#);
        assert_eq!(resolve_max_tokens(dir.path().to_str().unwrap(), 0), 32768);
    }

    #[test]
    fn auto_falls_back_to_default_when_config_missing() {
        // /tmp/nonexistent → no config.json → 262144 fallback.
        let bogus = std::env::temp_dir().join(format!("arle-no-config-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&bogus);
        assert_eq!(resolve_max_tokens(bogus.to_str().unwrap(), 0), 262_144);
    }

    #[test]
    fn read_model_max_context_returns_none_on_zero_value() {
        // A 0 in the config is nonsense — treat as "absent" and let the
        // caller's fallback fire instead of pinning the cap to 0.
        let dir = write_config(r#"{"max_position_embeddings": 0}"#);
        assert!(read_model_max_context(dir.path().to_str().unwrap()).is_none());
    }
}

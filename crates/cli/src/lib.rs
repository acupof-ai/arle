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
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod startup;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod tps;
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

        match run_args {
            Some(run_args) if run_args.prompt.is_some() || run_args.stdin => repl::run_one_shot(
                &mut engine,
                &backend_name,
                args.max_turns,
                args.max_tokens,
                args.temperature,
                &run_args,
            )?,
            _ => repl::run_repl(
                &mut engine,
                &backend_name,
                args.max_turns,
                args.max_tokens,
                args.temperature,
            )?,
        }

        Ok(())
    }
}

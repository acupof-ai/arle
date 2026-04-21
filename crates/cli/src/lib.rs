mod args;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod banner;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod download;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod hardware;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod hf_search;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod hub_discovery;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod model_catalog;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod model_picker;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod repl;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod startup;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod tps;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod welcome;

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::time::Instant;

use anyhow::Result;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use args::Args;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use clap::Parser;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer::server_engine::{InferenceEngine, LoadedInferenceEngine};

pub fn run() -> Result<()> {
    #[cfg(all(not(feature = "cuda"), not(feature = "metal"), not(feature = "cpu")))]
    {
        anyhow::bail!(
            "agent-infer requires a local inference backend. Rebuild with either \
             the default `cuda` feature, `--no-default-features --features metal,no-cuda`, \
             or `--no-default-features --features cpu,no-cuda`."
        );
    }

    #[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
    {
        use std::io::IsTerminal;

        infer::logging::init_default();
        let args = Args::parse();

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
                eprintln!("{err:#}");
                eprintln!();
                eprintln!(
                    "Hint: verify --model-path points to a model directory with config.json,"
                );
                eprintln!("or try: ./scripts/run_dflash.sh serve  (Metal, Apple Silicon)");
                eprintln!(
                    "         cargo run --release -p infer --bin metal_bench -- --model <path>"
                );
                return Err(err);
            }
        };
        let backend_name = engine.backend_name().to_string();

        let load_secs = load_start.elapsed().as_secs_f64();
        banner::print_model_loaded(engine.model_id(), &backend_name, load_secs);

        // First-run welcome banner (interactive only). On subsequent runs
        // this degrades to a 1-line model+mode reminder.
        if !args.non_interactive
            && std::io::stdin().is_terminal()
            && std::io::stderr().is_terminal()
        {
            let mode_label = if args.tools { "agent" } else { "chat" };
            welcome::print_welcome_banner(engine.model_id(), mode_label);
        }

        repl::run_repl(
            &mut engine,
            &backend_name,
            args.max_turns,
            args.max_tokens,
            args.temperature,
            args.tools,
        )?;

        Ok(())
    }
}

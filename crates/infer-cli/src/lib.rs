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
mod model_catalog;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod model_picker;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod repl;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod startup;

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
        infer::logging::init_default();
        let args = Args::parse();

        // Interactive startup: hardware detection + model picker + download.
        // Falls back to resolve_model_source() when non-interactive.
        let model_source = startup::resolve_model_interactive(&args)?;

        log::info!("Loading model from: {}", model_source);
        let load_start = Instant::now();
        let mut engine = LoadedInferenceEngine::load(&model_source, !args.no_cuda_graph)?;
        let backend_name = engine.backend_name().to_string();

        if let Some(max_kv) = args.max_gpu_kv {
            engine.set_max_gpu_kv(max_kv);
        }

        let load_secs = load_start.elapsed().as_secs_f64();
        banner::print_model_loaded(engine.model_id(), &backend_name, load_secs);

        repl::run_repl(
            &mut engine,
            &backend_name,
            args.max_turns,
            args.max_tokens,
            args.temperature,
        )?;

        Ok(())
    }
}

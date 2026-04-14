mod args;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod repl;

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::time::Instant;

use anyhow::Result;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use args::Args;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use clap::Parser;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer::hf_hub::resolve_model_source;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer::logging::init_default;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer::server_engine::{InferenceEngine, LoadedInferenceEngine};
#[cfg(all(not(feature = "cuda"), any(feature = "metal", feature = "cpu")))]
use log::warn;

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
        init_default();
        let args = Args::parse();
        let model_source = resolve_model_source(args.model_path.as_deref())?;
        log::info!("Loading model from: {}", model_source);
        let load_start = Instant::now();
        let mut engine = LoadedInferenceEngine::load(&model_source, !args.no_cuda_graph)?;
        let backend_name = engine.backend_name().to_string();

        if let Some(max_kv) = args.max_gpu_kv {
            #[cfg(feature = "cuda")]
            log::info!(
                "Setting max GPU KV to {} tokens (offload test mode)",
                max_kv
            );
            #[cfg(not(feature = "cuda"))]
            warn!("Ignoring --max-gpu-kv: only supported by the CUDA backend");
            engine.set_max_gpu_kv(max_kv);
        }
        log::info!(
            "Model loaded in {:.1}s (backend={}, model={})",
            load_start.elapsed().as_secs_f64(),
            engine.backend_name(),
            engine.model_id(),
        );
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

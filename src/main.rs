mod agent;
mod chat;
mod engine;
mod tools;

#[cfg(any(feature = "cuda", feature = "metal"))]
use std::io::{self, BufRead, Write};
#[cfg(any(feature = "cuda", feature = "metal"))]
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
#[cfg(any(feature = "cuda", feature = "metal"))]
use log::info;
#[cfg(all(not(feature = "cuda"), feature = "metal"))]
use log::warn;

#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::agent::{AgentSession, AgentSettings};
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::engine::{AgentEngine, LoadedAgentEngine};
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::tools::builtin_tools;

#[derive(Parser)]
#[command(name = "agent-infer", about = "Local LLM agent with tool use")]
struct Args {
    /// Path to model directory or HuggingFace model ID.
    /// If omitted, the CLI auto-detects a local model from common directories and HF cache.
    #[arg(long)]
    model_path: Option<String>,

    /// Maximum agent turns (generate-execute cycles) per query
    #[arg(long, default_value_t = 10)]
    max_turns: usize,

    /// Maximum tokens to generate per turn
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Disable CUDA graph (useful for debugging)
    #[arg(long, default_value_t = false)]
    no_cuda_graph: bool,

    /// Max KV cache tokens on GPU. Excess offloads to CPU.
    /// Use a small value (e.g. 512) to test KV offload behavior.
    #[arg(long)]
    max_gpu_kv: Option<usize>,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn run_repl(
    engine: &mut dyn AgentEngine,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) -> Result<()> {
    let tools = builtin_tools();
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut session = AgentSession::new();

    println!();
    println!("=== agent-infer REPL ===");
    println!("Model: {}", engine.model_id());
    println!(
        "Tools: {}",
        tools
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Max turns: {}, Max tokens: {}, Temperature: {}",
        max_turns, max_tokens, temperature
    );
    println!("Type your query and press Enter. Type '/reset' to clear history.");
    println!("Type 'quit' or 'exit' to stop.");
    println!();

    loop {
        print!("\x1b[1;32m> \x1b[0m");
        stdout.flush()?;

        let mut input = String::new();
        let bytes = stdin.lock().read_line(&mut input)?;
        if bytes == 0 {
            // EOF
            println!();
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/reset" {
            session.reset();
            println!("\x1b[2m(conversation reset)\x1b[0m");
            println!();
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        let start = Instant::now();
        match session.run_turn(
            engine,
            input,
            &tools,
            AgentSettings {
                max_turns,
                max_tokens,
                temperature,
            },
        ) {
            Ok(result) => {
                println!();
                println!("\x1b[1;34m{}\x1b[0m", result.text);
                if result.max_turns_reached {
                    println!("\x1b[2m(agent stopped after reaching max turns)\x1b[0m");
                }
                println!();
                let elapsed = start.elapsed();
                println!("\x1b[2m({:.1}s)\x1b[0m", elapsed.as_secs_f64());
                println!();
            }
            Err(e) => {
                eprintln!("\x1b[1;31mError: {e:#}\x1b[0m");
                println!();
            }
        }
    }

    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn resolve_cli_model_source(args: &Args) -> Result<String> {
    if let Some(model_path) = args.model_path.as_ref()
        && !model_path.trim().is_empty()
    {
        return Ok(model_path.clone());
    }

    if let Ok(model) = std::env::var("AGENT_INFER_MODEL")
        && !model.trim().is_empty()
    {
        return Ok(model);
    }

    if let Some((candidate, local_path)) = infer::hf_hub::discover_local_model() {
        info!(
            "Auto-detected local model '{}' at {}",
            candidate,
            local_path.display()
        );
        return Ok(candidate);
    }

    anyhow::bail!(
        "No model specified and no local model was auto-detected. Pass --model-path or set AGENT_INFER_MODEL."
    )
}

fn main() -> Result<()> {
    infer::logging::init_default();

    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    {
        anyhow::bail!(
            "agent-infer requires a local inference backend. Rebuild with either \
             the default `cuda` feature or `--no-default-features --features metal,no-cuda`."
        );
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        let args = Args::parse();
        let model_source = resolve_cli_model_source(&args)?;
        info!("Loading model from: {}", model_source);
        let load_start = Instant::now();
        let mut engine = LoadedAgentEngine::load(&model_source, !args.no_cuda_graph)?;

        if let Some(max_kv) = args.max_gpu_kv {
            #[cfg(feature = "cuda")]
            info!(
                "Setting max GPU KV to {} tokens (offload test mode)",
                max_kv
            );
            #[cfg(not(feature = "cuda"))]
            warn!("Ignoring --max-gpu-kv: only supported by the CUDA backend");
            engine.set_max_gpu_kv(max_kv);
        }
        info!(
            "Model loaded in {:.1}s (backend={}, model={})",
            load_start.elapsed().as_secs_f64(),
            engine.backend_name(),
            engine.model_id(),
        );
        run_repl(
            &mut engine,
            args.max_turns,
            args.max_tokens,
            args.temperature,
        )?;

        Ok(())
    }
}

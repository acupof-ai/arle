use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use infer::backend::metal::MetalBackend;
use infer::backend::{GenerateResult, InferenceBackend};
use infer::logging;
use infer::sampler::SamplingParams;
use infer_chat::{ChatMessage, messages_to_prompt};

fn parse_metal_top_k(raw: &str) -> Result<i32, String> {
    let parsed: i32 = raw
        .parse()
        .map_err(|_| format!("invalid integer for --top-k: {raw}"))?;
    if parsed == -1 || parsed == 1 {
        Ok(parsed)
    } else {
        Err("metal_request only supports --top-k -1 or --top-k 1".to_string())
    }
}

#[derive(Parser)]
#[command(
    name = "metal_request",
    about = "Single MLX/Metal inference request (greedy or temperature sampling)"
)]
struct Args {
    /// Model path (local directory) or HuggingFace repo ID.
    #[arg(long, short)]
    model: String,

    /// Prompt text as a user message.
    #[arg(long, conflicts_with = "prompt_file")]
    prompt: Option<String>,

    /// Read prompt text from file.
    #[arg(long, value_name = "PATH", conflicts_with = "prompt")]
    prompt_file: Option<String>,

    /// Optional system prompt when using chat formatting.
    #[arg(long)]
    system: Option<String>,

    /// Treat `--prompt` as a raw prompt and skip ChatML formatting.
    #[arg(long, default_value_t = false)]
    raw_prompt: bool,

    /// Stream generated text to stdout as it is produced.
    #[arg(long, default_value_t = false)]
    stream: bool,

    /// Number of warmup runs before the timed request.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,

    /// Sampling temperature. `0.0` = greedy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Top-K sampling. Metal currently supports only `-1` or `1`.
    #[arg(long, default_value_t = -1, value_parser = parse_metal_top_k)]
    top_k: i32,

    /// Ignore EOS and continue generating until `max_new_tokens`.
    #[arg(long, default_value_t = false)]
    ignore_eos: bool,
}

fn load_prompt(args: &Args) -> Result<String> {
    match (&args.prompt, &args.prompt_file) {
        (Some(prompt), None) => Ok(prompt.clone()),
        (None, Some(path)) => {
            fs::read_to_string(path).with_context(|| format!("failed to read prompt file: {path}"))
        }
        (Some(_), Some(_)) => unreachable!("clap enforces conflicts"),
        (None, None) => anyhow::bail!("one of --prompt or --prompt-file is required"),
    }
}

fn build_chat_prompt(user_prompt: &str, system_prompt: Option<&str>) -> String {
    let mut messages = Vec::new();
    if let Some(system) = system_prompt {
        messages.push(ChatMessage {
            role: "system".into(),
            content: Some(system.to_string()),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        });
    }
    messages.push(ChatMessage {
        role: "user".into(),
        content: Some(user_prompt.to_string()),
        tool_calls: vec![],
        tool_call_id: None,
        name: None,
    });
    messages_to_prompt(&messages, &[])
}

fn print_summary<W: Write>(
    mut out: W,
    args: &Args,
    result: &GenerateResult,
    wall_s: f64,
) -> Result<()> {
    writeln!(out, "Model:         {}", args.model)?;
    writeln!(
        out,
        "Prompt format: {}",
        if args.raw_prompt { "raw" } else { "chatml" }
    )?;
    writeln!(out, "Prompt tokens: {}", result.prompt_tokens)?;
    writeln!(out, "Output tokens: {}", result.completion_tokens)?;
    writeln!(out, "TTFT:          {:.1} ms", result.ttft_ms)?;
    writeln!(out, "Prompt TPS:    {:.1} tok/s", result.prompt_tps)?;
    writeln!(out, "Gen TPS:       {:.1} tok/s", result.generation_tps)?;
    writeln!(out, "Total time:    {:.1} ms", result.total_time_ms)?;
    writeln!(out, "Wall time:     {:.2} s", wall_s)?;
    writeln!(out, "Finish reason: {}", result.finish_reason)?;
    Ok(())
}

fn main() -> Result<()> {
    logging::init_default();

    #[cfg(not(feature = "metal"))]
    anyhow::bail!(
        "metal_request requires the `metal` feature. Rebuild with \
         --no-default-features --features metal,no-cuda"
    );

    #[cfg(feature = "metal")]
    {
        let args = Args::parse();
        let raw_prompt = load_prompt(&args)?;
        ensure!(!raw_prompt.trim().is_empty(), "prompt must not be empty");

        let prompt = if args.raw_prompt {
            raw_prompt
        } else {
            build_chat_prompt(&raw_prompt, args.system.as_deref())
        };

        let params = SamplingParams {
            temperature: args.temperature,
            top_k: args.top_k,
            ignore_eos: args.ignore_eos,
            max_new_tokens: Some(args.max_new_tokens),
            ..Default::default()
        };

        let mut backend = MetalBackend::new();
        let load_start = Instant::now();
        backend
            .load(Path::new(&args.model))
            .with_context(|| format!("failed to load model: {}", args.model))?;

        eprintln!("Model loaded in {:.2}s", load_start.elapsed().as_secs_f64());
        eprintln!(
            "Running {} warmup pass(es), then one timed request...",
            args.warmup
        );

        for _ in 0..args.warmup {
            let _ = backend.generate(&prompt, &params)?;
        }

        let t0 = Instant::now();
        let result = if args.stream {
            let mut stdout = io::stdout().lock();
            let result = backend.generate_stream(&prompt, &params, |chunk| {
                write!(stdout, "{chunk}").context("failed to write streamed chunk")?;
                stdout.flush().context("failed to flush streamed chunk")?;
                Ok(())
            })?;
            if !result.text.ends_with('\n') {
                writeln!(stdout).context("failed to terminate streamed output")?;
            }
            stdout
                .flush()
                .context("failed to flush final streamed output")?;
            result
        } else {
            backend.generate(&prompt, &params)?
        };
        let wall_s = t0.elapsed().as_secs_f64();

        if args.stream {
            print_summary(io::stderr().lock(), &args, &result, wall_s)?;
        } else {
            print_summary(io::stdout().lock(), &args, &result, wall_s)?;
            println!();
            println!("{}", result.text);
        }
    }

    Ok(())
}

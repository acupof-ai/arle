use clap::Parser;

#[derive(Parser)]
#[command(name = "agent-infer", about = "Local LLM agent with tool use")]
pub(crate) struct Args {
    /// Path to model directory or HuggingFace model ID.
    /// If omitted, the CLI auto-detects a local model from common directories and HF cache.
    #[arg(long)]
    pub(crate) model_path: Option<String>,

    /// Maximum agent turns (generate-execute cycles) per query
    #[arg(long, default_value_t = 10)]
    pub(crate) max_turns: usize,

    /// Maximum tokens to generate per turn
    #[arg(long, default_value_t = 4096)]
    pub(crate) max_tokens: usize,

    /// Sampling temperature (0.0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    pub(crate) temperature: f32,

    /// Disable CUDA graph (useful for debugging)
    #[arg(long, default_value_t = false)]
    pub(crate) no_cuda_graph: bool,

    /// Skip interactive model selection (use auto-discovery)
    #[arg(long, default_value_t = false)]
    pub(crate) non_interactive: bool,

    /// Start in tool-calling agent mode instead of streaming chat mode (default: chat).
    /// Inside the REPL, `/agent` and `/chat` switch between modes at any time.
    #[arg(long, default_value_t = false)]
    pub(crate) tools: bool,

    /// Deprecated compatibility flag. Legacy contiguous CPU KV offload has
    /// been retired, so this value is ignored.
    #[arg(long)]
    pub(crate) max_gpu_kv: Option<usize>,
}

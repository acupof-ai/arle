use std::path::PathBuf;

use clap::{ArgGroup, Args as ClapArgs, Parser, Subcommand, ValueEnum};

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("expected a positive integer, got '{value}'"))?;
    if parsed == 0 {
        return Err("value must be at least 1".to_string());
    }
    Ok(parsed)
}

/// Like `parse_positive_usize` but also accepts `0` as an "auto" sentinel —
/// used by `--max-tokens` so users can ask the CLI to read the model's
/// `max_position_embeddings` (or `context_length`) at startup instead of
/// pinning a fixed cap. Negative or non-integer input is still rejected.
fn parse_max_tokens_or_auto(value: &str) -> Result<usize, String> {
    let trimmed = value.trim();
    if trimmed == "auto" || trimmed == "0" {
        return Ok(0);
    }
    parse_positive_usize(trimmed)
}

fn parse_temperature(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("expected a finite number, got '{value}'"))?;
    if !parsed.is_finite() {
        return Err("temperature must be finite".to_string());
    }
    if parsed < 0.0 {
        return Err("temperature must be >= 0.0".to_string());
    }
    Ok(parsed)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum TracePromptsMode {
    On,
    Off,
}

// `keep_prompts` is only consumed by the trajectory writer, which is
// itself gated on a backend feature being active. Mirror that gate
// here so `cargo clippy -p cli -- -D warnings` on the no-backend
// build doesn't trip on `method never used`. (codex Phase-1 P1)
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl TracePromptsMode {
    pub(crate) fn keep_prompts(self) -> bool {
        matches!(self, Self::On)
    }
}

fn parse_trace_path(value: &str) -> Result<PathBuf, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("trace path must not be empty".to_string());
    }
    Ok(PathBuf::from(trimmed))
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum BackendArg {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl BackendArg {
    pub(crate) fn as_train_backend(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            Self::Cpu => Some("cpu"),
            Self::Metal => Some("metal"),
            Self::Cuda => Some("cuda"),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum ModelFamilyArg {
    Auto,
    Qwen35,
    Qwen3,
}

impl ModelFamilyArg {
    pub(crate) fn as_train_family(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Qwen35 => "qwen35",
            Self::Qwen3 => "qwen3",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum SaveDtypeArg {
    F32,
    Bf16,
}

impl SaveDtypeArg {
    pub(crate) fn as_train_dtype(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum DatasetFormatArg {
    Chat,
    Dolly,
    Alpaca,
    Sharegpt,
}

impl DatasetFormatArg {
    pub(crate) fn as_train_format(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Dolly => "dolly",
            Self::Alpaca => "alpaca",
            Self::Sharegpt => "sharegpt",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum MultiTurnObjectiveArg {
    #[value(name = "stepwise-grpo")]
    StepwiseGrpo,
    #[value(name = "gspo")]
    Gspo,
}

impl MultiTurnObjectiveArg {
    pub(crate) fn as_train_objective(self) -> &'static str {
        match self {
            Self::StepwiseGrpo => "stepwise-grpo",
            Self::Gspo => "gspo",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum PretrainPresetArg {
    #[value(name = "tiny-3m")]
    Tiny3m,
    #[value(name = "small-25m")]
    Small25m,
    #[value(name = "small-30m")]
    Small30m,
}

#[derive(Debug, Clone, ClapArgs)]
pub(crate) struct RenderArgs {
    /// Print the fully resolved execution plan without running the job.
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run: bool,

    /// Render `--dry-run` output as JSON for scripts and CI.
    #[arg(long, default_value_t = false, requires = "dry_run")]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
pub(crate) struct ExtraArgs {
    /// Forward additional advanced flags after `--` to the underlying train binary.
    #[arg(last = true, allow_hyphen_values = true)]
    pub(crate) extra_args: Vec<String>,
}

#[derive(Parser)]
#[command(
    name = "arle",
    about = "ARLE local agent, training, and dataset CLI",
    after_help = "Common flows:\n  arle                                       Start the interactive agent REPL.\n  arle run                                   Explicit alias for the interactive agent REPL.\n  arle run --prompt \"Summarize this repo\"    Run one prompt and exit.\n  arle run --stdin --json < prompt.txt       Read one prompt from stdin and emit JSON.\n  arle serve --model-path /path/to/model      Start the OpenAI-compatible server.\n  arle --doctor                              Inspect the local environment and model resolution.\n  arle train env                             Print train-time environment diagnostics.\n  arle train test --backend metal --json     Build the canonical tiny fixture and keep stdout machine-readable.",
    group(ArgGroup::new("inspection_mode").args(["doctor", "list_models"]))
)]
pub(crate) struct Args {
    /// Path to model directory or HuggingFace model ID.
    /// If omitted, the CLI auto-detects a local model from common directories and HF cache.
    #[arg(long)]
    pub(crate) model_path: Option<String>,

    /// Print a local environment/model-resolution diagnostic report and exit.
    #[arg(long, default_value_t = false)]
    pub(crate) doctor: bool,

    /// Print discovered and recommended models, then exit.
    #[arg(long, default_value_t = false)]
    pub(crate) list_models: bool,

    /// Render `--doctor` / `--list-models` output as JSON for scripts and CI.
    #[arg(long, default_value_t = false, requires = "inspection_mode")]
    pub(crate) json: bool,

    /// Fail with a non-zero exit code when `--doctor` reports warnings.
    #[arg(
        long,
        default_value_t = false,
        requires = "doctor",
        conflicts_with = "list_models"
    )]
    pub(crate) strict: bool,

    #[command(subcommand)]
    pub(crate) command: Option<CliCommand>,

    /// Maximum agent turns (generate-execute cycles) per query.
    /// 250 lets multi-step tool plans run to completion on long tasks
    /// (project surveys, refactors, audits). The agent still stops as
    /// soon as it produces a final answer, so a high cap costs nothing
    /// on short turns. Override with `--max-turns N`.
    #[arg(long, default_value_t = 250, value_parser = parse_positive_usize)]
    pub(crate) max_turns: usize,

    /// Maximum tokens to generate per turn. Default `0` means "auto" —
    /// the CLI reads `max_position_embeddings` (or `context_length` for
    /// GGUF) from the model's config at startup and uses that as the
    /// per-turn cap. Pass `--max-tokens N` to pin an explicit value.
    /// Pass `--max-tokens auto` (or `0`) to make the auto-resolution
    /// explicit. If config can't be read, falls back to 262144 (256K).
    #[arg(long, default_value_t = 0, value_parser = parse_max_tokens_or_auto)]
    pub(crate) max_tokens: usize,

    /// Sampling temperature (0.0 = greedy)
    #[arg(
        long,
        default_value_t = 0.0,
        value_parser = parse_temperature,
        allow_hyphen_values = true
    )]
    pub(crate) temperature: f32,

    /// Disable CUDA graph (useful for debugging)
    #[arg(long, default_value_t = false)]
    pub(crate) no_cuda_graph: bool,

    /// Disable built-in shell/python tools for the local agent runtime.
    #[arg(long, default_value_t = false)]
    pub(crate) no_tools: bool,

    /// Skip interactive model selection (use auto-discovery)
    #[arg(long, default_value_t = false)]
    pub(crate) non_interactive: bool,

    /// Path to a JSONL file that will receive one trajectory record per
    /// agent turn (Phase 1 / v1 schema). When unset, no trajectory is
    /// written. See `docs/projects/agent-trajectory-export.md` for the
    /// canonical schema.
    #[arg(long, value_parser = parse_trace_path)]
    pub(crate) trace: Option<PathBuf>,

    /// Whether to record the full ChatML prompt in each trajectory's
    /// `sub_turns[].prompt_text`. `off` writes JSON `null` for that
    /// field — useful when the prompt would dominate trace size or
    /// leak operator data.
    #[arg(long, value_enum, default_value_t = TracePromptsMode::On)]
    pub(crate) trace_prompts: TracePromptsMode,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum CliCommand {
    /// Agent REPL and one-shot prompt execution.
    Run(Box<RunArgs>),
    /// OpenAI-compatible serving through the matching backend binary.
    Serve(Box<ServeArgs>),
    /// Training jobs.
    Train(Box<TrainArgs>),
    /// Dataset utilities.
    Data(Box<DataArgs>),
}

#[derive(Debug, Clone, PartialEq, Eq, ClapArgs)]
#[command(
    group(ArgGroup::new("run_input").args(["prompt", "stdin"])),
    after_help = "Output:\n  Plain text is written to stdout by default.\n  `--json` emits one machine-readable document with model, backend, usage, and tool-call stats.\n\nExamples:\n  arle --model-path /path/to/model run\n  arle --model-path /path/to/model run --prompt \"Summarize this repo\"\n  arle --model-path /path/to/model run --stdin --json < prompt.txt\n  arle --model-path /path/to/model run --no-tools --prompt \"No tool execution\""
)]
pub(crate) struct RunArgs {
    /// Run a single prompt and exit.
    #[arg(long)]
    pub(crate) prompt: Option<String>,

    /// Read one prompt from stdin, run it, and exit.
    #[arg(long, default_value_t = false)]
    pub(crate) stdin: bool,

    /// Render one-shot output as JSON for scripts and CI.
    #[arg(long, default_value_t = false, requires = "run_input")]
    pub(crate) json: bool,

    /// Disable built-in shell/python tools for this run.
    #[arg(long, default_value_t = false)]
    pub(crate) no_tools: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "This is a thin front door over the backend serving binaries shipped in release artifacts.\nIt looks for `infer`, `metal_serve`, or `cpu_serve` next to the current `arle` binary first, then on PATH.\n\nExamples:\n  arle serve --model-path /path/to/Qwen3-4B\n  arle serve --backend metal --model-path mlx-community/Qwen3-0.6B-4bit --port 8010\n  arle serve --backend cuda --model-path /models/Qwen3-4B -- --num-slots 8"
)]
pub(crate) struct ServeArgs {
    /// Model directory or HuggingFace model ID. Defaults to the top-level --model-path.
    #[arg(long)]
    pub(crate) model_path: Option<String>,

    /// Serving backend to launch; `auto` selects the compiled backend.
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    pub(crate) port: u16,

    /// Host or IP address to bind to when the backend binary supports it.
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) bind: String,

    /// Optional upstream train control-plane URL to expose under `/v1/train/*`.
    #[arg(long)]
    pub(crate) train_control_url: Option<String>,

    /// Forward additional backend-specific flags after `--`.
    #[arg(last = true, allow_hyphen_values = true)]
    pub(crate) extra_args: Vec<String>,
}

#[derive(Debug, Clone, clap::Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  arle train env\n  arle train test --backend metal --json\n  arle train test --backend metal --out-dir /tmp/arle-fixture\n  arle train estimate-memory --tokenizer tokenizer.json --preset small-25m\n  arle train pretrain --corpus corpus.txt --tokenizer tokenizer.json --preset small-25m"
)]
pub(crate) struct TrainArgs {
    #[command(subcommand)]
    pub(crate) command: TrainCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum TrainCommand {
    /// Print train-time environment diagnostics.
    Env(TrainEnvArgs),
    /// Build the canonical tiny fixture via convert -> pretrain -> sft -> eval.
    Test(TrainTestArgs),
    /// Estimate parameter count and rough memory for scratch pretrain or LoRA SFT.
    EstimateMemory(TrainEstimateMemoryArgs),
    /// Scratch pretraining from a plain-text corpus.
    Pretrain(TrainPretrainArgs),
    /// Supervised fine-tuning from canonical chat JSONL.
    Sft(TrainSftArgs),
    /// Group-relative policy optimization.
    Grpo(TrainGrpoArgs),
    /// Multi-turn RL training.
    MultiTurn(TrainMultiTurnArgs),
    /// Evaluate a checkpoint on tokenized or chat JSONL.
    Eval(TrainEvalArgs),
}

#[derive(Debug, Clone, clap::Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  arle data download --repo tatsu-lab/alpaca --file alpaca_data.json\n  arle data convert --input alpaca.jsonl --format alpaca\n  arle data convert --input data.jsonl --format sharegpt --dry-run --json"
)]
pub(crate) struct DataArgs {
    #[command(subcommand)]
    pub(crate) command: DataCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum DataCommand {
    /// Download one dataset file from Hugging Face.
    Download(DataDownloadArgs),
    /// Convert instruction-tuning JSONL into canonical chat JSONL.
    Convert(DataConvertArgs),
}

#[derive(Debug, Clone, ClapArgs)]
pub(crate) struct TrainEnvArgs {
    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "This command builds the canonical tiny fixture through convert -> pretrain -> sft -> eval.\nUse `--keep-artifacts` or `--out-dir` when you want to keep the generated checkpoint.\nThe final checkpoint lives at `<root>/sft/latest` and can be passed directly to `arle --model-path` or `infer --model-path`."
)]
pub(crate) struct TrainTestArgs {
    /// Training backend to exercise; `auto` selects the compiled backend.
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    /// Keep the temporary fixture directory instead of deleting it.
    #[arg(long, default_value_t = false)]
    pub(crate) keep_artifacts: bool,

    /// Override the fixture output directory. Defaults to a temp folder.
    #[arg(long)]
    pub(crate) out_dir: Option<PathBuf>,

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Examples:\n  arle train estimate-memory --tokenizer tokenizer.json --preset small-25m\n  arle train estimate-memory --model checkpoints/base --lora-rank 32 --json"
)]
pub(crate) struct TrainEstimateMemoryArgs {
    /// Existing model directory to inspect for LoRA SFT / eval-style runs.
    #[arg(long, alias = "model-path")]
    pub(crate) model: Option<PathBuf>,

    /// Scratch tokenizer source (`tokenizer.json` or a local model dir containing it).
    #[arg(long)]
    pub(crate) tokenizer: Option<PathBuf>,

    /// Optional scratch preset for `pretrain`-style estimates.
    #[arg(long, value_enum)]
    pub(crate) preset: Option<PretrainPresetArg>,

    /// Override the scratch model family.
    #[arg(long, value_enum)]
    pub(crate) model_family: Option<ModelFamilyArg>,

    /// Token batch width used for the rough activation estimate.
    #[arg(long, default_value_t = 1, value_parser = parse_positive_usize)]
    pub(crate) batch: usize,

    /// Sequence length used for the rough activation estimate.
    #[arg(long, default_value_t = 512, value_parser = parse_positive_usize)]
    pub(crate) seq: usize,

    /// LoRA rank used for model-dir estimates.
    #[arg(long, default_value_t = 16, value_parser = parse_positive_usize)]
    pub(crate) lora_rank: usize,

    /// Save dtype used for checkpoint-size estimates.
    #[arg(long, value_enum, default_value_t = SaveDtypeArg::Bf16)]
    pub(crate) save_dtype: SaveDtypeArg,

    /// Scratch vocab size override.
    #[arg(long)]
    pub(crate) vocab_size: Option<usize>,

    /// Scratch hidden width override.
    #[arg(long)]
    pub(crate) hidden: Option<usize>,

    /// Scratch transformer layer count override.
    #[arg(long)]
    pub(crate) layers: Option<usize>,

    /// Scratch attention head count override.
    #[arg(long)]
    pub(crate) heads: Option<usize>,

    /// Scratch KV head count override.
    #[arg(long)]
    pub(crate) kv_heads: Option<usize>,

    /// Scratch per-head dimension override.
    #[arg(long)]
    pub(crate) head_dim: Option<usize>,

    /// Scratch MLP intermediate width override.
    #[arg(long)]
    pub(crate) intermediate: Option<usize>,

    /// Scratch maximum position embeddings override.
    #[arg(long)]
    pub(crate) max_pos: Option<usize>,

    /// For Qwen3.5 scratch estimates, insert one linear-attention layer every N layers.
    #[arg(long)]
    pub(crate) linear_attn_every: Option<usize>,

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Examples:\n  arle train pretrain --corpus corpus.txt --tokenizer tokenizer.json --preset small-25m\n  arle train pretrain --corpus corpus.txt --tokenizer tokenizer.json --dry-run --json\n\nPreset labels only pick the transformer shape. Final parameter count still depends on tokenizer vocab size; use `arle train estimate-memory ...` when you need the exact total.\n\nAdvanced pretrain flags still work after `--`, for example:\n  arle train pretrain --corpus corpus.txt --tokenizer tokenizer.json -- --bos-token <s>"
)]
pub(crate) struct TrainPretrainArgs {
    /// Plain-text training corpus.
    #[arg(long)]
    pub(crate) corpus: PathBuf,

    /// Tokenizer source (`tokenizer.json` or a local model dir containing it).
    #[arg(long)]
    pub(crate) tokenizer: PathBuf,

    /// Output checkpoint directory. Defaults to `runs/pretrain/<corpus-stem>`.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,

    /// Optional scratch preset.
    #[arg(long, value_enum)]
    pub(crate) preset: Option<PretrainPresetArg>,

    /// Override the scratch model family. Defaults to the train binary default.
    #[arg(long, value_enum)]
    pub(crate) model_family: Option<ModelFamilyArg>,

    /// Total optimizer steps.
    #[arg(long)]
    pub(crate) steps: Option<usize>,

    /// Micro-batch size in sequences per step.
    #[arg(long)]
    pub(crate) batch: Option<usize>,

    /// Sequence length in tokens.
    #[arg(long)]
    pub(crate) seq: Option<usize>,

    /// Learning rate.
    #[arg(long)]
    pub(crate) lr: Option<f32>,

    /// Gradient accumulation steps before each optimizer update.
    #[arg(long)]
    pub(crate) grad_accum_steps: Option<usize>,

    /// Log training metrics every N steps.
    #[arg(long)]
    pub(crate) log_every: Option<usize>,

    /// Write a checkpoint every N steps.
    #[arg(long)]
    pub(crate) save_every: Option<usize>,

    /// Run evaluation every N steps.
    #[arg(long)]
    pub(crate) eval_every: Option<usize>,

    /// Number of evaluation windows to sample per eval pass.
    #[arg(long)]
    pub(crate) eval_windows: Option<usize>,

    /// Fraction of the corpus reserved for evaluation.
    #[arg(long)]
    pub(crate) eval_frac: Option<f32>,

    /// Resume from an existing checkpoint directory.
    #[arg(long)]
    pub(crate) resume_from: Option<PathBuf>,

    /// Random seed.
    #[arg(long)]
    pub(crate) seed: Option<u64>,

    /// Gradient clipping norm.
    #[arg(long, conflicts_with = "no_grad_clip")]
    pub(crate) grad_clip: Option<f32>,

    /// Disable gradient clipping.
    #[arg(long, default_value_t = false)]
    pub(crate) no_grad_clip: bool,

    /// Training backend.
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    /// Checkpoint dtype to write.
    #[arg(long, value_enum)]
    pub(crate) save_dtype: Option<SaveDtypeArg>,

    /// Scratch vocab size override.
    #[arg(long)]
    pub(crate) vocab_size: Option<usize>,

    /// Scratch hidden width override.
    #[arg(long)]
    pub(crate) hidden: Option<usize>,

    /// Scratch transformer layer count override.
    #[arg(long)]
    pub(crate) layers: Option<usize>,

    /// Scratch attention head count override.
    #[arg(long)]
    pub(crate) heads: Option<usize>,

    /// Scratch KV head count override.
    #[arg(long)]
    pub(crate) kv_heads: Option<usize>,

    /// Scratch per-head dimension override.
    #[arg(long)]
    pub(crate) head_dim: Option<usize>,

    /// Scratch MLP intermediate width override.
    #[arg(long)]
    pub(crate) intermediate: Option<usize>,

    /// Scratch maximum position embeddings.
    #[arg(long)]
    pub(crate) max_pos: Option<usize>,

    /// RMSNorm epsilon.
    #[arg(long)]
    pub(crate) rms_eps: Option<f32>,

    /// RoPE theta.
    #[arg(long)]
    pub(crate) rope_theta: Option<f32>,

    /// Do not tie input embedding and LM head weights.
    #[arg(long, default_value_t = false)]
    pub(crate) no_tie_embed: bool,

    /// For Qwen3.5 scratch models, insert one linear-attention layer every N layers.
    #[arg(long)]
    pub(crate) linear_attn_every: Option<usize>,

    /// Override BOS token text.
    #[arg(long)]
    pub(crate) bos_token: Option<String>,

    /// Override EOS token text.
    #[arg(long)]
    pub(crate) eos_token: Option<String>,

    /// Override BOS token id.
    #[arg(long)]
    pub(crate) bos_token_id: Option<u32>,

    /// Override EOS token id.
    #[arg(long)]
    pub(crate) eos_token_id: Option<u32>,

    /// Append per-step metrics to this JSONL file.
    #[arg(long)]
    pub(crate) metrics_jsonl: Option<PathBuf>,

    /// Expose a small HTTP status endpoint on this port during training.
    #[arg(long)]
    pub(crate) serve: Option<u16>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,

    #[command(flatten)]
    pub(crate) extra: ExtraArgs,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Examples:\n  arle train sft --model models/base --data train.chat.jsonl\n  arle train sft --model models/base --data train.chat.jsonl --dry-run --json\n\nAdvanced SFT flags still work after `--`, for example:\n  arle train sft --model models/base --data train.chat.jsonl -- --resume-from runs/sft/step_000100"
)]
pub(crate) struct TrainSftArgs {
    /// Base checkpoint directory or HF model ID.
    #[arg(long)]
    pub(crate) model: PathBuf,

    /// Canonical chat JSONL dataset.
    #[arg(long)]
    pub(crate) data: PathBuf,

    /// Output checkpoint directory. Defaults to `runs/sft/<model-name>`.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,

    /// Override auto family resolution from `config.json`.
    #[arg(long, value_enum)]
    pub(crate) model_family: Option<ModelFamilyArg>,

    /// Total optimizer steps.
    #[arg(long)]
    pub(crate) steps: Option<usize>,

    /// Micro-batch size in sequences per step.
    #[arg(long)]
    pub(crate) batch: Option<usize>,

    /// Learning rate.
    #[arg(long)]
    pub(crate) lr: Option<f32>,

    /// Sequence length in tokens.
    #[arg(long)]
    pub(crate) seq_len: Option<usize>,

    /// Training backend.
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    /// Write a checkpoint every N steps.
    #[arg(long)]
    pub(crate) save_every: Option<usize>,

    /// Log training metrics every N steps.
    #[arg(long)]
    pub(crate) log_every: Option<usize>,

    /// Random seed.
    #[arg(long)]
    pub(crate) seed: Option<u64>,

    /// Checkpoint dtype to write.
    #[arg(long, value_enum)]
    pub(crate) save_dtype: Option<SaveDtypeArg>,

    /// Learning-rate schedule name forwarded to the train binary.
    #[arg(long)]
    pub(crate) lr_schedule: Option<String>,

    /// Warmup steps for the learning-rate schedule.
    #[arg(long)]
    pub(crate) warmup_steps: Option<u64>,

    /// Floor learning rate for schedules that decay.
    #[arg(long)]
    pub(crate) min_lr: Option<f32>,

    /// Gradient accumulation steps before each optimizer update.
    #[arg(long)]
    pub(crate) grad_accum_steps: Option<usize>,

    /// Append per-step metrics to this JSONL file.
    #[arg(long)]
    pub(crate) metrics_jsonl: Option<PathBuf>,

    /// Resume from an existing checkpoint directory.
    #[arg(long)]
    pub(crate) resume_from: Option<PathBuf>,

    /// LoRA rank.
    #[arg(long)]
    pub(crate) lora_rank: Option<usize>,

    /// LoRA alpha.
    #[arg(long)]
    pub(crate) lora_alpha: Option<f32>,

    /// Expose a small HTTP status endpoint on this port during training.
    #[arg(long)]
    pub(crate) serve: Option<u16>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,

    #[command(flatten)]
    pub(crate) extra: ExtraArgs,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Examples:\n  arle train eval --model checkpoints/base --data eval.chat.jsonl\n  arle train eval --model checkpoints/base --data eval.chat.jsonl --dry-run --json\n\nAdvanced eval flags still work after `--`, for example:\n  arle train eval --model checkpoints/base --data eval.chat.jsonl -- --metrics-jsonl metrics.jsonl"
)]
pub(crate) struct TrainEvalArgs {
    /// Checkpoint directory or HF model ID.
    #[arg(long, alias = "model-path")]
    pub(crate) model: PathBuf,

    /// Evaluation dataset (`.txt` or chat JSONL).
    #[arg(long)]
    pub(crate) data: PathBuf,

    /// Override auto family resolution from `config.json`.
    #[arg(long, value_enum)]
    pub(crate) model_family: Option<ModelFamilyArg>,

    /// Optional tokenizer override. Defaults to `<model>/tokenizer.json`.
    #[arg(long)]
    pub(crate) tokenizer: Option<PathBuf>,

    /// Sequence length in tokens.
    #[arg(long)]
    pub(crate) seq_len: Option<usize>,

    /// Evaluation backend.
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    /// Append evaluation metrics to this JSONL file.
    #[arg(long)]
    pub(crate) metrics_jsonl: Option<PathBuf>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,

    #[command(flatten)]
    pub(crate) extra: ExtraArgs,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Examples:\n  arle train grpo --grpo-iters 20 --batch-prompts 8 --group-size 4\n  arle train grpo --grpo-iters 20 --dry-run --json\n\nAdvanced GRPO flags still work after `--`, for example:\n  arle train grpo --grpo-iters 20 -- --resume-from runs/grpo/step_000010"
)]
pub(crate) struct TrainGrpoArgs {
    /// Scratch policy family override.
    #[arg(long, value_enum)]
    pub(crate) model_family: Option<ModelFamilyArg>,

    /// Warm-start SFT steps before GRPO.
    #[arg(long)]
    pub(crate) sft_steps: Option<usize>,

    /// Total GRPO iterations.
    #[arg(long)]
    pub(crate) grpo_iters: Option<usize>,

    /// Write a checkpoint every N iterations.
    #[arg(long)]
    pub(crate) save_every: Option<usize>,

    /// Prompt batch size per GRPO iteration.
    #[arg(long)]
    pub(crate) batch_prompts: Option<usize>,

    /// Number of rollouts sampled per prompt.
    #[arg(long)]
    pub(crate) group_size: Option<usize>,

    /// Prompt-plus-response sequence budget in tokens.
    #[arg(long)]
    pub(crate) seq: Option<usize>,

    /// Policy learning rate.
    #[arg(long)]
    pub(crate) lr: Option<f32>,

    /// KL penalty coefficient against the reference policy.
    #[arg(long)]
    pub(crate) kl_coef: Option<f32>,

    /// Sampling temperature for rollout generation.
    #[arg(long)]
    pub(crate) temperature: Option<f32>,

    /// Random seed.
    #[arg(long)]
    pub(crate) seed: Option<u64>,

    /// LoRA rank.
    #[arg(long)]
    pub(crate) lora_rank: Option<usize>,

    /// LoRA alpha.
    #[arg(long)]
    pub(crate) lora_alpha: Option<f32>,

    /// Training backend.
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    /// Gradient clipping norm.
    #[arg(long, conflicts_with = "no_grad_clip")]
    pub(crate) grad_clip: Option<f32>,

    /// Disable gradient clipping.
    #[arg(long, default_value_t = false)]
    pub(crate) no_grad_clip: bool,

    /// Append per-step metrics to this JSONL file.
    #[arg(long)]
    pub(crate) metrics_jsonl: Option<PathBuf>,

    /// Output checkpoint directory.
    #[arg(long)]
    pub(crate) save_path: Option<PathBuf>,

    /// Resume from an existing checkpoint directory.
    #[arg(long)]
    pub(crate) resume_from: Option<PathBuf>,

    /// Expose a small HTTP status endpoint on this port during training.
    #[arg(long)]
    pub(crate) serve: Option<u16>,

    /// For Qwen3.5 scratch policies, insert one linear-attention layer every N layers.
    #[arg(long)]
    pub(crate) linear_attn_every: Option<usize>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,

    #[command(flatten)]
    pub(crate) extra: ExtraArgs,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Examples:\n  arle train multi-turn --iters 20 --group-size 4 --turns 6\n  arle train multi-turn --iters 20 --dry-run --json\n\nAdvanced multi-turn flags still work after `--`, for example:\n  arle train multi-turn --iters 20 -- --resume-from runs/multi-turn/step_000010"
)]
pub(crate) struct TrainMultiTurnArgs {
    /// Total training iterations.
    #[arg(long)]
    pub(crate) iters: Option<usize>,

    /// Number of episodes sampled per prompt.
    #[arg(long)]
    pub(crate) group_size: Option<usize>,

    /// Per-turn agent token budget.
    #[arg(long)]
    pub(crate) agent_tokens: Option<usize>,

    /// Per-turn observation token budget.
    #[arg(long)]
    pub(crate) obs_tokens: Option<usize>,

    /// Maximum environment turns per episode.
    #[arg(long)]
    pub(crate) turns: Option<usize>,

    /// Initial prompt length in tokens.
    #[arg(long)]
    pub(crate) prompt_len: Option<usize>,

    /// Policy learning rate.
    #[arg(long)]
    pub(crate) lr: Option<f32>,

    /// KL penalty coefficient against the reference policy.
    #[arg(long)]
    pub(crate) kl_coef: Option<f32>,

    /// PPO-style clip epsilon.
    #[arg(long)]
    pub(crate) clip_eps: Option<f32>,

    /// Sampling temperature for rollout generation.
    #[arg(long)]
    pub(crate) temperature: Option<f32>,

    /// Reward discount factor.
    #[arg(long)]
    pub(crate) gamma: Option<f32>,

    /// LoRA rank.
    #[arg(long)]
    pub(crate) lora_rank: Option<usize>,

    /// LoRA alpha.
    #[arg(long)]
    pub(crate) lora_alpha: Option<f32>,

    /// Random seed.
    #[arg(long)]
    pub(crate) seed: Option<u64>,

    /// Scratch vocab size override.
    #[arg(long)]
    pub(crate) vocab: Option<usize>,

    /// Scratch synthetic target-range override.
    #[arg(long)]
    pub(crate) target_range: Option<usize>,

    /// Scratch hidden width override.
    #[arg(long)]
    pub(crate) d_model: Option<usize>,

    /// Scratch transformer layer count override.
    #[arg(long)]
    pub(crate) n_layers: Option<usize>,

    /// Scratch attention head count override.
    #[arg(long)]
    pub(crate) n_heads: Option<usize>,

    /// Scratch per-head dimension override.
    #[arg(long)]
    pub(crate) d_head: Option<usize>,

    /// Scratch MLP intermediate width override.
    #[arg(long)]
    pub(crate) d_ff: Option<usize>,

    /// For Qwen3.5 scratch policies, insert one linear-attention layer every N layers.
    #[arg(long)]
    pub(crate) linear_attn_every: Option<usize>,

    /// Run evaluation every N iterations.
    #[arg(long)]
    pub(crate) eval_every: Option<usize>,

    /// Number of evaluation prompts per eval pass.
    #[arg(long)]
    pub(crate) eval_prompts: Option<usize>,

    /// Sampling temperature for evaluation rollouts.
    #[arg(long)]
    pub(crate) eval_temperature: Option<f32>,

    /// Training backend.
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    /// Output checkpoint directory.
    #[arg(long)]
    pub(crate) save_path: Option<PathBuf>,

    /// Resume from an existing checkpoint directory.
    #[arg(long)]
    pub(crate) resume_from: Option<PathBuf>,

    /// Expose a small HTTP status endpoint on this port during training.
    #[arg(long)]
    pub(crate) serve: Option<u16>,

    /// Gradient clipping norm.
    #[arg(long, conflicts_with = "no_grad_clip")]
    pub(crate) grad_clip: Option<f32>,

    /// Disable gradient clipping.
    #[arg(long, default_value_t = false)]
    pub(crate) no_grad_clip: bool,

    /// Append per-step metrics to this JSONL file.
    #[arg(long)]
    pub(crate) metrics_jsonl: Option<PathBuf>,

    /// Reward aggregation objective.
    #[arg(long, value_enum)]
    pub(crate) objective: Option<MultiTurnObjectiveArg>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,

    #[command(flatten)]
    pub(crate) extra: ExtraArgs,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "The output path defaults to `<input-stem>.chat.jsonl`.\nExample:\n  arle data convert --input alpaca.jsonl --format alpaca"
)]
pub(crate) struct DataConvertArgs {
    /// Input JSONL file in a supported public schema.
    #[arg(long)]
    pub(crate) input: PathBuf,

    /// Input dataset schema.
    #[arg(long, value_enum)]
    pub(crate) format: DatasetFormatArg,

    /// Output canonical chat JSONL path.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Example:\n  arle data download --repo tatsu-lab/alpaca --file alpaca_data.json"
)]
pub(crate) struct DataDownloadArgs {
    /// Hugging Face dataset repo ID.
    #[arg(long)]
    pub(crate) repo: String,

    /// File path within the dataset repo.
    #[arg(long)]
    pub(crate) file: String,

    #[command(flatten)]
    pub(crate) render: RenderArgs,
}

#[cfg(test)]
mod tests {
    use super::{
        Args, CliCommand, DataCommand, DatasetFormatArg, ModelFamilyArg, RunArgs, TrainCommand,
        TrainPretrainArgs,
    };
    use clap::{CommandFactory, Parser};

    #[test]
    fn rejects_removed_max_gpu_kv_flag() {
        let err = Args::try_parse_from(["arle", "--max-gpu-kv", "256"])
            .err()
            .expect("removed flag should be rejected");
        let rendered = err.to_string();
        assert!(rendered.contains("--max-gpu-kv"));
    }

    #[test]
    fn rejects_removed_tools_flag() {
        let err = Args::try_parse_from(["arle", "--tools"])
            .err()
            .expect("removed flag should be rejected");
        assert!(err.to_string().contains("--tools"));
    }

    #[test]
    fn accepts_no_tools_flag() {
        let args = Args::try_parse_from(["arle", "--no-tools"])
            .expect("global no-tools flag should parse");
        assert!(args.no_tools);
    }

    #[test]
    fn accepts_run_no_tools_flag() {
        let args = Args::try_parse_from(["arle", "run", "--no-tools"])
            .expect("run no-tools flag should parse");
        match args.command.expect("run command") {
            CliCommand::Run(run) => assert!(run.no_tools),
            other => panic!("expected run command, got {other:?}"),
        }
    }

    #[test]
    fn accepts_serve_command() {
        let args = Args::try_parse_from([
            "arle",
            "serve",
            "--backend",
            "cpu",
            "--model-path",
            "models/tiny",
            "--port",
            "8010",
            "--",
            "--max-waiting",
            "8",
        ])
        .expect("serve command should parse");
        match args.command.expect("serve command") {
            CliCommand::Serve(serve) => {
                assert_eq!(serve.backend, super::BackendArg::Cpu);
                assert_eq!(serve.model_path.as_deref(), Some("models/tiny"));
                assert_eq!(serve.port, 8010);
                assert_eq!(serve.extra_args, ["--max-waiting", "8"]);
            }
            other => panic!("expected serve command, got {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_max_turns() {
        let err = Args::try_parse_from(["arle", "--max-turns", "0"])
            .err()
            .expect("zero max-turns should be rejected");
        assert!(err.to_string().contains("at least 1"));
    }

    #[test]
    fn accepts_zero_max_tokens_as_auto_sentinel() {
        // 0 is reserved as the "auto-resolve from model config" sentinel —
        // the CLI substitutes `max_position_embeddings` at startup before
        // any inference call, so the engine never sees a literal 0.
        let args = Args::try_parse_from(["arle", "--max-tokens", "0"])
            .expect("0 max-tokens should parse as auto");
        assert_eq!(args.max_tokens, 0);
    }

    #[test]
    fn accepts_auto_keyword_for_max_tokens() {
        let args = Args::try_parse_from(["arle", "--max-tokens", "auto"])
            .expect("'auto' should parse as the same sentinel");
        assert_eq!(args.max_tokens, 0);
    }

    #[test]
    fn rejects_negative_or_garbage_max_tokens() {
        // Anything other than 0/auto/positive-integer must still error out.
        for bad in ["-1", "1.5", "abc", " "] {
            let err = Args::try_parse_from(["arle", "--max-tokens", bad])
                .err()
                .unwrap_or_else(|| panic!("garbage value `{bad}` should be rejected"));
            let msg = err.to_string();
            assert!(
                msg.contains("expected") || msg.contains("at least 1"),
                "unexpected error for `{bad}`: {msg}"
            );
        }
    }

    #[test]
    fn rejects_negative_temperature() {
        let err = Args::try_parse_from(["arle", "--temperature", "-0.1"])
            .err()
            .expect("negative temperature should be rejected");
        assert!(err.to_string().contains("temperature must be >= 0.0"));
    }

    #[test]
    fn rejects_non_finite_temperature() {
        let err = Args::try_parse_from(["arle", "--temperature", "NaN"])
            .err()
            .expect("NaN temperature should be rejected");
        assert!(err.to_string().contains("temperature must be finite"));
    }

    #[test]
    fn accepts_doctor_flag() {
        let args = Args::try_parse_from(["arle", "--doctor"]).expect("doctor flag should parse");
        assert!(args.doctor);
    }

    #[test]
    fn accepts_list_models_flag() {
        let args =
            Args::try_parse_from(["arle", "--list-models"]).expect("list-models flag should parse");
        assert!(args.list_models);
    }

    #[test]
    fn rejects_doctor_and_list_models_together() {
        let err = Args::try_parse_from(["arle", "--doctor", "--list-models"])
            .err()
            .expect("doctor and list-models should conflict");
        assert!(err.to_string().contains("--list-models"));
    }

    #[test]
    fn accepts_doctor_json_flag() {
        let args = Args::try_parse_from(["arle", "--doctor", "--json"])
            .expect("doctor json flag should parse");
        assert!(args.doctor);
        assert!(args.json);
    }

    #[test]
    fn accepts_doctor_strict_flag() {
        let args = Args::try_parse_from(["arle", "--doctor", "--strict"])
            .expect("doctor strict flag should parse");
        assert!(args.doctor);
        assert!(args.strict);
    }

    #[test]
    fn accepts_list_models_json_flag() {
        let args = Args::try_parse_from(["arle", "--list-models", "--json"])
            .expect("list-models json flag should parse");
        assert!(args.list_models);
        assert!(args.json);
    }

    #[test]
    fn rejects_json_without_inspection_mode() {
        let err = Args::try_parse_from(["arle", "--json"])
            .err()
            .expect("--json without inspection mode should fail");
        assert!(err.to_string().contains("--doctor"));
    }

    #[test]
    fn rejects_strict_without_doctor() {
        let err = Args::try_parse_from(["arle", "--strict"])
            .err()
            .expect("--strict without doctor should fail");
        assert!(err.to_string().contains("--doctor"));
    }

    #[test]
    fn rejects_strict_with_list_models() {
        let err = Args::try_parse_from(["arle", "--list-models", "--strict"])
            .err()
            .expect("--strict with list-models should fail");
        let rendered = err.to_string();
        assert!(rendered.contains("--list-models"));
        assert!(rendered.contains("--strict"));
    }

    #[test]
    fn command_tree_is_valid() {
        Args::command().debug_assert();
    }

    #[test]
    fn accepts_train_pretrain_core_args() {
        let args = Args::try_parse_from([
            "arle",
            "train",
            "pretrain",
            "--corpus",
            "train.txt",
            "--tokenizer",
            "tok.json",
        ])
        .expect("train pretrain should parse");
        let Some(CliCommand::Train(train)) = args.command else {
            panic!("expected train command");
        };
        let TrainCommand::Pretrain(TrainPretrainArgs {
            corpus,
            tokenizer,
            out,
            ..
        }) = train.command
        else {
            panic!("expected pretrain command");
        };
        assert_eq!(corpus, std::path::PathBuf::from("train.txt"));
        assert_eq!(tokenizer, std::path::PathBuf::from("tok.json"));
        assert!(out.is_none());
    }

    #[test]
    fn accepts_run_prompt() {
        let args = Args::try_parse_from(["arle", "run", "--prompt", "hello"])
            .expect("run prompt should parse");
        let Some(CliCommand::Run(run)) = args.command else {
            panic!("expected run command");
        };
        assert_eq!(
            *run,
            RunArgs {
                prompt: Some("hello".to_string()),
                stdin: false,
                json: false,
                no_tools: false,
            }
        );
    }

    #[test]
    fn accepts_run_stdin_json() {
        let args = Args::try_parse_from(["arle", "run", "--stdin", "--json"])
            .expect("run stdin json should parse");
        let Some(CliCommand::Run(run)) = args.command else {
            panic!("expected run command");
        };
        assert!(run.stdin);
        assert!(run.json);
        assert!(run.prompt.is_none());
    }

    #[test]
    fn rejects_run_json_without_input() {
        let err = Args::try_parse_from(["arle", "run", "--json"])
            .err()
            .expect("run --json without input should fail");
        assert!(err.to_string().contains("--prompt"));
    }

    #[test]
    fn accepts_train_multi_turn_extra_args() {
        let args = Args::try_parse_from([
            "arle",
            "train",
            "multi-turn",
            "--iters",
            "2",
            "--",
            "--resume-from",
            "ckpt",
        ])
        .expect("train multi-turn should parse");
        let Some(CliCommand::Train(train)) = args.command else {
            panic!("expected train command");
        };
        let TrainCommand::MultiTurn(multi_turn) = train.command else {
            panic!("expected multi-turn command");
        };
        assert_eq!(multi_turn.iters, Some(2));
        assert_eq!(multi_turn.extra.extra_args, ["--resume-from", "ckpt"]);
    }

    #[test]
    fn accepts_data_convert_typed_args() {
        let args = Args::try_parse_from([
            "arle",
            "data",
            "convert",
            "--input",
            "raw.jsonl",
            "--format",
            "dolly",
        ])
        .expect("data convert should parse");
        let Some(CliCommand::Data(data)) = args.command else {
            panic!("expected data command");
        };
        let DataCommand::Convert(convert) = data.command else {
            panic!("expected convert command");
        };
        assert_eq!(convert.input, std::path::PathBuf::from("raw.jsonl"));
        assert_eq!(convert.format, DatasetFormatArg::Dolly);
        assert!(convert.output.is_none());
    }

    #[test]
    fn accepts_train_sft_model_family_override() {
        let args = Args::try_parse_from([
            "arle",
            "train",
            "sft",
            "--model",
            "base",
            "--data",
            "train.jsonl",
            "--model-family",
            "qwen35",
        ])
        .expect("train sft should parse");
        let Some(CliCommand::Train(train)) = args.command else {
            panic!("expected train command");
        };
        let TrainCommand::Sft(sft) = train.command else {
            panic!("expected sft command");
        };
        assert_eq!(sft.model_family, Some(ModelFamilyArg::Qwen35));
    }

    #[test]
    fn accepts_trace_with_prompts_off() {
        let args = Args::try_parse_from([
            "arle",
            "--trace",
            "/tmp/trace.jsonl",
            "--trace-prompts",
            "off",
        ])
        .expect("trace + trace-prompts should parse");
        assert_eq!(
            args.trace.as_deref(),
            Some(std::path::Path::new("/tmp/trace.jsonl"))
        );
        assert_eq!(args.trace_prompts, super::TracePromptsMode::Off);
    }

    #[test]
    fn trace_prompts_defaults_to_on() {
        let args = Args::try_parse_from(["arle"]).expect("default args");
        assert_eq!(args.trace_prompts, super::TracePromptsMode::On);
        assert!(args.trace.is_none());
    }

    #[test]
    fn rejects_empty_trace_path() {
        let err = Args::try_parse_from(["arle", "--trace", ""])
            .err()
            .expect("empty trace path should be rejected");
        assert!(err.to_string().contains("trace path must not be empty"));
    }

    #[test]
    fn rejects_train_pretrain_json_without_dry_run() {
        let err = Args::try_parse_from([
            "arle",
            "train",
            "pretrain",
            "--corpus",
            "train.txt",
            "--tokenizer",
            "tok.json",
            "--json",
        ])
        .err()
        .expect("--json without --dry-run must fail");
        assert!(err.to_string().contains("--dry-run"));
    }
}

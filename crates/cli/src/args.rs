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
    #[arg(last = true, trailing_var_arg = true, allow_hyphen_values = true)]
    pub(crate) extra_args: Vec<String>,
}

#[derive(Parser)]
#[command(
    name = "agent-infer",
    about = "Local LLM inference, training, and dataset CLI",
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

    /// Maximum agent turns (generate-execute cycles) per query
    #[arg(long, default_value_t = 10, value_parser = parse_positive_usize)]
    pub(crate) max_turns: usize,

    /// Maximum tokens to generate per turn
    #[arg(long, default_value_t = 4096, value_parser = parse_positive_usize)]
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

    /// Skip interactive model selection (use auto-discovery)
    #[arg(long, default_value_t = false)]
    pub(crate) non_interactive: bool,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum CliCommand {
    /// Training jobs.
    Train(Box<TrainArgs>),
    /// Dataset utilities.
    Data(Box<DataArgs>),
}

#[derive(Debug, Clone, clap::Args)]
#[command(arg_required_else_help = true)]
pub(crate) struct TrainArgs {
    #[command(subcommand)]
    pub(crate) command: TrainCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum TrainCommand {
    /// Print train-time environment diagnostics.
    Env(TrainEnvArgs),
    /// Run a local end-to-end smoke over convert -> pretrain -> sft -> eval.
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
#[command(arg_required_else_help = true)]
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
pub(crate) struct TrainTestArgs {
    /// Training backend to exercise; `auto` selects the compiled backend.
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    /// Keep the temporary smoke directory instead of deleting it.
    #[arg(long, default_value_t = false)]
    pub(crate) keep_artifacts: bool,

    /// Override the smoke output directory. Defaults to a temp folder.
    #[arg(long)]
    pub(crate) out_dir: Option<PathBuf>,

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
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

    #[arg(long)]
    pub(crate) vocab_size: Option<usize>,

    #[arg(long)]
    pub(crate) hidden: Option<usize>,

    #[arg(long)]
    pub(crate) layers: Option<usize>,

    #[arg(long)]
    pub(crate) heads: Option<usize>,

    #[arg(long)]
    pub(crate) kv_heads: Option<usize>,

    #[arg(long)]
    pub(crate) head_dim: Option<usize>,

    #[arg(long)]
    pub(crate) intermediate: Option<usize>,

    #[arg(long)]
    pub(crate) max_pos: Option<usize>,

    #[arg(long)]
    pub(crate) linear_attn_every: Option<usize>,

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Advanced pretrain flags still work after `--`, for example:\n  agent-infer train pretrain --corpus corpus.txt --tokenizer tokenizer.json -- --bos-token <s>"
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

    #[arg(long)]
    pub(crate) steps: Option<usize>,

    #[arg(long)]
    pub(crate) batch: Option<usize>,

    #[arg(long)]
    pub(crate) seq: Option<usize>,

    #[arg(long)]
    pub(crate) lr: Option<f32>,

    #[arg(long)]
    pub(crate) grad_accum_steps: Option<usize>,

    #[arg(long)]
    pub(crate) log_every: Option<usize>,

    #[arg(long)]
    pub(crate) save_every: Option<usize>,

    #[arg(long)]
    pub(crate) eval_every: Option<usize>,

    #[arg(long)]
    pub(crate) eval_windows: Option<usize>,

    #[arg(long)]
    pub(crate) eval_frac: Option<f32>,

    #[arg(long)]
    pub(crate) resume_from: Option<PathBuf>,

    #[arg(long)]
    pub(crate) seed: Option<u64>,

    #[arg(long, conflicts_with = "no_grad_clip")]
    pub(crate) grad_clip: Option<f32>,

    #[arg(long, default_value_t = false)]
    pub(crate) no_grad_clip: bool,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    #[arg(long, value_enum)]
    pub(crate) save_dtype: Option<SaveDtypeArg>,

    #[arg(long)]
    pub(crate) vocab_size: Option<usize>,

    #[arg(long)]
    pub(crate) hidden: Option<usize>,

    #[arg(long)]
    pub(crate) layers: Option<usize>,

    #[arg(long)]
    pub(crate) heads: Option<usize>,

    #[arg(long)]
    pub(crate) kv_heads: Option<usize>,

    #[arg(long)]
    pub(crate) head_dim: Option<usize>,

    #[arg(long)]
    pub(crate) intermediate: Option<usize>,

    #[arg(long)]
    pub(crate) max_pos: Option<usize>,

    #[arg(long)]
    pub(crate) rms_eps: Option<f32>,

    #[arg(long)]
    pub(crate) rope_theta: Option<f32>,

    #[arg(long, default_value_t = false)]
    pub(crate) no_tie_embed: bool,

    #[arg(long)]
    pub(crate) linear_attn_every: Option<usize>,

    #[arg(long)]
    pub(crate) bos_token: Option<String>,

    #[arg(long)]
    pub(crate) eos_token: Option<String>,

    #[arg(long)]
    pub(crate) bos_token_id: Option<u32>,

    #[arg(long)]
    pub(crate) eos_token_id: Option<u32>,

    #[arg(long)]
    pub(crate) metrics_jsonl: Option<PathBuf>,

    #[arg(long)]
    pub(crate) serve: Option<u16>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,

    #[command(flatten)]
    pub(crate) extra: ExtraArgs,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Advanced SFT flags still work after `--`, for example:\n  agent-infer train sft --model models/base --data train.chat.jsonl -- --resume-from runs/sft/step_000100"
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

    #[arg(long)]
    pub(crate) steps: Option<usize>,

    #[arg(long)]
    pub(crate) batch: Option<usize>,

    #[arg(long)]
    pub(crate) lr: Option<f32>,

    #[arg(long)]
    pub(crate) seq_len: Option<usize>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    #[arg(long)]
    pub(crate) save_every: Option<usize>,

    #[arg(long)]
    pub(crate) log_every: Option<usize>,

    #[arg(long)]
    pub(crate) seed: Option<u64>,

    #[arg(long, value_enum)]
    pub(crate) save_dtype: Option<SaveDtypeArg>,

    #[arg(long)]
    pub(crate) lr_schedule: Option<String>,

    #[arg(long)]
    pub(crate) warmup_steps: Option<u64>,

    #[arg(long)]
    pub(crate) min_lr: Option<f32>,

    #[arg(long)]
    pub(crate) grad_accum_steps: Option<usize>,

    #[arg(long)]
    pub(crate) metrics_jsonl: Option<PathBuf>,

    #[arg(long)]
    pub(crate) resume_from: Option<PathBuf>,

    #[arg(long)]
    pub(crate) lora_rank: Option<usize>,

    #[arg(long)]
    pub(crate) lora_alpha: Option<f32>,

    #[arg(long)]
    pub(crate) serve: Option<u16>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,

    #[command(flatten)]
    pub(crate) extra: ExtraArgs,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Advanced eval flags still work after `--`, for example:\n  agent-infer train eval --model checkpoints/base --data eval.chat.jsonl -- --metrics-jsonl metrics.jsonl"
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

    #[arg(long)]
    pub(crate) seq_len: Option<usize>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    #[arg(long)]
    pub(crate) metrics_jsonl: Option<PathBuf>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,

    #[command(flatten)]
    pub(crate) extra: ExtraArgs,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Advanced GRPO flags still work after `--`, for example:\n  agent-infer train grpo --grpo-iters 20 -- --resume-from runs/grpo/step_000010"
)]
pub(crate) struct TrainGrpoArgs {
    #[arg(long, value_enum)]
    pub(crate) model_family: Option<ModelFamilyArg>,

    #[arg(long)]
    pub(crate) sft_steps: Option<usize>,

    #[arg(long)]
    pub(crate) grpo_iters: Option<usize>,

    #[arg(long)]
    pub(crate) save_every: Option<usize>,

    #[arg(long)]
    pub(crate) batch_prompts: Option<usize>,

    #[arg(long)]
    pub(crate) group_size: Option<usize>,

    #[arg(long)]
    pub(crate) seq: Option<usize>,

    #[arg(long)]
    pub(crate) lr: Option<f32>,

    #[arg(long)]
    pub(crate) kl_coef: Option<f32>,

    #[arg(long)]
    pub(crate) temperature: Option<f32>,

    #[arg(long)]
    pub(crate) seed: Option<u64>,

    #[arg(long)]
    pub(crate) lora_rank: Option<usize>,

    #[arg(long)]
    pub(crate) lora_alpha: Option<f32>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    #[arg(long, conflicts_with = "no_grad_clip")]
    pub(crate) grad_clip: Option<f32>,

    #[arg(long, default_value_t = false)]
    pub(crate) no_grad_clip: bool,

    #[arg(long)]
    pub(crate) metrics_jsonl: Option<PathBuf>,

    #[arg(long)]
    pub(crate) save_path: Option<PathBuf>,

    #[arg(long)]
    pub(crate) resume_from: Option<PathBuf>,

    #[arg(long)]
    pub(crate) serve: Option<u16>,

    #[arg(long)]
    pub(crate) linear_attn_every: Option<usize>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,

    #[command(flatten)]
    pub(crate) extra: ExtraArgs,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Advanced multi-turn flags still work after `--`, for example:\n  agent-infer train multi-turn --iters 20 -- --resume-from runs/multi-turn/step_000010"
)]
pub(crate) struct TrainMultiTurnArgs {
    #[arg(long)]
    pub(crate) iters: Option<usize>,

    #[arg(long)]
    pub(crate) group_size: Option<usize>,

    #[arg(long)]
    pub(crate) agent_tokens: Option<usize>,

    #[arg(long)]
    pub(crate) obs_tokens: Option<usize>,

    #[arg(long)]
    pub(crate) turns: Option<usize>,

    #[arg(long)]
    pub(crate) prompt_len: Option<usize>,

    #[arg(long)]
    pub(crate) lr: Option<f32>,

    #[arg(long)]
    pub(crate) kl_coef: Option<f32>,

    #[arg(long)]
    pub(crate) clip_eps: Option<f32>,

    #[arg(long)]
    pub(crate) temperature: Option<f32>,

    #[arg(long)]
    pub(crate) gamma: Option<f32>,

    #[arg(long)]
    pub(crate) lora_rank: Option<usize>,

    #[arg(long)]
    pub(crate) lora_alpha: Option<f32>,

    #[arg(long)]
    pub(crate) seed: Option<u64>,

    #[arg(long)]
    pub(crate) vocab: Option<usize>,

    #[arg(long)]
    pub(crate) target_range: Option<usize>,

    #[arg(long)]
    pub(crate) d_model: Option<usize>,

    #[arg(long)]
    pub(crate) n_layers: Option<usize>,

    #[arg(long)]
    pub(crate) n_heads: Option<usize>,

    #[arg(long)]
    pub(crate) d_head: Option<usize>,

    #[arg(long)]
    pub(crate) d_ff: Option<usize>,

    #[arg(long)]
    pub(crate) linear_attn_every: Option<usize>,

    #[arg(long)]
    pub(crate) eval_every: Option<usize>,

    #[arg(long)]
    pub(crate) eval_prompts: Option<usize>,

    #[arg(long)]
    pub(crate) eval_temperature: Option<f32>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    #[arg(long)]
    pub(crate) save_path: Option<PathBuf>,

    #[arg(long)]
    pub(crate) resume_from: Option<PathBuf>,

    #[arg(long)]
    pub(crate) serve: Option<u16>,

    #[arg(long, conflicts_with = "no_grad_clip")]
    pub(crate) grad_clip: Option<f32>,

    #[arg(long, default_value_t = false)]
    pub(crate) no_grad_clip: bool,

    #[arg(long)]
    pub(crate) metrics_jsonl: Option<PathBuf>,

    #[arg(long, value_enum)]
    pub(crate) objective: Option<MultiTurnObjectiveArg>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,

    #[command(flatten)]
    pub(crate) extra: ExtraArgs,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(after_help = "The output path defaults to `<input-stem>.chat.jsonl`.")]
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
        Args, CliCommand, DataCommand, DatasetFormatArg, ModelFamilyArg, TrainCommand,
        TrainPretrainArgs,
    };
    use clap::{CommandFactory, Parser};

    #[test]
    fn rejects_removed_max_gpu_kv_flag() {
        let err = Args::try_parse_from(["agent-infer", "--max-gpu-kv", "256"])
            .err()
            .expect("removed flag should be rejected");
        let rendered = err.to_string();
        assert!(rendered.contains("--max-gpu-kv"));
    }

    #[test]
    fn rejects_removed_tools_flag() {
        let err = Args::try_parse_from(["agent-infer", "--tools"])
            .err()
            .expect("removed flag should be rejected");
        assert!(err.to_string().contains("--tools"));
    }

    #[test]
    fn rejects_zero_max_turns() {
        let err = Args::try_parse_from(["agent-infer", "--max-turns", "0"])
            .err()
            .expect("zero max-turns should be rejected");
        assert!(err.to_string().contains("at least 1"));
    }

    #[test]
    fn rejects_zero_max_tokens() {
        let err = Args::try_parse_from(["agent-infer", "--max-tokens", "0"])
            .err()
            .expect("zero max-tokens should be rejected");
        assert!(err.to_string().contains("at least 1"));
    }

    #[test]
    fn rejects_negative_temperature() {
        let err = Args::try_parse_from(["agent-infer", "--temperature", "-0.1"])
            .err()
            .expect("negative temperature should be rejected");
        assert!(err.to_string().contains("temperature must be >= 0.0"));
    }

    #[test]
    fn rejects_non_finite_temperature() {
        let err = Args::try_parse_from(["agent-infer", "--temperature", "NaN"])
            .err()
            .expect("NaN temperature should be rejected");
        assert!(err.to_string().contains("temperature must be finite"));
    }

    #[test]
    fn accepts_doctor_flag() {
        let args =
            Args::try_parse_from(["agent-infer", "--doctor"]).expect("doctor flag should parse");
        assert!(args.doctor);
    }

    #[test]
    fn accepts_list_models_flag() {
        let args = Args::try_parse_from(["agent-infer", "--list-models"])
            .expect("list-models flag should parse");
        assert!(args.list_models);
    }

    #[test]
    fn rejects_doctor_and_list_models_together() {
        let err = Args::try_parse_from(["agent-infer", "--doctor", "--list-models"])
            .err()
            .expect("doctor and list-models should conflict");
        assert!(err.to_string().contains("--list-models"));
    }

    #[test]
    fn accepts_doctor_json_flag() {
        let args = Args::try_parse_from(["agent-infer", "--doctor", "--json"])
            .expect("doctor json flag should parse");
        assert!(args.doctor);
        assert!(args.json);
    }

    #[test]
    fn accepts_doctor_strict_flag() {
        let args = Args::try_parse_from(["agent-infer", "--doctor", "--strict"])
            .expect("doctor strict flag should parse");
        assert!(args.doctor);
        assert!(args.strict);
    }

    #[test]
    fn accepts_list_models_json_flag() {
        let args = Args::try_parse_from(["agent-infer", "--list-models", "--json"])
            .expect("list-models json flag should parse");
        assert!(args.list_models);
        assert!(args.json);
    }

    #[test]
    fn rejects_json_without_inspection_mode() {
        let err = Args::try_parse_from(["agent-infer", "--json"])
            .err()
            .expect("--json without inspection mode should fail");
        assert!(err.to_string().contains("--doctor"));
    }

    #[test]
    fn rejects_strict_without_doctor() {
        let err = Args::try_parse_from(["agent-infer", "--strict"])
            .err()
            .expect("--strict without doctor should fail");
        assert!(err.to_string().contains("--doctor"));
    }

    #[test]
    fn rejects_strict_with_list_models() {
        let err = Args::try_parse_from(["agent-infer", "--list-models", "--strict"])
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
            "agent-infer",
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
    fn accepts_train_multi_turn_extra_args() {
        let args = Args::try_parse_from([
            "agent-infer",
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
            "agent-infer",
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
            "agent-infer",
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
    fn rejects_train_pretrain_json_without_dry_run() {
        let err = Args::try_parse_from([
            "agent-infer",
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

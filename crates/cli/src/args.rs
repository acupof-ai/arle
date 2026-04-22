use clap::{ArgGroup, Parser, Subcommand};

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

#[derive(Debug, Clone, Subcommand, PartialEq, Eq)]
pub(crate) enum CliCommand {
    /// Training jobs.
    Train(TrainArgs),
    /// Dataset utilities.
    Data(DataArgs),
}

#[derive(Debug, Clone, clap::Args, PartialEq, Eq)]
#[command(arg_required_else_help = true)]
pub(crate) struct TrainArgs {
    #[command(subcommand)]
    pub(crate) command: TrainCommand,
}

#[derive(Debug, Clone, Subcommand, PartialEq, Eq)]
pub(crate) enum TrainCommand {
    /// Scratch pretraining from a text corpus.
    Pretrain(ForwardedArgs),
    /// Supervised fine-tuning from chat JSONL.
    Sft(ForwardedArgs),
    /// Group-relative policy optimization.
    Grpo(ForwardedArgs),
    /// Multi-turn RL training.
    MultiTurn(ForwardedArgs),
    /// Evaluate a checkpoint on tokenized or chat JSONL.
    Eval(ForwardedArgs),
}

#[derive(Debug, Clone, clap::Args, PartialEq, Eq)]
#[command(arg_required_else_help = true)]
pub(crate) struct DataArgs {
    #[command(subcommand)]
    pub(crate) command: DataCommand,
}

#[derive(Debug, Clone, Subcommand, PartialEq, Eq)]
pub(crate) enum DataCommand {
    /// Download one dataset file from Hugging Face.
    Download(ForwardedArgs),
    /// Convert instruction-tuning JSONL into canonical chat JSONL.
    Convert(ForwardedArgs),
}

#[derive(Debug, Clone, clap::Args, PartialEq, Eq)]
pub(crate) struct ForwardedArgs {
    /// Remaining arguments are forwarded verbatim to the underlying training job.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub(crate) args: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::{Args, CliCommand, DataCommand, TrainCommand};
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
    fn accepts_train_pretrain_passthrough_args() {
        let args = Args::try_parse_from([
            "agent-infer",
            "train",
            "pretrain",
            "--corpus",
            "train.txt",
            "--tokenizer",
            "tok.json",
            "--out",
            "out",
        ])
        .expect("train pretrain should parse");
        let Some(CliCommand::Train(train)) = args.command else {
            panic!("expected train command");
        };
        let TrainCommand::Pretrain(forwarded) = train.command else {
            panic!("expected pretrain command");
        };
        assert_eq!(
            forwarded.args,
            [
                "--corpus",
                "train.txt",
                "--tokenizer",
                "tok.json",
                "--out",
                "out"
            ]
        );
    }

    #[test]
    fn accepts_train_multi_turn_passthrough_args() {
        let args = Args::try_parse_from([
            "agent-infer",
            "train",
            "multi-turn",
            "--iters",
            "2",
            "--backend",
            "metal",
        ])
        .expect("train multi-turn should parse");
        let Some(CliCommand::Train(train)) = args.command else {
            panic!("expected train command");
        };
        let TrainCommand::MultiTurn(forwarded) = train.command else {
            panic!("expected multi-turn command");
        };
        assert_eq!(forwarded.args, ["--iters", "2", "--backend", "metal"]);
    }

    #[test]
    fn accepts_data_convert_passthrough_args() {
        let args = Args::try_parse_from([
            "agent-infer",
            "data",
            "convert",
            "--input",
            "raw.jsonl",
            "--format",
            "dolly",
            "--output",
            "chat.jsonl",
        ])
        .expect("data convert should parse");
        let Some(CliCommand::Data(data)) = args.command else {
            panic!("expected data command");
        };
        let DataCommand::Convert(forwarded) = data.command else {
            panic!("expected convert command");
        };
        assert_eq!(
            forwarded.args,
            [
                "--input",
                "raw.jsonl",
                "--format",
                "dolly",
                "--output",
                "chat.jsonl",
            ]
        );
    }
}

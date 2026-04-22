use std::{
    env,
    path::{Path, PathBuf},
    process::ExitCode,
};

use autograd::{AutogradError, Tape, TensorStore};
use thiserror::Error;
use train::{
    CausalLm, GrpoPolicyConfig,
    causal_lm::build_registry,
    cli_args::{ArgError, BackendChoice, next_value, parse_value},
    eval_lm::{evaluate_examples, load_eval_examples},
    metrics::{MetricSample, TrainEvent, default_run_id, open_shared_sink},
    model_family::{ModelFamily, ModelFamilyError, resolve_model_family},
    qwen3::{Qwen3Config, Qwen3ConfigError, Qwen3Error, Qwen3Model},
    qwen35::{Qwen35Config, Qwen35ConfigError, Qwen35Error, Qwen35Model},
};

#[derive(Debug, Clone)]
struct CliArgs {
    model_family: ModelFamily,
    model_path: PathBuf,
    data: PathBuf,
    tokenizer: Option<PathBuf>,
    seq_len: usize,
    backend: BackendChoice,
    metrics_jsonl: Option<PathBuf>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            model_family: ModelFamily::Auto,
            model_path: PathBuf::new(),
            data: PathBuf::new(),
            tokenizer: None,
            seq_len: 1024,
            backend: BackendChoice::Cpu,
            metrics_jsonl: None,
        }
    }
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Arg(#[from] ArgError),
    #[error(transparent)]
    ModelFamily(#[from] ModelFamilyError),
    #[error(transparent)]
    Qwen3(#[from] Qwen3Error),
    #[error(transparent)]
    Qwen3Config(#[from] Qwen3ConfigError),
    #[error(transparent)]
    Qwen35(#[from] Qwen35Error),
    #[error(transparent)]
    Qwen35Config(#[from] Qwen35ConfigError),
    #[error(transparent)]
    EvalLm(#[from] train::eval_lm::EvalLmError),
    #[error("{0}")]
    Custom(String),
}

trait EvalFamily {
    type Config: GrpoPolicyConfig + Clone;
    type Model: CausalLm<Config = Self::Config>;

    fn family_name() -> &'static str;
    fn load_config(config_path: &Path) -> Result<Self::Config, CliError>;
    fn build_model(cfg: &Self::Config, store: &mut TensorStore) -> Result<Self::Model, CliError>;
}

struct Qwen3Family;

impl EvalFamily for Qwen3Family {
    type Config = Qwen3Config;
    type Model = Qwen3Model;

    fn family_name() -> &'static str {
        "qwen3"
    }

    fn load_config(config_path: &Path) -> Result<Self::Config, CliError> {
        Qwen3Config::from_json_file(config_path).map_err(Into::into)
    }

    fn build_model(cfg: &Self::Config, store: &mut TensorStore) -> Result<Self::Model, CliError> {
        Qwen3Model::new(cfg, store).map_err(Into::into)
    }
}

struct Qwen35Family;

impl EvalFamily for Qwen35Family {
    type Config = Qwen35Config;
    type Model = Qwen35Model;

    fn family_name() -> &'static str {
        "qwen35"
    }

    fn load_config(config_path: &Path) -> Result<Self::Config, CliError> {
        Qwen35Config::from_json_file(config_path).map_err(Into::into)
    }

    fn build_model(cfg: &Self::Config, store: &mut TensorStore) -> Result<Self::Model, CliError> {
        Qwen35Model::new_for_eval(cfg, store).map_err(Into::into)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[eval_lm] error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), CliError> {
    let args = parse_args()?;
    run_with_args(args)
}

#[allow(dead_code)]
pub(crate) fn dispatch_from_args<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let parsed = parse_args_from(args.into_iter()).map_err(|err| err.to_string())?;
    run_with_args(parsed).map_err(|err| err.to_string())
}

fn run_with_args(args: CliArgs) -> Result<(), CliError> {
    validate_args(&args)?;

    let config_path = args.model_path.join("config.json");
    let resolved_family = resolve_model_family(&config_path, args.model_family)?;
    match resolved_family {
        ModelFamily::Qwen35 => run_with_family::<Qwen35Family>(&args, &config_path),
        ModelFamily::Qwen3 => run_with_family::<Qwen3Family>(&args, &config_path),
        ModelFamily::Auto => unreachable!("auto must resolve to a concrete family"),
    }
}

fn run_with_family<F: EvalFamily>(args: &CliArgs, config_path: &Path) -> Result<(), CliError> {
    let cfg = F::load_config(config_path)?;
    let tokenizer_path = args
        .tokenizer
        .as_ref()
        .cloned()
        .unwrap_or_else(|| args.model_path.join("tokenizer.json"));
    let tokenizer_path = tokenizer_path.is_file().then_some(tokenizer_path);

    let mut store = TensorStore::with_backend(args.backend.build_backend_or_cpu("eval_lm")?);
    let model = F::build_model(&cfg, &mut store)?;
    let mut registry = build_registry(&model);
    let weights_path = args.model_path.join("model.safetensors");
    registry.load_into(&mut store, &weights_path)?;

    let examples = load_eval_examples(&args.data, tokenizer_path.as_deref(), args.seq_len)?;
    let mut tape = Tape::new();
    let summary = evaluate_examples(&model, &examples, &mut store, &mut tape)?;

    let output = serde_json::json!({
        "loss": summary.loss,
        "ppl": summary.ppl(),
        "tokens": summary.token_count,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);

    if let Some(metrics_jsonl) = args.metrics_jsonl.as_deref() {
        let sink = open_shared_sink(Some(metrics_jsonl), false)
            .map_err(|err| CliError::Custom(format!("metrics sink: {err}")))?;
        let run_id = default_run_id("eval_lm");
        let backend_name = args.backend.as_str();
        let model_path_string = args.model_path.display().to_string();
        let data_path_string = args.data.display().to_string();
        let run_start_strings = [
            ("run_id", run_id.as_str()),
            ("job", "eval_lm"),
            ("model_family", F::family_name()),
            ("backend", backend_name),
            ("model_path", model_path_string.as_str()),
            ("data", data_path_string.as_str()),
        ];
        let run_start_scalars = [("seq_len", args.seq_len as f64)];
        sink.emit_event(&TrainEvent {
            kind: "run_start",
            step: Some(0),
            strings: &run_start_strings,
            scalars: &run_start_scalars,
            bools: &[],
        });
        let fields = [
            ("eval_loss", summary.loss),
            ("eval_ppl", summary.ppl()),
            ("eval_tokens", summary.token_count as f64),
        ];
        sink.emit_metric(&MetricSample {
            step: 0,
            phase: "eval",
            fields: &fields,
        });
        let run_end_strings = [("run_id", run_id.as_str()), ("status", "completed")];
        let run_end_scalars = [
            ("eval_loss", summary.loss),
            ("eval_ppl", summary.ppl()),
            ("eval_tokens", summary.token_count as f64),
        ];
        sink.emit_event(&TrainEvent {
            kind: "run_end",
            step: Some(0),
            strings: &run_end_strings,
            scalars: &run_end_scalars,
            bools: &[],
        });
        sink.flush_blocking();
    }

    eprintln!(
        "[eval_lm] family={} loss={:.6} ppl={:.6} tokens={}",
        F::family_name(),
        summary.loss,
        summary.ppl(),
        summary.token_count
    );

    Ok(())
}

fn parse_args() -> Result<CliArgs, CliError> {
    parse_args_from(env::args().skip(1))
}

fn parse_args_from<I>(mut iter: I) -> Result<CliArgs, CliError>
where
    I: Iterator<Item = String>,
{
    let mut args = CliArgs::default();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model-family" => {
                args.model_family = next_value(&mut iter, &flag)?
                    .parse()
                    .map_err(|value| CliError::Arg(ArgError::InvalidValue { flag, value }))?;
            }
            "--model-path" | "--model" => {
                args.model_path = PathBuf::from(next_value(&mut iter, &flag)?)
            }
            "--data" => args.data = PathBuf::from(next_value(&mut iter, &flag)?),
            "--tokenizer" => args.tokenizer = Some(PathBuf::from(next_value(&mut iter, &flag)?)),
            "--seq-len" => args.seq_len = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--backend" => {
                args.backend = next_value(&mut iter, &flag)?
                    .parse()
                    .map_err(|value| CliError::Arg(ArgError::InvalidValue { flag, value }))?;
            }
            "--metrics-jsonl" => {
                args.metrics_jsonl = Some(PathBuf::from(next_value(&mut iter, &flag)?));
            }
            _ => return Err(CliError::Arg(ArgError::UnknownFlag(flag))),
        }
    }
    Ok(args)
}

fn validate_args(args: &CliArgs) -> Result<(), CliError> {
    if args.model_path.as_os_str().is_empty() {
        return Err(CliError::Custom("--model-path is required".into()));
    }
    if args.data.as_os_str().is_empty() {
        return Err(CliError::Custom("--data is required".into()));
    }
    if args.seq_len == 0 {
        return Err(CliError::Arg(ArgError::InvalidValue {
            flag: "--seq-len".into(),
            value: "0".into(),
        }));
    }
    Ok(())
}

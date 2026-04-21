use std::{
    env,
    path::{Path, PathBuf},
    process::ExitCode,
    str::FromStr,
    sync::Arc,
};

use autograd::{AutogradError, Backend, CpuBackend, Tape, TensorStore};
use thiserror::Error;
use train::{
    CausalLm, GrpoPolicyConfig,
    causal_lm::build_registry,
    cli_args::{ArgError, next_value, parse_value},
    eval_lm::{evaluate_examples, load_eval_examples},
    metrics::{MetricSample, TrainEvent, default_run_id, open_shared_sink},
    model_family::{ModelFamily, ModelFamilyError, resolve_model_family},
    qwen3::{Qwen3Config, Qwen3ConfigError, Qwen3Error, Qwen3Model},
    qwen35::{Qwen35Config, Qwen35ConfigError, Qwen35Error, Qwen35Model},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Cpu,
    Metal,
    Cuda,
}

impl FromStr for BackendChoice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => Err(format!("unknown backend: {value}")),
        }
    }
}

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
        Qwen35Model::new(cfg, store).map_err(Into::into)
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

    let mut store = TensorStore::with_backend(build_backend(args.backend)?);
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
        let backend_name = match args.backend {
            BackendChoice::Cpu => "cpu",
            BackendChoice::Metal => "metal",
            BackendChoice::Cuda => "cuda",
        };
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
    let mut args = CliArgs::default();
    let mut iter = env::args().skip(1);
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

fn build_backend(choice: BackendChoice) -> Result<Arc<dyn Backend>, CliError> {
    match choice {
        BackendChoice::Cpu => Ok(Arc::new(CpuBackend)),
        #[cfg(feature = "metal")]
        BackendChoice::Metal => Ok(Arc::new(autograd::backend_metal::MetalBackend)),
        #[cfg(not(feature = "metal"))]
        BackendChoice::Metal => {
            eprintln!(
                "[eval_lm] warning: metal backend requested without --features metal; falling back to cpu"
            );
            Ok(Arc::new(CpuBackend))
        }
        #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
        BackendChoice::Cuda => Ok(Arc::new(autograd::backend_cuda::CudaBackend::new(0)?)),
        #[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
        BackendChoice::Cuda => {
            eprintln!(
                "[eval_lm] warning: cuda backend requested without --features cuda; falling back to cpu"
            );
            Ok(Arc::new(CpuBackend))
        }
    }
}

use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Instant,
};

use autograd::{
    AutogradError, Backend, CpuBackend, SafetensorsRegistry, Tape, TensorId, TensorStore,
    ops::{gather_last_dim, log_softmax, mul, mul_scalar, sum},
    optim::AdamW,
};
use thiserror::Error;
use train::{
    dataset::LcgRng,
    qwen3::{Qwen3Config, Qwen3Error, Qwen3Model},
    sft_data::{TokenizedSft, load_jsonl, tokenize_example},
    tokenizer::ChatTokenizer,
};

const DEFAULT_BETAS: (f32, f32) = (0.9, 0.999);
const DEFAULT_EPS: f32 = 1.0e-8;
const DEFAULT_WEIGHT_DECAY: f32 = 0.01;

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

// bf16 is the default because infer/'s DeviceMatrix::from_safetensors
// reinterprets the safetensors bytes as `&[bf16]`. f32 is kept for bit-exact
// debugging — it round-trips inside autograd but infer will reject it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SaveDtype {
    F32,
    Bf16,
}

impl FromStr for SaveDtype {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "f32" => Ok(Self::F32),
            "bf16" => Ok(Self::Bf16),
            _ => Err(format!("unknown save dtype: {value}")),
        }
    }
}

#[derive(Debug, Clone)]
struct CliArgs {
    model: PathBuf,
    data: PathBuf,
    out: PathBuf,
    steps: usize,
    batch: usize,
    lr: f32,
    seq_len: usize,
    backend: BackendChoice,
    save_every: usize,
    log_every: usize,
    seed: u64,
    save_dtype: SaveDtype,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            data: PathBuf::new(),
            out: PathBuf::new(),
            steps: 100,
            batch: 1,
            lr: 2.0e-5,
            seq_len: 1024,
            backend: BackendChoice::Cpu,
            save_every: 50,
            log_every: 1,
            seed: 0,
            save_dtype: SaveDtype::Bf16,
        }
    }
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Qwen3(#[from] Qwen3Error),
    #[error("unknown flag {0}")]
    UnknownFlag(String),
    #[error("missing value for flag {0}")]
    MissingValue(String),
    #[error("invalid value for {flag}: {value}")]
    InvalidValue { flag: String, value: String },
    #[error("{0}")]
    Custom(String),
}

fn main() -> Result<(), CliError> {
    let args = parse_args()?;
    validate_args(&args)?;
    fs::create_dir_all(&args.out)?;

    let config_path = args.model.join("config.json");
    let tokenizer_path = args.model.join("tokenizer.json");
    let weights_path = args.model.join("model.safetensors");
    let shard_index = args.model.join("model.safetensors.index.json");
    if shard_index.exists() {
        eprintln!(
            "[train_sft] warning: sharded safetensors ({}) are not supported yet",
            shard_index.display()
        );
        return Err(CliError::Custom(
            "only single-file model.safetensors checkpoints are supported".into(),
        ));
    }

    let cfg = Qwen3Config::from_json_file(&config_path)?;
    let tokenizer = ChatTokenizer::from_file(&tokenizer_path)?;
    let mut store = TensorStore::with_backend(build_backend(args.backend)?);
    let model = Qwen3Model::new(&cfg, &mut store)?;
    let mut registry = build_registry(&model);
    registry.load_into(&mut store, &weights_path)?;

    let model_ids = live_tensor_ids(&store);
    let params = trainable_params(&model, &store);
    if params.is_empty() {
        return Err(CliError::Custom(
            "qwen3 model exposed no trainable parameters".into(),
        ));
    }

    let raw_examples = load_jsonl(&args.data)?;
    let dataset = raw_examples
        .iter()
        .enumerate()
        .filter_map(|(index, example)| match tokenize_example(example, &tokenizer, args.seq_len) {
            Ok(tokenized) if tokenized.input_ids.len() >= 2 && has_supervised_target(&tokenized) => {
                Some(Ok(tokenized))
            }
            Ok(_) => {
                eprintln!(
                    "[train_sft] warning: skipping example {} with no supervised assistant tokens after truncation",
                    index
                );
                None
            }
            Err(err) => Some(Err(CliError::Autograd(err))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dataset.is_empty() {
        return Err(CliError::Custom(
            "no usable SFT examples remained after tokenization".into(),
        ));
    }

    let mut optimizer = AdamW::new(args.lr, DEFAULT_BETAS, DEFAULT_EPS, DEFAULT_WEIGHT_DECAY);
    let mut tape = Tape::new();
    let mut rng = LcgRng::seed(args.seed);

    for step in 0..args.steps {
        let step_start = Instant::now();
        optimizer.zero_grad(&params, &mut store);

        let mut step_loss = 0.0_f32;
        // Each micro-step's loss is already averaged over its own supervised
        // tokens, so the gradient accumulated by tape.backward() is the
        // per-example gradient. Scale by 1/batch before backward so the
        // summed gradient equals the mean across micro-steps — otherwise
        // the effective learning rate grows linearly with --batch.
        let loss_scale = 1.0_f32 / args.batch.max(1) as f32;
        for micro_step in 0..args.batch {
            let example_index = sample_index(&mut rng, dataset.len(), step, micro_step);
            step_loss += train_on_example(
                &model,
                &dataset[example_index],
                &mut store,
                &mut tape,
                &cfg,
                loss_scale,
            )?;

            tape.entries.clear();
            tape.set_enabled(true);
            let keep = retained_ids(&model_ids, &params, &store);
            store.retain_ids(&keep);
        }

        optimizer.step(&params, &mut store);

        tape.entries.clear();
        tape.set_enabled(true);
        let keep = retained_ids(&model_ids, &params, &store);
        store.retain_ids(&keep);

        let mean_loss = step_loss / args.batch as f32;
        if step % args.log_every == 0 || step + 1 == args.steps {
            println!(
                "step={} loss={mean_loss:.6} lr={} ms={:.2}",
                step + 1,
                args.lr,
                step_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        if (step + 1) % args.save_every == 0 || step + 1 == args.steps {
            save_checkpoint(
                &args.model,
                &args.out,
                step + 1,
                &registry,
                &mut store,
                &config_path,
                &tokenizer_path,
                args.save_dtype,
            )?;
        }
    }

    Ok(())
}

fn parse_args() -> Result<CliArgs, CliError> {
    let mut args = CliArgs::default();
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model" => args.model = PathBuf::from(next_value(&mut iter, &flag)?),
            "--data" => args.data = PathBuf::from(next_value(&mut iter, &flag)?),
            "--out" => args.out = PathBuf::from(next_value(&mut iter, &flag)?),
            "--steps" => args.steps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--batch" => args.batch = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--lr" => args.lr = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--seq-len" => args.seq_len = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--backend" => {
                args.backend = next_value(&mut iter, &flag)?
                    .parse()
                    .map_err(|value| CliError::InvalidValue { flag, value })?;
            }
            "--save-every" => {
                args.save_every = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--log-every" => {
                args.log_every = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--seed" => args.seed = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--save-dtype" => {
                args.save_dtype = next_value(&mut iter, &flag)?
                    .parse()
                    .map_err(|value| CliError::InvalidValue { flag, value })?;
            }
            _ => return Err(CliError::UnknownFlag(flag)),
        }
    }

    Ok(args)
}

fn validate_args(args: &CliArgs) -> Result<(), CliError> {
    if args.model.as_os_str().is_empty() {
        return Err(CliError::Custom("--model is required".into()));
    }
    if args.data.as_os_str().is_empty() {
        return Err(CliError::Custom("--data is required".into()));
    }
    if args.out.as_os_str().is_empty() {
        return Err(CliError::Custom("--out is required".into()));
    }
    for (flag, value) in [
        ("--steps", args.steps),
        ("--batch", args.batch),
        ("--seq-len", args.seq_len),
        ("--save-every", args.save_every),
        ("--log-every", args.log_every),
    ] {
        if value == 0 {
            return Err(CliError::InvalidValue {
                flag: flag.into(),
                value: "0".into(),
            });
        }
    }
    Ok(())
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, CliError> {
    iter.next()
        .ok_or_else(|| CliError::MissingValue(flag.to_string()))
}

fn parse_value<T>(flag: &str, value: String) -> Result<T, CliError>
where
    T: FromStr,
{
    value.parse::<T>().map_err(|_| CliError::InvalidValue {
        flag: flag.to_string(),
        value,
    })
}

fn build_backend(choice: BackendChoice) -> Result<Arc<dyn Backend>, CliError> {
    match choice {
        BackendChoice::Cpu => Ok(Arc::new(CpuBackend)),
        #[cfg(feature = "metal")]
        BackendChoice::Metal => Ok(Arc::new(autograd::backend_metal::MetalBackend)),
        #[cfg(not(feature = "metal"))]
        BackendChoice::Metal => {
            eprintln!(
                "[train_sft] warning: metal backend requested without --features metal; falling back to cpu"
            );
            Ok(Arc::new(CpuBackend))
        }
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => Ok(Arc::new(autograd::backend_cuda::CudaBackend::new(0)?)),
        #[cfg(not(feature = "cuda"))]
        BackendChoice::Cuda => {
            eprintln!(
                "[train_sft] warning: cuda backend requested without --features cuda; falling back to cpu"
            );
            Ok(Arc::new(CpuBackend))
        }
    }
}

fn build_registry(model: &Qwen3Model) -> SafetensorsRegistry {
    let mut registry = SafetensorsRegistry::new();
    for (name, tensor_id) in model.param_name_map() {
        registry.insert(name, tensor_id);
    }
    registry
}

fn trainable_params(model: &Qwen3Model, store: &TensorStore) -> Vec<TensorId> {
    let mut params = model
        .param_name_map()
        .into_values()
        .collect::<HashSet<_>>()
        .into_iter()
        .filter(|tensor_id| {
            store
                .get(*tensor_id)
                .is_some_and(|tensor| tensor.requires_grad)
        })
        .collect::<Vec<_>>();
    params.sort_unstable();
    params
}

fn train_on_example(
    model: &Qwen3Model,
    example: &TokenizedSft,
    store: &mut TensorStore,
    tape: &mut Tape,
    cfg: &Qwen3Config,
    loss_scale: f32,
) -> Result<f32, CliError> {
    let input_len = example.input_ids.len() - 1;
    let position_ids = (0..input_len).map(|index| index as u32).collect::<Vec<_>>();
    let logits = model.forward(store, tape, &example.input_ids[..input_len], &position_ids)?;
    let loss_id = assistant_masked_causal_loss(logits, &example.labels[1..], store, tape, cfg)?;
    let loss = store.to_host(loss_id)?[0];
    // Scale the loss used for backward so the accumulated per-example
    // gradients average (rather than sum) across micro-steps. Reported loss
    // is the unscaled value so log output remains human-meaningful.
    let backward_id = if (loss_scale - 1.0).abs() > f32::EPSILON {
        mul_scalar(loss_id, loss_scale, store, tape)?
    } else {
        loss_id
    };
    tape.backward(backward_id, store)?;
    Ok(loss)
}

fn assistant_masked_causal_loss(
    logits: TensorId,
    labels: &[i32],
    store: &mut TensorStore,
    tape: &mut Tape,
    cfg: &Qwen3Config,
) -> Result<TensorId, CliError> {
    let logits_shape = store
        .get(logits)
        .ok_or(AutogradError::InvalidTensorId(logits))?
        .shape
        .clone();
    let target_count = logits_shape
        .iter()
        .take(logits_shape.len() - 1)
        .product::<usize>();
    if labels.len() != target_count {
        return Err(CliError::Custom(format!(
            "shifted label count {} does not match logits prefix size {}",
            labels.len(),
            target_count
        )));
    }

    let valid_count = labels.iter().filter(|&&label| label >= 0).count();
    if valid_count == 0 {
        return Err(CliError::Custom(
            "example has no supervised assistant tokens after shifting".into(),
        ));
    }

    let gather_indices = labels
        .iter()
        .map(|&label| match usize::try_from(label) {
            Ok(index) if index < cfg.vocab_size => Ok(index),
            Ok(index) => Err(CliError::Custom(format!(
                "label {index} is outside vocab size {}",
                cfg.vocab_size
            ))),
            Err(_) => Ok(0),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mask_values = labels
        .iter()
        .map(|&label| if label >= 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();

    let log_probs = log_softmax(logits, store, tape)?;
    let target_log_probs = gather_last_dim(log_probs, &gather_indices, store, tape)?;
    let target_shape = store
        .get(target_log_probs)
        .ok_or(AutogradError::InvalidTensorId(target_log_probs))?
        .shape
        .clone();
    let mask = store.from_slice(&mask_values, &target_shape)?;
    let masked = mul(target_log_probs, mask, store, tape)?;
    let total = sum(masked, store, tape)?;
    Ok(mul_scalar(total, -1.0 / valid_count as f32, store, tape)?)
}

fn save_checkpoint(
    model_dir: &Path,
    out_dir: &Path,
    step: usize,
    registry: &SafetensorsRegistry,
    store: &mut TensorStore,
    config_path: &Path,
    tokenizer_path: &Path,
    save_dtype: SaveDtype,
) -> Result<(), CliError> {
    let step_dir = out_dir.join(format!("step_{step}"));
    fs::create_dir_all(&step_dir)?;
    fs::copy(config_path, step_dir.join("config.json"))?;
    fs::copy(tokenizer_path, step_dir.join("tokenizer.json"))?;
    let weights_path = step_dir.join("model.safetensors");
    match save_dtype {
        SaveDtype::F32 => registry.save_from(store, &weights_path)?,
        SaveDtype::Bf16 => registry.save_from_bf16(store, &weights_path)?,
    }
    println!(
        "[train_sft] saved checkpoint for step {} to {} (source model dir: {}, dtype: {:?})",
        step,
        step_dir.display(),
        model_dir.display(),
        save_dtype
    );
    Ok(())
}

fn has_supervised_target(example: &TokenizedSft) -> bool {
    example.labels.iter().skip(1).any(|&label| label >= 0)
}

fn sample_index(rng: &mut LcgRng, upper: usize, step: usize, micro_step: usize) -> usize {
    if upper <= 1 {
        return 0;
    }
    let mix = rng
        .next_u64()
        .wrapping_add(step as u64)
        .wrapping_add((micro_step as u64) << 32);
    (mix % upper as u64) as usize
}

fn live_tensor_ids(store: &TensorStore) -> HashSet<TensorId> {
    store
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(tensor_id, slot)| slot.as_ref().map(|_| tensor_id))
        .collect()
}

fn retained_ids(
    model_ids: &HashSet<TensorId>,
    params: &[TensorId],
    store: &TensorStore,
) -> HashSet<TensorId> {
    let mut keep = model_ids.clone();
    for &param_id in params {
        keep.insert(param_id);
        if let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) {
            keep.insert(grad_id);
        }
    }
    keep
}

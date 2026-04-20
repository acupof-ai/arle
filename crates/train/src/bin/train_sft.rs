use std::{
    cell::RefCell,
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
    rc::Rc,
    str::FromStr,
    sync::Arc,
};

use autograd::{
    AutogradError, Backend, CpuBackend, SafetensorsRegistry, Tape, TensorId, TensorStore,
    ops::{gather_last_dim, log_softmax, mul, mul_scalar, sum},
    optim::AdamW,
};
use thiserror::Error;
use train::{
    StepOutcome, Trainer, TrainerConfig,
    cli_args::{ArgError, next_value, parse_value},
    dataset::LcgRng,
    grad_clip::NoClip,
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
    // Phase 2 acceptance flags (docs/plans/train-runtime-architecture-v1.md §9).
    lr_schedule: String,
    warmup_steps: u64,
    min_lr: f32,
    grad_accum_steps: Option<usize>,
    metrics_jsonl: Option<PathBuf>,
    resume_from: Option<PathBuf>,
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
            lr_schedule: "constant".to_string(),
            warmup_steps: 0,
            min_lr: 0.0,
            grad_accum_steps: None,
            metrics_jsonl: None,
            resume_from: None,
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
    #[error(transparent)]
    Arg(#[from] ArgError),
    #[error("{0}")]
    Custom(String),
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        // Display (not Debug): thiserror's #[error(transparent)] already
        // delegates to the inner error's Display impl, so io errors surface as
        // "No such file or directory (os error 2)" instead of the Debug blob
        // "Os { code: 2, kind: NotFound, message: \"...\" }".
        Err(err) => {
            eprintln!("[train_sft] error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), CliError> {
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
    let param_names = model
        .param_name_map()
        .into_iter()
        .filter(|(_, id)| params.contains(id))
        .map(|(name, id)| (id, name.to_string()))
        .collect::<Vec<_>>();

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

    // --- Trainer setup (Wave 3 migration) ---
    let optim = AdamW::new(args.lr, DEFAULT_BETAS, DEFAULT_EPS, DEFAULT_WEIGHT_DECAY);
    // `--grad-accum-steps` overrides `--batch` when passed; otherwise the
    // pre-migration semantics ("batch folds grads into a single optim step")
    // hold. max(1) to keep the zero-check above happy even if a user passes 0.
    let grad_accum = args.grad_accum_steps.unwrap_or(args.batch).max(1) as u64;
    let schedule: Box<dyn autograd::LrSchedule> = autograd::parse_lr_schedule(
        &args.lr_schedule,
        args.lr,
        args.warmup_steps,
        args.steps as u64,
        args.min_lr,
    )
    .map_err(|e| CliError::Custom(format!("bad --lr-schedule: {e}")))?;
    let metrics = train::metrics::open_sink(args.metrics_jsonl.as_deref(), true)
        .map_err(|e| CliError::Custom(format!("metrics sink: {e}")))?;
    // Enable the Trainer's built-in save path so every checkpoint round
    // gets `trainer_state.json + optimizer.safetensors` written next to the
    // bf16 model weights the on_step_end hook produces — without this wiring,
    // `--resume-from` could not reload optimizer state from our own output
    // (codex review 2026-04-20 on ad5568b, P1). Trainer lays files out at
    // `<save_dir>/step_{:06}/`; the hook below matches that format.
    let trainer_cfg = TrainerConfig {
        total_steps: args.steps as u64,
        grad_accum_steps: grad_accum,
        log_every: args.log_every.max(1) as u64,
        eval_every: None,
        save_every: Some(args.save_every as u64),
        save_dir: Some(args.out.clone()),
        resume_from: args.resume_from.clone(),
        rng_seed: args.seed,
    };
    let mut trainer = Trainer::new(optim, NoClip, schedule, metrics, trainer_cfg);

    // Resume if `--resume-from` was passed.
    //
    // Codex review 2026-04-20 on ad5568b (P1): `resume_if_configured` only
    // restores optimizer state + step counter — the Trainer is
    // architecture-agnostic and does not know how to reload model weights.
    // So we first overwrite the base `--model` weights with the checkpoint's
    // `model.safetensors`, then call the trainer-side restore. Without the
    // explicit weight reload, a resumed run would combine base-model
    // weights with resumed Adam moments + step state — a corrupt resume.
    if let Some(resume_dir) = &args.resume_from {
        let resume_weights = resume_dir.join("model.safetensors");
        if !resume_weights.is_file() {
            return Err(CliError::Custom(format!(
                "--resume-from {} has no model.safetensors",
                resume_dir.display(),
            )));
        }
        registry.load_into(&mut store, &resume_weights)?;
        let resumed = trainer
            .resume_if_configured(&param_names)
            .map_err(CliError::Autograd)?;
        eprintln!(
            "[train_sft] resumed from step {resumed} (weights from {})",
            resume_weights.display()
        );
    }

    let mut tape = Tape::new();
    let rng = Rc::new(RefCell::new(LcgRng::seed(args.seed)));
    let dataset_len = dataset.len();
    let total_steps = args.steps;
    let save_every = args.save_every;
    let save_dtype = args.save_dtype;

    // Step closure: forward + assistant-masked cross-entropy. Trainer handles
    // the `1/batch` loss-scale + backward + optimizer.step + cleanup.
    let rng_for_step = Rc::clone(&rng);
    let model_ref = &model;
    let cfg_ref = &cfg;
    let dataset_ref = &dataset;
    let step_fn = |ctx: &mut train::StepCtx<'_>| -> autograd::Result<StepOutcome> {
        let example_index = {
            let mut rng = rng_for_step.borrow_mut();
            sample_index(
                &mut rng,
                dataset_len,
                ctx.step as usize,
                ctx.micro_idx as usize,
            )
        };
        let example = &dataset_ref[example_index];
        let input_len = example.input_ids.len() - 1;
        let position_ids = (0..input_len).map(|index| index as u32).collect::<Vec<_>>();
        let logits = model_ref
            .forward(
                ctx.store,
                ctx.tape,
                &example.input_ids[..input_len],
                &position_ids,
            )
            .map_err(autograd_from_qwen3)?;
        let loss_id = assistant_masked_causal_loss(
            logits,
            &example.labels[1..],
            ctx.store,
            ctx.tape,
            cfg_ref,
        )?;
        Ok(StepOutcome {
            loss_id,
            token_count: input_len as u64,
        })
    };

    // Post-step hook: dump bf16/f32 model weights to `<out>/step_<N>/` on
    // `save_every` boundaries (and always on the final step). Mirrors the
    // pre-migration behavior byte-for-byte.
    let model_path = args.model.clone();
    let out_path = args.out.clone();
    let cfg_path = config_path.clone();
    let tok_path = tokenizer_path.clone();
    let registry_ref = &registry;
    let on_step_end = |step: u64, store: &mut TensorStore| -> autograd::Result<()> {
        let step_usize = step as usize;
        // Gate matches the Trainer's save_every + force-final behavior so the
        // bf16 weights file always lands in the same `step_{:06}/` directory
        // that the Trainer just populated with `trainer_state.json +
        // optimizer.safetensors`. Keep the two gates in sync.
        if step_usize.is_multiple_of(save_every) || step_usize == total_steps {
            save_checkpoint_via_registry(
                &model_path,
                &out_path,
                step_usize,
                registry_ref,
                store,
                &cfg_path,
                &tok_path,
                save_dtype,
            )
            .map_err(autograd_from_cli)?;
        }
        Ok(())
    };

    trainer
        .run_with_hooks(
            &mut store,
            &mut tape,
            params,
            param_names,
            model_ids,
            step_fn,
            on_step_end,
        )
        .map_err(CliError::Autograd)?;

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
                    .map_err(|value| CliError::Arg(ArgError::InvalidValue { flag, value }))?;
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
                    .map_err(|value| CliError::Arg(ArgError::InvalidValue { flag, value }))?;
            }
            "--lr-schedule" => {
                args.lr_schedule = next_value(&mut iter, &flag)?;
            }
            "--warmup-steps" => {
                args.warmup_steps = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--min-lr" => {
                args.min_lr = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--grad-accum-steps" => {
                args.grad_accum_steps = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--metrics-jsonl" => {
                args.metrics_jsonl = Some(PathBuf::from(next_value(&mut iter, &flag)?));
            }
            "--resume-from" => {
                args.resume_from = Some(PathBuf::from(next_value(&mut iter, &flag)?));
            }
            _ => return Err(CliError::Arg(ArgError::UnknownFlag(flag))),
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
            return Err(CliError::Arg(ArgError::InvalidValue {
                flag: flag.into(),
                value: "0".into(),
            }));
        }
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

fn assistant_masked_causal_loss(
    logits: TensorId,
    labels: &[i32],
    store: &mut TensorStore,
    tape: &mut Tape,
    cfg: &Qwen3Config,
) -> autograd::Result<TensorId> {
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
        return Err(leak_autograd_err(format!(
            "train_sft: shifted label count {} does not match logits prefix size {}",
            labels.len(),
            target_count
        )));
    }

    let valid_count = labels.iter().filter(|&&label| label >= 0).count();
    if valid_count == 0 {
        return Err(AutogradError::TapeInvariant(
            "train_sft: example has no supervised assistant tokens after shifting",
        ));
    }

    let mut gather_indices: Vec<usize> = Vec::with_capacity(labels.len());
    for &label in labels {
        let idx = match usize::try_from(label) {
            Ok(index) if index < cfg.vocab_size => index,
            Ok(index) => {
                return Err(leak_autograd_err(format!(
                    "train_sft: label {index} is outside vocab size {}",
                    cfg.vocab_size
                )));
            }
            Err(_) => 0,
        };
        gather_indices.push(idx);
    }
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
    mul_scalar(total, -1.0 / valid_count as f32, store, tape)
}

/// Leak a `String` into a `&'static str` for `AutogradError::TapeInvariant`.
/// Matches the pattern in `autograd/src/safetensors_io.rs` — fires at most
/// once per run because the error immediately propagates up and aborts.
fn leak_autograd_err(msg: String) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(msg.into_boxed_str()))
}

fn autograd_from_qwen3(err: Qwen3Error) -> AutogradError {
    leak_autograd_err(format!("train_sft: qwen3 forward: {err}"))
}

fn autograd_from_cli(err: CliError) -> AutogradError {
    leak_autograd_err(format!("train_sft: checkpoint save: {err}"))
}

fn save_checkpoint_via_registry(
    model_dir: &Path,
    out_dir: &Path,
    step: usize,
    registry: &SafetensorsRegistry,
    store: &mut TensorStore,
    config_path: &Path,
    tokenizer_path: &Path,
    save_dtype: SaveDtype,
) -> Result<(), CliError> {
    // Zero-padded 6-digit format matches the Trainer's own save_checkpoint
    // path (`step_{:06}`), so the bf16 weights file lands next to
    // `trainer_state.json + optimizer.safetensors` in a single directory —
    // required for `--resume-from` to roundtrip correctly (codex review
    // 2026-04-20 on ad5568b, P1).
    let step_dir = out_dir.join(format!("step_{step:06}"));
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

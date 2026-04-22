use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
    time::Instant,
};

#[cfg(test)]
use autograd::ops::{mul_scalar, sum};
use autograd::{
    AutogradError, Tape, TensorId, TensorStore,
    ops::{gather_last_dim, log_softmax, matmul, mean, mul},
    optim::AdamW,
};
use thiserror::Error;
use train::{
    CausalLm, GrpoPolicy, StepOutcome, Trainer, TrainerConfig,
    causal_lm::{
        build_adapter_registry, build_registry, live_tensor_ids, save_materialized_registry,
        trainable_params,
    },
    cli_args::{ArgError, BackendChoice, SaveDtype, next_value, parse_value},
    control::{
        TrainingController, emit_run_end, emit_run_start, open_run_metrics, serve_if_requested,
        sync_status,
    },
    grad_clip::NoClip,
    lora::{LoraAdapterConfig, LoraConfig},
    model_family::{ModelFamily, ModelFamilyError, resolve_model_family},
    qwen3::{Qwen3Config, Qwen3ConfigError, Qwen3Error, Qwen3Model},
    qwen3_checkpoint::{
        ConfigJsonSource, GenerationConfigSource, Qwen3CheckpointError, Qwen3StepCheckpoint,
        save_step_checkpoint,
    },
    qwen35::{Qwen35Config, Qwen35ConfigError, Qwen35Error, Qwen35Model},
    qwen35_checkpoint::{
        ConfigJsonSource as Qwen35ConfigJsonSource,
        GenerationConfigSource as Qwen35GenerationConfigSource, Qwen35CheckpointError,
        Qwen35StepCheckpoint, save_step_checkpoint as save_qwen35_step_checkpoint,
    },
    sft_data::{TokenizedSft, load_jsonl, tokenize_example},
    tokenizer::ChatTokenizer,
};

const DEFAULT_BETAS: (f32, f32) = (0.9, 0.999);
const DEFAULT_EPS: f32 = 1.0e-8;
const DEFAULT_WEIGHT_DECAY: f32 = 0.01;
const STOP_REQUESTED_ERR: &str = "train_sft: operator stop requested";

#[derive(Debug, Clone)]
struct CliArgs {
    model_family: ModelFamily,
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
    lora_rank: usize,
    lora_alpha: f32,
    serve: Option<u16>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            model_family: ModelFamily::Auto,
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
            lora_rank: 16,
            lora_alpha: 32.0,
            serve: None,
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
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Qwen3Checkpoint(#[from] Qwen3CheckpointError),
    #[error(transparent)]
    Qwen3(#[from] Qwen3Error),
    #[error(transparent)]
    Qwen3Config(#[from] Qwen3ConfigError),
    #[error(transparent)]
    Qwen35Checkpoint(#[from] Qwen35CheckpointError),
    #[error(transparent)]
    Qwen35(#[from] Qwen35Error),
    #[error(transparent)]
    Qwen35Config(#[from] Qwen35ConfigError),
    #[error(transparent)]
    ModelFamily(#[from] ModelFamilyError),
    #[error(transparent)]
    Arg(#[from] ArgError),
    #[error("{0}")]
    Custom(String),
}

trait SftFamily {
    type Config: train::GrpoPolicyConfig + Clone;
    type Model: CausalLm<Config = Self::Config>;

    fn family_name() -> &'static str;
    fn load_config(config_path: &Path) -> Result<Self::Config, CliError>;
    fn build_model(
        cfg: &Self::Config,
        lora: LoraConfig,
        store: &mut TensorStore,
    ) -> Result<Self::Model, CliError>;
    fn validate_resume_config(resume_dir: &Path, cfg: &Self::Config) -> Result<(), CliError>;
    fn save_checkpoint(
        model_dir: &Path,
        out_dir: &Path,
        step: usize,
        model: &Self::Model,
        store: &mut TensorStore,
        config_path: &Path,
        tokenizer_path: &Path,
        save_dtype: SaveDtype,
        lora: LoraConfig,
    ) -> Result<(), CliError>;
}

struct Qwen3Family;

impl SftFamily for Qwen3Family {
    type Config = Qwen3Config;
    type Model = Qwen3Model;

    fn family_name() -> &'static str {
        "qwen3"
    }

    fn load_config(config_path: &Path) -> Result<Self::Config, CliError> {
        Qwen3Config::from_json_file(config_path).map_err(Into::into)
    }

    fn build_model(
        cfg: &Self::Config,
        lora: LoraConfig,
        store: &mut TensorStore,
    ) -> Result<Self::Model, CliError> {
        Qwen3Model::new_with_lora(cfg, Some(lora), store).map_err(Into::into)
    }

    fn validate_resume_config(resume_dir: &Path, cfg: &Self::Config) -> Result<(), CliError> {
        validate_qwen3_resume_config(resume_dir, cfg)
    }

    fn save_checkpoint(
        model_dir: &Path,
        out_dir: &Path,
        step: usize,
        model: &Self::Model,
        store: &mut TensorStore,
        config_path: &Path,
        tokenizer_path: &Path,
        save_dtype: SaveDtype,
        lora: LoraConfig,
    ) -> Result<(), CliError> {
        save_qwen3_checkpoint(
            model_dir,
            out_dir,
            step,
            model,
            store,
            config_path,
            tokenizer_path,
            save_dtype,
            lora,
        )
    }
}

struct Qwen35Family;

impl SftFamily for Qwen35Family {
    type Config = Qwen35Config;
    type Model = Qwen35Model;

    fn family_name() -> &'static str {
        "qwen35"
    }

    fn load_config(config_path: &Path) -> Result<Self::Config, CliError> {
        Qwen35Config::from_json_file(config_path).map_err(Into::into)
    }

    fn build_model(
        cfg: &Self::Config,
        lora: LoraConfig,
        store: &mut TensorStore,
    ) -> Result<Self::Model, CliError> {
        Qwen35Model::new_with_lora(cfg, Some(lora), store).map_err(Into::into)
    }

    fn validate_resume_config(resume_dir: &Path, cfg: &Self::Config) -> Result<(), CliError> {
        validate_qwen35_resume_config(resume_dir, cfg)
    }

    fn save_checkpoint(
        model_dir: &Path,
        out_dir: &Path,
        step: usize,
        model: &Self::Model,
        store: &mut TensorStore,
        config_path: &Path,
        tokenizer_path: &Path,
        save_dtype: SaveDtype,
        lora: LoraConfig,
    ) -> Result<(), CliError> {
        save_qwen35_checkpoint(
            model_dir,
            out_dir,
            step,
            model,
            store,
            config_path,
            tokenizer_path,
            save_dtype,
            lora,
        )
    }
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
    let config_path = args.model.join("config.json");

    match resolve_model_family(&config_path, args.model_family)? {
        ModelFamily::Qwen35 => run_with_family::<Qwen35Family>(&args, &config_path),
        ModelFamily::Qwen3 => run_with_family::<Qwen3Family>(&args, &config_path),
        ModelFamily::Auto => unreachable!("auto must resolve to a concrete family"),
    }
}

fn run_with_family<F: SftFamily>(args: &CliArgs, config_path: &Path) -> Result<(), CliError> {
    fs::create_dir_all(&args.out)?;
    let lora = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };

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

    let cfg = F::load_config(config_path)?;
    let tokenizer = ChatTokenizer::from_file(&tokenizer_path)?;
    let mut store = TensorStore::with_backend(args.backend.build_backend_or_cpu("train_sft")?);
    let model = F::build_model(&cfg, lora, &mut store)?;
    let mut registry = build_registry(&model);
    registry.load_into(&mut store, &weights_path)?;

    let ones_by_seq = build_prefix_ones(&mut store, args.seq_len)?;
    let mut model_ids = live_tensor_ids(&store);
    model_ids.extend(ones_by_seq.iter().copied());
    let params = trainable_params(&model, &store);
    if params.is_empty() {
        return Err(CliError::Custom(format!(
            "{} model exposed no trainable parameters",
            F::family_name()
        )));
    }
    let param_names = trainable_param_names(&model, &params);

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
    // `--batch` is the true per-forward batch size; `--grad-accum-steps`
    // layers extra accumulation on top only when explicitly requested.
    let batch_size = args.batch.max(1);
    let grad_accum = args.grad_accum_steps.unwrap_or(1).max(1) as u64;
    let schedule: Box<dyn autograd::LrSchedule> = autograd::parse_lr_schedule(
        &args.lr_schedule,
        args.lr,
        args.warmup_steps,
        args.steps as u64,
        args.min_lr,
    )
    .map_err(|e| CliError::Custom(format!("bad --lr-schedule: {e}")))?;
    let controller = TrainingController::new();
    let metrics = open_run_metrics(args.metrics_jsonl.as_deref(), &controller)
        .map_err(|e| CliError::Custom(format!("metrics sink: {e}")))?;
    let run_id = train::metrics::default_run_id("train_sft");
    let run_timer = Instant::now();
    let backend_name = args.backend.as_str();

    // DX-1 follow-up (codex review 2026-04-20 on 8bde810, High): canonicalize
    // --resume-from once, up-front, so every subsequent read (weights, config,
    // trainer_state, optimizer.safetensors loaded via Trainer::resume_if_configured)
    // targets the same snapshot of the `latest` symlink. Without this, a
    // concurrent trainer repointing `latest` between our opens could let us
    // mix step N weights with step N+1 trainer/optimizer state.
    let resume_dir_canonical: Option<PathBuf> = match &args.resume_from {
        Some(resume_dir) => Some(resume_dir.canonicalize().map_err(|e| {
            CliError::Custom(format!(
                "failed to canonicalize --resume-from {}: {e} \
                 (is the path / symlink target missing?)",
                resume_dir.display()
            ))
        })?),
        None => None,
    };
    let model_path_string = args.model.display().to_string();
    let out_path_string = args.out.display().to_string();
    let resume_dir_string = resume_dir_canonical
        .as_ref()
        .map(|path| path.display().to_string());
    let mut run_start_strings = vec![
        ("model_family", F::family_name()),
        ("backend", backend_name),
        ("model", model_path_string.as_str()),
        ("out", out_path_string.as_str()),
    ];
    if let Some(path) = resume_dir_string.as_deref() {
        run_start_strings.push(("resume_from", path));
    }
    let run_start_scalars = [
        ("total_steps", args.steps as f64),
        ("batch_size", batch_size as f64),
        ("grad_accum_steps", grad_accum as f64),
        ("seq_len", args.seq_len as f64),
    ];
    let run_start_bools = [("resumed", resume_dir_canonical.is_some())];
    emit_run_start(
        &metrics,
        &run_id,
        "train_sft",
        0,
        &run_start_strings,
        &run_start_scalars,
        &run_start_bools,
    );
    sync_status(&controller, &metrics, |status| {
        status.total_iters = args.steps;
        status.started = true;
    });
    let _server_handle =
        serve_if_requested("train_sft", &controller, args.serve).map_err(CliError::Custom)?;

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
        resume_from: resume_dir_canonical.clone(),
        rng_seed: args.seed,
    };
    let mut trainer = Trainer::new(
        optim,
        NoClip,
        schedule,
        Box::new(metrics.clone()),
        trainer_cfg,
    );

    // Resume if `--resume-from` was passed.
    //
    // `resume_if_configured` only restores optimizer state + step counter —
    // the Trainer is architecture-agnostic and does not know how to reload
    // model weights. For LoRA-only SFT we keep the frozen base weights from
    // `--model` and reload the adapter tensors from the resume snapshot.
    //
    // The shape/config validation below still matters because a checkpoint
    // with mismatched architecture would otherwise crash on the next forward
    // pass or silently mis-route the optimizer state.
    if let Some(resume_dir) = resume_dir_canonical.as_deref() {
        F::validate_resume_config(resume_dir, &cfg)?;
        validate_adapter_resume_config(resume_dir, &args.model, F::family_name(), lora)?;
        let resume_adapter = resume_dir.join("adapter_model.safetensors");
        if !resume_adapter.is_file() {
            return Err(CliError::Custom(format!(
                "--resume-from {} has no adapter_model.safetensors",
                resume_dir.display(),
            )));
        }
        let mut adapter_registry = build_adapter_registry(&model);
        if adapter_registry.is_empty() {
            return Err(CliError::Custom(format!(
                "{} model exposed no LoRA adapter tensors",
                F::family_name()
            )));
        }
        adapter_registry.load_into_strict(&mut store, &resume_adapter)?;
        let resumed = trainer
            .resume_if_configured(&param_names)
            .map_err(CliError::Autograd)?;
        eprintln!(
            "[train_sft] resumed from step {resumed} (adapters from {})",
            resume_adapter.display()
        );
    }

    let mut tape = Tape::new();
    let dataset_len = dataset.len();
    let total_steps = args.steps;
    let save_every = args.save_every;
    let save_dtype = args.save_dtype;
    let seed = args.seed;
    let position_ids = (0..args.seq_len).collect::<Vec<_>>();

    // Step closure: forward + assistant-masked cross-entropy. Trainer handles
    // any extra `--grad-accum-steps` scaling plus backward/step/cleanup.
    //
    // Codex review 2026-04-20 on 49512b1 (#2, Medium): sample_index is
    // keyed on `(seed, step, micro_idx)` alone — NOT on a shared stateful
    // RNG. A stateful RNG would reset to position 0 on `--resume-from`
    // (we don't persist RNG state in the checkpoint codec), causing data
    // order to diverge from the interrupted run. With this derivation,
    // step/micro_idx is a natural resume cursor: a resumed run picks up
    // the same sequence it would have seen in a single uninterrupted run.
    let model_ref = &model;
    let dataset_ref = &dataset;
    let ones_by_seq_ref = &ones_by_seq;
    let mut batch_examples = Vec::with_capacity(batch_size);
    let mut collated = BatchedTokenizedSft::with_capacity(batch_size, args.seq_len);
    let mut input_ids = Vec::with_capacity(batch_size * args.seq_len);
    let step_fn = |ctx: &mut train::StepCtx<'_>| -> autograd::Result<StepOutcome> {
        batch_examples.clear();
        for row in 0..batch_size {
            let example_index = sample_index(
                seed,
                dataset_len,
                ctx.step as usize,
                (ctx.micro_idx as usize * batch_size) + row,
            );
            batch_examples.push(&dataset_ref[example_index]);
        }
        collate_tokenized_batch_into(&mut collated, &batch_examples);
        input_ids.clear();
        input_ids.extend(collated.input_ids.iter().map(|&token_id| token_id as usize));
        let logits = model_ref.forward_batch_tokens_with_positions(
            &input_ids,
            &position_ids[..collated.seq_len],
            batch_examples.len(),
            ctx.store,
            ctx.tape,
        )?;
        let loss_id = assistant_masked_causal_loss_batch_precomputed(
            logits,
            &collated.gather_indices,
            &collated.mask_values,
            &collated.inv_counts,
            batch_examples.len(),
            collated.seq_len,
            ones_by_seq_ref[collated.seq_len],
            ctx.store,
            ctx.tape,
        )?;
        Ok(StepOutcome {
            loss_id,
            token_count: collated.token_count,
        })
    };

    // Post-step hook: dump bf16/f32 model weights to `<out>/step_<N>/` on
    // `save_every` boundaries (and always on the final step). Mirrors the
    // pre-migration behavior byte-for-byte.
    let model_path = args.model.clone();
    let out_path = args.out.clone();
    let cfg_path = config_path;
    let tok_path = tokenizer_path.clone();
    let model_save_ref = &model;
    let metrics_for_hooks = metrics.clone();
    let controller_for_hooks = Arc::clone(&controller);
    let on_step_end = |step: u64, store: &mut TensorStore| -> autograd::Result<()> {
        let step_usize = step as usize;
        let save_requested = controller_for_hooks.take_save_request();
        // Gate matches the Trainer's save_every + force-final behavior so the
        // bf16 weights file always lands in the same `step_{:06}/` directory
        // that the Trainer just populated with `trainer_state.json +
        // optimizer.safetensors`. Keep the two gates in sync.
        if step_usize.is_multiple_of(save_every) || step_usize == total_steps || save_requested {
            F::save_checkpoint(
                &model_path,
                &out_path,
                step_usize,
                model_save_ref,
                store,
                cfg_path,
                &tok_path,
                save_dtype,
                lora,
            )
            .map_err(autograd_from_cli)?;
            let checkpoint_dir = out_path.join(format!("step_{step:06}"));
            let checkpoint_dir_string = checkpoint_dir.display().to_string();
            let strings = [
                ("path", checkpoint_dir_string.as_str()),
                ("artifact_model", "model.safetensors"),
                ("artifact_adapter", "adapter_model.safetensors"),
                ("artifact_adapter_config", "adapter_config.json"),
                ("artifact_config", "config.json"),
                ("artifact_generation_config", "generation_config.json"),
                ("artifact_tokenizer", "tokenizer.json"),
            ];
            metrics_for_hooks.emit_event(&train::metrics::TrainEvent {
                kind: "checkpoint",
                step: Some(step),
                strings: &strings,
                scalars: &[],
                bools: &[],
            });
        }
        sync_status(&controller_for_hooks, &metrics_for_hooks, |status| {
            status.iter = step as usize;
            status.wall_secs = run_timer.elapsed().as_secs_f32();
        });
        if controller_for_hooks.should_stop() {
            return Err(AutogradError::TapeInvariant(STOP_REQUESTED_ERR));
        }
        Ok(())
    };

    let run_result = trainer
        .run_with_hooks(
            &mut store,
            &mut tape,
            params,
            param_names,
            model_ids,
            step_fn,
            on_step_end,
        )
        .map_err(CliError::Autograd);

    let stopped = match run_result {
        Ok(()) => false,
        Err(CliError::Autograd(AutogradError::TapeInvariant(msg))) if msg == STOP_REQUESTED_ERR => {
            true
        }
        Err(err) => return Err(err),
    };

    let run_end_scalars = [
        ("completed_steps", trainer.step() as f64),
        ("dropped_metrics", metrics.dropped_metrics() as f64),
    ];
    let status = if stopped { "stopped" } else { "completed" };
    emit_run_end(&metrics, &run_id, status, trainer.step(), &run_end_scalars);
    sync_status(&controller, &metrics, |summary| {
        summary.iter = trainer.step() as usize;
        summary.total_iters = args.steps;
        summary.wall_secs = run_timer.elapsed().as_secs_f32();
        summary.finished = true;
    });
    metrics.flush_blocking();

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
            "--serve" => {
                args.serve = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--resume-from" => {
                args.resume_from = Some(PathBuf::from(next_value(&mut iter, &flag)?));
            }
            "--lora-rank" => {
                args.lora_rank = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--lora-alpha" => {
                args.lora_alpha = parse_value(&flag, next_value(&mut iter, &flag)?)?;
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
        ("--lora-rank", args.lora_rank),
    ] {
        if value == 0 {
            return Err(CliError::Arg(ArgError::InvalidValue {
                flag: flag.into(),
                value: "0".into(),
            }));
        }
    }
    if !(args.lora_alpha.is_finite() && args.lora_alpha > 0.0) {
        return Err(CliError::Arg(ArgError::InvalidValue {
            flag: "--lora-alpha".into(),
            value: args.lora_alpha.to_string(),
        }));
    }
    Ok(())
}

struct BatchedTokenizedSft {
    input_ids: Vec<u32>,
    gather_indices: Vec<usize>,
    mask_values: Vec<f32>,
    inv_counts: Vec<f32>,
    seq_len: usize,
    token_count: u64,
}

impl BatchedTokenizedSft {
    fn with_capacity(batch: usize, seq_len: usize) -> Self {
        Self {
            input_ids: Vec::with_capacity(batch * seq_len),
            gather_indices: Vec::with_capacity(batch * seq_len),
            mask_values: Vec::with_capacity(batch * seq_len),
            inv_counts: Vec::with_capacity(batch),
            seq_len: 0,
            token_count: 0,
        }
    }
}

fn collate_tokenized_batch_into(out: &mut BatchedTokenizedSft, examples: &[&TokenizedSft]) {
    let seq_len = examples
        .iter()
        .map(|example| example.input_ids.len().saturating_sub(1))
        .max()
        .unwrap_or(0)
        .max(1);
    let batch_elems = examples.len() * seq_len;
    out.input_ids.clear();
    out.input_ids.resize(batch_elems, 0);
    out.gather_indices.clear();
    out.mask_values.clear();
    out.inv_counts.clear();
    out.gather_indices.resize(batch_elems, 0);
    out.mask_values.resize(batch_elems, 0.0);
    out.inv_counts.resize(examples.len(), 0.0);
    out.seq_len = seq_len;
    out.token_count = 0;

    for (row, example) in examples.iter().enumerate() {
        let input_len = example.input_ids.len().saturating_sub(1);
        let base = row * seq_len;
        out.input_ids[base..base + input_len].copy_from_slice(&example.input_ids[..input_len]);
        let row_labels = &example.labels[1..1 + input_len];
        let valid_count = row_labels.iter().filter(|&&label| label >= 0).count();
        if valid_count > 0 {
            out.inv_counts[row] = -1.0 / valid_count as f32;
        }
        for (offset, &label) in row_labels.iter().enumerate() {
            if let Ok(index) = usize::try_from(label) {
                out.gather_indices[base + offset] = index;
                out.mask_values[base + offset] = 1.0;
            }
        }
        out.token_count += input_len as u64;
    }
}

fn build_prefix_ones(
    store: &mut TensorStore,
    max_seq_len: usize,
) -> autograd::Result<Vec<TensorId>> {
    let mut ones_by_seq = Vec::with_capacity(max_seq_len + 1);
    for seq_len in 0..=max_seq_len {
        let len = seq_len.max(1);
        ones_by_seq.push(store.from_slice(&vec![1.0f32; len], &[len, 1])?);
    }
    Ok(ones_by_seq)
}

#[cfg(test)]
fn assistant_masked_causal_loss(
    logits: TensorId,
    labels: &[i32],
    store: &mut TensorStore,
    tape: &mut Tape,
    vocab_size: usize,
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
            Ok(index) if index < vocab_size => index,
            Ok(index) => {
                return Err(leak_autograd_err(format!(
                    "train_sft: label {index} is outside vocab size {vocab_size}",
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

#[cfg(test)]
fn assistant_masked_causal_loss_batch(
    logits: TensorId,
    labels: &[i32],
    batch: usize,
    seq_len: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
    vocab_size: usize,
) -> autograd::Result<TensorId> {
    if labels.len() != batch * seq_len {
        return Err(leak_autograd_err(format!(
            "train_sft: batched label count {} does not match batch*seq {}",
            labels.len(),
            batch * seq_len
        )));
    }

    let mut gather_indices = Vec::with_capacity(labels.len());
    let mut mask_values = Vec::with_capacity(labels.len());
    let mut inv_counts = Vec::with_capacity(batch);
    for row in 0..batch {
        let row_labels = &labels[row * seq_len..(row + 1) * seq_len];
        let valid_count = row_labels.iter().filter(|&&label| label >= 0).count();
        if valid_count == 0 {
            return Err(AutogradError::TapeInvariant(
                "train_sft: batched example has no supervised assistant tokens after shifting",
            ));
        }
        inv_counts.push(-1.0 / valid_count as f32);
        for &label in row_labels {
            let idx = match usize::try_from(label) {
                Ok(index) if index < vocab_size => index,
                Ok(index) => {
                    return Err(leak_autograd_err(format!(
                        "train_sft: label {index} is outside vocab size {vocab_size}",
                    )));
                }
                Err(_) => 0,
            };
            gather_indices.push(idx);
            mask_values.push(if label >= 0 { 1.0 } else { 0.0 });
        }
    }

    let ones = store.from_slice(&vec![1.0f32; seq_len.max(1)], &[seq_len.max(1), 1])?;
    assistant_masked_causal_loss_batch_precomputed(
        logits,
        &gather_indices,
        &mask_values,
        &inv_counts,
        batch,
        seq_len,
        ones,
        store,
        tape,
    )
}

fn assistant_masked_causal_loss_batch_precomputed(
    logits: TensorId,
    gather_indices: &[usize],
    mask_values: &[f32],
    inv_counts: &[f32],
    batch: usize,
    seq_len: usize,
    ones: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> autograd::Result<TensorId> {
    if gather_indices.len() != batch * seq_len || mask_values.len() != batch * seq_len {
        return Err(leak_autograd_err(format!(
            "train_sft: precomputed buffers do not match batch*seq {}",
            batch * seq_len
        )));
    }
    if inv_counts.len() != batch {
        return Err(leak_autograd_err(format!(
            "train_sft: inv_counts len {} does not match batch {}",
            inv_counts.len(),
            batch
        )));
    }
    if inv_counts.iter().any(|&count| count >= 0.0) {
        return Err(AutogradError::TapeInvariant(
            "train_sft: batched example has no supervised assistant tokens after shifting",
        ));
    }

    let log_probs = log_softmax(logits, store, tape)?;
    let target_log_probs = gather_last_dim(log_probs, gather_indices, store, tape)?;
    let mask = store.from_slice(mask_values, &[batch, seq_len])?;
    let masked = mul(target_log_probs, mask, store, tape)?;
    let row_sums = matmul(masked, ones, store, tape)?;
    let inv_counts = store.from_slice(inv_counts, &[batch, 1])?;
    let per_example_loss = mul(row_sums, inv_counts, store, tape)?;
    mean(per_example_loss, store, tape)
}

/// Leak a `String` into a `&'static str` for `AutogradError::TapeInvariant`.
/// Matches the pattern in `autograd/src/safetensors_io.rs` — fires at most
/// once per run because the error immediately propagates up and aborts.
fn leak_autograd_err(msg: String) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(msg.into_boxed_str()))
}

fn autograd_from_cli(err: CliError) -> AutogradError {
    leak_autograd_err(format!("train_sft: checkpoint save: {err}"))
}

/// Verify the resume dir's `config.json` matches the live `--model` config
/// on the dimensions that determine tensor shapes — hidden size, layer
/// count, attention shape, vocab. A silent shape mismatch would otherwise
/// surface as a mid-training crash on the first forward pass; per the
/// codex review of 49512b1 (#1 High), fail fast before the run starts.
///
/// Missing `config.json` is a hard error: codex review 429efc3 (Medium)
/// flagged that silently skipping the config-match check when cfg.json
/// is absent reopens the tied/untied silent-corruption path this
/// function is meant to close. Older checkpoints that need to resume
/// can regenerate their `config.json` from the live `Qwen3Config` and
/// drop it into the resume dir.
fn validate_qwen3_resume_config(resume_dir: &Path, cfg: &Qwen3Config) -> Result<(), CliError> {
    let cfg_path = resume_dir.join("config.json");
    if !cfg_path.exists() {
        return Err(CliError::Custom(format!(
            "--resume-from {} has no config.json; refuse to resume without a config-match check \
             (would otherwise silently miss tie_word_embeddings / shape drift). \
             Regenerate config.json from the source model or fresh-start without --resume-from.",
            resume_dir.display()
        )));
    }
    let file_cfg: serde_json::Value = serde_json::from_str(&fs::read_to_string(&cfg_path)?)
        .map_err(|e| CliError::Custom(format!("resume config.json parse error: {e}")))?;
    let mut mismatches: Vec<String> = [
        ("hidden_size", cfg.hidden_size as i64),
        ("intermediate_size", cfg.intermediate_size as i64),
        ("num_hidden_layers", cfg.num_hidden_layers as i64),
        ("num_attention_heads", cfg.num_attention_heads as i64),
        ("num_key_value_heads", cfg.num_key_value_heads as i64),
        ("head_dim", cfg.head_dim as i64),
        ("vocab_size", cfg.vocab_size as i64),
        (
            "max_position_embeddings",
            cfg.max_position_embeddings as i64,
        ),
    ]
    .iter()
    .filter_map(|(k, v)| match file_cfg.get(*k).and_then(|x| x.as_i64()) {
        Some(seen) if seen != *v => Some(format!("{k}: ckpt={seen} live={v}")),
        _ => None,
    })
    .collect();

    // Codex review 2026-04-20 on d9eee61 (High): `tie_word_embeddings`
    // changes the live parameter map — when true, `embed_tokens.weight`
    // and `lm_head.weight` alias to the same `TensorId` in `param_ids`,
    // when false they are distinct. An untied checkpoint resumed against
    // a tied live config would pass the shape check above, then
    // `load_into_strict` would load two different file tensors into the
    // same live TensorId with the second one silently winning. Treat the
    // flag as a shape-determining field so the run fails fast.
    if let Some(saw) = file_cfg
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
    {
        if saw != cfg.tie_word_embeddings {
            mismatches.push(format!(
                "tie_word_embeddings: ckpt={saw} live={}",
                cfg.tie_word_embeddings
            ));
        }
    }

    if let Some(seen) = file_cfg.get("rope_theta").and_then(|v| v.as_f64()) {
        if seen != cfg.rope_theta as f64 {
            mismatches.push(format!("rope_theta: ckpt={seen} live={}", cfg.rope_theta));
        }
    }

    if !mismatches.is_empty() {
        return Err(CliError::Custom(format!(
            "--resume-from {} config mismatch with --model: {}",
            resume_dir.display(),
            mismatches.join(", ")
        )));
    }
    Ok(())
}

fn validate_qwen35_resume_config(resume_dir: &Path, cfg: &Qwen35Config) -> Result<(), CliError> {
    let cfg_path = resume_dir.join("config.json");
    if !cfg_path.exists() {
        return Err(CliError::Custom(format!(
            "--resume-from {} has no config.json; refuse to resume without a config-match check",
            resume_dir.display()
        )));
    }
    let file_cfg = Qwen35Config::from_json_file(&cfg_path).map_err(|err| {
        CliError::Custom(format!(
            "resume config {} does not parse as qwen3.5-family: {err}",
            cfg_path.display()
        ))
    })?;

    let mut mismatches = Vec::new();
    for (name, live, seen) in [
        ("hidden_size", cfg.hidden_size, file_cfg.hidden_size),
        (
            "intermediate_size",
            cfg.intermediate_size,
            file_cfg.intermediate_size,
        ),
        (
            "num_hidden_layers",
            cfg.num_hidden_layers,
            file_cfg.num_hidden_layers,
        ),
        (
            "num_attention_heads",
            cfg.num_attention_heads,
            file_cfg.num_attention_heads,
        ),
        (
            "num_key_value_heads",
            cfg.num_key_value_heads,
            file_cfg.num_key_value_heads,
        ),
        ("head_dim", cfg.head_dim, file_cfg.head_dim),
        ("vocab_size", cfg.vocab_size, file_cfg.vocab_size),
        (
            "linear_num_key_heads",
            cfg.linear_num_key_heads,
            file_cfg.linear_num_key_heads,
        ),
        (
            "linear_key_head_dim",
            cfg.linear_key_head_dim,
            file_cfg.linear_key_head_dim,
        ),
        (
            "linear_num_value_heads",
            cfg.linear_num_value_heads,
            file_cfg.linear_num_value_heads,
        ),
        (
            "linear_value_head_dim",
            cfg.linear_value_head_dim,
            file_cfg.linear_value_head_dim,
        ),
        (
            "linear_conv_kernel_dim",
            cfg.linear_conv_kernel_dim,
            file_cfg.linear_conv_kernel_dim,
        ),
    ] {
        if live != seen {
            mismatches.push(format!("{name}: ckpt={seen} live={live}"));
        }
    }
    if cfg.tie_word_embeddings != file_cfg.tie_word_embeddings {
        mismatches.push(format!(
            "tie_word_embeddings: ckpt={} live={}",
            file_cfg.tie_word_embeddings, cfg.tie_word_embeddings
        ));
    }
    if cfg.rope_theta != file_cfg.rope_theta {
        mismatches.push(format!(
            "rope_theta: ckpt={} live={}",
            file_cfg.rope_theta, cfg.rope_theta
        ));
    }
    if cfg.partial_rotary_factor != file_cfg.partial_rotary_factor {
        mismatches.push(format!(
            "partial_rotary_factor: ckpt={} live={}",
            file_cfg.partial_rotary_factor, cfg.partial_rotary_factor
        ));
    }
    if cfg.rope_cache_len_hint != file_cfg.rope_cache_len_hint {
        mismatches.push(format!(
            "rope_cache_len_hint: ckpt={:?} live={:?}",
            file_cfg.rope_cache_len_hint, cfg.rope_cache_len_hint
        ));
    }
    if cfg.layer_types != file_cfg.layer_types {
        mismatches.push("layer_types mismatch".to_string());
    }
    if !mismatches.is_empty() {
        return Err(CliError::Custom(format!(
            "--resume-from {} config mismatch with --model: {}",
            resume_dir.display(),
            mismatches.join(", ")
        )));
    }
    Ok(())
}

fn validate_adapter_resume_config(
    resume_dir: &Path,
    model_dir: &Path,
    family: &str,
    lora: LoraConfig,
) -> Result<(), CliError> {
    let adapter_config_path = resume_dir.join("adapter_config.json");
    if !adapter_config_path.is_file() {
        return Err(CliError::Custom(format!(
            "--resume-from {} has no adapter_config.json",
            resume_dir.display()
        )));
    }
    let value: LoraAdapterConfig = serde_json::from_str(&fs::read_to_string(&adapter_config_path)?)
        .map_err(|err| {
            CliError::Custom(format!(
                "adapter config {} parse error: {err}",
                adapter_config_path.display()
            ))
        })?;
    let mut mismatches = Vec::new();
    if value.model_family != family {
        mismatches.push(format!(
            "model_family: ckpt={} live={family}",
            value.model_family
        ));
    }
    if value.peft_type != "LORA" {
        mismatches.push(format!("peft_type: ckpt={} live=LORA", value.peft_type));
    }
    if value.task_type != "CAUSAL_LM" {
        mismatches.push(format!(
            "task_type: ckpt={} live=CAUSAL_LM",
            value.task_type
        ));
    }
    if value.r != lora.rank {
        mismatches.push(format!("r: ckpt={} live={}", value.r, lora.rank));
    }
    if (value.lora_alpha - lora.alpha).abs() > 1.0e-6 {
        mismatches.push(format!(
            "lora_alpha: ckpt={} live={}",
            value.lora_alpha, lora.alpha
        ));
    }
    if value.target_modules != ["all-linear"] {
        mismatches.push(format!(
            "target_modules: ckpt={:?} live=[\"all-linear\"]",
            value.target_modules
        ));
    }
    if !same_model_path(&value.base_model_name_or_path, model_dir) {
        mismatches.push(format!(
            "base_model_name_or_path: ckpt={} live={}",
            value.base_model_name_or_path,
            model_dir.display()
        ));
    }
    if !mismatches.is_empty() {
        return Err(CliError::Custom(format!(
            "--resume-from {} adapter mismatch: {}",
            resume_dir.display(),
            mismatches.join(", ")
        )));
    }
    Ok(())
}

fn same_model_path(saved: &str, live: &Path) -> bool {
    if saved == live.display().to_string() {
        return true;
    }
    let Ok(saved_path) = PathBuf::from(saved).canonicalize() else {
        return false;
    };
    let Ok(live_path) = live.canonicalize() else {
        return false;
    };
    saved_path == live_path
}

fn save_adapter_artifacts<M: CausalLm>(
    step_dir: &Path,
    model: &M,
    store: &mut TensorStore,
    model_dir: &Path,
    family: &str,
    lora: LoraConfig,
) -> Result<(), CliError> {
    let adapter_registry = build_adapter_registry(model);
    if adapter_registry.is_empty() {
        return Ok(());
    }
    adapter_registry.save_from(store, &step_dir.join("adapter_model.safetensors"))?;
    let adapter_config = LoraAdapterConfig::new(model_dir.display().to_string(), family, lora);
    fs::write(
        step_dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&adapter_config)?,
    )?;
    Ok(())
}

fn trainable_param_names<M: CausalLm>(model: &M, params: &[TensorId]) -> Vec<(TensorId, String)> {
    let param_set = params.iter().copied().collect::<HashSet<_>>();
    let mut names = model
        .adapter_name_map()
        .into_iter()
        .filter_map(|(name, id)| param_set.contains(&id).then_some((id, name.to_string())))
        .collect::<Vec<_>>();
    names.sort_unstable_by(|(id_a, name_a), (id_b, name_b)| {
        name_a.cmp(name_b).then_with(|| id_a.cmp(id_b))
    });
    names
}

fn save_qwen3_checkpoint(
    model_dir: &Path,
    out_dir: &Path,
    step: usize,
    model: &Qwen3Model,
    store: &mut TensorStore,
    config_path: &Path,
    tokenizer_path: &Path,
    save_dtype: SaveDtype,
    lora: LoraConfig,
) -> Result<(), CliError> {
    let keep_ids = live_tensor_ids(store);
    let step_dir = save_step_checkpoint(
        Qwen3StepCheckpoint {
            out_dir,
            step,
            tokenizer_path: Some(tokenizer_path),
            config_json: ConfigJsonSource::CopyFrom(config_path),
            generation_config: GenerationConfigSource::CopyOrSynthesize {
                source_path: &model_dir.join("generation_config.json"),
                fallback_config_path: config_path,
            },
        },
        |weights_path| {
            let mut tape = Tape::new();
            save_materialized_registry(
                model,
                store,
                &mut tape,
                weights_path,
                matches!(save_dtype, SaveDtype::Bf16),
            )
            .map_err(Into::into)
        },
    )?;
    store.retain_ids(&keep_ids);
    save_adapter_artifacts(&step_dir, model, store, model_dir, "qwen3", lora)?;

    println!(
        "[train_sft] saved checkpoint for step {} to {} (source model dir: {}, dtype: {:?})",
        step,
        step_dir.display(),
        model_dir.display(),
        save_dtype
    );
    Ok(())
}

fn save_qwen35_checkpoint(
    model_dir: &Path,
    out_dir: &Path,
    step: usize,
    model: &Qwen35Model,
    store: &mut TensorStore,
    config_path: &Path,
    tokenizer_path: &Path,
    save_dtype: SaveDtype,
    lora: LoraConfig,
) -> Result<(), CliError> {
    let keep_ids = live_tensor_ids(store);
    let step_dir = save_qwen35_step_checkpoint(
        Qwen35StepCheckpoint {
            out_dir,
            step,
            tokenizer_path: Some(tokenizer_path),
            config_json: Qwen35ConfigJsonSource::CopyFrom(config_path),
            generation_config: Qwen35GenerationConfigSource::CopyOrSynthesize {
                source_path: &model_dir.join("generation_config.json"),
                fallback_config_path: config_path,
            },
        },
        |weights_path| {
            let mut tape = Tape::new();
            save_materialized_registry(
                model,
                store,
                &mut tape,
                weights_path,
                matches!(save_dtype, SaveDtype::Bf16),
            )
            .map_err(Into::into)
        },
    )?;
    store.retain_ids(&keep_ids);
    save_adapter_artifacts(&step_dir, model, store, model_dir, "qwen35", lora)?;

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

/// Deterministic in `(seed, step, micro_step)` — stateless w.r.t. any
/// running RNG, so a `--resume-from` run picks up the same data stream a
/// single uninterrupted run would have produced. Mixing is SplitMix64 on
/// top of a `step * golden_ratio ^ (micro_step << 32)` combine so
/// consecutive triples spread uniformly across `[0, upper)` instead of
/// clustering on a single-bit flip.
///
/// Codex review 2026-04-20 on 49512b1 (#2 Medium): prior version drew
/// from a shared `LcgRng` whose position was not persisted in the
/// checkpoint, so a resumed run redrew from position 0 and diverged
/// immediately from the interrupted-run's data order.
fn sample_index(seed: u64, upper: usize, step: usize, micro_step: usize) -> usize {
    if upper <= 1 {
        return 0;
    }
    let mut h =
        seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((micro_step as u64) << 32);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    (h % upper as u64) as usize
}

#[cfg(test)]
mod lora_tests {
    use super::*;
    use qwen35_spec::{LayerType, Qwen35AttentionTensorNames};
    use serde_json::json;
    use tempfile::tempdir;
    use train::LoraAdapterConfig;
    use train::causal_lm::build_materialized_registry;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn tiny_qwen35_config() -> Qwen35Config {
        Qwen35Config {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            vocab_size: 128,
            rms_norm_eps: 1.0e-6,
            stop_token_ids: vec![2],
            bos_token_id: Some(1),
            eos_token_id: 2,
            tie_word_embeddings: false,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            linear_num_key_heads: 4,
            linear_key_head_dim: 8,
            linear_num_value_heads: 4,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            rotary_dim: 8,
            rope_cache_len_hint: Some(16),
            layer_types: vec![LayerType::FullAttention; 2],
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: Vec::new(),
        }
    }

    #[test]
    fn qwen35_save_checkpoint_writes_merged_weights_and_adapter_artifacts() -> TestResult {
        let tmp = tempdir().expect("tempdir");
        let model_dir = tmp.path().join("model");
        let out_dir = tmp.path().join("out");
        fs::create_dir_all(&model_dir).expect("create model dir");
        fs::create_dir_all(&out_dir).expect("create out dir");
        fs::write(
            model_dir.join("config.json"),
            serde_json::to_string_pretty(&json!({
                "bos_token_id": 1,
                "eos_token_id": 2,
            }))?,
        )
        .expect("write config");
        fs::write(model_dir.join("tokenizer.json"), "{}").expect("write tokenizer");

        let cfg = tiny_qwen35_config();
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
        };
        let layer_names = cfg.layer_tensor_names(0);
        let Qwen35AttentionTensorNames::Full(attn_names) = layer_names.attention else {
            unreachable!("test config uses full attention");
        };

        let mut expected_store = TensorStore::default();
        let expected_model = Qwen35Model::new_with_lora(&cfg, Some(lora), &mut expected_store)?;
        let mut save_store = TensorStore::default();
        let save_model = Qwen35Model::new_with_lora(&cfg, Some(lora), &mut save_store)?;

        for (model, store) in [
            (&expected_model, &mut expected_store),
            (&save_model, &mut save_store),
        ] {
            let adapter_map = model.adapter_name_map();
            let q_proj_a = *adapter_map
                .get(format!("{}.lora_a", attn_names.q_proj).as_str())
                .expect("adapter a");
            let q_proj_b = *adapter_map
                .get(format!("{}.lora_b", attn_names.q_proj).as_str())
                .expect("adapter b");
            store.get_mut(q_proj_a).expect("adapter a exists").data[0] = 1.0;
            store.get_mut(q_proj_b).expect("adapter b exists").data[0] = 2.0;
        }

        let mut expected_tape = Tape::new();
        let materialized =
            build_materialized_registry(&expected_model, &mut expected_store, &mut expected_tape)?;
        let expected_q = expected_store.to_host(
            materialized
                .get(attn_names.q_proj.as_str())
                .expect("materialized q proj"),
        )?;

        save_qwen35_checkpoint(
            &model_dir,
            &out_dir,
            3,
            &save_model,
            &mut save_store,
            &model_dir.join("config.json"),
            &model_dir.join("tokenizer.json"),
            SaveDtype::F32,
            lora,
        )?;
        let step_dir = out_dir.join("step_000003");

        assert!(step_dir.join("model.safetensors").is_file());
        assert!(step_dir.join("adapter_model.safetensors").is_file());
        assert!(step_dir.join("adapter_config.json").is_file());
        validate_adapter_resume_config(&step_dir, &model_dir, "qwen35", lora)?;
        let adapter_config: LoraAdapterConfig =
            serde_json::from_str(&fs::read_to_string(step_dir.join("adapter_config.json"))?)?;
        assert_eq!(
            adapter_config.base_model_name_or_path,
            model_dir.display().to_string()
        );
        assert_eq!(adapter_config.peft_type, "LORA");
        assert_eq!(adapter_config.task_type, "CAUSAL_LM");
        assert_eq!(
            adapter_config.target_modules,
            vec!["all-linear".to_string()]
        );

        let mut load_store = TensorStore::default();
        let load_model = Qwen35Model::new_with_lora(&cfg, None, &mut load_store)?;
        let mut registry = build_registry(&load_model);
        registry.load_into_strict(&mut load_store, &step_dir.join("model.safetensors"))?;
        let loaded_q = load_store.to_host(
            *load_model
                .param_name_map()
                .get(attn_names.q_proj.as_str())
                .expect("loaded q proj"),
        )?;
        assert_eq!(loaded_q, expected_q);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::error::Error;
    use tempfile::tempdir;

    type TestResult<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

    fn resume_cfg() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 24,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            max_position_embeddings: 32768,
        }
    }

    fn write_resume_config(dir: &Path, cfg: &Qwen3Config) {
        fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&json!({
                "bos_token_id": 1,
                "eos_token_id": 2,
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "head_dim": cfg.head_dim,
                "vocab_size": cfg.vocab_size,
                "rms_norm_eps": cfg.rms_norm_eps,
                "rope_theta": cfg.rope_theta,
                "tie_word_embeddings": cfg.tie_word_embeddings,
                "max_position_embeddings": cfg.max_position_embeddings,
            }))
            .expect("serialize config"),
        )
        .expect("write config");
    }

    fn tiny_cli_cfg() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            vocab_size: 32,
            rms_norm_eps: 1.0e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: false,
            max_position_embeddings: 64,
        }
    }

    fn write_tiny_model_dir(model_dir: &Path, cfg: &Qwen3Config) -> TestResult {
        fs::create_dir_all(model_dir)?;
        write_resume_config(model_dir, cfg);
        write_tiny_tokenizer(&model_dir.join("tokenizer.json"))?;

        let mut store = TensorStore::default();
        let model = Qwen3Model::new(cfg, &mut store)?;
        build_registry(&model).save_from(&mut store, &model_dir.join("model.safetensors"))?;
        Ok(())
    }

    fn write_tiny_tokenizer(path: &Path) -> TestResult {
        train::tokenizer::write_wordlevel_tokenizer(
            path,
            std::iter::empty::<String>(),
            [
                "<|im_start|>user\nhi<|im_end|>\n".to_string(),
                "<|im_start|>assistant\nhello<|im_end|>\n".to_string(),
                "<|im_start|>user\nadd<|im_end|>\n".to_string(),
                "<|im_start|>assistant\nsum<|im_end|>\n".to_string(),
                "<|im_start|>user\nbye<|im_end|>\n".to_string(),
                "<|im_start|>assistant\nlater<|im_end|>\n".to_string(),
            ],
        )?;
        Ok(())
    }

    fn write_tiny_sft_jsonl(path: &Path) -> TestResult {
        fs::write(
            path,
            concat!(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"hello\"}]}\n",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"add\"},{\"role\":\"assistant\",\"content\":\"sum\"}]}\n",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"bye\"},{\"role\":\"assistant\",\"content\":\"later\"}]}\n",
            ),
        )?;
        Ok(())
    }

    #[test]
    fn batched_assistant_loss_matches_single_example_mean() -> TestResult {
        let logits_data = vec![
            4.0f32, 1.0, 0.0, -1.0, 0.5, 3.0, 0.0, -2.0, // example 0
            2.0, 0.0, 1.0, -1.0, -0.5, 0.0, 4.0, 1.0, // example 1
        ];

        let mut batch_store = TensorStore::default();
        let mut batch_tape = Tape::new();
        let batch_logits = batch_store.from_slice(&logits_data, &[2, 2, 4])?;
        let batch_loss = assistant_masked_causal_loss_batch(
            batch_logits,
            &[1, 1, 0, -100],
            2,
            2,
            &mut batch_store,
            &mut batch_tape,
            4,
        )?;
        let batch_value = batch_store.to_host(batch_loss)?[0];

        let mut single_values = Vec::new();
        for (row, labels) in [vec![1, 1], vec![0, -100]].into_iter().enumerate() {
            let mut store = TensorStore::default();
            let mut tape = Tape::new();
            let start = row * 8;
            let logits = store.from_slice(&logits_data[start..start + 8], &[1, 2, 4])?;
            let loss = assistant_masked_causal_loss(logits, &labels, &mut store, &mut tape, 4)?;
            single_values.push(store.to_host(loss)?[0]);
        }
        let expected = single_values.iter().sum::<f32>() / single_values.len() as f32;
        assert!(
            (batch_value - expected).abs() < 1.0e-5,
            "batched loss {batch_value} != single-example mean {expected}"
        );
        Ok(())
    }

    fn tiny_cli_args(
        model_dir: &Path,
        data_path: &Path,
        out_dir: &Path,
        steps: usize,
        resume_from: Option<PathBuf>,
    ) -> CliArgs {
        CliArgs {
            model_family: ModelFamily::Qwen3,
            model: model_dir.to_path_buf(),
            data: data_path.to_path_buf(),
            out: out_dir.to_path_buf(),
            steps,
            batch: 1,
            lr: 5.0e-3,
            seq_len: 64,
            backend: BackendChoice::Cpu,
            save_every: 2,
            log_every: 1,
            seed: 0x5F54_5F53_4D4F_4B45,
            save_dtype: SaveDtype::F32,
            lr_schedule: "constant".to_string(),
            warmup_steps: 0,
            min_lr: 0.0,
            grad_accum_steps: None,
            metrics_jsonl: None,
            resume_from,
            lora_rank: 4,
            lora_alpha: 8.0,
            serve: None,
        }
    }

    fn latest_step_dir(out_dir: &Path) -> TestResult<PathBuf> {
        Ok(out_dir.join("latest").canonicalize()?)
    }

    fn assert_adamw_state_eq(
        lhs: &autograd::adamw_state::AdamWState,
        rhs: &autograd::adamw_state::AdamWState,
    ) {
        assert_eq!(lhs.step, rhs.step);
        assert_eq!(lhs.skipped_export, rhs.skipped_export);
        assert_eq!(lhs.params.len(), rhs.params.len());
        let mut lhs_params = lhs.params.iter().collect::<Vec<_>>();
        let mut rhs_params = rhs.params.iter().collect::<Vec<_>>();
        lhs_params.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        rhs_params.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        for (idx, (left, right)) in lhs_params.iter().zip(rhs_params.iter()).enumerate() {
            assert_eq!(left.name, right.name, "param name mismatch at {idx}");
            assert_eq!(left.shape, right.shape, "shape mismatch at {idx}");
            assert_eq!(left.m.len(), right.m.len(), "m length mismatch at {idx}");
            assert_eq!(left.v.len(), right.v.len(), "v length mismatch at {idx}");
            for (elem_idx, (lm, rm)) in left.m.iter().zip(right.m.iter()).enumerate() {
                assert_eq!(lm.to_bits(), rm.to_bits(), "m drift at {idx}:{elem_idx}");
            }
            for (elem_idx, (lv, rv)) in left.v.iter().zip(right.v.iter()).enumerate() {
                assert_eq!(lv.to_bits(), rv.to_bits(), "v drift at {idx}:{elem_idx}");
            }
        }
    }

    #[test]
    fn validate_resume_config_accepts_matching_config() {
        let tmp = tempdir().expect("tempdir");
        let resume_dir = tmp.path();
        let cfg = resume_cfg();
        write_resume_config(resume_dir, &cfg);

        validate_qwen3_resume_config(resume_dir, &cfg).expect("matching config should pass");
    }

    #[test]
    fn validate_resume_config_rejects_max_position_embeddings_mismatch() {
        let tmp = tempdir().expect("tempdir");
        let resume_dir = tmp.path();
        let cfg = resume_cfg();
        write_resume_config(resume_dir, &cfg);

        let live_cfg = Qwen3Config {
            max_position_embeddings: cfg.max_position_embeddings + 1,
            ..cfg
        };

        let err = validate_qwen3_resume_config(resume_dir, &live_cfg).expect_err("should reject");
        assert!(
            err.to_string().contains("max_position_embeddings"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_resume_config_rejects_rope_theta_mismatch() {
        let tmp = tempdir().expect("tempdir");
        let resume_dir = tmp.path();
        let cfg = resume_cfg();
        write_resume_config(resume_dir, &cfg);

        let live_cfg = Qwen3Config {
            rope_theta: cfg.rope_theta * 2.0,
            ..cfg
        };

        let err = validate_qwen3_resume_config(resume_dir, &live_cfg).expect_err("should reject");
        assert!(
            err.to_string().contains("rope_theta"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn qwen3_sft_resume_matches_uninterrupted_run() -> TestResult {
        let tmp = tempdir()?;
        let model_dir = tmp.path().join("model");
        let data_path = tmp.path().join("tiny_sft.jsonl");
        let continuous_out = tmp.path().join("continuous");
        let resumed_out = tmp.path().join("resumed");
        let cfg = tiny_cli_cfg();

        write_tiny_model_dir(&model_dir, &cfg)?;
        write_tiny_sft_jsonl(&data_path)?;

        let continuous_args = tiny_cli_args(&model_dir, &data_path, &continuous_out, 4, None);
        run_with_family::<Qwen3Family>(&continuous_args, &model_dir.join("config.json"))?;

        let first_leg_args = tiny_cli_args(&model_dir, &data_path, &resumed_out, 2, None);
        run_with_family::<Qwen3Family>(&first_leg_args, &model_dir.join("config.json"))?;

        let resume_args = tiny_cli_args(
            &model_dir,
            &data_path,
            &resumed_out,
            4,
            Some(resumed_out.join("latest")),
        );
        run_with_family::<Qwen3Family>(&resume_args, &model_dir.join("config.json"))?;

        let continuous_latest = latest_step_dir(&continuous_out)?;
        let resumed_latest = latest_step_dir(&resumed_out)?;
        assert_eq!(
            continuous_latest.file_name().and_then(|name| name.to_str()),
            Some("step_000004")
        );
        assert_eq!(
            resumed_latest.file_name().and_then(|name| name.to_str()),
            Some("step_000004")
        );

        for artifact in [
            "model.safetensors",
            "adapter_model.safetensors",
            "trainer_state.json",
        ] {
            assert_eq!(
                fs::read(continuous_latest.join(artifact))?,
                fs::read(resumed_latest.join(artifact))?,
                "resume drifted for {artifact}",
            );
        }

        let (continuous_doc, continuous_optim) =
            train::checkpoint::load_trainer_state_v2(&continuous_latest)?;
        let (resumed_doc, resumed_optim) =
            train::checkpoint::load_trainer_state_v2(&resumed_latest)?;
        assert_eq!(continuous_doc.step, resumed_doc.step);
        assert_eq!(continuous_doc.optim_schema, resumed_doc.optim_schema);
        assert_eq!(continuous_doc.schedule_name, resumed_doc.schedule_name);
        assert_eq!(continuous_doc.schedule_params, resumed_doc.schedule_params);
        assert_eq!(
            continuous_doc.grad_accum_current,
            resumed_doc.grad_accum_current
        );
        assert_eq!(continuous_doc.rng_seed, resumed_doc.rng_seed);
        assert_eq!(continuous_doc.codec_version, resumed_doc.codec_version);
        assert_adamw_state_eq(&continuous_optim, &resumed_optim);

        Ok(())
    }
}

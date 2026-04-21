use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use autograd::{
    AutogradError, ConstantLr, Result as AutogradResult, Tape, TensorId, TensorStore,
    adamw_state::AdamWState,
    optim::{AdamW, Optimizer},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use train::{
    CausalLm, StepCtx, StepOutcome, Trainer, TrainerConfig,
    causal_lm::{
        build_adapter_registry, build_registry, live_tensor_ids, save_materialized_registry,
        trainable_param_name_map,
    },
    checkpoint::{
        TRAINER_STATE_CODEC_VERSION, TrainerStateDoc, load_trainer_state_v2, save_trainer_state_v2,
    },
    cli_args::{ArgError, next_value, parse_value},
    dataset::{CopyDataset, Dataset, LcgRng},
    grad_clip::{GlobalNorm, GradClip, NoClip, clip_grad_norm},
    grpo::{GrpoConfig, group_advantages, grpo_loss, mean_sampled_kl},
    lora::{LoraAdapterConfig, LoraConfig},
    loss::cross_entropy_loss,
    model_family::{ModelFamily, synthetic_qwen3_config, synthetic_qwen35_dense_config},
    policy::{GrpoPolicy, GrpoPolicyConfig},
    policy_support::{retained_ids, trainable_param_ids},
    qwen3::{Qwen3Config, Qwen3Error, Qwen3Model},
    qwen3_checkpoint::{
        ConfigJsonSource as Qwen3ConfigJsonSource,
        GenerationConfigSource as Qwen3GenerationConfigSource, Qwen3CheckpointError,
        Qwen3StepCheckpoint, save_step_checkpoint as save_qwen3_step_checkpoint,
    },
    qwen35::{Qwen35Config, Qwen35Error, Qwen35Model},
    qwen35_checkpoint::{
        ConfigJsonSource as Qwen35ConfigJsonSource,
        GenerationConfigSource as Qwen35GenerationConfigSource, Qwen35CheckpointError,
        Qwen35StepCheckpoint, save_step_checkpoint as save_qwen35_step_checkpoint,
    },
    rollout::rollout_group,
};

#[derive(Debug, Clone)]
struct CliArgs {
    model_family: ModelFamily,
    sft_steps: usize,
    grpo_iters: usize,
    save_every: usize,
    batch_prompts: usize,
    group_size: usize,
    seq: usize,
    lr: f32,
    kl_coef: f32,
    temperature: f32,
    seed: u64,
    lora_rank: usize,
    lora_alpha: f32,
    grad_clip: Option<f32>,
    metrics_jsonl: Option<PathBuf>,
    save_path: Option<PathBuf>,
    resume_from: Option<PathBuf>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            model_family: ModelFamily::Qwen35,
            sft_steps: 30,
            grpo_iters: 20,
            save_every: 0,
            batch_prompts: 4,
            group_size: 4,
            seq: 16,
            lr: 1.0e-4,
            kl_coef: 0.02,
            temperature: 1.0,
            seed: 42,
            lora_rank: 8,
            lora_alpha: 16.0,
            grad_clip: Some(1.0),
            metrics_jsonl: None,
            save_path: None,
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
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Arg(#[from] ArgError),
    #[error(transparent)]
    Qwen3Checkpoint(#[from] Qwen3CheckpointError),
    #[error(transparent)]
    Qwen3(#[from] Qwen3Error),
    #[error(transparent)]
    Qwen35Checkpoint(#[from] Qwen35CheckpointError),
    #[error(transparent)]
    Qwen35(#[from] Qwen35Error),
    #[error("{0}")]
    Custom(String),
}

const GRPO_TRAIN_MODEL_FILENAME: &str = "train_model.safetensors";
const GRPO_REFERENCE_MODEL_FILENAME: &str = "reference_model.safetensors";
const GRPO_REFERENCE_ADAPTER_FILENAME: &str = "reference_adapter_model.safetensors";
const GRPO_BASELINE_SALT: u64 = 0x4752_504F_4241_5345;
const GRPO_ITER_SALT: u64 = 0x4752_504F_4954_4552;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct GrpoCheckpointMeta {
    family: String,
    lr: f32,
    kl_coef: f32,
    temperature: f32,
    batch_prompts: usize,
    group_size: usize,
    seq: usize,
    grad_clip: Option<f32>,
    best_mean_reward: f32,
    last_kl: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct ResumeState {
    start_iter: usize,
    best_mean_reward: f32,
    last_kl: f32,
}

trait SyntheticGrpoFamily {
    type Config: GrpoPolicyConfig + Clone;
    type Model: GrpoPolicy<Config = Self::Config> + CausalLm<Config = Self::Config>;

    fn family_name() -> &'static str;
    fn synthetic_model_uri() -> &'static str;
    fn build_config(seq: usize) -> Self::Config;
    fn build_model(
        cfg: &Self::Config,
        lora: LoraConfig,
        store: &mut TensorStore,
    ) -> Result<Self::Model, CliError>;
    fn validate_resume_config(resume_dir: &Path, cfg: &Self::Config) -> Result<(), CliError>;
    fn save_merged_checkpoint(
        out_dir: &Path,
        step: usize,
        cfg: &Self::Config,
        model: &Self::Model,
        store: &mut TensorStore,
    ) -> Result<PathBuf, CliError>;
}

struct Qwen3GrpoFamily;

impl SyntheticGrpoFamily for Qwen3GrpoFamily {
    type Config = Qwen3Config;
    type Model = Qwen3Model;

    fn family_name() -> &'static str {
        "qwen3"
    }

    fn synthetic_model_uri() -> &'static str {
        "synthetic://qwen3"
    }

    fn build_config(seq: usize) -> Self::Config {
        synthetic_qwen3_config(seq)
    }

    fn build_model(
        cfg: &Self::Config,
        lora: LoraConfig,
        store: &mut TensorStore,
    ) -> Result<Self::Model, CliError> {
        Qwen3Model::new_with_lora(cfg, Some(lora), store).map_err(Into::into)
    }

    fn validate_resume_config(resume_dir: &Path, cfg: &Self::Config) -> Result<(), CliError> {
        let cfg_path = resume_dir.join("config.json");
        if !cfg_path.is_file() {
            return Err(CliError::Custom(format!(
                "--resume-from {} has no config.json",
                resume_dir.display()
            )));
        }
        let saved = qwen3_spec::Qwen3Config::from_json_file(&cfg_path).map_err(|err| {
            CliError::Custom(format!(
                "resume config {} does not parse as qwen3-family: {err}",
                cfg_path.display()
            ))
        })?;
        if saved != *cfg {
            return Err(CliError::Custom(format!(
                "--resume-from {} config mismatch with live qwen3 setup",
                resume_dir.display()
            )));
        }
        Ok(())
    }

    fn save_merged_checkpoint(
        out_dir: &Path,
        step: usize,
        cfg: &Self::Config,
        model: &Self::Model,
        store: &mut TensorStore,
    ) -> Result<PathBuf, CliError> {
        save_qwen3_step_checkpoint(
            Qwen3StepCheckpoint {
                out_dir,
                step,
                tokenizer_path: None,
                config_json: Qwen3ConfigJsonSource::Synthesize {
                    cfg,
                    bos_token_id: 0,
                    eos_token_id: 255,
                    torch_dtype: "float32",
                },
                generation_config: Qwen3GenerationConfigSource::Synthesize {
                    bos_token_id: 0,
                    eos_token_id: 255,
                },
            },
            |weights_path| {
                let mut tape = Tape::new();
                save_materialized_registry(model, store, &mut tape, weights_path, false)
                    .map_err(Into::into)
            },
        )
        .map_err(Into::into)
    }
}

struct Qwen35GrpoFamily;

impl SyntheticGrpoFamily for Qwen35GrpoFamily {
    type Config = Qwen35Config;
    type Model = Qwen35Model;

    fn family_name() -> &'static str {
        "qwen35"
    }

    fn synthetic_model_uri() -> &'static str {
        "synthetic://qwen35"
    }

    fn build_config(seq: usize) -> Self::Config {
        synthetic_qwen35_dense_config(seq)
    }

    fn build_model(
        cfg: &Self::Config,
        lora: LoraConfig,
        store: &mut TensorStore,
    ) -> Result<Self::Model, CliError> {
        Qwen35Model::new_with_lora(cfg, Some(lora), store).map_err(Into::into)
    }

    fn validate_resume_config(resume_dir: &Path, cfg: &Self::Config) -> Result<(), CliError> {
        let cfg_path = resume_dir.join("config.json");
        if !cfg_path.is_file() {
            return Err(CliError::Custom(format!(
                "--resume-from {} has no config.json",
                resume_dir.display()
            )));
        }
        let saved = qwen35_spec::Qwen35Config::from_json_file(&cfg_path).map_err(|err| {
            CliError::Custom(format!(
                "resume config {} does not parse as qwen3.5-family: {err}",
                cfg_path.display()
            ))
        })?;
        if saved != *cfg {
            return Err(CliError::Custom(format!(
                "--resume-from {} config mismatch with live qwen3.5 setup",
                resume_dir.display()
            )));
        }
        Ok(())
    }

    fn save_merged_checkpoint(
        out_dir: &Path,
        step: usize,
        cfg: &Self::Config,
        model: &Self::Model,
        store: &mut TensorStore,
    ) -> Result<PathBuf, CliError> {
        save_qwen35_step_checkpoint(
            Qwen35StepCheckpoint {
                out_dir,
                step,
                tokenizer_path: None,
                config_json: Qwen35ConfigJsonSource::Synthesize {
                    cfg,
                    torch_dtype: "float32",
                },
                generation_config: Qwen35GenerationConfigSource::Synthesize {
                    bos_token_id: cfg.bos_token_id,
                    eos_token_id: cfg.eos_token_id,
                },
            },
            |weights_path| {
                let mut tape = Tape::new();
                save_materialized_registry(model, store, &mut tape, weights_path, false)
                    .map_err(Into::into)
            },
        )
        .map_err(Into::into)
    }
}

/// Phase 3 train_grpo migration choice, mirroring `pretrain.rs`:
/// `Trainer<O, C, S>` is generic on the clip policy, so
/// `--no-grad-clip` vs `--grad-clip N` needs to collapse to a single
/// concrete `C`. `NoClip`/`GlobalNorm` forward through this enum so we
/// don't monomorphise the Trainer twice.
enum GrpoClip {
    None(NoClip),
    Norm(GlobalNorm),
}

impl GradClip for GrpoClip {
    fn clip(&mut self, store: &mut TensorStore, params: &[TensorId]) -> AutogradResult<f32> {
        match self {
            Self::None(c) => c.clip(store, params),
            Self::Norm(c) => c.clip(store, params),
        }
    }
}

fn seeded_rng(seed: u64, salt: u64, major: u64, minor: u64) -> LcgRng {
    let mut mixed = seed ^ salt;
    mixed ^= major.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    mixed = mixed.rotate_left(17);
    mixed ^= minor.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = mixed.rotate_left(11);
    LcgRng::seed(mixed)
}

fn freeze_policy<P: GrpoPolicy>(model: &P, store: &mut TensorStore) {
    for tensor_id in model.all_parameter_ids() {
        if let Some(tensor) = store.get_mut(tensor_id) {
            tensor.requires_grad = false;
            tensor.grad = None;
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        // Display (not Debug): `#[error(transparent)]` delegates to the inner
        // error's Display impl, so the user sees the real message instead of
        // `Error: Custom("...")`. Mirrors the `train_sft.rs` pattern.
        Err(err) => {
            eprintln!("[train_grpo] error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), CliError> {
    let args = parse_args()?;
    validate_args(&args)?;

    match args.model_family {
        ModelFamily::Auto | ModelFamily::Qwen35 => run_with_family::<Qwen35GrpoFamily>(&args),
        ModelFamily::Qwen3 => run_with_family::<Qwen3GrpoFamily>(&args),
    }
}

fn run_with_family<F: SyntheticGrpoFamily>(args: &CliArgs) -> Result<(), CliError> {
    let config = F::build_config(args.seq);
    let lora = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };
    let resume_dir = args
        .resume_from
        .as_ref()
        .map(|path| {
            path.canonicalize().map_err(|err| {
                CliError::Custom(format!(
                    "failed to canonicalize --resume-from {}: {err}",
                    path.display()
                ))
            })
        })
        .transpose()?;
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = F::build_model(&config, lora, &mut store)?;
    let params = trainable_param_ids(&policy, &store);
    if params.is_empty() {
        return Err(CliError::Custom(format!(
            "{} model exposed no trainable parameters",
            F::family_name()
        )));
    }
    let keep_extra = policy
        .all_parameter_ids()
        .into_iter()
        .collect::<HashSet<_>>();
    let verifier = |prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]| {
        copy_reward(prompt_ids, full_ids, response_mask)
    };

    let (ref_model, mut optimizer, resume) = if let Some(resume_dir) = resume_dir.as_deref() {
        resume_grpo_checkpoint::<F>(resume_dir, args, &config, &policy, &mut store, lora)?
    } else {
        // ---- SFT warm-up phase (migrated onto Trainer) ----
        //
        // The hand-written loop also computed a teacher-forced accuracy
        // `reward` (via `store.to_host(logits)` + `teacher_forced_reward`) and
        // printed it alongside loss. That value was PRINT-ONLY and cost a full
        // device→host copy of logits per step; the migrated Trainer-format
        // metrics line drops it. GRPO-phase reward tracking (via
        // `mean_reward`) is unaffected.
        //
        // We export the final AdamW state so the GRPO-phase optimizer can
        // resume moments + step counter instead of starting cold (codex review
        // on 09c5c89 P1).
        let sft_optim_state = run_sft_phase(
            args,
            &policy,
            &params,
            keep_extra.clone(),
            &mut store,
            &mut tape,
        )?;

        let ref_model = policy.clone_frozen(&mut store);
        let mut baseline_rng = seeded_rng(args.seed, GRPO_BASELINE_SALT, 0, 0);
        let baseline_prompts =
            build_prompt_batch(args.batch_prompts, args.seq, 64, 255, &mut baseline_rng);
        let baseline_trajectories = rollout_group(
            &policy,
            &ref_model,
            &config,
            &baseline_prompts,
            args.group_size,
            args.temperature,
            &mut baseline_rng,
            &verifier,
            &mut store,
            &mut tape,
        )?;
        let baseline_reward = mean_reward(&baseline_trajectories);
        println!("baseline reward after sft: {baseline_reward:.4}");

        let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
        let param_names = trainable_param_name_map(&policy, &store);
        optimizer
            .import_state(&sft_optim_state, &param_names)
            .map_err(|e| CliError::Custom(format!("adamw sft→grpo handoff: {e}")))?;
        (
            ref_model,
            optimizer,
            ResumeState {
                start_iter: 0,
                best_mean_reward: baseline_reward,
                last_kl: 0.0,
            },
        )
    };

    // Phase 4 follow-up: extend `--metrics-jsonl` to cover the GRPO phase.
    // `run_sft_phase` already truncated the JSONL file (JsonlSink::create),
    // so the GRPO phase opens in APPEND mode to extend the same file
    // instead of clobbering SFT-phase samples. `also_stdout = false` keeps
    // the human-readable `grpo iter N:` println below as the stdout
    // contract; the MetricSample emit is JSONL-only tooling output.
    let mut grpo_metrics = train::metrics::open_sink_append(args.metrics_jsonl.as_deref(), false)
        .map_err(|e| CliError::Custom(format!("grpo metrics sink: {e}")))?;

    let grpo_cfg = GrpoConfig {
        clip_eps: 0.2,
        kl_coef: args.kl_coef,
        group_size: args.group_size,
    };
    let mut reward_trajectory =
        Vec::with_capacity(args.grpo_iters.saturating_sub(resume.start_iter));
    let mut last_kl = resume.last_kl;
    let mut best_mean_reward = resume.best_mean_reward;

    for iter in resume.start_iter..args.grpo_iters {
        let mut iter_rng = seeded_rng(args.seed, GRPO_ITER_SALT, iter as u64, 0);
        let prompts = build_prompt_batch(args.batch_prompts, args.seq, 64, 255, &mut iter_rng);
        let trajectories = rollout_group(
            &policy,
            &ref_model,
            &config,
            &prompts,
            args.group_size,
            args.temperature,
            &mut iter_rng,
            &verifier,
            &mut store,
            &mut tape,
        )?;
        let rewards = trajectories
            .iter()
            .map(|trajectory| trajectory.reward)
            .collect::<Vec<_>>();
        let mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let advantages = group_advantages(&rewards, args.group_size);

        tape.entries.clear();
        tape.set_enabled(true);
        let loss_id = grpo_loss(
            &policy,
            &trajectories,
            &advantages,
            &grpo_cfg,
            &config,
            &mut store,
            &mut tape,
        )?;
        let loss_value = store.to_host(loss_id)?[0];

        optimizer.zero_grad(&params, &mut store);
        tape.backward(loss_id, &mut store)?;
        // `clip_grad_norm` is a no-op when `max_norm` is non-positive / non-
        // finite (see its sanitize-at-boundary contract landed in 429efc3),
        // so `--grad-clip 0 / NaN / inf` collapse to "disabled" without
        // panicking. The `if let Some` gate covers the `--no-grad-clip`
        // case.
        if let Some(max_norm) = args.grad_clip {
            clip_grad_norm(&params, max_norm, &mut store);
        }
        optimizer.step(&params, &mut store);

        tape.entries.clear();
        tape.set_enabled(true);
        last_kl = mean_sampled_kl(&policy, &trajectories, &config, &mut store, &mut tape)?;
        let keep = retained_ids(&[&policy, &ref_model], &store);
        store.retain_ids(&keep);

        reward_trajectory.push(mean_reward);
        best_mean_reward = best_mean_reward.max(mean_reward);
        println!(
            "grpo iter {iter}: loss {loss_value:.4} mean_reward {mean_reward:.4} best_mean_reward {best_mean_reward:.4} mean_kl {last_kl:.4}"
        );
        // JSONL-only emit (see sink construction above): records the
        // GRPO-phase iter as `step = iter + 1` so the SFT-phase Trainer
        // samples (1-indexed from 1..=sft_steps) chain naturally with GRPO
        // samples (sft_steps+1..=sft_steps+grpo_iters) in the same file.
        // Field set is intentionally NOT a superset of the SFT schema —
        // GRPO doesn't have lr/grad_norm semantics the same way, so
        // downstream tools key off field presence to distinguish phases.
        let fields: [(&str, f64); 4] = [
            ("loss", loss_value as f64),
            ("mean_reward", mean_reward as f64),
            ("best_mean_reward", best_mean_reward as f64),
            ("mean_kl", last_kl as f64),
        ];
        grpo_metrics.emit(&train::metrics::MetricSample {
            step: (args.sft_steps as u64) + (iter as u64) + 1,
            fields: &fields,
        });

        if let Some(save_path) = args.save_path.as_deref() {
            if args.save_every > 0 && (iter + 1).is_multiple_of(args.save_every) {
                let step_dir = save_grpo_checkpoint::<F>(
                    save_path,
                    iter + 1,
                    args,
                    &config,
                    &policy,
                    &ref_model,
                    &optimizer,
                    &mut store,
                    lora,
                    best_mean_reward,
                    last_kl,
                )?;
                eprintln!("[train_grpo] checkpoint saved to {}", step_dir.display());
            }
        }
    }

    if let Some(save_path) = args.save_path.as_deref() {
        let step_dir = save_grpo_checkpoint::<F>(
            save_path,
            args.grpo_iters,
            args,
            &config,
            &policy,
            &ref_model,
            &optimizer,
            &mut store,
            lora,
            best_mean_reward,
            last_kl,
        )?;
        eprintln!(
            "[train_grpo] final checkpoint saved to {}",
            step_dir.display()
        );
    }

    println!("final kl {last_kl:.4}");
    println!("reward trajectory: {reward_trajectory:?}");
    Ok(())
}

/// SFT warm-up migrated onto `Trainer<AdamW, GrpoClip, ConstantLr>`. The
/// hand-written loop emitted `sft step N: loss L reward R`; the migrated
/// loop emits the shared Trainer metric format (`step=N loss=... lr=...
/// grad_norm=... ms_per_step=... tok_per_sec=...`). The teacher-forced
/// `reward` print field is dropped — it required a logits device→host
/// copy per step and only fed the println, no correctness concern.
fn run_sft_phase<P>(
    args: &CliArgs,
    policy: &P,
    params: &[TensorId],
    keep_extra: HashSet<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<AdamWState, CliError>
where
    P: GrpoPolicy + CausalLm,
{
    let optim = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    // Mirror pretrain.rs: `--grad-clip 0/NaN/inf` warns and falls through
    // to NoClip instead of panicking in `GlobalNorm::new`.
    let clip = match args.grad_clip {
        Some(max_norm) if max_norm > 0.0 && max_norm.is_finite() => {
            GrpoClip::Norm(GlobalNorm::new(max_norm))
        }
        Some(max_norm) => {
            eprintln!(
                "[train_grpo] warning: --grad-clip {max_norm} is non-positive/non-finite; disabling gradient clipping"
            );
            GrpoClip::None(NoClip)
        }
        None => GrpoClip::None(NoClip),
    };
    let schedule = ConstantLr(args.lr);
    let metrics = train::metrics::open_sink(args.metrics_jsonl.as_deref(), true)
        .map_err(|e| CliError::Custom(format!("metrics sink: {e}")))?;

    let trainer_cfg = TrainerConfig {
        total_steps: args.sft_steps as u64,
        grad_accum_steps: 1,
        log_every: 1,
        eval_every: None,
        save_every: None,
        save_dir: None,
        resume_from: None,
        rng_seed: args.seed,
    };
    let mut trainer = Trainer::new(optim, clip, schedule, metrics, trainer_cfg);

    let param_names = trainable_param_name_map(policy, store);

    let mut dataset = CopyDataset::with_vocab(args.batch_prompts, args.seq, args.seed, 64, 255);
    let batch_shape = dataset.batch_shape();
    let step_fn = |ctx: &mut StepCtx<'_>| -> AutogradResult<StepOutcome> {
        let (input_ids, target_ids) = dataset.sample();
        let (batch, seq_len) = batch_shape;
        let token_count = (batch * seq_len) as u64;
        let logits =
            policy.forward_batch_tokens(&input_ids, batch, seq_len, ctx.store, ctx.tape)?;
        let loss_id = cross_entropy_loss(logits, &target_ids, ctx.store, ctx.tape)?;
        Ok(StepOutcome {
            loss_id,
            token_count,
        })
    };

    trainer.run(
        store,
        tape,
        params.to_vec(),
        param_names.clone(),
        keep_extra,
        step_fn,
    )?;
    // Export optimizer state so the caller can re-import it into the
    // GRPO-phase AdamW and preserve moments + step counter across the
    // SFT→GRPO boundary (codex review on 09c5c89 P1).
    Ok(trainer.optim().export_state(&param_names))
}

fn validate_adapter_resume_config(
    resume_dir: &Path,
    synthetic_model_uri: &str,
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
    let expected = LoraAdapterConfig::new(synthetic_model_uri, family, lora);
    if value != expected {
        return Err(CliError::Custom(format!(
            "--resume-from {} adapter mismatch with live LoRA config",
            resume_dir.display()
        )));
    }
    Ok(())
}

fn resume_grpo_checkpoint<F: SyntheticGrpoFamily>(
    resume_dir: &Path,
    args: &CliArgs,
    cfg: &F::Config,
    policy: &F::Model,
    store: &mut TensorStore,
    lora: LoraConfig,
) -> Result<(F::Model, AdamW, ResumeState), CliError> {
    F::validate_resume_config(resume_dir, cfg)?;
    validate_adapter_resume_config(resume_dir, F::synthetic_model_uri(), F::family_name(), lora)?;

    let train_model_path = resume_dir.join(GRPO_TRAIN_MODEL_FILENAME);
    if !train_model_path.is_file() {
        return Err(CliError::Custom(format!(
            "--resume-from {} has no {GRPO_TRAIN_MODEL_FILENAME}",
            resume_dir.display()
        )));
    }
    let mut registry = build_registry(policy);
    registry.load_into_strict(store, &train_model_path)?;

    let mut adapter_registry = build_adapter_registry(policy);
    if !adapter_registry.is_empty() {
        let adapter_path = resume_dir.join("adapter_model.safetensors");
        if !adapter_path.is_file() {
            return Err(CliError::Custom(format!(
                "--resume-from {} has no adapter_model.safetensors",
                resume_dir.display()
            )));
        }
        adapter_registry.load_into_strict(store, &adapter_path)?;
    }

    let ref_model = {
        let loaded = F::build_model(cfg, lora, store)?;
        let mut ref_registry = build_registry(&loaded);
        let ref_model_path = resume_dir.join(GRPO_REFERENCE_MODEL_FILENAME);
        if !ref_model_path.is_file() {
            return Err(CliError::Custom(format!(
                "--resume-from {} has no {GRPO_REFERENCE_MODEL_FILENAME}",
                resume_dir.display()
            )));
        }
        ref_registry.load_into_strict(store, &ref_model_path)?;
        let mut ref_adapter_registry = build_adapter_registry(&loaded);
        if !ref_adapter_registry.is_empty() {
            let ref_adapter_path = resume_dir.join(GRPO_REFERENCE_ADAPTER_FILENAME);
            if !ref_adapter_path.is_file() {
                return Err(CliError::Custom(format!(
                    "--resume-from {} has no {GRPO_REFERENCE_ADAPTER_FILENAME}",
                    resume_dir.display()
                )));
            }
            ref_adapter_registry.load_into_strict(store, &ref_adapter_path)?;
        }
        freeze_policy(&loaded, store);
        loaded
    };

    let (trainer_doc, optim_state) = load_trainer_state_v2(resume_dir)
        .map_err(|err| CliError::Custom(format!("resume trainer state: {err}")))?;
    if trainer_doc.rng_seed != args.seed {
        return Err(CliError::Custom(format!(
            "--resume-from {} seed mismatch: checkpoint={} live={}",
            resume_dir.display(),
            trainer_doc.rng_seed,
            args.seed
        )));
    }
    if trainer_doc.schedule_name != "constant" {
        return Err(CliError::Custom(format!(
            "--resume-from {} unsupported schedule {}",
            resume_dir.display(),
            trainer_doc.schedule_name
        )));
    }
    let meta: GrpoCheckpointMeta =
        serde_json::from_value(trainer_doc.schedule_params).map_err(|err| {
            CliError::Custom(format!("resume checkpoint metadata parse error: {err}"))
        })?;
    let expected = GrpoCheckpointMeta {
        family: F::family_name().to_string(),
        lr: args.lr,
        kl_coef: args.kl_coef,
        temperature: args.temperature,
        batch_prompts: args.batch_prompts,
        group_size: args.group_size,
        seq: args.seq,
        grad_clip: args.grad_clip,
        best_mean_reward: meta.best_mean_reward,
        last_kl: meta.last_kl,
    };
    if meta != expected {
        return Err(CliError::Custom(format!(
            "--resume-from {} GRPO metadata mismatch with live args",
            resume_dir.display()
        )));
    }

    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let param_names = trainable_param_name_map(policy, store);
    let restored = optimizer
        .import_state(&optim_state, &param_names)
        .map_err(|err| CliError::Custom(format!("resume optimizer import failed: {err}")))?;
    eprintln!(
        "[train_grpo] resumed step {} with {restored} optimizer entries from {}",
        trainer_doc.step,
        resume_dir.display()
    );
    Ok((
        ref_model,
        optimizer,
        ResumeState {
            start_iter: trainer_doc.step as usize,
            best_mean_reward: meta.best_mean_reward,
            last_kl: meta.last_kl,
        },
    ))
}

fn save_grpo_checkpoint<F: SyntheticGrpoFamily>(
    out_dir: &Path,
    step: usize,
    args: &CliArgs,
    cfg: &F::Config,
    policy: &F::Model,
    ref_model: &F::Model,
    optimizer: &AdamW,
    store: &mut TensorStore,
    lora: LoraConfig,
    best_mean_reward: f32,
    last_kl: f32,
) -> Result<PathBuf, CliError> {
    let keep_ids = live_tensor_ids(store);
    let step_dir = F::save_merged_checkpoint(out_dir, step, cfg, policy, store)?;
    store.retain_ids(&keep_ids);

    build_registry(policy).save_from(store, &step_dir.join(GRPO_TRAIN_MODEL_FILENAME))?;
    build_registry(ref_model).save_from(store, &step_dir.join(GRPO_REFERENCE_MODEL_FILENAME))?;

    let adapter_registry = build_adapter_registry(policy);
    if !adapter_registry.is_empty() {
        adapter_registry.save_from(store, &step_dir.join("adapter_model.safetensors"))?;
        build_adapter_registry(ref_model)
            .save_from(store, &step_dir.join(GRPO_REFERENCE_ADAPTER_FILENAME))?;
        let adapter_config =
            LoraAdapterConfig::new(F::synthetic_model_uri(), F::family_name(), lora);
        fs::write(
            step_dir.join("adapter_config.json"),
            serde_json::to_string_pretty(&adapter_config)?,
        )?;
    }

    let trainer_doc = TrainerStateDoc {
        step: step as u64,
        optim_schema: optimizer.state_schema().to_string(),
        schedule_name: "constant".to_string(),
        schedule_params: serde_json::to_value(GrpoCheckpointMeta {
            family: F::family_name().to_string(),
            lr: args.lr,
            kl_coef: args.kl_coef,
            temperature: args.temperature,
            batch_prompts: args.batch_prompts,
            group_size: args.group_size,
            seq: args.seq,
            grad_clip: args.grad_clip,
            best_mean_reward,
            last_kl,
        })?,
        grad_accum_current: 0,
        rng_seed: args.seed,
        codec_version: TRAINER_STATE_CODEC_VERSION,
    };
    let optim_state = optimizer.export_state(&trainable_param_name_map(policy, store));
    save_trainer_state_v2(&step_dir, &trainer_doc, &optim_state)
        .map_err(|err| CliError::Custom(format!("save trainer state: {err}")))?;
    Ok(step_dir)
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
            "--sft-steps" => args.sft_steps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--grpo-iters" => {
                args.grpo_iters = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--save-every" => {
                args.save_every = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--batch-prompts" => {
                args.batch_prompts = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--group-size" => {
                args.group_size = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--seq" => args.seq = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--lr" => args.lr = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--kl-coef" => args.kl_coef = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--temperature" => {
                args.temperature = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--seed" => args.seed = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--lora-rank" => {
                args.lora_rank = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--lora-alpha" => {
                args.lora_alpha = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--grad-clip" => {
                args.grad_clip = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--no-grad-clip" => args.grad_clip = None,
            "--metrics-jsonl" => {
                args.metrics_jsonl = Some(PathBuf::from(next_value(&mut iter, &flag)?));
            }
            "--save-path" => {
                args.save_path = Some(PathBuf::from(next_value(&mut iter, &flag)?));
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
    if args.seq < 4 || !args.seq.is_multiple_of(2) {
        return Err(CliError::Autograd(AutogradError::InvalidRank {
            expected: "even sequence length >= 4",
            got: args.seq,
        }));
    }
    if args.group_size == 0 || args.batch_prompts == 0 {
        return Err(CliError::Autograd(AutogradError::InvalidRank {
            expected: "positive batch and group sizes",
            got: args.group_size.min(args.batch_prompts),
        }));
    }
    if args.lora_rank == 0 {
        return Err(CliError::Arg(ArgError::InvalidValue {
            flag: "--lora-rank".into(),
            value: "0".into(),
        }));
    }
    if !(args.lora_alpha.is_finite() && args.lora_alpha > 0.0) {
        return Err(CliError::Arg(ArgError::InvalidValue {
            flag: "--lora-alpha".into(),
            value: args.lora_alpha.to_string(),
        }));
    }
    if args.save_every > 0 && args.save_path.is_none() {
        return Err(CliError::Arg(ArgError::InvalidValue {
            flag: "--save-every".into(),
            value: "requires --save-path".into(),
        }));
    }
    Ok(())
}

fn build_prompt_batch(
    batch: usize,
    seq_len: usize,
    token_upper: usize,
    separator: usize,
    rng: &mut LcgRng,
) -> Vec<Vec<usize>> {
    let prefix_len = seq_len / 2;
    let mut prompts = Vec::with_capacity(batch);
    for _ in 0..batch {
        let mut prompt = vec![0usize; seq_len];
        for token in &mut prompt[..prefix_len] {
            *token = (rng.next_u64() % token_upper as u64) as usize;
        }
        prompt[prefix_len] = separator;
        prompts.push(prompt);
    }
    prompts
}

fn copy_reward(prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]) -> f32 {
    let prefix_len = prompt_ids.len() / 2;
    let mut correct = 0usize;
    let mut total = 0usize;
    for (position, masked) in response_mask.iter().enumerate() {
        if !*masked {
            continue;
        }
        total += 1;
        if full_ids[position] == prompt_ids[position - (prefix_len + 1)] {
            correct += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32
    }
}

fn mean_reward(trajectories: &[train::rollout::Trajectory]) -> f32 {
    if trajectories.is_empty() {
        0.0
    } else {
        trajectories
            .iter()
            .map(|trajectory| trajectory.reward)
            .sum::<f32>()
            / trajectories.len() as f32
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use qwen35_spec::Qwen35AttentionTensorNames;
    use tempfile::tempdir;

    fn tiny_args() -> CliArgs {
        CliArgs {
            model_family: ModelFamily::Qwen35,
            sft_steps: 2,
            grpo_iters: 4,
            save_every: 0,
            batch_prompts: 2,
            group_size: 2,
            seq: 16,
            lr: 1.0e-4,
            kl_coef: 0.02,
            temperature: 1.0,
            seed: 42,
            lora_rank: 2,
            lora_alpha: 4.0,
            grad_clip: Some(1.0),
            metrics_jsonl: None,
            save_path: None,
            resume_from: None,
        }
    }

    fn assert_adamw_state_eq(
        lhs: &autograd::adamw_state::AdamWState,
        rhs: &autograd::adamw_state::AdamWState,
    ) {
        assert_eq!(lhs.step, rhs.step);
        assert_eq!(lhs.skipped_export, rhs.skipped_export);
        assert_eq!(lhs.params.len(), rhs.params.len());
        for (left, right) in lhs.params.iter().zip(rhs.params.iter()) {
            assert_eq!(left.name, right.name);
            assert_eq!(left.shape, right.shape);
            assert_eq!(left.m, right.m);
            assert_eq!(left.v, right.v);
        }
    }

    #[test]
    fn trainable_param_name_map_uses_adapter_names() -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Qwen3Config {
            vocab_size: 128,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            max_position_embeddings: 16,
            rms_norm_eps: 1.0e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: false,
        };
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
        };
        let mut store = TensorStore::default();
        let model = Qwen3Model::new_with_lora(&cfg, Some(lora), &mut store)?;
        let params = trainable_param_ids(&model, &store);
        let names = trainable_param_name_map(&model, &store);
        assert_eq!(names.len(), params.len());
        assert!(names.iter().all(|(_, name)| name.contains(".lora_")));
        assert!(names.iter().all(|(_, name)| !name.starts_with("param.")));
        Ok(())
    }

    #[test]
    fn qwen35_grpo_checkpoint_roundtrip_restores_policy_ref_and_optimizer()
    -> Result<(), Box<dyn std::error::Error>> {
        let args = tiny_args();
        let cfg = synthetic_qwen35_dense_config(args.seq);
        let lora = LoraConfig {
            rank: args.lora_rank,
            alpha: args.lora_alpha,
        };
        let out_dir = tempdir()?;
        let layer_names = cfg.layer_tensor_names(0);
        let Qwen35AttentionTensorNames::Full(attn_names) = layer_names.attention else {
            unreachable!("synthetic qwen35 config uses full attention");
        };

        let mut save_store = TensorStore::default();
        let policy = Qwen35Model::new_with_lora(&cfg, Some(lora), &mut save_store)?;
        let ref_model = policy.clone_frozen(&mut save_store);
        let current_names = trainable_param_name_map(&policy, &save_store);
        let current_adapter_map = policy.adapter_name_map();
        let current_q = format!("{}.lora_a", attn_names.q_proj);
        let current_q_id = *policy
            .adapter_name_map()
            .get(current_q.as_str())
            .expect("policy adapter id");
        save_store
            .get_mut(current_q_id)
            .expect("policy adapter")
            .data[0] = 1.25;
        assert!(current_adapter_map.contains_key(current_q.as_str()));
        let ref_q = format!("{}.lora_a", attn_names.q_proj);
        let ref_q_id = *ref_model
            .adapter_name_map()
            .get(ref_q.as_str())
            .expect("ref adapter id");
        save_store.get_mut(ref_q_id).expect("ref adapter").data[0] = -0.75;

        let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
        let saved_state = autograd::adamw_state::AdamWState {
            step: 3,
            skipped_export: 0,
            params: current_names
                .iter()
                .map(|(tensor_id, name)| {
                    let shape = save_store
                        .get(*tensor_id)
                        .expect("tensor exists")
                        .shape
                        .clone();
                    let len = shape.iter().product::<usize>().max(1);
                    autograd::adamw_state::AdamWParamState {
                        name: name.clone(),
                        m: vec![0.25; len],
                        v: vec![0.5; len],
                        shape,
                    }
                })
                .collect(),
        };
        optimizer.import_state(&saved_state, &current_names)?;

        let saved_policy_a = save_store.to_host(current_q_id)?;
        let saved_ref_a = save_store.to_host(ref_q_id)?;

        let step_dir = save_grpo_checkpoint::<Qwen35GrpoFamily>(
            out_dir.path(),
            3,
            &args,
            &cfg,
            &policy,
            &ref_model,
            &optimizer,
            &mut save_store,
            lora,
            0.8,
            0.1,
        )?;

        assert!(step_dir.join(GRPO_TRAIN_MODEL_FILENAME).is_file());
        assert!(step_dir.join(GRPO_REFERENCE_MODEL_FILENAME).is_file());
        assert!(step_dir.join(GRPO_REFERENCE_ADAPTER_FILENAME).is_file());

        let mut resumed_store = TensorStore::default();
        let resumed_policy = Qwen35Model::new_with_lora(&cfg, Some(lora), &mut resumed_store)?;
        let (resumed_ref, resumed_optim, resume) = resume_grpo_checkpoint::<Qwen35GrpoFamily>(
            &step_dir,
            &args,
            &cfg,
            &resumed_policy,
            &mut resumed_store,
            lora,
        )?;
        assert_eq!(resume.start_iter, 3);
        assert_eq!(resume.best_mean_reward, 0.8);
        assert_eq!(resume.last_kl, 0.1);

        let resumed_policy_q = format!("{}.lora_a", attn_names.q_proj);
        let resumed_policy_q_id = *resumed_policy
            .adapter_name_map()
            .get(resumed_policy_q.as_str())
            .expect("resumed policy q id");
        let resumed_ref_q = format!("{}.lora_a", attn_names.q_proj);
        let resumed_ref_q_id = *resumed_ref
            .adapter_name_map()
            .get(resumed_ref_q.as_str())
            .expect("resumed ref q id");

        assert_eq!(resumed_store.to_host(resumed_policy_q_id)?, saved_policy_a);
        assert_eq!(resumed_store.to_host(resumed_ref_q_id)?, saved_ref_a);
        assert!(
            resumed_ref
                .all_parameter_ids()
                .iter()
                .all(|tensor_id| !resumed_store.get(*tensor_id).expect("tensor").requires_grad)
        );

        let resumed_names = trainable_param_name_map(&resumed_policy, &resumed_store);
        let resumed_state = resumed_optim.export_state(&resumed_names);
        assert_adamw_state_eq(&saved_state, &resumed_state);
        Ok(())
    }

    /// Codex review 2026-04-20 on 09c5c89 (P2): `fn main() -> Result<(),
    /// CliError>` leaked the Debug format through to stderr (`Error:
    /// Custom("...")`). We now wrap in `ExitCode` + `eprintln!("{err}")`
    /// which routes through `Display`. Pin the Display format so a future
    /// `derive(Debug)`-only regression can't silently re-introduce the
    /// Custom(...) wrapper.
    #[test]
    fn cli_error_display_does_not_leak_debug_wrapper() {
        let err = CliError::Custom(
            "metrics sink: failed to create JSONL metrics sink at /root/forbidden.jsonl: \
             No such file or directory (os error 2)"
                .to_string(),
        );
        let rendered = format!("{err}");
        assert!(
            !rendered.contains("Custom("),
            "Display for CliError::Custom must NOT wrap the payload in Custom(...); \
             got: {rendered}"
        );
        assert!(
            rendered.contains("metrics sink"),
            "Display should surface the underlying message verbatim; got: {rendered}"
        );
    }
}

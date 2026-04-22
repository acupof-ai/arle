//! Multi-turn trainer with an explicit objective switch:
//! - stepwise GRPO: rollout_episode -> discounted per-turn returns ->
//!   cross-episode group-normalize per turn -> per-position advantages ->
//!   grpo_loss_per_position -> AdamW.
//! - sequence-level GSPO: rollout_episode -> one scalar episode score (mean
//!   per-turn reward) -> group-normalize per episode -> grpo_loss -> AdamW.
//!
//! Mirrors `train_grpo` but on interleaved agent/observation episodes
//! instead of suffix-only rollouts.

use std::{env, path::PathBuf, process::ExitCode, str::FromStr, sync::Arc};

use autograd::{
    AutogradError, Backend, CpuBackend, Tape, TensorStore,
    optim::{AdamW, Optimizer},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use train::{
    causal_lm::{
        build_adapter_registry, build_registry, live_tensor_ids, save_materialized_registry,
        trainable_param_name_map,
    },
    checkpoint::{
        TRAINER_STATE_CODEC_VERSION, TrainerStateDoc, load_trainer_state_v2, save_trainer_state_v2,
    },
    cli_args::{ArgError, BackendChoice, adamw_for_backend, next_value, parse_value},
    control::{
        TrainingController, emit_run_end, emit_run_start, open_run_metrics, serve_if_requested,
        sync_status,
    },
    dataset::LcgRng,
    grad_clip::clip_grad_norm,
    grpo::{GrpoConfig, group_advantages, grpo_loss, grpo_loss_per_position, mean_sampled_kl},
    lora::{LoraAdapterConfig, LoraConfig},
    metrics::MetricSample,
    model_family::{Qwen35AttentionPattern, apply_qwen35_attention_pattern},
    multi_turn::{Environment, Episode, TurnSpec, rollout_episode, rollout_episode_group},
    policy::{GrpoPolicy, GrpoPolicyConfig},
    policy_support::{retained_ids, trainable_param_ids},
    qwen35::{LayerType, Qwen35Config, Qwen35Error, Qwen35Model},
    qwen35_checkpoint::{
        ConfigJsonSource, GenerationConfigSource, Qwen35CheckpointError, Qwen35StepCheckpoint,
        save_step_checkpoint,
    },
    reward::{discounted_returns, group_normalize, returns_to_per_position},
    rollout::Trajectory,
};

#[derive(Debug, Clone)]
struct CliArgs {
    iters: usize,
    group_size: usize,
    agent_tokens: usize,
    obs_tokens: usize,
    turns: usize,
    prompt_len: usize,
    lr: f32,
    kl_coef: f32,
    clip_eps: f32,
    temperature: f32,
    gamma: f32,
    lora_rank: usize,
    lora_alpha: f32,
    seed: u64,
    vocab_size: usize,
    target_range: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    d_head: usize,
    d_ff: usize,
    eval_every: usize,
    eval_prompts: usize,
    eval_temperature: f32,
    backend: BackendChoice,
    save_path: Option<String>,
    resume_from: Option<PathBuf>,
    serve: Option<u16>,
    grad_clip: Option<f32>,
    metrics_jsonl: Option<PathBuf>,
    objective: MultiTurnObjective,
    linear_attn_every: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiTurnObjective {
    StepwiseGrpo,
    SequenceGspo,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            iters: 20,
            group_size: 4,
            agent_tokens: 2,
            obs_tokens: 2,
            turns: 2,
            prompt_len: 4,
            lr: 1.0e-4,
            kl_coef: 0.02,
            clip_eps: 0.2,
            temperature: 1.0,
            gamma: 0.9,
            lora_rank: 8,
            lora_alpha: 16.0,
            seed: 42,
            vocab_size: 32,
            target_range: 8,
            d_model: 64,
            n_layers: 2,
            n_heads: 2,
            d_head: 32,
            d_ff: 128,
            eval_every: 0,
            eval_prompts: 16,
            eval_temperature: 0.3,
            backend: BackendChoice::Cpu,
            save_path: None,
            resume_from: None,
            serve: None,
            grad_clip: Some(1.0),
            metrics_jsonl: None,
            objective: MultiTurnObjective::StepwiseGrpo,
            linear_attn_every: 0,
        }
    }
}

impl MultiTurnObjective {
    fn as_str(self) -> &'static str {
        match self {
            MultiTurnObjective::StepwiseGrpo => "stepwise-grpo",
            MultiTurnObjective::SequenceGspo => "gspo",
        }
    }
}

impl FromStr for MultiTurnObjective {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "stepwise-grpo" | "grpo" | "stepwise" => Ok(MultiTurnObjective::StepwiseGrpo),
            "gspo" | "sequence-gspo" | "sequence" => Ok(MultiTurnObjective::SequenceGspo),
            _ => Err(format!("unknown objective: {s}")),
        }
    }
}

const TRAIN_MODEL_FILENAME: &str = "train_model.safetensors";
const MULTI_TURN_PROMPT_SALT: u64 = 0x4D55_4C54_5052_4F4D;
const MULTI_TURN_SAMPLE_SALT: u64 = 0x4D55_4C54_5341_4D50;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct MultiTurnCheckpointMeta {
    lr: f32,
    objective: String,
    best_reward: f32,
    last_kl: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct ResumeState {
    start_iter: usize,
    best_reward: f32,
    last_kl: f32,
}

fn build_backend(choice: BackendChoice) -> Result<Arc<dyn Backend>, CliError> {
    match choice {
        BackendChoice::Cpu => Ok(Arc::new(CpuBackend)),
        #[cfg(feature = "metal")]
        BackendChoice::Metal => Ok(Arc::new(autograd::backend_metal::MetalBackend)),
        #[cfg(not(feature = "metal"))]
        BackendChoice::Metal => Err(CliError::Arg(ArgError::InvalidValue {
            flag: "--backend".into(),
            value: "metal (build with --features metal)".into(),
        })),
        #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
        BackendChoice::Cuda => {
            let backend =
                autograd::backend_cuda::CudaBackend::new(0).map_err(CliError::Autograd)?;
            Ok(Arc::new(backend))
        }
        #[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
        BackendChoice::Cuda => Err(CliError::Arg(ArgError::InvalidValue {
            flag: "--backend".into(),
            value: "cuda (build with --features cuda and no no-cuda)".into(),
        })),
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
    Qwen35(#[from] Qwen35Error),
    #[error(transparent)]
    Qwen35Checkpoint(#[from] Qwen35CheckpointError),
    #[error("{0}")]
    Custom(String),
}

struct EchoSeparator(usize);

impl Environment for EchoSeparator {
    fn observation(
        &self,
        _history: &[usize],
        _agent_start: usize,
        _agent_end: usize,
        observation_tokens: usize,
    ) -> Vec<usize> {
        vec![self.0; observation_tokens]
    }
}

#[cfg(feature = "metal")]
fn metal_eval_reset() {
    autograd::backend_metal::reset_eval_count();
}

#[cfg(not(feature = "metal"))]
fn metal_eval_reset() {}

#[cfg(feature = "metal")]
fn metal_eval_snapshot() -> Option<u64> {
    Some(autograd::backend_metal::eval_count())
}

#[cfg(not(feature = "metal"))]
fn metal_eval_snapshot() -> Option<u64> {
    None
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        // Display (not Debug): `#[error(transparent)]` delegates to the inner
        // error's Display impl, so the user sees the real message instead of
        // `Error: Custom("...")`. Mirrors the `train_sft.rs` / `train_grpo.rs`
        // pattern.
        Err(err) => {
            eprintln!("[train_multi_turn] error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), CliError> {
    let args = parse_args()?;
    run_with_args(args)
}

pub(crate) fn dispatch_from_args<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let parsed = parse_args_from(args.into_iter()).map_err(|err| err.to_string())?;
    run_with_args(parsed).map_err(|err| err.to_string())
}

fn run_with_args(args: CliArgs) -> Result<(), CliError> {
    validate_args(&args)?;
    if args.grad_clip.is_none() {
        eprintln!("[train_multi_turn] gradient clipping disabled (--no-grad-clip)");
    }
    let controller = TrainingController::new();
    let metrics = open_run_metrics(args.metrics_jsonl.as_deref(), &controller)
        .map_err(|e| CliError::Custom(format!("metrics sink: {e}")))?;
    let run_id = train::metrics::default_run_id("train_multi_turn");

    let total_agent = args.agent_tokens * args.turns;
    let total_obs = args.obs_tokens * (args.turns.saturating_sub(1));
    let seq_len = args.prompt_len + total_agent + total_obs;
    let lora = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };

    let separator = (args.vocab_size - 1).min(31);
    let config = qwen35_config(&args, seq_len, separator)?;

    let backend = build_backend(args.backend)?;
    eprintln!(
        "[train_multi_turn] backend={:?} objective={}",
        backend.device(),
        args.objective.as_str()
    );
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
    let mut store = TensorStore::with_backend(Arc::clone(&backend));
    let mut tape = Tape::new();
    let policy = Qwen35Model::new_with_lora(&config, Some(lora), &mut store)?;
    let params = trainable_param_ids(&policy, &store);
    let mut optimizer = adamw_for_backend(args.lr, (0.9, 0.999), 1.0e-8, 0.0, Arc::clone(&backend));
    let resume = if let Some(resume_dir) = resume_dir.as_deref() {
        resume_multi_turn_checkpoint(
            resume_dir,
            &args,
            &config,
            &policy,
            &mut store,
            &mut optimizer,
            lora,
        )?
    } else {
        ResumeState::default()
    };
    let ref_model = policy.clone_frozen(&mut store);
    let eval_prompt_seed = args.seed ^ 0x4556_414C_5F50_524D;
    let eval_sample_seed = args.seed ^ 0x4556_414C_5350_4C52;

    let env = EchoSeparator(separator);
    let target_range = args
        .target_range
        .min(config.vocab_size.saturating_sub(2).max(1));
    let mut turns = Vec::with_capacity(args.turns);
    for turn_idx in 0..args.turns {
        let obs = if turn_idx + 1 < args.turns {
            args.obs_tokens
        } else {
            0
        };
        turns.push(TurnSpec {
            agent_tokens: args.agent_tokens,
            observation_tokens: obs,
        });
    }

    let grpo_cfg = GrpoConfig {
        clip_eps: args.clip_eps,
        kl_coef: args.kl_coef,
        group_size: args.group_size,
    };

    let mut reward_trajectory = Vec::with_capacity(args.iters.saturating_sub(resume.start_iter));
    let mut best_reward = resume.best_reward;
    let mut last_kl = resume.last_kl;
    let loop_start = std::time::Instant::now();
    let backend_name = match args.backend {
        BackendChoice::Cpu => "cpu",
        BackendChoice::Metal => "metal",
        BackendChoice::Cuda => "cuda",
    };
    let save_path_string = args.save_path.as_deref().unwrap_or_default().to_string();
    let resume_path_string = resume_dir.as_ref().map(|path| path.display().to_string());
    let mut run_start_strings = vec![
        ("model_family", "qwen35"),
        ("backend", backend_name),
        ("objective", args.objective.as_str()),
    ];
    if !save_path_string.is_empty() {
        run_start_strings.push(("save_path", save_path_string.as_str()));
    }
    if let Some(path) = resume_path_string.as_deref() {
        run_start_strings.push(("resume_from", path));
    }
    let run_start_scalars = [
        ("iters", args.iters as f64),
        ("group_size", args.group_size as f64),
        ("turns", args.turns as f64),
        ("prompt_len", args.prompt_len as f64),
    ];
    let run_start_bools = [("resumed", resume_dir.is_some())];
    emit_run_start(
        &metrics,
        &run_id,
        "train_multi_turn",
        resume.start_iter as u64,
        &run_start_strings,
        &run_start_scalars,
        &run_start_bools,
    );

    sync_status(&controller, &metrics, |s| {
        s.total_iters = args.iters;
        s.started = true;
        s.iter = resume.start_iter;
        s.best_reward = best_reward;
        s.last_kl = last_kl;
    });
    let _server_handle = serve_if_requested("train_multi_turn", &controller, args.serve)
        .map_err(CliError::Custom)?;

    let mut stopped_early = false;
    for iter in resume.start_iter..args.iters {
        if controller.should_stop() {
            eprintln!("[train_multi_turn] stop requested at iter {iter}");
            let strings = [("run_id", run_id.as_str()), ("reason", "operator_stop")];
            metrics.emit_event(&train::metrics::TrainEvent {
                kind: "status",
                step: Some(iter as u64),
                strings: &strings,
                scalars: &[],
                bools: &[("stop_requested", true)],
            });
            stopped_early = true;
            break;
        }
        let mut prompt_rng = seeded_rng(args.seed, MULTI_TURN_PROMPT_SALT, iter as u64, 0);
        let initial_prompt =
            build_prompt(args.prompt_len, separator, target_range, &mut prompt_rng);
        let mut sample_rngs = (0..args.group_size)
            .map(|episode_idx| {
                seeded_rng(
                    args.seed,
                    MULTI_TURN_SAMPLE_SALT,
                    iter as u64,
                    episode_idx as u64,
                )
            })
            .collect::<Vec<_>>();
        let episodes = rollout_episode_group(
            &policy,
            &ref_model,
            &initial_prompt,
            &turns,
            &env,
            args.temperature,
            &mut sample_rngs,
            &|_: &Episode| 0.0,
            &mut store,
            &mut tape,
        )?;

        let per_turn_rewards: Vec<Vec<f32>> = episodes
            .iter()
            .map(|episode| compute_per_turn_rewards(episode, &initial_prompt))
            .collect();

        let trajectories: Vec<_> = episodes
            .iter()
            .map(|e| e.clone().into_trajectory())
            .collect();
        let objective_rewards = shape_objective_rewards(
            args.objective,
            &episodes,
            &per_turn_rewards,
            args.gamma,
            args.group_size,
            seq_len,
        );

        tape.entries.clear();
        tape.set_enabled(true);
        metal_eval_reset();
        let eval_count_loop_start = metal_eval_snapshot();
        let loss_id = objective_loss(
            &policy,
            &trajectories,
            &grpo_cfg,
            &config,
            &objective_rewards,
            &mut store,
            &mut tape,
        )?;
        let loss_value = store.to_host(loss_id)?[0];
        let eval_count_after_forward = metal_eval_snapshot();

        optimizer.zero_grad(&params, &mut store);
        tape.backward(loss_id, &mut store)?;
        let eval_count_after_backward = metal_eval_snapshot();
        // `clip_grad_norm` is a no-op when `max_norm` is non-positive / non-
        // finite (see its sanitize-at-boundary contract landed in 429efc3),
        // so `--grad-clip 0 / NaN / inf` collapse to "disabled" without
        // panicking. The `if let Some` gate covers the `--no-grad-clip`
        // case.
        if let Some(max_norm) = args.grad_clip {
            clip_grad_norm(&params, max_norm, &mut store);
        }
        optimizer.step(&params, &mut store);
        let eval_count_after_step = metal_eval_snapshot();

        tape.entries.clear();
        tape.set_enabled(true);
        last_kl = mean_sampled_kl(&policy, &trajectories, &config, &mut store, &mut tape)?;
        let keep = retained_ids(&[&policy, &ref_model], &store);
        store.retain_ids(&keep);

        let mean_turn_reward = mean_per_turn(&per_turn_rewards);
        reward_trajectory.push(mean_turn_reward);
        best_reward = best_reward.max(mean_turn_reward);
        // Emit via the sink only; `open_sink(_, true)` already tees to
        // StdoutSink, so a parallel `println!` would double-print.
        // Metal per-step eval-count delta lets us verify M5.3b.1–19 lazy
        // work on real training shapes (not just test_device_handle).
        // `start` is snapshotted after reset, so we report absolute deltas
        // per phase. On non-Metal builds the snapshots are None and the
        // fields are elided.
        let metal_fwd_evals = match (eval_count_loop_start, eval_count_after_forward) {
            (Some(a), Some(b)) => Some(b.saturating_sub(a) as f64),
            _ => None,
        };
        let metal_bwd_evals = match (eval_count_after_forward, eval_count_after_backward) {
            (Some(a), Some(b)) => Some(b.saturating_sub(a) as f64),
            _ => None,
        };
        let metal_opt_evals = match (eval_count_after_backward, eval_count_after_step) {
            (Some(a), Some(b)) => Some(b.saturating_sub(a) as f64),
            _ => None,
        };

        let mut metric_fields: Vec<(&str, f64)> = vec![
            ("loss", loss_value as f64),
            ("mean_reward", mean_turn_reward as f64),
            ("best_reward", best_reward as f64),
            ("mean_kl", last_kl as f64),
        ];
        if let Some(v) = metal_fwd_evals {
            metric_fields.push(("metal_evals_fwd", v));
        }
        if let Some(v) = metal_bwd_evals {
            metric_fields.push(("metal_evals_bwd", v));
        }
        if let Some(v) = metal_opt_evals {
            metric_fields.push(("metal_evals_opt", v));
        }
        metrics.emit_metric(&MetricSample {
            step: iter as u64 + 1,
            phase: args.objective.as_str(),
            fields: &metric_fields,
        });

        let wall_so_far = loop_start.elapsed().as_secs_f32();
        sync_status(&controller, &metrics, |s| {
            s.iter = iter + 1;
            s.mean_reward = mean_turn_reward;
            s.best_reward = best_reward;
            s.last_kl = last_kl;
            s.last_loss = loss_value;
            s.wall_secs = wall_so_far;
        });

        if controller.take_save_request() {
            if let Some(path) = &args.save_path {
                let step_dir = save_qwen35_checkpoint(
                    path,
                    iter + 1,
                    &args,
                    &config,
                    &policy,
                    &optimizer,
                    &mut store,
                    lora,
                    best_reward,
                    last_kl,
                )?;
                eprintln!(
                    "[train_multi_turn] save requested → flushed to {}",
                    step_dir.display()
                );
                let step_dir_string = step_dir.display().to_string();
                let strings = [
                    ("run_id", run_id.as_str()),
                    ("path", step_dir_string.as_str()),
                    ("artifact_model", "model.safetensors"),
                    ("artifact_train_model", "train_model.safetensors"),
                    ("artifact_adapter", "adapter_model.safetensors"),
                    ("artifact_adapter_config", "adapter_config.json"),
                    ("artifact_state", "trainer_state.json"),
                    ("artifact_optimizer", "optimizer.safetensors"),
                ];
                metrics.emit_event(&train::metrics::TrainEvent {
                    kind: "checkpoint",
                    step: Some((iter + 1) as u64),
                    strings: &strings,
                    scalars: &[],
                    bools: &[],
                });
            } else {
                eprintln!(
                    "[train_multi_turn] save requested but no --save-path configured; ignoring"
                );
                let strings = [("run_id", run_id.as_str()), ("reason", "save_without_path")];
                metrics.emit_event(&train::metrics::TrainEvent {
                    kind: "status",
                    step: Some((iter + 1) as u64),
                    strings: &strings,
                    scalars: &[],
                    bools: &[("save_requested", true)],
                });
            }
        }

        if args.eval_every > 0 && (iter + 1).is_multiple_of(args.eval_every) {
            let mut eval_prompt_rng = LcgRng::seed(eval_prompt_seed);
            let mut eval_sample_rng = LcgRng::seed(eval_sample_seed ^ iter as u64);
            let (eval_reward, eval_passrate) = run_eval(
                &policy,
                &ref_model,
                args.eval_prompts,
                args.prompt_len,
                separator,
                target_range,
                &turns,
                &env,
                args.eval_temperature,
                &mut eval_prompt_rng,
                &mut eval_sample_rng,
                &mut store,
                &mut tape,
            )?;
            let keep = retained_ids(&[&policy, &ref_model], &store);
            store.retain_ids(&keep);
            println!(
                "eval @ iter {iter}: mean_reward {eval_reward:.4} pass@1 {eval_passrate:.4} \
                 (prompts={}, temperature={:.2})",
                args.eval_prompts, args.eval_temperature
            );
            let eval_fields = [
                ("eval_mean_reward", eval_reward as f64),
                ("eval_pass_rate", eval_passrate as f64),
                ("eval_prompts", args.eval_prompts as f64),
            ];
            metrics.emit_metric(&MetricSample {
                step: iter as u64 + 1,
                phase: "eval",
                fields: &eval_fields,
            });
        }
    }

    let wall_secs = loop_start.elapsed().as_secs_f32();
    let total_episodes = args.iters * args.group_size;
    let tokens_per_episode = seq_len;
    let total_tokens = total_episodes * tokens_per_episode;
    let iter_per_sec = args.iters as f32 / wall_secs.max(1e-6);
    let episodes_per_sec = total_episodes as f32 / wall_secs.max(1e-6);
    let tokens_per_sec = total_tokens as f32 / wall_secs.max(1e-6);
    println!("final kl {last_kl:.4}");
    println!("reward trajectory: {reward_trajectory:?}");
    println!(
        "bench: wall {wall_secs:.2}s | iter/s {iter_per_sec:.2} | episode/s {episodes_per_sec:.2} \
         | token/s {tokens_per_sec:.1} | seq_len {seq_len} | group {group}",
        group = args.group_size,
    );

    if let Some(path) = &args.save_path {
        let step_dir = save_qwen35_checkpoint(
            path,
            controller.snapshot().iter,
            &args,
            &config,
            &policy,
            &optimizer,
            &mut store,
            lora,
            best_reward,
            last_kl,
        )?;
        println!("checkpoint saved to {}", step_dir.display());
        let step_dir_string = step_dir.display().to_string();
        let strings = [
            ("run_id", run_id.as_str()),
            ("path", step_dir_string.as_str()),
            ("artifact_model", "model.safetensors"),
            ("artifact_train_model", "train_model.safetensors"),
            ("artifact_adapter", "adapter_model.safetensors"),
            ("artifact_adapter_config", "adapter_config.json"),
            ("artifact_state", "trainer_state.json"),
            ("artifact_optimizer", "optimizer.safetensors"),
        ];
        metrics.emit_event(&train::metrics::TrainEvent {
            kind: "checkpoint",
            step: Some(controller.snapshot().iter as u64),
            strings: &strings,
            scalars: &[],
            bools: &[],
        });
    }

    sync_status(&controller, &metrics, |s| {
        s.wall_secs = wall_secs;
        s.finished = true;
    });
    if stopped_early {
        eprintln!(
            "[train_multi_turn] training stopped early at iter {}",
            controller.snapshot().iter
        );
    }
    let status = if stopped_early {
        "stopped"
    } else {
        "completed"
    };
    let run_end_scalars = [
        ("completed_iters", controller.snapshot().iter as f64),
        ("best_reward", best_reward as f64),
        ("last_kl", last_kl as f64),
        ("wall_secs", wall_secs as f64),
        ("dropped_metrics", metrics.dropped_metrics() as f64),
    ];
    emit_run_end(
        &metrics,
        &run_id,
        status,
        controller.snapshot().iter as u64,
        &run_end_scalars,
    );
    metrics.flush_blocking();
    Ok(())
}

fn qwen35_config(
    args: &CliArgs,
    seq_len: usize,
    separator: usize,
) -> Result<Qwen35Config, CliError> {
    if args.d_model != args.n_heads * args.d_head {
        return Err(CliError::Custom(format!(
            "--d-model {} must equal --n-heads {} * --d-head {} for qwen3.5-family training",
            args.d_model, args.n_heads, args.d_head
        )));
    }
    let mut cfg = Qwen35Config {
        hidden_size: args.d_model,
        intermediate_size: args.d_ff,
        num_hidden_layers: args.n_layers,
        vocab_size: args.vocab_size,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![separator as u32],
        bos_token_id: Some(0),
        eos_token_id: separator as u32,
        tie_word_embeddings: false,
        num_attention_heads: args.n_heads,
        num_key_value_heads: args.n_heads,
        head_dim: args.d_head,
        linear_num_key_heads: args.n_heads,
        linear_key_head_dim: args.d_head,
        linear_num_value_heads: args.n_heads,
        linear_value_head_dim: args.d_head,
        linear_conv_kernel_dim: 4,
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 1.0,
        rotary_dim: args.d_head,
        rope_cache_len_hint: Some(seq_len.max(32)),
        layer_types: vec![LayerType::FullAttention; args.n_layers],
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
    };
    let pattern = if args.linear_attn_every == 0 {
        Qwen35AttentionPattern::Dense
    } else {
        Qwen35AttentionPattern::Hybrid {
            linear_attn_every: args.linear_attn_every,
        }
    };
    apply_qwen35_attention_pattern(&mut cfg, pattern)
        .map_err(|err| CliError::Custom(err.to_string()))?;
    Ok(cfg)
}

fn seeded_rng(seed: u64, salt: u64, major: u64, minor: u64) -> LcgRng {
    let mut mixed = seed ^ salt;
    mixed ^= major.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    mixed = mixed.rotate_left(17);
    mixed ^= minor.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = mixed.rotate_left(11);
    LcgRng::seed(mixed)
}

fn resume_multi_turn_checkpoint(
    resume_dir: &std::path::Path,
    args: &CliArgs,
    cfg: &Qwen35Config,
    model: &Qwen35Model,
    store: &mut TensorStore,
    optimizer: &mut AdamW,
    lora: LoraConfig,
) -> Result<ResumeState, CliError> {
    validate_qwen35_resume_config(resume_dir, cfg)?;
    validate_adapter_resume_config(resume_dir, lora)?;

    let train_model_path = resume_dir.join(TRAIN_MODEL_FILENAME);
    if !train_model_path.is_file() {
        return Err(CliError::Custom(format!(
            "--resume-from {} has no {TRAIN_MODEL_FILENAME}",
            resume_dir.display()
        )));
    }
    let mut registry = build_registry(model);
    registry.load_into_strict(store, &train_model_path)?;

    let adapter_path = resume_dir.join("adapter_model.safetensors");
    let mut adapter_registry = build_adapter_registry(model);
    if !adapter_registry.is_empty() {
        if !adapter_path.is_file() {
            return Err(CliError::Custom(format!(
                "--resume-from {} has no adapter_model.safetensors",
                resume_dir.display()
            )));
        }
        adapter_registry.load_into_strict(store, &adapter_path)?;
    }

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
    let meta: MultiTurnCheckpointMeta = serde_json::from_value(trainer_doc.schedule_params)
        .map_err(|err| {
            CliError::Custom(format!("resume checkpoint metadata parse error: {err}"))
        })?;
    if (meta.lr - args.lr).abs() > 1.0e-8 {
        return Err(CliError::Custom(format!(
            "--resume-from {} lr mismatch: checkpoint={} live={}",
            resume_dir.display(),
            meta.lr,
            args.lr
        )));
    }
    if meta.objective != args.objective.as_str() {
        return Err(CliError::Custom(format!(
            "--resume-from {} objective mismatch: checkpoint={} live={}",
            resume_dir.display(),
            meta.objective,
            args.objective.as_str()
        )));
    }
    let param_names = trainable_param_name_map(model, store);
    let restored = optimizer
        .import_state(&optim_state, &param_names)
        .map_err(|err| CliError::Custom(format!("resume optimizer import failed: {err}")))?;
    eprintln!(
        "[train_multi_turn] resumed step {} with {restored} optimizer entries from {}",
        trainer_doc.step,
        resume_dir.display()
    );
    Ok(ResumeState {
        start_iter: trainer_doc.step as usize,
        best_reward: meta.best_reward,
        last_kl: meta.last_kl,
    })
}

fn save_qwen35_checkpoint(
    out_dir: &str,
    step: usize,
    args: &CliArgs,
    cfg: &Qwen35Config,
    model: &Qwen35Model,
    optimizer: &AdamW,
    store: &mut TensorStore,
    lora: LoraConfig,
    best_reward: f32,
    last_kl: f32,
) -> Result<PathBuf, CliError> {
    let out_dir = PathBuf::from(out_dir);
    let keep_ids = live_tensor_ids(store);
    let step_dir = save_step_checkpoint(
        Qwen35StepCheckpoint {
            out_dir: out_dir.as_path(),
            step,
            tokenizer_path: None,
            config_json: ConfigJsonSource::Synthesize {
                cfg,
                torch_dtype: "float32",
            },
            generation_config: GenerationConfigSource::Synthesize {
                bos_token_id: cfg.bos_token_id,
                eos_token_id: cfg.eos_token_id,
            },
        },
        |weights_path| {
            let mut tape = Tape::new();
            save_materialized_registry(model, store, &mut tape, weights_path, false)
                .map_err(Into::into)
        },
    )?;
    store.retain_ids(&keep_ids);
    let train_registry = build_registry(model);
    train_registry.save_from(store, &step_dir.join(TRAIN_MODEL_FILENAME))?;
    let adapter_registry = build_adapter_registry(model);
    if !adapter_registry.is_empty() {
        adapter_registry.save_from(store, &step_dir.join("adapter_model.safetensors"))?;
        let adapter_config = LoraAdapterConfig::new("synthetic://qwen35", "qwen35", lora);
        std::fs::write(
            step_dir.join("adapter_config.json"),
            serde_json::to_string_pretty(&adapter_config)?,
        )?;
    }
    let trainer_doc = TrainerStateDoc {
        step: step as u64,
        optim_schema: optimizer.state_schema().to_string(),
        schedule_name: "constant".to_string(),
        schedule_params: serde_json::to_value(MultiTurnCheckpointMeta {
            lr: args.lr,
            objective: args.objective.as_str().to_string(),
            best_reward,
            last_kl,
        })?,
        grad_accum_current: 0,
        rng_seed: args.seed,
        codec_version: TRAINER_STATE_CODEC_VERSION,
    };
    let optim_state = optimizer.export_state(&trainable_param_name_map(model, store));
    save_trainer_state_v2(&step_dir, &trainer_doc, &optim_state)
        .map_err(|err| CliError::Custom(format!("save trainer state: {err}")))?;
    Ok(step_dir)
}

fn validate_qwen35_resume_config(
    resume_dir: &std::path::Path,
    cfg: &Qwen35Config,
) -> Result<(), CliError> {
    let cfg_path = resume_dir.join("config.json");
    if !cfg_path.is_file() {
        return Err(CliError::Custom(format!(
            "--resume-from {} has no config.json",
            resume_dir.display()
        )));
    }
    let saved = Qwen35Config::from_json_file(&cfg_path).map_err(|err| {
        CliError::Custom(format!(
            "resume config {} parse error: {err}",
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

fn validate_adapter_resume_config(
    resume_dir: &std::path::Path,
    lora: LoraConfig,
) -> Result<(), CliError> {
    let path = resume_dir.join("adapter_config.json");
    if !path.is_file() {
        return Err(CliError::Custom(format!(
            "--resume-from {} has no adapter_config.json",
            resume_dir.display()
        )));
    }
    let saved: LoraAdapterConfig =
        serde_json::from_str(&std::fs::read_to_string(&path)?).map_err(|err| {
            CliError::Custom(format!(
                "resume adapter config {} parse error: {err}",
                path.display()
            ))
        })?;
    let expected = LoraAdapterConfig::new("synthetic://qwen35", "qwen35", lora);
    if saved != expected {
        return Err(CliError::Custom(format!(
            "--resume-from {} adapter mismatch with live LoRA config",
            resume_dir.display()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_eval<P>(
    policy: &P,
    ref_model: &P,
    n_prompts: usize,
    prompt_len: usize,
    separator: usize,
    target_range: usize,
    turns: &[TurnSpec],
    env: &EchoSeparator,
    temperature: f32,
    prompt_rng: &mut LcgRng,
    sample_rng: &mut LcgRng,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<(f32, f32), CliError>
where
    P: GrpoPolicy,
    P::Config: GrpoPolicyConfig,
{
    if n_prompts == 0 {
        return Ok((0.0, 0.0));
    }
    let mut total_reward = 0.0_f32;
    let mut total_turns = 0.0_f32;
    let mut pass_count = 0.0_f32;
    for _ in 0..n_prompts {
        let initial_prompt = build_prompt(prompt_len, separator, target_range, prompt_rng);
        let episode = rollout_episode(
            policy,
            ref_model,
            &initial_prompt,
            turns,
            env,
            temperature,
            sample_rng,
            &|_: &Episode| 0.0,
            store,
            tape,
        )?;
        let per_turn = compute_per_turn_rewards(&episode, &initial_prompt);
        let n_turns = per_turn.len();
        if n_turns == 0 {
            continue;
        }
        let episode_mean: f32 = per_turn.iter().sum::<f32>() / n_turns as f32;
        total_reward += episode_mean * n_turns as f32;
        total_turns += n_turns as f32;
        if episode_mean >= 1.0 - 1.0e-4 {
            pass_count += 1.0;
        }
    }
    let mean_reward = if total_turns > 0.0 {
        total_reward / total_turns
    } else {
        0.0
    };
    let pass_rate = pass_count / n_prompts as f32;
    Ok((mean_reward, pass_rate))
}

fn compute_per_turn_rewards(episode: &Episode, initial_prompt: &[usize]) -> Vec<f32> {
    // Turn-t reward: fraction of agent tokens that copy `initial_prompt[t % prompt_len]`.
    let prompt_len = initial_prompt.len();
    let mut rewards = Vec::with_capacity(episode.turn_boundaries.len());
    for (turn_idx, (start, end)) in episode.turn_boundaries.iter().enumerate() {
        let target = initial_prompt[turn_idx % prompt_len];
        let mut hits = 0.0_f32;
        let mut total = 0.0_f32;
        for position in *start..*end {
            total += 1.0;
            if episode.full_ids[position] == target {
                hits += 1.0;
            }
        }
        rewards.push(if total == 0.0 { 0.0 } else { hits / total });
    }
    rewards
}

fn stepwise_advantages(
    episodes: &[Episode],
    per_turn_rewards: &[Vec<f32>],
    gamma: f32,
    group: usize,
    seq_len: usize,
) -> Vec<f32> {
    let n_turns = episodes[0].turn_boundaries.len();
    let mut returns_per_ep: Vec<Vec<f32>> = Vec::with_capacity(episodes.len());
    for rewards in per_turn_rewards {
        returns_per_ep.push(discounted_returns(rewards, gamma));
    }

    // Interleave so same-turn-across-episodes forms a normalization group.
    let mut stacked = Vec::with_capacity(group * n_turns);
    for turn_idx in 0..n_turns {
        for ep_returns in &returns_per_ep {
            stacked.push(ep_returns[turn_idx]);
        }
    }
    let normalized = group_normalize(&stacked, group);

    let mut per_position = Vec::with_capacity(group * seq_len);
    for (ep, episode) in episodes.iter().enumerate() {
        let mut row_adv = vec![0.0_f32; n_turns];
        for (turn_idx, slot) in row_adv.iter_mut().enumerate() {
            *slot = normalized[turn_idx * group + ep];
        }
        let row = returns_to_per_position(&row_adv, &episode.turn_boundaries, seq_len);
        per_position.extend_from_slice(&row);
    }
    per_position
}

#[derive(Debug, Clone, PartialEq)]
enum ObjectiveRewards {
    Stepwise { advantages_per_position: Vec<f32> },
    Sequence { advantages: Vec<f32> },
}

fn shape_objective_rewards(
    objective: MultiTurnObjective,
    episodes: &[Episode],
    per_turn_rewards: &[Vec<f32>],
    gamma: f32,
    group: usize,
    seq_len: usize,
) -> ObjectiveRewards {
    match objective {
        MultiTurnObjective::StepwiseGrpo => ObjectiveRewards::Stepwise {
            advantages_per_position: stepwise_advantages(
                episodes,
                per_turn_rewards,
                gamma,
                group,
                seq_len,
            ),
        },
        MultiTurnObjective::SequenceGspo => {
            let sequence_scores: Vec<f32> = per_turn_rewards
                .iter()
                .map(|rewards| episode_sequence_score(rewards))
                .collect();
            ObjectiveRewards::Sequence {
                advantages: group_advantages(&sequence_scores, group),
            }
        }
    }
}

fn objective_loss(
    policy: &Qwen35Model,
    trajectories: &[Trajectory],
    grpo_cfg: &GrpoConfig,
    config: &Qwen35Config,
    rewards: &ObjectiveRewards,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<autograd::TensorId, CliError> {
    match rewards {
        ObjectiveRewards::Stepwise {
            advantages_per_position,
        } => grpo_loss_per_position(
            policy,
            trajectories,
            advantages_per_position,
            grpo_cfg,
            config,
            store,
            tape,
        )
        .map_err(CliError::Autograd),
        ObjectiveRewards::Sequence { advantages } => grpo_loss(
            policy,
            trajectories,
            advantages,
            grpo_cfg,
            config,
            store,
            tape,
        )
        .map_err(CliError::Autograd),
    }
}

// GSPO sequence score: the mean per-turn reward for the full episode.
// This keeps the sequence-level objective grounded in the same verifier
// signal the stepwise path uses, but collapses the episode to one scalar
// before group normalization and GRPO broadcast.
fn episode_sequence_score(per_turn_rewards: &[f32]) -> f32 {
    if per_turn_rewards.is_empty() {
        0.0
    } else {
        per_turn_rewards.iter().sum::<f32>() / per_turn_rewards.len() as f32
    }
}

fn mean_per_turn(per_turn_rewards: &[Vec<f32>]) -> f32 {
    let mut sum = 0.0_f32;
    let mut count = 0.0_f32;
    for row in per_turn_rewards {
        for reward in row {
            sum += *reward;
            count += 1.0;
        }
    }
    if count == 0.0 { 0.0 } else { sum / count }
}

fn build_prompt(
    prompt_len: usize,
    separator: usize,
    target_range: usize,
    rng: &mut LcgRng,
) -> Vec<usize> {
    assert!(prompt_len >= 2, "prompt length must be ≥ 2");
    let span = target_range.max(1) as u64;
    let mut prompt = vec![0usize; prompt_len];
    for slot in &mut prompt[..prompt_len - 1] {
        *slot = 1 + (rng.next_u64() % span) as usize;
    }
    prompt[prompt_len - 1] = separator;
    prompt
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
            "--iters" => args.iters = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--group-size" => {
                args.group_size = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--agent-tokens" => {
                args.agent_tokens = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--obs-tokens" => {
                args.obs_tokens = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--turns" => args.turns = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--prompt-len" => {
                args.prompt_len = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--lr" => args.lr = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--kl-coef" => args.kl_coef = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--clip-eps" => args.clip_eps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--temperature" => {
                args.temperature = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--gamma" => args.gamma = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--lora-rank" => {
                args.lora_rank = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--lora-alpha" => {
                args.lora_alpha = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--seed" => args.seed = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--vocab" => {
                args.vocab_size = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--target-range" => {
                args.target_range = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--d-model" => args.d_model = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--n-layers" => args.n_layers = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--n-heads" => args.n_heads = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--d-head" => args.d_head = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--d-ff" => args.d_ff = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--linear-attn-every" => {
                args.linear_attn_every = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--eval-every" => {
                args.eval_every = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--eval-prompts" => {
                args.eval_prompts = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--eval-temperature" => {
                args.eval_temperature = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--backend" => {
                let value = next_value(&mut iter, &flag)?;
                args.backend = value.parse().map_err(|_| {
                    CliError::Arg(ArgError::InvalidValue {
                        flag: flag.clone(),
                        value,
                    })
                })?;
            }
            "--save-path" => {
                args.save_path = Some(next_value(&mut iter, &flag)?);
            }
            "--resume-from" => {
                args.resume_from = Some(PathBuf::from(next_value(&mut iter, &flag)?));
            }
            "--serve" => {
                args.serve = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--grad-clip" => {
                args.grad_clip = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--no-grad-clip" => args.grad_clip = None,
            "--metrics-jsonl" => {
                args.metrics_jsonl = Some(PathBuf::from(next_value(&mut iter, &flag)?));
            }
            "--objective" => {
                let value = next_value(&mut iter, &flag)?;
                args.objective = value.parse().map_err(|_| {
                    CliError::Arg(ArgError::InvalidValue {
                        flag: flag.clone(),
                        value,
                    })
                })?;
            }
            _ => return Err(CliError::Arg(ArgError::UnknownFlag(flag))),
        }
    }
    Ok(args)
}

fn validate_args(args: &CliArgs) -> Result<(), CliError> {
    if args.turns == 0 || args.group_size == 0 || args.agent_tokens == 0 {
        return Err(CliError::Autograd(AutogradError::InvalidRank {
            expected: "positive turns, group_size, agent_tokens",
            got: 0,
        }));
    }
    if args.prompt_len < 2 {
        return Err(CliError::Autograd(AutogradError::InvalidRank {
            expected: "prompt_len >= 2",
            got: args.prompt_len,
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
    if args.linear_attn_every > args.n_layers && args.n_layers > 0 {
        return Err(CliError::Custom(format!(
            "--linear-attn-every {} produces no linear-attention layers for --n-layers {}",
            args.linear_attn_every, args.n_layers
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen35_spec::{LayerType, Qwen35AttentionTensorNames};
    use tempfile::tempdir;
    use train::{LoraAdapterConfig, checkpoint::load_trainer_state_v2};

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn tiny_args() -> CliArgs {
        CliArgs {
            iters: 4,
            group_size: 2,
            agent_tokens: 2,
            obs_tokens: 1,
            turns: 2,
            prompt_len: 4,
            lr: 1.0e-4,
            kl_coef: 0.02,
            clip_eps: 0.2,
            temperature: 1.0,
            gamma: 0.9,
            lora_rank: 2,
            lora_alpha: 4.0,
            seed: 42,
            vocab_size: 128,
            target_range: 8,
            d_model: 32,
            n_layers: 2,
            n_heads: 4,
            d_head: 8,
            d_ff: 64,
            eval_every: 0,
            eval_prompts: 4,
            eval_temperature: 0.3,
            backend: BackendChoice::Cpu,
            save_path: None,
            resume_from: None,
            serve: None,
            grad_clip: Some(1.0),
            metrics_jsonl: None,
            objective: MultiTurnObjective::SequenceGspo,
            linear_attn_every: 0,
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
    fn objective_mode_parser_accepts_stepwise_and_sequence_modes() {
        assert_eq!(
            "stepwise-grpo"
                .parse::<MultiTurnObjective>()
                .expect("stepwise"),
            MultiTurnObjective::StepwiseGrpo
        );
        assert_eq!(
            "gspo".parse::<MultiTurnObjective>().expect("gspo"),
            MultiTurnObjective::SequenceGspo
        );
    }

    #[test]
    fn qwen35_config_builds_hybrid_layers() -> TestResult {
        let mut args = tiny_args();
        args.linear_attn_every = 2;
        let cfg = qwen35_config(&args, 16, 31)?;
        assert_eq!(
            cfg.layer_types,
            vec![LayerType::FullAttention, LayerType::LinearAttention]
        );
        assert!(cfg.rotary_dim < cfg.head_dim);
        cfg.validate_train_scratch_contract()?;
        Ok(())
    }

    fn synthetic_episode(turn_boundaries: &[(usize, usize)], response_mask: &[bool]) -> Episode {
        let len = response_mask.len();
        Episode {
            initial_prompt: vec![0; len],
            full_ids: vec![0; len],
            response_mask: response_mask.to_vec(),
            old_log_probs: vec![0.0; len],
            ref_log_probs: vec![0.0; len],
            reward: 0.0,
            turn_boundaries: turn_boundaries.to_vec(),
        }
    }

    #[test]
    fn objective_reward_shaping_uses_stepwise_returns_per_position() {
        let episodes = vec![
            synthetic_episode(&[(0, 2), (2, 4)], &[false, false, true, true]),
            synthetic_episode(&[(0, 2), (2, 4)], &[false, false, true, true]),
        ];
        let per_turn_rewards = vec![vec![1.0, 0.0], vec![0.0, 0.5]];
        let shaped = shape_objective_rewards(
            MultiTurnObjective::StepwiseGrpo,
            &episodes,
            &per_turn_rewards,
            0.5,
            2,
            4,
        );
        let expected = ObjectiveRewards::Stepwise {
            advantages_per_position: stepwise_advantages(&episodes, &per_turn_rewards, 0.5, 2, 4),
        };
        assert_eq!(shaped, expected);
    }

    #[test]
    fn objective_reward_shaping_uses_sequence_mean_per_episode() {
        let episodes = vec![
            synthetic_episode(&[(0, 2)], &[false, true, true, false]),
            synthetic_episode(&[(0, 2)], &[false, true, true, false]),
        ];
        let per_turn_rewards = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let shaped = shape_objective_rewards(
            MultiTurnObjective::SequenceGspo,
            &episodes,
            &per_turn_rewards,
            0.5,
            2,
            4,
        );
        let expected_scores: Vec<f32> = per_turn_rewards
            .iter()
            .map(|rewards| episode_sequence_score(rewards))
            .collect();
        let expected = ObjectiveRewards::Sequence {
            advantages: group_advantages(&expected_scores, 2),
        };
        assert_eq!(shaped, expected);
    }

    #[test]
    fn save_qwen35_checkpoint_writes_merged_weights_and_adapter_artifacts() -> TestResult {
        let tmp = tempdir().expect("tempdir");
        let out_dir = tmp.path().join("out");
        std::fs::create_dir_all(&out_dir).expect("create out dir");

        let cfg = Qwen35Config {
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
        };
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
        };
        let args = tiny_args();
        let layer_names = cfg.layer_tensor_names(0);
        let Qwen35AttentionTensorNames::Full(attn_names) = layer_names.attention else {
            unreachable!("test config uses full attention");
        };

        let mut expected_store = TensorStore::default();
        let expected_model = Qwen35Model::new_with_lora(&cfg, Some(lora), &mut expected_store)?;
        let mut save_store = TensorStore::default();
        let save_model = Qwen35Model::new_with_lora(&cfg, Some(lora), &mut save_store)?;
        let optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);

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
        let materialized = train::causal_lm::build_materialized_registry(
            &expected_model,
            &mut expected_store,
            &mut expected_tape,
        )?;
        let expected_q = expected_store.to_host(
            materialized
                .get(attn_names.q_proj.as_str())
                .expect("materialized q proj"),
        )?;

        let step_dir = save_qwen35_checkpoint(
            out_dir.to_str().expect("utf8 out dir"),
            4,
            &args,
            &cfg,
            &save_model,
            &optimizer,
            &mut save_store,
            lora,
            0.5,
            0.125,
        )?;

        assert!(step_dir.join("model.safetensors").is_file());
        assert!(step_dir.join(TRAIN_MODEL_FILENAME).is_file());
        assert!(step_dir.join("adapter_model.safetensors").is_file());
        assert!(step_dir.join("adapter_config.json").is_file());
        assert!(step_dir.join("trainer_state.json").is_file());
        assert!(step_dir.join("optimizer.safetensors").is_file());
        let adapter_config: LoraAdapterConfig = serde_json::from_str(&std::fs::read_to_string(
            step_dir.join("adapter_config.json"),
        )?)?;
        assert_eq!(adapter_config.base_model_name_or_path, "synthetic://qwen35");
        assert_eq!(
            adapter_config.target_modules,
            vec!["all-linear".to_string()]
        );
        let (trainer_doc, _) = load_trainer_state_v2(&step_dir)?;
        let meta: MultiTurnCheckpointMeta = serde_json::from_value(trainer_doc.schedule_params)?;
        assert_eq!(meta.objective, args.objective.as_str());
        assert_eq!(meta.best_reward, 0.5);
        assert_eq!(meta.last_kl, 0.125);

        let mut load_store = TensorStore::default();
        let load_model = Qwen35Model::new_with_lora(&cfg, None, &mut load_store)?;
        let mut registry = train::causal_lm::build_registry(&load_model);
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

    fn assert_resume_qwen35_checkpoint_restores_train_state_exactly(
        linear_attn_every: usize,
    ) -> TestResult {
        let tmp = tempdir().expect("tempdir");
        let out_dir = tmp.path().join("out");
        std::fs::create_dir_all(&out_dir).expect("create out dir");

        let mut args = tiny_args();
        args.linear_attn_every = linear_attn_every;
        let cfg = qwen35_config(&args, 16, 2)?;
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
        };
        let layer_names = cfg.layer_tensor_names(0);
        let Qwen35AttentionTensorNames::Full(attn_names) = layer_names.attention else {
            unreachable!("test config uses full attention");
        };

        let mut save_store = TensorStore::default();
        let save_model = Qwen35Model::new_with_lora(&cfg, Some(lora), &mut save_store)?;
        let q_proj = *save_model
            .param_name_map()
            .get(attn_names.q_proj.as_str())
            .expect("q proj");
        save_store.get_mut(q_proj).expect("q proj exists").data[0] = 3.25;
        let adapter_map = save_model.adapter_name_map();
        let q_proj_a = *adapter_map
            .get(format!("{}.lora_a", attn_names.q_proj).as_str())
            .expect("adapter a");
        let q_proj_b = *adapter_map
            .get(format!("{}.lora_b", attn_names.q_proj).as_str())
            .expect("adapter b");
        save_store.get_mut(q_proj_a).expect("adapter a exists").data[0] = 1.5;
        save_store.get_mut(q_proj_b).expect("adapter b exists").data[0] = -0.75;

        let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
        let param_names = trainable_param_name_map(&save_model, &save_store);
        let saved_state = autograd::adamw_state::AdamWState {
            step: 3,
            skipped_export: 0,
            params: param_names
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
        optimizer.import_state(&saved_state, &param_names)?;
        let saved_q = save_store.to_host(q_proj)?;
        let saved_a = save_store.to_host(q_proj_a)?;
        let saved_b = save_store.to_host(q_proj_b)?;

        let step_dir = save_qwen35_checkpoint(
            out_dir.to_str().expect("utf8 out dir"),
            3,
            &args,
            &cfg,
            &save_model,
            &optimizer,
            &mut save_store,
            lora,
            0.75,
            0.125,
        )?;

        let mut resumed_store = TensorStore::default();
        let resumed_model = Qwen35Model::new_with_lora(&cfg, Some(lora), &mut resumed_store)?;
        let mut resumed_optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
        let resume = resume_multi_turn_checkpoint(
            &step_dir,
            &args,
            &cfg,
            &resumed_model,
            &mut resumed_store,
            &mut resumed_optimizer,
            lora,
        )?;
        assert_eq!(resume.start_iter, 3);
        assert_eq!(resume.best_reward, 0.75);
        assert_eq!(resume.last_kl, 0.125);

        let resumed_q = resumed_store.to_host(
            *resumed_model
                .param_name_map()
                .get(attn_names.q_proj.as_str())
                .expect("resumed q proj"),
        )?;
        let resumed_adapter_map = resumed_model.adapter_name_map();
        let resumed_a = resumed_store.to_host(
            *resumed_adapter_map
                .get(format!("{}.lora_a", attn_names.q_proj).as_str())
                .expect("resumed adapter a"),
        )?;
        let resumed_b = resumed_store.to_host(
            *resumed_adapter_map
                .get(format!("{}.lora_b", attn_names.q_proj).as_str())
                .expect("resumed adapter b"),
        )?;
        assert_eq!(resumed_q, saved_q);
        assert_eq!(resumed_a, saved_a);
        assert_eq!(resumed_b, saved_b);

        let resumed_param_names = trainable_param_name_map(&resumed_model, &resumed_store);
        let resumed_state = resumed_optimizer.export_state(&resumed_param_names);
        assert_adamw_state_eq(&saved_state, &resumed_state);
        Ok(())
    }

    #[test]
    fn resume_qwen35_checkpoint_restores_train_state_exactly() -> TestResult {
        assert_resume_qwen35_checkpoint_restores_train_state_exactly(0)
    }

    #[test]
    fn resume_hybrid_qwen35_checkpoint_restores_train_state_exactly() -> TestResult {
        assert_resume_qwen35_checkpoint_restores_train_state_exactly(2)
    }
}

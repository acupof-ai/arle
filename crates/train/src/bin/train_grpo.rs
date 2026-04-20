use std::{collections::HashSet, env, path::PathBuf, process::ExitCode};

use autograd::{
    AutogradError, ConstantLr, Result as AutogradResult, Tape, TensorId, TensorStore,
    adamw_state::AdamWState, optim::AdamW,
};
use thiserror::Error;
use train::{
    StepCtx, StepOutcome, Trainer, TrainerConfig,
    cli_args::{ArgError, next_value, parse_value},
    dataset::{CopyDataset, Dataset, LcgRng},
    grad_clip::{GlobalNorm, GradClip, NoClip, clip_grad_norm},
    grpo::{GrpoConfig, group_advantages, grpo_loss, mean_sampled_kl},
    loss::cross_entropy_loss,
    qwen3::{Qwen3Config, Qwen3Error, Qwen3Model},
    qwen3_support::trainable_params,
    rollout::rollout_group,
};

#[derive(Debug, Clone)]
struct CliArgs {
    sft_steps: usize,
    grpo_iters: usize,
    batch_prompts: usize,
    group_size: usize,
    seq: usize,
    lr: f32,
    kl_coef: f32,
    temperature: f32,
    seed: u64,
    grad_clip: Option<f32>,
    metrics_jsonl: Option<PathBuf>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            sft_steps: 30,
            grpo_iters: 20,
            batch_prompts: 4,
            group_size: 4,
            seq: 16,
            lr: 1.0e-4,
            kl_coef: 0.02,
            temperature: 1.0,
            seed: 42,
            grad_clip: Some(1.0),
            metrics_jsonl: None,
        }
    }
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Arg(#[from] ArgError),
    #[error(transparent)]
    Qwen3(#[from] Qwen3Error),
    #[error("{0}")]
    Custom(String),
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

    let config = qwen3_config(args.seq);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Qwen3Model::new(&config, &mut store)?;
    let params = trainable_params(&policy, &store);
    let keep_extra = policy
        .all_parameter_ids()
        .into_iter()
        .collect::<HashSet<_>>();
    let mut prompt_rng = LcgRng::seed(args.seed ^ 0x4752_504F_5052_4F4D);
    let verifier = |prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]| {
        copy_reward(prompt_ids, full_ids, response_mask)
    };

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
        &args,
        &policy,
        &params,
        keep_extra.clone(),
        &mut store,
        &mut tape,
    )?;

    // Phase 4 follow-up: extend `--metrics-jsonl` to cover the GRPO phase.
    // `run_sft_phase` already truncated the JSONL file (JsonlSink::create),
    // so the GRPO phase opens in APPEND mode to extend the same file
    // instead of clobbering SFT-phase samples. `also_stdout = false` keeps
    // the human-readable `grpo iter N:` println below as the stdout
    // contract; the MetricSample emit is JSONL-only tooling output.
    let mut grpo_metrics = train::metrics::open_sink_append(args.metrics_jsonl.as_deref(), false)
        .map_err(|e| CliError::Custom(format!("grpo metrics sink: {e}")))?;

    // ---- GRPO phase (hand-written; see commit body for the "why") ----
    let ref_model = policy.clone_frozen(&mut store);
    let baseline_prompts =
        build_prompt_batch(args.batch_prompts, args.seq, 64, 255, &mut prompt_rng);
    let baseline_trajectories = rollout_group(
        &policy,
        &ref_model,
        &config,
        &baseline_prompts,
        args.group_size,
        args.temperature,
        &mut prompt_rng,
        &verifier,
        &mut store,
        &mut tape,
    )?;
    let baseline_reward = mean_reward(&baseline_trajectories);
    println!("baseline reward after sft: {baseline_reward:.4}");

    let grpo_cfg = GrpoConfig {
        clip_eps: 0.2,
        kl_coef: args.kl_coef,
        group_size: args.group_size,
    };
    // Hand AdamW's final SFT-phase state (moments + step counter) across
    // the SFT→GRPO boundary so bias correction keeps counting and the
    // second-moment buffers are warm-started, matching the pre-migration
    // single-instance behaviour. Codex review on 09c5c89 correctly rebutted
    // the earlier "no API" claim — `Trainer::optim()` + `Optimizer::
    // export_state/import_state` were already sufficient, no Trainer
    // surface changes needed.
    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let param_names: Vec<(TensorId, String)> = params
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, format!("param.{i:04}")))
        .collect();
    optimizer
        .import_state(&sft_optim_state, &param_names)
        .map_err(|e| CliError::Custom(format!("adamw sft→grpo handoff: {e}")))?;
    let mut reward_trajectory = Vec::with_capacity(args.grpo_iters);
    let mut last_kl = 0.0_f32;
    let mut best_mean_reward = baseline_reward;

    for iter in 0..args.grpo_iters {
        let prompts = build_prompt_batch(args.batch_prompts, args.seq, 64, 255, &mut prompt_rng);
        let trajectories = rollout_group(
            &policy,
            &ref_model,
            &config,
            &prompts,
            args.group_size,
            args.temperature,
            &mut prompt_rng,
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
fn run_sft_phase(
    args: &CliArgs,
    policy: &Qwen3Model,
    params: &[TensorId],
    keep_extra: HashSet<TensorId>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<AdamWState, CliError> {
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

    // Qwen3 exposes named parameters; synthesize stable
    // names purely to satisfy the Trainer API (no optimizer-state
    // persistence is wired — save_dir is None).
    let param_names: Vec<(TensorId, String)> = params
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, format!("param.{i:04}")))
        .collect();

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

fn parse_args() -> Result<CliArgs, CliError> {
    let mut args = CliArgs::default();
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--sft-steps" => args.sft_steps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--grpo-iters" => {
                args.grpo_iters = parse_value(&flag, next_value(&mut iter, &flag)?)?;
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
            "--grad-clip" => {
                args.grad_clip = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--no-grad-clip" => args.grad_clip = None,
            "--metrics-jsonl" => {
                args.metrics_jsonl = Some(PathBuf::from(next_value(&mut iter, &flag)?));
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

fn qwen3_config(seq: usize) -> Qwen3Config {
    Qwen3Config {
        vocab_size: 256,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        intermediate_size: 128,
        max_position_embeddings: seq,
        rms_norm_eps: 1.0e-6,
        rope_theta: 1_000_000.0,
        tie_word_embeddings: false,
    }
}

fn retained_ids(models: &[&Qwen3Model], store: &TensorStore) -> HashSet<TensorId> {
    let mut keep = HashSet::new();
    for model in models {
        for param_id in model.all_parameter_ids() {
            keep.insert(param_id);
            if let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) {
                keep.insert(grad_id);
            }
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

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

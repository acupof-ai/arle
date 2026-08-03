//! Data-driven OPD policy-update seam. An algorithm is a [`UpdatePreset`] —
//! six orthogonal fields over ONE update path — not a new enum arm; the named
//! constructors below are the algorithm table. The rollout/scoring harness is
//! preset-agnostic: it only reads [`UpdatePreset::needs`].

use autograd::ops::fused_linear_distill::{PgStats, WeightForm, pg_token_weight};
use autograd::{TensorId, TensorStore, optim::Optimizer};

use crate::grad_clip::finite_optimizer_step;
use crate::opd::{
    OpdError, Result, ValueCritic, WritebackLoss, capture_rollout_logprobs,
    masked_writeback_ce_step_dispatch, masked_writeback_step,
};
use crate::qwen35::Qwen35Model;

/// What the harness must collect for a preset: whether to keep failing
/// trajectories, and whether generation must return behavior-policy logprobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolloutNeeds {
    pub keep_failing: bool,
    pub behavior_logprobs: bool,
}

/// One scored rollout: the verl-style `(prompt, response, mask)` record plus its
/// scalar reward, group key, and generation-time behavior-policy logprobs.
#[derive(Clone, Debug)]
pub struct ScoredTrajectory {
    pub prompt_ids: Vec<u32>,
    pub response_ids: Vec<u32>,
    /// Skip-Observation mask: `1` = LLM token, `0` = tool/environment token.
    pub response_mask: Vec<u8>,
    /// Graded pytest reward in [0,1]; pass ⇔ `reward >= 1.0`.
    pub reward: f32,
    /// Generation-time π_behavior logprob per masked target position.
    /// Required iff `needs.behavior_logprobs`.
    pub behavior_logprobs: Option<Vec<f32>>,
    /// Per-prompt group key (task index) for [`Scope::Group`] baselines.
    pub group_id: usize,
    /// Hit the turn/token budget (input to the DAPO overlong filter).
    pub truncated: bool,
}

/// Which scored trajectories enter the update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFilter {
    /// Full passes only (`reward >= 1.0` — all fail_to_pass green).
    PassOnly,
    KeepAll,
    /// DAPO dynamic sampling: drop zero-reward-variance groups AND truncated
    /// trajectories (the paper's overlong filter travels with it).
    DropZeroAdvGroup,
    /// Drop budget-truncated trajectories only.
    DropTruncated,
}

/// Advantage baseline scope for [`Advantage::Mean`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Batch,
    /// Per-prompt group ([`ScoredTrajectory::group_id`]).
    Group,
}

/// Advantage estimator (per-trajectory scalar unless GAE).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Advantage {
    None,
    /// `A_i = reward_i − mean(scope rewards)`, `/ (std + 1e-6)` if `std_norm`.
    Mean {
        scope: Scope,
        std_norm: bool,
    },
    /// Learned value critic + Skip-Obs GAE (per-token credit).
    ValueGae {
        gamma: f32,
        lam: f32,
    },
}

/// Importance-ratio granularity vs the behavior policy π_b.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RatioGrain {
    None,
    PerToken,
    /// Length-normalized sequence ratio (GSPO): `exp(mean_t(logπθ − logπ_b))`,
    /// clipped at the sequence level and broadcast over tokens.
    PerSequence,
}

/// Ratio objective applied to a detached advantage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClipForm {
    /// Binary keep/drop gate: `1` iff `(1−lo) < r < (1+hi)` (SAO DIS).
    HardGate { lo: f32, hi: f32 },
    /// Detached `clamp(r, 1−lo, 1+hi)` coefficient (CISPO only).
    DetachedSoftClamp { lo: f32, hi: f32 },
    /// Sign-aware PPO clipped surrogate (GRPO/DAPO/Dr.GRPO/GSPO).
    PpoClip { lo: f32, hi: f32 },
}

impl ClipForm {
    fn weight_form(self) -> WeightForm {
        match self {
            Self::HardGate { lo, hi } => WeightForm::HardGate { lo, hi },
            Self::DetachedSoftClamp { lo, hi } => WeightForm::DetachedSoftClamp { lo, hi },
            Self::PpoClip { lo, hi } => WeightForm::PpoClip { lo, hi },
        }
    }
}

/// Loss normalization + optimizer-step grain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aggregation {
    /// Token-mean per trajectory, one optimizer step per trajectory.
    PerSeqTokenMean,
    /// Token-mean over the whole batch (÷ `norm_const` if set, else the batch's
    /// masked-token count); grads accumulate, ONE step per batch.
    GlobalTokenMean { norm_const: Option<usize> },
}

/// One `update` call's roll-up: the mean per-trajectory loss plus the
/// off-policy / critic / advantage diagnostics the metrics sink records.
#[derive(Clone, Copy, Debug, Default)]
pub struct UpdateReport {
    pub loss: f32,
    /// Trajectories that contributed a gradient.
    pub trained: usize,
    /// Masked tokens trained across them (0 on the CE path's own count below).
    pub tokens: usize,
    pub stats: PgStats,
    /// Mean critic MSE (0.0 without a critic).
    pub critic_mse: f32,
    pub adv_mean: f32,
    pub adv_std: f32,
}

/// One policy-update algorithm as data. Extend = a new constructor value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpdatePreset {
    pub filter: SampleFilter,
    pub advantage: Advantage,
    pub ratio: RatioGrain,
    pub clip: ClipForm,
    pub agg: Aggregation,
}

impl UpdatePreset {
    /// Shipped default: reject failures, masked CE on full passes.
    pub fn rejection_ce() -> Self {
        Self {
            filter: SampleFilter::PassOnly,
            advantage: Advantage::None,
            ratio: RatioGrain::None,
            // Unused: ratio None routes to the fused CE op.
            clip: ClipForm::HardGate { lo: 0.0, hi: 0.0 },
            agg: Aggregation::PerSeqTokenMean,
        }
    }

    /// SAO Phase 1: batch-centered advantage, hard-gated per-token IS ratio.
    pub fn sao_dis(lo: f32, hi: f32) -> Self {
        Self {
            filter: SampleFilter::KeepAll,
            advantage: Advantage::Mean {
                scope: Scope::Batch,
                std_norm: false,
            },
            ratio: RatioGrain::PerToken,
            clip: ClipForm::HardGate { lo, hi },
            agg: Aggregation::PerSeqTokenMean,
        }
    }

    /// SAO Phase 2: Skip-Obs GAE advantages from a learned value critic.
    pub fn sao_value(lo: f32, hi: f32, gamma: f32, lam: f32) -> Self {
        Self {
            advantage: Advantage::ValueGae { gamma, lam },
            ..Self::sao_dis(lo, hi)
        }
    }

    /// GRPO (Shao et al. 2024): group-normalized advantage, token-level
    /// sign-aware PPO clipped surrogate.
    pub fn grpo() -> Self {
        Self {
            filter: SampleFilter::KeepAll,
            advantage: Advantage::Mean {
                scope: Scope::Group,
                std_norm: true,
            },
            ratio: RatioGrain::PerToken,
            clip: ClipForm::PpoClip { lo: 0.2, hi: 0.2 },
            agg: Aggregation::PerSeqTokenMean,
        }
    }

    /// DAPO (Yu et al. 2025): clip-higher 0.2/0.28, dynamic sampling + overlong
    /// filter, no std norm, token-level (batch token-mean) loss, no KL.
    pub fn dapo() -> Self {
        Self {
            filter: SampleFilter::DropZeroAdvGroup,
            advantage: Advantage::Mean {
                scope: Scope::Group,
                std_norm: false,
            },
            clip: ClipForm::PpoClip { lo: 0.2, hi: 0.28 },
            agg: Aggregation::GlobalTokenMean { norm_const: None },
            ..Self::grpo()
        }
    }

    /// Dr.GRPO (Liu et al. 2025): GRPO minus its length/std biases — no std
    /// norm, fixed-constant normalizer (the generation budget).
    pub fn dr_grpo(norm_const: usize) -> Self {
        Self {
            advantage: Advantage::Mean {
                scope: Scope::Group,
                std_norm: false,
            },
            agg: Aggregation::GlobalTokenMean {
                norm_const: Some(norm_const),
            },
            ..Self::grpo()
        }
    }

    /// GSPO (Zheng et al. 2025): sequence-level ratio; paper clip 3e-4/4e-4.
    pub fn gspo() -> Self {
        Self {
            ratio: RatioGrain::PerSequence,
            clip: ClipForm::PpoClip { lo: 3e-4, hi: 4e-4 },
            ..Self::grpo()
        }
    }

    /// CISPO (MiniMax-M1 2025): detached clamped-IS weight, one-sided (lo=1 →
    /// floor 0, no lower clip); the upper bound mirrors the licensed SAO-DIS
    /// bound (the paper leaves ε_high^IS workload-tuned). Batch token-mean.
    pub fn cispo() -> Self {
        Self {
            clip: ClipForm::DetachedSoftClamp { lo: 1.0, hi: 3.0 },
            agg: Aggregation::GlobalTokenMean { norm_const: None },
            ..Self::grpo()
        }
    }

    /// Derived mechanically from the preset fields.
    pub fn needs(&self) -> RolloutNeeds {
        RolloutNeeds {
            keep_failing: self.filter != SampleFilter::PassOnly,
            behavior_logprobs: self.ratio != RatioGrain::None,
        }
    }

    /// True when the caller must build (and pass) a [`ValueCritic`].
    pub fn needs_value_critic(&self) -> bool {
        matches!(self.advantage, Advantage::ValueGae { .. })
    }

    fn filter_batch<'a>(&self, batch: &'a [ScoredTrajectory]) -> Vec<&'a ScoredTrajectory> {
        match self.filter {
            SampleFilter::PassOnly => batch.iter().filter(|t| t.reward >= 1.0).collect(),
            SampleFilter::KeepAll => batch.iter().collect(),
            SampleFilter::DropTruncated => batch.iter().filter(|t| !t.truncated).collect(),
            SampleFilter::DropZeroAdvGroup => {
                // Variance judged on the group as scored (before truncation drop).
                let live = |t: &ScoredTrajectory| {
                    batch
                        .iter()
                        .any(|o| o.group_id == t.group_id && o.reward != t.reward)
                };
                batch.iter().filter(|t| !t.truncated && live(t)).collect()
            }
        }
    }

    fn validated_training<'a>(
        &self,
        batch: &'a [ScoredTrajectory],
        vocab: usize,
        window: usize,
    ) -> Result<Vec<&'a ScoredTrajectory>> {
        validate_trajectory_structure(batch)?;
        if window == 0 {
            return Err(OpdError::InvalidInput(
                "masked writeback window_size must be > 0".to_owned(),
            ));
        }
        if vocab > i32::MAX as usize {
            return Err(OpdError::InvalidInput(format!(
                "masked writeback vocab {vocab} exceeds i32::MAX"
            )));
        }
        let training: Vec<_> = self
            .filter_batch(batch)
            .into_iter()
            .filter(|traj| !over_max_seq(traj) && masked_target_count(traj) > 0)
            .collect();
        validate_training_inputs(&training, vocab, self.needs().behavior_logprobs)?;
        Ok(training)
    }

    /// Number of trajectories that pass the preset and deterministic skip gates.
    pub fn planned_training_count(&self, batch: &[ScoredTrajectory]) -> usize {
        self.filter_batch(batch)
            .into_iter()
            .filter(|traj| !over_max_seq(traj) && masked_target_count(traj) > 0)
            .count()
    }

    /// Validate every deterministic input before model, critic, or optimizer work.
    pub fn preflight(&self, batch: &[ScoredTrajectory], vocab: usize, window: usize) -> Result<()> {
        self.validated_training(batch, vocab, window).map(drop)
    }

    /// Apply one round's update over `batch`, returning the mean per-trajectory
    /// loss (same reduction as the pre-preset per-trajectory writeback loop)
    /// plus the diagnostics the metrics sink records.
    #[allow(clippy::too_many_arguments)]
    pub fn update<O: Optimizer>(
        &self,
        batch: &[ScoredTrajectory],
        student: &Qwen35Model,
        all_params: &[TensorId],
        trainable: &[TensorId],
        opt: &mut O,
        critic: Option<&mut ValueCritic>,
        vocab: usize,
        window: usize,
        store: &mut TensorStore,
    ) -> Result<UpdateReport> {
        let survivors = self.validated_training(batch, vocab, window)?;
        if survivors.is_empty() {
            return Ok(UpdateReport::default());
        }
        if self.ratio == RatioGrain::None {
            return update_ce(
                &survivors, student, all_params, trainable, opt, vocab, window, store,
            );
        }
        self.update_pg(
            &survivors, student, all_params, trainable, opt, critic, vocab, window, store,
        )
    }

    /// Weighted policy-gradient path: advantage × aggregation norm is passed to
    /// the fused PG ABI; token PPO/CISPO/SAO semantics are selected by `clip`,
    /// while GSPO precomputes the sequence-level sign-aware coefficient.
    #[allow(clippy::too_many_arguments)]
    fn update_pg<O: Optimizer>(
        &self,
        survivors: &[&ScoredTrajectory],
        student: &Qwen35Model,
        all_params: &[TensorId],
        trainable: &[TensorId],
        opt: &mut O,
        mut critic: Option<&mut ValueCritic>,
        vocab: usize,
        window: usize,
        store: &mut TensorStore,
    ) -> Result<UpdateReport> {
        let scalar_advs: Vec<f32> = match self.advantage {
            Advantage::Mean { scope, std_norm } => centered_advantages(survivors, scope, std_norm),
            // GAE supplies per-token credit below; None = uniform PG weight.
            Advantage::ValueGae { .. } => Vec::new(),
            Advantage::None => vec![1.0; survivors.len()],
        };
        if self.needs_value_critic() && critic.is_none() {
            return Err(OpdError::InvalidInput(
                "ValueGae update requires a ValueCritic (caller must pass Some)".to_owned(),
            ));
        }
        // The policy writeback's `cleanup_after_backward` frees every live tensor
        // not in `all_model_params`; the critic weight isn't a student param, so
        // it must ride in this keep-set (kept, NOT stepped — `trainable` is still
        // LoRA-only) or `critic.update` below hits a freed weight.
        let all_with_critic: Vec<TensorId> = match &critic {
            Some(c) => all_params
                .iter()
                .copied()
                .chain(c.param_ids().iter().copied())
                .collect(),
            None => Vec::new(),
        };
        let params = if critic.is_some() {
            all_with_critic.as_slice()
        } else {
            all_params
        };

        // GlobalTokenMean counts only trajectories that passed preflight.
        let batch_tokens: usize = survivors.iter().map(|t| masked_target_count(t)).sum();
        let step_each = matches!(self.agg, Aggregation::PerSeqTokenMean);

        let mut loss_sum = 0.0f32;
        let mut steps = 0usize;
        let mut stats = PgStats::default();
        // Critic health telemetry: is V(s) learning (MSE ↓) and producing
        // non-trivial credit (|adv| > 0)? A null result is unattributable without it.
        let mut mse_sum = 0.0f32;
        let mut adv_abs_sum = 0.0f32;
        let mut adv_tokens = 0usize;
        // Advantage distribution over the values actually applied (per-token
        // under GAE, per-trajectory scalars otherwise).
        let mut adv_sum = 0.0f64;
        let mut adv_sq = 0.0f64;
        let mut adv_n = 0usize;
        // GlobalTokenMean accumulates gradients across trajectories into a single
        // step; any mid-loop error would otherwise leave those grads pending and
        // leak into the next batch. Run the accumulation as one fallible unit and
        // clear pending grads on any failure so a batch is all-or-nothing.
        let accumulate = (|| -> std::result::Result<(), OpdError> {
            for (i, traj) in survivors.iter().enumerate() {
                let behavior_logprobs = traj
                    .behavior_logprobs
                    .as_deref()
                    .expect("ratio-weighted batch validated before policy/critic forward");
                let (advantages, returns) = match critic.as_deref_mut() {
                    Some(c) => c.advantages(
                        student,
                        &traj.prompt_ids,
                        &traj.response_ids,
                        &traj.response_mask,
                        traj.reward,
                        store,
                    )?,
                    None => (vec![scalar_advs[i]; behavior_logprobs.len()], Vec::new()),
                };
                if advantages.is_empty() {
                    continue; // no LLM tokens to train
                }
                // No signal → no update: stepping on zero advantage lets AdamW weight
                // decay / momentum drift the weights on zero gradient.
                if critic.is_none() && scalar_advs[i] == 0.0 {
                    continue;
                }
                for &a in &advantages {
                    adv_sum += f64::from(a);
                    adv_sq += f64::from(a) * f64::from(a);
                }
                adv_n += advantages.len();

                // Token-mean the PG objective, mirroring CE's 1/N: per-trajectory
                // masked count, or the batch/fixed constant for GlobalTokenMean.
                let norm = token_mean_norm(self.agg, advantages.len(), batch_tokens);
                let mut weights: Vec<f32> = advantages.iter().map(|a| a / norm).collect();

                let form = match self.ratio {
                    RatioGrain::PerToken => self.clip.weight_form(),
                    RatioGrain::PerSequence => {
                        // GSPO: seq ratio at CURRENT θ (one tape-off capture pass),
                        // clipped at the sequence level, broadcast into the weights.
                        let current_lp = capture_rollout_logprobs(
                            student,
                            &traj.prompt_ids,
                            &traj.response_ids,
                            &traj.response_mask,
                            store,
                        )?;
                        let s = sequence_ratio(&current_lp, behavior_logprobs)?;
                        let mut clipped = false;
                        for w in &mut weights {
                            let (effective, token_clipped) =
                                pg_token_weight(self.clip.weight_form(), *w, s, 0.0);
                            *w = effective;
                            clipped |= token_clipped;
                        }
                        WeightForm::Precomputed { ratio: s, clipped }
                    }
                    RatioGrain::None => unreachable!("CE path handled in update()"),
                };

                let (loss, traj_stats, _) = masked_writeback_step(
                    WritebackLoss::Pg {
                        rollout_logprobs: behavior_logprobs,
                        weight: &weights,
                        form,
                        kl_coef: 0.0,
                    },
                    student,
                    params,
                    trainable,
                    opt,
                    step_each,
                    &traj.prompt_ids,
                    &traj.response_ids,
                    &traj.response_mask,
                    vocab,
                    window,
                    crate::context_parallel::CpContext::from_env(),
                    crate::context_parallel::DpContext::from_env(),
                    store,
                )?;
                stats.merge(traj_stats);
                if let Some(c) = critic.as_deref_mut() {
                    // Fit the critic toward the observed returns (frozen-attention MSE).
                    mse_sum += c.update(
                        student,
                        &traj.prompt_ids,
                        &traj.response_ids,
                        &traj.response_mask,
                        &returns,
                        store,
                    )?;
                    adv_abs_sum += advantages.iter().map(|a| a.abs()).sum::<f32>();
                    adv_tokens += advantages.len();
                }
                loss_sum += loss;
                if !loss_sum.is_finite() {
                    return Err(OpdError::InvalidInput(format!(
                        "policy loss became non-finite while accumulating trajectory {i}: {loss_sum}"
                    )));
                }
                steps += 1;
            }
            Ok(())
        })();
        if let Err(e) = accumulate {
            // Any accumulation failure aborts the whole batch: drop pending grads
            // so nothing leaks into the next GlobalTokenMean step.
            opt.zero_grad(store, trainable);
            return Err(e);
        }

        if !step_each && steps > 0 {
            finite_optimizer_step(loss_sum, trainable, 0.0, opt, store)?;
        }
        let critic_mse = if critic.is_some() && steps > 0 {
            mse_sum / steps as f32
        } else {
            0.0
        };
        if steps > 0 {
            let critic_suffix = if critic.is_some() {
                format!(
                    " mean_critic_mse={critic_mse:.4e} mean_adv_abs={:.4e}",
                    adv_abs_sum / adv_tokens.max(1) as f32,
                )
            } else {
                String::new()
            };
            eprintln!(
                "[update] trained={steps} mean_policy_loss={:.4} kl={:.4e} clip_frac={:.3}{critic_suffix}",
                loss_sum / steps as f32,
                stats.kl_mean(),
                stats.clip_frac(),
            );
        }
        let adv_mean = adv_sum / adv_n.max(1) as f64;
        let adv_std = (adv_sq / adv_n.max(1) as f64 - adv_mean * adv_mean).max(0.0);
        Ok(UpdateReport {
            loss: if steps > 0 {
                loss_sum / steps as f32
            } else {
                0.0
            },
            trained: steps,
            tokens: stats.tokens,
            stats,
            critic_mse,
            adv_mean: adv_mean as f32,
            adv_std: adv_std.sqrt() as f32,
        })
    }
}

fn validate_trajectory_structure(batch: &[ScoredTrajectory]) -> Result<()> {
    for (idx, traj) in batch.iter().enumerate() {
        if !traj.reward.is_finite() {
            return Err(OpdError::InvalidInput(format!(
                "trajectory {idx}: reward must be finite, got {}",
                traj.reward
            )));
        }
        if traj.response_ids.len() != traj.response_mask.len() {
            return Err(OpdError::InvalidInput(format!(
                "trajectory {idx}: response_ids len {} != response_mask len {}",
                traj.response_ids.len(),
                traj.response_mask.len()
            )));
        }
        if let Some((position, mask)) = traj
            .response_mask
            .iter()
            .copied()
            .enumerate()
            .find(|(_, mask)| *mask > 1)
        {
            return Err(OpdError::InvalidInput(format!(
                "trajectory {idx}: response_mask[{position}]={mask}, expected 0 or 1"
            )));
        }
    }
    Ok(())
}

fn masked_target_count(traj: &ScoredTrajectory) -> usize {
    traj.response_mask.iter().filter(|&&mask| mask == 1).count()
}

fn validate_training_inputs(
    training: &[&ScoredTrajectory],
    vocab: usize,
    needs_behavior: bool,
) -> Result<()> {
    for (idx, traj) in training.iter().enumerate() {
        if traj.prompt_ids.is_empty() {
            return Err(OpdError::InvalidInput(format!(
                "training trajectory {idx}: masked writeback requires a non-empty prompt"
            )));
        }
        let seq_len = traj
            .prompt_ids
            .len()
            .checked_add(traj.response_ids.len())
            .ok_or_else(|| {
                OpdError::InvalidInput(format!(
                    "training trajectory {idx}: prompt + response length overflow"
                ))
            })?;
        if seq_len > i32::MAX as usize {
            return Err(OpdError::InvalidInput(format!(
                "training trajectory {idx}: length {seq_len} exceeds i32::MAX position indices"
            )));
        }
        for (position, (&token, &mask)) in traj
            .response_ids
            .iter()
            .zip(&traj.response_mask)
            .enumerate()
        {
            if mask == 1 && token as usize >= vocab {
                return Err(OpdError::InvalidInput(format!(
                    "training trajectory {idx}: masked target token {token} at response index {position} exceeds vocab={vocab}"
                )));
            }
        }
        if needs_behavior {
            let behavior = traj.behavior_logprobs.as_deref().ok_or_else(|| {
                OpdError::InvalidInput(format!(
                    "training trajectory {idx}: ratio-weighted update requires generation-time behavior_logprobs sidecar"
                ))
            })?;
            let targets = masked_target_count(traj);
            if behavior.len() != targets {
                return Err(OpdError::InvalidInput(format!(
                    "training trajectory {idx}: behavior_logprobs len {} != masked token count {targets}",
                    behavior.len()
                )));
            }
            if let Some(position) = behavior.iter().position(|value| !value.is_finite()) {
                return Err(OpdError::InvalidInput(format!(
                    "training trajectory {idx}: behavior_logprobs[{position}] is not finite ({})",
                    behavior[position]
                )));
            }
        }
    }
    Ok(())
}

/// Reward-centered advantage per trajectory: `A_i = r_i − mean(scope)`,
/// `/ (std + 1e-6)` when `std_norm` (the GRPO convention).
fn centered_advantages(survivors: &[&ScoredTrajectory], scope: Scope, std_norm: bool) -> Vec<f32> {
    let group_of = |t: &ScoredTrajectory| match scope {
        Scope::Batch => 0usize,
        Scope::Group => t.group_id,
    };
    // (key, sum, sumsq, n) — small n, first-seen order, linear find.
    let mut stats: Vec<(usize, f32, f32, usize)> = Vec::new();
    for t in survivors {
        let key = group_of(t);
        match stats.iter_mut().find(|(k, ..)| *k == key) {
            Some((_, sum, sumsq, n)) => {
                *sum += t.reward;
                *sumsq += t.reward * t.reward;
                *n += 1;
            }
            None => stats.push((key, t.reward, t.reward * t.reward, 1)),
        }
    }
    survivors
        .iter()
        .map(|t| {
            let &(_, sum, sumsq, n) = stats
                .iter()
                .find(|(k, ..)| *k == group_of(t))
                .expect("group key inserted above");
            let mean = sum / n as f32;
            let adv = t.reward - mean;
            if std_norm {
                let var = (sumsq / n as f32 - mean * mean).max(0.0);
                adv / (var.sqrt() + 1e-6)
            } else {
                adv
            }
        })
        .collect()
}

/// PG token-mean denominator. GlobalTokenMean with a fixed `norm_const` is
/// Dr.GRPO's length-debiasing: a per-token divide by the constant generation
/// budget, independent of each trajectory's actual length (the length bias GRPO
/// carries via per-sequence `advantages.len()`). Group averaging is separate —
/// it lives in `centered_advantages`.
fn token_mean_norm(agg: Aggregation, seq_tokens: usize, batch_tokens: usize) -> f32 {
    match agg {
        Aggregation::PerSeqTokenMean => seq_tokens,
        Aggregation::GlobalTokenMean { norm_const } => norm_const.unwrap_or(batch_tokens),
    }
    .max(1) as f32
}

/// GSPO length-normalized sequence ratio `exp(mean_t(logπθ − logπ_b))`.
fn sequence_ratio(current_lp: &[f32], behavior_lp: &[f32]) -> Result<f32> {
    if current_lp.len() != behavior_lp.len() {
        return Err(OpdError::InvalidInput(format!(
            "seq-ratio capture len {} != behavior logprobs len {}",
            current_lp.len(),
            behavior_lp.len()
        )));
    }
    if current_lp.is_empty() {
        return Err(OpdError::InvalidInput(
            "seq-ratio capture must be non-empty".to_owned(),
        ));
    }
    let mean_delta = current_lp
        .iter()
        .zip(behavior_lp)
        .map(|(&current, &behavior)| f64::from(current) - f64::from(behavior))
        .sum::<f64>()
        / current_lp.len() as f64;
    let ratio = mean_delta.exp();
    if !ratio.is_finite() || ratio <= 0.0 || ratio > f64::from(f32::MAX) {
        return Err(OpdError::InvalidInput(
            "sequence importance ratio must be finite, positive, and fit f32".to_owned(),
        ));
    }
    Ok(ratio as f32)
}

/// The VRAM-wall gate as a pure predicate (no warn), so the update loop and the
/// GlobalTokenMean denominator agree on which trajectories train.
fn over_max_seq(traj: &ScoredTrajectory) -> bool {
    let cap = crate::runtime_flags::max_update_seq();
    cap != 0 && traj.prompt_ids.len() + traj.response_ids.len() > cap
}

/// Masked-CE loop over the surviving trajectories (`ratio == None`), mean loss.
#[allow(clippy::too_many_arguments)]
fn update_ce<O: Optimizer>(
    survivors: &[&ScoredTrajectory],
    student: &Qwen35Model,
    all_params: &[TensorId],
    trainable: &[TensorId],
    opt: &mut O,
    vocab: usize,
    window: usize,
    store: &mut TensorStore,
) -> Result<UpdateReport> {
    let mut loss_sum = 0.0f32;
    let mut tokens = 0usize;
    let mut trained = 0usize;
    for traj in survivors {
        // Dispatch (not `masked_writeback_step(Ce)` directly) so the default
        // path stays byte-identical, honoring `--writeback-frozen-prompt-kv`.
        loss_sum += masked_writeback_ce_step_dispatch(
            student,
            all_params,
            trainable,
            opt,
            &traj.prompt_ids,
            &traj.response_ids,
            &traj.response_mask,
            vocab,
            window,
            crate::context_parallel::CpContext::from_env(),
            crate::context_parallel::DpContext::from_env(),
            store,
        )?;
        tokens += traj.response_mask.iter().filter(|&&m| m == 1).count();
        trained += 1;
    }
    // Average over trained trajectories only (skips add no loss), mirroring
    // `update_pg`'s `steps` convention.
    Ok(UpdateReport {
        loss: if trained > 0 {
            loss_sum / trained as f32
        } else {
            0.0
        },
        trained,
        tokens,
        ..UpdateReport::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traj(reward: f32, group_id: usize) -> ScoredTrajectory {
        ScoredTrajectory {
            prompt_ids: vec![1],
            response_ids: vec![2],
            response_mask: vec![1],
            reward,
            behavior_logprobs: None,
            group_id,
            truncated: false,
        }
    }

    #[test]
    fn preset_needs_derive_mechanically() {
        let cases = [
            (UpdatePreset::rejection_ce(), false, false),
            (UpdatePreset::sao_dis(0.8, 3.0), true, true),
            (UpdatePreset::sao_value(0.8, 3.0, 1.0, 0.95), true, true),
            (UpdatePreset::grpo(), true, true),
            (UpdatePreset::dapo(), true, true),
            (UpdatePreset::dr_grpo(4096), true, true),
            (UpdatePreset::gspo(), true, true),
            (UpdatePreset::cispo(), true, true),
        ];
        for (preset, keep_failing, behavior_logprobs) in cases {
            assert_eq!(
                preset.needs(),
                RolloutNeeds {
                    keep_failing,
                    behavior_logprobs
                },
                "{preset:?}"
            );
        }
        assert!(UpdatePreset::sao_value(0.8, 3.0, 1.0, 0.95).needs_value_critic());
        assert!(!UpdatePreset::grpo().needs_value_critic());
    }

    #[test]
    fn group_vs_batch_advantage_scope() {
        // Group 0 rewards (1, 0); group 1 zero-variance (1, 1).
        let batch = [traj(1.0, 0), traj(0.0, 0), traj(1.0, 1), traj(1.0, 1)];
        let refs: Vec<&ScoredTrajectory> = batch.iter().collect();
        // Batch scope: mean 0.75.
        assert_eq!(
            centered_advantages(&refs, Scope::Batch, false),
            vec![0.25, -0.75, 0.25, 0.25]
        );
        // Group scope: group 0 mean 0.5 → ±0.5; group 1 zero-variance → 0.
        assert_eq!(
            centered_advantages(&refs, Scope::Group, false),
            vec![0.5, -0.5, 0.0, 0.0]
        );
        // std_norm (GRPO): group 0 std 0.5 → ±(0.5 / (0.5 + 1e-6)).
        let gn = centered_advantages(&refs, Scope::Group, true);
        assert!((gn[0] - 0.5 / (0.5 + 1e-6)).abs() < 1e-6, "{gn:?}");
        assert_eq!(gn[2], 0.0);
    }

    #[test]
    fn dr_grpo_norm_is_fixed_constant_independent_of_length() {
        // Dr.GRPO: fixed generation budget divides every token, regardless of a
        // trajectory's own length — a short and a long trajectory share the
        // denominator, so long ones are not down-weighted (GRPO's length bias).
        let dr = Aggregation::GlobalTokenMean {
            norm_const: Some(4096),
        };
        assert_eq!(token_mean_norm(dr, 10, 999), 4096.0);
        assert_eq!(token_mean_norm(dr, 4000, 999), 4096.0);
        // GRPO (no fixed const) falls back to the batch's actual token count.
        let grpo = Aggregation::GlobalTokenMean { norm_const: None };
        assert_eq!(token_mean_norm(grpo, 10, 512), 512.0);
        // PerSeqTokenMean divides by the trajectory's own length (the bias).
        assert_eq!(token_mean_norm(Aggregation::PerSeqTokenMean, 10, 512), 10.0);
        // Empty guard: never divide by zero.
        assert_eq!(token_mean_norm(Aggregation::PerSeqTokenMean, 0, 0), 1.0);
    }

    #[test]
    fn weight_builder_distinguishes_hard_cispo_and_ppo() {
        let hard = WeightForm::HardGate { lo: 0.2, hi: 0.5 };
        let cispo = WeightForm::DetachedSoftClamp { lo: 0.2, hi: 0.5 };
        let ppo = WeightForm::PpoClip { lo: 0.2, hi: 0.5 };
        assert_eq!(pg_token_weight(hard, 2.0, 1.0, 0.0), (2.0, false));
        assert_eq!(pg_token_weight(hard, 2.0, 1.6, 0.0), (0.0, true));

        // CISPO remains a detached clamped coefficient on both signs.
        assert_eq!(pg_token_weight(cispo, 2.0, 1.2, 0.0), (2.4, false));
        assert_eq!(pg_token_weight(cispo, 2.0, 3.0, 0.0), (3.0, true));
        assert_eq!(pg_token_weight(cispo, -2.0, 0.5, 0.0), (-1.6, true));

        // PPO's clipped branch depends on the advantage sign.
        let cases = [
            (2.0, 0.5, 1.0, false),
            (2.0, 1.0, 2.0, false),
            (2.0, 1.6, 0.0, true),
            (-2.0, 0.5, 0.0, true),
            (-2.0, 1.0, -2.0, false),
            (-2.0, 1.6, -3.2, false),
        ];
        for (base, ratio, expected, clipped) in cases {
            assert_eq!(
                pg_token_weight(ppo, base, ratio, 0.0),
                (expected, clipped),
                "base={base} ratio={ratio}"
            );
        }

        let (w, clipped) = pg_token_weight(
            WeightForm::Precomputed {
                ratio: 2.0,
                clipped: false,
            },
            1.0,
            2.0,
            0.1,
        );
        assert!((w - 0.9).abs() < 1e-6);
        assert!(!clipped);

        let s = sequence_ratio(&[0.0, 0.0], &[-0.2, 0.2]).unwrap();
        assert!((s - 1.0).abs() < 1e-6, "mean Δ=0 → s=1, got {s}");
        let s = sequence_ratio(&[0.5, 0.5], &[0.0, 0.0]).unwrap();
        assert!((s - 0.5f32.exp()).abs() < 1e-6, "got {s}");
        assert_eq!(pg_token_weight(ppo, 1.0, 1.6, 0.0), (0.0, true));
        assert_eq!(pg_token_weight(ppo, -1.0, 1.6, 0.0), (-1.6, false));
    }

    #[test]
    fn sequence_ratio_uses_f64_and_fails_closed() {
        let ratio = sequence_ratio(&[1.0e20, -1.0e20], &[0.0, 0.0]).unwrap();
        assert_eq!(ratio, 1.0);

        let overflow = sequence_ratio(&[1000.0], &[0.0]).unwrap_err();
        assert!(overflow.to_string().contains("sequence importance ratio"));
        let empty = sequence_ratio(&[], &[]).unwrap_err();
        assert!(empty.to_string().contains("must be non-empty"));
    }

    #[test]
    fn behavior_sidecars_fail_closed() {
        let mut batch = vec![traj(1.0, 0)];
        let err = UpdatePreset::grpo().preflight(&batch, 100, 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("requires generation-time behavior_logprobs")
        );

        batch[0].behavior_logprobs = Some(vec![f32::NAN]);
        let err = UpdatePreset::grpo().preflight(&batch, 100, 1).unwrap_err();
        assert!(err.to_string().contains("is not finite"));

        batch[0].behavior_logprobs = Some(vec![]);
        let err = UpdatePreset::grpo().preflight(&batch, 100, 1).unwrap_err();
        assert!(err.to_string().contains("masked token count 1"));
    }

    #[test]
    fn trajectory_structure_accepts_mixed_mask_and_rejects_mismatch() {
        let mut batch = vec![traj(1.0, 0)];
        batch[0].response_ids = vec![2, 3];
        batch[0].response_mask = vec![1, 0];
        assert!(validate_trajectory_structure(&batch).is_ok());

        batch[0].response_mask.pop();
        let err = validate_trajectory_structure(&batch).unwrap_err();
        assert!(
            err.to_string()
                .contains("response_ids len 2 != response_mask len 1")
        );

        batch[0].response_mask = vec![1, 2];
        let err = validate_trajectory_structure(&batch).unwrap_err();
        assert!(err.to_string().contains("expected 0 or 1"));
    }

    #[test]
    fn dapo_filtered_malformed_sidecars_do_not_block() {
        let mut batch = vec![traj(1.0, 0), traj(0.0, 0), traj(1.0, 1), traj(1.0, 1)];
        batch[0].truncated = true;
        batch[1].behavior_logprobs = Some(vec![-0.5]);
        batch[2].behavior_logprobs = Some(vec![f32::NAN]);
        UpdatePreset::dapo().preflight(&batch, 100, 1).unwrap();
    }

    #[test]
    fn preflight_rejects_deterministic_training_inputs() {
        for reward in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut batch = vec![traj(reward, 0)];
            batch[0].behavior_logprobs = Some(vec![-0.5]);
            assert!(
                UpdatePreset::grpo()
                    .preflight(&batch, 100, 1)
                    .unwrap_err()
                    .to_string()
                    .contains("reward must be finite")
            );
        }

        let mut batch = vec![traj(1.0, 0)];
        batch[0].behavior_logprobs = Some(vec![-0.5]);
        batch[0].prompt_ids.clear();
        assert!(
            UpdatePreset::grpo()
                .preflight(&batch, 100, 1)
                .unwrap_err()
                .to_string()
                .contains("non-empty prompt")
        );

        let mut batch = vec![traj(1.0, 0)];
        batch[0].behavior_logprobs = Some(vec![-0.5]);
        batch[0].response_ids[0] = 100;
        assert!(
            UpdatePreset::grpo()
                .preflight(&batch, 100, 1)
                .unwrap_err()
                .to_string()
                .contains("exceeds vocab=100")
        );
        assert!(
            UpdatePreset::grpo()
                .preflight(&batch, 101, 0)
                .unwrap_err()
                .to_string()
                .contains("window_size must be > 0")
        );
    }

    #[test]
    fn preflight_validates_second_training_trajectory_before_execution() {
        let mut batch = vec![traj(1.0, 0), traj(0.0, 0)];
        batch[0].behavior_logprobs = Some(vec![-0.5]);
        batch[1].behavior_logprobs = Some(vec![f32::NAN]);
        let err = UpdatePreset::grpo().preflight(&batch, 100, 1).unwrap_err();
        assert!(err.to_string().contains("training trajectory 1"));
    }

    #[test]
    fn skipped_trajectories_do_not_require_valid_sidecars() {
        let mut batch = vec![traj(1.0, 0), traj(0.0, 0), traj(1.0, 1), traj(1.0, 1)];
        batch[0].truncated = true;
        batch[1].behavior_logprobs = Some(vec![-0.5]);
        batch[2].behavior_logprobs = Some(vec![f32::NAN]);
        batch[3].response_mask[0] = 0;
        UpdatePreset::dapo().preflight(&batch, 100, 1).unwrap();

        let mut zero_target = traj(1.0, 0);
        zero_target.response_mask[0] = 0;
        zero_target.behavior_logprobs = Some(vec![f32::NAN]);
        UpdatePreset::grpo()
            .preflight(&[zero_target], 100, 1)
            .unwrap();
    }

    #[test]
    fn dapo_filter_drops_zero_variance_groups_and_truncated() {
        let mut batch = vec![traj(1.0, 0), traj(0.0, 0), traj(1.0, 1), traj(1.0, 1)];
        batch[0].truncated = true;
        let kept: Vec<f32> = UpdatePreset::dapo()
            .filter_batch(&batch)
            .iter()
            .map(|t| t.reward)
            .collect();
        // Group 1 zero-variance dropped; the truncated pass dropped; group 0's
        // failing arm survives (variance judged on the group as scored).
        assert_eq!(kept, vec![0.0]);
    }
}

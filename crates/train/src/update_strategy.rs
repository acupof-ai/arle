//! Data-driven OPD policy-update seam. An algorithm is a [`UpdatePreset`] —
//! six orthogonal fields over ONE update path — not a new enum arm; the named
//! constructors below are the algorithm table. The rollout/scoring harness is
//! preset-agnostic: it only reads [`UpdatePreset::needs`].
//!
//! See docs/plans/2026-07-16-agent-rl-unified-infra.md §C.

use autograd::ops::fused_linear_distill::{PgStats, WeightForm, pg_token_weight};
use autograd::{TensorId, TensorStore, optim::Optimizer};

use crate::opd::{
    OpdError, Result, ValueCritic, WritebackLoss, capture_rollout_logprobs,
    masked_writeback_ce_step_dispatch, masked_writeback_step,
};
use crate::qwen35::Qwen35Model;

/// What the harness must collect for a preset: whether to keep failing
/// trajectories, and whether to capture π_behavior logprobs before the update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolloutNeeds {
    pub keep_failing: bool,
    pub rollout_logprobs: bool,
}

/// One scored rollout: the verl-style `(prompt, response, mask)` record plus its
/// scalar reward, group key, and (for ratio-weighted presets) the captured
/// behavior-policy logprobs.
#[derive(Clone, Debug)]
pub struct ScoredTrajectory {
    pub prompt_ids: Vec<u32>,
    pub response_ids: Vec<u32>,
    /// Skip-Observation mask: `1` = LLM token, `0` = tool/environment token.
    pub response_mask: Vec<u8>,
    /// Graded pytest reward in [0,1]; pass ⇔ `reward >= 1.0`.
    pub reward: f32,
    /// π_behavior logprob per masked target position — the IS-ratio denominator
    /// (the policy the trajectory was SAMPLED from; equals πθ at V0 today,
    /// stale-θ under future async staleness). `Some` iff `needs.rollout_logprobs`.
    pub rollout_logprobs: Option<Vec<f32>>,
    /// Generation-time behavior logprobs from the serve sidecar (same order as
    /// `rollout_logprobs`). F.6 diagnostic input only for now: the IS ratio
    /// keeps the V0 recompute until F.6 licenses flipping the source.
    pub gen_logprobs: Option<Vec<f32>>,
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

/// Clip applied to the (detached) ratio before it weights the loss.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClipForm {
    /// Binary keep/drop gate: `1` iff `(1−lo) < r < (1+hi)` (SAO DIS).
    HardGate { lo: f32, hi: f32 },
    /// `clamp(r, 1−lo, 1+hi)` as the weight factor (CISPO family).
    SoftClamp { lo: f32, hi: f32 },
}

impl ClipForm {
    fn weight_form(self) -> WeightForm {
        match self {
            Self::HardGate { lo, hi } => WeightForm::HardGate { lo, hi },
            Self::SoftClamp { lo, hi } => WeightForm::SoftClamp { lo, hi },
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

/// k3 KL(π_b‖πθ) regularizer folded into the per-token weight; the reference is
/// the rollout/behavior policy (a teacher reference is the designed follow-up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KlReg {
    pub coef: f32,
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
    /// F.6 precision diagnostic over trajectories carrying BOTH the sidecar's
    /// generation-time logprobs and the V0 recompute: per-token
    /// `exp(logp_recompute − logp_sidecar)` mean / max, and the tokens covered
    /// (0 = no sidecar coverage this update).
    pub ratio_floor_mean: f32,
    pub ratio_floor_max: f32,
    pub ratio_floor_tokens: usize,
}

/// One policy-update algorithm as data. Extend = a new constructor value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpdatePreset {
    pub filter: SampleFilter,
    pub advantage: Advantage,
    pub ratio: RatioGrain,
    pub clip: ClipForm,
    pub agg: Aggregation,
    pub kl: Option<KlReg>,
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
            kl: None,
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
            kl: None,
        }
    }

    /// SAO Phase 2: Skip-Obs GAE advantages from a learned value critic.
    pub fn sao_value(lo: f32, hi: f32, gamma: f32, lam: f32) -> Self {
        Self {
            advantage: Advantage::ValueGae { gamma, lam },
            ..Self::sao_dis(lo, hi)
        }
    }

    /// GRPO (Shao et al. 2024): group-normalized advantage, clamped token ratio.
    pub fn grpo() -> Self {
        Self {
            filter: SampleFilter::KeepAll,
            advantage: Advantage::Mean {
                scope: Scope::Group,
                std_norm: true,
            },
            ratio: RatioGrain::PerToken,
            clip: ClipForm::SoftClamp { lo: 0.2, hi: 0.2 },
            agg: Aggregation::PerSeqTokenMean,
            kl: None,
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
            clip: ClipForm::SoftClamp { lo: 0.2, hi: 0.28 },
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
            clip: ClipForm::SoftClamp { lo: 3e-4, hi: 4e-4 },
            ..Self::grpo()
        }
    }

    /// CISPO (MiniMax-M1 2025): detached clamped-IS weight, one-sided (lo=1 →
    /// floor 0, no lower clip); the upper bound mirrors the licensed SAO-DIS
    /// bound (the paper leaves ε_high^IS workload-tuned). Batch token-mean.
    pub fn cispo() -> Self {
        Self {
            clip: ClipForm::SoftClamp { lo: 1.0, hi: 3.0 },
            agg: Aggregation::GlobalTokenMean { norm_const: None },
            ..Self::grpo()
        }
    }

    /// Derived mechanically from the preset fields.
    pub fn needs(&self) -> RolloutNeeds {
        RolloutNeeds {
            keep_failing: self.filter != SampleFilter::PassOnly,
            rollout_logprobs: self.ratio != RatioGrain::None,
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
        let survivors = self.filter_batch(batch);
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

    /// Weighted policy-gradient path: per-trajectory detached weights =
    /// advantage × aggregation norm × (seq-ratio factor for GSPO), the token
    /// ratio gate applied inside the fused op per `clip`.
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

        // GlobalTokenMean denominator: the surviving batch's masked-token count
        // (zero-advantage trajectories contribute 0 grad either way).
        let batch_tokens: usize = survivors
            .iter()
            .map(|t| t.rollout_logprobs.as_ref().map_or(0, Vec::len))
            .sum();
        let step_each = matches!(self.agg, Aggregation::PerSeqTokenMean);
        let kl_coef = self.kl.map_or(0.0, |k| k.coef);

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
        // F.6: per-token exp(logp_recompute − logp_sidecar) over trajectories
        // where both sources exist — measures the serve↔train numerics gap
        // (FP8 serve vs bf16 train) plus any θ staleness.
        let mut rf_sum = 0.0f64;
        let mut rf_max = 0.0f32;
        let mut rf_tokens = 0usize;

        for (i, traj) in survivors.iter().enumerate() {
            let rollout_logprobs = traj.rollout_logprobs.as_deref().ok_or_else(|| {
                OpdError::InvalidInput(
                    "ratio-weighted update requires rollout_logprobs (harness must capture \
                     π_behavior when needs.rollout_logprobs)"
                        .to_owned(),
                )
            })?;
            if let Some(gen_lp) = traj.gen_logprobs.as_deref()
                && gen_lp.len() == rollout_logprobs.len()
            {
                for (&recompute, &sidecar) in rollout_logprobs.iter().zip(gen_lp) {
                    let ratio = (recompute - sidecar).exp();
                    rf_sum += f64::from(ratio);
                    rf_max = rf_max.max(ratio);
                }
                rf_tokens += gen_lp.len();
            }
            let (advantages, returns) = match critic.as_deref_mut() {
                Some(c) => c.advantages(
                    student,
                    &traj.prompt_ids,
                    &traj.response_ids,
                    &traj.response_mask,
                    traj.reward,
                    store,
                )?,
                None => (vec![scalar_advs[i]; rollout_logprobs.len()], Vec::new()),
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
            let norm = match self.agg {
                Aggregation::PerSeqTokenMean => advantages.len(),
                Aggregation::GlobalTokenMean { norm_const } => norm_const.unwrap_or(batch_tokens),
            }
            .max(1) as f32;
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
                    let s = seq_ratio_weight(&current_lp, rollout_logprobs, self.clip)?;
                    for w in &mut weights {
                        *w *= s;
                    }
                    WeightForm::Precomputed
                }
                RatioGrain::None => unreachable!("CE path handled in update()"),
            };

            let (loss, traj_stats) = masked_writeback_step(
                WritebackLoss::Pg {
                    rollout_logprobs,
                    weight: &weights,
                    form,
                    kl_coef: kl_coef / norm,
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
            steps += 1;
        }

        if !step_each && steps > 0 {
            opt.step(store, trainable)?;
            opt.zero_grad(store, trainable);
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
            let rf_suffix = if rf_tokens > 0 {
                format!(
                    " ratio_floor_mean={:.4} ratio_floor_max={rf_max:.4} (F.6, {rf_tokens} tokens)",
                    rf_sum / rf_tokens as f64,
                )
            } else {
                String::new()
            };
            eprintln!(
                "[update] trained={steps} mean_policy_loss={:.4} kl={:.4e} clip_frac={:.3}{critic_suffix}{rf_suffix}",
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
            ratio_floor_mean: (rf_sum / rf_tokens.max(1) as f64) as f32,
            ratio_floor_max: rf_max,
            ratio_floor_tokens: rf_tokens,
        })
    }
}

/// P7 staleness: pin the IS-ratio denominator for a version-tagged group.
/// Staleness 0 leaves the batch alone (caller captures the V0 recompute —
/// θ == π_behavior). A STALE group must use the generation-time sidecar: the
/// V0 recompute reflects the post-merge policy (the missing-old-logits
/// failure), so a stale trajectory without sidecar coverage is dropped —
/// never trained with a fake ratio ≈ 1. Returns the number dropped.
pub fn apply_staleness_denominator(batch: &mut Vec<ScoredTrajectory>, staleness: u64) -> usize {
    if staleness == 0 {
        return 0;
    }
    let before = batch.len();
    batch.retain_mut(|t| {
        let Some(lp) = t.gen_logprobs.clone() else {
            return false;
        };
        t.rollout_logprobs = Some(lp);
        true
    });
    before - batch.len()
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

/// GSPO sequence weight: `s = exp(mean_t(logπθ − logπ_b))` gated/clamped at the
/// sequence level via the same [`pg_token_weight`] the op uses per token.
fn seq_ratio_weight(current_lp: &[f32], behavior_lp: &[f32], clip: ClipForm) -> Result<f32> {
    if current_lp.len() != behavior_lp.len() {
        return Err(OpdError::InvalidInput(format!(
            "seq-ratio capture len {} != behavior logprobs len {}",
            current_lp.len(),
            behavior_lp.len()
        )));
    }
    let n = current_lp.len().max(1) as f32;
    let s = (current_lp
        .iter()
        .zip(behavior_lp)
        .map(|(c, b)| c - b)
        .sum::<f32>()
        / n)
        .exp();
    Ok(pg_token_weight(clip.weight_form(), 1.0, s, 0.0).0)
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
            store,
        )?;
        tokens += traj.response_mask.iter().filter(|&&m| m == 1).count();
    }
    Ok(UpdateReport {
        loss: loss_sum / survivors.len() as f32,
        trained: survivors.len(),
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
            rollout_logprobs: None,
            gen_logprobs: None,
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
        for (preset, keep_failing, rollout_logprobs) in cases {
            assert_eq!(
                preset.needs(),
                RolloutNeeds {
                    keep_failing,
                    rollout_logprobs
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
    fn weight_builder_hard_soft_precomputed() {
        let hard = WeightForm::HardGate { lo: 0.2, hi: 0.5 };
        let soft = WeightForm::SoftClamp { lo: 0.2, hi: 0.5 };
        // HardGate: base passes untouched inside (1−lo, 1+hi), else 0.
        assert_eq!(pg_token_weight(hard, 2.0, 1.0, 0.0), (2.0, false));
        assert_eq!(pg_token_weight(hard, 2.0, 1.6, 0.0), (0.0, true));
        // SoftClamp: w = base · clamp(r, 1−lo, 1+hi), detached.
        assert_eq!(pg_token_weight(soft, 2.0, 1.2, 0.0), (2.4, false));
        assert_eq!(pg_token_weight(soft, 2.0, 3.0, 0.0), (3.0, true));
        assert_eq!(pg_token_weight(soft, 2.0, 0.5, 0.0), (1.6, true));
        // KL folds into the weight: w −= coef·(r−1).
        let (w, clipped) = pg_token_weight(WeightForm::Precomputed, 1.0, 2.0, 0.1);
        assert!((w - 0.9).abs() < 1e-6);
        assert!(!clipped);
        // Precomputed-seq (GSPO): s = exp(mean Δlogp), clamped at seq level.
        let clip = ClipForm::SoftClamp { lo: 3e-4, hi: 4e-4 };
        let s = seq_ratio_weight(&[0.0, 0.0], &[-0.2, 0.2], clip).unwrap();
        assert!((s - 1.0).abs() < 1e-6, "mean Δ=0 → s=1, got {s}");
        let s = seq_ratio_weight(&[0.5, 0.5], &[0.0, 0.0], clip).unwrap();
        assert!((s - 1.0004).abs() < 1e-6, "clamped to 1+hi, got {s}");
    }

    #[test]
    fn staleness_version_tag_and_sidecar_denominator() {
        // Group launched at policy version v, trained at v+1 → staleness 1.
        let (behavior_version, policy_version) = (3u64, 4u64);
        let staleness = policy_version - behavior_version;
        assert_eq!(staleness, 1);

        // Stale: the sidecar becomes the ratio denominator; the trajectory
        // without sidecar coverage is dropped (never trained at ratio ≈ 1).
        let mut batch = vec![traj(1.0, 0), traj(0.0, 0)];
        batch[0].gen_logprobs = Some(vec![-0.25]);
        assert_eq!(apply_staleness_denominator(&mut batch, staleness), 1);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].rollout_logprobs.as_deref(), Some(&[-0.25f32][..]));

        // On-policy (staleness 0): untouched — the V0 recompute stays the
        // denominator even when a sidecar is absent.
        let mut batch = vec![traj(1.0, 0)];
        assert_eq!(apply_staleness_denominator(&mut batch, 0), 0);
        assert!(batch[0].rollout_logprobs.is_none());
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

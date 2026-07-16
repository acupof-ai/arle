//! Pluggable OPD policy-update strategy (SAO Phase 1). The rollout/scoring
//! harness is strategy-agnostic; each algorithm is one closed-set enum variant
//! with a static-dispatch `update<O: Optimizer>` (a `dyn` trait would force the
//! generic `Optimizer` behind `dyn`). Extend = one new variant.
//!
//! See `docs/plans/2026-07-11-opd-pluggable-update-strategy.md`.

use autograd::ops::fused_linear_distill::WeightForm;
use autograd::{TensorId, TensorStore, optim::Optimizer};

use crate::opd::{
    OpdError, Result, ValueCritic, WritebackLoss, masked_writeback_ce_step_dispatch,
    masked_writeback_step,
};
use crate::qwen35::Qwen35Model;

/// What the harness must collect for a strategy: whether to keep failing
/// trajectories, and whether to capture π_rollout logprobs before the update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolloutNeeds {
    pub keep_failing: bool,
    pub rollout_logprobs: bool,
}

/// One scored rollout: the verl-style `(prompt, response, mask)` record plus its
/// scalar reward and (for advantage-weighted strategies) the captured π_rollout
/// logprobs — one per masked target position, `Some` iff `needs.rollout_logprobs`.
#[derive(Clone, Debug)]
pub struct ScoredTrajectory {
    pub prompt_ids: Vec<u32>,
    pub response_ids: Vec<u32>,
    /// Skip-Observation mask: `1` = LLM token, `0` = tool/environment token.
    pub response_mask: Vec<u8>,
    /// pytest reward: `1.0` pass / `0.0` fail.
    pub reward: f32,
    pub rollout_logprobs: Option<Vec<f32>>,
}

/// Closed set of policy-update algorithms, selected once by a CLI flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UpdateStrategy {
    /// Current default: reject (reward ≤ 0), masked CE on the survivors.
    RejectionCe,
    /// SAO Phase 1: DIS (per-token PG, clipped importance-ratio gate) with a
    /// batch-centered binary advantage.
    SaoDis { eps_low: f32, eps_high: f32 },
    /// SAO Phase 2: DIS with per-token Skip-Obs GAE advantages from a learned
    /// value critic (γ/λ live on the [`ValueCritic`]). The critic supplies the
    /// baseline (no batch-mean centering) and per-token credit; the caller passes
    /// `Some(critic)` to [`UpdateStrategy::update`].
    SaoValue { eps_low: f32, eps_high: f32 },
}

impl UpdateStrategy {
    pub fn needs(&self) -> RolloutNeeds {
        match self {
            Self::RejectionCe => RolloutNeeds {
                keep_failing: false,
                rollout_logprobs: false,
            },
            Self::SaoDis { .. } | Self::SaoValue { .. } => RolloutNeeds {
                keep_failing: true,
                rollout_logprobs: true,
            },
        }
    }

    /// Apply one round's update over `batch`, returning the mean per-trajectory
    /// loss (same reduction as the pre-strategy per-trajectory writeback loop).
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
    ) -> Result<f32> {
        match self {
            Self::RejectionCe => self.update_rejection_ce(
                batch, student, all_params, trainable, opt, vocab, window, store,
            ),
            Self::SaoDis { eps_low, eps_high } => self.update_sao_dis(
                batch, *eps_low, *eps_high, student, all_params, trainable, opt, vocab, window,
                store,
            ),
            Self::SaoValue { eps_low, eps_high } => {
                let critic = critic.ok_or_else(|| {
                    OpdError::InvalidInput(
                        "SaoValue update requires a ValueCritic (caller must pass Some)".to_owned(),
                    )
                })?;
                self.update_sao_value(
                    batch, *eps_low, *eps_high, student, all_params, trainable, opt, critic, vocab,
                    window, store,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_rejection_ce<O: Optimizer>(
        &self,
        batch: &[ScoredTrajectory],
        student: &Qwen35Model,
        all_params: &[TensorId],
        trainable: &[TensorId],
        opt: &mut O,
        vocab: usize,
        window: usize,
        store: &mut TensorStore,
    ) -> Result<f32> {
        let mut loss_sum = 0.0f32;
        let mut steps = 0usize;
        for traj in batch.iter().filter(|t| t.reward > 0.0) {
            // Dispatch (not `masked_writeback_step(Ce)` directly) so the default
            // path stays byte-identical to the prior closure, honoring the
            // `--writeback-frozen-prompt-kv` opt-in as before.
            let loss = masked_writeback_ce_step_dispatch(
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
            loss_sum += loss;
            steps += 1;
        }
        Ok(if steps > 0 {
            loss_sum / steps as f32
        } else {
            0.0
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn update_sao_dis<O: Optimizer>(
        &self,
        batch: &[ScoredTrajectory],
        eps_low: f32,
        eps_high: f32,
        student: &Qwen35Model,
        all_params: &[TensorId],
        trainable: &[TensorId],
        opt: &mut O,
        vocab: usize,
        window: usize,
        store: &mut TensorStore,
    ) -> Result<f32> {
        if batch.is_empty() {
            return Ok(0.0);
        }
        // Batch-centered binary advantage: A_i = reward_i − mean(reward).
        let mean_reward = batch.iter().map(|t| t.reward).sum::<f32>() / batch.len() as f32;

        let mut loss_sum = 0.0f32;
        let mut steps = 0usize;
        for traj in batch {
            let advantage = traj.reward - mean_reward;
            // No signal → no update. Binary rewards center to exactly 0 when the
            // batch is all-pass or all-fail; stepping anyway lets AdamW weight
            // decay / momentum drift the weights on zero gradient.
            if advantage == 0.0 {
                continue;
            }
            let rollout_logprobs = traj.rollout_logprobs.as_deref().ok_or_else(|| {
                OpdError::InvalidInput(
                    "SaoDis update requires rollout_logprobs (harness must capture π_rollout \
                     when needs.rollout_logprobs)"
                        .to_owned(),
                )
            })?;
            // Constant per-token weight (batch-centered scalar broadcast to
            // every masked position), token-meaned (÷ masked count) to mirror
            // CE's 1/N; SaoValue supplies Skip-Obs GAE instead.
            let n = rollout_logprobs.len().max(1) as f32;
            let weights = vec![advantage / n; rollout_logprobs.len()];
            let (loss, _stats) = masked_writeback_step(
                WritebackLoss::Pg {
                    rollout_logprobs,
                    weight: &weights,
                    form: WeightForm::HardGate {
                        lo: eps_low,
                        hi: eps_high,
                    },
                    kl_coef: 0.0,
                },
                student,
                all_params,
                trainable,
                opt,
                true,
                &traj.prompt_ids,
                &traj.response_ids,
                &traj.response_mask,
                vocab,
                window,
                store,
            )?;
            loss_sum += loss;
            steps += 1;
        }
        Ok(if steps > 0 {
            loss_sum / steps as f32
        } else {
            0.0
        })
    }

    /// SAO Phase 2: per-trajectory Skip-Obs GAE from the critic → per-token DIS
    /// PG on the policy, then one MSE step on the critic. No batch-mean centering
    /// (the critic IS the baseline). At cold start V≈0 so failing trajectories
    /// (reward 0) get ~0 advantage — degrading gracefully to rejection-CE — and
    /// gain a negative signal only as the critic learns to predict >0 on them.
    #[allow(clippy::too_many_arguments)]
    fn update_sao_value<O: Optimizer>(
        &self,
        batch: &[ScoredTrajectory],
        eps_low: f32,
        eps_high: f32,
        student: &Qwen35Model,
        all_params: &[TensorId],
        trainable: &[TensorId],
        opt: &mut O,
        critic: &mut ValueCritic,
        vocab: usize,
        window: usize,
        store: &mut TensorStore,
    ) -> Result<f32> {
        let mut loss_sum = 0.0f32;
        let mut steps = 0usize;
        // Critic health telemetry — the load-bearing question for Phase 2: is
        // V(s) learning (MSE ↓) and producing non-trivial credit (|adv| > 0)? A
        // null result is unattributable without it.
        let mut mse_sum = 0.0f32;
        let mut adv_abs_sum = 0.0f32;
        let mut adv_tokens = 0usize;
        // The policy writeback's `cleanup_after_backward` frees every live tensor
        // not in `all_model_params`; the critic weight isn't a student param, so
        // it must ride in this keep-set (kept, NOT stepped — `trainable` is still
        // LoRA-only) or `critic.update` below hits a freed weight.
        let mut all_with_critic = all_params.to_vec();
        all_with_critic.extend_from_slice(critic.param_ids());
        for traj in batch {
            let rollout_logprobs = traj.rollout_logprobs.as_deref().ok_or_else(|| {
                OpdError::InvalidInput(
                    "SaoValue update requires rollout_logprobs (harness must capture π_rollout \
                     when needs.rollout_logprobs)"
                        .to_owned(),
                )
            })?;
            let (advantages, returns) = critic.advantages(
                student,
                &traj.prompt_ids,
                &traj.response_ids,
                &traj.response_mask,
                traj.reward,
                store,
            )?;
            if advantages.is_empty() {
                continue; // no LLM tokens to train
            }
            // Token-mean the per-token GAE weights (÷ masked count), as CE's 1/N.
            let n = advantages.len() as f32;
            let weights: Vec<f32> = advantages.iter().map(|a| a / n).collect();
            let (loss, _stats) = masked_writeback_step(
                WritebackLoss::Pg {
                    rollout_logprobs,
                    weight: &weights,
                    form: WeightForm::HardGate {
                        lo: eps_low,
                        hi: eps_high,
                    },
                    kl_coef: 0.0,
                },
                student,
                &all_with_critic,
                trainable,
                opt,
                true,
                &traj.prompt_ids,
                &traj.response_ids,
                &traj.response_mask,
                vocab,
                window,
                store,
            )?;
            // Fit the critic toward the observed returns (frozen-attention MSE).
            let mse = critic.update(
                student,
                &traj.prompt_ids,
                &traj.response_ids,
                &traj.response_mask,
                &returns,
                store,
            )?;
            mse_sum += mse;
            adv_abs_sum += advantages.iter().map(|a| a.abs()).sum::<f32>();
            adv_tokens += advantages.len();
            loss_sum += loss;
            steps += 1;
        }
        if steps > 0 {
            eprintln!(
                "[sao-value] trained={steps} mean_policy_loss={:.4} mean_critic_mse={:.4e} mean_adv_abs={:.4e}",
                loss_sum / steps as f32,
                mse_sum / steps as f32,
                adv_abs_sum / adv_tokens.max(1) as f32,
            );
        }
        Ok(if steps > 0 {
            loss_sum / steps as f32
        } else {
            0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traj(reward: f32) -> ScoredTrajectory {
        ScoredTrajectory {
            prompt_ids: vec![1],
            response_ids: vec![2],
            response_mask: vec![1],
            reward,
            rollout_logprobs: None,
        }
    }

    #[test]
    fn needs_are_closed_set() {
        assert_eq!(
            UpdateStrategy::RejectionCe.needs(),
            RolloutNeeds {
                keep_failing: false,
                rollout_logprobs: false
            }
        );
        assert_eq!(
            UpdateStrategy::SaoDis {
                eps_low: 0.8,
                eps_high: 3.0
            }
            .needs(),
            RolloutNeeds {
                keep_failing: true,
                rollout_logprobs: true
            }
        );
    }

    #[test]
    fn sao_advantages_center_to_zero() {
        let batch = [traj(1.0), traj(0.0), traj(1.0), traj(0.0)];
        let mean = batch.iter().map(|t| t.reward).sum::<f32>() / batch.len() as f32;
        let sum: f32 = batch.iter().map(|t| t.reward - mean).sum();
        assert!(sum.abs() < 1e-6, "advantages must sum to ~0, got {sum}");
    }
}

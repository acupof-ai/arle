//! Pluggable OPD policy-update strategy (SAO Phase 1). The rollout/scoring
//! harness is strategy-agnostic; each algorithm is one closed-set enum variant
//! with a static-dispatch `update<O: Optimizer>` (a `dyn` trait would force the
//! generic `Optimizer` behind `dyn`). Extend = one new variant.
//!
//! See `docs/plans/2026-07-11-opd-pluggable-update-strategy.md`.

use autograd::{TensorId, TensorStore, optim::Optimizer};

use crate::opd::{
    OpdError, Result, WritebackLoss, masked_writeback_ce_step_dispatch, masked_writeback_step,
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
}

impl UpdateStrategy {
    pub fn needs(&self) -> RolloutNeeds {
        match self {
            Self::RejectionCe => RolloutNeeds {
                keep_failing: false,
                rollout_logprobs: false,
            },
            Self::SaoDis { .. } => RolloutNeeds {
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
            let loss = masked_writeback_step(
                WritebackLoss::Dis {
                    rollout_logprobs,
                    advantage,
                    eps_low,
                    eps_high,
                },
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

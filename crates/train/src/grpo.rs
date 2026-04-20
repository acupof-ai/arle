use autograd::{
    AutogradError, Result, Tape, TensorId, TensorStore,
    ops::{add, exp, gather_last_dim, log_softmax, matmul, mul, mul_scalar, sum},
};

use crate::{
    policy::{GrpoPolicy, GrpoPolicyConfig},
    rollout::Trajectory,
    sampling::log_prob_at_index,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrpoConfig {
    pub clip_eps: f32,
    pub kl_coef: f32,
    pub group_size: usize,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            clip_eps: 0.2,
            kl_coef: 0.02,
            group_size: 4,
        }
    }
}

pub fn group_advantages(rewards: &[f32], group_size: usize) -> Vec<f32> {
    assert!(group_size > 0, "group size must be positive");
    assert_eq!(
        rewards.len() % group_size,
        0,
        "reward count must be divisible by group size",
    );

    let mut advantages = Vec::with_capacity(rewards.len());
    for group in rewards.chunks(group_size) {
        let mean = group.iter().sum::<f32>() / group.len() as f32;
        let variance = group
            .iter()
            .map(|reward| {
                let centered = *reward - mean;
                centered * centered
            })
            .sum::<f32>()
            / group.len() as f32;
        let std = variance.sqrt();
        for reward in group {
            advantages.push((*reward - mean) / (std + 1.0e-6));
        }
    }
    advantages
}

pub fn grpo_loss<P>(
    policy: &P,
    trajectories: &[Trajectory],
    advantages: &[f32],
    cfg: &GrpoConfig,
    config: &P::Config,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId>
where
    P: GrpoPolicy,
{
    if trajectories.is_empty() {
        return store.from_slice(&[0.0], &[]);
    }
    if advantages.len() != trajectories.len() {
        return Err(AutogradError::InvalidIndicesLen {
            expected: trajectories.len(),
            got: advantages.len(),
        });
    }

    let seq_len = trajectories[0].full_ids.len();
    let mut per_position_advantages = Vec::with_capacity(trajectories.len() * seq_len);
    for (trajectory, advantage) in trajectories.iter().zip(advantages.iter().copied()) {
        per_position_advantages.extend(std::iter::repeat_n(advantage, trajectory.full_ids.len()));
    }

    grpo_loss_from_flat_advantages(
        policy,
        trajectories,
        &per_position_advantages,
        cfg,
        config,
        store,
        tape,
    )
}

pub fn grpo_loss_per_position<P>(
    policy: &P,
    trajectories: &[Trajectory],
    advantages_per_position: &[f32],
    cfg: &GrpoConfig,
    config: &P::Config,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId>
where
    P: GrpoPolicy,
{
    if trajectories.is_empty() {
        return store.from_slice(&[0.0], &[]);
    }

    grpo_loss_from_flat_advantages(
        policy,
        trajectories,
        advantages_per_position,
        cfg,
        config,
        store,
        tape,
    )
}

fn grpo_loss_from_flat_advantages<P>(
    policy: &P,
    trajectories: &[Trajectory],
    advantages_per_position: &[f32],
    cfg: &GrpoConfig,
    config: &P::Config,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId>
where
    P: GrpoPolicy,
{
    if cfg.group_size == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "positive group size",
            got: 0,
        });
    }
    if trajectories.is_empty() {
        return store.from_slice(&[0.0], &[]);
    }

    let batch = trajectories.len();
    let seq_len = trajectories[0].full_ids.len();
    validate_trajectories(trajectories, seq_len, config.max_seq_len())?;
    if advantages_per_position.len() != batch * seq_len {
        return Err(AutogradError::InvalidIndicesLen {
            expected: batch * seq_len,
            got: advantages_per_position.len(),
        });
    }

    let batch_data =
        GrpoBatch::from_trajectories(trajectories, advantages_per_position, cfg.kl_coef, seq_len);

    let logits_id =
        policy.forward_batch_tokens(&batch_data.full_ids, batch, seq_len, store, tape)?;
    let log_probs_id = log_softmax(logits_id, store, tape)?;
    let gathered_next_id = gather_last_dim(log_probs_id, &batch_data.next_token_ids, store, tape)?;
    let shift_id = store.from_slice(&shift_matrix(seq_len), &[seq_len, seq_len])?;
    let new_lp_id = matmul(gathered_next_id, shift_id, store, tape)?;

    let old_lp_tensor = store.from_slice(&batch_data.old_log_probs, &[batch, seq_len])?;
    let old_lp_delta = mul_scalar(old_lp_tensor, -1.0, store, tape)?;
    let ratio_input = add(new_lp_id, old_lp_delta, store, tape)?;
    let ratio = exp(ratio_input, store, tape)?;

    // PPO-clip active mask (host-space, detached). min(r*A, clip(r)*A) has
    // zero gradient wherever the clipped branch is binding, so we equivalently
    // multiply the unclipped surrogate by an indicator that blocks gradient
    // flow on clipped elements. Clipping condition: A>=0 & r>1+eps, or A<0 & r<1-eps.
    let new_lp_values = store.to_host(new_lp_id)?;
    let active_mask_values = ppo_active_mask(
        &new_lp_values,
        &batch_data.old_log_probs,
        &batch_data.advantages,
        &batch_data.mask,
        cfg.clip_eps,
    );

    let active_tensor = store.from_slice(&active_mask_values, &[batch, seq_len])?;
    let pg_scale_tensor = store.from_slice(&batch_data.pg_scales, &[batch, seq_len])?;
    let masked_scale = mul(pg_scale_tensor, active_tensor, store, tape)?;
    let masked_pg = mul(ratio, masked_scale, store, tape)?;
    let loss_pg = mul_scalar(sum(masked_pg, store, tape)?, -1.0, store, tape)?;

    let ref_lp_tensor = store.from_slice(&batch_data.ref_log_probs, &[batch, seq_len])?;
    let neg_new_lp = mul_scalar(new_lp_id, -1.0, store, tape)?;
    let kl_diff = add(ref_lp_tensor, neg_new_lp, store, tape)?;
    let kl_scale_tensor = store.from_slice(&batch_data.kl_scales, &[batch, seq_len])?;
    let kl_term = sum(mul(kl_diff, kl_scale_tensor, store, tape)?, store, tape)?;

    add(loss_pg, kl_term, store, tape)
}

pub fn ppo_active_mask(
    new_log_probs: &[f32],
    old_log_probs: &[f32],
    advantages: &[f32],
    response_mask: &[f32],
    clip_eps: f32,
) -> Vec<f32> {
    let upper = 1.0 + clip_eps;
    let lower = 1.0 - clip_eps;
    new_log_probs
        .iter()
        .zip(old_log_probs.iter())
        .zip(advantages.iter())
        .zip(response_mask.iter())
        .map(|(((new_lp, old_lp), adv), resp)| {
            if *resp <= 0.0 {
                return 0.0;
            }
            let ratio = (new_lp - old_lp).exp();
            let clipped = if *adv >= 0.0 {
                ratio > upper
            } else {
                ratio < lower
            };
            if clipped { 0.0 } else { 1.0 }
        })
        .collect()
}

pub fn mean_sampled_kl<P>(
    policy: &P,
    trajectories: &[Trajectory],
    config: &P::Config,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<f32>
where
    P: GrpoPolicy,
{
    if trajectories.is_empty() {
        return Ok(0.0);
    }

    let seq_len = trajectories[0].full_ids.len();
    validate_trajectories(trajectories, seq_len, config.max_seq_len())?;

    let was_enabled = tape.enabled;
    let result = (|| {
        tape.set_enabled(false);
        let batch = trajectories.len();
        let batch_ids = batch_full_ids(trajectories);
        let logits_id = policy.forward_batch_tokens(&batch_ids, batch, seq_len, store, tape)?;
        let logits = store.to_host(logits_id)?;
        let row_stride = seq_len * config.vocab_size();
        let mut total_kl = 0.0_f32;
        let mut count = 0usize;
        for (row, trajectory) in trajectories.iter().enumerate() {
            for position in 0..seq_len {
                if !trajectory.response_mask[position] {
                    continue;
                }

                let logits_base =
                    (row * row_stride) + position.saturating_sub(1) * config.vocab_size();
                let new_lp = log_prob_at_index(
                    &logits[logits_base..logits_base + config.vocab_size()],
                    1.0,
                    trajectory.full_ids[position],
                );
                total_kl += trajectory.ref_log_probs[position] - new_lp;
                count += 1;
            }
        }

        Ok(if count == 0 {
            0.0
        } else {
            total_kl / count as f32
        })
    })();
    tape.set_enabled(was_enabled);
    result
}

struct GrpoBatch {
    full_ids: Vec<usize>,
    next_token_ids: Vec<usize>,
    old_log_probs: Vec<f32>,
    ref_log_probs: Vec<f32>,
    advantages: Vec<f32>,
    mask: Vec<f32>,
    pg_scales: Vec<f32>,
    kl_scales: Vec<f32>,
}

impl GrpoBatch {
    fn from_trajectories(
        trajectories: &[Trajectory],
        advantages_per_position: &[f32],
        kl_coef: f32,
        seq_len: usize,
    ) -> Self {
        let total_positions = trajectories.len() * seq_len;
        let mut full_ids = Vec::with_capacity(total_positions);
        let mut next_token_ids = Vec::with_capacity(total_positions);
        let mut old_log_probs = Vec::with_capacity(total_positions);
        let mut ref_log_probs = Vec::with_capacity(total_positions);
        let mut advantages = Vec::with_capacity(total_positions);
        let mut mask = Vec::with_capacity(total_positions);
        let mut pg_scales = Vec::with_capacity(total_positions);
        let mut kl_scales = Vec::with_capacity(total_positions);

        let mut offset = 0usize;
        for trajectory in trajectories {
            let response_count = trajectory
                .response_mask
                .iter()
                .filter(|masked| **masked)
                .count();
            let response_scale = if response_count == 0 {
                0.0
            } else {
                1.0 / response_count as f32
            };
            for position in 0..seq_len {
                full_ids.push(trajectory.full_ids[position]);
                next_token_ids.push(if position + 1 < seq_len {
                    trajectory.full_ids[position + 1]
                } else {
                    0
                });
                old_log_probs.push(trajectory.old_log_probs[position]);
                ref_log_probs.push(trajectory.ref_log_probs[position]);
                let response_mask = if trajectory.response_mask[position] {
                    1.0
                } else {
                    0.0
                };
                mask.push(response_mask);
                let advantage = advantages_per_position[offset + position];
                advantages.push(advantage);
                let scale = response_mask * response_scale;
                pg_scales.push(advantage * scale);
                kl_scales.push(kl_coef * scale);
            }
            offset += seq_len;
        }

        Self {
            full_ids,
            next_token_ids,
            old_log_probs,
            ref_log_probs,
            advantages,
            mask,
            pg_scales,
            kl_scales,
        }
    }
}

fn validate_trajectories(
    trajectories: &[Trajectory],
    seq_len: usize,
    max_seq_len: usize,
) -> Result<()> {
    if seq_len > max_seq_len {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![max_seq_len],
            got: vec![seq_len],
        });
    }
    for trajectory in trajectories {
        validate_trajectory(trajectory, seq_len, max_seq_len)?;
    }
    Ok(())
}

fn validate_trajectory(trajectory: &Trajectory, seq_len: usize, max_seq_len: usize) -> Result<()> {
    if seq_len > max_seq_len {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![max_seq_len],
            got: vec![seq_len],
        });
    }
    if trajectory.prompt_ids.len() != seq_len
        || trajectory.response_mask.len() != seq_len
        || trajectory.full_ids.len() != seq_len
        || trajectory.old_log_probs.len() != seq_len
        || trajectory.ref_log_probs.len() != seq_len
    {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![seq_len],
            got: vec![
                trajectory.prompt_ids.len(),
                trajectory.response_mask.len(),
                trajectory.full_ids.len(),
                trajectory.old_log_probs.len(),
                trajectory.ref_log_probs.len(),
            ],
        });
    }
    Ok(())
}

fn shift_matrix(seq_len: usize) -> Vec<f32> {
    let mut data = vec![0.0; seq_len * seq_len];
    for position in 1..seq_len {
        data[((position - 1) * seq_len) + position] = 1.0;
    }
    data
}

fn batch_full_ids(trajectories: &[Trajectory]) -> Vec<usize> {
    let total = trajectories
        .iter()
        .map(|trajectory| trajectory.full_ids.len())
        .sum();
    let mut batch_ids = Vec::with_capacity(total);
    for trajectory in trajectories {
        batch_ids.extend_from_slice(&trajectory.full_ids);
    }
    batch_ids
}

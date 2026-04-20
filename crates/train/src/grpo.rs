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

    let mut loss_id = None;
    for (trajectory, advantage) in trajectories.iter().zip(advantages.iter().copied()) {
        let per_position = vec![advantage; trajectory.full_ids.len()];
        let trajectory_loss =
            grpo_loss_single(policy, trajectory, &per_position, cfg, config, store, tape)?;
        loss_id = Some(match loss_id {
            Some(current) => add(current, trajectory_loss, store, tape)?,
            None => trajectory_loss,
        });
    }

    Ok(loss_id.expect("non-empty trajectories already checked"))
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

    let mut offset = 0usize;
    let mut loss_id = None;
    for trajectory in trajectories {
        let seq_len = trajectory.full_ids.len();
        if offset + seq_len > advantages_per_position.len() {
            return Err(AutogradError::InvalidIndicesLen {
                expected: offset + seq_len,
                got: advantages_per_position.len(),
            });
        }
        let trajectory_loss = grpo_loss_single(
            policy,
            trajectory,
            &advantages_per_position[offset..offset + seq_len],
            cfg,
            config,
            store,
            tape,
        )?;
        offset += seq_len;
        loss_id = Some(match loss_id {
            Some(current) => add(current, trajectory_loss, store, tape)?,
            None => trajectory_loss,
        });
    }

    if offset != advantages_per_position.len() {
        return Err(AutogradError::InvalidIndicesLen {
            expected: offset,
            got: advantages_per_position.len(),
        });
    }

    Ok(loss_id.expect("non-empty trajectories already checked"))
}

fn grpo_loss_single<P>(
    policy: &P,
    trajectory: &Trajectory,
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
    let seq_len = trajectory.full_ids.len();
    if advantages_per_position.len() != seq_len {
        return Err(AutogradError::InvalidIndicesLen {
            expected: seq_len,
            got: advantages_per_position.len(),
        });
    }
    validate_trajectory(trajectory, seq_len, config.max_seq_len())?;
    let batch_data =
        GrpoBatch::from_trajectory_per_position(trajectory, advantages_per_position, seq_len);

    let logits_id = policy.forward_single(&batch_data.full_ids, store, tape)?;
    let log_probs_id = log_softmax(logits_id, store, tape)?;
    let gathered_next_id = gather_last_dim(log_probs_id, &batch_data.next_token_ids, store, tape)?;
    let shift_id = store.from_slice(&shift_matrix(seq_len), &[seq_len, seq_len])?;
    let new_lp_id = matmul(gathered_next_id, shift_id, store, tape)?;

    let old_lp_tensor = store.from_slice(&batch_data.old_log_probs, &[1, seq_len])?;
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

    let adv_tensor = store.from_slice(&batch_data.advantages, &[1, seq_len])?;
    let mask_tensor = store.from_slice(&batch_data.mask, &[1, seq_len])?;
    let active_tensor = store.from_slice(&active_mask_values, &[1, seq_len])?;
    let masked_adv = mul(adv_tensor, active_tensor, store, tape)?;
    let masked_pg = mul(ratio, masked_adv, store, tape)?;
    let loss_pg = mul_scalar(
        sum(masked_pg, store, tape)?,
        -1.0 / batch_data.n_response_positions as f32,
        store,
        tape,
    )?;

    let ref_lp_tensor = store.from_slice(&batch_data.ref_log_probs, &[1, seq_len])?;
    let neg_new_lp = mul_scalar(new_lp_id, -1.0, store, tape)?;
    let kl_diff = add(ref_lp_tensor, neg_new_lp, store, tape)?;
    let masked_kl = mul(kl_diff, mask_tensor, store, tape)?;
    let kl_term = mul_scalar(
        sum(masked_kl, store, tape)?,
        cfg.kl_coef / batch_data.n_response_positions as f32,
        store,
        tape,
    )?;

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
        let mut total_kl = 0.0_f32;
        let mut count = 0usize;
        for trajectory in trajectories {
            let logits_id = policy.forward_single(&trajectory.full_ids, store, tape)?;
            let logits = store.to_host(logits_id)?;
            for position in 0..seq_len {
                if !trajectory.response_mask[position] {
                    continue;
                }

                let logits_base = position.saturating_sub(1) * config.vocab_size();
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
    n_response_positions: usize,
}

impl GrpoBatch {
    fn from_trajectory_per_position(
        trajectory: &Trajectory,
        advantages_per_position: &[f32],
        seq_len: usize,
    ) -> Self {
        let mut next_token_ids = Vec::with_capacity(seq_len);
        let mut old_log_probs = Vec::with_capacity(seq_len);
        let mut ref_log_probs = Vec::with_capacity(seq_len);
        let mut mask = Vec::with_capacity(seq_len);
        let mut n_response_positions = 0usize;

        for position in 0..seq_len {
            next_token_ids.push(if position + 1 < seq_len {
                trajectory.full_ids[position + 1]
            } else {
                0
            });
            old_log_probs.push(trajectory.old_log_probs[position]);
            ref_log_probs.push(trajectory.ref_log_probs[position]);
            if trajectory.response_mask[position] {
                mask.push(1.0);
                n_response_positions += 1;
            } else {
                mask.push(0.0);
            }
        }

        Self {
            full_ids: trajectory.full_ids.clone(),
            next_token_ids,
            old_log_probs,
            ref_log_probs,
            advantages: advantages_per_position.to_vec(),
            mask,
            n_response_positions,
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

use autograd::{
    AutogradError, Result, Tape, TensorId, TensorStore,
    ops::{add, exp, gather_last_dim, log_softmax, matmul, mul, mul_scalar, sum},
};

use crate::{
    model::{TinyLM, TinyLMConfig},
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

pub fn grpo_loss(
    policy: &TinyLM,
    trajectories: &[Trajectory],
    advantages: &[f32],
    cfg: &GrpoConfig,
    config: &TinyLMConfig,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if trajectories.is_empty() {
        return store.from_slice(&[0.0], &[]);
    }
    if cfg.group_size == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "positive group size",
            got: 0,
        });
    }
    if advantages.len() != trajectories.len() {
        return Err(AutogradError::InvalidIndicesLen {
            expected: trajectories.len(),
            got: advantages.len(),
        });
    }
    let _ = cfg.clip_eps;

    let batch = trajectories.len();
    let seq_len = trajectories[0].full_ids.len();
    validate_trajectories(trajectories, seq_len, config.max_seq_len)?;
    let batch_data = GrpoBatch::from_trajectories(trajectories, advantages, seq_len);

    let logits_id = policy.forward(&batch_data.full_ids, batch, seq_len, store, tape)?;
    let log_probs_id = log_softmax(logits_id, store, tape)?;
    let gathered_next_id = gather_last_dim(log_probs_id, &batch_data.next_token_ids, store, tape)?;
    let shift_id = store.from_slice(&shift_matrix(seq_len), &[seq_len, seq_len])?;
    let new_lp_id = matmul(gathered_next_id, shift_id, store, tape)?;

    let old_lp_tensor = store.from_slice(&batch_data.old_log_probs, &[batch, seq_len])?;
    let old_lp_delta = mul_scalar(old_lp_tensor, -1.0, store, tape)?;
    let ratio_input = add(new_lp_id, old_lp_delta, store, tape)?;
    let ratio = exp(ratio_input, store, tape)?;

    let adv_tensor = store.from_slice(&batch_data.advantages, &[batch, seq_len])?;
    let mask_tensor = store.from_slice(&batch_data.mask, &[batch, seq_len])?;
    let masked_adv = mul(adv_tensor, mask_tensor, store, tape)?;
    let masked_pg = mul(ratio, masked_adv, store, tape)?;
    let loss_pg = mul_scalar(
        sum(masked_pg, store, tape)?,
        -1.0 / batch_data.n_response_positions as f32,
        store,
        tape,
    )?;

    let ref_lp_tensor = store.from_slice(&batch_data.ref_log_probs, &[batch, seq_len])?;
    let neg_new_lp = mul_scalar(new_lp_id, -1.0, store, tape)?;
    let kl_diff = add(ref_lp_tensor, neg_new_lp, store, tape)?;
    let masked_kl = mul(kl_diff, mask_tensor, store, tape)?;
    let kl_term = mul_scalar(
        sum(masked_kl, store, tape)?,
        cfg.kl_coef / batch_data.n_response_positions as f32,
        store,
        tape,
    )?;

    // PPO clipping is intentionally deferred for M4; M3 uses the unclipped GRPO ratio.
    add(loss_pg, kl_term, store, tape)
}

pub fn mean_sampled_kl(
    policy: &TinyLM,
    trajectories: &[Trajectory],
    config: &TinyLMConfig,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<f32> {
    if trajectories.is_empty() {
        return Ok(0.0);
    }

    let seq_len = trajectories[0].full_ids.len();
    validate_trajectories(trajectories, seq_len, config.max_seq_len)?;
    let batch = trajectories.len();
    let flat_ids = trajectories
        .iter()
        .flat_map(|trajectory| trajectory.full_ids.iter().copied())
        .collect::<Vec<_>>();

    let was_enabled = tape.enabled;
    let result = (|| {
        tape.set_enabled(false);
        let logits_id = policy.forward(&flat_ids, batch, seq_len, store, tape)?;
        let logits = store.to_host(logits_id)?;

        let mut total_kl = 0.0_f32;
        let mut count = 0usize;
        for (row, trajectory) in trajectories.iter().enumerate() {
            for position in 0..seq_len {
                if !trajectory.response_mask[position] {
                    continue;
                }

                let logits_row = row * seq_len + (position - 1);
                let logits_base = logits_row * config.vocab_size;
                let new_lp = log_prob_at_index(
                    &logits[logits_base..logits_base + config.vocab_size],
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
    fn from_trajectories(trajectories: &[Trajectory], advantages: &[f32], seq_len: usize) -> Self {
        let mut full_ids = Vec::with_capacity(trajectories.len() * seq_len);
        let mut next_token_ids = Vec::with_capacity(trajectories.len() * seq_len);
        let mut old_log_probs = Vec::with_capacity(trajectories.len() * seq_len);
        let mut ref_log_probs = Vec::with_capacity(trajectories.len() * seq_len);
        let mut expanded_advantages = Vec::with_capacity(trajectories.len() * seq_len);
        let mut mask = Vec::with_capacity(trajectories.len() * seq_len);
        let mut n_response_positions = 0usize;

        for (trajectory, advantage) in trajectories.iter().zip(advantages.iter()) {
            full_ids.extend_from_slice(&trajectory.full_ids);
            for position in 0..seq_len {
                next_token_ids.push(if position + 1 < seq_len {
                    trajectory.full_ids[position + 1]
                } else {
                    0
                });
                old_log_probs.push(trajectory.old_log_probs[position]);
                ref_log_probs.push(trajectory.ref_log_probs[position]);
                expanded_advantages.push(*advantage);
                if trajectory.response_mask[position] {
                    mask.push(1.0);
                    n_response_positions += 1;
                } else {
                    mask.push(0.0);
                }
            }
        }

        Self {
            full_ids,
            next_token_ids,
            old_log_probs,
            ref_log_probs,
            advantages: expanded_advantages,
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

use autograd::{AutogradError, Result, Tape, TensorStore};

use crate::{
    dataset::LcgRng,
    policy::{GrpoPolicy, GrpoPolicyConfig},
    policy_support::retained_ids,
    sampling::{log_prob_at_index, sample_categorical},
};

#[derive(Debug, Clone)]
pub struct Trajectory {
    pub prompt_ids: Vec<usize>,
    pub response_mask: Vec<bool>,
    pub full_ids: Vec<usize>,
    pub old_log_probs: Vec<f32>,
    pub ref_log_probs: Vec<f32>,
    pub reward: f32,
}

pub fn rollout_group<P>(
    policy: &P,
    ref_model: &P,
    config: &P::Config,
    prompts: &[Vec<usize>],
    group_size: usize,
    temperature: f32,
    rng: &mut LcgRng,
    verifier: &impl Fn(&[usize], &[usize], &[bool]) -> f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<Vec<Trajectory>>
where
    P: GrpoPolicy,
{
    if prompts.is_empty() || group_size == 0 {
        return Ok(Vec::new());
    }

    let seq_len = prompts[0].len();
    validate_prompt_batch(prompts, seq_len, config.max_seq_len())?;
    let response_mask = response_mask(seq_len)?;
    let position_ids = (0..seq_len).collect::<Vec<_>>();
    let was_enabled = tape.enabled;
    let trajectories = (|| {
        tape.set_enabled(false);

        let mut trajectories = Vec::with_capacity(prompts.len() * group_size);
        for prompt in prompts {
            for _ in 0..group_size {
                trajectories.push(Trajectory {
                    prompt_ids: prompt.clone(),
                    response_mask: response_mask.clone(),
                    full_ids: prompt.clone(),
                    old_log_probs: vec![0.0; seq_len],
                    ref_log_probs: vec![0.0; seq_len],
                    reward: 0.0,
                });
            }
        }
        let total_tokens: usize = trajectories
            .iter()
            .map(|trajectory| trajectory.full_ids.len())
            .sum();
        let mut batch_ids = Vec::with_capacity(total_tokens);
        let mut position_logits = Vec::with_capacity(trajectories.len() * config.vocab_size());

        for (position, masked) in response_mask.iter().enumerate() {
            if !*masked {
                continue;
            }

            fill_batch_full_ids(&mut batch_ids, &trajectories);
            let logits_id = policy.forward_batch_tokens_with_positions(
                &batch_ids,
                &position_ids,
                trajectories.len(),
                store,
                tape,
            )?;
            let logits = store.to_host(logits_id)?;
            fill_batch_position_logits(
                &mut position_logits,
                &logits,
                trajectories.len(),
                seq_len,
                position,
                config.vocab_size(),
            );
            let (sampled_ids, sampled_log_probs) = sample_categorical(
                &position_logits,
                (trajectories.len(), 1),
                config.vocab_size(),
                temperature,
                rng,
            );

            for (trajectory, (sampled_id, sampled_log_prob)) in trajectories
                .iter_mut()
                .zip(sampled_ids.into_iter().zip(sampled_log_probs))
            {
                trajectory.full_ids[position] = sampled_id;
                trajectory.old_log_probs[position] = sampled_log_prob;
            }

            let keep = retained_ids(&[policy, ref_model], store);
            store.retain_ids(&keep);
        }

        fill_batch_full_ids(&mut batch_ids, &trajectories);
        let ref_logits_id = ref_model.forward_batch_tokens_with_positions(
            &batch_ids,
            &position_ids,
            trajectories.len(),
            store,
            tape,
        )?;
        let ref_logits = store.to_host(ref_logits_id)?;
        for (row, trajectory) in trajectories.iter_mut().enumerate() {
            for (position, masked) in trajectory.response_mask.iter().enumerate() {
                if !*masked {
                    continue;
                }
                let logits_base =
                    ((row * seq_len) + position.saturating_sub(1)) * config.vocab_size();
                trajectory.ref_log_probs[position] = log_prob_at_index(
                    &ref_logits[logits_base..logits_base + config.vocab_size()],
                    1.0,
                    trajectory.full_ids[position],
                );
            }
            trajectory.reward = verifier(
                &trajectory.prompt_ids,
                &trajectory.full_ids,
                &trajectory.response_mask,
            );
        }

        Ok(trajectories)
    })();

    tape.set_enabled(was_enabled);
    if trajectories.is_ok() {
        let keep = retained_ids(&[policy, ref_model], store);
        store.retain_ids(&keep);
    }
    trajectories
}

fn batch_full_ids(trajectories: &[Trajectory]) -> Vec<usize> {
    let total = trajectories
        .iter()
        .map(|trajectory| trajectory.full_ids.len())
        .sum();
    let mut batch_ids = Vec::with_capacity(total);
    fill_batch_full_ids(&mut batch_ids, trajectories);
    batch_ids
}

fn fill_batch_full_ids(out: &mut Vec<usize>, trajectories: &[Trajectory]) {
    out.clear();
    for trajectory in trajectories {
        out.extend_from_slice(&trajectory.full_ids);
    }
}

fn batch_position_logits(
    logits: &[f32],
    batch: usize,
    seq_len: usize,
    position: usize,
    vocab_size: usize,
) -> Vec<f32> {
    let row_stride = seq_len * vocab_size;
    let mut position_logits = Vec::with_capacity(batch * vocab_size);
    fill_batch_position_logits(
        &mut position_logits,
        logits,
        batch,
        seq_len,
        position,
        vocab_size,
    );
    position_logits
}

fn fill_batch_position_logits(
    out: &mut Vec<f32>,
    logits: &[f32],
    batch: usize,
    seq_len: usize,
    position: usize,
    vocab_size: usize,
) {
    let row_stride = seq_len * vocab_size;
    out.clear();
    for row in 0..batch {
        let base = (row * row_stride) + position.saturating_sub(1) * vocab_size;
        out.extend_from_slice(&logits[base..base + vocab_size]);
    }
}

fn validate_prompt_batch(prompts: &[Vec<usize>], seq_len: usize, max_seq_len: usize) -> Result<()> {
    if seq_len > max_seq_len {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![max_seq_len],
            got: vec![seq_len],
        });
    }
    for prompt in prompts {
        if prompt.len() != seq_len {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![seq_len],
                got: vec![prompt.len()],
            });
        }
    }
    Ok(())
}

fn response_mask(seq_len: usize) -> Result<Vec<bool>> {
    let prefix_len = seq_len / 2;
    if prefix_len == 0 || prefix_len + 1 >= seq_len {
        return Err(AutogradError::InvalidRank {
            expected: "sequence with prompt, separator, and response positions",
            got: seq_len,
        });
    }

    let mut mask = vec![false; seq_len];
    for value in &mut mask[prefix_len + 1..] {
        *value = true;
    }
    Ok(mask)
}

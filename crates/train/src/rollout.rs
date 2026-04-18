use std::collections::HashSet;

use autograd::{AutogradError, Result, Tape, TensorStore};

use crate::{
    dataset::LcgRng,
    model::{TinyLM, TinyLMConfig},
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

pub fn rollout_group(
    policy: &TinyLM,
    ref_model: &TinyLM,
    config: &TinyLMConfig,
    prompts: &[Vec<usize>],
    group_size: usize,
    temperature: f32,
    rng: &mut LcgRng,
    verifier: &impl Fn(&[usize], &[usize], &[bool]) -> f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<Vec<Trajectory>> {
    if prompts.is_empty() || group_size == 0 {
        return Ok(Vec::new());
    }

    let seq_len = prompts[0].len();
    validate_prompt_batch(prompts, seq_len, config.max_seq_len)?;
    let response_mask = response_mask(seq_len)?;
    let batch = prompts.len() * group_size;
    let mut flat_prompts = Vec::with_capacity(batch * seq_len);
    let mut repeated_prompts = Vec::with_capacity(batch);
    for prompt in prompts {
        for _ in 0..group_size {
            flat_prompts.extend_from_slice(prompt);
            repeated_prompts.push(prompt.clone());
        }
    }

    let was_enabled = tape.enabled;
    let trajectories = (|| {
        tape.set_enabled(false);

        let logits_id = policy.forward(&flat_prompts, batch, seq_len, store, tape)?;
        let logits = store.to_host(logits_id)?;
        let (sampled_ids, sampled_log_probs) = sample_categorical(
            &logits,
            (batch, seq_len),
            config.vocab_size,
            temperature,
            rng,
        );

        let mut trajectories = Vec::with_capacity(batch);
        let mut full_batch = Vec::with_capacity(batch * seq_len);
        for (row, prompt_ids) in repeated_prompts.iter().enumerate() {
            let mut full_ids = prompt_ids.clone();
            let mut old_log_probs = vec![0.0; seq_len];

            for position in 0..seq_len {
                if !response_mask[position] {
                    continue;
                }

                // Causal next-token LM alignment: logits at position t - 1 predict token t.
                let prediction_index = (row * seq_len) + (position - 1);
                full_ids[position] = sampled_ids[prediction_index];
                old_log_probs[position] = sampled_log_probs[prediction_index];
            }

            full_batch.extend_from_slice(&full_ids);
            trajectories.push(Trajectory {
                prompt_ids: prompt_ids.clone(),
                response_mask: response_mask.clone(),
                full_ids,
                old_log_probs,
                ref_log_probs: vec![0.0; seq_len],
                reward: 0.0,
            });
        }

        let ref_logits_id = ref_model.forward(&full_batch, batch, seq_len, store, tape)?;
        let ref_logits = store.to_host(ref_logits_id)?;
        for (row, trajectory) in trajectories.iter_mut().enumerate() {
            for position in 0..seq_len {
                if !trajectory.response_mask[position] {
                    continue;
                }

                let logits_row = row * seq_len + (position - 1);
                let logits_base = logits_row * config.vocab_size;
                trajectory.ref_log_probs[position] = log_prob_at_index(
                    &ref_logits[logits_base..logits_base + config.vocab_size],
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
        let keep = retained_ids(policy, ref_model, store);
        store.retain_ids(&keep);
    }
    trajectories
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

fn retained_ids(policy: &TinyLM, ref_model: &TinyLM, store: &TensorStore) -> HashSet<usize> {
    let mut keep = HashSet::new();
    for param_id in policy
        .all_parameter_ids()
        .into_iter()
        .chain(ref_model.all_parameter_ids())
    {
        keep.insert(param_id);
        if let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) {
            keep.insert(grad_id);
        }
    }
    keep
}

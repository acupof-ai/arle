use std::collections::HashSet;

use autograd::{AutogradError, Result, Tape, TensorStore};

use crate::{
    dataset::LcgRng,
    policy::{GrpoPolicy, GrpoPolicyConfig},
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
    let was_enabled = tape.enabled;
    let trajectories = (|| {
        tape.set_enabled(false);

        let mut trajectories = Vec::with_capacity(prompts.len() * group_size);
        for prompt in prompts {
            for _ in 0..group_size {
                let mut trajectory = Trajectory {
                    prompt_ids: prompt.clone(),
                    response_mask: response_mask.clone(),
                    full_ids: prompt.clone(),
                    old_log_probs: vec![0.0; seq_len],
                    ref_log_probs: vec![0.0; seq_len],
                    reward: 0.0,
                };

                for (position, masked) in trajectory.response_mask.iter().enumerate() {
                    if !*masked {
                        continue;
                    }

                    let logits_id = policy.forward_single(&trajectory.full_ids, store, tape)?;
                    let logits = store.to_host(logits_id)?;
                    let logits_base = position.saturating_sub(1) * config.vocab_size();
                    let slice = &logits[logits_base..logits_base + config.vocab_size()];
                    let (sampled_ids, sampled_log_probs) =
                        sample_categorical(slice, (1, 1), config.vocab_size(), temperature, rng);
                    trajectory.full_ids[position] = sampled_ids[0];
                    trajectory.old_log_probs[position] = sampled_log_probs[0];

                    let keep = retained_ids(policy, ref_model, store);
                    store.retain_ids(&keep);
                }

                let ref_logits_id = ref_model.forward_single(&trajectory.full_ids, store, tape)?;
                let ref_logits = store.to_host(ref_logits_id)?;
                for (position, masked) in trajectory.response_mask.iter().enumerate() {
                    if !*masked {
                        continue;
                    }
                    let logits_base = position.saturating_sub(1) * config.vocab_size();
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
                trajectories.push(trajectory);
            }
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

fn retained_ids(
    policy: &impl GrpoPolicy,
    ref_model: &impl GrpoPolicy,
    store: &TensorStore,
) -> HashSet<usize> {
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

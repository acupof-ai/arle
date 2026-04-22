//! Multi-turn episode scaffolding for GRPO policies (M4.1).
//!
//! Single episode (unbatched) with interleaved agent turns and environment
//! observations. The episode flattens to a `Trajectory` that GRPO can train
//! on — the only change from M3 single-turn is that `response_mask` is an
//! arbitrary interleaved pattern rather than the suffix-only pattern from
//! `rollout::rollout_group`.
//!
//! Autoregressive sampling matches `rollout_group`'s pattern: for each agent
//! position, forward the known prefix and sample from logits at
//! (position - 1). Environment observations are written deterministically.

use autograd::{AutogradError, Result, Tape, TensorStore};

use crate::{
    dataset::LcgRng,
    policy::{GrpoPolicy, GrpoPolicyConfig},
    policy_support::retained_ids,
    rollout::Trajectory,
    sampling::{log_prob_at_index, sample_categorical_into},
};

#[derive(Debug, Clone, Copy)]
pub struct TurnSpec {
    pub agent_tokens: usize,
    pub observation_tokens: usize,
}

pub trait Environment {
    /// Produce `observation_tokens` deterministic tokens appended after the
    /// agent's slice `history[agent_start..agent_end]`. Must return a vector
    /// of exactly the length specified in the matching `TurnSpec`.
    fn observation(
        &self,
        history: &[usize],
        agent_start: usize,
        agent_end: usize,
        observation_tokens: usize,
    ) -> Vec<usize>;
}

impl<F> Environment for F
where
    F: Fn(&[usize], usize, usize, usize) -> Vec<usize>,
{
    fn observation(
        &self,
        history: &[usize],
        agent_start: usize,
        agent_end: usize,
        observation_tokens: usize,
    ) -> Vec<usize> {
        (self)(history, agent_start, agent_end, observation_tokens)
    }
}

#[derive(Debug, Clone)]
pub struct Episode {
    pub initial_prompt: Vec<usize>,
    pub full_ids: Vec<usize>,
    pub response_mask: Vec<bool>,
    pub old_log_probs: Vec<f32>,
    pub ref_log_probs: Vec<f32>,
    pub reward: f32,
    pub turn_boundaries: Vec<(usize, usize)>,
}

impl Episode {
    pub fn into_trajectory(self) -> Trajectory {
        let seq_len = self.full_ids.len();
        let mut prompt_ids = vec![0usize; seq_len];
        for (position, masked) in self.response_mask.iter().enumerate() {
            if !*masked {
                prompt_ids[position] = self.full_ids[position];
            }
        }
        Trajectory {
            prompt_ids,
            response_mask: self.response_mask,
            full_ids: self.full_ids,
            old_log_probs: self.old_log_probs,
            ref_log_probs: self.ref_log_probs,
            reward: self.reward,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rollout_episode<P>(
    policy: &P,
    ref_model: &P,
    initial_prompt: &[usize],
    turns: &[TurnSpec],
    env: &impl Environment,
    temperature: f32,
    rng: &mut LcgRng,
    verifier: &impl Fn(&Episode) -> f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<Episode>
where
    P: GrpoPolicy,
    P::Config: GrpoPolicyConfig,
{
    let config = policy.config();
    let max_seq_len = config.max_seq_len();
    let vocab_size = config.vocab_size();
    if turns.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least one turn",
            got: 0,
        });
    }
    let total_agent: usize = turns.iter().map(|t| t.agent_tokens).sum();
    let total_obs: usize = turns.iter().map(|t| t.observation_tokens).sum();
    let seq_len = initial_prompt.len() + total_agent + total_obs;
    if seq_len > max_seq_len {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![max_seq_len],
            got: vec![seq_len],
        });
    }
    if initial_prompt.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "non-empty initial prompt",
            got: 0,
        });
    }

    let mut full_ids = vec![0usize; seq_len];
    full_ids[..initial_prompt.len()].copy_from_slice(initial_prompt);
    let mut response_mask = vec![false; seq_len];
    let mut old_log_probs = vec![0.0f32; seq_len];
    let mut turn_boundaries = Vec::with_capacity(turns.len());
    let position_ids = (0..seq_len).collect::<Vec<_>>();

    let was_enabled = tape.enabled;
    let rollout_result = (|| {
        tape.set_enabled(false);
        let keep = retained_ids(&[policy, ref_model], store);
        let mut sampled_ids = Vec::with_capacity(1);
        let mut sampled_log_probs = Vec::with_capacity(1);

        let mut cursor = initial_prompt.len();
        for turn in turns {
            let agent_start = cursor;
            let agent_end = agent_start + turn.agent_tokens;

            for position in agent_start..agent_end {
                if position == 0 {
                    return Err(AutogradError::InvalidRank {
                        expected: "agent turn must start after at least one prompt token",
                        got: 0,
                    });
                }
                // Only the known prefix up to the current token matters for
                // autoregressive sampling; avoid re-forwarding the padded
                // future tail of the episode.
                let logits_id = policy.forward_batch_tokens_with_positions(
                    &full_ids[..position],
                    &position_ids[..position],
                    1,
                    store,
                    tape,
                )?;
                let logits = store.to_host(logits_id)?;
                let logits_base = (position - 1) * vocab_size;
                let slice = &logits[logits_base..logits_base + vocab_size];
                sample_categorical_into(
                    &mut sampled_ids,
                    &mut sampled_log_probs,
                    slice,
                    (1, 1),
                    vocab_size,
                    temperature,
                    rng,
                );
                full_ids[position] = sampled_ids[0];
                old_log_probs[position] = sampled_log_probs[0];
                response_mask[position] = true;

                store.retain_ids(&keep);
            }
            turn_boundaries.push((agent_start, agent_end));

            if turn.observation_tokens > 0 {
                let obs =
                    env.observation(&full_ids, agent_start, agent_end, turn.observation_tokens);
                if obs.len() != turn.observation_tokens {
                    return Err(AutogradError::ShapeMismatch {
                        expected: vec![turn.observation_tokens],
                        got: vec![obs.len()],
                    });
                }
                let obs_start = agent_end;
                let obs_end = obs_start + turn.observation_tokens;
                full_ids[obs_start..obs_end].copy_from_slice(&obs);
            }

            cursor = agent_end + turn.observation_tokens;
        }
        debug_assert_eq!(cursor, seq_len);

        let ref_logits_id = ref_model.forward_batch_tokens_with_positions(
            &full_ids,
            &position_ids,
            1,
            store,
            tape,
        )?;
        let ref_logits = store.to_host(ref_logits_id)?;
        let mut ref_log_probs = vec![0.0f32; seq_len];
        for (position, masked) in response_mask.iter().enumerate() {
            if !*masked || position == 0 {
                continue;
            }
            let logits_base = (position - 1) * vocab_size;
            ref_log_probs[position] = log_prob_at_index(
                &ref_logits[logits_base..logits_base + vocab_size],
                1.0,
                full_ids[position],
            );
        }

        Ok((
            full_ids,
            response_mask,
            old_log_probs,
            ref_log_probs,
            turn_boundaries,
        ))
    })();

    tape.set_enabled(was_enabled);

    let (full_ids, response_mask, old_log_probs, ref_log_probs, turn_boundaries) = rollout_result?;

    let mut episode = Episode {
        initial_prompt: initial_prompt.to_vec(),
        full_ids,
        response_mask,
        old_log_probs,
        ref_log_probs,
        reward: 0.0,
        turn_boundaries,
    };
    episode.reward = verifier(&episode);

    let keep = retained_ids(&[policy, ref_model], store);
    store.retain_ids(&keep);

    Ok(episode)
}

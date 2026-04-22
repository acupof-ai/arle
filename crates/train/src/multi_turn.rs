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
    sampling::{log_prob_at_index, sample_categorical_into, sample_categorical_rows_into},
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

#[allow(clippy::too_many_arguments)]
pub fn rollout_episode_group<P>(
    policy: &P,
    ref_model: &P,
    initial_prompt: &[usize],
    turns: &[TurnSpec],
    env: &impl Environment,
    temperature: f32,
    rngs: &mut [LcgRng],
    verifier: &impl Fn(&Episode) -> f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<Vec<Episode>>
where
    P: GrpoPolicy,
    P::Config: GrpoPolicyConfig,
{
    if rngs.is_empty() {
        return Ok(Vec::new());
    }

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

    let mut response_mask = vec![false; seq_len];
    let mut turn_boundaries = Vec::with_capacity(turns.len());
    let mut cursor = initial_prompt.len();
    for turn in turns {
        let agent_start = cursor;
        let agent_end = agent_start + turn.agent_tokens;
        for value in &mut response_mask[agent_start..agent_end] {
            *value = true;
        }
        turn_boundaries.push((agent_start, agent_end));
        cursor = agent_end + turn.observation_tokens;
    }

    let position_ids = (0..seq_len).collect::<Vec<_>>();
    let was_enabled = tape.enabled;
    let rollout_result = (|| {
        tape.set_enabled(false);
        let keep = retained_ids(&[policy, ref_model], store);

        let mut episodes = Vec::with_capacity(rngs.len());
        for _ in 0..rngs.len() {
            let mut full_ids = vec![0usize; seq_len];
            full_ids[..initial_prompt.len()].copy_from_slice(initial_prompt);
            episodes.push(Episode {
                initial_prompt: initial_prompt.to_vec(),
                full_ids,
                response_mask: response_mask.clone(),
                old_log_probs: vec![0.0; seq_len],
                ref_log_probs: vec![0.0; seq_len],
                reward: 0.0,
                turn_boundaries: turn_boundaries.clone(),
            });
        }

        let mut batch_ids = Vec::with_capacity(rngs.len() * seq_len);
        let mut position_logits = Vec::with_capacity(rngs.len() * vocab_size);
        let mut sampled_ids = Vec::with_capacity(rngs.len());
        let mut sampled_log_probs = Vec::with_capacity(rngs.len());

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
                fill_batch_prefix_ids(&mut batch_ids, &episodes, position);
                let logits_id = policy.forward_batch_tokens_with_positions(
                    &batch_ids,
                    &position_ids[..position],
                    episodes.len(),
                    store,
                    tape,
                )?;
                let logits = store.to_host(logits_id)?;
                fill_batch_last_position_logits(
                    &mut position_logits,
                    &logits,
                    episodes.len(),
                    position,
                    vocab_size,
                );
                sample_categorical_rows_into(
                    &mut sampled_ids,
                    &mut sampled_log_probs,
                    &position_logits,
                    (episodes.len(), 1),
                    vocab_size,
                    temperature,
                    rngs,
                );

                for (episode, (sampled_id, sampled_log_prob)) in episodes.iter_mut().zip(
                    sampled_ids
                        .iter()
                        .copied()
                        .zip(sampled_log_probs.iter().copied()),
                ) {
                    episode.full_ids[position] = sampled_id;
                    episode.old_log_probs[position] = sampled_log_prob;
                }

                store.retain_ids(&keep);
            }

            if turn.observation_tokens > 0 {
                for episode in &mut episodes {
                    let obs = env.observation(
                        &episode.full_ids,
                        agent_start,
                        agent_end,
                        turn.observation_tokens,
                    );
                    if obs.len() != turn.observation_tokens {
                        return Err(AutogradError::ShapeMismatch {
                            expected: vec![turn.observation_tokens],
                            got: vec![obs.len()],
                        });
                    }
                    let obs_start = agent_end;
                    let obs_end = obs_start + turn.observation_tokens;
                    episode.full_ids[obs_start..obs_end].copy_from_slice(&obs);
                }
            }

            cursor = agent_end + turn.observation_tokens;
        }
        debug_assert_eq!(cursor, seq_len);

        fill_batch_full_ids(&mut batch_ids, &episodes);
        let ref_logits_id = ref_model.forward_batch_tokens_with_positions(
            &batch_ids,
            &position_ids,
            episodes.len(),
            store,
            tape,
        )?;
        let ref_logits = store.to_host(ref_logits_id)?;
        let row_stride = seq_len * vocab_size;
        for (row, episode) in episodes.iter_mut().enumerate() {
            for (position, masked) in episode.response_mask.iter().enumerate() {
                if !*masked || position == 0 {
                    continue;
                }
                let logits_base = (row * row_stride) + (position - 1) * vocab_size;
                episode.ref_log_probs[position] = log_prob_at_index(
                    &ref_logits[logits_base..logits_base + vocab_size],
                    1.0,
                    episode.full_ids[position],
                );
            }
        }

        Ok(episodes)
    })();

    tape.set_enabled(was_enabled);

    let mut episodes = rollout_result?;
    for episode in &mut episodes {
        episode.reward = verifier(episode);
    }

    let keep = retained_ids(&[policy, ref_model], store);
    store.retain_ids(&keep);

    Ok(episodes)
}

fn fill_batch_prefix_ids(out: &mut Vec<usize>, episodes: &[Episode], prefix_len: usize) {
    out.clear();
    for episode in episodes {
        out.extend_from_slice(&episode.full_ids[..prefix_len]);
    }
}

fn fill_batch_full_ids(out: &mut Vec<usize>, episodes: &[Episode]) {
    out.clear();
    for episode in episodes {
        out.extend_from_slice(&episode.full_ids);
    }
}

fn fill_batch_last_position_logits(
    out: &mut Vec<f32>,
    logits: &[f32],
    batch: usize,
    seq_len: usize,
    vocab_size: usize,
) {
    let row_stride = seq_len * vocab_size;
    out.clear();
    for row in 0..batch {
        let base = (row * row_stride) + (seq_len - 1) * vocab_size;
        out.extend_from_slice(&logits[base..base + vocab_size]);
    }
}

#[cfg(test)]
mod tests {
    use autograd::{Tape, TensorStore};

    use super::{Episode, TurnSpec, rollout_episode, rollout_episode_group};
    use crate::{
        dataset::LcgRng,
        policy::{GrpoPolicy, GrpoPolicyConfig},
    };

    #[derive(Debug, Clone)]
    struct FakeConfig {
        max_seq_len: usize,
        vocab_size: usize,
    }

    impl GrpoPolicyConfig for FakeConfig {
        fn max_seq_len(&self) -> usize {
            self.max_seq_len
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    #[derive(Debug, Clone)]
    struct FakePolicy {
        config: FakeConfig,
    }

    impl FakePolicy {
        fn new(max_seq_len: usize, vocab_size: usize) -> Self {
            Self {
                config: FakeConfig {
                    max_seq_len,
                    vocab_size,
                },
            }
        }

        fn build_logits(&self, input_ids: &[usize], batch: usize, seq_len: usize) -> Vec<f32> {
            let vocab = self.config.vocab_size;
            let mut logits = Vec::with_capacity(batch * seq_len * vocab);
            for row in 0..batch {
                let row_tokens = &input_ids[row * seq_len..(row + 1) * seq_len];
                for position in 0..seq_len {
                    let prefix_sum = row_tokens[..=position].iter().copied().sum::<usize>();
                    let anchor = (prefix_sum + position) % vocab;
                    for token in 0..vocab {
                        let wrapped = (anchor + vocab - token) % vocab;
                        logits.push(-(wrapped as f32));
                    }
                }
            }
            logits
        }
    }

    impl GrpoPolicy for FakePolicy {
        type Config = FakeConfig;

        fn config(&self) -> &Self::Config {
            &self.config
        }

        fn forward_single(
            &self,
            input_ids: &[usize],
            store: &mut TensorStore,
            _tape: &mut Tape,
        ) -> autograd::Result<autograd::TensorId> {
            let logits = self.build_logits(input_ids, 1, input_ids.len());
            store.from_slice(&logits, &[1, input_ids.len(), self.config.vocab_size()])
        }

        fn forward_batch_tokens(
            &self,
            input_ids: &[usize],
            batch: usize,
            seq_len: usize,
            store: &mut TensorStore,
            _tape: &mut Tape,
        ) -> autograd::Result<autograd::TensorId> {
            let logits = self.build_logits(input_ids, batch, seq_len);
            store.from_slice(&logits, &[batch, seq_len, self.config.vocab_size()])
        }

        fn all_parameter_ids(&self) -> Vec<autograd::TensorId> {
            Vec::new()
        }

        fn clone_frozen(&self, _store: &mut TensorStore) -> Self {
            self.clone()
        }
    }

    #[test]
    fn batched_rollout_matches_serial_rollout() {
        let policy = FakePolicy::new(32, 8);
        let ref_model = policy.clone();
        let initial_prompt = vec![1, 2, 7];
        let turns = vec![
            TurnSpec {
                agent_tokens: 3,
                observation_tokens: 2,
            },
            TurnSpec {
                agent_tokens: 2,
                observation_tokens: 0,
            },
        ];
        let env = |history: &[usize],
                   _agent_start: usize,
                   agent_end: usize,
                   observation_tokens: usize| {
            let base = history[agent_end - 1] % 5;
            (0..observation_tokens)
                .map(|offset| (base + offset + 1) % 8)
                .collect::<Vec<_>>()
        };
        let verifier = |episode: &Episode| episode.full_ids.iter().copied().sum::<usize>() as f32;

        let mut serial_store = TensorStore::default();
        let mut serial_tape = Tape::new();
        let mut expected = Vec::new();
        let mut serial_rngs = vec![LcgRng::seed(11), LcgRng::seed(22), LcgRng::seed(33)];
        for rng in &mut serial_rngs {
            expected.push(
                rollout_episode(
                    &policy,
                    &ref_model,
                    &initial_prompt,
                    &turns,
                    &env,
                    0.8,
                    rng,
                    &verifier,
                    &mut serial_store,
                    &mut serial_tape,
                )
                .expect("serial rollout"),
            );
        }

        let mut batched_store = TensorStore::default();
        let mut batched_tape = Tape::new();
        let mut batched_rngs = vec![LcgRng::seed(11), LcgRng::seed(22), LcgRng::seed(33)];
        let actual = rollout_episode_group(
            &policy,
            &ref_model,
            &initial_prompt,
            &turns,
            &env,
            0.8,
            &mut batched_rngs,
            &verifier,
            &mut batched_store,
            &mut batched_tape,
        )
        .expect("batched rollout");

        assert_eq!(actual.len(), expected.len());
        for (actual_episode, expected_episode) in actual.iter().zip(expected.iter()) {
            assert_eq!(
                actual_episode.initial_prompt,
                expected_episode.initial_prompt
            );
            assert_eq!(actual_episode.full_ids, expected_episode.full_ids);
            assert_eq!(actual_episode.response_mask, expected_episode.response_mask);
            assert_eq!(actual_episode.old_log_probs, expected_episode.old_log_probs);
            assert_eq!(actual_episode.ref_log_probs, expected_episode.ref_log_probs);
            assert_eq!(
                actual_episode.turn_boundaries,
                expected_episode.turn_boundaries
            );
            assert_eq!(actual_episode.reward, expected_episode.reward);
        }
        for (actual_rng, expected_rng) in batched_rngs.iter_mut().zip(serial_rngs.iter_mut()) {
            assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());
        }
    }
}

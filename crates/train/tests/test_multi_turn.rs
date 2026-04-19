use autograd::{Tape, TensorStore, module::Module};
use train::{
    dataset::LcgRng,
    grpo::{GrpoConfig, group_advantages, grpo_loss},
    model::{Transformer, TransformerConfig},
    multi_turn::{Environment, Episode, TurnSpec, rollout_episode},
};

fn tiny_config() -> TransformerConfig {
    TransformerConfig {
        vocab_size: 16,
        d_model: 16,
        n_layers: 2,
        n_heads: 2,
        d_head: 8,
        d_ff: 32,
        max_seq_len: 32,
        lora: None,
    }
}

struct EchoSeparator(usize);

impl Environment for EchoSeparator {
    fn observation(
        &self,
        _history: &[usize],
        _agent_start: usize,
        _agent_end: usize,
        observation_tokens: usize,
    ) -> Vec<usize> {
        vec![self.0; observation_tokens]
    }
}

#[test]
fn rollout_episode_shapes_and_masks() {
    let config = tiny_config();
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Transformer::new(config, &mut store).expect("policy");
    let ref_model = policy.clone_frozen(&mut store);

    let initial_prompt = vec![1usize, 2, 3, 15];
    let turns = [
        TurnSpec {
            agent_tokens: 3,
            observation_tokens: 2,
        },
        TurnSpec {
            agent_tokens: 3,
            observation_tokens: 0,
        },
    ];
    let mut rng = LcgRng::seed(7);
    let env = EchoSeparator(15);
    let episode = rollout_episode(
        &policy,
        &ref_model,
        &config,
        &initial_prompt,
        &turns,
        &env,
        1.0,
        &mut rng,
        &|_: &Episode| 0.0,
        &mut store,
        &mut tape,
    )
    .expect("rollout");

    let expected_len = initial_prompt.len() + 3 + 2 + 3;
    assert_eq!(episode.full_ids.len(), expected_len);
    assert_eq!(episode.response_mask.len(), expected_len);
    assert_eq!(episode.old_log_probs.len(), expected_len);
    assert_eq!(episode.ref_log_probs.len(), expected_len);
    assert_eq!(episode.turn_boundaries, vec![(4, 7), (9, 12)]);

    // Agent positions masked, prompt + observation positions not masked.
    for position in 0..initial_prompt.len() {
        assert!(!episode.response_mask[position], "prompt pos {position}");
    }
    for position in 4..7 {
        assert!(episode.response_mask[position], "agent1 pos {position}");
    }
    for position in 7..9 {
        assert!(!episode.response_mask[position], "obs1 pos {position}");
    }
    for position in 9..12 {
        assert!(episode.response_mask[position], "agent2 pos {position}");
    }

    // Observation tokens are exactly what the env emitted.
    assert_eq!(episode.full_ids[7..9], [15, 15]);

    // log-probs only populated on agent positions.
    for (position, masked) in episode.response_mask.iter().enumerate() {
        if *masked {
            assert!(
                episode.old_log_probs[position].is_finite(),
                "old_lp pos {position}"
            );
        } else {
            assert_eq!(episode.old_log_probs[position], 0.0, "non-agent {position}");
            assert_eq!(episode.ref_log_probs[position], 0.0, "non-agent {position}");
        }
    }
}

#[test]
fn episode_trajectory_feeds_grpo_loss() {
    let config = tiny_config();
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Transformer::new(config, &mut store).expect("policy");
    let ref_model = policy.clone_frozen(&mut store);

    let initial_prompt = vec![1usize, 2, 3, 15];
    let turns = [
        TurnSpec {
            agent_tokens: 2,
            observation_tokens: 2,
        },
        TurnSpec {
            agent_tokens: 2,
            observation_tokens: 0,
        },
    ];
    let env = EchoSeparator(14);
    let mut rng = LcgRng::seed(19);

    let mut trajectories = Vec::with_capacity(4);
    let prompt_len = initial_prompt.len();
    let verifier = |episode: &Episode| -> f32 {
        let mut score = 0.0_f32;
        for (position, masked) in episode.response_mask.iter().enumerate() {
            if *masked
                && episode.full_ids[position] == episode.initial_prompt[position % prompt_len]
            {
                score += 1.0;
            }
        }
        score / episode.response_mask.iter().filter(|m| **m).count().max(1) as f32
    };
    for seed in 0..4 {
        let mut rng_i = LcgRng::seed(seed + 100);
        let episode = rollout_episode(
            &policy,
            &ref_model,
            &config,
            &initial_prompt,
            &turns,
            &env,
            1.0,
            &mut rng_i,
            &verifier,
            &mut store,
            &mut tape,
        )
        .expect("episode");
        trajectories.push(episode.into_trajectory());
        rng.next_u64();
    }

    let rewards: Vec<f32> = trajectories.iter().map(|t| t.reward).collect();
    let advantages = group_advantages(&rewards, 4);

    tape.entries.clear();
    tape.set_enabled(true);
    let loss = grpo_loss(
        &policy,
        &trajectories,
        &advantages,
        &GrpoConfig {
            clip_eps: 0.2,
            kl_coef: 0.02,
            group_size: 4,
        },
        &config,
        &mut store,
        &mut tape,
    )
    .expect("grpo loss");
    let loss_value = store.to_host(loss).expect("loss host")[0];
    assert!(
        loss_value.is_finite(),
        "loss should be finite: {loss_value}"
    );

    tape.backward(loss, &mut store).expect("backward");
    let params = policy.parameters();
    let any_grad = params.iter().any(|id| {
        store
            .get(*id)
            .and_then(|t| t.grad)
            .and_then(|g| store.get(g))
            .is_some_and(|g| g.data.iter().any(|v| v.abs() > 1e-7))
    });
    assert!(any_grad, "expected non-zero gradient on at least one param");
}

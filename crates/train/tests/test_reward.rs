use autograd::{Tape, TensorStore};
use train::{
    dataset::LcgRng,
    grpo::{GrpoConfig, grpo_loss_per_position},
    model::{TinyLM, TinyLMConfig},
    multi_turn::{Environment, Episode, TurnSpec, rollout_episode},
    reward::{discounted_returns, group_normalize, returns_to_per_position},
};

#[test]
fn discounted_returns_undiscounted_matches_suffix_sum() {
    let returns = discounted_returns(&[1.0, 2.0, 3.0], 1.0);
    assert_eq!(returns, vec![6.0, 5.0, 3.0]);
}

#[test]
fn discounted_returns_with_gamma() {
    let returns = discounted_returns(&[1.0, 2.0, 3.0], 0.5);
    // G_2 = 3; G_1 = 2 + 0.5*3 = 3.5; G_0 = 1 + 0.5*3.5 = 2.75
    let expected = [2.75, 3.5, 3.0];
    for (actual, expected) in returns.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "got {actual} expected {expected}"
        );
    }
}

#[test]
fn returns_to_per_position_fans_out() {
    let returns = vec![2.0, 5.0];
    let boundaries = vec![(1, 3), (5, 7)];
    let per_position = returns_to_per_position(&returns, &boundaries, 8);
    assert_eq!(per_position, vec![0.0, 2.0, 2.0, 0.0, 0.0, 5.0, 5.0, 0.0]);
}

#[test]
fn group_normalize_zero_mean_unit_std_per_group() {
    let returns = vec![1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0];
    let advantages = group_normalize(&returns, 4);

    // First group: std ≈ 1.118, advantages ≈ -1.34, -0.45, 0.45, 1.34
    let group1_mean: f32 = advantages[..4].iter().sum::<f32>() / 4.0;
    assert!(group1_mean.abs() < 1e-4);
    // Second group: all identical → std = 0 → all zeros (divided by eps).
    for value in &advantages[4..] {
        assert!(value.abs() < 1e-3, "got {value}");
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

fn tiny_config() -> TinyLMConfig {
    TinyLMConfig {
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

#[test]
fn stepwise_pipeline_feeds_grpo_loss_per_position() {
    use autograd::module::Module;

    let config = tiny_config();
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = TinyLM::new(config, &mut store).expect("policy");
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
    let env = EchoSeparator(15);
    let group = 4usize;
    let gamma = 0.9_f32;

    // Per-turn reward: fraction of agent tokens in this turn equal to 1.
    let per_turn_reward = |episode: &Episode, agent_start: usize, agent_end: usize| -> f32 {
        let mut hits = 0.0_f32;
        let mut count = 0.0_f32;
        for position in agent_start..agent_end {
            count += 1.0;
            if episode.full_ids[position] == 1 {
                hits += 1.0;
            }
        }
        if count == 0.0 { 0.0 } else { hits / count }
    };

    let mut episodes = Vec::with_capacity(group);
    for seed in 0..group {
        let mut rng = LcgRng::seed(seed as u64 + 257);
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
        .expect("episode");
        episodes.push(episode);
    }

    // Per-turn rewards per episode, then discounted returns.
    let mut returns_per_ep: Vec<Vec<f32>> = Vec::with_capacity(group);
    for episode in &episodes {
        let rewards: Vec<f32> = episode
            .turn_boundaries
            .iter()
            .map(|(start, end)| per_turn_reward(episode, *start, *end))
            .collect();
        returns_per_ep.push(discounted_returns(&rewards, gamma));
    }

    // Normalize returns across the group, per turn index. Stack as
    // [turn0_ep0, turn0_ep1, ..., turn0_ep3, turn1_ep0, ..., turn1_ep3].
    let n_turns = turns.len();
    let mut stacked = Vec::with_capacity(group * n_turns);
    for turn_idx in 0..n_turns {
        for ep_returns in &returns_per_ep {
            stacked.push(ep_returns[turn_idx]);
        }
    }
    let normalized = group_normalize(&stacked, group);
    // Pull advantages back into per-episode per-turn layout.
    let mut normalized_per_ep: Vec<Vec<f32>> = vec![vec![0.0; n_turns]; group];
    for turn_idx in 0..n_turns {
        for ep in 0..group {
            normalized_per_ep[ep][turn_idx] = normalized[turn_idx * group + ep];
        }
    }

    // Fan out to per-position advantages, then concatenate across the batch.
    let seq_len = episodes[0].full_ids.len();
    let mut advantages_per_position = Vec::with_capacity(group * seq_len);
    for (episode, adv_by_turn) in episodes.iter().zip(normalized_per_ep.iter()) {
        let row = returns_to_per_position(adv_by_turn, &episode.turn_boundaries, seq_len);
        advantages_per_position.extend_from_slice(&row);
    }
    assert_eq!(advantages_per_position.len(), group * seq_len);

    let trajectories: Vec<_> = episodes.into_iter().map(|e| e.into_trajectory()).collect();

    tape.entries.clear();
    tape.set_enabled(true);
    let loss = grpo_loss_per_position(
        &policy,
        &trajectories,
        &advantages_per_position,
        &GrpoConfig {
            clip_eps: 0.2,
            kl_coef: 0.02,
            group_size: group,
        },
        &config,
        &mut store,
        &mut tape,
    )
    .expect("grpo loss");
    let loss_value = store.to_host(loss).expect("loss host")[0];
    assert!(loss_value.is_finite(), "loss must be finite: {loss_value}");

    tape.backward(loss, &mut store).expect("backward");
    let params = policy.parameters();
    let any_grad = params.iter().any(|id| {
        store
            .get(*id)
            .and_then(|tensor| tensor.grad)
            .and_then(|grad_id| store.get(grad_id))
            .is_some_and(|grad| grad.data.iter().any(|value| value.abs() > 1e-7))
    });
    assert!(
        any_grad,
        "expected non-zero gradient from stepwise pipeline"
    );
}

use autograd::{Tape, TensorStore};
use train::{
    dataset::LcgRng,
    grpo::{GrpoConfig, group_advantages, grpo_loss},
    multi_turn::{Environment, Episode, TurnSpec, rollout_episode},
    qwen3::{Qwen3Config, Qwen3Model},
};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

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

fn tiny_qwen3_config() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 64,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        intermediate_size: 128,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
    }
}

#[test]
fn multi_turn_qwen3_rollout_and_grpo_smoke() -> TestResult {
    let cfg = tiny_qwen3_config();
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Qwen3Model::new(&cfg, &mut store)?;
    let ref_model = policy.clone_frozen(&mut store);

    let initial_prompt = vec![1usize, 2, 3, 63];
    let turns = [
        TurnSpec {
            agent_tokens: 2,
            observation_tokens: 1,
        },
        TurnSpec {
            agent_tokens: 2,
            observation_tokens: 0,
        },
    ];
    let env = EchoSeparator(63);

    let prompt_len = initial_prompt.len();
    let verifier = |episode: &Episode| -> f32 {
        let mut hits = 0.0_f32;
        let mut total = 0.0_f32;
        for (position, masked) in episode.response_mask.iter().enumerate() {
            if !*masked {
                continue;
            }
            total += 1.0;
            if episode.full_ids[position] == episode.initial_prompt[position % prompt_len] {
                hits += 1.0;
            }
        }
        if total == 0.0 { 0.0 } else { hits / total }
    };

    let mut trajectories = Vec::with_capacity(2);
    for seed in 0..2_u64 {
        let mut rng = LcgRng::seed(100 + seed);
        let episode = rollout_episode(
            &policy,
            &ref_model,
            &initial_prompt,
            &turns,
            &env,
            1.0,
            &mut rng,
            &verifier,
            &mut store,
            &mut tape,
        )?;
        assert_eq!(episode.turn_boundaries, vec![(4, 6), (7, 9)]);
        assert_eq!(episode.full_ids[6], 63);
        trajectories.push(episode.into_trajectory());
    }

    let rewards: Vec<f32> = trajectories
        .iter()
        .map(|trajectory| trajectory.reward)
        .collect();
    let advantages = group_advantages(&rewards, 2);

    tape.entries.clear();
    tape.set_enabled(true);
    let loss = grpo_loss(
        &policy,
        &trajectories,
        &advantages,
        &GrpoConfig {
            clip_eps: 0.2,
            kl_coef: 0.02,
            group_size: 2,
        },
        &cfg,
        &mut store,
        &mut tape,
    )?;
    let loss_value = store.to_host(loss)?[0];
    assert!(loss_value.is_finite());

    let grads = tape.backward(loss, &mut store)?;
    let any_grad = policy.param_name_map().into_values().any(|param_id| {
        grads.get(&param_id).is_some_and(|grad_id| {
            store
                .get(*grad_id)
                .is_some_and(|tensor| tensor.data.iter().any(|value| value.abs() > 1.0e-7))
        })
    });
    assert!(
        any_grad,
        "expected qwen3 multi-turn grpo to produce gradients"
    );

    Ok(())
}

use autograd::{Result, Tape, TensorId, TensorStore};
use train::{
    dataset::LcgRng,
    grpo::{GrpoConfig, group_advantages, grpo_loss},
    multi_turn::{Environment, Episode, TurnSpec, rollout_episode},
    policy::{GrpoPolicy, GrpoPolicyConfig},
    qwen35::{LayerType, Qwen35Config, Qwen35Model},
};

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

#[derive(Clone, Copy)]
struct MockPolicyConfig {
    vocab_size: usize,
    max_seq_len: usize,
}

impl GrpoPolicyConfig for MockPolicyConfig {
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[derive(Clone)]
struct MockPolicy {
    config: MockPolicyConfig,
}

impl GrpoPolicy for MockPolicy {
    type Config = MockPolicyConfig;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn forward_single(
        &self,
        input_ids: &[usize],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        self.forward_batch_tokens(input_ids, 1, input_ids.len(), store, tape)
    }

    fn forward_batch_tokens(
        &self,
        input_ids: &[usize],
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        _tape: &mut Tape,
    ) -> Result<TensorId> {
        assert_eq!(input_ids.len(), batch * seq_len);
        let vocab = self.config.vocab_size;
        let mut logits = vec![0.0f32; batch * seq_len * vocab];
        for (index, token) in input_ids.iter().copied().enumerate() {
            let base = index * vocab;
            let token = token % vocab;
            for value in 0..vocab {
                logits[base + value] = if value == token { 8.0 } else { -8.0 };
            }
        }
        store.from_slice(&logits, &[batch, seq_len, vocab])
    }

    fn all_parameter_ids(&self) -> Vec<TensorId> {
        Vec::new()
    }

    fn clone_frozen(&self, _store: &mut TensorStore) -> Self {
        self.clone()
    }
}

fn tiny_qwen35_config() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        vocab_size: 16,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![15],
        bos_token_id: Some(1),
        eos_token_id: 15,
        tie_word_embeddings: false,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 8,
        linear_num_key_heads: 2,
        linear_key_head_dim: 8,
        linear_num_value_heads: 2,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        rotary_dim: 8,
        rope_cache_len_hint: Some(32),
        layer_types: vec![LayerType::FullAttention; 2],
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
    }
}

fn tiny_hybrid_qwen35_config() -> Qwen35Config {
    let mut cfg = tiny_qwen35_config();
    cfg.partial_rotary_factor = 0.5;
    cfg.rotary_dim = cfg.head_dim / 2;
    cfg.linear_key_head_dim = cfg.rotary_dim;
    cfg.linear_value_head_dim = cfg.rotary_dim;
    cfg.layer_types = vec![LayerType::FullAttention, LayerType::LinearAttention];
    cfg.validate_train_scratch_contract()
        .expect("hybrid scratch config");
    cfg
}

#[test]
fn rollout_episode_shapes_and_masks() {
    let config = MockPolicyConfig {
        vocab_size: 16,
        max_seq_len: 32,
    };
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = MockPolicy { config };
    let ref_model = policy.clone();

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
fn rollout_episode_uses_generic_grpo_policy_path() {
    let config = MockPolicyConfig {
        vocab_size: 16,
        max_seq_len: 32,
    };
    let policy = MockPolicy { config };
    let ref_model = policy.clone();
    let mut store = TensorStore::default();
    let mut tape = Tape::new();

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
    let mut rng = LcgRng::seed(11);
    let env = EchoSeparator(15);
    let episode = rollout_episode(
        &policy,
        &ref_model,
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
    assert_eq!(episode.turn_boundaries, vec![(4, 7), (9, 12)]);
    assert_eq!(episode.full_ids[7..9], [15, 15]);

    for (position, masked) in episode.response_mask.iter().enumerate() {
        if *masked {
            assert!(episode.old_log_probs[position].is_finite());
            assert!(
                (episode.old_log_probs[position] - episode.ref_log_probs[position]).abs() < 1e-6,
                "position {position} diverged"
            );
        }
    }
}

#[test]
fn episode_trajectory_feeds_grpo_loss() {
    let config = tiny_qwen35_config();
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Qwen35Model::new(&config, &mut store).expect("policy");
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
    let params = policy.all_parameter_ids();
    let any_grad = params.iter().any(|id| {
        store
            .get(*id)
            .and_then(|t| t.grad)
            .and_then(|g| store.get(g))
            .is_some_and(|g| g.data.iter().any(|v| v.abs() > 1e-7))
    });
    assert!(any_grad, "expected non-zero gradient on at least one param");
}

#[test]
fn hybrid_episode_trajectory_feeds_grpo_loss() {
    let config = tiny_hybrid_qwen35_config();
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Qwen35Model::new(&config, &mut store).expect("hybrid policy");
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
        let mut rng_i = LcgRng::seed(seed + 300);
        let episode = rollout_episode(
            &policy,
            &ref_model,
            &initial_prompt,
            &turns,
            &env,
            1.0,
            &mut rng_i,
            &verifier,
            &mut store,
            &mut tape,
        )
        .expect("hybrid episode");
        trajectories.push(episode.into_trajectory());
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
    .expect("hybrid grpo loss");
    let loss_value = store.to_host(loss).expect("loss host")[0];
    assert!(
        loss_value.is_finite(),
        "hybrid loss should be finite: {loss_value}"
    );

    tape.backward(loss, &mut store).expect("backward");
    let params = policy.all_parameter_ids();
    let any_grad = params.iter().any(|id| {
        store
            .get(*id)
            .and_then(|t| t.grad)
            .and_then(|g| store.get(g))
            .is_some_and(|g| g.data.iter().any(|v| v.abs() > 1e-7))
    });
    assert!(
        any_grad,
        "expected non-zero gradient on at least one hybrid param"
    );
}

#[test]
fn multi_turn_gspo_uses_sequence_level_episode_scores() {
    let config = tiny_qwen35_config();
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Qwen35Model::new(&config, &mut store).expect("policy");
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
    let mut rng = LcgRng::seed(29);

    let mut trajectories = Vec::with_capacity(4);
    let mut sequence_scores = Vec::with_capacity(4);
    for seed in 0..4 {
        let mut rng_i = LcgRng::seed(seed + 200);
        let episode = rollout_episode(
            &policy,
            &ref_model,
            &initial_prompt,
            &turns,
            &env,
            1.0,
            &mut rng_i,
            &|_: &Episode| 0.0,
            &mut store,
            &mut tape,
        )
        .expect("episode");
        let sequence_score = episode_sequence_score(&episode, &initial_prompt);
        sequence_scores.push(sequence_score);
        trajectories.push(episode.into_trajectory());
        rng.next_u64();
    }

    let advantages = group_advantages(&sequence_scores, 4);

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
    .expect("gspo loss");
    let loss_value = store.to_host(loss).expect("loss host")[0];
    assert!(
        loss_value.is_finite(),
        "loss should be finite: {loss_value}"
    );

    tape.backward(loss, &mut store).expect("backward");
    let params = policy.all_parameter_ids();
    let any_grad = params.iter().any(|id| {
        store
            .get(*id)
            .and_then(|t| t.grad)
            .and_then(|g| store.get(g))
            .is_some_and(|g| g.data.iter().any(|v| v.abs() > 1e-7))
    });
    assert!(any_grad, "expected non-zero gradient on at least one param");
}

#[test]
fn hybrid_multi_turn_gspo_uses_sequence_level_episode_scores() {
    let config = tiny_hybrid_qwen35_config();
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Qwen35Model::new(&config, &mut store).expect("hybrid policy");
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

    let mut trajectories = Vec::with_capacity(4);
    let mut sequence_scores = Vec::with_capacity(4);
    for seed in 0..4 {
        let mut rng_i = LcgRng::seed(seed + 400);
        let episode = rollout_episode(
            &policy,
            &ref_model,
            &initial_prompt,
            &turns,
            &env,
            1.0,
            &mut rng_i,
            &|_: &Episode| 0.0,
            &mut store,
            &mut tape,
        )
        .expect("hybrid episode");
        let sequence_score = episode_sequence_score(&episode, &initial_prompt);
        sequence_scores.push(sequence_score);
        trajectories.push(episode.into_trajectory());
    }

    let advantages = group_advantages(&sequence_scores, 4);

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
    .expect("hybrid gspo loss");
    let loss_value = store.to_host(loss).expect("loss host")[0];
    assert!(
        loss_value.is_finite(),
        "hybrid loss should be finite: {loss_value}"
    );

    tape.backward(loss, &mut store).expect("backward");
    let params = policy.all_parameter_ids();
    let any_grad = params.iter().any(|id| {
        store
            .get(*id)
            .and_then(|t| t.grad)
            .and_then(|g| store.get(g))
            .is_some_and(|g| g.data.iter().any(|v| v.abs() > 1e-7))
    });
    assert!(
        any_grad,
        "expected non-zero gradient on at least one hybrid param"
    );
}

fn episode_sequence_score(episode: &Episode, initial_prompt: &[usize]) -> f32 {
    let prompt_len = initial_prompt.len().max(1);
    let mut score = 0.0_f32;
    let mut turns = 0.0_f32;
    for (turn_idx, (start, end)) in episode.turn_boundaries.iter().enumerate() {
        let target = initial_prompt[turn_idx % prompt_len];
        let mut hits = 0.0_f32;
        let mut total = 0.0_f32;
        for position in *start..*end {
            total += 1.0;
            if episode.full_ids[position] == target {
                hits += 1.0;
            }
        }
        score += if total == 0.0 { 0.0 } else { hits / total };
        turns += 1.0;
    }
    if turns == 0.0 { 0.0 } else { score / turns }
}

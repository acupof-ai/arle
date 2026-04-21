use std::time::Instant;

use autograd::{Tape, TensorStore, optim::AdamW};
use train::{
    dataset::{CopyDataset, Dataset, LcgRng},
    grpo::{GrpoConfig, group_advantages, grpo_loss, grpo_loss_per_position, ppo_active_mask},
    policy_support::retained_ids,
    qwen35::Qwen35Model,
    rollout::{Trajectory, rollout_group},
    trainer::{clip_grad_norm, cross_entropy_loss},
};

mod common;

use common::qwen35_test_support::{tiny_hybrid_qwen35_scratch_config, tiny_qwen35_scratch_config};

#[test]
fn ppo_active_mask_zeros_out_clipped_positions() {
    // ratio = exp(new_lp - old_lp); eps = 0.2 → active window [0.8, 1.2]
    let new_log_probs = vec![
        0.0,            // ratio = 1.0 (inside)
        (1.5_f32).ln(), // ratio = 1.5 (above upper)
        (0.5_f32).ln(), // ratio = 0.5 (below lower)
        0.0,            // ratio = 1.0 (inside) but response_mask = 0
        (1.5_f32).ln(), // ratio = 1.5, adv < 0 → NOT clipped (helpful direction)
        (0.5_f32).ln(), // ratio = 0.5, adv < 0 → clipped (hurts)
    ];
    let old_log_probs = vec![0.0; 6];
    let advantages = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0];
    let response_mask = vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0];

    let mask = ppo_active_mask(
        &new_log_probs,
        &old_log_probs,
        &advantages,
        &response_mask,
        0.2,
    );

    assert_eq!(mask, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn group_advantages_normalizes_per_group() {
    let rewards = vec![1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0];
    let advantages = group_advantages(&rewards, 4);
    let expected = [
        -1.341_640_7,
        -0.447_213_6,
        0.447_213_6,
        1.341_640_7,
        0.0,
        0.0,
        0.0,
        0.0,
    ];

    for (actual, expected) in advantages.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1.0e-3);
    }
}

#[test]
fn grpo_loss_gradient_non_zero() {
    let config = tiny_qwen35_scratch_config(8);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Qwen35Model::new(&config, &mut store).expect("build policy");
    let ref_model = policy.clone_frozen(&mut store);
    let params = policy.all_parameter_ids();
    {
        let tensor = store
            .get_mut(params[0])
            .expect("first parameter should remain mutable");
        for value in tensor.data.iter_mut().take(8) {
            *value += 0.01;
        }
    }

    let mut rng = LcgRng::seed(13);
    let prompts = build_prompt_batch(2, 4, 8, 15, &mut rng);
    let trajectories = rollout_group(
        &policy,
        &ref_model,
        &config,
        &prompts,
        2,
        1.0,
        &mut rng,
        &copy_reward,
        &mut store,
        &mut tape,
    )
    .expect("rollout");
    let rewards = trajectories
        .iter()
        .map(|trajectory| trajectory.reward)
        .collect::<Vec<_>>();
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
        &config,
        &mut store,
        &mut tape,
    )
    .expect("grpo loss");
    let loss_value = store.to_host(loss).expect("loss value")[0];
    assert!(
        loss_value.is_finite(),
        "loss should be finite, got {loss_value}"
    );

    tape.backward(loss, &mut store).expect("backward");
    let has_non_zero_grad = params.iter().any(|&param_id| {
        store
            .get(param_id)
            .and_then(|tensor| tensor.grad)
            .and_then(|grad_id| store.get(grad_id))
            .is_some_and(|grad| grad.data.iter().any(|value| value.abs() > 1.0e-7))
    });
    assert!(has_non_zero_grad, "expected a non-zero GRPO gradient");
}

#[test]
fn hybrid_grpo_loss_gradient_non_zero() {
    let config = tiny_hybrid_qwen35_scratch_config(8);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Qwen35Model::new(&config, &mut store).expect("build hybrid policy");
    let ref_model = policy.clone_frozen(&mut store);
    let params = policy.all_parameter_ids();
    {
        let tensor = store
            .get_mut(params[0])
            .expect("first parameter should remain mutable");
        for value in tensor.data.iter_mut().take(8) {
            *value += 0.01;
        }
    }

    let mut rng = LcgRng::seed(113);
    let prompts = build_prompt_batch(2, 4, 8, 15, &mut rng);
    let trajectories = rollout_group(
        &policy,
        &ref_model,
        &config,
        &prompts,
        2,
        1.0,
        &mut rng,
        &copy_reward,
        &mut store,
        &mut tape,
    )
    .expect("hybrid rollout");
    let rewards = trajectories
        .iter()
        .map(|trajectory| trajectory.reward)
        .collect::<Vec<_>>();
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
        &config,
        &mut store,
        &mut tape,
    )
    .expect("hybrid grpo loss");
    let loss_value = store.to_host(loss).expect("loss value")[0];
    assert!(
        loss_value.is_finite(),
        "hybrid loss should be finite, got {loss_value}"
    );

    tape.backward(loss, &mut store).expect("backward");
    let has_non_zero_grad = params.iter().any(|&param_id| {
        store
            .get(param_id)
            .and_then(|tensor| tensor.grad)
            .and_then(|grad_id| store.get(grad_id))
            .is_some_and(|grad| grad.data.iter().any(|value| value.abs() > 1.0e-7))
    });
    assert!(
        has_non_zero_grad,
        "expected a non-zero hybrid GRPO gradient"
    );
}

#[test]
fn gspo_scalar_advantages_broadcast_like_per_position_inputs() {
    let config = tiny_qwen35_scratch_config(8);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Qwen35Model::new(&config, &mut store).expect("build policy");
    let ref_model = policy.clone_frozen(&mut store);

    let mut rng = LcgRng::seed(31);
    let prompts = build_prompt_batch(2, 4, 8, 15, &mut rng);
    let trajectories = rollout_group(
        &policy,
        &ref_model,
        &config,
        &prompts,
        2,
        1.0,
        &mut rng,
        &copy_reward,
        &mut store,
        &mut tape,
    )
    .expect("rollout");

    let rewards = trajectories
        .iter()
        .map(|trajectory| trajectory.reward)
        .collect::<Vec<_>>();
    let scalar_advantages = group_advantages(&rewards, 2);
    let seq_len = trajectories[0].full_ids.len();
    let mut per_position_advantages = Vec::with_capacity(scalar_advantages.len() * seq_len);
    for advantage in &scalar_advantages {
        per_position_advantages.extend(std::iter::repeat_n(*advantage, seq_len));
    }

    tape.entries.clear();
    tape.set_enabled(true);
    let scalar_loss = grpo_loss(
        &policy,
        &trajectories,
        &scalar_advantages,
        &GrpoConfig {
            clip_eps: 0.2,
            kl_coef: 0.02,
            group_size: 2,
        },
        &config,
        &mut store,
        &mut tape,
    )
    .expect("scalar grpo loss");
    let scalar_loss_value = store.to_host(scalar_loss).expect("scalar loss")[0];

    tape.entries.clear();
    tape.set_enabled(true);
    let broadcast_loss = grpo_loss_per_position(
        &policy,
        &trajectories,
        &per_position_advantages,
        &GrpoConfig {
            clip_eps: 0.2,
            kl_coef: 0.02,
            group_size: 2,
        },
        &config,
        &mut store,
        &mut tape,
    )
    .expect("broadcast grpo loss");
    let broadcast_loss_value = store.to_host(broadcast_loss).expect("broadcast loss")[0];

    assert!(
        (scalar_loss_value - broadcast_loss_value).abs() < 1.0e-6,
        "scalar={scalar_loss_value} broadcast={broadcast_loss_value}"
    );
}

#[test]
fn grpo_smoke_reward_nondecreasing_on_trivial_task() {
    let started = Instant::now();
    let config = tiny_qwen35_scratch_config(8);

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = Qwen35Model::new(&config, &mut store).expect("build policy");
    let params = policy.all_parameter_ids();
    let mut optimizer = AdamW::new(1.0e-2, (0.9, 0.999), 1.0e-8, 0.0);
    let mut dataset = CopyDataset::with_vocab(2, 8, 7, 8, 15);

    for _ in 0..10 {
        let (inputs, targets) = dataset.sample();
        let (batch, seq_len) = dataset.batch_shape();

        tape.entries.clear();
        tape.set_enabled(true);
        let logits = policy
            .forward_batch_tokens(&inputs, batch, seq_len, &mut store, &mut tape)
            .expect("forward");
        let loss = cross_entropy_loss(logits, &targets, &mut store, &mut tape).expect("loss");

        optimizer.zero_grad(&params, &mut store);
        tape.backward(loss, &mut store).expect("backward");
        clip_grad_norm(&params, 1.0, &mut store);
        optimizer.step(&params, &mut store);

        let keep = retained_ids(&[&policy], &store);
        store.retain_ids(&keep);
    }

    let ref_model = policy.clone_frozen(&mut store);
    let grpo_cfg = GrpoConfig {
        clip_eps: 0.2,
        kl_coef: 0.02,
        group_size: 2,
    };
    let mut rng = LcgRng::seed(23);
    let mut rewards = Vec::with_capacity(10);

    for _ in 0..10 {
        let prompts = build_prompt_batch(2, 8, 8, 15, &mut rng);
        let trajectories = rollout_group(
            &policy,
            &ref_model,
            &config,
            &prompts,
            2,
            0.0,
            &mut rng,
            &copy_reward,
            &mut store,
            &mut tape,
        )
        .expect("rollout");
        let rollout_rewards = trajectories
            .iter()
            .map(|trajectory| trajectory.reward)
            .collect::<Vec<_>>();
        rewards.push(mean_reward(&trajectories));
        let advantages = group_advantages(&rollout_rewards, 2);

        tape.entries.clear();
        tape.set_enabled(true);
        let loss = grpo_loss(
            &policy,
            &trajectories,
            &advantages,
            &grpo_cfg,
            &config,
            &mut store,
            &mut tape,
        )
        .expect("grpo loss");

        optimizer.zero_grad(&params, &mut store);
        tape.backward(loss, &mut store).expect("backward");
        clip_grad_norm(&params, 1.0, &mut store);
        optimizer.step(&params, &mut store);

        let keep = retained_ids(&[&policy, &ref_model], &store);
        store.retain_ids(&keep);
    }

    assert!(
        rewards[9] >= rewards[0] - 0.05,
        "expected stable or improving reward, got {rewards:?}",
    );
    assert!(
        started.elapsed().as_secs_f32() < 30.0,
        "smoke test took too long: {:?}",
        started.elapsed(),
    );
}

fn build_prompt_batch(
    batch: usize,
    seq_len: usize,
    token_upper: usize,
    separator: usize,
    rng: &mut LcgRng,
) -> Vec<Vec<usize>> {
    let prefix_len = seq_len / 2;
    let mut prompts = Vec::with_capacity(batch);
    for _ in 0..batch {
        let mut prompt = vec![0usize; seq_len];
        for token in &mut prompt[..prefix_len] {
            *token = (rng.next_u64() % token_upper as u64) as usize;
        }
        prompt[prefix_len] = separator;
        prompts.push(prompt);
    }
    prompts
}

fn copy_reward(prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]) -> f32 {
    let prefix_len = prompt_ids.len() / 2;
    let mut correct = 0usize;
    let mut total = 0usize;
    for (position, masked) in response_mask.iter().enumerate() {
        if !*masked {
            continue;
        }
        total += 1;
        if full_ids[position] == prompt_ids[position - (prefix_len + 1)] {
            correct += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32
    }
}

fn mean_reward(trajectories: &[Trajectory]) -> f32 {
    trajectories
        .iter()
        .map(|trajectory| trajectory.reward)
        .sum::<f32>()
        / trajectories.len() as f32
}

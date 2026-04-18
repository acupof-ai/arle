use std::{collections::HashSet, env, str::FromStr};

use autograd::{AutogradError, Tape, TensorId, TensorStore, module::Module, optim::AdamW};
use thiserror::Error;
use train::{
    dataset::{CopyDataset, Dataset, LcgRng},
    grpo::{GrpoConfig, group_advantages, grpo_loss, mean_sampled_kl},
    lora::LoraConfig,
    model::{TinyLM, TinyLMConfig},
    rollout::rollout_group,
    trainer::{clip_grad_norm, cross_entropy_loss},
};

const GRAD_CLIP_NORM: f32 = 1.0;

#[derive(Debug, Clone)]
struct CliArgs {
    sft_steps: usize,
    grpo_iters: usize,
    batch_prompts: usize,
    group_size: usize,
    seq: usize,
    lr: f32,
    kl_coef: f32,
    temperature: f32,
    lora_rank: usize,
    lora_alpha: f32,
    seed: u64,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            sft_steps: 30,
            grpo_iters: 20,
            batch_prompts: 4,
            group_size: 4,
            seq: 16,
            lr: 1.0e-4,
            kl_coef: 0.02,
            temperature: 1.0,
            lora_rank: 0,
            lora_alpha: 0.0,
            seed: 42,
        }
    }
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error("unknown flag {0}")]
    UnknownFlag(String),
    #[error("missing value for flag {0}")]
    MissingValue(String),
    #[error("invalid value for {flag}: {value}")]
    InvalidValue { flag: String, value: String },
}

fn main() -> Result<(), CliError> {
    let args = parse_args()?;
    validate_args(&args)?;

    let mut config = TinyLMConfig {
        max_seq_len: args.seq,
        ..TinyLMConfig::default()
    };
    if args.lora_rank > 0 {
        let alpha = if args.lora_alpha > 0.0 {
            args.lora_alpha
        } else {
            (2 * args.lora_rank) as f32
        };
        config.lora = Some(LoraConfig {
            rank: args.lora_rank,
            alpha,
        });
    }

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let policy = TinyLM::new(config, &mut store)?;
    let params = policy.parameters();
    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let mut dataset = CopyDataset::with_vocab(args.batch_prompts, args.seq, args.seed, 64, 255);
    let mut prompt_rng = LcgRng::seed(args.seed ^ 0x4752_504F_5052_4F4D);
    let verifier = |prompt_ids: &[usize], full_ids: &[usize], response_mask: &[bool]| {
        copy_reward(prompt_ids, full_ids, response_mask)
    };

    for step in 0..args.sft_steps {
        let (inputs, targets) = dataset.sample();
        let (batch, seq_len) = dataset.batch_shape();

        tape.entries.clear();
        tape.set_enabled(true);

        let logits = policy.forward(&inputs, batch, seq_len, &mut store, &mut tape)?;
        let logits_data = store.to_host(logits)?;
        let loss = cross_entropy_loss(logits, &targets, &mut store, &mut tape)?;
        let loss_value = store.to_host(loss)?[0];
        let reward =
            teacher_forced_reward(&logits_data, &targets, batch, seq_len, config.vocab_size);

        optimizer.zero_grad(&params, &mut store);
        tape.backward(loss, &mut store)?;
        clip_grad_norm(&params, GRAD_CLIP_NORM, &mut store);
        optimizer.step(&params, &mut store);

        tape.entries.clear();
        tape.set_enabled(true);
        let keep = retained_ids(&[&policy], &store);
        store.retain_ids(&keep);

        println!("sft step {step}: loss {loss_value:.4} reward {reward:.4}");
    }

    let ref_model = policy.clone_frozen(&mut store);
    let baseline_prompts =
        build_prompt_batch(args.batch_prompts, args.seq, 64, 255, &mut prompt_rng);
    let baseline_trajectories = rollout_group(
        &policy,
        &ref_model,
        &config,
        &baseline_prompts,
        args.group_size,
        args.temperature,
        &mut prompt_rng,
        &verifier,
        &mut store,
        &mut tape,
    )?;
    let baseline_reward = mean_reward(&baseline_trajectories);
    println!("baseline reward after sft: {baseline_reward:.4}");

    let grpo_cfg = GrpoConfig {
        clip_eps: 0.2,
        kl_coef: args.kl_coef,
        group_size: args.group_size,
    };
    let mut reward_trajectory = Vec::with_capacity(args.grpo_iters);
    let mut last_kl = 0.0_f32;
    let mut best_mean_reward = baseline_reward;

    for iter in 0..args.grpo_iters {
        let prompts = build_prompt_batch(args.batch_prompts, args.seq, 64, 255, &mut prompt_rng);
        let trajectories = rollout_group(
            &policy,
            &ref_model,
            &config,
            &prompts,
            args.group_size,
            args.temperature,
            &mut prompt_rng,
            &verifier,
            &mut store,
            &mut tape,
        )?;
        let rewards = trajectories
            .iter()
            .map(|trajectory| trajectory.reward)
            .collect::<Vec<_>>();
        let mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let advantages = group_advantages(&rewards, args.group_size);

        tape.entries.clear();
        tape.set_enabled(true);
        let loss_id = grpo_loss(
            &policy,
            &trajectories,
            &advantages,
            &grpo_cfg,
            &config,
            &mut store,
            &mut tape,
        )?;
        let loss_value = store.to_host(loss_id)?[0];

        optimizer.zero_grad(&params, &mut store);
        tape.backward(loss_id, &mut store)?;
        clip_grad_norm(&params, GRAD_CLIP_NORM, &mut store);
        optimizer.step(&params, &mut store);

        tape.entries.clear();
        tape.set_enabled(true);
        last_kl = mean_sampled_kl(&policy, &trajectories, &config, &mut store, &mut tape)?;
        let keep = retained_ids(&[&policy, &ref_model], &store);
        store.retain_ids(&keep);

        reward_trajectory.push(mean_reward);
        best_mean_reward = best_mean_reward.max(mean_reward);
        println!(
            "grpo iter {iter}: loss {loss_value:.4} mean_reward {mean_reward:.4} best_mean_reward {best_mean_reward:.4} mean_kl {last_kl:.4}"
        );
    }

    println!("final kl {last_kl:.4}");
    println!("reward trajectory: {reward_trajectory:?}");
    Ok(())
}

fn parse_args() -> Result<CliArgs, CliError> {
    let mut args = CliArgs::default();
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--sft-steps" => args.sft_steps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--grpo-iters" => {
                args.grpo_iters = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--batch-prompts" => {
                args.batch_prompts = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--group-size" => {
                args.group_size = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--seq" => args.seq = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--lr" => args.lr = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--kl-coef" => args.kl_coef = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--temperature" => {
                args.temperature = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--lora-rank" => {
                args.lora_rank = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--lora-alpha" => {
                args.lora_alpha = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--seed" => args.seed = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            _ => return Err(CliError::UnknownFlag(flag)),
        }
    }
    Ok(args)
}

fn validate_args(args: &CliArgs) -> Result<(), CliError> {
    if args.seq < 4 || !args.seq.is_multiple_of(2) {
        return Err(CliError::Autograd(AutogradError::InvalidRank {
            expected: "even sequence length >= 4",
            got: args.seq,
        }));
    }
    if args.group_size == 0 || args.batch_prompts == 0 {
        return Err(CliError::Autograd(AutogradError::InvalidRank {
            expected: "positive batch and group sizes",
            got: args.group_size.min(args.batch_prompts),
        }));
    }
    Ok(())
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, CliError> {
    iter.next()
        .ok_or_else(|| CliError::MissingValue(flag.to_string()))
}

fn parse_value<T>(flag: &str, value: String) -> Result<T, CliError>
where
    T: FromStr,
{
    value.parse::<T>().map_err(|_| CliError::InvalidValue {
        flag: flag.to_string(),
        value,
    })
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

fn teacher_forced_reward(
    logits: &[f32],
    targets: &[usize],
    batch: usize,
    seq_len: usize,
    vocab: usize,
) -> f32 {
    let prefix_len = seq_len / 2;
    let mut correct = 0usize;
    let mut total = 0usize;
    for row in 0..batch {
        for position in prefix_len + 1..seq_len {
            let logits_row = row * seq_len + (position - 1);
            let logits_base = logits_row * vocab;
            let prediction = argmax(&logits[logits_base..logits_base + vocab]);
            if prediction == targets[row * seq_len + (position - 1)] {
                correct += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best_index = index;
        }
    }
    best_index
}

fn mean_reward(trajectories: &[train::rollout::Trajectory]) -> f32 {
    if trajectories.is_empty() {
        0.0
    } else {
        trajectories
            .iter()
            .map(|trajectory| trajectory.reward)
            .sum::<f32>()
            / trajectories.len() as f32
    }
}

fn retained_ids(models: &[&TinyLM], store: &TensorStore) -> HashSet<TensorId> {
    let mut keep = HashSet::new();
    for model in models {
        for param_id in model.all_parameter_ids() {
            keep.insert(param_id);
            if let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) {
                keep.insert(grad_id);
            }
        }
    }
    keep
}

//! Multi-turn GRPO trainer: rollout_episode -> discounted per-turn returns
//! -> cross-episode group-normalize per turn -> per-position advantages ->
//! grpo_loss_per_position -> AdamW. Mirrors `train_grpo` but on interleaved
//! agent/observation episodes instead of suffix-only rollouts.

use std::{collections::HashSet, env, str::FromStr, sync::Arc};

use autograd::{
    AutogradError, Backend, CpuBackend, Tape, TensorId, TensorStore, module::Module, optim::AdamW,
};
use thiserror::Error;
use train::{
    control::TrainingController,
    dataset::LcgRng,
    grpo::{GrpoConfig, grpo_loss_per_position, mean_sampled_kl},
    lora::LoraConfig,
    model::{Transformer, TransformerConfig},
    multi_turn::{Environment, Episode, TurnSpec, rollout_episode},
    reward::{discounted_returns, group_normalize, returns_to_per_position},
    server::bind_and_serve_on_thread,
    trainer::clip_grad_norm,
};

const GRAD_CLIP_NORM: f32 = 1.0;

#[derive(Debug, Clone)]
struct CliArgs {
    iters: usize,
    group_size: usize,
    agent_tokens: usize,
    obs_tokens: usize,
    turns: usize,
    prompt_len: usize,
    lr: f32,
    kl_coef: f32,
    clip_eps: f32,
    temperature: f32,
    gamma: f32,
    lora_rank: usize,
    lora_alpha: f32,
    seed: u64,
    vocab_size: usize,
    target_range: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    d_head: usize,
    d_ff: usize,
    eval_every: usize,
    eval_prompts: usize,
    eval_temperature: f32,
    backend: BackendChoice,
    save_path: Option<String>,
    serve: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Cpu,
    Metal,
    Cuda,
}

impl FromStr for BackendChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cpu" => Ok(BackendChoice::Cpu),
            "metal" => Ok(BackendChoice::Metal),
            "cuda" => Ok(BackendChoice::Cuda),
            _ => Err(format!("unknown backend: {s}")),
        }
    }
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            iters: 20,
            group_size: 4,
            agent_tokens: 2,
            obs_tokens: 2,
            turns: 2,
            prompt_len: 4,
            lr: 1.0e-4,
            kl_coef: 0.02,
            clip_eps: 0.2,
            temperature: 1.0,
            gamma: 0.9,
            lora_rank: 0,
            lora_alpha: 0.0,
            seed: 42,
            vocab_size: 32,
            target_range: 8,
            d_model: 64,
            n_layers: 2,
            n_heads: 2,
            d_head: 32,
            d_ff: 128,
            eval_every: 0,
            eval_prompts: 16,
            eval_temperature: 0.3,
            backend: BackendChoice::Cpu,
            save_path: None,
            serve: None,
        }
    }
}

fn build_backend(choice: BackendChoice) -> Result<Arc<dyn Backend>, CliError> {
    match choice {
        BackendChoice::Cpu => Ok(Arc::new(CpuBackend)),
        #[cfg(feature = "metal")]
        BackendChoice::Metal => Ok(Arc::new(autograd::backend_metal::MetalBackend)),
        #[cfg(not(feature = "metal"))]
        BackendChoice::Metal => Err(CliError::InvalidValue {
            flag: "--backend".into(),
            value: "metal (build with --features metal)".into(),
        }),
        #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
        BackendChoice::Cuda => {
            let backend =
                autograd::backend_cuda::CudaBackend::new(0).map_err(|e| CliError::Autograd(e))?;
            Ok(Arc::new(backend))
        }
        #[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
        BackendChoice::Cuda => Err(CliError::InvalidValue {
            flag: "--backend".into(),
            value: "cuda (build with --features cuda and no no-cuda)".into(),
        }),
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
    #[error("{0}")]
    Custom(String),
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

fn main() -> Result<(), CliError> {
    let args = parse_args()?;
    validate_args(&args)?;

    let total_agent = args.agent_tokens * args.turns;
    let total_obs = args.obs_tokens * (args.turns.saturating_sub(1));
    let seq_len = args.prompt_len + total_agent + total_obs;

    let mut config = TransformerConfig {
        vocab_size: args.vocab_size,
        d_model: args.d_model,
        n_layers: args.n_layers,
        n_heads: args.n_heads,
        d_head: args.d_head,
        d_ff: args.d_ff,
        max_seq_len: seq_len.max(32),
        lora: None,
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

    let backend = build_backend(args.backend)?;
    eprintln!("[train_multi_turn] backend={:?}", backend.device());
    let mut store = TensorStore::with_backend(backend);
    let mut tape = Tape::new();
    let policy = Transformer::new(config, &mut store)?;
    let params = policy.parameters();
    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let ref_model = policy.clone_frozen(&mut store);
    let mut prompt_rng = LcgRng::seed(args.seed ^ 0x4D55_4C54_5455_524E);
    let mut sample_rng = LcgRng::seed(args.seed.wrapping_add(17));
    let eval_prompt_seed = args.seed ^ 0x4556_414C_5F50_524D;
    let eval_sample_seed = args.seed ^ 0x4556_414C_5350_4C52;

    let separator = (config.vocab_size - 1).min(31);
    let env = EchoSeparator(separator);
    let target_range = args
        .target_range
        .min(config.vocab_size.saturating_sub(2).max(1));
    let mut turns = Vec::with_capacity(args.turns);
    for turn_idx in 0..args.turns {
        let obs = if turn_idx + 1 < args.turns {
            args.obs_tokens
        } else {
            0
        };
        turns.push(TurnSpec {
            agent_tokens: args.agent_tokens,
            observation_tokens: obs,
        });
    }

    let grpo_cfg = GrpoConfig {
        clip_eps: args.clip_eps,
        kl_coef: args.kl_coef,
        group_size: args.group_size,
    };

    let mut reward_trajectory = Vec::with_capacity(args.iters);
    let mut best_reward = 0.0_f32;
    let mut last_kl = 0.0_f32;
    let loop_start = std::time::Instant::now();

    let controller = TrainingController::new();
    controller.update(|s| {
        s.total_iters = args.iters;
        s.started = true;
    });
    let _server_handle = if let Some(port) = args.serve {
        let addr = format!("127.0.0.1:{port}");
        eprintln!("[train_multi_turn] control plane listening on {addr}");
        Some(
            bind_and_serve_on_thread(Arc::clone(&controller), addr)
                .map_err(|e| CliError::Custom(format!("train server bind failed: {e}")))?,
        )
    } else {
        None
    };

    let mut stopped_early = false;
    for iter in 0..args.iters {
        if controller.should_stop() {
            eprintln!("[train_multi_turn] stop requested at iter {iter}");
            stopped_early = true;
            break;
        }
        let initial_prompt =
            build_prompt(args.prompt_len, separator, target_range, &mut prompt_rng);
        let mut episodes = Vec::with_capacity(args.group_size);
        for _ in 0..args.group_size {
            let episode = rollout_episode(
                &policy,
                &ref_model,
                &config,
                &initial_prompt,
                &turns,
                &env,
                args.temperature,
                &mut sample_rng,
                &|_: &Episode| 0.0,
                &mut store,
                &mut tape,
            )?;
            episodes.push(episode);
        }

        let mut per_turn_rewards: Vec<Vec<f32>> = Vec::with_capacity(args.group_size);
        for episode in &episodes {
            per_turn_rewards.push(compute_per_turn_rewards(episode, &initial_prompt));
        }

        let advantages_per_position = stepwise_advantages(
            &episodes,
            &per_turn_rewards,
            args.gamma,
            args.group_size,
            seq_len,
        );

        let trajectories: Vec<_> = episodes
            .iter()
            .map(|e| e.clone().into_trajectory())
            .collect();

        tape.entries.clear();
        tape.set_enabled(true);
        let loss_id = grpo_loss_per_position(
            &policy,
            &trajectories,
            &advantages_per_position,
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

        let mean_turn_reward = mean_per_turn(&per_turn_rewards);
        reward_trajectory.push(mean_turn_reward);
        best_reward = best_reward.max(mean_turn_reward);
        println!(
            "mt iter {iter}: loss {loss_value:.4} mean_reward {mean_turn_reward:.4} \
             best_reward {best_reward:.4} mean_kl {last_kl:.4}"
        );

        let wall_so_far = loop_start.elapsed().as_secs_f32();
        controller.update(|s| {
            s.iter = iter + 1;
            s.mean_reward = mean_turn_reward;
            s.best_reward = best_reward;
            s.last_kl = last_kl;
            s.last_loss = loss_value;
            s.wall_secs = wall_so_far;
        });

        if controller.take_save_request() {
            if let Some(path) = &args.save_path {
                train::checkpoint::save(&policy, &config, &store, path)
                    .map_err(|e| CliError::Custom(format!("checkpoint save failed: {e}")))?;
                eprintln!("[train_multi_turn] save requested → flushed to {path}");
            } else {
                eprintln!(
                    "[train_multi_turn] save requested but no --save-path configured; ignoring"
                );
            }
        }

        if args.eval_every > 0 && (iter + 1).is_multiple_of(args.eval_every) {
            let mut eval_prompt_rng = LcgRng::seed(eval_prompt_seed);
            let mut eval_sample_rng = LcgRng::seed(eval_sample_seed ^ iter as u64);
            let (eval_reward, eval_passrate) = run_eval(
                &policy,
                &ref_model,
                &config,
                args.eval_prompts,
                args.prompt_len,
                separator,
                target_range,
                &turns,
                &env,
                args.eval_temperature,
                &mut eval_prompt_rng,
                &mut eval_sample_rng,
                &mut store,
                &mut tape,
            )?;
            let keep = retained_ids(&[&policy, &ref_model], &store);
            store.retain_ids(&keep);
            println!(
                "eval @ iter {iter}: mean_reward {eval_reward:.4} pass@1 {eval_passrate:.4} \
                 (prompts={}, temperature={:.2})",
                args.eval_prompts, args.eval_temperature
            );
        }
    }

    let wall_secs = loop_start.elapsed().as_secs_f32();
    let total_episodes = args.iters * args.group_size;
    let tokens_per_episode = seq_len;
    let total_tokens = total_episodes * tokens_per_episode;
    let iter_per_sec = args.iters as f32 / wall_secs.max(1e-6);
    let episodes_per_sec = total_episodes as f32 / wall_secs.max(1e-6);
    let tokens_per_sec = total_tokens as f32 / wall_secs.max(1e-6);
    println!("final kl {last_kl:.4}");
    println!("reward trajectory: {reward_trajectory:?}");
    println!(
        "bench: wall {wall_secs:.2}s | iter/s {iter_per_sec:.2} | episode/s {episodes_per_sec:.2} \
         | token/s {tokens_per_sec:.1} | seq_len {seq_len} | group {group}",
        group = args.group_size,
    );

    if let Some(path) = &args.save_path {
        train::checkpoint::save(&policy, &config, &store, path)
            .map_err(|e| CliError::Custom(format!("checkpoint save failed: {e}")))?;
        println!("checkpoint saved to {path}");
    }

    controller.update(|s| {
        s.wall_secs = wall_secs;
        s.finished = true;
    });
    if stopped_early {
        eprintln!(
            "[train_multi_turn] training stopped early at iter {}",
            controller.snapshot().iter
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_eval(
    policy: &Transformer,
    ref_model: &Transformer,
    config: &TransformerConfig,
    n_prompts: usize,
    prompt_len: usize,
    separator: usize,
    target_range: usize,
    turns: &[TurnSpec],
    env: &EchoSeparator,
    temperature: f32,
    prompt_rng: &mut LcgRng,
    sample_rng: &mut LcgRng,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<(f32, f32), CliError> {
    if n_prompts == 0 {
        return Ok((0.0, 0.0));
    }
    let mut total_reward = 0.0_f32;
    let mut total_turns = 0.0_f32;
    let mut pass_count = 0.0_f32;
    for _ in 0..n_prompts {
        let initial_prompt = build_prompt(prompt_len, separator, target_range, prompt_rng);
        let episode = rollout_episode(
            policy,
            ref_model,
            config,
            &initial_prompt,
            turns,
            env,
            temperature,
            sample_rng,
            &|_: &Episode| 0.0,
            store,
            tape,
        )?;
        let per_turn = compute_per_turn_rewards(&episode, &initial_prompt);
        let n_turns = per_turn.len();
        if n_turns == 0 {
            continue;
        }
        let episode_mean: f32 = per_turn.iter().sum::<f32>() / n_turns as f32;
        total_reward += episode_mean * n_turns as f32;
        total_turns += n_turns as f32;
        if episode_mean >= 1.0 - 1.0e-4 {
            pass_count += 1.0;
        }
    }
    let mean_reward = if total_turns > 0.0 {
        total_reward / total_turns
    } else {
        0.0
    };
    let pass_rate = pass_count / n_prompts as f32;
    Ok((mean_reward, pass_rate))
}

fn compute_per_turn_rewards(episode: &Episode, initial_prompt: &[usize]) -> Vec<f32> {
    // Turn-t reward: fraction of agent tokens that copy `initial_prompt[t % prompt_len]`.
    let prompt_len = initial_prompt.len();
    let mut rewards = Vec::with_capacity(episode.turn_boundaries.len());
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
        rewards.push(if total == 0.0 { 0.0 } else { hits / total });
    }
    rewards
}

fn stepwise_advantages(
    episodes: &[Episode],
    per_turn_rewards: &[Vec<f32>],
    gamma: f32,
    group: usize,
    seq_len: usize,
) -> Vec<f32> {
    let n_turns = episodes[0].turn_boundaries.len();
    let mut returns_per_ep: Vec<Vec<f32>> = Vec::with_capacity(episodes.len());
    for rewards in per_turn_rewards {
        returns_per_ep.push(discounted_returns(rewards, gamma));
    }

    // Interleave so same-turn-across-episodes forms a normalization group.
    let mut stacked = Vec::with_capacity(group * n_turns);
    for turn_idx in 0..n_turns {
        for ep_returns in &returns_per_ep {
            stacked.push(ep_returns[turn_idx]);
        }
    }
    let normalized = group_normalize(&stacked, group);

    let mut per_position = Vec::with_capacity(group * seq_len);
    for (ep, episode) in episodes.iter().enumerate() {
        let mut row_adv = vec![0.0_f32; n_turns];
        for (turn_idx, slot) in row_adv.iter_mut().enumerate() {
            *slot = normalized[turn_idx * group + ep];
        }
        let row = returns_to_per_position(&row_adv, &episode.turn_boundaries, seq_len);
        per_position.extend_from_slice(&row);
    }
    per_position
}

fn mean_per_turn(per_turn_rewards: &[Vec<f32>]) -> f32 {
    let mut sum = 0.0_f32;
    let mut count = 0.0_f32;
    for row in per_turn_rewards {
        for reward in row {
            sum += *reward;
            count += 1.0;
        }
    }
    if count == 0.0 { 0.0 } else { sum / count }
}

fn build_prompt(
    prompt_len: usize,
    separator: usize,
    target_range: usize,
    rng: &mut LcgRng,
) -> Vec<usize> {
    assert!(prompt_len >= 2, "prompt length must be ≥ 2");
    let span = target_range.max(1) as u64;
    let mut prompt = vec![0usize; prompt_len];
    for slot in &mut prompt[..prompt_len - 1] {
        *slot = 1 + (rng.next_u64() % span) as usize;
    }
    prompt[prompt_len - 1] = separator;
    prompt
}

fn parse_args() -> Result<CliArgs, CliError> {
    let mut args = CliArgs::default();
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--iters" => args.iters = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--group-size" => {
                args.group_size = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--agent-tokens" => {
                args.agent_tokens = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--obs-tokens" => {
                args.obs_tokens = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--turns" => args.turns = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--prompt-len" => {
                args.prompt_len = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--lr" => args.lr = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--kl-coef" => args.kl_coef = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--clip-eps" => args.clip_eps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--temperature" => {
                args.temperature = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--gamma" => args.gamma = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--lora-rank" => {
                args.lora_rank = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--lora-alpha" => {
                args.lora_alpha = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--seed" => args.seed = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--vocab" => {
                args.vocab_size = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--target-range" => {
                args.target_range = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--d-model" => args.d_model = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--n-layers" => args.n_layers = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--n-heads" => args.n_heads = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--d-head" => args.d_head = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--d-ff" => args.d_ff = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--eval-every" => {
                args.eval_every = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--eval-prompts" => {
                args.eval_prompts = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--eval-temperature" => {
                args.eval_temperature = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--backend" => {
                let value = next_value(&mut iter, &flag)?;
                args.backend = value.parse().map_err(|_| CliError::InvalidValue {
                    flag: flag.clone(),
                    value,
                })?;
            }
            "--save-path" => {
                args.save_path = Some(next_value(&mut iter, &flag)?);
            }
            "--serve" => {
                args.serve = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            _ => return Err(CliError::UnknownFlag(flag)),
        }
    }
    Ok(args)
}

fn validate_args(args: &CliArgs) -> Result<(), CliError> {
    if args.turns == 0 || args.group_size == 0 || args.agent_tokens == 0 {
        return Err(CliError::Autograd(AutogradError::InvalidRank {
            expected: "positive turns, group_size, agent_tokens",
            got: 0,
        }));
    }
    if args.prompt_len < 2 {
        return Err(CliError::Autograd(AutogradError::InvalidRank {
            expected: "prompt_len >= 2",
            got: args.prompt_len,
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

fn retained_ids(models: &[&Transformer], store: &TensorStore) -> HashSet<TensorId> {
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

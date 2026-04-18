//! Standalone inference binary for TinyLM checkpoints.
//!
//! Loads a checkpoint produced by `train_multi_turn --save-path …`,
//! runs greedy (or temperature-sampled) autoregressive decoding on a
//! prompt of comma-separated token ids, and prints the extended
//! sequence. Exists as the inference companion to the training binary
//! so the RL loop is end-to-end usable: train → save → load → generate.

use std::env;
use std::str::FromStr;
use std::sync::Arc;

use autograd::{AutogradError, Backend, CpuBackend, Tape, TensorStore};
use thiserror::Error;
use train::dataset::LcgRng;
use train::model::TinyLM;
use train::sampling::sample_categorical;

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Checkpoint(#[from] train::checkpoint::CheckpointError),
    #[error("{0}")]
    Custom(String),
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

struct CliArgs {
    load_path: String,
    prompt: Vec<usize>,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
    backend: BackendChoice,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            load_path: String::new(),
            prompt: vec![0, 1, 2, 15],
            max_new_tokens: 16,
            temperature: 0.0,
            seed: 1,
            backend: BackendChoice::Cpu,
        }
    }
}

fn build_backend(choice: BackendChoice) -> Result<Arc<dyn Backend>, CliError> {
    match choice {
        BackendChoice::Cpu => Ok(Arc::new(CpuBackend)),
        #[cfg(feature = "metal")]
        BackendChoice::Metal => Ok(Arc::new(autograd::backend_metal::MetalBackend)),
        #[cfg(not(feature = "metal"))]
        BackendChoice::Metal => Err(CliError::Custom(
            "metal backend requires --features metal".into(),
        )),
        #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
        BackendChoice::Cuda => Ok(Arc::new(autograd::backend_cuda::CudaBackend::new(0)?)),
        #[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
        BackendChoice::Cuda => Err(CliError::Custom(
            "cuda backend requires --features cuda without no-cuda".into(),
        )),
    }
}

fn parse_args() -> Result<CliArgs, CliError> {
    let mut args = CliArgs::default();
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--load-path" => args.load_path = next(&mut iter, &flag)?,
            "--prompt" => {
                let raw = next(&mut iter, &flag)?;
                args.prompt = raw
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| CliError::Custom(format!("--prompt parse: {e}")))?;
            }
            "--max-new-tokens" => args.max_new_tokens = parse(&flag, next(&mut iter, &flag)?)?,
            "--temperature" => args.temperature = parse(&flag, next(&mut iter, &flag)?)?,
            "--seed" => args.seed = parse(&flag, next(&mut iter, &flag)?)?,
            "--backend" => {
                let raw = next(&mut iter, &flag)?;
                args.backend = raw.parse().map_err(CliError::Custom)?;
            }
            _ => return Err(CliError::Custom(format!("unknown flag: {flag}"))),
        }
    }
    if args.load_path.is_empty() {
        return Err(CliError::Custom("--load-path is required".into()));
    }
    Ok(args)
}

fn next(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, CliError> {
    iter.next()
        .ok_or_else(|| CliError::Custom(format!("missing value for {flag}")))
}

fn parse<T: FromStr>(flag: &str, value: String) -> Result<T, CliError> {
    value
        .parse::<T>()
        .map_err(|_| CliError::Custom(format!("invalid value for {flag}: {value}")))
}

fn main() -> Result<(), CliError> {
    let args = parse_args()?;
    let backend = build_backend(args.backend)?;
    let mut store = TensorStore::with_backend(backend);

    let config = train::checkpoint::read_config(&args.load_path)?;
    println!(
        "checkpoint config: vocab={} d_model={} n_layers={} n_heads={} d_head={} d_ff={} max_seq_len={}",
        config.vocab_size,
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.d_head,
        config.d_ff,
        config.max_seq_len,
    );

    let policy = TinyLM::new(config, &mut store)?;
    train::checkpoint::load(&policy, &mut store, &args.load_path)?;
    println!("loaded checkpoint from {}", args.load_path);

    if args.prompt.iter().any(|t| *t >= config.vocab_size) {
        return Err(CliError::Custom(format!(
            "prompt contains ids outside vocab ({}): {:?}",
            config.vocab_size, args.prompt,
        )));
    }

    let mut tokens = args.prompt.clone();
    let mut rng = LcgRng::seed(args.seed);
    let mut tape = Tape::new();
    tape.set_enabled(false);

    for _ in 0..args.max_new_tokens {
        if tokens.len() >= config.max_seq_len {
            println!("reached max_seq_len={}, stopping", config.max_seq_len);
            break;
        }

        let seq_len = tokens.len();
        let logits = policy.forward(&tokens, 1, seq_len, &mut store, &mut tape)?;
        let host = store.to_host(logits)?;

        // Take the *last* position's logits — row major [B=1, S, V].
        let last_start = (seq_len - 1) * config.vocab_size;
        let last_logits = &host[last_start..last_start + config.vocab_size];
        let (sampled, _) = sample_categorical(
            last_logits,
            (1, 1),
            config.vocab_size,
            args.temperature,
            &mut rng,
        );
        tokens.push(sampled[0]);
    }

    println!("prompt:   {:?}", args.prompt);
    println!("sampled:  {:?}", &tokens[args.prompt.len()..]);
    println!("full:     {:?}", tokens);
    Ok(())
}

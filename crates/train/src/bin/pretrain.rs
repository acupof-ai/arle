use std::{collections::HashSet, env, fs, path::PathBuf, str::FromStr, sync::Arc};

use autograd::{
    AutogradError, Backend, CpuBackend, Tape, TensorId, TensorStore, module::Module, optim::AdamW,
};
use thiserror::Error;
use train::{
    dataset::{BytesDataset, CopyDataset, CorpusDataset, Dataset},
    lora::LoraConfig,
    model::{Transformer, TransformerConfig},
    tokenizer::ChatTokenizer,
    trainer::{clip_grad_norm, cross_entropy_loss},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetKind {
    Copy,
    Bytes,
    Corpus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Cpu,
    Metal,
    Cuda,
}

impl FromStr for BackendChoice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => Err(format!("unknown backend: {value}")),
        }
    }
}

#[derive(Debug, Clone)]
struct CliArgs {
    dataset: DatasetKind,
    steps: usize,
    batch: usize,
    seq: usize,
    lr: f32,
    log_every: usize,
    grad_clip: Option<f32>,
    lora_rank: usize,
    lora_alpha: f32,
    backend: BackendChoice,
    corpus: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    vocab_size: Option<usize>,
    d_model: Option<usize>,
    n_layers: Option<usize>,
    n_heads: Option<usize>,
    d_head: Option<usize>,
    d_ff: Option<usize>,
    max_seq_len: Option<usize>,
    seed: u64,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            dataset: DatasetKind::Bytes,
            steps: 50,
            batch: 4,
            seq: 64,
            lr: 3.0e-4,
            log_every: 10,
            grad_clip: Some(1.0),
            lora_rank: 0,
            lora_alpha: 0.0,
            backend: BackendChoice::Cpu,
            corpus: None,
            tokenizer: None,
            vocab_size: None,
            d_model: None,
            n_layers: None,
            n_heads: None,
            d_head: None,
            d_ff: None,
            max_seq_len: None,
            seed: 0xCAFEBABE,
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
    #[error("{0}")]
    Custom(String),
}

fn main() -> Result<(), CliError> {
    let args = parse_args()?;

    // Load tokenizer first — `--corpus` implies we tokenize the file now, and
    // the tokenizer's vocab size is the natural default for the model's
    // vocab_size when the user didn't override it.
    let tokenizer = match args.tokenizer.as_ref() {
        Some(path) => Some(ChatTokenizer::from_file(path)?),
        None => None,
    };

    let mut config = TransformerConfig::default();
    if let Some(tok) = tokenizer.as_ref() {
        config.vocab_size = tok.vocab_size();
    }
    if let Some(vocab) = args.vocab_size {
        config.vocab_size = vocab;
    }
    if let Some(d_model) = args.d_model {
        config.d_model = d_model;
    }
    if let Some(n_layers) = args.n_layers {
        config.n_layers = n_layers;
    }
    if let Some(n_heads) = args.n_heads {
        config.n_heads = n_heads;
    }
    if let Some(d_head) = args.d_head {
        config.d_head = d_head;
    }
    if let Some(d_ff) = args.d_ff {
        config.d_ff = d_ff;
    }
    if let Some(max_seq_len) = args.max_seq_len {
        config.max_seq_len = max_seq_len;
    }
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
    if args.seq > config.max_seq_len {
        return Err(CliError::Autograd(AutogradError::ShapeMismatch {
            expected: vec![config.max_seq_len],
            got: vec![args.seq],
        }));
    }

    let backend = build_backend(args.backend)?;
    println!("backend: {:?}", backend.device());
    println!(
        "config: vocab={} d_model={} n_layers={} n_heads={} d_head={} d_ff={} max_seq_len={}",
        config.vocab_size,
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.d_head,
        config.d_ff,
        config.max_seq_len,
    );
    let mut store = TensorStore::with_backend(backend);
    let mut tape = Tape::new();
    let model = Transformer::new(config, &mut store)?;
    let params = model.parameters();
    let base_params = model.base_parameter_ids();
    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1e-8, 0.01);
    let mut dataset = build_dataset(&args, tokenizer.as_ref(), config.vocab_size)?;

    if let Some(lora) = config.lora {
        let base_count = count_parameters(&base_params, &store);
        let trainable_count = count_parameters(&params, &store);
        println!(
            "base params: {:.2}M | trainable params: {:.2}M (LoRA rank={})",
            base_count as f64 / 1_000_000.0,
            trainable_count as f64 / 1_000_000.0,
            lora.rank
        );
    } else {
        let param_count = model.parameter_count(&store);
        println!(
            "params: {} ({:.2}M)",
            param_count,
            param_count as f64 / 1_000_000.0
        );
    }

    let mut final_loss = f32::INFINITY;
    for step in 0..args.steps {
        let (input_ids, target_ids) = dataset.sample();
        let (batch, seq_len) = dataset.batch_shape();

        tape.entries.clear();
        tape.set_enabled(true);

        let logits = model.forward(&input_ids, batch, seq_len, &mut store, &mut tape)?;
        let loss_id = cross_entropy_loss(logits, &target_ids, &mut store, &mut tape)?;
        final_loss = store.to_host(loss_id)?[0];

        optimizer.zero_grad(&params, &mut store);
        tape.backward(loss_id, &mut store)?;
        if let Some(max_norm) = args.grad_clip {
            clip_grad_norm(&params, max_norm, &mut store);
        }
        optimizer.step(&params, &mut store);

        tape.entries.clear();
        tape.set_enabled(true);
        let keep = retained_ids(&params, &base_params, &store);
        store.retain_ids(&keep);

        if step % args.log_every == 0 || step + 1 == args.steps {
            println!("step {step}: loss {final_loss:.4}");
        }
    }

    println!("final loss {final_loss:.4}");
    Ok(())
}

fn parse_args() -> Result<CliArgs, CliError> {
    let mut args = CliArgs::default();
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--dataset" => {
                let value = next_value(&mut iter, &flag)?;
                args.dataset = match value.as_str() {
                    "copy" => DatasetKind::Copy,
                    "bytes" => DatasetKind::Bytes,
                    "corpus" => DatasetKind::Corpus,
                    _ => {
                        return Err(CliError::InvalidValue { flag, value });
                    }
                };
            }
            "--corpus" => {
                args.corpus = Some(PathBuf::from(next_value(&mut iter, &flag)?));
                args.dataset = DatasetKind::Corpus;
            }
            "--tokenizer" => {
                args.tokenizer = Some(PathBuf::from(next_value(&mut iter, &flag)?));
            }
            "--vocab-size" => {
                args.vocab_size = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--d-model" => {
                args.d_model = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--n-layers" => {
                args.n_layers = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--n-heads" => {
                args.n_heads = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--d-head" => {
                args.d_head = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--d-ff" => {
                args.d_ff = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--max-seq-len" => {
                args.max_seq_len = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--seed" => args.seed = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--steps" => args.steps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--batch" => args.batch = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--seq" => args.seq = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--lr" => args.lr = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--log-every" => {
                args.log_every = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--lora-rank" => {
                args.lora_rank = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--lora-alpha" => {
                args.lora_alpha = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--grad-clip" => {
                args.grad_clip = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--no-grad-clip" => args.grad_clip = None,
            "--backend" => {
                args.backend = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            _ => return Err(CliError::UnknownFlag(flag)),
        }
    }

    Ok(args)
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

fn build_backend(choice: BackendChoice) -> Result<Arc<dyn Backend>, CliError> {
    match choice {
        BackendChoice::Cpu => Ok(Arc::new(CpuBackend)),
        #[cfg(feature = "metal")]
        BackendChoice::Metal => Ok(Arc::new(autograd::backend_metal::MetalBackend)),
        #[cfg(not(feature = "metal"))]
        BackendChoice::Metal => {
            eprintln!(
                "[pretrain] warning: metal backend requested without --features metal; falling back to cpu"
            );
            Ok(Arc::new(CpuBackend))
        }
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => Ok(Arc::new(autograd::backend_cuda::CudaBackend::new(0)?)),
        #[cfg(not(feature = "cuda"))]
        BackendChoice::Cuda => {
            eprintln!(
                "[pretrain] warning: cuda backend requested without --features cuda; falling back to cpu"
            );
            Ok(Arc::new(CpuBackend))
        }
    }
}

fn build_dataset(
    args: &CliArgs,
    tokenizer: Option<&ChatTokenizer>,
    vocab_size: usize,
) -> Result<Box<dyn Dataset>, CliError> {
    match args.dataset {
        DatasetKind::Copy => Ok(Box::new(CopyDataset::new(args.batch, args.seq))),
        DatasetKind::Bytes => {
            if vocab_size < 256 {
                return Err(CliError::Custom(format!(
                    "--dataset bytes emits byte ids 0..=255; --vocab-size {vocab_size} < 256 would overflow the embedding table"
                )));
            }
            Ok(Box::new(BytesDataset::new(args.batch, args.seq)))
        }
        DatasetKind::Corpus => {
            let path = args.corpus.as_ref().ok_or_else(|| {
                CliError::Custom("--dataset corpus requires --corpus <path>".into())
            })?;
            let tok = tokenizer.ok_or_else(|| {
                CliError::Custom("--dataset corpus requires --tokenizer <path>".into())
            })?;
            let text = fs::read_to_string(path).map_err(|e| {
                CliError::Custom(format!("failed to read corpus {}: {e}", path.display()))
            })?;
            let ids = tok.encode(&text, false)?;
            if ids.is_empty() {
                return Err(CliError::Custom(format!(
                    "corpus {} tokenized to 0 tokens",
                    path.display()
                )));
            }
            for &id in &ids {
                if (id as usize) >= vocab_size {
                    return Err(CliError::Custom(format!(
                        "token id {id} exceeds configured vocab_size {vocab_size}"
                    )));
                }
            }
            let tokens: Vec<usize> = ids.iter().map(|&id| id as usize).collect();
            if tokens.len() <= args.seq {
                return Err(CliError::Custom(format!(
                    "corpus has {} tokens but seq_len is {}; need more tokens",
                    tokens.len(),
                    args.seq
                )));
            }
            println!("corpus: {} tokens from {}", tokens.len(), path.display());
            Ok(Box::new(CorpusDataset::new(
                tokens, args.batch, args.seq, args.seed,
            )))
        }
    }
}

fn count_parameters(params: &[TensorId], store: &TensorStore) -> usize {
    params
        .iter()
        .map(|&id| store.get(id).map_or(0, |tensor| tensor.size))
        .sum()
}

fn retained_ids(
    params: &[TensorId],
    base_params: &[TensorId],
    store: &TensorStore,
) -> HashSet<TensorId> {
    let mut keep = HashSet::with_capacity((params.len() + base_params.len()) * 2);
    for &param_id in params.iter().chain(base_params.iter()) {
        keep.insert(param_id);
        if let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) {
            keep.insert(grad_id);
        }
    }
    keep
}

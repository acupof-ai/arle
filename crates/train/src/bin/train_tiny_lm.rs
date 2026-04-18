use std::{collections::HashSet, env, str::FromStr};

use autograd::{AutogradError, Tape, TensorId, TensorStore, module::Module, optim::AdamW};
use thiserror::Error;
use train::{
    dataset::{BytesDataset, CopyDataset, Dataset},
    model::{TinyLM, TinyLMConfig},
    trainer::{clip_grad_norm, cross_entropy_loss},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetKind {
    Copy,
    Bytes,
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
    let config = TinyLMConfig::default();
    if args.seq > config.max_seq_len {
        return Err(CliError::Autograd(AutogradError::ShapeMismatch {
            expected: vec![config.max_seq_len],
            got: vec![args.seq],
        }));
    }

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let model = TinyLM::new(config, &mut store)?;
    let params = model.parameters();
    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1e-8, 0.01);
    let mut dataset = build_dataset(args.dataset, args.batch, args.seq);

    let param_count = model.parameter_count(&store);
    println!(
        "params: {} ({:.2}M)",
        param_count,
        param_count as f64 / 1_000_000.0
    );

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
        let keep = retained_ids(&params, &store);
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
                    _ => {
                        return Err(CliError::InvalidValue { flag, value });
                    }
                };
            }
            "--steps" => args.steps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--batch" => args.batch = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--seq" => args.seq = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--lr" => args.lr = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--log-every" => {
                args.log_every = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--grad-clip" => {
                args.grad_clip = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--no-grad-clip" => args.grad_clip = None,
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

fn build_dataset(kind: DatasetKind, batch: usize, seq: usize) -> Box<dyn Dataset> {
    match kind {
        DatasetKind::Copy => Box::new(CopyDataset::new(batch, seq)),
        DatasetKind::Bytes => Box::new(BytesDataset::new(batch, seq)),
    }
}

fn retained_ids(params: &[TensorId], store: &TensorStore) -> HashSet<TensorId> {
    let mut keep = HashSet::with_capacity(params.len() * 2);
    for &param_id in params {
        keep.insert(param_id);
        if let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) {
            keep.insert(grad_id);
        }
    }
    keep
}

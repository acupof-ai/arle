use std::{collections::HashSet, env, fs, path::PathBuf, str::FromStr, sync::Arc};

use autograd::{
    AutogradError, Backend, ConstantLr, CpuBackend, Result as AutogradResult, Tape, TensorId,
    TensorStore, module::Module, optim::AdamW,
};
use thiserror::Error;
use train::{
    StepCtx, StepOutcome, Trainer, TrainerConfig,
    cli_args::{ArgError, next_value, parse_value},
    dataset::{BytesDataset, CopyDataset, CorpusDataset, Dataset},
    grad_clip::{GlobalNorm, GradClip, NoClip},
    lora::LoraConfig,
    loss::cross_entropy_loss,
    model::{Transformer, TransformerConfig},
    tokenizer::ChatTokenizer,
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
    metrics_jsonl: Option<PathBuf>,
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
            metrics_jsonl: None,
        }
    }
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Arg(#[from] ArgError),
    #[error("{0}")]
    Custom(String),
}

/// Wave 3 pretrain migration choice: `Trainer<O, C, S>` is generic on the
/// clip policy, so `--no-grad-clip` vs `--grad-clip N` needs to collapse
/// to a single concrete `C`. We keep `NoClip` + `GlobalNorm` as the
/// real impls and forward through this enum so we don't have to
/// monomorphise the Trainer twice.
enum PretrainClip {
    None(NoClip),
    Norm(GlobalNorm),
}

impl GradClip for PretrainClip {
    fn clip(&mut self, store: &mut TensorStore, params: &[TensorId]) -> AutogradResult<f32> {
        match self {
            Self::None(c) => c.clip(store, params),
            Self::Norm(c) => c.clip(store, params),
        }
    }
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

    // ---- Trainer setup ----
    // AdamW with the same defaults the hand-written loop used. `(0.9, 0.999)`
    // + 1e-8 + wd 0.01 is the historical default for this binary (verified
    // against the pre-migration commit) — flipping them would quietly change
    // every convergence baseline.
    let optim = AdamW::new(args.lr, (0.9, 0.999), 1e-8, 0.01);
    // Codex review 2026-04-20 on 6bd0211 (M1): pre-migration
    // `clip_grad_norm(params, max_norm <= 0.0, store)` returned early
    // as a no-op, so `--grad-clip 0` was legal and effectively disabled
    // clipping. The migrated path sends `max_norm` into
    // `GlobalNorm::new`, which asserts `max_norm > 0.0 && is_finite()`
    // — `--grad-clip 0` would panic there. Preserve the legacy
    // semantics by treating non-positive / non-finite values as NoClip
    // instead of panicking, matching the old behaviour bit-for-bit.
    let clip = match args.grad_clip {
        Some(max_norm) if max_norm > 0.0 && max_norm.is_finite() => {
            PretrainClip::Norm(GlobalNorm::new(max_norm))
        }
        Some(max_norm) => {
            eprintln!(
                "[pretrain] warning: --grad-clip {max_norm} is non-positive/non-finite; disabling gradient clipping"
            );
            PretrainClip::None(NoClip)
        }
        None => PretrainClip::None(NoClip),
    };
    let schedule = ConstantLr(args.lr);
    let metrics = train::metrics::open_sink(args.metrics_jsonl.as_deref(), true)
        .map_err(|e| CliError::Custom(format!("metrics sink: {e}")))?;

    // pretrain.rs did not support checkpoint save/resume in the hand-written
    // loop — the custom `Transformer` has no safetensors codec (unlike
    // `qwen3::Qwen3Model` in train_sft). Leaving both fields `None` preserves
    // that prior semantics. Follow-up work: add a `TransformerRegistry`
    // mirroring `SafetensorsRegistry` before wiring `save_every` here.
    let trainer_cfg = TrainerConfig {
        total_steps: args.steps as u64,
        // Each `dataset.sample()` already produces a full batch; the
        // hand-written loop ran one optimizer step per sample, so
        // `grad_accum_steps = 1` preserves the historical semantics.
        grad_accum_steps: 1,
        log_every: args.log_every.max(1) as u64,
        eval_every: None,
        save_every: None,
        save_dir: None,
        resume_from: None,
        rng_seed: args.seed,
    };
    let mut trainer = Trainer::new(optim, clip, schedule, metrics, trainer_cfg);

    // `param_name_map` — Transformer doesn't expose named parameters today;
    // optimizer state persistence isn't wired anyway (save_dir is None), so
    // we hand the Trainer synthetic stable names purely to satisfy the API.
    // When the safetensors codec lands for Transformer, replace this with
    // the real named-parameter map.
    let param_names: Vec<(TensorId, String)> = params
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, format!("param.{i:04}")))
        .collect();

    // keep_extra: base (frozen-or-not) parameter ids that aren't in the
    // trainable `params` list — LoRA leaves the base matmul weights alive
    // across backward passes, and the hand-written cleanup added them to the
    // retain set (`retained_ids`). Matches that behaviour exactly.
    let mut keep_extra: HashSet<TensorId> = HashSet::new();
    for &id in &base_params {
        keep_extra.insert(id);
    }

    let batch_shape = dataset.batch_shape();
    let model_ref = &model;
    let dataset_ref: &mut dyn Dataset = dataset.as_mut();
    // `Cell`-free shim: the closure borrows `dataset` as `FnMut` so we can
    // just call `dataset.sample()` directly each invocation.
    let mut dataset_holder = Some(dataset_ref);
    let step_fn = |ctx: &mut StepCtx<'_>| -> AutogradResult<StepOutcome> {
        let dataset = dataset_holder.as_mut().expect("dataset holder populated");
        let (input_ids, target_ids) = dataset.sample();
        let (batch, seq_len) = batch_shape;
        let token_count = (batch * seq_len) as u64;

        let logits = model_ref.forward(&input_ids, batch, seq_len, ctx.store, ctx.tape)?;
        let loss_id = cross_entropy_loss(logits, &target_ids, ctx.store, ctx.tape)?;
        Ok(StepOutcome {
            loss_id,
            token_count,
        })
    };

    trainer.run(
        &mut store,
        &mut tape,
        params,
        param_names,
        keep_extra,
        step_fn,
    )?;
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
                        return Err(CliError::Arg(ArgError::InvalidValue { flag, value }));
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
            "--metrics-jsonl" => {
                args.metrics_jsonl = Some(PathBuf::from(next_value(&mut iter, &flag)?));
            }
            _ => return Err(CliError::Arg(ArgError::UnknownFlag(flag))),
        }
    }

    Ok(args)
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

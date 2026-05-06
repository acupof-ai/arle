// DSV4 nano pretrain driver.
//
// This command owns the train-side DeepSeek nano smoke path: random-init the
// dense MLA nano config, train it on a plain-text corpus with autograd, and
// save an infer-shaped checkpoint directory (`model.safetensors`,
// `config.json`, `generation_config.json`, `tokenizer.json`). SKU-A/B stay on
// the external cold-path pretrain track from the substrate plan.

use std::{fs, path::PathBuf, str::FromStr, sync::Arc};

use autograd::{
    AutogradError, ConstantLr, Result as AutogradResult, SafetensorsRegistry, Tape, TensorStore,
};
use deepseek_spec::DeepSeekConfig;
use thiserror::Error;

use crate::{
    StepCtx, StepOutcome, Trainer, TrainerConfig,
    causal_lm::{live_tensor_ids, trainable_param_name_map, trainable_params},
    checkpoint::publish_latest_after_weights,
    cli_args::{ArgError, BackendChoice, SaveDtype, adamw_for_backend, next_value, parse_value},
    dataset::LcgRng,
    deepseek::DeepseekNanoModel,
    grad_clip::GlobalNorm,
    metrics::NullSink,
    tokenizer::ChatTokenizer,
    trainer::cross_entropy_loss,
};

const DEFAULT_BETAS: (f32, f32) = (0.9, 0.95);
const DEFAULT_EPS: f32 = 1.0e-8;
const DEFAULT_WEIGHT_DECAY: f32 = 0.1;

/// DSV4 SKU selector. Only `nano` is wired in-tree; larger SKUs remain an
/// external cold-path pretrain concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepseekSku {
    Nano,
}

impl FromStr for DeepseekSku {
    type Err = DsV4PretrainError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input {
            "nano" => Ok(Self::Nano),
            other => Err(DsV4PretrainError::UnknownSku(other.to_string())),
        }
    }
}

impl DeepseekSku {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nano => "nano",
        }
    }

    pub fn build_config(self) -> DeepSeekConfig {
        match self {
            Self::Nano => DeepSeekConfig::nano(),
        }
    }
}

/// Parsed CLI for `arle train pretrain-dsv4`.
#[derive(Debug, Clone)]
pub struct CliArgs {
    pub sku: DeepseekSku,
    pub corpus: PathBuf,
    pub tokenizer: PathBuf,
    pub out: PathBuf,
    pub seed: u64,
    pub steps: usize,
    pub batch: usize,
    pub seq: usize,
    pub lr: f32,
    pub log_every: usize,
    pub save_every: usize,
    pub backend: BackendChoice,
    pub save_dtype: SaveDtype,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            sku: DeepseekSku::Nano,
            corpus: PathBuf::new(),
            tokenizer: PathBuf::new(),
            out: PathBuf::new(),
            seed: 0xC0FFEE,
            steps: 10,
            batch: 1,
            seq: 64,
            lr: 3.0e-4,
            log_every: 1,
            save_every: 10,
            backend: BackendChoice::Cpu,
            save_dtype: SaveDtype::Bf16,
        }
    }
}

#[derive(Debug, Error)]
pub enum DsV4PretrainError {
    #[error("missing required argument: {0}")]
    MissingArg(&'static str),
    #[error("unknown DSV4 SKU `{0}` — only `nano` is wired today")]
    UnknownSku(String),
    #[error("argument `{flag}` requires a value")]
    MissingValue { flag: String },
    #[error("argument `{flag}` value `{value}` is not a valid {kind}")]
    InvalidValue {
        flag: String,
        value: String,
        kind: &'static str,
    },
    #[error(transparent)]
    Arg(#[from] ArgError),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error("{0}")]
    Custom(String),
}

pub fn parse_args_from<I>(args: I) -> Result<CliArgs, DsV4PretrainError>
where
    I: IntoIterator<Item = String>,
{
    let mut args_out = CliArgs::default();
    let mut iter = args.into_iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--deepseek-config" => {
                let value = iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                    flag: arg.to_string(),
                })?;
                args_out.sku = value.parse()?;
            }
            "--corpus" => {
                args_out.corpus =
                    PathBuf::from(iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                        flag: arg.to_string(),
                    })?);
            }
            "--tokenizer" => {
                args_out.tokenizer =
                    PathBuf::from(iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                        flag: arg.to_string(),
                    })?);
            }
            "--out" => {
                args_out.out =
                    PathBuf::from(iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                        flag: arg.to_string(),
                    })?);
            }
            "--seed" => {
                let value = iter.next().ok_or_else(|| DsV4PretrainError::MissingValue {
                    flag: arg.to_string(),
                })?;
                args_out.seed =
                    value
                        .parse::<u64>()
                        .map_err(|_| DsV4PretrainError::InvalidValue {
                            flag: arg.to_string(),
                            value,
                            kind: "u64",
                        })?;
            }
            "--steps" => args_out.steps = parse_value(&arg, next_value(&mut iter, &arg)?)?,
            "--batch" => args_out.batch = parse_value(&arg, next_value(&mut iter, &arg)?)?,
            "--seq" => args_out.seq = parse_value(&arg, next_value(&mut iter, &arg)?)?,
            "--lr" => args_out.lr = parse_value(&arg, next_value(&mut iter, &arg)?)?,
            "--log-every" => args_out.log_every = parse_value(&arg, next_value(&mut iter, &arg)?)?,
            "--save-every" => {
                args_out.save_every = parse_value(&arg, next_value(&mut iter, &arg)?)?
            }
            "--backend" => {
                let value = next_value(&mut iter, &arg)?;
                args_out.backend = value
                    .parse()
                    .map_err(|_| ArgError::InvalidValue { flag: arg, value })?;
            }
            "--save-dtype" => {
                let value = next_value(&mut iter, &arg)?;
                args_out.save_dtype = value
                    .parse()
                    .map_err(|_| ArgError::InvalidValue { flag: arg, value })?;
            }
            _ => {}
        }
    }

    if args_out.corpus.as_os_str().is_empty() {
        return Err(DsV4PretrainError::MissingArg("--corpus"));
    }
    if args_out.tokenizer.as_os_str().is_empty() {
        return Err(DsV4PretrainError::MissingArg("--tokenizer"));
    }
    if args_out.out.as_os_str().is_empty() {
        return Err(DsV4PretrainError::MissingArg("--out"));
    }
    Ok(args_out)
}

/// Public CLI entry. Same shape as `pretrain::dispatch_from_args` so
/// `train_cli.rs::run_train_command` can route through the same harness.
pub fn dispatch_from_args<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let parsed = parse_args_from(args).map_err(|err| err.to_string())?;
    run(&parsed).map_err(|err| err.to_string())
}

fn run(args: &CliArgs) -> Result<(), DsV4PretrainError> {
    fs::create_dir_all(&args.out).map_err(|err| {
        DsV4PretrainError::Custom(format!("create output dir {}: {err}", args.out.display()))
    })?;

    let tokenizer = ChatTokenizer::from_file(&args.tokenizer)
        .map_err(|err| DsV4PretrainError::Custom(format!("tokenizer: {err}")))?;
    let cfg = args.sku.build_config();
    if args.seq > cfg.max_position_embeddings {
        return Err(DsV4PretrainError::Custom(format!(
            "--seq {} exceeds DeepSeek nano context {}",
            args.seq, cfg.max_position_embeddings
        )));
    }

    let text = fs::read_to_string(&args.corpus).map_err(|err| {
        DsV4PretrainError::Custom(format!("read corpus {}: {err}", args.corpus.display()))
    })?;
    let token_ids = tokenizer
        .encode(&text, false)
        .map_err(|err| DsV4PretrainError::Custom(format!("tokenize corpus: {err}")))?;
    if token_ids.len() <= args.seq {
        return Err(DsV4PretrainError::Custom(format!(
            "corpus has {} tokens but --seq is {}; need at least seq+1 tokens",
            token_ids.len(),
            args.seq
        )));
    }
    if token_ids.iter().any(|&id| id as usize >= cfg.vocab_size) {
        return Err(DsV4PretrainError::Custom(format!(
            "tokenizer produced ids outside DeepSeek nano vocab {}",
            cfg.vocab_size
        )));
    }

    let backend = args.backend.build_backend_or_cpu("pretrain-dsv4")?;
    let mut store = TensorStore::with_backend(Arc::clone(&backend));
    let mut tape = Tape::new();
    let model = DeepseekNanoModel::new(&cfg, &mut store)
        .map_err(|err| DsV4PretrainError::Custom(err.to_string()))?;
    let params = trainable_params(&model, &store);
    let param_names = trainable_param_name_map(&model, &store);
    let model_ids = live_tensor_ids(&store);
    let position_ids = (0..args.seq).collect::<Vec<_>>();
    let batch = args.batch.max(1);
    let window = args.seq + 1;
    let upper = token_ids.len() - window + 1;
    let mut rng = LcgRng::seed(args.seed);
    let mut train_input_ids = Vec::with_capacity(batch * args.seq);
    let mut train_targets = Vec::with_capacity(batch * args.seq);

    println!(
        "[pretrain-dsv4] sku={} backend={} steps={} batch={} seq={} lr={} params={:.2}M corpus_tokens={} out={}",
        args.sku.as_str(),
        args.backend.as_str(),
        args.steps,
        batch,
        args.seq,
        args.lr,
        params
            .iter()
            .map(|&id| store.get(id).map_or(0, |tensor| tensor.size))
            .sum::<usize>() as f64
            / 1_000_000.0,
        token_ids.len(),
        args.out.display(),
    );

    let optim = adamw_for_backend(
        args.lr,
        DEFAULT_BETAS,
        DEFAULT_EPS,
        DEFAULT_WEIGHT_DECAY,
        Arc::clone(&backend),
    );
    let mut trainer = Trainer::new(
        optim,
        GlobalNorm::new(1.0),
        ConstantLr(args.lr),
        Box::new(NullSink),
        TrainerConfig {
            total_steps: args.steps as u64,
            grad_accum_steps: 1,
            log_every: args.log_every.max(1) as u64,
            eval_every: None,
            save_every: Some(args.save_every.max(1) as u64),
            save_dir: Some(args.out.clone()),
            resume_from: None,
            rng_seed: args.seed,
        },
    );

    let step_fn = |ctx: &mut StepCtx<'_>| -> AutogradResult<StepOutcome> {
        train_input_ids.clear();
        train_targets.clear();
        for _ in 0..batch {
            let start = (rng.next_u64() % upper as u64) as usize;
            let slice = &token_ids[start..start + window];
            train_input_ids.extend(slice[..args.seq].iter().map(|&id| id as usize));
            train_targets.extend(slice[1..].iter().map(|&id| id as usize));
        }
        let logits = model
            .forward_batch_tokens_with_positions(
                &train_input_ids,
                &position_ids,
                batch,
                ctx.store,
                ctx.tape,
            )
            .map_err(|err| {
                AutogradError::TapeInvariant(Box::leak(err.to_string().into_boxed_str()))
            })?;
        let loss_id = cross_entropy_loss(logits, &train_targets, ctx.store, ctx.tape)?;
        Ok(StepOutcome {
            loss_id,
            token_count: (batch * args.seq) as u64,
        })
    };

    let out_dir = args.out.clone();
    let tokenizer_path = args.tokenizer.clone();
    let cfg_for_save = cfg.clone();
    let save_dtype = args.save_dtype;
    let on_step_end = |step: u64, store: &mut TensorStore| -> AutogradResult<()> {
        if step.is_multiple_of(args.save_every.max(1) as u64) || step == args.steps as u64 {
            save_checkpoint(
                &out_dir,
                step as usize,
                &model,
                store,
                &cfg_for_save,
                &tokenizer_path,
                save_dtype,
            )
            .map_err(|err| {
                AutogradError::TapeInvariant(Box::leak(err.to_string().into_boxed_str()))
            })?;
        }
        Ok(())
    };

    trainer.run_with_hooks(
        &mut store,
        &mut tape,
        params,
        param_names,
        model_ids,
        step_fn,
        on_step_end,
    )?;
    Ok(())
}

fn save_checkpoint(
    out_dir: &std::path::Path,
    step: usize,
    model: &DeepseekNanoModel,
    store: &mut TensorStore,
    cfg: &DeepSeekConfig,
    tokenizer_path: &std::path::Path,
    save_dtype: SaveDtype,
) -> Result<(), DsV4PretrainError> {
    let basename = format!("step_{step:06}");
    let step_dir = out_dir.join(&basename);
    fs::create_dir_all(&step_dir).map_err(|err| {
        DsV4PretrainError::Custom(format!(
            "create checkpoint dir {}: {err}",
            step_dir.display()
        ))
    })?;

    let weights_path = step_dir.join("model.safetensors");
    let mut registry = SafetensorsRegistry::new();
    for (name, tensor_id) in model.param_name_map() {
        registry.insert(name, tensor_id);
    }
    match save_dtype {
        SaveDtype::F32 => registry.save_from(store, &weights_path)?,
        SaveDtype::Bf16 => registry.save_from_bf16(store, &weights_path)?,
    }

    let config = serde_json::to_string_pretty(cfg)
        .map_err(|err| DsV4PretrainError::Custom(format!("serialize config: {err}")))?;
    fs::write(step_dir.join("config.json"), config).map_err(|err| {
        DsV4PretrainError::Custom(format!(
            "write config.json in {}: {err}",
            step_dir.display()
        ))
    })?;
    let generation_config = serde_json::json!({
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
    });
    fs::write(
        step_dir.join("generation_config.json"),
        serde_json::to_string_pretty(&generation_config).unwrap(),
    )
    .map_err(|err| {
        DsV4PretrainError::Custom(format!(
            "write generation_config.json in {}: {err}",
            step_dir.display()
        ))
    })?;
    fs::copy(tokenizer_path, step_dir.join("tokenizer.json")).map_err(|err| {
        DsV4PretrainError::Custom(format!(
            "copy tokenizer {} into {}: {err}",
            tokenizer_path.display(),
            step_dir.display()
        ))
    })?;
    publish_latest_after_weights(out_dir, &basename).map_err(|err| {
        DsV4PretrainError::Custom(format!("publish latest checkpoint {}: {err}", basename))
    })?;
    println!(
        "[pretrain-dsv4] saved step {} to {} (dtype: {:?})",
        step,
        step_dir.display(),
        save_dtype,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_minimal_invocation() {
        let parsed = parse_args_from(vec_of(&[
            "--deepseek-config",
            "nano",
            "--corpus",
            "corpus.txt",
            "--tokenizer",
            "tokenizer.json",
            "--out",
            "/tmp/dsv4-nano",
        ]))
        .unwrap();

        assert_eq!(parsed.sku, DeepseekSku::Nano);
        assert_eq!(parsed.corpus, PathBuf::from("corpus.txt"));
        assert_eq!(parsed.tokenizer, PathBuf::from("tokenizer.json"));
        assert_eq!(parsed.out, PathBuf::from("/tmp/dsv4-nano"));
        assert_eq!(parsed.steps, 10);
    }

    #[test]
    fn parses_training_knobs() {
        let parsed = parse_args_from(vec_of(&[
            "--corpus",
            "corpus.txt",
            "--tokenizer",
            "tokenizer.json",
            "--out",
            "/tmp/out",
            "--steps",
            "2",
            "--batch",
            "3",
            "--seq",
            "16",
            "--lr",
            "0.001",
            "--save-every",
            "1",
            "--backend",
            "cpu",
            "--save-dtype",
            "f32",
        ]))
        .unwrap();
        assert_eq!(parsed.steps, 2);
        assert_eq!(parsed.batch, 3);
        assert_eq!(parsed.seq, 16);
        assert_eq!(parsed.lr, 0.001);
        assert_eq!(parsed.save_every, 1);
        assert_eq!(parsed.backend, BackendChoice::Cpu);
        assert_eq!(parsed.save_dtype, SaveDtype::F32);
    }

    #[test]
    fn defaults_sku_to_nano() {
        let parsed = parse_args_from(vec_of(&[
            "--corpus",
            "corpus.txt",
            "--tokenizer",
            "tokenizer.json",
            "--out",
            "/tmp/out",
        ]))
        .unwrap();
        assert_eq!(parsed.sku, DeepseekSku::Nano);
    }

    #[test]
    fn rejects_unknown_sku() {
        let err = parse_args_from(vec_of(&[
            "--deepseek-config",
            "tiny-dense",
            "--corpus",
            "c",
            "--tokenizer",
            "t",
            "--out",
            "o",
        ]))
        .unwrap_err();
        match err {
            DsV4PretrainError::UnknownSku(name) => assert_eq!(name, "tiny-dense"),
            other => panic!("expected UnknownSku, got {other:?}"),
        }
    }

    #[test]
    fn requires_corpus() {
        let err = parse_args_from(vec_of(&["--tokenizer", "t", "--out", "o"])).unwrap_err();
        match err {
            DsV4PretrainError::MissingArg(name) => assert_eq!(name, "--corpus"),
            other => panic!("expected MissingArg(--corpus), got {other:?}"),
        }
    }
}

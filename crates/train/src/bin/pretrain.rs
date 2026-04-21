// Pretrain a generic Qwen-family model from random init on a plain-text
// corpus, saving checkpoints in the safetensors + config.json +
// tokenizer.json layout that infer/ can load. Mirrors train_sft.rs's save
// pipeline but starts from scratch (no source model dir) and drives a
// packed 1D forward over random corpus windows.
//
// Phase 3 (2026-04-20): migrated onto the generic `Trainer<O, C, S>` loop.
// The hand-written optimizer-step / clip / backward / cleanup sequence now
// lives in `train::Trainer`; this binary owns only the data sampler, the
// forward+loss closure, the eval closure, and the model-weight checkpoint
// save pipeline (wired via `on_step_end`). See
// `docs/plans/train-runtime-architecture-v1.md` for context.

use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Instant,
};

use autograd::{
    AutogradError, Backend, ConstantLr, CpuBackend, Result as AutogradResult, SafetensorsRegistry,
    Tape, TensorId, TensorStore, optim::AdamW,
};
use qwen3_spec::Qwen3Config;
use thiserror::Error;
use train::{
    EvalOutcome, StepCtx, StepOutcome, Trainer, TrainerConfig,
    causal_lm::{build_registry, live_tensor_ids, trainable_param_name_map, trainable_params},
    cli_args::{ArgError, next_value, parse_value},
    control::{
        TrainingController, emit_run_end, emit_run_start, open_run_metrics, serve_if_requested,
        sync_status,
    },
    dataset::LcgRng,
    grad_clip::{GlobalNorm, GradClip, NoClip},
    model_family::{ModelFamily, synthetic_qwen35_dense_config},
    qwen3::{Qwen3ConfigError, Qwen3Error, Qwen3Model},
    qwen3_checkpoint::{
        ConfigJsonSource, GenerationConfigSource, Qwen3CheckpointError, Qwen3StepCheckpoint,
        save_step_checkpoint,
    },
    qwen35::{Qwen35Config, Qwen35Error, Qwen35Model},
    qwen35_checkpoint::{
        ConfigJsonSource as Qwen35ConfigJsonSource,
        GenerationConfigSource as Qwen35GenerationConfigSource, Qwen35CheckpointError,
        Qwen35StepCheckpoint, save_step_checkpoint as save_qwen35_step_checkpoint,
    },
    tokenizer::ChatTokenizer,
    trainer::cross_entropy_loss,
};

// Qwen3 tokenizer defaults — infer/Config requires both fields.
const DEFAULT_BOS_TOKEN_ID: u32 = 151_643;
const DEFAULT_EOS_TOKEN_ID: u32 = 151_645;

const DEFAULT_BETAS: (f32, f32) = (0.9, 0.999);
const DEFAULT_EPS: f32 = 1.0e-8;
const DEFAULT_WEIGHT_DECAY: f32 = 0.01;
const STOP_REQUESTED_ERR: &str = "pretrain: operator stop requested";

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SaveDtype {
    F32,
    Bf16,
}

impl FromStr for SaveDtype {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "f32" => Ok(Self::F32),
            "bf16" => Ok(Self::Bf16),
            _ => Err(format!("unknown save dtype: {value}")),
        }
    }
}

#[derive(Debug, Clone)]
struct CliArgs {
    model_family: ModelFamily,
    corpus: PathBuf,
    tokenizer: PathBuf,
    out: PathBuf,
    steps: usize,
    batch: usize,
    seq: usize,
    lr: f32,
    log_every: usize,
    save_every: usize,
    eval_every: usize,
    eval_windows: usize,
    eval_frac: f32,
    resume_from: Option<PathBuf>,
    seed: u64,
    grad_clip: Option<f32>,
    backend: BackendChoice,
    save_dtype: SaveDtype,
    // Model hyperparams — vocab_size defaults to tokenizer.vocab_size().
    vocab_size: Option<usize>,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    tie_word_embeddings: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    metrics_jsonl: Option<PathBuf>,
    serve: Option<u16>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            model_family: ModelFamily::Qwen35,
            corpus: PathBuf::new(),
            tokenizer: PathBuf::new(),
            out: PathBuf::new(),
            steps: 100,
            batch: 1,
            seq: 128,
            lr: 3.0e-4,
            log_every: 5,
            save_every: 50,
            eval_every: 0,
            eval_windows: 8,
            eval_frac: 0.1,
            resume_from: None,
            seed: 0xC0FFEE,
            grad_clip: Some(1.0),
            backend: BackendChoice::Cpu,
            save_dtype: SaveDtype::Bf16,
            vocab_size: None,
            hidden_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            intermediate_size: 512,
            max_position_embeddings: 512,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            bos_token_id: DEFAULT_BOS_TOKEN_ID,
            eos_token_id: DEFAULT_EOS_TOKEN_ID,
            metrics_jsonl: None,
            serve: None,
        }
    }
}

/// Phase 3 migration choice: `Trainer<O, C, S>` is generic on the clip policy,
/// so `--no-grad-clip` vs `--grad-clip N` needs to collapse to a single
/// concrete `C`. We keep `NoClip` + `GlobalNorm` as the real impls and
/// forward through this enum so we don't have to monomorphise the Trainer
/// twice. Mirrors `pretrain.rs::PretrainClip` (kept inline since it's a
/// CLI-adapter type, not a library surface).
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

fn cli_error_to_autograd(err: CliError, family: &str, context: &str) -> AutogradError {
    match err {
        CliError::Autograd(inner) => inner,
        other => {
            eprintln!("[pretrain] {family} {context} error: {other}");
            AutogradError::TapeInvariant("pretrain: family callback returned non-autograd error")
        }
    }
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Qwen3Checkpoint(#[from] Qwen3CheckpointError),
    #[error(transparent)]
    Qwen3(#[from] Qwen3Error),
    #[error(transparent)]
    Qwen3Config(#[from] Qwen3ConfigError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Qwen35Checkpoint(#[from] Qwen35CheckpointError),
    #[error(transparent)]
    Qwen35(#[from] Qwen35Error),
    #[error(transparent)]
    Qwen35Config(#[from] qwen35_spec::Qwen35ConfigError),
    #[error(transparent)]
    Arg(#[from] ArgError),
    #[error("{0}")]
    Custom(String),
}

trait PretrainFamily {
    type Config: Clone;
    type Model: train::CausalLm<Config = Self::Config>;

    fn family_name() -> &'static str;
    fn build_config(args: &CliArgs, vocab_size: usize) -> Result<Self::Config, CliError>;
    fn build_model(cfg: &Self::Config, store: &mut TensorStore) -> Result<Self::Model, CliError>;
    fn max_seq_len(cfg: &Self::Config) -> usize;
    fn describe_config(cfg: &Self::Config) -> String;
    fn forward(
        model: &Self::Model,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> Result<TensorId, CliError>;
    fn validate_resume_config(resume_dir: &Path, cfg: &Self::Config) -> Result<(), CliError>;
    fn save_checkpoint(
        out_dir: &Path,
        step: usize,
        model: &Self::Model,
        store: &mut TensorStore,
        cfg: &Self::Config,
        tokenizer_path: &Path,
        bos_token_id: u32,
        eos_token_id: u32,
        save_dtype: SaveDtype,
    ) -> Result<(), CliError>;
}

struct Qwen3Family;

impl PretrainFamily for Qwen3Family {
    type Config = Qwen3Config;
    type Model = Qwen3Model;

    fn family_name() -> &'static str {
        "qwen3"
    }

    fn build_config(args: &CliArgs, vocab_size: usize) -> Result<Self::Config, CliError> {
        let cfg = Qwen3Config {
            vocab_size,
            hidden_size: args.hidden_size,
            num_hidden_layers: args.num_hidden_layers,
            num_attention_heads: args.num_attention_heads,
            num_key_value_heads: args.num_kv_heads,
            head_dim: args.head_dim,
            intermediate_size: args.intermediate_size,
            max_position_embeddings: args.max_position_embeddings,
            rms_norm_eps: args.rms_norm_eps,
            rope_theta: args.rope_theta,
            tie_word_embeddings: args.tie_word_embeddings,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn build_model(cfg: &Self::Config, store: &mut TensorStore) -> Result<Self::Model, CliError> {
        Qwen3Model::new(cfg, store).map_err(Into::into)
    }

    fn max_seq_len(cfg: &Self::Config) -> usize {
        cfg.max_position_embeddings
    }

    fn describe_config(cfg: &Self::Config) -> String {
        format!(
            "vocab={} hidden={} layers={} heads={} kv_heads={} head_dim={} ffn={} max_pos={} tie_embed={}",
            cfg.vocab_size,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.intermediate_size,
            cfg.max_position_embeddings,
            cfg.tie_word_embeddings,
        )
    }

    fn forward(
        model: &Self::Model,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> Result<TensorId, CliError> {
        model
            .forward(store, tape, input_ids, position_ids)
            .map_err(Into::into)
    }

    fn validate_resume_config(resume_dir: &Path, cfg: &Self::Config) -> Result<(), CliError> {
        validate_qwen3_resume_config(resume_dir, cfg)
    }

    fn save_checkpoint(
        out_dir: &Path,
        step: usize,
        model: &Self::Model,
        store: &mut TensorStore,
        cfg: &Self::Config,
        tokenizer_path: &Path,
        bos_token_id: u32,
        eos_token_id: u32,
        save_dtype: SaveDtype,
    ) -> Result<(), CliError> {
        save_qwen3_checkpoint(
            out_dir,
            step,
            model,
            store,
            cfg,
            tokenizer_path,
            bos_token_id,
            eos_token_id,
            save_dtype,
        )
    }
}

struct Qwen35Family;

impl PretrainFamily for Qwen35Family {
    type Config = Qwen35Config;
    type Model = Qwen35Model;

    fn family_name() -> &'static str {
        "qwen35"
    }

    fn build_config(args: &CliArgs, vocab_size: usize) -> Result<Self::Config, CliError> {
        let mut cfg = synthetic_qwen35_dense_config(args.max_position_embeddings);
        cfg.vocab_size = vocab_size;
        cfg.hidden_size = args.hidden_size;
        cfg.num_hidden_layers = args.num_hidden_layers;
        cfg.num_attention_heads = args.num_attention_heads;
        cfg.num_key_value_heads = args.num_kv_heads.max(1);
        cfg.head_dim = args.head_dim;
        cfg.intermediate_size = args.intermediate_size;
        cfg.rms_norm_eps = args.rms_norm_eps;
        cfg.rope_theta = args.rope_theta;
        cfg.tie_word_embeddings = args.tie_word_embeddings;
        cfg.stop_token_ids = vec![args.eos_token_id];
        cfg.bos_token_id = Some(args.bos_token_id);
        cfg.eos_token_id = args.eos_token_id;
        cfg.rope_cache_len_hint = Some(args.max_position_embeddings);
        cfg.layer_types = vec![qwen35_spec::LayerType::FullAttention; args.num_hidden_layers];
        cfg.linear_num_key_heads = args.num_attention_heads;
        cfg.linear_key_head_dim = args.head_dim;
        cfg.linear_num_value_heads = args.num_attention_heads;
        cfg.linear_value_head_dim = args.head_dim;
        cfg.linear_conv_kernel_dim = cfg.linear_conv_kernel_dim.max(4);
        cfg.partial_rotary_factor = 1.0;
        cfg.rotary_dim = args.head_dim;
        cfg.num_experts = 0;
        cfg.num_experts_per_tok = 0;
        cfg.moe_intermediate_size = 0;
        cfg.shared_expert_intermediate_size = 0;
        cfg.norm_topk_prob = true;
        cfg.mlp_only_layers = Vec::new();
        cfg.validate()?;
        Ok(cfg)
    }

    fn build_model(cfg: &Self::Config, store: &mut TensorStore) -> Result<Self::Model, CliError> {
        Qwen35Model::new(cfg, store).map_err(Into::into)
    }

    fn max_seq_len(cfg: &Self::Config) -> usize {
        cfg.rope_cache_len_hint.unwrap_or(cfg.rotary_dim.max(1))
    }

    fn describe_config(cfg: &Self::Config) -> String {
        format!(
            "vocab={} hidden={} layers={} heads={} kv_heads={} head_dim={} ffn={} max_pos={} tie_embed={}",
            cfg.vocab_size,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.intermediate_size,
            cfg.rope_cache_len_hint.unwrap_or_default(),
            cfg.tie_word_embeddings,
        )
    }

    fn forward(
        model: &Self::Model,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
    ) -> Result<TensorId, CliError> {
        model
            .forward(store, tape, input_ids, position_ids)
            .map_err(Into::into)
    }

    fn validate_resume_config(resume_dir: &Path, cfg: &Self::Config) -> Result<(), CliError> {
        validate_qwen35_resume_config(resume_dir, cfg)
    }

    fn save_checkpoint(
        out_dir: &Path,
        step: usize,
        model: &Self::Model,
        store: &mut TensorStore,
        cfg: &Self::Config,
        tokenizer_path: &Path,
        bos_token_id: u32,
        eos_token_id: u32,
        save_dtype: SaveDtype,
    ) -> Result<(), CliError> {
        save_qwen35_checkpoint(
            out_dir,
            step,
            model,
            store,
            cfg,
            tokenizer_path,
            bos_token_id,
            eos_token_id,
            save_dtype,
        )
    }
}

fn main() -> Result<(), CliError> {
    run()
}

fn run() -> Result<(), CliError> {
    let args = parse_args()?;
    validate_args(&args)?;
    let tokenizer = ChatTokenizer::from_file(&args.tokenizer)?;
    let vocab_size = args.vocab_size.unwrap_or_else(|| tokenizer.vocab_size());

    match resolve_pretrain_family(args.model_family) {
        PretrainModelFamily::Qwen3 => run_with_family::<Qwen3Family>(&args, tokenizer, vocab_size),
        PretrainModelFamily::Qwen35 => {
            run_with_family::<Qwen35Family>(&args, tokenizer, vocab_size)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PretrainModelFamily {
    Qwen3,
    Qwen35,
}

fn resolve_pretrain_family(requested: ModelFamily) -> PretrainModelFamily {
    match requested {
        ModelFamily::Qwen3 => PretrainModelFamily::Qwen3,
        ModelFamily::Qwen35 | ModelFamily::Auto => PretrainModelFamily::Qwen35,
    }
}

fn run_with_family<F: PretrainFamily>(
    args: &CliArgs,
    tokenizer: ChatTokenizer,
    vocab_size: usize,
) -> Result<(), CliError> {
    fs::create_dir_all(&args.out)?;

    let cfg = F::build_config(args, vocab_size)?;
    if args.seq > F::max_seq_len(&cfg) {
        return Err(CliError::Custom(format!(
            "--seq {} exceeds family max seq {}",
            args.seq,
            F::max_seq_len(&cfg)
        )));
    }

    let backend = build_backend(args.backend)?;
    println!("backend: {:?}", backend.device());
    println!("family={}", F::family_name());
    println!("config: {}", F::describe_config(&cfg));

    let mut store = TensorStore::with_backend(backend);
    let mut tape = Tape::new();
    let model = F::build_model(&cfg, &mut store)?;
    let mut registry = build_registry(&model);

    let resume_dir_canonical = args
        .resume_from
        .as_ref()
        .map(|resume_dir| {
            resume_dir.canonicalize().map_err(|e| {
                CliError::Custom(format!(
                    "failed to canonicalize --resume-from {}: {e} (is the path / symlink target missing?)",
                    resume_dir.display()
                ))
            })
        })
        .transpose()?;

    let start_step = if let Some(resume_dir) = &resume_dir_canonical {
        let step = resume_from_checkpoint::<F>(resume_dir, &mut registry, &mut store, &cfg)?;
        println!(
            "[pretrain] resumed {} from {} at step {}",
            F::family_name(),
            resume_dir.display(),
            step
        );
        step
    } else {
        0
    };

    let text = fs::read_to_string(&args.corpus).map_err(|e| {
        CliError::Custom(format!(
            "failed to read corpus {}: {e}",
            args.corpus.display()
        ))
    })?;
    let token_ids = tokenizer.encode(&text, false)?;
    if token_ids.len() <= args.seq {
        return Err(CliError::Custom(format!(
            "corpus has {} tokens but --seq is {}; need more tokens",
            token_ids.len(),
            args.seq
        )));
    }
    for &id in &token_ids {
        if (id as usize) >= vocab_size {
            return Err(CliError::Custom(format!(
                "token id {id} exceeds configured vocab_size {}",
                vocab_size
            )));
        }
    }

    let eval_len = ((token_ids.len() as f32) * args.eval_frac).floor() as usize;
    let (train_tokens, eval_tokens) = if args.eval_every > 0 && eval_len > args.seq {
        let split = token_ids.len() - eval_len;
        let (train, eval) = token_ids.split_at(split);
        (train.to_vec(), eval.to_vec())
    } else {
        (token_ids.clone(), Vec::new())
    };
    if train_tokens.len() <= args.seq {
        return Err(CliError::Custom(format!(
            "train slice has {} tokens after eval split but --seq is {}; reduce --eval-frac or grow corpus",
            train_tokens.len(),
            args.seq
        )));
    }
    println!(
        "corpus: {} tokens from {} (train={} eval={})",
        token_ids.len(),
        args.corpus.display(),
        train_tokens.len(),
        eval_tokens.len(),
    );

    let model_ids = live_tensor_ids(&store);
    let params = trainable_params(&model, &store);
    if params.is_empty() {
        return Err(CliError::Custom(format!(
            "{} model exposed no trainable parameters",
            F::family_name()
        )));
    }
    let param_names = trainable_param_name_map(&model, &store);
    let param_count: usize = params
        .iter()
        .map(|&id| store.get(id).map_or(0, |t| t.size))
        .sum();
    println!(
        "params: {param_count} ({:.2}M)",
        param_count as f64 / 1_000_000.0
    );

    let optim = AdamW::new(args.lr, DEFAULT_BETAS, DEFAULT_EPS, DEFAULT_WEIGHT_DECAY);
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
    let total_steps = start_step as u64 + args.steps as u64;
    let schedule = ConstantLr(args.lr);
    let run_timer = Instant::now();
    let controller = TrainingController::new();
    let metrics = open_run_metrics(args.metrics_jsonl.as_deref(), &controller)
        .map_err(|e| CliError::Custom(format!("metrics sink: {e}")))?;
    let run_id = train::metrics::default_run_id("pretrain");
    let backend_name = match args.backend {
        BackendChoice::Cpu => "cpu",
        BackendChoice::Metal => "metal",
        BackendChoice::Cuda => "cuda",
    };
    let out_dir_string = args.out.display().to_string();
    let resume_dir_string = resume_dir_canonical
        .as_ref()
        .map(|path| path.display().to_string());
    let mut run_start_strings = vec![
        ("model_family", F::family_name()),
        ("backend", backend_name),
        ("out", out_dir_string.as_str()),
    ];
    if let Some(path) = resume_dir_string.as_deref() {
        run_start_strings.push(("resume_from", path));
    }
    let run_start_scalars = [
        ("total_steps", total_steps as f64),
        ("grad_accum_steps", args.batch.max(1) as f64),
        ("seq", args.seq as f64),
    ];
    let run_start_bools = [("resumed", resume_dir_canonical.is_some())];
    emit_run_start(
        &metrics,
        &run_id,
        "pretrain",
        start_step as u64,
        &run_start_strings,
        &run_start_scalars,
        &run_start_bools,
    );
    sync_status(&controller, &metrics, |status| {
        status.iter = start_step;
        status.total_iters = total_steps as usize;
        status.started = true;
    });
    let _server_handle =
        serve_if_requested("pretrain", &controller, args.serve).map_err(CliError::Custom)?;

    let trainer_cfg = TrainerConfig {
        total_steps,
        grad_accum_steps: args.batch.max(1) as u64,
        log_every: args.log_every.max(1) as u64,
        eval_every: if eval_tokens.is_empty() || args.eval_every == 0 {
            None
        } else {
            Some(args.eval_every as u64)
        },
        save_every: Some(args.save_every.max(1) as u64),
        save_dir: Some(args.out.clone()),
        resume_from: resume_dir_canonical.clone(),
        rng_seed: args.seed,
    };
    let mut trainer = Trainer::new(
        optim,
        clip,
        schedule,
        Box::new(metrics.clone()),
        trainer_cfg,
    );

    if let Some(resume_dir) = &resume_dir_canonical {
        let resumed = trainer
            .resume_if_configured(&param_names)
            .map_err(CliError::Autograd)?;
        if resumed as usize != start_step {
            return Err(CliError::Custom(format!(
                "trainer resume step {} did not match checkpoint step {}",
                resumed, start_step
            )));
        }
        println!(
            "[pretrain] trainer optimizer state resumed from {} at step {}",
            resume_dir.display(),
            resumed
        );
    }

    let eval_keep: Vec<TensorId> = params.clone();
    let eval_model_ids: HashSet<TensorId> = model_ids.clone();
    let mut rng = LcgRng::seed(args.seed ^ start_step as u64);
    let mut eval_rng = LcgRng::seed(args.seed ^ 0x4556_414C_5F50_5245);
    let window = args.seq + 1;
    let upper = train_tokens.len().saturating_sub(window) + 1;
    let eval_upper = eval_tokens.len().saturating_sub(window) + 1;
    let position_ids: Vec<u32> = (0..args.seq as u32).collect();
    let token_count_per_micro = args.seq as u64;
    let model_ref = &model;
    let train_tokens_ref: &[u32] = &train_tokens;
    let eval_tokens_ref: &[u32] = &eval_tokens;
    let position_ids_ref: &[u32] = &position_ids;
    let family = F::family_name();

    let step_fn = |ctx: &mut StepCtx<'_>| -> AutogradResult<StepOutcome> {
        let start = (rng.next_u64() % upper as u64) as usize;
        let slice = &train_tokens_ref[start..start + window];
        let input_ids: Vec<u32> = slice[..args.seq].to_vec();
        let targets: Vec<usize> = slice[1..].iter().map(|&t| t as usize).collect();

        let logits = F::forward(model_ref, ctx.store, ctx.tape, &input_ids, position_ids_ref)
            .map_err(|err| cli_error_to_autograd(err, family, "forward"))?;
        let loss_id = cross_entropy_loss(logits, &targets, ctx.store, ctx.tape)?;
        Ok(StepOutcome {
            loss_id,
            token_count: token_count_per_micro,
        })
    };

    let eval_windows = args.eval_windows;
    let seq = args.seq;
    let eval_fn = move |store: &mut TensorStore, tape: &mut Tape| -> AutogradResult<EvalOutcome> {
        if eval_upper == 0 || eval_tokens_ref.is_empty() {
            return Ok(EvalOutcome {
                loss: f32::NAN,
                token_count: 0,
            });
        }
        let mut sum = 0.0_f32;
        let mut count: u64 = 0;
        tape.entries.clear();
        tape.set_enabled(false);
        for _ in 0..eval_windows {
            let start = (eval_rng.next_u64() % eval_upper as u64) as usize;
            let slice = &eval_tokens_ref[start..start + window];
            let input_ids: Vec<u32> = slice[..seq].to_vec();
            let targets: Vec<usize> = slice[1..].iter().map(|&t| t as usize).collect();

            let logits = F::forward(model_ref, store, tape, &input_ids, position_ids_ref)
                .map_err(|err| cli_error_to_autograd(err, family, "eval forward"))?;
            let loss_id = cross_entropy_loss(logits, &targets, store, tape)?;
            sum += store.to_host(loss_id)?[0];
            count += 1;
            tape.entries.clear();
            train::cleanup_after_backward(store, tape, &eval_keep, &eval_model_ids);
            tape.set_enabled(false);
        }
        tape.set_enabled(true);
        Ok(EvalOutcome {
            loss: sum / count.max(1) as f32,
            token_count: count * seq as u64,
        })
    };

    let out_dir = args.out.clone();
    let tokenizer_path = args.tokenizer.clone();
    let save_dtype = args.save_dtype;
    let bos_token_id = args.bos_token_id;
    let eos_token_id = args.eos_token_id;
    let cfg_ref = cfg.clone();
    let metrics_for_hooks = metrics.clone();
    let controller_for_hooks = Arc::clone(&controller);
    let on_step_end = |trainer_step: u64, store: &mut TensorStore| -> AutogradResult<()> {
        let is_final = trainer_step == total_steps;
        let save_requested = controller_for_hooks.take_save_request();
        if trainer_step.is_multiple_of(args.save_every as u64) || is_final || save_requested {
            F::save_checkpoint(
                &out_dir,
                trainer_step as usize,
                &model,
                store,
                &cfg_ref,
                &tokenizer_path,
                bos_token_id,
                eos_token_id,
                save_dtype,
            )
            .map_err(|err| cli_error_to_autograd(err, family, "save_checkpoint"))?;
            let checkpoint_dir = out_dir.join(format!("step_{trainer_step:06}"));
            let checkpoint_dir_string = checkpoint_dir.display().to_string();
            let strings = [
                ("path", checkpoint_dir_string.as_str()),
                ("artifact_model", "model.safetensors"),
                ("artifact_config", "config.json"),
                ("artifact_generation_config", "generation_config.json"),
                ("artifact_tokenizer", "tokenizer.json"),
            ];
            metrics_for_hooks.emit_event(&train::metrics::TrainEvent {
                kind: "checkpoint",
                step: Some(trainer_step),
                strings: &strings,
                scalars: &[],
                bools: &[],
            });
        }
        sync_status(&controller_for_hooks, &metrics_for_hooks, |status| {
            status.iter = trainer_step as usize;
            status.wall_secs = run_timer.elapsed().as_secs_f32();
        });
        if controller_for_hooks.should_stop() {
            return Err(AutogradError::TapeInvariant(STOP_REQUESTED_ERR));
        }
        Ok(())
    };

    let has_eval = !eval_tokens.is_empty() && args.eval_every > 0;
    let run_result = if has_eval {
        trainer.run_with_eval_and_hooks(
            &mut store,
            &mut tape,
            params,
            param_names,
            model_ids,
            step_fn,
            eval_fn,
            on_step_end,
        )
    } else {
        trainer.run_with_hooks(
            &mut store,
            &mut tape,
            params,
            param_names,
            model_ids,
            step_fn,
            on_step_end,
        )
    };

    let stopped = match run_result {
        Ok(()) => false,
        Err(AutogradError::TapeInvariant(msg)) if msg == STOP_REQUESTED_ERR => true,
        Err(err) => return Err(CliError::Autograd(err)),
    };

    let run_end_scalars = [
        ("completed_steps", trainer.step() as f64),
        ("dropped_metrics", metrics.dropped_metrics() as f64),
    ];
    let status = if stopped { "stopped" } else { "completed" };
    emit_run_end(&metrics, &run_id, status, trainer.step(), &run_end_scalars);
    sync_status(&controller, &metrics, |summary| {
        summary.iter = trainer.step() as usize;
        summary.total_iters = total_steps as usize;
        summary.wall_secs = run_timer.elapsed().as_secs_f32();
        summary.finished = true;
    });
    metrics.flush_blocking();

    Ok(())
}

fn resume_from_checkpoint<F: PretrainFamily>(
    resume_dir: &Path,
    registry: &mut SafetensorsRegistry,
    store: &mut TensorStore,
    cfg: &F::Config,
) -> Result<usize, CliError> {
    let resume_dir = resume_dir.canonicalize().map_err(|e| {
        CliError::Custom(format!(
            "failed to canonicalize --resume-from {}: {e} (is the path / symlink target missing?)",
            resume_dir.display()
        ))
    })?;

    let weights = resume_dir.join("model.safetensors");
    if !weights.exists() {
        return Err(CliError::Custom(format!(
            "resume path {} has no model.safetensors",
            resume_dir.display()
        )));
    }

    F::validate_resume_config(&resume_dir, cfg)?;
    registry.load_into_strict(store, &weights)?;

    let start_step = resume_dir
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|s| s.strip_prefix("step_"))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    Ok(start_step)
}

fn validate_qwen3_resume_config(resume_dir: &Path, cfg: &Qwen3Config) -> Result<(), CliError> {
    let saved = Qwen3Config::from_json_file(resume_dir.join("config.json"))?;
    if saved != *cfg {
        return Err(CliError::Custom(format!(
            "resume config {} does not match live qwen3 config",
            resume_dir.join("config.json").display()
        )));
    }
    Ok(())
}

fn validate_qwen35_resume_config(resume_dir: &Path, cfg: &Qwen35Config) -> Result<(), CliError> {
    let saved = Qwen35Config::from_json_file(resume_dir.join("config.json"))?;
    if saved != *cfg {
        return Err(CliError::Custom(format!(
            "resume config {} does not match live qwen35 config",
            resume_dir.join("config.json").display()
        )));
    }
    Ok(())
}

fn parse_args() -> Result<CliArgs, CliError> {
    parse_args_from(env::args().skip(1))
}

fn parse_args_from<I>(mut iter: I) -> Result<CliArgs, CliError>
where
    I: Iterator<Item = String>,
{
    let mut args = CliArgs::default();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model-family" => {
                args.model_family = next_value(&mut iter, &flag)?.parse().map_err(|value| {
                    CliError::Arg(ArgError::InvalidValue {
                        flag: flag.clone(),
                        value,
                    })
                })?;
            }
            "--corpus" => args.corpus = PathBuf::from(next_value(&mut iter, &flag)?),
            "--tokenizer" => args.tokenizer = PathBuf::from(next_value(&mut iter, &flag)?),
            "--out" => args.out = PathBuf::from(next_value(&mut iter, &flag)?),
            "--steps" => args.steps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--batch" => args.batch = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--seq" => args.seq = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--lr" => args.lr = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--log-every" => args.log_every = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--save-every" => args.save_every = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--eval-every" => args.eval_every = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--eval-windows" => {
                args.eval_windows = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--eval-frac" => args.eval_frac = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--resume-from" | "--resume" => {
                args.resume_from = Some(PathBuf::from(next_value(&mut iter, &flag)?))
            }
            "--seed" => args.seed = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--grad-clip" => {
                args.grad_clip = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--no-grad-clip" => args.grad_clip = None,
            "--backend" => {
                args.backend = next_value(&mut iter, &flag)?.parse().map_err(|value| {
                    CliError::Arg(ArgError::InvalidValue {
                        flag: flag.clone(),
                        value,
                    })
                })?;
            }
            "--save-dtype" => {
                args.save_dtype = next_value(&mut iter, &flag)?.parse().map_err(|value| {
                    CliError::Arg(ArgError::InvalidValue {
                        flag: flag.clone(),
                        value,
                    })
                })?;
            }
            "--vocab-size" => {
                args.vocab_size = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            "--hidden" => args.hidden_size = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--layers" => {
                args.num_hidden_layers = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--heads" => {
                args.num_attention_heads = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--kv-heads" => args.num_kv_heads = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--head-dim" => args.head_dim = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--intermediate" => {
                args.intermediate_size = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--max-pos" => {
                args.max_position_embeddings = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--rms-eps" => args.rms_norm_eps = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--rope-theta" => args.rope_theta = parse_value(&flag, next_value(&mut iter, &flag)?)?,
            "--no-tie-embed" => args.tie_word_embeddings = false,
            "--bos-token-id" => {
                args.bos_token_id = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--eos-token-id" => {
                args.eos_token_id = parse_value(&flag, next_value(&mut iter, &flag)?)?;
            }
            "--metrics-jsonl" => {
                args.metrics_jsonl = Some(PathBuf::from(next_value(&mut iter, &flag)?));
            }
            "--serve" => {
                args.serve = Some(parse_value(&flag, next_value(&mut iter, &flag)?)?);
            }
            _ => return Err(CliError::Arg(ArgError::UnknownFlag(flag))),
        }
    }
    Ok(args)
}

fn validate_args(args: &CliArgs) -> Result<(), CliError> {
    if args.corpus.as_os_str().is_empty() {
        return Err(CliError::Custom("--corpus is required".into()));
    }
    if args.tokenizer.as_os_str().is_empty() {
        return Err(CliError::Custom("--tokenizer is required".into()));
    }
    if args.out.as_os_str().is_empty() {
        return Err(CliError::Custom("--out is required".into()));
    }
    for (flag, value) in [
        ("--steps", args.steps),
        ("--batch", args.batch),
        ("--seq", args.seq),
        ("--log-every", args.log_every),
        ("--save-every", args.save_every),
        ("--hidden", args.hidden_size),
        ("--layers", args.num_hidden_layers),
        ("--heads", args.num_attention_heads),
        ("--kv-heads", args.num_kv_heads),
        ("--head-dim", args.head_dim),
        ("--intermediate", args.intermediate_size),
        ("--max-pos", args.max_position_embeddings),
    ] {
        if value == 0 {
            return Err(CliError::Arg(ArgError::InvalidValue {
                flag: flag.into(),
                value: "0".into(),
            }));
        }
    }
    if !(0.0..1.0).contains(&args.eval_frac) {
        return Err(CliError::Arg(ArgError::InvalidValue {
            flag: "--eval-frac".into(),
            value: args.eval_frac.to_string(),
        }));
    }
    if args.eval_every > 0 && args.eval_windows == 0 {
        return Err(CliError::Arg(ArgError::InvalidValue {
            flag: "--eval-windows".into(),
            value: "0".into(),
        }));
    }
    Ok(())
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

#[allow(clippy::too_many_arguments)]
fn save_qwen3_checkpoint(
    out_dir: &Path,
    step: usize,
    model: &Qwen3Model,
    store: &mut TensorStore,
    cfg: &Qwen3Config,
    tokenizer_path: &Path,
    bos_token_id: u32,
    eos_token_id: u32,
    save_dtype: SaveDtype,
) -> Result<(), CliError> {
    let torch_dtype = match save_dtype {
        SaveDtype::F32 => "float32",
        SaveDtype::Bf16 => "bfloat16",
    };
    let step_dir = save_step_checkpoint(
        Qwen3StepCheckpoint {
            out_dir,
            step,
            tokenizer_path: Some(tokenizer_path),
            config_json: ConfigJsonSource::Synthesize {
                cfg,
                bos_token_id,
                eos_token_id,
                torch_dtype,
            },
            generation_config: GenerationConfigSource::Synthesize {
                bos_token_id,
                eos_token_id,
            },
        },
        |weights_path| {
            let mut tape = Tape::new();
            let registry = train::causal_lm::build_materialized_registry(model, store, &mut tape)
                .map_err(Qwen3CheckpointError::from)?;
            match save_dtype {
                SaveDtype::F32 => registry
                    .save_from(store, weights_path)
                    .map_err(Qwen3CheckpointError::from),
                SaveDtype::Bf16 => registry
                    .save_from_bf16(store, weights_path)
                    .map_err(Qwen3CheckpointError::from),
            }
        },
    )?;

    println!(
        "[pretrain] saved qwen3 step {} to {} (dtype: {:?})",
        step,
        step_dir.display(),
        save_dtype
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn save_qwen35_checkpoint(
    out_dir: &Path,
    step: usize,
    model: &Qwen35Model,
    store: &mut TensorStore,
    cfg: &Qwen35Config,
    tokenizer_path: &Path,
    bos_token_id: u32,
    eos_token_id: u32,
    save_dtype: SaveDtype,
) -> Result<(), CliError> {
    let torch_dtype = match save_dtype {
        SaveDtype::F32 => "float32",
        SaveDtype::Bf16 => "bfloat16",
    };
    let step_dir = save_qwen35_step_checkpoint(
        Qwen35StepCheckpoint {
            out_dir,
            step,
            tokenizer_path: Some(tokenizer_path),
            config_json: Qwen35ConfigJsonSource::Synthesize { cfg, torch_dtype },
            generation_config: Qwen35GenerationConfigSource::Synthesize {
                bos_token_id: Some(bos_token_id).or(cfg.bos_token_id),
                eos_token_id: eos_token_id.max(cfg.eos_token_id),
            },
        },
        |weights_path| {
            let mut tape = Tape::new();
            let registry = train::causal_lm::build_materialized_registry(model, store, &mut tape)
                .map_err(Qwen35CheckpointError::from)?;
            match save_dtype {
                SaveDtype::F32 => registry
                    .save_from(store, weights_path)
                    .map_err(Qwen35CheckpointError::from),
                SaveDtype::Bf16 => registry
                    .save_from_bf16(store, weights_path)
                    .map_err(Qwen35CheckpointError::from),
            }
        },
    )?;

    println!(
        "[pretrain] saved qwen35 step {} to {} (dtype: {:?})",
        step,
        step_dir.display(),
        save_dtype
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use autograd::{ConstantLr, Tape, TensorStore, optim::AdamW};
    use tempfile::tempdir;
    use train::{
        StepOutcome, Trainer, TrainerConfig, qwen35::LayerType, trainer::cross_entropy_loss,
    };

    fn tiny_args() -> CliArgs {
        CliArgs {
            model_family: ModelFamily::Qwen35,
            corpus: PathBuf::new(),
            tokenizer: PathBuf::new(),
            out: PathBuf::new(),
            steps: 1,
            batch: 1,
            seq: 3,
            lr: 1.0e-3,
            log_every: 1,
            save_every: 1,
            eval_every: 0,
            eval_windows: 1,
            eval_frac: 0.1,
            resume_from: None,
            seed: 123,
            grad_clip: Some(1.0),
            backend: BackendChoice::Cpu,
            save_dtype: SaveDtype::Bf16,
            vocab_size: Some(32),
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            max_position_embeddings: 8,
            rms_norm_eps: 1.0e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            metrics_jsonl: None,
            serve: None,
        }
    }

    fn assert_adamw_state_eq(
        lhs: &autograd::adamw_state::AdamWState,
        rhs: &autograd::adamw_state::AdamWState,
    ) {
        assert_eq!(lhs.step, rhs.step);
        assert_eq!(lhs.skipped_export, rhs.skipped_export);
        assert_eq!(lhs.params.len(), rhs.params.len());
        let mut lhs_params = lhs.params.iter().collect::<Vec<_>>();
        let mut rhs_params = rhs.params.iter().collect::<Vec<_>>();
        lhs_params.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        rhs_params.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        for (idx, (left, right)) in lhs_params.iter().zip(rhs_params.iter()).enumerate() {
            assert_eq!(left.name, right.name, "param name mismatch at {idx}");
            assert_eq!(left.shape, right.shape, "shape mismatch at {idx}");
            assert_eq!(left.m.len(), right.m.len(), "m length mismatch at {idx}");
            assert_eq!(left.v.len(), right.v.len(), "v length mismatch at {idx}");
            for (elem_idx, (lm, rm)) in left.m.iter().zip(right.m.iter()).enumerate() {
                assert_eq!(lm.to_bits(), rm.to_bits(), "m drift at {idx}:{elem_idx}");
            }
            for (elem_idx, (lv, rv)) in left.v.iter().zip(right.v.iter()).enumerate() {
                assert_eq!(lv.to_bits(), rv.to_bits(), "v drift at {idx}:{elem_idx}");
            }
        }
    }

    #[test]
    fn model_family_dispatch_defaults_to_qwen35() {
        assert!(matches!(
            resolve_pretrain_family(ModelFamily::Auto),
            PretrainModelFamily::Qwen35
        ));
        assert!(matches!(
            resolve_pretrain_family(ModelFamily::Qwen35),
            PretrainModelFamily::Qwen35
        ));
        assert!(matches!(
            resolve_pretrain_family(ModelFamily::Qwen3),
            PretrainModelFamily::Qwen3
        ));
    }

    #[test]
    fn qwen35_family_builds_dense_full_attention_config() -> Result<(), CliError> {
        let args = tiny_args();
        let cfg = Qwen35Family::build_config(&args, args.vocab_size.unwrap())?;
        assert!(
            cfg.layer_types
                .iter()
                .all(|ty| *ty == LayerType::FullAttention)
        );
        assert_eq!(cfg.num_experts, 0);
        assert_eq!(cfg.rope_cache_len_hint, Some(args.max_position_embeddings));
        assert_eq!(cfg.bos_token_id, Some(args.bos_token_id));
        assert_eq!(cfg.eos_token_id, args.eos_token_id);
        Ok(())
    }

    #[test]
    fn parse_args_accepts_resume_from_and_resume_alias() {
        let canonical = parse_args_from(
            vec![
                "--corpus",
                "corpus.txt",
                "--tokenizer",
                "tok.json",
                "--out",
                "ckpt",
                "--resume-from",
                "ckpt/step_000007",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .expect("parse canonical resume-from");
        assert_eq!(
            canonical.resume_from,
            Some(PathBuf::from("ckpt/step_000007"))
        );

        let alias = parse_args_from(
            vec![
                "--corpus",
                "corpus.txt",
                "--tokenizer",
                "tok.json",
                "--out",
                "ckpt",
                "--resume",
                "ckpt/step_000008",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .expect("parse legacy resume alias");
        assert_eq!(alias.resume_from, Some(PathBuf::from("ckpt/step_000008")));
    }

    #[test]
    #[ignore = "replaced by exact resume checkpoint test below"]
    fn qwen35_pretrain_resume_restores_optimizer_state() -> Result<(), CliError> {
        let args = tiny_args();
        let cfg = Qwen35Family::build_config(&args, args.vocab_size.unwrap())?;

        let train_root = tempdir().expect("tempdir");
        let tokenizer_path = train_root.path().join("tokenizer.json");
        fs::write(&tokenizer_path, "{}").expect("write tokenizer");

        let mut store = TensorStore::default();
        let model = Qwen35Family::build_model(&cfg, &mut store)?;
        let params = trainable_params(&model, &store);
        let param_names = trainable_param_name_map(&model, &store);
        assert_eq!(param_names.len(), params.len(), "deduped param-name map");
        assert!(
            param_names
                .iter()
                .any(|(_, name)| name == "model.language_model.embed_tokens.weight")
        );

        let model_ids = live_tensor_ids(&store);
        let mut tape = Tape::new();
        let optim = AdamW::new(args.lr, DEFAULT_BETAS, DEFAULT_EPS, DEFAULT_WEIGHT_DECAY);
        let trainer_cfg = TrainerConfig {
            total_steps: 1,
            grad_accum_steps: 1,
            log_every: 1,
            eval_every: None,
            save_every: Some(1),
            save_dir: Some(train_root.path().to_path_buf()),
            resume_from: None,
            rng_seed: args.seed,
        };
        let mut trainer = Trainer::new(
            optim,
            NoClip,
            ConstantLr(args.lr),
            Box::new(train::metrics::NullSink),
            trainer_cfg,
        );

        trainer
            .run_with_hooks(
                &mut store,
                &mut tape,
                params.clone(),
                param_names.clone(),
                model_ids,
                |ctx| {
                    let logits = model
                        .forward(ctx.store, ctx.tape, &[1, 2, 3], &[0, 1, 2])
                        .map_err(|err| {
                            cli_error_to_autograd(CliError::Qwen35(err), "qwen35", "test forward")
                        })?;
                    let loss_id = cross_entropy_loss(logits, &[2, 3, 4], ctx.store, ctx.tape)?;
                    Ok(StepOutcome {
                        loss_id,
                        token_count: 3,
                    })
                },
                |_step, _store| Ok(()),
            )
            .expect("first pass trainer");

        let step_dir = train_root.path().join("step_000001");
        save_qwen35_checkpoint(
            train_root.path(),
            1,
            &model,
            &mut store,
            &cfg,
            &tokenizer_path,
            args.bos_token_id,
            args.eos_token_id,
            SaveDtype::F32,
        )?;

        let mut resumed_store = TensorStore::default();
        let resumed_model = Qwen35Family::build_model(&cfg, &mut resumed_store)?;
        let resumed_param_names = trainable_param_name_map(&resumed_model, &resumed_store);
        let mut resumed_registry = build_registry(&resumed_model);
        let (loaded_doc, loaded_optim) =
            train::checkpoint::load_trainer_state_v2(&step_dir).expect("load trainer state v2");
        assert_eq!(loaded_doc.step, 1);
        let loaded_step = resume_from_checkpoint::<Qwen35Family>(
            &step_dir,
            &mut resumed_registry,
            &mut resumed_store,
            &cfg,
        )?;
        assert_eq!(loaded_step, 1);
        let loaded_embed = resumed_model
            .param_name_map()
            .get("model.language_model.embed_tokens.weight")
            .copied()
            .expect("embed token param");
        assert_eq!(
            store.to_host(
                *model
                    .param_name_map()
                    .get("model.language_model.embed_tokens.weight")
                    .expect("embed token param")
            )?,
            resumed_store.to_host(loaded_embed)?
        );

        let mut resumed_trainer = Trainer::new(
            AdamW::new(args.lr, DEFAULT_BETAS, DEFAULT_EPS, DEFAULT_WEIGHT_DECAY),
            NoClip,
            ConstantLr(args.lr),
            Box::new(train::metrics::NullSink),
            TrainerConfig {
                total_steps: 2,
                grad_accum_steps: 1,
                log_every: 1,
                eval_every: None,
                save_every: Some(1),
                save_dir: Some(train_root.path().to_path_buf()),
                resume_from: Some(step_dir.clone()),
                rng_seed: args.seed,
            },
        );
        let resumed_step = resumed_trainer
            .resume_if_configured(&resumed_param_names)
            .expect("resume trainer state");
        assert_eq!(resumed_step, 1);
        let resumed_state = resumed_trainer.optim().export_state(&resumed_param_names);
        assert_adamw_state_eq(&loaded_optim, &resumed_state);

        Ok(())
    }

    #[test]
    fn qwen35_pretrain_resume_restores_checkpoint_state_exactly() -> Result<(), CliError> {
        let args = tiny_args();
        let cfg = Qwen35Family::build_config(&args, args.vocab_size.unwrap())?;

        let train_root = tempdir().expect("tempdir");
        let tokenizer_path = train_root.path().join("tokenizer.json");
        fs::write(&tokenizer_path, "{}").expect("write tokenizer");

        let mut store = TensorStore::default();
        let model = Qwen35Family::build_model(&cfg, &mut store)?;
        let param_names = trainable_param_name_map(&model, &store);
        let params = trainable_params(&model, &store);
        assert_eq!(param_names.len(), params.len());

        let step_dir = train_root.path().join("step_000007");
        save_qwen35_checkpoint(
            train_root.path(),
            7,
            &model,
            &mut store,
            &cfg,
            &tokenizer_path,
            args.bos_token_id,
            args.eos_token_id,
            SaveDtype::F32,
        )?;

        let resume_state = autograd::adamw_state::AdamWState {
            step: 7,
            skipped_export: 0,
            params: param_names
                .iter()
                .enumerate()
                .map(|(idx, (tensor_id, name))| {
                    let shape = store.get(*tensor_id).expect("tensor exists").shape.clone();
                    let len = shape.iter().product::<usize>();
                    let base = idx as f32 + 0.25;
                    autograd::adamw_state::AdamWParamState {
                        name: name.clone(),
                        m: (0..len).map(|offset| base + offset as f32 * 0.01).collect(),
                        v: (0..len)
                            .map(|offset| base * 2.0 + offset as f32 * 0.02)
                            .collect(),
                        shape,
                    }
                })
                .collect(),
        };
        let resume_doc = train::checkpoint::TrainerStateDoc {
            step: 7,
            optim_schema: "adamw-v1".to_string(),
            schedule_name: "constant".to_string(),
            schedule_params: serde_json::json!({ "lr": args.lr }),
            grad_accum_current: 1,
            rng_seed: args.seed,
            codec_version: train::checkpoint::TRAINER_STATE_CODEC_VERSION,
        };
        train::checkpoint::save_trainer_state_v2(&step_dir, &resume_doc, &resume_state)
            .expect("save trainer state");

        let (loaded_doc, loaded_optim) =
            train::checkpoint::load_trainer_state_v2(&step_dir).expect("load trainer state");
        assert_eq!(loaded_doc.step, 7);
        assert_adamw_state_eq(&resume_state, &loaded_optim);

        let mut resumed_store = TensorStore::default();
        let resumed_model = Qwen35Family::build_model(&cfg, &mut resumed_store)?;
        let resumed_param_names = trainable_param_name_map(&resumed_model, &resumed_store);
        let mut resumed_registry = build_registry(&resumed_model);
        let loaded_step = resume_from_checkpoint::<Qwen35Family>(
            &step_dir,
            &mut resumed_registry,
            &mut resumed_store,
            &cfg,
        )?;
        assert_eq!(loaded_step, 7);

        let loaded_embed = resumed_model
            .param_name_map()
            .get("model.language_model.embed_tokens.weight")
            .copied()
            .expect("embed token param");
        assert_eq!(
            store.to_host(
                *model
                    .param_name_map()
                    .get("model.language_model.embed_tokens.weight")
                    .expect("embed token param")
            )?,
            resumed_store.to_host(loaded_embed)?
        );

        let mut resumed_trainer = Trainer::new(
            AdamW::new(args.lr, DEFAULT_BETAS, DEFAULT_EPS, DEFAULT_WEIGHT_DECAY),
            NoClip,
            ConstantLr(args.lr),
            Box::new(train::metrics::NullSink),
            TrainerConfig {
                total_steps: 8,
                grad_accum_steps: 1,
                log_every: 1,
                eval_every: None,
                save_every: None,
                save_dir: Some(train_root.path().to_path_buf()),
                resume_from: Some(step_dir.clone()),
                rng_seed: args.seed,
            },
        );
        let resumed_step = resumed_trainer
            .resume_if_configured(&resumed_param_names)
            .expect("resume trainer state");
        assert_eq!(resumed_step, 7);
        let resumed_state = resumed_trainer.optim().export_state(&resumed_param_names);
        assert_adamw_state_eq(&resume_state, &resumed_state);

        Ok(())
    }
}

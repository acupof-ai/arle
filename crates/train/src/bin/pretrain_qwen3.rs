// Pretrain a Qwen3-layout model from random init on a plain-text corpus,
// saving checkpoints in the safetensors + config.json + tokenizer.json
// layout that infer/ can load. Mirrors train_sft.rs's save pipeline but
// starts from scratch (no source model dir) and drives a packed 1D forward
// over random corpus windows.
//
// Phase 3 (2026-04-20): migrated onto the generic `Trainer<O, C, S>` loop.
// The hand-written optimizer-step / clip / backward / cleanup sequence now
// lives in `train::Trainer`; this binary owns only the data sampler, the
// forward+loss closure, the eval closure, and the model-weight checkpoint
// save pipeline (wired via `on_step_end`). See `pretrain.rs` for the
// template and `docs/plans/train-runtime-architecture-v1.md` for context.

use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use autograd::{
    AutogradError, Backend, ConstantLr, CpuBackend, Result as AutogradResult, SafetensorsRegistry,
    Tape, TensorId, TensorStore, optim::AdamW,
};
use qwen3_spec::Qwen3Config;
use serde_json::json;
use thiserror::Error;
use train::{
    EvalOutcome, StepCtx, StepOutcome, Trainer, TrainerConfig,
    checkpoint::publish_latest_after_weights,
    cli_args::{ArgError, next_value, parse_value},
    dataset::LcgRng,
    grad_clip::{GlobalNorm, GradClip, NoClip},
    qwen3::{Qwen3Error, Qwen3Model},
    tokenizer::ChatTokenizer,
    trainer::cross_entropy_loss,
};

// Qwen3 tokenizer defaults — infer/Config requires both fields.
const DEFAULT_BOS_TOKEN_ID: u32 = 151_643;
const DEFAULT_EOS_TOKEN_ID: u32 = 151_645;

const DEFAULT_BETAS: (f32, f32) = (0.9, 0.999);
const DEFAULT_EPS: f32 = 1.0e-8;
const DEFAULT_WEIGHT_DECAY: f32 = 0.01;

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
    resume: Option<PathBuf>,
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
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
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
            resume: None,
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

/// `Qwen3Model::forward` returns `Result<_, Qwen3Error>`, but the Trainer
/// step/eval closures must return `Result<_, AutogradError>`. `Qwen3Error`
/// has `From<AutogradError>` (one-way), so unwrap the inner autograd error
/// when possible and otherwise stash the display string on stderr and
/// surface a generic TapeInvariant so the outer loop can still unwind.
fn qwen3_to_autograd(err: Qwen3Error) -> AutogradError {
    match err {
        Qwen3Error::Autograd(inner) => inner,
        other => {
            eprintln!("[pretrain_qwen3] qwen3 forward error: {other}");
            AutogradError::TapeInvariant(
                "pretrain_qwen3: qwen3 forward returned non-autograd error",
            )
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
    Qwen3(#[from] Qwen3Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Arg(#[from] ArgError),
    #[error("{0}")]
    Custom(String),
}

fn main() -> Result<(), CliError> {
    let args = parse_args()?;
    validate_args(&args)?;
    fs::create_dir_all(&args.out)?;

    let tokenizer = ChatTokenizer::from_file(&args.tokenizer)?;
    let vocab_size = args.vocab_size.unwrap_or_else(|| tokenizer.vocab_size());

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
    if args.seq > cfg.max_position_embeddings {
        return Err(CliError::Custom(format!(
            "--seq {} exceeds --max-pos {}",
            args.seq, cfg.max_position_embeddings
        )));
    }

    let backend = build_backend(args.backend)?;
    println!("backend: {:?}", backend.device());
    println!(
        "config: vocab={} hidden={} layers={} heads={} kv_heads={} head_dim={} ffn={} max_pos={} tie_embed={}",
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.max_position_embeddings,
        cfg.tie_word_embeddings,
    );

    let mut store = TensorStore::with_backend(backend);
    let mut tape = Tape::new();
    let model = Qwen3Model::new(&cfg, &mut store)?;
    let mut registry = build_registry(&model);

    // --resume loads weights from a prior checkpoint (step_N dir). Optimizer
    // state (AdamW m/v) is NOT persisted yet — the optimizer reinitializes
    // with zeroed moments, so you may want a brief LR warmup after resume.
    let start_step = if let Some(resume_dir) = &args.resume {
        resume_from_checkpoint(resume_dir, &mut registry, &mut store, &cfg)?
    } else {
        0
    };
    if start_step > 0 {
        println!(
            "[pretrain_qwen3] resumed from {} at step {} (optimizer state reset)",
            args.resume.as_ref().unwrap().display(),
            start_step
        );
    }

    // Tokenize corpus once and validate IDs against configured vocab.
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
        if (id as usize) >= cfg.vocab_size {
            return Err(CliError::Custom(format!(
                "token id {id} exceeds configured vocab_size {}",
                cfg.vocab_size
            )));
        }
    }
    // Hold out the last `eval_frac` slice for eval. The boundary is
    // deterministic from corpus length + eval_frac, so eval and train stay
    // reproducible across resumes of the same corpus.
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
        return Err(CliError::Custom(
            "qwen3 model exposed no trainable parameters".into(),
        ));
    }
    let param_count: usize = params
        .iter()
        .map(|&id| store.get(id).map_or(0, |t| t.size))
        .sum();
    println!(
        "params: {param_count} ({:.2}M)",
        param_count as f64 / 1_000_000.0
    );

    // ---- Trainer setup ----
    // AdamW with the historical pretrain_qwen3 defaults (betas=(0.9,0.999),
    // eps=1e-8, wd=0.01). Preserve bit-for-bit against the pre-migration run.
    let optim = AdamW::new(args.lr, DEFAULT_BETAS, DEFAULT_EPS, DEFAULT_WEIGHT_DECAY);

    // Codex review 2026-04-20 on 6bd0211 (M1): pre-migration
    // `clip_grad_norm(params, max_norm <= 0.0, store)` returned early as a
    // no-op, so `--grad-clip 0 / NaN / inf` were legal and effectively
    // disabled clipping. The migrated path routes through
    // `GlobalNorm::new`, which panics on non-positive / non-finite
    // `max_norm`. Preserve legacy semantics by treating those values as
    // NoClip with a one-shot warning.
    let clip = match args.grad_clip {
        Some(max_norm) if max_norm > 0.0 && max_norm.is_finite() => {
            PretrainClip::Norm(GlobalNorm::new(max_norm))
        }
        Some(max_norm) => {
            eprintln!(
                "[pretrain_qwen3] warning: --grad-clip {max_norm} is non-positive/non-finite; disabling gradient clipping"
            );
            PretrainClip::None(NoClip)
        }
        None => PretrainClip::None(NoClip),
    };
    let schedule = ConstantLr(args.lr);
    let metrics = train::metrics::open_sink(args.metrics_jsonl.as_deref(), true)
        .map_err(|e| CliError::Custom(format!("metrics sink: {e}")))?;

    // Resume relative-step note: the hand-written loop logged absolute step
    // numbers (`start_step + step_offset`). The Trainer's internal `self.step`
    // starts at 0 and counts `1..=total_steps` regardless of where weights
    // were loaded from (no public API to advance the internal counter).
    // Compromise: set `total_steps = args.steps` and treat Trainer's step as
    // relative — the model-weight checkpoint writer below reconstructs the
    // absolute step as `start_step + trainer_step` so the on-disk
    // `step_<N>/` directories stay numbered absolutely. Per-step metric
    // samples will show the relative step number, which is a minor
    // deviation from the old log format; accept it (same choice as
    // `pretrain.rs`, which has no resume at all).
    //
    // `save_every` / `save_dir` are left `None` on purpose: the Trainer's
    // built-in save pipeline writes `trainer_state.json +
    // optimizer.safetensors` into `step_<N>/`, which is NEW state this
    // binary didn't persist before the migration. Adding it is a deliberate
    // follow-up; the existing weight-only save runs from `on_step_end`
    // instead so `step_<N>/` stays bit-compatible with pre-migration output.
    let trainer_cfg = TrainerConfig {
        total_steps: args.steps as u64,
        grad_accum_steps: args.batch.max(1) as u64,
        log_every: args.log_every.max(1) as u64,
        eval_every: if eval_tokens.is_empty() || args.eval_every == 0 {
            None
        } else {
            Some(args.eval_every as u64)
        },
        save_every: None,
        save_dir: None,
        resume_from: None,
        rng_seed: args.seed,
    };
    let mut trainer = Trainer::new(optim, clip, schedule, metrics, trainer_cfg);

    // `param_names` — synthetic stable names. The Trainer requires a
    // `(TensorId, String)` map for optimizer state persistence; since
    // `save_every` is None (no trainer_state.json yet), these names are
    // never written. When optimizer persistence lands, swap these for the
    // real `model.param_name_map()` entries.
    let param_names: Vec<(TensorId, String)> = params
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, format!("param.{i:04}")))
        .collect();

    // `keep_extra`: every model tensor id the Trainer's cleanup should leave
    // alive. The hand-written `retained_ids(&model_ids, &params, ...)`
    // returned `model_ids ∪ params ∪ grads`; the Trainer already keeps
    // `params ∪ grads` itself, so we pass `model_ids` as the extra set.
    let keep_extra: HashSet<TensorId> = model_ids.clone();

    // Codex review 2026-04-20 on bd5e277 (High): the eval closure needs its
    // own copy of `params` and the model-id set so it can call
    // `train::cleanup_after_backward` per eval window. Clone before the
    // Trainer consumes `params` / `keep_extra`.
    let eval_keep: Vec<TensorId> = params.clone();
    let eval_model_ids: HashSet<TensorId> = model_ids.clone();

    // Sampler RNGs live outside the closures so their state persists across
    // step_fn invocations. The seed derivations match the hand-written loop
    // exactly: `args.seed ^ start_step` for train windows,
    // `args.seed ^ 0x4556_414C_5F50_5245` (ASCII "EVAL_PRE") for eval.
    let mut rng = LcgRng::seed(args.seed ^ start_step as u64);
    let mut eval_rng = LcgRng::seed(args.seed ^ 0x4556_414C_5F50_5245);
    let window = args.seq + 1;
    let upper = train_tokens.len().saturating_sub(window) + 1;
    let eval_upper = eval_tokens.len().saturating_sub(window) + 1;
    let position_ids: Vec<u32> = (0..args.seq as u32).collect();
    let token_count_per_micro = args.seq as u64;

    // Borrow everything the closures need by reference; the borrow checker
    // enforces their lifetimes against `trainer.run_*`.
    let model_ref = &model;
    let train_tokens_ref: &[u32] = &train_tokens;
    let eval_tokens_ref: &[u32] = &eval_tokens;
    let position_ids_ref: &[u32] = &position_ids;

    let step_fn = |ctx: &mut StepCtx<'_>| -> AutogradResult<StepOutcome> {
        // Sample one training window per micro-batch. The Trainer's internal
        // micro-batch loop invokes this closure `grad_accum_steps` times per
        // optimizer step — matches the hand-written `for _micro in
        // 0..args.batch` exactly.
        let start = (rng.next_u64() % upper as u64) as usize;
        let slice = &train_tokens_ref[start..start + window];
        let input_ids: Vec<u32> = slice[..args.seq].to_vec();
        let targets: Vec<usize> = slice[1..].iter().map(|&t| t as usize).collect();

        let logits = model_ref
            .forward(ctx.store, ctx.tape, &input_ids, position_ids_ref)
            .map_err(qwen3_to_autograd)?;
        let loss_id = cross_entropy_loss(logits, &targets, ctx.store, ctx.tape)?;
        // Return the *unscaled* loss — the Trainer applies
        // `loss_scale = 1/grad_accum_steps` via `mul_scalar` before backward.
        Ok(StepOutcome {
            loss_id,
            token_count: token_count_per_micro,
        })
    };

    // Eval closure — runs `args.eval_windows` held-out windows and returns
    // mean loss. The hand-written `run_eval` also toggled `tape.set_enabled`
    // around the forward; we replicate the same no-grad pattern here. The
    // Trainer's internal cleanup runs after the closure returns.
    let eval_windows = args.eval_windows;
    let seq = args.seq;
    let eval_fn = move |store: &mut TensorStore, tape: &mut Tape| -> AutogradResult<EvalOutcome> {
        if eval_upper == 0 || eval_tokens_ref.is_empty() {
            return Ok(EvalOutcome {
                loss: f32::NAN,
                token_count: 0,
            });
        }
        // Disable the tape for eval so we don't build a graph for a
        // backward pass that never comes. Re-enable on the way out so the
        // Trainer's next train step finds the tape live.
        let mut sum = 0.0_f32;
        let mut count: u64 = 0;
        tape.entries.clear();
        tape.set_enabled(false);
        for _ in 0..eval_windows {
            let start = (eval_rng.next_u64() % eval_upper as u64) as usize;
            let slice = &eval_tokens_ref[start..start + window];
            let input_ids: Vec<u32> = slice[..seq].to_vec();
            let targets: Vec<usize> = slice[1..].iter().map(|&t| t as usize).collect();

            let logits = model_ref
                .forward(store, tape, &input_ids, position_ids_ref)
                .map_err(qwen3_to_autograd)?;
            let loss_id = cross_entropy_loss(logits, &targets, store, tape)?;
            sum += store.to_host(loss_id)?[0];
            count += 1;
            tape.entries.clear();
            // Codex review 2026-04-20 on bd5e277 (High): without this prune,
            // the forward graph for every eval window accumulates in the
            // store, so `--eval-windows N` with a large `--seq` OOMs.
            // `cleanup_after_backward` re-enables the tape internally (it
            // was designed for the training-path post-backward call site);
            // flip it back off for the next window.
            train::cleanup_after_backward(store, tape, &eval_keep, &eval_model_ids);
            tape.set_enabled(false);
        }
        tape.set_enabled(true);
        Ok(EvalOutcome {
            loss: sum / count.max(1) as f32,
            token_count: count * seq as u64,
        })
    };

    // `on_step_end` drives the model-weight save pipeline. Absolute step =
    // `start_step + trainer_step`, matching the pre-migration
    // `step_<N>/` directory numbering. Force-save on the final step so a
    // run that ends between save boundaries still leaves a resumable
    // checkpoint on disk (parity with `trainer.save_every`'s own
    // is_final logic).
    let save_every = args.save_every as u64;
    let total_steps = args.steps as u64;
    let out_dir = args.out.clone();
    let tokenizer_path = args.tokenizer.clone();
    let bos_token_id = args.bos_token_id;
    let eos_token_id = args.eos_token_id;
    let save_dtype = args.save_dtype;
    let cfg_ref = cfg.clone();
    let registry_ref = &registry;
    let on_step_end = |trainer_step: u64, store: &mut TensorStore| -> AutogradResult<()> {
        let abs_step = start_step as u64 + trainer_step;
        let is_final = trainer_step == total_steps;
        if trainer_step.is_multiple_of(save_every) || is_final {
            save_checkpoint(
                &out_dir,
                abs_step as usize,
                registry_ref,
                store,
                &cfg_ref,
                &tokenizer_path,
                bos_token_id,
                eos_token_id,
                save_dtype,
            )
            .map_err(|err| {
                // save_checkpoint returns CliError; funnel non-autograd
                // failures through TapeInvariant with stderr context so the
                // Trainer Result<()> stays AutogradError-shaped.
                if let CliError::Autograd(e) = err {
                    return e;
                }
                eprintln!("[pretrain_qwen3] save_checkpoint failed: {err}");
                AutogradError::TapeInvariant("pretrain_qwen3: save_checkpoint failed")
            })?;
        }
        Ok(())
    };

    // Dispatch: eval-and-hooks when eval is configured, otherwise just hooks
    // (Trainer's eval branch is gated on `cfg.eval_every`, but we pick the
    // narrower API when possible to document intent and avoid a dead closure).
    let has_eval = !eval_tokens.is_empty() && args.eval_every > 0;
    if has_eval {
        trainer.run_with_eval_and_hooks(
            &mut store,
            &mut tape,
            params,
            param_names,
            keep_extra,
            step_fn,
            eval_fn,
            on_step_end,
        )?;
    } else {
        trainer.run_with_hooks(
            &mut store,
            &mut tape,
            params,
            param_names,
            keep_extra,
            step_fn,
            on_step_end,
        )?;
    }

    Ok(())
}

// Phase 3 migration note: `run_eval(..)` was inlined into the eval closure
// handed to `Trainer::run_with_eval_and_hooks`. `retained_ids(..)` was
// removed because the Trainer's internal `cleanup_after_backward` performs
// the equivalent `keep_extra ∪ params ∪ grads` retention itself. See the
// closure in `main` for the reconstructed eval body.
fn resume_from_checkpoint(
    resume_dir: &Path,
    registry: &mut SafetensorsRegistry,
    store: &mut TensorStore,
    cfg: &Qwen3Config,
) -> Result<usize, CliError> {
    // DX-1 follow-up (codex review 2026-04-20 on 8bde810, High): canonicalize
    // at function entry so every subsequent read (weights, config, step
    // derivation) refers to the same snapshot of the `latest` symlink. Without
    // this, a concurrent trainer repointing `latest` between our opens could
    // let us mix step N weights with step N+1 metadata. Bubble the error
    // instead of silently falling back to step 0 — a broken symlink at resume
    // time is the user's bug, not something we should paper over.
    let resume_dir = resume_dir.canonicalize().map_err(|e| {
        CliError::Custom(format!(
            "failed to canonicalize --resume {}: {e} (is the path / symlink target missing?)",
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

    // Verify the checkpoint's config.json matches the requested config so a
    // silent shape mismatch doesn't surface as a mid-training crash.
    // Codex review 429efc3 (Medium): missing config.json was previously a
    // silent skip, which reopened the tied/untied silent-corruption path
    // whenever the checkpoint lacked a config file. Hard-error instead.
    let cfg_path = resume_dir.join("config.json");
    if !cfg_path.exists() {
        return Err(CliError::Custom(format!(
            "resume path {} has no config.json; refuse to resume without a config-match check \
             (would otherwise silently miss tie_word_embeddings / shape drift). \
             Regenerate config.json from the expected live Qwen3Config or fresh-start.",
            resume_dir.display()
        )));
    }
    {
        let file_cfg: serde_json::Value = serde_json::from_str(&fs::read_to_string(&cfg_path)?)?;
        let mut mismatches: Vec<String> = [
            ("hidden_size", cfg.hidden_size as i64),
            ("intermediate_size", cfg.intermediate_size as i64),
            ("num_hidden_layers", cfg.num_hidden_layers as i64),
            ("num_attention_heads", cfg.num_attention_heads as i64),
            ("num_key_value_heads", cfg.num_key_value_heads as i64),
            ("head_dim", cfg.head_dim as i64),
            ("vocab_size", cfg.vocab_size as i64),
        ]
        .iter()
        .filter_map(|(k, v)| match file_cfg.get(*k).and_then(|x| x.as_i64()) {
            Some(seen) if seen != *v => Some(format!("{k}: ckpt={seen} cli={v}")),
            _ => None,
        })
        .collect();

        // Codex review feb715f (High): `tie_word_embeddings` determines
        // whether `embed_tokens.weight` and `lm_head.weight` alias to one
        // TensorId or stay distinct. An untied checkpoint loaded into a
        // tied live config would pass the numeric-shape check, then
        // silently merge two file tensors into one live slot.
        // `load_into_strict` below also rejects missing names, but we
        // fail fast on the config flag so the error points at the root
        // cause (config divergence) rather than at a missing tensor.
        if let Some(saw) = file_cfg
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            && saw != cfg.tie_word_embeddings
        {
            mismatches.push(format!(
                "tie_word_embeddings: ckpt={saw} cli={}",
                cfg.tie_word_embeddings
            ));
        }

        if !mismatches.is_empty() {
            return Err(CliError::Custom(format!(
                "resume config mismatch: {}",
                mismatches.join(", ")
            )));
        }
    }

    registry.load_into_strict(store, &weights)?;

    // Derive absolute step from the canonical dir name `step_<N>`.
    // `resume_dir` is already canonicalized above, so `file_name()` resolves
    // to the `step_NNNNNN` target directly (not the literal "latest").
    let start_step = resume_dir
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|s| s.strip_prefix("step_"))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    Ok(start_step)
}

fn parse_args() -> Result<CliArgs, CliError> {
    let mut args = CliArgs::default();
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
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
            "--resume" => args.resume = Some(PathBuf::from(next_value(&mut iter, &flag)?)),
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
                "[pretrain_qwen3] warning: metal backend requested without --features metal; falling back to cpu"
            );
            Ok(Arc::new(CpuBackend))
        }
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => Ok(Arc::new(autograd::backend_cuda::CudaBackend::new(0)?)),
        #[cfg(not(feature = "cuda"))]
        BackendChoice::Cuda => {
            eprintln!(
                "[pretrain_qwen3] warning: cuda backend requested without --features cuda; falling back to cpu"
            );
            Ok(Arc::new(CpuBackend))
        }
    }
}

fn build_registry(model: &Qwen3Model) -> SafetensorsRegistry {
    let mut registry = SafetensorsRegistry::new();
    for (name, tensor_id) in model.param_name_map() {
        registry.insert(name, tensor_id);
    }
    registry
}

fn trainable_params(model: &Qwen3Model, store: &TensorStore) -> Vec<TensorId> {
    let mut params = model
        .param_name_map()
        .into_values()
        .collect::<HashSet<_>>()
        .into_iter()
        .filter(|id| store.get(*id).is_some_and(|t| t.requires_grad))
        .collect::<Vec<_>>();
    params.sort_unstable();
    params
}

fn live_tensor_ids(store: &TensorStore) -> HashSet<TensorId> {
    store
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(id, slot)| slot.as_ref().map(|_| id))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn save_checkpoint(
    out_dir: &Path,
    step: usize,
    registry: &SafetensorsRegistry,
    store: &mut TensorStore,
    cfg: &Qwen3Config,
    tokenizer_path: &Path,
    bos_token_id: u32,
    eos_token_id: u32,
    save_dtype: SaveDtype,
) -> Result<(), CliError> {
    // DX-1: zero-padded 6-digit to match Trainer::save_checkpoint and
    // train_sft::save_checkpoint_via_registry. Consistent padding makes
    // resume-path lookup a single glob + lex-max pattern instead of
    // per-binary branches.
    let step_basename = format!("step_{step:06}");
    let step_dir = out_dir.join(&step_basename);
    fs::create_dir_all(&step_dir)?;

    let config_json = json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "max_position_embeddings": cfg.max_position_embeddings,
        "torch_dtype": match save_dtype { SaveDtype::F32 => "float32", SaveDtype::Bf16 => "bfloat16" },
    });
    fs::write(
        step_dir.join("config.json"),
        serde_json::to_string_pretty(&config_json)?,
    )?;
    fs::copy(tokenizer_path, step_dir.join("tokenizer.json"))?;

    let weights_path = step_dir.join("model.safetensors");
    match save_dtype {
        SaveDtype::F32 => registry.save_from(store, &weights_path)?,
        SaveDtype::Bf16 => registry.save_from_bf16(store, &weights_path)?,
    }
    // DX-1: refresh `<out>/latest` symlink after weights write so the
    // just-written step becomes directly addressable for `infer` / resume.
    // `publish_latest_after_weights` asserts `model.safetensors` exists in
    // `step_dir` before flipping the symlink — codifies the publish-last
    // contract so a future refactor that moves this call above the weight
    // write (or drops it) would fail a targeted unit test instead of silently
    // exposing an incomplete checkpoint dir.
    publish_latest_after_weights(out_dir, &step_basename)?;

    println!(
        "[pretrain_qwen3] saved step {} to {} (dtype: {:?})",
        step,
        step_dir.display(),
        save_dtype
    );
    Ok(())
}

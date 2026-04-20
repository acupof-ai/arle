//! Phase 2 `Trainer<O, C, S>` — generic training loop that owns LR schedule
//! ticking, gradient accumulation, clipping, optimizer step, metrics emission,
//! and checkpoint scheduling. Binaries supply the forward+loss closure and
//! (optionally) an eval closure; everything else lives here.
//!
//! See `docs/plans/train-runtime-architecture-v1.md` §6 for the design.
//!
//! The thin re-exports at the top of this file preserve the legacy import
//! paths used by the un-migrated binaries (`trainer::clip_grad_norm` /
//! `trainer::cross_entropy_loss`). Wave 3 migrates those binaries onto the
//! struct below.

pub use crate::grad_clip::clip_grad_norm;
pub use crate::loss::cross_entropy_loss;

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use autograd::{
    AutogradError, LrSchedule, Optimizer, Result, Tape, TensorId, TensorStore, ops::mul_scalar,
};

use crate::checkpoint::{
    CheckpointError, TRAINER_STATE_CODEC_VERSION, TrainerStateDoc, load_trainer_state_v2,
    save_trainer_state_v2,
};
use crate::grad_accum::GradAccumulator;
use crate::grad_clip::GradClip;
use crate::metrics::{MetricSample, MetricSink};

/// Scalar config that drives the Trainer loop. The public Trainer API takes
/// ownership of this at construction.
pub struct TrainerConfig {
    /// Number of optimizer steps to run (not micro-batches). Resuming from
    /// step `k` runs for `total_steps - k` further steps.
    pub total_steps: u64,
    /// Micro-batches per optimizer step. Must be >= 1.
    pub grad_accum_steps: u64,
    /// Emit a metric sample every N optimizer steps. Must be >= 1.
    pub log_every: u64,
    /// If `Some(n)`, call the eval closure (if provided) every N optimizer
    /// steps. Ignored by `run` (only honored by `run_with_eval`).
    pub eval_every: Option<u64>,
    /// If `Some(n)`, save a checkpoint every N optimizer steps. Requires
    /// `save_dir` to be `Some`.
    pub save_every: Option<u64>,
    /// Root directory for `step_{NNNNNN}/{trainer_state.json,optimizer.safetensors}`.
    pub save_dir: Option<PathBuf>,
    /// If `Some(path)`, `resume_if_configured` will load the trainer state
    /// doc + AdamW moments from this directory.
    pub resume_from: Option<PathBuf>,
    /// Persisted across resumes via `TrainerStateDoc.rng_seed`. The Trainer
    /// does not itself seed any RNG — binaries own data iteration.
    pub rng_seed: u64,
}

/// Context handed to the caller's forward+loss closure for a single
/// micro-batch. `store` and `tape` are borrowed mutably from the Trainer so
/// the closure can build graph and return the scalar loss tensor id.
pub struct StepCtx<'a> {
    /// Optimizer-step index (0-based).
    pub step: u64,
    /// Which micro-batch inside the current accumulation window, 0-indexed.
    pub micro_idx: u64,
    /// Informational `1.0 / grad_accum_steps`. The Trainer auto-applies this
    /// scale via `mul_scalar` before `tape.backward`, so the closure's
    /// returned `loss_id` should be the *pre-scale* loss for readability.
    pub loss_scale: f32,
    pub store: &'a mut TensorStore,
    pub tape: &'a mut Tape,
}

/// What the per-micro-batch forward closure returns.
pub struct StepOutcome {
    /// Scalar loss tensor in `store`/`tape`. The Trainer will `mul_scalar`
    /// this by `1/N` (when N > 1) and then call `tape.backward` on the
    /// scaled id.
    pub loss_id: TensorId,
    /// Tokens processed in this micro-batch, for the `tok_per_sec` metric.
    pub token_count: u64,
}

/// What an eval closure returns.
pub struct EvalOutcome {
    pub loss: f32,
    pub token_count: u64,
}

/// The generic training loop. Parameterised on optimizer `O`, clip policy
/// `C`, and LR schedule `S` so each binary picks its own concrete types
/// without monomorphising through a `dyn` hole.
pub struct Trainer<O: Optimizer, C: GradClip, S: LrSchedule> {
    optim: O,
    clip: C,
    schedule: S,
    accum: GradAccumulator,
    metrics: Box<dyn MetricSink>,
    cfg: TrainerConfig,
    step: u64,
}

impl<O: Optimizer, C: GradClip, S: LrSchedule> Trainer<O, C, S> {
    pub fn new(
        optim: O,
        clip: C,
        schedule: S,
        metrics: Box<dyn MetricSink>,
        cfg: TrainerConfig,
    ) -> Self {
        assert!(
            cfg.grad_accum_steps >= 1,
            "Trainer requires grad_accum_steps >= 1 (got 0)"
        );
        assert!(
            cfg.log_every >= 1,
            "Trainer requires log_every >= 1 (got 0)"
        );
        let accum = GradAccumulator::new(cfg.grad_accum_steps);
        Self {
            optim,
            clip,
            schedule,
            accum,
            metrics,
            cfg,
            step: 0,
        }
    }

    /// Current optimizer step index (0 until the first `step()`).
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Borrow the underlying optimizer for inspection (e.g. `optim.lr()` or
    /// `optim.export_state(...)`). Kept read-only so callers cannot bypass
    /// the loop's bookkeeping (LR ticking, moment updates).
    pub fn optim(&self) -> &O {
        &self.optim
    }

    /// Resume scalar + optimizer state from `cfg.resume_from`. The caller is
    /// responsible for loading model weights separately (the Trainer is
    /// architecture-agnostic). Returns the step index the next `run` will
    /// start at.
    ///
    /// Wraps `CheckpointError` into `AutogradError::TapeInvariant` with a
    /// `checkpoint: ...` prefix so the caller only has to handle a single
    /// error type. Schema mismatches between the saved doc and the live
    /// optimizer flow through the same wrapper.
    pub fn resume_if_configured(&mut self, name_map: &[(TensorId, String)]) -> Result<u64> {
        let Some(dir) = self.cfg.resume_from.as_ref() else {
            return Ok(0);
        };

        let (doc, optim_state) = load_trainer_state_v2(dir).map_err(wrap_checkpoint_err)?;

        let live_schema = self.optim.state_schema();
        if doc.optim_schema != live_schema {
            // Leak the message as a static string via a boxed leak is
            // allowed here because the Trainer is long-lived and this path
            // is hit at most once per run. We return a plain TapeInvariant
            // with a generic message and the schema details end up in logs
            // through the `wrap_checkpoint_err` fallback instead.
            eprintln!(
                "[trainer] optimizer schema mismatch on resume: saved={saved}, live={live}",
                saved = doc.optim_schema,
                live = live_schema,
            );
            return Err(AutogradError::TapeInvariant(
                "checkpoint: optimizer schema mismatch with live optimizer",
            ));
        }

        // P2 (codex review 2026-04-20): validate the saved LR schedule matches
        // the live one so a CLI flag flip between runs cannot silently resume
        // under a different schedule (e.g. switching `linear-warmup` for
        // `cosine-with-warmup` preserves steps but yields wrong LRs).
        let live_describe = self.schedule.describe();
        if doc.schedule_name != live_describe {
            eprintln!(
                "[trainer] lr schedule mismatch on resume: saved={saved:?}, live={live:?}",
                saved = doc.schedule_name,
                live = live_describe,
            );
            return Err(AutogradError::TapeInvariant(
                "checkpoint: lr schedule mismatch with live schedule (re-run with matching flags)",
            ));
        }

        self.optim
            .import_state(&optim_state, name_map)
            .map_err(|err| {
                // `anyhow::Error` from `import_state` — stringify into a
                // TapeInvariant so the Result type stays AutogradError.
                eprintln!("[trainer] optim import_state failed: {err}");
                AutogradError::TapeInvariant("checkpoint: optim.import_state failed")
            })?;

        self.step = doc.step;
        Ok(self.step)
    }

    /// Run `total_steps - resumed_step` optimizer steps, invoking `step_fn`
    /// for each micro-batch. Weights are NOT saved by this loop — the
    /// Trainer is architecture-agnostic. Binaries that want weight
    /// checkpoints must persist them from inside `step_fn` (or via a
    /// separate callback layered on top).
    ///
    /// `keep_extra` is the set of tensor ids the caller wants preserved
    /// across cleanup (model weights, frozen buffers, tokenizer caches —
    /// anything referenced by future micro-batches). The Trainer always
    /// keeps `params` + their `.grad` entries in addition. Toy tests can
    /// pass `HashSet::new()` when params == entire live set.
    pub fn run<F>(
        &mut self,
        store: &mut TensorStore,
        tape: &mut Tape,
        params: Vec<TensorId>,
        param_names: Vec<(TensorId, String)>,
        keep_extra: HashSet<TensorId>,
        step_fn: F,
    ) -> Result<()>
    where
        F: FnMut(&mut StepCtx<'_>) -> Result<StepOutcome>,
    {
        // Supply a no-op eval closure that is never invoked because
        // `cfg.eval_every` is read below; to be safe we also gate the call
        // itself on eval_every being set.
        let eval_fn = |_: &mut TensorStore, _: &mut Tape| -> Result<EvalOutcome> {
            Ok(EvalOutcome {
                loss: 0.0,
                token_count: 0,
            })
        };
        // Force eval_every off so the no-op eval closure is unreachable.
        let saved_eval_every = self.cfg.eval_every.take();
        let result = self.run_inner(
            store,
            tape,
            params,
            param_names,
            keep_extra,
            step_fn,
            eval_fn,
        );
        self.cfg.eval_every = saved_eval_every;
        result
    }

    /// Same as [`run`] but additionally calls `eval_fn` every `eval_every`
    /// optimizer steps and emits the eval loss as a metric sample.
    pub fn run_with_eval<F, E>(
        &mut self,
        store: &mut TensorStore,
        tape: &mut Tape,
        params: Vec<TensorId>,
        param_names: Vec<(TensorId, String)>,
        keep_extra: HashSet<TensorId>,
        step_fn: F,
        eval_fn: E,
    ) -> Result<()>
    where
        F: FnMut(&mut StepCtx<'_>) -> Result<StepOutcome>,
        E: FnMut(&mut TensorStore, &mut Tape) -> Result<EvalOutcome>,
    {
        self.run_inner(
            store,
            tape,
            params,
            param_names,
            keep_extra,
            step_fn,
            eval_fn,
        )
    }

    // Monomorphised shared body. `run` supplies a no-op eval closure plus
    // `eval_every = None` so the eval branch is never hit.
    fn run_inner<F, E>(
        &mut self,
        store: &mut TensorStore,
        tape: &mut Tape,
        params: Vec<TensorId>,
        param_names: Vec<(TensorId, String)>,
        keep_extra: HashSet<TensorId>,
        mut step_fn: F,
        mut eval_fn: E,
    ) -> Result<()>
    where
        F: FnMut(&mut StepCtx<'_>) -> Result<StepOutcome>,
        E: FnMut(&mut TensorStore, &mut Tape) -> Result<EvalOutcome>,
    {
        let n = self.cfg.grad_accum_steps;
        let loss_scale = 1.0_f32 / n as f32;

        // Rolling counters for `log_every`-gated metric emission.
        let mut log_timer = Instant::now();
        let mut tokens_since_last_log: u64 = 0;
        let mut steps_since_last_log: u64 = 0;

        while self.step < self.cfg.total_steps {
            let lr = self.schedule.lr(self.step);
            self.optim.set_lr(lr);

            // ---- micro-batch loop: accumulate grads ----
            let mut running_loss_sum: f32 = 0.0;
            let mut running_tokens: u64 = 0;
            let mut micro_idx: u64 = 0;

            loop {
                let outcome = {
                    let mut ctx = StepCtx {
                        step: self.step,
                        micro_idx,
                        loss_scale,
                        store,
                        tape,
                    };
                    step_fn(&mut ctx)?
                };

                let StepOutcome {
                    loss_id,
                    token_count,
                } = outcome;

                let scaled_id = if n > 1 {
                    mul_scalar(loss_id, loss_scale, store, tape)?
                } else {
                    loss_id
                };

                // Read back the scaled scalar for the logging sum. For a
                // scalar tensor `to_host` is a cheap 1-element copy.
                let host = store.to_host(scaled_id)?;
                let scaled_val = host.first().copied().unwrap_or(0.0);
                running_loss_sum += scaled_val;
                running_tokens += token_count;

                tape.backward(scaled_id, store)?;

                // P1 (codex review 2026-04-20): clear tape + prune the
                // TensorStore after each backward. Grads have already been
                // accumulated into param `.grad` fields by the backward pass,
                // so the per-micro activation graph is dead weight — leaving
                // it in place grows `store` (and therefore RSS) linearly
                // with `total_steps * grad_accum_steps`.
                cleanup_after_backward(store, tape, &params, &keep_extra);

                let ready = self.accum.observe_and_check_ready();
                micro_idx += 1;
                if ready {
                    break;
                }
            }

            // ---- optimizer step ----
            let grad_norm = self.clip.clip(store, &params)?;
            self.optim.step(store, &params)?;
            self.optim.zero_grad(store, &params);
            self.accum.reset_after_step();
            // Prune again after zero_grad so any temporaries allocated by
            // clip/step/zero_grad don't leak into the next step.
            cleanup_after_backward(store, tape, &params, &keep_extra);

            self.step += 1;
            tokens_since_last_log += running_tokens;
            steps_since_last_log += 1;

            // ---- metrics emission ----
            if self.step.is_multiple_of(self.cfg.log_every) {
                let elapsed = log_timer.elapsed();
                let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
                let elapsed_s = elapsed.as_secs_f64().max(1e-9);
                let tok_per_sec = tokens_since_last_log as f64 / elapsed_s;
                // Divide ms_per_step by the number of steps folded into
                // this window so the metric stays meaningful when
                // `log_every > 1`.
                let ms_per_step = elapsed_ms / steps_since_last_log.max(1) as f64;

                let fields: [(&str, f64); 6] = [
                    ("loss", running_loss_sum as f64),
                    ("lr", lr as f64),
                    ("grad_norm", grad_norm as f64),
                    ("ms_per_step", ms_per_step),
                    ("tok_per_sec", tok_per_sec),
                    ("step", self.step as f64),
                ];
                self.metrics.emit(&MetricSample {
                    step: self.step,
                    fields: &fields,
                });

                log_timer = Instant::now();
                tokens_since_last_log = 0;
                steps_since_last_log = 0;
            }

            // ---- eval ----
            if let Some(eval_n) = self.cfg.eval_every
                && self.step.is_multiple_of(eval_n)
            {
                let eval = eval_fn(store, tape)?;
                let fields: [(&str, f64); 3] = [
                    ("eval_loss", eval.loss as f64),
                    ("eval_tokens", eval.token_count as f64),
                    ("step", self.step as f64),
                ];
                self.metrics.emit(&MetricSample {
                    step: self.step,
                    fields: &fields,
                });
            }

            // ---- save ----
            if let Some(save_n) = self.cfg.save_every
                && self.step.is_multiple_of(save_n)
            {
                self.save_checkpoint(&param_names)?;
            }
        }

        self.metrics.flush();
        Ok(())
    }

    fn save_checkpoint(&self, param_names: &[(TensorId, String)]) -> Result<()> {
        let root = self.cfg.save_dir.as_ref().ok_or({
            AutogradError::TapeInvariant("checkpoint: save_every set but save_dir is None")
        })?;
        let dir = root.join(format!("step_{:06}", self.step));
        std::fs::create_dir_all(&dir).map_err(|err| {
            eprintln!("[trainer] create_dir_all({:?}) failed: {err}", &dir);
            AutogradError::TapeInvariant("checkpoint: create_dir_all failed")
        })?;

        let optim_state = self.optim.export_state(param_names);
        // Persist the full `describe()` so `resume_if_configured` can
        // reject mismatched schedule flags at load time. Per codex review
        // 2026-04-20 (P2) a bare prefix lost base_lr/warmup/total information
        // and silently let one schedule impersonate another on resume.
        let schedule_name = self.schedule.describe();

        let doc = TrainerStateDoc {
            step: self.step,
            optim_schema: self.optim.state_schema().to_string(),
            schedule_name,
            schedule_params: serde_json::json!({}),
            grad_accum_current: 0,
            rng_seed: self.cfg.rng_seed,
            codec_version: TRAINER_STATE_CODEC_VERSION,
        };

        save_trainer_state_v2(&dir, &doc, &optim_state).map_err(wrap_checkpoint_err)
    }
}

fn wrap_checkpoint_err(err: CheckpointError) -> AutogradError {
    eprintln!("[trainer] checkpoint error: {err}");
    AutogradError::TapeInvariant("checkpoint: v2 codec failure (see stderr)")
}

/// Post-backward cleanup: clear the tape, re-enable it for the next
/// micro-batch, then prune the store down to `keep_extra ∪ params ∪ grads`.
///
/// Matches the `tape.entries.clear(); tape.set_enabled(true);
/// store.retain_ids(...)` idiom used across the hand-written training
/// binaries (`pretrain.rs`, `train_sft.rs`, `train_grpo.rs`, …).
fn cleanup_after_backward(
    store: &mut TensorStore,
    tape: &mut Tape,
    params: &[TensorId],
    keep_extra: &HashSet<TensorId>,
) {
    tape.entries.clear();
    tape.set_enabled(true);
    let mut keep = keep_extra.clone();
    for &param_id in params {
        keep.insert(param_id);
        if let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) {
            keep.insert(grad_id);
        }
    }
    store.retain_ids(&keep);
}

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
    save_trainer_state_v2, write_latest_symlink,
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
    /// Emit a metric sample every N optimizer steps. Must be >= 1. The loop
    /// additionally force-emits on the very first step (step == 1) and the
    /// final step (step == total_steps) so CLI-consumers always see a first
    /// progress line and a closing summary regardless of `N`.
    pub log_every: u64,
    /// If `Some(n)`, call the eval closure (if provided) every N optimizer
    /// steps. Honored by `run_with_eval` and `run_with_eval_and_hooks`;
    /// ignored by `run` / `run_with_hooks` (those methods don't receive an
    /// eval closure and internally reset this field to `None` for the run).
    pub eval_every: Option<u64>,
    /// If `Some(n)`, save a checkpoint every N optimizer steps. Requires
    /// `save_dir` to be `Some`. The loop also force-saves on the final step
    /// (step == total_steps) — mirrors the metrics force-emit pattern so a
    /// training run that ends between save boundaries still produces a final,
    /// resumable checkpoint.
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
///
/// **Metric contract.** The Trainer emits `ppl = exp(loss)` alongside
/// `loss` on every training metric sample, treating `loss_id` as a
/// token-mean cross-entropy in natural-log space. Binaries that return a
/// non-CE scalar (MSE, rollout returns, contrastive losses) will still see
/// the `ppl` field populated with `exp(raw_scalar)`, which is mathematically
/// defined but semantically meaningless — downstream consumers should
/// ignore the `ppl` column in that case. See
/// `docs/plans/train-runtime-architecture-v1.md` §Phase 4.
pub struct StepOutcome {
    /// Scalar loss tensor in `store`/`tape`. The Trainer will `mul_scalar`
    /// this by `1/N` (when N > 1) and then call `tape.backward` on the
    /// scaled id. Interpreted as token-mean CE in nats for the `ppl` emit
    /// (see struct-level contract).
    pub loss_id: TensorId,
    /// Tokens processed in this micro-batch, for the `tok_per_sec` metric.
    pub token_count: u64,
}

/// What an eval closure returns.
///
/// **Metric contract.** Same as [`StepOutcome`]: the Trainer emits
/// `eval_ppl = exp(loss)` on eval metric samples, treating `loss` as a
/// token-mean CE in nats. Non-CE eval losses populate `eval_ppl` with a
/// mathematically defined but semantically meaningless number.
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
        //
        // Legacy compat (codex review 2026-04-20 on 3d9125d, P1): pre-P2
        // checkpoints stored a bare name like `"constant"` (the
        // `describe().split(['(', ' ']).next()` prefix). Accept that form when
        // its prefix matches the live `describe()` prefix so old checkpoints
        // still resume; strict full-describe match otherwise.
        let live_describe = self.schedule.describe();
        let live_prefix = live_describe
            .split(['(', ' '])
            .next()
            .unwrap_or(&live_describe);
        let saved = &doc.schedule_name;
        let saved_looks_legacy = !saved.contains('(') && !saved.contains(' ');
        let ok = saved == &live_describe || (saved_looks_legacy && saved == live_prefix);
        if !ok {
            eprintln!(
                "[trainer] lr schedule mismatch on resume: saved={saved:?}, live={live:?}",
                saved = doc.schedule_name,
                live = live_describe,
            );
            return Err(AutogradError::TapeInvariant(
                "checkpoint: lr schedule mismatch with live schedule (re-run with matching flags)",
            ));
        }

        // Codex review 2026-04-20 on d9eee61 (Medium): the checkpoint
        // persists `rng_seed` (see `TrainerStateDoc`), but the previous
        // version of this function never compared it to `self.cfg.rng_seed`.
        // Binaries now derive their sampler directly from the live CLI
        // `--seed` (e.g. the stateless `sample_index(seed, step, micro_step)`
        // in `train_sft`). If the operator resumes with a different `--seed`
        // than the interrupted run used, the sampler silently consumes a
        // different data stream from the resume step onward. Reject the
        // mismatch so the operator either restores the original seed or
        // starts a fresh run.
        if doc.rng_seed != self.cfg.rng_seed {
            eprintln!(
                "[trainer] rng_seed mismatch on resume: saved={saved}, live={live}",
                saved = doc.rng_seed,
                live = self.cfg.rng_seed,
            );
            return Err(AutogradError::TapeInvariant(
                "checkpoint: rng_seed mismatch with live --seed (re-run with matching --seed or start fresh)",
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
        self.run_with_hooks(
            store,
            tape,
            params,
            param_names,
            keep_extra,
            step_fn,
            |_, _| Ok(()),
        )
    }

    /// Same as [`run`] but calls `on_step_end(step, store)` after every
    /// optimizer step (after cleanup + LR tick, before the next step). Used
    /// by binaries that need to save model weights or tokenizer caches
    /// synchronously with the optimizer step — the Trainer's built-in
    /// `save_every` only writes `trainer_state.json + optimizer.safetensors`.
    pub fn run_with_hooks<F, H>(
        &mut self,
        store: &mut TensorStore,
        tape: &mut Tape,
        params: Vec<TensorId>,
        param_names: Vec<(TensorId, String)>,
        keep_extra: HashSet<TensorId>,
        step_fn: F,
        on_step_end: H,
    ) -> Result<()>
    where
        F: FnMut(&mut StepCtx<'_>) -> Result<StepOutcome>,
        H: FnMut(u64, &mut TensorStore) -> Result<()>,
    {
        let eval_fn = |_: &mut TensorStore, _: &mut Tape| -> Result<EvalOutcome> {
            Ok(EvalOutcome {
                loss: 0.0,
                token_count: 0,
            })
        };
        let saved_eval_every = self.cfg.eval_every.take();
        let result = self.run_inner(
            store,
            tape,
            params,
            param_names,
            keep_extra,
            step_fn,
            eval_fn,
            on_step_end,
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
            |_, _| Ok(()),
        )
    }

    /// Combined surface: both an `eval_fn` (invoked every `eval_every` steps)
    /// and an `on_step_end` hook (invoked after every optimizer step). Needed
    /// by binaries that own their own model-weight save pipeline AND want
    /// eval-loss metrics — splitting the two would force the caller to
    /// re-implement one or the other.
    pub fn run_with_eval_and_hooks<F, E, H>(
        &mut self,
        store: &mut TensorStore,
        tape: &mut Tape,
        params: Vec<TensorId>,
        param_names: Vec<(TensorId, String)>,
        keep_extra: HashSet<TensorId>,
        step_fn: F,
        eval_fn: E,
        on_step_end: H,
    ) -> Result<()>
    where
        F: FnMut(&mut StepCtx<'_>) -> Result<StepOutcome>,
        E: FnMut(&mut TensorStore, &mut Tape) -> Result<EvalOutcome>,
        H: FnMut(u64, &mut TensorStore) -> Result<()>,
    {
        self.run_inner(
            store,
            tape,
            params,
            param_names,
            keep_extra,
            step_fn,
            eval_fn,
            on_step_end,
        )
    }

    // Monomorphised shared body. `run` supplies a no-op eval closure plus
    // `eval_every = None` so the eval branch is never hit.
    fn run_inner<F, E, H>(
        &mut self,
        store: &mut TensorStore,
        tape: &mut Tape,
        params: Vec<TensorId>,
        param_names: Vec<(TensorId, String)>,
        keep_extra: HashSet<TensorId>,
        mut step_fn: F,
        mut eval_fn: E,
        mut on_step_end: H,
    ) -> Result<()>
    where
        F: FnMut(&mut StepCtx<'_>) -> Result<StepOutcome>,
        E: FnMut(&mut TensorStore, &mut Tape) -> Result<EvalOutcome>,
        H: FnMut(u64, &mut TensorStore) -> Result<()>,
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
            // Codex review 44a7e19 (medium): pre-migration train_sft always
            // logged step 1 and the final step regardless of `log_every`.
            // Force-emit on both boundaries so the CLI contract is preserved
            // (first progress line + final summary) across log_every values.
            let is_final = self.step == self.cfg.total_steps;
            let log_now =
                self.step == 1 || is_final || self.step.is_multiple_of(self.cfg.log_every);
            if log_now {
                let elapsed = log_timer.elapsed();
                let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
                let elapsed_s = elapsed.as_secs_f64().max(1e-9);
                let tok_per_sec = tokens_since_last_log as f64 / elapsed_s;
                // Divide ms_per_step by the number of steps folded into
                // this window so the metric stays meaningful when
                // `log_every > 1`.
                let ms_per_step = elapsed_ms / steps_since_last_log.max(1) as f64;

                // Codex review 44a7e19 (low): don't include "step" in the
                // fields array — MetricSample carries it in `sample.step`
                // already, and both StdoutSink and JsonlSink emit it from
                // there. Redundant "step" produced `step=5 ... step=5.000000`
                // in stdout + a duplicate JSON key.
                let loss_f64 = running_loss_sum as f64;
                // Phase 4: `ppl = exp(loss)` assumes `loss` is token-mean CE
                // in nats — see the StepOutcome metric contract. Non-CE
                // callers still get a numerically defined `ppl` field; it
                // is on them to ignore it. Overflow → +inf, which JsonlSink
                // null-falls back on via `Number::from_f64` and StdoutSink
                // prints readably.
                let fields: [(&str, f64); 6] = [
                    ("loss", loss_f64),
                    ("ppl", loss_f64.exp()),
                    ("lr", lr as f64),
                    ("grad_norm", grad_norm as f64),
                    ("ms_per_step", ms_per_step),
                    ("tok_per_sec", tok_per_sec),
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
            // Codex review 2026-04-20 on bd5e277 (Medium): also fire on the
            // final step so `--steps` not divisible by `--eval-every` still
            // gets the closing eval sample. Mirrors the save branch below.
            if let Some(eval_n) = self.cfg.eval_every
                && (self.step.is_multiple_of(eval_n) || is_final)
            {
                let eval = eval_fn(store, tape)?;
                // Codex review 2026-04-20 on bd5e277 (High): defensive
                // post-eval cleanup. Multi-forward eval closures (see
                // pretrain_qwen3's `--eval-windows N` path) accumulate
                // forward temporaries across windows; even single-call evals
                // can leave scratch tensors in the store. Prune down to
                // `params ∪ grads ∪ keep_extra` so the next training step
                // doesn't inherit eval temporaries.
                cleanup_after_backward(store, tape, &params, &keep_extra);
                // Codex review 44a7e19 (low): same redundant-step issue as the
                // training-metrics emit — drop "step" from fields, sinks read
                // it from sample.step.
                let eval_loss_f64 = eval.loss as f64;
                // Phase 4: same CE-in-nats contract as the training emit
                // above — see EvalOutcome doc.
                let fields: [(&str, f64); 3] = [
                    ("eval_loss", eval_loss_f64),
                    ("eval_ppl", eval_loss_f64.exp()),
                    ("eval_tokens", eval.token_count as f64),
                ];
                self.metrics.emit(&MetricSample {
                    step: self.step,
                    fields: &fields,
                });
            }

            // ---- save ----
            // Codex review 2026-04-20 on ad5568b (P1): force-save on the final
            // step too — otherwise a run with save_every=5 + total_steps=12
            // would save at 5/10 but drop the real "training done" state at
            // 12, leaving resume unable to pick up where the run ended.
            if let Some(save_n) = self.cfg.save_every
                && (self.step.is_multiple_of(save_n) || is_final)
            {
                self.save_checkpoint(&param_names)?;
            }

            // ---- post-step callback (weight save, custom logging, …) ----
            // Runs after the Trainer's own save_every hook so both the v2
            // state file and any binary-supplied weight file land in the
            // same checkpoint round.
            on_step_end(self.step, store)?;
        }

        self.metrics.flush();
        Ok(())
    }

    fn save_checkpoint(&self, param_names: &[(TensorId, String)]) -> Result<()> {
        let root = self.cfg.save_dir.as_ref().ok_or({
            AutogradError::TapeInvariant("checkpoint: save_every set but save_dir is None")
        })?;
        let step_basename = format!("step_{:06}", self.step);
        let dir = root.join(&step_basename);
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

        save_trainer_state_v2(&dir, &doc, &optim_state).map_err(wrap_checkpoint_err)?;

        // DX-1: Maintain `<save_dir>/latest` symlink pointing at this step so
        // `infer --model-path <out>/latest` and `--resume-from <out>/latest`
        // roundtrip without the caller knowing the step number. On non-unix
        // this is a no-op; symlink failures on unix surface as a hard error
        // because the pointer is part of the shipped DX contract.
        write_latest_symlink(root, &step_basename).map_err(|err| {
            eprintln!("[trainer] write_latest_symlink({:?}) failed: {err}", root);
            AutogradError::TapeInvariant("checkpoint: latest symlink failed")
        })?;

        Ok(())
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
///
/// Exposed `pub` so eval closures that produce multi-forward activations
/// (e.g. `pretrain_qwen3`'s `--eval-windows N` loop) can prune the store
/// between windows. Note: this unconditionally re-enables the tape, which
/// is correct for the post-backward path but NOT for an eval loop that
/// wants the tape disabled across windows — the caller must re-disable
/// with `tape.set_enabled(false)` after each invocation in that case.
pub fn cleanup_after_backward(
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

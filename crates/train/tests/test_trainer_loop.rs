//! Trainer loop integration tests. Toy model: a single parameter tensor `p`
//! with `loss = mean(p * p)`, so grad = 2*p/N. We verify loop semantics
//! (step counting, accumulation, LR ticking, resume, checkpoint dir, metric
//! emission) without pulling any Transformer wiring in.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use autograd::adamw_state::{AdamWParamState, AdamWState};
use autograd::ops::{mean, mul};
use autograd::{
    AdamW, ConstantLr, LinearWarmup, LrSchedule, Optimizer, Tape, Tensor, TensorId, TensorStore,
};
use tempfile::tempdir;
use train::checkpoint::{TRAINER_STATE_CODEC_VERSION, TrainerStateDoc, save_trainer_state_v2};
use train::grad_clip::NoClip;
use train::metrics::{MetricSample, MetricSink, NullSink};
use train::{StepCtx, StepOutcome, Trainer, TrainerConfig};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `TensorStore` with a single 1-D parameter initialised to `init`
/// and return its id.
fn setup_param(init: &[f32]) -> (TensorStore, TensorId) {
    let mut store = TensorStore::default();
    let id = store.alloc(Tensor::new(init.to_vec(), vec![init.len()], true).expect("alloc param"));
    (store, id)
}

/// Build `loss = mean(p * p)` and return the scalar tensor id.
fn squared_mean_loss(
    param: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> autograd::Result<TensorId> {
    let sq = mul(param, param, store, tape)?;
    mean(sq, store, tape)
}

/// Collecting `MetricSink` — stores every emit into an `Arc<Mutex<_>>` the
/// test can inspect after the run. The `Arc<Mutex<_>>` is `Send`, satisfying
/// the `MetricSink: Send` bound.
#[derive(Default)]
struct OwnedSample {
    step: u64,
    fields: Vec<(String, f64)>,
}

struct VecSink {
    buf: Arc<Mutex<Vec<OwnedSample>>>,
}

impl MetricSink for VecSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        let fields = sample
            .fields
            .iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect();
        self.buf.lock().unwrap().push(OwnedSample {
            step: sample.step,
            fields,
        });
    }
}

fn default_cfg(total_steps: u64) -> TrainerConfig {
    TrainerConfig {
        total_steps,
        grad_accum_steps: 1,
        log_every: 1,
        eval_every: None,
        save_every: None,
        save_dir: None,
        resume_from: None,
        rng_seed: 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// 5 optimizer steps, grad_accum=1 — verify the loop honours total_steps
/// and actually moves the parameter toward zero (gradient = 2*p, so AdamW
/// with any positive LR must decrease |p| monotonically on a quadratic).
#[test]
fn trainer_runs_total_steps() {
    let (mut store, p) = setup_param(&[1.0, -1.0, 0.5, -0.5]);
    let mut tape = Tape::new();

    let optim = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
    let cfg = default_cfg(5);
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-2), Box::new(NullSink), cfg);

    let initial_abs_sum: f32 = store.to_host(p).unwrap().iter().map(|v| v.abs()).sum();

    trainer
        .run(
            &mut store,
            &mut tape,
            vec![p],
            vec![(p, "p".to_string())],
            HashSet::new(),
            |ctx| {
                let loss = squared_mean_loss(p, ctx.store, ctx.tape)?;
                Ok(StepOutcome {
                    loss_id: loss,
                    token_count: 1,
                })
            },
        )
        .expect("trainer.run");

    assert_eq!(trainer.step(), 5, "trainer should have run 5 steps");
    let final_abs_sum: f32 = store.to_host(p).unwrap().iter().map(|v| v.abs()).sum();
    assert!(
        final_abs_sum < initial_abs_sum,
        "|p| should decrease ({initial_abs_sum} -> {final_abs_sum})"
    );
}

/// grad_accum_steps=4, total_steps=2 — step_fn must be invoked 8 times and
/// the optimizer step only 2 times. We prove the optimizer-step count by
/// inspecting AdamWState.step after the run (AdamW increments its internal
/// step counter once per `optim.step`).
#[test]
fn grad_accum_triggers_step_every_n() {
    let (mut store, p) = setup_param(&[0.3, -0.2]);
    let mut tape = Tape::new();

    let optim = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        total_steps: 2,
        grad_accum_steps: 4,
        log_every: 1,
        ..default_cfg(2)
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-2), Box::new(NullSink), cfg);

    let step_fn_calls = Arc::new(Mutex::new(0u64));
    let step_fn_calls_hook = Arc::clone(&step_fn_calls);

    trainer
        .run(
            &mut store,
            &mut tape,
            vec![p],
            vec![(p, "p".to_string())],
            HashSet::new(),
            |ctx: &mut StepCtx<'_>| {
                *step_fn_calls_hook.lock().unwrap() += 1;
                let loss = squared_mean_loss(p, ctx.store, ctx.tape)?;
                Ok(StepOutcome {
                    loss_id: loss,
                    token_count: 1,
                })
            },
        )
        .expect("trainer.run");

    assert_eq!(
        *step_fn_calls.lock().unwrap(),
        8,
        "step_fn should have been called 2 * 4 = 8 times"
    );
    assert_eq!(
        trainer.step(),
        2,
        "trainer should have run 2 optimizer steps"
    );
    let exported = trainer_optim_state(&trainer, p);
    assert_eq!(
        exported.step, 2,
        "AdamW internal step counter should match optimizer-step count"
    );
}

/// LinearWarmup(base_lr=0.1, warmup_steps=3): steps 0..3 are in-warmup
/// (lr < base_lr), steps >= 3 saturate at base_lr. After 5 steps we expect
/// `optim.lr() == 0.1`.
#[test]
fn lr_schedule_drives_optimizer_lr() {
    let (mut store, p) = setup_param(&[0.5]);
    let mut tape = Tape::new();

    let optim = AdamW::new(0.0, (0.9, 0.999), 1e-8, 0.0);
    let cfg = default_cfg(5);
    let schedule = LinearWarmup {
        base_lr: 0.1,
        warmup_steps: 3,
    };
    let mut trainer = Trainer::new(optim, NoClip, schedule, Box::new(NullSink), cfg);

    trainer
        .run(
            &mut store,
            &mut tape,
            vec![p],
            vec![(p, "p".to_string())],
            HashSet::new(),
            |ctx| {
                let loss = squared_mean_loss(p, ctx.store, ctx.tape)?;
                Ok(StepOutcome {
                    loss_id: loss,
                    token_count: 1,
                })
            },
        )
        .expect("trainer.run");

    // After step=5 we called `set_lr(schedule.lr(4))` last. LinearWarmup
    // with warmup_steps=3 saturates at base_lr once step >= warmup_steps.
    assert_eq!(
        trainer_lr(&trainer),
        0.1,
        "optimizer LR should saturate at base_lr after warmup"
    );
}

/// Codex review 2026-04-20 on ad5568b (P1): end-to-end save→resume
/// roundtrip test. Run trainer with save_every=N to land a real
/// checkpoint dir, then spin up a fresh trainer pointed at that dir and
/// confirm `resume_if_configured` accepts the output this code just
/// produced. Would have caught the format-mismatch + missing-file bugs
/// that shipped in ad5568b and were fixed in 49512b1.
#[test]
fn trainer_save_then_resume_roundtrip() {
    let tmp = tempdir().expect("tempdir");
    let save_dir: PathBuf = tmp.path().join("ckpt");

    // --- first run: train for 3 steps, save_every=3 so a checkpoint lands
    //     at `<save_dir>/step_000003/{trainer_state.json,optimizer.safetensors}`.
    let (mut store, p) = setup_param(&[0.25, -0.5]);
    let mut tape = Tape::new();
    let optim = AdamW::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        total_steps: 3,
        grad_accum_steps: 1,
        log_every: 1,
        eval_every: None,
        save_every: Some(3),
        save_dir: Some(save_dir.clone()),
        resume_from: None,
        rng_seed: 0,
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-3), Box::new(NullSink), cfg);
    trainer
        .run(
            &mut store,
            &mut tape,
            vec![p],
            vec![(p, "p".to_string())],
            HashSet::new(),
            |ctx| {
                let loss = squared_mean_loss(p, ctx.store, ctx.tape)?;
                Ok(StepOutcome {
                    loss_id: loss,
                    token_count: 1,
                })
            },
        )
        .expect("trainer.run (first pass)");

    let produced_dir = save_dir.join("step_000003");
    assert!(
        produced_dir.join("trainer_state.json").is_file(),
        "first-pass trainer must have written trainer_state.json"
    );
    assert!(
        produced_dir.join("optimizer.safetensors").is_file(),
        "first-pass trainer must have written optimizer.safetensors"
    );

    // --- second run: fresh trainer, `resume_from = <produced_dir>`.
    //     Should restore step to 3 so subsequent `run` starts from 3.
    let (_store_b, p_b) = setup_param(&[0.25, -0.5]);
    let optim_b = AdamW::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    let cfg_b = TrainerConfig {
        total_steps: 5,
        grad_accum_steps: 1,
        log_every: 1,
        eval_every: None,
        save_every: None,
        save_dir: None,
        resume_from: Some(produced_dir),
        rng_seed: 0,
    };
    let mut trainer_b = Trainer::new(optim_b, NoClip, ConstantLr(1e-3), Box::new(NullSink), cfg_b);
    let resumed = trainer_b
        .resume_if_configured(&[(p_b, "p".to_string())])
        .expect("resume_if_configured on own output");
    assert_eq!(resumed, 3, "should resume at step 3 from save_every=3");
    assert_eq!(trainer_b.step(), 3);
}

/// Hand-craft a TrainerStateDoc with step=42 + AdamWState that matches
/// param name "p", then resume. `resume_if_configured` should return 42
/// and `trainer.step()` should read 42.
#[test]
fn resume_restores_step() {
    let tmp = tempdir().expect("tempdir");
    let ckpt_dir = tmp.path().join("step_000042");
    std::fs::create_dir_all(&ckpt_dir).unwrap();

    // Must match the live `ConstantLr(1e-3).describe()` string exactly —
    // trainer.rs now validates the full describe() on resume (codex P2).
    let doc = TrainerStateDoc {
        step: 42,
        optim_schema: "adamw-v1".to_string(),
        schedule_name: ConstantLr(1e-3).describe(),
        schedule_params: serde_json::json!({}),
        grad_accum_current: 0,
        rng_seed: 7,
        codec_version: TRAINER_STATE_CODEC_VERSION,
    };
    let optim_state = AdamWState {
        step: 42,
        skipped_export: 0,
        params: vec![AdamWParamState {
            name: "p".to_string(),
            m: vec![0.1, -0.2, 0.3],
            v: vec![1e-3, 2e-3, 3e-3],
            shape: vec![3],
        }],
    };
    save_trainer_state_v2(&ckpt_dir, &doc, &optim_state).expect("save v2");

    let (_store, p) = setup_param(&[0.0, 0.0, 0.0]);
    let optim = AdamW::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    // `rng_seed: 7` matches the doc above. codex review 2026-04-20 on
    // d9eee61 (Medium): `resume_if_configured` now rejects a mismatch,
    // so this test exercises the happy-path seed-match branch.
    let cfg = TrainerConfig {
        resume_from: Some(ckpt_dir.clone()),
        rng_seed: 7,
        ..default_cfg(100)
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-3), Box::new(NullSink), cfg);

    let resumed = trainer
        .resume_if_configured(&[(p, "p".to_string())])
        .expect("resume_if_configured");
    assert_eq!(resumed, 42);
    assert_eq!(trainer.step(), 42);
}

/// save_every=2 + total_steps=2 — the run should write
/// `{save_dir}/step_000002/trainer_state.json`.
#[test]
fn checkpoint_save_writes_directory() {
    let tmp = tempdir().expect("tempdir");
    let save_dir: PathBuf = tmp.path().join("ckpt");

    let (mut store, p) = setup_param(&[0.25, -0.5]);
    let mut tape = Tape::new();
    let optim = AdamW::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        total_steps: 2,
        grad_accum_steps: 1,
        log_every: 1,
        eval_every: None,
        save_every: Some(2),
        save_dir: Some(save_dir.clone()),
        resume_from: None,
        rng_seed: 0,
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-3), Box::new(NullSink), cfg);

    trainer
        .run(
            &mut store,
            &mut tape,
            vec![p],
            vec![(p, "p".to_string())],
            HashSet::new(),
            |ctx| {
                let loss = squared_mean_loss(p, ctx.store, ctx.tape)?;
                Ok(StepOutcome {
                    loss_id: loss,
                    token_count: 1,
                })
            },
        )
        .expect("trainer.run");

    let expected = save_dir.join("step_000002").join("trainer_state.json");
    assert!(
        expected.is_file(),
        "expected checkpoint JSON at {expected:?}"
    );
    let optim_path = save_dir.join("step_000002").join("optimizer.safetensors");
    assert!(optim_path.is_file(), "expected optimizer safetensors");
}

/// Codex review 2026-04-20 on ad5568b (P1): the Trainer must force-save on
/// the final step even when it isn't a multiple of `save_every`. Otherwise
/// a run with `save_every=5 --steps 7` would save at step 5 only and lose
/// the 2 steps of progress between the last save boundary and termination.
#[test]
fn checkpoint_save_forces_final_step_when_not_multiple() {
    let tmp = tempdir().expect("tempdir");
    let save_dir: PathBuf = tmp.path().join("ckpt");

    let (mut store, p) = setup_param(&[0.25, -0.5]);
    let mut tape = Tape::new();
    let optim = AdamW::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        total_steps: 7,
        grad_accum_steps: 1,
        log_every: 1,
        eval_every: None,
        save_every: Some(5),
        save_dir: Some(save_dir.clone()),
        resume_from: None,
        rng_seed: 0,
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-3), Box::new(NullSink), cfg);

    trainer
        .run(
            &mut store,
            &mut tape,
            vec![p],
            vec![(p, "p".to_string())],
            HashSet::new(),
            |ctx| {
                let loss = squared_mean_loss(p, ctx.store, ctx.tape)?;
                Ok(StepOutcome {
                    loss_id: loss,
                    token_count: 1,
                })
            },
        )
        .expect("trainer.run");

    let step5 = save_dir.join("step_000005").join("trainer_state.json");
    let step7 = save_dir.join("step_000007").join("trainer_state.json");
    assert!(step5.is_file(), "expected step 5 checkpoint at {step5:?}");
    assert!(
        step7.is_file(),
        "expected forced final-step 7 checkpoint at {step7:?}"
    );
}

/// Codex review 2026-04-20 on bd6c871 (nit): the eval-path metrics emit
/// dropped `"step"` from its fields array alongside the training path, but
/// only the training side had a dedicated test. Pin the eval-side behavior
/// so a future regression cannot silently re-introduce `step=N ... step=N`
/// duplication in eval sinks.
#[test]
fn eval_metrics_fields_omit_step() {
    let buf: Arc<Mutex<Vec<OwnedSample>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = VecSink {
        buf: Arc::clone(&buf),
    };

    let (mut store, p) = setup_param(&[0.3]);
    let mut tape = Tape::new();
    let optim = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        total_steps: 4,
        grad_accum_steps: 1,
        log_every: 100, // suppress training emits except forced (1 + 4)
        eval_every: Some(2),
        save_every: None,
        save_dir: None,
        resume_from: None,
        rng_seed: 0,
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-2), Box::new(sink), cfg);

    trainer
        .run_with_eval(
            &mut store,
            &mut tape,
            vec![p],
            vec![(p, "p".to_string())],
            HashSet::new(),
            |ctx| {
                let loss = squared_mean_loss(p, ctx.store, ctx.tape)?;
                Ok(StepOutcome {
                    loss_id: loss,
                    token_count: 1,
                })
            },
            |_store, _tape| {
                Ok(train::EvalOutcome {
                    loss: 0.5,
                    token_count: 42,
                })
            },
        )
        .expect("trainer.run_with_eval");

    let samples = buf.lock().unwrap();
    // Identify eval samples by their distinctive key set.
    let eval_samples: Vec<&OwnedSample> = samples
        .iter()
        .filter(|s| s.fields.iter().any(|(k, _)| k == "eval_loss"))
        .collect();
    assert!(
        !eval_samples.is_empty(),
        "expected at least one eval sample (eval_every=2 total_steps=4)"
    );
    for s in eval_samples {
        let expected = ["eval_loss", "eval_tokens"];
        for k in expected {
            assert!(
                s.fields.iter().any(|(name, _)| name == k),
                "eval sample at step {} missing field {k}",
                s.step
            );
        }
        assert!(
            !s.fields.iter().any(|(name, _)| name == "step"),
            "eval sample at step {} still carries redundant `step` field",
            s.step
        );
    }
}

/// total_steps=4 + log_every=2 should produce exactly 2 metric samples.
#[test]
fn metrics_emit_at_log_every() {
    let buf: Arc<Mutex<Vec<OwnedSample>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = VecSink {
        buf: Arc::clone(&buf),
    };

    let (mut store, p) = setup_param(&[0.3]);
    let mut tape = Tape::new();
    let optim = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        total_steps: 4,
        grad_accum_steps: 1,
        log_every: 2,
        eval_every: None,
        save_every: None,
        save_dir: None,
        resume_from: None,
        rng_seed: 0,
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-2), Box::new(sink), cfg);

    trainer
        .run(
            &mut store,
            &mut tape,
            vec![p],
            vec![(p, "p".to_string())],
            HashSet::new(),
            |ctx| {
                let loss = squared_mean_loss(p, ctx.store, ctx.tape)?;
                Ok(StepOutcome {
                    loss_id: loss,
                    token_count: 7,
                })
            },
        )
        .expect("trainer.run");

    let samples = buf.lock().unwrap();
    // Codex review 44a7e19 (medium): the Trainer now always force-emits on
    // step 1 and on the final step in addition to the `log_every` rule, to
    // match the pre-migration train_sft CLI contract. So log_every=2 +
    // total_steps=4 yields 3 samples: step 1 (forced), step 2
    // (is_multiple_of), step 4 (is_multiple_of AND final).
    assert_eq!(
        samples.len(),
        3,
        "total_steps=4 / log_every=2 should emit 3 samples (1 forced + 2 + 4), got {}",
        samples.len()
    );
    assert_eq!(samples[0].step, 1);
    assert_eq!(samples[1].step, 2);
    assert_eq!(samples[2].step, 4);
    // Every sample carries the documented field set. `"step"` intentionally
    // not in this list anymore — sinks read it from `MetricSample.step`
    // (codex review 44a7e19 low).
    let expected_keys = ["loss", "lr", "grad_norm", "ms_per_step", "tok_per_sec"];
    for s in samples.iter() {
        for k in expected_keys {
            assert!(
                s.fields.iter().any(|(name, _)| name == k),
                "sample at step {} missing field {k}",
                s.step
            );
        }
        assert!(
            !s.fields.iter().any(|(name, _)| name == "step"),
            "sample at step {} still carries redundant `step` field",
            s.step
        );
    }
}

/// Codex review 44a7e19 (medium): the Trainer must always force-emit on
/// step 1 and the final step regardless of `log_every`. Pins the CLI
/// contract inherited from pre-migration train_sft so `--log-every 5
/// --steps 12` still produces a first progress line and a final summary,
/// not just steps 5 and 10.
#[test]
fn metrics_force_emit_on_first_and_final_step() {
    let buf: Arc<Mutex<Vec<OwnedSample>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = VecSink {
        buf: Arc::clone(&buf),
    };

    let (mut store, p) = setup_param(&[0.3]);
    let mut tape = Tape::new();
    let optim = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        total_steps: 12,
        grad_accum_steps: 1,
        log_every: 5,
        eval_every: None,
        save_every: None,
        save_dir: None,
        resume_from: None,
        rng_seed: 0,
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-2), Box::new(sink), cfg);

    trainer
        .run(
            &mut store,
            &mut tape,
            vec![p],
            vec![(p, "p".to_string())],
            HashSet::new(),
            |ctx| {
                let loss = squared_mean_loss(p, ctx.store, ctx.tape)?;
                Ok(StepOutcome {
                    loss_id: loss,
                    token_count: 3,
                })
            },
        )
        .expect("trainer.run");

    let samples = buf.lock().unwrap();
    // Expected: step 1 (forced), step 5 (log_every), step 10 (log_every),
    // step 12 (forced final). Final is not a multiple of 5, so the force
    // rule is load-bearing here.
    let emitted_steps: Vec<u64> = samples.iter().map(|s| s.step).collect();
    assert_eq!(
        emitted_steps,
        vec![1, 5, 10, 12],
        "expected forced emits on step 1 and final (12) plus log_every=5"
    );
}

/// `run_with_hooks` must invoke `on_step_end` exactly once per optimizer step
/// (never per micro-batch), after cleanup, with the post-update step index.
/// This is the hook train_sft uses to save bf16 model weights.
#[test]
fn run_with_hooks_fires_after_each_optimizer_step() {
    let (mut store, p) = setup_param(&[0.1, 0.2]);
    let mut tape = Tape::new();

    let optim = AdamW::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        total_steps: 3,
        grad_accum_steps: 2, // ensure hook is NOT called per micro-batch
        log_every: 1,
        ..default_cfg(3)
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-3), Box::new(NullSink), cfg);

    let hook_calls: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let hook_calls_hook = Arc::clone(&hook_calls);

    trainer
        .run_with_hooks(
            &mut store,
            &mut tape,
            vec![p],
            vec![(p, "p".to_string())],
            HashSet::new(),
            |ctx| {
                let loss = squared_mean_loss(p, ctx.store, ctx.tape)?;
                Ok(StepOutcome {
                    loss_id: loss,
                    token_count: 1,
                })
            },
            |step, _store| {
                hook_calls_hook.lock().unwrap().push(step);
                Ok(())
            },
        )
        .expect("run_with_hooks");

    let observed = hook_calls.lock().unwrap();
    assert_eq!(
        observed.as_slice(),
        &[1, 2, 3],
        "hook must fire once per optimizer step with post-update index"
    );
}

/// Codex review P1 (2026-04-20): the Trainer loop must prune `TensorStore`
/// after every backward + every optimizer step so activation allocations
/// don't grow linearly with `total_steps * grad_accum_steps`. The toy
/// `mean(p * p)` forward allocates at least one temporary (`sq = p * p`)
/// that must be gone by the time we read the store after the run.
#[test]
fn trainer_cleans_up_activations_after_each_step() {
    let (mut store, p) = setup_param(&[0.25, -0.5, 0.1]);
    let mut tape = Tape::new();

    // Snapshot the live-id count before the loop — this is the baseline
    // the cleanup target must respect (param + its eventual grad only).
    let pre_run_live_ids: usize = (0..store.tensors.len())
        .filter(|&i| store.get(i).is_some())
        .count();

    let optim = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        total_steps: 20,
        grad_accum_steps: 3,
        log_every: 1,
        ..default_cfg(20)
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-2), Box::new(NullSink), cfg);

    trainer
        .run(
            &mut store,
            &mut tape,
            vec![p],
            vec![(p, "p".to_string())],
            HashSet::new(),
            |ctx| {
                let loss = squared_mean_loss(p, ctx.store, ctx.tape)?;
                Ok(StepOutcome {
                    loss_id: loss,
                    token_count: 1,
                })
            },
        )
        .expect("trainer.run");

    // Post-run: tape must be empty (never carries forward across steps) and
    // the live-id count bounded — param (+ its grad) is the only thing
    // we guarantee; everything else must have been pruned. 20 steps × 3
    // micro would be ~60 allocations if cleanup silently regressed.
    assert!(tape.entries.is_empty(), "tape must be empty after run");
    let post_run_live_ids: usize = (0..store.tensors.len())
        .filter(|&i| store.get(i).is_some())
        .count();
    assert!(
        post_run_live_ids <= pre_run_live_ids + 2,
        "store grew unboundedly: pre={pre_run_live_ids}, post={post_run_live_ids} (expected <= pre+2 for param+grad)"
    );
}

/// Codex review P2 (2026-04-20): a saved schedule_name that doesn't match the
/// live schedule's `describe()` must cause `resume_if_configured` to error
/// rather than silently resuming under mismatched LR flags.
#[test]
fn resume_rejects_mismatched_schedule() {
    let tmp = tempdir().expect("tempdir");
    let ckpt_dir = tmp.path().join("step_000010");
    std::fs::create_dir_all(&ckpt_dir).unwrap();

    // Save a doc that claims the schedule was `LinearWarmup(base_lr=0.1,
    // warmup_steps=5)` …
    let doc = TrainerStateDoc {
        step: 10,
        optim_schema: "adamw-v1".to_string(),
        schedule_name: LinearWarmup {
            base_lr: 0.1,
            warmup_steps: 5,
        }
        .describe(),
        schedule_params: serde_json::json!({}),
        grad_accum_current: 0,
        rng_seed: 0,
        codec_version: TRAINER_STATE_CODEC_VERSION,
    };
    let optim_state = AdamWState {
        step: 10,
        skipped_export: 0,
        params: vec![AdamWParamState {
            name: "p".to_string(),
            m: vec![0.0],
            v: vec![0.0],
            shape: vec![1],
        }],
    };
    save_trainer_state_v2(&ckpt_dir, &doc, &optim_state).expect("save v2");

    // … but construct the live trainer with a *different* schedule.
    let (_store, p) = setup_param(&[0.0]);
    let optim = AdamW::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        resume_from: Some(ckpt_dir.clone()),
        ..default_cfg(100)
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-3), Box::new(NullSink), cfg);

    let err = trainer
        .resume_if_configured(&[(p, "p".to_string())])
        .expect_err("mismatched schedule must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("lr schedule mismatch"),
        "expected schedule-mismatch error, got: {msg}"
    );
    assert_eq!(
        trainer.step(),
        0,
        "failed resume must leave trainer.step() at 0"
    );
}

/// Codex review 2026-04-20 on d9eee61 (Medium): resuming with a different
/// `--seed` than the interrupted run used would silently consume a different
/// data stream (the sampler is derived directly from the live seed).
/// `resume_if_configured` must reject the mismatch.
#[test]
fn resume_rejects_mismatched_rng_seed() {
    let tmp = tempdir().expect("tempdir");
    let ckpt_dir = tmp.path().join("step_000020");
    std::fs::create_dir_all(&ckpt_dir).unwrap();

    let doc = TrainerStateDoc {
        step: 20,
        optim_schema: "adamw-v1".to_string(),
        schedule_name: ConstantLr(1e-3).describe(),
        schedule_params: serde_json::json!({}),
        grad_accum_current: 0,
        // Saved with seed=123…
        rng_seed: 123,
        codec_version: TRAINER_STATE_CODEC_VERSION,
    };
    let optim_state = AdamWState {
        step: 20,
        skipped_export: 0,
        params: vec![AdamWParamState {
            name: "p".to_string(),
            m: vec![0.0],
            v: vec![0.0],
            shape: vec![1],
        }],
    };
    save_trainer_state_v2(&ckpt_dir, &doc, &optim_state).expect("save v2");

    // … resumed with seed=999 (live TrainerConfig.rng_seed = 999).
    let (_store, p) = setup_param(&[0.0]);
    let optim = AdamW::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        resume_from: Some(ckpt_dir.clone()),
        rng_seed: 999,
        ..default_cfg(100)
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-3), Box::new(NullSink), cfg);

    let err = trainer
        .resume_if_configured(&[(p, "p".to_string())])
        .expect_err("mismatched rng_seed must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("rng_seed mismatch"),
        "expected rng_seed-mismatch error, got: {msg}"
    );
    assert_eq!(
        trainer.step(),
        0,
        "failed resume must leave trainer.step() at 0"
    );
}

/// Legacy-compat (codex review 3d9125d P1): checkpoints written before the P2
/// describe()-full-match rollout stored a bare prefix (e.g. `"constant"`).
/// Those must still resume when the live schedule's describe() starts with
/// that same prefix, otherwise a v2-codec bump would be required.
#[test]
fn resume_accepts_legacy_bare_schedule_name() {
    let tmp = tempdir().expect("tempdir");
    let ckpt_dir = tmp.path().join("step_000007");
    std::fs::create_dir_all(&ckpt_dir).unwrap();

    let doc = TrainerStateDoc {
        step: 7,
        optim_schema: "adamw-v1".to_string(),
        // Legacy bare-prefix format — pre-P2 writers produced exactly this.
        schedule_name: "constant".to_string(),
        schedule_params: serde_json::json!({}),
        grad_accum_current: 0,
        rng_seed: 0,
        codec_version: TRAINER_STATE_CODEC_VERSION,
    };
    let optim_state = AdamWState {
        step: 7,
        skipped_export: 0,
        params: vec![AdamWParamState {
            name: "p".to_string(),
            m: vec![0.0],
            v: vec![0.0],
            shape: vec![1],
        }],
    };
    save_trainer_state_v2(&ckpt_dir, &doc, &optim_state).expect("save v2");

    let (_store, p) = setup_param(&[0.0]);
    let optim = AdamW::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        resume_from: Some(ckpt_dir.clone()),
        ..default_cfg(100)
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-3), Box::new(NullSink), cfg);

    let resumed = trainer
        .resume_if_configured(&[(p, "p".to_string())])
        .expect("legacy bare prefix must still resume");
    assert_eq!(resumed, 7);
    assert_eq!(trainer.step(), 7);
}

/// Second legacy-compat case (codex review on bdde441 suggestion): make sure
/// the prefix path also accepts `"linear-warmup"` against a live schedule
/// whose `describe()` is `"linear-warmup(base_lr=..., warmup=...)"`. Pins
/// behavior for the non-constant branch of `parse_lr_schedule`.
#[test]
fn resume_accepts_legacy_bare_linear_warmup_name() {
    let tmp = tempdir().expect("tempdir");
    let ckpt_dir = tmp.path().join("step_000003");
    std::fs::create_dir_all(&ckpt_dir).unwrap();

    let doc = TrainerStateDoc {
        step: 3,
        optim_schema: "adamw-v1".to_string(),
        schedule_name: "linear-warmup".to_string(),
        schedule_params: serde_json::json!({}),
        grad_accum_current: 0,
        rng_seed: 0,
        codec_version: TRAINER_STATE_CODEC_VERSION,
    };
    let optim_state = AdamWState {
        step: 3,
        skipped_export: 0,
        params: vec![AdamWParamState {
            name: "p".to_string(),
            m: vec![0.0],
            v: vec![0.0],
            shape: vec![1],
        }],
    };
    save_trainer_state_v2(&ckpt_dir, &doc, &optim_state).expect("save v2");

    let (_store, p) = setup_param(&[0.0]);
    let optim = AdamW::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        resume_from: Some(ckpt_dir.clone()),
        ..default_cfg(100)
    };
    let live = LinearWarmup {
        base_lr: 1e-3,
        warmup_steps: 5,
    };
    let mut trainer = Trainer::new(optim, NoClip, live, Box::new(NullSink), cfg);

    let resumed = trainer
        .resume_if_configured(&[(p, "p".to_string())])
        .expect("legacy bare linear-warmup prefix must still resume");
    assert_eq!(resumed, 3);
    assert_eq!(trainer.step(), 3);
}

// ---------------------------------------------------------------------------
// Small shims on top of `Trainer::optim()` so the assertion-time call sites
// stay focused on what they're proving.
// ---------------------------------------------------------------------------

fn trainer_optim_state(trainer: &Trainer<AdamW, NoClip, ConstantLr>, p: TensorId) -> AdamWState {
    trainer.optim().export_state(&[(p, "p".to_string())])
}

fn trainer_lr(trainer: &Trainer<AdamW, NoClip, LinearWarmup>) -> f32 {
    trainer.optim().lr()
}

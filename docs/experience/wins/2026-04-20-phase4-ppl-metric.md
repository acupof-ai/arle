# Phase 4: perplexity derived from loss in Trainer metric pipeline

## Context

Phase 4 of `docs/plans/train-runtime-architecture-v1.md`: eval +
observability tightening. Trainer already emits `loss`, `lr`,
`grad_norm`, `ms_per_step`, `tok_per_sec` on training samples and
`eval_loss`, `eval_tokens` on eval samples. Perplexity was the
cheapest remaining item: pure metric derivation on the loss value
the Trainer already reads back to host for logging, no hot-path
numerical change.

## What Worked

- `trainer.rs` metric-emit sites now compute `ppl = loss.exp()`
  alongside `loss` (training) and `eval_ppl = eval_loss.exp()`
  alongside `eval_loss` (eval). Both arrays widen by one field.
- `running_loss_sum` is already the micro-batch-averaged per-step
  cross-entropy (`loss_scale = 1.0 / grad_accum_steps` applied
  inside the accumulation loop), so `exp(loss)` is the step
  perplexity directly — no additional averaging layer needed.
- Non-finite handling inherited from existing sinks: `JsonlSink`
  null-falls back through `serde_json::Number::from_f64` so inf/NaN
  becomes `null` (file stays parseable); `StdoutSink` formats `inf`
  readably. No new guard required, even if a cold-start loss
  overflows `f64::exp`.
- Smoke (`cargo run --release -p train --bin pretrain --
  --dataset copy --steps 5 --vocab-size 300 --log-every 1`):
  loss=5.76→5.27→4.95→4.60→4.29 produces ppl=317→194→141→99→73,
  matching `loss.exp()` to the displayed precision.
- Tests: 21 `test_trainer_loop` green (was 20). New
  `ppl_metric_equals_exp_of_loss_on_every_sample` drives
  `run_with_eval` with `total_steps=3 / log_every=1 / eval_every=3`
  and asserts `ppl == exp(loss)` on every training sample + 
  `eval_ppl == exp(eval_loss)` on the eval sample (tolerance
  `1e-12 * |expected|` to survive f64 round-trip).
  `metrics_emit_at_log_every` + `eval_metrics_fields_omit_step`
  widened their `expected_keys` arrays to include the new fields.
- `cargo clippy --release -p train --no-default-features
  -- -D warnings` clean.

## Rule

**Derive, don't recompute.** When a metric is a pure function of one
the Trainer already emits, derive it at the emit site — don't push
the derivation into every sink consumer or every downstream tool.
Sinks should see the full, already-computed field set; non-finite
handling is the sink's job, not the metric producer's.

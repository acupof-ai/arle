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
  `ppl_field_equals_exp_of_loss_field_in_metric_plumbing` (renamed
  from `ppl_metric_equals_exp_of_loss_on_every_sample` in cbdd7f9
  to reflect that it pins mechanical derivation, not semantic ppl
  correctness) drives `run_with_eval` with `total_steps=3 /
  log_every=1 / eval_every=3` and asserts `ppl == exp(loss)` on
  every training sample + `eval_ppl == exp(eval_loss)` on the eval
  sample (tolerance `1e-12 * |expected|` to survive f64 round-trip).
  `metrics_emit_at_log_every` + `eval_metrics_fields_omit_step`
  widened their `expected_keys` arrays to include the new fields.
- `cargo clippy --release -p train --no-default-features
  -- -D warnings` clean.

### Follow-ups (same day)

- **cbdd7f9** — CE-in-nats contract documented at three layers
  (`StepOutcome` / `EvalOutcome` docstrings, trainer emit-site
  comments, `docs/plans/train-runtime-architecture-v1.md` Phase 4
  row). Closes the 83c6ed2 codex review Medium finding: `ppl =
  exp(loss)` is only semantically valid when `loss` is token-mean
  cross-entropy in natural-log space; non-CE callers still get a
  numerically defined `ppl` field but must ignore it. The test was
  renamed to make the plumbing-vs-semantics distinction obvious.
- **60f7183** — `--metrics-jsonl` extended to cover the GRPO phase
  of `train_grpo`. Added `JsonlSink::open_append` (OpenOptions
  append+create) + `open_sink_append` factory, so the hand-written
  GRPO loop extends the JSONL that `run_sft_phase`'s Trainer
  already truncated + wrote, instead of clobbering it. GRPO
  samples chain step as `sft_steps + iter + 1` so downstream
  tooling sorts cleanly. Existing `grpo iter N:` stdout line kept
  as human contract (`also_stdout=false` on the GRPO sink).
- **2dd8607** — Two tests lock in the truncate-vs-append contract:
  `jsonl_sink_open_append_extends_existing_file` (direct API) +
  `open_sink_append_factory_extends_and_creates` (factory path,
  exactly what `train_grpo` calls). Codex review clean (no
  findings) on both 60f7183 and 2dd8607.

## Rule

**Derive, don't recompute.** When a metric is a pure function of one
the Trainer already emits, derive it at the emit site — don't push
the derivation into every sink consumer or every downstream tool.
Sinks should see the full, already-computed field set; non-finite
handling is the sink's job, not the metric producer's.

**Document the precondition at the emit site, not only in tests.**
When a derived metric (`ppl = exp(loss)`) only makes semantic sense
under a contract on its input (loss in nats, token-mean CE),
document that contract next to the emit code — callers who reuse
the Trainer under a different loss shouldn't have to reverse-engineer
it from a test name. Tests pin mechanical plumbing; doc comments
pin semantics.

**Multi-phase binaries truncate once, extend thereafter.** When a
binary runs two phases and both want to write to the same JSONL,
the second phase must `open_append`, not `create`. Otherwise
phase-1 samples silently vanish on phase-2 startup. Test coverage
must assert append, not just "3 lines exist" (a truncate regression
that writes 3 phase-2 samples would pass that assertion).

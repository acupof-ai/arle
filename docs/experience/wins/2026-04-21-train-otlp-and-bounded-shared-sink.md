# Train OTLP And Bounded Shared Sink

## Context

The train-side observability surface already had JSONL/stdout, `/v1/train/events`,
and an MLflow adapter, but two gaps remained:

- no vendor-neutral remote export path for collectors / Phoenix-style stacks
- no explicit backpressure policy inside `SharedSink`; metric emission used an
  unbounded channel

This left the runtime API in a good shape but the export layer still too
provider-specific and too trusting of queue growth.

## What Worked

- Added `OtlpLogSink` on top of the existing `MetricSink` / `TrainEvent`
  surface, using OTLP/HTTP logs so the train worker can export structured
  metric + lifecycle records to any compatible collector without changing the
  hot path.
- Kept the same shared async architecture: `Trainer` and the hand-written RL
  loops still emit into `SharedSink`; OTLP sits behind that adapter boundary
  rather than becoming a new runtime API.
- Tightened `SharedSink` from an unbounded channel to a bounded queue with an
  explicit overload policy:
  - scalar metrics use `try_send` and may drop with a warning counter
  - lifecycle / artifact events still block into the queue so `checkpoint` and
    `run_end` are not silently lost
- End-to-end OTLP smoke worked for both supervised and RL entrypoints using a
  local HTTP mock:
  - `pretrain` emitted OTLP log requests and completed with a checkpoint
  - `train_grpo` emitted OTLP log requests across SFT + GRPO and completed with
    a checkpoint

## Verification

- `cargo test -p train --release`
- `cargo clippy -p train --all-targets -- -D warnings`
- `python3` local OTLP mock + `cargo run -p train --release --bin pretrain -- ...`
- `python3` local OTLP mock + `cargo run -p train --release --bin train_grpo -- ...`

## Rule

The train runtime API is the event stream. Remote observability vendors sit
behind adapters on the shared async sink, and that sink must always have an
explicit backpressure policy instead of unbounded growth.

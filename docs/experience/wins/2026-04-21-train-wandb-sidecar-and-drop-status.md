# Train W&B Sidecar And Drop Status

## Context

The train-side observability stack already had:

- local JSONL / stdout as the canonical truth
- shared async export via `SharedSink`
- MLflow metrics + checkpoint artifacts
- OTLP/HTTP log export

The remaining practical gap was W&B. The official docs recommend using the SDK
with `WANDB_MODE=offline` for disconnected machines and syncing later with
`wandb sync`, while W&B itself already writes asynchronously and keeps a local
run directory. Re-implementing the `.wandb` datastore in Rust would have been
the wrong abstraction.

## What Worked

- Added `WandbProcessSink` as an optional adapter under `crates/train/src/metrics.rs`.
- Kept the hot path Rust-only: train loops still emit `MetricSample` /
  `TrainEvent` into `SharedSink`; the W&B adapter runs as a background sidecar
  process around the official SDK.
- Defaulted the sidecar to `WANDB_MODE=offline` and made it consume the same
  lifecycle/checkpoint events already used by MLflow and OTLP.
- Surfaced bounded-queue loss into the live control plane by reporting
  `dropped_metrics` in `/v1/train/status` and terminal `run_end` events.
- Added a repo-local helper script at
  `crates/train/scripts/wandb_sink_helper.py` instead of embedding Python into
  `Trainer`.

## Rule

When an observability vendor already owns a local/offline persistence format,
adapt to the official SDK off the hot path instead of cloning that format in
Rust. The runtime API remains the train event stream; vendors stay adapters.

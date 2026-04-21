# Train Observability v1

Last updated: 2026-04-21

## Goal

Give the Rust training stack a single observability/export contract that:

- keeps the training loop non-blocking and Rust-native
- preserves the current lightweight `MetricSink` fast path
- supports experiment tracking and artifact lineage for W&B / MLflow
- supports vendor-neutral telemetry export through OpenTelemetry / OTLP
- leaves room for trace-first LLM observability stacks (Phoenix, Langfuse,
  Braintrust) without hard-coding a SaaS SDK into the core trainer

This is an architecture plan, not a declaration that every exporter has
already landed.

## Current state

Today the train-side observability surface is intentionally minimal:

- `crates/train/src/metrics.rs` defines `MetricSample { step, fields }`,
  `MetricSink`, `NullSink`, `StdoutSink`, `JsonlSink`, and `MultiSink`.
- `Trainer` emits shared supervised metrics (`loss`, `ppl`, `lr`,
  `grad_norm`, `ms_per_step`, `tok_per_sec`, `eval_*`).
- `train_grpo` appends its RL metrics to JSONL during the GRPO phase.
- `train_multi_turn` emits its own RL metrics and exposes the only live train
  control plane (`/v1/train/status|stop|save`).

What is missing:

- no run lifecycle (`run_start`, `run_end`)
- no checkpoint/artifact events
- no stable run metadata (`run_id`, `backend`, `model_family`, tags, config)
- no timestamped event schema
- no async exporter thread / queue / retry policy
- no external vendor integration surface

## Constraints

- Rust-only hot path: no Python dependency in the training loop.
- Export must not stall training; remote I/O belongs on a background worker.
- Checkpoint directories remain the artifact truth (`model.safetensors`,
  `optimizer.safetensors`, `trainer_state.json`, `config.json`,
  `tokenizer.json`).
- Metric emission stays host-scalar and post-step; do not pull extra device
  syncs into the loop just for logging.

## Industry fit

The current tool ecosystem splits into two practical buckets:

1. Experiment tracking + artifacts
   - W&B
   - MLflow
2. Trace-first LLM / agent observability
   - OpenTelemetry / OTLP with GenAI semantic conventions
   - Phoenix / OpenInference
   - Langfuse
   - Braintrust

For this repository, the right order is:

- first stabilize a train event/export contract
- then ship vendor-neutral OTLP export
- then add experiment/artifact adapters for W&B and MLflow
- then decide which trace-first stack to target for eval / agent traces

This avoids baking provider-specific assumptions into `Trainer`.

## Proposed surface

Keep `MetricSink` for the hot-path scalar case, but add a higher-level
exporter surface above it:

```rust
pub enum TrainEvent<'a> {
    RunStart(RunMeta<'a>),
    Metric(MetricEvent<'a>),
    Checkpoint(CheckpointEvent<'a>),
    Status(StatusEvent<'a>),
    RunEnd(RunSummary<'a>),
}

pub trait TrainEventSink: Send {
    fn emit(&mut self, event: &TrainEvent<'_>);
    fn flush(&mut self) {}
}
```

### Required event payloads

`RunMeta`
- `run_id`
- `job_kind` (`pretrain`, `sft`, `grpo`, `multi_turn`, `eval`)
- `backend`
- `model_family`
- `model_path` / base checkpoint
- flattened config / tags
- git commit when available

`MetricEvent`
- `step`
- `phase` (`train`, `eval`, `rollout`, `reward`, `optimizer`)
- timestamp
- scalar fields

`CheckpointEvent`
- `step`
- output directory
- paths to `model.safetensors`, `adapter_model.safetensors`,
  `optimizer.safetensors`, `trainer_state.json`, `tokenizer.json`
- optional metadata (`merged=true`, `reference_model=true`, etc.)

`StatusEvent`
- save / stop / resume notifications from the control plane

`RunSummary`
- final status (`completed`, `stopped`, `failed`)
- wall time
- final/best metrics

## Export architecture

### 1. Canonical local truth

Every run continues to write:

- stdout
- JSONL metrics
- checkpoint directories

This remains the local source of truth and the fallback when remote export is
disabled or unavailable.

### 2. Async exporter worker

`Trainer` and hand-written RL loops should push `TrainEvent`s into a bounded
channel. A background worker owns remote export:

- batching
- retries/backoff
- rate limiting
- network failure isolation
- final flush on shutdown

If the queue fills, the policy should be explicit and safe:

- metrics may drop with a warning counter
- checkpoint and run-end events must block briefly or spool to disk

### 3. Vendor-neutral first

OpenTelemetry / OTLP should be the first remote target:

- MLflow explicitly supports OpenTelemetry-compatible tracing for GenAI apps
- Phoenix is built on OpenTelemetry + OpenInference
- OTel GenAI semantic conventions now cover events, metrics, model spans, and
  agent spans

For train-side scalar metrics, OTLP gives a vendor-neutral wire format.
Artifact lineage still needs explicit checkpoint events.

### 4. Experiment tracking adapters

W&B and MLflow both need:

- run metadata/config
- step metrics
- checkpoint artifacts
- final summary

Best practice here is not to embed Python into `Trainer`.
Instead:

- expose a stable event stream in Rust
- implement vendor adapters as optional sinks or sidecars
- keep them outside the hot path

## Recommended rollout

### Phase A — schema stabilization

- Add `TrainEvent` and lifecycle metadata.
- Unify metric names across `Trainer`, `train_grpo`, and `train_multi_turn`.
- Emit eval metrics through the same sink path instead of `println!`.

### Phase B — async export

- Add a bounded event queue and background worker.
- Keep JSONL/stdout as local truth.
- Add queue-depth and dropped-event counters.

### Phase C — OTLP

- Add `OtlpSink` for metrics/logs/traces as appropriate.
- Map scalar training metrics into OTLP metrics/log records.
- Map run/checkpoint lifecycle into structured log records or spans.

### Phase D — experiment tracking

- Add W&B adapter:
  - run config
  - step metrics
  - checkpoint artifacts
  - final summary
- Add MLflow adapter with the same contract.

### Phase E — trace-first agent observability

Use the same event model for:

- eval runs
- reward-model traces
- agent multi-turn rollouts
- tool/action traces

This is where Phoenix / Langfuse / Braintrust become most valuable. They are
less urgent for scalar supervised training than for agent/eval workflows.

## Immediate implementation target

The smallest safe next step is:

1. widen the event schema beyond `step + &[(&str, f64)]`
2. normalize metrics across `Trainer`, `train_grpo`, and `train_multi_turn`
3. introduce a background exporter thread
4. make checkpoint save emit artifact events

Only after that should we wire W&B / MLflow / OTLP endpoints.

## Rule

Do not let any single observability vendor become the training runtime API.
The runtime API is the event stream; vendors are adapters.

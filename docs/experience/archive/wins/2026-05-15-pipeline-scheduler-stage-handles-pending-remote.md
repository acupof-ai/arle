# CUDA pipeline scheduler stage handles - 2026-05-15

## Goal

- Continue P3 of the CPU/GPU pipeline plan by turning scheduler-owned pending
  prefill and readback work into typed GPU stage handles that expose stage
  lifecycle state and request-local fence ownership.

## Hypothesis

- The scheduler already has the right high-level shape: prefill and decode
  readback can stay pending across loop turns while CPU planning, admission,
  and emit work continue. Adding typed `GpuStageHandle`s should make that
  pipeline boundary explicit without changing model numerics or claiming a
  performance result. The useful acceptance signal for this tranche is
  observability and ownership clarity; CUDA nsys evidence is still required
  before attributing overlap or latency wins.

## Command

Local non-GPU validation:

```bash
rustfmt --edition 2024 \
  infer/src/scheduler/cuda/core.rs \
  infer/src/scheduler/cuda/core/state_types.rs \
  infer/src/scheduler/cuda/prefill.rs \
  infer/src/scheduler/cuda/decode.rs \
  infer/src/metrics.rs \
  infer/src/metrics/render.rs

cargo check -p infer --no-default-features --features no-cuda --lib

cargo test -p infer --no-default-features --features no-cuda \
  server_metrics_ -- --nocapture

cargo clippy -p infer --no-default-features --features no-cuda --lib -- -D warnings

CUDARC_CUDA_VERSION=13010 \
cargo check -p infer --no-default-features --features cuda,no-cuda
```

GPU verification TODO for a CUDA Codex:

```bash
CUDARC_CUDA_VERSION=13010 \
cargo check -p infer --no-default-features --features cuda,no-cuda

CUDA_HOME=/usr/local/cuda \
cargo test --release -p infer --features cuda --test e2e_qwen35 -- --nocapture

scripts/profile_nsys_signal.sh pipeline-scheduler-stage-handles \
  --server-args "--model-path infer/models/Qwen3.5-4B --port 8000 --max-seq-len 8192" \
  --fast \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3.5-4B
```

## Environment

- **Backend:** local Rust typecheck only; CUDA runtime validation pending.
- **Model:** not loaded locally.
- **Hardware:** Apple Silicon/macOS local development host; Linux CUDA host
  pending.
- **Commit before change:** `521b8a10`.
- **Feature set:** `--no-default-features --features no-cuda` plus
  `--no-default-features --features cuda,no-cuda` typecheck.
- **Non-default flags / env vars:** `CUDARC_CUDA_VERSION=13010` for CUDA Rust
  typecheck.
- **Server launch:** pending CUDA host.

## Params

| Param | Value |
|---|---|
| Change type | CUDA scheduler pipeline handle substrate |
| Stage handle | `GpuStageHandle { id, kind, state, slots }` |
| Stage kinds | `prefill`, `readback` |
| Stage lifecycle | `queued -> in-flight -> ready -> completed` |
| Request-local drain | `slot_has_pending_gpu_work()` checks handle slot membership |
| Metrics | `infer_scheduler_pipeline_stage_depth`, `infer_scheduler_pipeline_stage_total` |
| Perf status | `pending-remote`, no performance conclusion claimed |

## Results

| Check | Result |
|---|---|
| targeted `rustfmt --edition 2024` | PASS |
| `cargo check -p infer --no-default-features --features no-cuda --lib` | PASS |
| `cargo test -p infer --no-default-features --features no-cuda server_metrics_ -- --nocapture` | PASS |
| `cargo clippy -p infer --no-default-features --features no-cuda --lib -- -D warnings` | PASS |
| `CUDARC_CUDA_VERSION=13010 cargo check -p infer --no-default-features --features cuda,no-cuda` | PASS with unrelated existing DSV4/ops warnings |
| Directed `codex review --base HEAD` | PASS, no actionable correctness findings |
| CUDA Qwen3.5 runtime test | TODO on GPU host |
| nsys stage timeline validation | TODO on GPU host |

## Problems

- This local host cannot execute CUDA runtime tests, so the new stage-handle
  substrate still needs a GPU-host e2e and nsys trace.
- Queued depth is expected to be short-lived because this scheduler still
  dispatches one GPU command immediately after planning. The durable queued
  signal is the queued transition counter; in-flight depth is the live pending
  stage gauge across loop turns.
- The current worktree has unrelated DSV4 dirty changes. They are intentionally
  not touched or staged by this tranche.

## Learnings

- The scheduler already had a pipeline shape, but the ownership was implicit in
  `Option<PendingPrefill>` and `Option<PendingDecode>`. A typed handle makes
  the fence owner, affected slots, and lifecycle state visible to cleanup,
  metrics, and future H2D/D2H extensions.
- Mixed decode+prefill needs two logical stages: a readback handle for sampled
  decode tokens and a prefill handle for prefilling rows completed by the same
  GPU launch. Keeping both handles prevents mixed batches from disappearing
  from prefill stage metrics.
- Request cancellation should not wait globally. Handle slot membership keeps
  the existing local-drain behavior while making it auditable as a pipeline
  contract.

## Delta vs baseline

- **Baseline:** [`2026-05-15-pipeline-d2h-readback-qwen35-pending-remote.md`](2026-05-15-pipeline-d2h-readback-qwen35-pending-remote.md).
- **Delta:** not measured; pending CUDA GuideLLM and nsys run.

## Artefacts

- Raw GuideLLM artefacts: pending.
- nsys report: pending.
- CUDA runtime test log: pending.

## Notes

- This is a bench stub per the runtime-change rule. It records local type
  checks and explicitly defers performance attribution to a CUDA host.

# CUDA pipeline D2H readback unification - 2026-05-15

## Goal

- Continue P2 of the CPU/GPU pipeline plan by moving Qwen3.5 greedy batched
  decode readback from full compute-stream synchronization to the same
  copy-stream event-ring pattern used by Qwen3.

## Hypothesis

- Qwen3.5's greedy decode readback can use a model-owned async D2H ring without
  changing token selection semantics because the argmax/logprob kernel output
  is snapshotted before copy-stream readback. This should remove the hot-path
  `ctx.sync()` from Qwen3.5 greedy readback, but this commit does not claim a
  throughput or latency win until CUDA nsys and GuideLLM evidence confirms the
  stream timeline.

## Command

Local non-GPU validation:

```bash
rustfmt --edition 2024 \
  infer/src/model/qwen35/batch_decode.rs \
  infer/src/model/qwen35/forward.rs \
  infer/src/model/qwen3/batch_decode.rs \
  infer/src/metrics.rs \
  infer/src/metrics/render.rs \
  infer/src/scheduler/cuda/core/state_types.rs \
  infer/src/scheduler/cuda/decode.rs

CUDARC_CUDA_VERSION=13010 \
cargo check -p infer --no-default-features --features cuda,no-cuda

cargo check -p infer --no-default-features --features no-cuda --lib

cargo test -p infer --no-default-features --features no-cuda \
  server_metrics_ -- --nocapture

git diff -- infer/src/model/qwen35/batch_decode.rs \
  infer/src/model/qwen35/forward.rs \
  infer/src/model/qwen3/batch_decode.rs \
  infer/src/metrics.rs \
  infer/src/metrics/render.rs \
  infer/src/scheduler/cuda/core.rs \
  infer/src/scheduler/cuda/core/state_types.rs \
  infer/src/scheduler/cuda/decode.rs \
  infer/src/scheduler/cuda/execution.rs \
  docs/plans/cpu-gpu-pipeline-sync-stream.md \
  docs/experience/wins/2026-05-15-pipeline-d2h-readback-qwen35-pending-remote.md \
  | codex review -
```

GPU verification TODO for a CUDA Codex:

```bash
CUDA_HOME=/usr/local/cuda \
cargo test --release -p infer --features cuda --test e2e_qwen35 -- --nocapture

CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda

scripts/profile_nsys_signal.sh pipeline-d2h-readback-qwen35 \
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
- **Commit before change:** `b1ac132d`.
- **Feature set:** `--no-default-features --features cuda,no-cuda` and
  `--no-default-features --features no-cuda`.
- **Non-default flags / env vars:** `CUDARC_CUDA_VERSION=13010` for local CUDA
  Rust typecheck without CUDA runtime.
- **Server launch:** pending CUDA host.

## Params

| Param | Value |
|---|---|
| Change type | CUDA D2H readback substrate |
| Qwen3 path | existing async readback ring, plus context bind before event query |
| Qwen3.5 path | new async argmax/logprob D2H ring on copy stream plus sampled-token GPU handoff |
| Scheduler telemetry | `d2h_latency_us`, `d2h_wait_us`, `readback_poll_not_ready` |
| Distributed guard | distributed deferred decode waits for coordinated readback before next launch |
| DeepSeek status | explicit fallback, untouched in this tranche |
| Perf status | `pending-remote`, no performance conclusion claimed |

## Results

| Check | Result |
|---|---|
| targeted `rustfmt --edition 2024` | PASS |
| `cargo check -p infer --no-default-features --features cuda,no-cuda` | PASS with unrelated existing warnings |
| `cargo check -p infer --no-default-features --features no-cuda --lib` | PASS |
| `cargo test -p infer --no-default-features --features no-cuda server_metrics_ -- --nocapture` | PASS |
| `cargo clippy -p infer --no-default-features --features no-cuda --lib -- -D warnings` | PASS |
| Directed `codex review` | PASS after fixing Qwen3.5 sampled-token handoff and distributed follower-token guard |
| CUDA Qwen3.5 runtime test | TODO on GPU host |
| nsys stream/event validation | TODO on GPU host |

## Problems

- This local host cannot execute CUDA runtime tests, so the Qwen3.5 async
  readback path still needs GPU-host correctness and nsys validation.
- The current worktree has unrelated DSV4 dirty changes. They contribute
  warnings in the CUDA/no-cuda typecheck and are intentionally not touched or
  staged by this tranche.
- `sample_batch_greedy()` remains a blocking compatibility path for direct
  callers, but now waits on the copy stream after launching the async readback
  instead of synchronizing the compute stream. The scheduler hot path uses
  `sample_batch_greedy_launch()` / `sample_batch_greedy_readback()`.
- DeepSeek is not migrated in this step. The plan keeps it as an explicit
  fallback until DSV4 correctness parity is stable.
- Distributed scheduler requests cannot blindly consume the local sampled-token
  handoff on follower ranks. If `coordinate_decode_token()` replaces the local
  token with rank0's token, the scheduler invalidates that slot's handoff so the
  next decode input comes from the CPU-visible coordinated token.

## Learnings

- D2H readback needs a consumer-visible event slot for each in-flight greedy
  sample. Returning a dummy slot hides whether the model actually enqueued
  readback work and makes scheduler polling ambiguous.
- The pre-copy dependency is real for D2H: copy stream must wait for compute
  before reading argmax/logprob outputs. Unlike H2D staging, this wait is a
  data dependency, not a blanket serialization hazard.
- Metrics need both latency and polling visibility. `d2h_latency_us` tracks
  launch-to-ready time; `d2h_wait_us` tracks scheduler time spent checking or
  completing readback; `readback_poll_not_ready` shows whether polling is
  actually non-blocking.
- Async token handoff is not just a D2D copy. It has request-level semantics:
  normal Qwen3.5 greedy rows can decode one step ahead from the GPU-staged
  argmax, but distributed rows must invalidate the handoff when rank0
  coordination changes the sampled token.

## Delta vs baseline

- **Baseline:** [`2026-05-14-pipeline-h2d-copy-stream-pending-remote.md`](2026-05-14-pipeline-h2d-copy-stream-pending-remote.md).
- **Delta:** not measured; pending CUDA GuideLLM and nsys run.

## Artefacts

- Raw GuideLLM artefacts: pending.
- nsys report: pending.
- CUDA runtime test log: pending.

## Notes

- This is a bench stub per the runtime-change rule. It records local type
  checks and explicitly defers performance attribution to a CUDA host.

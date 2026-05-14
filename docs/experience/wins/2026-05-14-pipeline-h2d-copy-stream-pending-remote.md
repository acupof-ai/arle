# CUDA pipeline H2D copy-stream helpers - 2026-05-14

## Goal

- Continue the CPU/GPU pipeline plan by adding H2D helpers that enqueue on the
  CUDA copy stream and return explicit fences for compute-stream consumers.

## Hypothesis

- A typed copy-stream upload API lets request metadata and staged-prefix paths
  move off the compute stream in later tranches without introducing hidden
  synchronization. This commit does not claim a performance win until CUDA nsys
  evidence confirms the stream placement and wait edges.
- The safe API must stay narrow until call sites prove their ownership model:
  a copy-stream-owned device allocation can race with compute-stream readers
  during free, so this tranche exposes only a pinned-source, existing-allocation
  helper with an explicit safety contract.

## Command

Local non-GPU validation:

```bash
cargo fmt --manifest-path crates/cuda-kernels/Cargo.toml --package cuda-kernels -- --check

CUDARC_CUDA_VERSION=13010 \
cargo check -p cuda-kernels --no-default-features --features cuda,no-cuda

CUDARC_CUDA_VERSION=13010 \
cargo clippy -p cuda-kernels --no-default-features --features cuda,no-cuda -- -D warnings

git diff -- crates/cuda-kernels/src/tensor.rs \
  docs/plans/cpu-gpu-pipeline-sync-stream.md \
  docs/experience/wins/2026-05-14-pipeline-h2d-copy-stream-pending-remote.md \
  | codex review -
```

GPU verification TODO for a CUDA Codex:

```bash
CUDA_HOME=/usr/local/cuda \
cargo test -p cuda-kernels --no-default-features --features cuda \
  pinned_copy_stream_h2d_helper_returns_compute_waitable_fence -- --nocapture

CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda

scripts/profile_nsys_signal.sh pipeline-h2d-copy-stream \
  --server-args "--model-path infer/models/Qwen3-4B --port 8000 --max-seq-len 8192" \
  --fast \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B
```

## Environment

- **Backend:** local Rust typecheck only; CUDA runtime validation pending.
- **Model:** not loaded locally.
- **Hardware:** Apple Silicon/macOS local development host; Linux CUDA host
  pending.
- **Commit before change:** `485b32dc`.
- **Feature set:** `--no-default-features --features cuda,no-cuda`.
- **Non-default flags / env vars:** `CUDARC_CUDA_VERSION=13010` for local CUDA
  Rust typecheck without CUDA runtime.
- **Server launch:** pending CUDA host.

## Params

| Param | Value |
|---|---|
| Change type | CUDA H2D copy-stream substrate |
| Code path | `crates/cuda-kernels/src/tensor.rs` |
| New API | `unsafe DeviceContext::memcpy_pinned_htod_on_copy_stream` |
| Source / destination contract | pinned host source, pre-existing device allocation |
| Fence producer | `CudaPipelineStreamKind::Copy` |
| Perf status | `pending-remote`, no performance conclusion claimed |

## Results

| Check | Result |
|---|---|
| `cargo fmt --manifest-path crates/cuda-kernels/Cargo.toml --package cuda-kernels -- --check` | PASS |
| `cargo check -p cuda-kernels --no-default-features --features cuda,no-cuda` | PASS |
| `cargo clippy -p cuda-kernels --no-default-features --features cuda,no-cuda -- -D warnings` | PASS |
| Directed `codex review` | PASS after narrowing the helper API and removing the implicit pre-copy compute wait |
| CUDA H2D helper runtime test | TODO on GPU host |
| nsys stream/event validation | TODO on GPU host |

## Problems

- This local host cannot execute CUDA runtime tests. The new
  `pinned_copy_stream_h2d_helper_returns_compute_waitable_fence` test is
  intentionally left as a GPU-host TODO.
- Full-worktree `cargo fmt --all -- --check` is currently blocked by unrelated
  uncommitted DeepSeek changes in `infer/src/model/deepseek/mlp.rs`; the
  `cuda-kernels` package-level fmt gate for this tranche passes.
- No model metadata call site is switched in this tranche. That keeps the
  variable isolated to the helper API before moving Qwen3/Qwen3.5 metadata
  uploads in a follow-up change.
- Directed review rejected the initial generic safe helpers because a
  copy-stream-owned returned allocation can be freed before compute-stream
  consumers finish, an existing destination can race with prior compute-stream
  allocation or writes, and raw copy-stream memcpy needs an explicit CUDA
  context bind. The helper was narrowed to a pinned-source existing-allocation
  upload and binds the context before enqueue.
- Directed review also caught that an unconditional pre-copy
  `copy_waits_for_compute()` would serialize H2D behind unrelated compute and
  defeat the overlap goal. The final helper keeps pre-copy ordering
  caller-controlled: call sites add a wait only when the destination allocation
  or prior writes actually require it.

## Learnings

- H2D overlap needs a producer fence, not an implicit helper-side compute wait.
  The API returns the copy fence and leaves the compute wait at the consumer
  boundary.
- Safe copy-stream allocation helpers need a stream-owned device-buffer type or
  a deferred-free policy. Until that exists, low-level H2D helpers should only
  write into caller-owned allocations with explicit safety rules.
- The producer fence should be unconditional; the pre-copy dependency should
  not be. Otherwise the substrate becomes a serialized compute -> copy ->
  compute chain instead of an overlap point.
- Weight loading should stay on the existing compute-stream helpers for now;
  the near-term target is request metadata and staged-prefix payloads.

## Delta vs baseline

- **Baseline:** [`2026-05-13-pipeline-fence-substrate-pending-remote.md`](2026-05-13-pipeline-fence-substrate-pending-remote.md).
- **Delta:** not measured; pending CUDA GuideLLM and nsys run.

## Artefacts

- Raw GuideLLM artefacts: pending.
- nsys report: pending.
- CUDA runtime test log: pending.

## Notes

- This is a bench stub per the runtime-change rule. It records local type
  checks and explicitly defers performance attribution to a CUDA host.

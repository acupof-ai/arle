# CUDA pipeline fence substrate - 2026-05-13

## Goal

- Land the first code tranche of the CPU/GPU pipeline plan: an explicit CUDA
  stream/event fence substrate that later H2D, D2H, and scheduler stages can
  pass around without hidden synchronization.

## Hypothesis

- Replacing ad hoc compute/copy wait helpers with a typed fence wrapper should
  preserve current behavior while making cross-stream dependencies auditable.
  It should not claim a performance improvement until a CUDA host runs the
  nsys and GuideLLM gates.

## Command

Local non-GPU validation:

```bash
cargo fmt --manifest-path crates/cuda-kernels/Cargo.toml --all -- --check

CUDARC_CUDA_VERSION=13010 \
cargo check -p cuda-kernels --no-default-features --features cuda,no-cuda

CUDARC_CUDA_VERSION=13010 \
cargo clippy -p cuda-kernels --no-default-features --features cuda,no-cuda -- -D warnings

CUDARC_CUDA_VERSION=13010 \
cargo check -p infer --no-default-features --features cuda,no-cuda

cargo check -p infer --no-default-features --features no-cuda --lib

git diff -- crates/cuda-kernels/src/tensor.rs \
  docs/plans/cpu-gpu-pipeline-sync-stream.md \
  docs/experience/wins/2026-05-13-pipeline-fence-substrate-pending-remote.md \
  | codex review -
```

GPU verification TODO for a CUDA Codex:

```bash
CUDA_HOME=/usr/local/cuda \
cargo test -p cuda-kernels --no-default-features --features cuda \
  pipeline_fence_orders_compute_and_copy_streams -- --nocapture

CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda

scripts/profile_nsys_signal.sh pipeline-fence-substrate \
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
- **Commit before change:** `027f5852`.
- **Feature set:** `--no-default-features --features cuda,no-cuda` and
  `--no-default-features --features no-cuda`.
- **Non-default flags / env vars:** `CUDARC_CUDA_VERSION=13010` for local CUDA
  Rust typecheck without CUDA runtime.
- **Server launch:** pending CUDA host.

## Params

| Param | Value |
|---|---|
| Change type | CUDA fence substrate |
| Code path | `crates/cuda-kernels/src/tensor.rs` |
| New API | `CudaPipelineFence`, `CudaPipelineStreamKind`, `CudaPipelineFenceStatus` |
| Compatibility helpers | `copy_waits_for_compute`, `compute_waits_for_copy` |
| Perf status | `pending-remote`, no performance conclusion claimed |

## Results

| Check | Result |
|---|---|
| `cargo fmt --manifest-path crates/cuda-kernels/Cargo.toml --all -- --check` | PASS |
| `cargo check -p cuda-kernels --no-default-features --features cuda,no-cuda` | PASS |
| `cargo clippy -p cuda-kernels --no-default-features --features cuda,no-cuda -- -D warnings` | PASS |
| `cargo check -p infer --no-default-features --features cuda,no-cuda` | PASS with pre-existing DeepSeek warnings |
| `cargo check -p infer --no-default-features --features no-cuda --lib` | PASS |
| Review pass 1 | Found P2: `wait_on_pipeline_fence` bypassed cudarc context binding before `cuStreamWaitEvent`; fixed by using `CudaStream::wait(&CudaEvent)` |
| Review pass 2 | Found P2: `CudaPipelineFence::query` polled without binding the event context; fixed by binding `event.context()` before `cuEventQuery` |
| CUDA fence runtime test | TODO on GPU host |
| nsys stream/event validation | TODO on GPU host |

## Problems

- This local host cannot execute CUDA runtime tests, so the new
  `pipeline_fence_orders_compute_and_copy_streams` test is intentionally left
  as a GPU-host TODO.
- A local attempt to compile the CUDA test binary under `cuda,no-cuda` cannot
  link on this macOS host because CUDA C symbols and `/usr/local/cuda/lib64`
  stubs are unavailable. The GPU-host command above is still the runtime gate.
- The full `infer` CUDA typecheck passes on the current worktree but emits
  pre-existing DeepSeek warnings unrelated to this fence substrate tranche.
- No H2D helper or readback hot path was switched in this tranche beyond the
  existing compatibility helpers. That is deliberate: the next tranche must
  isolate H2D behavior as its own A/B variable.

## Learnings

- The fence substrate belongs in `cuda-kernels::tensor`, not in scheduler code:
  it owns CUDA events and streams, while higher layers should only receive a
  typed edge.
- The old helpers can be preserved as compatibility shims, but their
  implementation now exercises the same explicit fence API planned for the
  pipeline stages.
- Public fence wait/query APIs must bind the owning CUDA context before driver
  calls. This is required for cross-thread pipeline handoff, not just
  same-thread compatibility helper behavior.

## Delta vs baseline

- **Baseline:** [`2026-05-13-numa-pipeline-runtime-pending-bench.md`](2026-05-13-numa-pipeline-runtime-pending-bench.md).
- **Delta:** not measured; pending CUDA GuideLLM and nsys run.

## Artefacts

- Raw GuideLLM artefacts: pending.
- nsys report: pending.
- CUDA runtime test log: pending.

## Notes

- This is a bench stub per the runtime-change rule. It records local type
  checks and explicitly defers performance attribution to a CUDA host.

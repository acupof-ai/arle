# DSv4 CUDA allocation trace substrate - 2026-05-16

## Goal

- Add a default-off diagnostic substrate for Phase 1 DSv4 decode memory-access
  work so single-in-flight current-main profile runs can report CUDA allocation
  callsites before any preallocation fix is licensed.

## Hypothesis

- `nsys --cudabacktrace=memory` does not attribute driver async allocations on
  this path, so source-level callsite counters are the cheaper SOLID evidence
  path for the current-main top allocation callers.
- With `ARLE_CUDA_ALLOC_TRACE` unset, behavior should match baseline except for
  a cheap env-gated branch after successful allocations.
- With `ARLE_CUDA_ALLOC_TRACE=1`, single-in-flight profiling should include a
  `cuda_alloc_trace_process_delta` object in `request_trace` JSON. It contains
  scope metadata plus a top-20 list with file, line, kind, label, type, calls,
  and bytes for the process-global delta during that request window.

## Command

Local validation:

```bash
CUDARC_CUDA_VERSION=13010 \
cargo check -p infer --no-default-features --features cuda,no-cuda

CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo check -p infer --features cuda

CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo clippy -p cuda-kernels --features cuda -- -D warnings

cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings

CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda --lib \
  test_dsv4_fp4_batched_gemv -- --nocapture
```

Remote Phase 1 evidence command:

```bash
ARLE_CUDA_ALLOC_TRACE=1 \
./scripts/profile_dsv4_single_decode_nsys.sh \
  --out docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-alloc-trace
```

## Environment

- **Backend:** CUDA, DSv4 decode profiling substrate.
- **Hardware:** local SM89 compile/test host; target Phase 1 profile still
  requires the 8xH20 `/root/DeepSeek-V4-Flash` environment.
- **Model:** local unit test uses repo CUDA tests; remote profile uses
  `/root/DeepSeek-V4-Flash`.
- **Commit before change:** `ccd5f723`.
- **Feature set:** `--features cuda` and `--no-default-features --features
  cuda,no-cuda`.
- **Non-default flags / env vars:** `CUDARC_CUDA_VERSION=13010`,
  `NVCC_CCBIN=/usr/bin/g++-14`, `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `TORCH_CUDA_ARCH_LIST=8.9`; remote evidence additionally sets
  `ARLE_CUDA_ALLOC_TRACE=1`.

## Params

| Param | Value |
|---|---|
| Change type | diagnostic instrumentation |
| Runtime default | off |
| Enable switch | `ARLE_CUDA_ALLOC_TRACE=1` |
| Trace surface | `infer::request_trace` JSON |
| Allocation wrappers | `DeviceVec::zeros`, `HiddenStates::zeros`, `HiddenStates::uninit`, DSv4 `forward.rs`, DSv4 `mlp.rs`, DSv4 `state.rs`, DSv4 `weights.rs`, shared `ops/linear.rs`, shared `ops/sampling.rs` |
| Request output | top 20 callsites by calls, then bytes |
| Profile artifacts | `request-traces.json`, `cuda-alloc-trace-process-delta.json`, `cuda-alloc-trace-process-delta.csv` |
| Perf status | `pending-remote`, no optimization conclusion claimed |

## Results

| Check | Result |
|---|---|
| Local CUDA/no-cuda typecheck | PASS with existing DeepSeek warnings |
| Local CUDA typecheck | PASS with existing DeepSeek warnings |
| Local cuda-kernels clippy | PASS |
| Local no-cuda infer clippy | PASS |
| Local CUDA targeted regression test | PASS: `test_dsv4_fp4_batched_gemv` |
| Local `cargo clippy -p infer --features cuda -- -D warnings` | FAIL on pre-existing unused/dead-code/clippy warnings outside this substrate |
| Remote 8xH20 DSv4 allocation caller table | pending |
| Remote wall-clock attribution | pending |

## Problems

- The local machine does not have `/root/DeepSeek-V4-Flash` or the 8xH20
  topology used by the Phase 1 decode trace, so the top-20 caller table is
  intentionally pending.
- Running the CUDA/no-cuda typecheck without `CUDARC_CUDA_VERSION=13010` fails
  before this crate is checked because local CUDA 13.2 is newer than cudarc
  0.18.2's build-script version allow-list. The passing command above pins the
  repo's existing reproduction handle.
- `cargo clippy -p infer --features cuda -- -D warnings` currently fails on
  existing DSv4 and scheduler warnings such as unused imports/dead code in
  `infer/src/model/deepseek/*`, redundant clones, and needless returns. This
  tranche does not touch those unrelated files.
- The trace payload is explicitly labeled
  `process_global_delta_since_request_trace_start` and
  `single_inflight_profiling_only`. It is not a concurrency-safe per-request
  allocator profiler.
- A local nsys probe showed CUDA runtime memory backtraces were recorded for
  `cudaMalloc`/`cudaFree`, but `cuMemAllocAsync`/`cuMemFreeAsync` rows had no
  callchain IDs. This makes nsys callchain-only attribution insufficient for
  this Phase 1 root-cause gate.
- This tranche does not preallocate or remove any allocation site. It only
  creates the caller-count evidence surface needed to license one fix at a
  time.

## Learnings

- Phase 1 must start from current-main caller counts, not the stale 2026-05-14
  allocation totals. The substrate keeps the next decision tied to
  single-in-flight process-delta evidence.
- Default-off instrumentation is acceptable here because it is a measurement
  substrate, not a claimed performance improvement. The next tranche must use
  wall-clock/request framing before licensing a preallocation fix.

## Delta vs baseline

- **Baseline:** [`2026-05-14-dsv4-decode-nccl-bottleneck.md`](../errors/2026-05-14-dsv4-decode-nccl-bottleneck.md).
- **Delta:** not measured; pending remote DSv4 allocation trace.

## Artefacts

- Remote trace directory: pending
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-alloc-trace`.
- Remote allocation caller artifacts: pending
  `cuda-alloc-trace-process-delta.{json,csv}` from the same trace directory.
- Local compile/test logs: terminal output from the commands listed above.

## Notes

- This is a bench stub per the runtime-change rule. It records local compile
  gates and explicitly defers performance attribution until the DSv4 remote
  profile produces caller counts and wall-clock framing.

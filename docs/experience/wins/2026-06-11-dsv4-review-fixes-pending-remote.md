# DSv4 Review Fixes — pending remote validation, 2026-06-11

## SLO-shape probed? N

Not run locally. The changes touch DSv4 CUDA runtime paths that require an
8xH20 pod for wall-clock and correctness validation.

## Roofline check

Deferred. No local H20 trace was collected for this fix batch.

| Op | Achieved | Peak (this HW) | % | Verdict |
|---|---:|---:|---:|---|
| DSv4 FP8 contiguous MoE prefill | pending | H20 remote | pending | deferred: pending remote |
| DSv4 whole-step graph + TP comm | pending | H20 remote | pending | deferred: pending remote |
| DSv4 multiproc decode loop | pending | H20 remote | pending | deferred: pending remote |

## Goal

- Land review fixes for DSv4 MoE packing, CustomAllreduce graph safety,
  multiproc broadcast failure handling, HIP/tooling setup, and CUDA prebuilt
  manifest consistency.

## Hypothesis

- The fixes are correctness and fail-closed changes. They should not be used as
  a performance win until the H20 gates below pass.

## Command

```bash
# Local typecheck / unit-test gates run on macOS:
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,nccl,no-cuda --lib
cargo test -p infer-server --release
cargo test -p infer-cuda --release --no-default-features --features no-cuda
cargo test -p hip-sys --release --features hip
cargo build -p agent-infer --release --no-default-features --features cli,hip,no-cuda --bin arle
```

Remote gates still required:

```bash
scripts/dsv4_lever_gate.sh contigmoe ARLE_DSV4_GPU_ROUTER=1
ARLE_COMM_BACKEND=auto ARLE_DSV4_DECODE_GRAPH=1 ARLE_DSV4_WHOLE_STEP_GRAPH=1 arle serve ...
ARLE_COMM_BACKEND=nccl ARLE_DSV4_DECODE_GRAPH=1 ARLE_DSV4_WHOLE_STEP_GRAPH=1 arle serve ...
```

## Environment

- **Backend:** cuda, hip tooling, multiproc server
- **Model:** DSv4-Flash for remote validation
- **Hardware:** local macOS for typecheck/stub tests; H20 remote pending
- **Commit:** pending
- **Feature set:** `cuda,no-cuda`, `cuda,nccl,no-cuda`, `cli,hip,no-cuda`
- **Non-default flags / env vars:** see commands above

## Results

| Gate | Result |
|---|---:|
| cuda/no-cuda infer-api typecheck | pass |
| cuda+nccl/no-cuda infer-api typecheck | pass |
| infer-server tests | 25 pass |
| infer-cuda no-cuda tests | 65 pass |
| hip-sys `--features hip` off-box | 1 pass |
| arle `cli,hip,no-cuda` build | pass |
| script syntax | pass |
| `cargo fmt --check` | pass |
| `git diff --check` | pass |

## Problems

- H20 runtime gates are still pending.
- Graph fail-closed behavior and one-shot cleanup need real 8-GPU process
  validation.

## Learnings

- Review fixes that change DSv4 runtime behavior can be locally typechecked, but
  the performance and graph-safety verdict remains remote-only.

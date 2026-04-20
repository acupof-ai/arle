# CUDA paged-prefill contract fix bench stub — pending remote

**Status:** `pending-remote`  
**Plan anchor:** [`docs/plans/2026-04-20-project-constitution-and-refactor-plan.md`](../../plans/2026-04-20-project-constitution-and-refactor-plan.md)  
**Change scope:** `infer/src/model/qwen3/forward.rs`, `infer/src/scheduler/cuda/core.rs`

## Goal

- **Type:** regression
- Record the required post-change CUDA regression bench for the paged-prefill single-token fix and the decode warmup page-geometry fix.

## Hypothesis

- The fix should preserve steady-state throughput within noise while removing a correctness risk on short paged-prefill chunks and aligning graph warmup with the real pool geometry.

## Command

```bash
scripts/bench_guidellm.sh cuda-qwen3-paged-prefill-contract-fix
```

Canonical params are locked by [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md).

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** pending remote CUDA machine
- **Commit:** `c4baf7b` plus uncommitted local refactor tranche
- **Feature set:** `cargo build --release`
- **Non-default flags / env vars:** pending remote machine selection
- **Server launch:** `scripts/start_infer.sh Qwen/Qwen3-4B 8000`

## Results

- Pending remote execution.
- This entry exists because the current workstation is Apple Silicon and cannot run the required CUDA serving benchmark locally.

## Problems

- Local validation could only reach `cargo check -p infer --no-default-features --features cuda,no-cuda --lib`.
- A targeted `cargo test` invocation hit expected local CUDA link limitations on macOS (`/usr/local/cuda/lib64/stubs` unavailable and unresolved CUDA symbols), so no trustworthy runtime numbers were collected here.

## Learnings

- Runtime-facing CUDA fixes still need a committed benchmark narrative even when the development machine cannot execute the CUDA path.
- The correct fallback is a dated `pending-remote` wins entry, not a silent omission.

## Delta vs baseline

- **Baseline:** pending remote selection against the latest CUDA Qwen3 guidellm snapshot
- **Delta table:** pending remote execution

## Follow-up

- Run the canonical CUDA guidellm sweep on the next available NVIDIA host.
- Update this stub by adding a new dated wins entry with the actual numbers and a delta against the chosen baseline.

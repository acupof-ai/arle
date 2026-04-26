# CUDA Mixed Packed Prefill Review Fixes -- guidellm sweep, 2026-04-26

## Goal

- Type: regression-check.
- Validate the follow-up fixes for packed mixed decode+prefill:
  non-greedy decode sampling fallback, COW page-budget test coverage, and
  mixed TC workspace/row bounding.

## Hypothesis

- Mixed BF16 decode+prefill should keep greedy and non-greedy decode sampling
  correct while avoiding large single-step TC prefill plans that exceed
  FlashInfer workspace.

## Command

```bash
scripts/bench_guidellm.sh cuda-qwen3-mixed-packed-prefill-review-fixes
```

Invoked via: pending remote CUDA host.

## Environment

- Backend: cuda
- Model: Qwen/Qwen3-4B
- Hardware: pending remote NVIDIA host
- Commit: pending
- Feature set: `CUDA_HOME=/usr/local/cuda cargo build --release`
- Non-default flags / env vars: none expected
- Server launch: `scripts/start_infer.sh models/Qwen3-4B 8000` or equivalent

## Results

- Status: pending-remote. Local machine cannot execute CUDA kernels.

## Problems

- Local validation is limited to Rust type-checking and no-CUDA tests because
  this workspace is on macOS without NVIDIA/CUDA execution.

## Learnings

- Mixed decode+prefill must materialize per-slot decode logits before the
  scheduler falls back to `select_tokens_batch`; the greedy path can keep the
  batched argmax shortcut.
- Packed mixed prefill rows need an explicit total-QO cap and a large
  FlashInfer split-KV workspace. Per-request caps alone still allow many rows
  to combine into one oversized TC plan.

## Delta vs Baseline

- Baseline:
  `docs/experience/wins/2026-04-26-bench-guidellm-cuda-mixed-packed-prefill-pending-remote.md`
- Delta table: pending remote CUDA run.

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- What changed in the code since baseline: mixed non-greedy launch now prepares
  per-slot sampling fallback logits; mixed launch selection caps total prefill
  tokens at the long-prefill cap; mixed Qwen3 FlashInfer metadata allocates the
  large split-KV workspace; COW page-budget unit expectations were corrected.

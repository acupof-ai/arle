# Phase-4 file splits — guidellm regression check, pending-remote, 2026-04-27

> Bench stub for the structural-only file-split refactor that lands on
> 2026-04-27. CLAUDE.md mandates a wins/ entry for any in-scope diff;
> structural splits are in-scope because the touched files are on the
> hot path (HTTP server entry, CUDA scheduler runtime + core, Metal
> request_state). Pure structural refactor — no behavior change is
> expected, this is a regression check only.

## Goal

- Confirm that the four file splits (`http_server.rs`,
  `scheduler/cuda/core.rs`, `scheduler/cuda/runtime.rs`,
  `backend/metal/request_state.rs`) introduce no measurable runtime
  regression versus the prior baseline.

## Hypothesis

- Zero numeric delta. Refactor moved code between files, did not change
  any logic, branches, or visibility narrowing in the hot path.

## Status

- **pending-remote.** Mac dev box cannot exercise the CUDA scheduler
  hot path (`scheduler/cuda/{core,runtime}` rebuild needs a CUDA host).
  Metal half (`backend/metal/request_state`) is runnable locally but
  paired here with the CUDA half so the full sweep lands together.

## Command (when run remotely)

```bash
scripts/bench_guidellm.sh phase4-file-splits-cuda \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B
```

```bash
scripts/bench_guidellm.sh phase4-file-splits-metal \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B
```

## Environment (planned)

- **Backends:** cuda (remote), metal (local M-series follow-up)
- **Model:** Qwen/Qwen3-4B
- **Commit:** four-commit chain landing 2026-04-27
- **Feature set:** `--release`, default features per backend
- **Non-default flags / env vars:** none

## Δ vs baseline

- **Baseline (cuda):** the most recent
  `docs/experience/wins/*-bench-guidellm-cuda-*.md` snapshot before
  these commits. Latest stub: `2026-04-24-bench-guidellm-cuda-mixed-packed-prefill-compile-closure-pending-remote.md`.
- **Baseline (metal):** the most recent
  `docs/experience/wins/*-bench-guidellm-metal-*.md` snapshot before
  these commits.
- Threshold for "no regression": ≤ 1 % on TTFT p50 @ synchronous and
  out tok/s @ saturation. Anything outside that bound on a structural
  split is a real signal — investigate before declaring done.

## Notes

- Will be filled in once a CUDA host runs the sweep and a Metal host
  reruns the local sweep against the same commit.
- If a regression appears, the four commits are individually
  bisectable (one file per commit) — start by reverting whichever
  commit's domain matches the regressing metric.

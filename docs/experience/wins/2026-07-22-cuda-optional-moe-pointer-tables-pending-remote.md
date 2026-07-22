# Optional MoE pointer-table unification — CUDA, 2026-07-22

> Status: pending-remote

## Goal

Preserve CUDA MoE serving behavior while replacing four duplicated optional pointer-table implementations with one typed helper.

## Hypothesis

The refactor is behavior-neutral: wrappers retain their selectors, labels, expert order, and error text, so the final combined L4 gate should match its baseline.

## Parameters

Remote benchmark parameters are deferred to the planned final combined L4 gate.

- Baseline: `pending-remote`
- Treatment: commit containing this entry
- Workload / trials: final combined L4 gate

## Environment

Mac verification uses `CUDARC_CUDA_VERSION=12080` with `cuda,no-cuda`; CUDA hardware, model, TP/EP, slots, KV, and server flags are deferred to the final combined L4 gate.

## Results

No metrics claimed. CUDA execution is unavailable on Mac.

## Problems

`cargo test -p cuda-kernels --release --no-default-features --features cuda,no-cuda --lib` cannot link because `no-cuda` intentionally omits native CUDA symbols.

## Learnings

`pending-remote`. Local CUDA type checks pass; performance and runtime correctness remain part of the planned final combined L4 gate.

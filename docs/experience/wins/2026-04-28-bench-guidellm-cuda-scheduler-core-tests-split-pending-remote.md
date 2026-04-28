# Pending Remote Benchmark: CUDA Scheduler Core Tests Split

## Context

Runtime-adjacent change: moved CUDA scheduler core unit tests from
`infer/src/scheduler/cuda/core.rs` into
`infer/src/scheduler/cuda/core/tests.rs`.

This is a behavior-preserving test-module split under `infer/src/scheduler`;
CUDA scheduler construction, admission, prefix retention, and spill/drain logic
are unchanged.

## Goal

Confirm no TTFT, ITL, tok/s, or throughput regression from the CUDA scheduler
core module layout cleanup.

## Hypothesis

No measurable change. The patch only moves `#[cfg(test)]` code into a sibling
test module and leaves non-test runtime code unchanged.

## Params

- Tool: `scripts/bench_guidellm.sh cuda-scheduler-core-tests-split`
- Backend: CUDA
- Model: most recent CUDA scheduler baseline model
- Feature set: `--features cuda`
- Comparison: latest CUDA GuideLLM baseline for the same backend and model

## Env

Status: `pending-remote`

Local verification covers compile/test shape only. Run the canonical CUDA
GuideLLM regression on the CUDA benchmark host before closing the performance
gate.

## Results

Pending remote run.

## Problems

None observed locally. Remote benchmark is required by project policy for
`infer/src/scheduler/` changes.

## Learnings

Pending remote run.

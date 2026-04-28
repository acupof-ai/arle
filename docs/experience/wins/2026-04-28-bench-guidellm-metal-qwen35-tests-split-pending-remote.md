# Pending Remote Benchmark: Metal Qwen3.5 Tests Split

## Context

Runtime-adjacent change: moved the Metal Qwen3.5 unit tests from
`infer/src/backend/metal/qwen35.rs` into
`infer/src/backend/metal/qwen35/tests.rs`.

This is a behavior-preserving test-module split under `infer/src/backend/metal`;
the Qwen3.5 runtime path, MLX bridge calls, DFlash dispatch, and packed decode
logic are unchanged.

## Goal

Confirm no TTFT, ITL, tok/s, or throughput regression from the Metal Qwen3.5
module layout cleanup.

## Hypothesis

No measurable change. The patch only moves `#[cfg(test)]` code into a sibling
test module and leaves non-test runtime code unchanged.

## Params

- Tool: `scripts/bench_guidellm.sh metal-qwen35-tests-split`
- Backend: Metal
- Model: most recent Metal Qwen3.5 baseline model
- Feature set: `--no-default-features --features metal,no-cuda`
- Comparison: latest Metal GuideLLM baseline for the same backend and model

## Env

Status: `pending-remote`

Local verification covers compile/test shape only. Run the canonical Metal
GuideLLM regression on the Metal benchmark host before closing the performance
gate.

## Results

Pending remote run.

## Problems

None observed locally. Remote benchmark is required by project policy for
`infer/src/backend/metal/` changes.

## Learnings

Pending remote run.

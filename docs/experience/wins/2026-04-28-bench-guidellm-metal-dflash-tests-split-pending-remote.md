# Pending Remote Benchmark: Metal DFlash Tests Split

## Context

Runtime-adjacent change: moved the Metal DFlash unit tests from
`infer/src/backend/metal/dflash.rs` into
`infer/src/backend/metal/dflash/tests.rs`.

This is a behavior-preserving test-module split under `infer/src/backend/metal`;
the DFlash runtime, MLX bridge calls, speculative verification, and packed
decode logic are unchanged.

## Goal

Confirm no TTFT, ITL, tok/s, or throughput regression from the Metal DFlash
module layout cleanup.

## Hypothesis

No measurable change. The patch only moves `#[cfg(test)]` code into a sibling
test module and leaves non-test runtime code unchanged.

## Params

- Tool: `scripts/bench_guidellm.sh metal-dflash-tests-split`
- Backend: Metal
- Model: most recent Metal Qwen3.5 + DFlash baseline model
- Feature set: `--no-default-features --features metal,no-cuda`
- Comparison: latest Metal GuideLLM baseline for the same backend, model, and DFlash config

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

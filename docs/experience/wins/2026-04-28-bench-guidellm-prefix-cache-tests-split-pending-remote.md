# Pending Remote Benchmark: Prefix Cache Tests Split

## Context

Runtime-adjacent change: moved the `prefix_cache.rs` unit tests into
`infer/src/prefix_cache/tests.rs`.

This is a behavior-preserving test-module split under `infer/src/`; the
production prefix cache implementation is unchanged.

## Goal

Confirm no TTFT, ITL, tok/s, or throughput regression from the prefix-cache
module layout cleanup.

## Hypothesis

No measurable change. The patch only moves `#[cfg(test)]` code into a sibling
test module and leaves the compiled non-test runtime path unchanged.

## Params

- Tool: `scripts/bench_guidellm.sh prefix-cache-tests-split`
- Backend: CUDA
- Model: most recent CUDA Qwen baseline model
- Feature set: release CUDA runtime feature set from the active benchmark host
- Comparison: latest CUDA GuideLLM baseline for the same backend and model

## Env

Status: `pending-remote`

Local machine for this edit does not provide the canonical CUDA benchmark host.

## Results

Pending remote run.

## Problems

None observed locally. Remote benchmark is required by project policy for
`infer/src/` changes.

## Learnings

Pending remote run.

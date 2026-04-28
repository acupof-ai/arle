# Pending Remote Benchmark: GGUF Tests Split

## Context

Runtime-adjacent change: moved the GGUF unit tests from `infer/src/gguf.rs`
into `infer/src/gguf/tests.rs`.

This is a behavior-preserving test-module split under `infer/src`; GGUF parsing,
tensor metadata handling, and dequantization code are unchanged.

## Goal

Confirm no TTFT, ITL, tok/s, or throughput regression from the GGUF module
layout cleanup.

## Hypothesis

No measurable change. The patch only moves `#[cfg(test)]` code into a sibling
test module and leaves non-test runtime code unchanged.

## Params

- Tool: `scripts/bench_guidellm.sh gguf-tests-split`
- Backend: most recent affected GGUF baseline backend
- Model: most recent GGUF baseline model
- Feature set: matching latest GGUF baseline feature set
- Comparison: latest GuideLLM baseline for the same backend and model

## Env

Status: `pending-remote`

Local verification covers compile/test shape only. Run the canonical GuideLLM
regression on the benchmark host before closing the performance gate.

## Results

Pending remote run.

## Problems

None observed locally. Remote benchmark is required by project policy for
`infer/src/` changes.

## Learnings

Pending remote run.

# Pending Remote Benchmark: OpenAI v1 Tests Split

## Context

Runtime-adjacent change: moved the OpenAI v1 protocol unit tests from
`infer/src/http_server/openai_v1.rs` into
`infer/src/http_server/openai_v1/tests.rs`.

This is a behavior-preserving test-module split under `infer/src/`; request
translation, validation, streaming chunks, and response JSON shapes are
unchanged.

## Goal

Confirm no TTFT, ITL, tok/s, or throughput regression from the HTTP protocol
module layout cleanup.

## Hypothesis

No measurable change. The patch only moves `#[cfg(test)]` code into a sibling
test module and leaves non-test runtime code unchanged.

## Params

- Tool: `scripts/bench_guidellm.sh openai-v1-tests-split`
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

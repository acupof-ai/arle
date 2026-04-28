# Pending Remote Benchmark: Metrics Render Split

## Context

Runtime-adjacent change: moved Prometheus and human-readable metrics rendering
from `infer/src/metrics.rs` into `infer/src/metrics/render.rs`.

This is a behavior-preserving module split under `infer/src/`, so it carries a
GuideLLM regression entry even though request scheduling and model execution are
unchanged.

## Goal

Confirm no TTFT, ITL, tok/s, or throughput regression from the metrics module
layout cleanup.

## Hypothesis

No measurable change. The patch only moves `ServerMetrics::render_prometheus`
and `ServerMetrics::render_summary` into a sibling module; hot-path counters,
histograms, and scheduler updates are untouched.

## Params

- Tool: `scripts/bench_guidellm.sh metrics-render-split`
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

# Pending Remote Benchmark: CUDA Scheduler Stats State

## Context

Runtime change: grouped CUDA scheduler runtime counters, step-timing EMA fields,
and throttled GPU memory query state into `SchedulerRuntimeStats`.

This is a behavior-preserving state layout refactor under `infer/src/`, so it
requires a GuideLLM regression entry.

## Goal

Confirm no throughput, TTFT, ITL, or tok/s regression from the scheduler state
layout cleanup.

## Hypothesis

No measurable change. The patch only changes field ownership from root
`Scheduler` fields to `SchedulerRuntimeStats` and keeps update timing and
metrics emission unchanged.

## Params

- Tool: `scripts/bench_guidellm.sh cuda-scheduler-stats-state`
- Backend: CUDA
- Model: most recent CUDA Qwen baseline model
- Feature set: release CUDA runtime feature set from the active benchmark host
- Comparison: latest CUDA GuideLLM baseline for the same backend and model

## Env

Status: `pending-remote`

Local machine for this edit does not provide the canonical CUDA benchmark host.
Run on the CUDA benchmark machine before considering the performance gate
closed.

## Results

Pending remote run.

## Problems

None observed locally. Remote benchmark is required because this touches
`infer/src/scheduler/cuda/`.

## Learnings

Pending remote run.

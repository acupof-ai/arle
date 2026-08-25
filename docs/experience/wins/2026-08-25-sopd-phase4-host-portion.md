# SOPD Phase-4 host portion: C1 graph invalidation + trajectory channel — 2026-08-25

> Status: pending-remote

## Goal

#97 host portions: C1 graph-pointer staleness mitigation (seam method +
re-merge closure call) and 4b trajectory-emit channel (agent → OPD data
buffer). C5 (batch drain) was already implemented (`drain_control` +
`run_on_engine` control closure).

## What landed

- `BackendExecutor::invalidate_decode_graph` (infer-seam, default no-op):
  called inside `remerge_student_lora`'s control closure after the weight
  mutation + prefix-cache drop, so a captured decode graph that baked the old
  weight pointers is dropped before the next step. The CUDA override (drop +
  force re-capture) is pod-only.
- `TrajectoryChannel` (train/update_strategy.rs): a bounded sync channel
  carrying `ScoredTrajectory` from the agent/tools side into the OPD data
  buffer. `drain_batch(max)` is non-blocking; `emit` applies backpressure
  when the buffer is full. 3 cpu-lane unit tests.

## Parameters

```bash
# pending-remote: live-serve self-update loop on H20
# - needle after a live re-merge proves no graph-pointer corruption (C1 gate)
# - trajectory channel feeds live agent rollouts into the OPD step (4b gate)
```

- Baseline: `ba1d464c3` (no graph invalidation, no trajectory channel)
- Treatment: this commit
- Trials: pending-remote

## Environment

- Host / GPU: H20 pod (pending-remote)

## Rule

A default-no-op seam method is the migration seam for graph invalidation:
backends without a captured decode graph keep the default, and the CUDA
override lands independently. A bounded sync channel is the simplest
backpressure-correct trajectory pipe.

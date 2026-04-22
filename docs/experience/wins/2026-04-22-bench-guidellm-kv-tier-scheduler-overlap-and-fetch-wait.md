# Bench Stub — KV Tier Scheduler Overlap And Fetch-Wait Idle

## Context

This runtime tranche changes the live CUDA scheduler loop in two ways:

- `execution.rs::step()` now keeps decode readback pending across loop turns so
  CPU-side intake/admission can overlap the previous decode launch
- `runtime.rs::run()` no longer busy-spins when every active request is parked
  in `Phase::WaitingFetch`; it blocks on coordinator events with a short timeout

## What Worked

- The scheduler compile/test lanes stayed green after the loop reordering.
- The runtime flow doc now reflects the actual per-iteration ordering and the
  `WaitingFetch` wakeup rule.

## Rule

Status: `pending-remote`

Remote CUDA / guidellm validation is still required because this changes the
live scheduler hot path under `infer/src/scheduler/cuda/{runtime,execution}.rs`.

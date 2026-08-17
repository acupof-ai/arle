# observe simplification: dead state + broken latest endpoint

## Context

The observability feature (2026-08-17-observability-infra) shipped with dead
state and a redundant endpoint. A /simplify pass (4-agent review) flagged
them; one finding was also a multiproc correctness bug.

## What Worked

- `GpuSample.stale` / `sampled_at_ms` and `HostSample.ram_total_mb` were
  written but never read — deleted.
- The host-sampler thread (RwLock + OnceLock + dedicated thread) was pure
  indirection: its only reader was the observe task that could compute the
  sample inline. Inlined; -1 thread, -35 lines.
- `/v1/observe/latest` + the `LATEST` static duplicated the query tail. In
  multiproc (the primary CUDA serving mode) the coordinator is engine-less,
  so `LATEST` was permanently `None` — dashboard stat tiles never populated.
  Deleted the endpoint; tiles now read `samples[samples.length-1]` from
  `/v1/observe/query`, which works in every mode (a flock-winning worker
  writes JSONL).
- Dropped `.truncate(false)` in append_sample (redundant under
  `.append(true)`); the lock-file open keeps it to satisfy
  clippy::suspicious_open_options.

Net -62 lines. No perf delta (background sampling + dashboard, not the
inference hot path). Device-neutral crates checked with `-D warnings`;
observe tests pass. `gpu_sample.rs` typechecks in CI's `cuda,no-cuda` lane.

## Rule

A "latest sample" static in a process that may not own the writer is broken
by construction — derive latest from the time series' last element.

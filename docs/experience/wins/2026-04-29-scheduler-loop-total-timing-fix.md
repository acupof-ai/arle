# Scheduler Loop Total Timing Fix

## Context

`infer_scheduler_loop_total_microseconds` claimed to report full scheduler-loop
wall time, but the sample was taken immediately after `cleanup()`. Per-tick
metrics updates, KV coordinator gauges, tier wait gauges, KV pool gauges, and
the throttled GPU memory query were not included.

## What Worked

Kept the metric name and help semantics as full loop wall time, and moved the
sample to the end of the loop body after the metrics and memory-query work.
The slow-iteration log now uses the same full-loop value.

## Verification

```bash
CARGO_HOME=/tmp/arle-cargo-home CARGO_TARGET_DIR=/tmp/arle-target \
  ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig CUDA_HOME=/usr/local/cuda \
  cargo test -p infer --features cuda loop_total_timing_includes_work_after_step_phase --lib
```

Result: 1 passed.

## Rule

Metric names and help text must match the sampling window. If a gauge is named
`loop_total`, sample it after the loop-body work it claims to include.

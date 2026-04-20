# Qwen3.6 Metal bench must run serially

## Context

While investigating Qwen3.6-35B-A3B Metal prefill performance, I mixed bench runs
with parallel agent/tool activity and extra shell work in the same turn.

## Root Cause

The measurement task was latency-sensitive, but I treated it like a generic
exploration task and overlapped unrelated work. That adds noise and makes it
harder to trust wall-clock comparisons between runs.

## Fix

For Qwen3.6 Metal performance checks, run benchmarks serially:

- one bench command at a time
- no parallel agent waits
- no concurrent shell probes
- compare like-for-like command shapes only

## Rule

If the user corrects bench methodology, record it immediately and rerun with the
requested measurement discipline before drawing conclusions.

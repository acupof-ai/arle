# Task #43 — W4A16 fragmentation hypothesis disproven

## Context

Task #43 tested whether W4A16 sustained-load failures share the PF8.3 root cause:
per-call `cudarc` allocator fragmentation when `INFER_PREFILL_GRAPH` is not set
and Marlin prefill scratch falls back to per-call allocation.

The single-variable A/B:

- Arm A: `INFER_PREFILL_GRAPH=1`, expected to route Marlin scratch through the
  prefill graph path and avoid per-call allocation.
- Arm B: no `INFER_PREFILL_GRAPH`, expected to reproduce the fallback allocator
  failure.
- Workload: Qwen3-4B W4A16 Marlin, `4096-in / 128-out`, `concurrency=4`,
  60 seconds.

## Result

The first run was invalid because the scaffold did not pass
`--max-seq-len 5120`; both arms rejected `4097`-token prompts with
`scheduler max_input=1997`.

After fixing the scaffold, the result inverted the hypothesis:

| arm | env | guidellm successful | live kernel failures | TTFT p50 | ITL p50 | verdict |
|---|---|---:|---:|---:|---:|---|
| A | `INFER_PREFILL_GRAPH=1` | 71 | 36 | 834.6 ms | 7.35 ms | substrate kill |
| B | unset | 56 | 0 | 2381.7 ms | 11.36 ms | healthy |

Representative Arm A live failure:

```text
Request 1: prefill batch failed: Alloc failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

Arm B had only the expected startup warmup backoff warning and no live kernel
failure.

## Root Cause

The original fragmentation hypothesis is not supported. The failing arm is the
prefill-graph/scratch-enabled arm, not the eager fallback arm. This points to
prefill-graph memory footprint or persistent graph-resource cache pressure, not
per-call `cudarc` allocator fragmentation in the eager fallback.

The scaffold also had two diagnostic bugs:

- It omitted `--max-seq-len 5120`, so the first run measured prompt rejection.
- `pf83_bench_health.sh` did not understand `guidellm 0.6.0`
  `metrics.request_totals.successful` and counted successful benches as zero.

## Fix

This entry does not change runtime behavior. It fixes the diagnostic scaffold:

- `scripts/task43_hypothesis_test.sh` now starts the server with
  `--num-slots 8 --max-seq-len 5120`.
- `scripts/pf83_bench_health.sh` now parses `metrics.request_totals` and
  separates live kernel failures from intentional Pass 3 warmup backoff.
- The Task #43 verdict branch now treats Arm A kill + Arm B healthy as an
  inverse failure that disproves the allocator-fragmentation hypothesis.

## Rule

If a hypothesis test depends on long prompts, the server envelope is part of the
experiment. Verify the server log for prompt admission before interpreting
benchmark metrics. For health checks, distinguish live-request failures from
startup warmup probes that intentionally back off on OOM.

# KV-Tier Stage Hot Path Removal

## Context

The CUDA tiered-KV scheduler still claimed a live staged readmission path:
requests with radix hits in T1/T2 parked behind coordinator `Stage*` events
and re-entered admission after a synchronous local echo. Review found that the
path was not correct under concurrency and did not produce a runnable prefix
surface for admission.

This change removes the broken scheduler hot-path staging semantics and keeps
the runtime honest:

- live prefix reuse remains `T0`-only
- `T1/T2` remain live for spill / persist
- staged readmission now explicitly waits on a future attach / ownership model

## What Worked

- Deleted the scheduler's parked `stage_waiting` admission path.
- Removed request parking / promotion logic that could not safely support
  concurrent stage hits.
- Kept the verified `T0 -> T1 -> T2` spill / persist flow intact.
- Updated the tiered-KV docs to describe the current runtime contract instead
  of the intended future one.

## Rule

Do not claim a live multilayer restore path until the runtime has a real
attach / ownership model that makes staged bytes runnable for admission.

## Benchmark

Status: `pending-remote`

Required follow-up on a CUDA host:

```bash
scripts/bench_guidellm.sh kv-tier-stage-path-removal
```

Compare against the most recent CUDA baseline for the same model / slot count
and record TTFT / ITL deltas.

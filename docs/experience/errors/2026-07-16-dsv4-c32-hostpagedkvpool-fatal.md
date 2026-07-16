# DSv4 c32: HostPagedKvPool exhaustion is FATAL instead of preempting

## Context

Arm C (`672b8ac08` slot hoist, slots 2→59) var-c32 bench: 59/64 requests
completed, then at ~101 s (tick #3073) all 4 ranks died together:

```
worker rank 3 step (tick #3073): KV alloc retry failed after reclaiming 2
pages: first error: HostPagedKvPool out of pages: slot 6 needs 2, free 0;
retry error: ... free 1
[coordinator] tearing down the serve (worker group unwound by the child guard)
```

The 5 "errored" guidellm requests all carry `dequeued` == teardown timestamp —
killed by the teardown, not individual failures. Unreachable before the slot
hoist: clamp 2 could never oversubscribe the pool.

## Root Cause

**Not yet attributed — hypothesis only:** scheduler admission budgets against
engine `total_pages` (8192) while the DSv4 shared comp pool holds 1324 pages /
84736 tokens; 59 slots × ~2.7k-token prompts oversubscribe the comp pool, and
the step-loop alloc failure path is fatal rather than parking a decode. This
contradicts the engine invariant (#162: admission reserves prompt+max_tokens
AND preempts under KV pressure). Which pool `HostPagedKvPool` names, and why
reclaim freed only 1 of the 2 needed pages, must come from the code + a
decoded repro before any fix.

## Fix

Pending (tracked in the GitHub issue). Constraint for the fix: the step-loop
alloc failure must degrade to preemption/parking, and admission must budget
the pool that actually binds.

## Rule

A capacity fix that raises concurrency is not accepted until the NEW regime's
failure path is exercised — clamp-2 masked a fatal alloc path for months;
the first c32 run found it in 101 s. Bench the failure boundary, not just the
happy path.

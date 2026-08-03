# The 1.2 ms host tail was an O(cached-pages) scan, 3× per token — 2026-08-03

> Status: **Shipped, default path.** c=1 W8A16 decode ITL p50 **20.19 → 18.98
> ms (−6.0%)**; p99 21.32 → 19.52. Cumulative with #196: **26.88 → 18.98
> (−29.4%)**. Greedy byte-identical; graphed lane intact.

## What it was

`HostPagedKvPool::resident_evictable_pages()` walked the **entire** page-ref
map and did a second hash lookup per entry. Per decode token it ran three
times: once in the planner's admission tick, twice more via
`publish_counters` (called on both the stepping and the post-admission path).
With a warm prefix cache holding ~20k retained pages that is ~0.3 ms per
scan — **~0.9–1.2 ms/token, growing with cache warmth.**

Now an O(1) counter maintained at the four retain/release/attach/detach
transitions. Three smaller cuts rode along: the redundant post-admission
`publish_counters`, a per-step `getenv` (`ARLE_STEP_DIAG`) cached in a
`OnceLock`, and an early return in `admit_waiting` when nothing is waiting
(placed **after** the TP collective so per-rank collective counts are
unchanged — gating a collective on a rank-local condition is the 2026-07-05
admission-livelock hazard).

## Why it hid for so long

**It is invisible in every cold measurement.** The cost is proportional to
retained prefix-cache pages, so a fresh serve, a micro-bench, or a short
smoke run all show ~0 — it only appears in the steady state that production
actually runs in. It also never showed in the CUDA profile: no kernel, no API
call, just host time between one step's tokens and the next submit. It took
a per-phase host-time split of the decode step (executor 0.05 ms, engine tail
1.2 ms) to prove the time was outside the executor at all, and a read of the
call graph to find it.

**Rule:** when a per-token cost scales with a *cache* rather than the request,
the cold bench cannot see it. Measure with a warm cache, and split host time
by phase before assuming the GPU owns the step.

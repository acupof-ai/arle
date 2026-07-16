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

Attributed (code-level). The crash is NOT a wrong admission capacity — it is
an incomplete KV-pressure plan repair that lets a step-loop alloc error
propagate fatally.

1. **The pool is the right one.** `HostPagedKvPool`
   (`crates/infer-seam/src/host_paged_kv_pool.rs:187`) is the host admission
   mirror of DSv4's demand-paged FlashMLA comp pool: for demand-paged models
   `flashmla_total_pages()` returns `flashmla_pool_tokens / 64` = 1324 pages
   (`crates/infer-cuda/src/attention/kv_layout.rs:1037-1041`), plumbed via
   `effective_total_pages` (`crates/infer-cuda/src/executor.rs:729`) into
   `cuda_admission_total_pages` (`crates/infer-api/src/loaded.rs:1765-1770`,
   pool built at `loaded.rs:2100`). Admission budgets exactly the 1324-page
   comp capacity — the "84736 ≪ 195k" smell was real oversubscription, but
   BY DESIGN: `admit_waiting` re-derives `remaining_pages` from
   `kv.free_pages()` each tick (`crates/infer-core/src/lib.rs:1121`), so the
   prompt+max_tokens reservation (`prefix.rs:16`) lives only within one
   tick's admit loop. Chunked prefill allocates lazily, so 59 slots ×
   ~3.3k-token commitments admit fine and grow into the pool later — vLLM's
   admit-on-current-free + preempt-on-pressure model. That model is only
   crash-free if the pressure path is complete. It wasn't:

2. **`retract_decode_to_fit` (pre-fix `crates/infer-core/src/planner.rs:111`)
   had two holes**: it never retracted the LAST decode row
   (`plan.decode_rows.len() > 1` floor) and never shed PREFILL rows at all.
   At tick #3073 free=0 and the plan carried a prefill chunk for slot 6
   crossing 2 page boundaries; retract could not repair that, so the plan
   reached `allocate_for_plan` (`lib.rs:666 → 1372`) →
   `alloc_with_prefix_reclaim` (`prefix.rs:206`) → `HostPagedKvPool::alloc`
   bailed "slot 6 needs 2, free 0".

3. **Why reclaim "freed" 2 but retry saw free 1:** `reclaimed` counts
   radix-evicted pages (`prefix.rs:376`), but `release_pages` only returns a
   page to `free` when its refcount hits 0
   (`host_paged_kv_pool.rs:283-292`); one of the 2 evicted cache pages was
   still attached to a live slot → only 1 entered `free` → retry error
   "needs 2, free 1".

4. **Why fatal:** the `Err` propagates out of `Engine::step`
   (`lib.rs:666`) → `serve_multiproc.rs:361` worker `?` → worker thread
   returns → child guard unwinds the TP group → coordinator teardown.

Unreachable before the slot hoist because clamp-2 slots could never
oversubscribe 1324 pages.

## Fix

`crates/infer-core/src/planner.rs` — complete `retract_decode_to_fit` so a
plan can NEVER reach a failing `allocate_for_plan` (`plan_new_pages_needed`
mirrors the allocs exactly, so post-repair allocation is total):

- shed prefill rows first (a dropped chunk retries next tick with zero state
  change — the same deferral `apply_step_budget` already performs);
- then preempt-requeue decode victims down to an EMPTY plan (floor removed) —
  each retraction is the existing #162 park/recompute path
  (`requeue_preempted_decode`) and frees/reclaims the victim's pages, so the
  loop terminates and later ticks fit. An idle plan for one tick beats
  unwinding the TP group.

No new mechanism, no admission change (per-tick budgeting already targets the
binding 1324-page pool), no device sync; inputs (lockstep host pool + plan)
are the same the pre-fix retract used, so SPMD determinism is unchanged.
GPU-free regression test:
`kv_exhaustion_sheds_prefill_and_parks_last_decode_instead_of_fatal`
(`crates/infer-core/src/lib.rs`). Residual (out of scope, noted): the
spec-decode extra-token alloc in `apply_output` (`lib.rs:945`) can still
error under exhaustion — only reachable with MTP on.

**Pending gate:** pod var-c32 repro (4×H20 TP=4/EP=4, 59 slots, ~3.3k-tok
prompts) must run past the ~101 s exhaustion point with preemptions logged
and zero teardowns.

## Rule

A capacity fix that raises concurrency is not accepted until the NEW regime's
failure path is exercised — clamp-2 masked a fatal alloc path for months;
the first c32 run found it in 101 s. Bench the failure boundary, not just the
happy path.

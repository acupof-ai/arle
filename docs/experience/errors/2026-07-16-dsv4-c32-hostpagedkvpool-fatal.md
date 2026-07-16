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

## Residual (2026-07-17)

The 459ed5000 repair (`fit_plan_to_kv_pages` budgets `tp_sync_min(free +
resident_evictable_pages)`, sheds/preempts on shortfall) did NOT survive the
pod acceptance run: 4×H20 TP=4 eager c=32, prefix hit_rate 0.925, tick #8340,
all ranks identical — `KV alloc retry failed after reclaiming 2 pages ...
needs 2, free 0 ... retry ... free 0`, with ZERO park/preempt/requeue in the
log (the repair never computed a shortfall). Attributed (code-level, verified
against `serve-accept-crash-last80.log`: L2 host tier ON, 59 slots):

1. **Capacity model ≠ allocator — the pool refcount is blind to live-slot
   attachment.** `HostPagedKvPool::resident_evictable_pages`
   (`host_paged_kv_pool.rs:130` pre-fix) counted `page_refs == 1`. But
   `apply_output`'s prompt-seal publish (`lib.rs:930-936`) puts a LIVE
   decoding slot's freshly-prefilled pages into the radix at exactly
   `page_refs == 1` (radix `ref_count == 0`, no `retain_blocks`, not in
   `reused_prefix_pages`) while they still sit in the slot's `slot_pages`.
   With c=32 decodes × 256-token outputs, ~32 requests' non-reused prompt
   tail pages sat in that state at any tick — phantom capacity. The repair
   (`planner.rs:139`) therefore saw `plan_new_pages_needed ≤ capacity` and
   shed nothing (zero park/preempt ✓).

2. **The evictor freed live pages, seeding refcount drift.**
   `evict_prefix_cache_for_pages` (`prefix.rs:376` pre-fix) victims came from
   radix LRU only (`is_evictable_leaf`: radix ref 0), never consulting slot
   attachment. Evicting/demoting a live-attached page ran `release_pages` →
   refs 1→0 → the page RETURNED TO FREE while the slot still wrote it
   (`host_paged_kv_pool.rs:283-292` pre-fix). Consequences cascade: the page
   is re-allocated to a second slot (aliasing); the first slot's `free_slot`
   pushes it AGAIN (`reclaim_page`: refs 0) — duplicate free entries; a
   duplicate later gets published under a second token path and
   `RadixCache::insert`'s `page_to_node.insert` overwrite (`radix.rs:392/406`)
   orphans the older node WITHOUT releasing its ref. Evicting the mapped node
   frees the single ref; the orphan remains an `is_evictable_leaf` whose later
   eviction calls `release_pages` on a page absent from `page_refs` — a
   no-op.

3. **Why "reclaimed 2, free 0" twice:** `alloc_with_prefix_reclaim`
   (`prefix.rs:216` pre-fix) counted `reclaimed` = radix-severed NODES, not
   pages actually freed, and stopped after one bounded batch. At tick #8340
   the LRU frontier's coldest entries were drift orphans (mechanism 2): both
   eviction attempts severed 2 nodes, freed 0 pages, and gave up while
   genuinely freeable pages sat deeper in the LRU. Deterministic lockstep →
   identical on all 4 ranks. The failing call site is
   `allocate_for_plan` (`lib.rs:1379`, prefill chunk crossing 2 page
   boundaries); the other step-path allocs (spec extras `lib.rs:950`,
   `restore_swapped_slot` `planner.rs:359`) already degraded.

Fix (single source, degrade-never-propagate):

- `KvQuery::page_is_evictable(page)` — new predicate: retained exactly once
  AND attached to no live slot. `HostPagedKvPool` now tracks per-page
  `slot_attach` counts (alloc/attach_pages/free_slot/truncate/evict_slot_page)
  and both `resident_evictable_pages` (capacity) and the evictor filter are
  this one predicate.
- `release_pages` frees a page only when retains AND attachments are zero —
  the radix evicting a live-attached page can no longer recycle it (the
  corruption seed of mechanism 2 is closed at the pool level).
- `evict_prefix_cache_for_pages` is evict-until-freed: filters victims by the
  predicate, skips non-freeing pages, re-queries the frontier each round, and
  returns pages ACTUALLY freed.
- `allocate_for_plan` degrades instead of propagating: a failing prefill row
  is shed, a failing decode row parks via `requeue_preempted_decode`;
  `attach_prefix_to_request`'s attach/grow allocs degrade to full recompute.
  Invariant: no step-path alloc failure can unwind the TP group.

GPU-free regressions (production `HostPagedKvPool`):
`evict_skips_live_attached_lru_head_and_frees_deeper_page`,
`plan_repair_sees_shortfall_when_cached_pages_are_live_attached`
(`crates/infer-core/src/lib.rs`),
`release_of_attached_page_defers_free_until_slot_detach`
(`crates/infer-seam/src/host_paged_kv_pool.rs`).

**Pending gate:** pod c=32 acceptance rerun past the ~700 s crash point with
zero teardowns.

## Rule

A capacity fix that raises concurrency is not accepted until the NEW regime's
failure path is exercised — clamp-2 masked a fatal alloc path for months;
the first c32 run found it in 101 s. Bench the failure boundary, not just the
happy path.

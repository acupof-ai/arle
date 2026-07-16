# #164 first fix traded a crash for a livelock — evictable ≠ free, and free_pages is rank-local

## Context

The first #164 fix (`74b6aafd3`) made `fit_plan_to_kv_pages` shed prefill rows
and preempt decodes whenever `plan_new_pages_needed > kv.free_pages()`. A
high-effort adversarial review (before any pod deployment) confirmed four
defects in it.

## Root Cause

1. **Evictable ≠ free.** `allocate_for_plan` → `alloc_with_prefix_reclaim`
   evicts radix-cache pages on demand, so the pool's true capacity is
   `free + resident_evictable`. Budgeting against `free_pages` alone turned a
   warm cache (free=0, plenty evictable) into a permanent prefill stall —
   the repair popped the chunk every tick and the eviction path (inside
   allocation) was never reached. Preemption cascaded for the same reason:
   victims' pages are published INTO the cache, barely raising `free`.
2. **`free_pages` is rank-local** (KV-tier demote residuals) — admission
   tp_sync_min's it for exactly this reason; the repair read the raw value,
   so ranks could shed different rows → divergent ForwardPlans → collective
   shape mismatch → TP hang.
3. Blind `pop()` deferred zero-demand (fully cached) prefill rows for nothing.
4. The MTP extra-token alloc in `apply_output` remained a fatal path outside
   the repair's coverage.

## Fix

`459ed5000`: capacity = `tp_sync_min(free + resident_evictable)` (accessor
already existed), re-synced after each preemption; shed only demand-reducing
rows (`rposition(append_pages_needed > 0)`); spec extra-token alloc failure
degrades to the #162 park path. 6 GPU-free regression tests cover
warm-cache-no-shed / zero-demand-survives / true-exhaustion-idles / spec-park.

## Rule

- A plan-repair predicate must model the ALLOCATOR's full capacity, not one
  counter — enumerate every source the failing alloc can draw from (free,
  evictable, reclaim-on-demand) before writing `demand > X`.
- Any scheduler decision input must be rank-consistent by construction;
  grep for existing `tp_sync_min` uses of the same counter before reading it
  raw.
- A fatal-path fix is complete only when every alloc call site on the step
  path is enumerated (the MTP `apply_output` alloc was outside the repaired
  window).

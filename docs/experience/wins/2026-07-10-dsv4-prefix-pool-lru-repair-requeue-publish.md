# DSv4 prefix pool LRU + canonical repair; requeue publishes committed sequence

> Status: pending-remote — CUDA bench requires the 8×H20 pod; local gates only.

## Context

Conformance review of the #154/#157 KV-reuse plan found two performance
hazards:

1. **Pool FIFO evicted the hottest prefix → floor-0 self-lock.** Over-cap
   eviction was publish-order FIFO (`read_entry` never touched), so under
   DRAM pressure the first-published shared preamble entry dropped first.
   Once any low page's entry was gone, `reusable_prefix_blocks` broke at the
   missing page → whole chain floor-0; recompute deduped onto canonical
   pages → `newly_cached=∅` → nothing ever re-attached entries to the
   canonical chain until radix eviction.
2. **Preemption requeue published prompt-only.** Generated pages' provisional
   entries dropped at `free_slot_pages`; resume / follow-up turns recomputed
   the whole generated region (plan claimed ≤128-token resume cost).

## What Worked

- `Dsv4PrefixStatePool`: eviction index `(confirmed, stamp, page_id)` —
  provisional-first, LRU within tier, touch-on-read (kv_tier `host_lru`
  pattern).
- Confirm-time repair (`adopt_canonical`): where radix dedup diverged the
  canonical chain from the slot's own pages and the canonical entry is
  missing, re-key the slot's provisional entry (content-identical by
  construction) to the canonical id and confirm. Seam
  `save_prefix_sidecar` gained a `slot_pages` param to carry the pairing.
- `requeue_preempted_decode_with_bias` publishes the full committed
  sequence (prompt + generated) — the finish boundary — before
  `free_slot_pages`, on both the page-tier and plain-recompute arms.

Local evidence: `cargo test -p infer-core --profile release-fast` 92 pass
(2 new: `requeue_publishes_generated_pages`, pool LRU/repair pair in
`prefix_state.rs`); cli cpu lane 208 pass; Mac cuda,no-cuda typecheck clean.

## Rule

Reuse-pool eviction must be recency-and-value ordered (LRU + confirmed >
provisional), never publish-order; every dedup site needs a repair path that
can re-attach recomputed state to the canonical chain, or one eviction
self-locks the whole prefix.

Pod follow-up: `scripts/bench_guidellm.sh` DSv4 A/B vs latest baseline under
DRAM-pressure pool budget; verify preempt-resume attach depth.

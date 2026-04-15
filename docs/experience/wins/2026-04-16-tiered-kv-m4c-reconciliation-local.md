# 2026-04-16 · Tiered KV M4c local reconciliation

## Context

M4c closes the local gap between serde-based radix snapshots and the M4
session-load design: `RadixCache` can no longer treat deserialized
`BlockId`s as durable identity, and process-local runtime counters should not
leak across a restore boundary.

## What Worked

`infer/src/prefix_cache.rs` now exposes a `ReconcileReport` plus
`RadixCache::reconcile(...)` so a restored radix snapshot can remap known
fingerprints into a fresh pool, tombstone missing fingerprints, clear
fingerprint-less orphans, and rebuild the private `block_index` in one pass.
Serde now skips runtime-only prefix-cache state (`ref_count`, LRU clocks, and
soft-pin deadlines), so restore starts from a clean scheduler epoch while
preserving the structural/tier metadata that actually belongs in the snapshot.
The no-cuda tests cover remap, tombstone, orphan clearing, and an end-to-end
serialize → deserialize → reconcile → `lookup_or_stage` flow.

## Rule

When a radix snapshot crosses a process boundary, persist structural identity
and durable metadata, then rebuild every allocator-local or clock-relative
field from the new runtime. `BlockFingerprint` is the restore key; `BlockId`
and logical-clock state are not.

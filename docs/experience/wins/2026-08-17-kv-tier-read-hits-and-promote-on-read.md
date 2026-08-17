# KV tier read-hit telemetry + L3→L2 promote-on-read — all backends, 2026-08-17

> Status: Shipped

## Context

L2/L3 KV tier audit (workflow `wrnsbj8br`, 5 parallel mapping agents +
synthesis) found the tier stack structurally sound but with two high-severity
observability and efficiency gaps:

1. **Read-hit telemetry stubbed on 3/4 arms.** Only DSv4's prefix-state pool
   tracked `kv_tier_read_hits`. Metal, Qwen35, and Qwen returned
   `KvTierReadHits::default()` — L2-vs-L3 hit attribution read as permanently
   zero, so tier effectiveness could not be measured on the arms that most
   needed it.
2. **No L3→L2 promote-on-read.** A cold key spilled to NVMe was re-read from
   disk on every promote; disk hits never re-populated host L2. Read
   amplification under repeated prefix hits, on every backend.

Additional medium-severity fixes: DSv4 `kv_tier_io_stats` omitted `slot_tier`
(whole-slot spill IO invisible), `kv_tier_location()` returned `None` on
Qwen35/DSv4, stale warning text in `loaded.rs`, stale comment in `executor.rs`,
and a probe-miss fallback that over-granted 4 GiB on non-Linux.

## What Worked

**Read-hit tracking on the store itself.** Added `host_read_hits` /
`disk_read_hits` counters to `KvTierStore`, incremented in `read()` and
`read_many()`. Exposed via `KvTierStore::read_hits() -> KvTierReadHits`.
Each executor arm delegates:
- Qwen35: merges `slot_tier` + `recall_tier`
- Qwen: `tier.read_hits()`
- DSv4: merges `prefix_state.read_hits()` + `slot_tier.read_hits()`
- Metal: `page_store.tier.read_hits()`

**L3→L2 promote-on-read in the shared read path.** On a disk hit in
`KvTierStore::read()`, the payload is re-inserted into host L2 via
`insert_inner(key, payload.clone())` (evict-if-full). The caller still gets
`Cow::Owned(payload)` — the extra clone is negligible vs the NVMe read
eliminated on the next promote. `read_many_concat` inherits the behavior
since it calls `read()` in a loop.

**DSv4 io_stats aggregation.** `kv_tier_io_stats` now sums `prefix_state` +
`slot_tier` byte/ns/failure counters (was prefix-state only).

**kv_tier_location delegation.** Qwen35 checks `slot_tier` then `recall_tier`;
DSv4 checks `slot_tier` then `prefix_state`.

**Non-Linux probe-miss fix.** `dram_l2_budget` returns 0 on non-Linux (was
returning the 4 GiB floor when `/proc` is unavailable — harmless on Metal
where L2 is disabled, but would silently over-grant on a future non-Linux
CUDA host).

## Rule

- Track read hits on the store, not the executor — one delegation point per
  arm, zero hot-path cost.
- Promote-on-read belongs in the shared `KvTierStore::read()` path, not in
  each executor's promote hook — one change covers all backends.
- `Cow::Owned` on disk hits is the correct return type for promote-on-read:
  the caller gets owned bytes, the host gets a copy, no borrow-checker escape
  hatches needed.

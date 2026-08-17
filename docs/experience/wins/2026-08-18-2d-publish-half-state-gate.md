# 2D publish half-state gate — CUDA, 2026-08-18

> Status: pending-remote (typecheck green on infer-core; 2D GPU needle ladder pending pod)

## Context

c8c8b5d71 skipped the prefix **attach** path under 2D (attn_tp × attn_cp, world ≥ 4)
to break a cross-communicator deadlock, but left the **publish** path running.
Under 2D the radix is write-only — attach is skipped, so no rank ever reads the
cached prefix. `publish_prefix_blocks` still burned one `tp_sync_min` collective
plus `retain_pages` + sidecar save per request completion, all for blocks no
rank ever re-attaches.

## What Worked

- **Early return on `kv_shard_spec().is_some()`.** The same predicate c8c8b5d71
  used for attach, applied at the top of `publish_prefix_blocks`. Symmetric on
  every rank (config property, not runtime state), so the collective count
  stays rank-invariant — no deadlock.
- **Dead min-reduce block removed.** With the 2D gate, `cp_shard_identity()`
  always returns `ShardSpec::default()` (rank=0, size=1) past the gate, so the
  `if shard.size > 1 { tp_sync_min }` branch was unreachable. Collapsed to
  `common_extent = shard.rank + local_sealed * shard.size`.

## Rule

When a sharding mode makes a cache write-only, gate the write path with the
same predicate that gated the read path — a half-gated cache burns collectives
and retains pages for blocks that are never read. The attach and publish gates
must use the same predicate so the collective count stays symmetric.

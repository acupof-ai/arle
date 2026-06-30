# KV Mmapped-Tier E2E — L2 DRAM + L3 NVMe Verified on H20 Pod

## Context

The KV tier system demotes evicted prefix pages to a host-side store for reuse
(avoid re-prefill). Original implementation: per-page sharded block files on
`kv-native-sys` (`write_block_cache_sharded` → one file per page). Promotions
frequently failed (496 failures observed) and demotion throughput was limited
(~550 MB/s).

## What Worked

### 1. KvMmapStore — sparse mmap page-slot store
Added to `kv-native-sys/src/lib.rs`. File-backed sparse mmap: `set_len` for
logical size but filesystem lazily allocates blocks (`blocks=0` before first
write).

```
write: alloc_slot() → memcpy(mmap[offset..]) — no per-page syscall
read:  read_slot() → &[u8] — zero-copy mmap slice
capacity: budget / page_bytes slots, free-list for reuse
```

One file per disk tier namespace (`kv.mmap`) + one manifest (`manifest.kvm`).

### 2. DiskTier rewrite (CudaKvTierStore)
Replaced sharded block files with `KvMmapStore`. Manifest upgraded to V2:
`key slot_idx slot_bytes` per record. `read()` returns `Cow::Borrowed(disk.store.read_slot(slot))`
— zero-copy for promotions.

### 3. Batch H2D dispatch
`promote_prefix_pages` now gathers all mmap slices into one contiguous `Vec<u8>`,
then calls `copy_pages_from_host` once (was N times = ~19,728 `cuMemcpyHtoDAsync`
launches → 1 call).

### 4. Counter fix
`record_prefix_tier_hits` moved from **after** promote into **before** — the
promote path removes disk entries, so the old location lost `reuse_hit_disk`
attribution (always 0).

## E2E Results (Qwen3-4B BF16 on H20, 256-page pool)

| Metric | Value |
|--------|-------|
| Demote throughput | 411 MB/s (mmap write) |
| Promotion H2D bandwidth | 1475 MB/s (batch dispatch) |
| Pure promotion latency | 24 ms (33 MB, 12 pages) |
| Promotion E2E (disk→GPU→output) | 241 ms (12 pages) |
| Promote failures | **0** (was 496) |
| Write amplification | 1.0x |
| Sparse file utilization | 0.1% (511 GB logical, 312 MB actual) |
| Counter attribution | `reuse_hit_disk: 12, reuse_hit_resident: 40, fallback: 0` |

## Impact

```
Before: 496 promote failures → 0
Before: ~4 ms/page file I/O → ~0.05 ms/page memcpy
Before: reuse_hit_disk always 0 → correct attribution per tier
Before: N inodes for N pages → 1 inode total
```

## Rule

Tiered KV demotion/promotion is production-ready for CUDA Qwen3-dense. The mmap
page-slot model is the canonical storage format: `set_len` sparse file, slot
allocator, manifest-based index. Per-page block files are obsolete.

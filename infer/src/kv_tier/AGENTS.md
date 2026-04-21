# `infer::kv_tier` — Agent Guide

Hierarchical KV cache skeleton: T0 GPU HBM → T1 host pinned DRAM → T2 NVMe →
T3 remote (NIXL/Mooncake/UCX). **Status: skeleton only** — no production
callers today, everything is constructed by unit tests or the in-module
coordinator plumbing.

Load this file before editing anything under `kv_tier/`, and re-read
`docs/projects/tiered-kv-cache.md` before making any design-visible change.

## Tier numbering (2026-04-15 revision)

| Tier | Medium            | Latency  | Status in this module |
|------|-------------------|----------|-----------------------|
| T0   | GPU HBM           | kernel   | **Not here.** Owned by `TokenKVPool` in `crates/cuda-kernels/src/paged_kv.rs`. |
| T1   | Host pinned DRAM  | ~10 µs   | M3 (CUDA only). `host_pool.rs` skeleton exists, locally-verifiable bookkeeping only. |
| T2   | NVMe SSD          | 10–100 µs| `transport/disk.rs` is wired into coordinator spill/rehydrate handling; scheduler watermark callers are still pending. |
| T3   | Remote (NIXL)     | 1–50 µs  | `transport/nixl.rs` stub behind `rdma-nixl` feature. |

**Apple Silicon skips T1.** MLX unified memory makes host↔GPU a self-memcpy.
Metal joins at M4 for T2 disk (bounded wired-memory KV pool).

## Module layout

```
kv_tier.rs              — module root, public re-exports
kv_tier/id.rs           — re-export of crate::types::BlockId (u32)
kv_tier/tier.rs         — Tier enum, BlockLocation, RemoteBlockDesc, TransportId, MemKind
kv_tier/host_pool.rs    — HostPinnedPool, HostPinnedRegion (bookkeeping-only today)
kv_tier/transport.rs    — KVTransport trait + TransferOp + TransportError
kv_tier/transport/disk.rs       — DiskStore (Rust adapter over kv-native-sys Zig object store + future descriptor substrate)
kv_tier/transport/local_cuda.rs — LocalCudaTransport (local-lane plumbing)
kv_tier/transport/nixl.rs       — NixlTransport stub, #[cfg(feature = "rdma-nixl")]
kv_tier/coordinator.rs  — Coordinator, CoordinatorCommand (Demote/Promote/Shutdown), handle + event channel
```

**Do not reintroduce `directory.rs`.** The former `TierDirectory` /
`BlockDescriptor` was deleted in M1 — its fields (`ref_count`, `last_access`,
`session_id`, `pin_until`, `tier`, `location`, `byte_len`) now live on
`crate::prefix_cache::RadixCache`'s private `Node`. One source of truth.

## Invariants (hard — the design hinges on these)

1. **`BlockId` is a pool slot identifier (`u32`), not a content hash.**
   Canonical definition in `crate::types::BlockId`. Content-addressable
   identity uses `crate::types::BlockFingerprint` and only exists at
   persist (M4) or migrate (M5) boundaries.
2. **Only the coordinator moves blocks between tiers.** The scheduler emits
   intents (`Demote`, `Promote`, `Pin`, `Unpin`); the coordinator owns the
   CUDA copy stream and the disk/remote IO queue. Scheduler code **must
   not** issue `TransferOp`s directly.
3. **MR registration stability.** NIXL requires registered memory regions
   to be allocation-stable. `HostPinnedPool` must be allocated once at
   engine init and never reallocated. See `tiered-kv-cache.md §4.2` inv 5
   and §8 pitfall 2.
4. **No `#[cfg(feature = "cuda")]` in this module.** The skeleton is
   always-on so `cargo check --features no-cuda` and `--features metal`
   both validate it. CUDA types (cudarc handles, FlashInfer metadata) live
   in `backend/cuda/` and `crates/cuda-kernels/`.
5. **Coordinator locking.** `RadixCache` is scheduler-thread-owned today.
   It will grow a reader lock only when the M3 coordinator thread starts
   issuing promote/demote writes from a separate OS thread. Do not
   preemptively shard it or wrap it in `dashmap`.
6. **`Tier` ordering is load-bearing.** `Gpu < HostPinned < Disk < Remote`
   is the distance-from-compute order; eviction policies compare tiers with
   this ordering.

## Remote payload opacity

`RemoteBlockDesc.payload` is opaque per-transport bytes. Cross-backend code
must **never parse the payload directly** — only the transport that
produced it can decode. Example payloads documented in `tier.rs`:

- `NixlTransport` (M5): bincode of `(remote_agent_name, addr, len, mem_type, dev_id)`.
- `MooncakeTransport` (post-M5, trigger-gated): bincode of its own handle.

## Pointers

- `docs/projects/tiered-kv-cache.md` — live design doc, revised 2026-04-15.
  §4.1 (tier model), §4.2 (invariants), §5.2 (why the directory was removed),
  §8 (pitfalls).
- `docs/plans/tiered-kv-cache-tasks.md` — milestone ledger (local Mac / remote
  GPU / parallel-GPU lanes).
- `docs/plans/tiered-kv-cache-m2b-remote-acceptance.md` — remote CUDA
  acceptance checklist for the 2026-04-15 M2b batch.
- `docs/experience/errors/2026-04-14-p0-page16-blocker.md` — the NHD/HND
  blocker hit at page_size=16.
- `docs/experience/wins/2026-04-15-tiered-kv-m2b-local.md` — what shipped
  at M2b local (selector flip, resurrection, retain hard cap, tombstone GC).

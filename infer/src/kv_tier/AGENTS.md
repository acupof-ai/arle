# `infer::kv_tier` — Agent Guide

Hierarchical KV cache shape: T0 GPU HBM → T1 host pinned DRAM → T2 NVMe →
T3 remote (shared-fs today; NIXL/Mooncake/UCX later). **Status: partially
live on the CUDA lane** — the scheduler now uses `prefix_cache + paged_kv +
HostPinnedPool + Coordinator + DiskStore/SharedFsStore` for one unified local
path: direct GPU prefix attachment and decode-time COW on T0, host-arena spill
buffering on T1, staged readmission (`host/disk/shared-fs -> host -> T0`),
T1→T2 persistence, and a live `ServerMetrics` surface for coordinator
fetch/store queue depth, waiters, backpressure, and cancellation. Only the
RDMA-class remote transports remain skeletal.

Load this file before editing anything under `kv_tier/`, and re-read
`docs/projects/tiered-kv-cache.md` before making any design-visible change.

## Refactor posture

- Keep KV-tier code simple and uniform. Prefer deletion-style refactors:
  remove speculative side paths, collapse duplicate ownership/state tracking,
  and keep one canonical spill/readmission story instead of partial shadows.

## Tier numbering (2026-04-15 revision)

| Tier | Medium            | Latency  | Status in this module |
|------|-------------------|----------|-----------------------|
| T0   | GPU HBM           | kernel   | **Not here.** Owned by `TokenKVPool` in `crates/cuda-kernels/src/paged_kv.rs`. |
| T1   | Host pinned DRAM  | ~10 µs   | live on CUDA: scheduler demotes GPU blocks into the `kv-native-sys` host arena via `host_pool.rs`, and staged host hits promote back into T0 through `ReadmissionPlan + FetchTicket + WaitingFetch` |
| T2   | NVMe SSD          | 10–100 µs| `transport/disk.rs` is wired into coordinator spill/persist, session restore plumbing, and local staged readmission (`disk -> host -> T0`) |
| T3   | Remote (NIXL)     | 1–50 µs  | `transport/nixl.rs` via `rdma-nixl` (stub) or `rdma-nixl-real`. |

**Apple Silicon skips T1.** MLX unified memory makes host↔GPU a self-memcpy.
Metal joins at M4 for T2 disk (bounded wired-memory KV pool).

## Module layout

```
kv_tier.rs              — module root, public re-exports
kv_tier/backend.rs      — KVBackend trait (node-local / cluster-shared slower-tier surface)
kv_tier/chunk.rs        — KVBlock / KVSpan / KVHandle + index/store/request state enums
kv_tier/id.rs           — re-export of crate::types::BlockId (u32)
kv_tier/io.rs           — KVPayload / KVPayloadRef / backend request-response payloads
kv_tier/lookup.rs       — HitKind / LookupBlock / LookupOutcome / LookupHeuristics: prefix-classification structs that scheduler ingress uses to score radix vs host vs disk hits
kv_tier/policy.rs       — PrefetchPolicy / WritePolicy enums (BestEffort vs WaitComplete; WriteThroughSelective vs WriteBack). The scheduler-side wiring lives in `scheduler/cuda/policy.rs`; this file owns the policy enums themselves so coordinator + scheduler agree on shape.
kv_tier/readmission.rs  — ReadmissionPlan / ReadmissionSource / dedupe keys
kv_tier/tier.rs         — Tier enum, BlockLocation, RemoteBlockDesc, TransportId, MemKind
kv_tier/host_pool.rs    — HostPinnedPool, HostPinnedRegion (Rust wrapper over the `kv-native-sys` native host arena)
kv_tier/transport.rs    — KVTransport trait + TransferOp + TransportError
kv_tier/transport/disk.rs       — DiskStore (Rust adapter over the `kv-native-sys` native object store + descriptor substrate)
kv_tier/transport/local_cuda.rs — LocalCudaTransport (local-lane plumbing; future P0' NVLink peer hop)
kv_tier/transport/nixl.rs       — NixlTransport remote-tier surface, compiled via `rdma-nixl` (stub) or `rdma-nixl-real`
kv_tier/transport/shared_fs.rs  — SharedFsStore: shared-filesystem remote backend (POSIX-visible mount), used as the M4-era cluster-shared transport while RDMA work lands
kv_tier/coordinator.rs + kv_tier/coordinator/  — Coordinator entry surface plus internal split: `builder.rs` (engine-init wiring), `control.rs` (command channel + queue stats / cancellation / backpressure), `events.rs` (completion event fanout), `types.rs` (StoreTarget / QueueControlStats / CoordinatorQueueStats), `bench.rs` (in-tree micro-bench), `tests.rs`. The coordinator owns plan/fetch/store queues, including shared-fs remote fetch/store.
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
2. **Only the coordinator moves blocks between tiers.** The scheduler decides
   which blocks should spill; the coordinator owns the byte movement and
   completion events. Scheduler code **must not** issue `TransferOp`s
   directly.
3. **MR registration stability.** NIXL requires registered memory regions
   to be allocation-stable. `HostPinnedPool` must be allocated once at
   engine init and never reallocated. See `tiered-kv-cache.md §4.2` inv 5
   and §8 pitfall 2.
4. **No `#[cfg(feature = "cuda")]` in this module.** The skeleton is
   always-on so `cargo check --features no-cuda` and `--features metal`
   both validate it. CUDA types (cudarc handles, TileLang metadata) live
   in `backend/cuda/` and `crates/cuda-kernels/`.
5. **Coordinator locking.** `RadixCache` is scheduler-thread-owned today.
   It will grow a reader lock only when the M3 coordinator thread starts
   issuing promote/demote writes from a separate OS thread. Do not
   preemptively shard it or wrap it in `dashmap`.
6. **`Tier` ordering is load-bearing.** `Gpu < HostPinned < Disk < Remote`
   is the distance-from-compute order; eviction policies compare tiers with
   this ordering.
7. **Policy enums vs scheduler wiring split.** `kv_tier::policy` owns
   `PrefetchPolicy` / `WritePolicy` so the coordinator and scheduler share
   one shape. The scheduler-side gate (soft-saturation thresholds,
   prefetch/store decisions) lives in `infer::scheduler::cuda::policy`;
   it must not branch on tier-movement state directly. Add a knob here,
   wire it on the scheduler side, never the reverse.
8. **Lookup classification is centralized in `lookup.rs`.** `HitKind`
   discriminates radix-only / host-staged / disk-staged / remote-staged
   prefix hits; the scheduler calls into `LookupHeuristics` from
   `cuda/runtime/admission.rs`. New tier or staging tier ⇒ extend
   `HitKind` here, not by adding a parallel enum elsewhere.

## Active priority — P2 staged readmission

This module is the live focus of P2 (tiered KV cache validated staged
readmission and remote/shared backends). Current truth:

- M0–M2b local CUDA path live (radix-backed shared pages, tombstone GC,
  retain hard cap). Status snapshot:
  [`docs/experience/wins/2026-04-15-tiered-kv-m2b-local.md`](../../../docs/experience/wins/2026-04-15-tiered-kv-m2b-local.md).
- T1 host pinned + T2 disk + shared-fs remote all wired through the
  coordinator with queue stats / cancellation / backpressure.
- M5 RDMA-class remote transports (`transport/nixl.rs`) remain
  skeletal — design-ready, blocked on M4 stabilization.
- Apple Silicon still skips T1 (unified memory). Metal joins at M4 for
  T2 disk only.

When extending this module, re-read
[`docs/projects/tiered-kv-cache.md`](../../../docs/projects/tiered-kv-cache.md)
and [`docs/plans/tiered-kv-hicache-readmission.md`](../../../docs/plans/tiered-kv-hicache-readmission.md)
first — the milestone ledger and current staged-readmission plan
take precedence over this file when they disagree.

## Remote payload opacity

`RemoteBlockDesc.payload` is opaque per-transport bytes. Cross-backend code
must **never parse the payload directly** — only the transport that
produced it can decode. Example payloads documented in `tier.rs`:

- `NixlTransport` (M5): bincode of `(remote_agent_name, addr, len, mem_type, dev_id)`.
- `MooncakeTransport` (post-M5, trigger-gated): bincode of its own handle.

## Pointers

- `docs/projects/tiered-kv-cache.md` — live design doc and milestone ledger.
  §4.1 (tier model), §4.2 (invariants), §5.2 (why the directory was removed),
  §6 (M0–M5 milestones), §8 (pitfalls).
- `docs/plans/tiered-kv-hicache-readmission.md` — current staged-readmission
  + remote/shared backend plan.
- `docs/experience/wins/2026-04-15-tiered-kv-m2b-local.md` — what shipped
  at M2b local (selector flip, resurrection, retain hard cap, tombstone GC).

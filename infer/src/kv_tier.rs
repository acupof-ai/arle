//! Tiered KV cache — hierarchical KV block tracking across GPU / pinned
//! DRAM / NVMe / remote nodes.
//!
//! This module is the **structural skeleton** for the Tiered KV Cache
//! project. See `docs/projects/tiered-kv-cache.md` for the live design
//! (revised 2026-04-15 after an internal survey + 7-system industry
//! comparison). This file intentionally only documents what is
//! *currently* shipped and what is *actually* in the module tree; every
//! forward-looking plan lives in the design doc, not in rustdoc comments.
//!
//! # Tier model (2026-04-15 numbering, matches project doc §4.1)
//!
//! | Tier | Medium             | Latency class | Status in this module |
//! |------|--------------------|---------------|------------------------|
//! | T0   | GPU HBM            | ~0 (kernel)   | owned by `TokenKVPool` in `backend/cuda/paged_kv.rs`, not represented here |
//! | T1   | Host pinned DRAM   | ~10 µs PCIe   | live on the CUDA lane via `HostPinnedPool` for demotion / spill buffering; no live readmission path yet |
//! | T2   | NVMe SSD           | 10–100 µs     | `transport/disk.rs` backs coordinator spill / persist and session restore plumbing |
//! | T3   | Remote (NIXL/RDMA) | 1–50 µs       | `transport/nixl.rs` stub exists behind `rdma-nixl` feature |
//!
//! The earlier project-doc draft used T0/T2/T3/T4 with T1 intentionally
//! cut; the 2026-04-15 revision renamed to T0/T1/T2/T3 for alignment
//! with vLLM / SGLang HiCache / Mooncake / NVIDIA KVBM documentation.
//! No semantic change — T1 (host pinned DRAM) is still the first
//! non-GPU tier.
//!
//! Apple Silicon skips T1 entirely (MLX unified memory makes host↔GPU
//! a self-memcpy); Metal backend joins the hierarchy only at M4 for T2
//! disk to enforce a bounded wired-memory KV pool. See project doc §10.
//!
//! # Current status (what this module actually contains as of 2026-04-15)
//!
//! **Partially live on the CUDA lane.** The CUDA scheduler now owns one
//! multilayer KV spill/persist path:
//! - radix metadata in `prefix_cache`
//! - T0 page ownership in `paged_kv`
//! - T1 demotion buffering in `HostPinnedPool`
//! - T1→T2 spill and disk persistence in `Coordinator` + `DiskStore`
//!
//! Live staged readmission was removed from the scheduler hot path until the
//! runtime grows a real attach/ownership model. `KVTransport` and remote tiers
//! remain skeletal.
//!
//! The former `directory::TierDirectory` / `BlockDescriptor` holding
//! area was removed in M1 per project doc §5.2. Block metadata that
//! used to live in `BlockDescriptor` (`ref_count`, `last_access`,
//! `session_id`, `pin_until`, `tier`, `location`, `byte_len`) now
//! belongs on [`crate::prefix_cache::RadixCache`]'s private `Node`
//! struct, so there is a single source of truth for in-flight prefix
//! blocks and eviction candidates. Do not reintroduce a parallel
//! directory.
//!
//! # Invariants (current, after the 2026-04-15 revision)
//!
//! 1. **`BlockId` is a pool slot identifier, `u32`.** It is *not* a
//!    content hash. It identifies which slot in which pool holds the
//!    block; it is not stable across restarts or across nodes. The
//!    canonical definition lives at [`crate::types::BlockId`]; this
//!    module re-exports it through [`id::BlockId`].
//!
//! 2. **Content-addressable identity uses [`crate::types::BlockFingerprint`].**
//!    Only constructed when a block is actually persisted (M4 disk
//!    tier) or migrated cross-node (M5 remote tier). Radix-tree nodes
//!    carry `Option<BlockFingerprint>` — `None` for transient
//!    in-memory blocks.
//!
//! 3. **Only the coordinator moves blocks between tiers.** The scheduler
//!    decides *which* blocks should spill; the coordinator owns the
//!    byte-moving work and emits completion events.
//!
//! 4. **MR registration stability.** The NIXL transport requires
//!    registered memory regions to be allocation-stable. The planned
//!    T1 `HostPinnedPool` must be allocated once at engine init and
//!    never reallocated; see project doc §4.2 invariant 5 and §8
//!    pitfall 2.
//!
//! 5. **No cuda dependencies here.** This skeleton is always-on — not
//!    gated behind `#[cfg(feature = "cuda")]` — so `cargo check
//!    --features no-cuda` and `cargo check --features metal` both
//!    validate it. Cuda-specific types (cudarc handles, FlashInfer
//!    metadata) live in `infer/src/backend/cuda/` and in
//!    `crates/cuda-kernels`.
//!
//! # Layout
//!
//! Flat submodule layout:
//! - [`id`] — re-export of [`crate::types::BlockId`] for backward
//!   compatibility with code that imports `kv_tier::BlockId`. The
//!   former `BlockHashCtx` struct was deleted in the 2026-04-15 M0.1
//!   BlockId unification; its content-hash role now belongs to
//!   [`crate::types::BlockFingerprint`].
//! - [`tier`] — [`Tier`], [`BlockLocation`], [`RemoteBlockDesc`],
//!   [`TransportId`], [`MemKind`].
//! - [`transport`] — [`KVTransport`] trait, [`TransferOp`],
//!   [`TransportError`]. `DiskStore` is implemented in
//!   `transport::disk`; `NixlTransport` is a stub in `transport::nixl`
//!   behind `rdma-nixl`.
//!
//! The former `directory` submodule was deleted in M1; its fields
//! (`ref_count`, `last_access`, `session_id`, `pin_until`, `tier`,
//! `location`, `byte_len`) now live on [`crate::prefix_cache::RadixCache`]'s
//! private `Node` struct so there is a single source of truth for
//! in-flight prefix blocks. See project doc §5.2.
//!
//! All publicly useful types are re-exported at the `crate::kv_tier::`
//! root so downstream callers do not need to know the submodule they
//! live in.
//!
//! # Locking strategy
//!
//! Locking for prefix cache state is owned by [`crate::prefix_cache::RadixCache`]
//! (scheduler-thread-owned today; will grow a reader lock when the
//! M3 coordinator thread starts issuing promote/demote writes from a
//! separate OS thread). The 2026-04-13 note about "revisit with
//! `dashmap` or sharded map" is retired: the deleted directory was the
//! only source of write contention it referred to, and its
//! replacement inside `RadixCache` is single-writer by construction.

#[allow(clippy::match_same_arms)]
pub mod coordinator;
pub mod host_pool;
pub mod id;
pub mod lookup;
pub mod tier;
pub mod transport;

pub use coordinator::{
    Coordinator, CoordinatorCommand, CoordinatorEvent, CoordinatorHandle, SpillRequest, SpillTicket,
};
pub use host_pool::{HostPinnedPool, HostPinnedRegion, SharedHostPinnedPool};
pub use id::BlockId;
pub use lookup::{HitKind, LookupBlock, LookupHeuristics, LookupOutcome};
pub use tier::{BlockLocation, MemKind, RemoteBlockDesc, Tier, TransportId};
pub use transport::{KVTransport, TransferOp, TransportError};

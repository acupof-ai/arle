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
//! | T1   | Host pinned DRAM   | ~10 µs PCIe   | planned for M3 (CUDA-only) |
//! | T2   | NVMe SSD           | 10–100 µs     | `transport/disk.rs` impl exists, not yet wired into a coordinator |
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
//! **Skeleton, zero production callers.** Every type defined here is
//! either constructed only by unit tests or imported only by in-module
//! dependencies. No file under `infer/src/scheduler/`,
//! `infer/src/server_engine.rs`, `infer/src/model/`, or
//! `infer/src/backend/` calls into `kv_tier::TierDirectory`,
//! `kv_tier::KVTransport`, or `kv_tier::EvictionPolicy` from the hot
//! path.
//!
//! The project doc calls for M1 to delete [`directory::TierDirectory`]
//! entirely and move its `BlockDescriptor` fields onto the private
//! `Node` struct inside `infer/src/prefix_cache.rs`. Until that lands,
//! the directory remains here as a legacy holding area. Do not extend
//! it; add new fields to the radix-cache node instead.
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
//! 3. **Only the coordinator (not yet implemented, M3+) moves blocks
//!    between tiers.** The scheduler's role is to emit intents
//!    (`Demote`, `Promote`, `Pin`, `Unpin`); the coordinator owns the
//!    CUDA copy stream and the disk / remote IO queue.
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
//!    `crates/infer-cuda-kernels`.
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
//! - [`directory`] — [`BlockDescriptor`], [`TierDirectory`],
//!   [`DirectoryError`]. Scheduled for deletion in M1 per project
//!   doc §5.2 (fields move onto `RadixCache::Node`). Zero production
//!   callers today.
//! - [`transport`] — [`KVTransport`] trait, [`TransferOp`],
//!   [`TransportError`]. `DiskStore` is implemented in
//!   `transport::disk`; `NixlTransport` is a stub in `transport::nixl`
//!   behind `rdma-nixl`.
//!
//! All publicly useful types are re-exported at the `crate::kv_tier::`
//! root so downstream callers do not need to know the submodule they
//! live in.
//!
//! # Locking strategy (legacy, being retired in M1)
//!
//! [`TierDirectory`] currently uses `RwLock<HashMap<..>>` for
//! simplicity. The 2026-04-13 note about "revisit with `dashmap` or
//! sharded map in M3 when the coordinator thread introduces write
//! contention" is now moot: the directory is being deleted in M1 and
//! its replacement lives inside `RadixCache`, whose locking is
//! scheduler-thread-owned. No `dashmap` dependency is needed.

pub mod directory;
pub mod id;
pub mod tier;
pub mod transport;

pub use directory::{BlockDescriptor, DirectoryError, TierDirectory};
pub use id::BlockId;
pub use tier::{BlockLocation, MemKind, RemoteBlockDesc, Tier, TransportId};
pub use transport::{KVTransport, TransferOp, TransportError};

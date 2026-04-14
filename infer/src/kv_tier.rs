//! Tiered KV cache — hierarchical KV block tracking across GPU / pinned
//! DRAM / NVMe / remote nodes.
//!
//! This module is the **structural skeleton** for the Tiered KV Cache
//! project (see `docs/projects/tiered-kv-cache.md`). It establishes the
//! data types and trait surface that P2–P5 will flesh out:
//!
//! | Tier | Medium             | Latency class | Phase that lights it up |
//! |------|--------------------|---------------|-------------------------|
//! | T0   | GPU HBM            | ~0 (kernel)   | P1 directory, P2 evict  |
//! | T2   | Host pinned DRAM   | ~10 µs PCIe   | P2 (CUDA-only)          |
//! | T3   | NVMe SSD           | 10–100 µs     | P3 (CUDA + Metal)       |
//! | T4   | Remote (NIXL/RDMA) | 1–50 µs       | P5 (stub) / P6 (real)   |
//!
//! T1 (GPU-warm) is intentionally cut — same hardware as T0, no capacity
//! gain. Metal backends get a T0 + T3 subset in Phase 3; the rest of the
//! hierarchy is CUDA-only.
//!
//! # Invariants (enforced as phases land)
//!
//! 1. **RadixCache nodes carry `BlockId`, not slots.** The radix tree
//!    never touches physical memory; slot resolution goes through
//!    [`TierDirectory::resolve`].
//! 2. **Only the coordinator moves blocks between tiers.** The scheduler
//!    emits intents (`Demote`, `Promote`, `Pin`, `Unpin`); the
//!    coordinator owns the copy streams and tier transitions.
//! 3. **The directory is the single source of truth for `tier`.** A
//!    block's tier changes atomically at directory commit.
//! 4. **`BlockId` must be stable across restarts and across nodes.** A
//!    content hash (not a monotonic counter) is the plan, so two nodes
//!    that independently prefill the same prefix see the same id. The
//!    actual hashing function lands in P3/P5; until then
//!    [`BlockId::derive`] is a `todo!` stub and the directory accepts
//!    caller-assigned ids.
//! 5. **MR registration stability.** Phase 5 (NIXL transport) requires
//!    the T0 GPU pool and T2 host pool to be allocated once at engine
//!    init and never reallocated, so registered memory regions never
//!    move under the hardware.
//!
//! This skeleton is **always-on** — not gated behind `#[cfg(feature = "cuda")]`
//! — so `cargo check --features no-cuda` and `cargo check --features metal`
//! both validate it. No cuda-specific types (cudarc, FlashInfer) are
//! imported here.
//!
//! # Layout
//!
//! Flat submodule layout:
//! - [`id`] — [`BlockId`], [`BlockHashCtx`]
//! - [`tier`] — [`Tier`], [`BlockLocation`], [`RemoteBlockDesc`],
//!   [`TransportId`], [`MemKind`]
//! - [`directory`] — [`BlockDescriptor`], [`TierDirectory`],
//!   [`DirectoryError`]
//! - [`transport`] — [`KVTransport`] trait, [`TransferOp`],
//!   [`TransportError`]
//!
//! All public types are re-exported at the `crate::kv_tier::` root so
//! downstream callers do not need to know the submodule they live in.
//!
//! # P2+ notes
//!
//! The [`TierDirectory`] currently uses `RwLock<HashMap<...>>` for
//! simplicity and to avoid pulling in `dashmap` at skeleton time. When P2
//! introduces a background coordinator thread the contention profile
//! changes; revisit with `dashmap` or a sharded map at that point.
//! Similarly, [`RemoteBlockDesc::payload`] uses `Vec<u8>` for now;
//! P5/NIXL may want `SmallVec<[u8; 32]>` to keep inline short names.

pub mod directory;
pub mod id;
pub mod tier;
pub mod transport;

pub use directory::{BlockDescriptor, DirectoryError, TierDirectory};
pub use id::BlockId;
pub use tier::{BlockLocation, MemKind, RemoteBlockDesc, Tier, TransportId};
pub use transport::{KVTransport, TransferOp, TransportError};

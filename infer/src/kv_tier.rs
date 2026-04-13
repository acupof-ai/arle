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
//! # P2+ notes
//!
//! The [`TierDirectory`] currently uses `RwLock<HashMap<...>>` for
//! simplicity and to avoid pulling in `dashmap` at skeleton time. When P2
//! introduces a background coordinator thread the contention profile
//! changes; revisit with `dashmap` or a sharded map at that point.
//! Similarly, [`RemoteBlockDesc::payload`] uses `Vec<u8>` for now;
//! P5/NIXL may want `SmallVec<[u8; 32]>` to keep inline short names.

use std::collections::HashMap;
use std::sync::RwLock;
use std::task::Poll;

use infer_core::SessionId;

// ============================================================================
// Block identity
// ============================================================================

/// Content-addressable KV block identifier.
///
/// Stable across processes and across nodes once [`BlockId::derive`] is
/// implemented in P5. Phase 1–4 use caller-assigned values (simple integer
/// counters or hashes of convenience); the type remains opaque so callers
/// can swap implementations without touching downstream code.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BlockId(pub u64);

/// Context for deterministic [`BlockId`] derivation. Locked in at P5; kept
/// public here so P1/P2/P3 call sites can already thread the values through
/// even while [`BlockId::derive`] is a stub.
#[derive(Debug, Clone, Copy)]
pub struct BlockHashCtx {
    /// Fingerprint of the model (architecture + weight digest + numeric
    /// profile). Two different models must produce different hashes even
    /// for the same tokens.
    pub model_fingerprint: u64,
    /// Layer index in the model stack.
    pub layer_idx: u16,
    /// KV dtype selector — so bf16 and int8 variants of the same layer
    /// hash to different ids.
    pub kv_format_tag: u8,
    /// Hash of the parent block along the radix path. Chains the tree
    /// walk into the content so divergence at block granularity is
    /// detectable without walking up the tree.
    pub parent_hash: u64,
}

impl BlockId {
    /// Deterministic derivation used by [`TierDirectory`] insertion.
    ///
    /// **Phase gate**: lands in P5 (content-addressable Tiered KV Cache).
    /// Until then, callers pass caller-assigned ids directly.
    pub fn derive(_ctx: &BlockHashCtx, _tokens: &[u32]) -> Self {
        todo!("P5: blake3 of (model_fingerprint, layer_idx, kv_format_tag, parent_hash, tokens)")
    }
}

// ============================================================================
// Tier taxonomy
// ============================================================================

/// Storage medium for a KV block. Ordering (`Gpu < HostPinned < Disk <
/// Remote`) reflects the distance from compute — nearer first.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Tier {
    /// T0 — GPU HBM. Kernel-accessible.
    Gpu,
    /// T2 — Host pinned DRAM. Coordinator-accessible only.
    HostPinned,
    /// T3 — Local NVMe SSD.
    Disk,
    /// T4 — Remote node, reached over NIXL / Mooncake / UCX.
    Remote,
}

/// Physical location of a block's bytes. Variants match [`Tier`] 1:1.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BlockLocation {
    /// GPU pool slot index. Interpretation is pool-specific.
    Gpu { slot: u32 },
    /// Byte offset within the pinned host pool.
    HostPinned { offset: u64 },
    /// `(file_id, offset)` pair within the disk store. The `file_id` is a
    /// logical handle assigned by the disk transport; the disk store maps
    /// ids to actual filenames.
    Disk { file_id: u32, offset: u64 },
    /// Remote block, opaque per-transport payload.
    Remote { desc: RemoteBlockDesc },
}

impl BlockLocation {
    /// Returns the tier this location lives in.
    pub fn tier(&self) -> Tier {
        match self {
            BlockLocation::Gpu { .. } => Tier::Gpu,
            BlockLocation::HostPinned { .. } => Tier::HostPinned,
            BlockLocation::Disk { .. } => Tier::Disk,
            BlockLocation::Remote { .. } => Tier::Remote,
        }
    }
}

/// Opaque remote descriptor. The `transport` tag identifies which
/// transport impl is responsible for decoding `payload`. Cross-backend
/// code must never parse the payload directly.
///
/// Example payloads:
/// - `NixlTransport` (P5): bincode of `(remote_agent_name, addr, len,
///   mem_type, dev_id)` — fits in ~24–32 bytes for short agent names.
/// - `MooncakeTransport` (P6): bincode of `(segment_handle, offset,
///   length)` = 24 bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteBlockDesc {
    pub transport: TransportId,
    pub payload: Vec<u8>,
}

/// Discriminator for [`RemoteBlockDesc::payload`]. Kept small so the
/// enum fits in one byte and serialization is cheap.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum TransportId {
    /// NVIDIA NIXL — lands in P5 as a stub.
    Nixl = 0,
    /// Mooncake `TransferEngine` — deferred to P6.
    Mooncake = 1,
    /// Reserved for future transports (UCX direct, libfabric, etc.).
    Reserved = 255,
}

/// Memory kind used by [`KVTransport::register`]. Maps 1:1 to NIXL's
/// `MemType` enum (Dram / Vram / Block / Object / File) so future
/// backends have room to grow.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemKind {
    /// CPU-visible pinned DRAM (`cudaHostAlloc` / `cudaHostRegister`).
    Host,
    /// GPU device memory; `device` is the CUDA/ROCm ordinal.
    Vram { device: u32 },
    /// Block-device backed storage (reserved for GPUDirect Storage).
    Block { volume: u32 },
}

// ============================================================================
// Directory
// ============================================================================

/// Metadata for a single KV block across all tiers. This is the only
/// per-block state shared between the radix cache, the scheduler, the
/// coordinator, and any transport backend.
#[derive(Clone, Debug)]
pub struct BlockDescriptor {
    pub id: BlockId,
    pub tier: Tier,
    pub location: BlockLocation,
    /// Byte length of the block's KV data (including any quantization
    /// scales that live alongside the values).
    pub byte_len: u32,
    /// Radix refcount + in-flight request count. `0` means evictable.
    pub ref_count: u32,
    /// Monotonic tick of the last read or promote. Larger = more recent.
    pub last_access: u64,
    /// Owning session for agent-sticky routing and keepalive policies.
    /// `None` for cross-session shared prefixes.
    pub session_id: Option<SessionId>,
    /// Soft pin deadline — the coordinator treats the block as pinned
    /// until this tick elapses, even if `ref_count == 0`. Phase 4
    /// (session keepalive) uses this to avoid evicting warm blocks right
    /// after a turn ends.
    pub pin_until: Option<u64>,
}

/// Single source of truth for block metadata across tiers.
///
/// Phase 1 behavior: a plain `RwLock<HashMap<...>>`. The scheduler thread
/// owns all writes; other threads read through the lock. Phase 2 may swap
/// the backing store for `dashmap::DashMap` when the coordinator
/// introduces real write contention. The public API does not change.
pub struct TierDirectory {
    blocks: RwLock<HashMap<BlockId, BlockDescriptor>>,
}

/// Error surface for [`TierDirectory`] operations. Kept as a small enum
/// so callers can match on it; avoid leaking `anyhow` across the module
/// boundary.
#[derive(Debug, Eq, PartialEq)]
pub enum DirectoryError {
    /// A block with the requested id does not exist in the directory.
    NotFound(BlockId),
    /// An `insert` tried to add a block whose id already exists.
    AlreadyExists(BlockId),
}

impl std::fmt::Display for DirectoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DirectoryError::NotFound(id) => write!(f, "block {id:?} not found in tier directory"),
            DirectoryError::AlreadyExists(id) => {
                write!(f, "block {id:?} already exists in tier directory")
            }
        }
    }
}

impl std::error::Error for DirectoryError {}

impl TierDirectory {
    /// Create an empty directory.
    pub fn new() -> Self {
        Self {
            blocks: RwLock::new(HashMap::new()),
        }
    }

    /// Insert a new block descriptor. Fails if the id already exists —
    /// callers should use [`TierDirectory::promote`] / [`demote`] to
    /// move existing blocks between tiers.
    pub fn insert(&self, desc: BlockDescriptor) -> Result<(), DirectoryError> {
        let mut guard = self.blocks.write().expect("TierDirectory poisoned");
        if guard.contains_key(&desc.id) {
            return Err(DirectoryError::AlreadyExists(desc.id));
        }
        guard.insert(desc.id, desc);
        Ok(())
    }

    /// Return a clone of the descriptor for `id`, or `None` if unknown.
    /// Cloning is cheap — `BlockDescriptor` has no deep fields.
    pub fn resolve(&self, id: BlockId) -> Option<BlockDescriptor> {
        self.blocks
            .read()
            .expect("TierDirectory poisoned")
            .get(&id)
            .cloned()
    }

    /// Atomically rewrite a block's tier and location. Used by the
    /// coordinator after a completed copy has promoted the block to a
    /// higher tier (e.g. T2 → T0 after a prefetch) or demoted it.
    pub fn relocate(
        &self,
        id: BlockId,
        new_tier: Tier,
        new_location: BlockLocation,
    ) -> Result<(), DirectoryError> {
        debug_assert_eq!(
            new_location.tier(),
            new_tier,
            "relocate: new_tier must match new_location.tier()"
        );
        let mut guard = self.blocks.write().expect("TierDirectory poisoned");
        let desc = guard.get_mut(&id).ok_or(DirectoryError::NotFound(id))?;
        desc.tier = new_tier;
        desc.location = new_location;
        Ok(())
    }

    /// Convenience: promote to a higher tier (nearer to GPU).
    pub fn promote(
        &self,
        id: BlockId,
        new_tier: Tier,
        new_location: BlockLocation,
    ) -> Result<(), DirectoryError> {
        self.relocate(id, new_tier, new_location)
    }

    /// Convenience: demote to a lower tier (farther from GPU).
    pub fn demote(
        &self,
        id: BlockId,
        new_tier: Tier,
        new_location: BlockLocation,
    ) -> Result<(), DirectoryError> {
        self.relocate(id, new_tier, new_location)
    }

    /// Update the block's `last_access`. No-op if the block is unknown.
    pub fn touch(&self, id: BlockId, now: u64) {
        let mut guard = self.blocks.write().expect("TierDirectory poisoned");
        if let Some(desc) = guard.get_mut(&id) {
            desc.last_access = now;
        }
    }

    /// Pin the block soft-style until `until_tick`. The coordinator
    /// checks `pin_until > now` when scoring eviction candidates.
    pub fn pin(&self, id: BlockId, until_tick: u64) {
        let mut guard = self.blocks.write().expect("TierDirectory poisoned");
        if let Some(desc) = guard.get_mut(&id) {
            desc.pin_until = Some(until_tick);
        }
    }

    /// Clear any soft pin on the block.
    pub fn unpin(&self, id: BlockId) {
        let mut guard = self.blocks.write().expect("TierDirectory poisoned");
        if let Some(desc) = guard.get_mut(&id) {
            desc.pin_until = None;
        }
    }

    /// Number of block descriptors in the directory. Cheap.
    pub fn len(&self) -> usize {
        self.blocks.read().expect("TierDirectory poisoned").len()
    }

    /// True if the directory has no descriptors.
    pub fn is_empty(&self) -> bool {
        self.blocks
            .read()
            .expect("TierDirectory poisoned")
            .is_empty()
    }
}

impl Default for TierDirectory {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Transport trait (frozen at P5 — see docs/plans/tiered-kv-cache-tasks.md §6)
// ============================================================================

/// One batched transfer instruction handed to the transport. The
/// coordinator builds these and submits them via
/// [`KVTransport::put_batch`] or [`KVTransport::get_batch`].
#[derive(Clone, Debug)]
pub struct TransferOp {
    pub src: BlockLocation,
    pub dst: BlockLocation,
    pub len: u32,
}

/// Transport-layer errors. Intentionally coarse — each impl can decorate
/// the inner string with its own diagnostic; cross-backend code only
/// needs to distinguish the four kinds below.
#[derive(Debug)]
pub enum TransportError {
    /// MR registration failed (out of memory, invalid pointer, hardware
    /// bounds). Typically unrecoverable for this region.
    Registration(String),
    /// A submitted transfer completed with an error (remote failure,
    /// checksum mismatch, local copy engine fault).
    Transfer(String),
    /// An in-flight operation was cancelled via
    /// [`KVTransport::abort`] and then polled to completion.
    Aborted,
    /// Catch-all for transport-specific errors that don't fit the
    /// above. Keep the string short.
    Other(String),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::Registration(msg) => write!(f, "registration failed: {msg}"),
            TransportError::Transfer(msg) => write!(f, "transfer failed: {msg}"),
            TransportError::Aborted => write!(f, "transfer aborted"),
            TransportError::Other(msg) => write!(f, "transport error: {msg}"),
        }
    }
}

impl std::error::Error for TransportError {}

/// Backend-agnostic async KV transfer trait.
///
/// Phase gates:
/// - P2 — `LocalCudaTransport` (cudaMemcpyAsync on a dedicated copy stream)
/// - P3 — `DiskTransport` (tokio::fs default, io_uring behind a feature flag)
/// - P5 — `NixlTransport` (stub via `nixl-sys` with `stub-api` feature)
/// - P6 — `MooncakeTransport` (direct TransferEngine binding)
///
/// **Shape locked** per `docs/plans/tiered-kv-cache-tasks.md §6.3`:
/// `type Op: Send` (NOT `type Completion: Future`) because NIXL has no
/// native `Future` — all four stacks expose polling completion. Keeping
/// the trait Future-free lets each backend hide its own completion model;
/// an adapter `TransportFuture<T>` lives in `infer-engine`, not here.
///
/// **Cancel-safety**: dropping an [`KVTransport::Op`] handle before
/// [`KVTransport::poll`] returns `Ready` is unsound — the underlying
/// hardware may still DMA into the registered buffer. Callers must first
/// call [`KVTransport::abort`] and then poll until `Ready` before
/// dropping the handle or freeing the buffer.
pub trait KVTransport: Send + Sync {
    /// Drop-guarded memory-region handle. Registration is expensive
    /// (page-table pinning + HCA key caching), so callers hold these
    /// across many transfers.
    type Region: Send + Sync;

    /// Per-operation handle. Callers poll it via [`KVTransport::poll`].
    type Op: Send;

    /// Register a byte range as an MR.
    ///
    /// # Safety
    /// `ptr` must remain valid and unmapped for the lifetime of the
    /// returned `Region`. The transport may install the pointer in
    /// hardware page tables; reallocating or freeing the backing pool
    /// while a `Region` is outstanding will cause use-after-free in the
    /// NIC or copy engine. See the Tiered KV Cache invariant 5 at the
    /// top of this module.
    unsafe fn register(
        &self,
        ptr: *mut u8,
        len: usize,
        kind: MemKind,
    ) -> Result<Self::Region, TransportError>;

    /// Drop a region. Default no-op to match backends where registration
    /// is free.
    fn invalidate_region(&self, _region: &Self::Region) -> Result<(), TransportError> {
        Ok(())
    }

    /// Submit a batch of write operations. Returns an opaque handle that
    /// callers poll via [`KVTransport::poll`].
    fn put_batch(&self, ops: &[TransferOp]) -> Result<Self::Op, TransportError>;

    /// Submit a batch of read operations. Same semantics as `put_batch`.
    fn get_batch(&self, ops: &[TransferOp]) -> Result<Self::Op, TransportError>;

    /// Non-blocking poll. Returns `Pending` while the batch is in
    /// flight, `Ready(Ok(()))` on success, or `Ready(Err(_))` on
    /// failure. After `Ready(_)`, the op handle is exhausted; do not
    /// poll it again.
    fn poll(&self, op: &mut Self::Op) -> Poll<Result<(), TransportError>>;

    /// Best-effort cancel. The handle must still be polled to
    /// completion before the caller drops it — see the cancel-safety
    /// note on the trait. Some backends (RDMA) cannot actually stop an
    /// in-flight operation; they record the cancellation and return
    /// [`TransportError::Aborted`] the next time the op is polled.
    fn abort(&self, op: &mut Self::Op);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_desc(id: u64, tier: Tier, location: BlockLocation) -> BlockDescriptor {
        BlockDescriptor {
            id: BlockId(id),
            tier,
            location,
            byte_len: 8192,
            ref_count: 0,
            last_access: 1000,
            session_id: None,
            pin_until: None,
        }
    }

    #[test]
    fn tier_variants_round_trip() {
        // Every tier variant clones cleanly and has a stable Debug
        // representation — enough to catch accidental enum renames.
        for tier in [Tier::Gpu, Tier::HostPinned, Tier::Disk, Tier::Remote] {
            let cloned = tier;
            assert_eq!(cloned, tier);
            assert!(!format!("{tier:?}").is_empty());
        }
    }

    #[test]
    fn block_location_reports_its_tier() {
        let g = BlockLocation::Gpu { slot: 3 };
        let h = BlockLocation::HostPinned { offset: 4096 };
        let d = BlockLocation::Disk {
            file_id: 1,
            offset: 0,
        };
        let r = BlockLocation::Remote {
            desc: RemoteBlockDesc {
                transport: TransportId::Nixl,
                payload: vec![0; 16],
            },
        };
        assert_eq!(g.tier(), Tier::Gpu);
        assert_eq!(h.tier(), Tier::HostPinned);
        assert_eq!(d.tier(), Tier::Disk);
        assert_eq!(r.tier(), Tier::Remote);
    }

    #[test]
    fn directory_insert_resolve_round_trip() {
        let dir = TierDirectory::new();
        assert!(dir.is_empty());
        let desc = sample_desc(100, Tier::Gpu, BlockLocation::Gpu { slot: 0 });
        dir.insert(desc).expect("insert");
        assert_eq!(dir.len(), 1);

        let resolved = dir.resolve(BlockId(100)).expect("resolve");
        assert_eq!(resolved.tier, Tier::Gpu);
        assert_eq!(resolved.byte_len, 8192);
        assert_eq!(resolved.ref_count, 0);
    }

    #[test]
    fn directory_rejects_duplicate_insert() {
        let dir = TierDirectory::new();
        let desc = sample_desc(7, Tier::Gpu, BlockLocation::Gpu { slot: 0 });
        dir.insert(desc.clone()).expect("first insert");
        let err = dir.insert(desc).expect_err("duplicate insert");
        assert_eq!(err, DirectoryError::AlreadyExists(BlockId(7)));
    }

    #[test]
    fn directory_promote_demote_roundtrip() {
        let dir = TierDirectory::new();
        dir.insert(sample_desc(42, Tier::Gpu, BlockLocation::Gpu { slot: 0 }))
            .expect("insert");

        // Promote is semantically identical to relocate; here we demote
        // GPU → HostPinned to free GPU slot 0.
        dir.demote(
            BlockId(42),
            Tier::HostPinned,
            BlockLocation::HostPinned { offset: 0 },
        )
        .expect("demote");
        let after_demote = dir.resolve(BlockId(42)).expect("resolve after demote");
        assert_eq!(after_demote.tier, Tier::HostPinned);
        assert!(matches!(
            after_demote.location,
            BlockLocation::HostPinned { offset: 0 }
        ));

        // Promote back to GPU on a different slot.
        dir.promote(BlockId(42), Tier::Gpu, BlockLocation::Gpu { slot: 5 })
            .expect("promote");
        let after_promote = dir.resolve(BlockId(42)).expect("resolve after promote");
        assert_eq!(after_promote.tier, Tier::Gpu);
        assert!(matches!(
            after_promote.location,
            BlockLocation::Gpu { slot: 5 }
        ));
    }

    #[test]
    fn directory_touch_updates_last_access() {
        let dir = TierDirectory::new();
        dir.insert(sample_desc(
            1,
            Tier::HostPinned,
            BlockLocation::HostPinned { offset: 0 },
        ))
        .expect("insert");
        assert_eq!(dir.resolve(BlockId(1)).unwrap().last_access, 1000);
        dir.touch(BlockId(1), 5000);
        assert_eq!(dir.resolve(BlockId(1)).unwrap().last_access, 5000);
    }

    #[test]
    fn directory_pin_unpin_toggles_pin_until() {
        let dir = TierDirectory::new();
        dir.insert(sample_desc(9, Tier::Gpu, BlockLocation::Gpu { slot: 2 }))
            .expect("insert");
        assert!(dir.resolve(BlockId(9)).unwrap().pin_until.is_none());
        dir.pin(BlockId(9), 12345);
        assert_eq!(dir.resolve(BlockId(9)).unwrap().pin_until, Some(12345));
        dir.unpin(BlockId(9));
        assert!(dir.resolve(BlockId(9)).unwrap().pin_until.is_none());
    }

    #[test]
    fn directory_resolve_missing_returns_none() {
        let dir = TierDirectory::new();
        assert!(dir.resolve(BlockId(404)).is_none());
    }

    #[test]
    fn directory_relocate_missing_returns_not_found() {
        let dir = TierDirectory::new();
        let err = dir
            .relocate(BlockId(404), Tier::Gpu, BlockLocation::Gpu { slot: 0 })
            .expect_err("relocate on missing id");
        assert_eq!(err, DirectoryError::NotFound(BlockId(404)));
    }

    #[test]
    fn session_id_on_descriptor_is_optional_and_clonable() {
        let dir = TierDirectory::new();
        let desc = BlockDescriptor {
            id: BlockId(55),
            tier: Tier::Gpu,
            location: BlockLocation::Gpu { slot: 0 },
            byte_len: 4096,
            ref_count: 1,
            last_access: 0,
            session_id: Some(SessionId::from("agent-session-xyz")),
            pin_until: None,
        };
        dir.insert(desc).expect("insert");
        let out = dir.resolve(BlockId(55)).expect("resolve");
        assert_eq!(
            out.session_id.as_ref().map(|s| s.as_str()),
            Some("agent-session-xyz")
        );
    }
}

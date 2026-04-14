//! [`TierDirectory`] and [`BlockDescriptor`] — the single source of truth
//! for block metadata across tiers.
//!
//! See `crate::kv_tier` for the module-level design notes.

use std::collections::HashMap;
use std::sync::RwLock;

use crate::types::SessionId;

use super::id::BlockId;
use super::tier::{BlockLocation, Tier};

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
    /// callers should use [`TierDirectory::promote`] / [`TierDirectory::demote`] to
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
            out.session_id.as_ref().map(SessionId::as_str),
            Some("agent-session-xyz")
        );
    }
}

//! Core domain types shared across infer workspace crates.

use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Opaque identifier for a KV block that is currently resident in some tier.
///
/// Scope: lives only as long as the block is in memory or on disk; **not**
/// stable across restarts, **not** stable across nodes. Used by the radix
/// tree, the paged KV pool, the block manager, and the tier-aware metadata
/// on `RadixCache::Node`.
///
/// u32 is sufficient for the block counts we expect in any single process:
/// worst-case `page_size=16` + 80 GB T0 HBM + 1 TB T1 host pinned on
/// DeepSeek-V3 (64 layers × 8 KV heads × 128 head_dim × bf16 ≈ 256 KB/block)
/// is roughly 4.5 M blocks — well below 2³². vLLM and SGLang also both use
/// 32-bit block ids. Keeping `BlockId` at `u32` makes radix-tree nodes
/// cache-line friendly.
///
/// **Do not confuse with [`BlockFingerprint`]**, which is the *content*
/// hash used for persistence and cross-node reuse. `BlockId` is a pool
/// slot id, not a content identifier.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct BlockId(pub u32);

/// Content-addressable fingerprint for a KV block's semantic identity.
///
/// The **design target** is a cross-process / cross-node stable hash that
/// lets two independent prefills of the same prefix agree on identity —
/// the foundation for session save/load (tiered-kv-cache M4) and
/// cross-node remote-tier reuse (M5+). The **current implementation**
/// (`compute_from_tokens`) is a local placeholder: two `DefaultHasher`
/// passes with different salt strings packed into 16 bytes. That gives
/// within-process determinism — same tokens → same fingerprint — but
/// **NOT** stability across processes or Rust toolchain upgrades, because
/// `DefaultHasher` is explicitly allowed to change.
///
/// Upgrading to a stable hash (BLAKE3 / xxHash3 + full construction
/// inputs: model fingerprint, layer index, KV format, parent fingerprint,
/// block tokens) is the M4 call site's responsibility. Until then,
/// fingerprints are meaningful only inside a single process run — just
/// enough for the publish-time `RadixCache::insert_with_fingerprints`
/// path and the local `DiskStore` round-trip test.
///
/// Radix-tree nodes carry `Option<BlockFingerprint>`; `None` means a
/// transient in-memory block that never went through the publish path.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct BlockFingerprint(pub [u8; 16]);

impl BlockFingerprint {
    /// Compute a 16-byte content hash over a full-block token slice.
    /// Local placeholder — final hash function (BLAKE3 vs xxHash3) deferred to M4.
    /// Stability within a single process run is the only requirement.
    pub fn compute_from_tokens(tokens: &[u32]) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut h1 = DefaultHasher::new();
        tokens.hash(&mut h1);
        "pegainfer-kv-v1-a".hash(&mut h1);
        let v1 = h1.finish();

        let mut h2 = DefaultHasher::new();
        tokens.hash(&mut h2);
        "pegainfer-kv-v1-b".hash(&mut h2);
        let v2 = h2.finish();

        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&v1.to_le_bytes());
        bytes[8..].copy_from_slice(&v2.to_le_bytes());
        BlockFingerprint(bytes)
    }
}

/// Stable request identifier across scheduler/runtime boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

/// Client-supplied session/conversation identifier used by the scheduler to
/// route subsequent turns of the same agent session to the slot or radix
/// subtree that already holds their KV prefix.
///
/// The value is opaque to the engine. Callers should treat it as a stable
/// per-conversation key (typically a UUID or hash of the conversation array).
/// It is the protocol-level plumbing for
/// `docs/projects/agent-first-architecture.md::A2` (session-sticky routing);
/// admission logic consumes it once `A1` wires the RadixCache into the
/// CUDA/Metal schedulers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(Arc<str>);

impl SessionId {
    /// Wrap an arbitrary string as a session id. Empty strings are rejected by
    /// callers; this type does not enforce non-emptiness because higher layers
    /// decide what to do with invalid input.
    pub fn new(id: impl Into<Arc<str>>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for SessionId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for SessionId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for SessionId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SessionId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from(value))
    }
}

/// Canonical inference mode used by control/data-plane orchestration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMode {
    Prefill,
    Decode,
}

/// Lifecycle action emitted by scheduler-like components.
///
/// These events are intentionally action-oriented instead of state-oriented so
/// consumers can distinguish initial admission from preemption/requeue, and
/// lifecycle transitions from chunked work units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestEventKind {
    Enqueued,
    Requeued,
    PrefillStarted,
    DecodeStep,
    Evicted,
    Completed,
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_id_is_hashable_and_copy() {
        let a = RequestId(7);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn block_id_is_copy_and_ordered() {
        let a = BlockId(1);
        let b = a;
        assert_eq!(a, b);
        assert!(BlockId(1) < BlockId(2));
    }

    #[test]
    fn block_fingerprint_round_trips() {
        let bytes = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let fp = BlockFingerprint(bytes);
        assert_eq!(fp.0, bytes);
        assert_eq!(fp, BlockFingerprint(bytes));
    }

    #[test]
    fn fingerprint_is_deterministic_and_sensitive() {
        let a = BlockFingerprint::compute_from_tokens(&[1, 2, 3, 4]);
        let b = BlockFingerprint::compute_from_tokens(&[1, 2, 3, 4]);
        let c = BlockFingerprint::compute_from_tokens(&[1, 2, 3, 5]);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn request_event_kind_progression_example() {
        let events = [
            RequestEventKind::Enqueued,
            RequestEventKind::Requeued,
            RequestEventKind::PrefillStarted,
            RequestEventKind::DecodeStep,
            RequestEventKind::Evicted,
            RequestEventKind::Completed,
            RequestEventKind::Cancelled,
        ];
        assert_eq!(events.len(), 7);
    }

    #[test]
    fn session_id_round_trips_from_str_and_string() {
        let from_str = SessionId::from("abc-123");
        let from_string = SessionId::from(String::from("abc-123"));
        assert_eq!(from_str, from_string);
        assert_eq!(from_str.as_str(), "abc-123");
        assert_eq!(from_str.to_string(), "abc-123");
    }

    #[test]
    fn session_id_is_hashable_and_cheap_clone() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let a = SessionId::new("agent-session-1");
        let b = a.clone();
        set.insert(a);
        assert!(set.contains(&b));
    }
}

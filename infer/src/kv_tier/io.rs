//! Data-plane types for the HiCache-aligned KV tier surface.
//!
//! These types describe *where the bytes live* and *what a backend moves*.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::{
    chunk::{KVBlock, KVHandle},
    tier::{BlockLocation, Tier},
};

pub type KVBytes = Arc<[u8]>;

/// Half-open byte range `[offset, offset + len)` within a payload.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct KVByteRange {
    pub offset: u64,
    pub len: u64,
}

impl KVByteRange {
    pub fn new(offset: u64, len: u64) -> Self {
        Self { offset, len }
    }

    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Byte-oriented reference into a concrete location.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KVPayloadRef {
    pub location: BlockLocation,
    pub range: KVByteRange,
}

impl KVPayloadRef {
    pub fn new(location: BlockLocation, range: KVByteRange) -> Self {
        Self { location, range }
    }

    pub fn whole(location: BlockLocation, len: u64) -> Self {
        Self {
            location,
            range: KVByteRange::new(0, len),
        }
    }

    pub fn len(&self) -> u64 {
        self.range.len
    }

    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    pub fn tier(&self) -> Tier {
        self.location.tier()
    }
}

/// Owned byte payload used by the backend surface.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KVPayload {
    pub bytes: KVBytes,
    pub reference: Option<KVPayloadRef>,
}

impl KVPayload {
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Arc::<[u8]>::from(bytes),
            reference: None,
        }
    }

    pub fn with_reference(bytes: Vec<u8>, reference: KVPayloadRef) -> Self {
        Self {
            bytes: Arc::<[u8]>::from(bytes),
            reference: Some(reference),
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn as_slice(&self) -> &[u8] {
        self.bytes.as_ref()
    }
}

/// Backend write request: control-plane metadata + data-plane payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KVBackendStore {
    pub handle: KVHandle,
    pub block: KVBlock,
    pub kv_format_tag: u8,
    pub payload: KVPayload,
}

/// Backend read request: resolve bytes for an existing handle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KVBackendFetch {
    pub handle: KVHandle,
}

/// Backend delete request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KVBackendDelete {
    pub handle: KVHandle,
}

/// Completion from an object backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KVBackendCompletion {
    Stored(KVHandle),
    Loaded {
        handle: KVHandle,
        payload: KVPayload,
    },
    Deleted(KVHandle),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_ref_whole_sets_full_range() {
        let payload = KVPayloadRef::whole(BlockLocation::Gpu { slot: 4 }, 8192);
        assert_eq!(payload.range.offset, 0);
        assert_eq!(payload.range.len, 8192);
        assert_eq!(payload.tier(), Tier::Gpu);
    }

    #[test]
    fn payload_wraps_bytes_without_copying_at_callsite() {
        let payload = KVPayload::from_vec(vec![1, 2, 3, 4]);
        assert_eq!(payload.len(), 4);
        assert_eq!(payload.as_slice(), &[1, 2, 3, 4]);
        assert!(payload.reference.is_none());
    }
}

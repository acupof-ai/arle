//! Canonical control-plane KV objects for the HiCache-aligned tier surface.
//!
//! These types describe *what* the cache is tracking. They intentionally do
//! not own raw payload bytes; data-plane buffers live in [`super::io`].

use serde::{Deserialize, Serialize};

use crate::types::{BlockFingerprint, BlockId};

use super::tier::{BlockLocation, Tier};

/// Index-visible readiness state for a cached block/span entry.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Default)]
pub enum IndexEntryState {
    #[default]
    Ready,
    Pending,
    Evicting,
}

/// Request-visible state for one staged chunk/span.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Default)]
pub enum RequestChunkState {
    #[default]
    New,
    Planned,
    Fetching,
    Ready,
    Consumed,
}

/// Store-side persistence state for a cached block/span entry.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Default)]
pub enum StoreState {
    #[default]
    Idle,
    Pending,
    Storing,
    Stored,
    Failed,
}

/// Stable in-process identifier for a radix-edge-aligned KV span.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct KVSpanId(pub u64);

/// Half-open layer range `[start, end)` covered by a block or span.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct LayerRange {
    pub start: u16,
    pub end: u16,
}

impl LayerRange {
    pub fn new(start: u16, end: u16) -> Self {
        assert!(start <= end, "LayerRange start must be <= end");
        Self { start, end }
    }

    pub fn len(&self) -> u16 {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Half-open token range `[start, end)` covered by a block or span.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct TokenRange {
    pub start: u32,
    pub end: u32,
}

impl TokenRange {
    pub fn new(start: u32, end: u32) -> Self {
        assert!(start <= end, "TokenRange start must be <= end");
        Self { start, end }
    }

    pub fn len(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Smallest control-plane unit tracked by allocators and persistence.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KVBlock {
    pub block_id: BlockId,
    pub layer_range: LayerRange,
    pub token_range: TokenRange,
    pub byte_len: u64,
    pub fingerprint: Option<BlockFingerprint>,
}

impl KVBlock {
    pub fn new(
        block_id: BlockId,
        layer_range: LayerRange,
        token_range: TokenRange,
        byte_len: u64,
    ) -> Self {
        Self {
            block_id,
            layer_range,
            token_range,
            byte_len,
            fingerprint: None,
        }
    }

    #[must_use]
    pub fn with_fingerprint(mut self, fingerprint: BlockFingerprint) -> Self {
        self.fingerprint = Some(fingerprint);
        self
    }
}

/// Continuous prefix segment backed by one or more blocks.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KVSpan {
    pub span_id: KVSpanId,
    pub prefix_fingerprint: Option<BlockFingerprint>,
    pub blocks: Vec<KVBlock>,
    pub epoch: u64,
}

impl KVSpan {
    pub fn new(span_id: KVSpanId, blocks: Vec<KVBlock>, epoch: u64) -> Self {
        Self {
            span_id,
            prefix_fingerprint: None,
            blocks,
            epoch,
        }
    }

    pub fn byte_len(&self) -> u64 {
        self.blocks.iter().map(|block| block.byte_len).sum()
    }

    pub fn token_range(&self) -> Option<TokenRange> {
        let first = self.blocks.first()?;
        let last = self.blocks.last()?;
        Some(TokenRange::new(
            first.token_range.start,
            last.token_range.end,
        ))
    }
}

/// Lightweight control-plane reference that points at a concrete location.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KVHandle {
    pub span_id: KVSpanId,
    pub block_id: BlockId,
    pub location: BlockLocation,
    pub epoch: u64,
    pub byte_len: u64,
}

impl KVHandle {
    pub fn new(
        span_id: KVSpanId,
        block_id: BlockId,
        location: BlockLocation,
        epoch: u64,
        byte_len: u64,
    ) -> Self {
        Self {
            span_id,
            block_id,
            location,
            epoch,
            byte_len,
        }
    }

    #[must_use]
    pub fn with_location(&self, location: BlockLocation) -> Self {
        Self {
            span_id: self.span_id,
            block_id: self.block_id,
            location,
            epoch: self.epoch,
            byte_len: self.byte_len,
        }
    }

    pub fn tier(&self) -> Tier {
        self.location.tier()
    }
}

/// Queue-dedup key for async span work.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SpanTaskKey {
    pub span_id: KVSpanId,
    pub epoch: u64,
    pub dst_tier: Tier,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_byte_len_and_token_range_follow_blocks() {
        let blocks = vec![
            KVBlock::new(
                BlockId(1),
                LayerRange::new(0, 2),
                TokenRange::new(0, 16),
                4096,
            ),
            KVBlock::new(
                BlockId(2),
                LayerRange::new(0, 2),
                TokenRange::new(16, 32),
                4096,
            ),
        ];
        let span = KVSpan::new(KVSpanId(7), blocks, 3);

        assert_eq!(span.byte_len(), 8192);
        assert_eq!(span.token_range(), Some(TokenRange::new(0, 32)));
    }

    #[test]
    fn handle_location_updates_do_not_mutate_original() {
        let handle = KVHandle::new(
            KVSpanId(11),
            BlockId(5),
            BlockLocation::Gpu { slot: 3 },
            9,
            1024,
        );

        let moved = handle.with_location(BlockLocation::HostPinned { offset: 4096 });

        assert_eq!(handle.location, BlockLocation::Gpu { slot: 3 });
        assert_eq!(moved.location, BlockLocation::HostPinned { offset: 4096 });
        assert_eq!(moved.span_id, handle.span_id);
        assert_eq!(moved.block_id, handle.block_id);
    }
}

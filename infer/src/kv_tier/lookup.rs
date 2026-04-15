//! Lookup/staging result types for tiered KV cache M3b.
//!
//! These types let the radix cache describe "ready now" vs "needs staging"
//! without teaching the scheduler every tier-specific detail.

use crate::types::BlockId;

use super::tier::BlockLocation;

/// Scheduler-visible result of matching one cached block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitKind {
    /// Bytes are already in T0 and can be decoded immediately.
    ReadyOnGpu,
    /// Bytes live in T1 and must be copied back to T0 first.
    StagingFromHost,
    /// Bytes live in T2/T3 and must be fetched before decode.
    StagingFromDisk,
    /// Only a tombstone or index entry exists; no reusable bytes remain.
    Miss,
}

/// One matched block from a lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupBlock {
    /// `None` means the radix tree only held an index/tombstone entry.
    pub block_id: Option<BlockId>,
    pub hit_kind: HitKind,
}

/// Concrete staging request handed to the coordinator/planner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageRequest {
    pub block_id: BlockId,
    pub from: BlockLocation,
    pub byte_len: u32,
}

/// Opaque ticket representing one staged lookup batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StageTicket(pub u64);

/// Full lookup result surfaced by `RadixCache::lookup_or_stage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupOutcome {
    /// Reusable prefix length in tokens.
    pub matched_len: usize,
    pub blocks: Vec<LookupBlock>,
    /// Present when a planner/coordinator accepted staging work.
    pub staging_ticket: Option<StageTicket>,
    /// True when fetching staged blocks is likely slower than recomputing.
    pub recompute_advised: bool,
}

impl LookupOutcome {
    pub fn new(
        matched_len: usize,
        blocks: Vec<LookupBlock>,
        staging_ticket: Option<StageTicket>,
        recompute_advised: bool,
    ) -> Self {
        Self {
            matched_len,
            blocks,
            staging_ticket,
            recompute_advised,
        }
    }
}

/// Read-only lookup-time bandwidth model used for recompute-vs-fetch advice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LookupHeuristics {
    pub prefill_tokens_per_sec: f32,
    pub host_bandwidth_bytes_per_sec: f32,
    pub disk_bandwidth_bytes_per_sec: f32,
}

impl Default for LookupHeuristics {
    fn default() -> Self {
        Self {
            prefill_tokens_per_sec: 30_000.0,
            host_bandwidth_bytes_per_sec: 25.0 * 1024.0 * 1024.0 * 1024.0,
            disk_bandwidth_bytes_per_sec: 3.0 * 1024.0 * 1024.0 * 1024.0,
        }
    }
}

impl LookupHeuristics {
    /// Returns true when recomputing the staged prefix is likely cheaper than
    /// fetching it from the current tier.
    pub fn advise_recompute(
        &self,
        hit_kind: HitKind,
        staging_tokens: usize,
        staging_bytes: u64,
    ) -> bool {
        let bandwidth = match hit_kind {
            HitKind::StagingFromHost => self.host_bandwidth_bytes_per_sec,
            HitKind::StagingFromDisk => self.disk_bandwidth_bytes_per_sec,
            HitKind::ReadyOnGpu | HitKind::Miss => return false,
        };

        if staging_tokens == 0
            || staging_bytes == 0
            || self.prefill_tokens_per_sec <= 0.0
            || bandwidth <= 0.0
        {
            return false;
        }

        let fetch_seconds = staging_bytes as f32 / bandwidth;
        let recompute_seconds = staging_tokens as f32 / self.prefill_tokens_per_sec;
        fetch_seconds > recompute_seconds
    }
}

/// Minimal abstraction used by `RadixCache::lookup_or_stage` to request
/// background staging without depending directly on a concrete coordinator.
pub trait StagePlanner {
    fn stage(&self, requests: &[StageRequest]) -> Option<StageTicket>;
}

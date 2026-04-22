//! Lookup result types for tiered KV cache M3b.
//!
//! These types let the radix cache classify reusable prefix blocks without
//! dragging raw storage/RDMA calls into the scheduler hot path. The local CUDA
//! runtime turns staged hits into `ReadmissionPlan + FetchTicket + WaitingFetch`
//! after this classification step.

use crate::types::BlockId;

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

/// Full lookup result surfaced by `RadixCache::lookup_or_stage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupOutcome {
    /// Reusable prefix length in tokens.
    pub matched_len: usize,
    pub blocks: Vec<LookupBlock>,
    /// True when fetching staged blocks is likely slower than recomputing.
    pub recompute_advised: bool,
}

impl LookupOutcome {
    pub fn new(matched_len: usize, blocks: Vec<LookupBlock>, recompute_advised: bool) -> Self {
        Self {
            matched_len,
            blocks,
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
            // Use end-to-end long-prefill throughput rather than an optimistic
            // kernel-only number. The scheduler compares staged slower-tier
            // recall against a full cold-prefill fallback; on current Qwen3
            // CUDA hosts that path is in the low-thousands tok/s once queueing,
            // launch, and model overheads are included.
            prefill_tokens_per_sec: 5_000.0,
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

#[cfg(test)]
mod tests {
    use super::{HitKind, LookupHeuristics};

    #[test]
    fn default_heuristics_stage_qwen3_sized_disk_blocks() {
        let heuristics = LookupHeuristics::default();
        assert!(
            !heuristics.advise_recompute(HitKind::StagingFromDisk, 16, 2 * 1024 * 1024),
            "default heuristics should prefer staged recall over cold recompute for a 2MiB / 16-token disk block",
        );
    }

    #[test]
    fn explicit_fast_prefill_and_slow_disk_can_still_recompute() {
        let heuristics = LookupHeuristics {
            prefill_tokens_per_sec: 200_000.0,
            host_bandwidth_bytes_per_sec: 25.0 * 1024.0 * 1024.0 * 1024.0,
            disk_bandwidth_bytes_per_sec: 50.0 * 1024.0 * 1024.0,
        };
        assert!(heuristics.advise_recompute(HitKind::StagingFromDisk, 16, 2 * 1024 * 1024));
    }
}

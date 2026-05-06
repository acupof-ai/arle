//! Minimal policy layer for tiered-KV prefetch and write-back decisions.
//!
//! The policy enums themselves (`PrefetchPolicy`, `WritePolicy`) live in
//! `crate::kv_tier::policy` since they describe coordinator-owned
//! tier-movement decisions, not scheduler-only concerns. This module
//! holds the scheduler's wiring of those policies into the tiered-KV
//! decision points.

use crate::kv_tier::coordinator::{CoordinatorQueueStats, QueueControlStats, StoreTarget};
use crate::kv_tier::policy::{PrefetchPolicy, WritePolicy};
use crate::kv_tier::{BlockId, KvTierAdapter, Tier};
use crate::prefix_cache::BlockMetadata;

#[derive(Clone, Debug)]
pub(super) struct TieredKvPolicy {
    fetch_soft_limit: f64,
    store_soft_limit: f64,
    remote_store_min_hits: u32,
    prefetch_policy: PrefetchPolicy,
    write_policy: WritePolicy,
}

impl Default for TieredKvPolicy {
    fn default() -> Self {
        Self {
            fetch_soft_limit: 0.75,
            store_soft_limit: 0.75,
            remote_store_min_hits: 2,
            prefetch_policy: PrefetchPolicy::BestEffort,
            write_policy: WritePolicy::WriteThroughSelective,
        }
    }
}

impl TieredKvPolicy {
    pub(super) fn allow_prefetch(&self, queue: QueueControlStats) -> bool {
        match self.prefetch_policy {
            PrefetchPolicy::BestEffort => !queue.soft_saturated(self.fetch_soft_limit),
            PrefetchPolicy::WaitComplete => true,
        }
    }

    pub(super) fn allow_store(&self, queue: QueueControlStats) -> bool {
        !queue.soft_saturated(self.store_soft_limit)
    }

    pub(super) fn choose_store_target(
        &self,
        metadata: &BlockMetadata,
        stats: CoordinatorQueueStats,
        cluster_backend_ready: bool,
    ) -> StoreTarget {
        if !cluster_backend_ready {
            return StoreTarget::Disk;
        }
        match self.write_policy {
            WritePolicy::WriteThrough => StoreTarget::Remote,
            WritePolicy::WriteThroughSelective => {
                if metadata.hit_count >= self.remote_store_min_hits && self.allow_store(stats.store)
                {
                    StoreTarget::Remote
                } else {
                    StoreTarget::Disk
                }
            }
        }
    }
}

impl KvTierAdapter for TieredKvPolicy {
    fn paged_pool_pressure(&self) -> f64 {
        0.0
    }

    fn submit_demote(&self, _block_id: BlockId) -> anyhow::Result<()> {
        Ok(())
    }

    fn submit_promote(&self, _block_id: BlockId, _tier: Tier) -> anyhow::Result<()> {
        Ok(())
    }
}

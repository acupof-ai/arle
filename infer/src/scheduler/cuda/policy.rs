//! Minimal policy layer for tiered-KV prefetch and write-back decisions.

use crate::kv_tier::coordinator::{CoordinatorQueueStats, QueueControlStats, StoreTarget};
use crate::prefix_cache::BlockMetadata;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(dead_code)] // `WaitComplete` reserved for future policy mode.
pub(super) enum PrefetchMode {
    BestEffort,
    WaitComplete,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(dead_code)] // `WriteThrough` reserved for future policy mode.
pub(super) enum WriteBackMode {
    WriteThrough,
    WriteThroughSelective,
}

#[derive(Clone, Debug)]
pub(super) struct TieredKvPolicy {
    fetch_soft_limit: f64,
    store_soft_limit: f64,
    remote_store_min_hits: u32,
    prefetch_mode: PrefetchMode,
    write_back_mode: WriteBackMode,
}

impl Default for TieredKvPolicy {
    fn default() -> Self {
        Self {
            fetch_soft_limit: 0.75,
            store_soft_limit: 0.75,
            remote_store_min_hits: 2,
            prefetch_mode: PrefetchMode::BestEffort,
            write_back_mode: WriteBackMode::WriteThroughSelective,
        }
    }
}

impl TieredKvPolicy {
    pub(super) fn allow_prefetch(&self, queue: QueueControlStats) -> bool {
        match self.prefetch_mode {
            PrefetchMode::BestEffort => !queue.soft_saturated(self.fetch_soft_limit),
            PrefetchMode::WaitComplete => true,
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
        match self.write_back_mode {
            WriteBackMode::WriteThrough => StoreTarget::Remote,
            WriteBackMode::WriteThroughSelective => {
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

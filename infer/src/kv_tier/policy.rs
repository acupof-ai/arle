//! Minimal policy layer for tiered-KV prefetch and write-back decisions.

use super::coordinator::{CoordinatorQueueStats, QueueControlStats, StoreTarget};
use crate::prefix_cache::BlockMetadata;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PrefetchMode {
    BestEffort,
    WaitComplete,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WriteBackMode {
    WriteThrough,
    WriteThroughSelective,
}

#[derive(Clone, Debug)]
pub struct TieredKvPolicy {
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
    pub fn allow_prefetch(&self, queue: QueueControlStats) -> bool {
        match self.prefetch_mode {
            PrefetchMode::BestEffort => !queue.soft_saturated(self.fetch_soft_limit),
            PrefetchMode::WaitComplete => true,
        }
    }

    pub fn allow_store(&self, queue: QueueControlStats) -> bool {
        !queue.soft_saturated(self.store_soft_limit)
    }

    pub fn choose_store_target(
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

//! Read-only KV pool queries.
//!
//! Host-indexed observation surface: slot lengths, page maps, free capacity,
//! and occupant epochs. The scheduler reads through this trait without needing
//! the mutating allocation or prefix-store surfaces.

/// Read-only host-indexed KV pool queries visible to engine-core.
///
/// Every method is expressed in host slot ids, page ids, token counts, and
/// logical positions. The trait is dyn-safe.
pub trait KvQuery {
    fn is_active(&self) -> bool;

    fn page_size(&self) -> usize;

    fn free_pages(&self) -> usize;

    /// Pages currently resident in the fast working pool. Default 0 for pools
    /// that do not expose a standard host-indexed page pool.
    fn resident_pages(&self) -> usize {
        0
    }

    /// Resident pages retained only by the prefix cache and therefore
    /// reclaimable by cache eviction. Default 0 for non-standard pools.
    ///
    /// Must count exactly the pages for which [`KvQuery::page_is_evictable`]
    /// returns true — the scheduler's capacity repair budgets against this
    /// count and the evictor filters victims with the predicate, so a
    /// divergent pair silently masks a shortfall (#164 residual).
    fn resident_evictable_pages(&self) -> usize {
        0
    }

    /// Whether evicting `page` from the prefix cache would actually return it
    /// to the free pool: retained exactly once (the cache's own ref) and not
    /// attached to any live slot. Releasing a page a live slot still writes
    /// must never recycle it — that aliases two slots onto one physical page.
    /// Default `true` for pools without per-page tracking.
    fn page_is_evictable(&self, _page: u32) -> bool {
        true
    }

    fn seq_len(&self, slot: usize) -> usize;

    fn slot_epoch(&self, slot: usize) -> u64;

    fn append_pages_needed(&self, slot: usize, tokens: usize) -> usize;

    /// For fixed-band pools (DSv4), the number of physical pages each slot is
    /// pre-allocated. `None` for token-grown pools (Qwen dense/Metal).
    fn fixed_pages_per_slot(&self) -> Option<usize> {
        None
    }

    fn page_indices(&self, slot: usize) -> &[u32];

    fn page_indices_for_token_range(&self, slot: usize, len: usize) -> &[u32];

    /// Number of pages this pool holds per `global_pages` total pages.
    /// Under CP sequence-sharding, returns the local shard's share; without
    /// sharding, returns `global_pages`.
    fn shard_local_page_count(&self, global_pages: usize) -> usize {
        global_pages
    }
}

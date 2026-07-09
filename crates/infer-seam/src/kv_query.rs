//! Read-only KV pool queries.
//!
//! Host-indexed observation surface: slot lengths, page maps, free capacity,
//! and occupant epochs. The scheduler reads through this trait without needing
//! the mutating allocation or prefix-store surfaces.

/// Read-only host-indexed KV pool queries visible to engine-core.
///
/// Implementations may own GPU, Metal, CPU, or remote buffers internally, but
/// every method is expressed in host slot ids, page ids, token counts, and
/// logical positions. The trait is dyn-safe.
pub trait KvQuery {
    /// Return whether the pool has live storage backing it.
    fn is_active(&self) -> bool;

    /// Return the number of tokens stored per physical page.
    fn page_size(&self) -> usize;

    /// Return the number of free physical pages.
    fn free_pages(&self) -> usize;

    /// Return the number of logical tokens still allocatable without eviction.
    fn free_tokens(&self) -> usize;

    /// Return pages currently resident in the fast working pool. Default 0 for
    /// pools that do not expose a standard host-indexed page pool.
    fn resident_pages(&self) -> usize {
        0
    }

    /// Return resident pages retained only by the prefix cache and therefore
    /// reclaimable by cache eviction. Default 0 for non-standard pools.
    fn resident_evictable_pages(&self) -> usize {
        0
    }

    /// Return the logical sequence length for `slot`.
    fn seq_len(&self, slot: usize) -> usize;

    /// Return the current logical occupant epoch for `slot`.
    fn slot_epoch(&self, slot: usize) -> u64;

    /// Return the number of extra pages needed to append `tokens` to `slot`.
    fn append_pages_needed(&self, slot: usize, tokens: usize) -> usize;

    /// For fixed-band pools (DSv4), return the number of physical pages each
    /// slot is pre-allocated. `None` for token-grown pools (Qwen dense/Metal).
    fn fixed_pages_per_slot(&self) -> Option<usize> {
        None
    }

    /// Return physical page ids for `slot` in logical-page order.
    fn page_indices(&self, slot: usize) -> &[u32];

    /// Return physical page ids that cover a logical token range in `slot`.
    fn page_indices_for_token_range(&self, slot: usize, start: usize, len: usize) -> &[u32];
}

//! Mutating KV pool allocation.
//!
//! Slot growth, detached-page allocation, truncation, and release —
//! the write surface the scheduler drives as requests advance.

/// Mutating host-indexed KV pool allocation visible to engine-core.
///
/// Every method is expressed in host slot ids, page ids, and token counts. The
/// trait is dyn-safe.
pub trait KvAllocator {
    fn alloc(&mut self, slot: usize, tokens: usize) -> anyhow::Result<()>;

    fn alloc_detached_pages(&mut self, pages: usize) -> anyhow::Result<Vec<u32>>;

    /// Return detached pages (from [`KvAllocator::alloc_detached_pages`]) to
    /// the free pool. Must only be called with pages that were never attached
    /// to a slot or retained by the prefix store.
    fn free_detached_pages(&mut self, pages: &[u32]);

    fn free_slot(&mut self, slot: usize);

    /// **Write-through evict-drop** — release one *middle* physical page of a
    /// live slot back to the free pool while the slot keeps decoding, leaving its
    /// logical length unchanged (the dropped page's KV is already mirrored to the
    /// tier by `KvTier::write_through`). The page is identified by its *logical*
    /// page index within the slot (`token_pos / page_size`); the implementation
    /// replaces that logical slot with an evicted sentinel so the remaining pages
    /// stay logically addressable, and returns the freed physical page id (or
    /// `None` if that logical page was already evicted / pinned / out of range).
    ///
    /// This is the one allocator-level operation that makes HBM bounded as a
    /// session grows: it is the *real* page free behind the write-through tiered
    /// KV model. No in-tree caller today — the `--kv-recall` driver was deleted
    /// (3f826c204); kept as the hole-tolerance primitive a remote L3 needs, so
    /// the default impl is a no-op and the baseline stays byte-identical.
    fn evict_slot_page(&mut self, _slot: usize, _logical_page: usize) -> Option<u32> {
        None
    }

    /// Inverse of [`KvAllocator::evict_slot_page`]: claim a free page back into
    /// `logical_page`. `None` if it is not evicted or the pool is empty.
    /// Default no-op (no in-tree caller; see `evict_slot_page`).
    fn reinstate_slot_page(&mut self, _slot: usize, _logical_page: usize) -> Option<u32> {
        None
    }

    fn truncate_slot(&mut self, slot: usize, new_len: usize) -> anyhow::Result<()>;
}

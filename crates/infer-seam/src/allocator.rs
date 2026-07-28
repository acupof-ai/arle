//! Mutating KV pool allocation.
//!
//! Slot growth, detached-page allocation, truncation, release, and migration —
//! the write surface the scheduler drives as requests advance.

/// Mutating host-indexed KV pool allocation visible to engine-core.
///
/// Every method is expressed in host slot ids, page ids, and token counts. The
/// trait is dyn-safe.
pub trait KvAllocator {
    /// Allocate `tokens` logical tokens for `slot`.
    fn alloc(&mut self, slot: usize, tokens: usize) -> anyhow::Result<()>;

    /// Allocate physical pages that are not yet attached to a slot.
    fn alloc_detached_pages(&mut self, pages: usize) -> anyhow::Result<Vec<u32>>;

    /// Return detached pages (from [`KvAllocator::alloc_detached_pages`]) to
    /// the free pool. Must only be called with pages that were never attached
    /// to a slot or retained by the prefix store.
    fn free_detached_pages(&mut self, pages: &[u32]);

    /// Free all pages currently attached to `slot`.
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
    /// KV model. It is NEVER called on the default decode path — only the opt-in
    /// `--kv-recall` executor calls it for non-pinned middle pages — so the
    /// default impl is a no-op and the baseline stays byte-identical.
    fn evict_slot_page(&mut self, _slot: usize, _logical_page: usize) -> Option<u32> {
        None
    }

    /// Inverse of [`KvAllocator::evict_slot_page`] — claim a free physical page
    /// and put it back at `logical_page`, replacing the evicted sentinel. The
    /// caller is responsible for refilling the page's KV from the tier; the
    /// allocator only hands back the id. `None` when that logical page is not
    /// evicted, is out of range, or the pool has no free page. Same opt-in
    /// footprint as `evict_slot_page` (`--kv-recall` prefetch only), so the
    /// default impl is a no-op.
    fn reinstate_slot_page(&mut self, _slot: usize, _logical_page: usize) -> Option<u32> {
        None
    }

    /// Truncate a slot to `new_len` logical tokens.
    fn truncate_slot(&mut self, slot: usize, new_len: usize) -> anyhow::Result<()>;

    /// Migrate a logical token range for `slot` into this pool.
    fn migrate(&mut self, slot: usize, start: usize, len: usize) -> anyhow::Result<()>;
}

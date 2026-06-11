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

    /// Truncate a slot to `new_len` logical tokens.
    fn truncate_slot(&mut self, slot: usize, new_len: usize) -> anyhow::Result<()>;

    /// Migrate a logical token range for `slot` into this pool.
    fn migrate(&mut self, slot: usize, start: usize, len: usize) -> anyhow::Result<()>;
}

//! Prefix-cache page retention.
//!
//! Retain/release ref-counting and attach of externally retained pages — the
//! surface a `PrefixManager` drives to share prompt prefixes across slots.

/// Prefix-cache page retention visible to engine-core.
///
/// Every method is expressed in host page ids and counts. The trait is
/// dyn-safe.
pub trait KvPrefixStore {
    /// Retain pages for an external owner such as a prefix cache.
    fn retain_pages(&mut self, pages: &[u32]);

    /// Release externally retained pages.
    fn release_pages(&mut self, pages: &[u32]);

    /// Return the number of pages retained by an external owner.
    fn retained_count(&self) -> usize;

    /// Attach retained physical pages to an empty slot.
    fn attach_pages(
        &mut self,
        slot: usize,
        pages: &[u32],
        token_count: usize,
    ) -> anyhow::Result<()>;
}

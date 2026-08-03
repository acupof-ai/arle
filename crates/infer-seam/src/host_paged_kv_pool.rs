//! Backend-neutral host-side paged KV bookkeeping.
//!
//! [`HostPagedKvPool`] owns only logical slot/page/token accounting. Device KV
//! buffers and backend-specific physical layouts stay below [`BackendExecutor`]
//! implementations; this pool is the production host allocator shared by
//! backends that use the standard paged-KV seam.

use std::collections::HashMap;

use anyhow::bail;

use crate::{KvAllocator, KvPrefixStore, KvQuery};

/// Logical-page slot marker for a page that has been **evict-dropped** out of
/// HBM under the write-through tiered KV model (`KvAllocator::evict_slot_page`).
///
/// A recall slot keeps its `slot_pages` vector at full *logical* length (one
/// entry per `ceil(seq_len / page_size)` page) so that token positions still map
/// to the right logical page index; an evicted middle page leaves this sentinel
/// in its logical slot while its physical page id is returned to the free pool.
/// The sentinel never names a real page (`total_pages` is far below `u32::MAX`),
/// and it only ever appears on the opt-in recall path — the default decode path
/// has byte-identical contiguous `slot_pages` with no sentinels.
pub const EVICTED_PAGE: u32 = u32::MAX;

/// Host-side paged KV bookkeeping for a backend executor.
///
/// Pages are logical `u32` ids. The executor decides how those ids map to
/// device buffers.
#[derive(Debug)]
pub struct HostPagedKvPool {
    page_size: usize,
    total_pages: usize,
    /// Free page ids, used as a LIFO stack.
    free: Vec<u32>,
    /// Per-slot page ids in logical order.
    slot_pages: Vec<Vec<u32>>,
    /// Per-slot logical token length.
    slot_len: Vec<usize>,
    /// Per-slot occupant epoch (bumped on free/attach).
    slot_epoch: Vec<u64>,
    /// Ref counts for pages retained by an external owner such as prefix cache.
    page_refs: HashMap<u32, u32>,
    /// Per-page live-slot attachment counts (a shared prefix page can be
    /// attached to several slots at once). A page returns to `free` only when
    /// BOTH its retain count and attach count are zero: recycling a page a
    /// live slot still writes aliases two slots onto one physical page, then
    /// double-frees on slot free and drifts `page_refs` until eviction frees
    /// nothing while the evictable count stays positive (#164 residual).
    slot_attach: HashMap<u32, u32>,
    /// Live count of pages matching `page_is_evictable`, maintained at every
    /// retain/release/attach/detach transition — the scheduler reads it on
    /// every decode tick, where a full `page_refs` scan is O(cached pages).
    evictable: usize,
    fixed_pages_per_slot: Option<usize>,
}

impl HostPagedKvPool {
    /// Build a pool with `num_slots` logical slots and `total_pages` physical pages.
    #[must_use]
    pub fn new(num_slots: usize, total_pages: usize, page_size: usize) -> Self {
        let page_size = page_size.max(1);
        let free: Vec<u32> = (0..total_pages as u32).rev().collect();
        Self {
            page_size,
            total_pages,
            free,
            slot_pages: vec![Vec::new(); num_slots],
            slot_len: vec![0; num_slots],
            slot_epoch: vec![0; num_slots],
            page_refs: HashMap::new(),
            slot_attach: HashMap::new(),
            evictable: 0,
            fixed_pages_per_slot: None,
        }
    }

    /// Configure fixed-band allocation for slots whose backend needs a full page
    /// table independent of the current logical token cursor (DSv4 FlashMLA).
    pub fn set_fixed_pages_per_slot(&mut self, pages: usize) {
        self.fixed_pages_per_slot = (pages > 0).then_some(pages);
    }

    fn pages_for_tokens(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page_size)
    }

    fn attach_count(&self, page: u32) -> u32 {
        self.slot_attach.get(&page).copied().unwrap_or(0)
    }

    fn attach(&mut self, page: u32) {
        let was = self.page_is_evictable(page);
        *self.slot_attach.entry(page).or_insert(0) += 1;
        self.sync_evictable(page, was);
    }

    /// Re-fold `page` into the `evictable` counter after its retain or attach
    /// count changed. `was` is the predicate value read before the mutation.
    fn sync_evictable(&mut self, page: u32, was: bool) {
        match (was, self.page_is_evictable(page)) {
            (false, true) => self.evictable += 1,
            (true, false) => self.evictable = self.evictable.saturating_sub(1),
            _ => {}
        }
    }

    /// Drop one slot attachment for `page`, recycling it once neither a slot
    /// nor the prefix cache holds it. Skips the evict-drop sentinel (its
    /// physical page was already returned to `free` at evict time — pushing
    /// it back would corrupt the free stack).
    fn detach(&mut self, page: u32) {
        if page == EVICTED_PAGE {
            return;
        }
        let was = self.page_is_evictable(page);
        if let Some(count) = self.slot_attach.get_mut(&page) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.slot_attach.remove(&page);
                self.maybe_free(page);
            }
        }
        self.sync_evictable(page, was);
    }

    /// Push `page` to the free stack iff no retain AND no attachment holds it.
    /// Called exactly at ref-decrement transitions, so a page frees once.
    fn maybe_free(&mut self, page: u32) {
        if self.page_refs.get(&page).copied().unwrap_or(0) == 0 && self.attach_count(page) == 0 {
            self.free.push(page);
        }
    }

    fn alloc_fixed_band(&mut self, slot: usize, pages: usize, tokens: usize) -> anyhow::Result<()> {
        if slot >= self.slot_pages.len() {
            bail!("HostPagedKvPool fixed alloc: slot {slot} out of range");
        }
        if self.slot_pages[slot].is_empty() {
            if pages > self.free.len() {
                bail!(
                    "HostPagedKvPool out of fixed-band pages: slot {slot} needs {pages}, free {}",
                    self.free.len()
                );
            }
            for _ in 0..pages {
                let page = self.free.pop().expect("checked free >= fixed pages");
                self.attach(page);
                self.slot_pages[slot].push(page);
            }
        }
        self.slot_len[slot] = self.slot_len[slot].saturating_add(tokens);
        Ok(())
    }
}

impl KvQuery for HostPagedKvPool {
    fn is_active(&self) -> bool {
        self.total_pages > 0
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn free_pages(&self) -> usize {
        self.free.len()
    }

    fn free_tokens(&self) -> usize {
        self.free.len() * self.page_size
    }

    fn resident_pages(&self) -> usize {
        self.total_pages.saturating_sub(self.free.len())
    }

    fn resident_evictable_pages(&self) -> usize {
        self.evictable
    }

    fn page_is_evictable(&self, page: u32) -> bool {
        self.page_refs.get(&page).copied().unwrap_or(0) == 1 && self.attach_count(page) == 0
    }

    fn seq_len(&self, slot: usize) -> usize {
        self.slot_len.get(slot).copied().unwrap_or(0)
    }

    fn slot_epoch(&self, slot: usize) -> u64 {
        self.slot_epoch.get(slot).copied().unwrap_or(0)
    }

    fn append_pages_needed(&self, slot: usize, tokens: usize) -> usize {
        if let Some(pages) = self.fixed_pages_per_slot {
            return if self.slot_pages.get(slot).is_some_and(Vec::is_empty) {
                pages
            } else {
                0
            };
        }
        let have = self.slot_pages.get(slot).map_or(0, Vec::len);
        let after = self.pages_for_tokens(self.seq_len(slot) + tokens);
        after.saturating_sub(have)
    }

    fn fixed_pages_per_slot(&self) -> Option<usize> {
        self.fixed_pages_per_slot
    }

    fn page_indices(&self, slot: usize) -> &[u32] {
        self.slot_pages.get(slot).map_or(&[], Vec::as_slice)
    }

    fn page_indices_for_token_range(&self, slot: usize, start: usize, len: usize) -> &[u32] {
        let Some(pages) = self.slot_pages.get(slot) else {
            return &[];
        };
        let start_page = start / self.page_size;
        let end_page = (start + len).div_ceil(self.page_size).min(pages.len());
        if start_page >= end_page {
            return &[];
        }
        &pages[start_page..end_page]
    }
}

impl KvAllocator for HostPagedKvPool {
    fn alloc(&mut self, slot: usize, tokens: usize) -> anyhow::Result<()> {
        if let Some(pages) = self.fixed_pages_per_slot {
            return self.alloc_fixed_band(slot, pages, tokens);
        }
        if slot >= self.slot_pages.len() {
            bail!("HostPagedKvPool alloc: slot {slot} out of range");
        }
        let need = self.append_pages_needed(slot, tokens);
        if need > self.free.len() {
            bail!(
                "HostPagedKvPool out of pages: slot {slot} needs {need}, free {}",
                self.free.len()
            );
        }
        for _ in 0..need {
            let page = self.free.pop().expect("checked free >= need");
            self.attach(page);
            self.slot_pages[slot].push(page);
        }
        self.slot_len[slot] += tokens;
        Ok(())
    }

    fn alloc_detached_pages(&mut self, pages: usize) -> anyhow::Result<Vec<u32>> {
        if pages > self.free.len() {
            bail!(
                "HostPagedKvPool out of pages: detached request {pages}, free {}",
                self.free.len()
            );
        }
        Ok((0..pages)
            .map(|_| self.free.pop().expect("checked free >= pages"))
            .collect())
    }

    fn free_detached_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            debug_assert!(
                self.page_refs.get(&page).copied().unwrap_or(0) == 0,
                "free_detached_pages: page {page} is retained"
            );
            debug_assert!(
                self.attach_count(page) == 0,
                "free_detached_pages: page {page} is slot-attached"
            );
            self.free.push(page);
        }
    }

    fn free_slot(&mut self, slot: usize) {
        let Some(pages) = self.slot_pages.get_mut(slot) else {
            return;
        };
        let taken = std::mem::take(pages);
        for page in taken {
            // `detach` skips the evict sentinel (already freed at evict time).
            self.detach(page);
        }
        self.slot_len[slot] = 0;
        self.slot_epoch[slot] = self.slot_epoch[slot].wrapping_add(1);
    }

    fn evict_slot_page(&mut self, slot: usize, logical_page: usize) -> Option<u32> {
        let pages = self.slot_pages.get_mut(slot)?;
        let page = *pages.get(logical_page)?;
        // Already evicted (sentinel) or retained by the prefix store (a pinned
        // sink page lives in the radix) → not a free-able middle page.
        if page == EVICTED_PAGE || self.page_refs.get(&page).copied().unwrap_or(0) != 0 {
            return None;
        }
        // Keep the logical length intact so token→logical-page mapping stays
        // valid for the surviving pages; the physical page goes back to the pool.
        pages[logical_page] = EVICTED_PAGE;
        self.detach(page);
        Some(page)
    }

    fn reinstate_slot_page(&mut self, slot: usize, logical_page: usize) -> Option<u32> {
        if *self.slot_pages.get(slot)?.get(logical_page)? != EVICTED_PAGE {
            return None; // already resident
        }
        let page = self.free.pop()?;
        self.attach(page);
        self.slot_pages[slot][logical_page] = page;
        Some(page)
    }

    fn truncate_slot(&mut self, slot: usize, new_len: usize) -> anyhow::Result<()> {
        if self.fixed_pages_per_slot.is_some() {
            if slot >= self.slot_pages.len() {
                bail!("truncate_slot: slot {slot} out of range");
            }
            self.slot_len[slot] = new_len;
            return Ok(());
        }
        let keep_pages = self.pages_for_tokens(new_len);
        let pages = self
            .slot_pages
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("truncate_slot: slot {slot} out of range"))?;
        let cut = keep_pages.min(pages.len());
        let removed: Vec<u32> = pages.split_off(cut);
        for page in removed {
            self.detach(page);
        }
        self.slot_len[slot] = new_len;
        Ok(())
    }

    fn migrate(&mut self, _slot: usize, _start: usize, _len: usize) -> anyhow::Result<()> {
        Ok(())
    }
}

impl KvPrefixStore for HostPagedKvPool {
    fn retain_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            let was = self.page_is_evictable(page);
            *self.page_refs.entry(page).or_insert(0) += 1;
            self.sync_evictable(page, was);
        }
    }

    fn release_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            let was = self.page_is_evictable(page);
            if let Some(c) = self.page_refs.get_mut(&page) {
                *c = c.saturating_sub(1);
                if *c == 0 {
                    self.page_refs.remove(&page);
                    self.maybe_free(page);
                }
            }
            self.sync_evictable(page, was);
        }
    }

    fn retained_count(&self) -> usize {
        self.page_refs.values().filter(|&&c| c > 0).count()
    }

    fn attach_pages(
        &mut self,
        slot: usize,
        pages: &[u32],
        token_count: usize,
    ) -> anyhow::Result<()> {
        if slot >= self.slot_pages.len() {
            bail!("attach_pages: slot {slot} out of range");
        }
        if let Some(fixed) = self.fixed_pages_per_slot {
            let top_up = fixed.saturating_sub(pages.len());
            if top_up > self.free.len() {
                bail!(
                    "attach_pages: fixed-band slot {slot} needs {top_up} more pages, free {}",
                    self.free.len()
                );
            }
            let stale = std::mem::take(&mut self.slot_pages[slot]);
            for page in stale {
                self.detach(page);
            }
            self.slot_pages[slot].extend_from_slice(pages);
            for _ in 0..top_up {
                let page = self.free.pop().expect("checked free >= top_up");
                self.attach(page);
                self.slot_pages[slot].push(page);
            }
        } else {
            self.slot_pages[slot].extend_from_slice(pages);
        }
        for &page in pages {
            self.attach(page);
        }
        self.slot_len[slot] = self.slot_len[slot].max(token_count);
        self.slot_epoch[slot] = self.slot_epoch[slot].wrapping_add(1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_grows_and_free_returns_pages() {
        let mut pool = HostPagedKvPool::new(2, 8, 16);
        assert_eq!(pool.free_pages(), 8);
        pool.alloc(0, 16).unwrap();
        assert_eq!(pool.seq_len(0), 16);
        assert_eq!(pool.page_indices(0).len(), 1);
        assert_eq!(pool.free_pages(), 7);
        pool.alloc(0, 1).unwrap();
        assert_eq!(pool.page_indices(0).len(), 2);
        pool.free_slot(0);
        assert_eq!(pool.seq_len(0), 0);
        assert_eq!(pool.free_pages(), 8);
    }

    #[test]
    fn out_of_pages_does_not_mutate_len() {
        let mut pool = HostPagedKvPool::new(1, 1, 16);
        assert!(pool.alloc(0, 17).is_err());
        assert_eq!(pool.seq_len(0), 0);
    }

    #[test]
    fn retained_pages_survive_free_slot_then_release() {
        let mut pool = HostPagedKvPool::new(2, 8, 16);
        pool.alloc(0, 32).unwrap();
        let prefix: Vec<u32> = pool.page_indices(0).to_vec();
        pool.retain_pages(&prefix);
        assert_eq!(pool.retained_count(), 2);
        assert_eq!(pool.resident_pages(), 2);
        assert_eq!(
            pool.resident_evictable_pages(),
            0,
            "live-attached pages are not evictable capacity"
        );
        let free_before = pool.free_pages();
        pool.free_slot(0);
        assert_eq!(pool.free_pages(), free_before);
        assert_eq!(
            pool.resident_evictable_pages(),
            2,
            "evictable once detached"
        );
        pool.attach_pages(1, &prefix, 32).unwrap();
        pool.retain_pages(&prefix);
        assert_eq!(pool.resident_evictable_pages(), 0);
        assert_eq!(pool.page_indices(1), prefix.as_slice());
        assert_eq!(pool.seq_len(1), 32);
        pool.free_slot(1);
        pool.release_pages(&prefix);
        assert_eq!(pool.resident_evictable_pages(), 2);
        pool.release_pages(&prefix);
        assert_eq!(pool.retained_count(), 0);
        assert_eq!(pool.resident_pages(), 0);
        assert_eq!(pool.free_pages(), 8);
    }

    /// #164 residual: the prefix cache releasing a page a live slot still
    /// writes must NOT recycle it — recycling aliases two slots onto one
    /// physical page and double-frees on slot free. The free is deferred to
    /// the slot's own detach.
    #[test]
    fn release_of_attached_page_defers_free_until_slot_detach() {
        let mut pool = HostPagedKvPool::new(1, 4, 16);
        pool.alloc(0, 16).unwrap();
        let pages = pool.page_indices(0).to_vec();
        pool.retain_pages(&pages); // mid-flight prompt-seal publish
        assert!(!pool.page_is_evictable(pages[0]));
        pool.release_pages(&pages); // radix evicts while the slot still lives
        assert_eq!(pool.free_pages(), 3, "attached page must not recycle");
        pool.free_slot(0);
        assert_eq!(pool.free_pages(), 4, "freed exactly once, at slot free");
    }

    #[test]
    fn truncate_frees_tail_pages() {
        let mut pool = HostPagedKvPool::new(1, 8, 16);
        pool.alloc(0, 48).unwrap();
        assert_eq!(pool.page_indices(0).len(), 3);
        pool.truncate_slot(0, 16).unwrap();
        assert_eq!(pool.page_indices(0).len(), 1);
        assert_eq!(pool.seq_len(0), 16);
        assert_eq!(pool.free_pages(), 7);
    }

    #[test]
    fn fixed_band_truncate_keeps_slot_table() {
        let mut pool = HostPagedKvPool::new(1, 8, 16);
        pool.set_fixed_pages_per_slot(4);
        pool.alloc(0, 48).unwrap();
        let pages = pool.page_indices(0).to_vec();
        assert_eq!(pages.len(), 4);
        let free_after_alloc = pool.free_pages();
        pool.truncate_slot(0, 16).unwrap();
        assert_eq!(pool.seq_len(0), 16);
        assert_eq!(pool.page_indices(0), pages.as_slice());
        assert_eq!(pool.free_pages(), free_after_alloc);
        pool.alloc(0, 1).unwrap();
        assert_eq!(pool.seq_len(0), 17);
        assert_eq!(pool.page_indices(0), pages.as_slice());
    }

    #[test]
    fn fixed_band_attach_tops_up_short_page_list() {
        let mut pool = HostPagedKvPool::new(2, 8, 16);
        pool.set_fixed_pages_per_slot(4);
        pool.alloc(0, 64).unwrap();
        let reused: Vec<u32> = pool.page_indices(0)[..2].to_vec();
        pool.attach_pages(1, &reused, 32).unwrap();
        assert_eq!(pool.page_indices(1).len(), 4);
        assert_eq!(&pool.page_indices(1)[..2], reused.as_slice());
        assert_eq!(pool.seq_len(1), 32);
    }

    #[test]
    fn fixed_band_attach_rejects_when_free_cannot_top_up() {
        let mut pool = HostPagedKvPool::new(2, 4, 16);
        pool.set_fixed_pages_per_slot(4);
        pool.alloc(0, 64).unwrap();
        assert!(pool.attach_pages(1, &[], 0).is_err());
    }

    #[test]
    fn evict_slot_page_returns_real_page_and_keeps_logical_length() {
        // 4 pages allocated (page_size 16, 64 tokens). Evict the logical middle
        // page (index 1): the physical page returns to the free pool, the logical
        // slot keeps length 4 (sentinel in the freed slot), seq_len is unchanged.
        let mut pool = HostPagedKvPool::new(1, 8, 16);
        pool.alloc(0, 64).unwrap();
        let pages: Vec<u32> = pool.page_indices(0).to_vec();
        assert_eq!(pages.len(), 4);
        let free_before = pool.free_pages();
        let freed = pool.evict_slot_page(0, 1).expect("middle page evicts");
        assert_eq!(
            freed, pages[1],
            "returned the physical page that was at logical 1"
        );
        // The physical page actually came back to the pool (the VRAM win).
        assert_eq!(pool.free_pages(), free_before + 1, "page freed to the pool");
        // Logical length intact so surviving pages stay logically addressable.
        assert_eq!(pool.page_indices(0).len(), 4);
        assert_eq!(pool.page_indices(0)[1], EVICTED_PAGE);
        assert_eq!(pool.page_indices(0)[0], pages[0]);
        assert_eq!(pool.page_indices(0)[2], pages[2]);
        assert_eq!(pool.seq_len(0), 64);
        // Re-evicting the same logical page is a no-op (already a sentinel).
        assert_eq!(pool.evict_slot_page(0, 1), None);
        assert_eq!(pool.free_pages(), free_before + 1);
        // free_slot must not double-free the sentinel slot.
        pool.free_slot(0);
        assert_eq!(
            pool.free_pages(),
            8,
            "all live pages back, sentinel not double-freed"
        );
    }

    #[test]
    fn evicting_middle_pages_keeps_resident_set_bounded_as_session_grows() {
        // The flat-VRAM invariant at the allocator level: as the session grows,
        // evicting the coldest middle page each time a new page rolls keeps the
        // number of physically-resident pages bounded, even though the logical
        // length (and the logical slot-page count) climbs without limit.
        let total_pages = 64;
        let mut pool = HostPagedKvPool::new(1, total_pages, 16);
        let budget_pages = 8; // sink(1) + local-ish window; just a bound to prove.
        let mut resident_high_water = 0usize;
        // Grow to 50 pages' worth of tokens, evicting to stay within budget.
        for page_no in 0..50 {
            pool.alloc(0, 16).unwrap(); // one fresh page each iteration
            let logical = pool.page_indices(0).len();
            assert_eq!(logical, page_no + 1, "logical length climbs every step");
            // Count physically-resident (non-sentinel) pages.
            let resident = pool
                .page_indices(0)
                .iter()
                .filter(|&&p| p != EVICTED_PAGE)
                .count();
            resident_high_water = resident_high_water.max(resident);
            // Over budget → evict the oldest still-resident non-pinned middle page
            // (logical index 1; logical 0 is the "sink" we keep), proving the
            // page is RELEASED, not merely un-attended.
            if resident > budget_pages {
                let evict_at = (1..logical)
                    .find(|&i| pool.page_indices(0)[i] != EVICTED_PAGE)
                    .expect("a resident middle page to evict");
                let freed = pool.evict_slot_page(0, evict_at);
                assert!(freed.is_some(), "a real page was freed");
            }
        }
        // Resident set stayed bounded near the budget despite a 50-page session.
        assert!(
            resident_high_water <= budget_pages + 1,
            "resident pages {resident_high_water} stayed bounded near budget {budget_pages}"
        );
        // And the pool still has plenty of free pages — HBM did not grow with history.
        let resident_now = pool
            .page_indices(0)
            .iter()
            .filter(|&&p| p != EVICTED_PAGE)
            .count();
        assert!(resident_now <= budget_pages + 1);
        assert!(
            pool.free_pages() >= total_pages - budget_pages - 1,
            "freed pages returned to the pool (flat VRAM)"
        );
    }
}

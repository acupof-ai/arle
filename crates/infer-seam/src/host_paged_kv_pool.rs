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
/// A recall slot keeps its `slot_pages` vector at full *logical* length so that
/// token positions still map to the right logical page index; an evicted middle
/// page leaves this sentinel in its logical slot while its physical page id is
/// returned to the free pool. The sentinel never names a real page
/// (`total_pages` is far below `u32::MAX`), and it only ever appears on the
/// opt-in recall path.
pub const EVICTED_PAGE: u32 = u32::MAX;

/// Host-side paged KV bookkeeping for a backend executor.
///
/// Pages are logical `u32` ids; the executor decides how those ids map to
/// device buffers.
#[derive(Debug)]
pub struct HostPagedKvPool {
    page_size: usize,
    total_pages: usize,
    free: Vec<u32>,
    slot_pages: Vec<Vec<u32>>,
    slot_len: Vec<usize>,
    slot_epoch: Vec<u64>,
    /// Ref counts for pages retained by an external owner such as prefix cache.
    page_refs: HashMap<u32, u32>,
    /// Per-page live-slot attachment counts. A page returns to `free` only when
    /// BOTH its retain count and attach count are zero: recycling a page a live
    /// slot still writes aliases two slots onto one physical page, then
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

    /// Fixed-band allocation for slots whose backend needs a full page table
    /// independent of the current logical token cursor (DSv4 FlashMLA).
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

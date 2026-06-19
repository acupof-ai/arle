//! Backend-neutral host-side paged KV bookkeeping.
//!
//! [`HostPagedKvPool`] owns only logical slot/page/token accounting. Device KV
//! buffers and backend-specific physical layouts stay below [`BackendExecutor`]
//! implementations; this pool is the production host allocator shared by
//! backends that use the standard paged-KV seam.

use std::collections::HashMap;

use anyhow::bail;

use crate::{KvAllocator, KvPrefixStore, KvQuery};

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
        }
    }

    fn pages_for_tokens(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page_size)
    }

    fn reclaim_page(&mut self, page: u32) {
        if self.page_refs.get(&page).copied().unwrap_or(0) == 0 {
            self.free.push(page);
        }
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
        self.page_refs.values().filter(|&&count| count == 1).count()
    }

    fn seq_len(&self, slot: usize) -> usize {
        self.slot_len.get(slot).copied().unwrap_or(0)
    }

    fn slot_epoch(&self, slot: usize) -> u64 {
        self.slot_epoch.get(slot).copied().unwrap_or(0)
    }

    fn append_pages_needed(&self, slot: usize, tokens: usize) -> usize {
        let have = self.slot_pages.get(slot).map_or(0, Vec::len);
        let after = self.pages_for_tokens(self.seq_len(slot) + tokens);
        after.saturating_sub(have)
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
            self.free.push(page);
        }
    }

    fn free_slot(&mut self, slot: usize) {
        let Some(pages) = self.slot_pages.get_mut(slot) else {
            return;
        };
        let taken = std::mem::take(pages);
        for page in taken {
            self.reclaim_page(page);
        }
        self.slot_len[slot] = 0;
        self.slot_epoch[slot] = self.slot_epoch[slot].wrapping_add(1);
    }

    fn truncate_slot(&mut self, slot: usize, new_len: usize) -> anyhow::Result<()> {
        let keep_pages = self.pages_for_tokens(new_len);
        let pages = self
            .slot_pages
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("truncate_slot: slot {slot} out of range"))?;
        let cut = keep_pages.min(pages.len());
        let removed: Vec<u32> = pages.split_off(cut);
        for page in removed {
            self.reclaim_page(page);
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
            *self.page_refs.entry(page).or_insert(0) += 1;
        }
    }

    fn release_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            if let Some(c) = self.page_refs.get_mut(&page) {
                *c = c.saturating_sub(1);
                if *c == 0 {
                    self.page_refs.remove(&page);
                    self.free.push(page);
                }
            }
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
        let dst = self
            .slot_pages
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("attach_pages: slot {slot} out of range"))?;
        dst.extend_from_slice(pages);
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
        assert_eq!(pool.resident_evictable_pages(), 2);
        let free_before = pool.free_pages();
        pool.free_slot(0);
        assert_eq!(pool.free_pages(), free_before);
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
}

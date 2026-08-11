//! Host KV bookkeeping for the Vulkan backend.
//!
//! Mirrors the HIP/Metal backends' page accounting shape: LIFO free
//! pages, retained pages survive `free_slot`, and device buffers are hidden
//! from the seam. P2 allocates host page ids only; real Vulkan arenas land
//! with the first numeric model path.

use std::collections::HashMap;

use infer_seam::{KvAllocator, KvPrefixStore, KvQuery};

pub struct VulkanKvPool {
    page_size: usize,
    total_pages: usize,
    max_seq_len: usize,
    free: Vec<u32>,
    slot_pages: Vec<Vec<u32>>,
    slot_len: Vec<usize>,
    slot_epoch: Vec<u64>,
    page_refs: HashMap<u32, u32>,
}

impl VulkanKvPool {
    pub fn new(num_slots: usize, total_pages: usize, page_size: usize, max_seq_len: usize) -> Self {
        let page_size = page_size.max(1);
        let free: Vec<u32> = (0..total_pages as u32).rev().collect();
        Self {
            page_size,
            total_pages,
            max_seq_len,
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

impl KvQuery for VulkanKvPool {
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

impl KvAllocator for VulkanKvPool {
    fn alloc(&mut self, slot: usize, tokens: usize) -> anyhow::Result<()> {
        if self.seq_len(slot) + tokens > self.max_seq_len {
            anyhow::bail!(
                "VulkanKvPool slot {slot} exceeds max_seq_len {} (have {}, append {tokens})",
                self.max_seq_len,
                self.seq_len(slot)
            );
        }
        let pages = self
            .slot_pages
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("alloc: slot {slot} out of range"))?;
        let have = pages.len();
        let after = (self.slot_len[slot] + tokens).div_ceil(self.page_size);
        let need = after.saturating_sub(have);
        if need > self.free.len() {
            anyhow::bail!(
                "VulkanKvPool out of pages: slot {slot} needs {need}, free {}",
                self.free.len()
            );
        }
        for _ in 0..need {
            pages.push(self.free.pop().expect("checked free >= need"));
        }
        self.slot_len[slot] += tokens;
        Ok(())
    }

    fn alloc_detached_pages(&mut self, pages: usize) -> anyhow::Result<Vec<u32>> {
        if pages > self.free.len() {
            anyhow::bail!(
                "VulkanKvPool out of pages: detached request {pages}, free {}",
                self.free.len()
            );
        }
        Ok((0..pages)
            .map(|_| self.free.pop().expect("checked free >= pages"))
            .collect())
    }

    fn free_detached_pages(&mut self, pages: &[u32]) {
        self.free.extend_from_slice(pages);
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

impl KvPrefixStore for VulkanKvPool {
    fn retain_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            *self.page_refs.entry(page).or_insert(0) += 1;
        }
    }

    fn release_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            if let Some(count) = self.page_refs.get_mut(&page) {
                *count = count.saturating_sub(1);
                if *count == 0 {
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

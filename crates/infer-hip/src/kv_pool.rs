//! DSv4 per-slot KV shapes + host-bookkeeping [`KvPool`] for the HIP lane.
//!
//! Buffer enumeration mirrors `infer-cuda/src/attention.rs`
//! `Dsv4LayerAttentionState::new` (l.1518-1564) + `Dsv4CompressorState::new`
//! (l.83-112): per slot, EVERY layer holds an SW window ring
//! (`sliding_window * head_dim` bf16); compressed layers (ratio > 0) add
//! pending/prev-overlap/compressed; CSA layers (1..16) add the same family
//! at indexer width (overlap always on). Opt-in CUDA states (FlashMLA,
//! fused-wqkv, DSA-official, DeepGEMM scratch) are datacenter-only and
//! excluded.
//!
//! Host bookkeeping matches `infer-metal/src/kv_pool.rs` semantics exactly
//! (LIFO free pages, retained pages survive `free_slot`, no extra
//! prefix-store features). Device buffers gate on `hip` (pending-remote).

use std::collections::HashMap;

use deepseek_spec::v4::DeepSeekV4Config;
use infer_seam::{KvAllocator, KvPrefixStore, KvQuery};

/// Element counts (bf16) of one compressor/indexer ring family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dsv4CompressorShapeElems {
    pub pending_kv: usize,
    pub pending_score: usize,
    pub prev_overlap_kv: usize,
    pub prev_overlap_score: usize,
    pub compressed: usize,
}

impl Dsv4CompressorShapeElems {
    fn new(head_dim: usize, ratio: usize, overlap: bool, max_seq_len: usize) -> Self {
        let width = if overlap { 2 * head_dim } else { head_dim };
        let pending = ratio * width;
        let prev = ratio * head_dim;
        Self {
            pending_kv: pending,
            pending_score: pending,
            prev_overlap_kv: prev,
            prev_overlap_score: prev,
            compressed: head_dim * max_seq_len.div_ceil(ratio).max(1),
        }
    }

    pub fn total(&self) -> usize {
        self.pending_kv
            + self.pending_score
            + self.prev_overlap_kv
            + self.prev_overlap_score
            + self.compressed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dsv4LayerSlotShape {
    pub compress_ratio: usize,
    pub sw_window: usize,
    pub compressor: Option<Dsv4CompressorShapeElems>,
    pub indexer: Option<Dsv4CompressorShapeElems>,
}

impl Dsv4LayerSlotShape {
    pub fn total_elems(&self) -> usize {
        self.sw_window
            + self.compressor.map_or(0, |c| c.total())
            + self.indexer.map_or(0, |c| c.total())
    }
}

#[derive(Debug, Clone)]
pub struct Dsv4SlotShape {
    pub layers: Vec<Dsv4LayerSlotShape>,
    pub max_seq_len: usize,
}

impl Dsv4SlotShape {
    pub fn new(config: &DeepSeekV4Config, max_seq_len: usize) -> Self {
        let layers = config
            .compress_ratios
            .iter()
            .take(config.num_hidden_layers)
            .map(|&ratio| {
                let compressor = (ratio > 0).then(|| {
                    Dsv4CompressorShapeElems::new(config.head_dim, ratio, ratio < 16, max_seq_len)
                });
                let indexer = (ratio > 0 && ratio < 16).then(|| {
                    Dsv4CompressorShapeElems::new(config.index_head_dim, ratio, true, max_seq_len)
                });
                Dsv4LayerSlotShape {
                    compress_ratio: ratio,
                    sw_window: config.sliding_window * config.head_dim,
                    compressor,
                    indexer,
                }
            })
            .collect();
        Self {
            layers,
            max_seq_len,
        }
    }

    pub fn total_elems(&self) -> usize {
        self.layers
            .iter()
            .map(Dsv4LayerSlotShape::total_elems)
            .sum()
    }

    pub fn total_bytes(&self) -> usize {
        self.total_elems() * 2
    }
}

#[cfg(feature = "hip")]
struct SlotDeviceState {
    /// One bf16 arena per slot covering all per-layer buffers (laid out in
    /// `Dsv4SlotShape::layers` order). Allocation only — pending-remote.
    _arena: hip_sys::DeviceBuffer,
}

/// Host-side paged KV bookkeeping for the HIP backend.
pub struct HipKvPool {
    page_size: usize,
    total_pages: usize,
    free: Vec<u32>,
    slot_pages: Vec<Vec<u32>>,
    slot_len: Vec<usize>,
    slot_epoch: Vec<u64>,
    page_refs: HashMap<u32, u32>,
    shape: Dsv4SlotShape,
    #[cfg(feature = "hip")]
    slot_device: Vec<Option<SlotDeviceState>>,
}

impl HipKvPool {
    pub fn new(
        config: &DeepSeekV4Config,
        num_slots: usize,
        total_pages: usize,
        page_size: usize,
        max_seq_len: usize,
    ) -> Self {
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
            shape: Dsv4SlotShape::new(config, max_seq_len),
            #[cfg(feature = "hip")]
            slot_device: (0..num_slots).map(|_| None).collect(),
        }
    }

    pub fn slot_shape(&self) -> &Dsv4SlotShape {
        &self.shape
    }

    /// Allocate the per-slot device arena sized from the shape. Real run
    /// pending-remote (#77 on-box).
    #[cfg(feature = "hip")]
    pub fn ensure_slot_device(&mut self, slot: usize) -> anyhow::Result<()> {
        if slot >= self.slot_device.len() {
            anyhow::bail!("slot {slot} out of range");
        }
        if self.slot_device[slot].is_none() {
            let arena = hip_sys::DeviceBuffer::alloc(self.shape.total_bytes())
                .map_err(|e| anyhow::anyhow!("slot {slot} arena alloc failed: {e}"))?;
            self.slot_device[slot] = Some(SlotDeviceState { _arena: arena });
        }
        Ok(())
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

impl KvQuery for HipKvPool {
    fn is_active(&self) -> bool {
        self.total_pages > 0
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn free_pages(&self) -> usize {
        self.free.len()
    }

    fn resident_pages(&self) -> usize {
        self.total_pages.saturating_sub(self.free.len())
    }

    fn resident_evictable_pages(&self) -> usize {
        self.page_refs.values().filter(|&&count| count == 1).count()
    }

    fn page_is_evictable(&self, page: u32) -> bool {
        // No slot-attach tracking in the HIP lane yet: same predicate as the
        // count above so capacity and evictor stay one source.
        self.page_refs.get(&page).copied().unwrap_or(0) == 1
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

    fn page_indices_for_token_range(&self, slot: usize, len: usize) -> &[u32] {
        let Some(pages) = self.slot_pages.get(slot) else {
            return &[];
        };
        let end_page = len.div_ceil(self.page_size).min(pages.len());
        &pages[..end_page]
    }
}

impl KvAllocator for HipKvPool {
    fn alloc(&mut self, slot: usize, tokens: usize) -> anyhow::Result<()> {
        if self.seq_len(slot) + tokens > self.shape.max_seq_len {
            anyhow::bail!(
                "HipKvPool slot {slot} exceeds max_seq_len {} (have {}, append {tokens})",
                self.shape.max_seq_len,
                self.seq_len(slot)
            );
        }
        let need = self.append_pages_needed(slot, tokens);
        if need > self.free.len() {
            anyhow::bail!(
                "HipKvPool out of pages: slot {slot} needs {need}, free {}",
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
            anyhow::bail!(
                "HipKvPool out of pages: detached request {pages}, free {}",
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
}

impl KvPrefixStore for HipKvPool {
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

//! Whole-slot spill tier (host DRAM plus opt-in NVMe) and its counters: a
//! demoted slot's complete device image out and back.

use super::*;

impl Dsv4CudaExecutor {
    /// Attach the opt-in NVMe disk spill level (pre-serve only). The cap is
    /// soft: over it, publish drops the oldest entries — capacity never blocks
    /// a forward and never enters the reuse license.
    pub(crate) fn set_kv_tier_disk(
        &mut self,
        root: std::path::PathBuf,
        budget_bytes: usize,
    ) -> bool {
        let prefix_ok = self.prefix_state.set_disk(root.clone(), budget_bytes);
        let slot_ok = self
            .slot_tier
            .set_disk(root, budget_bytes, BLOB_CHUNK_BYTES);
        prefix_ok && slot_ok
    }

    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        self.prefix_state.set_budget_bytes(bytes);
        self.slot_tier = KvTierStore::with_budget(bytes, BLOB_CHUNK_BYTES);
    }

    pub(crate) fn kv_tier_host_demoted_pages(&self) -> usize {
        self.prefix_state.host_pages() + self.slot_tier.host_demoted_pages()
    }

    pub(crate) fn kv_tier_disk_pages(&self) -> usize {
        self.prefix_state.disk_pages() + self.slot_tier.disk_pages()
    }

    pub(crate) fn kv_tier_read_hits(&self) -> infer_seam::KvTierReadHits {
        let mut hits = self.prefix_state.read_hits();
        let slot = self.slot_tier.read_hits();
        hits.host_demoted += slot.host_demoted;
        hits.disk += slot.disk;
        hits
    }

    pub(crate) fn kv_tier_io_stats(&self) -> infer_seam::KvTierIoStats {
        let prefix = self.prefix_state.io_stats();
        let slot = self.slot_tier.io_stats();
        let stats = kv_native_sys::TierIoStats {
            useful_read_bytes: prefix.useful_read_bytes + slot.useful_read_bytes,
            useful_write_bytes: prefix.useful_write_bytes + slot.useful_write_bytes,
            submitted_read_bytes: prefix.submitted_read_bytes + slot.submitted_read_bytes,
            submitted_write_bytes: prefix.submitted_write_bytes + slot.submitted_write_bytes,
            metadata_write_bytes: prefix.metadata_write_bytes + slot.metadata_write_bytes,
            failures: prefix.failures + slot.failures,
            completion_wait_ns: prefix.completion_wait_ns + slot.completion_wait_ns,
            ..prefix
        };
        infer_seam::KvTierIoStats {
            mode: match stats.mode {
                kv_native_sys::DiskIoMode::Disabled => infer_seam::KvTierIoMode::Disabled,
                kv_native_sys::DiskIoMode::Mmap => infer_seam::KvTierIoMode::Mmap,
                kv_native_sys::DiskIoMode::Direct => infer_seam::KvTierIoMode::Direct,
            },
            useful_read_bytes: stats.useful_read_bytes,
            useful_write_bytes: stats.useful_write_bytes,
            submitted_read_bytes: stats.submitted_read_bytes,
            submitted_write_bytes: stats.submitted_write_bytes,
            metadata_write_bytes: stats.metadata_write_bytes,
            failures: stats.failures,
            completion_wait_ns: stats.completion_wait_ns,
        }
    }

    pub(crate) fn kv_tier_location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        self.slot_tier
            .location(key)
            .or_else(|| self.prefix_state.location(key))
    }

    pub(crate) fn demote_slot(&mut self, slot: usize, key: u64) -> Result<bool> {
        ensure!(
            slot < self.num_slots,
            "DSv4 demote slot {slot} outside executor slots {}",
            self.num_slots
        );
        let image = {
            let Self {
                model,
                slots,
                kv_adapter,
                ..
            } = &mut *self;
            slots[slot].swap_out_image(&model.ctx, kv_adapter, slot)
        };
        let capture_ok = usize::from(image.is_ok());
        if self.tp_min_usize(capture_ok, "dsv4 slot demote capture")? == 0 {
            return Err(image
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("peer rank failed DSv4 slot demote capture")));
        }
        let bytes = image?.to_bytes();
        let inserted = self
            .slot_tier
            .insert_chunked(NS_SLOT, NS_SLOT_CHUNK, key, &bytes);
        if self.tp_min_usize(usize::from(inserted), "dsv4 slot demote insert")? == 0 {
            if inserted {
                self.slot_tier.remove_chunked(NS_SLOT, NS_SLOT_CHUNK, key);
            }
            // Capture was non-destructive, so the victim keeps decoding.
            return Ok(false);
        }
        let Self {
            slots, kv_adapter, ..
        } = &mut *self;
        slots[slot].release_swapped_out(kv_adapter)?;
        Ok(true)
    }

    pub(crate) fn promote_slot(
        &mut self,
        key: u64,
        slot: usize,
        _slot_pages: &[u32],
    ) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "DSv4 promote slot {slot} outside executor slots {}",
            self.num_slots
        );
        let image = self
            .slot_tier
            .read_chunked(NS_SLOT, NS_SLOT_CHUNK, key)
            .map_err(|e| anyhow::anyhow!("DSv4 whole-slot tier read key {key}: {e}"))
            .and_then(|bytes| crate::dsv4::Dsv4SlotImage::from_bytes(&bytes));
        let image_ok = usize::from(image.is_ok());
        if self.tp_min_usize(image_ok, "dsv4 slot promote read")? == 0 {
            return Err(image
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("peer rank failed DSv4 slot promote read")));
        }
        let image = image?;
        let restored = {
            let Self {
                model,
                slots,
                kv_adapter,
                ..
            } = &mut *self;
            slots[slot].swap_in_image(&model.ctx, kv_adapter, slot, &image)
        };
        let restore_ok = usize::from(restored.is_ok());
        if self.tp_min_usize(restore_ok, "dsv4 slot promote restore")? == 0 {
            return Err(restored
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("peer rank failed DSv4 slot promote restore")));
        }
        restored
    }

    pub(crate) fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
        for &key in keys {
            self.slot_tier.remove_chunked(NS_SLOT, NS_SLOT_CHUNK, key);
        }
    }
}

//! Metal KV page store: resident MLX mirrors + the shared disk tier.
//!
//! The disk level is the backend-neutral [`kv_native_sys::KvTierStore`] run
//! disk-only (host budget 0 — on unified memory demoting device→host frees
//! nothing; OS pressure already spills via SSD) in its durable, fixed-size
//! namespace. Everything on disk is addressed by the engine's prefix content
//! key, so a restarted server attaches a prompt's pages and restore snapshot
//! without re-prefilling. This module owns only the MLX payload codec and the
//! live-slot↔tier bridge; budgeting, eviction, and mmap I/O live in the store.

use kv_native_sys::{KvTierStore, tier_key};

use super::*;

/// Store key namespaces (top byte); the sub-key is the content key. Pages are
/// plain fixed-slot inserts, prefix snapshots are variable-size chunked blobs.
const NS_PAGE: u64 = 1;
const NS_PREFIX: u64 = 2;
const NS_PREFIX_CHUNK: u64 = 3;

/// Default `--kv-disk` size on Metal: a fixed, self-evicting 1 GiB file. The
/// tier is a restart cache, so a bounded footprint beats a disk-fraction one.
const DEFAULT_DISK_BUDGET_BYTES: usize = 1 << 30;

/// Default L3 budget for a Metal `--kv-disk` root, clamped to what the disk can
/// actually give (the signature matches the CUDA disk-fraction default).
#[must_use]
pub fn default_t2_budget_bytes(root: &std::path::Path, ssd_fraction: f64) -> usize {
    DEFAULT_DISK_BUDGET_BYTES.min(kv_native_sys::default_t2_budget_bytes(root, ssd_fraction))
}

/// Slot headroom over the raw KV page bytes for the codec's per-array headers
/// (dtype/ndim/dims/len ≈ 56 B × O(100) arrays); generous so a page payload
/// never trips the store's oversize refusal.
const PAGE_CODEC_HEADROOM: usize = 64 << 10;

pub struct MetalPageStore {
    pub(super) pages: HashMap<u32, MetalPageBlock>,
    pub(super) prefixes: HashMap<Vec<u64>, MetalPrefixSnapshot>,
    pub(super) next_logical_id: u64,
    /// Disk-only durable tier store (`--kv-disk`).
    tier: Option<KvTierStore>,
    /// Engine page size (content keys are chained per page); 0 until the tier
    /// is attached.
    page_size: usize,
}

impl Default for MetalPageStore {
    fn default() -> Self {
        Self {
            pages: HashMap::new(),
            prefixes: HashMap::new(),
            next_logical_id: 1,
            tier: None,
            page_size: 0,
        }
    }
}

#[derive(Clone)]
pub struct MetalPageBlock {
    pub(super) logical_id: u64,
    pub(super) owner: Option<MetalPageOwner>,
    /// Engine prefix content key, known once the page's tokens are published
    /// (sidecar save) or when it was promoted from the tier.
    pub(super) content_key: Option<u64>,
    pub(super) kv_flat: Vec<mlx::MlxArray>,
}

#[derive(Clone, Copy)]
pub struct MetalPageOwner {
    pub(super) slot: usize,
    pub(super) slot_epoch: u64,
    pub(super) page_idx: usize,
}

#[derive(Clone)]
pub struct MetalPrefixSnapshot {
    pub(super) cache_len: usize,
    pub(super) gdr_flat: Vec<mlx::MlxArray>,
}

/// Write-through one page under its content key (idempotent: content keys
/// name immutable content, so an existing record is kept as-is).
fn write_page(tier: &mut KvTierStore, content_key: u64, block: &MetalPageBlock) -> bool {
    let key = tier_key(NS_PAGE, content_key);
    if tier.contains(key) {
        return true;
    }
    match encode_page_payload(&block.kv_flat) {
        Ok(bytes) => tier.insert(key, bytes),
        Err(err) => {
            log::warn!("Metal KV T2 page encode failed for content key {content_key}: {err:#}");
            false
        }
    }
}

impl MetalPageStore {
    /// Attach the durable disk namespace for `epoch` (model + KV layout tag):
    /// a restart replays its manifest and serves prior prompts by content key.
    pub(super) fn set_ssd(
        &mut self,
        root: PathBuf,
        budget_bytes: usize,
        bytes_per_page: usize,
        page_size: usize,
        epoch: &str,
    ) -> bool {
        self.page_size = page_size.max(1);
        let slot_bytes = bytes_per_page.max(1).saturating_add(PAGE_CODEC_HEADROOM);
        // Host budget pinned to 0: unified memory makes a DRAM L2 a no-op.
        let mut store = KvTierStore::with_budget(0, slot_bytes);
        if !store.load(root, budget_bytes, slot_bytes, epoch.to_string(), 0, 1, 0) {
            return false;
        }
        eprintln!(
            "[infer-metal] KV T2 disk tier: budget_bytes={}, capacity_pages={}, restored_pages={}",
            budget_bytes,
            store.capacity_pages(),
            store.disk_page_count()
        );
        self.tier = Some(store);
        true
    }

    pub(super) fn kv_tier_capacity_pages(&self) -> usize {
        self.tier.as_ref().map_or(0, KvTierStore::capacity_pages)
    }

    pub(super) fn kv_tier_page_bytes(&self) -> usize {
        self.tier.as_ref().map_or(0, KvTierStore::page_bytes)
    }

    pub(super) fn kv_tier_disk_pages(&self) -> usize {
        self.tier.as_ref().map_or(0, KvTierStore::disk_page_count)
    }

    pub(super) fn kv_tier_read_hits(&self) -> infer_seam::KvTierReadHits {
        self.tier
            .as_ref()
            .map_or_else(infer_seam::KvTierReadHits::default, KvTierStore::read_hits)
    }

    pub(super) fn kv_tier_io_stats(&self) -> infer_seam::KvTierIoStats {
        let stats = self
            .tier
            .as_ref()
            .map_or_else(kv_native_sys::TierIoStats::default, KvTierStore::io_stats);
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

    pub(super) fn kv_tier_location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        self.tier.as_ref()?.location(tier_key(NS_PAGE, key))
    }

    #[cfg(test)]
    pub(super) fn has_disk_prefix(&self, content_key: u64) -> bool {
        self.tier
            .as_ref()
            .is_some_and(|tier| tier.contains_chunked(NS_PREFIX, NS_PREFIX_CHUNK, content_key))
    }

    /// Largest leading block count for which Metal has a complete restore
    /// image: a resident snapshot for the block's logical key, or a durable
    /// one under the block's content key. Demoted keys are checked before
    /// promotion, so an unusable prefix tail is never read back from T2.
    pub(super) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        (1..=blocks.len())
            .rev()
            .find(|&k| {
                if self
                    .logical_key_for_prefix_blocks(&blocks[..k])
                    .is_some_and(|key| self.prefixes.contains_key(&key))
                {
                    return true;
                }
                let content_key = match blocks[k - 1] {
                    PrefixBlock::ResidentPage(page) => {
                        self.pages.get(&page).and_then(|b| b.content_key)
                    }
                    PrefixBlock::DemotedKey(key) => Some(key),
                };
                match (content_key, self.tier.as_ref()) {
                    (Some(key), Some(tier)) => {
                        tier.contains_chunked(NS_PREFIX, NS_PREFIX_CHUNK, key)
                    }
                    _ => false,
                }
            })
            .unwrap_or(0)
    }

    pub(super) fn release_pages(&mut self, pages: &[u32]) {
        if pages.is_empty() {
            return;
        }
        let released_logical_ids: Vec<u64> = pages
            .iter()
            .filter_map(|page| self.pages.remove(page).map(|b| b.logical_id))
            .collect();
        if released_logical_ids.is_empty() {
            return;
        }
        self.prefixes.retain(|key, _| {
            !released_logical_ids
                .iter()
                .any(|logical_id| key.contains(logical_id))
        });
    }

    pub(super) fn next_logical_id(&mut self) -> u64 {
        let id = self.next_logical_id.max(1);
        self.next_logical_id = id.saturating_add(1);
        id
    }

    pub(super) fn logical_key_for_pages(&self, pages: &[u32]) -> Option<Vec<u64>> {
        pages
            .iter()
            .map(|page| self.pages.get(page).map(|b| b.logical_id))
            .collect()
    }

    fn logical_key_for_prefix_blocks(&self, blocks: &[PrefixBlock]) -> Option<Vec<u64>> {
        blocks
            .iter()
            .map(|block| match *block {
                PrefixBlock::ResidentPage(page) => Some(self.pages.get(&page)?.logical_id),
                PrefixBlock::DemotedKey(_) => None,
            })
            .collect()
    }

    /// Make the restore snapshot for the resident page run `key` available in
    /// memory, reading it from the durable tier under `content_key` if needed.
    fn ensure_prefix_snapshot_resident(
        &mut self,
        key: &[u64],
        content_key: Option<u64>,
    ) -> anyhow::Result<()> {
        if self.prefixes.contains_key(key) {
            return Ok(());
        }
        let content_key = content_key
            .ok_or_else(|| anyhow::anyhow!("Metal prefix snapshot {key:?} is not resident"))?;
        let tier = self
            .tier
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Metal KV T2 store is not configured"))?;
        let bytes = tier.read_chunked(NS_PREFIX, NS_PREFIX_CHUNK, content_key)?;
        let snapshot = decode_prefix_payload(&bytes, content_key)?;
        self.prefixes.insert(key.to_vec(), snapshot);
        Ok(())
    }

    /// Bind published pages to their engine content keys and write them (plus
    /// the prefix restore snapshot ending at the last page) through to the
    /// durable tier.
    pub(super) fn save_prefix_sidecar(&mut self, tokens: &[u32], prefix_pages: &[u32]) {
        if self.page_size == 0 {
            return;
        }
        // A prefix larger than the store would evict its own head while
        // writing its tail, leaving nothing reusable and paying the writes for
        // it. Chunked snapshot rides along, hence the margin.
        let capacity = self.kv_tier_capacity_pages();
        if capacity > 0 && prefix_pages.len().saturating_add(16) > capacity {
            log::debug!(
                "Metal KV T2 sidecar skipped: {} pages exceed the {capacity}-page store",
                prefix_pages.len()
            );
            return;
        }
        let keys = infer_seam::prefix_content_keys(tokens, self.page_size);
        let mut written = 0usize;
        for (page, key) in prefix_pages.iter().zip(keys.iter().copied()) {
            let Some(block) = self.pages.get_mut(page) else {
                break;
            };
            block.content_key = Some(key);
            if let Some(tier) = self.tier.as_mut()
                && !write_page(tier, key, block)
            {
                break;
            }
            written += 1;
        }
        // Persist every restore image we hold, not just the one at the end:
        // `publish_slot` leaves one per prefill chunk, and a later prompt that
        // shares only a leading run can restore at any of those boundaries.
        let boundaries: Vec<(u64, MetalPrefixSnapshot)> = (1..=written)
            .filter_map(|n| {
                let logical_key = self.logical_key_for_pages(&prefix_pages[..n])?;
                Some((keys[n - 1], self.prefixes.get(&logical_key)?.clone()))
            })
            .collect();
        log::debug!(
            "Metal KV T2 sidecar: tokens={} pages={} written={written} snapshots={}",
            tokens.len(),
            prefix_pages.len(),
            boundaries.len()
        );
        for (content_key, snapshot) in boundaries {
            self.write_disk_prefix(content_key, &snapshot);
        }
        // Publish is rare (once per prompt) and a kill skips Drop, so make the
        // just-written pages recoverable now rather than at shutdown.
        if let Some(tier) = self.tier.as_mut() {
            tier.sync_manifest();
        }
    }

    pub(super) fn demote_prefix_pages(&mut self, entries: &[(u32, u64)]) -> anyhow::Result<usize> {
        let Some(tier) = self.tier.as_mut() else {
            return Ok(0);
        };
        let mut accepted = 0usize;
        for &(page, key) in entries {
            let Some(block) = self.pages.get_mut(&page) else {
                break;
            };
            if !write_page(tier, key, block) {
                break;
            }
            block.content_key = Some(key);
            accepted = accepted.saturating_add(1);
        }
        Ok(accepted)
    }

    pub(super) fn promote_prefix_pages(&mut self, entries: &[(u64, u32)]) -> anyhow::Result<()> {
        for &(key, dst_page) in entries {
            let block = self.read_tier_page(key)?;
            if let Some(old) = self.pages.insert(dst_page, block) {
                let old_id = old.logical_id;
                self.prefixes.retain(|key, _| !key.contains(&old_id));
            }
        }
        Ok(())
    }

    fn read_tier_page(&mut self, key: u64) -> anyhow::Result<MetalPageBlock> {
        let tier = self
            .tier
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Metal KV T2 store is not configured"))?;
        let bytes = tier.read(tier_key(NS_PAGE, key))?;
        let kv_flat = decode_page_payload(&bytes)?;
        Ok(MetalPageBlock {
            logical_id: self.next_logical_id(),
            owner: None,
            content_key: Some(key),
            kv_flat,
        })
    }

    pub(super) fn publish_slot(
        &mut self,
        slot: &MetalSlotState,
        kv: &dyn KvPool,
    ) -> anyhow::Result<()> {
        let page_size = kv.page_size().max(1);
        let full_pages = slot.cache_len / page_size;
        if full_pages == 0 {
            return Ok(());
        }

        let page_ids = kv.page_indices(slot.slot);
        let publish_pages = full_pages.min(page_ids.len());
        let mut overwritten_logical_ids = Vec::new();
        for (page_idx, page_id) in page_ids.iter().take(publish_pages).enumerate() {
            let start = page_idx * page_size;
            let end = start + page_size;
            let kv_flat = slot
                .kv_flat
                .iter()
                .map(|array| slice_kv_tokens(array, start, end))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let owner = MetalPageOwner {
                slot: slot.slot,
                slot_epoch: slot.slot_epoch,
                page_idx,
            };
            let logical_id = if self
                .pages
                .get(page_id)
                .and_then(|block| block.owner)
                .is_some_and(|old| {
                    old.slot == owner.slot
                        && old.slot_epoch == owner.slot_epoch
                        && old.page_idx == owner.page_idx
                }) {
                self.pages
                    .get(page_id)
                    .expect("page checked above")
                    .logical_id
            } else {
                self.next_logical_id()
            };
            // Host page ids may be reused after the seam frees a slot. Overwrite
            // with the current slot's contents; retained/shared pages cannot be
            // reallocated by the host pool, so this does not corrupt live reuse.
            let content_key = self
                .pages
                .get(page_id)
                .filter(|block| block.logical_id == logical_id)
                .and_then(|block| block.content_key);
            let block = MetalPageBlock {
                logical_id,
                owner: Some(owner),
                content_key,
                kv_flat,
            };
            if let Some(old) = self.pages.insert(*page_id, block)
                && old.logical_id != logical_id
            {
                overwritten_logical_ids.push(old.logical_id);
            }
        }
        // Alias hazard: overwriting a logical page means this page id was
        // recycled to a new occupant. Any surviving prefix containing the old
        // logical id would pair NEW K/V with a STALE restore snapshot. Keep
        // only exact prefixes of the live occupant's logical page list.
        if !overwritten_logical_ids.is_empty()
            && let Some(live_key) = self.logical_key_for_pages(&page_ids[..publish_pages])
        {
            self.prefixes.retain(|key, _| {
                !overwritten_logical_ids
                    .iter()
                    .any(|logical_id| key.contains(logical_id))
                    || (key.len() <= live_key.len() && live_key[..key.len()] == key[..])
            });
        }

        // A reusable prefix boundary is valid only when the page-id prefix and
        // every prefix-wide side state describe the same token length. Publish
        // that restore image only at exact page boundaries.
        if slot.cache_len.is_multiple_of(page_size)
            && publish_pages == full_pages
            && let Some(key) = self.logical_key_for_pages(&page_ids[..full_pages])
        {
            self.prefixes.insert(
                key,
                MetalPrefixSnapshot {
                    cache_len: slot.cache_len,
                    gdr_flat: slot.gdr_flat.clone(),
                },
            );
        }

        Ok(())
    }

    /// Write-through one prefix restore image under the content key of its
    /// last block (idempotent: an existing blob is kept as-is).
    fn write_disk_prefix(&mut self, content_key: u64, snapshot: &MetalPrefixSnapshot) {
        let Some(tier) = self.tier.as_mut() else {
            return;
        };
        if tier.contains(tier_key(NS_PREFIX, content_key)) {
            return;
        }
        match encode_prefix_payload(content_key, snapshot) {
            Ok(bytes) => {
                if !tier.insert_chunked(NS_PREFIX, NS_PREFIX_CHUNK, content_key, &bytes) {
                    log::debug!("Metal KV T2 prefix write refused for content key {content_key}");
                }
            }
            Err(err) => {
                log::warn!(
                    "Metal KV T2 prefix encode failed for content key {content_key}: {err:#}"
                );
            }
        }
    }

    pub(super) fn materialize_slot_from_prefix(
        &mut self,
        slot: usize,
        slot_epoch: u64,
        kv: &dyn KvPool,
        prefix_tokens: usize,
        capacity_tokens: usize,
    ) -> anyhow::Result<MetalSlotState> {
        let page_size = kv.page_size().max(1);
        anyhow::ensure!(
            prefix_tokens.is_multiple_of(page_size),
            "Metal prefix attach requires page-aligned prefix: prefix_tokens={}, page_size={}",
            prefix_tokens,
            page_size
        );
        let prefix_pages = prefix_tokens / page_size;
        let slot_pages = kv.page_indices(slot);
        anyhow::ensure!(
            slot_pages.len() >= prefix_pages,
            "Metal prefix attach for slot {slot} needs {prefix_pages} pages, host slot has {}",
            slot_pages.len()
        );
        let key = self
            .logical_key_for_pages(&slot_pages[..prefix_pages])
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Metal prefix attach missing resident logical pages for slot {slot}, prefix_tokens={prefix_tokens}"
                )
            })?;
        let content_key = slot_pages
            .get(prefix_pages - 1)
            .and_then(|page| self.pages.get(page))
            .and_then(|block| block.content_key);
        self.ensure_prefix_snapshot_resident(&key, content_key)?;
        let snapshot = self.prefixes.get(&key).ok_or_else(|| {
            anyhow::anyhow!(
                "Metal prefix attach missing GDR snapshot for slot {slot}, prefix_tokens={prefix_tokens}, key={key:?}"
            )
        })?;
        anyhow::ensure!(
            snapshot.cache_len == prefix_tokens,
            "Metal prefix snapshot length mismatch for slot {slot}: requested={}, snapshot={}",
            prefix_tokens,
            snapshot.cache_len
        );

        let first_page = key
            .first()
            .ok_or_else(|| anyhow::anyhow!("Metal prefix attach got empty logical key"))?;
        let first_physical_page = slot_pages
            .first()
            .ok_or_else(|| anyhow::anyhow!("Metal prefix attach got empty page key"))?;
        let first_block = self.pages.get(first_physical_page).ok_or_else(|| {
            anyhow::anyhow!(
                "Metal prefix attach missing K/V page {first_physical_page} for slot {slot}, logical={first_page}"
            )
        })?;

        let capacity = round_up_capacity(capacity_tokens.max(prefix_tokens)) as usize;
        let kv_flat = (0..first_block.kv_flat.len())
            .map(|array_idx| -> anyhow::Result<_> {
                let page_arrays = slot_pages[..prefix_pages]
                    .iter()
                    .map(|page| -> anyhow::Result<_> {
                        let block = self.pages.get(page).ok_or_else(|| {
                            anyhow::anyhow!(
                                "Metal prefix attach missing K/V page {page} for slot {slot}"
                            )
                        })?;
                        block.kv_flat.get(array_idx).cloned().ok_or_else(|| {
                            anyhow::anyhow!(
                                "Metal prefix attach K/V page {page} is missing array index {array_idx}"
                            )
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let prefix_array = concatenate_or_single(page_arrays);
                let shape = prefix_array.shape().to_vec();
                anyhow::ensure!(
                    shape.len() == 4 && shape[2] as usize == prefix_tokens,
                    "Metal prefix K/V materialization shape mismatch for slot {slot}: shape={shape:?}, prefix_tokens={prefix_tokens}"
                );
                Ok(if capacity > prefix_tokens {
                    let mut zero_shape = shape;
                    zero_shape[2] = usize_to_i32(capacity - prefix_tokens)?;
                    let zeros = mlx::zeros(&zero_shape, prefix_array.dtype());
                    mlx::concatenate_axis(&[prefix_array, zeros], 2)
                } else {
                    prefix_array
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(MetalSlotState::from_arrays(
            slot,
            slot_epoch,
            prefix_tokens,
            kv_flat,
            snapshot.gdr_flat.clone(),
        ))
    }
}

// All-u64 LE framing. Page payload: `[n_arrays][arrays…]`. Prefix payload:
// `[content_key][cache_len][n_arrays][arrays…]` — the key echo catches a
// namespace collision at decode. Array: `[dtype][ndim][dims…][len][bytes]`.

fn encode_page_payload(arrays: &[mlx::MlxArray]) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::new();
    encode_arrays(&mut out, arrays)?;
    Ok(out)
}

fn decode_page_payload(bytes: &[u8]) -> anyhow::Result<Vec<mlx::MlxArray>> {
    let mut reader = PayloadReader { bytes, offset: 0 };
    let arrays = decode_arrays(&mut reader)?;
    reader.finish()?;
    Ok(arrays)
}

fn encode_prefix_payload(
    content_key: u64,
    snapshot: &MetalPrefixSnapshot,
) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&content_key.to_le_bytes());
    out.extend_from_slice(&(snapshot.cache_len as u64).to_le_bytes());
    encode_arrays(&mut out, &snapshot.gdr_flat)?;
    Ok(out)
}

fn decode_prefix_payload(bytes: &[u8], content_key: u64) -> anyhow::Result<MetalPrefixSnapshot> {
    let mut reader = PayloadReader { bytes, offset: 0 };
    let recorded = reader.u64()?;
    anyhow::ensure!(
        recorded == content_key,
        "Metal KV T2 prefix key mismatch: requested={content_key}, record={recorded}"
    );
    let cache_len = reader.usize()?;
    let gdr_flat = decode_arrays(&mut reader)?;
    reader.finish()?;
    Ok(MetalPrefixSnapshot {
        cache_len,
        gdr_flat,
    })
}

fn encode_arrays(out: &mut Vec<u8>, arrays: &[mlx::MlxArray]) -> anyhow::Result<()> {
    out.extend_from_slice(&(arrays.len() as u64).to_le_bytes());
    for array in arrays {
        let dtype = array.dtype();
        let shape = array.shape().to_vec();
        let bytes = array.export_bytes();
        anyhow::ensure!(
            bytes.len() == expected_array_nbytes(&shape, dtype)?,
            "Metal KV T2 encode byte size mismatch for shape={shape:?} dtype={dtype:?}"
        );
        out.extend_from_slice(&(dtype.to_raw() as i64 as u64).to_le_bytes());
        out.extend_from_slice(&(shape.len() as u64).to_le_bytes());
        for dim in &shape {
            out.extend_from_slice(&(*dim as i64 as u64).to_le_bytes());
        }
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&bytes);
    }
    Ok(())
}

fn decode_arrays(reader: &mut PayloadReader<'_>) -> anyhow::Result<Vec<mlx::MlxArray>> {
    let array_count = reader.usize()?;
    (0..array_count)
        .map(|_| {
            let dtype_raw = reader.i64()?;
            let dtype = i32::try_from(dtype_raw)
                .ok()
                .and_then(mlx::Dtype::from_raw)
                .ok_or_else(|| anyhow::anyhow!("Metal KV T2 unknown dtype {dtype_raw}"))?;
            let ndim = reader.usize()?;
            let shape: Vec<i32> = (0..ndim)
                .map(|_| {
                    let dim = reader.i64()?;
                    i32::try_from(dim)
                        .map_err(|_| anyhow::anyhow!("Metal KV T2 array dim {dim} exceeds i32"))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            let byte_len = reader.usize()?;
            anyhow::ensure!(
                byte_len == expected_array_nbytes(&shape, dtype)?,
                "Metal KV T2 array byte size mismatch for shape={shape:?} dtype={dtype:?}"
            );
            let data = reader.take(byte_len)?;
            Ok(mlx::MlxArray::from_bytes(data, &shape, dtype))
        })
        .collect()
}

struct PayloadReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PayloadReader<'a> {
    fn take(&mut self, len: usize) -> anyhow::Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("Metal KV T2 payload cursor overflow"))?;
        anyhow::ensure!(
            end <= self.bytes.len(),
            "Metal KV T2 truncated payload: need {len} bytes at offset {}",
            self.offset
        );
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn u64(&mut self) -> anyhow::Result<u64> {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(raw))
    }

    fn i64(&mut self) -> anyhow::Result<i64> {
        self.u64().map(|v| v as i64)
    }

    fn usize(&mut self) -> anyhow::Result<usize> {
        let value = self.u64()?;
        usize::try_from(value)
            .map_err(|_| anyhow::anyhow!("Metal KV T2 payload field {value} exceeds usize"))
    }

    fn finish(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.offset == self.bytes.len(),
            "Metal KV T2 payload has {} trailing bytes",
            self.bytes.len().saturating_sub(self.offset)
        );
        Ok(())
    }
}

pub fn expected_array_nbytes(shape: &[i32], dtype: mlx::Dtype) -> anyhow::Result<usize> {
    let mut elements = 1usize;
    for dim in shape {
        anyhow::ensure!(*dim >= 0, "negative MLX array dimension {dim}");
        elements = elements
            .checked_mul(*dim as usize)
            .ok_or_else(|| anyhow::anyhow!("MLX array shape overflows usize: {shape:?}"))?;
    }
    elements
        .checked_mul(dtype_size(dtype))
        .ok_or_else(|| anyhow::anyhow!("MLX array byte size overflows usize: {shape:?}"))
}

pub fn dtype_size(dtype: mlx::Dtype) -> usize {
    match dtype {
        mlx::Dtype::Bool | mlx::Dtype::Uint8 | mlx::Dtype::Int8 => 1,
        mlx::Dtype::Uint16 | mlx::Dtype::Int16 | mlx::Dtype::Float16 | mlx::Dtype::Bfloat16 => 2,
        mlx::Dtype::Uint32 | mlx::Dtype::Int32 | mlx::Dtype::Float32 => 4,
        mlx::Dtype::Uint64 | mlx::Dtype::Int64 | mlx::Dtype::Float64 | mlx::Dtype::Complex64 => 8,
    }
}

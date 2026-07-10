//! DSv4 cross-request prefix-state pool (#154 Phase 2 reland).
//!
//! One HOST-resident entry per (host page id): everything a restore at token
//! boundary `page_tokens·(k+1)` needs from every layer, D2H'd once when the
//! page completes. Content identity rides the host page id — the radix
//! dedupes prefix chains, so matching prefixes share host page ids and
//! non-matching content never collides (the D1 flaw is unrepresentable).
//! Zero HBM footprint: the pool IS the L2 tier (`CudaKvTierStore` host level);
//! L3 is the same store's mmap spill.
//!
//! Entry sections split by observability: content sections (staging/dsa
//! rows) are positional storage, capturable post-forward for every completed
//! page; boundary sections (overlap registers + ring) are transient — the
//! registers are overwritten every `ratio` tokens and the ring wraps every
//! `sliding_window` tokens, so they exist only at the forward's own end.
//! v1 therefore restores at chunk/tick-end boundaries (the planner already
//! forces prefill chunk ends onto `sliding_window` alignment). v2 (per-page
//! boundary granularity) needs a kernel-side per-block overlap output buffer
//! — the CUDA compressor's `overlap_page_stride` param can target a per-slot
//! capture buffer (no sharing, no D1 risk); design-only, not implemented.
//!
//! Under TP every rank redundantly holds its own shard's entries (state is
//! rank-replicated by construction) — DRAM cost is ×world; a rank-0-only
//! layout is a later optimization. Weight updates: DSv4 has no live-reload
//! path (both executor arms bail), the L3 namespace is per-process ephemeral,
//! and the engine's `invalidate_prefix_cache` drops every entry through the
//! `release_prefix_pages` seam — no cross-epoch reuse window.

use std::collections::BTreeMap;

use super::*;
use crate::kv_tier::{CudaKvTierStore, NS_PREFIX_STATE, tier_key};

/// Rows per FP8 DSA key-cache page (`dsv4_dsa_official.cu` fused-store:
/// `kPageBytes = 132 << 6` = 64 rows × (128 B data + 4 B f32 scale)).
const DSA_PAGE_ROWS: usize = 64;

/// One layer's share of a per-page entry. Content sections (`staging`,
/// `dsa_*`) are captured for EVERY completed page; boundary sections
/// (`overlap_*`, `idx_overlap_*`, `ring`) only when the forward ended
/// exactly at the page end (the registers/ring for an overshot boundary are
/// already advanced past it and unrecoverable). Empty vec = section absent
/// for this layer/page.
///
/// The FP8 compressed band is NOT captured: it is derived state (staging
/// quantized), only written on the DECODE lane (`flashmla_pack_compressed_delta`),
/// so at prefill-time capture it is still the `zero_slot_band` zeros —
/// restoring it corrupted every warm decode (E2, pod-proven 2026-07-09:
/// publish fingerprints showed `band=0/196224` for every prefill-published
/// page). Restore instead leaves `fp8_kv_comp_packed_rows=0` and the first
/// post-restore decode's bulk pack rebuilds the band from the restored
/// staging.
#[derive(Default, PartialEq)]
pub(crate) struct Dsv4LayerPageState {
    /// Main compressor bf16 staging rows (A2: read as full history by the
    /// hybrid CSA/HCA attention). Indexer staging is NOT captured — its only
    /// reader drains the delta `[packed_rows, seq_len)`, empty at a boundary.
    pub(super) staging: Vec<half::bf16>,
    /// FP8 DSA key-cache rows (paged data bytes, B1-B3 layout).
    pub(super) dsa_data: Vec<u8>,
    /// FP8 DSA key-cache f32 scales for the same rows.
    pub(super) dsa_scale: Vec<u8>,
    /// Main compressor `prev_overlap_kv` at the page-end boundary.
    pub(super) overlap_kv: Vec<half::bf16>,
    /// Main compressor `prev_overlap_score` at the page-end boundary.
    pub(super) overlap_score: Vec<half::bf16>,
    /// Indexer `prev_overlap_kv` at the page-end boundary.
    pub(super) idx_overlap_kv: Vec<half::bf16>,
    /// Indexer `prev_overlap_score` at the page-end boundary.
    pub(super) idx_overlap_score: Vec<half::bf16>,
    /// Full bf16 SW ring at the page-end boundary (FP8 ring region is NOT
    /// stored: restore sets `fp8_kv_sw_bootstrapped=false` and the existing
    /// bootstrap repacks from this ring — the bf16 ring stays the single
    /// source, A14 resolved by design).
    pub(super) ring: Vec<half::bf16>,
}

/// One host page's captured state across all layers.
pub(crate) struct Dsv4PrefixPageEntry {
    /// Slot-logical page index the entry was captured at. KV content is
    /// position-dependent, so a restore must see the same index — a mismatch
    /// means the host page id was recycled into different content.
    pub(crate) page_index: u32,
    /// Boundary sections present (forward ended exactly at this page's end).
    pub(crate) boundary: bool,
    pub(crate) layers: Vec<Dsv4LayerPageState>,
}

const ENTRY_MAGIC: &[u8; 4] = b"DSPP";

fn push_bytes(buf: &mut Vec<u8>, v: &[u8]) {
    buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
    buf.extend_from_slice(v);
}

fn push_bf16(buf: &mut Vec<u8>, v: &[half::bf16]) {
    let byte_len = v.len() * 2;
    buf.extend_from_slice(&(byte_len as u32).to_le_bytes());
    // SAFETY: half::bf16 is #[repr(transparent)] over u16; byte view is valid.
    let raw = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, byte_len) };
    buf.extend_from_slice(raw);
}

fn read_bytes(pos: &mut usize, bytes: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        *pos + 4 <= bytes.len(),
        "prefix-state entry truncated at len"
    );
    let len = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap()) as usize;
    *pos += 4;
    ensure!(
        *pos + len <= bytes.len(),
        "prefix-state entry truncated at section"
    );
    let v = bytes[*pos..*pos + len].to_vec();
    *pos += len;
    Ok(v)
}

fn read_bf16(pos: &mut usize, bytes: &[u8]) -> Result<Vec<half::bf16>> {
    let raw = read_bytes(pos, bytes)?;
    ensure!(
        raw.len().is_multiple_of(2),
        "prefix-state bf16 section has odd byte length {}",
        raw.len()
    );
    Ok(raw
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
        .collect())
}

impl Dsv4PrefixPageEntry {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.host_bytes() + 64);
        buf.extend_from_slice(ENTRY_MAGIC);
        buf.extend_from_slice(&self.page_index.to_le_bytes());
        buf.push(u8::from(self.boundary));
        buf.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for layer in &self.layers {
            push_bf16(&mut buf, &layer.staging);
            push_bytes(&mut buf, &layer.dsa_data);
            push_bytes(&mut buf, &layer.dsa_scale);
            push_bf16(&mut buf, &layer.overlap_kv);
            push_bf16(&mut buf, &layer.overlap_score);
            push_bf16(&mut buf, &layer.idx_overlap_kv);
            push_bf16(&mut buf, &layer.idx_overlap_score);
            push_bf16(&mut buf, &layer.ring);
        }
        buf
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= 13 && &bytes[..4] == ENTRY_MAGIC,
            "bad prefix-state entry header"
        );
        let page_index = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let boundary = bytes[8] != 0;
        let n_layers = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
        let mut pos = 13usize;
        let layers = (0..n_layers)
            .map(|_| {
                Ok(Dsv4LayerPageState {
                    staging: read_bf16(&mut pos, bytes)?,
                    dsa_data: read_bytes(&mut pos, bytes)?,
                    dsa_scale: read_bytes(&mut pos, bytes)?,
                    overlap_kv: read_bf16(&mut pos, bytes)?,
                    overlap_score: read_bf16(&mut pos, bytes)?,
                    idx_overlap_kv: read_bf16(&mut pos, bytes)?,
                    idx_overlap_score: read_bf16(&mut pos, bytes)?,
                    ring: read_bf16(&mut pos, bytes)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(pos == bytes.len(), "prefix-state entry has trailing bytes");
        Ok(Self {
            page_index,
            boundary,
            layers,
        })
    }

    pub(crate) fn host_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                l.dsa_data.len()
                    + l.dsa_scale.len()
                    + (l.staging.len()
                        + l.overlap_kv.len()
                        + l.overlap_score.len()
                        + l.idx_overlap_kv.len()
                        + l.idx_overlap_score.len()
                        + l.ring.len())
                        * 2
            })
            .sum()
    }
}

/// STATIC upper bound on one encoded entry — MUST mirror the capture fns +
/// codec framing (kept adjacent so drift is visible). Sizes the pool store's
/// fixed page slots (worst case = boundary entry with ring).
pub(crate) fn dsv4_prefix_entry_max_bytes(
    config: &DeepSeekV4Config,
    layer_specs: &[(DeepSeekV4AttentionMode, usize)],
    page_tokens: usize,
) -> usize {
    let bf16 = 2usize;
    let mut total = 13usize; // entry header
    for &(mode, ratio) in layer_specs {
        total += 8 * 4; // section length prefixes
        // ring — every layer has an SW window cache.
        total += config.sliding_window * config.head_dim * bf16;
        // ceil, not floor: for ratio > page_tokens a page can still complete
        // one row (page_row_span), and the predictor must never undersize.
        if mode.has_compressor() && ratio > 0 {
            let rpp = page_tokens.div_ceil(ratio);
            total += rpp * config.head_dim * bf16; // staging
            total += 2 * ratio * config.head_dim * bf16; // overlap kv+score
        }
        if mode.has_indexer() {
            let index_ratio = if mode == DeepSeekV4AttentionMode::SparseIndexed {
                1
            } else {
                ratio.max(1)
            };
            let rpp = page_tokens.div_ceil(index_ratio);
            total += rpp * (config.index_head_dim + 4); // dsa data+scale
            total += 2 * index_ratio * config.index_head_dim * bf16; // idx overlap
        }
    }
    total
}

#[derive(Clone, Copy)]
pub(crate) struct Dsv4PrefixPageMeta {
    /// Entry carries the boundary sections (restore may commit here).
    pub(crate) boundary: bool,
    /// Radix publish confirmed this page (executor `save_prefix_sidecar` arm).
    /// A provisional entry's page id may recycle after an unpublished free —
    /// its content stays write-only (restore rejects) until confirmed.
    pub(crate) confirmed: bool,
    /// Last publish/read stamp — the entry's key into `lru`.
    stamp: u64,
}

/// Host-resident content-keyed pool: page id → encoded [`Dsv4PrefixPageEntry`].
/// Storage rides ONE dedicated [`CudaKvTierStore`] (host DRAM level = L2;
/// `set_disk` adds the mmap L3). `meta` is the exact host index — the store
/// never drops keys on its own, so the two cannot drift.
pub(crate) struct Dsv4PrefixStatePool {
    store: CudaKvTierStore,
    meta: BTreeMap<u32, Dsv4PrefixPageMeta>,
    /// Over-cap eviction index `(confirmed, stamp, page id)`: provisional
    /// entries drop before confirmed ones (a confirmed boundary entry is the
    /// most valuable byte in the pool), LRU within each tier. Reads touch
    /// (restore traffic keeps a hot shared preamble resident); a FIFO here
    /// evicted the hottest first-published preamble under DRAM pressure and
    /// floor-0-locked the whole chain.
    lru: std::collections::BTreeSet<(bool, u64, u32)>,
    clock: u64,
    entry_bytes: usize,
}

impl Dsv4PrefixStatePool {
    pub(crate) fn new(budget_bytes: usize, entry_bytes: usize) -> Self {
        Self {
            store: CudaKvTierStore::with_budget(budget_bytes, entry_bytes.max(1)),
            meta: BTreeMap::new(),
            lru: std::collections::BTreeSet::new(),
            clock: 0,
            entry_bytes: entry_bytes.max(1),
        }
    }

    /// Pre-serve re-budget rebuilds the store, dropping every entry (same
    /// contract as the slot tier's re-budget).
    pub(crate) fn set_budget_bytes(&mut self, bytes: usize) {
        self.store = CudaKvTierStore::with_budget(bytes, self.entry_bytes);
        self.meta.clear();
        self.lru.clear();
    }

    fn next_stamp(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    /// Move `page_id` to the back of its LRU tier (and optionally across
    /// tiers on confirm). Meta must exist.
    fn touch(&mut self, page_id: u32, confirm: bool) {
        let stamp = self.next_stamp();
        let Some(meta) = self.meta.get_mut(&page_id) else {
            return;
        };
        self.lru.remove(&(meta.confirmed, meta.stamp, page_id));
        meta.confirmed |= confirm;
        meta.stamp = stamp;
        self.lru.insert((meta.confirmed, stamp, page_id));
    }

    pub(crate) fn set_disk(&mut self, root: std::path::PathBuf, budget_bytes: usize) -> bool {
        self.store.set_disk(root, budget_bytes, self.entry_bytes)
    }

    /// No capacity on either level ⇒ publish is a no-op (skip the D2H cost).
    pub(crate) fn is_inactive(&self) -> bool {
        self.store.capacity_pages() == 0
    }

    pub(crate) fn host_pages(&self) -> usize {
        self.store.host_demoted_pages()
    }

    pub(crate) fn disk_pages(&self) -> usize {
        self.store.disk_pages()
    }

    /// Insert (LAST producer wins). Host page ids recycle when freed, so a
    /// republish under a recycled id MUST overwrite — a page a slot completes
    /// is either radix-shared (content-identical) or slot-exclusive, so the
    /// newest content is always the correct one. When both levels are full,
    /// drop oldest entries and retry (capacity never blocks the forward).
    pub(crate) fn publish(&mut self, page_id: u32, entry: &Dsv4PrefixPageEntry) -> bool {
        let bytes = entry.to_bytes();
        let key = tier_key(NS_PREFIX_STATE, u64::from(page_id));
        let mut payload = bytes;
        loop {
            match self.store.insert(key, payload) {
                true => break,
                false => {
                    let Some(oldest) = self.pop_oldest_other(page_id) else {
                        return false;
                    };
                    self.remove_pages(&[oldest]);
                    payload = entry.to_bytes();
                }
            }
        }
        // Carry `confirmed` forward: a confirmed id still present is still
        // radix-held (evict would have dropped it), so any overwrite is a
        // content-identical sharer's republish.
        let confirmed = self.meta.get(&page_id).is_some_and(|m| m.confirmed);
        if let Some(old) = self.meta.get(&page_id) {
            self.lru.remove(&(old.confirmed, old.stamp, page_id));
        }
        let stamp = self.next_stamp();
        self.lru.insert((confirmed, stamp, page_id));
        self.meta.insert(
            page_id,
            Dsv4PrefixPageMeta {
                boundary: entry.boundary,
                confirmed,
                stamp,
            },
        );
        true
    }

    /// Radix publish confirmed these pages (they are now radix-retained, so
    /// their ids cannot recycle while the entries live).
    pub(crate) fn confirm_pages(&mut self, pages: &[u32]) {
        for &page_id in pages {
            if self.meta.get(&page_id).is_some_and(|m| !m.confirmed) {
                self.touch(page_id, true);
            }
        }
    }

    /// #157 repair — the floor-0 self-lock breaker. Radix dedup keeps the
    /// CANONICAL page id; a recomputing slot publishes onto its OWN page id,
    /// so once a canonical entry evicts, nothing ever re-attaches state to
    /// the canonical chain. At confirm time, adopt the finishing slot's
    /// provisional entry for the same position: content is identical by
    /// construction (same token positions, recomputed). Guard: never clobber
    /// an existing confirmed canonical entry.
    pub(crate) fn adopt_canonical(&mut self, canonical: u32, own: u32) {
        match self.meta.get(&canonical) {
            Some(m) if m.confirmed => return,
            // Canonical id is radix-retained (never recycled while cached), so
            // a provisional entry under it holds the canonical content —
            // confirm in place.
            Some(_) => {
                self.touch(canonical, true);
                return;
            }
            None => {}
        }
        // Re-key the slot's own provisional entry to the canonical id.
        let Some(own_meta) = self.meta.get(&own).copied().filter(|m| !m.confirmed) else {
            return;
        };
        let boundary = own_meta.boundary;
        let own_key = tier_key(NS_PREFIX_STATE, u64::from(own));
        let Ok(bytes) = self.store.read(own_key).map(|b| b.into_owned()) else {
            return;
        };
        self.remove_pages(&[own]);
        let canonical_key = tier_key(NS_PREFIX_STATE, u64::from(canonical));
        if !self.store.insert(canonical_key, bytes) {
            return;
        }
        let stamp = self.next_stamp();
        self.lru.insert((true, stamp, canonical));
        self.meta.insert(
            canonical,
            Dsv4PrefixPageMeta {
                boundary,
                confirmed: true,
                stamp,
            },
        );
    }

    fn pop_oldest_other(&mut self, publishing: u32) -> Option<u32> {
        self.lru
            .iter()
            .map(|&(_, _, id)| id)
            .find(|&id| id != publishing)
    }

    pub(crate) fn page_meta(&self, page_id: u32) -> Option<Dsv4PrefixPageMeta> {
        self.meta.get(&page_id).copied()
    }

    pub(crate) fn read_entry(&mut self, page_id: u32) -> Result<Dsv4PrefixPageEntry> {
        ensure!(
            self.meta.get(&page_id).is_some_and(|m| m.confirmed),
            "prefix-state pool has no confirmed entry for host page {page_id}"
        );
        let key = tier_key(NS_PREFIX_STATE, u64::from(page_id));
        let promote = matches!(
            self.store.location(key),
            Some(infer_seam::KvTierLocation::Disk)
        );
        let bytes = self.store.read(key)?.into_owned();
        let entry = Dsv4PrefixPageEntry::from_bytes(&bytes)?;
        // Read-on-miss promote: a disk-resident entry that restores is hot —
        // re-insert to the host level (its LRU spills something colder), then
        // drop the now-superseded disk record (else the key is double-resident
        // and the soft-cap math counts it twice). Only when the insert actually
        // landed in HOST — a full host level routes the insert back to disk,
        // and removing the record then would lose the entry.
        if promote
            && self.store.insert(key, bytes)
            && matches!(
                self.store.location(key),
                Some(infer_seam::KvTierLocation::HostDemoted)
            )
        {
            self.store.remove_disk_only(key);
        }
        // Touch-on-read: a restored entry is hot — LRU, not publish-order FIFO.
        self.touch(page_id, false);
        Ok(entry)
    }

    /// Radix evicted these host pages: drop their entries (eviction rides the
    /// radix; the pool holds no independent lifetime).
    pub(crate) fn remove_pages(&mut self, pages: &[u32]) {
        let keys: Vec<u64> = pages
            .iter()
            .filter_map(|&p| {
                let meta = self.meta.remove(&p)?;
                self.lru.remove(&(meta.confirmed, meta.stamp, p));
                Some(tier_key(NS_PREFIX_STATE, u64::from(p)))
            })
            .collect();
        if !keys.is_empty() {
            self.store.remove(&keys);
        }
    }

    /// A slot freed these pages back to the pool: drop only the NON-confirmed
    /// entries (write-only, never radix-published). A freed id recycles, and a
    /// lingering provisional entry could later be confirmed as if it held the
    /// new occupant's content. Confirmed entries are radix-retained (their
    /// pages never free while the entry lives) and stay.
    pub(crate) fn remove_provisional_pages(&mut self, pages: &[u32]) {
        let provisional: Vec<u32> = pages
            .iter()
            .filter(|p| self.meta.get(p).is_some_and(|m| !m.confirmed))
            .copied()
            .collect();
        self.remove_pages(&provisional);
    }
}

impl Dsv4LayerAttentionState {
    /// D2H one host page's share of this layer's state. `boundary` additionally
    /// captures the page-end registers + ring — valid only when the forward
    /// ended exactly at `page_tokens·(page_index+1)` (the caller's contract).
    /// Host vectors are stream-ordered; the caller syncs once after all layers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_prefix_page(
        &self,
        ctx: &DeviceContext,
        pool: &Dsv4LayerKvLayout,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        index_head_dim: usize,
        page_tokens: usize,
        page_index: usize,
        boundary: bool,
    ) -> Result<Dsv4LayerPageState> {
        let mut out = Dsv4LayerPageState::default();
        if let Some(c) = &self.compressor {
            if let Some(range) = staging_row_range(c, compress_ratio, page_tokens, page_index)? {
                out.staging = ctx
                    .stream
                    .clone_dtoh(&c.compressed.data.slice(range))
                    .map_err(|e| anyhow!("DSv4 prefix capture staging D2H failed: {e}"))?;
            }
            if boundary {
                out.overlap_kv = ctx
                    .stream
                    .clone_dtoh(&c.prev_overlap_kv)
                    .map_err(|e| anyhow!("DSv4 prefix capture overlap kv D2H failed: {e}"))?;
                out.overlap_score = ctx
                    .stream
                    .clone_dtoh(&c.prev_overlap_score)
                    .map_err(|e| anyhow!("DSv4 prefix capture overlap score D2H failed: {e}"))?;
            }
        }
        if let Some(ix) = &self.indexer
            && boundary
        {
            out.idx_overlap_kv = ctx
                .stream
                .clone_dtoh(&ix.prev_overlap_kv)
                .map_err(|e| anyhow!("DSv4 prefix capture idx overlap kv D2H failed: {e}"))?;
            out.idx_overlap_score = ctx
                .stream
                .clone_dtoh(&ix.prev_overlap_score)
                .map_err(|e| anyhow!("DSv4 prefix capture idx overlap score D2H failed: {e}"))?;
        }
        if let Some(dsa) = &self.dsa_official
            && let Some((data_range, scale_range)) = dsa_row_ranges(
                pool,
                dsa,
                mode,
                compress_ratio,
                index_head_dim,
                page_tokens,
                page_index,
            )?
        {
            let cache = pool
                .dsa_key_cache
                .as_ref()
                .ok_or_else(|| anyhow!("DSv4 prefix capture: DSA shared key-cache missing"))?;
            ensure!(
                data_range.end <= cache.len() && scale_range.end <= cache.len(),
                "DSv4 prefix capture DSA range outside cache bytes {}",
                cache.len()
            );
            out.dsa_data = ctx
                .stream
                .clone_dtoh(&cache.slice(data_range))
                .map_err(|e| anyhow!("DSv4 prefix capture DSA data D2H failed: {e}"))?;
            out.dsa_scale = ctx
                .stream
                .clone_dtoh(&cache.slice(scale_range))
                .map_err(|e| anyhow!("DSv4 prefix capture DSA scale D2H failed: {e}"))?;
        }
        if boundary {
            out.ring = ctx
                .stream
                .clone_dtoh(&self.sw_window_cache)
                .map_err(|e| anyhow!("DSv4 prefix capture ring D2H failed: {e}"))?;
        }
        Ok(out)
    }

    /// H2D the inverse of [`Self::capture_prefix_page`]. `boundary` restores
    /// the page-end registers + ring (final matched page only). Every section
    /// length is checked against the live shape before any byte moves.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn restore_prefix_page(
        &mut self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        index_head_dim: usize,
        page_tokens: usize,
        page_index: usize,
        state: &Dsv4LayerPageState,
        boundary: bool,
    ) -> Result<()> {
        if let Some(c) = &mut self.compressor {
            if let Some(range) = staging_row_range(c, compress_ratio, page_tokens, page_index)? {
                ensure!(
                    state.staging.len() == range.len(),
                    "DSv4 prefix restore staging section {} != live rows {}",
                    state.staging.len(),
                    range.len()
                );
                let mut view = c.compressed.data.slice_mut(range);
                ctx.stream
                    .memcpy_htod(&state.staging, &mut view)
                    .map_err(|e| anyhow!("DSv4 prefix restore staging H2D failed: {e}"))?;
            } else {
                // Capture derives the same span from the same config — a
                // stored section here means capture-era shape drift.
                ensure!(
                    state.staging.is_empty(),
                    "DSv4 prefix restore: entry has staging rows for a page the live shape completes none"
                );
            }
            if boundary {
                ensure!(
                    state.overlap_kv.len() == c.prev_overlap_kv.len()
                        && state.overlap_score.len() == c.prev_overlap_score.len(),
                    "DSv4 prefix restore overlap sections mismatch"
                );
                ctx.stream
                    .memcpy_htod(&state.overlap_kv, &mut c.prev_overlap_kv)
                    .map_err(|e| anyhow!("DSv4 prefix restore overlap kv H2D failed: {e}"))?;
                ctx.stream
                    .memcpy_htod(&state.overlap_score, &mut c.prev_overlap_score)
                    .map_err(|e| anyhow!("DSv4 prefix restore overlap score H2D failed: {e}"))?;
            }
        }
        if let Some(ix) = &mut self.indexer
            && boundary
        {
            ensure!(
                state.idx_overlap_kv.len() == ix.prev_overlap_kv.len()
                    && state.idx_overlap_score.len() == ix.prev_overlap_score.len(),
                "DSv4 prefix restore idx overlap sections mismatch"
            );
            ctx.stream
                .memcpy_htod(&state.idx_overlap_kv, &mut ix.prev_overlap_kv)
                .map_err(|e| anyhow!("DSv4 prefix restore idx overlap kv H2D failed: {e}"))?;
            ctx.stream
                .memcpy_htod(&state.idx_overlap_score, &mut ix.prev_overlap_score)
                .map_err(|e| anyhow!("DSv4 prefix restore idx overlap score H2D failed: {e}"))?;
        }
        if let Some(dsa) = &self.dsa_official
            && let Some((data_range, scale_range)) = dsa_row_ranges(
                pool,
                dsa,
                mode,
                compress_ratio,
                index_head_dim,
                page_tokens,
                page_index,
            )?
        {
            ensure!(
                state.dsa_data.len() == data_range.len()
                    && state.dsa_scale.len() == scale_range.len(),
                "DSv4 prefix restore DSA section {}+{} != live {}+{}",
                state.dsa_data.len(),
                state.dsa_scale.len(),
                data_range.len(),
                scale_range.len()
            );
            let cache = pool
                .dsa_key_cache
                .as_mut()
                .ok_or_else(|| anyhow!("DSv4 prefix restore: DSA shared key-cache missing"))?;
            ensure!(
                data_range.end <= cache.len() && scale_range.end <= cache.len(),
                "DSv4 prefix restore DSA range outside cache bytes {}",
                cache.len()
            );
            {
                let mut data = cache.slice_mut(data_range);
                ctx.stream
                    .memcpy_htod(&state.dsa_data, &mut data)
                    .map_err(|e| anyhow!("DSv4 prefix restore DSA data H2D failed: {e}"))?;
            }
            let mut scale = cache.slice_mut(scale_range);
            ctx.stream
                .memcpy_htod(&state.dsa_scale, &mut scale)
                .map_err(|e| anyhow!("DSv4 prefix restore DSA scale H2D failed: {e}"))?;
        }
        if boundary {
            ensure!(
                state.ring.len() == self.sw_window_cache.len(),
                "DSv4 prefix restore ring section {} != live ring {}",
                state.ring.len(),
                self.sw_window_cache.len()
            );
            ctx.stream
                .memcpy_htod(&state.ring, &mut self.sw_window_cache)
                .map_err(|e| anyhow!("DSv4 prefix restore ring H2D failed: {e}"))?;
        }
        Ok(())
    }

    /// Host counters for a restore at `matched_len` (a page-aligned boundary):
    /// staging/indexer row counts, DSA packed rows, and the FlashMLA FP8
    /// counters. `fp8_kv_sw_bootstrapped=false` forces the SW-ring repack from
    /// the restored bf16 ring (A14); `fp8_kv_comp_packed_rows=0` forces the
    /// first post-restore decode's bulk pack to rebuild the FP8 band from the
    /// restored staging (the band itself is never captured — see
    /// [`Dsv4LayerPageState`]). `pending_kv/score` need no restore:
    /// `matched_len % ratio == 0` ⇒ no partial block, so the next compress
    /// overwrites before any read.
    pub(crate) fn restore_prefix_counters(
        &mut self,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        matched_len: usize,
    ) {
        let comp_rows = if mode == DeepSeekV4AttentionMode::SlidingWindow {
            0
        } else {
            matched_len / compress_ratio.max(1)
        };
        let index_ratio = if mode == DeepSeekV4AttentionMode::SparseIndexed {
            1
        } else {
            compress_ratio.max(1)
        };
        let index_rows = if mode.has_indexer() {
            matched_len / index_ratio
        } else {
            0
        };
        if let Some(c) = &mut self.compressor {
            c.compressed.seq_len = comp_rows;
        }
        if let Some(ix) = &mut self.indexer {
            ix.compressed.seq_len = index_rows;
        }
        if let Some(dsa) = &mut self.dsa_official {
            dsa.packed_rows = index_rows;
        }
        if let Some(flash) = &mut self.flashmla {
            flash.fp8_kv_comp_packed_rows = 0;
            flash.fp8_kv_sw_bootstrapped = false;
        }
    }
}

/// Compressed rows newly COMPLETED by host page `page_index`:
/// `[page_start/ratio, page_end/ratio)`. Uniform over every ratio — cr=4 →
/// 16 rows/page; cr=128 → a row lands on the page whose tokens complete it
/// (0 rows on the other page); ratio=1 → 64 rows/page.
fn page_row_span(page_tokens: usize, ratio: usize, page_index: usize) -> (usize, usize) {
    let start = (page_index * page_tokens) / ratio;
    let end = ((page_index + 1) * page_tokens) / ratio;
    (start, end - start)
}

/// Element range of one host page's rows in the main compressor's bf16
/// staging (`compressed.data`, row-major `[row][head_dim]`); `None` when the
/// page completes no row.
fn staging_row_range(
    c: &Dsv4CompressorState,
    compress_ratio: usize,
    page_tokens: usize,
    page_index: usize,
) -> Result<Option<std::ops::Range<usize>>> {
    ensure!(
        c.ring_rows == c.compressed_capacity,
        "DSv4 prefix staging: main compressor staging must be full-history, not a ring"
    );
    ensure!(compress_ratio > 0, "DSv4 prefix staging: zero ratio");
    let (row0, count) = page_row_span(page_tokens, compress_ratio, page_index);
    if count == 0 {
        return Ok(None);
    }
    ensure!(
        row0 + count <= c.compressed_capacity,
        "DSv4 prefix staging rows {row0}..{} outside capacity {}",
        row0 + count,
        c.compressed_capacity
    );
    let head_dim = c.compressed.data.len() / c.ring_rows.max(1);
    ensure!(
        head_dim * c.ring_rows == c.compressed.data.len(),
        "DSv4 prefix staging: data len {} not row-divisible by {}",
        c.compressed.data.len(),
        c.ring_rows
    );
    Ok(Some(row0 * head_dim..(row0 + count) * head_dim))
}

/// Byte ranges of one host page's rows in the slot's FP8 DSA key-cache band
/// (`None` when the page completes no row): paged layout
/// `[64×index_head_dim data][64×f32 scales]` per page — the B1-B3 lesson:
/// never flat `row × (dim+4)` math.
#[allow(clippy::type_complexity)]
fn dsa_row_ranges(
    pool: &Dsv4LayerKvLayout,
    dsa: &Dsv4DsaOfficialState,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    index_head_dim: usize,
    page_tokens: usize,
    page_index: usize,
) -> Result<Option<(std::ops::Range<usize>, std::ops::Range<usize>)>> {
    let index_ratio = if mode == DeepSeekV4AttentionMode::SparseIndexed {
        1
    } else {
        compress_ratio
    };
    ensure!(index_ratio > 0, "DSv4 prefix DSA: zero index ratio");
    let (row0, count) = page_row_span(page_tokens, index_ratio, page_index);
    if count == 0 {
        return Ok(None);
    }
    ensure!(
        row0 / DSA_PAGE_ROWS == (row0 + count - 1) / DSA_PAGE_ROWS,
        "DSv4 prefix DSA rows {row0}..{} straddle a {DSA_PAGE_ROWS}-row cache page",
        row0 + count
    );
    let cache_page = row0 / DSA_PAGE_ROWS;
    let in_row = row0 % DSA_PAGE_ROWS;
    let page_bytes = DSA_PAGE_ROWS * (index_head_dim + std::mem::size_of::<f32>());
    let slot_range = pool.dsa_slot_range(dsa.slot_idx)?;
    let page_base = slot_range.start + cache_page * page_bytes;
    ensure!(
        page_base + page_bytes <= slot_range.end,
        "DSv4 prefix DSA page {cache_page} outside slot band {slot_range:?}"
    );
    let data_start = page_base + in_row * index_head_dim;
    let scale_start =
        page_base + DSA_PAGE_ROWS * index_head_dim + in_row * std::mem::size_of::<f32>();
    Ok(Some((
        data_start..data_start + count * index_head_dim,
        scale_start..scale_start + count * std::mem::size_of::<f32>(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end-shaped codec gate: a two-layer entry (one boundary-bearing,
    /// one content-only, one section empty) must round-trip byte-exact through
    /// the pool's publish/read path.
    #[test]
    fn entry_codec_round_trips_through_pool() {
        let bf = |v: &[f32]| v.iter().copied().map(half::bf16::from_f32).collect();
        let entry = Dsv4PrefixPageEntry {
            page_index: 7,
            boundary: true,
            layers: vec![
                Dsv4LayerPageState {
                    staging: bf(&[0.5, -1.25]),
                    dsa_data: vec![5, 6],
                    dsa_scale: vec![0, 0, 128, 63],
                    overlap_kv: bf(&[2.0]),
                    overlap_score: bf(&[-3.0]),
                    idx_overlap_kv: bf(&[0.125]),
                    idx_overlap_score: bf(&[4.0]),
                    ring: bf(&[1.0, 2.0, 3.0]),
                },
                Dsv4LayerPageState {
                    ring: bf(&[7.0]),
                    ..Default::default()
                },
            ],
        };
        let mut pool = Dsv4PrefixStatePool::new(1 << 20, 1 << 16);
        assert!(pool.publish(42, &entry));
        assert!(pool.page_meta(42).is_some_and(|m| m.boundary));
        assert!(
            pool.read_entry(42).is_err(),
            "provisional entries are write-only until radix publish confirms"
        );
        pool.confirm_pages(&[42]);
        let back = pool.read_entry(42).expect("entry readable");
        assert_eq!(back.page_index, entry.page_index);
        assert_eq!(back.boundary, entry.boundary);
        assert_eq!(back.layers.len(), 2);
        assert!(back.layers[0] == entry.layers[0] && back.layers[1] == entry.layers[1]);
        pool.remove_pages(&[42]);
        assert!(pool.page_meta(42).is_none());
        assert!(pool.read_entry(42).is_err());
    }

    fn small_entry(page_index: u32) -> Dsv4PrefixPageEntry {
        Dsv4PrefixPageEntry {
            page_index,
            boundary: true,
            layers: vec![Dsv4LayerPageState {
                ring: vec![half::bf16::from_f32(page_index as f32)],
                ..Default::default()
            }],
        }
    }

    /// Over-cap eviction is provisional-first LRU, not publish-order FIFO: a
    /// confirmed + recently-read entry (the hot shared preamble) survives a
    /// younger provisional one.
    #[test]
    fn eviction_prefers_provisional_and_respects_read_lru() {
        // Host level fits exactly 2 entries; no disk.
        let entry_bytes = small_entry(0).to_bytes().len();
        let mut pool = Dsv4PrefixStatePool::new(2 * entry_bytes, entry_bytes);
        assert!(pool.publish(1, &small_entry(0))); // oldest, will be confirmed
        assert!(pool.publish(2, &small_entry(1))); // younger, provisional
        pool.confirm_pages(&[1]);
        pool.read_entry(1).expect("confirmed entry readable");
        // Pool full: publishing a third entry must evict the provisional page
        // 2, NOT the older-but-confirmed-and-hot page 1.
        assert!(pool.publish(3, &small_entry(2)));
        assert!(pool.page_meta(1).is_some_and(|m| m.confirmed));
        assert!(pool.page_meta(2).is_none(), "provisional evicted first");
        assert!(pool.page_meta(3).is_some());
    }

    /// #157 repair: canonical entry evicted → recompute publishes onto the
    /// slot's OWN page id → radix dedup keeps the canonical id → confirm-time
    /// adoption re-keys the provisional entry back to the canonical id, so the
    /// canonical chain is readable again (self-lock broken).
    #[test]
    fn adopt_canonical_rekeys_provisional_onto_canonical_chain() {
        let entry_bytes = small_entry(0).to_bytes().len();
        let mut pool = Dsv4PrefixStatePool::new(8 * entry_bytes, entry_bytes);
        // Canonical page 10 confirmed, then evicted (radix chain intact).
        assert!(pool.publish(10, &small_entry(0)));
        pool.confirm_pages(&[10]);
        pool.remove_pages(&[10]);
        assert!(pool.read_entry(10).is_err(), "canonical entry gone");
        // Recompute lands on the slot's own page 77 (provisional).
        assert!(pool.publish(77, &small_entry(0)));
        pool.adopt_canonical(10, 77);
        let back = pool.read_entry(10).expect("canonical chain repaired");
        assert_eq!(back.page_index, 0);
        assert!(back.boundary);
        assert!(pool.page_meta(77).is_none(), "own id re-keyed away");
        // Guard: an existing confirmed canonical entry is never clobbered.
        assert!(pool.publish(78, &small_entry(1)));
        pool.adopt_canonical(10, 78);
        assert_eq!(pool.read_entry(10).expect("kept").page_index, 0);
        // A provisional entry already under the canonical id confirms in place.
        assert!(pool.publish(20, &small_entry(3)));
        pool.adopt_canonical(20, 78);
        assert!(pool.page_meta(20).is_some_and(|m| m.confirmed));
        assert_eq!(pool.read_entry(20).expect("confirmed").page_index, 3);
    }
}

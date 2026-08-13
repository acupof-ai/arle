//! DSv4 cross-request prefix-state pool (#154 Phase 2 reland).
//!
//! One HOST-resident entry per (host page id): everything a restore at token
//! boundary `page_tokens·(k+1)` needs from every layer, D2H'd once when the
//! page completes. Content identity rides the host page id — the radix
//! dedupes prefix chains, so matching prefixes share host page ids and
//! non-matching content never collides (the D1 flaw is unrepresentable).
//! Zero HBM footprint: the pool IS the L2 tier (`KvTierStore` host level);
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
use kv_native_sys::{KvTierStore, tier_key};

/// DSv4 cross-request prefix-state entries (key = host page id) — the
/// content-keyed per-page pool (#154 Phase 2). Registry note: NS 1-4 (slot
/// park + sidecar) live in `executor.rs`.
const NS_PREFIX_STATE: u64 = 5;

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
    /// Frontier-tail sections (finish write-through, `--dsv4-decode-reuse`):
    /// the sub-page `tail` the radix match can't cover (`[matched_len,
    /// finish_len)`, < page_tokens). Present ONLY on the frontier entry when a
    /// finish landed off a page boundary; empty otherwise (aligned finishes and
    /// every non-frontier page). Captured after the finish forward, so the whole
    /// frontier carry (overlap/idx_overlap/ring/pending) reflects `finish_len`.
    ///
    /// `pending_kv/score`: the incomplete compress block's raw rows (`finish_len
    /// % ratio` tokens × width) — the next forward derives `pending_len =
    /// finish_len % ratio` and reads exactly these.
    pub(super) pending_kv: Vec<half::bf16>,
    pub(super) pending_score: Vec<half::bf16>,
    /// Indexer pending of the same tail (`finish_len % index_ratio` tokens ×
    /// indexer width) — #165: without it an off-ratio restore left the prior
    /// occupant's rows in the indexer's bf16 pending.
    pub(super) idx_pending_kv: Vec<half::bf16>,
    pub(super) idx_pending_score: Vec<half::bf16>,
    /// Completed compress rows of the tail page (`[matched_len/ratio,
    /// finish_len/ratio)`), restored past the frontier page's own rows.
    pub(super) tail_staging: Vec<half::bf16>,
    /// FP8 DSA key-cache rows of the tail page (`[matched_len/ir,
    /// finish_len/ir)`, no cache-page straddle: starts at a 16-row multiple,
    /// < 16 rows).
    pub(super) tail_dsa_data: Vec<u8>,
    pub(super) tail_dsa_scale: Vec<u8>,
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

// Magic doubles as format version: bumped on layout change so stale entries
// fail-close at the header instead of misparsing positional sections.
const ENTRY_MAGIC: &[u8; 4] = b"DSP2";

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
            push_bf16(&mut buf, &layer.pending_kv);
            push_bf16(&mut buf, &layer.pending_score);
            push_bf16(&mut buf, &layer.idx_pending_kv);
            push_bf16(&mut buf, &layer.idx_pending_score);
            push_bf16(&mut buf, &layer.tail_staging);
            push_bytes(&mut buf, &layer.tail_dsa_data);
            push_bytes(&mut buf, &layer.tail_dsa_scale);
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
                    pending_kv: read_bf16(&mut pos, bytes)?,
                    pending_score: read_bf16(&mut pos, bytes)?,
                    idx_pending_kv: read_bf16(&mut pos, bytes)?,
                    idx_pending_score: read_bf16(&mut pos, bytes)?,
                    tail_staging: read_bf16(&mut pos, bytes)?,
                    tail_dsa_data: read_bytes(&mut pos, bytes)?,
                    tail_dsa_scale: read_bytes(&mut pos, bytes)?,
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
                    + l.tail_dsa_data.len()
                    + l.tail_dsa_scale.len()
                    + (l.staging.len()
                        + l.overlap_kv.len()
                        + l.overlap_score.len()
                        + l.idx_overlap_kv.len()
                        + l.idx_overlap_score.len()
                        + l.ring.len()
                        + l.pending_kv.len()
                        + l.pending_score.len()
                        + l.idx_pending_kv.len()
                        + l.idx_pending_score.len()
                        + l.tail_staging.len())
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
    let mut total = 13usize; // entry header (magic + page_index + boundary + n_layers)
    for &(mode, ratio) in layer_specs {
        total += 15 * 4; // section length prefixes
        // ring — every layer has an SW window cache.
        total += config.sliding_window * config.head_dim * bf16;
        // ceil, not floor: for ratio > page_tokens a page can still complete
        // one row (page_row_span), and the predictor must never undersize.
        if mode.has_compressor() && ratio > 0 {
            let rpp = page_tokens.div_ceil(ratio);
            // staging + tail_staging (both ≤ one page of compress rows).
            total += 2 * rpp * config.head_dim * bf16;
            total += 2 * ratio * config.head_dim * bf16; // overlap kv+score
            // pending kv+score: `ratio·width`, width ≤ 2·head_dim (overlap).
            total += 2 * ratio * (2 * config.head_dim) * bf16;
        }
        if mode.has_indexer() {
            let index_ratio = if mode == DeepSeekV4AttentionMode::SparseIndexed {
                1
            } else {
                ratio.max(1)
            };
            let rpp = page_tokens.div_ceil(index_ratio);
            // dsa data+scale + tail dsa data+scale (both ≤ one page of rows).
            total += 2 * rpp * (config.index_head_dim + 4);
            total += 2 * index_ratio * config.index_head_dim * bf16; // idx overlap
            // idx pending kv+score: indexer is built with overlap=true, width
            // = 2·index_head_dim.
            total += 2 * index_ratio * (2 * config.index_head_dim) * bf16;
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
/// Storage rides ONE dedicated [`KvTierStore`] (host DRAM level = L2;
/// `set_disk` adds the mmap L3). `meta` is the exact host index — the store
/// never drops keys on its own, so the two cannot drift.
pub(crate) struct Dsv4PrefixStatePool {
    store: KvTierStore,
    meta: BTreeMap<u32, Dsv4PrefixPageMeta>,
    /// Over-cap eviction index `(confirmed, stamp, page id)`: provisional
    /// entries drop before confirmed ones (a confirmed boundary entry is the
    /// most valuable byte in the pool), LRU within each tier. Reads touch
    /// (restore traffic keeps a hot shared preamble resident); a FIFO here
    /// evicted the hottest first-published preamble under DRAM pressure and
    /// floor-0-locked the whole chain.
    lru: std::collections::BTreeSet<(bool, u64, u32)>,
    /// Frontier page id → the sub-page tail token ids `[matched_len, finish_len)`
    /// carried by that page's finish-write-through entry. The content-identity
    /// index: the radix proves identity only to the page boundary, so a restore
    /// that extends into the tail MUST verify these against the requesting
    /// prompt (a `&self` reusable scan can't `read_entry`, which is `&mut`, so
    /// the ids live here in memory, not just in the serialized entry).
    frontier_tails: BTreeMap<u32, Vec<u32>>,
    clock: u64,
    entry_bytes: usize,
    host_read_hits: u64,
    disk_read_hits: u64,
}

impl Dsv4PrefixStatePool {
    pub(crate) fn new(budget_bytes: usize, entry_bytes: usize) -> Self {
        Self {
            store: KvTierStore::with_budget(budget_bytes, entry_bytes.max(1)),
            meta: BTreeMap::new(),
            lru: std::collections::BTreeSet::new(),
            frontier_tails: BTreeMap::new(),
            clock: 0,
            entry_bytes: entry_bytes.max(1),
            host_read_hits: 0,
            disk_read_hits: 0,
        }
    }

    /// Pre-serve re-budget rebuilds the store, dropping every entry (same
    /// contract as the slot tier's re-budget).
    pub(crate) fn set_budget_bytes(&mut self, bytes: usize) {
        self.store = KvTierStore::with_budget(bytes, self.entry_bytes);
        self.meta.clear();
        self.lru.clear();
        self.frontier_tails.clear();
    }

    /// Record the finish-frontier tail token ids for `page_id` (empty ⇒ clear).
    /// Set by `capture_finish_frontier` after publishing the frontier entry.
    pub(crate) fn set_frontier_tail(&mut self, page_id: u32, tokens: Vec<u32>) {
        if tokens.is_empty() {
            self.frontier_tails.remove(&page_id);
        } else {
            self.frontier_tails.insert(page_id, tokens);
        }
    }

    /// The finish-frontier tail token ids for `page_id`, if it carries one.
    pub(crate) fn frontier_tail_tokens(&self, page_id: u32) -> Option<&[u32]> {
        self.frontier_tails.get(&page_id).map(Vec::as_slice)
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

    pub(crate) fn read_hits(&self) -> infer_seam::KvTierReadHits {
        infer_seam::KvTierReadHits {
            host_demoted: self.host_read_hits,
            disk: self.disk_read_hits,
        }
    }

    pub(crate) fn io_stats(&self) -> kv_native_sys::TierIoStats {
        self.store.io_stats()
    }

    /// Insert (LAST producer wins). Host page ids recycle when freed, so a
    /// republish under a recycled id MUST overwrite — a page a slot completes
    /// is either radix-shared (content-identical) or slot-exclusive, so the
    /// newest content is always the correct one. When both levels are full,
    /// drop oldest entries and retry (capacity never blocks the forward).
    /// Eviction skips `protected` (the publishing slot's chain) — else page k
    /// evicts its own provisional page k-1 and the chain never confirms.
    pub(crate) fn publish(
        &mut self,
        page_id: u32,
        entry: &Dsv4PrefixPageEntry,
        protected: &[u32],
    ) -> bool {
        let bytes = entry.to_bytes();
        let key = tier_key(NS_PREFIX_STATE, u64::from(page_id));
        let mut payload = bytes;
        loop {
            match self.store.insert(key, payload) {
                true => break,
                false => {
                    let Some(oldest) = self.pop_oldest_excluding(page_id, protected) else {
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
        let old = self.meta.get(&page_id).copied();
        let confirmed = old.is_some_and(|m| m.confirmed);
        if let Some(old) = old {
            let clears_tail = self.frontier_tails.contains_key(&page_id);
            if (old.boundary && !entry.boundary) || clears_tail {
                log::info!(
                    "prefix-pool republish page {page_id}: boundary {}->{} tail_cleared={clears_tail}",
                    old.boundary,
                    entry.boundary
                );
            }
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
        // A republish overwrites content; the finish tail (if any) is set
        // separately by `set_frontier_tail` right after — clear any stale one.
        self.frontier_tails.remove(&page_id);
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

    /// Fill a canonical id's frontier tail from the finishing slot's own id
    /// when the canonical lacks one (content-identical by construction) — never
    /// clobber an existing tail. Lets a finish tail attach to an already-present
    /// canonical entry across the `adopt_canonical` early returns (#159).
    fn adopt_frontier_tail(&mut self, canonical: u32, own: u32) {
        if !self.frontier_tails.contains_key(&canonical)
            && let Some(tail) = self.frontier_tails.get(&own).cloned()
        {
            self.frontier_tails.insert(canonical, tail);
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
            // Canonical entry already exists (confirmed or provisional). Carry
            // the finishing slot's frontier tail onto it — else a continuation
            // after radix dedup loses the sub-page tail reuse (#159). Content is
            // identical by construction; only fill a gap, never clobber.
            Some(m) if m.confirmed => {
                self.adopt_frontier_tail(canonical, own);
                return;
            }
            // Canonical id is radix-retained (never recycled while cached), so
            // a provisional entry under it holds the canonical content —
            // confirm in place.
            Some(_) => {
                self.adopt_frontier_tail(canonical, own);
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
        // Carry the frontier tail across the re-key (remove_pages drops own's).
        let own_tail = self.frontier_tails.get(&own).cloned();
        self.remove_pages(&[own]);
        let canonical_key = tier_key(NS_PREFIX_STATE, u64::from(canonical));
        if !self.store.insert(canonical_key, bytes) {
            return;
        }
        if let Some(tail) = own_tail {
            self.frontier_tails.insert(canonical, tail);
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

    fn pop_oldest_excluding(&self, publishing: u32, protected: &[u32]) -> Option<u32> {
        self.lru
            .iter()
            .map(|&(_, _, id)| id)
            .find(|&id| id != publishing && !protected.contains(&id))
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
        if promote {
            self.disk_read_hits = self.disk_read_hits.saturating_add(1);
        } else {
            self.host_read_hits = self.host_read_hits.saturating_add(1);
        }
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
                self.frontier_tails.remove(&p);
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
    /// Host vectors are stream-ordered; the caller fences after all layers.
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

    /// D2H the frontier-tail sections onto `out` (already holding this frontier
    /// page's own content+carry): the sub-page tail the radix match can't cover,
    /// `[matched_len, finish_len)`. `pending_kv/score` = the incomplete compress
    /// block (`finish_len % ratio` tokens); `tail_staging`/`tail_dsa_*` = the
    /// tail page's completed compress / DSA rows. The caller fences after all
    /// layers before reading them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_frontier_tail(
        &self,
        ctx: &DeviceContext,
        pool: &Dsv4LayerKvLayout,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        index_head_dim: usize,
        matched_len: usize,
        finish_len: usize,
        out: &mut Dsv4LayerPageState,
    ) -> Result<()> {
        if let Some(c) = &self.compressor {
            let ratio = compress_ratio.max(1);
            (out.pending_kv, out.pending_score) = capture_pending_tail(
                ctx,
                &c.pending_kv,
                &c.pending_score,
                ratio,
                finish_len,
                "pending",
            )?;
            if let Some(range) = staging_tail_range(c, ratio, matched_len, finish_len)? {
                out.tail_staging = ctx
                    .stream
                    .clone_dtoh(&c.compressed.data.slice(range))
                    .map_err(|e| anyhow!("DSv4 frontier tail staging D2H failed: {e}"))?;
            }
        }
        if let Some(ix) = &self.indexer {
            (out.idx_pending_kv, out.idx_pending_score) = capture_pending_tail(
                ctx,
                &ix.pending_kv,
                &ix.pending_score,
                dsa_index_ratio(mode, compress_ratio),
                finish_len,
                "idx pending",
            )?;
        }
        if let Some(dsa) = &self.dsa_official
            && let Some((data_range, scale_range)) = dsa_tail_ranges(
                pool,
                dsa,
                mode,
                compress_ratio,
                index_head_dim,
                matched_len,
                finish_len,
            )?
        {
            let cache = pool
                .dsa_key_cache
                .as_ref()
                .ok_or_else(|| anyhow!("DSv4 frontier tail: DSA shared key-cache missing"))?;
            ensure!(
                data_range.end <= cache.len() && scale_range.end <= cache.len(),
                "DSv4 frontier tail DSA range outside cache bytes {}",
                cache.len()
            );
            out.tail_dsa_data = ctx
                .stream
                .clone_dtoh(&cache.slice(data_range))
                .map_err(|e| anyhow!("DSv4 frontier tail DSA data D2H failed: {e}"))?;
            out.tail_dsa_scale = ctx
                .stream
                .clone_dtoh(&cache.slice(scale_range))
                .map_err(|e| anyhow!("DSv4 frontier tail DSA scale D2H failed: {e}"))?;
        }
        Ok(())
    }

    /// H2D the inverse of [`Self::capture_frontier_tail`]. Every section length
    /// is checked against the live tail shape before any byte moves; an absent
    /// section must be empty (else capture-era shape drift).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn restore_frontier_tail(
        &mut self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        index_head_dim: usize,
        matched_len: usize,
        finish_len: usize,
        state: &Dsv4LayerPageState,
    ) -> Result<()> {
        if let Some(c) = &mut self.compressor {
            let ratio = compress_ratio.max(1);
            restore_pending_tail(
                ctx,
                &mut c.pending_kv,
                &mut c.pending_score,
                ratio,
                finish_len,
                &state.pending_kv,
                &state.pending_score,
                "pending",
            )?;
            match staging_tail_range(c, ratio, matched_len, finish_len)? {
                Some(range) => {
                    ensure!(
                        state.tail_staging.len() == range.len(),
                        "DSv4 frontier tail restore staging {} != live {}",
                        state.tail_staging.len(),
                        range.len()
                    );
                    let mut view = c.compressed.data.slice_mut(range);
                    ctx.stream
                        .memcpy_htod(&state.tail_staging, &mut view)
                        .map_err(|e| anyhow!("DSv4 frontier tail staging H2D failed: {e}"))?;
                }
                None => ensure!(
                    state.tail_staging.is_empty(),
                    "DSv4 frontier tail restore: entry has staging for a tail the live shape completes none"
                ),
            }
        }
        if let Some(ix) = &mut self.indexer {
            restore_pending_tail(
                ctx,
                &mut ix.pending_kv,
                &mut ix.pending_score,
                dsa_index_ratio(mode, compress_ratio),
                finish_len,
                &state.idx_pending_kv,
                &state.idx_pending_score,
                "idx pending",
            )?;
        } else {
            ensure!(
                state.idx_pending_kv.is_empty() && state.idx_pending_score.is_empty(),
                "DSv4 frontier tail restore: entry has idx pending for a layer without an indexer"
            );
        }
        if let Some(dsa) = &self.dsa_official {
            match dsa_tail_ranges(
                pool,
                dsa,
                mode,
                compress_ratio,
                index_head_dim,
                matched_len,
                finish_len,
            )? {
                Some((data_range, scale_range)) => {
                    ensure!(
                        state.tail_dsa_data.len() == data_range.len()
                            && state.tail_dsa_scale.len() == scale_range.len(),
                        "DSv4 frontier tail restore DSA {}+{} != live {}+{}",
                        state.tail_dsa_data.len(),
                        state.tail_dsa_scale.len(),
                        data_range.len(),
                        scale_range.len()
                    );
                    let cache = pool.dsa_key_cache.as_mut().ok_or_else(|| {
                        anyhow!("DSv4 frontier tail restore: DSA key-cache missing")
                    })?;
                    ensure!(
                        data_range.end <= cache.len() && scale_range.end <= cache.len(),
                        "DSv4 frontier tail restore DSA range outside cache bytes {}",
                        cache.len()
                    );
                    {
                        let mut data = cache.slice_mut(data_range);
                        ctx.stream
                            .memcpy_htod(&state.tail_dsa_data, &mut data)
                            .map_err(|e| anyhow!("DSv4 frontier tail DSA data H2D failed: {e}"))?;
                    }
                    let mut scale = cache.slice_mut(scale_range);
                    ctx.stream
                        .memcpy_htod(&state.tail_dsa_scale, &mut scale)
                        .map_err(|e| anyhow!("DSv4 frontier tail DSA scale H2D failed: {e}"))?;
                }
                None => ensure!(
                    state.tail_dsa_data.is_empty() && state.tail_dsa_scale.is_empty(),
                    "DSv4 frontier tail restore: entry has DSA for a tail the live shape completes none"
                ),
            }
        }
        Ok(())
    }

    /// Host counters for a restore at `restore_len` (the frontier position:
    /// page-aligned `matched_len` for a plain restore, or `finish_len` for a
    /// finish write-through — the formula floors either way): staging/indexer
    /// row counts, DSA packed rows, and the FlashMLA FP8 counters.
    /// `fp8_kv_sw_bootstrapped=false` forces the SW-ring repack from the restored
    /// bf16 ring (A14); `fp8_kv_comp_packed_rows=0` forces the first post-restore
    /// decode's bulk pack to rebuild the FP8 band from the restored staging (the
    /// band itself is never captured — see [`Dsv4LayerPageState`]). `pending_kv/
    /// score` are NOT zeroed here: an off-`ratio` `restore_len` carries a
    /// sub-`ratio` tail that [`Self::restore_frontier_tail`] wrote and the next
    /// forward reads (`pending_len = restore_len % ratio`, `attention.rs:7638`).
    pub(crate) fn restore_prefix_counters(
        &mut self,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        restore_len: usize,
    ) {
        let comp_rows = if mode == DeepSeekV4AttentionMode::SlidingWindow {
            0
        } else {
            restore_len / compress_ratio.max(1)
        };
        let index_ratio = if mode == DeepSeekV4AttentionMode::SparseIndexed {
            1
        } else {
            compress_ratio.max(1)
        };
        let index_rows = if mode.has_indexer() {
            restore_len / index_ratio
        } else {
            0
        };
        // The restore rewrote the bf16 carry (overlap registers + pending tail)
        // while the FP32 probe carry still holds the previous occupant — the
        // next probe reseeds FP32 from the restored bf16 (cross-request
        // contamination otherwise).
        if let Some(c) = &mut self.compressor {
            c.compressed.seq_len = comp_rows;
            c.fp32_carry_stale = true;
        }
        if let Some(ix) = &mut self.indexer {
            ix.compressed.seq_len = index_rows;
            ix.fp32_carry_stale = true;
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

/// Element range of compressor staging rows `[row0, row0+count)` in
/// `compressed.data` (row-major `[row][head_dim]`).
fn staging_elem_range(
    c: &Dsv4CompressorState,
    row0: usize,
    count: usize,
) -> Result<std::ops::Range<usize>> {
    ensure!(
        c.ring_rows == c.compressed_capacity,
        "DSv4 prefix staging: main compressor staging must be full-history, not a ring"
    );
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
    Ok(row0 * head_dim..(row0 + count) * head_dim)
}

/// Element range of one host page's rows in the main compressor's bf16
/// staging; `None` when the page completes no row.
fn staging_row_range(
    c: &Dsv4CompressorState,
    compress_ratio: usize,
    page_tokens: usize,
    page_index: usize,
) -> Result<Option<std::ops::Range<usize>>> {
    ensure!(compress_ratio > 0, "DSv4 prefix staging: zero ratio");
    let (row0, count) = page_row_span(page_tokens, compress_ratio, page_index);
    if count == 0 {
        return Ok(None);
    }
    Ok(Some(staging_elem_range(c, row0, count)?))
}

/// Element range of the frontier tail's completed compress rows
/// `[matched_len/ratio, finish_len/ratio)`; `None` when the tail completes no
/// row (aligned finish or a sub-`ratio` tail).
fn staging_tail_range(
    c: &Dsv4CompressorState,
    compress_ratio: usize,
    matched_len: usize,
    finish_len: usize,
) -> Result<Option<std::ops::Range<usize>>> {
    ensure!(compress_ratio > 0, "DSv4 prefix staging tail: zero ratio");
    let row0 = matched_len / compress_ratio;
    let count = finish_len / compress_ratio - row0;
    if count == 0 {
        return Ok(None);
    }
    Ok(Some(staging_elem_range(c, row0, count)?))
}

/// Byte ranges of one host page's rows in the slot's FP8 DSA key-cache band
/// (`None` when the page completes no row): paged layout
/// `[64×index_head_dim data][64×f32 scales]` per page — the B1-B3 lesson:
/// never flat `row × (dim+4)` math.
#[allow(clippy::type_complexity)]
fn dsa_byte_ranges(
    pool: &Dsv4LayerKvLayout,
    dsa: &Dsv4DsaOfficialState,
    index_head_dim: usize,
    row0: usize,
    count: usize,
) -> Result<(std::ops::Range<usize>, std::ops::Range<usize>)> {
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
    Ok((
        data_start..data_start + count * index_head_dim,
        scale_start..scale_start + count * std::mem::size_of::<f32>(),
    ))
}

/// D2H the `finish_len % ratio` valid tail rows of a ratio-grouped pending
/// kv+score pair; `(empty, empty)` when the finish is on-ratio.
fn capture_pending_tail(
    ctx: &DeviceContext,
    kv: &CudaSlice<half::bf16>,
    score: &CudaSlice<half::bf16>,
    ratio: usize,
    finish_len: usize,
    label: &str,
) -> Result<(Vec<half::bf16>, Vec<half::bf16>)> {
    let pending_len = finish_len % ratio;
    if pending_len == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    ensure!(
        kv.len().is_multiple_of(ratio),
        "DSv4 frontier tail: {label} buffer {} not ratio-{ratio} divisible",
        kv.len()
    );
    let n = pending_len * (kv.len() / ratio);
    ensure!(
        n <= kv.len() && n <= score.len(),
        "DSv4 frontier tail: {label} rows {n} outside buffer {}",
        kv.len()
    );
    Ok((
        ctx.stream
            .clone_dtoh(&kv.slice(0..n))
            .map_err(|e| anyhow!("DSv4 frontier tail {label} kv D2H failed: {e}"))?,
        ctx.stream
            .clone_dtoh(&score.slice(0..n))
            .map_err(|e| anyhow!("DSv4 frontier tail {label} score D2H failed: {e}"))?,
    ))
}

/// H2D the inverse of [`capture_pending_tail`]; section lengths must equal the
/// live tail row count (0 for an on-ratio finish).
#[allow(clippy::too_many_arguments)]
fn restore_pending_tail(
    ctx: &DeviceContext,
    kv: &mut CudaSlice<half::bf16>,
    score: &mut CudaSlice<half::bf16>,
    ratio: usize,
    finish_len: usize,
    src_kv: &[half::bf16],
    src_score: &[half::bf16],
    label: &str,
) -> Result<()> {
    let pending_len = finish_len % ratio;
    let n = if pending_len > 0 {
        pending_len * (kv.len() / ratio)
    } else {
        0
    };
    ensure!(
        src_kv.len() == n && src_score.len() == n,
        "DSv4 frontier tail restore {label} {}+{} != live {n}",
        src_kv.len(),
        src_score.len()
    );
    if n == 0 {
        return Ok(());
    }
    let mut kv_view = kv.slice_mut(0..n);
    ctx.stream
        .memcpy_htod(src_kv, &mut kv_view)
        .map_err(|e| anyhow!("DSv4 frontier tail {label} kv H2D failed: {e}"))?;
    let mut score_view = score.slice_mut(0..n);
    ctx.stream
        .memcpy_htod(src_score, &mut score_view)
        .map_err(|e| anyhow!("DSv4 frontier tail {label} score H2D failed: {e}"))?;
    Ok(())
}

fn dsa_index_ratio(mode: DeepSeekV4AttentionMode, compress_ratio: usize) -> usize {
    if mode == DeepSeekV4AttentionMode::SparseIndexed {
        1
    } else {
        compress_ratio
    }
}

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
    let index_ratio = dsa_index_ratio(mode, compress_ratio);
    ensure!(index_ratio > 0, "DSv4 prefix DSA: zero index ratio");
    let (row0, count) = page_row_span(page_tokens, index_ratio, page_index);
    if count == 0 {
        return Ok(None);
    }
    Ok(Some(dsa_byte_ranges(
        pool,
        dsa,
        index_head_dim,
        row0,
        count,
    )?))
}

/// DSA byte ranges of the frontier tail rows `[matched_len/ir, finish_len/ir)`.
/// No cache-page straddle: `matched_len/ir` is a 16-row multiple and the tail
/// completes < 16 rows.
#[allow(clippy::type_complexity)]
fn dsa_tail_ranges(
    pool: &Dsv4LayerKvLayout,
    dsa: &Dsv4DsaOfficialState,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    index_head_dim: usize,
    matched_len: usize,
    finish_len: usize,
) -> Result<Option<(std::ops::Range<usize>, std::ops::Range<usize>)>> {
    let index_ratio = dsa_index_ratio(mode, compress_ratio);
    ensure!(index_ratio > 0, "DSv4 prefix DSA tail: zero index ratio");
    let row0 = matched_len / index_ratio;
    let count = finish_len / index_ratio - row0;
    if count == 0 {
        return Ok(None);
    }
    Ok(Some(dsa_byte_ranges(
        pool,
        dsa,
        index_head_dim,
        row0,
        count,
    )?))
}

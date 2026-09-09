//! Whole-slot and recurrent-sidecar tier lifecycle: the host half of KV
//! tiering. The backend keeps the device adapters (snapshot D2H/H2D, page
//! mirror, whole-slot swap); this module owns key derivation, boundary math,
//! chunking, eviction coordination, and the off-thread serialization.

use kv_native_sys::{BLOB_CHUNK_BYTES, KvTierStore, chunk_manifest, chunk_sub, tier_key};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;

/// Namespace keys for the tier store. `NS_SLOT` holds whole-slot images;
/// `NS_SIDECAR` holds recurrent-state blobs. The chunk namespaces carry each
/// blob's 16 MiB pieces. One store holds both, so the numbers must stay
/// disjoint.
pub const NS_SLOT: u64 = 1;
pub const NS_SLOT_CHUNK: u64 = 2;
pub const NS_SIDECAR: u64 = 3;
pub const NS_SIDECAR_CHUNK: u64 = 4;

/// FNV-1a hash of a token id slice, folded to the tier store's key space —
/// keys a sidecar blob to its prefix. The store packs the chunk index into the
/// low 16 bits and the namespace into the top 8 (`kv_native_sys::{chunk_sub,
/// tier_key}`), so a key must fit 40 bits.
pub fn hash_prefix_tokens(tokens: &[u32]) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut h = FNV_OFFSET;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h & ((1 << 40) - 1)
}

/// The L* boundary: the last page-aligned position strictly inside the
/// prompt — the exact-resend restore target.
pub fn sidecar_lstar(tokens_len: usize, page: usize) -> usize {
    tokens_len.saturating_sub(1) / page * page
}

/// Page-aligned matched length, clamped to the slot's materialized state and
/// the prompt. 0 when below one page.
pub fn sidecar_mat_len(
    matched_len: usize,
    slot_seq_len: usize,
    tokens_len: usize,
    page: usize,
) -> usize {
    matched_len.min(slot_seq_len).min(tokens_len) / page * page
}

/// A periodic snapshot is worth saving only inside the matched prefix:
/// position 0 carries no state, and a position past `mat_len` would never be
/// probed.
pub fn sidecar_periodic_savable(pos: usize, mat_len: usize) -> bool {
    pos != 0 && pos <= mat_len
}

/// Probe order for a sidecar restore: the exact matched length first, then
/// stride boundaries descending. Every save key is page-aligned, so
/// `hash(tokens[..b])` rendezvous with the probe at the boundaries.
///
/// `stride` must be > 0.
pub fn sidecar_candidates(matched_len: usize, stride: usize) -> Vec<usize> {
    debug_assert!(stride > 0);
    let mut candidates = Vec::new();
    if matched_len > 0 {
        candidates.push(matched_len);
    }
    let mut b = matched_len / stride * stride;
    while b >= stride {
        if b != matched_len {
            candidates.push(b);
        }
        b -= stride;
    }
    candidates
}

/// Map the store's raw IO stats to the seam type. Pure host — shared by the
/// Qwen3.6 slot tier and the DSv4 prefix+slot aggregate.
pub fn tier_io_stats(s: &kv_native_sys::TierIoStats) -> infer_seam::KvTierIoStats {
    infer_seam::KvTierIoStats {
        mode: match s.mode {
            kv_native_sys::DiskIoMode::Direct => infer_seam::KvTierIoMode::Direct,
            kv_native_sys::DiskIoMode::Mmap => infer_seam::KvTierIoMode::Mmap,
            _ => infer_seam::KvTierIoMode::Disabled,
        },
        useful_read_bytes: s.useful_read_bytes,
        useful_write_bytes: s.useful_write_bytes,
        submitted_read_bytes: s.submitted_read_bytes,
        submitted_write_bytes: s.submitted_write_bytes,
        metadata_write_bytes: s.metadata_write_bytes,
        failures: s.failures,
        completion_wait_ns: s.completion_wait_ns,
    }
}

/// A recurrent snapshot's serialization. Implemented by the backend's
/// snapshot type so the tier's background thread can serialize off the engine
/// step and the restore probe can skip a corrupt payload without naming the
/// model's type.
pub trait SidecarSnapshot: Send + 'static {
    fn serialize(self) -> Vec<u8>;
    fn deserialize(bytes: &[u8]) -> Option<Self>
    where
        Self: Sized;
}

/// A blob whose `serialize()` + chunking finished on the background thread.
struct SidecarBlob {
    pos: usize,
    key: u64,
    chunks: usize,
    entries: Vec<(u64, Vec<u8>)>,
    prefix_pages: Vec<u32>,
}

/// A batch of snapshots to serialize on the background thread.
struct SidecarWork<S> {
    items: Vec<(usize, u64, S)>,
    prefix_pages: Vec<u32>,
}

/// The executor's whole-slot and sidecar tier: one [`KvTierStore`] plus the
/// sidecar eviction map and the serialization thread.
///
/// Whole-slot images (demote/promote) live under `NS_SLOT`; recurrent sidecar
/// blobs live under `NS_SIDECAR`, each keyed by `hash_prefix_tokens` at its
/// boundary. A sidecar's lifetime rides the radix blocks: the tail page of
/// the prefix it covers owns its eviction.
pub struct KvSlotTier<S: SidecarSnapshot> {
    store: KvTierStore,
    page: usize,
    /// Tail host-pool page of each published prefix -> its sidecar key.
    sidecar_page_key: HashMap<u32, u64>,
    sidecar_sync: bool,
    sidecar_work_tx: mpsc::Sender<SidecarWork<S>>,
    sidecar_rx: mpsc::Receiver<SidecarBlob>,
}

/// Split a serialized snapshot into a manifest entry plus 16 MiB chunk
/// entries, keyed under the sidecar namespace.
fn build_blob_entries(key: u64, bytes: Vec<u8>) -> (usize, Vec<(u64, Vec<u8>)>) {
    let chunks = bytes.len().div_ceil(BLOB_CHUNK_BYTES);
    let manifest_key = tier_key(NS_SIDECAR, key);
    let manifest = chunk_manifest(chunks, bytes.len());
    let mut entries = Vec::with_capacity(chunks + 1);
    entries.push((manifest_key, manifest));
    entries.extend(
        bytes
            .chunks(BLOB_CHUNK_BYTES)
            .enumerate()
            .map(|(idx, chunk)| {
                (
                    tier_key(NS_SIDECAR_CHUNK, chunk_sub(key, idx)),
                    chunk.to_vec(),
                )
            }),
    );
    (chunks, entries)
}

impl<S: SidecarSnapshot> KvSlotTier<S> {
    /// `budget_bytes` is the per-rank L2 cap. `sync` serializes inline
    /// (A/B control); otherwise a dedicated thread does it.
    pub fn new(budget_bytes: usize, page: usize, sync: bool) -> Self {
        let (blob_tx, blob_rx) = mpsc::channel::<SidecarBlob>();
        let (work_tx, work_rx) = mpsc::channel::<SidecarWork<S>>();
        if !sync {
            // One thread (not spawn-per-call) bounds memory-bandwidth
            // contention at c=1.
            std::thread::spawn(move || {
                while let Ok(work) = work_rx.recv() {
                    for (pos, key, snap) in work.items {
                        let (chunks, entries) = build_blob_entries(key, snap.serialize());
                        if blob_tx
                            .send(SidecarBlob {
                                pos,
                                key,
                                chunks,
                                entries,
                                prefix_pages: work.prefix_pages.clone(),
                            })
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            });
        }
        Self {
            store: KvTierStore::with_budget(budget_bytes, BLOB_CHUNK_BYTES),
            page,
            sidecar_page_key: HashMap::new(),
            sidecar_sync: sync,
            sidecar_work_tx: work_tx,
            sidecar_rx: blob_rx,
        }
    }

    /// Pre-serve only: replace the store with one at the new cap.
    pub fn set_budget(&mut self, bytes: usize) {
        self.store = KvTierStore::with_budget(bytes, BLOB_CHUNK_BYTES);
    }

    /// Pre-serve only. The budget is a per-store cap, not a reservation —
    /// the store is a sparse mmap, so disk is consumed only by actual spill.
    pub fn set_disk(&mut self, root: PathBuf, budget_bytes: usize) -> bool {
        self.store.set_disk(root, budget_bytes, BLOB_CHUNK_BYTES)
    }

    pub fn host_demoted_pages(&self) -> usize {
        self.store.host_demoted_pages()
    }

    pub fn disk_pages(&self) -> usize {
        self.store.disk_pages()
    }

    pub fn io_stats(&self) -> infer_seam::KvTierIoStats {
        tier_io_stats(&self.store.io_stats())
    }

    pub fn read_hits(&self) -> infer_seam::KvTierReadHits {
        self.store.read_hits()
    }

    pub fn location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        self.store.location(key)
    }

    /// Store a whole-slot image. Returns false when the tier is at budget.
    pub fn store_image(&mut self, key: u64, bytes: &[u8]) -> bool {
        self.store
            .insert_chunked(NS_SLOT, NS_SLOT_CHUNK, key, bytes)
    }

    pub fn load_image(&mut self, key: u64) -> anyhow::Result<Vec<u8>> {
        self.store
            .read_chunked(NS_SLOT, NS_SLOT_CHUNK, key)
            .map_err(|e| anyhow::anyhow!("whole-slot tier read key {key}: {e}"))
    }

    pub fn remove_image(&mut self, key: u64) {
        self.store.remove_chunked(NS_SLOT, NS_SLOT_CHUNK, key);
    }

    pub fn drop_images(&mut self, keys: &[u64]) {
        for &key in keys {
            self.remove_image(key);
        }
    }

    /// Insert a sidecar blob and coordinate its eviction off the last radix
    /// page it covers: leaves evict deepest-first, so the blob drops as its
    /// own prefix erodes.
    fn store_sidecar_blob(
        &mut self,
        pos: usize,
        key: u64,
        chunks: usize,
        entries: Vec<(u64, Vec<u8>)>,
        prefix_pages: &[u32],
    ) {
        if !self
            .store
            .insert_prechunked(NS_SIDECAR, key, chunks, entries)
        {
            return;
        }
        let cover_idx = (pos / self.page).saturating_sub(1);
        if let Some(&tail) = prefix_pages.get(cover_idx).or_else(|| prefix_pages.last())
            && let Some(old) = self.sidecar_page_key.insert(tail, key)
            && old != key
        {
            self.store.remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, old);
        }
    }

    /// Serialize and store every snapshot, inline or on the background thread
    /// per the `sync` construction flag.
    pub fn submit_sidecar(&mut self, items: Vec<(usize, u64, S)>, prefix_pages: &[u32]) {
        if items.is_empty() {
            return;
        }
        if self.sidecar_sync {
            for (pos, key, snap) in items {
                let (chunks, entries) = build_blob_entries(key, snap.serialize());
                self.store_sidecar_blob(pos, key, chunks, entries, prefix_pages);
            }
            return;
        }
        let _ = self.sidecar_work_tx.send(SidecarWork {
            items,
            prefix_pages: prefix_pages.to_vec(),
        });
    }

    /// Drain completed background serializations into the store.
    pub fn poll_sidecar(&mut self) {
        while let Ok(blob) = self.sidecar_rx.try_recv() {
            self.store_sidecar_blob(
                blob.pos,
                blob.key,
                blob.chunks,
                blob.entries,
                &blob.prefix_pages,
            );
        }
    }

    /// Drop sidecar blobs keyed to evicted radix pages — eviction rides the
    /// radix, no independent sidecar LRU.
    pub fn release_sidecar_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            if let Some(key) = self.sidecar_page_key.remove(&page) {
                self.store.remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key);
            }
        }
    }

    /// Drop every tracked sidecar blob — the weight epoch changed, so a
    /// skipped capture must never serve old-epoch state.
    pub fn drop_all_sidecar(&mut self) {
        for (_, key) in self.sidecar_page_key.drain() {
            self.store.remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key);
        }
    }

    /// Largest boundary whose sidecar is present and deserializes, probing
    /// [`sidecar_candidates`] order. A corrupt/foreign payload is skipped, so
    /// the next candidate still has its chance.
    pub fn probe_sidecar(
        &mut self,
        tokens: &[u32],
        matched_len: usize,
        stride: usize,
    ) -> Option<(usize, S)> {
        sidecar_candidates(matched_len, stride)
            .into_iter()
            .find_map(|b| {
                let key = hash_prefix_tokens(&tokens[..b]);
                self.store
                    .read_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key)
                    .ok()
                    .and_then(|bytes| S::deserialize(&bytes))
                    .map(|snap| (b, snap))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 16;
    const STRIDE: usize = 1024;

    /// A snapshot that fails to deserialize empty bytes, so the corrupt-blob
    /// skip path is exercisable.
    #[derive(Debug, PartialEq)]
    struct TestSnap(Vec<u8>);

    impl SidecarSnapshot for TestSnap {
        fn serialize(self) -> Vec<u8> {
            self.0
        }
        fn deserialize(bytes: &[u8]) -> Option<Self> {
            (!bytes.is_empty()).then(|| TestSnap(bytes.to_vec()))
        }
    }

    fn tier() -> KvSlotTier<TestSnap> {
        KvSlotTier::new(1 << 30, PAGE, true)
    }

    #[test]
    fn boundary_math_aligns_and_orders() {
        assert_eq!(sidecar_lstar(256, PAGE), 240);
        assert_eq!(sidecar_lstar(1, PAGE), 0);
        assert_eq!(sidecar_mat_len(1000, 500, 9999, PAGE), 496);
        assert_eq!(sidecar_mat_len(31, 9999, 9999, PAGE), 16);
        assert_eq!(sidecar_mat_len(15, 9999, 9999, PAGE), 0);
        assert!(sidecar_periodic_savable(512, 1024));
        assert!(!sidecar_periodic_savable(0, 1024));
        assert!(!sidecar_periodic_savable(1025, 1024));
        // matched first, then stride boundaries descending, no duplicates.
        assert_eq!(sidecar_candidates(3000, STRIDE), vec![3000, 2048, 1024]);
        assert_eq!(sidecar_candidates(2048, STRIDE), vec![2048, 1024]);
        assert_eq!(sidecar_candidates(100, STRIDE), vec![100]);
    }

    #[test]
    fn slot_image_round_trips_and_drops() {
        let mut t = tier();
        assert!(t.store_image(7, b"slot-image"));
        assert_eq!(t.load_image(7).unwrap(), b"slot-image");
        t.remove_image(7);
        assert!(t.load_image(7).is_err());
        t.store_image(8, b"a");
        t.store_image(9, b"b");
        t.drop_images(&[8, 9]);
        assert!(t.load_image(8).is_err() && t.load_image(9).is_err());
    }

    #[test]
    fn sidecar_eviction_rides_the_tail_page() {
        let mut t = tier();
        let tokens: Vec<u32> = (0..1024).collect();
        let key = hash_prefix_tokens(&tokens[..1024]);
        // pos 1024 covers page idx 63; the blob rides page 64's predecessor.
        t.submit_sidecar(vec![(1024, key, TestSnap(b"snap".to_vec()))], &[10, 11, 63]);
        assert!(t.probe_sidecar(&tokens, 1024, STRIDE).is_some());
        // Evicting an unrelated page keeps the blob; evicting the tail drops it.
        t.release_sidecar_pages(&[11]);
        assert!(t.probe_sidecar(&tokens, 1024, STRIDE).is_some());
        t.release_sidecar_pages(&[63]);
        assert!(t.probe_sidecar(&tokens, 1024, STRIDE).is_none());
    }

    #[test]
    fn probe_skips_corrupt_and_falls_to_the_next_boundary() {
        let mut t = tier();
        let tokens: Vec<u32> = (0..3000).collect();
        // Corrupt blob at the exact matched length (empty bytes fail
        // deserialize); a valid one at the stride boundary below it.
        t.submit_sidecar(
            vec![
                (
                    3000,
                    hash_prefix_tokens(&tokens[..3000]),
                    TestSnap(Vec::new()),
                ),
                (
                    2048,
                    hash_prefix_tokens(&tokens[..2048]),
                    TestSnap(b"ok".to_vec()),
                ),
            ],
            &[187],
        );
        let (boundary, snap) = t.probe_sidecar(&tokens, 3000, STRIDE).unwrap();
        assert_eq!(boundary, 2048);
        assert_eq!(snap, TestSnap(b"ok".to_vec()));
    }

    #[test]
    fn async_submit_drains_through_poll() {
        let mut t = KvSlotTier::new(1 << 30, PAGE, false);
        let tokens: Vec<u32> = (0..1024).collect();
        let key = hash_prefix_tokens(&tokens[..1024]);
        t.submit_sidecar(vec![(1024, key, TestSnap(b"snap".to_vec()))], &[63]);
        // Not visible until the thread's blob is polled.
        for _ in 0..100 {
            t.poll_sidecar();
            if t.probe_sidecar(&tokens, 1024, STRIDE).is_some() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("async sidecar never became visible after poll");
    }

    #[test]
    fn drop_all_sidecar_clears_every_blob() {
        let mut t = tier();
        let tokens: Vec<u32> = (0..2048).collect();
        t.submit_sidecar(
            vec![
                (
                    1024,
                    hash_prefix_tokens(&tokens[..1024]),
                    TestSnap(b"a".to_vec()),
                ),
                (
                    2048,
                    hash_prefix_tokens(&tokens[..2048]),
                    TestSnap(b"b".to_vec()),
                ),
            ],
            &[63, 127],
        );
        t.drop_all_sidecar();
        assert!(t.probe_sidecar(&tokens, 2048, STRIDE).is_none());
    }
}

//! Two-level host store for demoted KV blocks.
//!
//! Host-demoted blocks live in a capacity-capped in-RAM map (default-on, 4 GiB).
//! Disk spill is optional on the `kv-native-sys` block substrate
//! (`--kv-disk`, opt-in): when host RAM fills, the coldest host entry spills
//! to a file-backed mmap page-slot store, so the capacity the engine sees is
//! host-demoted + disk slots. Payloads are opaque fixed-limit blocks (paged-KV
//! pages or DSv4 slot-image chunks); this module never touches the device.
//!
//! ## Mmap store
//!
//! The disk tier uses [`kv_native_sys::KvMmapStore`] — one file per namespace,
//! fixed-size page slots. Writes memcpy into the mapping (no per-page syscall);
//! reads return `&[u8]` slices directly from the mapping (zero-copy). This
//! replaces the prior sharded per-page block-file approach (~4 ms/page) with
//! a single mmap (~0.05 ms/page write, sub-μs read).
//!
//! ## Durability (recall tier)
//!
//! The prefix tier uses a per-process *ephemeral* disk namespace that is wiped
//! on drop (a crash-safe cache the engine can always rebuild). The session
//! KV-recall tier instead opts into a **durable** disk namespace ([`set_disk`]
//! with `durable = true`): the namespace is stable across restarts, survives
//! drop, and carries a [`MANIFEST_FILE`] persisting each disk-resident block's
//! `{key, slot_idx}` plus an epoch tag. [`CudaKvTierStore::load`] replays the
//! manifest so a prior session's evicted KV is addressable again; a stale epoch
//! (e.g. after an OPD weight update) discards the prior memory.
//!
//! [`set_disk`]: CudaKvTierStore::set_disk

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use infer_seam::{DramTierPolicy, NvmeTierPolicy, dram_l2_budget, nvme_l3_budget};

/// Manifest filename under a durable disk namespace. Records the epoch tag plus
/// one `key slot_idx len slot_bytes` line per disk-resident block so a restart
/// can rebuild the in-memory disk index and replay slot allocations.
const MANIFEST_FILE: &str = "manifest.kvm";

/// Manifest header magic + version. A mismatch — or a missing manifest — makes
/// [`CudaKvTierStore::load`] start cold rather than trust a foreign layout.
const MANIFEST_MAGIC: &str = "ARLE-KVTIER-MANIFEST-V2";

static DISK_TIER_NAMESPACE_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

// ---- OS probes (unchanged) ----

fn meminfo_field_bytes(field: &str) -> Option<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let prefix = format!("{field}:");
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix(&prefix) {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

fn available_ram_bytes() -> Option<usize> {
    meminfo_field_bytes("MemAvailable")
}

fn total_ram_bytes() -> Option<usize> {
    meminfo_field_bytes("MemTotal")
}

#[cfg(unix)]
fn disk_free_total_bytes(path: &Path) -> Option<(usize, usize)> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let stat = unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut stat) != 0 {
            return None;
        }
        stat
    };
    let frsize = u128::from(stat.f_frsize);
    let free = usize::try_from(frsize.saturating_mul(u128::from(stat.f_bavail))).ok()?;
    let total = usize::try_from(frsize.saturating_mul(u128::from(stat.f_blocks))).ok()?;
    Some((free, total))
}

#[cfg(not(unix))]
fn disk_free_total_bytes(_path: &Path) -> Option<(usize, usize)> {
    None
}

// ---- Budget helpers (unchanged) ----

pub(crate) fn default_t1_budget_bytes(dram_fraction: f64) -> usize {
    let budget = dram_l2_budget(
        available_ram_bytes(),
        total_ram_bytes(),
        DramTierPolicy {
            fraction: dram_fraction,
            ..DramTierPolicy::default()
        },
    );
    log::info!(
        "L2 (host DRAM) KV tier budget {} MiB (dram_fraction {dram_fraction}, MemAvailable {} MiB, \
         MemTotal {} MiB)",
        budget / (1 << 20),
        available_ram_bytes().map_or(0, |b| b / (1 << 20)),
        total_ram_bytes().map_or(0, |b| b / (1 << 20)),
    );
    budget
}

/// Per-rank L2 budget from a deployment-total request. `Off` → 0 (tier
/// disabled); `Fraction` resolves against measured MemAvailable via
/// `default_t1_budget_bytes` before the world split.
pub fn resolve_dram_budget_bytes(budget: infer_seam::KvTierBudget, world: usize) -> usize {
    let world = world.max(1);
    match budget {
        infer_seam::KvTierBudget::Off => 0,
        infer_seam::KvTierBudget::Bytes(b) => b / world,
        infer_seam::KvTierBudget::Fraction(f) => default_t1_budget_bytes(f) / world,
    }
}

pub fn default_t2_budget_bytes(root: &Path, ssd_fraction: f64) -> usize {
    let (free, total) = disk_free_total_bytes(root).unzip();
    let budget = nvme_l3_budget(
        free,
        total,
        NvmeTierPolicy {
            fraction: ssd_fraction,
            ..NvmeTierPolicy::default()
        },
    );
    log::info!(
        "L3 (NVMe) KV tier budget {} MiB at {} (ssd_fraction {ssd_fraction}, free_disk {} MiB, \
         total_disk {} MiB)",
        budget / (1 << 20),
        root.display(),
        free.map_or(0, |b| b / (1 << 20)),
        total.map_or(0, |b| b / (1 << 20)),
    );
    budget
}

// ---- chunked-blob key layout ----

/// One u64 key space partitioned by a top-byte namespace so features sharing a
/// store never collide; callers own the namespace constants. Manifest keys
/// carry the feature key in the low bits; chunk keys carry
/// `key << CHUNK_IDX_BITS | idx` (a blob is ≤ 2^16 × page_bytes).
pub(crate) const TIER_NS_SHIFT: u32 = 56;
pub(crate) const CHUNK_IDX_BITS: u32 = 16;
/// Canonical chunk/page size for whole-slot blob stores (DSv4 + Qwen3.6).
pub(crate) const BLOB_CHUNK_BYTES: usize = 16 << 20;
/// DSv4 cross-request prefix-state entries (key = host page id) — the
/// content-keyed per-page pool (#154 Phase 2, `attention/prefix_state.rs`).
/// Registry note: NS 1-4 (slot park + sidecar) live in `executor.rs`.
#[allow(dead_code)] // wired by the publish executor commit in this series
pub(crate) const NS_PREFIX_STATE: u64 = 5;

pub(crate) fn tier_key(ns: u64, sub: u64) -> u64 {
    debug_assert!(
        sub >> TIER_NS_SHIFT == 0,
        "tier sub-key overflows namespace"
    );
    (ns << TIER_NS_SHIFT) | sub
}

pub(crate) fn chunk_sub(key: u64, idx: usize) -> u64 {
    (key << CHUNK_IDX_BITS) | idx as u64
}

fn chunk_manifest(chunks: usize, bytes: usize) -> Vec<u8> {
    format!("DSCHUNK {chunks} {bytes}\n").into_bytes()
}

fn parse_chunk_manifest(bytes: &[u8]) -> Result<(usize, usize)> {
    let text =
        std::str::from_utf8(bytes).map_err(|e| anyhow!("chunked-blob manifest utf8: {e}"))?;
    let mut fields = text.split_whitespace();
    anyhow::ensure!(
        fields.next() == Some("DSCHUNK"),
        "bad chunked-blob manifest"
    );
    let chunks = fields
        .next()
        .ok_or_else(|| anyhow!("chunked-blob manifest missing chunks"))?
        .parse()
        .map_err(|e| anyhow!("chunked-blob manifest chunks: {e}"))?;
    let bytes = fields
        .next()
        .ok_or_else(|| anyhow!("chunked-blob manifest missing bytes"))?
        .parse()
        .map_err(|e| anyhow!("chunked-blob manifest bytes: {e}"))?;
    Ok((chunks, bytes))
}

// ---- CudaKvTierStore ----

pub(crate) struct CudaKvTierStore {
    host_capacity_pages: usize,
    bytes_per_page: usize,
    host: BTreeMap<u64, HostDemotedEntry>,
    host_lru: BTreeSet<(u64, u64)>,
    clock: u64,
    disk: Option<DiskTier>,
}

struct HostDemotedEntry {
    stamp: u64,
    payload: Vec<u8>,
}

// ---- DiskTier (mmap-backed) ----

struct DiskTier {
    /// Namespace directory (contains `kv.mmap` + `manifest.kvm`).
    root_dir: PathBuf,
    /// Fixed-size page-slot mmap store.
    store: kv_native_sys::KvMmapStore,
    /// Key -> slot index plus valid payload length in the mmap store.
    keys: BTreeMap<u64, DiskRecord>,
    /// Durable tier: suppress drop-time wipe, persist manifest on mutation.
    durable: bool,
    /// Model/weights-version tag written into the manifest.
    epoch: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DiskRecord {
    slot: u32,
    len: usize,
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        if !self.durable {
            // KvMmapStore drops first (field order), unmapping the file.
            // Linux tolerates unlinking a still-open file, so best-effort.
            let _ = std::fs::remove_dir_all(&self.root_dir);
        }
    }
}

impl DiskTier {
    #[allow(dead_code)]
    fn mmap_path(&self) -> PathBuf {
        self.root_dir.join("kv.mmap")
    }

    fn manifest_path(&self) -> PathBuf {
        self.root_dir.join(MANIFEST_FILE)
    }

    fn write_manifest(&self) -> Result<()> {
        let mut buf = String::with_capacity(64 + self.keys.len() * 32);
        buf.push_str(MANIFEST_MAGIC);
        buf.push('\n');
        buf.push_str(&self.epoch);
        buf.push('\n');
        let slot_bytes = self.store.slot_bytes();
        for (key, record) in &self.keys {
            writeln!(
                &mut buf,
                "{key} {} {} {slot_bytes}",
                record.slot, record.len
            )
            .expect("write to String never fails");
        }
        kv_native_sys::write_file_atomic_cache(&self.manifest_path(), buf.as_bytes())
            .with_context(|| format!("KV recall manifest write under {}", self.root_dir.display()))
    }

    fn parse_manifest(bytes: &[u8]) -> Option<(String, Vec<(u64, DiskRecord)>)> {
        // Accept current four-field records plus the earlier three-field V2
        // records (len=0 means the full mmap slot is valid).
        let text = std::str::from_utf8(bytes).ok()?;
        let mut lines = text.lines();
        let magic = lines.next()?;
        if magic != MANIFEST_MAGIC {
            return None;
        }
        let epoch = lines.next()?.to_string();
        let records: Vec<(u64, DiskRecord)> = lines
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() {
                    return None;
                }
                let mut parts = line.split_whitespace();
                let key: u64 = parts.next()?.parse().ok()?;
                let slot_idx: u32 = parts.next()?.parse().ok()?;
                let third = parts.next()?;
                let fourth = parts.next();
                let (len, _slot_bytes) = match fourth {
                    Some(slot_bytes) => (
                        third.parse::<usize>().ok()?,
                        slot_bytes.parse::<u32>().ok().unwrap_or(0),
                    ),
                    None => (0, third.parse::<u32>().ok().unwrap_or(0)),
                };
                Some((
                    key,
                    DiskRecord {
                        slot: slot_idx,
                        len,
                    },
                ))
            })
            .collect();
        Some((epoch, records))
    }
}

// ---- CudaKvTierStore impl ----

impl Drop for CudaKvTierStore {
    fn drop(&mut self) {
        self.persist();
    }
}

impl CudaKvTierStore {
    pub(crate) fn with_budget(budget_bytes: usize, bytes_per_page: usize) -> Self {
        let host_capacity_pages = budget_bytes.checked_div(bytes_per_page).unwrap_or(0);
        Self {
            host_capacity_pages,
            bytes_per_page,
            host: BTreeMap::new(),
            host_lru: BTreeSet::new(),
            clock: 0,
            disk: None,
        }
    }

    pub(crate) fn set_disk(
        &mut self,
        root: PathBuf,
        budget_bytes: usize,
        bytes_per_page: usize,
    ) -> bool {
        let namespace = self.ephemeral_namespace(root);
        self.attach_disk(
            namespace,
            budget_bytes,
            bytes_per_page,
            false,
            String::new(),
        )
    }

    pub(crate) fn set_disk_durable(
        &mut self,
        root: PathBuf,
        budget_bytes: usize,
        bytes_per_page: usize,
        epoch: String,
    ) -> bool {
        let namespace = Self::durable_namespace(root);
        self.attach_disk(namespace, budget_bytes, bytes_per_page, true, epoch)
    }

    fn attach_disk(
        &mut self,
        namespace: PathBuf,
        budget_bytes: usize,
        bytes_per_page: usize,
        durable: bool,
        epoch: String,
    ) -> bool {
        let capacity_pages = budget_bytes.checked_div(bytes_per_page).unwrap_or(0);
        if capacity_pages == 0 {
            log::warn!(
                "KV disk mmap store: zero capacity (budget {budget_bytes} / page {bytes_per_page})"
            );
            return false;
        }
        let mmap_path = namespace.join("kv.mmap");
        if let Err(err) = std::fs::create_dir_all(&namespace) {
            log::warn!(
                "KV disk namespace creation failed under {}: {err}",
                namespace.display()
            );
            return false;
        }
        match kv_native_sys::KvMmapStore::create(&mmap_path, capacity_pages, bytes_per_page) {
            Ok(store) => {
                self.disk = Some(DiskTier {
                    root_dir: namespace,
                    store,
                    keys: BTreeMap::new(),
                    durable,
                    epoch,
                });
                true
            }
            Err(err) => {
                log::warn!(
                    "KV disk mmap store creation failed at {}: {err}",
                    mmap_path.display()
                );
                false
            }
        }
    }

    pub(crate) fn load(
        &mut self,
        root: PathBuf,
        budget_bytes: usize,
        bytes_per_page: usize,
        epoch: String,
    ) -> bool {
        let namespace = Self::durable_namespace(root.clone());
        let manifest_path = namespace.join(MANIFEST_FILE);
        let mmap_path = namespace.join("kv.mmap");
        let capacity_pages = budget_bytes.checked_div(bytes_per_page).unwrap_or(0);
        if capacity_pages == 0 {
            return false;
        }

        // Always (re)attach the durable disk level first.
        if let Err(err) = std::fs::create_dir_all(&namespace) {
            log::warn!("KV durable namespace creation failed: {err}");
            return false;
        }

        // Try to open existing mmap store.
        let mut store =
            match kv_native_sys::KvMmapStore::open(&mmap_path, capacity_pages, bytes_per_page) {
                Ok(s) => s,
                Err(_) => {
                    // No existing store — create fresh.
                    match kv_native_sys::KvMmapStore::create(
                        &mmap_path,
                        capacity_pages,
                        bytes_per_page,
                    ) {
                        Ok(s) => s,
                        Err(err) => {
                            log::warn!("KV mmap store create failed: {err}");
                            return false;
                        }
                    }
                }
            };

        let parsed = std::fs::read(&manifest_path)
            .ok()
            .and_then(|bytes| DiskTier::parse_manifest(&bytes));

        let mut keys = BTreeMap::new();
        if let Some((manifest_epoch, records)) = parsed {
            if manifest_epoch != epoch {
                log::warn!(
                    "KV recall manifest epoch mismatch under {} (manifest={manifest_epoch:?}, \
                     requested={epoch:?}); discarding stale memory",
                    namespace.display()
                );
            } else {
                let mut indices = Vec::with_capacity(records.len());
                for (key, record) in &records {
                    if keys.len() >= capacity_pages {
                        break;
                    }
                    if (record.slot as usize) < capacity_pages {
                        let len = record.len.min(bytes_per_page);
                        keys.insert(
                            *key,
                            DiskRecord {
                                slot: record.slot,
                                len,
                            },
                        );
                        indices.push(record.slot);
                    }
                }
                store.reserve_indices(&indices);
            }
        }

        self.disk = Some(DiskTier {
            root_dir: namespace,
            store,
            keys,
            durable: true,
            epoch,
        });
        true
    }

    pub(crate) fn persist(&self) {
        if let Some(disk) = self.disk.as_ref() {
            if disk.durable {
                if let Err(err) = disk.write_manifest() {
                    log::warn!("KV recall manifest persist failed: {err}");
                }
            }
        }
    }

    fn ephemeral_namespace(&mut self, root: PathBuf) -> PathBuf {
        let counter =
            DISK_TIER_NAMESPACE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        root.join(format!(
            "arle-kv-tier-{}-{nanos}-{counter}",
            std::process::id()
        ))
    }

    fn durable_namespace(root: PathBuf) -> PathBuf {
        root.join(format!("arle-kv-recall-{}", std::process::id()))
    }

    pub(crate) fn capacity_pages(&self) -> usize {
        self.host_capacity_pages.saturating_add(
            self.disk
                .as_ref()
                .map_or(0, |d| d.store.num_slots() as usize),
        )
    }

    pub(crate) fn available_pages(&self) -> usize {
        self.capacity_pages()
            .saturating_sub(self.host.len().saturating_add(self.disk_pages()))
    }

    pub(crate) fn page_bytes(&self) -> usize {
        self.bytes_per_page
    }

    pub(crate) fn host_demoted_pages(&self) -> usize {
        self.host.len()
    }

    pub(crate) fn disk_pages(&self) -> usize {
        self.disk.as_ref().map_or(0, |d| d.keys.len())
    }

    pub(crate) fn location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        if self.host.contains_key(&key) {
            return Some(infer_seam::KvTierLocation::HostDemoted);
        }
        self.disk.as_ref().and_then(|disk| {
            disk.keys
                .contains_key(&key)
                .then_some(infer_seam::KvTierLocation::Disk)
        })
    }

    pub(crate) fn is_full(&self) -> bool {
        let host_full = self.host.len() >= self.host_capacity_pages;
        let disk_full = self
            .disk
            .as_ref()
            .is_none_or(|d| d.keys.len() >= d.store.num_slots() as usize);
        host_full && disk_full
    }

    pub(crate) fn contains(&self, key: u64) -> bool {
        self.host.contains_key(&key)
            || self
                .disk
                .as_ref()
                .is_some_and(|disk| disk.keys.contains_key(&key))
    }

    pub(crate) fn insert(&mut self, key: u64, payload: Vec<u8>) -> bool {
        // ONE size contract for both levels: a payload that cannot land in a
        // disk slot is refused up front. Without this, the host level accepts
        // it and a later cold-spill (or a `--kv-dram 0` direct write) hits
        // `KvMmapStore::write_slot`'s assert — a process abort, not a miss.
        if payload.len() > self.bytes_per_page {
            log::warn!(
                "KV tier insert refused: payload {} B > page {} B (key {key})",
                payload.len(),
                self.bytes_per_page
            );
            return false;
        }
        if self.host.len() < self.host_capacity_pages {
            self.insert_host(key, payload);
            return true;
        }
        if self.host_capacity_pages > 0 && self.spill_coldest_to_disk() {
            self.insert_host(key, payload);
            return true;
        }
        self.write_to_disk(key, &payload)
    }

    fn next_stamp(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    fn insert_host(&mut self, key: u64, payload: Vec<u8>) {
        let stamp = self.next_stamp();
        if let Some(old) = self.host.insert(key, HostDemotedEntry { stamp, payload }) {
            self.host_lru.remove(&(old.stamp, key));
        }
        self.host_lru.insert((stamp, key));
    }

    fn write_to_disk(&mut self, key: u64, payload: &[u8]) -> bool {
        let Some(disk) = &mut self.disk else {
            return false;
        };
        let already_present = disk.keys.contains_key(&key);
        let slot = if already_present {
            disk.keys[&key].slot
        } else {
            match disk.store.alloc_slot() {
                Some(s) => s,
                None => return false,
            }
        };
        if let Err(err) = disk.store.write_slot(slot, payload) {
            log::warn!("KV mmap write failed for key {key} slot {slot}: {err}");
            if !already_present {
                disk.store.free_slot(slot);
            }
            return false;
        }
        disk.keys.insert(
            key,
            DiskRecord {
                slot,
                len: payload.len(),
            },
        );
        if disk.durable {
            if let Err(err) = disk.write_manifest() {
                log::warn!("KV recall manifest update failed for key {key}: {err}");
            }
        }
        true
    }

    fn spill_coldest_to_disk(&mut self) -> bool {
        let Some((stamp, key)) = self.host_lru.iter().next().copied() else {
            return false;
        };
        self.host_lru.remove(&(stamp, key));
        let Some(entry) = self.host.remove(&key) else {
            return false;
        };
        debug_assert_eq!(entry.stamp, stamp);
        if self.write_to_disk(key, &entry.payload) {
            true
        } else {
            self.host.insert(key, entry);
            self.host_lru.insert((stamp, key));
            false
        }
    }

    /// Fetch a payload for promotion. Host hit bumps recency and returns the
    /// owned payload. Disk hit returns a **zero-copy** mmap slice — no allocation,
    /// no copy. The caller (promote) copies the slice into a device buffer, so
    /// the borrowed lifetime is sufficient.
    pub(crate) fn read(&mut self, key: u64) -> Result<Cow<'_, [u8]>> {
        // Host hit: bump LRU, return owned payload.
        if let Some(old_stamp) = self.host.get(&key).map(|entry| entry.stamp) {
            let stamp = self.next_stamp();
            self.host_lru.remove(&(old_stamp, key));
            self.host_lru.insert((stamp, key));
            let entry = self.host.get_mut(&key).expect("key observed above");
            entry.stamp = stamp;
            return Ok(Cow::Borrowed(entry.payload.as_slice()));
        }
        // Disk hit: zero-copy mmap slice.
        if let Some(disk) = self.disk.as_ref() {
            if let Some(record) = disk.keys.get(&key) {
                let len = if record.len == 0 {
                    self.bytes_per_page
                } else {
                    record.len.min(self.bytes_per_page)
                };
                return Ok(Cow::Borrowed(&disk.store.read_slot(record.slot)[..len]));
            }
        }
        Err(anyhow!("KV tier store has no entry for key {key}"))
    }

    /// Rank-LOCAL chunked-blob insert: one manifest page under
    /// `tier_key(ns, key)` + N chunk pages (`bytes_per_page` each) under
    /// `tier_key(ns_chunk, chunk_sub(key, i))`. On any failed insert removes
    /// everything already added and returns false. No collectives — callers
    /// own any TP consensus.
    pub(crate) fn insert_chunked(
        &mut self,
        ns: u64,
        ns_chunk: u64,
        key: u64,
        bytes: &[u8],
    ) -> bool {
        debug_assert!(!bytes.is_empty(), "chunked blob must be non-empty");
        let chunks = bytes.len().div_ceil(self.bytes_per_page);
        // Refuse blobs whose chunk index would alias the next key's chunks
        // (chunk_sub packs idx into CHUNK_IDX_BITS) — unreachable at 16 MiB
        // pages, load-bearing for small-page callers of this general API.
        if chunks > (1 << CHUNK_IDX_BITS) || self.available_pages() <= chunks {
            return false;
        }
        let manifest_key = tier_key(ns, key);
        if !self.insert(manifest_key, chunk_manifest(chunks, bytes.len())) {
            return false;
        }
        let mut added = Vec::with_capacity(chunks + 1);
        added.push(manifest_key);
        for (idx, chunk) in bytes.chunks(self.bytes_per_page).enumerate() {
            let chunk_key = tier_key(ns_chunk, chunk_sub(key, idx));
            if !self.insert(chunk_key, chunk.to_vec()) {
                self.remove(&added);
                return false;
            }
            added.push(chunk_key);
        }
        true
    }

    /// Rank-LOCAL chunked-blob read-back (host hit or zero-copy mmap slices,
    /// assembled owned). Err on any missing piece.
    pub(crate) fn read_chunked(&mut self, ns: u64, ns_chunk: u64, key: u64) -> Result<Vec<u8>> {
        let manifest = self.read(tier_key(ns, key))?.into_owned();
        let (chunks, total) = parse_chunk_manifest(&manifest)?;
        let mut bytes = Vec::with_capacity(total);
        for idx in 0..chunks {
            bytes.extend_from_slice(&self.read(tier_key(ns_chunk, chunk_sub(key, idx)))?);
        }
        bytes.truncate(total);
        anyhow::ensure!(
            bytes.len() == total,
            "chunked blob {key} truncated: {} < {total}",
            bytes.len()
        );
        Ok(bytes)
    }

    /// Rank-LOCAL chunked-blob drop (manifest + chunks). Tolerates a missing
    /// or unparsable manifest (drops whatever resolves).
    pub(crate) fn remove_chunked(&mut self, ns: u64, ns_chunk: u64, key: u64) {
        let manifest_key = tier_key(ns, key);
        let chunks = self
            .read(manifest_key)
            .ok()
            .and_then(|m| parse_chunk_manifest(m.as_ref()).ok())
            .map_or(0, |(chunks, _)| chunks);
        let mut all = Vec::with_capacity(chunks + 1);
        all.push(manifest_key);
        all.extend((0..chunks).map(|idx| tier_key(ns_chunk, chunk_sub(key, idx))));
        self.remove(&all);
    }

    /// Drop entries from both levels. In the mmap store, freed slots return to
    /// the free list (no file unlink needed — the slot bytes are simply
    /// overwritten on next allocation).
    pub(crate) fn remove(&mut self, keys: &[u64]) {
        let mut disk_index_changed = false;
        for key in keys {
            if let Some(entry) = self.host.remove(key) {
                self.host_lru.remove(&(entry.stamp, *key));
            }
            if let Some(disk) = &mut self.disk {
                if let Some(record) = disk.keys.remove(key) {
                    disk.store.free_slot(record.slot);
                    disk_index_changed = true;
                }
            }
        }
        if disk_index_changed {
            if let Some(disk) = self.disk.as_ref() {
                if disk.durable {
                    if let Err(err) = disk.write_manifest() {
                        log::warn!("KV recall manifest update after remove failed: {err}");
                    }
                }
            }
        }
    }
}

/// Cheap content-version tag for a model checkpoint directory.
pub(crate) fn weights_epoch_tag(model_path: &Path) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    let mut any = false;
    if let Ok(entries) = std::fs::read_dir(model_path) {
        let mut files: Vec<std::fs::DirEntry> = entries.flatten().collect();
        files.sort_by_key(std::fs::DirEntry::file_name);
        for entry in files {
            let name = entry.file_name();
            let Some(name_str) = name.to_str() else {
                continue;
            };
            if !name_str.ends_with(".safetensors") {
                continue;
            }
            any = true;
            mix(name_str.as_bytes());
            if let Ok(meta) = entry.metadata() {
                mix(&meta.len().to_le_bytes());
                if let Ok(modified) = meta.modified() {
                    if let Ok(dur) = modified.duration_since(std::time::UNIX_EPOCH) {
                        mix(&dur.as_secs().to_le_bytes());
                        mix(&dur.subsec_nanos().to_le_bytes());
                    }
                }
            }
        }
    }
    if any {
        format!("st-{hash:016x}")
    } else {
        mix(model_path.to_string_lossy().as_bytes());
        format!("path-{hash:016x}")
    }
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "arle_kv_tier_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("create temp root");
        root
    }

    #[test]
    fn host_only_store_caps_at_budget() {
        let mut store = CudaKvTierStore::with_budget(16, 8);
        assert_eq!(store.capacity_pages(), 2);
        assert!(store.insert(1, vec![1; 8]));
        assert!(store.insert(2, vec![2; 8]));
        assert!(store.is_full());
        assert!(!store.insert(3, vec![3; 8]), "no disk level: reject");
        assert_eq!(store.read(1).expect("host read").as_ref(), &[1u8; 8]);
    }

    #[test]
    fn chunked_blob_round_trips_and_accounts() {
        // 32-byte pages (the manifest must fit ONE page); an 80-byte blob =
        // 1 manifest + 3 chunks = 4 pages.
        let mut store = CudaKvTierStore::with_budget(32 * 10, 32);
        let baseline = store.available_pages();
        let blob: Vec<u8> = (0..80u8).collect();
        assert!(store.insert_chunked(3, 4, 7, &blob));
        assert_eq!(store.available_pages(), baseline - 4);
        assert_eq!(store.read_chunked(3, 4, 7).expect("read back"), blob);
        store.remove_chunked(3, 4, 7);
        assert_eq!(
            store.available_pages(),
            baseline,
            "remove returns the store to baseline"
        );
        assert!(store.read_chunked(3, 4, 7).is_err(), "blob gone");
    }

    /// The leak class the pod verify surfaced: a superseded blob (identical
    /// prompt re-captured under a fresh key) must be fully reclaimed —
    /// supersede-without-remove is invisible functionally and shows up only
    /// as monotonic page counts.
    #[test]
    fn superseded_chunked_blob_removal_returns_store_to_baseline() {
        let mut store = CudaKvTierStore::with_budget(32 * 20, 32);
        let baseline = store.available_pages();
        let blob: Vec<u8> = (0..80u8).collect();
        assert!(store.insert_chunked(3, 4, 1, &blob), "v1 inserts");
        assert!(store.insert_chunked(3, 4, 2, &blob), "v2 inserts alongside");
        assert_eq!(store.available_pages(), baseline - 8);
        store.remove_chunked(3, 4, 1); // supersede: drop v1
        assert_eq!(
            store.available_pages(),
            baseline - 4,
            "exactly one blob's pages remain"
        );
        assert_eq!(store.read_chunked(3, 4, 2).expect("v2 intact"), blob);
    }

    #[test]
    fn chunked_blob_spills_to_disk_and_survives_supersede() {
        // Zero host pages: everything goes straight to the mmap level
        // (the --kv-dram 0 shape used on the pod).
        let root = temp_root("chunked_disk");
        let mut store = CudaKvTierStore::with_budget(0, 32);
        assert!(store.set_disk(root.clone(), 32 * 20, 32));
        let baseline = store.available_pages();
        let blob: Vec<u8> = (0..80u8).collect();
        assert!(store.insert_chunked(3, 4, 1, &blob));
        assert_eq!(store.disk_pages(), 4, "manifest + 3 chunks on disk");
        assert!(store.insert_chunked(3, 4, 2, &blob));
        store.remove_chunked(3, 4, 1);
        assert_eq!(store.disk_pages(), 4, "superseded disk blob reclaimed");
        assert_eq!(store.available_pages(), baseline - 4);
        assert_eq!(store.read_chunked(3, 4, 2).expect("v2 via mmap"), blob);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn remove_chunked_tolerates_missing_manifest() {
        let mut store = CudaKvTierStore::with_budget(32 * 10, 32);
        let baseline = store.available_pages();
        store.remove_chunked(3, 4, 99); // never inserted
        assert_eq!(store.available_pages(), baseline);
    }

    /// Saturation probe, NOT a unit test (`cargo test ... -- --ignored`):
    /// streams 2 GiB of production-shaped chunked blobs (32 MiB blob = two
    /// 16 MiB chunks + manifest) through the disk level under `--kv-dram 0`
    /// semantics and prints GB/s. Expectations: burst writes ≈ page-cache
    /// memcpy speed (the store is ephemeral and never msyncs — the kernel
    /// writes back behind us, exactly the serve behavior); warm reads ≈
    /// memcpy. Cold-device numbers come from a `dd`/`fio` baseline beside it.
    /// Root override: `ARLE_KV_BENCH_ROOT` (point it at the real NVMe).
    #[test]
    #[ignore = "bandwidth probe — run on the target box with --ignored"]
    fn bench_chunked_blob_disk_bandwidth() {
        const BLOB: usize = 32 << 20;
        const NUM: u64 = 64; // 2 GiB payload total
        let page = 16 << 20;
        // Own a subdirectory so the cleanup below can never delete a
        // user-supplied tree (ARLE_KV_BENCH_ROOT may be a live data root).
        let root = std::env::var("ARLE_KV_BENCH_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| temp_root("bw"))
            .join(format!("arle-bw-probe-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("bench root");
        let mut store = CudaKvTierStore::with_budget(0, page);
        assert!(store.set_disk(root.clone(), (NUM as usize + 8) * 3 * page, page));
        let blob = vec![0xA5u8; BLOB];

        let t0 = std::time::Instant::now();
        for key in 0..NUM {
            assert!(store.insert_chunked(3, 4, key, &blob), "insert {key}");
        }
        let write = t0.elapsed();
        let t1 = std::time::Instant::now();
        let mut total = 0usize;
        for key in 0..NUM {
            total += store.read_chunked(3, 4, key).expect("read").len();
        }
        let read = t1.elapsed();
        assert_eq!(total, NUM as usize * BLOB);

        let gib = (NUM as f64 * BLOB as f64) / (1u64 << 30) as f64;
        println!(
            "chunked-blob disk bandwidth over {gib:.1} GiB: \
             write {:.2} GiB/s ({write:?}), read(warm) {:.2} GiB/s ({read:?})",
            gib / write.as_secs_f64(),
            gib / read.as_secs_f64(),
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// Round-3 pod finding: an oversized payload must be REFUSED, not abort
    /// the process. The disk level's `write_slot` asserts `len <= slot_bytes`;
    /// before the insert-side guard, a host-accepted oversize payload
    /// panicked there on cold-spill (or immediately under `--kv-dram 0`).
    #[test]
    fn oversized_payload_fails_closed_on_both_levels() {
        // Disk-only shape (--kv-dram 0): refusal, not the write_slot assert.
        let root = temp_root("oversize");
        let mut store = CudaKvTierStore::with_budget(0, 8);
        assert!(store.set_disk(root.clone(), 64, 8));
        assert!(!store.insert(1, vec![0; 13]), "oversize refused, no panic");
        assert_eq!(store.disk_pages(), 0);
        // Host shape: refused up front too (one contract for both levels).
        let mut host_store = CudaKvTierStore::with_budget(64, 8);
        assert!(!host_store.insert(1, vec![0; 13]));
        assert_eq!(host_store.host_demoted_pages(), 0);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn chunk_manifest_round_trips() {
        assert_eq!(
            parse_chunk_manifest(&chunk_manifest(3, 33_554_433)).unwrap(),
            (3, 33_554_433)
        );
    }

    #[test]
    fn host_overflow_spills_coldest_to_disk_and_reads_back() {
        let root = temp_root("spill");
        let mut store = CudaKvTierStore::with_budget(16, 8);
        assert!(store.set_disk(root.clone(), 32, 8));
        assert_eq!(store.capacity_pages(), 2 + 4);

        assert!(store.insert(1, vec![1; 8]));
        assert!(store.insert(2, vec![2; 8]));
        // Touch key 1 so key 2 is coldest.
        store.read(1).expect("touch");
        assert!(store.insert(3, vec![3; 8]));
        assert_eq!(store.host.len(), 2);
        assert_eq!(store.disk_pages(), 1, "coldest entry spilled");

        // Spilled entry reads back byte-identical (zero-copy mmap).
        assert_eq!(store.read(2).expect("disk read").as_ref(), &[2u8; 8]);

        // Removal frees the slot.
        store.remove(&[2]);
        assert_eq!(store.disk_pages(), 0);
        assert!(store.read(2).is_err());

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn disk_read_is_zero_copy_mmap_slice() {
        let root = temp_root("mmap_read");
        let mut store = CudaKvTierStore::with_budget(0, 8);
        assert!(store.set_disk(root.clone(), 32, 8));
        assert!(store.insert(1, vec![1; 8]));

        {
            let first = store.read(1).expect("first disk read");
            assert!(
                matches!(&first, Cow::Borrowed(_)),
                "disk read borrows mmap slice"
            );
            assert_eq!(first.as_ref(), &[1u8; 8]);
        }
        // Second read still works (mmap unchanged).
        {
            let second = store.read(1).expect("second disk read");
            assert_eq!(second.as_ref(), &[1u8; 8]);
        }

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn disk_read_respects_recorded_payload_length() {
        let root = temp_root("disk_len");
        let mut store = CudaKvTierStore::with_budget(0, 16);
        assert!(store.set_disk(root.clone(), 32, 16));
        assert!(store.insert(7, b"short".to_vec()));
        let read = store.read(7).expect("disk read");
        assert_eq!(read.as_ref(), b"short");
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn disk_full_allows_replacing_existing_key() {
        let root = temp_root("replace");
        let mut store = CudaKvTierStore::with_budget(0, 8);
        assert!(store.set_disk(root.clone(), 8, 8));

        assert!(store.insert(1, vec![1; 8]));
        assert!(store.is_full());
        assert!(
            store.insert(1, vec![9; 8]),
            "replace does not consume capacity"
        );
        assert_eq!(store.disk_pages(), 1);
        assert_eq!(
            store.read(1).expect("replaced disk read").as_ref(),
            &[9u8; 8]
        );

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn disk_only_config_writes_straight_to_disk() {
        let root = temp_root("disk_only");
        let mut store = CudaKvTierStore::with_budget(0, 8);
        assert!(store.set_disk(root.clone(), 16, 8));
        assert_eq!(store.capacity_pages(), 2);

        assert!(store.insert(1, vec![1; 8]), "insert lands on disk");
        assert_eq!(store.host.len(), 0);
        assert_eq!(store.disk_pages(), 1);
        assert_eq!(store.read(1).expect("disk read").as_ref(), &[1u8; 8]);
        assert!(store.insert(2, vec![2; 8]));
        assert!(store.is_full());
        assert!(!store.insert(3, vec![3; 8]), "disk full rejects");

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn available_pages_counts_host_and_disk_blocks() {
        let root = temp_root("available_pages");
        let mut store = CudaKvTierStore::with_budget(8, 4);
        assert!(store.set_disk(root.clone(), 8, 4));
        assert_eq!(store.capacity_pages(), 4);
        assert_eq!(store.available_pages(), 4);

        assert!(store.insert(1, vec![1; 4]));
        assert!(store.insert(2, vec![2; 4]));
        assert_eq!(store.available_pages(), 2);

        assert!(store.insert(3, vec![3; 4]));
        assert_eq!(store.host_demoted_pages(), 2);
        assert_eq!(store.disk_pages(), 1);
        assert_eq!(store.available_pages(), 1);

        store.remove(&[3]);
        assert_eq!(store.available_pages(), 2);

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn full_means_both_levels_full() {
        let root = temp_root("full");
        let mut store = CudaKvTierStore::with_budget(8, 8);
        assert!(store.set_disk(root.clone(), 8, 8));
        assert!(store.insert(1, vec![1; 8]));
        assert!(!store.is_full(), "disk still has room");
        assert!(store.insert(2, vec![2; 8]), "spills 1 to disk");
        assert!(store.is_full());
        assert!(!store.insert(3, vec![3; 8]));

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn disk_slot_is_reused_after_remove() {
        let root = temp_root("reuse");
        let mut store = CudaKvTierStore::with_budget(0, 8);
        assert!(store.set_disk(root.clone(), 8, 8));
        assert!(store.insert(1, vec![1; 8]));
        assert_eq!(store.disk_pages(), 1);
        store.remove(&[1]);
        assert_eq!(store.disk_pages(), 0);
        // Same slot should be re-allocated.
        assert!(store.insert(2, vec![2; 8]));
        assert_eq!(store.disk_pages(), 1);

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    const GIB: usize = 1 << 30;

    #[test]
    fn host_default_budget_at_least_floor() {
        let b = default_t1_budget_bytes(0.5);
        assert!(
            b == 0 || b >= 4 * GIB,
            "L2 DRAM budget {b} is neither 0 nor ≥ floor"
        );
    }

    #[test]
    fn disk_default_budget_runs_and_is_bounded_by_free() {
        let root = temp_root("disk_probe");
        let b = default_t2_budget_bytes(&root, 0.5);
        if let Some((free, _total)) = disk_free_total_bytes(&root) {
            assert!(b <= free, "L3 budget {b} exceeds free disk {free}");
        }
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn disk_free_total_probe_reports_some_on_real_dir() {
        let root = temp_root("statvfs");
        let probe = disk_free_total_bytes(&root);
        assert!(
            probe.is_some(),
            "statvfs on an existing dir should report free+total bytes"
        );
        if let Some((free, total)) = probe {
            assert!(total >= free, "total disk {total} < free {free}");
        }
        assert!(disk_free_total_bytes(&root.join("does-not-exist/x")).is_none());
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn weights_epoch_tag_is_stable_and_path_falls_back() {
        let root = temp_root("epoch");
        let a = weights_epoch_tag(&root);
        let b = weights_epoch_tag(&root);
        assert_eq!(a, b, "epoch tag must be deterministic for an unchanged dir");
        assert!(
            a.starts_with("path-"),
            "empty dir falls back to path tag: {a}"
        );

        std::fs::write(root.join("model.safetensors"), b"weights-v1").expect("write st");
        let c = weights_epoch_tag(&root);
        assert!(
            c.starts_with("st-"),
            "safetensors present → content tag: {c}"
        );
        assert_ne!(a, c, "adding weights must change the tag");
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn durable_manifest_round_trips_disk_index_across_restart() {
        let root = temp_root("durable_mmap");
        let epoch = "epoch-A".to_string();

        // session 1: write three blocks, then drop.
        let namespace;
        {
            let mut store = CudaKvTierStore::with_budget(0, 8);
            assert!(store.set_disk_durable(root.clone(), 32, 8, epoch.clone()));
            namespace = root.join(format!("arle-kv-recall-{}", std::process::id()));
            assert!(namespace.exists());
            assert!(namespace.join("kv.mmap").exists());

            assert!(store.insert(1, vec![1; 8]));
            assert!(store.insert(2, vec![2; 8]));
            assert!(store.insert(3, vec![3; 8]));
            assert!(namespace.join(MANIFEST_FILE).exists());
            assert_eq!(store.disk_pages(), 3);
        }
        // Durable namespace survives drop.
        assert!(namespace.exists());

        // session 2: load() rebuilds index from manifest.
        {
            let mut store = CudaKvTierStore::with_budget(0, 8);
            let reloaded = store.load(root.clone(), 32, 8, epoch.clone());
            assert!(reloaded, "load reports prior blocks reloaded");
            assert_eq!(store.disk_pages(), 3, "all three keys re-indexed");
            assert_eq!(store.read(1).expect("reload read 1").as_ref(), &[1u8; 8]);
            assert_eq!(store.read(3).expect("reload read 3").as_ref(), &[3u8; 8]);
        }

        // session 3: mismatched epoch discards stale memory.
        {
            let mut store = CudaKvTierStore::with_budget(0, 8);
            let reloaded = store.load(root.clone(), 32, 8, "epoch-B".to_string());
            assert!(reloaded, "durable disk tier still attaches");
            assert_eq!(store.disk_pages(), 0, "stale-epoch index starts cold");
            assert!(store.read(1).is_err(), "stale-epoch key must not resolve");
        }

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn durable_manifest_round_trips_variable_payload_length() {
        let root = temp_root("durable_varlen");
        let epoch = "epoch-A".to_string();

        {
            let mut store = CudaKvTierStore::with_budget(0, 16);
            assert!(store.set_disk_durable(root.clone(), 32, 16, epoch.clone()));
            assert!(store.insert(11, b"tiny".to_vec()));
            assert!(store.insert(12, b"payload-123".to_vec()));
            let disk = store.disk.as_ref().expect("disk tier");
            assert_eq!(
                disk.keys.get(&11),
                Some(&DiskRecord { slot: 1, len: 4 }),
                "latest alloc pops highest free slot first"
            );
            assert_eq!(disk.keys.get(&12), Some(&DiskRecord { slot: 0, len: 11 }),);
        }

        {
            let mut store = CudaKvTierStore::with_budget(0, 16);
            let reloaded = store.load(root.clone(), 32, 16, epoch.clone());
            assert!(reloaded, "load reports prior blocks reloaded");
            assert_eq!(store.read(11).expect("reload read 11").as_ref(), b"tiny");
            assert_eq!(
                store.read(12).expect("reload read 12").as_ref(),
                b"payload-123"
            );
            let disk = store.disk.as_ref().expect("disk tier");
            assert_eq!(disk.keys.get(&11), Some(&DiskRecord { slot: 1, len: 4 }),);
            assert_eq!(disk.keys.get(&12), Some(&DiskRecord { slot: 0, len: 11 }),);
        }

        std::fs::remove_dir_all(&root).expect("cleanup");
    }
}

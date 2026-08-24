//! Two-level (host RAM + disk) store for demoted KV blocks — backend-neutral.
//!
//! Host-demoted blocks live in a capacity-capped in-RAM map (default-on, 4 GiB).
//! Disk spill is optional on the `KvMmapStore` block substrate
//! (`--kv-disk`, opt-in): when host RAM fills, the coldest host entry spills
//! to a file-backed mmap page-slot store, so the capacity the engine sees is
//! host-demoted + disk slots. Payloads are opaque fixed-limit blocks (paged-KV
//! pages, slot-image chunks, or encoded MLX pages); this module never touches
//! the device. Metal runs the disk-only degenerate case (host budget 0 —
//! unified memory makes a DRAM L2 a no-op).
//!
//! ## Mmap store
//!
//! The disk tier uses [`crate::KvMmapStore`] — one file per namespace,
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
//! `{key, slot_idx}` plus an epoch tag. [`KvTierStore::load`] replays the
//! manifest so a prior session's evicted KV is addressable again; a stale epoch
//! (e.g. after an OPD weight update) discards the prior memory.
//!
//! [`set_disk`]: KvTierStore::set_disk

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;

use anyhow::{Context, Result, anyhow};
use infer_seam::{DramTierPolicy, NvmeTierPolicy, dram_l2_budget, nvme_l3_budget};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DiskIoMode {
    #[default]
    Disabled,
    Mmap,
    Direct,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TierIoStats {
    pub mode: DiskIoMode,
    pub useful_read_bytes: u64,
    pub useful_write_bytes: u64,
    pub submitted_read_bytes: u64,
    pub submitted_write_bytes: u64,
    pub metadata_write_bytes: u64,
    pub read_ops: u64,
    pub write_ops: u64,
    pub failures: u64,
    pub completion_wait_ns: u64,
    pub fallback_count: u64,
    pub fallback_reason: Option<String>,
}

/// Manifest filename under a durable disk namespace. Records the epoch tag plus
/// one `key slot_idx len slot_bytes` line per disk-resident block so a restart
/// can rebuild the in-memory disk index and replay slot allocations.
const MANIFEST_FILE: &str = "manifest.kvm";

/// Manifest header magic + version. A mismatch — or a missing manifest — makes
/// [`KvTierStore::load`] start cold rather than trust a foreign layout.
const MANIFEST_MAGIC: &str = "ARLE-KVTIER-MANIFEST-V2";

static DISK_TIER_NAMESPACE_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

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
    // SAFETY: zeroed statvfs out-param filled by libc; rc checked before use.
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

pub fn default_t1_budget_bytes(dram_fraction: f64) -> usize {
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

/// One u64 key space partitioned by a top-byte namespace so features sharing a
/// store never collide; callers own the namespace constants. Manifest keys
/// carry the feature key in the low bits; chunk keys carry
/// `key << CHUNK_IDX_BITS | idx` (a blob is ≤ 2^16 × page_bytes).
pub const TIER_NS_SHIFT: u32 = 56;
pub const CHUNK_IDX_BITS: u32 = 16;
/// Canonical chunk/page size for whole-slot blob stores (DSv4 + Qwen3.6).
pub const BLOB_CHUNK_BYTES: usize = 16 << 20;

pub fn tier_key(ns: u64, sub: u64) -> u64 {
    debug_assert!(
        sub >> TIER_NS_SHIFT == 0,
        "tier sub-key overflows namespace"
    );
    (ns << TIER_NS_SHIFT) | sub
}

pub fn chunk_sub(key: u64, idx: usize) -> u64 {
    (key << CHUNK_IDX_BITS) | idx as u64
}

pub fn chunk_manifest(chunks: usize, bytes: usize) -> Vec<u8> {
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

pub struct KvTierStore {
    host_capacity_pages: usize,
    bytes_per_page: usize,
    host: BTreeMap<u64, HostDemotedEntry>,
    host_lru: BTreeSet<(u64, u64)>,
    clock: u64,
    disk: Option<DiskTier>,
    host_read_hits: u64,
    disk_read_hits: u64,
}

struct HostDemotedEntry {
    stamp: u64,
    payload: Vec<u8>,
}

enum DiskStore {
    Mmap(crate::KvMmapStore),
    #[cfg(target_os = "linux")]
    Direct(Box<crate::direct_store::DirectStore>),
}

impl DiskStore {
    fn num_slots(&self) -> u32 {
        match self {
            Self::Mmap(store) => store.num_slots(),
            #[cfg(target_os = "linux")]
            Self::Direct(store) => store.num_slots(),
        }
    }

    fn slot_bytes(&self) -> usize {
        match self {
            Self::Mmap(store) => store.slot_bytes(),
            #[cfg(target_os = "linux")]
            Self::Direct(store) => store.slot_bytes(),
        }
    }

    fn alloc_slot(&mut self) -> Option<u32> {
        match self {
            Self::Mmap(store) => store.alloc_slot(),
            #[cfg(target_os = "linux")]
            Self::Direct(store) => store.alloc_slot(),
        }
    }

    fn available_slots(&self) -> usize {
        match self {
            Self::Mmap(store) => store.available_slots(),
            #[cfg(target_os = "linux")]
            Self::Direct(store) => store.available_slots(),
        }
    }

    fn free_slot(&mut self, slot: u32) {
        match self {
            Self::Mmap(store) => store.free_slot(slot),
            #[cfg(target_os = "linux")]
            Self::Direct(store) => store.free_slot(slot),
        }
    }

    /// Flush written slots to disk so a following durable-manifest write cannot
    /// name a slot whose bytes are still only in the page cache. O_DIRECT writes
    /// bypass the page cache and are on-device at io_uring completion, so the
    /// Direct arm is a no-op.
    fn flush(&self) -> std::io::Result<()> {
        match self {
            Self::Mmap(store) => store.flush(),
            #[cfg(target_os = "linux")]
            Self::Direct(_) => Ok(()),
        }
    }

    fn reserve_indices(&mut self, indices: &[u32]) {
        match self {
            Self::Mmap(store) => store.reserve_indices(indices),
            #[cfg(target_os = "linux")]
            Self::Direct(store) => store.reserve_indices(indices),
        }
    }

    fn write_payloads(&mut self, writes: &[(u32, &[u8])]) -> std::io::Result<WriteIo> {
        match self {
            Self::Mmap(store) => {
                for (slot, bytes) in writes {
                    store.write_slot(*slot, bytes)?;
                }
                Ok(WriteIo {
                    submitted_bytes: writes.iter().map(|(_, bytes)| bytes.len() as u64).sum(),
                    wait_ns: 0,
                })
            }
            #[cfg(target_os = "linux")]
            Self::Direct(store) => store.write_slots(writes).map(|batch| WriteIo {
                submitted_bytes: batch.submitted_bytes,
                wait_ns: batch.wait_ns,
            }),
        }
    }
}

struct DiskTier {
    /// Namespace directory (contains `kv.mmap` + `manifest.kvm`).
    root_dir: PathBuf,
    /// Fixed-size page-slot store selected once at attach.
    store: DiskStore,
    /// Key -> slot index plus valid payload length in the mmap store.
    keys: BTreeMap<u64, DiskRecord>,
    /// Durable tier: suppress drop-time wipe, persist manifest on mutation.
    durable: bool,
    /// Model/weights-version tag written into the manifest.
    epoch: String,
    stats: TierIoStats,
    worker_stats: Arc<WorkerIoStats>,
    pending: BTreeMap<u64, PendingRecord>,
    next_generation: u64,
    accepting_writes: bool,
    manifest_dirty: bool,
    worker_tx: Option<SyncSender<DiskWork>>,
    completion_rx: Receiver<DiskCompletion>,
    worker: Option<JoinHandle<()>>,
    _lock: Option<std::fs::File>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DiskRecord {
    slot: u32,
    len: usize,
}

struct PendingRecord {
    generation: u64,
    slot: u32,
    #[allow(
        clippy::rc_buffer,
        reason = "moves Vec payload without copying its allocation"
    )]
    payload: Arc<Vec<u8>>,
    useful: bool,
}

struct WriteItem {
    key: u64,
    generation: u64,
    slot: u32,
    #[allow(
        clippy::rc_buffer,
        reason = "moves Vec payload without copying its allocation"
    )]
    payload: Arc<Vec<u8>>,
    useful: bool,
}

enum DiskWork {
    Write(Vec<WriteItem>),
    Manifest(Vec<u8>),
}

enum DiskCompletion {
    Write {
        items: Vec<WriteItem>,
        result: std::io::Result<()>,
    },
    Manifest(std::io::Result<()>),
}

#[derive(Clone, Copy)]
struct WriteIo {
    submitted_bytes: u64,
    wait_ns: u64,
}

#[derive(Default)]
struct WorkerIoStats {
    submitted_write_bytes: AtomicU64,
    write_wait_ns: AtomicU64,
    metadata_write_bytes: AtomicU64,
    failures: AtomicU64,
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        let active = self.worker.is_some();
        self.finish_worker();
        if active
            && self.durable
            && let Err(err) = self.write_manifest()
        {
            log::warn!("KV final manifest persist failed: {err}");
        }
        if let Some(lock) = self._lock.take()
            && let Err(err) = lock.unlock()
        {
            log::warn!("KV durable namespace unlock failed: {err}");
        }
        if !self.durable {
            // KvMmapStore drops first (field order), unmapping the file.
            // Linux tolerates unlinking a still-open file, so best-effort.
            let _ = std::fs::remove_dir_all(&self.root_dir);
        }
    }
}

impl DiskTier {
    fn finish_worker(&mut self) {
        self.worker_tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.drain_completions();
    }

    fn lock_namespace(namespace: &Path) -> Result<std::fs::File> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(namespace.join("owner.lock"))?;
        file.try_lock()
            .map_err(|err| anyhow!("namespace already owned: {err}"))?;
        Ok(file)
    }

    fn manifest_path(&self) -> PathBuf {
        self.root_dir.join(MANIFEST_FILE)
    }

    fn write_manifest(&mut self) -> Result<()> {
        // Data-before-manifest ordering barrier: msync the slots this manifest
        // names, then fsync the manifest. Without it a crash replays the
        // manifest onto unflushed (zero) slots → silent wrong KV. Only the
        // durable tier persists a manifest, so this path never runs volatile.
        if let Err(err) = self.store.flush() {
            self.stats.failures = self.stats.failures.saturating_add(1);
            return Err(anyhow::Error::new(err)).with_context(|| {
                format!("KV recall data flush under {}", self.root_dir.display())
            });
        }
        let buf = self.manifest_bytes();
        let bytes = buf.len() as u64;
        let result = crate::write_file_atomic_durable(&self.manifest_path(), &buf)
            .with_context(|| format!("KV recall manifest write under {}", self.root_dir.display()));
        if result.is_ok() {
            self.stats.metadata_write_bytes = self.stats.metadata_write_bytes.saturating_add(bytes);
        } else {
            self.stats.failures = self.stats.failures.saturating_add(1);
        }
        result
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

    fn manifest_bytes(&self) -> Vec<u8> {
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
        buf.into_bytes()
    }

    fn enqueue(
        &mut self,
        entries: Vec<(u64, Vec<u8>, bool)>,
    ) -> Result<(), Vec<(u64, Vec<u8>, bool)>> {
        self.drain_completions();
        if entries.is_empty() {
            return Ok(());
        }
        if !self.accepting_writes {
            return Err(entries);
        }
        let mut items: Vec<WriteItem> = Vec::with_capacity(entries.len());
        let mut entries = entries.into_iter();
        while let Some((key, payload, useful)) = entries.next() {
            let Some(slot) = self.store.alloc_slot() else {
                let mut rejected = Vec::with_capacity(items.len() + entries.len() + 1);
                rejected.extend(self.reject(items));
                rejected.push((key, payload, useful));
                rejected.extend(entries);
                return Err(rejected);
            };
            self.next_generation = self.next_generation.saturating_add(1);
            items.push(WriteItem {
                key,
                generation: self.next_generation,
                slot,
                payload: Arc::new(payload),
                useful,
            });
        }
        let Some(tx) = self.worker_tx.as_ref() else {
            return Err(self.reject(items));
        };
        let pending = items
            .iter()
            .map(|item| {
                (
                    item.key,
                    PendingRecord {
                        generation: item.generation,
                        slot: item.slot,
                        payload: Arc::clone(&item.payload),
                        useful: item.useful,
                    },
                )
            })
            .collect::<Vec<_>>();
        match tx.try_send(DiskWork::Write(items)) {
            Ok(()) => {
                self.stats.write_ops = self.stats.write_ops.saturating_add(pending.len() as u64);
                self.stats.useful_write_bytes = self.stats.useful_write_bytes.saturating_add(
                    pending
                        .iter()
                        .filter(|(_, record)| record.useful)
                        .map(|(_, record)| record.payload.len() as u64)
                        .sum::<u64>(),
                );
                for (key, record) in pending {
                    self.pending.insert(key, record);
                }
                Ok(())
            }
            Err(TrySendError::Full(DiskWork::Write(items))) => {
                drop(pending);
                Err(self.reject(items))
            }
            Err(TrySendError::Disconnected(DiskWork::Write(items))) => {
                drop(pending);
                self.accepting_writes = false;
                Err(self.reject(items))
            }
            Err(_) => unreachable!("enqueue only submits disk writes"),
        }
    }

    fn recover_item(item: WriteItem) -> (u64, Vec<u8>, bool) {
        let payload = Arc::try_unwrap(item.payload).expect("pending references dropped");
        (item.key, payload, item.useful)
    }

    fn reject(&mut self, items: Vec<WriteItem>) -> Vec<(u64, Vec<u8>, bool)> {
        items
            .into_iter()
            .map(|item| {
                self.store.free_slot(item.slot);
                Self::recover_item(item)
            })
            .collect()
    }

    fn drain_completions(&mut self) {
        while let Ok(completion) = self.completion_rx.try_recv() {
            match completion {
                DiskCompletion::Write { items, result } => match result {
                    Ok(()) => {
                        for item in items {
                            let pending = self.pending.get(&item.key);
                            if pending.is_some_and(|entry| {
                                entry.generation == item.generation && entry.slot == item.slot
                            }) {
                                self.pending.remove(&item.key);
                                if let Some(old) = self.keys.insert(
                                    item.key,
                                    DiskRecord {
                                        slot: item.slot,
                                        len: item.payload.len(),
                                    },
                                ) {
                                    self.store.free_slot(old.slot);
                                }
                                self.manifest_dirty |= self.durable;
                            } else {
                                self.store.free_slot(item.slot);
                            }
                        }
                    }
                    Err(err) => {
                        log::warn!("KV async disk writer failed: {err}");
                        self.accepting_writes = false;
                        for item in items {
                            if !self.pending.get(&item.key).is_some_and(|entry| {
                                entry.generation == item.generation && entry.slot == item.slot
                            }) {
                                self.store.free_slot(item.slot);
                            }
                        }
                    }
                },
                DiskCompletion::Manifest(result) => {
                    if let Err(err) = result {
                        log::warn!("KV async manifest writer failed: {err}");
                        self.accepting_writes = false;
                        self.manifest_dirty = true;
                    }
                }
            }
        }
        self.try_enqueue_manifest();
    }

    fn try_enqueue_manifest(&mut self) {
        if !self.durable || !self.manifest_dirty || !self.accepting_writes {
            return;
        }
        let bytes = self.manifest_bytes();
        let Some(tx) = self.worker_tx.as_ref() else {
            return;
        };
        match tx.try_send(DiskWork::Manifest(bytes)) {
            Ok(()) => self.manifest_dirty = false,
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => {
                self.accepting_writes = false;
            }
        }
    }

    fn logical_pages(&self) -> usize {
        self.keys
            .keys()
            .chain(self.pending.keys())
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
    }

    fn read_payloads(&mut self, reads: &[(u32, usize)]) -> std::io::Result<Vec<Vec<u8>>> {
        let useful = reads.iter().map(|(_, len)| *len as u64).sum::<u64>();
        let result = match &mut self.store {
            DiskStore::Mmap(store) => {
                self.stats.submitted_read_bytes =
                    self.stats.submitted_read_bytes.saturating_add(useful);
                Ok(reads
                    .iter()
                    .map(|(slot, len)| store.read_slot(*slot)[..*len].to_vec())
                    .collect())
            }
            #[cfg(target_os = "linux")]
            DiskStore::Direct(store) => store.read_slots(reads).map(|(values, batch)| {
                self.stats.submitted_read_bytes = self
                    .stats
                    .submitted_read_bytes
                    .saturating_add(batch.submitted_bytes);
                self.stats.completion_wait_ns =
                    self.stats.completion_wait_ns.saturating_add(batch.wait_ns);
                values
            }),
        };
        self.stats.read_ops = self.stats.read_ops.saturating_add(reads.len() as u64);
        if result.is_ok() {
            self.stats.useful_read_bytes = self.stats.useful_read_bytes.saturating_add(useful);
        } else {
            self.stats.failures = self.stats.failures.saturating_add(1);
        }
        result
    }

    fn read_payload(&mut self, slot: u32, len: usize) -> std::io::Result<Cow<'_, [u8]>> {
        match &mut self.store {
            DiskStore::Mmap(store) => {
                self.stats.read_ops = self.stats.read_ops.saturating_add(1);
                self.stats.useful_read_bytes =
                    self.stats.useful_read_bytes.saturating_add(len as u64);
                self.stats.submitted_read_bytes =
                    self.stats.submitted_read_bytes.saturating_add(len as u64);
                Ok(Cow::Borrowed(&store.read_slot(slot)[..len]))
            }
            #[cfg(target_os = "linux")]
            DiskStore::Direct(store) => {
                self.stats.read_ops = self.stats.read_ops.saturating_add(1);
                match store.read_slots(&[(slot, len)]) {
                    Ok((mut values, batch)) => {
                        self.stats.useful_read_bytes =
                            self.stats.useful_read_bytes.saturating_add(len as u64);
                        self.stats.submitted_read_bytes = self
                            .stats
                            .submitted_read_bytes
                            .saturating_add(batch.submitted_bytes);
                        self.stats.completion_wait_ns =
                            self.stats.completion_wait_ns.saturating_add(batch.wait_ns);
                        Ok(Cow::Owned(values.pop().expect("one disk read")))
                    }
                    Err(err) => {
                        self.stats.failures = self.stats.failures.saturating_add(1);
                        Err(err)
                    }
                }
            }
        }
    }
}

fn requested_disk_mode() -> Option<DiskIoMode> {
    parse_disk_mode(std::env::var("ARLE_KV_DISK_IO").ok().as_deref())
}

fn parse_disk_mode(value: Option<&str>) -> Option<DiskIoMode> {
    match value {
        Some("mmap") => Some(DiskIoMode::Mmap),
        Some("direct") => Some(DiskIoMode::Direct),
        _ => None,
    }
}

fn mmap_stats(reason: Option<String>) -> TierIoStats {
    TierIoStats {
        mode: DiskIoMode::Mmap,
        fallback_count: u64::from(reason.is_some()),
        fallback_reason: reason,
        ..TierIoStats::default()
    }
}

fn create_disk_store(
    path: &Path,
    num_slots: usize,
    slot_bytes: usize,
) -> std::io::Result<(DiskStore, TierIoStats)> {
    if requested_disk_mode() == Some(DiskIoMode::Direct) {
        #[cfg(target_os = "linux")]
        match crate::direct_store::DirectStore::create(path, num_slots, slot_bytes) {
            Ok(store) => {
                return Ok((
                    DiskStore::Direct(Box::new(store)),
                    TierIoStats {
                        mode: DiskIoMode::Direct,
                        ..TierIoStats::default()
                    },
                ));
            }
            Err(err) => {
                let reason = err.to_string();
                log::warn!(
                    "KV O_DIRECT/io_uring unavailable at {}: {reason}; using mmap",
                    path.display()
                );
                return crate::KvMmapStore::create(path, num_slots, slot_bytes)
                    .map(|store| (DiskStore::Mmap(store), mmap_stats(Some(reason))));
            }
        }
        #[cfg(not(target_os = "linux"))]
        if requested_disk_mode() == Some(DiskIoMode::Direct) {
            let reason = "O_DIRECT/io_uring requires Linux".to_string();
            return crate::KvMmapStore::create(path, num_slots, slot_bytes)
                .map(|store| (DiskStore::Mmap(store), mmap_stats(Some(reason))));
        }
    }
    crate::KvMmapStore::create(path, num_slots, slot_bytes)
        .map(|store| (DiskStore::Mmap(store), mmap_stats(None)))
}

fn open_disk_store(
    path: &Path,
    num_slots: usize,
    slot_bytes: usize,
) -> std::io::Result<(DiskStore, TierIoStats)> {
    if requested_disk_mode() == Some(DiskIoMode::Direct) {
        #[cfg(target_os = "linux")]
        match crate::direct_store::DirectStore::open(path, num_slots, slot_bytes) {
            Ok(store) => {
                return Ok((
                    DiskStore::Direct(Box::new(store)),
                    TierIoStats {
                        mode: DiskIoMode::Direct,
                        ..TierIoStats::default()
                    },
                ));
            }
            Err(err) => {
                let reason = err.to_string();
                log::warn!(
                    "KV O_DIRECT/io_uring unavailable at {}: {reason}; using mmap",
                    path.display()
                );
                return crate::KvMmapStore::open(path, num_slots, slot_bytes)
                    .map(|store| (DiskStore::Mmap(store), mmap_stats(Some(reason))));
            }
        }
        #[cfg(not(target_os = "linux"))]
        if requested_disk_mode() == Some(DiskIoMode::Direct) {
            let reason = "O_DIRECT/io_uring requires Linux".to_string();
            return crate::KvMmapStore::open(path, num_slots, slot_bytes)
                .map(|store| (DiskStore::Mmap(store), mmap_stats(Some(reason))));
        }
    }
    crate::KvMmapStore::open(path, num_slots, slot_bytes)
        .map(|store| (DiskStore::Mmap(store), mmap_stats(None)))
}

fn open_writer_store(
    path: &Path,
    num_slots: usize,
    slot_bytes: usize,
    mode: DiskIoMode,
) -> std::io::Result<DiskStore> {
    #[cfg(not(target_os = "linux"))]
    let _ = mode;
    #[cfg(target_os = "linux")]
    if mode == DiskIoMode::Direct {
        return crate::direct_store::DirectStore::open(path, num_slots, slot_bytes)
            .map(|store| DiskStore::Direct(Box::new(store)));
    }
    crate::KvMmapStore::open(path, num_slots, slot_bytes).map(DiskStore::Mmap)
}

fn spawn_disk_worker(
    mut store: DiskStore,
    manifest_path: PathBuf,
) -> (
    SyncSender<DiskWork>,
    Receiver<DiskCompletion>,
    JoinHandle<()>,
    Arc<WorkerIoStats>,
) {
    let depth = std::env::var("ARLE_KV_ASYNC_QUEUE_DEPTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|depth| *depth > 0)
        .unwrap_or(8);
    let (work_tx, work_rx) = sync_channel(depth);
    let (completion_tx, completion_rx) = std::sync::mpsc::channel();
    let worker_stats = Arc::new(WorkerIoStats::default());
    let counters = Arc::clone(&worker_stats);
    let worker = std::thread::Builder::new()
        .name("arle-kv-disk".to_string())
        .spawn(move || {
            let mut failed = false;
            while let Ok(work) = work_rx.recv() {
                match work {
                    DiskWork::Write(items) => {
                        let result = if failed {
                            Err(std::io::Error::other(
                                "disk writer stopped after prior failure",
                            ))
                        } else {
                            let writes = items
                                .iter()
                                .map(|item| (item.slot, item.payload.as_slice()))
                                .collect::<Vec<_>>();
                            store.write_payloads(&writes)
                        };
                        let result = match result {
                            Ok(io) => {
                                counters
                                    .submitted_write_bytes
                                    .fetch_add(io.submitted_bytes, Ordering::Relaxed);
                                counters
                                    .write_wait_ns
                                    .fetch_add(io.wait_ns, Ordering::Relaxed);
                                Ok(())
                            }
                            Err(err) => {
                                counters.failures.fetch_add(1, Ordering::Relaxed);
                                Err(err)
                            }
                        };
                        failed |= result.is_err();
                        let _ = completion_tx.send(DiskCompletion::Write { items, result });
                    }
                    DiskWork::Manifest(bytes) => {
                        let len = bytes.len();
                        let result = if failed {
                            Err(std::io::Error::other(
                                "disk writer stopped after prior failure",
                            ))
                        } else {
                            // Data-before-manifest barrier: flush the slots this
                            // manifest names before persisting it. Only the durable
                            // tier enqueues manifests, so this is durable-only.
                            store.flush().and_then(|()| {
                                crate::write_file_atomic_durable(&manifest_path, &bytes)
                            })
                        };
                        if result.is_ok() {
                            counters
                                .metadata_write_bytes
                                .fetch_add(len as u64, Ordering::Relaxed);
                        } else {
                            counters.failures.fetch_add(1, Ordering::Relaxed);
                        }
                        failed |= result.is_err();
                        let _ = completion_tx.send(DiskCompletion::Manifest(result));
                    }
                }
            }
        })
        .expect("spawn KV disk writer");
    (work_tx, completion_rx, worker, worker_stats)
}

impl Drop for KvTierStore {
    fn drop(&mut self) {
        self.persist();
    }
}

impl KvTierStore {
    pub fn with_budget(budget_bytes: usize, bytes_per_page: usize) -> Self {
        let host_capacity_pages = budget_bytes.checked_div(bytes_per_page).unwrap_or(0);
        Self {
            host_capacity_pages,
            bytes_per_page,
            host: BTreeMap::new(),
            host_lru: BTreeSet::new(),
            clock: 0,
            disk: None,
            host_read_hits: 0,
            disk_read_hits: 0,
        }
    }

    pub fn set_disk(&mut self, root: PathBuf, budget_bytes: usize, bytes_per_page: usize) -> bool {
        let namespace = self.ephemeral_namespace(root);
        self.attach_disk(
            namespace,
            budget_bytes,
            bytes_per_page,
            false,
            String::new(),
        )
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
        let namespace_lock = if durable {
            match DiskTier::lock_namespace(&namespace) {
                Ok(lock) => Some(lock),
                Err(err) => {
                    log::warn!(
                        "KV durable namespace lock failed under {}: {err}",
                        namespace.display()
                    );
                    return false;
                }
            }
        } else {
            None
        };
        match create_disk_store(&mmap_path, capacity_pages, bytes_per_page) {
            Ok((store, stats)) => {
                crate::gds::log_gate(&namespace, stats.mode);
                let writer =
                    match open_writer_store(&mmap_path, capacity_pages, bytes_per_page, stats.mode)
                    {
                        Ok(writer) => writer,
                        Err(err) => {
                            log::warn!("KV async writer open failed: {err}");
                            return false;
                        }
                    };
                let (worker_tx, completion_rx, worker, worker_stats) =
                    spawn_disk_worker(writer, namespace.join(MANIFEST_FILE));
                self.disk = Some(DiskTier {
                    root_dir: namespace,
                    store,
                    keys: BTreeMap::new(),
                    durable,
                    epoch,
                    stats,
                    worker_stats,
                    pending: BTreeMap::new(),
                    next_generation: 0,
                    accepting_writes: true,
                    manifest_dirty: false,
                    worker_tx: Some(worker_tx),
                    completion_rx,
                    worker: Some(worker),
                    _lock: namespace_lock,
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

    pub fn load(
        &mut self,
        root: PathBuf,
        budget_bytes: usize,
        bytes_per_page: usize,
        epoch: String,
        format_tag: u8,
        world_size: usize,
        rank: usize,
    ) -> bool {
        let namespace = Self::durable_namespace(
            root.clone(),
            &epoch,
            format_tag,
            world_size,
            rank,
            bytes_per_page,
        );
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
        let namespace_lock = match DiskTier::lock_namespace(&namespace) {
            Ok(lock) => lock,
            Err(err) => {
                log::warn!(
                    "KV durable namespace lock failed under {}: {err}",
                    namespace.display()
                );
                return false;
            }
        };

        let (mut store, stats, replay_manifest) =
            match open_disk_store(&mmap_path, capacity_pages, bytes_per_page) {
                Ok((store, stats)) => (store, stats, true),
                Err(_) => match create_disk_store(&mmap_path, capacity_pages, bytes_per_page) {
                    Ok((store, stats)) => (store, stats, false),
                    Err(err) => {
                        log::warn!("KV mmap store create failed: {err}");
                        return false;
                    }
                },
            };
        crate::gds::log_gate(&namespace, stats.mode);
        if !replay_manifest
            && let Err(err) = std::fs::remove_file(&manifest_path)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            log::warn!("KV stale manifest removal failed: {err}");
            return false;
        }

        let parsed = replay_manifest
            .then(|| std::fs::read(&manifest_path).ok())
            .flatten()
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
        log::info!(
            "KV recall restored {} pages from {}",
            keys.len(),
            namespace.display()
        );

        let writer = match open_writer_store(&mmap_path, capacity_pages, bytes_per_page, stats.mode)
        {
            Ok(writer) => writer,
            Err(err) => {
                log::warn!("KV async writer open failed: {err}");
                return false;
            }
        };
        let (worker_tx, completion_rx, worker, worker_stats) =
            spawn_disk_worker(writer, namespace.join(MANIFEST_FILE));

        self.disk = Some(DiskTier {
            root_dir: namespace,
            store,
            keys,
            durable: true,
            epoch,
            stats,
            worker_stats,
            pending: BTreeMap::new(),
            next_generation: 0,
            accepting_writes: true,
            manifest_dirty: false,
            worker_tx: Some(worker_tx),
            completion_rx,
            worker: Some(worker),
            _lock: Some(namespace_lock),
        });
        true
    }

    pub fn persist(&mut self) {
        if let Some(disk) = self.disk.as_mut() {
            disk.finish_worker();
            if disk.durable
                && let Err(err) = disk.write_manifest()
            {
                log::warn!("KV recall manifest persist failed: {err}");
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

    fn durable_namespace(
        root: PathBuf,
        epoch: &str,
        format_tag: u8,
        world_size: usize,
        rank: usize,
        bytes_per_page: usize,
    ) -> PathBuf {
        root.join(format!(
            "arle-kv-recall-{epoch}-format-{format_tag}-world-{world_size}-rank-{rank}-page-{bytes_per_page}"
        ))
    }

    pub fn capacity_pages(&self) -> usize {
        self.host_capacity_pages.saturating_add(
            self.disk
                .as_ref()
                .map_or(0, |d| d.store.num_slots() as usize),
        )
    }

    pub fn available_pages(&self) -> usize {
        let disk_available = self
            .disk
            .as_ref()
            .map_or(0, |disk| disk.store.available_slots());
        self.host_capacity_pages
            .saturating_sub(self.host.len())
            .saturating_add(disk_available)
    }

    pub fn page_bytes(&self) -> usize {
        self.bytes_per_page
    }

    pub fn host_demoted_pages(&self) -> usize {
        self.host.len()
    }

    pub fn disk_pages(&self) -> usize {
        self.disk.as_ref().map_or(0, DiskTier::logical_pages)
    }

    pub fn io_stats(&self) -> TierIoStats {
        self.disk
            .as_ref()
            .map_or_else(TierIoStats::default, |disk| {
                let mut stats = disk.stats.clone();
                stats.submitted_write_bytes = stats.submitted_write_bytes.saturating_add(
                    disk.worker_stats
                        .submitted_write_bytes
                        .load(Ordering::Relaxed),
                );
                stats.metadata_write_bytes = stats.metadata_write_bytes.saturating_add(
                    disk.worker_stats
                        .metadata_write_bytes
                        .load(Ordering::Relaxed),
                );
                stats.completion_wait_ns = stats
                    .completion_wait_ns
                    .saturating_add(disk.worker_stats.write_wait_ns.load(Ordering::Relaxed));
                stats.failures = stats
                    .failures
                    .saturating_add(disk.worker_stats.failures.load(Ordering::Relaxed));
                stats
            })
    }

    pub fn location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        if self.host.contains_key(&key) {
            return Some(infer_seam::KvTierLocation::HostDemoted);
        }
        self.disk.as_ref().and_then(|disk| {
            (disk.keys.contains_key(&key) || disk.pending.contains_key(&key))
                .then_some(infer_seam::KvTierLocation::Disk)
        })
    }

    pub fn read_hits(&self) -> infer_seam::KvTierReadHits {
        infer_seam::KvTierReadHits {
            host_demoted: self.host_read_hits,
            disk: self.disk_read_hits,
        }
    }

    pub fn is_full(&self) -> bool {
        let host_full = self.host.len() >= self.host_capacity_pages;
        let disk_full = self
            .disk
            .as_ref()
            .is_none_or(|d| d.store.available_slots() == 0 || !d.accepting_writes);
        host_full && disk_full
    }

    pub fn contains(&self, key: u64) -> bool {
        self.host.contains_key(&key)
            || self
                .disk
                .as_ref()
                .is_some_and(|disk| disk.keys.contains_key(&key) || disk.pending.contains_key(&key))
    }

    pub fn insert(&mut self, key: u64, payload: Vec<u8>) -> bool {
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
        self.insert_inner(key, payload)
    }

    fn insert_inner(&mut self, key: u64, payload: Vec<u8>) -> bool {
        if self.host.len() < self.host_capacity_pages {
            self.insert_host(key, payload);
            return true;
        }
        if self.host_capacity_pages > 0 && self.spill_coldest_to_disk() {
            self.insert_host(key, payload);
            return true;
        }
        self.write_to_disk(key, payload).is_ok()
    }

    pub fn insert_many(&mut self, mut entries: Vec<(u64, Vec<u8>)>) -> bool {
        let keys = entries.iter().map(|(key, _)| *key).collect::<BTreeSet<_>>();
        if entries
            .iter()
            .any(|(_, payload)| payload.len() > self.bytes_per_page)
            || keys.len() != entries.len()
            || keys.iter().any(|key| self.contains(*key))
        {
            return false;
        }
        if self.host_capacity_pages == 0 {
            return self.write_many_to_disk(entries);
        }
        if self.insert_many_overflow(&mut entries) {
            return true;
        }
        let mut inserted = Vec::with_capacity(entries.len());
        for (key, payload) in entries {
            if !self.insert_inner(key, payload) {
                self.remove(&inserted);
                return false;
            }
            inserted.push(key);
        }
        true
    }

    fn insert_many_overflow(&mut self, entries: &mut [(u64, Vec<u8>)]) -> bool {
        if entries.is_empty() || self.host_capacity_pages == 0 || self.disk.is_none() {
            return false;
        }
        let keys = entries.iter().map(|(key, _)| *key).collect::<BTreeSet<_>>();
        if keys.len() != entries.len()
            || keys
                .iter()
                .any(|key| self.host.contains_key(key) || self.contains_disk(*key))
        {
            return false;
        }
        let overflow = self
            .host
            .len()
            .saturating_add(entries.len())
            .saturating_sub(self.host_capacity_pages);
        if overflow == 0 {
            for (key, payload) in entries {
                self.insert_host(*key, std::mem::take(payload));
            }
            return true;
        }
        let old_spills = self
            .host_lru
            .iter()
            .take(overflow.min(self.host.len()))
            .map(|(_, key)| *key)
            .collect::<Vec<_>>();
        let incoming_spills = overflow.saturating_sub(old_spills.len());
        let Some(disk) = self.disk.as_mut() else {
            return false;
        };
        let mut spills = old_spills
            .iter()
            .map(|key| (*key, self.host[key].payload.clone(), true))
            .collect::<Vec<_>>();
        spills.extend(
            entries
                .iter()
                .take(incoming_spills)
                .map(|(key, payload)| (*key, payload.clone(), true)),
        );
        if disk.enqueue(spills).is_err() {
            return false;
        }
        for key in old_spills {
            if let Some(entry) = self.host.remove(&key) {
                self.host_lru.remove(&(entry.stamp, key));
            }
        }
        for (key, payload) in entries.iter_mut().skip(incoming_spills) {
            self.insert_host(*key, std::mem::take(payload));
        }
        true
    }

    fn contains_disk(&self, key: u64) -> bool {
        self.disk
            .as_ref()
            .is_some_and(|disk| disk.keys.contains_key(&key) || disk.pending.contains_key(&key))
    }

    fn discount_disk_useful_write(&mut self, key: u64, bytes: usize) {
        if let Some(disk) = self.disk.as_mut() {
            if let Some(pending) = disk.pending.get_mut(&key) {
                if pending.useful {
                    disk.stats.useful_write_bytes =
                        disk.stats.useful_write_bytes.saturating_sub(bytes as u64);
                }
                pending.useful = false;
            } else if disk.keys.contains_key(&key) {
                disk.stats.useful_write_bytes =
                    disk.stats.useful_write_bytes.saturating_sub(bytes as u64);
            }
        }
    }

    fn discount_disk_useful_read(&mut self, key: u64, bytes: usize) {
        if let Some(disk) = self.disk.as_mut()
            && (disk.keys.contains_key(&key) || disk.pending.contains_key(&key))
        {
            disk.stats.useful_read_bytes =
                disk.stats.useful_read_bytes.saturating_sub(bytes as u64);
        }
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

    fn write_to_disk(&mut self, key: u64, payload: Vec<u8>) -> Result<(), Vec<u8>> {
        let Some(disk) = &mut self.disk else {
            return Err(payload);
        };
        disk.enqueue(vec![(key, payload, true)])
            .map_err(|mut entries| entries.pop().expect("single rejected disk write").1)
    }

    fn write_many_to_disk(&mut self, entries: Vec<(u64, Vec<u8>)>) -> bool {
        let Some(disk) = &mut self.disk else {
            return entries.is_empty();
        };
        disk.enqueue(
            entries
                .into_iter()
                .map(|(key, payload)| (key, payload, true))
                .collect(),
        )
        .is_ok()
    }

    fn spill_coldest_to_disk(&mut self) -> bool {
        let Some((stamp, key)) = self.host_lru.iter().next().copied() else {
            return false;
        };
        self.host_lru.remove(&(stamp, key));
        let Some(mut entry) = self.host.remove(&key) else {
            return false;
        };
        debug_assert_eq!(entry.stamp, stamp);
        let payload = entry.payload;
        match self.write_to_disk(key, payload) {
            Ok(()) => true,
            Err(payload) => {
                entry.payload = payload;
                self.host.insert(key, entry);
                self.host_lru.insert((stamp, key));
                false
            }
        }
    }

    /// Fetch a payload for promotion. Host and mmap hits borrow their payload;
    /// direct-I/O hits own the aligned read result. Disk hits are re-inserted
    /// into host L2 (evict-if-full) so repeated promotes don't re-read NVMe.
    pub fn read(&mut self, key: u64) -> Result<Cow<'_, [u8]>> {
        // Host hit: bump LRU, return borrowed payload.
        if let Some(old_stamp) = self.host.get(&key).map(|entry| entry.stamp) {
            self.host_read_hits += 1;
            let stamp = self.next_stamp();
            self.host_lru.remove(&(old_stamp, key));
            self.host_lru.insert((stamp, key));
            let entry = self.host.get_mut(&key).expect("key observed above");
            entry.stamp = stamp;
            return Ok(Cow::Borrowed(entry.payload.as_slice()));
        }
        let disk_payload = if let Some(disk) = self.disk.as_mut() {
            disk.drain_completions();
            if disk.pending.contains_key(&key) {
                let len = disk.pending[&key].payload.len();
                disk.stats.read_ops += 1;
                disk.stats.useful_read_bytes += len as u64;
                self.disk_read_hits += 1;
                (*disk.pending[&key].payload).clone()
            } else {
                let Some(record) = disk.keys.get(&key).copied() else {
                    return Err(anyhow!("KV tier store has no entry for key {key}"));
                };
                let len = if record.len == 0 {
                    self.bytes_per_page
                } else {
                    record.len.min(self.bytes_per_page)
                };
                let payload = disk.read_payload(record.slot, len)?.into_owned();
                self.disk_read_hits += 1;
                payload
            }
        } else {
            return Err(anyhow!("KV tier store has no entry for key {key}"));
        };
        // Promote to host L2 (best-effort: evict-if-full via insert_inner).
        self.insert_inner(key, disk_payload.clone());
        Ok(Cow::Owned(disk_payload))
    }

    pub fn read_many(&mut self, keys: &[u64]) -> Result<Vec<Vec<u8>>> {
        if let Some(disk) = self.disk.as_mut() {
            disk.drain_completions();
        }
        let mut values = vec![None; keys.len()];
        let mut disk_reads = Vec::new();
        for (index, key) in keys.iter().copied().enumerate() {
            if let Some(old_stamp) = self.host.get(&key).map(|entry| entry.stamp) {
                self.host_read_hits += 1;
                let stamp = self.next_stamp();
                self.host_lru.remove(&(old_stamp, key));
                self.host_lru.insert((stamp, key));
                let entry = self.host.get_mut(&key).expect("key observed above");
                entry.stamp = stamp;
                values[index] = Some(entry.payload.clone());
                continue;
            }
            if let Some(payload) = self
                .disk
                .as_ref()
                .and_then(|disk| disk.pending.get(&key))
                .map(|pending| pending.payload.as_ref().clone())
            {
                let disk = self
                    .disk
                    .as_mut()
                    .expect("pending disk payload observed above");
                disk.stats.read_ops += 1;
                disk.stats.useful_read_bytes += payload.len() as u64;
                self.disk_read_hits += 1;
                values[index] = Some(payload);
                continue;
            }
            let record = self
                .disk
                .as_ref()
                .and_then(|disk| disk.keys.get(&key))
                .copied()
                .ok_or_else(|| anyhow!("KV tier store has no entry for key {key}"))?;
            self.disk_read_hits += 1;
            let len = if record.len == 0 {
                self.bytes_per_page
            } else {
                record.len.min(self.bytes_per_page)
            };
            disk_reads.push((index, record.slot, len));
        }
        if !disk_reads.is_empty() {
            let reads = disk_reads
                .iter()
                .map(|(_, slot, len)| (*slot, *len))
                .collect::<Vec<_>>();
            let disk_values = self
                .disk
                .as_mut()
                .expect("disk records observed above")
                .read_payloads(&reads)?;
            for ((index, _, _), value) in disk_reads.into_iter().zip(disk_values) {
                values[index] = Some(value);
            }
        }
        Ok(values
            .into_iter()
            .map(|value| value.expect("every requested key resolved"))
            .collect())
    }

    /// Rank-LOCAL chunked-blob insert: one manifest page under
    /// `tier_key(ns, key)` + N chunk pages (`bytes_per_page` each) under
    /// `tier_key(ns_chunk, chunk_sub(key, i))`. On any failed insert removes
    /// everything already added and returns false. No collectives — callers
    /// own any TP consensus.
    pub fn insert_chunked(&mut self, ns: u64, ns_chunk: u64, key: u64, bytes: &[u8]) -> bool {
        debug_assert!(!bytes.is_empty(), "chunked blob must be non-empty");
        let chunks = bytes.len().div_ceil(self.bytes_per_page);
        // Refuse blobs whose chunk index would alias the next key's chunks
        // (chunk_sub packs idx into CHUNK_IDX_BITS) — unreachable at 16 MiB
        // pages, load-bearing for small-page callers of this general API.
        if chunks > (1 << CHUNK_IDX_BITS) || self.available_pages() <= chunks {
            return false;
        }
        let manifest_key = tier_key(ns, key);
        let manifest = chunk_manifest(chunks, bytes.len());
        let manifest_bytes = manifest.len();
        let mut entries = Vec::with_capacity(chunks + 1);
        entries.push((manifest_key, manifest));
        entries.extend(
            bytes
                .chunks(self.bytes_per_page)
                .enumerate()
                .map(|(idx, chunk)| (tier_key(ns_chunk, chunk_sub(key, idx)), chunk.to_vec())),
        );
        let inserted = self.insert_many(entries);
        if inserted {
            self.discount_disk_useful_write(manifest_key, manifest_bytes);
        }
        inserted
    }

    /// Insert pre-chunked entries (manifest first, then chunks) produced off-thread
    /// by [`chunk_manifest`] + `tier_key`/`chunk_sub`. Same capacity and dedup
    /// guards as `insert_chunked`; the expensive `to_vec` chunking ran elsewhere.
    pub fn insert_prechunked(
        &mut self,
        ns: u64,
        key: u64,
        chunks: usize,
        entries: Vec<(u64, Vec<u8>)>,
    ) -> bool {
        if chunks > (1 << CHUNK_IDX_BITS) || self.available_pages() <= chunks {
            return false;
        }
        let manifest_key = tier_key(ns, key);
        let manifest_bytes = entries.first().map(|(_, v)| v.len()).unwrap_or(0);
        let inserted = self.insert_many(entries);
        if inserted {
            self.discount_disk_useful_write(manifest_key, manifest_bytes);
        }
        inserted
    }

    /// Rank-LOCAL chunked-blob read-back, assembled owned. Err on any missing
    /// piece.
    pub fn read_chunked(&mut self, ns: u64, ns_chunk: u64, key: u64) -> Result<Vec<u8>> {
        let manifest_key = tier_key(ns, key);
        let manifest = self.read(manifest_key)?.into_owned();
        self.discount_disk_useful_read(manifest_key, manifest.len());
        let (chunks, total) = parse_chunk_manifest(&manifest)?;
        let keys = (0..chunks)
            .map(|idx| tier_key(ns_chunk, chunk_sub(key, idx)))
            .collect::<Vec<_>>();
        let mut bytes = Vec::with_capacity(total);
        for chunk in self.read_many(&keys)? {
            bytes.extend_from_slice(&chunk);
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
    pub fn remove_chunked(&mut self, ns: u64, ns_chunk: u64, key: u64) {
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

    /// Drop a key's DISK record only (host copy untouched) — the read-on-miss
    /// promote's cleanup: after the host re-insert, keeping the disk record
    /// would double-count the key in `disk_pages` and pin a spill slot.
    pub fn remove_disk_only(&mut self, key: u64) {
        let Some(disk) = &mut self.disk else {
            return;
        };
        disk.drain_completions();
        disk.pending.remove(&key);
        if let Some(record) = disk.keys.remove(&key) {
            disk.store.free_slot(record.slot);
            disk.manifest_dirty |= disk.durable;
        }
        disk.try_enqueue_manifest();
    }

    /// Drop entries from both levels. In the mmap store, freed slots return to
    /// the free list (no file unlink needed — the slot bytes are simply
    /// overwritten on next allocation).
    pub fn remove(&mut self, keys: &[u64]) {
        let mut disk_index_changed = false;
        if let Some(disk) = self.disk.as_mut() {
            disk.drain_completions();
        }
        for key in keys {
            if let Some(entry) = self.host.remove(key) {
                self.host_lru.remove(&(entry.stamp, *key));
            }
            if let Some(disk) = &mut self.disk {
                disk.pending.remove(key);
                if let Some(record) = disk.keys.remove(key) {
                    disk.store.free_slot(record.slot);
                    disk_index_changed = true;
                }
            }
        }
        if let Some(disk) = self.disk.as_mut() {
            disk.manifest_dirty |= disk_index_changed && disk.durable;
            disk.try_enqueue_manifest();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with_disk(host_pages: usize, page_bytes: usize) -> (KvTierStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let mut store = KvTierStore::with_budget(host_pages * page_bytes, page_bytes);
        assert!(store.set_disk(dir.path().to_path_buf(), 1024 * 1024, page_bytes));
        (store, dir)
    }

    fn payload(page_bytes: usize, fill: u8) -> Vec<u8> {
        vec![fill; page_bytes]
    }

    #[test]
    fn l2_insert_read_tracks_host_hits() {
        let mut store = KvTierStore::with_budget(4096, 1024);
        assert!(store.insert(1, payload(1024, 0xAA)));
        assert_eq!(store.host_demoted_pages(), 1);
        assert_eq!(store.read_hits().host_demoted, 0);

        let data = store.read(1).unwrap();
        assert_eq!(data.len(), 1024);
        assert_eq!(data[0], 0xAA);
        assert_eq!(store.read_hits().host_demoted, 1);
        assert_eq!(store.read_hits().disk, 0);

        // Second read increments again.
        let _ = store.read(1).unwrap();
        assert_eq!(store.read_hits().host_demoted, 2);
    }

    #[test]
    fn l3_spills_coldest_to_disk_on_host_full() {
        // Host capacity = 2 pages.
        let (mut store, _dir) = store_with_disk(2, 1024);
        assert!(store.insert(1, payload(1024, 0x01)));
        assert!(store.insert(2, payload(1024, 0x02)));
        assert_eq!(store.host_demoted_pages(), 2);
        assert_eq!(store.disk_pages(), 0);

        // Third insert spills key 1 (coldest) to disk.
        assert!(store.insert(3, payload(1024, 0x03)));
        assert_eq!(store.host_demoted_pages(), 2);
        assert_eq!(store.disk_pages(), 1);
        assert_eq!(store.location(1), Some(infer_seam::KvTierLocation::Disk));
        assert_eq!(
            store.location(2),
            Some(infer_seam::KvTierLocation::HostDemoted)
        );
    }

    #[test]
    fn l3_disk_read_promotes_back_to_host() {
        let (mut store, _dir) = store_with_disk(1, 1024);
        assert!(store.insert(1, payload(1024, 0x01)));

        // Second insert spills key 1 to disk (host capacity = 1).
        assert!(store.insert(2, payload(1024, 0x02)));
        assert_eq!(store.location(1), Some(infer_seam::KvTierLocation::Disk));
        assert_eq!(store.read_hits().disk, 0);

        // Read key 1 from disk → promote-on-read re-inserts into host.
        let data = store.read(1).unwrap();
        assert_eq!(data[0], 0x01);
        assert_eq!(store.read_hits().disk, 1);
        // Key 1 is now back in host (key 2 was evicted to disk).
        assert_eq!(
            store.location(1),
            Some(infer_seam::KvTierLocation::HostDemoted)
        );

        // Second read is a host hit, not a disk hit.
        let _ = store.read(1).unwrap();
        assert_eq!(store.read_hits().disk, 1);
        assert_eq!(store.read_hits().host_demoted, 1);
    }

    #[test]
    fn read_many_tracks_mixed_hits() {
        let (mut store, _dir) = store_with_disk(2, 1024);
        assert!(store.insert(1, payload(1024, 0x01)));
        assert!(store.insert(2, payload(1024, 0x02)));
        // Spill key 1 to disk.
        assert!(store.insert(3, payload(1024, 0x03)));

        let values = store.read_many(&[1, 2, 3]).unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0][0], 0x01); // disk hit (promoted to host)
        assert_eq!(values[1][0], 0x02); // host hit
        assert_eq!(values[2][0], 0x03); // host hit
        assert_eq!(store.read_hits().disk, 1);
        assert_eq!(store.read_hits().host_demoted, 2);
    }

    #[test]
    fn location_returns_none_for_missing_key() {
        let (store, _dir) = store_with_disk(2, 1024);
        assert_eq!(store.location(999), None);
    }

    #[test]
    fn remove_drops_entries_from_both_levels() {
        let (mut store, _dir) = store_with_disk(2, 1024);
        assert!(store.insert(1, payload(1024, 0x01)));
        assert!(store.insert(2, payload(1024, 0x02)));
        assert!(store.insert(3, payload(1024, 0x03))); // spills key 1

        store.remove(&[1, 2]);
        assert!(!store.contains(1));
        assert!(!store.contains(2));
        assert!(store.contains(3));
        assert_eq!(store.host_demoted_pages(), 1);
        assert_eq!(store.disk_pages(), 0);
    }

    #[test]
    fn host_only_store_works_without_disk() {
        let mut store = KvTierStore::with_budget(4096, 1024);
        assert!(store.insert(1, payload(1024, 0xAA)));
        assert!(store.insert(2, payload(1024, 0xBB)));
        assert_eq!(store.host_demoted_pages(), 2);
        assert_eq!(store.disk_pages(), 0);

        let data = store.read(1).unwrap();
        assert_eq!(data[0], 0xAA);
        assert_eq!(store.read_hits().host_demoted, 1);
        assert_eq!(store.read_hits().disk, 0);
    }
}

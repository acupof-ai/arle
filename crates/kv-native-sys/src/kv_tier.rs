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

// ---- Budget helpers (unchanged) ----

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

// ---- chunked-blob key layout ----

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

// ---- KvTierStore ----

pub struct KvTierStore {
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

// ---- DiskTier ----

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

    fn free_slot(&mut self, slot: u32) {
        match self {
            Self::Mmap(store) => store.free_slot(slot),
            #[cfg(target_os = "linux")]
            Self::Direct(store) => store.free_slot(slot),
        }
    }

    fn reserve_indices(&mut self, indices: &[u32]) {
        match self {
            Self::Mmap(store) => store.reserve_indices(indices),
            #[cfg(target_os = "linux")]
            Self::Direct(store) => store.reserve_indices(indices),
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
    _lock: Option<nix::fcntl::Flock<std::fs::File>>,
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
    fn lock_namespace(namespace: &Path) -> Result<nix::fcntl::Flock<std::fs::File>> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(namespace.join("owner.lock"))?;
        nix::fcntl::Flock::lock(file, nix::fcntl::FlockArg::LockExclusiveNonblock)
            .map_err(|(_, err)| anyhow!("namespace already owned: {err}"))
    }

    fn manifest_path(&self) -> PathBuf {
        self.root_dir.join(MANIFEST_FILE)
    }

    fn write_manifest(&mut self) -> Result<()> {
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
        let bytes = buf.len() as u64;
        let result = crate::write_file_atomic_cache(&self.manifest_path(), buf.as_bytes())
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

    fn write_payloads(&mut self, writes: &[(u32, &[u8])]) -> std::io::Result<()> {
        let useful = writes
            .iter()
            .map(|(_, bytes)| bytes.len() as u64)
            .sum::<u64>();
        let result = match &mut self.store {
            DiskStore::Mmap(store) => {
                for (slot, bytes) in writes {
                    store.write_slot(*slot, bytes)?;
                }
                self.stats.submitted_write_bytes =
                    self.stats.submitted_write_bytes.saturating_add(useful);
                Ok(())
            }
            #[cfg(target_os = "linux")]
            DiskStore::Direct(store) => store.write_slots(writes).map(|batch| {
                self.stats.submitted_write_bytes = self
                    .stats
                    .submitted_write_bytes
                    .saturating_add(batch.submitted_bytes);
                self.stats.completion_wait_ns =
                    self.stats.completion_wait_ns.saturating_add(batch.wait_ns);
            }),
        };
        self.stats.write_ops = self.stats.write_ops.saturating_add(writes.len() as u64);
        if result.is_ok() {
            self.stats.useful_write_bytes = self.stats.useful_write_bytes.saturating_add(useful);
        } else {
            self.stats.failures = self.stats.failures.saturating_add(1);
        }
        result
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

// ---- KvTierStore impl ----

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

    pub fn set_disk_durable(
        &mut self,
        root: PathBuf,
        budget_bytes: usize,
        bytes_per_page: usize,
        epoch: String,
        format_tag: u8,
        world_size: usize,
        rank: usize,
    ) -> bool {
        let namespace =
            Self::durable_namespace(root, &epoch, format_tag, world_size, rank, bytes_per_page);
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
                self.disk = Some(DiskTier {
                    root_dir: namespace,
                    store,
                    keys: BTreeMap::new(),
                    durable,
                    epoch,
                    stats,
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

        self.disk = Some(DiskTier {
            root_dir: namespace,
            store,
            keys,
            durable: true,
            epoch,
            stats,
            _lock: Some(namespace_lock),
        });
        true
    }

    pub fn persist(&mut self) {
        if let Some(disk) = self.disk.as_mut()
            && disk.durable
            && let Err(err) = disk.write_manifest()
        {
            log::warn!("KV recall manifest persist failed: {err}");
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
        self.capacity_pages()
            .saturating_sub(self.host.len().saturating_add(self.disk_pages()))
    }

    pub fn page_bytes(&self) -> usize {
        self.bytes_per_page
    }

    pub fn host_demoted_pages(&self) -> usize {
        self.host.len()
    }

    pub fn disk_pages(&self) -> usize {
        self.disk.as_ref().map_or(0, |d| d.keys.len())
    }

    pub fn io_stats(&self) -> TierIoStats {
        self.disk
            .as_ref()
            .map_or_else(TierIoStats::default, |disk| disk.stats.clone())
    }

    pub fn location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        if self.host.contains_key(&key) {
            return Some(infer_seam::KvTierLocation::HostDemoted);
        }
        self.disk.as_ref().and_then(|disk| {
            disk.keys
                .contains_key(&key)
                .then_some(infer_seam::KvTierLocation::Disk)
        })
    }

    pub fn is_full(&self) -> bool {
        let host_full = self.host.len() >= self.host_capacity_pages;
        let disk_full = self
            .disk
            .as_ref()
            .is_none_or(|d| d.keys.len() >= d.store.num_slots() as usize);
        host_full && disk_full
    }

    pub fn contains(&self, key: u64) -> bool {
        self.host.contains_key(&key)
            || self
                .disk
                .as_ref()
                .is_some_and(|disk| disk.keys.contains_key(&key))
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
        self.insert_inner(key, payload, true)
    }

    fn insert_inner(&mut self, key: u64, payload: Vec<u8>, commit_manifest: bool) -> bool {
        if self.host.len() < self.host_capacity_pages {
            self.insert_host(key, payload);
            return true;
        }
        if self.host_capacity_pages > 0 && self.spill_coldest_to_disk(commit_manifest) {
            self.insert_host(key, payload);
            return true;
        }
        self.write_to_disk(key, &payload, commit_manifest)
    }

    pub fn insert_many(&mut self, mut entries: Vec<(u64, Vec<u8>)>) -> bool {
        if entries
            .iter()
            .any(|(_, payload)| payload.len() > self.bytes_per_page)
        {
            return false;
        }
        if self.host_capacity_pages == 0 && self.write_many_to_disk(&entries) {
            return true;
        }
        if self.insert_many_overflow(&mut entries) {
            return true;
        }
        let mut inserted = Vec::with_capacity(entries.len());
        for (key, payload) in entries {
            if !self.insert_inner(key, payload, false) {
                self.remove(&inserted);
                return false;
            }
            inserted.push(key);
        }
        self.commit_manifest();
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
        let mut slots = Vec::with_capacity(overflow);
        for _ in 0..overflow {
            let Some(slot) = disk.store.alloc_slot() else {
                for slot in slots {
                    disk.store.free_slot(slot);
                }
                return false;
            };
            slots.push(slot);
        }
        let mut writes = Vec::with_capacity(overflow);
        let mut records = Vec::with_capacity(overflow);
        for (key, slot) in old_spills.iter().copied().zip(slots.iter().copied()) {
            let payload = &self.host[&key].payload;
            writes.push((slot, payload.as_slice()));
            records.push((key, slot, payload.len()));
        }
        for ((key, payload), slot) in entries
            .iter()
            .take(incoming_spills)
            .zip(slots.iter().copied().skip(old_spills.len()))
        {
            writes.push((slot, payload.as_slice()));
            records.push((*key, slot, payload.len()));
        }
        if let Err(err) = disk.write_payloads(&writes) {
            log::warn!("KV overflow batch disk write failed: {err}");
            for slot in slots {
                disk.store.free_slot(slot);
            }
            return false;
        }
        for (key, slot, len) in records {
            disk.keys.insert(key, DiskRecord { slot, len });
        }
        if disk.durable
            && let Err(err) = disk.write_manifest()
        {
            log::warn!("KV recall batch manifest update failed: {err}");
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
            .is_some_and(|disk| disk.keys.contains_key(&key))
    }

    fn discount_disk_useful_write(&mut self, key: u64, bytes: usize) {
        if let Some(disk) = self.disk.as_mut()
            && disk.keys.contains_key(&key)
        {
            disk.stats.useful_write_bytes =
                disk.stats.useful_write_bytes.saturating_sub(bytes as u64);
        }
    }

    fn discount_disk_useful_read(&mut self, key: u64, bytes: usize) {
        if let Some(disk) = self.disk.as_mut()
            && disk.keys.contains_key(&key)
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

    fn write_to_disk(&mut self, key: u64, payload: &[u8], commit_manifest: bool) -> bool {
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
        if let Err(err) = disk.write_payloads(&[(slot, payload)]) {
            log::warn!("KV disk write failed for key {key} slot {slot}: {err}");
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
        if commit_manifest
            && disk.durable
            && let Err(err) = disk.write_manifest()
        {
            log::warn!("KV recall manifest update failed for key {key}: {err}");
        }
        true
    }

    fn write_many_to_disk(&mut self, entries: &[(u64, Vec<u8>)]) -> bool {
        let Some(disk) = &mut self.disk else {
            return entries.is_empty();
        };
        let unique = entries.iter().map(|(key, _)| *key).collect::<BTreeSet<_>>();
        if unique.len() != entries.len() {
            return false;
        }
        let mut allocated = Vec::new();
        let mut records = Vec::with_capacity(entries.len());
        for (key, payload) in entries {
            let already_present = disk.keys.get(key).copied();
            let slot = match already_present {
                Some(record) => record.slot,
                None => match disk.store.alloc_slot() {
                    Some(slot) => {
                        allocated.push(slot);
                        slot
                    }
                    None => {
                        for slot in allocated {
                            disk.store.free_slot(slot);
                        }
                        return false;
                    }
                },
            };
            records.push((*key, slot, payload.len()));
        }
        let writes = records
            .iter()
            .zip(entries)
            .map(|((_, slot, _), (_, payload))| (*slot, payload.as_slice()))
            .collect::<Vec<_>>();
        if let Err(err) = disk.write_payloads(&writes) {
            log::warn!("KV batch disk write failed: {err}");
            for slot in allocated {
                disk.store.free_slot(slot);
            }
            return false;
        }
        for (key, slot, len) in records {
            disk.keys.insert(key, DiskRecord { slot, len });
        }
        if disk.durable
            && let Err(err) = disk.write_manifest()
        {
            log::warn!("KV recall batch manifest update failed: {err}");
        }
        true
    }

    fn commit_manifest(&mut self) {
        if let Some(disk) = self.disk.as_mut()
            && disk.durable
            && let Err(err) = disk.write_manifest()
        {
            log::warn!("KV recall batch manifest update failed: {err}");
        }
    }

    fn spill_coldest_to_disk(&mut self, commit_manifest: bool) -> bool {
        let Some((stamp, key)) = self.host_lru.iter().next().copied() else {
            return false;
        };
        self.host_lru.remove(&(stamp, key));
        let Some(entry) = self.host.remove(&key) else {
            return false;
        };
        debug_assert_eq!(entry.stamp, stamp);
        if self.write_to_disk(key, &entry.payload, commit_manifest) {
            true
        } else {
            self.host.insert(key, entry);
            self.host_lru.insert((stamp, key));
            false
        }
    }

    /// Fetch a payload for promotion. Host and mmap hits borrow their payload;
    /// direct-I/O hits own the aligned read result.
    pub fn read(&mut self, key: u64) -> Result<Cow<'_, [u8]>> {
        // Host hit: bump LRU, return owned payload.
        if let Some(old_stamp) = self.host.get(&key).map(|entry| entry.stamp) {
            let stamp = self.next_stamp();
            self.host_lru.remove(&(old_stamp, key));
            self.host_lru.insert((stamp, key));
            let entry = self.host.get_mut(&key).expect("key observed above");
            entry.stamp = stamp;
            return Ok(Cow::Borrowed(entry.payload.as_slice()));
        }
        if let Some(disk) = self.disk.as_mut()
            && let Some(record) = disk.keys.get(&key).copied()
        {
            let len = if record.len == 0 {
                self.bytes_per_page
            } else {
                record.len.min(self.bytes_per_page)
            };
            return Ok(disk.read_payload(record.slot, len)?);
        }
        Err(anyhow!("KV tier store has no entry for key {key}"))
    }

    pub fn read_many(&mut self, keys: &[u64]) -> Result<Vec<Vec<u8>>> {
        let mut values = vec![None; keys.len()];
        let mut disk_reads = Vec::new();
        for (index, key) in keys.iter().copied().enumerate() {
            if let Some(old_stamp) = self.host.get(&key).map(|entry| entry.stamp) {
                let stamp = self.next_stamp();
                self.host_lru.remove(&(old_stamp, key));
                self.host_lru.insert((stamp, key));
                let entry = self.host.get_mut(&key).expect("key observed above");
                entry.stamp = stamp;
                values[index] = Some(entry.payload.clone());
                continue;
            }
            let record = self
                .disk
                .as_ref()
                .and_then(|disk| disk.keys.get(&key))
                .copied()
                .ok_or_else(|| anyhow!("KV tier store has no entry for key {key}"))?;
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
        let Some(record) = disk.keys.remove(&key) else {
            return;
        };
        disk.store.free_slot(record.slot);
        if disk.durable
            && let Err(err) = disk.write_manifest()
        {
            log::warn!("KV recall manifest update failed dropping key {key}: {err}");
        }
    }

    /// Drop entries from both levels. In the mmap store, freed slots return to
    /// the free list (no file unlink needed — the slot bytes are simply
    /// overwritten on next allocation).
    pub fn remove(&mut self, keys: &[u64]) {
        let mut disk_index_changed = false;
        for key in keys {
            if let Some(entry) = self.host.remove(key) {
                self.host_lru.remove(&(entry.stamp, *key));
            }
            if let Some(disk) = &mut self.disk
                && let Some(record) = disk.keys.remove(key)
            {
                disk.store.free_slot(record.slot);
                disk_index_changed = true;
            }
        }
        if disk_index_changed
            && let Some(disk) = self.disk.as_mut()
            && disk.durable
            && let Err(err) = disk.write_manifest()
        {
            log::warn!("KV recall manifest update after remove failed: {err}");
        }
    }
}

/// Cheap content-version tag for a model checkpoint directory.
pub fn weights_epoch_tag(model_path: &Path) -> String {
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
                if let Ok(modified) = meta.modified()
                    && let Ok(dur) = modified.duration_since(std::time::UNIX_EPOCH)
                {
                    mix(&dur.as_secs().to_le_bytes());
                    mix(&dur.subsec_nanos().to_le_bytes());
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
        let mut store = KvTierStore::with_budget(16, 8);
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
        let mut store = KvTierStore::with_budget(32 * 10, 32);
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
        let mut store = KvTierStore::with_budget(32 * 20, 32);
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
        let mut store = KvTierStore::with_budget(0, 32);
        assert!(store.set_disk(root.clone(), 32 * 20, 32));
        let baseline = store.available_pages();
        let blob: Vec<u8> = (0..80u8).collect();
        assert!(store.insert_chunked(3, 4, 1, &blob));
        assert_eq!(store.disk_pages(), 4, "manifest + 3 chunks on disk");
        assert_eq!(store.io_stats().useful_write_bytes, blob.len() as u64);
        assert_eq!(store.read_chunked(3, 4, 1).expect("batch read"), blob);
        assert_eq!(store.io_stats().useful_read_bytes, blob.len() as u64);
        assert!(store.insert_chunked(3, 4, 2, &blob));
        store.remove_chunked(3, 4, 1);
        assert_eq!(store.disk_pages(), 4, "superseded disk blob reclaimed");
        assert_eq!(store.available_pages(), baseline - 4);
        assert_eq!(store.read_chunked(3, 4, 2).expect("v2 via mmap"), blob);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn remove_chunked_tolerates_missing_manifest() {
        let mut store = KvTierStore::with_budget(32 * 10, 32);
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
        let mut store = KvTierStore::with_budget(0, page);
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
        let stats = store.io_stats();
        let write_wa = stats
            .submitted_write_bytes
            .saturating_add(stats.metadata_write_bytes) as f64
            / stats.useful_write_bytes.max(1) as f64;
        let read_amplification =
            stats.submitted_read_bytes as f64 / stats.useful_read_bytes.max(1) as f64;
        println!(
            "chunked-blob disk bandwidth over {gib:.1} GiB: \
             write {:.2} GiB/s ({write:?}), read(warm) {:.2} GiB/s ({read:?}); \
             mode={:?} useful_read={} useful_write={} submitted_read={} submitted_write={} \
             metadata_write={} write_wa={write_wa:.4} read_amplification={read_amplification:.4} \
             completion_wait_ns={} fallback={:?}",
            gib / write.as_secs_f64(),
            gib / read.as_secs_f64(),
            stats.mode,
            stats.useful_read_bytes,
            stats.useful_write_bytes,
            stats.submitted_read_bytes,
            stats.submitted_write_bytes,
            stats.metadata_write_bytes,
            stats.completion_wait_ns,
            stats.fallback_reason,
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
        let mut store = KvTierStore::with_budget(0, 8);
        assert!(store.set_disk(root.clone(), 64, 8));
        assert!(!store.insert(1, vec![0; 13]), "oversize refused, no panic");
        assert_eq!(store.disk_pages(), 0);
        // Host shape: refused up front too (one contract for both levels).
        let mut host_store = KvTierStore::with_budget(64, 8);
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
    fn disk_mode_defaults_to_mmap() {
        assert_eq!(parse_disk_mode(None), None);
        assert_eq!(parse_disk_mode(Some("mmap")), Some(DiskIoMode::Mmap));
        assert_eq!(parse_disk_mode(Some("direct")), Some(DiskIoMode::Direct));
    }

    #[test]
    fn host_overflow_spills_coldest_to_disk_and_reads_back() {
        let root = temp_root("spill");
        let mut store = KvTierStore::with_budget(16, 8);
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
    fn batch_overflow_submits_disk_pages_together() {
        let root = temp_root("batch_overflow");
        let mut store = KvTierStore::with_budget(16, 8);
        assert!(store.set_disk(root.clone(), 32, 8));
        assert!(store.insert(1, vec![1; 8]));
        assert!(store.insert_many(vec![(2, vec![2; 8]), (3, vec![3; 8]), (4, vec![4; 8])]));
        assert_eq!(store.disk_pages(), 2);
        assert_eq!(
            store.read_many(&[1, 2, 3, 4]).expect("batch read")[0],
            vec![1; 8]
        );
        let stats = store.io_stats();
        assert_eq!(stats.write_ops, 2);
        assert_eq!(stats.useful_write_bytes, 16);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn disk_read_is_zero_copy_mmap_slice() {
        let root = temp_root("mmap_read");
        let mut store = KvTierStore::with_budget(0, 8);
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
        let mut store = KvTierStore::with_budget(0, 16);
        assert!(store.set_disk(root.clone(), 32, 16));
        assert!(store.insert(7, b"short".to_vec()));
        let read = store.read(7).expect("disk read");
        assert_eq!(read.as_ref(), b"short");
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn disk_full_allows_replacing_existing_key() {
        let root = temp_root("replace");
        let mut store = KvTierStore::with_budget(0, 8);
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
        let mut store = KvTierStore::with_budget(0, 8);
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
        let mut store = KvTierStore::with_budget(8, 4);
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
        let mut store = KvTierStore::with_budget(8, 8);
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
        let mut store = KvTierStore::with_budget(0, 8);
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
    fn durable_manifest_round_trips_disk_index_across_processes() {
        const MODE: &str = "ARLE_KV_TIER_CROSS_PROCESS_MODE";
        const ROOT: &str = "ARLE_KV_TIER_CROSS_PROCESS_ROOT";
        let epoch = "epoch-A".to_string();
        if let Some((mode, root)) = std::env::var_os(MODE).zip(std::env::var_os(ROOT)) {
            let root = PathBuf::from(root);
            let mode = mode.to_string_lossy();
            std::fs::write(
                root.join(format!("{mode}.pid")),
                std::process::id().to_string(),
            )
            .expect("write child pid");
            let mut store = KvTierStore::with_budget(0, 8);
            match mode.as_ref() {
                "write" => {
                    assert!(store.set_disk_durable(root, 32, 8, epoch, 1, 4, 3));
                    assert!(store.insert(1, vec![1; 8]));
                    assert!(store.insert(2, vec![2; 8]));
                    assert!(store.insert(3, vec![3; 8]));
                }
                "read" => {
                    assert!(store.load(root, 32, 8, epoch, 1, 4, 3));
                    assert_eq!(store.disk_pages(), 3);
                    assert_eq!(store.read(1).expect("reload read 1").as_ref(), &[1u8; 8]);
                    assert_eq!(store.read(3).expect("reload read 3").as_ref(), &[3u8; 8]);
                    assert!(store.insert(4, vec![4; 8]));
                    assert_eq!(store.read(4).expect("new read 4").as_ref(), &[4u8; 8]);
                }
                "contend" => assert!(!store.set_disk_durable(root, 32, 8, epoch, 1, 4, 3)),
                mode => panic!("unknown cross-process mode {mode}"),
            }
            return;
        }

        let root = temp_root("durable_mmap");
        let test = "kv_tier::tests::durable_manifest_round_trips_disk_index_across_processes";
        let run = |mode| {
            std::process::Command::new(std::env::current_exe().expect("test binary"))
                .args(["--exact", test])
                .env(MODE, mode)
                .env(ROOT, &root)
                .status()
                .expect("spawn test child")
        };
        let mut owner = KvTierStore::with_budget(0, 8);
        assert!(owner.set_disk_durable(root.clone(), 32, 8, epoch, 1, 4, 3));
        assert!(run("contend").success(), "lock contender child failed");
        drop(owner);
        for mode in ["write", "read"] {
            assert!(run(mode).success(), "{mode} child failed");
        }
        assert!(
            root.join("arle-kv-recall-epoch-A-format-1-world-4-rank-3-page-8")
                .exists()
        );
        let writer = std::fs::read_to_string(root.join("write.pid")).expect("writer pid");
        let reader = std::fs::read_to_string(root.join("read.pid")).expect("reader pid");
        assert_ne!(
            writer, reader,
            "write and reload must run in different PIDs"
        );
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn durable_manifest_round_trips_variable_payload_length() {
        let root = temp_root("durable_varlen");
        let epoch = "epoch-A".to_string();

        {
            let mut store = KvTierStore::with_budget(0, 16);
            assert!(store.set_disk_durable(root.clone(), 32, 16, epoch.clone(), 1, 1, 0));
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
            let mut store = KvTierStore::with_budget(0, 16);
            let reloaded = store.load(root.clone(), 32, 16, epoch.clone(), 1, 1, 0);
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

    #[test]
    fn durable_load_with_changed_capacity_starts_cold() {
        let root = temp_root("durable_geometry");
        let epoch = "epoch-A".to_string();
        {
            let mut store = KvTierStore::with_budget(0, 8);
            assert!(store.set_disk_durable(root.clone(), 32, 8, epoch.clone(), 1, 1, 0));
            assert!(store.insert(1, vec![1; 8]));
        }
        let mut store = KvTierStore::with_budget(0, 8);
        assert!(store.load(root.clone(), 64, 8, epoch, 1, 1, 0));
        assert_eq!(store.disk_pages(), 0);
        assert!(store.read(1).is_err());
        assert!(store.insert(2, vec![2; 8]));
        drop(store);
        std::fs::remove_dir_all(&root).expect("cleanup");
    }
}

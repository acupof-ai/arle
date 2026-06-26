//! Two-level host store for demoted prefix-KV pages.
//!
//! Host-demoted pages live in a capacity-capped in-RAM map (default-on, 4 GiB).
//! Disk spill is optional on the `kv-native-sys` block substrate
//! (`--kv-ssd-path`, opt-in): when host RAM fills, the coldest host entry spills
//! to a fingerprint-named block file, so the capacity the engine sees is
//! host-demoted + disk pages. Payloads are full
//! `PagedKVPool` page images; this module never touches the device.
//!
//! ## Durability (recall tier)
//!
//! The prefix tier uses a per-process *ephemeral* disk namespace that is wiped
//! on drop (a crash-safe cache the engine can always rebuild). The session
//! KV-recall tier instead opts into a **durable** disk namespace ([`set_disk`]
//! with `durable = true`): the namespace is stable across restarts, survives
//! drop, and carries a [`MANIFEST_FILE`] persisting each disk-resident block's
//! `{key, byte_len}` plus an epoch tag. [`CudaKvTierStore::load`] replays the
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
/// one `key byte_len` line per disk-resident block so a restart can rebuild the
/// in-memory disk index and re-address the otherwise-orphaned sharded blobs.
const MANIFEST_FILE: &str = "manifest.kvm";

/// Manifest header magic + version. A mismatch — or a missing manifest — makes
/// [`CudaKvTierStore::load`] start cold rather than trust a foreign layout.
const MANIFEST_MAGIC: &str = "ARLE-KVTIER-MANIFEST-V1";

static DISK_TIER_NAMESPACE_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

/// Read a `/proc/meminfo` field (e.g. `MemAvailable`, `MemTotal`) in bytes.
///
/// `None` off Linux or when the field is unreadable, so a `/proc`-less build
/// (a Mac CUDA-typecheck) makes [`dram_l2_budget`] fall back to its floor.
fn meminfo_field_bytes(field: &str) -> Option<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let prefix = format!("{field}:");
    for line in meminfo.lines() {
        // "MemAvailable:   12345678 kB"
        if let Some(rest) = line.strip_prefix(&prefix) {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// Bytes of *available* system RAM (Linux `/proc/meminfo` `MemAvailable`).
fn available_ram_bytes() -> Option<usize> {
    meminfo_field_bytes("MemAvailable")
}

/// Bytes of *total* system RAM (Linux `/proc/meminfo` `MemTotal`) — the base
/// for the L2 reserve (`reserve = max(64 GiB, 0.15 × total_ram)`).
fn total_ram_bytes() -> Option<usize> {
    meminfo_field_bytes("MemTotal")
}

/// One `statvfs` of `path`, returning `(free_bytes, total_bytes)` for the
/// filesystem holding it. `None` on non-unix or probe failure → the L3 budget
/// falls back to its floor.
///
/// - free  = `f_bavail × f_frsize` (blocks available to unprivileged users)
/// - total = `f_blocks × f_frsize` (all data blocks — the base for the reserve)
#[cfg(unix)]
fn disk_free_total_bytes(path: &Path) -> Option<(usize, usize)> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated path; `statvfs` writes into
    // the zeroed buffer and returns 0 on success (non-zero ⇒ bail to `None`).
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

/// Machine-derived **L2 (host DRAM)** budget for demoted prefix/recall pages —
/// replaces the hardcoded [1,4] GiB cap. Default-on (ckl 2026-06-11): evicted
/// pages demote into host RAM and promote back on the next hit instead of
/// re-prefilling (`--kv-t1-budget-bytes 0` opts out). Profiled from measured
/// hardware with a reserve so the **shared, pageable** host store never swaps or
/// OOM-kills co-tenants: `budget = clamp(dram_fraction × MemAvailable, [4 GiB,
/// MemAvailable − reserve])`, `reserve = max(64 GiB, 0.15 × MemTotal)`. A probe
/// miss (off Linux) falls back to the 4 GiB floor.
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

/// Machine-derived **L3 (NVMe)** spill budget for the disk tier rooted at `root`
/// — replaces the hardcoded 20 GiB cap. Applies when `--kv-ssd-path` is set
/// without `--kv-ssd-max-bytes`: `budget = clamp(ssd_fraction × free_disk,
/// [8 GiB, free_disk − reserve])`, `reserve = max(50 GiB, 0.1 × total_disk)`.
/// A probe miss falls back to the 8 GiB floor.
pub fn default_t2_budget_bytes(root: &Path, ssd_fraction: f64) -> usize {
    let (free, total) = match disk_free_total_bytes(root) {
        Some((free, total)) => (Some(free), Some(total)),
        None => (None, None),
    };
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

pub(crate) struct CudaKvTierStore {
    host_capacity_pages: usize,
    bytes_per_page: usize,
    /// Host-demoted entries: key -> touch stamp + page payload.
    host: BTreeMap<u64, HostDemotedEntry>,
    /// Ordered by (touch stamp, key) for O(log n) coldest selection.
    host_lru: BTreeSet<(u64, u64)>,
    clock: u64,
    disk: Option<DiskTier>,
    read_scratch: Vec<u8>,
}

struct HostDemotedEntry {
    stamp: u64,
    payload: Vec<u8>,
}

struct DiskTier {
    /// Process-owned namespace under the operator-provided root.
    root: PathBuf,
    capacity_pages: usize,
    /// Keys whose payloads currently live on disk, mapped to each blob's byte
    /// length (the metadata the manifest persists alongside the key so a
    /// restart can re-address + size-check the sharded blob).
    keys: BTreeMap<u64, u64>,
    /// Durable tier (the recall store): suppress the drop-time wipe, persist a
    /// manifest under the namespace, and stamp it with `epoch`. Ephemeral
    /// (prefix-tier) namespaces leave this `false` and are wiped on drop.
    durable: bool,
    /// Model/weights-version tag written into the manifest. A `load` whose
    /// requested epoch differs ignores the prior manifest (stale-memory guard).
    epoch: String,
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        // Durable namespaces (recall) MUST survive a restart — never wipe.
        // Ephemeral prefix-tier namespaces are a rebuildable cache, so the
        // process that owns them cleans up on drop.
        if !self.durable {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
}

impl DiskTier {
    fn manifest_path(&self) -> PathBuf {
        self.root.join(MANIFEST_FILE)
    }

    /// Rewrite the manifest from the current key set (atomic temp+rename via
    /// `kv-native-sys`). Cheap: one small text file, only the disk index — not
    /// the blobs. Called on every disk insert/remove for a durable tier and on
    /// the explicit `persist()`.
    fn write_manifest(&self) -> Result<()> {
        let mut buf = String::with_capacity(64 + self.keys.len() * 24);
        buf.push_str(MANIFEST_MAGIC);
        buf.push('\n');
        buf.push_str(&self.epoch);
        buf.push('\n');
        for (key, byte_len) in &self.keys {
            // One record per line: `<key> <byte_len>`.
            writeln!(&mut buf, "{key} {byte_len}").expect("write to String never fails");
        }
        kv_native_sys::write_file_atomic_cache(&self.manifest_path(), buf.as_bytes())
            .with_context(|| format!("KV recall manifest write under {}", self.root.display()))
    }

    /// Parse a manifest file's bytes into `(epoch, [(key, byte_len)])`. Returns
    /// `None` for a missing/empty/foreign-magic file (caller starts cold).
    #[allow(dead_code)] // WIP: durable KV-recall manifest replay, not yet wired
    fn parse_manifest(bytes: &[u8]) -> Option<(String, Vec<(u64, u64)>)> {
        let text = std::str::from_utf8(bytes).ok()?;
        let mut lines = text.lines();
        if lines.next()? != MANIFEST_MAGIC {
            return None;
        }
        let epoch = lines.next()?.to_string();
        let mut records = Vec::new();
        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let key: u64 = parts.next()?.parse().ok()?;
            let byte_len: u64 = parts.next()?.parse().ok()?;
            records.push((key, byte_len));
        }
        Some((epoch, records))
    }
}

fn fingerprint(key: u64) -> [u8; 16] {
    let mut f = [0u8; 16];
    f[..8].copy_from_slice(&key.to_le_bytes());
    f
}

/// Cheap content-version tag for a model checkpoint directory, used to stamp the
/// durable KV-recall manifest. Hashes each `*.safetensors` file's name, length,
/// and mtime (FNV-1a, deterministic, no crypto needed). An OPD weight update
/// rewrites the checkpoint so its lengths/mtimes shift, the tag changes, and a
/// restart drops the now-stale recalled KV (per prefix-cache-stale-across-weight
/// -epochs). A probe miss (unreadable dir) yields a stable fallback tag so recall
/// still works; it just won't auto-invalidate on a silent weight swap.
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
        // Collect + sort so directory-order non-determinism never changes the
        // tag for an unchanged checkpoint.
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
        // No safetensors enumerated (probe miss): a stable, path-derived tag so
        // recall still functions across restarts of the same model dir.
        mix(model_path.to_string_lossy().as_bytes());
        format!("path-{hash:016x}")
    }
}

impl Drop for CudaKvTierStore {
    fn drop(&mut self) {
        // Flush the durable manifest one last time on a clean shutdown
        // (idempotent with the incremental writes on insert/remove). No-op for
        // an ephemeral tier or when no disk level is attached.
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
            read_scratch: Vec::new(),
        }
    }

    /// Attach the **ephemeral** disk spill level (opt-in, the prefix tier).
    /// Pre-serve only. The namespace is wiped on drop — a rebuildable cache.
    ///
    /// Each serve process gets its own namespace under the operator-provided
    /// root because tier keys are engine-local. Sharing a flat root across two
    /// processes would let key 0 in one process overwrite key 0 in another.
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

    /// Attach a **durable**, NVMe-backed disk spill level for the recall tier
    /// (`--kv-recall` + `--kv-ssd-path`). Pre-serve only. The namespace is
    /// *stable across restarts* (derived from the process id only, not a nanos
    /// nonce) and is NOT wiped on drop, so a prior session's evicted KV blobs
    /// survive. A manifest under the namespace records every disk-resident
    /// block's `{key, byte_len}` plus `epoch`; [`Self::load`] replays it.
    ///
    /// Returns `false` only when the namespace directory cannot be created.
    #[allow(dead_code)] // WIP: durable KV-recall persistence, not yet wired
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
        if let Err(err) = std::fs::create_dir_all(&namespace) {
            log::warn!(
                "KV disk namespace creation failed under {}: {err}",
                namespace.display()
            );
            return false;
        }
        self.disk = Some(DiskTier {
            root: namespace,
            capacity_pages,
            keys: BTreeMap::new(),
            durable,
            epoch,
        });
        true
    }

    /// Rebuild the in-memory disk index from a durable namespace's manifest so a
    /// prior session's evicted KV is addressable again. Re-attaches the disk
    /// level (durable) at `root` and replays the manifest's records into
    /// `keys`. A missing manifest, or one whose epoch tag does not match
    /// `epoch` (stale memory after an OPD weight update — see
    /// prefix-cache-stale-across-weight-epochs), starts cold: the namespace is
    /// re-created empty and any orphaned blobs are left for the OS/operator (a
    /// new epoch never reads stale KV).
    ///
    /// Returns whether any block was reloaded.
    #[allow(dead_code)] // WIP: durable KV-recall manifest replay, not yet wired
    pub(crate) fn load(
        &mut self,
        root: PathBuf,
        budget_bytes: usize,
        bytes_per_page: usize,
        epoch: String,
    ) -> bool {
        let namespace = Self::durable_namespace(root.clone());
        let manifest = namespace.join(MANIFEST_FILE);
        let parsed = std::fs::read(&manifest)
            .ok()
            .and_then(|bytes| DiskTier::parse_manifest(&bytes));
        // Always (re)attach the durable disk level first so the store is wired
        // regardless of whether the manifest was usable. `set_disk_durable`
        // re-derives the same per-process namespace from the operator root.
        if !self.set_disk_durable(root, budget_bytes, bytes_per_page, epoch.clone()) {
            return false;
        }
        let Some((manifest_epoch, records)) = parsed else {
            return false;
        };
        if manifest_epoch != epoch {
            log::warn!(
                "KV recall manifest epoch mismatch under {} (manifest={manifest_epoch:?}, \
                 requested={epoch:?}); discarding stale memory",
                namespace.display()
            );
            return false;
        }
        let Some(disk) = self.disk.as_mut() else {
            return false;
        };
        let capacity = disk.capacity_pages;
        let mut reloaded = 0usize;
        for (key, byte_len) in records {
            if disk.keys.len() >= capacity {
                break;
            }
            disk.keys.insert(key, byte_len);
            reloaded += 1;
        }
        reloaded > 0
    }

    /// Explicitly flush the durable manifest. No-op for an ephemeral (prefix)
    /// tier or when no disk level is attached. Best-effort: logs on failure.
    pub(crate) fn persist(&self) {
        if let Some(disk) = self.disk.as_ref() {
            if disk.durable {
                if let Err(err) = disk.write_manifest() {
                    log::warn!("KV recall manifest persist failed: {err}");
                }
            }
        }
    }

    /// Per-process *ephemeral* namespace: a nanos+counter nonce guarantees a
    /// fresh dir each attach (wiped on drop). Prefix-tier path.
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

    /// Per-process *durable* namespace: stable across restarts (process id
    /// only). The recall tier re-derives the same path on load to re-address
    /// the prior session's blobs + manifest.
    #[allow(dead_code)] // WIP: durable KV-recall namespace, not yet wired
    fn durable_namespace(root: PathBuf) -> PathBuf {
        root.join(format!("arle-kv-recall-{}", std::process::id()))
    }

    /// Total pages the store can hold (host-demoted + disk) — what the engine budgets
    /// demotion against.
    pub(crate) fn capacity_pages(&self) -> usize {
        self.host_capacity_pages
            .saturating_add(self.disk.as_ref().map_or(0, |d| d.capacity_pages))
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
            .is_none_or(|d| d.keys.len() >= d.capacity_pages);
        host_full && disk_full
    }

    pub(crate) fn contains(&self, key: u64) -> bool {
        self.host.contains_key(&key)
            || self
                .disk
                .as_ref()
                .is_some_and(|disk| disk.keys.contains_key(&key))
    }

    /// Store a page payload. Host RAM takes it when it has room; a full (or
    /// disabled, `--kv-t1-budget-bytes 0`) host tier spills its coldest entry to
    /// disk — or, with no host tier at all, the payload writes straight to disk.
    /// Returns `false` (payload dropped) only when no level has room or the
    /// disk write failed.
    pub(crate) fn insert(&mut self, key: u64, payload: Vec<u8>) -> bool {
        if self.host.len() < self.host_capacity_pages {
            self.insert_host(key, payload);
            return true;
        }
        if self.host_capacity_pages > 0 && self.spill_coldest_to_disk() {
            self.insert_host(key, payload);
            return true;
        }
        // Disk-only configuration (or the spill failed): write directly.
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
        if !already_present && disk.keys.len() >= disk.capacity_pages {
            return false;
        }
        match kv_native_sys::write_block_cache_sharded(&disk.root, fingerprint(key), payload) {
            Ok(()) => {
                // Insert/refresh the key->byte_len index entry, then persist the
                // manifest for a durable tier so a restart can re-address this
                // blob (a fresh insert grows the index; a replace may change the
                // recorded byte_len). Manifest write is best-effort: the blob is
                // already durable, a missed manifest update just under-reports
                // on the next load.
                disk.keys.insert(key, payload.len() as u64);
                if disk.durable {
                    if let Err(err) = disk.write_manifest() {
                        log::warn!("KV recall manifest update failed for key {key}: {err}");
                    }
                }
                true
            }
            Err(err) => {
                log::warn!(
                    "KV disk write failed for key {key} under {}: {err}",
                    disk.root.display()
                );
                false
            }
        }
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
            // Keep the entry in RAM rather than lose it; report no room.
            self.host.insert(key, entry);
            self.host_lru.insert((stamp, key));
            false
        }
    }

    /// Fetch a payload for promotion (host hit bumps recency; disk reads from
    /// disk without re-warming — the engine drops promoted keys right after).
    pub(crate) fn read(&mut self, key: u64) -> Result<Cow<'_, [u8]>> {
        if let Some(old_stamp) = self.host.get(&key).map(|entry| entry.stamp) {
            let stamp = self.next_stamp();
            self.host_lru.remove(&(old_stamp, key));
            self.host_lru.insert((stamp, key));
            let entry = self.host.get_mut(&key).expect("key observed above");
            entry.stamp = stamp;
            return Ok(Cow::Borrowed(entry.payload.as_slice()));
        }
        let disk_root = self.disk.as_ref().and_then(|disk| {
            if disk.keys.contains_key(&key) {
                Some(disk.root.clone())
            } else {
                None
            }
        });
        if let Some(root) = disk_root {
            kv_native_sys::read_block_into_sharded(&root, fingerprint(key), &mut self.read_scratch)
                .with_context(|| format!("KV disk read for key {key}"))?;
            return Ok(Cow::Borrowed(self.read_scratch.as_slice()));
        }
        Err(anyhow!("KV tier store has no entry for key {key}"))
    }

    /// Drop entries from both levels (disk files unlinked best-effort).
    pub(crate) fn remove(&mut self, keys: &[u64]) {
        let mut disk_index_changed = false;
        for key in keys {
            if let Some(entry) = self.host.remove(key) {
                self.host_lru.remove(&(entry.stamp, *key));
            }
            if let Some(disk) = &mut self.disk {
                if disk.keys.remove(key).is_some() {
                    let _ =
                        kv_native_sys::remove_block_sharded(&disk.root, fingerprint(*key), true);
                    disk_index_changed = true;
                }
            }
        }
        // Re-persist the durable manifest once after the batch so the dropped
        // keys are not re-addressed on the next load.
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

    #[cfg(test)]
    fn host_len(&self) -> usize {
        self.host.len()
    }

    #[cfg(test)]
    fn disk_len(&self) -> usize {
        self.disk.as_ref().map_or(0, |d| d.keys.len())
    }

    #[cfg(test)]
    fn disk_root(&self) -> Option<&Path> {
        self.disk.as_ref().map(|d| d.root.as_path())
    }

    #[cfg(test)]
    fn coldest_host_key(&self) -> Option<u64> {
        self.host_lru.iter().next().map(|(_, key)| *key)
    }

    #[cfg(test)]
    fn read_scratch_capacity(&self) -> usize {
        self.read_scratch.capacity()
    }

    #[cfg(test)]
    fn disk_is_durable(&self) -> bool {
        self.disk.as_ref().is_some_and(|d| d.durable)
    }

    #[cfg(test)]
    fn disk_byte_len(&self, key: u64) -> Option<u64> {
        self.disk.as_ref().and_then(|d| d.keys.get(&key).copied())
    }
}

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
        // 2 pages of 8 bytes.
        let mut store = CudaKvTierStore::with_budget(16, 8);
        assert_eq!(store.capacity_pages(), 2);
        assert!(store.insert(1, vec![1; 8]));
        assert!(store.insert(2, vec![2; 8]));
        assert!(store.is_full());
        assert!(!store.insert(3, vec![3; 8]), "no disk level: reject");
        assert_eq!(store.read(1).expect("host read").as_ref(), &[1u8; 8]);
    }

    #[test]
    fn host_overflow_spills_coldest_to_disk_and_reads_back() {
        let root = temp_root("spill");
        let mut store = CudaKvTierStore::with_budget(16, 8);
        assert!(store.set_disk(root.clone(), 32, 8));
        assert_eq!(store.capacity_pages(), 2 + 4);

        assert!(store.insert(1, vec![1; 8]));
        assert!(store.insert(2, vec![2; 8]));
        assert_eq!(store.coldest_host_key(), Some(1));
        // Touch key 1 so key 2 is the coldest when 3 arrives.
        store.read(1).expect("touch");
        assert_eq!(store.coldest_host_key(), Some(2));
        assert!(store.insert(3, vec![3; 8]));
        assert_eq!(store.host_len(), 2);
        assert_eq!(store.disk_len(), 1, "coldest entry spilled");

        // The spilled entry reads back from disk byte-identical.
        assert_eq!(store.read(2).expect("disk read").as_ref(), &[2u8; 8]);

        // Removal unlinks the block file.
        store.remove(&[2]);
        assert_eq!(store.disk_len(), 0);
        assert!(store.read(2).is_err());

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn disk_read_reuses_store_scratch_buffer() {
        let root = temp_root("scratch");
        let mut store = CudaKvTierStore::with_budget(0, 8);
        assert!(store.set_disk(root.clone(), 32, 8));
        assert!(store.insert(1, vec![1; 8]));

        {
            let first = store.read(1).expect("first disk read");
            assert!(
                matches!(&first, Cow::Borrowed(_)),
                "disk read borrows scratch"
            );
            assert_eq!(first.as_ref(), &[1u8; 8]);
        }
        let cap = store.read_scratch_capacity();

        {
            let second = store.read(1).expect("second disk read");
            assert_eq!(second.as_ref(), &[1u8; 8]);
        }
        assert_eq!(
            store.read_scratch_capacity(),
            cap,
            "second disk read reused scratch allocation"
        );

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
        assert_eq!(store.disk_len(), 1);
        assert_eq!(
            store.read(1).expect("replaced disk read").as_ref(),
            &[9u8; 8]
        );

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn disk_only_config_writes_straight_to_disk() {
        // --kv-t1-budget-bytes 0 --kv-ssd-path ...: host tier disabled, disk active.
        let root = temp_root("disk_only");
        let mut store = CudaKvTierStore::with_budget(0, 8);
        assert!(store.set_disk(root.clone(), 16, 8));
        assert_eq!(store.capacity_pages(), 2);

        assert!(store.insert(1, vec![1; 8]), "insert lands on disk");
        assert_eq!(store.host_len(), 0);
        assert_eq!(store.disk_len(), 1);
        assert_eq!(store.read(1).expect("disk read").as_ref(), &[1u8; 8]);
        assert!(store.insert(2, vec![2; 8]));
        assert!(store.is_full());
        assert!(!store.insert(3, vec![3; 8]), "disk full rejects");

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn disk_namespace_shards_and_cleans_up_process_owned_cache() {
        let root = temp_root("namespace");
        let namespace;
        let block_path;

        {
            let mut store = CudaKvTierStore::with_budget(0, 8);
            assert!(store.set_disk(root.clone(), 16, 8));
            namespace = store.disk_root().expect("disk namespace").to_path_buf();

            assert!(namespace.starts_with(&root));
            assert_ne!(namespace, root, "disk tier must not write into shared root");
            assert!(
                namespace
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("arle-kv-tier-")),
                "unexpected namespace path {}",
                namespace.display()
            );

            assert!(store.insert(1, vec![1; 8]));
            block_path = kv_native_sys::block_path_sharded(&namespace, fingerprint(1)).unwrap();
            assert!(
                block_path.starts_with(namespace.join("01").join("00")),
                "block path should shard under namespace: {}",
                block_path.display()
            );
            assert!(block_path.exists());
        }

        assert!(
            !namespace.exists(),
            "dropping disk tier should remove its process namespace"
        );
        assert!(root.exists(), "operator-provided root must survive");
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

    const GIB: usize = 1 << 30;

    #[test]
    fn host_default_budget_at_least_floor() {
        // Probe miss (Mac, no /proc) → 4 GiB floor; Linux MemAvailable →
        // clamp(avail×0.5, [4 GiB, avail − reserve]). Either way ≥ 0 and, when
        // DRAM is present, never below the 4 GiB floor unless the reserve eats
        // the whole box (a constrained machine, where 0 is correct). On the CI
        // hosts (ample RAM) we expect the 4 GiB floor or higher.
        let b = default_t1_budget_bytes(0.5);
        // No upper cap any more — the L2 tier grows into the box. Just prove the
        // probe path runs and the result is sane (≥ floor on a real machine, or
        // 0 only if available ≤ reserve).
        assert!(
            b == 0 || b >= 4 * GIB,
            "L2 DRAM budget {b} is neither 0 nor ≥ floor"
        );
    }

    #[test]
    fn disk_default_budget_runs_and_is_bounded_by_free() {
        // statvfs on a real temp dir (or probe-miss fallback). No proven cap any
        // more: the budget grows into free disk, bounded by free − reserve.
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
        // A nonexistent path fails the syscall → None (caller falls back to floor).
        assert!(disk_free_total_bytes(&root.join("does-not-exist/x")).is_none());
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn weights_epoch_tag_is_stable_and_path_falls_back() {
        // No safetensors in an empty dir → a stable path-derived tag (same dir
        // → same tag across calls).
        let root = temp_root("epoch");
        let a = weights_epoch_tag(&root);
        let b = weights_epoch_tag(&root);
        assert_eq!(a, b, "epoch tag must be deterministic for an unchanged dir");
        assert!(
            a.starts_with("path-"),
            "empty dir falls back to path tag: {a}"
        );

        // A safetensors file flips the tag to the content branch.
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
        // Disk-only durable tier (mirrors the recall store: host RAM may be 0,
        // everything lands on NVMe). 8-byte pages, room for 4 blocks.
        let root = temp_root("durable_manifest");
        let epoch = "epoch-A".to_string();

        // --- session 1: write three blocks, then drop (simulated restart) ---
        let namespace;
        {
            let mut store = CudaKvTierStore::with_budget(0, 8);
            assert!(store.set_disk_durable(root.clone(), 32, 8, epoch.clone()));
            assert!(store.disk_is_durable());
            namespace = store.disk_root().expect("durable namespace").to_path_buf();
            assert!(namespace.starts_with(&root));

            assert!(store.insert(1, vec![1; 8]));
            assert!(store.insert(2, vec![2; 8]));
            assert!(store.insert(3, vec![3; 8]));
            assert_eq!(store.disk_len(), 3);
            // The manifest was written incrementally on each insert.
            assert!(namespace.join(MANIFEST_FILE).exists(), "manifest persisted");
        }
        // Durable namespace + blobs survive the drop (NOT wiped).
        assert!(namespace.exists(), "durable namespace must survive drop");

        // --- session 2: load() rebuilds the in-memory index from the manifest ---
        {
            let mut store = CudaKvTierStore::with_budget(0, 8);
            let reloaded = store.load(root.clone(), 32, 8, epoch.clone());
            assert!(reloaded, "load reports prior blocks reloaded");
            assert_eq!(store.disk_len(), 3, "all three keys re-indexed");
            assert_eq!(store.disk_byte_len(2), Some(8), "byte_len preserved");
            // Blobs are addressable again, byte-identical.
            assert_eq!(store.read(1).expect("reload read 1").as_ref(), &[1u8; 8]);
            assert_eq!(store.read(3).expect("reload read 3").as_ref(), &[3u8; 8]);
        }

        // --- session 3: a mismatched epoch discards the stale memory ---
        {
            let mut store = CudaKvTierStore::with_budget(0, 8);
            let reloaded = store.load(root.clone(), 32, 8, "epoch-B".to_string());
            assert!(!reloaded, "stale epoch must not reload");
            assert_eq!(store.disk_len(), 0, "stale-epoch index starts cold");
            assert!(store.disk_is_durable(), "still a durable tier, just empty");
        }

        std::fs::remove_dir_all(&root).expect("cleanup");
    }
}

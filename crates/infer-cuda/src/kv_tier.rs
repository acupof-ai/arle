//! Two-level host store for demoted prefix-KV pages.
//!
//! Host-demoted pages live in a capacity-capped in-RAM map (default-on, 4 GiB).
//! Disk spill is optional on the `kv-native-sys` block substrate
//! (`--kv-ssd-path`, opt-in): when host RAM fills, the coldest host entry spills
//! to a fingerprint-named block file, so the capacity the engine sees is
//! host-demoted + disk pages. Payloads are full
//! `PagedKVPool` page images; this module never touches the device.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use infer_seam::{HostTierPolicy, split_host_tiers};

static DISK_TIER_NAMESPACE_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

/// Bytes of *available* system RAM (Linux `/proc/meminfo` `MemAvailable`).
///
/// `None` off Linux or when the field is unreadable — [`split_host_tiers`]
/// then falls back to the proven cap, so a probe miss never over-shrinks the
/// tier (a Mac CUDA-typecheck build, having no `/proc`, lands on the cap).
fn available_ram_bytes() -> Option<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        // "MemAvailable:   12345678 kB"
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// Bytes free on the filesystem holding `path` (POSIX `statvfs`,
/// unprivileged-available blocks). `None` on non-unix or probe failure →
/// [`split_host_tiers`] falls back to the proven cap.
#[cfg(unix)]
fn free_disk_bytes(path: &Path) -> Option<usize> {
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
    // f_bavail = blocks available to unprivileged users; f_frsize = fragment size.
    let bytes = u128::from(stat.f_frsize).saturating_mul(u128::from(stat.f_bavail));
    usize::try_from(bytes).ok()
}

#[cfg(not(unix))]
fn free_disk_bytes(_path: &Path) -> Option<usize> {
    None
}

/// Machine-derived host-RAM budget for demoted prefix pages — replaces the
/// hardcoded 4 GiB constant. Default-on (ckl 2026-06-11): evicted prefix pages
/// demote into host RAM and promote back on the next prefix hit instead of
/// re-prefilling (`--kv-t1-budget-bytes 0` opts out). Probes `MemAvailable`;
/// the neutral [`split_host_tiers`] policy caps it at the proven 4 GiB (so an
/// ample host is byte-identical to the old default) and scales it down on a
/// constrained one, with a 1 GiB floor.
pub(crate) fn default_t1_budget_bytes() -> usize {
    split_host_tiers(
        available_ram_bytes(),
        None,
        false,
        HostTierPolicy::default(),
    )
    .ram_bytes
}

/// Machine-derived SSD spill budget for the disk tier rooted at `root` —
/// replaces the hardcoded 20 GiB constant. Applies when `--kv-ssd-path` is set
/// without `--kv-ssd-max-bytes`. Probes free disk at the spill path; the
/// neutral [`split_host_tiers`] policy caps it at the proven 20 GiB and scales
/// it down when free disk is scarce.
pub fn default_t2_budget_bytes(root: &Path) -> usize {
    split_host_tiers(None, free_disk_bytes(root), true, HostTierPolicy::default()).ssd_bytes
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
    /// Keys whose payloads currently live on disk.
    keys: BTreeSet<u64>,
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn fingerprint(key: u64) -> [u8; 16] {
    let mut f = [0u8; 16];
    f[..8].copy_from_slice(&key.to_le_bytes());
    f
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

    /// Attach the disk spill level (opt-in). Pre-serve only.
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
        let capacity_pages = budget_bytes.checked_div(bytes_per_page).unwrap_or(0);
        let root = self.disk_namespace(root);
        if let Err(err) = std::fs::create_dir_all(&root) {
            log::warn!(
                "KV disk namespace creation failed under {}: {err}",
                root.display()
            );
            return false;
        }
        self.disk = Some(DiskTier {
            root,
            capacity_pages,
            keys: BTreeSet::new(),
        });
        true
    }

    fn disk_namespace(&mut self, root: PathBuf) -> PathBuf {
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
                .contains(&key)
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
                .is_some_and(|disk| disk.keys.contains(&key))
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
        let already_present = disk.keys.contains(&key);
        if !already_present && disk.keys.len() >= disk.capacity_pages {
            return false;
        }
        match kv_native_sys::write_block_cache_sharded(&disk.root, fingerprint(key), payload) {
            Ok(()) => {
                if !already_present {
                    disk.keys.insert(key);
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
            if disk.keys.contains(&key) {
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
        for key in keys {
            if let Some(entry) = self.host.remove(key) {
                self.host_lru.remove(&(entry.stamp, *key));
            }
            if let Some(disk) = &mut self.disk {
                if disk.keys.remove(key) {
                    let _ =
                        kv_native_sys::remove_block_sharded(&disk.root, fingerprint(*key), true);
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
    fn host_default_budget_within_policy_envelope() {
        // Probe miss (Mac) → 4 GiB cap; Linux MemAvailable → clamp(avail×0.25,
        // 1 GiB floor, 4 GiB cap). Either way: floored at 1 GiB, capped at 4 GiB.
        let b = default_t1_budget_bytes();
        assert!(
            (GIB..=4 * GIB).contains(&b),
            "host-demoted budget {b} outside [1 GiB, 4 GiB]"
        );
    }

    #[test]
    fn disk_default_budget_capped_at_proven_constant() {
        // statvfs on a real temp dir (or probe-miss fallback) → never exceeds
        // the proven 20 GiB cap.
        let root = temp_root("disk_probe");
        let b = default_t2_budget_bytes(&root);
        assert!(b <= 20 * GIB, "disk budget {b} exceeds 20 GiB cap");
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn free_disk_probe_reports_some_on_real_dir() {
        let root = temp_root("statvfs");
        assert!(
            free_disk_bytes(&root).is_some(),
            "statvfs on an existing dir should report free bytes"
        );
        // A nonexistent path fails the syscall → None (caller falls back to cap).
        assert!(free_disk_bytes(&root.join("does-not-exist/x")).is_none());
        std::fs::remove_dir_all(&root).expect("cleanup");
    }
}

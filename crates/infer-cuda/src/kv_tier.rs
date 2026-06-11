//! Two-level host store for demoted prefix-KV pages.
//!
//! T1 is a capacity-capped in-RAM map (default-on, 4 GiB). T2 is an optional
//! disk spill on the `kv-native-sys` block substrate (`--kv-ssd-path`,
//! opt-in): when T1 fills, the coldest T1 entry spills to a fingerprint-named
//! block file, so the capacity the engine sees is T1 + T2. Payloads are full
//! `PagedKVPool` page images; this module never touches the device.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};

/// Default host-RAM budget for the T1 prefix tier. Default-on (ckl
/// 2026-06-11): evicted prefix pages demote into host RAM and promote back on
/// the next prefix hit instead of re-prefilling; `--kv-t1-budget-bytes 0`
/// opts out.
pub(crate) const DEFAULT_KV_TIER_BUDGET_BYTES: usize = 4 << 30;

/// Default T2 disk budget when `--kv-ssd-path` is set without
/// `--kv-ssd-max-bytes` (the monolith-era 20 GiB LRU budget).
pub const DEFAULT_KV_SSD_BUDGET_BYTES: usize = 20 << 30;

pub(crate) struct CudaKvTierStore {
    t1_capacity_pages: usize,
    /// T1 entries: key -> (touch stamp, page payload).
    t1: BTreeMap<u64, (u64, Vec<u8>)>,
    clock: u64,
    disk: Option<DiskTier>,
}

struct DiskTier {
    root: PathBuf,
    capacity_pages: usize,
    /// Keys whose payloads currently live on disk.
    keys: BTreeSet<u64>,
}

fn fingerprint(key: u64) -> [u8; 16] {
    let mut f = [0u8; 16];
    f[..8].copy_from_slice(&key.to_le_bytes());
    f
}

impl CudaKvTierStore {
    pub(crate) fn with_budget(budget_bytes: usize, bytes_per_page: usize) -> Self {
        let t1_capacity_pages = budget_bytes.checked_div(bytes_per_page).unwrap_or(0);
        Self {
            t1_capacity_pages,
            t1: BTreeMap::new(),
            clock: 0,
            disk: None,
        }
    }

    /// Attach the T2 disk level (opt-in). Pre-serve only.
    pub(crate) fn set_disk(&mut self, root: PathBuf, budget_bytes: usize, bytes_per_page: usize) {
        let capacity_pages = budget_bytes.checked_div(bytes_per_page).unwrap_or(0);
        self.disk = Some(DiskTier {
            root,
            capacity_pages,
            keys: BTreeSet::new(),
        });
    }

    /// Total pages the store can hold (T1 + T2) — what the engine budgets
    /// demotion against.
    pub(crate) fn capacity_pages(&self) -> usize {
        self.t1_capacity_pages
            .saturating_add(self.disk.as_ref().map_or(0, |d| d.capacity_pages))
    }

    pub(crate) fn is_full(&self) -> bool {
        let t1_full = self.t1.len() >= self.t1_capacity_pages;
        let disk_full = self
            .disk
            .as_ref()
            .is_none_or(|d| d.keys.len() >= d.capacity_pages);
        t1_full && disk_full
    }

    /// Store a page payload. T1 takes it when it has room; a full (or
    /// disabled, `--kv-t1-budget-bytes 0`) T1 spills its coldest entry to
    /// disk — or, with no T1 at all, the payload writes straight to disk.
    /// Returns `false` (payload dropped) only when no level has room or the
    /// disk write failed.
    pub(crate) fn insert(&mut self, key: u64, payload: Vec<u8>) -> bool {
        if self.t1.len() < self.t1_capacity_pages {
            self.clock += 1;
            self.t1.insert(key, (self.clock, payload));
            return true;
        }
        if self.t1_capacity_pages > 0 && self.spill_coldest_to_disk() {
            self.clock += 1;
            self.t1.insert(key, (self.clock, payload));
            return true;
        }
        // Disk-only configuration (or the spill failed): write directly.
        self.write_to_disk(key, &payload)
    }

    fn write_to_disk(&mut self, key: u64, payload: &[u8]) -> bool {
        let Some(disk) = &mut self.disk else {
            return false;
        };
        if disk.keys.len() >= disk.capacity_pages {
            return false;
        }
        match kv_native_sys::write_block_atomic(&disk.root, fingerprint(key), payload) {
            Ok(()) => {
                disk.keys.insert(key);
                true
            }
            Err(err) => {
                log::warn!(
                    "KV T2 write failed for key {key} under {}: {err}",
                    disk.root.display()
                );
                false
            }
        }
    }

    fn spill_coldest_to_disk(&mut self) -> bool {
        let Some((&key, _)) = self.t1.iter().min_by_key(|(_, (stamp, _))| *stamp) else {
            return false;
        };
        let (stamp, payload) = self.t1.remove(&key).expect("key just observed");
        if self.write_to_disk(key, &payload) {
            true
        } else {
            // Keep the entry in RAM rather than lose it; report no room.
            self.t1.insert(key, (stamp, payload));
            false
        }
    }

    /// Fetch a payload for promotion (T1 hit bumps recency; T2 reads from
    /// disk without re-warming — the engine drops promoted keys right after).
    pub(crate) fn read(&mut self, key: u64) -> Result<Cow<'_, [u8]>> {
        if self.t1.contains_key(&key) {
            self.clock += 1;
            let entry = self.t1.get_mut(&key).expect("contains_key above");
            entry.0 = self.clock;
            return Ok(Cow::Borrowed(entry.1.as_slice()));
        }
        if let Some(disk) = &self.disk {
            if disk.keys.contains(&key) {
                let payload = kv_native_sys::read_block(&disk.root, fingerprint(key))
                    .with_context(|| format!("KV T2 read for key {key}"))?;
                return Ok(Cow::Owned(payload));
            }
        }
        Err(anyhow!("KV tier store has no entry for key {key}"))
    }

    /// Drop entries from both levels (disk files unlinked best-effort).
    pub(crate) fn remove(&mut self, keys: &[u64]) {
        for key in keys {
            self.t1.remove(key);
            if let Some(disk) = &mut self.disk {
                if disk.keys.remove(key) {
                    if let Ok(path) = kv_native_sys::block_path(&disk.root, fingerprint(*key)) {
                        let _ = std::fs::remove_file(path);
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn t1_len(&self) -> usize {
        self.t1.len()
    }

    #[cfg(test)]
    fn disk_len(&self) -> usize {
        self.disk.as_ref().map_or(0, |d| d.keys.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn t1_only_store_caps_at_budget() {
        // 2 pages of 8 bytes.
        let mut store = CudaKvTierStore::with_budget(16, 8);
        assert_eq!(store.capacity_pages(), 2);
        assert!(store.insert(1, vec![1; 8]));
        assert!(store.insert(2, vec![2; 8]));
        assert!(store.is_full());
        assert!(!store.insert(3, vec![3; 8]), "no disk level: reject");
        assert_eq!(store.read(1).expect("t1 read").as_ref(), &[1u8; 8]);
    }

    #[test]
    fn t1_overflow_spills_coldest_to_disk_and_reads_back() {
        let root = temp_root("spill");
        let mut store = CudaKvTierStore::with_budget(16, 8);
        store.set_disk(root.clone(), 32, 8);
        assert_eq!(store.capacity_pages(), 2 + 4);

        assert!(store.insert(1, vec![1; 8]));
        assert!(store.insert(2, vec![2; 8]));
        // Touch key 1 so key 2 is the coldest when 3 arrives.
        store.read(1).expect("touch");
        assert!(store.insert(3, vec![3; 8]));
        assert_eq!(store.t1_len(), 2);
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
    fn disk_only_config_writes_straight_to_disk() {
        // --kv-t1-budget-bytes 0 --kv-ssd-path ...: T1 disabled, disk active.
        let root = temp_root("disk_only");
        let mut store = CudaKvTierStore::with_budget(0, 8);
        store.set_disk(root.clone(), 16, 8);
        assert_eq!(store.capacity_pages(), 2);

        assert!(store.insert(1, vec![1; 8]), "insert lands on disk");
        assert_eq!(store.t1_len(), 0);
        assert_eq!(store.disk_len(), 1);
        assert_eq!(store.read(1).expect("disk read").as_ref(), &[1u8; 8]);
        assert!(store.insert(2, vec![2; 8]));
        assert!(store.is_full());
        assert!(!store.insert(3, vec![3; 8]), "disk full rejects");

        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn full_means_both_levels_full() {
        let root = temp_root("full");
        let mut store = CudaKvTierStore::with_budget(8, 8);
        store.set_disk(root.clone(), 8, 8);
        assert!(store.insert(1, vec![1; 8]));
        assert!(!store.is_full(), "disk still has room");
        assert!(store.insert(2, vec![2; 8]), "spills 1 to disk");
        assert!(store.is_full());
        assert!(!store.insert(3, vec![3; 8]));

        std::fs::remove_dir_all(&root).expect("cleanup");
    }
}

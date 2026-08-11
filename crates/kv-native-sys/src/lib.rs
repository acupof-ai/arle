//! Pure-Rust persistence substrate for the KV tier.
//!
//! POSIX-only (Linux and macOS). mmap is the default; Linux can opt into
//! O_DIRECT with io_uring and mmap fallback. [`KvTierStore`] is the backend-neutral
//! two-level store shared by the CUDA and Metal executors.

#[cfg(target_os = "linux")]
mod direct_store;
mod gds;
mod kv_tier;

pub use kv_tier::{
    BLOB_CHUNK_BYTES, CHUNK_IDX_BITS, DiskIoMode, KvTierStore, TIER_NS_SHIFT, TierIoStats,
    chunk_sub, default_t1_budget_bytes, default_t2_budget_bytes, resolve_dram_budget_bytes,
    tier_key, weights_epoch_tag,
};

use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::path::PathBuf;

/// Atomic replacement for cache payloads that are safe to rebuild after a
/// crash. Keeps the temp-write + rename property, but skips data and directory
/// fsyncs to avoid turning cache spill into a durability workload.
pub fn write_file_atomic_cache(path: &Path, bytes: &[u8]) -> io::Result<()> {
    write_file_atomic_impl(path, bytes, false)
}

/// Atomic replacement for payloads that must survive a crash — the durable
/// recall manifest. `sync_data` + parent-dir `sync_all` make the rename
/// durable on power loss. Callers must flush the data the manifest references
/// (e.g. [`KvMmapStore::flush`]) BEFORE calling this, so a crash never replays
/// a manifest onto unflushed slots.
pub fn write_file_atomic_durable(path: &Path, bytes: &[u8]) -> io::Result<()> {
    write_file_atomic_impl(path, bytes, true)
}

fn write_file_atomic_impl(path: &Path, bytes: &[u8], durable: bool) -> io::Result<()> {
    if path.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "write_file_atomic: empty path",
        ));
    }
    // Compose `<path>.tmp` by appending bytes to the OsString (no extension
    // semantics — the suffix is literal, matching what the on-disk layout
    // expects).
    let mut tmp_os = path.as_os_str().to_owned();
    tmp_os.push(".tmp");
    let tmp_path = PathBuf::from(tmp_os);

    let result = (|| -> io::Result<()> {
        {
            let mut tmp = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o644)
                .open(&tmp_path)?;
            tmp.write_all(bytes)?;
            if durable {
                tmp.sync_data()?;
            }
        }
        std::fs::rename(&tmp_path, path)?;
        if durable {
            // fsync the parent directory so the rename is durable on power loss.
            let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
            let parent = parent.unwrap_or_else(|| Path::new("."));
            let dir = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECTORY)
                .open(parent)?;
            dir.sync_all()?;
        }
        Ok(())
    })();

    if result.is_err() {
        // Best-effort cleanup of the staging file; ignore secondary errors.
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
}

/// File-backed mmap page-slot store — one file per disk tier namespace, a
/// fixed-size slot per page. Writes memcpy into the mapping (no per-page
/// syscall); reads return `&[u8]` slices directly from the mapping (zero-copy).
/// Built once at disk-tier attach time and held for the process lifetime.
///
/// The backing file is created as a **sparse** file — `fallocate` is never
/// called, so the filesystem only allocates blocks for actually-written pages.
/// A 274 GB store with 10 pages occupied costs ~23 MB on disk, not 274 GB.
pub struct KvMmapStore {
    /// The file keeping the backing store alive (mmap pins it open).
    _file: std::fs::File,
    /// Mutable mapping over the full slot array.
    mapping: memmap2::MmapMut,
    /// Size of one slot in bytes.
    slot_bytes: usize,
    stride_bytes: usize,
    /// Total number of slots.
    num_slots: u32,
    /// Indices of freed slots, available for re-use.
    free_list: Vec<u32>,
}

impl KvMmapStore {
    /// Create a new page-slot sparse mmap file at `path` with `num_slots`
    /// slots of `slot_bytes` each. The file is set to the logical size but
    /// NOT pre-allocated — the filesystem lazily allocates blocks for
    /// slots that are actually written.
    pub fn create(path: &Path, num_slots: usize, slot_bytes: usize) -> io::Result<Self> {
        let stride_bytes = aligned_slot_bytes(slot_bytes)?;
        let num_slots = u32::try_from(num_slots).map_err(|_| invalid("num_slots exceeds u32"))?;
        if num_slots == 0 {
            return Err(invalid("num_slots must be > 0"));
        }
        let total_bytes = (num_slots as u64)
            .checked_mul(stride_bytes as u64)
            .and_then(|t| usize::try_from(t).ok())
            .ok_or_else(|| invalid("total bytes overflow"))?;

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o644)
            .open(path)?;
        // set_len creates a sparse file: logical size without block allocation.
        file.set_len(total_bytes as u64)?;

        // SAFETY: the file was just created+truncated by this process and sized
        // to exactly `total_bytes` via set_len; `_file` keeps it open for the
        // mapping's lifetime. Sound as long as no external process truncates the
        // backing file while mapped (mmap's inherent contract).
        let mapping = unsafe {
            memmap2::MmapOptions::new()
                .len(total_bytes)
                .map_mut(&file)?
        };
        // Write-burst ceiling on the sparse mapping is first-touch soft faults
        // (~524k 4KiB faults per 2 GiB streamed). MADV_HUGEPAGE collapses them
        // 512x where THP allows; opt-in probe until an A/B licenses a default.
        #[cfg(target_os = "linux")]
        if std::env::var_os("ARLE_KV_MMAP_HUGEPAGE").is_some_and(|v| v == "1") {
            let _ = mapping.advise(memmap2::Advice::HugePage);
        }

        let free_list: Vec<u32> = (0..num_slots).collect();

        Ok(Self {
            _file: file,
            mapping,
            slot_bytes,
            stride_bytes,
            num_slots,
            free_list,
        })
    }

    /// Open an existing page-slot mmap file. Caller must replay the manifest to
    /// mark allocated slots via [`reserve`] — all slots are free on return.
    pub fn open(path: &Path, num_slots: usize, slot_bytes: usize) -> io::Result<Self> {
        let stride_bytes = aligned_slot_bytes(slot_bytes)?;
        let num_slots = u32::try_from(num_slots).map_err(|_| invalid("num_slots exceeds u32"))?;
        let total_bytes = (num_slots as u64)
            .checked_mul(stride_bytes as u64)
            .and_then(|t| usize::try_from(t).ok())
            .ok_or_else(|| invalid("total bytes overflow"))?;

        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let actual = file.metadata()?.len() as usize;
        if actual != total_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("mmap store file {actual}B != expected {total_bytes}B"),
            ));
        }

        // SAFETY: the file length was verified == `total_bytes` above and
        // `_file` keeps it open for the mapping's lifetime. Sound as long as no
        // external process truncates the backing file while mapped (mmap's
        // inherent contract).
        let mapping = unsafe {
            memmap2::MmapOptions::new()
                .len(total_bytes)
                .map_mut(&file)?
        };

        Ok(Self {
            _file: file,
            mapping,
            slot_bytes,
            stride_bytes,
            num_slots,
            free_list: (0..num_slots).collect(),
        })
    }

    pub fn num_slots(&self) -> u32 {
        self.num_slots
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Allocate a free slot index. Returns `None` when full.
    pub fn alloc_slot(&mut self) -> Option<u32> {
        self.free_list.pop()
    }

    pub fn available_slots(&self) -> usize {
        self.free_list.len()
    }

    /// Memcpy `data` into `slot` (`data.len() <= slot_bytes`).
    /// Trailing bytes are left untouched; callers must track the valid length.
    pub fn write_slot(&mut self, slot: u32, data: &[u8]) -> io::Result<()> {
        assert!(
            data.len() <= self.slot_bytes,
            "write_slot: data len {} > slot_bytes {}",
            data.len(),
            self.slot_bytes,
        );
        let offset = (slot as usize) * self.stride_bytes;
        self.mapping[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Return a borrowed slice over the slot — **zero-copy** mmap read.
    pub fn read_slot(&self, slot: u32) -> &[u8] {
        assert!(
            (slot as usize) < self.num_slots as usize,
            "read_slot: slot {slot} >= num_slots {}",
            self.num_slots,
        );
        let offset = (slot as usize) * self.stride_bytes;
        &self.mapping[offset..offset + self.slot_bytes]
    }

    /// msync the mapping (`MS_SYNC`) so all written slots reach disk. Cost is
    /// proportional to dirty pages, not the sparse file's logical size. The
    /// durable recall tier calls this before persisting the manifest that names
    /// the slots — the data-before-manifest ordering barrier.
    pub fn flush(&self) -> io::Result<()> {
        self.mapping.flush()
    }

    /// Return a slot to the free list.
    pub fn free_slot(&mut self, slot: u32) {
        if !self.free_list.contains(&slot) {
            self.free_list.push(slot);
        }
    }

    /// Reserve `indices` as allocated (manifest replay on load).
    pub fn reserve_indices(&mut self, indices: &[u32]) {
        self.free_list.retain(|i| !indices.contains(i));
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

fn aligned_slot_bytes(bytes: usize) -> io::Result<usize> {
    bytes
        .checked_add(4095)
        .map(|value| value / 4096 * 4096)
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid("slot bytes must be > 0 and fit usize"))
}

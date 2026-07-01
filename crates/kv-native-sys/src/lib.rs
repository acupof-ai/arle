//! Pure-Rust persistence substrate for the KV tier.
//!
//! POSIX-only (Linux + macOS); uses `nix`, `memmap2`, and `libc` directly
//! with no FFI of its own. Surface: file/block I/O, WAL, file mmap, POSIX
//! shm, and a host arena (anonymous mmap + bump pointer + free-list).

use std::fs::OpenOptions;
use std::io;
use std::io::Read;
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

pub fn read_file(path: &Path) -> io::Result<Vec<u8>> {
    if path.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "read_file: empty path",
        ));
    }
    // 0-byte file returns Ok(empty); missing file surfaces as NotFound.
    // `std::fs::read` already handles both correctly.
    std::fs::read(path)
}

pub fn read_file_into(path: &Path, dst: &mut Vec<u8>) -> io::Result<()> {
    if path.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "read_file_into: empty path",
        ));
    }
    dst.clear();
    let mut file = OpenOptions::new().read(true).open(path)?;
    if let Ok(metadata) = file.metadata() {
        if let Ok(len) = usize::try_from(metadata.len()) {
            dst.reserve(len);
        }
    }
    if let Err(err) = file.read_to_end(dst) {
        dst.clear();
        return Err(err);
    }
    Ok(())
}

pub fn remove_file(path: &Path, ignore_not_found: bool) -> io::Result<()> {
    if path.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "remove_file: empty path",
        ));
    }
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound && ignore_not_found => Ok(()),
        Err(err) => Err(err),
    }
}

pub fn block_path_sharded(root: &Path, fingerprint: [u8; 16]) -> io::Result<PathBuf> {
    let filename = block_filename(fingerprint);
    Ok(root
        .join(&filename[0..2])
        .join(&filename[2..4])
        .join(filename))
}

fn block_filename(fingerprint: [u8; 16]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    fingerprint
        .iter()
        .flat_map(|b| {
            [
                HEX[(b >> 4) as usize] as char,
                HEX[(b & 0xf) as usize] as char,
            ]
        })
        .chain(".kv".chars())
        .collect()
}

pub fn write_block_cache_sharded(
    root: &Path,
    fingerprint: [u8; 16],
    bytes: &[u8],
) -> io::Result<()> {
    let path = block_path_sharded(root, fingerprint)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_file_atomic_cache(&path, bytes)
}

pub fn read_block_into_sharded(
    root: &Path,
    fingerprint: [u8; 16],
    dst: &mut Vec<u8>,
) -> io::Result<()> {
    let path = block_path_sharded(root, fingerprint)?;
    read_file_into(&path, dst)
}

pub fn remove_block_sharded(
    root: &Path,
    fingerprint: [u8; 16],
    ignore_not_found: bool,
) -> io::Result<()> {
    let path = block_path_sharded(root, fingerprint)?;
    remove_file(&path, ignore_not_found)
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
        let num_slots = u32::try_from(num_slots).map_err(|_| invalid("num_slots exceeds u32"))?;
        if num_slots == 0 {
            return Err(invalid("num_slots must be > 0"));
        }
        let total_bytes = (num_slots as u64)
            .checked_mul(slot_bytes as u64)
            .and_then(|t| usize::try_from(t).ok())
            .ok_or_else(|| invalid("total bytes overflow"))?;

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
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

        let mapping = unsafe {
            memmap2::MmapOptions::new()
                .len(total_bytes)
                .map_mut(&file)?
        };

        let free_list: Vec<u32> = (0..num_slots).collect();

        Ok(Self {
            _file: file,
            mapping,
            slot_bytes,
            num_slots,
            free_list,
        })
    }

    /// Open an existing page-slot mmap file. Caller must replay the manifest to
    /// mark allocated slots via [`reserve`] — all slots are free on return.
    pub fn open(path: &Path, num_slots: usize, slot_bytes: usize) -> io::Result<Self> {
        let num_slots = u32::try_from(num_slots).map_err(|_| invalid("num_slots exceeds u32"))?;
        let total_bytes = (num_slots as u64)
            .checked_mul(slot_bytes as u64)
            .and_then(|t| usize::try_from(t).ok())
            .ok_or_else(|| invalid("total bytes overflow"))?;

        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let actual = file.metadata()?.len() as usize;
        if actual < total_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("mmap store file {actual}B < expected {total_bytes}B"),
            ));
        }

        let mapping = unsafe {
            memmap2::MmapOptions::new()
                .len(total_bytes)
                .map_mut(&file)?
        };

        Ok(Self {
            _file: file,
            mapping,
            slot_bytes,
            num_slots,
            free_list: Vec::with_capacity(num_slots as usize),
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

    /// Memcpy `data` into `slot` (`data.len() <= slot_bytes`).
    /// Trailing bytes are left untouched; callers must track the valid length.
    pub fn write_slot(&mut self, slot: u32, data: &[u8]) -> io::Result<()> {
        assert!(
            data.len() <= self.slot_bytes,
            "write_slot: data len {} > slot_bytes {}",
            data.len(),
            self.slot_bytes,
        );
        let offset = (slot as usize) * self.slot_bytes;
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
        let offset = (slot as usize) * self.slot_bytes;
        &self.mapping[offset..offset + self.slot_bytes]
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

    /// Flush the store to disk (msync). Best-effort for cache durability.
    pub fn flush(&self) -> io::Result<()> {
        self.mapping.flush()
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn mmap_store_create_write_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mmap");
        let mut store = KvMmapStore::create(&path, 4, 16).unwrap();
        assert_eq!(store.num_slots(), 4);
        let s = store.alloc_slot().unwrap();
        store.write_slot(s, b"0123456789abcdef").unwrap();
        assert_eq!(store.read_slot(s), b"0123456789abcdef");
    }

    #[test]
    fn mmap_store_short_write_preserves_prefix() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("short.mmap");
        let mut store = KvMmapStore::create(&path, 2, 16).unwrap();
        let slot = store.alloc_slot().unwrap();
        store.write_slot(slot, b"short").unwrap();
        assert_eq!(&store.read_slot(slot)[..5], b"short");
    }

    #[test]
    fn mmap_store_free_and_reuse() {
        let dir = tempdir().unwrap();
        let mut s = KvMmapStore::create(&dir.path().join("f.mmap"), 2, 8).unwrap();
        let s0 = s.alloc_slot().unwrap();
        let _s1 = s.alloc_slot().unwrap();
        assert!(s.alloc_slot().is_none());
        s.free_slot(s0);
        assert_eq!(s.alloc_slot().unwrap(), s0);
    }

    #[test]
    fn mmap_store_flush_is_best_effort() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("fl.mmap");
        let mut s = KvMmapStore::create(&path, 2, 8).unwrap();
        let ss = s.alloc_slot().unwrap();
        s.write_slot(ss, b"testdata").unwrap();
        s.flush().unwrap();
        drop(s);
        assert!(path.exists());
    }
}

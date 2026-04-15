//! T2 disk backend: [`DiskStore`] provides both a key-based blob store
//! and a [`BlockLocation::Disk`]-aware block allocator for the tiered KV
//! cache skeleton.
//!
//! Tier number: the 2026-04-15 revision renamed T3 → T2 (project doc
//! §4.1) to match industry convention; the old T3 label is preserved
//! in some test/bench file names but the code uses T2.
//!
//! This is the **skeleton implementation** for the M4 disk tier
//! (formerly P3, renamed in the 2026-04-15 revision). It lands now to
//! prove two things on the local Mac lane:
//! 1. `std::fs` I/O is sufficient for the skeleton (no tokio runtime
//!    boilerplate, no async/await on the hot path).
//! 2. The `BlockLocation::Disk { file_id, offset }` addressing model
//!    survives a round trip through a real filesystem — [`DiskStore`]'s
//!    `put_block` / `get_block` pair issue `BlockLocation` values that
//!    the scheduler's prefix-cache layer can reference directly. The
//!    pre-M1 `TierDirectory` holding area no longer exists; location
//!    metadata now lives on the `RadixCache` node instead.
//!
//! # Scope (intentional)
//!
//! - Two layers on top of one root directory:
//!   - **Keyed API** — `write` / `read` / `remove` use caller-supplied
//!     relative paths and reject absolute paths or `..` traversal. Used
//!     by the HTTP session save/load routes (M4 behavior PR) where the
//!     caller already has a stable key like the session id.
//!   - **Block API** — `put_block` / `get_block` / `delete_block`
//!     allocate sequential `file_id`s (via `AtomicU32`) and hand back a
//!     [`BlockLocation::Disk`] for each put. Used by the coordinator
//!     eviction path where the radix cache doesn't have its own name
//!     for the block yet.
//! - Sequential `file_id` allocation. Not content-addressable yet —
//!   that arrives in M4 behavior PR together with blake3 hashing (via
//!   `crate::types::BlockFingerprint`).
//! - One file per block; every block starts at offset `0` of its own
//!   file. `offset` stays in [`BlockLocation::Disk`] for forward
//!   compatibility with a future packed-file format.
//! - Synchronous I/O via `std::fs`. The `KVTransport` trait is sync
//!   too, so this is the natural fit; the coordinator does its own
//!   thread management.
//! - No checksum / integrity verification yet. Relies on fs reliability.
//!
//! See `docs/plans/tiered-kv-cache-tasks.md §4.2` for research — LMCache
//! and SGLang both use the "one file per chunk, in-memory index, rebuild
//! on startup" pattern, and this matches.
//!
//! # Non-scope (deferred)
//!
//! - `KVTransport` trait impl over this store — lands in M4 behavior PR
//!   once the coordinator owns a registered host buffer to copy
//!   to/from.
//! - Content-addressable filenames (blake3 hash of bytes via
//!   `crate::types::BlockFingerprint`) — M4 behavior PR.
//! - `O_DIRECT` / `F_NOCACHE` — optional optimization, only matters
//!   when the store is large enough that page-cache pollution hurts.
//! - `tokio::fs` async path — only useful once the coordinator drives
//!   I/O from an async context.
//! - `io_uring` — deferred indefinitely per §4.3 course corrections.
//! - Rebuilding `next_file_id` by scanning the directory on startup —
//!   every new [`DiskStore`] instance starts at `file_id = 0`, which
//!   works for the skeleton but will collide with an existing directory
//!   containing `0.blk`. M4 behavior PR adds a `rescan()` that walks
//!   the root and bumps the counter past the highest observed id.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use super::super::tier::BlockLocation;

/// Stable local filesystem backing for persisted KV blobs.
///
/// Not `Clone` — the `AtomicU32` file-id allocator is a single source of
/// truth and cloning it would hand out colliding ids. Callers that need
/// to share a store across threads should wrap it in [`std::sync::Arc`].
#[derive(Debug)]
pub struct DiskStore {
    root: PathBuf,
    next_file_id: AtomicU32,
}

impl DiskStore {
    /// Create a store rooted at `root`. Does **not** create the
    /// directory — call [`DiskStore::create_root`] if it might not
    /// exist yet.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            next_file_id: AtomicU32::new(0),
        }
    }

    /// Root directory used by this store.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the canonical storage path for `key`.
    ///
    /// Keys are treated as relative paths and rejected if they escape
    /// the root. This is the defense the HTTP session save/load routes
    /// use to keep untrusted `session_id` values from reaching
    /// arbitrary files.
    pub fn path_for(&self, key: impl AsRef<Path>) -> io::Result<PathBuf> {
        let key = key.as_ref();
        if key.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "disk store key must be relative",
            ));
        }
        if key
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "disk store key cannot contain parent traversal",
            ));
        }
        Ok(self.root.join(key))
    }

    /// Ensures the store root exists.
    pub fn create_root(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root)
    }

    // ── Keyed API ────────────────────────────────────────────────────

    /// Writes `bytes` to `key`, creating parent directories as needed.
    pub fn write(&self, key: impl AsRef<Path>, bytes: &[u8]) -> io::Result<()> {
        let path = self.path_for(key)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, bytes)
    }

    /// Reads the payload stored at `key`.
    pub fn read(&self, key: impl AsRef<Path>) -> io::Result<Vec<u8>> {
        fs::read(self.path_for(key)?)
    }

    /// Removes the payload stored at `key`. Missing files are ignored.
    pub fn remove(&self, key: impl AsRef<Path>) -> io::Result<()> {
        let path = self.path_for(key)?;
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err),
        }
    }

    // ── Block API (BlockLocation::Disk aware) ────────────────────────

    /// Relative filename for a given `file_id`. Kept private so the
    /// block-api caller never has to think about directory layout.
    fn block_filename(file_id: u32) -> String {
        format!("{file_id}.blk")
    }

    /// Write `bytes` as a new block and return the resulting
    /// [`BlockLocation`] so the caller can register it with the tier
    /// directory. Each call allocates a fresh `file_id`, so callers do
    /// not need to worry about collisions.
    pub fn put_block(&self, bytes: &[u8]) -> io::Result<BlockLocation> {
        self.create_root()?;
        let file_id = self.next_file_id.fetch_add(1, Ordering::SeqCst);
        self.write(Self::block_filename(file_id), bytes)?;
        Ok(BlockLocation::Disk { file_id, offset: 0 })
    }

    /// Read a block back from the store.
    ///
    /// `len` is the number of bytes to return starting at the block's
    /// `offset`. Pass `None` to read the whole file — the skeleton
    /// stores one block per file so this is usually what the caller
    /// wants. Returns `Err(io::ErrorKind::InvalidInput)` if `location`
    /// is not a `Disk` variant.
    pub fn get_block(&self, location: &BlockLocation, len: Option<u64>) -> io::Result<Vec<u8>> {
        let BlockLocation::Disk { file_id, offset } = location else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "disk store: block location is not a Disk variant",
            ));
        };
        let path = self.path_for(Self::block_filename(*file_id))?;
        let bytes = fs::read(&path)?;
        let offset = *offset as usize;
        if offset > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "disk store: offset past end of block file",
            ));
        }
        let available = bytes.len() - offset;
        let want = match len {
            Some(n) => {
                let n = n as usize;
                if n > available {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "disk store: requested len past end of block file",
                    ));
                }
                n
            }
            None => available,
        };
        Ok(bytes[offset..offset + want].to_vec())
    }

    /// Delete a block's backing file. Missing files are ignored so the
    /// operation is idempotent — useful for the eviction path where
    /// coordinator bookkeeping may race with an explicit delete.
    pub fn delete_block(&self, location: &BlockLocation) -> io::Result<()> {
        let BlockLocation::Disk { file_id, .. } = location else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "disk store: block location is not a Disk variant",
            ));
        };
        self.remove(Self::block_filename(*file_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // ── Keyed API tests ──────────────────────────────────────────────

    #[test]
    fn rejects_escape_paths() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        assert!(store.path_for("../escape").is_err());
        assert!(store.path_for("/tmp/escape").is_err());
    }

    #[test]
    fn round_trips_bytes() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        store.write("session/kv.bin", b"hello").unwrap();
        assert_eq!(store.read("session/kv.bin").unwrap(), b"hello");
        store.remove("session/kv.bin").unwrap();
        assert!(store.read("session/kv.bin").is_err());
    }

    // ── Block API tests ──────────────────────────────────────────────

    #[test]
    fn put_block_and_get_block_roundtrip() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let payload = b"the quick brown fox jumps over the lazy dog";

        let location = store.put_block(payload).expect("put_block");
        match location {
            BlockLocation::Disk { file_id, offset } => {
                assert_eq!(file_id, 0, "first put_block should get file_id=0");
                assert_eq!(offset, 0);
            }
            _ => panic!("put_block should return a Disk location"),
        }

        let read_back = store.get_block(&location, None).expect("get_block");
        assert_eq!(read_back, payload);
    }

    #[test]
    fn sequential_put_blocks_advance_file_id() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let a = store.put_block(b"alpha").unwrap();
        let b = store.put_block(b"beta").unwrap();
        let c = store.put_block(b"gamma").unwrap();

        let ids: Vec<u32> = [a, b, c]
            .iter()
            .map(|loc| match loc {
                BlockLocation::Disk { file_id, .. } => *file_id,
                _ => panic!("put_block returned non-Disk location"),
            })
            .collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn get_block_honors_explicit_len() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let loc = store.put_block(b"hello world").unwrap();
        let prefix = store.get_block(&loc, Some(5)).expect("prefix read");
        assert_eq!(prefix, b"hello");
    }

    #[test]
    fn get_block_out_of_bounds_errors() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let loc = store.put_block(b"abc").unwrap();
        let err = store
            .get_block(&loc, Some(100))
            .expect_err("should fail past end");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn block_api_rejects_non_disk_location() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let bad = BlockLocation::Gpu { slot: 42 };
        let get_err = store.get_block(&bad, None).expect_err("get rejects Gpu");
        assert_eq!(get_err.kind(), io::ErrorKind::InvalidInput);
        let del_err = store.delete_block(&bad).expect_err("delete rejects Gpu");
        assert_eq!(del_err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn delete_block_is_idempotent() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let loc = store.put_block(b"ephemeral").unwrap();
        store.delete_block(&loc).expect("first delete");
        store.delete_block(&loc).expect("second delete is a no-op");
        let err = store.get_block(&loc, None).expect_err("get after delete");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn disk_store_round_trip_preserves_fingerprint() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let payload: Vec<u8> = (0..4096u64).map(|i| (i % 256) as u8).collect();
        let fingerprint_for = |bytes: &[u8]| {
            crate::types::BlockFingerprint::compute_from_tokens(
                &bytes
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<_>>(),
            )
        };

        let location = store.put_block(&payload).expect("put_block");
        let read_back = store.get_block(&location, None).expect("get_block");

        assert_eq!(fingerprint_for(&payload), fingerprint_for(&read_back));
        assert_eq!(read_back, payload);
    }
}

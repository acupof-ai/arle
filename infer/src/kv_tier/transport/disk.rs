//! T2 disk backend: [`DiskStore`] provides both a keyed blob store and a
//! content-addressable block store for the tiered KV cache skeleton.
//!
//! Tier number: the 2026-04-15 revision renamed T3 → T2 (project doc
//! §4.1) to match industry convention; the old T3 label is preserved in
//! some historical notes but the code uses T2.
//!
//! This file now carries the M4b local disk format:
//! 1. block files are named by [`crate::types::BlockFingerprint`] so a
//!    restarted process can reconcile persisted content by semantic
//!    identity instead of transient pool slot ids;
//! 2. each file starts with a postcard-encoded [`DiskBlockHeader`]
//!    followed by the raw KV payload bytes.
//!
//! # Scope (intentional)
//!
//! - Two layers on top of one root directory:
//!   - **Keyed API** — `write` / `read` / `remove` use caller-supplied
//!     relative paths and reject absolute paths or `..` traversal. Used
//!     by session save/load code where the caller already has a stable key
//!     such as a session id.
//!   - **Block API** — `put_block` / `get_block` / `delete_block`
//!     allocate content-addressable files named by the 16-byte
//!     fingerprint (`32` lowercase hex chars + `.kv`) and return a
//!     [`DiskBlockLocation`] for later reads/deletes.
//! - One file per block. The payload is still an opaque raw byte blob
//!   owned by the caller; only the header is interpreted here.
//! - Synchronous I/O via `std::fs`. The `KVTransport` trait is sync
//!   too, so this is the natural fit; the coordinator does its own
//!   thread management.
//! - Header validation on read: magic, version, payload length, and
//!   optional fingerprint match.
//!
//! # Non-scope (deferred)
//!
//! - `KVTransport` trait impl over this store — lands once the
//!   coordinator owns a registered host buffer to copy to/from.
//! - `O_DIRECT` / `F_NOCACHE` — optional optimization, only matters
//!   when the store is large enough that page-cache pollution hurts.
//! - `tokio::fs` async path — only useful once the coordinator drives
//!   I/O from an async context.
//! - `io_uring` — deferred indefinitely per §4.3 course corrections.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::types::BlockFingerprint;

const DISK_BLOCK_MAGIC: [u8; 8] = *b"PEGAKV01";
const DISK_BLOCK_VERSION: u16 = 1;

/// Stable on-disk location for a persisted KV block.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DiskBlockLocation {
    pub path: PathBuf,
    pub payload_len: u64,
    pub fingerprint: BlockFingerprint,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct DiskBlockHeader {
    magic: [u8; 8],
    version: u16,
    fingerprint: [u8; 16],
    kv_format_tag: u8,
    payload_len: u64,
}

impl DiskBlockHeader {
    fn new(fingerprint: BlockFingerprint, kv_format_tag: u8, payload_len: u64) -> Self {
        Self {
            magic: DISK_BLOCK_MAGIC,
            version: DISK_BLOCK_VERSION,
            fingerprint: fingerprint.0,
            kv_format_tag,
            payload_len,
        }
    }

    fn decode(bytes: &[u8]) -> io::Result<(Self, &[u8])> {
        let (header, payload) = postcard::take_from_bytes::<DiskBlockHeader>(bytes)
            .map_err(|err| invalid_data(format!("disk store: failed to decode header: {err}")))?;
        if header.magic != DISK_BLOCK_MAGIC {
            return Err(invalid_data("disk store: invalid block magic"));
        }
        if header.version != DISK_BLOCK_VERSION {
            return Err(invalid_data("disk store: unsupported block version"));
        }
        let payload_len = usize::try_from(header.payload_len)
            .map_err(|_| invalid_data("disk store: payload length exceeds platform usize"))?;
        if payload.len() != payload_len {
            return Err(invalid_data("disk store: payload length mismatch"));
        }
        Ok((header, payload))
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Stable local filesystem backing for persisted KV blobs.
#[derive(Debug)]
pub struct DiskStore {
    root: PathBuf,
}

impl DiskStore {
    /// Create a store rooted at `root`. Does **not** create the
    /// directory — call [`DiskStore::create_root`] if it might not
    /// exist yet.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Root directory used by this store.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the canonical storage path for `key`.
    ///
    /// Keys are treated as relative paths and rejected if they escape
    /// the root. This is the defense the session save/load routes use
    /// to keep untrusted `session_id` values from reaching arbitrary
    /// files.
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

    // ── Block API (content-addressable) ──────────────────────────────

    fn block_filename(fingerprint: BlockFingerprint) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";

        let mut filename = String::with_capacity(35);
        for byte in fingerprint.0 {
            filename.push(char::from(HEX[(byte >> 4) as usize]));
            filename.push(char::from(HEX[(byte & 0x0f) as usize]));
        }
        filename.push_str(".kv");
        filename
    }

    /// Public canonical path for a fingerprinted block. Re-derived on
    /// every read/write from the store root + fingerprint — callers
    /// **cannot** influence the final path by tampering with a stored
    /// `DiskBlockLocation`. This is the defense against session
    /// snapshot–driven path traversal (M4 review finding B2).
    pub fn block_path_for(&self, fingerprint: BlockFingerprint) -> io::Result<PathBuf> {
        self.path_for(Self::block_filename(fingerprint))
    }

    /// Writes a content-addressed block file and returns its stable disk
    /// location metadata. Writes are **crash-safe**: the payload is
    /// written to `<path>.tmp` first and then atomically renamed onto
    /// `<path>`, so a mid-write crash never leaves a partially-written
    /// file under the canonical name (M4 review finding B3).
    pub fn put_block(
        &self,
        fingerprint: BlockFingerprint,
        kv_format_tag: u8,
        payload: &[u8],
    ) -> io::Result<DiskBlockLocation> {
        self.create_root()?;

        let payload_len = u64::try_from(payload.len())
            .map_err(|_| invalid_data("disk store: payload too large"))?;
        let header = DiskBlockHeader::new(fingerprint, kv_format_tag, payload_len);
        let header_bytes = postcard::to_allocvec(&header)
            .map_err(|err| invalid_data(format!("disk store: failed to encode header: {err}")))?;
        let mut file_bytes = Vec::with_capacity(header_bytes.len() + payload.len());
        file_bytes.extend_from_slice(&header_bytes);
        file_bytes.extend_from_slice(payload);

        let path = self.block_path_for(fingerprint)?;
        // Crash-safe atomic write: stage to a .tmp sibling, then rename.
        let mut tmp_path = path.clone();
        let tmp_name = format!(
            "{}.tmp",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("block")
        );
        tmp_path.set_file_name(tmp_name);
        fs::write(&tmp_path, &file_bytes)?;
        if let Err(err) = fs::rename(&tmp_path, &path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(err);
        }

        Ok(DiskBlockLocation {
            path,
            payload_len,
            fingerprint,
        })
    }

    /// Reads a block back from the store and validates its postcard
    /// header before returning the raw payload bytes.
    ///
    /// **Path re-rooting (M4 review finding B2)**: the read path is
    /// always re-derived from `location.fingerprint` via
    /// [`DiskStore::block_path_for`]; `location.path` is treated as
    /// advisory and may **not** point outside the store root. If the
    /// two disagree, the call errors out with `InvalidData` — a
    /// tampered session snapshot that carries a doctored `path` cannot
    /// drive the store to read an arbitrary file.
    pub fn get_block(
        &self,
        location: &DiskBlockLocation,
        expected_fingerprint: Option<BlockFingerprint>,
    ) -> io::Result<Vec<u8>> {
        if let Some(expected) = expected_fingerprint
            && location.fingerprint != expected
        {
            return Err(invalid_data(
                "disk store: fingerprint mismatch (location vs expected)",
            ));
        }
        let canonical = self.block_path_for(location.fingerprint)?;
        if location.path != canonical {
            return Err(invalid_data(
                "disk store: refused location.path outside canonical root",
            ));
        }

        let bytes = fs::read(&canonical)?;
        let (header, payload) = DiskBlockHeader::decode(&bytes)?;

        if header.fingerprint != location.fingerprint.0 {
            return Err(invalid_data(
                "disk store: on-disk fingerprint does not match location",
            ));
        }

        Ok(payload.to_vec())
    }

    /// Delete a block's backing file. Missing files are ignored so the
    /// operation is idempotent — useful for eviction flows where
    /// bookkeeping may race with an explicit delete.
    ///
    /// The path is re-derived from `location.fingerprint`; the
    /// advisory `location.path` is not trusted (same reasoning as
    /// [`DiskStore::get_block`]).
    pub fn delete_block(&self, location: &DiskBlockLocation) -> io::Result<()> {
        let canonical = self.block_path_for(location.fingerprint)?;
        match fs::remove_file(&canonical) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::KvContentContext;
    use tempfile::tempdir;

    const TEST_MODEL_FINGERPRINT: &[u8] = b"qwen3-4b";
    const TEST_PARENT_FINGERPRINT: BlockFingerprint = BlockFingerprint([0xA5; 16]);

    fn payload_words(payload: &[u8]) -> Vec<u32> {
        let mut chunks = payload.chunks_exact(4);
        let words = chunks
            .by_ref()
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        assert!(
            chunks.remainder().is_empty(),
            "payload must be divisible by 4 for fingerprint test"
        );
        words
    }

    fn fingerprint_for_payload(payload: &[u8], kv_format_tag: u8) -> BlockFingerprint {
        BlockFingerprint::compute(
            KvContentContext {
                model_fingerprint: TEST_MODEL_FINGERPRINT,
                kv_format_tag,
                parent: Some(TEST_PARENT_FINGERPRINT),
            },
            &payload_words(payload),
        )
    }

    fn tampered_fingerprint(fingerprint: BlockFingerprint) -> BlockFingerprint {
        let mut bytes = fingerprint.0;
        bytes[0] ^= 0xFF;
        BlockFingerprint(bytes)
    }

    fn write_raw_block(path: &Path, header: &DiskBlockHeader, payload: &[u8]) {
        let mut bytes = postcard::to_allocvec(header).expect("serialize header");
        bytes.extend_from_slice(payload);
        fs::write(path, bytes).expect("write block file");
    }

    fn read_header_and_payload(path: &Path) -> (DiskBlockHeader, Vec<u8>) {
        let bytes = fs::read(path).expect("read block file");
        let (header, payload) =
            postcard::take_from_bytes::<DiskBlockHeader>(&bytes).expect("decode block file");
        (header, payload.to_vec())
    }

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
        let kv_format_tag = 3;
        let payload = b"0123456789abcdef0123456789abcdef".to_vec();
        let fingerprint = fingerprint_for_payload(&payload, kv_format_tag);

        let location = store
            .put_block(fingerprint, kv_format_tag, &payload)
            .expect("put_block");

        assert_eq!(
            location.path,
            dir.path().join(DiskStore::block_filename(fingerprint))
        );
        assert_eq!(location.payload_len, payload.len() as u64);
        assert_eq!(location.fingerprint, fingerprint);

        let read_back = store
            .get_block(&location, Some(fingerprint))
            .expect("get_block");
        assert_eq!(read_back, payload);

        let (header, stored_payload) = read_header_and_payload(&location.path);
        assert_eq!(header.magic, DISK_BLOCK_MAGIC);
        assert_eq!(header.version, DISK_BLOCK_VERSION);
        assert_eq!(header.fingerprint, fingerprint.0);
        assert_eq!(header.kv_format_tag, kv_format_tag);
        assert_eq!(header.payload_len, payload.len() as u64);
        assert_eq!(stored_payload, payload);
    }

    #[test]
    fn delete_block_is_idempotent() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let kv_format_tag = 1;
        let payload = b"deadbeef".to_vec();
        let fingerprint = fingerprint_for_payload(&payload, kv_format_tag);
        let location = store
            .put_block(fingerprint, kv_format_tag, &payload)
            .unwrap();

        store.delete_block(&location).expect("first delete");
        store
            .delete_block(&location)
            .expect("second delete is a no-op");

        let err = store
            .get_block(&location, Some(fingerprint))
            .expect_err("get after delete");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn disk_store_round_trip_preserves_fingerprint() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let kv_format_tag = 9;
        let payload: Vec<u8> = (0..4096u64).map(|i| (i % 256) as u8).collect();
        let fingerprint = fingerprint_for_payload(&payload, kv_format_tag);

        let location = store
            .put_block(fingerprint, kv_format_tag, &payload)
            .expect("put_block");

        let read_back = store
            .get_block(&location, Some(fingerprint))
            .expect("get_block with matching fingerprint");
        let wrong_fingerprint = tampered_fingerprint(fingerprint);
        let err = store
            .get_block(&location, Some(wrong_fingerprint))
            .expect_err("tampered fingerprint should fail");

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(read_back, payload);

        let (header, stored_payload) = read_header_and_payload(&location.path);
        assert_eq!(header.magic, DISK_BLOCK_MAGIC);
        assert_eq!(header.version, DISK_BLOCK_VERSION);
        assert_eq!(header.fingerprint, fingerprint.0);
        assert_eq!(header.kv_format_tag, kv_format_tag);
        assert_eq!(header.payload_len, payload.len() as u64);
        assert_eq!(stored_payload, payload);
    }

    #[test]
    fn disk_store_rejects_wrong_magic() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let kv_format_tag = 4;
        let payload = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let fingerprint = fingerprint_for_payload(&payload, kv_format_tag);
        let path = store.path_for("bad-magic.kv").unwrap();
        store.create_root().unwrap();

        write_raw_block(
            &path,
            &DiskBlockHeader {
                magic: *b"BADMAGC!",
                version: DISK_BLOCK_VERSION,
                fingerprint: fingerprint.0,
                kv_format_tag,
                payload_len: payload.len() as u64,
            },
            &payload,
        );

        let err = store
            .get_block(
                &DiskBlockLocation {
                    path,
                    payload_len: payload.len() as u64,
                    fingerprint,
                },
                Some(fingerprint),
            )
            .expect_err("bad magic should fail");

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn disk_store_rejects_version_mismatch() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let kv_format_tag = 5;
        let payload = vec![8, 9, 10, 11, 12, 13, 14, 15];
        let fingerprint = fingerprint_for_payload(&payload, kv_format_tag);
        let path = store.path_for("wrong-version.kv").unwrap();
        store.create_root().unwrap();

        write_raw_block(
            &path,
            &DiskBlockHeader {
                magic: DISK_BLOCK_MAGIC,
                version: DISK_BLOCK_VERSION + 1,
                fingerprint: fingerprint.0,
                kv_format_tag,
                payload_len: payload.len() as u64,
            },
            &payload,
        );

        let err = store
            .get_block(
                &DiskBlockLocation {
                    path,
                    payload_len: payload.len() as u64,
                    fingerprint,
                },
                Some(fingerprint),
            )
            .expect_err("version mismatch should fail");

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn disk_store_filename_is_fingerprint_hex() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let kv_format_tag = 7;
        let payload_a = b"abcdefghijklmnop".to_vec();
        let payload_b = b"ponmlkjihgfedcba".to_vec();
        let fingerprint_a = fingerprint_for_payload(&payload_a, kv_format_tag);
        let fingerprint_b = fingerprint_for_payload(&payload_b, kv_format_tag);

        let location_a = store
            .put_block(fingerprint_a, kv_format_tag, &payload_a)
            .unwrap();
        let location_b = store
            .put_block(fingerprint_b, kv_format_tag, &payload_b)
            .unwrap();
        let location_a_again = store
            .put_block(fingerprint_a, kv_format_tag, &payload_a)
            .unwrap();

        assert_ne!(fingerprint_a, fingerprint_b);
        assert_ne!(location_a.path, location_b.path);
        assert_eq!(location_a.path, location_a_again.path);
        assert_eq!(
            location_a
                .path
                .file_name()
                .and_then(std::ffi::OsStr::to_str),
            Some(DiskStore::block_filename(fingerprint_a).as_str())
        );
        assert_eq!(
            location_b
                .path
                .file_name()
                .and_then(std::ffi::OsStr::to_str),
            Some(DiskStore::block_filename(fingerprint_b).as_str())
        );
        assert_eq!(
            store
                .get_block(&location_a_again, Some(fingerprint_a))
                .unwrap(),
            payload_a
        );
    }
}

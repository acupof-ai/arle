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
//! 2. each file starts with a postcard-encoded `DiskBlockHeader`
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
//! - Synchronous I/O via the `kv-native-sys` Zig file engine. The
//!   `KVTransport` trait is sync too, so this remains the natural fit;
//!   the coordinator does its own thread management.
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

use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::task::Poll;

use super::super::{
    backend::{KVBackend, KVBackendScope},
    io::{
        KVBackendCompletion, KVBackendDelete, KVBackendFetch, KVBackendStore, KVPayload,
        KVPayloadRef,
    },
    tier::BlockLocation,
};
use super::TransportError;
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
        let (header, payload) = Self::decode_prefix(bytes)?;
        let payload_len = usize::try_from(header.payload_len)
            .map_err(|_| invalid_data("disk store: payload length exceeds platform usize"))?;
        if payload.len() != payload_len {
            return Err(invalid_data("disk store: payload length mismatch"));
        }
        Ok((header, payload))
    }

    fn decode_prefix(bytes: &[u8]) -> io::Result<(Self, &[u8])> {
        let (header, payload) = postcard::take_from_bytes::<DiskBlockHeader>(bytes)
            .map_err(|err| invalid_data(format!("disk store: failed to decode header: {err}")))?;
        if header.magic != DISK_BLOCK_MAGIC {
            return Err(invalid_data("disk store: invalid block magic"));
        }
        if header.version != DISK_BLOCK_VERSION {
            return Err(invalid_data("disk store: unsupported block version"));
        }
        Ok((header, payload))
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn block_fingerprint_from_filename(name: &str) -> Option<BlockFingerprint> {
    let stem = name.strip_suffix(".kv")?;
    if stem.len() != 32 {
        return None;
    }
    let mut bytes = [0u8; 16];
    for (idx, pair) in stem.as_bytes().chunks_exact(2).enumerate() {
        let hi = hex_value(pair[0])?;
        let lo = hex_value(pair[1])?;
        bytes[idx] = (hi << 4) | lo;
    }
    Some(BlockFingerprint(bytes))
}

fn same_existing_path(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

fn read_file_prefix(path: &Path, max_len: usize) -> io::Result<Vec<u8>> {
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::with_capacity(max_len.min(8192));
    let limit = u64::try_from(max_len).unwrap_or(u64::MAX);
    file.by_ref().take(limit).read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn disk_error(err: &io::Error) -> TransportError {
    TransportError::Transfer(err.to_string())
}

fn disk_location_error(message: impl Into<String>) -> TransportError {
    TransportError::Other(message.into())
}

/// Stable local filesystem backing for persisted KV blobs.
#[derive(Debug)]
pub struct DiskStore {
    root: PathBuf,
}

#[derive(Debug)]
enum DiskBackendOpState {
    Ready(Result<KVBackendCompletion, TransportError>),
    Exhausted,
}

#[derive(Debug)]
pub struct DiskBackendOp {
    state: DiskBackendOpState,
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
        std::fs::create_dir_all(&self.root)
    }

    // ── Keyed API ────────────────────────────────────────────────────

    /// Writes `bytes` to `key`, creating parent directories as needed.
    pub fn write(&self, key: impl AsRef<Path>, bytes: &[u8]) -> io::Result<()> {
        let path = self.path_for(key)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        kv_native_sys::write_file(&path, bytes)
    }

    /// Reads the payload stored at `key`.
    pub fn read(&self, key: impl AsRef<Path>) -> io::Result<Vec<u8>> {
        let path = self.path_for(key)?;
        kv_native_sys::read_file(&path)
    }

    /// Removes the payload stored at `key`. Missing files are ignored.
    pub fn remove(&self, key: impl AsRef<Path>) -> io::Result<()> {
        let path = self.path_for(key)?;
        kv_native_sys::remove_file(&path, true)
    }

    // ── Block API (content-addressable) ──────────────────────────────

    /// Public canonical path for a fingerprinted block. Re-derived on
    /// every read/write from the store root + fingerprint — callers
    /// **cannot** influence the final path by tampering with a stored
    /// `DiskBlockLocation`. This is the defense against session
    /// snapshot–driven path traversal (M4 review finding B2).
    pub fn block_path_for(&self, fingerprint: BlockFingerprint) -> io::Result<PathBuf> {
        kv_native_sys::block_path(self.root(), fingerprint.0)
    }

    pub fn contains_block(&self, fingerprint: BlockFingerprint) -> io::Result<bool> {
        self.block_path_for(fingerprint)?.try_exists()
    }

    /// Visits every valid fingerprint-named block file currently under
    /// the store root. Invalid filenames are ignored; block contents are
    /// still validated with the normal postcard header before the
    /// visitor sees the opaque payload.
    pub fn visit_blocks(
        &self,
        mut visit: impl FnMut(DiskBlockLocation, &[u8]) -> io::Result<()>,
    ) -> io::Result<()> {
        let entries = match std::fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(err) => return Err(err),
        };

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if !entry.file_type()?.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(std::ffi::OsStr::to_str) else {
                continue;
            };
            let Some(fingerprint) = block_fingerprint_from_filename(name) else {
                continue;
            };

            let canonical = self.block_path_for(fingerprint)?;
            if !same_existing_path(&path, &canonical) {
                return Err(invalid_data(
                    "disk store: discovered block path outside canonical root",
                ));
            }

            let bytes = match kv_native_sys::read_block(self.root(), fingerprint.0) {
                Ok(bytes) => bytes,
                Err(err) => {
                    log::debug!(
                        "disk store: skipping unreadable block {}: {err}",
                        path.display()
                    );
                    continue;
                }
            };
            let (header, payload) = match DiskBlockHeader::decode(&bytes) {
                Ok(decoded) => decoded,
                Err(err) => {
                    log::debug!(
                        "disk store: skipping malformed block {}: {err}",
                        path.display()
                    );
                    continue;
                }
            };
            if header.fingerprint != fingerprint.0 {
                log::debug!("disk store: on-disk fingerprint does not match filename");
                continue;
            }

            visit(
                DiskBlockLocation {
                    path: canonical,
                    payload_len: header.payload_len,
                    fingerprint,
                },
                payload,
            )?;
        }

        Ok(())
    }

    /// Visits valid fingerprint-named block files but reads at most
    /// `max_payload_prefix_len` bytes from each payload. This is intended for
    /// startup indexing paths that only need caller-owned metadata near the
    /// front of the payload and must not page in full KV snapshots.
    pub fn visit_block_payload_prefixes(
        &self,
        max_payload_prefix_len: usize,
        mut visit: impl FnMut(DiskBlockLocation, &[u8]) -> io::Result<()>,
    ) -> io::Result<()> {
        const BLOCK_HEADER_READ_SLOP: usize = 512;
        let entries = match std::fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(err) => return Err(err),
        };

        let max_file_prefix_len = max_payload_prefix_len.saturating_add(BLOCK_HEADER_READ_SLOP);
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if !entry.file_type()?.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(std::ffi::OsStr::to_str) else {
                continue;
            };
            let Some(fingerprint) = block_fingerprint_from_filename(name) else {
                continue;
            };

            let canonical = self.block_path_for(fingerprint)?;
            if !same_existing_path(&path, &canonical) {
                return Err(invalid_data(
                    "disk store: discovered block path outside canonical root",
                ));
            }

            let bytes = match read_file_prefix(&canonical, max_file_prefix_len) {
                Ok(bytes) => bytes,
                Err(err) => {
                    log::debug!(
                        "disk store: skipping unreadable block {}: {err}",
                        path.display()
                    );
                    continue;
                }
            };
            let (header, payload_prefix) = match DiskBlockHeader::decode_prefix(&bytes) {
                Ok(decoded) => decoded,
                Err(err) => {
                    log::debug!(
                        "disk store: skipping malformed block {}: {err}",
                        path.display()
                    );
                    continue;
                }
            };
            if header.fingerprint != fingerprint.0 {
                log::debug!("disk store: on-disk fingerprint does not match filename");
                continue;
            }

            let header_len = bytes.len().saturating_sub(payload_prefix.len());
            let Some(expected_file_len) = u64::try_from(header_len)
                .ok()
                .and_then(|len| len.checked_add(header.payload_len))
            else {
                log::debug!(
                    "disk store: skipping block {} with overflowing payload length",
                    path.display()
                );
                continue;
            };
            let actual_file_len = match std::fs::metadata(&canonical) {
                Ok(metadata) => metadata.len(),
                Err(err) => {
                    log::debug!(
                        "disk store: skipping unstatable block {}: {err}",
                        path.display()
                    );
                    continue;
                }
            };
            if actual_file_len != expected_file_len {
                log::debug!(
                    "disk store: skipping block {} with payload length mismatch",
                    path.display()
                );
                continue;
            }

            let payload_prefix_len = payload_prefix
                .len()
                .min(max_payload_prefix_len)
                .min(usize::try_from(header.payload_len).unwrap_or(usize::MAX));
            visit(
                DiskBlockLocation {
                    path: canonical,
                    payload_len: header.payload_len,
                    fingerprint,
                },
                &payload_prefix[..payload_prefix_len],
            )?;
        }

        Ok(())
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
        self.put_block_with_fsync(fingerprint, kv_format_tag, payload, true)
    }

    /// Writes a content-addressed block, optionally skipping data and parent-dir
    /// fsync. `fsync_each_block=false` still uses write-to-temp + rename so
    /// readers never observe a partially written canonical block, but a crash
    /// may lose the most recent cache entry. That tradeoff is only appropriate
    /// for opportunistic caches that can fall back to recomputation.
    pub fn put_block_with_fsync(
        &self,
        fingerprint: BlockFingerprint,
        kv_format_tag: u8,
        payload: &[u8],
        fsync_each_block: bool,
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
        if fsync_each_block {
            kv_native_sys::write_block_atomic(self.root(), fingerprint.0, &file_bytes)?;
        } else {
            let tmp_path = path.with_extension("kv.tmp");
            let write_result = std::fs::write(&tmp_path, &file_bytes)
                .and_then(|()| std::fs::rename(&tmp_path, &path));
            if let Err(err) = write_result {
                let _ = std::fs::remove_file(&tmp_path);
                return Err(err);
            }
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

        let bytes = kv_native_sys::read_block(self.root(), location.fingerprint.0)?;
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
        if location.path != canonical {
            return Err(invalid_data(
                "disk store: refused location.path outside canonical root",
            ));
        }
        kv_native_sys::remove_block(self.root(), location.fingerprint.0, true)
    }

    fn ready_op(result: Result<KVBackendCompletion, TransportError>) -> DiskBackendOp {
        DiskBackendOp {
            state: DiskBackendOpState::Ready(result),
        }
    }

    fn location_from_handle(
        &self,
        handle: &super::super::chunk::KVHandle,
    ) -> Result<DiskBlockLocation, TransportError> {
        match &handle.location {
            BlockLocation::Disk {
                fingerprint,
                payload_len,
            } => {
                let path = self
                    .block_path_for(*fingerprint)
                    .map_err(|err| disk_error(&err))?;
                Ok(DiskBlockLocation {
                    path,
                    payload_len: *payload_len,
                    fingerprint: *fingerprint,
                })
            }
            _ => Err(disk_location_error(
                "disk backend requires a Disk block location on fetch/delete",
            )),
        }
    }
}

impl KVBackend for DiskStore {
    type Op = DiskBackendOp;

    fn backend_id(&self) -> &'static str {
        "disk"
    }

    fn scope(&self) -> KVBackendScope {
        KVBackendScope::NodeLocal
    }

    fn tier(&self) -> super::super::tier::Tier {
        super::super::tier::Tier::Disk
    }

    fn store(&self, req: KVBackendStore) -> Result<Self::Op, TransportError> {
        let fingerprint = req.block.fingerprint.ok_or_else(|| {
            disk_location_error("disk backend store requires a block fingerprint")
        })?;
        let payload_len = u64::try_from(req.payload.len())
            .map_err(|_| disk_location_error("disk backend store payload length exceeds u64"))?;
        self.put_block(fingerprint, req.kv_format_tag, req.payload.as_slice())
            .map_err(|err| disk_error(&err))?;
        let handle = req.handle.with_location(BlockLocation::Disk {
            fingerprint,
            payload_len,
        });
        Ok(Self::ready_op(Ok(KVBackendCompletion::Stored(handle))))
    }

    fn fetch(&self, req: KVBackendFetch) -> Result<Self::Op, TransportError> {
        let location = self.location_from_handle(&req.handle)?;
        let bytes = self
            .get_block(&location, Some(location.fingerprint))
            .map_err(|err| disk_error(&err))?;
        let payload_len = bytes.len() as u64;
        let payload = KVPayload::with_reference(
            bytes,
            KVPayloadRef::whole(req.handle.location.clone(), payload_len),
        );
        Ok(Self::ready_op(Ok(KVBackendCompletion::Loaded {
            handle: req.handle,
            payload,
        })))
    }

    fn delete(&self, req: KVBackendDelete) -> Result<Self::Op, TransportError> {
        let location = self.location_from_handle(&req.handle)?;
        self.delete_block(&location)
            .map_err(|err| disk_error(&err))?;
        Ok(Self::ready_op(Ok(KVBackendCompletion::Deleted(req.handle))))
    }

    fn exists(&self, handle: &super::super::chunk::KVHandle) -> Result<bool, TransportError> {
        match &handle.location {
            BlockLocation::Disk { fingerprint, .. } => self
                .contains_block(*fingerprint)
                .map_err(|err| disk_error(&err)),
            _ => Err(disk_location_error(
                "disk backend requires a Disk block location on exists",
            )),
        }
    }

    fn poll(&self, op: &mut Self::Op) -> Poll<Result<KVBackendCompletion, TransportError>> {
        match std::mem::replace(&mut op.state, DiskBackendOpState::Exhausted) {
            DiskBackendOpState::Ready(result) => Poll::Ready(result),
            DiskBackendOpState::Exhausted => Poll::Ready(Err(TransportError::Other(
                "disk backend op already completed".into(),
            ))),
        }
    }

    fn abort(&self, op: &mut Self::Op) {
        op.state = DiskBackendOpState::Ready(Err(TransportError::Aborted));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::task::Poll;

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

        assert_eq!(location.path, dir.path().join(block_filename(fingerprint)));
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
    fn put_block_with_fsync_false_roundtrips() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let kv_format_tag = 4;
        let payload = b"relaxed-cache-write00000".to_vec();
        let fingerprint = fingerprint_for_payload(&payload, kv_format_tag);

        let location = store
            .put_block_with_fsync(fingerprint, kv_format_tag, &payload, false)
            .expect("put relaxed block");

        assert_eq!(
            store
                .get_block(&location, Some(fingerprint))
                .expect("get relaxed block"),
            payload
        );
        assert!(!location.path.with_extension("kv.tmp").exists());
    }

    #[test]
    fn contains_block_tracks_fingerprint_presence() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let kv_format_tag = 7;
        let payload = b"exists-check".to_vec();
        let fingerprint = fingerprint_for_payload(&payload, kv_format_tag);

        assert!(
            !store
                .contains_block(fingerprint)
                .expect("missing before write")
        );
        let location = store
            .put_block(fingerprint, kv_format_tag, &payload)
            .expect("put block");
        assert!(
            store
                .contains_block(fingerprint)
                .expect("present after write")
        );
        store.delete_block(&location).expect("delete block");
        assert!(
            !store
                .contains_block(fingerprint)
                .expect("missing after delete")
        );
    }

    #[test]
    fn visit_blocks_returns_valid_fingerprint_blocks() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let payload_a = b"payload-a000".to_vec();
        let payload_b = b"payload-b000".to_vec();
        let fingerprint_a = fingerprint_for_payload(&payload_a, 1);
        let fingerprint_b = fingerprint_for_payload(&payload_b, 1);
        store.put_block(fingerprint_a, 1, &payload_a).unwrap();
        store.put_block(fingerprint_b, 1, &payload_b).unwrap();
        fs::write(dir.path().join("not-a-block.txt"), b"ignored").unwrap();

        let mut visited = Vec::new();
        store
            .visit_blocks(|location, payload| {
                visited.push((location.fingerprint, payload.to_vec()));
                Ok(())
            })
            .expect("visit blocks");
        visited.sort_by_key(|(fingerprint, _)| fingerprint.0);

        let mut expected = vec![(fingerprint_a, payload_a), (fingerprint_b, payload_b)];
        expected.sort_by_key(|(fingerprint, _)| fingerprint.0);
        assert_eq!(visited, expected);
    }

    #[test]
    fn visit_blocks_accepts_trailing_slash_roots() {
        let dir = tempdir().unwrap();
        let root = PathBuf::from(format!("{}/", dir.path().display()));
        let store = DiskStore::new(root);
        let payload = b"payload-trailing-slash00".to_vec();
        let fingerprint = fingerprint_for_payload(&payload, 1);
        store.put_block(fingerprint, 1, &payload).unwrap();

        let mut visited = Vec::new();
        store
            .visit_blocks(|location, payload| {
                visited.push((location.fingerprint, payload.to_vec()));
                Ok(())
            })
            .expect("visit trailing slash root");

        assert_eq!(visited, vec![(fingerprint, payload)]);
    }

    #[test]
    fn visit_block_payload_prefixes_avoids_full_payload_reads() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let payload = vec![7u8; 4096];
        let fingerprint = fingerprint_for_payload(&payload, 1);
        store.put_block(fingerprint, 1, &payload).unwrap();

        let mut visited = Vec::new();
        store
            .visit_block_payload_prefixes(32, |location, payload_prefix| {
                visited.push((location.fingerprint, payload_prefix.len()));
                Ok(())
            })
            .expect("visit block payload prefixes");

        assert_eq!(visited, vec![(fingerprint, 32)]);
    }

    #[test]
    fn visit_block_payload_prefixes_skips_payload_len_mismatch() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        store.create_root().unwrap();
        let payload = vec![7u8; 64];
        let fingerprint = fingerprint_for_payload(&payload, 1);
        let header = DiskBlockHeader {
            magic: DISK_BLOCK_MAGIC,
            version: DISK_BLOCK_VERSION,
            fingerprint: fingerprint.0,
            kv_format_tag: 1,
            payload_len: 1024,
        };
        write_raw_block(
            &store.block_path_for(fingerprint).unwrap(),
            &header,
            &payload,
        );

        let mut visited = 0usize;
        store
            .visit_block_payload_prefixes(32, |_, _| {
                visited += 1;
                Ok(())
            })
            .expect("visit block payload prefixes");

        assert_eq!(visited, 0);
    }

    #[test]
    fn visit_blocks_ignores_missing_root() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path().join("missing"));
        let mut visited = 0usize;
        store
            .visit_blocks(|_, _| {
                visited += 1;
                Ok(())
            })
            .expect("missing root is empty");
        assert_eq!(visited, 0);
    }

    #[test]
    fn visit_blocks_skips_malformed_fingerprint_blocks() {
        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let payload = b"valid-payload000".to_vec();
        let valid_fingerprint = fingerprint_for_payload(&payload, 1);
        store.put_block(valid_fingerprint, 1, &payload).unwrap();

        let malformed_fingerprint = BlockFingerprint([0x11; 16]);
        store.create_root().unwrap();
        let malformed_path = store.block_path_for(malformed_fingerprint).unwrap();
        fs::write(malformed_path, b"not a disk block").unwrap();

        let mut visited = Vec::new();
        store
            .visit_blocks(|location, payload| {
                visited.push((location.fingerprint, payload.to_vec()));
                Ok(())
            })
            .expect("malformed block is skipped");
        assert_eq!(visited, vec![(valid_fingerprint, payload)]);
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
            Some(block_filename(fingerprint_a).as_str())
        );
        assert_eq!(
            location_b
                .path
                .file_name()
                .and_then(std::ffi::OsStr::to_str),
            Some(block_filename(fingerprint_b).as_str())
        );
        assert_eq!(
            store
                .get_block(&location_a_again, Some(fingerprint_a))
                .unwrap(),
            payload_a
        );
    }

    #[test]
    fn disk_backend_store_fetch_delete_roundtrip() {
        use crate::kv_tier::{
            KVBackend, KVBackendCompletion, KVBackendDelete, KVBackendFetch, KVBackendStore,
            KVBlock, KVHandle, KVPayload, KVSpanId, LayerRange, TokenRange,
        };

        let dir = tempdir().unwrap();
        let store = DiskStore::new(dir.path());
        let kv_format_tag = 7;
        let payload = b"hicache-backend!".to_vec();
        let fingerprint = fingerprint_for_payload(&payload, kv_format_tag);
        let block = KVBlock::new(
            crate::types::BlockId(4),
            LayerRange::new(0, 2),
            TokenRange::new(0, 16),
            payload.len() as u64,
        )
        .with_fingerprint(fingerprint);
        let host_handle = KVHandle::new(
            Some(KVSpanId(9)),
            block.block_id,
            BlockLocation::HostPinned { offset: 0 },
            1,
            payload.len() as u64,
        );

        let mut store_op = store
            .store(KVBackendStore {
                handle: host_handle.clone(),
                block: block.clone(),
                kv_format_tag,
                payload: KVPayload::from_vec(payload.clone()),
            })
            .unwrap();
        let stored = match store.poll(&mut store_op) {
            Poll::Ready(Ok(KVBackendCompletion::Stored(handle))) => handle,
            other => panic!("expected stored completion, got {other:?}"),
        };

        let mut fetch_op = store
            .fetch(KVBackendFetch {
                handle: stored.clone(),
            })
            .unwrap();
        match store.poll(&mut fetch_op) {
            Poll::Ready(Ok(KVBackendCompletion::Loaded {
                handle,
                payload: got,
            })) => {
                assert_eq!(handle, stored);
                assert_eq!(got.as_slice(), payload.as_slice());
            }
            other => panic!("expected loaded completion, got {other:?}"),
        }

        let mut delete_op = store.delete(KVBackendDelete { handle: stored }).unwrap();
        match store.poll(&mut delete_op) {
            Poll::Ready(Ok(KVBackendCompletion::Deleted(_))) => {}
            other => panic!("expected deleted completion, got {other:?}"),
        }
    }
}

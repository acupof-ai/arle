//! Chunked snapshot object store for slower KV tiers.
//!
//! This is backend-neutral: callers provide codec metadata plus one or more
//! byte parts. The store splits each part into content-addressed chunks, writes
//! only missing chunks, and commits a small manifest last. Backends such as
//! Metal or CUDA own their KV codec; this module owns the disk layout, chunk
//! dedupe, and bounded-parallel file I/O.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::crc32c;
use crate::types::BlockFingerprint;

const MANIFEST_MAGIC: [u8; 8] = *b"KVSNAP01";
const MANIFEST_VERSION: u16 = 1;
const DEFAULT_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug)]
pub struct ChunkedSnapshotStore {
    root: PathBuf,
    chunk_bytes: usize,
    read_parallelism: usize,
    write_parallelism: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedSnapshotLocation {
    pub manifest_id: BlockFingerprint,
    pub path: PathBuf,
    pub payload_len: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedSnapshotPartWrite {
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedSnapshotWrite {
    pub manifest_id: BlockFingerprint,
    pub namespace: String,
    pub metadata: Vec<u8>,
    pub parts: Vec<ChunkedSnapshotPartWrite>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedSnapshotPartRead {
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedSnapshotRead {
    pub manifest: ChunkedSnapshotManifest,
    pub parts: Vec<ChunkedSnapshotPartRead>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChunkedSnapshotManifest {
    pub namespace: String,
    pub metadata: Vec<u8>,
    pub payload_len: u64,
    pub metadata_crc32c: u32,
    pub metadata_checksum: [u8; 32],
    pub parts: Vec<ChunkedSnapshotPartManifest>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChunkedSnapshotPartManifest {
    pub name: String,
    pub byte_len: u64,
    pub chunks: Vec<ChunkedSnapshotChunkRef>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChunkedSnapshotChunkRef {
    pub chunk_id: BlockFingerprint,
    pub offset: u64,
    pub len: u64,
    pub crc32c: u32,
    pub segment_id: Option<BlockFingerprint>,
    pub segment_offset: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedSnapshotPutStats {
    pub chunks_written: usize,
    pub chunks_reused: usize,
    pub logical_payload_bytes: u64,
    pub physical_chunk_bytes_written: u64,
    pub manifest_bytes_written: u64,
}

impl ChunkedSnapshotStore {
    pub fn new(root: impl AsRef<Path>) -> Self {
        let parallelism = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .clamp(1, 8);
        Self {
            root: root.as_ref().to_path_buf(),
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            read_parallelism: parallelism,
            write_parallelism: parallelism,
        }
    }

    #[must_use]
    pub fn with_chunk_bytes(mut self, chunk_bytes: usize) -> Self {
        self.chunk_bytes = chunk_bytes.max(4096);
        self
    }

    #[must_use]
    pub fn with_parallelism(mut self, parallelism: usize) -> Self {
        let parallelism = parallelism.max(1);
        self.read_parallelism = parallelism;
        self.write_parallelism = parallelism;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn create_root(&self) -> io::Result<()> {
        fs::create_dir_all(self.chunks_dir())?;
        fs::create_dir_all(self.segments_dir())?;
        fs::create_dir_all(self.manifests_dir())
    }

    pub fn manifest_path_for(&self, manifest_id: BlockFingerprint) -> PathBuf {
        self.manifests_dir()
            .join(format!("{}.manifest", fingerprint_hex(manifest_id)))
    }

    pub fn chunk_path_for(&self, chunk_id: BlockFingerprint) -> PathBuf {
        self.chunks_dir()
            .join(format!("{}.chunk", fingerprint_hex(chunk_id)))
    }

    pub fn segment_path_for(&self, segment_id: BlockFingerprint) -> PathBuf {
        self.segments_dir()
            .join(format!("{}.segment", fingerprint_hex(segment_id)))
    }

    pub fn put_snapshot(
        &self,
        write: ChunkedSnapshotWrite,
        fsync_manifest: bool,
    ) -> io::Result<(ChunkedSnapshotLocation, ChunkedSnapshotPutStats)> {
        self.create_root()?;
        validate_manifest_id(write.manifest_id)?;
        if write.namespace.is_empty() {
            return Err(invalid_data("chunked snapshot namespace must not be empty"));
        }

        let metadata_len = u64::try_from(write.metadata.len())
            .map_err(|_| invalid_data("chunked snapshot metadata too large"))?;
        let existing_chunks = self.live_chunk_locations()?;
        let mut new_chunks = Vec::new();
        let mut parts = Vec::with_capacity(write.parts.len());
        let mut logical_payload_bytes = 0u64;
        let mut seen_chunks: HashMap<BlockFingerprint, PendingChunkStorage> = HashMap::new();
        let mut new_chunk_fixups = Vec::new();

        for part in &write.parts {
            if part.name.is_empty() {
                return Err(invalid_data("chunked snapshot part name must not be empty"));
            }
            let part_idx = parts.len();
            let mut chunks = Vec::new();
            let byte_len = u64::try_from(part.bytes.len())
                .map_err(|_| invalid_data("chunked snapshot part too large"))?;
            logical_payload_bytes = logical_payload_bytes
                .checked_add(byte_len)
                .ok_or_else(|| invalid_data("chunked snapshot payload length overflow"))?;
            for (idx, bytes) in part.bytes.chunks(self.chunk_bytes).enumerate() {
                let offset = u64::try_from(idx)
                    .ok()
                    .and_then(|idx| idx.checked_mul(self.chunk_bytes as u64))
                    .ok_or_else(|| invalid_data("chunked snapshot chunk offset overflow"))?;
                let chunk_id = chunk_fingerprint(&write.namespace, &part.name, offset, bytes);
                let len = u64::try_from(bytes.len())
                    .map_err(|_| invalid_data("chunked snapshot chunk too large"))?;
                let mut chunk_ref = ChunkedSnapshotChunkRef {
                    chunk_id,
                    offset,
                    len,
                    crc32c: crc32c::checksum(bytes),
                    segment_id: None,
                    segment_offset: 0,
                };
                let chunk_storage = if let Some(storage) = seen_chunks.get(&chunk_id).copied() {
                    storage
                } else if let Some(existing) = existing_chunks.get(&chunk_id).copied() {
                    seen_chunks.insert(chunk_id, PendingChunkStorage::Existing(existing));
                    PendingChunkStorage::Existing(existing)
                } else {
                    let new_idx = new_chunks.len();
                    new_chunks.push(PendingSegmentChunk {
                        chunk_id,
                        bytes: Arc::<[u8]>::from(bytes),
                        segment_offset: 0,
                    });
                    let storage = PendingChunkStorage::New(new_idx);
                    seen_chunks.insert(chunk_id, storage);
                    storage
                };
                match chunk_storage {
                    PendingChunkStorage::Existing(storage) => {
                        chunk_ref.segment_id = storage.segment_id;
                        chunk_ref.segment_offset = storage.segment_offset;
                    }
                    PendingChunkStorage::New(new_idx) => {
                        new_chunk_fixups.push((part_idx, chunks.len(), new_idx));
                    }
                }
                chunks.push(chunk_ref);
            }
            parts.push(ChunkedSnapshotPartManifest {
                name: part.name.clone(),
                byte_len,
                chunks,
            });
        }

        let chunks_written = new_chunks.len();
        let (segment_id, physical_chunk_bytes_written) =
            self.write_segment_for_new_chunks(write.manifest_id, &mut new_chunks, fsync_manifest)?;
        if let Some(segment_id) = segment_id {
            for (part_idx, chunk_idx, new_idx) in new_chunk_fixups {
                parts[part_idx].chunks[chunk_idx].segment_id = Some(segment_id);
                parts[part_idx].chunks[chunk_idx].segment_offset =
                    new_chunks[new_idx].segment_offset;
            }
        }

        let logical_snapshot_bytes = metadata_len
            .checked_add(logical_payload_bytes)
            .ok_or_else(|| invalid_data("chunked snapshot total length overflow"))?;
        let metadata_crc32c = crc32c::checksum(&write.metadata);
        let metadata_checksum = *blake3::hash(&write.metadata).as_bytes();
        let manifest = ChunkedSnapshotManifest {
            namespace: write.namespace,
            metadata: write.metadata,
            payload_len: logical_payload_bytes,
            metadata_crc32c,
            metadata_checksum,
            parts,
        };
        let manifest_bytes = encode_manifest(&manifest)?;
        let manifest_bytes_written = manifest_bytes.len() as u64;
        let path = self.manifest_path_for(write.manifest_id);
        write_file_atomic(&path, &manifest_bytes, fsync_manifest)?;

        let referenced_chunks = manifest
            .parts
            .iter()
            .map(|part| part.chunks.len())
            .sum::<usize>();
        let chunks_reused = referenced_chunks.saturating_sub(chunks_written);
        Ok((
            ChunkedSnapshotLocation {
                manifest_id: write.manifest_id,
                path,
                payload_len: logical_snapshot_bytes,
            },
            ChunkedSnapshotPutStats {
                chunks_written,
                chunks_reused,
                logical_payload_bytes: logical_snapshot_bytes,
                physical_chunk_bytes_written,
                manifest_bytes_written,
            },
        ))
    }

    pub fn get_snapshot(
        &self,
        location: &ChunkedSnapshotLocation,
        expected_manifest_id: Option<BlockFingerprint>,
    ) -> io::Result<ChunkedSnapshotRead> {
        if let Some(expected) = expected_manifest_id
            && location.manifest_id != expected
        {
            return Err(invalid_data("chunked snapshot manifest id mismatch"));
        }
        let canonical = self.manifest_path_for(location.manifest_id);
        if location.path != canonical {
            return Err(invalid_data(
                "chunked snapshot refused location.path outside canonical root",
            ));
        }
        let manifest = decode_manifest(&fs::read(&canonical)?)?;
        validate_manifest(&manifest)?;
        let parts = self.read_parts_parallel(&manifest)?;
        Ok(ChunkedSnapshotRead { manifest, parts })
    }

    pub fn visit_manifests(
        &self,
        mut visit: impl FnMut(ChunkedSnapshotLocation, &ChunkedSnapshotManifest) -> io::Result<()>,
    ) -> io::Result<()> {
        let entries = match fs::read_dir(self.manifests_dir()) {
            Ok(entries) => entries,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(err) => return Err(err),
        };

        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let path = entry.path();
            let Some(name) = path.file_name().and_then(std::ffi::OsStr::to_str) else {
                continue;
            };
            let Some(stem) = name.strip_suffix(".manifest") else {
                continue;
            };
            let Some(manifest_id) = fingerprint_from_hex(stem) else {
                continue;
            };
            let canonical = self.manifest_path_for(manifest_id);
            if path != canonical {
                return Err(invalid_data(
                    "chunked snapshot discovered manifest outside canonical root",
                ));
            }
            let manifest = match decode_manifest(&fs::read(&canonical)?) {
                Ok(manifest) => manifest,
                Err(err) => {
                    log::debug!(
                        "chunked snapshot: skipping malformed manifest {}: {err}",
                        canonical.display()
                    );
                    continue;
                }
            };
            if let Err(err) = validate_manifest(&manifest) {
                log::debug!(
                    "chunked snapshot: skipping invalid manifest {}: {err}",
                    canonical.display()
                );
                continue;
            }
            visit(
                ChunkedSnapshotLocation {
                    manifest_id,
                    path: canonical,
                    payload_len: u64::try_from(manifest.metadata.len())
                        .map_err(|_| invalid_data("chunked snapshot metadata too large"))?
                        .checked_add(manifest.payload_len)
                        .ok_or_else(|| invalid_data("chunked snapshot total length overflow"))?,
                },
                &manifest,
            )?;
        }
        Ok(())
    }

    pub fn delete_snapshot(&self, location: &ChunkedSnapshotLocation) -> io::Result<()> {
        let canonical = self.manifest_path_for(location.manifest_id);
        if location.path != canonical {
            return Err(invalid_data(
                "chunked snapshot refused delete outside canonical root",
            ));
        }
        match fs::remove_file(canonical) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err),
        }
    }

    pub fn collect_orphan_chunks(&self) -> io::Result<usize> {
        let mut referenced = HashSet::new();
        let mut referenced_segments = HashSet::new();
        self.visit_manifests(|_, manifest| {
            for part in &manifest.parts {
                for chunk in &part.chunks {
                    referenced.insert(chunk.chunk_id);
                    if let Some(segment_id) = chunk.segment_id {
                        referenced_segments.insert(segment_id);
                    }
                }
            }
            Ok(())
        })?;

        let entries = match fs::read_dir(self.chunks_dir()) {
            Ok(entries) => entries,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(err) => return Err(err),
        };
        let mut removed = 0usize;
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let path = entry.path();
            let Some(name) = path.file_name().and_then(std::ffi::OsStr::to_str) else {
                continue;
            };
            let Some(stem) = name.strip_suffix(".chunk") else {
                continue;
            };
            let Some(chunk_id) = fingerprint_from_hex(stem) else {
                continue;
            };
            if !referenced.contains(&chunk_id) {
                fs::remove_file(path)?;
                removed += 1;
            }
        }
        let segment_entries = match fs::read_dir(self.segments_dir()) {
            Ok(entries) => entries,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(removed),
            Err(err) => return Err(err),
        };
        for entry in segment_entries {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let path = entry.path();
            let Some(name) = path.file_name().and_then(std::ffi::OsStr::to_str) else {
                continue;
            };
            let Some(stem) = name.strip_suffix(".segment") else {
                continue;
            };
            let Some(segment_id) = fingerprint_from_hex(stem) else {
                continue;
            };
            if !referenced_segments.contains(&segment_id) {
                fs::remove_file(path)?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    fn live_chunk_locations(&self) -> io::Result<HashMap<BlockFingerprint, ChunkStorageLocation>> {
        let mut out = HashMap::new();
        self.visit_manifests(|_, manifest| {
            for part in &manifest.parts {
                for chunk in &part.chunks {
                    let storage = ChunkStorageLocation {
                        segment_id: chunk.segment_id,
                        segment_offset: chunk.segment_offset,
                    };
                    let exists = if let Some(segment_id) = storage.segment_id {
                        self.segment_path_for(segment_id).try_exists()?
                    } else {
                        self.chunk_path_for(chunk.chunk_id).try_exists()?
                    };
                    if exists {
                        out.entry(chunk.chunk_id).or_insert(storage);
                    }
                }
            }
            Ok(())
        })?;
        Ok(out)
    }

    fn write_segment_for_new_chunks(
        &self,
        manifest_id: BlockFingerprint,
        chunks: &mut [PendingSegmentChunk],
        fsync_segment: bool,
    ) -> io::Result<(Option<BlockFingerprint>, u64)> {
        if chunks.is_empty() {
            return Ok((None, 0));
        }
        let segment_id = segment_fingerprint(manifest_id, chunks);
        let path = self.segment_path_for(segment_id);
        if path.try_exists()? {
            let mut offset = 0u64;
            for chunk in chunks {
                chunk.segment_offset = offset;
                offset = offset
                    .checked_add(u64::try_from(chunk.bytes.len()).map_err(|_| {
                        invalid_data("chunked snapshot segment chunk length exceeds u64")
                    })?)
                    .ok_or_else(|| invalid_data("chunked snapshot segment offset overflow"))?;
            }
            return Ok((Some(segment_id), 0));
        }

        let total_len = chunks.iter().try_fold(0u64, |acc, chunk| {
            acc.checked_add(
                u64::try_from(chunk.bytes.len())
                    .map_err(|_| invalid_data("chunked snapshot segment chunk too large"))?,
            )
            .ok_or_else(|| invalid_data("chunked snapshot segment length overflow"))
        })?;
        let mut bytes = Vec::with_capacity(
            usize::try_from(total_len)
                .map_err(|_| invalid_data("chunked snapshot segment exceeds usize"))?,
        );
        for chunk in chunks {
            chunk.segment_offset = u64::try_from(bytes.len())
                .map_err(|_| invalid_data("chunked snapshot segment offset exceeds u64"))?;
            bytes.extend_from_slice(&chunk.bytes);
        }
        write_file_atomic(&path, &bytes, fsync_segment)?;
        Ok((Some(segment_id), total_len))
    }

    fn chunks_dir(&self) -> PathBuf {
        self.root.join("chunks")
    }

    fn segments_dir(&self) -> PathBuf {
        self.root.join("segments")
    }

    fn manifests_dir(&self) -> PathBuf {
        self.root.join("manifests")
    }

    fn read_parts_parallel(
        &self,
        manifest: &ChunkedSnapshotManifest,
    ) -> io::Result<Vec<ChunkedSnapshotPartRead>> {
        let mut tasks = Vec::new();
        for (part_idx, part) in manifest.parts.iter().enumerate() {
            for (chunk_idx, chunk) in part.chunks.iter().enumerate() {
                let (storage, path) = if let Some(segment_id) = chunk.segment_id {
                    (
                        ChunkReadStorage::Segment(segment_id),
                        self.segment_path_for(segment_id),
                    )
                } else {
                    (
                        ChunkReadStorage::File(chunk.chunk_id),
                        self.chunk_path_for(chunk.chunk_id),
                    )
                };
                tasks.push(ChunkReadTask {
                    namespace: manifest.namespace.clone(),
                    part_name: part.name.clone(),
                    part_idx,
                    chunk_idx,
                    chunk: chunk.clone(),
                    storage,
                    path,
                });
            }
        }
        let results = read_chunks_parallel(tasks, self.read_parallelism)?;
        let mut part_chunks = manifest
            .parts
            .iter()
            .map(|part| vec![Vec::<u8>::new(); part.chunks.len()])
            .collect::<Vec<_>>();
        for result in results {
            part_chunks[result.part_idx][result.chunk_idx] = result.bytes;
        }

        let mut out = Vec::with_capacity(manifest.parts.len());
        for (part, chunks) in manifest.parts.iter().zip(part_chunks) {
            let mut bytes = Vec::with_capacity(
                usize::try_from(part.byte_len)
                    .map_err(|_| invalid_data("chunked snapshot part exceeds usize"))?,
            );
            for chunk in chunks {
                bytes.extend_from_slice(&chunk);
            }
            if bytes.len() as u64 != part.byte_len {
                return Err(invalid_data("chunked snapshot part length mismatch"));
            }
            out.push(ChunkedSnapshotPartRead {
                name: part.name.clone(),
                bytes,
            });
        }
        Ok(out)
    }
}

#[derive(Clone)]
struct PendingSegmentChunk {
    chunk_id: BlockFingerprint,
    bytes: Arc<[u8]>,
    segment_offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChunkStorageLocation {
    segment_id: Option<BlockFingerprint>,
    segment_offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PendingChunkStorage {
    Existing(ChunkStorageLocation),
    New(usize),
}

#[derive(Clone)]
struct ChunkReadTask {
    namespace: String,
    part_name: String,
    part_idx: usize,
    chunk_idx: usize,
    chunk: ChunkedSnapshotChunkRef,
    storage: ChunkReadStorage,
    path: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum ChunkReadStorage {
    Segment(BlockFingerprint),
    File(BlockFingerprint),
}

#[derive(Clone)]
struct ChunkReadGroup {
    storage: ChunkReadStorage,
    tasks: Vec<ChunkReadTask>,
}

struct ChunkReadResult {
    part_idx: usize,
    chunk_idx: usize,
    bytes: Vec<u8>,
}

fn read_chunks_parallel(
    chunks: Vec<ChunkReadTask>,
    parallelism: usize,
) -> io::Result<Vec<ChunkReadResult>> {
    let mut group_by_storage = HashMap::<ChunkReadStorage, Vec<ChunkReadTask>>::new();
    for task in chunks {
        group_by_storage.entry(task.storage).or_default().push(task);
    }
    let groups = group_by_storage
        .into_iter()
        .map(|(storage, tasks)| ChunkReadGroup { storage, tasks })
        .collect::<Vec<_>>();
    let grouped = run_parallel(groups, parallelism, read_chunk_group)?;
    Ok(grouped.into_iter().flatten().collect())
}

fn read_chunk_group(group: ChunkReadGroup) -> io::Result<Vec<ChunkReadResult>> {
    let mut out = Vec::with_capacity(group.tasks.len());
    match group.storage {
        ChunkReadStorage::Segment(segment_id) => {
            let path = group
                .tasks
                .first()
                .ok_or_else(|| invalid_data("chunk read group is empty"))?
                .path
                .clone();
            if !path.ends_with(format!("{}.segment", fingerprint_hex(segment_id))) {
                return Err(invalid_data("chunked snapshot segment path mismatch"));
            }
            let mut file = fs::File::open(&path)?;
            let mut tasks = group.tasks;
            tasks.sort_by_key(|task| task.chunk.segment_offset);
            for task in tasks {
                file.seek(SeekFrom::Start(task.chunk.segment_offset))?;
                let len = usize::try_from(task.chunk.len)
                    .map_err(|_| invalid_data("chunked snapshot chunk exceeds usize"))?;
                let mut bytes = vec![0u8; len];
                file.read_exact(&mut bytes)?;
                out.push(validate_read_chunk(task, bytes)?);
            }
        }
        ChunkReadStorage::File(_chunk_id) => {
            for task in group.tasks {
                let bytes = fs::read(&task.path)?;
                out.push(validate_read_chunk(task, bytes)?);
            }
        }
    }
    Ok(out)
}

fn validate_read_chunk(task: ChunkReadTask, bytes: Vec<u8>) -> io::Result<ChunkReadResult> {
    if bytes.len() as u64 != task.chunk.len {
        return Err(invalid_data("chunked snapshot chunk length mismatch"));
    }
    if crc32c::checksum(&bytes) != task.chunk.crc32c {
        return Err(invalid_data("chunked snapshot chunk crc32c mismatch"));
    }
    let actual = chunk_fingerprint(&task.namespace, &task.part_name, task.chunk.offset, &bytes);
    if task.chunk.chunk_id != actual {
        return Err(invalid_data("chunked snapshot chunk checksum mismatch"));
    }
    Ok(ChunkReadResult {
        part_idx: task.part_idx,
        chunk_idx: task.chunk_idx,
        bytes,
    })
}

fn run_parallel<T, R, F>(items: Vec<T>, parallelism: usize, op: F) -> io::Result<Vec<R>>
where
    T: Send + Clone,
    R: Send,
    F: Fn(T) -> io::Result<R> + Sync,
{
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let parallelism = parallelism.max(1).min(items.len());
    if parallelism == 1 {
        return items.into_iter().map(op).collect();
    }

    let mut shards = vec![Vec::new(); parallelism];
    for (idx, item) in items.into_iter().enumerate() {
        shards[idx % parallelism].push(item);
    }
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(parallelism);
        for shard in shards {
            let op_ref = &op;
            handles.push(scope.spawn(move || {
                shard
                    .into_iter()
                    .map(op_ref)
                    .collect::<io::Result<Vec<_>>>()
            }));
        }
        let mut out = Vec::new();
        for handle in handles {
            let mut shard = handle
                .join()
                .map_err(|_| invalid_data("chunked snapshot worker panicked"))??;
            out.append(&mut shard);
        }
        Ok(out)
    })
}

fn validate_manifest(manifest: &ChunkedSnapshotManifest) -> io::Result<()> {
    if manifest.namespace.is_empty() {
        return Err(invalid_data("chunked snapshot manifest namespace is empty"));
    }
    if manifest.metadata_crc32c != crc32c::checksum(&manifest.metadata) {
        return Err(invalid_data("chunked snapshot metadata crc32c mismatch"));
    }
    if manifest.metadata_checksum != *blake3::hash(&manifest.metadata).as_bytes() {
        return Err(invalid_data("chunked snapshot metadata checksum mismatch"));
    }
    let mut payload_len = 0u64;
    for part in &manifest.parts {
        let mut offset = 0u64;
        for chunk in &part.chunks {
            if chunk.offset != offset {
                return Err(invalid_data("chunked snapshot chunk offset mismatch"));
            }
            if chunk.segment_id.is_none() && chunk.segment_offset != 0 {
                return Err(invalid_data(
                    "chunked snapshot standalone chunk has non-zero segment offset",
                ));
            }
            offset = offset
                .checked_add(chunk.len)
                .ok_or_else(|| invalid_data("chunked snapshot chunk length overflow"))?;
        }
        if offset != part.byte_len {
            return Err(invalid_data("chunked snapshot part byte length mismatch"));
        }
        payload_len = payload_len
            .checked_add(part.byte_len)
            .ok_or_else(|| invalid_data("chunked snapshot payload length overflow"))?;
    }
    if payload_len != manifest.payload_len {
        return Err(invalid_data("chunked snapshot payload length mismatch"));
    }
    Ok(())
}

fn encode_manifest(manifest: &ChunkedSnapshotManifest) -> io::Result<Vec<u8>> {
    let manifest_bytes = postcard::to_allocvec(manifest)
        .map_err(|err| invalid_data(format!("encode chunked snapshot manifest: {err}")))?;
    let mut out = Vec::with_capacity(MANIFEST_MAGIC.len() + 2 + manifest_bytes.len());
    out.extend_from_slice(&MANIFEST_MAGIC);
    out.extend_from_slice(&MANIFEST_VERSION.to_le_bytes());
    out.extend_from_slice(&manifest_bytes);
    Ok(out)
}

fn decode_manifest(bytes: &[u8]) -> io::Result<ChunkedSnapshotManifest> {
    if bytes.len() < MANIFEST_MAGIC.len() + 2 {
        return Err(invalid_data("chunked snapshot manifest too short"));
    }
    if bytes[..MANIFEST_MAGIC.len()] != MANIFEST_MAGIC {
        return Err(invalid_data("chunked snapshot manifest magic mismatch"));
    }
    let version = u16::from_le_bytes(
        bytes[MANIFEST_MAGIC.len()..MANIFEST_MAGIC.len() + 2]
            .try_into()
            .map_err(|_| invalid_data("chunked snapshot manifest version truncated"))?,
    );
    if version != MANIFEST_VERSION {
        return Err(invalid_data("chunked snapshot manifest version mismatch"));
    }
    postcard::from_bytes(&bytes[MANIFEST_MAGIC.len() + 2..])
        .map_err(|err| invalid_data(format!("decode chunked snapshot manifest: {err}")))
}

fn write_file_atomic(path: &Path, bytes: &[u8], fsync: bool) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| invalid_data("chunked snapshot path has no parent"))?;
    fs::create_dir_all(parent)?;
    let tmp = path.with_extension("tmp");
    let result = (|| {
        fs::write(&tmp, bytes)?;
        if fsync {
            let file = fs::File::open(&tmp)?;
            file.sync_data()?;
        }
        fs::rename(&tmp, path)?;
        if fsync {
            let dir = fs::File::open(parent)?;
            dir.sync_all()?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

fn chunk_fingerprint(
    namespace: &str,
    part_name: &str,
    offset: u64,
    bytes: &[u8],
) -> BlockFingerprint {
    let mut h = blake3::Hasher::new();
    h.update(b"arle-kv-snapshot-chunk-v1\0");
    h.update(&(namespace.len() as u64).to_le_bytes());
    h.update(namespace.as_bytes());
    h.update(&(part_name.len() as u64).to_le_bytes());
    h.update(part_name.as_bytes());
    h.update(&offset.to_le_bytes());
    h.update(&(bytes.len() as u64).to_le_bytes());
    h.update(bytes);
    let hash = h.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    BlockFingerprint(out)
}

fn segment_fingerprint(
    manifest_id: BlockFingerprint,
    chunks: &[PendingSegmentChunk],
) -> BlockFingerprint {
    let mut h = blake3::Hasher::new();
    h.update(b"arle-kv-snapshot-segment-v1\0");
    h.update(&manifest_id.0);
    h.update(&(chunks.len() as u64).to_le_bytes());
    for chunk in chunks {
        h.update(&chunk.chunk_id.0);
        h.update(&(chunk.bytes.len() as u64).to_le_bytes());
    }
    let hash = h.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    BlockFingerprint(out)
}

fn validate_manifest_id(id: BlockFingerprint) -> io::Result<()> {
    if id == BlockFingerprint([0; 16]) {
        return Err(invalid_data(
            "chunked snapshot manifest id must not be zero",
        ));
    }
    Ok(())
}

fn fingerprint_hex(fingerprint: BlockFingerprint) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(32);
    for byte in fingerprint.0 {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn fingerprint_from_hex(hex: &str) -> Option<BlockFingerprint> {
    if hex.len() != 32 {
        return None;
    }
    let mut bytes = [0u8; 16];
    for (idx, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let hi = hex_value(pair[0])?;
        let lo = hex_value(pair[1])?;
        bytes[idx] = (hi << 4) | lo;
    }
    Some(BlockFingerprint(bytes))
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn repeated_chunk_bytes(chunks: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(chunks.len() * 4096);
        for &chunk in chunks {
            bytes.extend(std::iter::repeat_n(chunk, 4096));
        }
        bytes
    }

    #[test]
    fn chunked_snapshot_store_reuses_prefix_chunks() {
        let dir = tempdir().expect("tempdir");
        let store = ChunkedSnapshotStore::new(dir.path())
            .with_chunk_bytes(4096)
            .with_parallelism(2);
        let id_a = BlockFingerprint([1; 16]);
        let id_b = BlockFingerprint([2; 16]);
        let bytes_a = repeated_chunk_bytes(b"ab");
        let bytes_b = repeated_chunk_bytes(b"abc");

        let (_, stats_a) = store
            .put_snapshot(
                ChunkedSnapshotWrite {
                    manifest_id: id_a,
                    namespace: "test".into(),
                    metadata: b"meta-a".to_vec(),
                    parts: vec![ChunkedSnapshotPartWrite {
                        name: "kv.0".into(),
                        bytes: bytes_a,
                    }],
                },
                false,
            )
            .expect("write a");
        assert_eq!(stats_a.chunks_written, 2);
        assert_eq!(stats_a.chunks_reused, 0);

        let (_, stats_b) = store
            .put_snapshot(
                ChunkedSnapshotWrite {
                    manifest_id: id_b,
                    namespace: "test".into(),
                    metadata: b"meta-b".to_vec(),
                    parts: vec![ChunkedSnapshotPartWrite {
                        name: "kv.0".into(),
                        bytes: bytes_b.clone(),
                    }],
                },
                false,
            )
            .expect("write b");
        assert_eq!(stats_b.chunks_written, 1);
        assert_eq!(stats_b.chunks_reused, 2);

        let reloaded = store
            .get_snapshot(
                &ChunkedSnapshotLocation {
                    manifest_id: id_b,
                    path: store.manifest_path_for(id_b),
                    payload_len: bytes_b.len() as u64 + 6,
                },
                Some(id_b),
            )
            .expect("reload b");
        assert_eq!(reloaded.manifest.metadata, b"meta-b");
        assert_eq!(reloaded.parts[0].bytes, bytes_b);
    }

    #[test]
    fn chunked_snapshot_gc_keeps_referenced_chunks() {
        let dir = tempdir().expect("tempdir");
        let store = ChunkedSnapshotStore::new(dir.path()).with_chunk_bytes(4096);
        let id_a = BlockFingerprint([3; 16]);
        let id_b = BlockFingerprint([4; 16]);
        let bytes_a = repeated_chunk_bytes(b"ab");
        let bytes_b = repeated_chunk_bytes(b"abc");

        let (loc_a, _) = store
            .put_snapshot(
                ChunkedSnapshotWrite {
                    manifest_id: id_a,
                    namespace: "test".into(),
                    metadata: b"meta-a".to_vec(),
                    parts: vec![ChunkedSnapshotPartWrite {
                        name: "part".into(),
                        bytes: bytes_a,
                    }],
                },
                false,
            )
            .expect("write a");
        let (loc_b, _) = store
            .put_snapshot(
                ChunkedSnapshotWrite {
                    manifest_id: id_b,
                    namespace: "test".into(),
                    metadata: b"meta-b".to_vec(),
                    parts: vec![ChunkedSnapshotPartWrite {
                        name: "part".into(),
                        bytes: bytes_b,
                    }],
                },
                false,
            )
            .expect("write b");

        store.delete_snapshot(&loc_a).expect("delete a");
        let removed = store.collect_orphan_chunks().expect("gc chunks");
        assert_eq!(removed, 0, "shared chunks still referenced by b");
        store.get_snapshot(&loc_b, Some(id_b)).expect("b survives");
    }

    #[test]
    fn chunked_snapshot_store_rejects_corrupt_chunk_by_crc32c() {
        let dir = tempdir().expect("tempdir");
        let store = ChunkedSnapshotStore::new(dir.path()).with_chunk_bytes(4);
        let id = BlockFingerprint([5; 16]);
        let (location, _) = store
            .put_snapshot(
                ChunkedSnapshotWrite {
                    manifest_id: id,
                    namespace: "test".into(),
                    metadata: b"meta".to_vec(),
                    parts: vec![ChunkedSnapshotPartWrite {
                        name: "part".into(),
                        bytes: b"aaaabbbb".to_vec(),
                    }],
                },
                false,
            )
            .expect("write snapshot");

        let mut chunk_storage = None;
        store
            .visit_manifests(|_, manifest| {
                chunk_storage = manifest
                    .parts
                    .first()
                    .and_then(|part| part.chunks.first())
                    .and_then(|chunk| {
                        chunk
                            .segment_id
                            .map(|segment_id| (segment_id, chunk.segment_offset))
                    });
                Ok(())
            })
            .expect("visit manifest");
        let (segment_id, segment_offset) = chunk_storage.expect("segment chunk storage");
        let segment_path = store.segment_path_for(segment_id);
        let mut bytes = fs::read(&segment_path).expect("read segment");
        bytes[segment_offset as usize] ^= 0x01;
        fs::write(&segment_path, bytes).expect("corrupt segment");

        let err = store
            .get_snapshot(&location, Some(id))
            .expect_err("corrupt chunk must fail");
        assert!(
            err.to_string().contains("crc32c"),
            "unexpected corrupt chunk error: {err}"
        );
    }
}

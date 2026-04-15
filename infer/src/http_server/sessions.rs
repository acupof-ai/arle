use std::collections::HashMap;
use std::io;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::kv_tier::transport::DiskStore;
use crate::kv_tier::transport::disk::DiskBlockLocation;
use crate::prefix_cache::{BlockId, RadixCache, ReconcileReport};
use crate::types::BlockFingerprint;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub version: u16,
    pub session_id: String,
    pub kv_format_tag: u8,
    pub radix_bytes: Vec<u8>,
    pub persisted_blocks: Vec<PersistedBlockEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedBlockEntry {
    pub fingerprint_hex: String,
    pub location: SerializablePath,
    pub payload_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializablePath(pub String);

#[derive(Debug, Error)]
pub enum SessionSnapshotError {
    #[error("session snapshot I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("failed to serialize radix snapshot: {0}")]
    SerializeRadix(serde_json::Error),
    #[error("failed to deserialize radix snapshot: {0}")]
    DeserializeRadix(serde_json::Error),
    #[error("failed to serialize session manifest: {0}")]
    SerializeManifest(String),
    #[error("failed to deserialize session manifest: {0}")]
    DeserializeManifest(String),
    #[error("missing disk block {fingerprint_hex} at {path}")]
    MissingDiskBlock {
        fingerprint_hex: String,
        path: String,
    },
    #[error("disk block mismatch for {fingerprint_hex} at {path}: {reason}")]
    DiskBlockMismatch {
        fingerprint_hex: String,
        path: String,
        reason: String,
    },
}

pub struct LoadedSession {
    pub radix: RadixCache,
    pub kv_payloads: HashMap<BlockFingerprint, Vec<u8>>,
    pub report: ReconcileReport,
}

pub fn save_session<F>(
    session_id: &str,
    kv_format_tag: u8,
    radix: &RadixCache,
    disk: &DiskStore,
    mut payload_for: F,
    fingerprints: &[BlockFingerprint],
) -> Result<SessionSnapshot, SessionSnapshotError>
where
    F: FnMut(BlockFingerprint) -> Option<Vec<u8>>,
{
    let radix_bytes = serde_json::to_vec(radix).map_err(SessionSnapshotError::SerializeRadix)?;
    let mut persisted_blocks = Vec::new();

    for &fingerprint in fingerprints {
        let Some(payload) = payload_for(fingerprint) else {
            continue;
        };

        let location = disk.put_block(fingerprint, kv_format_tag, &payload)?;
        persisted_blocks.push(PersistedBlockEntry {
            fingerprint_hex: fingerprint_to_hex(fingerprint),
            location: SerializablePath(location.path.to_string_lossy().into_owned()),
            payload_len: location.payload_len,
        });
    }

    Ok(SessionSnapshot {
        version: 1,
        session_id: session_id.to_string(),
        kv_format_tag,
        radix_bytes,
        persisted_blocks,
    })
}

pub fn load_session<F>(
    snapshot: &SessionSnapshot,
    disk: &DiskStore,
    mut allocate_block_id: F,
) -> Result<LoadedSession, SessionSnapshotError>
where
    F: FnMut(BlockFingerprint) -> BlockId,
{
    if snapshot.version != 1 {
        return Err(SessionSnapshotError::DeserializeManifest(format!(
            "unsupported session snapshot version {}",
            snapshot.version
        )));
    }

    let mut radix: RadixCache = serde_json::from_slice(&snapshot.radix_bytes)
        .map_err(SessionSnapshotError::DeserializeRadix)?;
    let mut kv_payloads = HashMap::with_capacity(snapshot.persisted_blocks.len());
    let mut known = HashMap::with_capacity(snapshot.persisted_blocks.len());

    for entry in &snapshot.persisted_blocks {
        let fingerprint = parse_fingerprint_hex(&entry.fingerprint_hex)?;
        let location = DiskBlockLocation {
            path: PathBuf::from(&entry.location.0),
            payload_len: entry.payload_len,
            fingerprint,
        };
        let payload = match disk.get_block(&location, Some(fingerprint)) {
            Ok(payload) => payload,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                return Err(SessionSnapshotError::MissingDiskBlock {
                    fingerprint_hex: entry.fingerprint_hex.clone(),
                    path: entry.location.0.clone(),
                });
            }
            Err(err) if err.kind() == io::ErrorKind::InvalidData => {
                return Err(SessionSnapshotError::DiskBlockMismatch {
                    fingerprint_hex: entry.fingerprint_hex.clone(),
                    path: entry.location.0.clone(),
                    reason: err.to_string(),
                });
            }
            Err(err) => return Err(SessionSnapshotError::Io(err)),
        };

        let payload_len = payload.len() as u64;
        if payload_len != entry.payload_len {
            return Err(SessionSnapshotError::DiskBlockMismatch {
                fingerprint_hex: entry.fingerprint_hex.clone(),
                path: entry.location.0.clone(),
                reason: format!(
                    "payload length mismatch: manifest={} disk={payload_len}",
                    entry.payload_len
                ),
            });
        }

        kv_payloads.insert(fingerprint, payload);
        known.insert(fingerprint, allocate_block_id(fingerprint));
    }

    let report = radix.reconcile(&known);
    Ok(LoadedSession {
        radix,
        kv_payloads,
        report,
    })
}

fn fingerprint_to_hex(fingerprint: BlockFingerprint) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut out = String::with_capacity(32);
    for byte in fingerprint.0 {
        out.push(char::from(HEX[(byte >> 4) as usize]));
        out.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    out
}

fn parse_fingerprint_hex(hex: &str) -> Result<BlockFingerprint, SessionSnapshotError> {
    if hex.len() != 32 {
        return Err(SessionSnapshotError::DeserializeManifest(format!(
            "fingerprint hex must be 32 chars, got {}",
            hex.len()
        )));
    }

    let mut bytes = [0u8; 16];
    for (idx, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
        let hi = decode_hex_nibble(chunk[0]).ok_or_else(|| {
            SessionSnapshotError::DeserializeManifest(format!(
                "invalid fingerprint hex nibble '{}'",
                char::from(chunk[0])
            ))
        })?;
        let lo = decode_hex_nibble(chunk[1]).ok_or_else(|| {
            SessionSnapshotError::DeserializeManifest(format!(
                "invalid fingerprint hex nibble '{}'",
                char::from(chunk[1])
            ))
        })?;
        bytes[idx] = (hi << 4) | lo;
    }

    Ok(BlockFingerprint(bytes))
}

fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;

    use tempfile::tempdir;

    use super::{SessionSnapshotError, load_session, save_session};
    use crate::prefix_cache::{BlockId, RadixCache};
    use crate::types::BlockFingerprint;

    #[test]
    fn save_then_load_round_trips_radix_and_payloads() {
        let dir = tempdir().expect("tempdir");
        let disk = crate::kv_tier::transport::DiskStore::new(dir.path());
        let mut radix = RadixCache::new(4);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let fp_a = BlockFingerprint([0x11; 16]);
        let fp_b = BlockFingerprint([0x22; 16]);
        let original_blocks = [BlockId(10), BlockId(20)];
        radix.insert_with_fingerprints(&tokens, &original_blocks, &[fp_a, fp_b]);

        let expected_payloads =
            HashMap::from([(fp_a, b"payload-a".to_vec()), (fp_b, b"payload-b".to_vec())]);
        let snapshot = save_session(
            "session-1",
            7,
            &radix,
            &disk,
            |fingerprint| expected_payloads.get(&fingerprint).cloned(),
            &[fp_a, fp_b],
        )
        .expect("save session");

        let mut next_block_id = 100u32;
        let mut loaded = load_session(&snapshot, &disk, |_| {
            let block_id = BlockId(next_block_id);
            next_block_id += 1;
            block_id
        })
        .expect("load session");

        assert_eq!(loaded.report.remapped, 2);
        assert_eq!(loaded.report.tombstoned, 0);
        assert_eq!(loaded.kv_payloads, expected_payloads);

        let (matched_len, blocks) = loaded.radix.lookup(&tokens);
        assert_eq!(matched_len, 8);
        assert_eq!(blocks, vec![BlockId(100), BlockId(101)]);
        assert_ne!(blocks, original_blocks);
    }

    #[test]
    fn save_skips_blocks_with_no_payload() {
        let dir = tempdir().expect("tempdir");
        let disk = crate::kv_tier::transport::DiskStore::new(dir.path());
        let mut radix = RadixCache::new(4);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let fp_a = BlockFingerprint([0x33; 16]);
        let fp_b = BlockFingerprint([0x44; 16]);
        radix.insert_with_fingerprints(&tokens, &[BlockId(1), BlockId(2)], &[fp_a, fp_b]);

        let snapshot = save_session(
            "session-2",
            9,
            &radix,
            &disk,
            |fingerprint| (fingerprint == fp_a).then(|| b"payload-a".to_vec()),
            &[fp_a, fp_b],
        )
        .expect("save session");

        assert_eq!(snapshot.persisted_blocks.len(), 1);
        assert_eq!(snapshot.persisted_blocks[0].fingerprint_hex.len(), 32);

        let mut loaded = load_session(&snapshot, &disk, |_| BlockId(77)).expect("load session");
        assert_eq!(loaded.report.remapped, 1);
        assert_eq!(loaded.report.tombstoned, 1);
        assert_eq!(loaded.kv_payloads.len(), 1);
        assert_eq!(loaded.kv_payloads.get(&fp_a), Some(&b"payload-a".to_vec()));
        assert!(loaded.kv_payloads.get(&fp_b).is_none());

        let (matched_len, blocks) = loaded.radix.lookup(&tokens);
        assert_eq!(matched_len, 4);
        assert_eq!(blocks, vec![BlockId(77)]);
    }

    #[test]
    fn load_errors_on_tampered_disk_payload() {
        let dir = tempdir().expect("tempdir");
        let disk = crate::kv_tier::transport::DiskStore::new(dir.path());
        let mut radix = RadixCache::new(4);
        let tokens = [1, 2, 3, 4];
        let fingerprint = BlockFingerprint([0x55; 16]);
        radix.insert_with_fingerprints(&tokens, &[BlockId(9)], &[fingerprint]);

        let snapshot = save_session(
            "session-3",
            3,
            &radix,
            &disk,
            |fp| (fp == fingerprint).then(|| b"payload".to_vec()),
            &[fingerprint],
        )
        .expect("save session");

        let disk_path = PathBuf::from(&snapshot.persisted_blocks[0].location.0);
        fs::write(&disk_path, b"wrong-bytes").expect("tamper disk payload");

        match load_session(&snapshot, &disk, |_| BlockId(99)) {
            Err(SessionSnapshotError::DiskBlockMismatch { .. }) => {}
            Err(other) => panic!("expected DiskBlockMismatch, got {other:?}"),
            Ok(_) => panic!("tampered payload should fail"),
        }
    }
}

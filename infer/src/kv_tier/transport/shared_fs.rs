//! Minimal cluster-shared backend backed by a shared filesystem root.
//!
//! This keeps the remote/backend contract small: bytes are still persisted via
//! the same content-addressed block format as `DiskStore`, but handles surface
//! as `BlockLocation::Remote` so the control plane can treat the backend as a
//! cluster-shared tier rather than a node-local disk path.

use std::io;
use std::path::{Path, PathBuf};
use std::task::Poll;

use serde::{Deserialize, Serialize};

use super::TransportError;
use super::disk::{DiskBlockLocation, DiskStore};
use crate::kv_tier::{
    KVHandle,
    backend::{KVBackend, KVBackendScope},
    io::{
        KVBackendCompletion, KVBackendDelete, KVBackendFetch, KVBackendStore, KVPayload,
        KVPayloadRef,
    },
    tier::{BlockLocation, RemoteBlockDesc, Tier, TransportId},
};
use crate::types::BlockFingerprint;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Eq, PartialEq)]
struct SharedFsDescriptor {
    fingerprint: BlockFingerprint,
    payload_len: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SharedFsBlockLocation {
    pub fingerprint: BlockFingerprint,
    pub payload_len: u64,
}

impl SharedFsBlockLocation {
    pub fn new(fingerprint: BlockFingerprint, payload_len: u64) -> Self {
        Self {
            fingerprint,
            payload_len,
        }
    }

    pub fn into_remote_desc(self) -> Result<RemoteBlockDesc, TransportError> {
        let payload = postcard::to_allocvec(&SharedFsDescriptor {
            fingerprint: self.fingerprint,
            payload_len: self.payload_len,
        })
        .map_err(|err| {
            TransportError::Other(format!("shared-fs descriptor encode failed: {err}"))
        })?;
        Ok(RemoteBlockDesc {
            transport: TransportId::SharedFilesystem,
            payload,
        })
    }

    pub fn into_block_location(self) -> Result<BlockLocation, TransportError> {
        Ok(BlockLocation::Remote {
            desc: self.into_remote_desc()?,
        })
    }

    pub fn from_remote_desc(desc: &RemoteBlockDesc) -> Result<Self, TransportError> {
        if desc.transport != TransportId::SharedFilesystem {
            return Err(TransportError::Other(format!(
                "shared-fs descriptor requires {:?}, got {:?}",
                TransportId::SharedFilesystem,
                desc.transport
            )));
        }
        let descriptor =
            postcard::from_bytes::<SharedFsDescriptor>(&desc.payload).map_err(|err| {
                TransportError::Other(format!("shared-fs descriptor decode failed: {err}"))
            })?;
        Ok(Self::new(descriptor.fingerprint, descriptor.payload_len))
    }

    pub fn from_block_location(location: &BlockLocation) -> Result<Self, TransportError> {
        match location {
            BlockLocation::Remote { desc } => Self::from_remote_desc(desc),
            _ => Err(TransportError::Other(
                "shared-fs backend requires a remote block location".into(),
            )),
        }
    }
}

#[derive(Debug)]
enum SharedFsBackendOpState {
    Ready(Result<KVBackendCompletion, TransportError>),
    Exhausted,
}

#[derive(Debug)]
pub struct SharedFsBackendOp {
    state: SharedFsBackendOpState,
}

#[derive(Debug)]
pub struct SharedFsStore {
    inner: DiskStore,
}

impl SharedFsStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            inner: DiskStore::new(root),
        }
    }

    pub fn root(&self) -> &Path {
        self.inner.root()
    }

    pub fn create_root(&self) -> io::Result<()> {
        self.inner.create_root()
    }

    pub fn block_path_for(&self, fingerprint: BlockFingerprint) -> PathBuf {
        self.inner.block_path_for(fingerprint)
    }

    pub fn contains_block(&self, fingerprint: BlockFingerprint) -> io::Result<bool> {
        self.inner.contains_block(fingerprint)
    }

    pub fn put_block(
        &self,
        fingerprint: BlockFingerprint,
        kv_format_tag: u8,
        payload: &[u8],
    ) -> io::Result<SharedFsBlockLocation> {
        let location = self.inner.put_block(fingerprint, kv_format_tag, payload)?;
        Ok(SharedFsBlockLocation::new(
            location.fingerprint,
            location.payload_len,
        ))
    }

    pub fn get_block(
        &self,
        location: SharedFsBlockLocation,
        expected_fingerprint: Option<BlockFingerprint>,
    ) -> io::Result<Vec<u8>> {
        let disk_location = DiskBlockLocation {
            path: self.block_path_for(location.fingerprint),
            payload_len: location.payload_len,
            fingerprint: location.fingerprint,
        };
        self.inner.get_block(&disk_location, expected_fingerprint)
    }

    pub fn delete_block(&self, location: SharedFsBlockLocation) -> io::Result<()> {
        let disk_location = DiskBlockLocation {
            path: self.block_path_for(location.fingerprint),
            payload_len: location.payload_len,
            fingerprint: location.fingerprint,
        };
        self.inner.delete_block(&disk_location)
    }

    fn ready_op(result: Result<KVBackendCompletion, TransportError>) -> SharedFsBackendOp {
        SharedFsBackendOp {
            state: SharedFsBackendOpState::Ready(result),
        }
    }

    fn location_from_handle(handle: &KVHandle) -> Result<SharedFsBlockLocation, TransportError> {
        SharedFsBlockLocation::from_block_location(&handle.location)
    }
}

impl KVBackend for SharedFsStore {
    type Op = SharedFsBackendOp;

    fn backend_id(&self) -> &'static str {
        "shared-fs"
    }

    fn scope(&self) -> KVBackendScope {
        KVBackendScope::ClusterShared
    }

    fn tier(&self) -> Tier {
        Tier::Remote
    }

    fn store(&self, req: KVBackendStore) -> Result<Self::Op, TransportError> {
        let fingerprint = req.block.fingerprint.ok_or_else(|| {
            TransportError::Other("shared-fs backend store requires a block fingerprint".into())
        })?;
        let payload_len = u64::try_from(req.payload.len()).map_err(|_| {
            TransportError::Other("shared-fs backend store payload length exceeds u64".into())
        })?;
        let location = self
            .put_block(fingerprint, req.kv_format_tag, req.payload.as_slice())
            .map_err(|err| TransportError::Transfer(err.to_string()))?;
        let handle = req.handle.with_location(location.into_block_location()?);
        debug_assert_eq!(handle.byte_len, payload_len);
        Ok(Self::ready_op(Ok(KVBackendCompletion::Stored(handle))))
    }

    fn fetch(&self, req: KVBackendFetch) -> Result<Self::Op, TransportError> {
        let location = Self::location_from_handle(&req.handle)?;
        let bytes = self
            .get_block(location, Some(location.fingerprint))
            .map_err(|err| TransportError::Transfer(err.to_string()))?;
        let payload_len = u64::try_from(bytes.len())
            .map_err(|_| TransportError::Other("shared-fs payload length exceeds u64".into()))?;
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
        let location = Self::location_from_handle(&req.handle)?;
        self.delete_block(location)
            .map_err(|err| TransportError::Transfer(err.to_string()))?;
        Ok(Self::ready_op(Ok(KVBackendCompletion::Deleted(req.handle))))
    }

    fn exists(&self, handle: &KVHandle) -> Result<bool, TransportError> {
        let location = Self::location_from_handle(handle)?;
        self.contains_block(location.fingerprint)
            .map_err(|err| TransportError::Transfer(err.to_string()))
    }

    fn poll(&self, op: &mut Self::Op) -> Poll<Result<KVBackendCompletion, TransportError>> {
        match std::mem::replace(&mut op.state, SharedFsBackendOpState::Exhausted) {
            SharedFsBackendOpState::Ready(result) => Poll::Ready(result),
            SharedFsBackendOpState::Exhausted => Poll::Ready(Err(TransportError::Other(
                "shared-fs backend op already completed".into(),
            ))),
        }
    }

    fn abort(&self, op: &mut Self::Op) {
        op.state = SharedFsBackendOpState::Ready(Err(TransportError::Aborted));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_tier::{KVBlock, KVSpanId, LayerRange, TokenRange};
    use tempfile::tempdir;

    #[test]
    fn descriptor_roundtrip_uses_remote_location() {
        let location = SharedFsBlockLocation::new(BlockFingerprint([0x11; 16]), 4096);
        let remote = location.into_block_location().expect("remote location");
        let decoded = SharedFsBlockLocation::from_block_location(&remote).expect("decode location");

        match remote {
            BlockLocation::Remote { desc } => {
                assert_eq!(desc.transport, TransportId::SharedFilesystem);
            }
            other => panic!("expected remote location, got {other:?}"),
        }
        assert_eq!(decoded, location);
    }

    #[test]
    fn descriptor_rejects_other_transport_payloads() {
        let desc = RemoteBlockDesc {
            transport: TransportId::Nixl,
            payload: vec![1, 2, 3],
        };
        let err = SharedFsBlockLocation::from_remote_desc(&desc).expect_err("transport mismatch");
        assert!(matches!(err, TransportError::Other(_)));
    }

    #[test]
    fn shared_fs_backend_store_fetch_delete_roundtrip() {
        let dir = tempdir().unwrap();
        let store = SharedFsStore::new(dir.path());
        let fingerprint = BlockFingerprint([0x7C; 16]);
        let payload = b"shared-fs-remote".to_vec();
        let block = KVBlock::new(
            crate::types::BlockId(17),
            LayerRange::new(0, 2),
            TokenRange::new(0, 16),
            payload.len() as u64,
        )
        .with_fingerprint(fingerprint);
        let handle = KVHandle::new(
            Some(KVSpanId(3)),
            block.block_id,
            BlockLocation::HostPinned { offset: 0 },
            1,
            payload.len() as u64,
        );

        let mut store_op = store
            .store(KVBackendStore {
                handle: handle.clone(),
                block: block.clone(),
                kv_format_tag: 5,
                payload: KVPayload::from_vec(payload.clone()),
            })
            .unwrap();
        let stored = match store.poll(&mut store_op) {
            Poll::Ready(Ok(KVBackendCompletion::Stored(handle))) => handle,
            other => panic!("expected stored completion, got {other:?}"),
        };
        match &stored.location {
            BlockLocation::Remote { desc } => {
                assert_eq!(desc.transport, TransportId::SharedFilesystem);
            }
            other => panic!("expected remote handle, got {other:?}"),
        }

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

        let mut delete_op = store
            .delete(KVBackendDelete {
                handle: stored.clone(),
            })
            .unwrap();
        match store.poll(&mut delete_op) {
            Poll::Ready(Ok(KVBackendCompletion::Deleted(handle))) => {
                assert_eq!(handle, stored);
            }
            other => panic!("expected deleted completion, got {other:?}"),
        }

        let remote_location =
            SharedFsBlockLocation::from_block_location(&stored.location).expect("decode location");
        let err = store
            .get_block(remote_location, Some(fingerprint))
            .expect_err("deleted payload should be gone");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn shared_fs_backend_exists_tracks_remote_presence() {
        let dir = tempdir().unwrap();
        let store = SharedFsStore::new(dir.path());
        let fingerprint = BlockFingerprint([0x52; 16]);
        let handle = KVHandle::new(
            Some(KVSpanId(5)),
            crate::types::BlockId(13),
            SharedFsBlockLocation::new(fingerprint, 6)
                .into_block_location()
                .expect("remote location"),
            1,
            6,
        );

        assert!(!store.exists(&handle).expect("missing before write"));
        store
            .put_block(fingerprint, 1, b"foobar")
            .expect("put block");
        assert!(store.exists(&handle).expect("present after write"));
    }

    #[test]
    fn shared_fs_backend_abort_surfaces_aborted() {
        let dir = tempdir().unwrap();
        let store = SharedFsStore::new(dir.path());
        let mut op = SharedFsStore::ready_op(Ok(KVBackendCompletion::Deleted(KVHandle::new(
            Some(KVSpanId(0)),
            crate::types::BlockId(0),
            BlockLocation::HostPinned { offset: 0 },
            0,
            0,
        ))));

        store.abort(&mut op);
        assert!(matches!(
            store.poll(&mut op),
            Poll::Ready(Err(TransportError::Aborted))
        ));
    }
}

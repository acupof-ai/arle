//! Object-backend surface for HiCache-aligned slower tiers.
//!
//! `KVTransport` remains the local byte-copy engine contract. `KVBackend`
//! is the higher-level object-store contract used for T2/T3 style storage.

use std::path::PathBuf;
use std::sync::Arc;
use std::task::Poll;

#[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
use super::transport::nixl::{NixlOp, NixlTransport};
use super::transport::shared_fs::SharedFsBackendOp;
use super::{
    chunk::KVHandle,
    io::{KVBackendCompletion, KVBackendDelete, KVBackendFetch, KVBackendStore},
    tier::{BlockLocation, Tier},
    transport::{
        TransportError,
        shared_fs::{SharedFsBlockLocation, SharedFsStore},
    },
};
use crate::types::BlockFingerprint;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum KVBackendScope {
    NodeLocal,
    ClusterShared,
}

/// Control-plane contract for slower KV tiers.
pub trait KVBackend: Send + Sync {
    type Op: Send;

    fn backend_id(&self) -> &'static str;
    fn scope(&self) -> KVBackendScope;
    fn tier(&self) -> Tier;

    fn store(&self, req: KVBackendStore) -> Result<Self::Op, TransportError>;
    fn fetch(&self, req: KVBackendFetch) -> Result<Self::Op, TransportError>;
    fn delete(&self, req: KVBackendDelete) -> Result<Self::Op, TransportError>;
    fn exists(&self, handle: &KVHandle) -> Result<bool, TransportError>;

    fn poll(&self, op: &mut Self::Op) -> Poll<Result<KVBackendCompletion, TransportError>>;
    fn abort(&self, op: &mut Self::Op);

    fn is_cluster_shared(&self) -> bool {
        matches!(self.scope(), KVBackendScope::ClusterShared)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClusterSharedBackendConfig {
    SharedFilesystem {
        root: PathBuf,
    },
    #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
    Nixl {
        agent_name: String,
    },
}

impl ClusterSharedBackendConfig {
    pub fn build(&self) -> ClusterSharedBackend {
        match self {
            Self::SharedFilesystem { root } => {
                ClusterSharedBackend::SharedFs(Arc::new(SharedFsStore::new(root.clone())))
            }
            #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
            Self::Nixl { agent_name } => {
                ClusterSharedBackend::Nixl(Arc::new(NixlTransport::new(agent_name.clone())))
            }
        }
    }
}

#[derive(Debug)]
pub enum ClusterSharedBackendOp {
    SharedFs(SharedFsBackendOp),
    #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
    Nixl(NixlOp),
}

#[derive(Clone, Debug)]
pub enum ClusterSharedBackend {
    SharedFs(Arc<SharedFsStore>),
    #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
    Nixl(Arc<NixlTransport>),
}

impl ClusterSharedBackend {
    pub fn backend_id(&self) -> &'static str {
        match self {
            Self::SharedFs(store) => store.backend_id(),
            #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
            Self::Nixl(store) => store.backend_id(),
        }
    }

    pub fn tier(&self) -> Tier {
        match self {
            Self::SharedFs(store) => store.tier(),
            #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
            Self::Nixl(store) => store.tier(),
        }
    }

    pub fn exists(&self, handle: &KVHandle) -> Result<bool, TransportError> {
        match self {
            Self::SharedFs(store) => store.exists(handle),
            #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
            Self::Nixl(store) => store.exists(handle),
        }
    }

    pub fn remote_location_for(
        &self,
        fingerprint: BlockFingerprint,
        payload_len: u64,
    ) -> Result<BlockLocation, TransportError> {
        match self {
            Self::SharedFs(_) => {
                SharedFsBlockLocation::new(fingerprint, payload_len).into_block_location()
            }
            #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
            Self::Nixl(_) => Err(TransportError::Other(
                "nixl remote descriptor synthesis requires real backend metadata".into(),
            )),
        }
    }

    pub fn store(&self, req: KVBackendStore) -> Result<ClusterSharedBackendOp, TransportError> {
        match self {
            Self::SharedFs(store) => store.store(req).map(ClusterSharedBackendOp::SharedFs),
            #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
            Self::Nixl(store) => store.store(req).map(ClusterSharedBackendOp::Nixl),
        }
    }

    pub fn fetch(&self, req: KVBackendFetch) -> Result<ClusterSharedBackendOp, TransportError> {
        match self {
            Self::SharedFs(store) => store.fetch(req).map(ClusterSharedBackendOp::SharedFs),
            #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
            Self::Nixl(store) => store.fetch(req).map(ClusterSharedBackendOp::Nixl),
        }
    }

    pub fn delete(&self, req: KVBackendDelete) -> Result<ClusterSharedBackendOp, TransportError> {
        match self {
            Self::SharedFs(store) => store.delete(req).map(ClusterSharedBackendOp::SharedFs),
            #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
            Self::Nixl(store) => store.delete(req).map(ClusterSharedBackendOp::Nixl),
        }
    }

    pub fn poll(
        &self,
        op: &mut ClusterSharedBackendOp,
    ) -> Poll<Result<KVBackendCompletion, TransportError>> {
        match (self, op) {
            (Self::SharedFs(store), ClusterSharedBackendOp::SharedFs(op)) => store.poll(op),
            #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
            (Self::Nixl(store), ClusterSharedBackendOp::Nixl(op)) => store.poll(op),
            #[allow(unreachable_patterns)]
            _ => Poll::Ready(Err(TransportError::Other(
                "cluster-shared backend/op mismatch".into(),
            ))),
        }
    }

    pub fn abort(&self, op: &mut ClusterSharedBackendOp) {
        match (self, op) {
            (Self::SharedFs(store), ClusterSharedBackendOp::SharedFs(op)) => store.abort(op),
            #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
            (Self::Nixl(store), ClusterSharedBackendOp::Nixl(op)) => store.abort(op),
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_tier::TransportId;
    use tempfile::tempdir;

    #[test]
    fn shared_fs_config_builds_cluster_backend() {
        let dir = tempdir().expect("tempdir");
        let backend = ClusterSharedBackendConfig::SharedFilesystem {
            root: dir.path().to_path_buf(),
        }
        .build();

        assert_eq!(backend.backend_id(), "shared-fs");
        let location = backend
            .remote_location_for(crate::types::BlockFingerprint([0x44; 16]), 64)
            .expect("remote location");
        match location {
            BlockLocation::Remote { desc } => {
                assert_eq!(desc.transport, TransportId::SharedFilesystem);
            }
            other => panic!("expected remote location, got {other:?}"),
        }
    }

    #[cfg(any(feature = "rdma-nixl", feature = "rdma-nixl-real"))]
    #[test]
    fn nixl_config_builds_cluster_backend() {
        let backend = ClusterSharedBackendConfig::Nixl {
            agent_name: "agent-0".into(),
        }
        .build();

        assert_eq!(backend.backend_id(), "nixl");
        let err = backend
            .remote_location_for(crate::types::BlockFingerprint([0x55; 16]), 32)
            .expect_err("nixl descriptor synthesis should stay gated");
        assert!(matches!(err, TransportError::Other(_)));
    }
}

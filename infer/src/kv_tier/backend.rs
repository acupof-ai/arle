//! Object-backend surface for HiCache-aligned slower tiers.
//!
//! `KVTransport` remains the local byte-copy engine contract. `KVBackend`
//! is the higher-level object-store contract used for T2/T3 style storage.

use std::task::Poll;

use super::{
    io::{KVBackendCompletion, KVBackendDelete, KVBackendFetch, KVBackendStore},
    tier::Tier,
    transport::TransportError,
};

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

    fn poll(&self, op: &mut Self::Op) -> Poll<Result<KVBackendCompletion, TransportError>>;
    fn abort(&self, op: &mut Self::Op);

    fn is_cluster_shared(&self) -> bool {
        matches!(self.scope(), KVBackendScope::ClusterShared)
    }
}

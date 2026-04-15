//! [`KVTransport`] trait — backend-agnostic KV transfer surface.
//!
//! Shape frozen per `docs/plans/tiered-kv-cache-tasks.md §6.3`: the trait
//! exposes `type Op` plus explicit `poll` and `abort` methods, NOT
//! `type Completion: Future`. NIXL, Mooncake, and UCX all expose polling
//! completion; keeping the trait Future-free lets each backend hide its
//! own completion model.
//!
//! See `crate::kv_tier` for the module-level design notes.
//!
//! # Backend submodules
//!
//! - [`disk`] — [`DiskStore`], the T3 NVMe / SSD backend. Pure `std::fs`;
//!   cross-platform (macOS tests run on `tokio::fs`-free paths). The
//!   Phase-3 follow-up PR will wrap it in a real `KVTransport` impl.
//! - [`nixl`] — [`nixl::NixlTransport`] stub for the T4 remote tier.
//!   Only compiled under `#[cfg(feature = "rdma-nixl")]`; references
//!   `nixl-sys` types so a `cargo check --features rdma-nixl` on macOS
//!   proves the trait shape is forward-compatible with real NIXL.

pub mod disk;
pub mod local_cuda;
#[cfg(feature = "rdma-nixl")]
pub mod nixl;

pub use disk::DiskStore;
pub use local_cuda::LocalCudaTransport;
#[cfg(feature = "rdma-nixl")]
pub use nixl::NixlTransport;

use std::task::Poll;

use super::tier::{BlockLocation, MemKind};

/// One batched transfer instruction handed to the transport. The
/// coordinator builds these and submits them via
/// [`KVTransport::put_batch`] or [`KVTransport::get_batch`].
#[derive(Clone, Debug)]
pub struct TransferOp {
    pub src: BlockLocation,
    pub dst: BlockLocation,
    pub len: u32,
}

/// Transport-layer errors. Intentionally coarse — each impl can decorate
/// the inner string with its own diagnostic; cross-backend code only
/// needs to distinguish the four kinds below.
#[derive(Debug)]
pub enum TransportError {
    /// MR registration failed (out of memory, invalid pointer, hardware
    /// bounds). Typically unrecoverable for this region.
    Registration(String),
    /// A submitted transfer completed with an error (remote failure,
    /// checksum mismatch, local copy engine fault).
    Transfer(String),
    /// An in-flight operation was cancelled via
    /// [`KVTransport::abort`] and then polled to completion.
    Aborted,
    /// Catch-all for transport-specific errors that don't fit the
    /// above. Keep the string short.
    Other(String),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::Registration(msg) => write!(f, "registration failed: {msg}"),
            TransportError::Transfer(msg) => write!(f, "transfer failed: {msg}"),
            TransportError::Aborted => write!(f, "transfer aborted"),
            TransportError::Other(msg) => write!(f, "transport error: {msg}"),
        }
    }
}

impl std::error::Error for TransportError {}

/// Backend-agnostic async KV transfer trait.
///
/// Milestone gates (tiered-kv-cache project, 2026-04-15 revision):
/// - **M3** — `LocalCudaTransport` (cudaMemcpyAsync on a dedicated copy
///   stream), not yet implemented
/// - **M4** — `DiskStore` (tokio::fs default, io_uring behind a feature
///   flag); a skeleton store already lives at [`disk::DiskStore`] but
///   is not yet wired into a coordinator or this trait
/// - **M5** — `NixlTransport` stub via `nixl-sys` with `stub-api`
///   feature; the real impl behind `rdma-nixl-real` is trigger-gated
/// - **Post-M5, trigger-gated** — Mooncake `TransferEngine` binding,
///   reachable either as a direct impl or through NIXL's Mooncake plugin
///
/// **Shape locked** per the 2026-04-13 research notes:
/// `type Op: Send` (NOT `type Completion: Future`) because NIXL has no
/// native `Future` — all four stacks expose polling completion. Keeping
/// the trait Future-free lets each backend hide its own completion model;
/// an adapter `TransportFuture<T>` lives in `infer-engine`, not here.
///
/// **Cancel-safety**: dropping an [`KVTransport::Op`] handle before
/// [`KVTransport::poll`] returns `Ready` is unsound — the underlying
/// hardware may still DMA into the registered buffer. Callers must first
/// call [`KVTransport::abort`] and then poll until `Ready` before
/// dropping the handle or freeing the buffer.
pub trait KVTransport: Send + Sync {
    /// Drop-guarded memory-region handle. Registration is expensive
    /// (page-table pinning + HCA key caching), so callers hold these
    /// across many transfers.
    type Region: Send + Sync;

    /// Per-operation handle. Callers poll it via [`KVTransport::poll`].
    type Op: Send;

    /// Register a byte range as an MR.
    ///
    /// # Safety
    /// `ptr` must remain valid and unmapped for the lifetime of the
    /// returned `Region`. The transport may install the pointer in
    /// hardware page tables; reallocating or freeing the backing pool
    /// while a `Region` is outstanding will cause use-after-free in the
    /// NIC or copy engine. See the Tiered KV Cache invariant 5 in the
    /// module-level docs.
    unsafe fn register(
        &self,
        ptr: *mut u8,
        len: usize,
        kind: MemKind,
    ) -> Result<Self::Region, TransportError>;

    /// Drop a region. Default no-op to match backends where registration
    /// is free.
    fn invalidate_region(&self, _region: &Self::Region) -> Result<(), TransportError> {
        Ok(())
    }

    /// Submit a batch of write operations. Returns an opaque handle that
    /// callers poll via [`KVTransport::poll`].
    fn put_batch(&self, ops: &[TransferOp]) -> Result<Self::Op, TransportError>;

    /// Submit a batch of read operations. Same semantics as `put_batch`.
    fn get_batch(&self, ops: &[TransferOp]) -> Result<Self::Op, TransportError>;

    /// Non-blocking poll. Returns `Pending` while the batch is in
    /// flight, `Ready(Ok(()))` on success, or `Ready(Err(_))` on
    /// failure. After `Ready(_)`, the op handle is exhausted; do not
    /// poll it again.
    fn poll(&self, op: &mut Self::Op) -> Poll<Result<(), TransportError>>;

    /// Best-effort cancel. The handle must still be polled to
    /// completion before the caller drops it — see the cancel-safety
    /// note on the trait. Some backends (RDMA) cannot actually stop an
    /// in-flight operation; they record the cancellation and return
    /// [`TransportError::Aborted`] the next time the op is polled.
    fn abort(&self, op: &mut Self::Op);
}

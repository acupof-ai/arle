//! [`NixlTransport`] — M5 stub impl of [`KVTransport`] over NVIDIA NIXL.
//!
//! Compiled only under `#[cfg(feature = "rdma-nixl")]`. The `rdma-nixl`
//! feature transitively enables `nixl-sys`'s `stub-api` feature, so
//! `cargo check --features rdma-nixl` succeeds on macOS without
//! `libnixl.so` installed — the stub compiles a small C++ wrapper and
//! defers symbol resolution to `dlopen` at runtime. On production CUDA
//! hosts with the real library, enable the sibling `rdma-nixl-real`
//! feature instead (it pulls the same dep without `stub-api`).
//!
//! # Status — M5 stub only
//!
//! This file ships the **stub half** of the tiered-kv-cache M5
//! milestone (formerly P5 in the 2026-04-13 plan; renamed in the
//! 2026-04-15 revision — see `docs/projects/tiered-kv-cache.md` §6 M5).
//! The "real RDMA" half of M5 is **trigger-gated** per project doc §6:
//! it executes only when one of these fires:
//!
//! - prefill / decode disaggregation requires KV migration between
//!   worker pools
//! - cluster-wide session roaming asks KV to follow a session across
//!   nodes
//! - a second consumer of the kernel layer needs functional remote
//!   tier I/O to integrate
//!
//! In the absence of any trigger, the real impl stays unwritten and
//! the stub stays as a compile gate.
//!
//! # What this skeleton proves
//!
//! - `nixl-sys` compiles into the infer workspace (types are in scope,
//!   the build.rs does not hard-fail when the native library is
//!   absent).
//! - Our [`KVTransport`] trait shape survives contact with NIXL's
//!   polling-completion model. The 2026-04-13 research specifically
//!   locked in `type Op` + explicit `poll` / `abort` because NIXL has
//!   no native `Future`; this file is where that shape gets validated.
//! - The feature-gate and cargo feature glue work end-to-end: default
//!   builds do not pay the `nixl-sys` build cost, `rdma-nixl` adds it,
//!   and the always-on `kv_tier` module still compiles on every
//!   backend.
//!
//! # What it does NOT do
//!
//! - No runtime behavior. Every [`KVTransport`] method returns
//!   [`TransportError::Other`] with an "M5 stub" diagnostic. The real
//!   M5 impl (deferred per the trigger discipline above) will actually
//!   call [`nixl_sys::Agent::register_memory`] and friends.
//! - No actual transfer. NIXL's `Agent::new` can succeed on Mac with
//!   `stub-api` (it just records the name), but `create_backend` /
//!   `register_memory` / `post_xfer_req` will `dlopen` at runtime and
//!   fail because the real library is absent — so this skeleton
//!   specifically avoids calling them.
//!
//! See `docs/projects/tiered-kv-cache.md` §6 M5 for the trigger
//! discipline and the migration shape from stub to real impl.

use std::task::Poll;

use super::super::tier::MemKind;
use super::{KVTransport, TransferOp, TransportError};

/// M5-stub transport. Holds no state; the real implementation will own
/// a `nixl_sys::Agent` and a set of registered regions when one of the
/// project doc §6 M5 triggers fires.
#[derive(Debug, Default)]
pub struct NixlTransport {
    /// Logical agent name that a future real impl will pass to
    /// [`nixl_sys::Agent::new`]. Kept as an owned String so callers can
    /// configure their node identity today and not have to revisit
    /// call sites when the real impl lands.
    name: String,
}

/// Opaque registered-region handle. The real impl will replace the
/// `()` payload with a `nixl_sys::RegistrationHandle`; keeping the
/// wrapper type lets call sites avoid importing `nixl-sys` directly.
#[derive(Debug)]
pub struct NixlRegion {
    _stub: (),
}

// Safety: the stub holds no memory; the `Send + Sync` bounds on
// `KVTransport::Region` are trivially satisfied.
unsafe impl Send for NixlRegion {}
unsafe impl Sync for NixlRegion {}

/// Opaque in-flight operation handle. The real impl will replace
/// `()` with a `nixl_sys::XferRequest`.
#[derive(Debug)]
pub struct NixlOp {
    _stub: (),
}

// Safety: `Send` bound on `KVTransport::Op` is trivially satisfied for
// the stub; the real impl will rely on NIXL's own thread-safety
// guarantees.
unsafe impl Send for NixlOp {}

impl NixlTransport {
    /// Construct a stub transport. Does not touch `nixl_sys` yet —
    /// constructing a real [`nixl_sys::Agent`] is deferred to the real
    /// M5 impl so the skeleton compiles uniformly on every platform.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// The agent name this transport was configured with. The real
    /// impl will pass it to `nixl_sys::Agent::new`.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Compile-time proof that `nixl-sys` is a reachable dependency under
/// the `rdma-nixl` feature. This function is never called; its body
/// only exists so the compiler instantiates the type and verifies the
/// feature glue is wired correctly. When the real M5 impl arrives and
/// wants to call `Agent::new`, it can delete this fn and use the type
/// directly.
#[allow(dead_code)]
fn _assert_nixl_sys_types_resolve() {
    // Referencing the type is enough — the skeleton never actually
    // constructs an Agent. On macOS with the stub-api feature this
    // compiles fine; on CUDA + real NIXL it also compiles fine. At
    // runtime neither path calls into NIXL, so there is no dlopen
    // failure to worry about.
    type _AgentAlias = nixl_sys::Agent;
    type _XferStatus = nixl_sys::XferStatus;
    type _XferOp = nixl_sys::XferOp;
    type _MemType = nixl_sys::MemType;
}

impl KVTransport for NixlTransport {
    type Region = NixlRegion;
    type Op = NixlOp;

    unsafe fn register(
        &self,
        _ptr: *mut u8,
        _len: usize,
        _kind: MemKind,
    ) -> Result<Self::Region, TransportError> {
        Err(TransportError::Other(
            "NixlTransport::register is an M5 stub; real impl calls nixl_sys::Agent::register_memory".into(),
        ))
    }

    fn put_batch(&self, _ops: &[TransferOp]) -> Result<Self::Op, TransportError> {
        Err(TransportError::Other(
            "NixlTransport::put_batch is an M5 stub; real impl calls Agent::post_xfer_req(XferOp::Write)".into(),
        ))
    }

    fn get_batch(&self, _ops: &[TransferOp]) -> Result<Self::Op, TransportError> {
        Err(TransportError::Other(
            "NixlTransport::get_batch is an M5 stub; real impl calls Agent::post_xfer_req(XferOp::Read)".into(),
        ))
    }

    fn poll(&self, _op: &mut Self::Op) -> Poll<Result<(), TransportError>> {
        Poll::Ready(Err(TransportError::Other(
            "NixlTransport::poll is an M5 stub; real impl calls Agent::get_xfer_status".into(),
        )))
    }

    fn abort(&self, _op: &mut Self::Op) {
        // No-op stub. Real impl records the cancellation; the next poll
        // will surface TransportError::Aborted.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_transport_reports_its_name() {
        let t = NixlTransport::new("infer-agent-0");
        assert_eq!(t.name(), "infer-agent-0");
    }

    #[test]
    fn stub_put_and_get_return_m5_stub_error() {
        let t = NixlTransport::new("infer-agent-0");
        let err = t
            .put_batch(&[])
            .expect_err("put_batch should be a stub error");
        match err {
            TransportError::Other(msg) => assert!(
                msg.contains("M5 stub"),
                "expected M5 stub marker, got: {msg}"
            ),
            other => panic!("expected Other, got {other:?}"),
        }
        let err = t
            .get_batch(&[])
            .expect_err("get_batch should be a stub error");
        assert!(matches!(err, TransportError::Other(_)));
    }

    #[test]
    fn stub_register_returns_m5_stub_error() {
        let t = NixlTransport::new("infer-agent-0");
        let mut dummy = [0u8; 16];
        // Safety: pointer is valid for the duration of the unsafe call;
        // the stub does not dereference it.
        let err = unsafe { t.register(dummy.as_mut_ptr(), dummy.len(), MemKind::Host) }
            .expect_err("register should be a stub error");
        assert!(matches!(err, TransportError::Other(_)));
    }

    #[test]
    fn stub_poll_returns_ready_error() {
        let t = NixlTransport::new("infer-agent-0");
        let mut op = NixlOp { _stub: () };
        match t.poll(&mut op) {
            Poll::Ready(Err(TransportError::Other(msg))) => {
                assert!(msg.contains("M5 stub"));
            }
            other => panic!("expected Ready(Err(Other)), got {other:?}"),
        }
    }
}

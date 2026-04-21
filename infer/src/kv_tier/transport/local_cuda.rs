//! Local CUDA transport skeleton for M3a.
//!
//! The real implementation will own two dedicated copy streams and perform
//! `cudaMemcpyAsync` between GPU HBM and the host pinned pool. This local file
//! freezes the trait surface and error plumbing without claiming the copies are
//! implemented on non-CUDA hosts.

use std::task::Poll;

use super::{KVTransport, TransferOp, TransportError};
use crate::kv_tier::tier::{BlockLocation, MemKind};

#[derive(Debug, Clone)]
pub struct LocalCudaRegion {
    pub ptr_addr: usize,
    pub len: usize,
    pub kind: MemKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalCudaOpState {
    Pending,
    Aborted,
}

#[derive(Debug, Clone)]
pub struct LocalCudaOp {
    pub ops: Vec<TransferOp>,
    pub state: LocalCudaOpState,
}

#[derive(Debug, Default)]
pub struct LocalCudaTransport;

impl LocalCudaTransport {
    pub fn new() -> Self {
        Self
    }

    fn validate_ops(ops: &[TransferOp]) -> Result<(), TransportError> {
        if ops.is_empty() {
            return Err(TransportError::Other(
                "LocalCudaTransport requires at least one transfer op".into(),
            ));
        }
        for op in ops {
            if op.is_empty() {
                return Err(TransportError::Other(
                    "LocalCudaTransport rejects zero-length transfers".into(),
                ));
            }
            match (&op.src.location, &op.dst.location) {
                (BlockLocation::Gpu { .. }, BlockLocation::HostPinned { .. })
                | (BlockLocation::HostPinned { .. }, BlockLocation::Gpu { .. }) => {}
                _ => {
                    return Err(TransportError::Other(
                        "LocalCudaTransport only supports local GPU <-> host pinned transfers"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl KVTransport for LocalCudaTransport {
    type Region = LocalCudaRegion;
    type Op = LocalCudaOp;

    unsafe fn register(
        &self,
        ptr: *mut u8,
        len: usize,
        kind: MemKind,
    ) -> Result<Self::Region, TransportError> {
        if ptr.is_null() || len == 0 {
            return Err(TransportError::Registration(
                "null pointer or zero-length registration".into(),
            ));
        }
        Ok(LocalCudaRegion {
            ptr_addr: ptr as usize,
            len,
            kind,
        })
    }

    fn put_batch(&self, ops: &[TransferOp]) -> Result<Self::Op, TransportError> {
        Self::validate_ops(ops)?;
        Ok(LocalCudaOp {
            ops: ops.to_vec(),
            state: LocalCudaOpState::Pending,
        })
    }

    fn get_batch(&self, ops: &[TransferOp]) -> Result<Self::Op, TransportError> {
        Self::validate_ops(ops)?;
        Ok(LocalCudaOp {
            ops: ops.to_vec(),
            state: LocalCudaOpState::Pending,
        })
    }

    fn poll(&self, op: &mut Self::Op) -> Poll<Result<(), TransportError>> {
        match op.state {
            LocalCudaOpState::Aborted => Poll::Ready(Err(TransportError::Aborted)),
            LocalCudaOpState::Pending => Poll::Ready(Err(TransportError::Other(
                "GPU required: LocalCudaTransport poll is a structural stub on this lane".into(),
            ))),
        }
    }

    fn abort(&self, op: &mut Self::Op) {
        op.state = LocalCudaOpState::Aborted;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_tier::tier::BlockLocation;

    #[test]
    fn register_rejects_null_and_zero_length() {
        let transport = LocalCudaTransport::new();

        let err =
            unsafe { transport.register(std::ptr::null_mut(), 16, MemKind::Vram { device: 0 }) }
                .expect_err("null pointers must be rejected");
        assert!(matches!(err, TransportError::Registration(_)));

        let mut byte = 0_u8;
        let err = unsafe { transport.register(&raw mut byte, 0, MemKind::Vram { device: 0 }) }
            .expect_err("zero-length registrations must be rejected");
        assert!(matches!(err, TransportError::Registration(_)));
    }

    #[test]
    fn put_batch_rejects_non_gpu_host_pairs() {
        let transport = LocalCudaTransport::new();
        let err = transport
            .put_batch(&[TransferOp::new(
                BlockLocation::Disk {
                    fingerprint: crate::types::BlockFingerprint([0x11; 16]),
                    payload_len: 4096,
                },
                BlockLocation::Gpu { slot: 1 },
                4096,
            )])
            .expect_err("disk->gpu must stay out of the local CUDA transport");
        assert!(matches!(err, TransportError::Other(_)));
    }

    #[test]
    fn abort_marks_pending_op_as_aborted() {
        let transport = LocalCudaTransport::new();
        let mut op = transport
            .put_batch(&[TransferOp::new(
                BlockLocation::Gpu { slot: 1 },
                BlockLocation::HostPinned { offset: 7 },
                4096,
            )])
            .unwrap();

        transport.abort(&mut op);
        assert_eq!(op.state, LocalCudaOpState::Aborted);
        assert!(matches!(
            transport.poll(&mut op),
            Poll::Ready(Err(TransportError::Aborted))
        ));
    }
}

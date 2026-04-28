//! `CollectiveBackend` trait: pluggable communication primitive.
//!
//! F0 ships `NcclBackend` skeleton; F7 adds CustomAR / mscclpp / quick_ar /
//! symm_mem impls behind the same trait. The trait method set is taken from
//! actual F1+ callers (LayerCommunicator AR, PP send/recv, MoE all-to-all
//! via group_start/end) — see plans/2026-04-28-single-node-multi-gpu.md §4.2.

use anyhow::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum ReduceOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum DType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    I32 = 3,
}

/// Pluggable collective transport. Implementations: `NcclBackend` (F0/F1),
/// CustomAR / mscclpp / quick_ar / symm_mem (F7).
///
/// All buffer/stream pointer methods are `unsafe`: callers must ensure the
/// pointer is a valid GPU allocation on this backend's device and the stream
/// belongs to the same device.
pub trait CollectiveBackend: Send + Sync {
    fn world_size(&self) -> usize;
    fn rank(&self) -> usize;

    /// In-place all-reduce.
    /// In-place all-reduce.
    ///
    /// # Safety
    /// `buffer` must be a valid GPU pointer holding `count` elements of
    /// `dtype`; `stream` must be a valid stream for this backend's device.
    unsafe fn all_reduce(
        &self,
        buffer: *mut std::ffi::c_void,
        count: usize,
        dtype: DType,
        op: ReduceOp,
        stream: *mut std::ffi::c_void,
    ) -> Result<()>;

    /// All-gather: every rank contributes `sendcount` elements; `recvbuf`
    /// receives `sendcount * world_size` elements.
    ///
    /// # Safety
    /// See `all_reduce`. `recvbuf` must hold `sendcount * world_size`
    /// elements of `dtype`.
    unsafe fn all_gather(
        &self,
        sendbuf: *const std::ffi::c_void,
        recvbuf: *mut std::ffi::c_void,
        sendcount: usize,
        dtype: DType,
        stream: *mut std::ffi::c_void,
    ) -> Result<()>;

    /// Reduce-scatter: input is reduced across ranks, then sliced;
    /// each rank receives `recvcount` elements.
    ///
    /// # Safety
    /// See `all_reduce`. `sendbuf` must hold `recvcount * world_size`
    /// elements of `dtype`.
    unsafe fn reduce_scatter(
        &self,
        sendbuf: *const std::ffi::c_void,
        recvbuf: *mut std::ffi::c_void,
        recvcount: usize,
        dtype: DType,
        op: ReduceOp,
        stream: *mut std::ffi::c_void,
    ) -> Result<()>;

    /// In-place broadcast from `root` to all ranks.
    ///
    /// # Safety
    /// See `all_reduce`.
    unsafe fn broadcast(
        &self,
        buffer: *mut std::ffi::c_void,
        count: usize,
        dtype: DType,
        root: usize,
        stream: *mut std::ffi::c_void,
    ) -> Result<()>;

    /// Point-to-point send to `peer`.
    ///
    /// # Safety
    /// See `all_reduce`. Must be paired with a matching `recv` on `peer`
    /// (or wrapped in `group_start`/`group_end`).
    unsafe fn send(
        &self,
        sendbuf: *const std::ffi::c_void,
        count: usize,
        dtype: DType,
        peer: usize,
        stream: *mut std::ffi::c_void,
    ) -> Result<()>;

    /// Point-to-point recv from `peer`.
    ///
    /// # Safety
    /// See `send`.
    unsafe fn recv(
        &self,
        recvbuf: *mut std::ffi::c_void,
        count: usize,
        dtype: DType,
        peer: usize,
        stream: *mut std::ffi::c_void,
    ) -> Result<()>;

    fn group_start(&self) -> Result<()>;
    fn group_end(&self) -> Result<()>;

    /// Whether this backend's collectives are safe to capture inside a CUDA
    /// graph. NCCL: false (F8 will revisit). CustomAR / SymmMem: true.
    fn supports_graph_capture(&self) -> bool;
}

#[cfg(feature = "nccl")]
pub use nccl_backend::NcclBackend;

#[cfg(feature = "nccl")]
mod nccl_backend {
    use super::{CollectiveBackend, DType, ReduceOp};
    use crate::ffi::nccl;
    use anyhow::Result;

    pub struct NcclBackend {
        comm: nccl::ncclComm_t,
        world_size: usize,
        rank: usize,
    }

    // SAFETY: NCCL communicators are thread-safe per the NCCL API contract;
    // the pointer itself is opaque and is only dereferenced through FFI
    // calls that NCCL serializes internally.
    unsafe impl Send for NcclBackend {}
    unsafe impl Sync for NcclBackend {}

    impl NcclBackend {
        /// Construct from an externally-acquired unique_id (rank 0 calls
        /// `ncclGetUniqueId` and broadcasts via the TCP rendezvous in
        /// `infer::distributed::init_method`).
        pub fn init_rank(
            unique_id: nccl::ncclUniqueId,
            world_size: usize,
            rank: usize,
        ) -> Result<Self> {
            let mut comm: nccl::ncclComm_t = std::ptr::null_mut();
            let res = unsafe {
                nccl::ncclCommInitRank(&mut comm, world_size as i32, unique_id, rank as i32)
            };
            nccl::check(res)?;
            Ok(Self {
                comm,
                world_size,
                rank,
            })
        }

        fn map_dtype(dtype: DType) -> nccl::ncclDataType_t {
            match dtype {
                DType::F16 => nccl::ncclDataType_t::Float16,
                DType::BF16 => nccl::ncclDataType_t::Bfloat16,
                DType::F32 => nccl::ncclDataType_t::Float32,
                DType::I32 => nccl::ncclDataType_t::Int32,
            }
        }

        fn map_op(op: ReduceOp) -> nccl::ncclRedOp_t {
            match op {
                ReduceOp::Sum => nccl::ncclRedOp_t::Sum,
                ReduceOp::Prod => nccl::ncclRedOp_t::Prod,
                ReduceOp::Max => nccl::ncclRedOp_t::Max,
                ReduceOp::Min => nccl::ncclRedOp_t::Min,
                ReduceOp::Avg => nccl::ncclRedOp_t::Avg,
            }
        }
    }

    impl CollectiveBackend for NcclBackend {
        fn world_size(&self) -> usize {
            self.world_size
        }

        fn rank(&self) -> usize {
            self.rank
        }

        unsafe fn all_reduce(
            &self,
            buffer: *mut std::ffi::c_void,
            count: usize,
            dtype: DType,
            op: ReduceOp,
            stream: *mut std::ffi::c_void,
        ) -> Result<()> {
            let res = unsafe {
                nccl::ncclAllReduce(
                    buffer as *const _,
                    buffer,
                    count,
                    Self::map_dtype(dtype),
                    Self::map_op(op),
                    self.comm,
                    stream,
                )
            };
            nccl::check(res)
        }

        unsafe fn all_gather(
            &self,
            sendbuf: *const std::ffi::c_void,
            recvbuf: *mut std::ffi::c_void,
            sendcount: usize,
            dtype: DType,
            stream: *mut std::ffi::c_void,
        ) -> Result<()> {
            let res = unsafe {
                nccl::ncclAllGather(
                    sendbuf,
                    recvbuf,
                    sendcount,
                    Self::map_dtype(dtype),
                    self.comm,
                    stream,
                )
            };
            nccl::check(res)
        }

        unsafe fn reduce_scatter(
            &self,
            sendbuf: *const std::ffi::c_void,
            recvbuf: *mut std::ffi::c_void,
            recvcount: usize,
            dtype: DType,
            op: ReduceOp,
            stream: *mut std::ffi::c_void,
        ) -> Result<()> {
            let res = unsafe {
                nccl::ncclReduceScatter(
                    sendbuf,
                    recvbuf,
                    recvcount,
                    Self::map_dtype(dtype),
                    Self::map_op(op),
                    self.comm,
                    stream,
                )
            };
            nccl::check(res)
        }

        unsafe fn broadcast(
            &self,
            buffer: *mut std::ffi::c_void,
            count: usize,
            dtype: DType,
            root: usize,
            stream: *mut std::ffi::c_void,
        ) -> Result<()> {
            let res = unsafe {
                nccl::ncclBroadcast(
                    buffer as *const _,
                    buffer,
                    count,
                    Self::map_dtype(dtype),
                    root as i32,
                    self.comm,
                    stream,
                )
            };
            nccl::check(res)
        }

        unsafe fn send(
            &self,
            sendbuf: *const std::ffi::c_void,
            count: usize,
            dtype: DType,
            peer: usize,
            stream: *mut std::ffi::c_void,
        ) -> Result<()> {
            let res = unsafe {
                nccl::ncclSend(
                    sendbuf,
                    count,
                    Self::map_dtype(dtype),
                    peer as i32,
                    self.comm,
                    stream,
                )
            };
            nccl::check(res)
        }

        unsafe fn recv(
            &self,
            recvbuf: *mut std::ffi::c_void,
            count: usize,
            dtype: DType,
            peer: usize,
            stream: *mut std::ffi::c_void,
        ) -> Result<()> {
            let res = unsafe {
                nccl::ncclRecv(
                    recvbuf,
                    count,
                    Self::map_dtype(dtype),
                    peer as i32,
                    self.comm,
                    stream,
                )
            };
            nccl::check(res)
        }

        fn group_start(&self) -> Result<()> {
            nccl::check(unsafe { nccl::ncclGroupStart() })
        }

        fn group_end(&self) -> Result<()> {
            nccl::check(unsafe { nccl::ncclGroupEnd() })
        }

        fn supports_graph_capture(&self) -> bool {
            false
        }
    }

    impl Drop for NcclBackend {
        fn drop(&mut self) {
            unsafe {
                let _ = nccl::ncclCommDestroy(self.comm);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_enum_size() {
        assert_eq!(std::mem::size_of::<DType>(), 4);
    }

    #[test]
    fn reduce_op_enum_size() {
        assert_eq!(std::mem::size_of::<ReduceOp>(), 4);
    }
}

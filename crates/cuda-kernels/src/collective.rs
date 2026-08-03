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
    // SAFETY: `&self` methods only hand `comm` back to NCCL, which is thread-safe
    // per its API contract; matching collective *order* across ranks is the
    // caller's job (unchecked here) but is a hang, not memory unsafety.
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
            // SAFETY: `comm` is a live out-param; `unique_id` is an owned 128-byte
            // POD. Caller guarantees every rank calls with the same id (unchecked).
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

        /// Split this comm into a sub-comm via `ncclCommSplit`. COLLECTIVE over
        /// the parent comm: every parent rank must call `split` together, in the
        /// same order. Ranks sharing `color` join the same sub-comm, ordered by
        /// `key`. Returns a fresh `NcclBackend` owning the sub-comm: its `Drop`
        /// independently calls `ncclCommDestroy` on the sub-comm, so there is no
        /// aliasing with — and no double-free of — the parent's comm.
        ///
        /// `world_size`/`rank` of the returned backend are the sub-comm's local
        /// membership: the number of ranks sharing `color` and this rank's `key`
        /// within that color (the caller knows both from the partition).
        ///
        /// Consumed by `autograd::CudaBackend::new_with_mesh` (CP×DP seq subgroup).
        ///
        /// # Errors
        /// Propagates the NCCL `ncclCommSplit` error.
        pub fn split(
            &self,
            color: i32,
            key: i32,
            sub_world: usize,
            sub_rank: usize,
        ) -> Result<Self> {
            let mut newcomm: nccl::ncclComm_t = std::ptr::null_mut();
            // `config = null` ⇒ the sub-comm inherits the parent's config.
            // SAFETY: `self.comm` is live for `&self`, `newcomm` is a live out-param,
            // and a null config is the documented "inherit" value.
            let res = unsafe {
                nccl::ncclCommSplit(self.comm, color, key, &mut newcomm, std::ptr::null_mut())
            };
            nccl::check(res)?;
            Ok(Self {
                comm: newcomm,
                world_size: sub_world,
                rank: sub_rank,
            })
        }

        /// Out-of-place all-reduce (the trait method is in-place; the comm
        /// bench needs distinct send/recv buffers so every arm runs the same
        /// input→output dataflow).
        ///
        /// # Safety
        /// `sendbuf`/`recvbuf` must be valid device allocations of `count`
        /// elements on this rank's device; `stream` must belong to it.
        pub unsafe fn all_reduce_out_of_place(
            &self,
            sendbuf: *const std::ffi::c_void,
            recvbuf: *mut std::ffi::c_void,
            count: usize,
            dtype: DType,
            op: ReduceOp,
            stream: *mut std::ffi::c_void,
        ) -> Result<()> {
            // SAFETY: per this fn's contract sendbuf/recvbuf are live device
            // allocations of `count` elements on `self.comm`'s device, as is `stream`.
            let res = unsafe {
                nccl::ncclAllReduce(
                    sendbuf,
                    recvbuf,
                    count,
                    Self::map_dtype(dtype),
                    Self::map_op(op),
                    self.comm,
                    stream,
                )
            };
            nccl::check(res)
        }

        /// NCCL ≥ 2.27 symmetric-memory pair: `ncclMemAlloc` a buffer and
        /// register it as a `NCCL_WIN_COLL_SYMMETRIC` window over this comm.
        /// COLLECTIVE: every rank must call together with the same `bytes`.
        /// Collectives whose operands are all symmetric windows take NCCL's
        /// low-latency symmetric kernels (the C4 arm of the comm bench).
        /// Returns the raw device pointer + window handle; the caller owns
        /// both (deregister + `ncclMemFree` on teardown).
        ///
        /// # Safety
        /// The returned pointer is a raw device allocation; the caller must
        /// not use it after `ncclMemFree`, and must deregister before freeing.
        pub unsafe fn mem_alloc_symmetric_window(
            &self,
            bytes: usize,
        ) -> Result<(*mut std::ffi::c_void, nccl::ncclWindow_t)> {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: `ptr` is a live out-param; NCCL owns the allocation it writes there.
            nccl::check(unsafe { nccl::ncclMemAlloc(&mut ptr, bytes) })?;
            let mut win: nccl::ncclWindow_t = std::ptr::null_mut();
            // SAFETY: `ptr` is the `bytes`-sized allocation just returned above, `win` a
            // live out-param, `self.comm` live for `&self`.
            let res = unsafe {
                nccl::ncclCommWindowRegister(
                    self.comm,
                    ptr,
                    bytes,
                    &mut win,
                    nccl::NCCL_WIN_COLL_SYMMETRIC,
                )
            };
            if let Err(err) = nccl::check(res) {
                // Roll back the allocation so a failed registration (e.g. NCCL
                // < 2.27 at runtime) doesn't leak device memory.
                // SAFETY: `ptr` came from `ncclMemAlloc` above and registration failed,
                // so no window still references it.
                let _ = unsafe { nccl::ncclMemFree(ptr) };
                return Err(err);
            }
            Ok((ptr, win))
        }

        /// Tear down a [`Self::mem_alloc_symmetric_window`] pair.
        ///
        /// # Safety
        /// `ptr`/`win` must come from `mem_alloc_symmetric_window` on this comm
        /// and must not be used afterwards.
        pub unsafe fn free_symmetric_window(
            &self,
            ptr: *mut std::ffi::c_void,
            win: nccl::ncclWindow_t,
        ) -> Result<()> {
            // SAFETY: caller guarantees `ptr`/`win` came from
            // `mem_alloc_symmetric_window` on this comm and are unused after this
            // (unchecked); deregistering before the free is the required order.
            nccl::check(unsafe { nccl::ncclCommWindowDeregister(self.comm, win) })?;
            // SAFETY: `ptr` is the `ncclMemAlloc` allocation, now deregistered.
            nccl::check(unsafe { nccl::ncclMemFree(ptr) })?;
            Ok(())
        }

        /// Host-visible all-gather for small byte payloads such as CUDA IPC
        /// handles. The payload is staged through an `i32` device buffer so this
        /// reuses NCCL all-gather without adding a host-side rendezvous path.
        #[cfg(feature = "cuda")]
        pub fn all_gather_bytes(
            &self,
            ctx: &crate::prelude::DeviceContext,
            input: &[u8],
            per_rank_bytes: usize,
        ) -> anyhow::Result<Vec<u8>> {
            use anyhow::{bail, ensure};
            use cudarc::driver::{DevicePtr, DevicePtrMut};

            ensure!(
                input.len() == per_rank_bytes,
                "all_gather_bytes input len {} must equal per-rank bytes {per_rank_bytes}",
                input.len()
            );
            ensure!(
                per_rank_bytes.is_multiple_of(4),
                "all_gather_bytes currently requires 4-byte alignment, got {per_rank_bytes}"
            );
            if per_rank_bytes == 0 {
                return Ok(Vec::new());
            }

            let send_words: Vec<i32> = input
                .chunks_exact(4)
                .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("4-byte chunk")))
                .collect();
            let per_rank_words = send_words.len();
            let send = ctx.stream.clone_htod(&send_words)?;
            let mut recv = ctx
                .stream
                .alloc_zeros::<i32>(per_rank_words * self.world_size)?;
            {
                let (src, _src_guard) = send.device_ptr(&ctx.stream);
                let (dst, _dst_guard) = recv.device_ptr_mut(&ctx.stream);
                // SAFETY: src/dst are live for the guards held in this scope — send holds
                // per_rank_words i32s, recv per_rank_words*world_size — and the stream owns both.
                unsafe {
                    self.all_gather(
                        src as *const std::ffi::c_void,
                        dst as *mut std::ffi::c_void,
                        per_rank_words,
                        DType::I32,
                        ctx.stream.cu_stream().cast::<std::ffi::c_void>(),
                    )?;
                }
            }
            ctx.stream.synchronize()?;
            let recv_words = ctx.stream.clone_dtoh(&recv)?;
            let out: Vec<u8> = recv_words.iter().flat_map(|w| w.to_ne_bytes()).collect();
            if out.len() != per_rank_bytes * self.world_size {
                bail!(
                    "all_gather_bytes produced {} bytes, expected {}",
                    out.len(),
                    per_rank_bytes * self.world_size
                );
            }
            Ok(out)
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
            // SAFETY: trait contract — `buffer` holds `count` `dtype` elements on this
            // comm's device and `stream` belongs to it; NCCL allows send==recv in-place.
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
            // SAFETY: trait contract — `sendbuf` holds `sendcount` elements and `recvbuf`
            // `sendcount * world_size`, both on this comm's device, `stream` on it too.
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
            // SAFETY: trait contract — `sendbuf` holds `recvcount * world_size` elements
            // and `recvbuf` `recvcount`, both on this comm's device, as is `stream`.
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
            // SAFETY: trait contract — `buffer` holds `count` `dtype` elements on this
            // comm's device; `root` is a caller-supplied rank, unchecked against world_size.
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
            // SAFETY: trait contract — `sendbuf` holds `count` `dtype` elements on this
            // comm's device; caller pairs this with a matching `recv` on `peer` (unchecked).
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
            // SAFETY: trait contract — `recvbuf` is writable for `count` `dtype` elements
            // on this comm's device; caller pairs this with a `send` on `peer` (unchecked).
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
            // SAFETY: no arguments; NCCL only flips its thread-local group depth. Pairing
            // with `group_end` on the same thread is the caller's job.
            nccl::check(unsafe { nccl::ncclGroupStart() })
        }

        fn group_end(&self) -> Result<()> {
            // SAFETY: no arguments; flushes the thread-local group opened by
            // `group_start`, whose buffers the caller keeps live until this returns.
            nccl::check(unsafe { nccl::ncclGroupEnd() })
        }

        fn supports_graph_capture(&self) -> bool {
            false
        }
    }

    impl Drop for NcclBackend {
        fn drop(&mut self) {
            // SAFETY: `comm` is owned solely by this backend (`split` returns a distinct
            // sub-comm), so this is its one and only destroy.
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

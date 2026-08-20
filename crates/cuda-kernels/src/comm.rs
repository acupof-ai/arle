//! Typed launchers for the custom all-reduce production ABI (`ffi/comm.rs`,
//! `csrc/comm/custom_all_reduce.cu`): IPC region/peer lifecycle plus the bf16
//! collectives over the registered scratch. Placement, overlap policy, and
//! buffer choice stay with the consumer (`infer-cuda/src/tp.rs`). The
//! bench-scoped rendezvous symbols (`arle_car_bootstrap` set) stay raw for
//! `comm_bench`.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::sys::{CUresult, CUstream};

use crate::ffi::comm as ffi;
use crate::tensor::DeviceContext;

/// Bind `ctx` to the calling thread; required before any IPC alloc/open on a
/// thread that has not touched this device yet.
pub fn bind(ctx: &DeviceContext) -> Result<()> {
    ctx.ctx
        .bind_to_thread()
        .map_err(|err| anyhow!("bind symmetric IPC CUDA context failed: {err}"))
}

fn check(res: CUresult, what: &str) -> Result<()> {
    ensure!(res == CUresult::CUDA_SUCCESS, "{what} failed: {res:?}");
    Ok(())
}

/// A cudaMalloc'd region with an exported `cudaIpcMemHandle_t`, freed on drop
/// unless [`Self::disarm`]ed (ownership transferred to [`CustomAllReduce`]).
pub struct SharedRegion {
    ptr: u64,
    label: &'static str,
}

impl SharedRegion {
    pub fn alloc(
        ctx: &DeviceContext,
        bytes: usize,
        handle: &mut [u8; 64],
        label: &'static str,
    ) -> Result<Self> {
        bind(ctx)?;
        let mut ptr = 0u64;
        check(
            // SAFETY: `ptr` and the 64-byte `handle` are live out-params and `bind`
            // above made `ctx` the current context.
            unsafe { ffi::arle_car_alloc_shared(bytes, &mut ptr, handle.as_mut_ptr()) },
            label,
        )?;
        Ok(Self { ptr, label })
    }

    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    pub fn disarm(&mut self) {
        self.ptr = 0;
    }
}

impl Drop for SharedRegion {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        // SAFETY: non-zero `ptr` is this region's own `alloc_shared` result and
        // `disarm` clears it once ownership moved, so this frees at most once.
        let res = unsafe { ffi::arle_car_free_shared(self.ptr) };
        if res != CUresult::CUDA_SUCCESS {
            log::warn!(
                "[cuda-ipc] cleanup {} ptr=0x{:x} failed: {res:?}",
                self.label,
                self.ptr
            );
        }
    }
}

/// A peer rank's IPC handle opened into this address space, closed on drop
/// unless [`Self::disarm`]ed (ownership transferred to [`CustomAllReduce`]).
pub struct PeerMapping {
    ptr: u64,
    label: &'static str,
}

impl PeerMapping {
    /// Open a peer's 64-byte IPC handle. Failure here IS the no-P2P probe.
    pub fn open(ctx: &DeviceContext, handle: &[u8; 64], label: &'static str) -> Result<Self> {
        bind(ctx)?;
        let mut ptr = 0u64;
        check(
            // SAFETY: `ptr` is a live out-param, `handle` holds the 64 bytes the
            // callee reads, and `bind` above made `ctx` current.
            unsafe { ffi::arle_car_open_peer(handle.as_ptr(), &mut ptr) },
            label,
        )?;
        Ok(Self { ptr, label })
    }

    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    pub fn disarm(&mut self) {
        self.ptr = 0;
    }
}

impl Drop for PeerMapping {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        // SAFETY: non-zero `ptr` is this mapping's own `open_peer` result and
        // `disarm` clears it once ownership moved, so this closes at most once.
        let res = unsafe { ffi::arle_car_close_peer(self.ptr) };
        if res != CUresult::CUDA_SUCCESS {
            log::warn!(
                "[cuda-ipc] cleanup {} ptr=0x{:x} failed: {res:?}",
                self.label,
                self.ptr
            );
        }
    }
}

/// Algorithm selector for [`CustomAllReduce::allreduce_bf16_into`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum AllReduceAlgo {
    /// One-shot below 256 KB at world 8, two-shot above.
    Auto = 0,
    OneShot = 1,
    TwoShot = 2,
}

/// One rank's booted `CustomAllreduce` over `world` signal/input regions;
/// owns every transferred pointer and destroys them on drop.
///
/// Not `Send`/`Sync`: the consumer decides its threading discipline.
pub struct CustomAllReduce {
    handle: *mut std::os::raw::c_void,
}

impl CustomAllReduce {
    /// Build over `world` signal/input device pointers (index = rank; this
    /// rank's entries are its own [`SharedRegion`]s, others opened
    /// [`PeerMapping`]s). Registers the input set and takes OWNERSHIP of every
    /// pointer — on success the caller must `disarm` the regions and mappings.
    pub fn create(
        rank: usize,
        world: usize,
        signal_ptrs: &[u64],
        input_ptrs: &[u64],
    ) -> Result<Self> {
        ensure!(
            signal_ptrs.len() >= world && input_ptrs.len() >= world,
            "CustomAllReduce::create pointer sets do not cover world={world}: \
             signals={} inputs={}",
            signal_ptrs.len(),
            input_ptrs.len()
        );
        // SAFETY: length checks above; the slices stay live through the call.
        let handle = unsafe {
            ffi::arle_car_create(
                i32::try_from(rank)?,
                i32::try_from(world)?,
                signal_ptrs.as_ptr(),
                input_ptrs.as_ptr(),
            )
        };
        ensure!(
            !handle.is_null(),
            "arle_car_create returned null (rank {rank} world {world})"
        );
        Ok(Self { handle })
    }

    /// bf16 sum allreduce: this rank's registered input scratch → `out_ptr`
    /// (any local buffer).
    ///
    /// # Safety
    /// `out_ptr` is a live device allocation of `elems` bf16 on this rank's
    /// device, the registered scratch holds `elems` bf16 of input, and
    /// `stream` belongs to this rank's device.
    pub unsafe fn allreduce_bf16_into(
        &self,
        stream: CUstream,
        out_ptr: u64,
        elems: usize,
        algo: AllReduceAlgo,
    ) -> Result<()> {
        check(
            // SAFETY: this fn's contract.
            unsafe {
                ffi::arle_car_allreduce_bf16_into(
                    self.handle,
                    stream,
                    out_ptr,
                    i32::try_from(elems)?,
                    algo as i32,
                )
            },
            "arle_car_allreduce_bf16_into",
        )
        .map_err(|e| anyhow!("{e} (elems={elems})"))
    }

    /// bf16 one-shot all-gather of every rank's registered scratch chunk into
    /// rank-major `out_ptr` (`[world × per_rank_elems]`, ncclAllGather layout).
    ///
    /// # Safety
    /// `out_ptr` is a live device allocation of `world * per_rank_elems` bf16
    /// on this rank's device, every rank's scratch holds `per_rank_elems`
    /// bf16, and `stream` belongs to this rank's device.
    pub unsafe fn allgather_bf16_into(
        &self,
        stream: CUstream,
        out_ptr: u64,
        per_rank_elems: usize,
    ) -> Result<()> {
        check(
            // SAFETY: this fn's contract.
            unsafe {
                ffi::arle_car_allgather_bf16_into(
                    self.handle,
                    stream,
                    out_ptr,
                    i32::try_from(per_rank_elems)?,
                )
            },
            "arle_car_allgather_bf16_into",
        )
        .map_err(|e| anyhow!("{e} (per_rank_elems={per_rank_elems})"))
    }
}

impl Drop for CustomAllReduce {
    fn drop(&mut self) {
        // SAFETY: `handle` came from `arle_car_create` and is owned solely here
        // (the struct is not `Clone`), so this destroys once.
        unsafe { ffi::arle_car_destroy_prod(self.handle) };
    }
}

/// Deterministic per-seed bf16 fill (correctness reference inputs).
///
/// # Safety
/// `ptr` is a live device allocation of `elems` bf16 on `stream`'s device.
pub unsafe fn fill_bf16(stream: CUstream, ptr: u64, elems: usize, seed: i32) -> Result<()> {
    check(
        // SAFETY: this fn's contract.
        unsafe { ffi::arle_car_fill_bf16(stream, ptr, i32::try_from(elems)?, seed) },
        "arle_car_fill_bf16",
    )
    .map_err(|e| anyhow!("{e} (elems={elems})"))
}

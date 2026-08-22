//! CUDA backend via cuBLAS SGEMM plus NVRTC-compiled point kernels.
//!
//! Row-major dispatch uses the cuBLAS swap-and-transpose trick: for row-major
//! `C[M,N] = A[M,K] @ B[K,N]`, call SGEMM with args swapped (A=B_data,
//! B=A_data) and m=N, n=M, k=K so cuBLAS's column-major view of the output
//! buffer matches the row-major layout on host. Batched (rank-3) uses
//! `sgemm_strided_batched` with the same swap.

#[cfg(not(feature = "no-cuda"))]
use crate::{
    AutogradError,
    backend::{
        CudaBf16Storage, CudaFp4E2M1GroupStorage, CudaFp8BlockScaledStorage, CudaStorage,
        LinearAttentionDeviceParams, cpu_causal_sdpa_recompute_backward,
        dequantize_fp8_block_scaled_host, matmul_bt_output_shape, matmul_output_shape,
        validate_broadcast, validate_decode_gqa_cache_shapes, validate_decode_gqa_shapes,
        validate_fp8_block_scaled, validate_qwen_decode_prepare_kv_shapes,
        validate_qwen_decode_prepare_q_shapes, validate_slice_shape,
    },
};
use crate::{
    Result,
    backend::{
        Backend, CausalSdpaDeviceBackwardArgs, CausalSdpaDeviceGradTriplet, CommAxis, Device,
        DeviceGradClipResult, DeviceHandle, LinearAttentionDeviceBackwardArgs,
        LinearAttentionDeviceBackwardResult, LinearAttentionDeviceBoundaryArgs,
        LinearAttentionDeviceForwardArgs, LinearAttentionDeviceForwardResult,
        LinearAttentionScanBackwardArgs, LinearAttentionScanBackwardGrads, RingBlockDims,
    },
};
#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/kernels.rs"]
mod kernels;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/handle.rs"]
mod handle;
#[cfg(not(feature = "no-cuda"))]
use self::handle::*;

#[path = "backend_cuda/state.rs"]
mod state;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/matmul.rs"]
mod matmul;
#[cfg(not(feature = "no-cuda"))]
use self::matmul::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/checkpoint.rs"]
mod checkpoint;
#[cfg(not(feature = "no-cuda"))]
use self::checkpoint::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/collective.rs"]
mod collective;
#[cfg(not(feature = "no-cuda"))]
use self::collective::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/broadcast.rs"]
mod broadcast;
#[cfg(not(feature = "no-cuda"))]
use self::broadcast::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/elementwise.rs"]
mod elementwise;
#[cfg(not(feature = "no-cuda"))]
use self::elementwise::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/embedding.rs"]
mod embedding;
#[cfg(not(feature = "no-cuda"))]
use self::embedding::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/gather.rs"]
mod gather;
#[cfg(not(feature = "no-cuda"))]
use self::gather::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/layout.rs"]
mod layout;
#[cfg(not(feature = "no-cuda"))]
use self::layout::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/norm.rs"]
mod norm;
#[cfg(not(feature = "no-cuda"))]
use self::norm::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/optim.rs"]
mod optim;
#[cfg(not(feature = "no-cuda"))]
use self::optim::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/reduce.rs"]
mod reduce;
#[cfg(not(feature = "no-cuda"))]
use self::reduce::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/rope.rs"]
mod rope;
#[cfg(not(feature = "no-cuda"))]
use self::rope::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/matmul_backward.rs"]
mod matmul_backward;
#[cfg(not(feature = "no-cuda"))]
use self::matmul_backward::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/sdpa_prefill.rs"]
mod sdpa_prefill;
#[cfg(not(feature = "no-cuda"))]
use self::sdpa_prefill::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/sdpa_decode.rs"]
mod sdpa_decode;
#[cfg(not(feature = "no-cuda"))]
use self::sdpa_decode::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/ring_attn.rs"]
mod ring_attn;
#[cfg(not(feature = "no-cuda"))]
use self::ring_attn::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/linear_attention_forward.rs"]
mod linear_attention_forward;
#[cfg(not(feature = "no-cuda"))]
use self::linear_attention_forward::*;

#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/linear_attention_backward.rs"]
mod linear_attention_backward;
#[cfg(not(feature = "no-cuda"))]
use self::linear_attention_backward::*;

#[cfg(not(feature = "no-cuda"))]
use self::kernels::{KernelCache, launch_1d, launch_rows};
#[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
use cuda_kernels::collective::{CollectiveBackend, DType, NcclBackend, ReduceOp};
#[cfg(not(feature = "no-cuda"))]
use cuda_kernels::ffi;
#[cfg(not(feature = "no-cuda"))]
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
#[cfg(not(feature = "no-cuda"))]
use cudarc::cublas::sys::cublasOperation_t;
#[cfg(not(feature = "no-cuda"))]
use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
#[cfg(not(feature = "no-cuda"))]
use cudarc::driver::sys::{
    CUdeviceptr, CUresult, CUstream, cuEventRecord, cuMemcpyDtoD_v2, cuMemcpyDtoDAsync_v2,
    cuStreamWaitEvent,
};
#[cfg(not(feature = "no-cuda"))]
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, PinnedHostSlice,
    PushKernelArg, result,
};
use std::sync::atomic::{AtomicU8, Ordering};
#[cfg(not(feature = "no-cuda"))]
use std::sync::{Arc, Mutex};

use crate::TapeDtype;

#[cfg(not(feature = "no-cuda"))]
type Fp8BlockScaledView<'a> = (
    &'a CudaSlice<u8>,
    &'a CudaSlice<f32>,
    usize,
    usize,
    usize,
    usize,
);

#[cfg(not(feature = "no-cuda"))]
const CUBLASLT_BF16_GEMMEX_MIN_N: usize = 32;

/// Complete artifact identity of one autograd NVRTC compile unit — the
/// training analogue of the serving side's `KERNEL_BUILD_ID`. Two runs with
/// equal identity load numerically identical cubins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvrtcIdentity {
    /// FNV-1a 64 hex hash of the dtype prelude + concatenated kernel sources.
    pub source_hash: String,
    pub compile_flags: String,
    pub sm_arch: &'static str,
    pub tape_dtype: TapeDtype,
    pub nvrtc_version: (i32, i32),
    pub cuda_driver_version: i32,
}

#[cfg(not(feature = "no-cuda"))]
struct DequantCacheEntry {
    key: usize,
    bf16: Arc<cudarc::driver::CudaSlice<u16>>,
    shape: Vec<usize>,
}

pub struct CudaBackend {
    tape_dtype: AtomicU8,
    #[cfg(not(feature = "no-cuda"))]
    stream: Arc<CudaStream>,
    #[cfg(not(feature = "no-cuda"))]
    blas: Arc<CudaBlas>,
    #[cfg(not(feature = "no-cuda"))]
    kernels: KernelCache,
    #[cfg(not(feature = "no-cuda"))]
    pinned_checkpoints: Mutex<PinnedCheckpointPool>,
    /// Last frozen weight dequantized to bf16, keyed by its source buffer. A
    /// chunked backward asks for the same weight once per chunk (16 at rank seq
    /// 65,536); the weights are frozen, so one entry turns that into one dequant.
    #[cfg(not(feature = "no-cuda"))]
    dequant_cache: Mutex<Option<DequantCacheEntry>>,
    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    nccl: Option<Arc<NcclBackend>>,
    /// Seq-collective comm: split subgroup when composed, else same as `nccl`.
    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    nccl_seq: Option<Arc<NcclBackend>>,
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("CudaBackend");
        #[cfg(not(feature = "no-cuda"))]
        {
            debug.field("device", &"cuda");
        }
        #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
        {
            debug.field("nccl", &self.nccl.is_some());
        }
        debug.finish_non_exhaustive()
    }
}

impl Backend for CudaBackend {
    fn device(&self) -> Device {
        Device::Cuda
    }

    fn set_tape_dtype(&self, dtype: TapeDtype) {
        self.tape_dtype.store(dtype as u8, Ordering::Relaxed);
        // Declared warmup: compile the dtype's NVRTC module now instead of on
        // the first hot-path kernel lookup. Best-effort — a failure here
        // resurfaces as a hard error at first use.
        #[cfg(not(feature = "no-cuda"))]
        if let Err(err) = self.kernels.warm_dtype(dtype) {
            log::warn!("autograd cuda dtype-module warmup failed: {err}");
        }
        // One line per run so a log can be matched to the exact NVRTC artifact.
        #[cfg(not(feature = "no-cuda"))]
        match self.kernels.nvrtc_identity(dtype) {
            Ok(id) => log::info!(
                "autograd cuda kernels: src={} arch={} dtype={:?} nvrtc={}.{} driver={}",
                id.source_hash,
                id.sm_arch,
                id.tape_dtype,
                id.nvrtc_version.0,
                id.nvrtc_version.1,
                id.cuda_driver_version
            ),
            Err(err) => log::warn!("autograd cuda kernel identity unavailable: {err}"),
        }
    }

    fn tape_dtype(&self) -> TapeDtype {
        if self.tape_dtype.load(Ordering::Relaxed) == TapeDtype::Bf16 as u8 {
            TapeDtype::Bf16
        } else {
            TapeDtype::F32
        }
    }

    fn device_synchronize(&self) -> Result<()> {
        #[cfg(feature = "no-cuda")]
        {
            Ok(())
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            // `cuCtxSynchronize` on the train backend's context: drains every
            // stream so outstanding train ops (incl. async-pool allocs/frees)
            // are ordered ahead of an infer-thread weight reload/offload that
            // touches the same shared device pool. See the trait doc.
            self.stream
                .context()
                .synchronize()
                .map_err(|_| AutogradError::TapeInvariant("cuda device_synchronize failed"))
        }
    }

    fn stream_synchronize(&self) -> Result<()> {
        #[cfg(feature = "no-cuda")]
        {
            Ok(())
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            // `cuStreamSynchronize` on THIS backend's own default stream only —
            // does NOT drain a co-resident foreign context's streams (unlike
            // `device_synchronize`/`cuCtxSynchronize`). Required for the
            // `--share-frozen-base` handoff fence: the shared device primary
            // context's other streams belong to the idle-parked rollout engine
            // (event-tracking disabled), so a context-wide sync deadlocks.
            self.stream
                .synchronize()
                .map_err(|_| AutogradError::TapeInvariant("cuda stream_synchronize failed"))
        }
    }

    fn linear_attention_head_geometry_supported(&self, h: usize, hg: usize) -> bool {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (h, hg);
            true
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            crate::backend_cuda::linear_attention_backward::flashqla_gdr_symbols(h, hg).is_ok()
        }
    }

    fn device_mem_info(&self) -> Option<(usize, usize)> {
        self.mem_get_info().ok()
    }

    #[cfg(not(feature = "no-cuda"))]
    fn mem_pool_stats(&self) -> Option<(u64, u64)> {
        // SAFETY: pool is this backend's live context; each read writes one u64.
        unsafe {
            let pool = result::device::get_mem_pool(self.stream.context().cu_device()).ok()?;
            let read = |attr| -> Option<u64> {
                let mut v: u64 = 0;
                result::mem_pool::get_attribute(pool, attr, (&mut v as *mut u64).cast()).ok()?;
                Some(v)
            };
            use cudarc::driver::sys::CUmemPool_attribute::*;
            Some((
                read(CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
                read(CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
            ))
        }
    }

    #[cfg(not(feature = "no-cuda"))]
    fn mem_pool_used_high(&self) -> Option<u64> {
        // SAFETY: pool is this backend's live context; the read writes one u64.
        unsafe {
            let pool = result::device::get_mem_pool(self.stream.context().cu_device()).ok()?;
            let mut high: u64 = 0;
            result::mem_pool::get_attribute(
                pool,
                cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                (&mut high as *mut u64).cast(),
            )
            .ok()?;
            Some(high)
        }
    }

    #[cfg(not(feature = "no-cuda"))]
    fn reset_mem_pool_used_high(&self) -> Result<()> {
        // The driver accepts only 0 here; it means "rebase onto current used".
        let mut zero: u64 = 0;
        // SAFETY: pool belongs to this backend's context; the write reads one u64.
        unsafe {
            let pool = result::device::get_mem_pool(self.stream.context().cu_device())
                .map_err(|_| AutogradError::TapeInvariant("cuda get_mem_pool failed"))?;
            result::mem_pool::set_attribute(
                pool,
                cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                (&mut zero as *mut u64).cast(),
            )
            .map_err(|_| AutogradError::TapeInvariant("cuda USED_MEM_HIGH reset failed"))
        }
    }

    fn upload(&self, host: &[f32], shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (host, shape);
            todo!("GPU required: cuda upload is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            Ok(DeviceHandle::Cuda(CudaStorage::new(
                self.upload_slice(host, shape)?,
            )))
        }
    }

    fn upload_bf16_bits(&self, host: &[u16], shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (host, shape);
            todo!("GPU required: cuda bf16 upload is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(
                self.upload_bf16_bits_slice(host, shape)?,
            )))
        }
    }

    fn quantize_frozen_to_bf16(
        &self,
        handle: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (handle, shape);
            todo!("GPU required: cuda bf16 quantize is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_quantize_frozen_to_bf16(self, handle, shape)
        }
    }

    fn upload_fp8_block_scaled(
        &self,
        weight: &[u8],
        scales: &[f32],
        shape: &[usize],
        block_m: usize,
        block_k: usize,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (weight, scales, shape, block_m, block_k);
            todo!("GPU required: cuda fp8 block-scaled upload is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_upload_fp8_block_scaled(self, weight, scales, shape, block_m, block_k)
        }
    }

    fn upload_fp4_e2m1_group(
        &self,
        weight: &[u8],
        scales: &[u8],
        global_scale: f32,
        shape: &[usize],
        group_size: usize,
        scale_cols: usize,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (weight, scales, global_scale, shape, group_size, scale_cols);
            todo!("GPU required: cuda nvfp4 upload is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_upload_fp4_e2m1_group(
                self,
                weight,
                scales,
                global_scale,
                shape,
                group_size,
                scale_cols,
            )
        }
    }

    fn import_bf16_device_ptr_as_f32(
        &self,
        src_device_ptr: u64,
        src_stream: u64,
        len: usize,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (src_device_ptr, src_stream, len, shape);
            todo!("GPU required: cuda bf16 device import is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_import_bf16_device_ptr_as_f32(self, src_device_ptr, src_stream, len, shape)
        }
    }

    fn import_fp8_block_scaled_device_ptr(
        &self,
        weight_device_ptr: u64,
        scale_device_ptr: u64,
        shape: &[usize],
        block_m: usize,
        block_k: usize,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (weight_device_ptr, scale_device_ptr, shape, block_m, block_k);
            todo!(
                "GPU required: cuda fp8 block-scaled device import is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_import_fp8_block_scaled_device_ptr(
                self,
                weight_device_ptr,
                scale_device_ptr,
                shape,
                block_m,
                block_k,
            )
        }
    }

    fn import_fp4_marlin_device_ptr(
        &self,
        weight_device_ptr: u64,
        scale_tail_device_ptr: u64,
        global_scale: f32,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (
                weight_device_ptr,
                scale_tail_device_ptr,
                global_scale,
                shape,
            );
            todo!(
                "GPU required: cuda marlin nvfp4 device import is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_import_fp4_marlin_device_ptr(
                self,
                weight_device_ptr,
                scale_tail_device_ptr,
                global_scale,
                shape,
            )
        }
    }

    fn import_bf16_device_ptr(&self, device_ptr: u64, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (device_ptr, shape);
            todo!("GPU required: cuda bf16 device import is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_import_bf16_device_ptr(self, device_ptr, shape)
        }
    }

    fn zeros(&self, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = shape;
            todo!("GPU required: cuda zeros is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            let size = shape_size(shape);
            let slice = alloc_zeros_retry::<f32>(self, size).map_err(|e| {
                cuda_alloc_failed_rich(self, "zeros", size * std::mem::size_of::<f32>(), &e)
            })?;
            Ok(DeviceHandle::Cuda(CudaStorage::new(slice)))
        }
    }

    fn readback(&self, handle: &DeviceHandle) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = handle;
            todo!("GPU required: cuda readback is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_readback(self, handle)
        }
    }

    fn readback_into(&self, handle: &DeviceHandle, dst: &mut [f32]) -> Result<()> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (handle, dst);
            todo!("GPU required: cuda readback_into is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_readback_into(self, handle, dst)
        }
    }

    fn eval(&self, handles: &[&DeviceHandle]) -> Result<()> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = handles;
            todo!("GPU required: cuda eval is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            for handle in handles {
                self.validate_cuda_handle_kind(handle)?;
            }
            self.stream
                .synchronize()
                .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))
        }
    }

    fn checkpoint_pin_offload(&self, handle: &DeviceHandle, len: usize) -> Result<Option<u32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (handle, len);
            todo!("GPU required: cuda checkpoint_pin_offload is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_checkpoint_pin_offload(self, handle, len)
        }
    }

    fn checkpoint_pin_reload(&self, slot: u32, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (slot, shape);
            todo!("GPU required: cuda checkpoint_pin_reload is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_checkpoint_pin_reload(self, slot, shape)
        }
    }

    fn checkpoint_pin_readback(&self, slot: u32, dst: &mut [f32]) -> Result<()> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (slot, dst);
            todo!("GPU required: cuda checkpoint_pin_readback is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_checkpoint_pin_readback(self, slot, dst)
        }
    }

    fn checkpoint_pin_release(&self, slot: u32) {
        #[cfg(feature = "no-cuda")]
        {
            let _ = slot;
            todo!("GPU required: cuda checkpoint_pin_release is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        if let Ok(mut pool) = self.pinned_checkpoints.lock() {
            pool.release(slot);
        }
    }

    fn linear_attention_scan_backward(
        &self,
        args: LinearAttentionScanBackwardArgs<'_>,
    ) -> Result<Option<LinearAttentionScanBackwardGrads>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = args;
            todo!(
                "GPU required: cuda linear_attention_scan_backward is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_linear_attention_scan_backward(self, args)
        }
    }

    fn linear_attention_forward_device(
        &self,
        args: LinearAttentionDeviceForwardArgs<'_>,
    ) -> Result<Option<LinearAttentionDeviceForwardResult>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = args;
            todo!(
                "GPU required: cuda linear_attention_forward_device is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_linear_attention_forward_device(self, args)
        }
    }

    fn linear_attention_boundary_device(
        &self,
        args: LinearAttentionDeviceBoundaryArgs<'_>,
    ) -> Result<Option<DeviceHandle>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = args;
            todo!(
                "GPU required: cuda linear_attention_boundary_device is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_linear_attention_boundary_device(self, args)
        }
    }

    fn causal_sdpa_prefill_device(
        &self,
        q: &DeviceHandle,
        q_shape: &[usize],
        k: &DeviceHandle,
        k_shape: &[usize],
        v: &DeviceHandle,
        v_shape: &[usize],
        q_start: usize,
    ) -> Result<Option<(DeviceHandle, Vec<usize>)>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (q, q_shape, k, k_shape, v, v_shape, q_start);
            todo!(
                "GPU required: cuda causal_sdpa_prefill_device is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_causal_sdpa_prefill_device(self, q, q_shape, k, k_shape, v, v_shape, q_start)
        }
    }

    fn linear_attention_backward_device(
        &self,
        args: LinearAttentionDeviceBackwardArgs<'_>,
    ) -> Result<Option<LinearAttentionDeviceBackwardResult>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = args;
            todo!(
                "GPU required: cuda linear_attention_backward_device is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_linear_attention_backward_device(self, args)
        }
    }

    fn causal_sdpa_recompute_backward_device(
        &self,
        args: CausalSdpaDeviceBackwardArgs<'_>,
    ) -> Result<CausalSdpaDeviceGradTriplet> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = args;
            todo!(
                "GPU required: cuda causal_sdpa_recompute_backward_device is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_causal_sdpa_recompute_backward_device(self, args)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ring_block_fwd_merge(
        &self,
        q: &DeviceHandle,
        k_blk: &DeviceHandle,
        v_blk: &DeviceHandle,
        acc_m: &DeviceHandle,
        acc_l: &DeviceHandle,
        acc_o: &DeviceHandle,
        q_pos: &DeviceHandle,
        k_pos: &DeviceHandle,
        q_pos_host: &[usize],
        k_pos_host: &[usize],
        dims: RingBlockDims,
    ) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (
                q, k_blk, v_blk, acc_m, acc_l, acc_o, q_pos, k_pos, q_pos_host, k_pos_host, dims,
            );
            todo!("GPU required: cuda ring_block_fwd_merge is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_ring_block_fwd_merge(
                self, q, k_blk, v_blk, acc_m, acc_l, acc_o, q_pos, k_pos, q_pos_host, k_pos_host,
                dims,
            )
        }
    }

    fn ring_block_finalize(
        &self,
        acc_m: &DeviceHandle,
        acc_l: &DeviceHandle,
        acc_o: &DeviceHandle,
        total_rows: usize,
        head_dim: usize,
    ) -> Result<(DeviceHandle, DeviceHandle)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (acc_m, acc_l, acc_o, total_rows, head_dim);
            todo!("GPU required: cuda ring_block_finalize is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_ring_block_finalize(self, acc_m, acc_l, acc_o, total_rows, head_dim)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ring_block_bwd(
        &self,
        q: &DeviceHandle,
        k_blk: &DeviceHandle,
        v_blk: &DeviceHandle,
        out: &DeviceHandle,
        lse: &DeviceHandle,
        d_out: &DeviceHandle,
        grad_q: &DeviceHandle,
        q_pos: &DeviceHandle,
        k_pos: &DeviceHandle,
        q_pos_host: &[usize],
        k_pos_host: &[usize],
        dims: RingBlockDims,
    ) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (
                q, k_blk, v_blk, out, lse, d_out, grad_q, q_pos, k_pos, q_pos_host, k_pos_host,
                dims,
            );
            todo!("GPU required: cuda ring_block_bwd is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_ring_block_bwd(
                self, q, k_blk, v_blk, out, lse, d_out, grad_q, q_pos, k_pos, q_pos_host,
                k_pos_host, dims,
            )
        }
    }

    fn trim_memory_pool(&self) -> Result<bool> {
        #[cfg(feature = "no-cuda")]
        {
            Ok(false)
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            self.stream.context().bind_to_thread().map_err(|_| {
                AutogradError::TapeInvariant("cuda bind failed before mempool trim")
            })?;
            self.stream.synchronize().map_err(|_| {
                AutogradError::TapeInvariant("cuda synchronize failed before mempool trim")
            })?;
            // SAFETY: the context is bound above.
            let pool = unsafe { result::device::get_mem_pool(self.stream.context().cu_device()) }
                .map_err(|_| AutogradError::TapeInvariant("cuda get_mem_pool failed"))?;
            // SAFETY: the pool belongs to this context.
            unsafe { result::mem_pool::trim_to(pool, 0) }
                .map_err(|_| AutogradError::TapeInvariant("cuda mem_pool trim_to(0) failed"))?;
            Ok(true)
        }
    }

    fn matmul(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (a, a_shape, b, b_shape);
            todo!("GPU required: cuda lazy matmul is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            let a_op = self.f32_operand(a, "matmul")?;
            let a = a_op.get();
            let b_op = self.f32_operand(b, "matmul")?;
            let b = b_op.get();
            let (out, out_shape) = self.matmul_device(a, a_shape, b, b_shape)?;
            Ok((DeviceHandle::Cuda(CudaStorage::new(out)), out_shape))
        }
    }

    fn matmul_bt(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (a, a_shape, b, b_shape);
            todo!("GPU required: cuda lazy matmul_bt is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_matmul_bt(self, a, a_shape, b, b_shape)
        }
    }

    fn add(&self, a: &DeviceHandle, b: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (a, b, shape);
            todo!("GPU required: cuda lazy add is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_binary_1d_device(self, a, b, shape, "add_f32", "add")
        }
    }

    fn sum_all(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda sum_all is unavailable under feature no-cuda")
        }

        // On-device recursive block reduce: the host-reduce alternative was a
        // full-tensor DtoH (~32 MB/chunk at vocab=248320) + blocking sync per
        // mean in the CE/KL head, which serialized the GPU behind a CPU reduce.
        // The result stays on-device; no synchronize — the caller's terminal
        // eval owns the fence (batched-eval contract).
        #[cfg(not(feature = "no-cuda"))]
        {
            let imported;
            let x = match x {
                DeviceHandle::CudaBf16(storage) => {
                    let bits = self.cuda_bf16_storage_slice(storage)?;
                    imported = DeviceHandle::Cuda(CudaStorage::new(
                        self.import_local_bf16_as_f32(bits, bits.len())?,
                    ));
                    &imported
                }
                _ => x,
            };
            let slice = self.cuda_slice(x, "sum_all")?;
            let size = shape_size(shape);
            if slice.len() != size {
                return Err(AutogradError::DataLengthMismatch {
                    len: slice.len(),
                    shape: shape.to_vec(),
                    size,
                });
            }
            cuda_sum_all_device(self, x, size)
        }
    }

    fn sum_squares(&self, x: &DeviceHandle, shape: &[usize]) -> Result<f64> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda sum_squares is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_sum_squares(self, x, shape)
        }
    }

    fn clip_grad_norm_device(
        &self,
        grads: &[(DeviceHandle, Vec<usize>)],
        max_norm: f32,
    ) -> Result<Option<DeviceGradClipResult>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (grads, max_norm);
            todo!("GPU required: cuda clip_grad_norm_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_clip_grad_norm_device(self, grads, max_norm).map(Some)
        }
    }

    fn matmul_forward(
        &self,
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (a, a_shape, b, b_shape);
            todo!("GPU required: cuda matmul_forward is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            let a_handle = self.upload(a, a_shape)?;
            let b_handle = self.upload(b, b_shape)?;
            let (out_handle, out_shape) = self.matmul(&a_handle, a_shape, &b_handle, b_shape)?;
            self.eval(&[&out_handle])?;
            let out = self.readback(&out_handle)?;
            Ok((out, out_shape))
        }
    }

    fn matmul_backward(
        &self,
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
        grad_out: &[f32],
        grad_out_shape: &[usize],
        need_grad_a: bool,
        need_grad_b: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (
                a,
                a_shape,
                b,
                b_shape,
                grad_out,
                grad_out_shape,
                need_grad_a,
                need_grad_b,
            );
            todo!("GPU required: cuda matmul_backward is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_matmul_backward(
                self,
                a,
                a_shape,
                b,
                b_shape,
                grad_out,
                grad_out_shape,
                need_grad_a,
                need_grad_b,
            )
        }
    }

    /// Device-resident matmul backward: consumes device handles and returns
    /// unevaluated slices — no host roundtrip; the terminal eval in
    /// `AdamW::step_device` is the single host fence (batched-eval contract).
    fn matmul_backward_device(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
        grad_out: &DeviceHandle,
        grad_out_shape: &[usize],
        need_grad_a: bool,
        need_grad_b: bool,
    ) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (
                a,
                a_shape,
                b,
                b_shape,
                grad_out,
                grad_out_shape,
                need_grad_a,
                need_grad_b,
            );
            todo!("GPU required: cuda matmul_backward_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_matmul_backward_device(
                self,
                a,
                a_shape,
                b,
                b_shape,
                grad_out,
                grad_out_shape,
                need_grad_a,
                need_grad_b,
            )
        }
    }

    fn matmul_bt_backward_device(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
        grad_out: &DeviceHandle,
        grad_out_shape: &[usize],
        need_grad_a: bool,
        need_grad_b: bool,
    ) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (
                a,
                a_shape,
                b,
                b_shape,
                grad_out,
                grad_out_shape,
                need_grad_a,
                need_grad_b,
            );
            todo!(
                "GPU required: cuda matmul_bt_backward_device is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_matmul_bt_backward_device(
                self,
                a,
                a_shape,
                b,
                b_shape,
                grad_out,
                grad_out_shape,
                need_grad_a,
                need_grad_b,
            )
        }
    }

    fn matmul_bt_input_grad_device(
        &self,
        b: &DeviceHandle,
        b_shape: &[usize],
        grad_out: &DeviceHandle,
        grad_out_shape: &[usize],
        input_shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (b, b_shape, grad_out, grad_out_shape, input_shape);
            todo!(
                "GPU required: cuda matmul_bt_input_grad_device is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_matmul_bt_input_grad_device(
                self,
                b,
                b_shape,
                grad_out,
                grad_out_shape,
                input_shape,
            )
        }
    }

    fn add_into_device(
        &self,
        dest: &DeviceHandle,
        src: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (dest, src, shape);
            todo!("GPU required: cuda add_into_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_add_into_device(self, dest, src, shape)
        }
    }

    fn accumulate_into_device(
        &self,
        dest: &DeviceHandle,
        src: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (dest, src, shape);
            todo!("GPU required: cuda accumulate_into_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_accumulate_into_device(self, dest, src, shape)
        }
    }

    fn all_reduce_sum_device(
        &self,
        x: &DeviceHandle,
        shape: &[usize],
        axis: CommAxis,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape, axis);
            todo!("GPU required: cuda all_reduce_sum_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_all_reduce_sum_device(self, x, shape, axis)
        }
    }

    fn all_gather_seq_device(
        &self,
        x: &DeviceHandle,
        local_shape: &[usize],
        axis: CommAxis,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, local_shape, axis);
            todo!("GPU required: cuda all_gather_seq_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_all_gather_seq_device(self, x, local_shape, axis)
        }
    }

    fn reduce_scatter_sum_device(
        &self,
        x: &DeviceHandle,
        local_shape: &[usize],
        axis: CommAxis,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, local_shape, axis);
            todo!(
                "GPU required: cuda reduce_scatter_sum_device is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_reduce_scatter_sum_device(self, x, local_shape, axis)
        }
    }

    fn ring_send_recv_kv(
        &self,
        block: &DeviceHandle,
        block_shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (block, block_shape);
            todo!("GPU required: cuda ring_send_recv_kv is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_ring_send_recv_kv(self, block, block_shape)
        }
    }

    fn cp_send_device(&self, handle: &DeviceHandle, len: usize, peer: usize) -> Result<()> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (handle, len, peer);
            todo!("GPU required: cuda cp_send_device is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_cp_send(self, handle, len, peer)
        }
    }

    fn cp_recv_device(&self, len: usize, peer: usize) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (len, peer);
            todo!("GPU required: cuda cp_recv_device is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_cp_recv(self, len, peer)
        }
    }

    fn all_to_all_device(
        &self,
        x: &DeviceHandle,
        in_shape: &[usize],
        scatter_axis: usize,
        gather_axis: usize,
        axis: CommAxis,
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, in_shape, scatter_axis, gather_axis, axis);
            todo!("GPU required: cuda all_to_all_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_all_to_all_device(self, x, in_shape, scatter_axis, gather_axis, axis)
        }
    }

    /// Returns an unevaluated handle per the batched-eval contract.
    fn mul_scalar_backward_device(
        &self,
        upstream_grad: &DeviceHandle,
        scale: f32,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream_grad, scale, shape);
            todo!(
                "GPU required: cuda mul_scalar_backward_device is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_mul_scalar_backward_device(self, upstream_grad, scale, shape)
        }
    }

    /// `upstream_grad` is a rank-0 device handle; returns an unevaluated handle.
    fn mean_backward_device(
        &self,
        upstream_grad: &DeviceHandle,
        output_shape: &[usize],
        elem_count: usize,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream_grad, output_shape, elem_count);
            todo!("GPU required: cuda mean_backward_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_mean_backward_device(self, upstream_grad, output_shape, elem_count)
        }
    }

    fn sum_backward_device(
        &self,
        upstream_grad: &DeviceHandle,
        output_shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream_grad, output_shape);
            todo!("GPU required: cuda sum_backward_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_sum_backward_device(self, upstream_grad, output_shape)
        }
    }

    fn softmax_forward_last_axis(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda softmax is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_softmax_like(self, x, shape, "softmax_last_axis_f32")
        }
    }

    fn log_softmax_forward_last_axis(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda log_softmax is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_softmax_like(self, x, shape, "log_softmax_last_axis_f32")
        }
    }

    /// Device-resident row-wise softmax: the default readback→host→upload
    /// fallback dominates per-step wall time at `[B,S,V]` ≈ 1 GB. Reuses the
    /// existing NVRTC kernel; no synchronize — eval belongs to the caller.
    fn softmax_last_axis(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda softmax_last_axis is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_softmax_like_device(self, x, shape, "softmax_last_axis_f32")
        }
    }

    /// Device-resident row-wise log-softmax; same rationale as `softmax_last_axis`.
    fn log_softmax_last_axis(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda log_softmax_last_axis is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_softmax_like_device(self, x, shape, "log_softmax_last_axis_f32")
        }
    }

    fn softmax_last_axis_backward(
        &self,
        upstream: &DeviceHandle,
        softmax_output: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, softmax_output, shape);
            todo!(
                "GPU required: cuda softmax_last_axis_backward is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_softmax_last_axis_backward(self, upstream, softmax_output, shape)
        }
    }

    fn mul_forward(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (a, b);
            todo!("GPU required: cuda mul is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_binary_1d(self, a, b, "mul_f32")
        }
    }

    fn mul_scalar_forward(&self, a: &[f32], s: f32) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (a, s);
            todo!("GPU required: cuda mul_scalar is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_scalar_1d(self, a, s, "mul_scalar_f32")
        }
    }

    fn silu(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda silu is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_unary_1d_device(self, x, shape, "silu_f32", "silu")
        }
    }

    fn sigmoid(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda sigmoid is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_unary_1d_device(self, x, shape, "sigmoid_f32", "sigmoid")
        }
    }

    fn abs(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda abs is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_unary_1d_device(self, x, shape, "abs_f32", "abs")
        }
    }

    fn mul(&self, a: &DeviceHandle, b: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (a, b, shape);
            todo!("GPU required: cuda mul is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_binary_1d_device(self, a, b, shape, "mul_f32", "mul")
        }
    }

    fn mul_scalar(&self, x: &DeviceHandle, s: f32, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, s, shape);
            todo!("GPU required: cuda mul_scalar is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_scalar_1d_device(self, x, s, shape, "mul_scalar_f32", "mul_scalar")
        }
    }

    fn add_broadcast_forward(
        &self,
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (a, a_shape, b, b_shape);
            todo!("GPU required: cuda add_broadcast is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_add_broadcast(self, a, a_shape, b, b_shape)
        }
    }

    fn add_broadcast(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (a, a_shape, b, b_shape);
            todo!("GPU required: cuda add_broadcast is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_add_broadcast_device(self, a, a_shape, b, b_shape)
        }
    }

    fn broadcast_expand(
        &self,
        src: &DeviceHandle,
        src_shape: &[usize],
        target_shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (src, src_shape, target_shape);
            todo!("GPU required: cuda broadcast_expand is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_broadcast_expand_device(self, src, src_shape, target_shape)
        }
    }

    fn exp_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = a;
            todo!("GPU required: cuda exp is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_unary_1d(self, a, "exp_f32")
        }
    }

    fn neg_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = a;
            todo!("GPU required: cuda neg is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_unary_1d(self, a, "neg_f32")
        }
    }

    fn gelu_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = a;
            todo!("GPU required: cuda gelu is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_unary_1d(self, a, "gelu_f32")
        }
    }

    fn silu_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = a;
            todo!("GPU required: cuda silu is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_unary_1d(self, a, "silu_f32")
        }
    }

    fn rms_norm_forward(
        &self,
        x: &[f32],
        weight: &[f32],
        shape: &[usize],
        eps: f32,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, weight, shape, eps);
            todo!("GPU required: cuda rms_norm is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_rms_norm(self, x, weight, shape, eps)
        }
    }

    fn rms_norm(
        &self,
        x: &DeviceHandle,
        weight: &[f32],
        shape: &[usize],
        eps: f32,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, weight, shape, eps);
            todo!("GPU required: cuda rms_norm is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_rms_norm_device(self, x, weight, shape, eps)
        }
    }

    fn embedding_forward(
        &self,
        weight: &[f32],
        vocab: usize,
        dim: usize,
        ids: &[i32],
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (weight, vocab, dim, ids);
            todo!("GPU required: cuda embedding is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_embedding(self, weight, vocab, dim, ids)
        }
    }

    fn embedding(
        &self,
        table: &DeviceHandle,
        table_shape: &[usize],
        ids: &[i32],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (table, table_shape, ids);
            todo!("GPU required: cuda embedding is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_embedding_device(self, table, table_shape, ids)
        }
    }

    fn embedding_from_f32_ids(
        &self,
        table: &DeviceHandle,
        table_shape: &[usize],
        ids: &DeviceHandle,
        n_ids: usize,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (table, table_shape, ids, n_ids);
            todo!("GPU required: cuda embedding_from_f32_ids is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_embedding_from_f32_ids_device(self, table, table_shape, ids, n_ids)
        }
    }

    fn argmax_last_dim(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda argmax_last_dim is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_argmax_last_dim(self, x, shape)
        }
    }

    fn write_scalar_at(
        &self,
        dest: &DeviceHandle,
        src: &DeviceHandle,
        len: usize,
        index: usize,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (dest, src, len, index);
            todo!("GPU required: cuda write_scalar_at is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_write_scalar_at(self, dest, src, len, index)
        }
    }

    fn sum_last_axis_forward(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda sum is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_reduce_last_axis(self, x, shape, "sum_last_axis_f32")
        }
    }

    fn mean_last_axis_forward(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda mean is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_reduce_last_axis(self, x, shape, "mean_last_axis_f32")
        }
    }

    fn rope_forward(
        &self,
        x: &[f32],
        x_shape: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, x_shape, cos, sin);
            todo!("GPU required: cuda rope is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_rope(self, x, x_shape, cos, sin)
        }
    }

    fn rope(
        &self,
        x: &DeviceHandle,
        x_shape: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, x_shape, cos, sin);
            todo!("GPU required: cuda rope is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_rope_device(self, x, x_shape, cos, sin)
        }
    }

    fn gather_last_dim_forward(
        &self,
        src: &[f32],
        src_shape: &[usize],
        ids: &[i32],
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (src, src_shape, ids);
            todo!("GPU required: cuda gather is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_gather_last_dim(self, src, src_shape, ids)
        }
    }

    /// Device-resident gather along the last axis: keeps the `[B,S,V]` logits
    /// on-device through the CE-loss chain instead of materializing the full
    /// ~1 GB tensor on host between `log_softmax` and `gather`.
    fn gather_last_dim(
        &self,
        src: &DeviceHandle,
        src_shape: &[usize],
        ids: &[i32],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (src, src_shape, ids);
            todo!("GPU required: cuda gather_last_dim is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_gather_last_dim_device(self, src, src_shape, ids)
        }
    }

    /// Device-resident backward for `log_softmax_last_axis`: consumes the
    /// saved forward output and upstream directly from device — kills the
    /// 1 015 MB log_softmax-grad readback nsys identified as the largest
    /// transfer per training step. Unevaluated handle per the batched-eval
    /// contract.
    fn log_softmax_last_axis_backward(
        &self,
        upstream: &DeviceHandle,
        log_softmax_output: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, log_softmax_output, shape);
            todo!(
                "GPU required: cuda log_softmax_last_axis_backward is unavailable under feature no-cuda"
            )
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_log_softmax_last_axis_backward(self, upstream, log_softmax_output, shape)
        }
    }

    /// Device-resident backward for `gather_last_dim`: one thread per prefix
    /// row writes its upstream scalar at `(row, ids[row])` — no atomics,
    /// since indices across rows touch disjoint slots.
    fn gather_last_dim_backward(
        &self,
        upstream: &DeviceHandle,
        indices: &[i32],
        src_shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, indices, src_shape);
            todo!(
                "GPU required: cuda gather_last_dim_backward is unavailable under feature no-cuda"
            )
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_gather_last_dim_backward(self, upstream, indices, src_shape)
        }
    }

    fn reshape(&self, x: &DeviceHandle, new_shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, new_shape);
            todo!("GPU required: cuda reshape is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            let expected = shape_size(new_shape);
            let len = match x {
                DeviceHandle::CudaBf16(storage) => self.cuda_bf16_storage_slice(storage)?.len(),
                _ => self.cuda_slice(x, "reshape")?.len(),
            };
            if len != expected {
                return Err(AutogradError::DataLengthMismatch {
                    len,
                    shape: new_shape.to_vec(),
                    size: expected,
                });
            }
            Ok(x.clone())
        }
    }

    fn transpose_axes_swap(
        &self,
        x: &DeviceHandle,
        old_shape: &[usize],
        axis1: usize,
        axis2: usize,
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, old_shape, axis1, axis2);
            todo!("GPU required: cuda transpose is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_transpose_axes_swap_device(self, x, old_shape, axis1, axis2)
        }
    }

    fn slice(
        &self,
        x: &DeviceHandle,
        old_shape: &[usize],
        starts: &[usize],
        ends: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, old_shape, starts, ends);
            todo!("GPU required: cuda slice is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_slice_device(self, x, old_shape, starts, ends)
        }
    }

    fn concat_axis2(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (a, a_shape, b, b_shape);
            todo!("GPU required: cuda concat_axis2 is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_concat_axis2_device(self, a, a_shape, b, b_shape)
        }
    }

    fn concat(
        &self,
        parts: &[(&DeviceHandle, &[usize])],
        axis: usize,
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (parts, axis);
            todo!("GPU required: cuda concat is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_concat_axis(self, parts, axis)
        }
    }

    fn kv_cache_write_axis2(
        &self,
        dst: &DeviceHandle,
        dst_shape: &[usize],
        src: &DeviceHandle,
        src_shape: &[usize],
        seq_offset: usize,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (dst, dst_shape, src, src_shape, seq_offset);
            todo!("GPU required: cuda kv_cache_write_axis2 is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_kv_cache_write_axis2(self, dst, dst_shape, src, src_shape, seq_offset)
        }
    }

    fn causal_sdpa_decode_gqa(
        &self,
        q: &DeviceHandle,
        q_shape: &[usize],
        k: &DeviceHandle,
        k_shape: &[usize],
        v: &DeviceHandle,
        v_shape: &[usize],
        q_start: usize,
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (q, q_shape, k, k_shape, v, v_shape, q_start);
            todo!("GPU required: cuda causal_sdpa_decode_gqa is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_causal_sdpa_decode_gqa(self, q, q_shape, k, k_shape, v, v_shape, q_start)
        }
    }

    fn causal_sdpa_decode_gqa_cache(
        &self,
        q: &DeviceHandle,
        q_shape: &[usize],
        k: &DeviceHandle,
        k_shape: &[usize],
        v: &DeviceHandle,
        v_shape: &[usize],
        kv_len: usize,
        q_start: usize,
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (q, q_shape, k, k_shape, v, v_shape, kv_len, q_start);
            todo!(
                "GPU required: cuda causal_sdpa_decode_gqa_cache is unavailable under feature no-cuda"
            )
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_causal_sdpa_decode_gqa_cache(
                self, q, q_shape, k, k_shape, v, v_shape, kv_len, q_start,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen_decode_prepare_q(
        &self,
        q_full: &DeviceHandle,
        q_full_shape: &[usize],
        q_norm_weight: &DeviceHandle,
        q_norm_weight_shape: &[usize],
        cos: &DeviceHandle,
        cos_shape: &[usize],
        sin: &DeviceHandle,
        sin_shape: &[usize],
        query_heads: usize,
        head_dim: usize,
        gated: bool,
        eps: f32,
    ) -> Result<(DeviceHandle, Option<DeviceHandle>, Vec<usize>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (
                q_full,
                q_full_shape,
                q_norm_weight,
                q_norm_weight_shape,
                cos,
                cos_shape,
                sin,
                sin_shape,
                query_heads,
                head_dim,
                gated,
                eps,
            );
            todo!("GPU required: cuda qwen_decode_prepare_q is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_qwen_decode_prepare_q(
                self,
                q_full,
                q_full_shape,
                q_norm_weight,
                q_norm_weight_shape,
                cos,
                cos_shape,
                sin,
                sin_shape,
                query_heads,
                head_dim,
                gated,
                eps,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen_decode_prepare_kv(
        &self,
        k_full: &DeviceHandle,
        k_full_shape: &[usize],
        v_full: &DeviceHandle,
        v_full_shape: &[usize],
        k_norm_weight: &DeviceHandle,
        k_norm_weight_shape: &[usize],
        cos: &DeviceHandle,
        cos_shape: &[usize],
        sin: &DeviceHandle,
        sin_shape: &[usize],
        kv_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(DeviceHandle, DeviceHandle, Vec<usize>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (
                k_full,
                k_full_shape,
                v_full,
                v_full_shape,
                k_norm_weight,
                k_norm_weight_shape,
                cos,
                cos_shape,
                sin,
                sin_shape,
                kv_heads,
                head_dim,
                eps,
            );
            todo!("GPU required: cuda qwen_decode_prepare_kv is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_qwen_decode_prepare_kv(
                self,
                k_full,
                k_full_shape,
                v_full,
                v_full_shape,
                k_norm_weight,
                k_norm_weight_shape,
                cos,
                cos_shape,
                sin,
                sin_shape,
                kv_heads,
                head_dim,
                eps,
            )
        }
    }

    fn slice_backward_device(
        &self,
        upstream: &DeviceHandle,
        input_shape: &[usize],
        starts: &[usize],
        ends: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, input_shape, starts, ends);
            todo!("GPU required: cuda slice_backward is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_slice_backward_device(self, upstream, input_shape, starts, ends)
        }
    }

    fn permute_seq_blocks_device(
        &self,
        x: &DeviceHandle,
        batch: usize,
        num_blocks: usize,
        block_elems: usize,
        perm: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, batch, num_blocks, block_elems, perm);
            todo!(
                "GPU required: cuda permute_seq_blocks_device is unavailable under feature no-cuda"
            )
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_permute_seq_blocks_device(self, x, batch, num_blocks, block_elems, perm)
        }
    }

    fn accumulate_slice_device(
        &self,
        dest: &DeviceHandle,
        upstream: &DeviceHandle,
        input_shape: &[usize],
        starts: &[usize],
        ends: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (dest, upstream, input_shape, starts, ends);
            todo!("GPU required: cuda accumulate_slice_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_accumulate_slice_device(self, dest, upstream, input_shape, starts, ends)
        }
    }

    fn write_slice_device(
        &self,
        dest: &DeviceHandle,
        upstream: &DeviceHandle,
        input_shape: &[usize],
        starts: &[usize],
        ends: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (dest, upstream, input_shape, starts, ends);
            todo!("GPU required: cuda write_slice_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_write_slice_device(self, dest, upstream, input_shape, starts, ends)
        }
    }

    fn scatter_add_rows_forward(
        &self,
        upstream: &[f32],
        prefix_rows: usize,
        feature_dim: usize,
        indices: &[i32],
        vocab: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, prefix_rows, feature_dim, indices, vocab);
            todo!("GPU required: cuda scatter_add_rows is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_scatter_add_rows(self, upstream, prefix_rows, feature_dim, indices, vocab)
        }
    }

    /// Device-resident embedding backward: atomicAdd-scatter is mandatory
    /// for the duplicate-token correctness guarantee. No synchronize —
    /// terminal eval is the caller's.
    fn embedding_backward_device(
        &self,
        upstream_grad: &DeviceHandle,
        indices: &[i32],
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream_grad, indices, vocab_size, hidden_dim);
            todo!(
                "GPU required: cuda embedding_backward_device is unavailable under feature no-cuda"
            )
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_embedding_backward_device(self, upstream_grad, indices, vocab_size, hidden_dim)
        }
    }

    /// Device-resident add_broadcast backward; mirrors the forward layout
    /// contract (right-aligned `b_strides` of length `out_rank`, stride-0
    /// entries for contracted axes).
    fn add_broadcast_backward_device(
        &self,
        upstream: &DeviceHandle,
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, a_shape, b_shape);
            todo!(
                "GPU required: cuda add_broadcast_backward_device is unavailable under feature no-cuda"
            )
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_add_broadcast_backward_device(self, upstream, a_shape, b_shape)
        }
    }

    /// Fused on-device AdamW: replaces the default host-loop fallback
    /// (`readback × 3 + cpu_adamw_step_in_place + upload × 3` per param per
    /// step) with one NVRTC kernel, mutating the existing param/m/v buffers
    /// in place. Matches `backend.rs::cpu_adamw_step_in_place` to ≤1e-4
    /// rel-error after 5 steps (`tests/test_cuda_adamw_step.rs`).
    #[allow(clippy::too_many_arguments)]
    fn adamw_step(
        &self,
        param: &DeviceHandle,
        m: &DeviceHandle,
        v: &DeviceHandle,
        grad: &[f32],
        shape: &[usize],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        bc1: f32,
        bc2: f32,
    ) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (
                param, m, v, grad, shape, lr, beta1, beta2, eps, wd, bc1, bc2,
            );
            todo!("GPU required: cuda adamw_step is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_adamw_step(
                self, param, m, v, grad, shape, lr, beta1, beta2, eps, wd, bc1, bc2,
            )
        }
    }

    /// Device-grad fused AdamW: same kernel as `adamw_step` but the gradient
    /// is sourced from the caller's `DeviceHandle::Cuda` — no `clone_htod`,
    /// killing the per-grad-accum-step DtoH from device-resident
    /// `embedding_backward` / `add_broadcast_backward` grads.
    #[allow(clippy::too_many_arguments)]
    fn adamw_step_device(
        &self,
        param: &DeviceHandle,
        m: &DeviceHandle,
        v: &DeviceHandle,
        grad: &DeviceHandle,
        shape: &[usize],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        bc1: f32,
        bc2: f32,
    ) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (
                param, m, v, grad, shape, lr, beta1, beta2, eps, wd, bc1, bc2,
            );
            todo!("GPU required: cuda adamw_step_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_adamw_step_device(
                self, param, m, v, grad, shape, lr, beta1, beta2, eps, wd, bc1, bc2,
            )
        }
    }

    /// Device-resident backward for `silu(x)`; returned handle is unevaluated.
    fn silu_backward_device(
        &self,
        upstream: &DeviceHandle,
        x: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, x, shape);
            todo!("GPU required: cuda silu_backward_device is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_silu_backward_device(self, upstream, x, shape)
        }
    }

    /// Device-resident backward for `gelu(x)` (erf form).
    fn gelu_backward_device(
        &self,
        upstream: &DeviceHandle,
        x: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, x, shape);
            todo!("GPU required: cuda gelu_backward_device is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_gelu_backward_device(self, upstream, x, shape)
        }
    }

    /// Device-resident backward for `sigmoid(x)`; consumes the saved output
    /// `y` (not the input).
    fn sigmoid_backward_device(
        &self,
        upstream: &DeviceHandle,
        y: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, y, shape);
            todo!("GPU required: cuda sigmoid_backward_device is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_sigmoid_backward_device(self, upstream, y, shape)
        }
    }

    /// Device-resident backward for `abs(x)`; consumes the saved input `x`,
    /// with `sign(0) = 0`.
    fn abs_backward_device(
        &self,
        upstream: &DeviceHandle,
        x: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, x, shape);
            todo!("GPU required: cuda abs_backward_device is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_abs_backward_device(self, upstream, x, shape)
        }
    }

    /// Device-resident backward for `exp(x)`; consumes the saved output
    /// `y = exp(x)` (not the input).
    fn exp_backward_device(
        &self,
        upstream: &DeviceHandle,
        y: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, y, shape);
            todo!("GPU required: cuda exp_backward_device is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_exp_backward_device(self, upstream, y, shape)
        }
    }

    fn mul_backward_device(
        &self,
        upstream: &DeviceHandle,
        a: &DeviceHandle,
        b: &DeviceHandle,
        shape: &[usize],
        need_grad_a: bool,
        need_grad_b: bool,
    ) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, a, b, shape, need_grad_a, need_grad_b);
            todo!("GPU required: cuda mul_backward_device is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_mul_backward_device(self, upstream, a, b, shape, need_grad_a, need_grad_b)
        }
    }

    fn rms_norm_backward_device(
        &self,
        upstream: &DeviceHandle,
        x: &DeviceHandle,
        weight: &DeviceHandle,
        shape: &[usize],
        eps: f32,
        need_grad_x: bool,
        need_grad_w: bool,
    ) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, x, weight, shape, eps, need_grad_x, need_grad_w);
            todo!(
                "GPU required: cuda rms_norm_backward_device is unavailable under feature no-cuda"
            )
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_rms_norm_backward_device(
                self,
                upstream,
                x,
                weight,
                shape,
                eps,
                need_grad_x,
                need_grad_w,
            )
        }
    }

    /// Device-resident backward for `rope`: same kernel body as the forward
    /// with the `sin` sign negated; `cos`/`sin` are uploaded fresh (tiny:
    /// `[seq, head_dim/2]`).
    fn rope_backward_device(
        &self,
        upstream: &DeviceHandle,
        x_shape: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (upstream, x_shape, cos, sin);
            todo!("GPU required: cuda rope_backward_device is unavailable under feature no-cuda")
        }
        #[cfg(not(feature = "no-cuda"))]
        {
            cuda_rope_backward_device(self, upstream, x_shape, cos, sin)
        }
    }
}

#[cfg(all(test, not(feature = "no-cuda")))]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::ops::ring_attention::ring_forward_tile;

    // The device ring kernel (ring_block_fwd_merge + finalize) had NO coverage:
    // CUDA world==1 takes the CPU path and the host multi-block test never touches
    // the .cu. This drives the device kernel over TWO distinct zigzag blocks on one
    // GPU (no NCCL — the merge/GQA/position-mask are what's under test, not the
    // transport) and compares to the verified host `ring_forward_tile`. Config
    // mirrors the pod failure: head_dim=128, GQA heads=4/kv=2, a zigzag shard whose
    // back chunk (the rows that ONLY attend the remote block) is where CE blew up.
    #[test]
    fn device_ring_two_blocks_matches_host_reference_gqa_hd128() -> Result<()> {
        let backend = CudaBackend::new(0)?;
        let (heads, kv_heads, d) = (4usize, 2usize, 128usize);
        let blk = 4usize; // rows per rank shard
        let gqa = heads / kv_heads;

        // Deterministic synthetic q/k/v, quantized to EXACT bf16 values (8-bit
        // mantissa) so f32 host == bf16 device and the tolerance isolates a real
        // algorithm bug from round-trip noise.
        let synth = |n: usize, seed: u64| -> Vec<f32> {
            let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    // f32 → bf16 (truncate low 16 mantissa bits) → f32: exact on device.
                    let raw = (((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 0.5;
                    f32::from_bits(raw.to_bits() & 0xffff_0000)
                })
                .collect()
        };

        // rank0 zigzag shard for seq=8, cp=2: owns rows {0,1,6,7}. Block 0 = its own
        // KV (positions 0,1,6,7); block 1 = rank1's KV (positions 2,3,4,5, delivered
        // by the ring). q rows carry the same positions as block 0.
        let q_pos = [0usize, 1, 6, 7];
        let blk0_pos = [0usize, 1, 6, 7];
        let blk1_pos = [2usize, 3, 4, 5];

        let q = synth(heads * blk * d, 1); // [heads, blk, d]
        let k0 = synth(kv_heads * blk * d, 2); // [kv_heads, blk, d]
        let v0 = synth(kv_heads * blk * d, 3);
        let k1 = synth(kv_heads * blk * d, 4);
        let v1 = synth(kv_heads * blk * d, 5);
        let scale = 1.0 / (d as f32).sqrt();

        // Host reference: per q-head, attend its GQA kv-head's block0 then block1.
        let mut ref_out = vec![0.0f32; heads * blk * d];
        for qh in 0..heads {
            let kvh = qh / gqa;
            let kv_tile = kvh * blk * d;
            let blocks: [(&[f32], &[f32], &[usize]); 2] = [
                (
                    &k0[kv_tile..kv_tile + blk * d],
                    &v0[kv_tile..kv_tile + blk * d],
                    &blk0_pos,
                ),
                (
                    &k1[kv_tile..kv_tile + blk * d],
                    &v1[kv_tile..kv_tile + blk * d],
                    &blk1_pos,
                ),
            ];
            let (o, _lse) = ring_forward_tile(
                &q[qh * blk * d..(qh + 1) * blk * d],
                &blocks,
                blk,
                d,
                scale,
                &q_pos,
            );
            ref_out[qh * blk * d..(qh + 1) * blk * d].copy_from_slice(&o);
        }

        // Device: init accumulators, feed block0 then block1 through the kernel.
        let rows = heads * blk;
        let up = |h: &[f32]| backend.upload(h, &[h.len()]);
        let q_h = up(&q)?;
        let qpos_h = up(&q_pos.iter().map(|&p| p as f32).collect::<Vec<_>>())?;
        let mut acc_m = up(&vec![f32::NEG_INFINITY; rows])?;
        let mut acc_l = up(&vec![0.0f32; rows])?;
        let mut acc_o = up(&vec![0.0f32; rows * d])?;
        let dims = RingBlockDims {
            num_q_tiles: heads,
            num_q_heads: heads,
            num_kv_heads: kv_heads,
            head_dim: d,
            q_rows: blk,
            blk_len: blk,
            sm_scale: scale,
        };
        for (k, v, kpos) in [(&k0, &v0, &blk0_pos), (&k1, &v1, &blk1_pos)] {
            let k_h = up(k)?;
            let v_h = up(v)?;
            let kpos_h = up(&kpos.iter().map(|&p| p as f32).collect::<Vec<_>>())?;
            let (m2, l2, o2) = backend.ring_block_fwd_merge(
                &q_h, &k_h, &v_h, &acc_m, &acc_l, &acc_o, &qpos_h, &kpos_h, &q_pos, kpos, dims,
            )?;
            acc_m = m2;
            acc_l = l2;
            acc_o = o2;
        }
        let (out_h, _lse_h) = backend.ring_block_finalize(&acc_m, &acc_l, &acc_o, rows, d)?;
        let got = backend.readback(&out_h)?;

        let max_diff = ref_out
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Inputs are exact bf16, so only f32 expf/accumulation differences remain
        // (~1e-4) — far below the 5.2% pod divergence this localizes.
        assert!(
            max_diff < 1.0e-3,
            "device ring != host reference: max_diff={max_diff}"
        );
        Ok(())
    }

    #[test]
    fn bf16_device_import_roundtrip_preserves_d2d_bytes_and_widens() -> Result<()> {
        let backend = CudaBackend::new(0)?;
        let bf16_bits: Vec<u16> = vec![0x3f80, 0xc020, 0x0000, 0x3e80, 0x7f7f];
        // Source on a second stream of the same context to exercise the
        // cross-stream event handshake (the production teacher/student shape).
        let alt_stream = backend
            .stream
            .context()
            .new_stream()
            .map_err(|_| AutogradError::TapeInvariant("new_stream failed (bf16 test)"))?;
        let src = alt_stream
            .clone_htod(&bf16_bits)
            .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (bf16 test)"))?;
        let (src_ptr, _src_guard) = src.device_ptr(&alt_stream);
        let src_stream = alt_stream.cu_stream() as u64;

        let staging =
            backend.copy_bf16_device_ptr_to_local(src_ptr, src_stream, bf16_bits.len())?;
        let copied = backend
            .stream
            .clone_dtoh(&staging)
            .map_err(|_| AutogradError::TapeInvariant("cuda dtoh copy failed (bf16 test)"))?;
        backend
            .stream
            .synchronize()
            .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed (bf16 test)"))?;
        assert_eq!(copied, bf16_bits);

        let handle = backend.import_bf16_device_ptr_as_f32(
            src_ptr,
            src_stream,
            bf16_bits.len(),
            &[bf16_bits.len()],
        )?;
        let widened = backend.readback(&handle)?;
        let expected: Vec<f32> = bf16_bits
            .iter()
            .map(|&bits| f32::from_bits((bits as u32) << 16))
            .collect();
        assert_eq!(widened, expected);
        Ok(())
    }

    #[test]
    fn bf16_bridge_timing_realistic_logits() -> Result<()> {
        let backend = CudaBackend::new(0)?;
        let alt_stream = backend
            .stream
            .context()
            .new_stream()
            .map_err(|_| AutogradError::TapeInvariant("new_stream failed (bf16 timing)"))?;
        // Realistic teacher logits: [512, 151936] bf16 ~= 155 MB.
        let seq_len = 512usize;
        let vocab = 151_936usize;
        let len = seq_len * vocab;
        let src: CudaSlice<u16> = alt_stream
            .alloc_zeros(len)
            .map_err(|_| AutogradError::TapeInvariant("alloc failed (bf16 timing)"))?;
        let (src_ptr, _src_guard) = src.device_ptr(&alt_stream);
        let src_stream = alt_stream.cu_stream() as u64;
        let shape = vec![seq_len, vocab];

        // A/B in one process, same device state: src_stream=0 is the legacy
        // sync path (cuMemcpyDtoD_v2 + context.synchronize), src_stream=alt
        // is the event-ordered async path.
        for (label, stream) in [("legacy-sync", 0u64), ("event-ordered", src_stream)] {
            let _ = backend.import_bf16_device_ptr_as_f32(src_ptr, stream, len, &shape)?;
            backend
                .stream
                .synchronize()
                .map_err(|_| AutogradError::TapeInvariant("sync failed (bf16 timing)"))?;

            let start = std::time::Instant::now();
            for _ in 0..5 {
                let _ = backend.import_bf16_device_ptr_as_f32(src_ptr, stream, len, &shape)?;
            }
            backend
                .stream
                .synchronize()
                .map_err(|_| AutogradError::TapeInvariant("sync failed (bf16 timing)"))?;
            let elapsed = start.elapsed();
            println!(
                "bf16_bridge_timing[{label}]: {:.3} ms/call (5 calls, {:.1} MB each)",
                elapsed.as_secs_f64() * 1000.0 / 5.0,
                (len * 2) as f64 / 1e6,
            );
        }
        Ok(())
    }
}

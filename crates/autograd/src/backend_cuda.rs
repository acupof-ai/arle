//! CUDA backend via cuBLAS SGEMM plus NVRTC-compiled point kernels.
//!
//! PENDING REMOTE CUDA VERIFICATION — user validates on GPU box.
//! Type-checks on Mac under `--no-default-features --features cuda,no-cuda`;
//! actual execution paths unreachable without a device are marked with
//! `todo!("GPU required: ...")` so a CPU-only binary fails loudly.
//!
//! Row-major dispatch uses the standard cuBLAS swap-and-transpose trick:
//! for row-major `C[M,N] = A[M,K] @ B[K,N]`, call SGEMM with args swapped
//! (A=B_data, B=A_data) and m=N, n=M, k=K so cuBLAS's column-major view
//! of the output buffer matches the row-major layout we want on host.
//! Batched (rank-3) uses `sgemm_strided_batched` with the same swap.

#[cfg(not(feature = "no-cuda"))]
use crate::{
    AutogradError,
    backend::{
        CudaBf16Storage, CudaFp8BlockScaledStorage, CudaStorage, LinearAttentionDeviceParams,
        cpu_causal_sdpa_recompute_backward, dequantize_fp8_block_scaled_host,
        matmul_bt_output_shape, matmul_output_shape, validate_broadcast,
        validate_decode_gqa_cache_shapes, validate_decode_gqa_shapes, validate_fp8_block_scaled,
        validate_qwen_decode_prepare_kv_shapes, validate_qwen_decode_prepare_q_shapes,
        validate_slice_shape,
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
use cudarc::driver::sys::{CUdeviceptr, CUresult, cuMemcpyDtoD_v2};
#[cfg(not(feature = "no-cuda"))]
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, PushKernelArg, result,
};
#[cfg(not(feature = "no-cuda"))]
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::TapeDtype;

/// Borrowed FP8 block-scaled tensor parts: (weight bytes, scales, rows, cols, block_m, block_k).
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

#[cfg(not(feature = "no-cuda"))]
fn check_cuda_ffi(status: CUresult, label: &'static str) -> Result<()> {
    if status == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(AutogradError::TapeInvariant(Box::leak(
            format!("{label} failed with CUDA status {status:?}").into_boxed_str(),
        )))
    }
}

#[cfg(not(feature = "no-cuda"))]
fn linear_attention_debug_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("ARLE_LINEAR_ATTENTION_DEBUG_TIMING").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        )
    })
}

#[cfg(not(feature = "no-cuda"))]
fn linear_attention_debug_stage_start() -> Option<std::time::Instant> {
    linear_attention_debug_timing_enabled().then(std::time::Instant::now)
}

#[cfg(not(feature = "no-cuda"))]
fn linear_attention_debug_stage_done(
    backend: &CudaBackend,
    label: &'static str,
    started: Option<std::time::Instant>,
) -> Result<()> {
    if let Some(started) = started {
        backend.stream.synchronize().map_err(|err| {
            AutogradError::TapeInvariant(Box::leak(
                format!("cuda synchronize failed (linear_attention {label}): {err:?}")
                    .into_boxed_str(),
            ))
        })?;
        eprintln!(
            "cuda linear_attention_forward stage={label} seconds={:.6}",
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

#[cfg(not(feature = "no-cuda"))]
fn linear_attention_gdr_chunkwise_prefill_enabled() -> bool {
    crate::runtime_flags::gdr_chunkwise_prefill()
}

/// A/B escape hatch: force the legacy monolithic chunked-scan backward (one
/// block per batch x value_head) instead of the staged chunk-parallel path.
#[cfg(not(feature = "no-cuda"))]
fn linear_attention_mono_backward_forced() -> bool {
    crate::runtime_flags::la_backward_mono()
}

/// Max concurrent chunk lanes in the stage-3 grad kernel. Bounds the per-block
/// history slab at `wave x rows x 64 x state_elems` f32 (1.6 GiB at 48 heads,
/// 128x128 state) independent of seq_len; 8 lanes x 48 rows = 384 blocks fills
/// H20's ~624-resident-block budget without oversubscribing it.
#[cfg(not(feature = "no-cuda"))]
const LA_BWD_CHUNK_WAVE: usize = 8;

/// cuBLAS-backed matmul plus NVRTC-compiled point kernels. Holds an
/// `Arc<CudaStream>` + `CudaBlas` so the context lives as long as the backend;
/// safe to share across threads.
#[cfg(not(feature = "no-cuda"))]
enum F32Operand<'a> {
    Borrowed(&'a CudaSlice<f32>),
    Imported(CudaSlice<f32>),
}

#[cfg(not(feature = "no-cuda"))]
impl F32Operand<'_> {
    fn get(&self) -> &CudaSlice<f32> {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Imported(slice) => slice,
        }
    }
}

#[cfg(not(feature = "no-cuda"))]
enum Bf16Operand<'a> {
    Borrowed(&'a CudaSlice<u16>),
    Quantized(CudaSlice<u16>),
}

#[cfg(not(feature = "no-cuda"))]
impl Bf16Operand<'_> {
    fn get(&self) -> &CudaSlice<u16> {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Quantized(slice) => slice,
        }
    }
}

pub struct CudaBackend {
    tape_dtype: AtomicU8,
    #[cfg(not(feature = "no-cuda"))]
    stream: Arc<CudaStream>,
    #[cfg(not(feature = "no-cuda"))]
    blas: Arc<CudaBlas>,
    #[cfg(not(feature = "no-cuda"))]
    kernels: KernelCache,
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

impl CudaBackend {
    /// Create a backend bound to the CUDA device at `ordinal`.
    ///
    /// # Errors
    /// Returns an error if the device cannot be opened, cuBLAS cannot be
    /// initialised, or the autograd CUDA kernels fail NVRTC compilation.
    pub fn new(ordinal: usize) -> Result<Self> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = ordinal;
            todo!("GPU required: CudaBackend::new is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            let ctx = CudaContext::new(ordinal).map_err(|_| {
                AutogradError::TapeInvariant("CudaContext::new failed (is a GPU present?)")
            })?;
            let stream = ctx.default_stream();
            let blas = CudaBlas::new(stream.clone())
                .map_err(|_| AutogradError::TapeInvariant("CudaBlas::new failed"))?;
            let kernels = KernelCache::new(stream.context())?;
            Ok(Self {
                tape_dtype: AtomicU8::new(TapeDtype::F32 as u8),
                stream,
                blas: Arc::new(blas),
                kernels,
                #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
                nccl: None,
                #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
                nccl_seq: None,
            })
        }
    }

    /// Create a CUDA backend with an NCCL all-reduce communicator attached.
    ///
    /// The communicator uses the same default stream/context as autograd CUDA
    /// kernels, so surrounding ops and collectives are naturally ordered.
    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    pub fn new_with_nccl(
        ordinal: usize,
        unique_id: cuda_kernels::ffi::nccl::ncclUniqueId,
        world_size: usize,
        rank: usize,
    ) -> Result<Self> {
        let mut backend = Self::new(ordinal)?;
        let nccl = NcclBackend::init_rank(unique_id, world_size, rank).map_err(|err| {
            AutogradError::TapeInvariant(Box::leak(
                format!("NcclBackend::init_rank failed for autograd: {err:#}").into_boxed_str(),
            ))
        })?;
        let nccl = Arc::new(nccl);
        // Default seq comm = world comm; a composed mesh overwrites it.
        backend.nccl_seq = Some(nccl.clone());
        backend.nccl = Some(nccl);
        Ok(backend)
    }

    /// Mesh backend: grad/count all-reduces on the world comm, seq collectives
    /// on a `ncclCommSplit` subgroup `seq_group = (color, size, rank)`. The
    /// caller derives the spec from the one mesh — this crate stays
    /// layout-agnostic. `None` = no subgroup (identical to `new_with_nccl`).
    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    pub fn new_with_mesh(
        ordinal: usize,
        unique_id: cuda_kernels::ffi::nccl::ncclUniqueId,
        world_size: usize,
        world_rank: usize,
        seq_group: Option<(usize, usize, usize)>,
    ) -> Result<Self> {
        let mut backend = Self::new_with_nccl(ordinal, unique_id, world_size, world_rank)?;
        if let Some((color, size, rank)) = seq_group {
            let world_comm = backend.nccl.as_ref().expect("new_with_nccl sets nccl");
            // Collective over the world comm — every rank constructs together.
            let seq = world_comm
                .split(color as i32, rank as i32, size, rank)
                .map_err(|err| {
                    AutogradError::TapeInvariant(Box::leak(
                        format!("ncclCommSplit for the seq subgroup failed: {err:#}")
                            .into_boxed_str(),
                    ))
                })?;
            backend.nccl_seq = Some(Arc::new(seq));
        }
        Ok(backend)
    }

    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    fn comm(&self, axis: CommAxis) -> Option<&Arc<NcclBackend>> {
        match axis {
            CommAxis::World | CommAxis::Expert => self.nccl.as_ref(),
            CommAxis::Seq => self.nccl_seq.as_ref(),
        }
    }

    /// No-GPU stub for no-cuda builds that still type-check the nccl feature.
    #[cfg(all(feature = "nccl", feature = "no-cuda"))]
    pub fn new_with_nccl(
        ordinal: usize,
        unique_id: cuda_kernels::ffi::nccl::ncclUniqueId,
        world_size: usize,
        rank: usize,
    ) -> Result<Self> {
        let _ = (ordinal, unique_id, world_size, rank);
        todo!("GPU required: CudaBackend::new_with_nccl is unavailable under feature no-cuda")
    }

    /// No-GPU stub for no-cuda builds that still type-check the nccl feature.
    #[cfg(all(feature = "nccl", feature = "no-cuda"))]
    pub fn new_with_mesh(
        ordinal: usize,
        unique_id: cuda_kernels::ffi::nccl::ncclUniqueId,
        world_size: usize,
        world_rank: usize,
        seq_group: Option<(usize, usize, usize)>,
    ) -> Result<Self> {
        let _ = (ordinal, unique_id, world_size, world_rank, seq_group);
        todo!("GPU required: CudaBackend::new_with_mesh is unavailable under feature no-cuda")
    }

    /// Query device VRAM `(free_bytes, total_bytes)` for the backend's CUDA
    /// context. Used by OPD attribution tooling to log per-phase peak VRAM
    /// without shelling out to `nvidia-smi`.
    ///
    /// # Errors
    /// Returns an error if the driver `cuMemGetInfo` call fails.
    #[cfg(not(feature = "no-cuda"))]
    pub fn mem_get_info(&self) -> Result<(usize, usize)> {
        self.stream
            .context()
            .mem_get_info()
            .map_err(|_| AutogradError::TapeInvariant("cuda mem_get_info failed"))
    }

    /// No-GPU stub — VRAM query is unavailable without a CUDA device.
    #[cfg(feature = "no-cuda")]
    pub fn mem_get_info(&self) -> Result<(usize, usize)> {
        todo!("GPU required: CudaBackend::mem_get_info is unavailable under feature no-cuda")
    }

    /// Whether an NCCL communicator is attached (multi-rank collectives work).
    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    pub fn has_collective(&self) -> bool {
        self.nccl.is_some()
    }

    /// Whether an NCCL communicator is attached (multi-rank collectives work).
    #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
    pub fn has_collective(&self) -> bool {
        false
    }

    #[cfg(not(feature = "no-cuda"))]
    #[track_caller]
    fn upload_slice(&self, host: &[f32], shape: &[usize]) -> Result<CudaSlice<f32>> {
        let size = shape_size(shape);
        if host.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: shape.to_vec(),
                size,
            });
        }

        if let Ok(raw_threshold) = std::env::var("ARLE_CUDA_UPLOAD_TRACE_MIN_ELEMS")
            && let Ok(threshold) = raw_threshold.parse::<usize>()
            && host.len() >= threshold
        {
            let caller = std::panic::Location::caller();
            eprintln!(
                "arle_cuda_upload_trace caller={}:{} shape={shape:?} len={} bytes={} backtrace={:?}",
                caller.file(),
                caller.line(),
                host.len(),
                host.len().saturating_mul(std::mem::size_of::<f32>()),
                std::backtrace::Backtrace::force_capture()
            );
        }

        self.stream.clone_htod(host).map_err(|err| {
            let bytes = host.len().saturating_mul(std::mem::size_of::<f32>());
            AutogradError::TapeInvariant(Box::leak(
                format!(
                    "cuda htod copy failed: shape={shape:?} len={} bytes={} err={err:?}",
                    host.len(),
                    bytes
                )
                .into_boxed_str(),
            ))
        })
    }

    #[cfg(not(feature = "no-cuda"))]
    fn cuda_storage_slice<'a>(&self, storage: &'a CudaStorage) -> Result<&'a CudaSlice<f32>> {
        let slice = storage.slice();
        // Reject handles that live on a different cudarc context/ordinal —
        // submitting foreign device pointers on our stream surfaces as
        // invalid-context driver errors. PENDING REMOTE CUDA VERIFICATION.
        if slice.context() != self.stream.context() {
            return Err(AutogradError::TapeInvariant(
                "cuda handle from different context/ordinal",
            ));
        }
        Ok(slice)
    }

    #[cfg(not(feature = "no-cuda"))]
    fn upload_bf16_bits_slice(&self, host: &[u16], shape: &[usize]) -> Result<CudaSlice<u16>> {
        let size = shape_size(shape);
        if host.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: shape.to_vec(),
                size,
            });
        }

        self.stream.clone_htod(host).map_err(|err| {
            let bytes = host.len().saturating_mul(std::mem::size_of::<u16>());
            AutogradError::TapeInvariant(Box::leak(
                format!(
                    "cuda bf16 htod copy failed: shape={shape:?} len={} bytes={} err={err:?}",
                    host.len(),
                    bytes
                )
                .into_boxed_str(),
            ))
        })
    }

    #[cfg(not(feature = "no-cuda"))]
    fn cuda_bf16_storage_slice<'a>(
        &self,
        storage: &'a CudaBf16Storage,
    ) -> Result<&'a CudaSlice<u16>> {
        let slice = storage.slice();
        if slice.context() != self.stream.context() {
            return Err(AutogradError::TapeInvariant(
                "cuda bf16 handle from different context/ordinal",
            ));
        }
        Ok(slice)
    }

    #[cfg(not(feature = "no-cuda"))]
    fn upload_fp8_bytes_slice(&self, host: &[u8], shape: &[usize]) -> Result<CudaSlice<u8>> {
        let size = shape_size(shape);
        if host.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: shape.to_vec(),
                size,
            });
        }

        self.stream.clone_htod(host).map_err(|err| {
            let bytes = host.len();
            AutogradError::TapeInvariant(Box::leak(
                format!(
                    "cuda fp8 htod copy failed: shape={shape:?} len={} bytes={} err={err:?}",
                    host.len(),
                    bytes
                )
                .into_boxed_str(),
            ))
        })
    }

    #[cfg(not(feature = "no-cuda"))]
    fn cuda_fp8_block_scaled_storage<'a>(
        &self,
        storage: &'a CudaFp8BlockScaledStorage,
    ) -> Result<Fp8BlockScaledView<'a>> {
        let weight = storage.weight();
        let scales = storage.scales();
        if weight.context() != self.stream.context() || scales.context() != self.stream.context() {
            return Err(AutogradError::TapeInvariant(
                "cuda fp8 block-scaled handle from different context/ordinal",
            ));
        }
        Ok((
            weight,
            scales,
            storage.rows(),
            storage.cols(),
            storage.block_m(),
            storage.block_k(),
        ))
    }

    #[cfg(not(feature = "no-cuda"))]
    fn copy_bf16_device_ptr_to_local(
        &self,
        src_device_ptr: u64,
        len: usize,
    ) -> Result<CudaSlice<u16>> {
        let mut staging = self
            .stream
            .alloc_zeros::<u16>(len)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (bf16 bridge)"))?;
        if len == 0 {
            return Ok(staging);
        }

        let byte_count =
            len.checked_mul(std::mem::size_of::<u16>())
                .ok_or(AutogradError::TapeInvariant(
                    "bf16 bridge byte count overflow",
                ))?;
        {
            let (dst_ptr, _dst_guard) = staging.device_ptr_mut(&self.stream);
            // SAFETY: dst spans `staging`'s len*2 = byte_count bytes (`_dst_guard` held); the
            // caller guarantees src_device_ptr covers byte_count bytes until the sync below.
            let status = unsafe {
                cuMemcpyDtoD_v2(
                    dst_ptr as CUdeviceptr,
                    src_device_ptr as CUdeviceptr,
                    byte_count,
                )
            };
            if status != CUresult::CUDA_SUCCESS {
                return Err(AutogradError::TapeInvariant(
                    "cuda D2D copy failed (bf16 bridge)",
                ));
            }
        };
        // `cuMemcpyDtoD_v2` issues on the per-thread default stream, which is
        // NOT host-blocking when the context disables event tracking (the OPD
        // infer/train contexts do — see `DeviceContext::on_device`). The source
        // pointer belongs to a *foreign* allocator (the infer engine's logits
        // buffer); its owner frees it via `cuMemFreeAsync` as soon as this
        // bridge returns. Without a fence the async free races the still-running
        // D2D read → use-after-free / CUDA_ERROR_ILLEGAL_ADDRESS at the next
        // sync (confirmed by compute-sanitizer: "Use-after-free ... accessed
        // after it is free'd" at cuMemcpyDtoD_v2 vs cuMemFreeAsync, and the
        // fault vanishes under CUDA_LAUNCH_BLOCKING=1). Drain the copy before
        // the source can be freed.
        self.stream
            .context()
            .synchronize()
            .map_err(|_| AutogradError::TapeInvariant("cuda D2D bridge sync failed"))?;
        Ok(staging)
    }

    /// First `len` elements of `buf` (identity when unpadded — no copy).
    #[cfg(not(feature = "no-cuda"))]
    fn f32_prefix(&self, buf: CudaSlice<f32>, len: usize) -> Result<CudaSlice<f32>> {
        if buf.len() == len {
            return Ok(buf);
        }
        let mut out = self
            .stream
            .alloc_zeros::<f32>(len)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (f32 prefix)"))?;
        self.stream
            .memcpy_dtod(&buf.slice(0..len), &mut out)
            .map_err(|_| AutogradError::TapeInvariant("cuda f32 prefix copy failed"))?;
        Ok(out)
    }

    #[cfg(not(feature = "no-cuda"))]
    fn import_local_bf16_as_f32(
        &self,
        staging: &CudaSlice<u16>,
        len: usize,
    ) -> Result<CudaSlice<f32>> {
        let mut out = self
            .stream
            .alloc_zeros::<f32>(len)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (bf16 import)"))?;
        if len == 0 {
            return Ok(out);
        }

        let n_u64 = len as u64;
        {
            let (src_ptr, _src_guard) = staging.device_ptr(&self.stream);
            let (dst_ptr, _dst_guard) = out.device_ptr_mut(&self.stream);
            launch_1d(
                &self.stream,
                self.kernels.function("bf16_bits_to_f32")?,
                len,
                |mut builder| {
                    builder.arg(&src_ptr).arg(&dst_ptr).arg(&n_u64);
                    builder
                },
            )?;
        }
        Ok(out)
    }

    #[cfg(not(feature = "no-cuda"))]
    fn local_f32_as_bf16(&self, src: &CudaSlice<f32>, len: usize) -> Result<CudaSlice<u16>> {
        let mut out = self
            .stream
            .alloc_zeros::<u16>(len)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (f32->bf16)"))?;
        if len == 0 {
            return Ok(out);
        }

        let n_u64 = len as u64;
        {
            let (src_ptr, _src_guard) = src.device_ptr(&self.stream);
            let (dst_ptr, _dst_guard) = out.device_ptr_mut(&self.stream);
            launch_1d(
                &self.stream,
                self.kernels.function("f32_to_bf16_bits")?,
                len,
                |mut builder| {
                    builder.arg(&src_ptr).arg(&dst_ptr).arg(&n_u64);
                    builder
                },
            )?;
        }
        Ok(out)
    }

    #[cfg(not(feature = "no-cuda"))]
    fn cublaslt_bf16_gemm_n(rows: usize) -> usize {
        if rows == 0 {
            0
        } else {
            rows.max(CUBLASLT_BF16_GEMMEX_MIN_N)
        }
    }

    #[cfg(not(feature = "no-cuda"))]
    fn checked_bf16_len(rows: usize, cols: usize, op: &'static str) -> Result<usize> {
        rows.checked_mul(cols)
            .ok_or(AutogradError::TapeInvariant(op))
    }

    #[cfg(not(feature = "no-cuda"))]
    fn maybe_pad_bf16_gemm_n(
        &self,
        src: &CudaSlice<u16>,
        rows: usize,
        cols: usize,
        padded_rows: usize,
        op: &'static str,
    ) -> Result<Option<CudaSlice<u16>>> {
        if padded_rows == rows {
            return Ok(None);
        }
        let padded_len = Self::checked_bf16_len(padded_rows, cols, op)?;
        let mut padded = self
            .stream
            .alloc_zeros::<u16>(padded_len)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (bf16 pad)"))?;
        self.stream
            .memcpy_dtod(src, &mut padded)
            .map_err(|_| AutogradError::TapeInvariant("cuda D2D copy failed (bf16 pad)"))?;
        Ok(Some(padded))
    }

    #[cfg(not(feature = "no-cuda"))]
    fn cuda_slice<'a>(
        &self,
        handle: &'a DeviceHandle,
        op: &'static str,
    ) -> Result<&'a CudaSlice<f32>> {
        match handle {
            DeviceHandle::Cuda(storage) => self.cuda_storage_slice(storage),
            DeviceHandle::CudaBf16(_) => Err(AutogradError::TapeInvariant(match op {
                "matmul" => "cuda backend cannot matmul a bf16 handle on this f32-only path",
                "matmul_bt" => "cuda backend cannot use bf16 handle as lhs on this matmul_bt path",
                "embedding_from_f32_ids" => "cuda backend cannot use bf16 handle for f32 token ids",
                _ => "cuda backend cannot operate on a bf16 handle on this f32-only path",
            })),
            DeviceHandle::CudaFp8BlockScaled(_) => Err(AutogradError::TapeInvariant(match op {
                "matmul" => {
                    "cuda backend cannot matmul a fp8 block-scaled handle on this f32-only path"
                }
                "matmul_bt" => {
                    "cuda backend cannot use fp8 block-scaled handle as lhs on this matmul_bt path"
                }
                _ => {
                    "cuda backend cannot operate on a fp8 block-scaled handle on this f32-only path"
                }
            })),
            DeviceHandle::Cpu(_) => Err(AutogradError::TapeInvariant(match op {
                "add" => "cuda backend cannot add a cpu device handle",
                "matmul" => "cuda backend cannot matmul a cpu device handle",
                _ => "cuda backend cannot operate on a cpu device handle",
            })),
            #[cfg(feature = "metal")]
            DeviceHandle::Metal(_) => Err(AutogradError::TapeInvariant(match op {
                "add" => "cuda backend cannot add a metal device handle",
                "matmul" => "cuda backend cannot matmul a metal device handle",
                _ => "cuda backend cannot operate on a metal device handle",
            })),
        }
    }

    #[cfg(not(feature = "no-cuda"))]
    fn cuda_bf16_slice<'a>(
        &self,
        handle: &'a DeviceHandle,
        op: &'static str,
    ) -> Result<&'a CudaSlice<u16>> {
        match handle {
            DeviceHandle::CudaBf16(storage) => self.cuda_bf16_storage_slice(storage),
            _ => Err(AutogradError::TapeInvariant(Box::leak(
                format!("cuda backend expected bf16 handle for {op}").into_boxed_str(),
            ))),
        }
    }

    #[cfg(not(feature = "no-cuda"))]
    fn tape_bf16(&self) -> bool {
        self.tape_dtype.load(Ordering::Relaxed) == TapeDtype::Bf16 as u8
    }

    /// f32 view of a handle: borrows f32 storage, imports (exact widen) bf16.
    #[cfg(not(feature = "no-cuda"))]
    fn f32_operand<'a>(
        &self,
        handle: &'a DeviceHandle,
        op: &'static str,
    ) -> Result<F32Operand<'a>> {
        match handle {
            DeviceHandle::CudaBf16(storage) => {
                let bits = self.cuda_bf16_storage_slice(storage)?;
                Ok(F32Operand::Imported(
                    self.import_local_bf16_as_f32(bits, bits.len())?,
                ))
            }
            _ => Ok(F32Operand::Borrowed(self.cuda_slice(handle, op)?)),
        }
    }

    /// Like `f32_operand`, but under bf16 tape dtype an f32 handle is
    /// round-tripped through bf16 so backward reads the value forward saw.
    #[cfg(not(feature = "no-cuda"))]
    fn f32_operand_tape_quantized<'a>(
        &self,
        handle: &'a DeviceHandle,
        op: &'static str,
    ) -> Result<F32Operand<'a>> {
        match handle {
            DeviceHandle::Cuda(_) if self.tape_bf16() => {
                let src = self.cuda_slice(handle, op)?;
                let bits = self.local_f32_as_bf16(src, src.len())?;
                Ok(F32Operand::Imported(
                    self.import_local_bf16_as_f32(&bits, bits.len())?,
                ))
            }
            _ => self.f32_operand(handle, op),
        }
    }

    /// bf16 view of a handle: borrows bf16 storage, quantizes f32.
    #[cfg(not(feature = "no-cuda"))]
    fn bf16_operand<'a>(
        &self,
        handle: &'a DeviceHandle,
        op: &'static str,
    ) -> Result<Bf16Operand<'a>> {
        match handle {
            DeviceHandle::CudaBf16(storage) => Ok(Bf16Operand::Borrowed(
                self.cuda_bf16_storage_slice(storage)?,
            )),
            _ => {
                let src = self.cuda_slice(handle, op)?;
                Ok(Bf16Operand::Quantized(
                    self.local_f32_as_bf16(src, src.len())?,
                ))
            }
        }
    }

    #[cfg(not(feature = "no-cuda"))]
    fn validate_cuda_handle_kind(&self, handle: &DeviceHandle) -> Result<()> {
        match handle {
            DeviceHandle::Cpu(_)
            | DeviceHandle::Cuda(_)
            | DeviceHandle::CudaBf16(_)
            | DeviceHandle::CudaFp8BlockScaled(_) => Ok(()),
            #[cfg(feature = "metal")]
            DeviceHandle::Metal(_) => Err(AutogradError::TapeInvariant(
                "cuda backend cannot evaluate a metal device handle",
            )),
        }
    }

    #[cfg(not(feature = "no-cuda"))]
    fn matmul_device(
        &self,
        a: &CudaSlice<f32>,
        a_shape: &[usize],
        b: &CudaSlice<f32>,
        b_shape: &[usize],
    ) -> Result<(CudaSlice<f32>, Vec<usize>)> {
        if a.len() != shape_size(a_shape) || b.len() != shape_size(b_shape) {
            return Err(AutogradError::TapeInvariant(
                "cuda backend matmul handle size does not match shape",
            ));
        }

        let out_shape = matmul_output_shape(a_shape, b_shape)?;
        match (a_shape.len(), b_shape.len()) {
            (2, 2) => {
                let m = a_shape[0];
                let k = a_shape[1];
                let n = b_shape[1];
                let mut c = self
                    .stream
                    .alloc_zeros::<f32>(m * n)
                    .map_err(|_| cuda_alloc_failed("matmul", vec![m, n]))?;

                let cfg = GemmConfig::<f32> {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: m as i32,
                    k: k as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: k as i32,
                    beta: 0.0,
                    ldc: n as i32,
                };

                // Safety: shapes validated above; device buffers outlive the call.
                unsafe {
                    self.blas
                        .gemm(cfg, b, a, &mut c)
                        .map_err(|_| AutogradError::TapeInvariant("cuBLAS sgemm failed"))?;
                }
                Ok((c, out_shape))
            }
            (3, 3) => {
                let batch = a_shape[0];
                let m = a_shape[1];
                let k = a_shape[2];
                let n = b_shape[2];
                let mut c = self
                    .stream
                    .alloc_zeros::<f32>(batch * m * n)
                    .map_err(|_| cuda_alloc_failed("matmul_batched", vec![batch, m, n]))?;

                let gemm = GemmConfig::<f32> {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: m as i32,
                    k: k as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: k as i32,
                    beta: 0.0,
                    ldc: n as i32,
                };
                let cfg = StridedBatchedConfig::<f32> {
                    gemm,
                    batch_size: batch as i32,
                    stride_a: (k * n) as i64,
                    stride_b: (m * k) as i64,
                    stride_c: (m * n) as i64,
                };

                // Safety: shapes validated above; device buffers outlive the call.
                unsafe {
                    self.blas
                        .gemm_strided_batched(cfg, b, a, &mut c)
                        .map_err(|_| {
                            AutogradError::TapeInvariant("cuBLAS sgemm_strided_batched failed")
                        })?;
                }
                Ok((c, out_shape))
            }
            _ => Err(AutogradError::InvalidRank {
                expected: "both operands must be rank-2 or rank-3",
                got: a_shape.len().max(b_shape.len()),
            }),
        }
    }

    #[cfg(not(feature = "no-cuda"))]
    fn matmul_bt_device_f32_bf16(
        &self,
        a: &CudaSlice<f32>,
        a_shape: &[usize],
        b: &CudaSlice<u16>,
        b_shape: &[usize],
    ) -> Result<(CudaSlice<f32>, Vec<usize>)> {
        if a.len() != shape_size(a_shape) || b.len() != shape_size(b_shape) {
            return Err(AutogradError::TapeInvariant(
                "cuda backend bf16 matmul_bt handle size does not match shape",
            ));
        }

        let out_shape = matmul_bt_output_shape(a_shape, b_shape)?;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: "both operands must be rank-2",
                got: a_shape.len().max(b_shape.len()),
            });
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[0];
        let a_bf16 = self.local_f32_as_bf16(a, a.len())?;
        if m == 0 || n == 0 || k == 0 {
            let c = self
                .stream
                .alloc_zeros::<f32>(m * n)
                .map_err(|_| cuda_alloc_failed("matmul_bt_bf16_empty", vec![m, n]))?;
            return Ok((c, out_shape));
        }

        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let k_i32 = i32::try_from(k)
            .map_err(|_| AutogradError::TapeInvariant("bf16 matmul_bt K exceeds i32"))?;
        let n_i32 = i32::try_from(n)
            .map_err(|_| AutogradError::TapeInvariant("bf16 matmul_bt N exceeds i32"))?;
        let padded_m = Self::cublaslt_bf16_gemm_n(m);
        let padded_m_i32 = i32::try_from(padded_m)
            .map_err(|_| AutogradError::TapeInvariant("bf16 matmul_bt padded M exceeds i32"))?;
        let padded_a = self.maybe_pad_bf16_gemm_n(
            &a_bf16,
            m,
            k,
            padded_m,
            "bf16 matmul_bt padded lhs length overflow",
        )?;
        let a_for_gemm = padded_a.as_ref().unwrap_or(&a_bf16);
        let c_len =
            Self::checked_bf16_len(padded_m, n, "bf16 matmul_bt padded output length overflow")?;
        let mut c_out = self
            .stream
            .alloc_zeros::<f32>(c_len)
            .map_err(|_| cuda_alloc_failed("matmul_bt_bf16", vec![padded_m, n]))?;
        {
            let (b_ptr, _b_guard) = b.device_ptr(&self.stream);
            let (a_ptr, _a_guard) = a_for_gemm.device_ptr(&self.stream);
            let (c_ptr, _c_guard) = c_out.device_ptr_mut(&self.stream);

            // Same row-major cuBLAS trick as the f32 path: swap operands so the
            // column-major output view is the row-major [M, N] buffer. Operand B
            // is stored as BF16; the activation is rounded to BF16 on-device,
            // accumulated in FP32, and converted back to FP32 for downstream
            // autograd ops. CUDA 12.9 cuBLASLt can SIGFPE inside
            // AlgoGetHeuristic for BF16 large-M skinny-N shapes such as
            // lm_head [vocab, hidden] x [hidden, 8]. Padding the cuBLAS N
            // dimension with zero activation rows avoids that heuristic bug;
            // only the real row prefix is returned.
            // SAFETY: b/a/c derive from live guarded slices — b [n,k] validated at entry, a
            // padded to [padded_m,k], c allocated [padded_m,n] — matching the dims passed.
            unsafe {
                cublas_result::gemm_ex(
                    *self.blas.handle(),
                    cublasOperation_t::CUBLAS_OP_T,
                    cublasOperation_t::CUBLAS_OP_N,
                    n_i32,
                    padded_m_i32,
                    k_i32,
                    (&alpha) as *const f32 as *const _,
                    b_ptr as *const _,
                    cublas_sys::cudaDataType_t::CUDA_R_16BF,
                    k_i32,
                    a_ptr as *const _,
                    cublas_sys::cudaDataType_t::CUDA_R_16BF,
                    k_i32,
                    (&beta) as *const f32 as *const _,
                    c_ptr as *mut _,
                    cublas_sys::cudaDataType_t::CUDA_R_32F,
                    n_i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .map_err(|_| {
                    AutogradError::TapeInvariant("cuBLAS gemm_ex failed (bf16 matmul_bt)")
                })?;
            }
        }

        let c = self.f32_prefix(c_out, m * n)?;
        Ok((c, out_shape))
    }

    #[cfg(not(feature = "no-cuda"))]
    fn matmul_device_f32_bf16(
        &self,
        a: &CudaSlice<f32>,
        a_shape: &[usize],
        b: &CudaSlice<u16>,
        b_shape: &[usize],
    ) -> Result<(CudaSlice<f32>, Vec<usize>)> {
        if a.len() != shape_size(a_shape) || b.len() != shape_size(b_shape) {
            return Err(AutogradError::TapeInvariant(
                "cuda backend bf16 matmul handle size does not match shape",
            ));
        }

        let out_shape = matmul_output_shape(a_shape, b_shape)?;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: "both operands must be rank-2",
                got: a_shape.len().max(b_shape.len()),
            });
        }

        let m = a_shape[0];
        let n = a_shape[1];
        let k = b_shape[1];
        let a_bf16 = self.local_f32_as_bf16(a, a.len())?;
        if m == 0 || n == 0 || k == 0 {
            let c = self
                .stream
                .alloc_zeros::<f32>(m * k)
                .map_err(|_| cuda_alloc_failed("matmul_bf16_empty", vec![m, k]))?;
            return Ok((c, out_shape));
        }

        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let n_i32 = i32::try_from(n)
            .map_err(|_| AutogradError::TapeInvariant("bf16 matmul N exceeds i32"))?;
        let k_i32 = i32::try_from(k)
            .map_err(|_| AutogradError::TapeInvariant("bf16 matmul K exceeds i32"))?;
        let padded_m = Self::cublaslt_bf16_gemm_n(m);
        let padded_m_i32 = i32::try_from(padded_m)
            .map_err(|_| AutogradError::TapeInvariant("bf16 matmul padded M exceeds i32"))?;
        let padded_a = self.maybe_pad_bf16_gemm_n(
            &a_bf16,
            m,
            n,
            padded_m,
            "bf16 matmul padded lhs length overflow",
        )?;
        let a_for_gemm = padded_a.as_ref().unwrap_or(&a_bf16);
        let c_len =
            Self::checked_bf16_len(padded_m, k, "bf16 matmul padded output length overflow")?;
        let mut c_out = self
            .stream
            .alloc_zeros::<f32>(c_len)
            .map_err(|_| cuda_alloc_failed("matmul_bf16", vec![padded_m, k]))?;
        {
            let (b_ptr, _b_guard) = b.device_ptr(&self.stream);
            let (a_ptr, _a_guard) = a_for_gemm.device_ptr(&self.stream);
            let (c_ptr, _c_guard) = c_out.device_ptr_mut(&self.stream);

            // Row-major C[M,K] = A[M,N] @ B[N,K], using cuBLAS's column-major
            // view as C_col[K,M] = B_col[K,N] @ A_col[N,M]. See
            // `matmul_bt_device_f32_bf16` for the skinny-N padding rationale.
            // SAFETY: b/a/c derive from live guarded slices — b [n,k] validated at entry, a
            // padded to [padded_m,n], c allocated [padded_m,k] — matching the dims passed.
            unsafe {
                cublas_result::gemm_ex(
                    *self.blas.handle(),
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    k_i32,
                    padded_m_i32,
                    n_i32,
                    (&alpha) as *const f32 as *const _,
                    b_ptr as *const _,
                    cublas_sys::cudaDataType_t::CUDA_R_16BF,
                    k_i32,
                    a_ptr as *const _,
                    cublas_sys::cudaDataType_t::CUDA_R_16BF,
                    n_i32,
                    (&beta) as *const f32 as *const _,
                    c_ptr as *mut _,
                    cublas_sys::cudaDataType_t::CUDA_R_32F,
                    k_i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .map_err(|_| AutogradError::TapeInvariant("cuBLAS gemm_ex failed (bf16 matmul)"))?;
            }
        }

        let c = self.f32_prefix(c_out, m * k)?;
        Ok((c, out_shape))
    }

    /// Dequantize an FP8 block-scaled weight to a BF16 device buffer (returns
    /// the buffer + its `[rows, cols]` shape). One memory-bound elementwise
    /// launch; GEMMs then ride the tensor-core cuBLAS BF16 path instead of the
    /// naive per-output-element FP8 kernel this replaced (~290× on 27B OPD).
    #[cfg(not(feature = "no-cuda"))]
    fn fp8_block_scaled_as_bf16(
        &self,
        storage: &CudaFp8BlockScaledStorage,
    ) -> Result<(CudaSlice<u16>, Vec<usize>)> {
        let (weight, scales, rows, cols, block_m, block_k) =
            self.cuda_fp8_block_scaled_storage(storage)?;
        let total = rows * cols;
        let scale_cols = cols.div_ceil(block_k);
        if weight.len() != total || scales.len() != rows.div_ceil(block_m) * scale_cols {
            return Err(AutogradError::TapeInvariant(
                "cuda backend fp8 dequant handle size does not match shape",
            ));
        }
        let mut out = self.stream.alloc_zeros::<u16>(total).map_err(|e| {
            // Surface the real driver error: alloc failure here is either true
            // OOM or a prior async fault turning sticky — indistinguishable
            // without the code (smoke 2026-07-03 hit both attribution paths).
            eprintln!("[autograd] alloc_zeros {total} x u16 failed (fp8 dequant): {e}");
            AutogradError::TapeInvariant("cuda alloc_zeros failed (fp8 dequant)")
        })?;
        let total_i32 = i32::try_from(total)
            .map_err(|_| AutogradError::TapeInvariant("fp8 dequant total exceeds i32"))?;
        let cols_i32 = i32::try_from(cols)
            .map_err(|_| AutogradError::TapeInvariant("fp8 dequant cols exceeds i32"))?;
        let block_m_i32 = i32::try_from(block_m)
            .map_err(|_| AutogradError::TapeInvariant("fp8 dequant block_m exceeds i32"))?;
        let block_k_i32 = i32::try_from(block_k)
            .map_err(|_| AutogradError::TapeInvariant("fp8 dequant block_k exceeds i32"))?;
        let scale_cols_i32 = i32::try_from(scale_cols)
            .map_err(|_| AutogradError::TapeInvariant("fp8 dequant scale_cols exceeds i32"))?;
        launch_1d(
            &self.stream,
            self.kernels.function("fp8_block_scaled_to_bf16")?,
            total,
            |mut builder| {
                builder
                    .arg(weight)
                    .arg(scales)
                    .arg(&mut out)
                    .arg(&total_i32)
                    .arg(&cols_i32)
                    .arg(&block_m_i32)
                    .arg(&block_k_i32)
                    .arg(&scale_cols_i32);
                builder
            },
        )?;
        Ok((out, vec![rows, cols]))
    }

    #[cfg(not(feature = "no-cuda"))]
    fn matmul_bt_device_f32_fp8_block_scaled(
        &self,
        a: &CudaSlice<f32>,
        a_shape: &[usize],
        storage: &CudaFp8BlockScaledStorage,
    ) -> Result<(CudaSlice<f32>, Vec<usize>)> {
        let (b_bf16, b_shape) = self.fp8_block_scaled_as_bf16(storage)?;
        self.matmul_bt_device_f32_bf16(a, a_shape, &b_bf16, &b_shape)
    }

    #[cfg(not(feature = "no-cuda"))]
    fn matmul_device_f32_fp8_block_scaled(
        &self,
        a: &CudaSlice<f32>,
        a_shape: &[usize],
        storage: &CudaFp8BlockScaledStorage,
    ) -> Result<(CudaSlice<f32>, Vec<usize>)> {
        let (b_bf16, b_shape) = self.fp8_block_scaled_as_bf16(storage)?;
        self.matmul_device_f32_bf16(a, a_shape, &b_bf16, &b_shape)
    }
}

impl Backend for CudaBackend {
    fn device(&self) -> Device {
        Device::Cuda
    }

    fn set_tape_dtype(&self, dtype: TapeDtype) {
        self.tape_dtype.store(dtype as u8, Ordering::Relaxed);
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
            // Idempotent: non-f32-Cuda handles pass through unchanged.
            let DeviceHandle::Cuda(_) = handle else {
                return Ok(handle.clone());
            };
            let src = self.cuda_slice(handle, "quantize_frozen_to_bf16")?;
            let bits = self.local_f32_as_bf16(src, shape_size(shape))?;
            Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(bits)))
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
            validate_fp8_block_scaled(weight, scales, shape, block_m, block_k)?;
            Ok(DeviceHandle::CudaFp8BlockScaled(
                CudaFp8BlockScaledStorage::new(
                    self.upload_fp8_bytes_slice(weight, shape)?,
                    self.upload_slice(scales, &[scales.len()])?,
                    shape[0],
                    shape[1],
                    block_m,
                    block_k,
                ),
            ))
        }
    }

    fn import_bf16_device_ptr_as_f32(
        &self,
        src_device_ptr: u64,
        len: usize,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (src_device_ptr, len, shape);
            todo!("GPU required: cuda bf16 device import is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            if shape_size(shape) != len {
                return Err(AutogradError::DataLengthMismatch {
                    len,
                    shape: shape.to_vec(),
                    size: shape_size(shape),
                });
            }
            let staging = self.copy_bf16_device_ptr_to_local(src_device_ptr, len)?;
            let f32_slice = self.import_local_bf16_as_f32(&staging, len)?;
            Ok(DeviceHandle::Cuda(CudaStorage::new(f32_slice)))
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
            if shape.len() != 2 {
                return Err(AutogradError::InvalidRank {
                    expected: "2",
                    got: shape.len(),
                });
            }
            if block_m == 0 || block_k == 0 {
                return Err(AutogradError::TapeInvariant(
                    "fp8 block-scaled import block_m/block_k must be non-zero",
                ));
            }
            let rows = shape[0];
            let cols = shape[1];
            let weight_len = rows.checked_mul(cols).ok_or(AutogradError::TapeInvariant(
                "fp8 import weight len overflow",
            ))?;
            let scale_len = rows
                .div_ceil(block_m)
                .checked_mul(cols.div_ceil(block_k))
                .ok_or(AutogradError::TapeInvariant(
                    "fp8 import scale len overflow",
                ))?;

            // NON-OWNING view over the foreign engine's resident FP8 base bytes.
            // `upgrade_device_ptr` wraps the raw `CUdeviceptr` in a `CudaSlice`
            // bound to THIS backend's stream/primary context — so the
            // context-equality guard in `cuda_fp8_block_scaled_storage`
            // (`weight.context() == self.stream.context()`) passes (both engines
            // retain the same device primary context on the same ordinal). The
            // `CudaFp8BlockScaledStorage::new_borrowed` handle leaks these slices
            // on drop instead of freeing — the infer engine owns the bytes.
            //
            // Safety: the caller (`--share-frozen-base` loader wiring) guarantees
            // `weight_device_ptr` / `scale_device_ptr` are valid resident
            // allocations on this ordinal of at least `weight_len` u8 /
            // `scale_len` f32 elements, kept resident for the handle's lifetime
            // (the OPD Phase-B offload skips the shared base when sharing).
            let weight_slice: CudaSlice<u8> = unsafe {
                self.stream
                    .upgrade_device_ptr::<u8>(weight_device_ptr as CUdeviceptr, weight_len)
            };
            // SAFETY: same caller contract as the weight slice — scale_device_ptr is resident
            // for scale_len f32s and never freed by the borrowed handle.
            let scale_slice: CudaSlice<f32> = unsafe {
                self.stream
                    .upgrade_device_ptr::<f32>(scale_device_ptr as CUdeviceptr, scale_len)
            };
            Ok(DeviceHandle::CudaFp8BlockScaled(
                crate::backend::CudaFp8BlockScaledStorage::new_borrowed(
                    weight_slice,
                    scale_slice,
                    rows,
                    cols,
                    block_m,
                    block_k,
                ),
            ))
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
            let slice = self
                .stream
                .alloc_zeros::<f32>(size)
                .map_err(|_| cuda_alloc_failed("zeros", shape.to_vec()))?;
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
            match handle {
                DeviceHandle::Cpu(data) => Ok(data.clone()),
                DeviceHandle::Cuda(storage) => {
                    let slice = self.cuda_storage_slice(storage)?;
                    let mut host = vec![0.0f32; slice.len()];
                    self.stream
                        .memcpy_dtoh(slice, &mut host)
                        .map_err(|_| AutogradError::TapeInvariant("cuda dtoh copy failed"))?;
                    // cudarc 0.18 routes memcpy_dtoh through cuMemcpyDtoHAsync_v2
                    // (async DMA); callers do not always eval() first, so this
                    // single host fence is required. PENDING REMOTE CUDA VERIFICATION.
                    self.stream
                        .synchronize()
                        .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))?;
                    Ok(host)
                }
                DeviceHandle::CudaBf16(storage) => {
                    let slice = self.cuda_bf16_storage_slice(storage)?;
                    let mut host = vec![0u16; slice.len()];
                    self.stream
                        .memcpy_dtoh(slice, &mut host)
                        .map_err(|_| AutogradError::TapeInvariant("cuda bf16 dtoh copy failed"))?;
                    self.stream
                        .synchronize()
                        .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))?;
                    Ok(host
                        .into_iter()
                        .map(crate::backend::bf16_bits_to_f32)
                        .collect())
                }
                DeviceHandle::CudaFp8BlockScaled(storage) => {
                    let (weight, scales, rows, cols, block_m, block_k) =
                        self.cuda_fp8_block_scaled_storage(storage)?;
                    let mut host_weight = vec![0u8; weight.len()];
                    let mut host_scales = vec![0.0f32; scales.len()];
                    self.stream
                        .memcpy_dtoh(weight, &mut host_weight)
                        .map_err(|_| AutogradError::TapeInvariant("cuda fp8 dtoh copy failed"))?;
                    self.stream
                        .memcpy_dtoh(scales, &mut host_scales)
                        .map_err(|_| {
                            AutogradError::TapeInvariant("cuda fp8 scales dtoh copy failed")
                        })?;
                    self.stream
                        .synchronize()
                        .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))?;
                    dequantize_fp8_block_scaled_host(
                        &host_weight,
                        &host_scales,
                        &[rows, cols],
                        block_m,
                        block_k,
                    )
                }
                #[cfg(feature = "metal")]
                DeviceHandle::Metal(_) => Err(AutogradError::TapeInvariant(
                    "cuda backend cannot read back a metal device handle",
                )),
            }
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
            match handle {
                DeviceHandle::Cpu(data) => {
                    if data.len() != dst.len() {
                        return Err(AutogradError::DataLengthMismatch {
                            len: data.len(),
                            shape: vec![dst.len()],
                            size: dst.len(),
                        });
                    }
                    dst.copy_from_slice(data);
                    Ok(())
                }
                DeviceHandle::Cuda(storage) => {
                    let slice = self.cuda_storage_slice(storage)?;
                    if slice.len() != dst.len() {
                        return Err(AutogradError::DataLengthMismatch {
                            len: slice.len(),
                            shape: vec![dst.len()],
                            size: dst.len(),
                        });
                    }
                    self.stream.memcpy_dtoh(slice, dst).map_err(|_| {
                        AutogradError::TapeInvariant("cuda dtoh copy failed (readback_into)")
                    })?;
                    self.stream.synchronize().map_err(|_| {
                        AutogradError::TapeInvariant("cuda synchronize failed (readback_into)")
                    })
                }
                _ => {
                    let src = self.readback(handle)?;
                    if src.len() != dst.len() {
                        return Err(AutogradError::DataLengthMismatch {
                            len: src.len(),
                            shape: vec![dst.len()],
                            size: dst.len(),
                        });
                    }
                    dst.copy_from_slice(&src);
                    Ok(())
                }
            }
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
            let b = self.cuda_slice(b, "matmul")?;
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
            let out_shape = matmul_bt_output_shape(a_shape, b_shape)?;
            let d_a_op = self.f32_operand(a, "matmul_bt")?;
            let d_a = d_a_op.get();
            if let DeviceHandle::CudaBf16(storage) = b {
                let d_b = self.cuda_bf16_storage_slice(storage)?;
                if d_a.len() != shape_size(a_shape) || d_b.len() != shape_size(b_shape) {
                    return Err(AutogradError::TapeInvariant(
                        "cuda backend bf16 matmul_bt handle size does not match shape",
                    ));
                }
                let (c, out_shape) = self.matmul_bt_device_f32_bf16(d_a, a_shape, d_b, b_shape)?;
                return Ok((DeviceHandle::Cuda(CudaStorage::new(c)), out_shape));
            }
            if let DeviceHandle::CudaFp8BlockScaled(storage) = b {
                let (weight, _, rows, cols, _, _) = self.cuda_fp8_block_scaled_storage(storage)?;
                if b_shape != [rows, cols] {
                    return Err(AutogradError::ShapeMismatch {
                        expected: vec![rows, cols],
                        got: b_shape.to_vec(),
                    });
                }
                if d_a.len() != shape_size(a_shape) || weight.len() != shape_size(b_shape) {
                    return Err(AutogradError::TapeInvariant(
                        "cuda backend fp8 matmul_bt handle size does not match shape",
                    ));
                }
                let (c, out_shape) =
                    self.matmul_bt_device_f32_fp8_block_scaled(d_a, a_shape, storage)?;
                return Ok((DeviceHandle::Cuda(CudaStorage::new(c)), out_shape));
            }

            let d_b = self.cuda_slice(b, "matmul_bt")?;
            if d_a.len() != shape_size(a_shape) || d_b.len() != shape_size(b_shape) {
                return Err(AutogradError::TapeInvariant(
                    "cuda backend matmul_bt handle size does not match shape",
                ));
            }
            let m = a_shape[0];
            let k = a_shape[1];
            let n = b_shape[0];
            let mut c = self
                .stream
                .alloc_zeros::<f32>(m * n)
                .map_err(|_| cuda_alloc_failed("matmul_bt", vec![m, n]))?;

            let cfg = GemmConfig::<f32> {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                lda: k as i32,
                ldb: k as i32,
                beta: 0.0,
                ldc: n as i32,
            };

            // Safety: shapes validated above; device buffers outlive the call.
            unsafe {
                self.blas
                    .gemm(cfg, d_b, d_a, &mut c)
                    .map_err(|_| AutogradError::TapeInvariant("cuBLAS sgemm failed (matmul_bt)"))?;
            }
            Ok((DeviceHandle::Cuda(CudaStorage::new(c)), out_shape))
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

        // Device-resident reduction. The host-reduce alternative downloads the whole
        // `x` (e.g. the `[seq, vocab]` KL-loss intermediate, ~32 MB/chunk at
        // vocab=248320) to host, sums single-threaded, and re-uploads the
        // scalar — a full-tensor DtoH + blocking `synchronize()` per `mean`
        // in the OPD CE/KL head (`log_softmax → mul → mean`, see
        // `ops::reduce::mean_device_lazy`). That serialized the GPU behind a
        // CPU reduce every chunk and was the host-bound bottleneck.
        //
        // Now: a `sum_partial_f32` block reduce produces one f32 per block,
        // then the partials are recursively reduced on-device until a single
        // scalar remains. The result handle never leaves the GPU; the only
        // host transfer is the final 4-byte loss scalar in `tape.backward`'s
        // `ensure_host`. No `synchronize()` — the caller's terminal eval owns
        // it, so this composes into the existing device-resident chain.
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

    /// Device-resident matmul backward for the
    /// device-resident gradient tape. Mirrors the cuBLAS dispatch of the
    /// host-buffer `matmul_backward` (`grad_a = dC @ B^T`,
    /// `grad_b = A^T @ dC` via two SGEMMs with `OP_T` on the transposed
    /// operand) but consumes existing device handles and returns
    /// unevaluated `CudaSlice<f32>` outputs — no host roundtrip on either
    /// side. The terminal `backend.eval(...)` in `AdamW::step_device`
    /// performs the single host fence per training step (batched-eval
    /// contract).
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

    /// Device-resident backward for `C = A @ B^T` where A:[M,K], B:[N,K].
    /// Uses `grad_a = dC @ B` through the existing row-major matmul helper
    /// and `grad_b = dC^T @ A` via one cuBLAS SGEMM with OP_T on dC.
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
            let len = shape_size(shape);
            let src = self.cuda_slice(x, "all_reduce_sum")?;
            if src.len() != len {
                return Err(AutogradError::DataLengthMismatch {
                    len: src.len(),
                    shape: shape.to_vec(),
                    size: len,
                });
            }
            let mut out = self
                .stream
                .alloc_zeros::<f32>(len)
                .map_err(|_| AutogradError::TapeInvariant("cuda all_reduce alloc failed"))?;
            self.stream
                .memcpy_dtod(src, &mut out)
                .map_err(|_| AutogradError::TapeInvariant("cuda all_reduce D2D copy failed"))?;

            #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
            if let Some(nccl) = self.comm(axis) {
                let (dst_ptr, _dst_guard) = out.device_ptr_mut(&self.stream);
                unsafe {
                    nccl.all_reduce(
                        dst_ptr as *mut _,
                        len,
                        DType::F32,
                        ReduceOp::Sum,
                        self.stream.cu_stream().cast(),
                    )
                    .map_err(|_| AutogradError::TapeInvariant("NCCL all_reduce_sum failed"))?;
                }
            }

            Ok(DeviceHandle::Cuda(CudaStorage::new(out)))
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
            let local_len = shape_size(local_shape);
            let src = self.cuda_slice(x, "all_gather_seq")?;
            if src.len() != local_len {
                return Err(AutogradError::DataLengthMismatch {
                    len: src.len(),
                    shape: local_shape.to_vec(),
                    size: local_len,
                });
            }

            // No communicator (single-process / CPU parity path): identity — the
            // gathered full sequence equals this rank's local shard (world==1).
            // Device buffers are functional (write-fresh; see add_into_device), and
            // this branch skips the in-place NCCL write, so sharing the input Arc is
            // safe — no alloc, no D2D copy.
            #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
            let world = self.comm(axis).map_or(1, |nccl| nccl.world_size());
            #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
            let world = 1usize;
            if world <= 1 {
                return Ok(x.clone());
            }

            // NCCL all-gather: shards are equal-length (seq % world == 0), so the
            // full [1, S, H] is the rank-order concatenation of each [1, S/N, H].
            #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
            {
                let full_len = local_len * world;
                let mut out = self.stream.alloc_zeros::<f32>(full_len).map_err(|_| {
                    AutogradError::TapeInvariant("cuda all_gather_seq full alloc failed")
                })?;
                let nccl = self.comm(axis).expect("world>1 implies nccl present");
                // Scope the device-ptr guards so their SyncOnDrop borrow of `out`
                // ends before `out` is moved into the handle (mirrors the implicit
                // drop in all_reduce_sum_device's `if let` block).
                {
                    let (src_ptr, _src_guard) = src.device_ptr(&self.stream);
                    let (dst_ptr, _dst_guard) = out.device_ptr_mut(&self.stream);
                    unsafe {
                        nccl.all_gather(
                            src_ptr as *const _,
                            dst_ptr as *mut _,
                            local_len,
                            DType::F32,
                            self.stream.cu_stream().cast(),
                        )
                        .map_err(|_| AutogradError::TapeInvariant("NCCL all_gather_seq failed"))?;
                    }
                }
                return Ok(DeviceHandle::Cuda(CudaStorage::new(out)));
            }
            #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
            unreachable!("world>1 without nccl feature")
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
            let local_len = shape_size(local_shape);
            let src = self.cuda_slice(x, "reduce_scatter_sum")?;

            #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
            let world = self.comm(axis).map_or(1, |nccl| nccl.world_size());
            #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
            let world = 1usize;
            if world <= 1 {
                // Identity: input already this rank's [1, S/N, H] (== full at N=1).
                // Share the input Arc (functional buffers) — no alloc, no D2D copy.
                if src.len() != local_len {
                    return Err(AutogradError::DataLengthMismatch {
                        len: src.len(),
                        shape: local_shape.to_vec(),
                        size: local_len,
                    });
                }
                return Ok(x.clone());
            }

            #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
            {
                if src.len() != local_len * world {
                    return Err(AutogradError::DataLengthMismatch {
                        len: src.len(),
                        shape: local_shape.to_vec(),
                        size: local_len * world,
                    });
                }
                let mut out = self.stream.alloc_zeros::<f32>(local_len).map_err(|_| {
                    AutogradError::TapeInvariant("cuda reduce_scatter_sum alloc failed")
                })?;
                let nccl = self.comm(axis).expect("world>1 implies nccl present");
                // Scope the device-ptr guards so their SyncOnDrop borrow of `out`
                // ends before `out` is moved into the handle.
                {
                    let (src_ptr, _src_guard) = src.device_ptr(&self.stream);
                    let (dst_ptr, _dst_guard) = out.device_ptr_mut(&self.stream);
                    unsafe {
                        nccl.reduce_scatter(
                            src_ptr as *const _,
                            dst_ptr as *mut _,
                            local_len,
                            DType::F32,
                            ReduceOp::Sum,
                            self.stream.cu_stream().cast(),
                        )
                        .map_err(|_| {
                            AutogradError::TapeInvariant("NCCL reduce_scatter_sum failed")
                        })?;
                    }
                }
                return Ok(DeviceHandle::Cuda(CudaStorage::new(out)));
            }
            #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
            unreachable!("world>1 without nccl feature")
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
            let len = shape_size(block_shape);
            let src = self.cuda_slice(block, "ring_send_recv_kv")?;
            if src.len() != len {
                return Err(AutogradError::DataLengthMismatch {
                    len: src.len(),
                    shape: block_shape.to_vec(),
                    size: len,
                });
            }

            #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
            let world = self.comm(CommAxis::Seq).map_or(1, |nccl| nccl.world_size());
            #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
            let world = 1usize;
            // Single rank: the ring degenerates to the local block (identity).
            if world <= 1 {
                return Ok(block.clone());
            }

            // Ring rotation: send this block to (rank+1)%world, receive the block
            // from (rank-1+world)%world, both inside one group so NCCL pairs the
            // matched send/recv without deadlock. Blocks are equal-length (the
            // launcher pads seq to a multiple of world), so recv fills `len`.
            #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
            {
                let nccl = self
                    .comm(CommAxis::Seq)
                    .expect("world>1 implies nccl present");
                let rank = nccl.rank();
                let next = (rank + 1) % world;
                let prev = (rank + world - 1) % world;
                let mut out = self.stream.alloc_zeros::<f32>(len).map_err(|_| {
                    AutogradError::TapeInvariant("cuda ring_send_recv_kv alloc failed")
                })?;
                {
                    let (src_ptr, _src_guard) = src.device_ptr(&self.stream);
                    let (dst_ptr, _dst_guard) = out.device_ptr_mut(&self.stream);
                    let stream = self.stream.cu_stream().cast();
                    nccl.group_start()
                        .map_err(|_| AutogradError::TapeInvariant("ring group_start failed"))?;
                    unsafe {
                        nccl.send(src_ptr as *const _, len, DType::F32, next, stream)
                            .map_err(|_| AutogradError::TapeInvariant("ring send failed"))?;
                        nccl.recv(dst_ptr as *mut _, len, DType::F32, prev, stream)
                            .map_err(|_| AutogradError::TapeInvariant("ring recv failed"))?;
                    }
                    nccl.group_end()
                        .map_err(|_| AutogradError::TapeInvariant("ring group_end failed"))?;
                }
                return Ok(DeviceHandle::Cuda(CudaStorage::new(out)));
            }
            #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
            unreachable!("world>1 without nccl feature")
        }
    }

    fn comm_world_rank(&self, axis: CommAxis) -> (usize, usize) {
        #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
        {
            self.comm(axis)
                .map_or((1, 0), |nccl| (nccl.world_size(), nccl.rank()))
        }
        #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
        {
            let _ = axis;
            (1, 0)
        }
    }

    fn ep_exchange_rows_device(
        &self,
        x: &DeviceHandle,
        dim: usize,
        send_counts: &[usize],
        recv_counts: &[usize],
        axis: CommAxis,
    ) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, dim, send_counts, recv_counts, axis);
            todo!("GPU required: cuda ep_exchange_rows_device is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            let send_rows: usize = send_counts.iter().sum();
            let src = self.cuda_slice(x, "ep_exchange_rows")?;
            if src.len() != send_rows * dim {
                return Err(AutogradError::DataLengthMismatch {
                    len: src.len(),
                    shape: vec![send_rows, dim],
                    size: send_rows * dim,
                });
            }

            #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
            let world = self.comm(axis).map_or(1, |nccl| nccl.world_size());
            #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
            let world = 1usize;
            if world <= 1 {
                return Ok(x.clone());
            }

            #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
            {
                if send_counts.len() != world || recv_counts.len() != world {
                    return Err(AutogradError::TapeInvariant(
                        "ep_exchange_rows: counts length must equal the group size",
                    ));
                }
                let nccl = self.comm(axis).expect("world>1 implies nccl present");
                let rank = nccl.rank();
                let recv_rows: usize = recv_counts.iter().sum();
                let mut out = self
                    .stream
                    .alloc_zeros::<f32>(recv_rows * dim)
                    .map_err(|_| AutogradError::TapeInvariant("ep_exchange recv alloc failed"))?;
                let send_offs: Vec<usize> = send_counts
                    .iter()
                    .scan(0, |o, &c| {
                        let s = *o;
                        *o += c * dim;
                        Some(s)
                    })
                    .collect();
                let recv_offs: Vec<usize> = recv_counts
                    .iter()
                    .scan(0, |o, &c| {
                        let s = *o;
                        *o += c * dim;
                        Some(s)
                    })
                    .collect();
                // Own segment moves by D2D; peers pair inside one NCCL group.
                if send_counts[rank] != recv_counts[rank] {
                    return Err(AutogradError::TapeInvariant(
                        "ep_exchange_rows: self send/recv counts must match",
                    ));
                }
                if send_counts[rank] > 0 {
                    let seg = send_offs[rank]..send_offs[rank] + send_counts[rank] * dim;
                    let mut dst =
                        out.slice_mut(recv_offs[rank]..recv_offs[rank] + recv_counts[rank] * dim);
                    self.stream
                        .memcpy_dtod(&src.slice(seg), &mut dst)
                        .map_err(|_| AutogradError::TapeInvariant("ep_exchange self D2D failed"))?;
                }
                {
                    let stream = self.stream.cu_stream().cast();
                    let (src_ptr, _sg) = src.device_ptr(&self.stream);
                    let (dst_ptr, _dg) = out.device_ptr_mut(&self.stream);
                    nccl.group_start()
                        .map_err(|_| AutogradError::TapeInvariant("ep_exchange group_start"))?;
                    for j in 0..world {
                        if j == rank {
                            continue;
                        }
                        // SAFETY: offsets stay inside src/out (built from the same
                        // counts the length checks above validated).
                        unsafe {
                            if send_counts[j] > 0 {
                                nccl.send(
                                    (src_ptr as *const f32).add(send_offs[j]).cast(),
                                    send_counts[j] * dim,
                                    DType::F32,
                                    j,
                                    stream,
                                )
                                .map_err(|_| {
                                    AutogradError::TapeInvariant("ep_exchange send failed")
                                })?;
                            }
                            if recv_counts[j] > 0 {
                                nccl.recv(
                                    (dst_ptr as *mut f32).add(recv_offs[j]).cast(),
                                    recv_counts[j] * dim,
                                    DType::F32,
                                    j,
                                    stream,
                                )
                                .map_err(|_| {
                                    AutogradError::TapeInvariant("ep_exchange recv failed")
                                })?;
                            }
                        }
                    }
                    nccl.group_end()
                        .map_err(|_| AutogradError::TapeInvariant("ep_exchange group_end"))?;
                }
                return Ok(DeviceHandle::Cuda(CudaStorage::new(out)));
            }
            #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
            unreachable!("world>1 without nccl feature")
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
            let rank_n = in_shape.len();
            if scatter_axis == gather_axis || scatter_axis >= rank_n || gather_axis >= rank_n {
                return Err(AutogradError::TapeInvariant(
                    "all_to_all: scatter/gather axes must be distinct and in range",
                ));
            }
            let len = shape_size(in_shape);
            let src = self.cuda_slice(x, "all_to_all")?;
            if src.len() != len {
                return Err(AutogradError::DataLengthMismatch {
                    len: src.len(),
                    shape: in_shape.to_vec(),
                    size: len,
                });
            }

            #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
            let world = self.comm(axis).map_or(1, |nccl| nccl.world_size());
            #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
            let world = 1usize;
            // Single rank: no rank to shuffle to — identity on shape and value.
            // Share the input Arc (functional buffers) — no alloc, no D2D copy.
            if world <= 1 {
                return Ok((x.clone(), in_shape.to_vec()));
            }

            #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
            {
                let n = world;
                if !in_shape[gather_axis].is_multiple_of(n) {
                    return Err(AutogradError::TapeInvariant(
                        "all_to_all: gather axis not divisible by world",
                    ));
                }
                // out[scatter] *= N, out[gather] /= N. Invertible under axis-swap, so
                // the generic backward (collective.rs) reproduces in_shape exactly.
                let g = in_shape[gather_axis] / n;
                let mut out_shape = in_shape.to_vec();
                out_shape[scatter_axis] *= n;
                out_shape[gather_axis] = g;

                // Send side: slice the gather axis into N equal chunks; chunk j is
                // rank j's share (heads are outer within the axis, so a contiguous
                // g-wide slice == that rank's head range).
                let mut send: Vec<DeviceHandle> = Vec::with_capacity(n);
                for j in 0..n {
                    let mut starts = vec![0usize; rank_n];
                    let mut ends = in_shape.to_vec();
                    starts[gather_axis] = j * g;
                    ends[gather_axis] = (j + 1) * g;
                    send.push(cuda_slice_device(self, x, in_shape, &starts, &ends)?);
                }
                let chunk_len = len / n;

                // Transport: one NCCL group of send/recv pairs. Own chunk (j==rank)
                // is reused from `send` — no self NCCL op, no self copy, no deadlock.
                let nccl = self.comm(axis).expect("world>1 implies nccl present");
                let rank = nccl.rank();
                let mut recv: Vec<Option<CudaSlice<f32>>> = (0..n).map(|_| None).collect();
                for (j, slot) in recv.iter_mut().enumerate() {
                    if j != rank {
                        *slot = Some(self.stream.alloc_zeros::<f32>(chunk_len).map_err(|_| {
                            AutogradError::TapeInvariant("all_to_all recv alloc failed")
                        })?);
                    }
                }
                {
                    let stream = self.stream.cu_stream().cast();
                    let mut guards = Vec::new();
                    let mut send_ptrs = Vec::new();
                    for (j, chunk) in send.iter().enumerate() {
                        if j != rank {
                            let s = self.cuda_slice(chunk, "all_to_all")?;
                            let (p, guard) = s.device_ptr(&self.stream);
                            send_ptrs.push((j, p));
                            guards.push(guard);
                        }
                    }
                    let mut recv_ptrs = Vec::new();
                    for (j, slot) in recv.iter_mut().enumerate() {
                        if let Some(buf) = slot {
                            let (p, guard) = buf.device_ptr_mut(&self.stream);
                            recv_ptrs.push((j, p));
                            guards.push(guard);
                        }
                    }
                    nccl.group_start()
                        .map_err(|_| AutogradError::TapeInvariant("a2a group_start failed"))?;
                    // SAFETY: every ptr is live for the group (its guard is held in
                    // `guards`), each chunk_len matches its buffer, and send/recv are
                    // symmetric across ranks so NCCL pairs them inside the one group.
                    unsafe {
                        for &(j, p) in &send_ptrs {
                            nccl.send(p as *const _, chunk_len, DType::F32, j, stream)
                                .map_err(|_| AutogradError::TapeInvariant("a2a send failed"))?;
                        }
                        for &(j, p) in &recv_ptrs {
                            nccl.recv(p as *mut _, chunk_len, DType::F32, j, stream)
                                .map_err(|_| AutogradError::TapeInvariant("a2a recv failed"))?;
                        }
                    }
                    nccl.group_end()
                        .map_err(|_| AutogradError::TapeInvariant("a2a group_end failed"))?;
                }

                // Recv assembly: concat the N chunks (source-rank order) along the
                // scatter axis, on-device (see `cuda_concat_axis`).
                let recv_handles: Vec<DeviceHandle> = (0..n)
                    .map(|j| {
                        if j == rank {
                            send[rank].clone()
                        } else {
                            DeviceHandle::Cuda(CudaStorage::new(recv[j].take().unwrap()))
                        }
                    })
                    .collect();
                let mut chunk_shape = in_shape.to_vec();
                chunk_shape[gather_axis] = g;
                let parts: Vec<(&DeviceHandle, &[usize])> = recv_handles
                    .iter()
                    .map(|h| (h, chunk_shape.as_slice()))
                    .collect();
                let (out_h, produced) = cuda_concat_axis(self, &parts, scatter_axis)?;
                debug_assert_eq!(produced, out_shape);
                return Ok((out_h, out_shape));
            }
            #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
            unreachable!("world>1 without nccl feature")
        }
    }

    /// Device-resident backward for `mul_scalar`. Pure elementwise
    /// `grad_x[i] = upstream[i] * k` via a 1D NVRTC kernel; returns an
    /// unevaluated handle per the batched-eval contract.
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

    /// Device-resident backward for `mean`. Scalar `upstream_grad`
    /// (rank-0 device handle) broadcast-divided by `elem_count` across
    /// `output_shape`. Returns an unevaluated handle.
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

    /// Device-resident row-wise softmax over the last axis. The
    /// default trait implementation falls back to
    /// `readback → host compute → upload`, which on production shapes
    /// (`[B, S, V] = 2 × 512 × 248070 × 4 B ≈ 1 GB`) dominates per-step
    /// wall time. Here we reuse the existing NVRTC kernel
    /// (`softmax_last_axis_f32` in `backend_cuda/kernels/softmax.cu`) but
    /// keep the result on-device so the CE-loss chain (softmax → gather)
    /// stays lazy. No `synchronize()` — the eval contract belongs to the
    /// caller (`Tape::backward` / `AdamW::step_device`).
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

    /// Device-resident row-wise log-softmax over the last axis.
    /// Same rationale as `softmax_last_axis` (no host roundtrip; the
    /// existing `log_softmax_last_axis_f32` NVRTC kernel runs against
    /// the device-side slice in place).
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

    /// Device-resident gather along the last axis. Reuses the
    /// existing `gather_last_dim_f32` NVRTC kernel against the
    /// device-side `src` slice, returning a fresh `CudaSlice<f32>` of
    /// length `product(src_shape[..-1])` without a host roundtrip. The
    /// CE-loss chain is the production caller: keeps the
    /// `[B,S,V]` logits on-device through the per-row gather instead of
    /// materializing the full ~1 GB tensor on the host between
    /// `log_softmax` and `gather`.
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

    /// Device-resident backward for
    /// `log_softmax_last_axis`. Consumes the saved forward output
    /// directly from its `DeviceHandle` (no DtoH) and the upstream gradient
    /// directly from device — kills the `1 015 MB` log_softmax-grad readback
    /// nsys identified as the single largest transfer per training step.
    /// Returns an unevaluated `CudaSlice<f32>` handle per the batched-eval
    /// contract — `Tape::backward`'s terminal eval (or the
    /// AdamW step) does the single host fence.
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

    /// Device-resident backward for `gather_last_dim`. Produces a
    /// zero-filled `[B, S, V]` (or any `src_shape`) grad on-device and
    /// writes the per-prefix upstream scalar at `(row, ids[row])` — one
    /// thread per prefix row, no atomics needed since indices across rows
    /// touch disjoint slots. Keeps the post-gather backward chain
    /// device-resident so the upstream gradient flowing into
    /// `log_softmax_last_axis_backward` never goes through host.
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

    /// Device-resident embedding backward.
    /// Allocates a zero-filled `[vocab, hidden]` grad on-device and
    /// atomicAdd-scatters the per-token-position upstream slice into
    /// `grad_table[ids[row], :]`. `atomicAdd` is mandatory for the
    /// duplicate-token correctness guarantee. No `synchronize()` — terminal
    /// eval is the caller's.
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

    /// Device-resident add_broadcast backward.
    /// Reduces the upstream `[a_shape]` tensor along broadcast axes into
    /// a `[b_shape]` grad via a per-output-element shared-memory block
    /// reduction. Mirrors the `add_broadcast` forward layout contract
    /// (right-aligned `b_strides` of length `out_rank`, stride-0 entries
    /// for contracted axes).
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

    /// Fused on-device AdamW per-parameter update. Replaces the default
    /// `Backend::adamw_step` host-loop fallback (which does
    /// `readback × 3 + cpu_adamw_step_in_place + upload × 3` per param per
    /// step) with a single NVRTC kernel launch. The CUDA override mutates
    /// the existing param/m/v device buffers in place and returns Arc-cloned
    /// handles to those same buffers, avoiding the former 3x allocation +
    /// DtoD seed-copy cost per tensor. Matches the formula in
    /// `crates/autograd/src/backend.rs::cpu_adamw_step_in_place` to
    /// floating-point rounding (validated by
    /// `tests/test_cuda_adamw_step.rs` to ≤1e-4 rel-error after 5 steps).
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

    /// Device-grad fused AdamW. Same kernel as `adamw_step`
    /// (`adamw_step_f32`) but the gradient is sourced directly from
    /// the caller's `DeviceHandle::Cuda` — **no `clone_htod`**. This kills
    /// the per-param-per-grad-accum-step DtoH incurred when
    /// `embedding_backward` /
    /// `add_broadcast_backward` produce device-resident grads.
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

    /// Device-resident backward for `silu(x)`. Single 1D NVRTC
    /// kernel `dx[i] = upstream[i] * silu'(x[i])`; both `upstream` and the
    /// saved input `x` stay on-device. Returned handle is unevaluated.
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

    /// Device-resident backward for `sigmoid(x)`. Consumes the
    /// saved output `y`: `dx[i] = upstream[i] * y[i] * (1 - y[i])`.
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

    /// Device-resident backward for `exp(x)`. Consumes the saved
    /// output `y = exp(x)`: `dx[i] = upstream[i] * y[i]`.
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

    /// Device-resident backward for `mul(a, b)`. Two 1D NVRTC
    /// kernels — one per side — gated by `need_grad_a` / `need_grad_b`.
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

    /// Device-resident backward for `rms_norm`. Three NVRTC
    /// kernels: per-row `inv_rms`, per-row `grad_x` with shared-mem `dot`
    /// reduction, per-col `grad_w` reduction.
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

    /// Device-resident backward for `rope`. Single NVRTC kernel
    /// — same body as `rope_f32` with the `sin` sign inlined-negated.
    /// `cos`/`sin` are uploaded fresh (tiny: `[seq, head_dim/2]`).
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

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_adamw_step(
    backend: &CudaBackend,
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
    let size = shape_size(shape);
    if grad.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: grad.len(),
            shape: shape.to_vec(),
            size,
        });
    }
    let param_slice = backend.cuda_slice(param, "adamw_step")?;
    let m_slice = backend.cuda_slice(m, "adamw_step")?;
    let v_slice = backend.cuda_slice(v, "adamw_step")?;
    if param_slice.len() != size || m_slice.len() != size || v_slice.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: param_slice.len().min(m_slice.len()).min(v_slice.len()),
            shape: shape.to_vec(),
            size,
        });
    }

    // grad arrives host-side from autograd's host-authoritative gradient
    // path (matmul_backward still returns Vec<f32>); upload it once.
    let d_grad = backend
        .stream
        .clone_htod(grad)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (adamw grad)"))?;

    let n = i32::try_from(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda adamw length exceeds i32"))?;
    launch_1d(
        &backend.stream,
        backend.kernels.function("adamw_step_f32")?,
        size,
        |mut builder| {
            // `PushKernelArg<&CudaSlice<T>>` passes the raw CUdeviceptr.
            // The kernel parameters are mutable `float*`, so CUDA updates
            // the existing buffers in place. This deliberately avoids
            // cloning the `CudaSlice`: `CudaSlice::clone()` is a device copy.
            builder
                .arg(param_slice)
                .arg(m_slice)
                .arg(v_slice)
                .arg(&d_grad)
                .arg(&n)
                .arg(&lr)
                .arg(&beta1)
                .arg(&beta2)
                .arg(&eps)
                .arg(&wd)
                .arg(&bc1)
                .arg(&bc2);
            builder
        },
    )?;

    // Per the Backend::adamw_step eval contract : return
    // unevaluated handles. These are Arc clones of the same in-place
    // buffers, not fresh allocations.
    Ok((param.clone(), m.clone(), v.clone()))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_adamw_step_device(
    backend: &CudaBackend,
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
    let size = shape_size(shape);
    let param_slice = backend.cuda_slice(param, "adamw_step_device")?;
    let m_slice = backend.cuda_slice(m, "adamw_step_device")?;
    let v_slice = backend.cuda_slice(v, "adamw_step_device")?;
    let grad_slice = backend.cuda_slice(grad, "adamw_step_device")?;
    if param_slice.len() != size
        || m_slice.len() != size
        || v_slice.len() != size
        || grad_slice.len() != size
    {
        return Err(AutogradError::DataLengthMismatch {
            len: param_slice
                .len()
                .min(m_slice.len())
                .min(v_slice.len())
                .min(grad_slice.len()),
            shape: shape.to_vec(),
            size,
        });
    }

    // Crucially: no `clone_htod(grad)`. The grad already lives on-device;
    // we pass the existing `&CudaSlice<f32>` straight into the kernel.
    let n = i32::try_from(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda adamw length exceeds i32"))?;
    launch_1d(
        &backend.stream,
        backend.kernels.function("adamw_step_f32")?,
        size,
        |mut builder| {
            // In-place update: see `cuda_adamw_step` above. Passing the
            // borrowed slices avoids `CudaSlice::clone()`, which is a DtoD
            // allocation+copy in cudarc, not an Arc ref-count bump.
            builder
                .arg(param_slice)
                .arg(m_slice)
                .arg(v_slice)
                .arg(grad_slice)
                .arg(&n)
                .arg(&lr)
                .arg(&beta1)
                .arg(&beta2)
                .arg(&eps)
                .arg(&wd)
                .arg(&bc1)
                .arg(&bc2);
            builder
        },
    )?;

    // Eval contract : return unevaluated; caller batches the
    // terminal `stream.synchronize()` for the whole optimizer step.
    Ok((param.clone(), m.clone(), v.clone()))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_softmax_like(
    backend: &CudaBackend,
    x: &[f32],
    shape: &[usize],
    kernel_name: &'static str,
) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected: usize = shape.iter().product();
    if x.len() != expected {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![x.len()],
        });
    }

    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda softmax cols exceeds i32"))?;
    let d_in = backend.upload_slice(x, shape)?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(expected)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
    launch_rows(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder.arg(&mut d_out).arg(&d_in).arg(&cols);
            builder
        },
    )?;

    let mut host = vec![0.0f32; expected];
    backend
        .stream
        .memcpy_dtoh(&d_out, &mut host)
        .map_err(|_| AutogradError::TapeInvariant("cuda dtoh copy failed"))?;
    backend
        .stream
        .synchronize()
        .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))?;
    Ok(host)
}

// Device-resident sibling of `cuda_softmax_like`: same NVRTC kernel + same
// 256-thread shared-mem reduction, but takes the input as a borrowed
// `CudaSlice<f32>` and returns a fresh `CudaSlice<f32>` instead of doing
// `upload → kernel → readback`. No `synchronize()` — the caller owns the
// terminal eval (Tape::backward / AdamW::step_device batched flush per the
// batched-eval contract). Reused for both `softmax_last_axis_f32` and
// `log_softmax_last_axis_f32` (selected by `kernel_name`).
#[cfg(not(feature = "no-cuda"))]
fn cuda_softmax_like_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    shape: &[usize],
    kernel_name: &'static str,
) -> Result<DeviceHandle> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected = shape_size(shape);
    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda softmax cols exceeds i32"))?;
    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
    if let DeviceHandle::CudaBf16(storage) = x {
        let d_in = backend.cuda_bf16_storage_slice(storage)?;
        if d_in.len() != expected {
            return Err(AutogradError::DataLengthMismatch {
                len: d_in.len(),
                shape: shape.to_vec(),
                size: expected,
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(expected)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
        let func = backend.kernels.function_for(kernel_name, TapeDtype::Bf16)?;
        launch_rows(
            &backend.stream,
            &func,
            rows,
            BLOCK,
            SHARED,
            |mut builder| {
                builder.arg(&mut d_out).arg(d_in).arg(&cols);
                builder
            },
        )?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }
    let d_in = backend.cuda_slice(x, "softmax_last_axis")?;
    if d_in.len() != expected {
        return Err(AutogradError::DataLengthMismatch {
            len: d_in.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }

    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(expected)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    launch_rows(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder.arg(&mut d_out).arg(d_in).arg(&cols);
            builder
        },
    )?;

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

// Device-resident sibling of `cuda_gather_last_dim`: reuses the same
// `gather_last_dim_f32` NVRTC kernel against a borrowed device slice and
// returns the per-prefix output on-device. Only the int32 `ids` array
// crosses PCIe; the `[B*S*V]` source stays on-device. No `synchronize()` —
// caller owns the terminal eval.
#[cfg(not(feature = "no-cuda"))]
fn cuda_gather_last_dim_device(
    backend: &CudaBackend,
    src: &DeviceHandle,
    src_shape: &[usize],
    ids: &[i32],
) -> Result<DeviceHandle> {
    if src_shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }
    let vocab = *src_shape.last().expect("non-empty shape above");
    let prefix: usize = src_shape[..src_shape.len() - 1]
        .iter()
        .product::<usize>()
        .max(1);
    let expected: usize = src_shape.iter().product();
    let d_src = backend.cuda_slice(src, "gather_last_dim")?;
    if d_src.len() != expected {
        return Err(AutogradError::DataLengthMismatch {
            len: d_src.len(),
            shape: src_shape.to_vec(),
            size: expected,
        });
    }
    if ids.len() != prefix {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix,
            got: ids.len(),
        });
    }
    let d_ids = backend
        .stream
        .clone_htod(ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(prefix)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    let n_i32 = i32::try_from(prefix)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather n exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather vocab exceeds i32"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("gather_last_dim_f32")?,
        prefix,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_src)
                .arg(&d_ids)
                .arg(&n_i32)
                .arg(&vocab_i32);
            builder
        },
    )?;

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

// Device-resident log_softmax
// backward. `upstream` and `log_softmax_output` arrive as borrowed CUDA
// slices via the `Backend::log_softmax_last_axis_backward` contract; the
// fresh grad allocation stays device-resident and is returned unevaluated
// for the tape's terminal eval (mirrors the forward helper pattern).
// Same 256-thread shared-mem reduce shape as `softmax_last_axis_f32`.
#[cfg(not(feature = "no-cuda"))]
fn cuda_log_softmax_last_axis_backward(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    log_softmax_output: &DeviceHandle,
    shape: &[usize],
) -> Result<DeviceHandle> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected = shape_size(shape);
    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda log_softmax_backward cols exceeds i32"))?;
    // Lane follows the saved forward output's dtype; the other operand is
    // harmonized so the adjoint reads the values forward produced.
    if let DeviceHandle::CudaBf16(storage) = log_softmax_output {
        let d_out = backend.cuda_bf16_storage_slice(storage)?;
        let d_up_op = backend.bf16_operand(upstream, "log_softmax_last_axis_backward")?;
        let d_up = d_up_op.get();
        if d_up.len() != expected || d_out.len() != expected {
            return Err(AutogradError::DataLengthMismatch {
                len: d_up.len().min(d_out.len()),
                shape: shape.to_vec(),
                size: expected,
            });
        }
        let mut d_grad = backend.stream.alloc_zeros::<u16>(expected).map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (log_softmax_bwd)")
        })?;
        const BLOCK: u32 = 256;
        const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
        let func = backend
            .kernels
            .function_for("log_softmax_last_axis_backward_f32", TapeDtype::Bf16)?;
        launch_rows(
            &backend.stream,
            &func,
            rows,
            BLOCK,
            SHARED,
            |mut builder| {
                builder.arg(&mut d_grad).arg(d_up).arg(d_out).arg(&cols);
                builder
            },
        )?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_grad)));
    }
    let d_up_op = backend.f32_operand(upstream, "log_softmax_last_axis_backward")?;
    let d_up = d_up_op.get();
    let d_out = backend.cuda_slice(log_softmax_output, "log_softmax_last_axis_backward")?;
    if d_up.len() != expected {
        return Err(AutogradError::DataLengthMismatch {
            len: d_up.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    if d_out.len() != expected {
        return Err(AutogradError::DataLengthMismatch {
            len: d_out.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }

    let mut d_grad = backend.stream.alloc_zeros::<f32>(expected).map_err(|e| {
        eprintln!("[autograd] alloc_zeros {expected} x f32 failed (log_softmax_bwd): {e}");
        AutogradError::TapeInvariant("cuda alloc_zeros failed (log_softmax_bwd)")
    })?;

    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
    launch_rows(
        &backend.stream,
        backend
            .kernels
            .function("log_softmax_last_axis_backward_f32")?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder.arg(&mut d_grad).arg(d_up).arg(d_out).arg(&cols);
            builder
        },
    )?;

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_softmax_last_axis_backward(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    softmax_output: &DeviceHandle,
    shape: &[usize],
) -> Result<DeviceHandle> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected = shape_size(shape);
    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda softmax_backward cols exceeds i32"))?;
    if let DeviceHandle::CudaBf16(storage) = softmax_output {
        let d_out = backend.cuda_bf16_storage_slice(storage)?;
        let d_up_op = backend.bf16_operand(upstream, "softmax_last_axis_backward")?;
        let d_up = d_up_op.get();
        if d_up.len() != expected || d_out.len() != expected {
            return Err(AutogradError::DataLengthMismatch {
                len: d_up.len().min(d_out.len()),
                shape: shape.to_vec(),
                size: expected,
            });
        }
        let mut d_grad = backend
            .stream
            .alloc_zeros::<u16>(expected)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (softmax_bwd)"))?;
        const BLOCK: u32 = 256;
        const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
        let func = backend
            .kernels
            .function_for("softmax_last_axis_backward_f32", TapeDtype::Bf16)?;
        launch_rows(
            &backend.stream,
            &func,
            rows,
            BLOCK,
            SHARED,
            |mut builder| {
                builder.arg(&mut d_grad).arg(d_up).arg(d_out).arg(&cols);
                builder
            },
        )?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_grad)));
    }
    let d_up_op = backend.f32_operand(upstream, "softmax_last_axis_backward")?;
    let d_up = d_up_op.get();
    let d_out = backend.cuda_slice(softmax_output, "softmax_last_axis_backward")?;
    if d_up.len() != expected {
        return Err(AutogradError::DataLengthMismatch {
            len: d_up.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    if d_out.len() != expected {
        return Err(AutogradError::DataLengthMismatch {
            len: d_out.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }

    let mut d_grad = backend
        .stream
        .alloc_zeros::<f32>(expected)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (softmax_bwd)"))?;

    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
    launch_rows(
        &backend.stream,
        backend.kernels.function("softmax_last_axis_backward_f32")?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder.arg(&mut d_grad).arg(d_up).arg(d_out).arg(&cols);
            builder
        },
    )?;

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
}

// Device-resident backward for gather_last_dim. Allocates a
// zero-filled `[product(src_shape)]` grad on-device and scatters the
// per-prefix upstream scalar into `(row, ids[row])`. Only the int32
// `indices` array crosses PCIe; the `[prefix_rows]` upstream slice stays
// on-device. No `synchronize()` — terminal eval is the caller's.
#[cfg(not(feature = "no-cuda"))]
fn cuda_gather_last_dim_backward(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    indices: &[i32],
    src_shape: &[usize],
) -> Result<DeviceHandle> {
    if src_shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }
    let vocab = *src_shape.last().expect("non-empty shape above");
    let prefix: usize = src_shape[..src_shape.len() - 1]
        .iter()
        .product::<usize>()
        .max(1);
    let total = prefix * vocab;
    let d_up = backend.cuda_slice(upstream, "gather_last_dim_backward")?;
    if d_up.len() != prefix {
        return Err(AutogradError::DataLengthMismatch {
            len: d_up.len(),
            shape: src_shape[..src_shape.len() - 1].to_vec(),
            size: prefix,
        });
    }
    if indices.len() != prefix {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix,
            got: indices.len(),
        });
    }
    let d_ids = backend
        .stream
        .clone_htod(indices)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (gather_bwd ids)"))?;
    // alloc_zeros gives us the zero-fill for free — kernel only writes the
    // single (row, ids[row]) slot per prefix row.
    let mut d_grad = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (gather_bwd grad)"))?;

    let prefix_i32 = i32::try_from(prefix)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather_bwd prefix exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather_bwd vocab exceeds i32"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("gather_last_dim_backward_f32")?,
        prefix,
        |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_ids)
                .arg(&prefix_i32)
                .arg(&vocab_i32);
            builder
        },
    )?;

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
}

// --- Context-parallel ring attention (Track A device path) ---
// One-block-pure kernels + on-device flash-2 merge; the ring rotation and tape
// live in ops/ring_attention.rs. q/k/v arrive as f32 handles (tape tensors),
// converted to bf16 for the kernel (matching the training activation precision);
// the (m,l,o) accumulators and grads stay f32 for a stable merge.
#[cfg(not(feature = "no-cuda"))]
fn ring_i32(v: usize, label: &'static str) -> Result<i32> {
    i32::try_from(v).map_err(|_| AutogradError::TapeInvariant(label))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_ring_block_fwd_merge(
    backend: &CudaBackend,
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
    let q_slice = backend.cuda_slice(q, "ring q")?;
    let k_slice = backend.cuda_slice(k_blk, "ring k")?;
    let v_slice = backend.cuda_slice(v_blk, "ring v")?;
    let q_bf16 = backend.local_f32_as_bf16(q_slice, q_slice.len())?;
    let k_bf16 = backend.local_f32_as_bf16(k_slice, k_slice.len())?;
    let v_bf16 = backend.local_f32_as_bf16(v_slice, v_slice.len())?;
    let m_in = backend.cuda_slice(acc_m, "ring acc_m")?;
    let l_in = backend.cuda_slice(acc_l, "ring acc_l")?;
    let o_in = backend.cuda_slice(acc_o, "ring acc_o")?;
    if ring_fa3_route(backend, dims, q_pos_host, k_pos_host) {
        return cuda_ring_block_fwd_merge_fa3(
            backend, &q_bf16, &k_bf16, &v_bf16, m_in, l_in, o_in, q_pos_host, k_pos_host, dims,
        );
    }
    let qpos_slice = backend.cuda_slice(q_pos, "ring q_pos")?;
    let kpos_slice = backend.cuda_slice(k_pos, "ring k_pos")?;
    let rows = dims.num_q_tiles * dims.q_rows;
    let mut m_out = backend
        .stream
        .alloc_zeros::<f32>(rows)
        .map_err(|_| cuda_alloc_failed("ring m_out", vec![rows]))?;
    let mut l_out = backend
        .stream
        .alloc_zeros::<f32>(rows)
        .map_err(|_| cuda_alloc_failed("ring l_out", vec![rows]))?;
    let mut o_out = backend
        .stream
        .alloc_zeros::<f32>(rows * dims.head_dim)
        .map_err(|_| cuda_alloc_failed("ring o_out", vec![rows * dims.head_dim]))?;
    {
        let (q_ptr, _qg) = q_bf16.device_ptr(&backend.stream);
        let (k_ptr, _kg) = k_bf16.device_ptr(&backend.stream);
        let (v_ptr, _vg) = v_bf16.device_ptr(&backend.stream);
        let (mi_ptr, _mig) = m_in.device_ptr(&backend.stream);
        let (li_ptr, _lig) = l_in.device_ptr(&backend.stream);
        let (oi_ptr, _oig) = o_in.device_ptr(&backend.stream);
        let (mo_ptr, _mog) = m_out.device_ptr_mut(&backend.stream);
        let (lo_ptr, _log) = l_out.device_ptr_mut(&backend.stream);
        let (oo_ptr, _oog) = o_out.device_ptr_mut(&backend.stream);
        let (qpos_ptr, _qpg) = qpos_slice.device_ptr(&backend.stream);
        let (kpos_ptr, _kpg) = kpos_slice.device_ptr(&backend.stream);
        check_cuda_ffi(
            // SAFETY: q/k/v are live bf16 copies; acc_*_in are f32 handles of the right
            // length; *_out are freshly allocated rows / rows*hd; q_pos/k_pos are f32
            // handles of q_rows / blk_len; dims mirror the shapes.
            unsafe {
                ffi::ring_block_attention_fwd_merge_cuda(
                    q_ptr as *const ffi::Half,
                    k_ptr as *const ffi::Half,
                    v_ptr as *const ffi::Half,
                    mi_ptr as *const f32,
                    li_ptr as *const f32,
                    oi_ptr as *const f32,
                    mo_ptr as *mut f32,
                    lo_ptr as *mut f32,
                    oo_ptr as *mut f32,
                    qpos_ptr as *const f32,
                    kpos_ptr as *const f32,
                    ring_i32(dims.num_q_tiles, "ring num_q_tiles i32")?,
                    ring_i32(dims.num_q_heads, "ring num_q_heads i32")?,
                    ring_i32(dims.num_kv_heads, "ring num_kv_heads i32")?,
                    ring_i32(dims.head_dim, "ring head_dim i32")?,
                    ring_i32(dims.q_rows, "ring q_rows i32")?,
                    ring_i32(dims.blk_len, "ring blk_len i32")?,
                    dims.sm_scale,
                    backend.stream.cu_stream(),
                )
            },
            "ring_block_attention_fwd_merge_cuda",
        )?;
    }
    Ok((
        DeviceHandle::Cuda(CudaStorage::new(m_out)),
        DeviceHandle::Cuda(CudaStorage::new(l_out)),
        DeviceHandle::Cuda(CudaStorage::new(o_out)),
    ))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_ring_block_finalize(
    backend: &CudaBackend,
    acc_m: &DeviceHandle,
    acc_l: &DeviceHandle,
    acc_o: &DeviceHandle,
    total_rows: usize,
    head_dim: usize,
) -> Result<(DeviceHandle, DeviceHandle)> {
    let m_slice = backend.cuda_slice(acc_m, "ring fin m")?;
    let l_slice = backend.cuda_slice(acc_l, "ring fin l")?;
    let o_slice = backend.cuda_slice(acc_o, "ring fin o")?;
    let out_len = total_rows * head_dim;
    let mut out_f32 = backend
        .stream
        .alloc_zeros::<f32>(out_len)
        .map_err(|_| cuda_alloc_failed("ring finalize out", vec![out_len]))?;
    let mut lse = backend
        .stream
        .alloc_zeros::<f32>(total_rows)
        .map_err(|_| cuda_alloc_failed("ring finalize lse", vec![total_rows]))?;
    {
        let (m_ptr, _mg) = m_slice.device_ptr(&backend.stream);
        let (l_ptr, _lg) = l_slice.device_ptr(&backend.stream);
        let (o_ptr, _og) = o_slice.device_ptr(&backend.stream);
        let (out_ptr, _outg) = out_f32.device_ptr_mut(&backend.stream);
        let (lse_ptr, _lseg) = lse.device_ptr_mut(&backend.stream);
        check_cuda_ffi(
            // SAFETY: acc_* are f32 handles length rows / rows*hd; out is rows*hd f32;
            // lse is rows f32; dims mirror the shapes.
            unsafe {
                ffi::ring_block_attention_finalize_cuda(
                    m_ptr as *const f32,
                    l_ptr as *const f32,
                    o_ptr as *const f32,
                    out_ptr as *mut f32,
                    lse_ptr as *mut f32,
                    ring_i32(total_rows, "ring total_rows i32")?,
                    ring_i32(head_dim, "ring head_dim i32")?,
                    backend.stream.cu_stream(),
                )
            },
            "ring_block_attention_finalize_cuda",
        )?;
    }
    Ok((
        DeviceHandle::Cuda(CudaStorage::new(out_f32)),
        DeviceHandle::Cuda(CudaStorage::new(lse)),
    ))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_ring_block_bwd(
    backend: &CudaBackend,
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
    let q_slice = backend.cuda_slice(q, "ring bwd q")?;
    let k_slice = backend.cuda_slice(k_blk, "ring bwd k")?;
    let v_slice = backend.cuda_slice(v_blk, "ring bwd v")?;
    let out_slice = backend.cuda_slice(out, "ring bwd out")?;
    let lse_slice = backend.cuda_slice(lse, "ring bwd lse")?;
    let do_slice = backend.cuda_slice(d_out, "ring bwd d_out")?;
    let gq_slice = backend.cuda_slice(grad_q, "ring bwd grad_q")?;
    let q_bf16 = backend.local_f32_as_bf16(q_slice, q_slice.len())?;
    let k_bf16 = backend.local_f32_as_bf16(k_slice, k_slice.len())?;
    let v_bf16 = backend.local_f32_as_bf16(v_slice, v_slice.len())?;
    let do_bf16 = backend.local_f32_as_bf16(do_slice, do_slice.len())?;
    // grad_q is in/out: copy the running accumulator so the kernel's += lands on a
    // fresh handle (the tape input stays immutable).
    let mut gq_out = backend
        .stream
        .alloc_zeros::<f32>(gq_slice.len())
        .map_err(|_| cuda_alloc_failed("ring grad_q out", vec![gq_slice.len()]))?;
    backend
        .stream
        .memcpy_dtod(gq_slice, &mut gq_out)
        .map_err(|_| AutogradError::TapeInvariant("cuda D2D copy failed (ring grad_q carry)"))?;
    let blk_elems = dims.num_q_tiles / (dims.num_q_heads / dims.num_kv_heads).max(1)
        * dims.blk_len
        * dims.head_dim;
    let mut gk = backend
        .stream
        .alloc_zeros::<f32>(blk_elems)
        .map_err(|_| cuda_alloc_failed("ring grad_k", vec![blk_elems]))?;
    let mut gv = backend
        .stream
        .alloc_zeros::<f32>(blk_elems)
        .map_err(|_| cuda_alloc_failed("ring grad_v", vec![blk_elems]))?;
    if ring_fa3_route(backend, dims, q_pos_host, k_pos_host) {
        return cuda_ring_block_bwd_fa3(
            backend, &q_bf16, &k_bf16, &v_bf16, out_slice, lse_slice, &do_bf16, gq_out, gk, gv,
            q_pos_host, k_pos_host, dims,
        );
    }
    let qpos_slice = backend.cuda_slice(q_pos, "ring bwd q_pos")?;
    let kpos_slice = backend.cuda_slice(k_pos, "ring bwd k_pos")?;
    {
        let (q_ptr, _qg) = q_bf16.device_ptr(&backend.stream);
        let (k_ptr, _kg) = k_bf16.device_ptr(&backend.stream);
        let (v_ptr, _vg) = v_bf16.device_ptr(&backend.stream);
        let (out_ptr, _og) = out_slice.device_ptr(&backend.stream);
        let (lse_ptr, _lg) = lse_slice.device_ptr(&backend.stream);
        let (do_ptr, _dg) = do_bf16.device_ptr(&backend.stream);
        let (gq_ptr, _gqg) = gq_out.device_ptr_mut(&backend.stream);
        let (gk_ptr, _gkg) = gk.device_ptr_mut(&backend.stream);
        let (gv_ptr, _gvg) = gv.device_ptr_mut(&backend.stream);
        let (qpos_ptr, _qpg) = qpos_slice.device_ptr(&backend.stream);
        let (kpos_ptr, _kpg) = kpos_slice.device_ptr(&backend.stream);
        check_cuda_ffi(
            // SAFETY: bf16 copies + f32 out/lse/grad handles of matching length; gk/gv
            // sized to one block's [Tkv, blk_len, hd]; q_pos/k_pos f32 of q_rows/blk_len;
            // dims mirror the shapes.
            unsafe {
                ffi::ring_block_attention_bwd_cuda(
                    q_ptr as *const ffi::Half,
                    k_ptr as *const ffi::Half,
                    v_ptr as *const ffi::Half,
                    out_ptr as *const f32,
                    lse_ptr as *const f32,
                    do_ptr as *const ffi::Half,
                    gq_ptr as *mut f32,
                    gk_ptr as *mut f32,
                    gv_ptr as *mut f32,
                    qpos_ptr as *const f32,
                    kpos_ptr as *const f32,
                    ring_i32(dims.num_q_tiles, "ring bwd num_q_tiles i32")?,
                    ring_i32(dims.num_q_heads, "ring bwd num_q_heads i32")?,
                    ring_i32(dims.num_kv_heads, "ring bwd num_kv_heads i32")?,
                    ring_i32(dims.head_dim, "ring bwd head_dim i32")?,
                    ring_i32(dims.q_rows, "ring bwd q_rows i32")?,
                    ring_i32(dims.blk_len, "ring bwd blk_len i32")?,
                    dims.sm_scale,
                    backend.stream.cu_stream(),
                )
            },
            "ring_block_attention_bwd_cuda",
        )?;
    }
    Ok((
        DeviceHandle::Cuda(CudaStorage::new(gq_out)),
        DeviceHandle::Cuda(CudaStorage::new(gk)),
        DeviceHandle::Cuda(CudaStorage::new(gv)),
    ))
}

// --- CP ring FA3 pair route (hd256 / sm_90) ---
// Decomposes the (q_pos, k_pos) block into rectangular FA3 calls per the run
// classification in ops/ring_attention.rs: each visible (q_run, k_run) pair is
// one non-varlen batch=1 FA3 launch over strided head-major views of the
// tile-major [tiles, seq, d] bf16 buffers (head_stride = seq*d, row_stride =
// d, run pointer offset = run_row*d). Forward merges each pair's normalized
// (o, lse) into the running accumulators as block stats (m = lse, l = 1);
// backward mirrors ring-flash-attention's global-LSE trick (final lse + final
// o + upstream d_out, per-pair causal flag). Scalar kernels remain the
// non-sm90 / non-hd256 / ARLE_CP_RING_FA3=0 fallback.

#[cfg(not(feature = "no-cuda"))]
fn ring_fa3_route(
    backend: &CudaBackend,
    dims: RingBlockDims,
    q_pos: &[usize],
    k_pos: &[usize],
) -> bool {
    // SAFETY: pure host query exported by both the real shim and the stub.
    let real = unsafe { ffi::arle_fa3_real_kernel_marker_cuda() } == 1;
    dims.head_dim == 256
        && q_pos.len() == dims.q_rows
        && k_pos.len() == dims.blk_len
        && crate::runtime_flags::cp_ring_fa3()
        && real
        && ring_fa3_is_sm90(backend)
}

/// The vendored FA3 units are sm_90a-only — dispatch strictly on 9.0.
#[cfg(not(feature = "no-cuda"))]
fn ring_fa3_is_sm90(backend: &CudaBackend) -> bool {
    use cudarc::driver::sys::CUdevice_attribute as Attr;
    let ctx = backend.stream.context();
    ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .is_ok_and(|v| v == 9)
        && ctx
            .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .is_ok_and(|v| v == 0)
}

#[cfg(not(feature = "no-cuda"))]
struct RingFa3Pair {
    q: crate::ops::ring_attention::PosRun,
    k: crate::ops::ring_attention::PosRun,
    causal: bool,
}

/// All visible (q_run, k_run) pairs, classified up front so a mis-aligned
/// shard errors loudly before any state is touched.
#[cfg(not(feature = "no-cuda"))]
fn ring_fa3_pairs(q_pos: &[usize], k_pos: &[usize]) -> Result<Vec<RingFa3Pair>> {
    use crate::ops::ring_attention::{PairClass, classify_pair, contiguous_pos_runs};
    let k_runs = contiguous_pos_runs(k_pos);
    let mut pairs = Vec::new();
    for q_run in contiguous_pos_runs(q_pos) {
        for &k_run in &k_runs {
            match classify_pair(q_run, k_run)? {
                PairClass::Full => pairs.push(RingFa3Pair {
                    q: q_run,
                    k: k_run,
                    causal: false,
                }),
                PairClass::Causal => pairs.push(RingFa3Pair {
                    q: q_run,
                    k: k_run,
                    causal: true,
                }),
                PairClass::Skip => {}
            }
        }
    }
    Ok(pairs)
}

/// Non-varlen causal metadata ceiling: `round_up(batch=1, 4) * 4 + 1` (see the
/// fwd shim's carve) — covers every pair class.
#[cfg(not(feature = "no-cuda"))]
const RING_FA3_FWD_METADATA_I32: usize = 17;

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_ring_block_fwd_merge_fa3(
    backend: &CudaBackend,
    q_bf16: &CudaSlice<u16>,
    k_bf16: &CudaSlice<u16>,
    v_bf16: &CudaSlice<u16>,
    m_in: &CudaSlice<f32>,
    l_in: &CudaSlice<f32>,
    o_in: &CudaSlice<f32>,
    q_pos: &[usize],
    k_pos: &[usize],
    dims: RingBlockDims,
) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
    let pairs = ring_fa3_pairs(q_pos, k_pos)?;
    let (h, hk, d, s, blk) = (
        dims.num_q_heads,
        dims.num_kv_heads,
        dims.head_dim,
        dims.q_rows,
        dims.blk_len,
    );
    let b = dims.num_q_tiles / h.max(1);
    let rows = dims.num_q_tiles * s;
    // Fresh accumulator copies (functional like the scalar merge); each pair
    // then rescales only its run rows in place.
    let mut m_out = backend
        .stream
        .alloc_zeros::<f32>(rows)
        .map_err(|_| cuda_alloc_failed("ring fa3 m_out", vec![rows]))?;
    let mut l_out = backend
        .stream
        .alloc_zeros::<f32>(rows)
        .map_err(|_| cuda_alloc_failed("ring fa3 l_out", vec![rows]))?;
    let mut o_out = backend
        .stream
        .alloc_zeros::<f32>(rows * d)
        .map_err(|_| cuda_alloc_failed("ring fa3 o_out", vec![rows * d]))?;
    let carry = AutogradError::TapeInvariant("cuda D2D copy failed (ring fa3 acc carry)");
    backend
        .stream
        .memcpy_dtod(m_in, &mut m_out)
        .map_err(|_| carry.clone())?;
    backend
        .stream
        .memcpy_dtod(l_in, &mut l_out)
        .map_err(|_| carry.clone())?;
    backend
        .stream
        .memcpy_dtod(o_in, &mut o_out)
        .map_err(|_| carry)?;

    for pair in &pairs {
        let (lq, lk) = (pair.q.len, pair.k.len);
        // Per-pair scratch, reused across the batch loop (stream-ordered).
        let mut o_pair = backend
            .stream
            .alloc_zeros::<u16>(h * lq * d)
            .map_err(|_| cuda_alloc_failed("ring fa3 o_pair", vec![h, lq, d]))?;
        let mut lse_pair = backend
            .stream
            .alloc_zeros::<f32>(h * lq)
            .map_err(|_| cuda_alloc_failed("ring fa3 lse_pair", vec![h, lq]))?;
        let mut meta = backend
            .stream
            .alloc_zeros::<i32>(RING_FA3_FWD_METADATA_I32)
            .map_err(|_| cuda_alloc_failed("ring fa3 meta", vec![RING_FA3_FWD_METADATA_I32]))?;
        for bi in 0..b {
            {
                let (q_ptr, _qg) = q_bf16.device_ptr(&backend.stream);
                let (k_ptr, _kg) = k_bf16.device_ptr(&backend.stream);
                let (v_ptr, _vg) = v_bf16.device_ptr(&backend.stream);
                let (op_ptr, _og) = o_pair.device_ptr_mut(&backend.stream);
                let (lse_ptr, _lg) = lse_pair.device_ptr_mut(&backend.stream);
                let (meta_ptr, _mg) = meta.device_ptr_mut(&backend.stream);
                let args = ffi::ArleFa3FwdHd256Args {
                    q: (q_ptr + (((bi * h * s) + pair.q.row) * d * 2) as u64) as *const ffi::Half,
                    k: (k_ptr + (((bi * hk * blk) + pair.k.row) * d * 2) as u64)
                        as *const ffi::Half,
                    v: (v_ptr + (((bi * hk * blk) + pair.k.row) * d * 2) as u64)
                        as *const ffi::Half,
                    o: op_ptr as *mut ffi::Half,
                    softmax_lse: lse_ptr as *mut f32,
                    out_accum: std::ptr::null_mut(),
                    softmax_lse_accum: std::ptr::null_mut(),
                    tile_count_semaphore: meta_ptr as *mut i32,
                    metadata_capacity: RING_FA3_FWD_METADATA_I32 as i32,
                    cu_seqlens_q: std::ptr::null(),
                    seqused_k: std::ptr::null(),
                    batch: 1,
                    total_q: ring_i32(lq, "ring fa3 total_q i32")?,
                    seqlen_q: ring_i32(lq, "ring fa3 seqlen_q i32")?,
                    seqlen_k: ring_i32(lk, "ring fa3 seqlen_k i32")?,
                    num_heads: ring_i32(h, "ring fa3 num_heads i32")?,
                    num_heads_k: ring_i32(hk, "ring fa3 num_heads_k i32")?,
                    head_dim: 256,
                    q_row_stride: d as i64,
                    k_row_stride: d as i64,
                    v_row_stride: d as i64,
                    o_row_stride: d as i64,
                    q_head_stride: (s * d) as i64,
                    k_head_stride: (blk * d) as i64,
                    v_head_stride: (blk * d) as i64,
                    o_head_stride: (lq * d) as i64,
                    softmax_scale: dims.sm_scale,
                    is_causal: pair.causal as i32,
                    num_splits: 1,
                    page_table: std::ptr::null(),
                    page_table_batch_stride: 0,
                    page_size: 0,
                    num_pages: 0,
                    k_page_stride: 0,
                    v_page_stride: 0,
                };
                check_cuda_ffi(
                    // SAFETY: strided head-major views of live bf16 buffers; o/lse
                    // are compact [h, lq(, d)] scratch; meta covers the shim's carve.
                    unsafe { ffi::arle_fa3_fwd_hd256_bf16_cuda(&args, backend.stream.cu_stream()) },
                    "arle_fa3_fwd_hd256_bf16_cuda",
                )?;
            }
            let (mo_ptr, _mog) = m_out.device_ptr_mut(&backend.stream);
            let (lo_ptr, _log) = l_out.device_ptr_mut(&backend.stream);
            let (oo_ptr, _oog) = o_out.device_ptr_mut(&backend.stream);
            let (op_ptr, _og2) = o_pair.device_ptr(&backend.stream);
            let (lse_ptr, _lg2) = lse_pair.device_ptr(&backend.stream);
            check_cuda_ffi(
                // SAFETY: acc buffers are full [tiles, s(, d)]; pair buffers compact.
                unsafe {
                    ffi::ring_fa3_merge_pair_cuda(
                        mo_ptr as *mut f32,
                        lo_ptr as *mut f32,
                        oo_ptr as *mut f32,
                        lse_ptr as *const f32,
                        op_ptr as *const ffi::Half,
                        ring_i32(h, "ring fa3 merge heads i32")?,
                        ring_i32(bi * h, "ring fa3 merge tile_base i32")?,
                        ring_i32(s, "ring fa3 merge seq i32")?,
                        ring_i32(pair.q.row, "ring fa3 merge run_start i32")?,
                        ring_i32(lq, "ring fa3 merge run_len i32")?,
                        ring_i32(d, "ring fa3 merge head_dim i32")?,
                        backend.stream.cu_stream(),
                    )
                },
                "ring_fa3_merge_pair_cuda",
            )?;
        }
    }
    Ok((
        DeviceHandle::Cuda(CudaStorage::new(m_out)),
        DeviceHandle::Cuda(CudaStorage::new(l_out)),
        DeviceHandle::Cuda(CudaStorage::new(o_out)),
    ))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_ring_block_bwd_fa3(
    backend: &CudaBackend,
    q_bf16: &CudaSlice<u16>,
    k_bf16: &CudaSlice<u16>,
    v_bf16: &CudaSlice<u16>,
    out_slice: &CudaSlice<f32>,
    lse_slice: &CudaSlice<f32>,
    do_bf16: &CudaSlice<u16>,
    mut gq_out: CudaSlice<f32>,
    mut gk: CudaSlice<f32>,
    mut gv: CudaSlice<f32>,
    q_pos: &[usize],
    k_pos: &[usize],
    dims: RingBlockDims,
) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
    let pairs = ring_fa3_pairs(q_pos, k_pos)?;
    let (h, hk, d, s, blk) = (
        dims.num_q_heads,
        dims.num_kv_heads,
        dims.head_dim,
        dims.q_rows,
        dims.blk_len,
    );
    let b = dims.num_q_tiles / h.max(1);
    let gqa = h != hk;
    // FA3 bwd wants o/d_out bf16; o is saved f32 — one quantize copy per call
    // (d_out was converted by the caller).
    let o_bf16 = backend.local_f32_as_bf16(out_slice, out_slice.len())?;

    for pair in &pairs {
        let (lq, lk) = (pair.q.len, pair.k.len);
        // hd256 sm90 bwd tiles: kBlockM=64 / kBlockN=80 (see the bwd shim).
        let sq_r = lq.div_ceil(64) * 64;
        let sk_r = lk.div_ceil(80) * 80;
        let mut lse_c = backend
            .stream
            .alloc_zeros::<f32>(h * lq)
            .map_err(|_| cuda_alloc_failed("ring fa3 bwd lse", vec![h, lq]))?;
        let mut dq = backend
            .stream
            .alloc_zeros::<u16>(h * lq * d)
            .map_err(|_| cuda_alloc_failed("ring fa3 dq", vec![h, lq, d]))?;
        let mut dk = backend
            .stream
            .alloc_zeros::<u16>(hk * lk * d)
            .map_err(|_| cuda_alloc_failed("ring fa3 dk", vec![hk, lk, d]))?;
        let mut dv = backend
            .stream
            .alloc_zeros::<u16>(hk * lk * d)
            .map_err(|_| cuda_alloc_failed("ring fa3 dv", vec![hk, lk, d]))?;
        let mut softmax_d = backend
            .stream
            .alloc_zeros::<f32>(h * sq_r)
            .map_err(|_| cuda_alloc_failed("ring fa3 softmax_d", vec![h, sq_r]))?;
        let mut lse_log2 = backend
            .stream
            .alloc_zeros::<f32>(h * sq_r)
            .map_err(|_| cuda_alloc_failed("ring fa3 lse_log2", vec![h, sq_r]))?;
        let mut dq_accum = backend
            .stream
            .alloc_zeros::<f32>(h * sq_r * 256)
            .map_err(|_| cuda_alloc_failed("ring fa3 dq_accum", vec![h, sq_r, 256]))?;
        let mut dkv_accum = if gqa {
            let dk_a = backend
                .stream
                .alloc_zeros::<f32>(hk * sk_r * 256)
                .map_err(|_| cuda_alloc_failed("ring fa3 dk_accum", vec![hk, sk_r, 256]))?;
            let dv_a = backend
                .stream
                .alloc_zeros::<f32>(hk * sk_r * 256)
                .map_err(|_| cuda_alloc_failed("ring fa3 dv_accum", vec![hk, sk_r, 256]))?;
            Some((dk_a, dv_a))
        } else {
            None
        };
        let dq_sem_len = lq.div_ceil(64) * h;
        let mut dq_sem = backend
            .stream
            .alloc_zeros::<i32>(dq_sem_len)
            .map_err(|_| cuda_alloc_failed("ring fa3 dq_semaphore", vec![dq_sem_len]))?;
        for bi in 0..b {
            {
                let (lse_ptr, _lg) = lse_slice.device_ptr(&backend.stream);
                let (lsec_ptr, _lcg) = lse_c.device_ptr_mut(&backend.stream);
                check_cuda_ffi(
                    // SAFETY: lse is full [tiles, s]; dst is compact [h, lq].
                    unsafe {
                        ffi::ring_fa3_gather_lse_cuda(
                            lsec_ptr as *mut f32,
                            lse_ptr as *const f32,
                            ring_i32(h, "ring fa3 gather heads i32")?,
                            ring_i32(bi * h, "ring fa3 gather tile_base i32")?,
                            ring_i32(s, "ring fa3 gather seq i32")?,
                            ring_i32(pair.q.row, "ring fa3 gather run_start i32")?,
                            ring_i32(lq, "ring fa3 gather run_len i32")?,
                            backend.stream.cu_stream(),
                        )
                    },
                    "ring_fa3_gather_lse_cuda",
                )?;
            }
            {
                let (q_ptr, _qg) = q_bf16.device_ptr(&backend.stream);
                let (k_ptr, _kg) = k_bf16.device_ptr(&backend.stream);
                let (v_ptr, _vg) = v_bf16.device_ptr(&backend.stream);
                let (o_ptr, _og) = o_bf16.device_ptr(&backend.stream);
                let (do_ptr, _dg) = do_bf16.device_ptr(&backend.stream);
                let (lsec_ptr, _lcg) = lse_c.device_ptr(&backend.stream);
                let (dq_ptr, _dqg) = dq.device_ptr_mut(&backend.stream);
                let (dk_ptr, _dkg) = dk.device_ptr_mut(&backend.stream);
                let (dv_ptr, _dvg) = dv.device_ptr_mut(&backend.stream);
                let (sd_ptr, _sdg) = softmax_d.device_ptr_mut(&backend.stream);
                let (ll2_ptr, _llg) = lse_log2.device_ptr_mut(&backend.stream);
                let (dqa_ptr, _dqag) = dq_accum.device_ptr_mut(&backend.stream);
                let (dka_ptr, dva_ptr, _dkag, _dvag) = match dkv_accum.as_mut() {
                    Some((dk_a, dv_a)) => {
                        let (dka, dkag) = dk_a.device_ptr_mut(&backend.stream);
                        let (dva, dvag) = dv_a.device_ptr_mut(&backend.stream);
                        (dka as *mut f32, dva as *mut f32, Some(dkag), Some(dvag))
                    }
                    None => (std::ptr::null_mut(), std::ptr::null_mut(), None, None),
                };
                let (sem_ptr, _semg) = dq_sem.device_ptr_mut(&backend.stream);
                let q_off = (((bi * h * s) + pair.q.row) * d * 2) as u64;
                let kv_off = (((bi * hk * blk) + pair.k.row) * d * 2) as u64;
                let args = ffi::ArleFa3BwdHd256Args {
                    q: (q_ptr + q_off) as *const ffi::Half,
                    k: (k_ptr + kv_off) as *const ffi::Half,
                    v: (v_ptr + kv_off) as *const ffi::Half,
                    o: (o_ptr + q_off) as *const ffi::Half,
                    dout: (do_ptr + q_off) as *const ffi::Half,
                    softmax_lse: lsec_ptr as *const f32,
                    dq: dq_ptr as *mut ffi::Half,
                    dk: dk_ptr as *mut ffi::Half,
                    dv: dv_ptr as *mut ffi::Half,
                    softmax_d: sd_ptr as *mut f32,
                    softmax_lse_log2: ll2_ptr as *mut f32,
                    dq_accum: dqa_ptr as *mut f32,
                    dk_accum: dka_ptr,
                    dv_accum: dva_ptr,
                    dq_semaphore: sem_ptr as *mut i32,
                    softmax_d_capacity: (h * sq_r) as i64,
                    softmax_lse_log2_capacity: (h * sq_r) as i64,
                    dq_accum_capacity: (h * sq_r * 256) as i64,
                    dk_accum_capacity: if gqa { (hk * sk_r * 256) as i64 } else { 0 },
                    dv_accum_capacity: if gqa { (hk * sk_r * 256) as i64 } else { 0 },
                    dq_semaphore_capacity: dq_sem_len as i64,
                    seqlen_q: ring_i32(lq, "ring fa3 bwd seqlen_q i32")?,
                    seqlen_k: ring_i32(lk, "ring fa3 bwd seqlen_k i32")?,
                    num_heads: ring_i32(h, "ring fa3 bwd num_heads i32")?,
                    num_heads_k: ring_i32(hk, "ring fa3 bwd num_heads_k i32")?,
                    head_dim: 256,
                    q_row_stride: d as i64,
                    k_row_stride: d as i64,
                    v_row_stride: d as i64,
                    o_row_stride: d as i64,
                    do_row_stride: d as i64,
                    dq_row_stride: d as i64,
                    dk_row_stride: d as i64,
                    dv_row_stride: d as i64,
                    q_head_stride: (s * d) as i64,
                    k_head_stride: (blk * d) as i64,
                    v_head_stride: (blk * d) as i64,
                    o_head_stride: (s * d) as i64,
                    do_head_stride: (s * d) as i64,
                    dq_head_stride: (lq * d) as i64,
                    dk_head_stride: (lk * d) as i64,
                    dv_head_stride: (lk * d) as i64,
                    softmax_scale: dims.sm_scale,
                    is_causal: pair.causal as i32,
                };
                check_cuda_ffi(
                    // SAFETY: strided head-major views of live bf16 buffers; compact
                    // dq/dk/dv outputs; scratch sized per the bwd shim's contract
                    // (round_up 64/80 tiles, GQA fp32 accum, dq semaphore).
                    unsafe { ffi::arle_fa3_bwd_hd256_bf16_cuda(&args, backend.stream.cu_stream()) },
                    "arle_fa3_bwd_hd256_bf16_cuda",
                )?;
            }
            // Accumulate the pair's compact bf16 grads into the running f32
            // buffers (dq += into gq carry; dk/dv += into this block's grads).
            let accum = |dst: &mut CudaSlice<f32>,
                         src: &CudaSlice<u16>,
                         heads: usize,
                         tile_base: usize,
                         seq: usize,
                         run: &crate::ops::ring_attention::PosRun|
             -> Result<()> {
                let (dst_ptr, _dg) = dst.device_ptr_mut(&backend.stream);
                let (src_ptr, _sg) = src.device_ptr(&backend.stream);
                check_cuda_ffi(
                    // SAFETY: dst full [tiles, seq, d]; src compact [heads, run.len, d].
                    unsafe {
                        ffi::ring_fa3_accum_grad_bf16_cuda(
                            dst_ptr as *mut f32,
                            src_ptr as *const ffi::Half,
                            ring_i32(heads, "ring fa3 accum heads i32")?,
                            ring_i32(tile_base, "ring fa3 accum tile_base i32")?,
                            ring_i32(seq, "ring fa3 accum seq i32")?,
                            ring_i32(run.row, "ring fa3 accum run_start i32")?,
                            ring_i32(run.len, "ring fa3 accum run_len i32")?,
                            ring_i32(d, "ring fa3 accum head_dim i32")?,
                            backend.stream.cu_stream(),
                        )
                    },
                    "ring_fa3_accum_grad_bf16_cuda",
                )
            };
            accum(&mut gq_out, &dq, h, bi * h, s, &pair.q)?;
            accum(&mut gk, &dk, hk, bi * hk, blk, &pair.k)?;
            accum(&mut gv, &dv, hk, bi * hk, blk, &pair.k)?;
        }
    }
    Ok((
        DeviceHandle::Cuda(CudaStorage::new(gq_out)),
        DeviceHandle::Cuda(CudaStorage::new(gk)),
        DeviceHandle::Cuda(CudaStorage::new(gv)),
    ))
}

// Fused causal prefill SDPA — the production inference kernel
// (`nonpaged_prefill_attention_cuda`: bf16, online softmax, GQA native)
// adopted for the training forward. Layout bridge: training q `[1, h, s, d]`
// transposes to the kernel's token-major `[s, h, d]`; k/v `[1, h_kv, kv, d]`
// contiguous already match its head-major cache view (`max_seq_len = kv`).
#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_causal_sdpa_prefill_device(
    backend: &CudaBackend,
    q: &DeviceHandle,
    q_shape: &[usize],
    k: &DeviceHandle,
    k_shape: &[usize],
    v: &DeviceHandle,
    v_shape: &[usize],
    q_start: usize,
) -> Result<Option<(DeviceHandle, Vec<usize>)>> {
    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape != k_shape {
        return Ok(None);
    }
    let (batch, heads, seq, dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (kv_heads, kv_len) = (k_shape[1], k_shape[2]);
    // Kernel envelope: head_dim ∈ {128, 256}, grid.y = seq ≤ 65535, exact
    // causal window, whole GQA groups. Outside it => compose from primitives.
    if batch != 1
        || k_shape[0] != 1
        || k_shape[3] != dim
        || !(dim == 128 || dim == 256)
        || seq > 65_535
        || kv_heads == 0
        || heads % kv_heads != 0
        || q_start + seq != kv_len
    {
        return Ok(None);
    }

    let (q_t, _) = backend.transpose_axes_swap(q, q_shape, 1, 2)?; // [1, s, h, d]
    let q_slice = backend.cuda_slice(&q_t, "sdpa_prefill q")?;
    let k_slice = backend.cuda_slice(k, "sdpa_prefill k")?;
    let v_slice = backend.cuda_slice(v, "sdpa_prefill v")?;
    let q_bf16 = backend.local_f32_as_bf16(q_slice, q_slice.len())?;
    let k_bf16 = backend.local_f32_as_bf16(k_slice, k_slice.len())?;
    let v_bf16 = backend.local_f32_as_bf16(v_slice, v_slice.len())?;
    let out_len = seq * heads * dim;
    let mut out_bf16 = backend
        .stream
        .alloc_zeros::<u16>(out_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sdpa prefill out)"))?;

    let as_i32 = |value: usize, label: &'static str| {
        i32::try_from(value).map_err(|_| AutogradError::TapeInvariant(label))
    };
    let heads_i32 = as_i32(heads, "sdpa prefill heads exceeds i32")?;
    let kv_heads_i32 = as_i32(kv_heads, "sdpa prefill kv_heads exceeds i32")?;
    let dim_i32 = as_i32(dim, "sdpa prefill head_dim exceeds i32")?;
    let seq_i32 = as_i32(seq, "sdpa prefill seq exceeds i32")?;
    let kv_i32 = as_i32(kv_len, "sdpa prefill kv_len exceeds i32")?;
    {
        let (q_ptr, _q_guard) = q_bf16.device_ptr(&backend.stream);
        let (k_ptr, _k_guard) = k_bf16.device_ptr(&backend.stream);
        let (v_ptr, _v_guard) = v_bf16.device_ptr(&backend.stream);
        let (out_ptr, _out_guard) = out_bf16.device_ptr_mut(&backend.stream);
        check_cuda_ffi(
            // SAFETY: q/k/v are live guarded bf16 copies of tape tensors whose shapes passed the
            // envelope check; out is allocated seq*heads*dim; the dims passed mirror those shapes.
            unsafe {
                ffi::nonpaged_prefill_attention_cuda(
                    q_ptr as *const ffi::Half,
                    k_ptr as *const ffi::Half,
                    v_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    heads_i32,
                    kv_heads_i32,
                    dim_i32,
                    seq_i32,
                    kv_i32,
                    kv_i32,
                    (dim as f32).sqrt().recip(),
                    backend.stream.cu_stream(),
                )
            },
            "nonpaged_prefill_attention_cuda",
        )?;
    }

    // Trace-gated sync: pin an async kernel fault here, not at the next alloc.
    if std::env::var("ARLE_SDPA_TRACE").is_ok() {
        backend.stream.synchronize().map_err(|e| {
            leak_err(format!(
                "sdpa fused async fault: seq={seq} dim={dim} err={e:?}"
            ))
        })?;
    }

    let out_f32 = backend.import_local_bf16_as_f32(&out_bf16, out_len)?;
    let out_handle = DeviceHandle::Cuda(CudaStorage::new(out_f32));
    let (out, out_shape) = backend.transpose_axes_swap(&out_handle, &[1, seq, heads, dim], 1, 2)?; // [1, h, s, d]
    Ok(Some((out, out_shape)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_row_len(len: usize, rows: usize) -> Result<usize> {
    if rows == 0 || !len.is_multiple_of(rows) {
        return Err(AutogradError::TapeInvariant(
            "linear_attention batched dispatch len not divisible by batch",
        ));
    }
    Ok(len / rows)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_copy_range<T: DeviceRepr>(
    backend: &CudaBackend,
    src: &CudaSlice<T>,
    start: usize,
    len: usize,
) -> Result<CudaSlice<T>> {
    // SAFETY: every byte is overwritten by the D2D copy below.
    let mut out = unsafe { backend.stream.alloc::<T>(len) }
        .map_err(|_| cuda_alloc_failed("la row slice", vec![len]))?;
    backend
        .stream
        .memcpy_dtod(&src.slice(start..start + len), &mut out)
        .map_err(|_| AutogradError::TapeInvariant("cuda D2D copy failed (la row slice)"))?;
    Ok(out)
}

/// Concat N f32 device tensors along `axis`, on-device. `cuda_concat_parts` only
/// does the outermost axis, so transpose `axis`→0, concat, transpose back
/// (identity when axis==0). Shared by `all_to_all_device` assembly and `concat`.
#[cfg(not(feature = "no-cuda"))]
fn cuda_concat_axis(
    backend: &CudaBackend,
    parts: &[(&DeviceHandle, &[usize])],
    axis: usize,
) -> Result<(DeviceHandle, Vec<usize>)> {
    let transposed: Vec<(DeviceHandle, Vec<usize>)> = parts
        .iter()
        .map(|(h, s)| backend.transpose_axes_swap(h, s, 0, axis))
        .collect::<Result<_>>()?;
    let slices: Vec<&CudaSlice<f32>> = transposed
        .iter()
        .map(|(h, _)| backend.cuda_slice(h, "concat"))
        .collect::<Result<_>>()?;
    let cat = cuda_concat_parts(backend, &slices)?;
    let mut cat_shape = transposed[0].1.clone();
    cat_shape[0] = transposed.iter().map(|(_, s)| s[0]).sum();
    let cat_h = DeviceHandle::Cuda(CudaStorage::new(cat));
    backend.transpose_axes_swap(&cat_h, &cat_shape, 0, axis)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_concat_parts<T: DeviceRepr>(
    backend: &CudaBackend,
    parts: &[&CudaSlice<T>],
) -> Result<CudaSlice<T>> {
    let total: usize = parts.iter().map(|part| part.len()).sum();
    // SAFETY: the copies below cover the full buffer.
    let mut out = unsafe { backend.stream.alloc::<T>(total) }
        .map_err(|_| cuda_alloc_failed("la row concat", vec![total]))?;
    let mut offset = 0;
    for part in parts {
        backend
            .stream
            .memcpy_dtod(*part, &mut out.slice_mut(offset..offset + part.len()))
            .map_err(|_| AutogradError::TapeInvariant("cuda D2D copy failed (la row concat)"))?;
        offset += part.len();
    }
    Ok(out)
}

/// Copy batch row `row` of a batch-leading device tensor into a fresh
/// device buffer of the same dtype (row length inferred as `len / rows`).
#[cfg(not(feature = "no-cuda"))]
fn cuda_row_slice(
    backend: &CudaBackend,
    src: &DeviceHandle,
    row: usize,
    rows: usize,
) -> Result<DeviceHandle> {
    match src {
        DeviceHandle::Cuda(storage) => {
            let src = backend.cuda_storage_slice(storage)?;
            let row_len = cuda_row_len(src.len(), rows)?;
            let out = cuda_copy_range(backend, src, row * row_len, row_len)?;
            Ok(DeviceHandle::Cuda(CudaStorage::new(out)))
        }
        DeviceHandle::CudaBf16(storage) => {
            let src = backend.cuda_bf16_storage_slice(storage)?;
            let row_len = cuda_row_len(src.len(), rows)?;
            let out = cuda_copy_range(backend, src, row * row_len, row_len)?;
            Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(out)))
        }
        _ => Err(AutogradError::TapeInvariant(
            "linear_attention batched dispatch expects f32/bf16 cuda handles",
        )),
    }
}

/// Concatenate same-dtype device rows into one contiguous batch buffer.
#[cfg(not(feature = "no-cuda"))]
fn cuda_concat_rows(backend: &CudaBackend, rows: &[&DeviceHandle]) -> Result<DeviceHandle> {
    match rows.first() {
        Some(DeviceHandle::Cuda(_)) => {
            let parts = rows
                .iter()
                .map(|handle| backend.cuda_slice(handle, "la row concat"))
                .collect::<Result<Vec<_>>>()?;
            let out = cuda_concat_parts(backend, &parts)?;
            Ok(DeviceHandle::Cuda(CudaStorage::new(out)))
        }
        Some(DeviceHandle::CudaBf16(_)) => {
            let parts = rows
                .iter()
                .map(|handle| backend.cuda_bf16_slice(handle, "la row concat"))
                .collect::<Result<Vec<_>>>()?;
            let out = cuda_concat_parts(backend, &parts)?;
            Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(out)))
        }
        _ => Err(AutogradError::TapeInvariant(
            "linear_attention batched concat expects f32/bf16 cuda handles",
        )),
    }
}

/// Shape gate for the device LA path. Only the head DIM is baked into the
/// kernels (GDR_KEY_DIM/VAL_DIM=128; num_value_heads stays a runtime param);
/// batch>1 rides per-row dispatch because the chunked kernels' chunk_state
/// carries no batch stride.
#[cfg(not(feature = "no-cuda"))]
fn cuda_la_device_supported(p: LinearAttentionDeviceParams) -> bool {
    p.key_dim == 128 && p.value_dim == 128 && p.conv_kernel > 0 && p.conv_kernel <= 5 && p.batch > 0
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_linear_attention_forward_device(
    backend: &CudaBackend,
    args: LinearAttentionDeviceForwardArgs<'_>,
) -> Result<Option<LinearAttentionDeviceForwardResult>> {
    let p = args.params;
    if !cuda_la_device_supported(p) {
        return Ok(None);
    }
    if p.batch == 1 {
        return cuda_linear_attention_forward_device_row(backend, args).map(Some);
    }

    // batch > 1: per-row dispatch to the proven batch==1 path. Every input and
    // result tensor is batch-leading, so row slicing and reassembly are
    // contiguous-range D2D copies; weights pass through whole.
    let row_params = LinearAttentionDeviceParams { batch: 1, ..p };
    let rows = (0..p.batch)
        .map(|row| {
            let slice = |src| cuda_row_slice(backend, src, row, p.batch);
            // Carry is batch-leading like the other inputs — slice per row too.
            let initial_state = args.initial_state.map(slice).transpose()?;
            let initial_conv_window = args.initial_conv_window.map(slice).transpose()?;
            cuda_linear_attention_forward_device_row(
                backend,
                LinearAttentionDeviceForwardArgs {
                    params: row_params,
                    qkv: &slice(args.qkv)?,
                    z: &slice(args.z)?,
                    b_proj: &slice(args.b_proj)?,
                    a_proj: &slice(args.a_proj)?,
                    conv1d_weight: args.conv1d_weight,
                    dt_bias: args.dt_bias,
                    a_log: args.a_log,
                    norm_weight: args.norm_weight,
                    initial_state: initial_state.as_ref(),
                    initial_conv_window: initial_conv_window.as_ref(),
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let concat = |field: fn(&LinearAttentionDeviceForwardResult) -> &DeviceHandle| {
        let parts: Vec<&DeviceHandle> = rows.iter().map(field).collect();
        cuda_concat_rows(backend, &parts)
    };
    Ok(Some(LinearAttentionDeviceForwardResult {
        output: concat(|r| &r.output)?,
        preact: concat(|r| &r.preact)?,
        qkv_conv: concat(|r| &r.qkv_conv)?,
        q: concat(|r| &r.q)?,
        k: concat(|r| &r.k)?,
        v: concat(|r| &r.v)?,
        g: concat(|r| &r.g)?,
        g_cumsum: concat(|r| &r.g_cumsum)?,
        beta: concat(|r| &r.beta)?,
        a_inv: concat(|r| &r.a_inv)?,
        chunk_state: concat(|r| &r.chunk_state)?,
        raw_output: concat(|r| &r.raw_output)?,
    }))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_linear_attention_boundary_device(
    backend: &CudaBackend,
    args: LinearAttentionDeviceBoundaryArgs<'_>,
) -> Result<Option<DeviceHandle>> {
    let p = args.params;
    if !cuda_la_device_supported(p) {
        return Ok(None);
    }
    if p.batch == 1 {
        return cuda_linear_attention_boundary_device_row(backend, args).map(Some);
    }

    let row_params = LinearAttentionDeviceParams { batch: 1, ..p };
    let rows = (0..p.batch)
        .map(|row| {
            let slice = |src| cuda_row_slice(backend, src, row, p.batch);
            let qkv = slice(args.qkv)?;
            let b_proj = slice(args.b_proj)?;
            let a_proj = slice(args.a_proj)?;
            let initial_state = args.initial_state.map(slice).transpose()?;
            let initial_conv_window = args.initial_conv_window.map(slice).transpose()?;
            cuda_linear_attention_boundary_device_row(
                backend,
                LinearAttentionDeviceBoundaryArgs {
                    params: row_params,
                    qkv: &qkv,
                    b_proj: &b_proj,
                    a_proj: &a_proj,
                    conv1d_weight: args.conv1d_weight,
                    dt_bias: args.dt_bias,
                    a_log: args.a_log,
                    initial_state: initial_state.as_ref(),
                    initial_conv_window: initial_conv_window.as_ref(),
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let parts = rows.iter().collect::<Vec<_>>();
    cuda_concat_rows(backend, &parts).map(Some)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_linear_attention_boundary_device_row(
    backend: &CudaBackend,
    args: LinearAttentionDeviceBoundaryArgs<'_>,
) -> Result<DeviceHandle> {
    let p = args.params;
    debug_assert_eq!(p.batch, 1);
    let q_dim = p.num_key_heads * p.key_dim;
    let qkv_dim = q_dim * 2 + p.num_value_heads * p.value_dim;
    let qkv_len = p.seq_len * qkv_dim;
    let head_len = p.seq_len * p.num_value_heads;
    let state_len = p.num_value_heads * p.key_dim * p.value_dim;
    let conv_tail_len = p.conv_kernel - 1;
    let qkv = backend.cuda_slice(args.qkv, "linear_attention_boundary qkv")?;
    let b_proj = backend.cuda_slice(args.b_proj, "linear_attention_boundary b_proj")?;
    let a_proj = backend.cuda_slice(args.a_proj, "linear_attention_boundary a_proj")?;
    let conv1d_weight = backend.cuda_slice(
        args.conv1d_weight,
        "linear_attention_boundary conv1d_weight",
    )?;
    let dt_bias = backend.cuda_slice(args.dt_bias, "linear_attention_boundary dt_bias")?;
    let a_log = backend.cuda_slice(args.a_log, "linear_attention_boundary a_log")?;
    let carry_state = args
        .initial_state
        .map(|h| backend.cuda_slice(h, "linear_attention_boundary initial_state"))
        .transpose()?;
    let carry_conv = args
        .initial_conv_window
        .map(|h| backend.cuda_slice(h, "linear_attention_boundary initial_conv_window"))
        .transpose()?;

    for (got, expected) in [
        (qkv.len(), qkv_len),
        (b_proj.len(), head_len),
        (a_proj.len(), head_len),
        (conv1d_weight.len(), qkv_dim * p.conv_kernel),
        (dt_bias.len(), p.num_value_heads),
        (a_log.len(), p.num_value_heads),
    ] {
        if got != expected {
            return Err(AutogradError::TapeInvariant(
                "linear_attention_boundary input length mismatch",
            ));
        }
    }
    if carry_state.is_some_and(|x| x.len() != state_len)
        || carry_conv.is_some_and(|x| x.len() != conv_tail_len * qkv_dim)
    {
        return Err(AutogradError::TapeInvariant(
            "linear_attention_boundary carry length mismatch",
        ));
    }

    let b_bf16 = backend.local_f32_as_bf16(b_proj, head_len)?;
    let a_bf16 = backend.local_f32_as_bf16(a_proj, head_len)?;
    let dt_bf16 = backend.local_f32_as_bf16(dt_bias, p.num_value_heads)?;
    let chunk_rows = p.seq_len.min(64);
    let mut qkv_chunk = backend
        .stream
        .alloc_zeros::<u16>(chunk_rows * qkv_dim)
        .map_err(|_| cuda_alloc_failed("la boundary qkv", vec![chunk_rows, qkv_dim]))?;
    let mut raw_chunk = backend
        .stream
        .alloc_zeros::<u16>(chunk_rows * p.num_value_heads * p.value_dim)
        .map_err(|_| {
            cuda_alloc_failed(
                "la boundary raw",
                vec![chunk_rows, p.num_value_heads, p.value_dim],
            )
        })?;
    let mut state = backend
        .stream
        .alloc_zeros::<f32>(state_len)
        .map_err(|_| cuda_alloc_failed("la boundary state", vec![state_len]))?;
    if let Some(initial) = carry_state {
        backend
            .stream
            .memcpy_dtod(initial, &mut state)
            .map_err(|_| AutogradError::TapeInvariant("la boundary state seed failed"))?;
    }

    let qkv_dim_i32 = i32::try_from(qkv_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention qkv_dim exceeds i32"))?;
    let conv_kernel_i32 = i32::try_from(p.conv_kernel)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention conv_kernel exceeds i32"))?;
    let num_key_heads_i32 = i32::try_from(p.num_key_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key heads exceeds i32"))?;
    let num_value_heads_i32 = i32::try_from(p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value heads exceeds i32"))?;
    let key_dim_i32 = i32::try_from(p.key_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key_dim exceeds i32"))?;
    let value_dim_i32 = i32::try_from(p.value_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value_dim exceeds i32"))?;
    let conv_tail = carry_conv.map(|x| x.device_ptr(&backend.stream));
    let conv_tail_ptr = conv_tail.as_ref().map_or(0u64, |(ptr, _)| *ptr);
    let conv_tail_len_i32 = if carry_conv.is_some() {
        i32::try_from(conv_tail_len)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention conv tail exceeds i32"))?
    } else {
        0
    };

    for start in (0..p.seq_len).step_by(64) {
        let rows = (p.seq_len - start).min(64);
        let head_start = start * p.num_value_heads;
        let total = rows * qkv_dim;
        let total_u64 = total as u64;
        let start_i32 = i32::try_from(start)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention start exceeds i32"))?;
        {
            let (out_ptr, _out_guard) = qkv_chunk.device_ptr_mut(&backend.stream);
            let (qkv_ptr, _qkv_guard) = qkv.device_ptr(&backend.stream);
            let (conv_ptr, _conv_guard) = conv1d_weight.device_ptr(&backend.stream);
            launch_1d(
                &backend.stream,
                backend
                    .kernels
                    .function("linear_attention_conv1d_silu_boundary_f32_to_bf16")?,
                total,
                |mut builder| {
                    builder
                        .arg(&out_ptr)
                        .arg(&qkv_ptr)
                        .arg(&conv_ptr)
                        .arg(&total_u64)
                        .arg(&start_i32)
                        .arg(&qkv_dim_i32)
                        .arg(&conv_kernel_i32)
                        .arg(&conv_tail_ptr)
                        .arg(&conv_tail_len_i32);
                    builder
                },
            )?;
        }
        let rows_i32 = i32::try_from(rows)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention rows exceeds i32"))?;
        let (qkv_ptr, _qkv_guard) = qkv_chunk.device_ptr(&backend.stream);
        let (b_ptr, _b_guard) = b_bf16.device_ptr(&backend.stream);
        let (a_ptr, _a_guard) = a_bf16.device_ptr(&backend.stream);
        let (dt_ptr, _dt_guard) = dt_bf16.device_ptr(&backend.stream);
        let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
        let (state_ptr, _state_guard) = state.device_ptr_mut(&backend.stream);
        let (raw_ptr, _raw_guard) = raw_chunk.device_ptr_mut(&backend.stream);
        check_cuda_ffi(
            // SAFETY: all pointers cover the dimensions passed below.
            unsafe {
                ffi::gated_delta_rule_prefill_recurrent_cuda(
                    qkv_ptr as *const ffi::Half,
                    (b_ptr as *const ffi::Half).add(head_start),
                    (a_ptr as *const ffi::Half).add(head_start),
                    dt_ptr as *const ffi::Half,
                    a_log_ptr as *const f32,
                    state_ptr as *mut f32,
                    raw_ptr as *mut ffi::Half,
                    num_key_heads_i32,
                    num_value_heads_i32,
                    key_dim_i32,
                    value_dim_i32,
                    rows_i32,
                    backend.stream.cu_stream(),
                )
            },
            "gated_delta_rule_prefill_recurrent_cuda",
        )?;
    }
    Ok(DeviceHandle::Cuda(CudaStorage::new(state)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_linear_attention_forward_device_row(
    backend: &CudaBackend,
    args: LinearAttentionDeviceForwardArgs<'_>,
) -> Result<LinearAttentionDeviceForwardResult> {
    let p = args.params;
    debug_assert_eq!(p.batch, 1);
    let q_dim = p.num_key_heads * p.key_dim;
    let qkv_dim = q_dim * 2 + p.num_value_heads * p.value_dim;
    let qkv_len = p.batch * p.seq_len * qkv_dim;
    let z_len = p.batch * p.seq_len * p.num_value_heads * p.value_dim;
    let head_len = p.batch * p.seq_len * p.num_value_heads;
    let conv_len = qkv_dim * p.conv_kernel;
    let num_chunks = p.seq_len.div_ceil(64);
    let q_len = p.seq_len * p.num_value_heads * p.key_dim;
    let v_len = p.seq_len * p.num_value_heads * p.value_dim;
    let a_len = p.seq_len * p.num_value_heads * 64;
    let state_len = p.num_value_heads * p.key_dim * p.value_dim;
    let chunk_state_len = num_chunks * state_len;

    let qkv = backend.cuda_slice(args.qkv, "linear_attention_forward qkv")?;
    let z = backend.cuda_slice(args.z, "linear_attention_forward z")?;
    let b_proj = backend.cuda_slice(args.b_proj, "linear_attention_forward b_proj")?;
    let a_proj = backend.cuda_slice(args.a_proj, "linear_attention_forward a_proj")?;
    let conv1d_weight =
        backend.cuda_slice(args.conv1d_weight, "linear_attention_forward conv1d_weight")?;
    let dt_bias = backend.cuda_slice(args.dt_bias, "linear_attention_forward dt_bias")?;
    let a_log = backend.cuda_slice(args.a_log, "linear_attention_forward a_log")?;
    let norm_weight =
        backend.cuda_slice(args.norm_weight, "linear_attention_forward norm_weight")?;

    // OPD carry (None = default zero-seed). initial_state seeds final_state (→ chunk_state[0]);
    // conv_tail feeds the conv1d boundary taps. tail_len = conv_kernel-1 rows of qkv_dim channels.
    let conv_tail_len = p.conv_kernel - 1;
    let carry_state = args
        .initial_state
        .map(|h| backend.cuda_slice(h, "linear_attention_forward initial_state"))
        .transpose()?;
    let carry_conv = args
        .initial_conv_window
        .map(|h| backend.cuda_slice(h, "linear_attention_forward initial_conv_window"))
        .transpose()?;

    for (label, got, expected) in [
        ("qkv", Some(qkv.len()), qkv_len),
        ("z", Some(z.len()), z_len),
        ("b_proj", Some(b_proj.len()), head_len),
        ("a_proj", Some(a_proj.len()), head_len),
        ("conv1d_weight", Some(conv1d_weight.len()), conv_len),
        ("dt_bias", Some(dt_bias.len()), p.num_value_heads),
        ("a_log", Some(a_log.len()), p.num_value_heads),
        ("norm_weight", Some(norm_weight.len()), p.value_dim),
        ("initial_state", carry_state.map(|s| s.len()), state_len),
        (
            "initial_conv_window",
            carry_conv.map(|s| s.len()),
            conv_tail_len * qkv_dim,
        ),
    ] {
        if let Some(got) = got
            && got != expected
        {
            return Err(AutogradError::TapeInvariant(Box::leak(
                format!(
                    "cuda linear_attention_forward_device {label} len mismatch: got={got} expected={expected}"
                )
                .into_boxed_str(),
            )));
        }
    }

    let b_bf16 = backend.local_f32_as_bf16(b_proj, head_len)?;
    let a_bf16 = backend.local_f32_as_bf16(a_proj, head_len)?;
    let dt_bf16 = backend.local_f32_as_bf16(dt_bias, p.num_value_heads)?;

    let mut preact = backend
        .stream
        .alloc_zeros::<f32>(qkv_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la preact)"))?;
    let mut qkv_conv = backend
        .stream
        .alloc_zeros::<u16>(qkv_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la qkv_conv)"))?;
    let mut q = backend
        .stream
        .alloc_zeros::<u16>(q_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la q)"))?;
    let mut k = backend
        .stream
        .alloc_zeros::<u16>(q_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la k)"))?;
    let mut v = backend
        .stream
        .alloc_zeros::<u16>(v_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la v)"))?;
    let mut g = backend
        .stream
        .alloc_zeros::<f32>(head_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la g)"))?;
    let mut g_cumsum = backend
        .stream
        .alloc_zeros::<f32>(head_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la g_cumsum)"))?;
    let mut beta = backend
        .stream
        .alloc_zeros::<f32>(head_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la beta)"))?;
    let mut a_tril = backend
        .stream
        .alloc_zeros::<f32>(a_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la a_tril)"))?;
    let mut a_inv = backend
        .stream
        .alloc_zeros::<u16>(a_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la a_inv)"))?;
    let mut w = backend
        .stream
        .alloc_zeros::<u16>(q_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la w)"))?;
    let mut u = backend
        .stream
        .alloc_zeros::<u16>(v_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la u)"))?;
    let mut initial_state = backend
        .stream
        .alloc_zeros::<f32>(state_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la initial_state)"))?;
    let mut chunk_state = backend
        .stream
        .alloc_zeros::<f32>(chunk_state_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la chunk_state)"))?;
    let mut v_new = backend
        .stream
        .alloc_zeros::<u16>(v_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la v_new)"))?;
    let mut final_state = backend
        .stream
        .alloc_zeros::<f32>(state_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la final_state)"))?;
    // The `seq_len <= 32` clamp (abb1ed995) worked around a WGMMA deadlock in the
    // pre-FlashQLA chunk kernel, which 778fef873 replaced — the flag alone gates it now.
    let use_chunkwise = linear_attention_gdr_chunkwise_prefill_enabled();
    // Seed carry so chunk_state[0] = carry. Only the taken branch's buffer needs it: the
    // recurrent branch runs final_state → chunk_state[0], the chunkwise branch reads initial_state
    // (final_state is output-only there). Seed the one the branch consumes — the other is dead.
    if let Some(state) = carry_state {
        let dst = if use_chunkwise {
            &mut initial_state
        } else {
            &mut final_state
        };
        backend
            .stream
            .memcpy_dtod(state, dst)
            .map_err(|_| AutogradError::TapeInvariant("cuda D2D copy failed (la carry seed)"))?;
    }
    let mut raw_output = backend
        .stream
        .alloc_zeros::<u16>(v_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la raw_output)"))?;
    let mut output = backend
        .stream
        .alloc_zeros::<f32>(z_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la output)"))?;

    let seq_len_i32 = i32::try_from(p.seq_len)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention seq_len exceeds i32"))?;
    let qkv_dim_i32 = i32::try_from(qkv_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention qkv_dim exceeds i32"))?;
    let conv_kernel_i32 = i32::try_from(p.conv_kernel)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention conv_kernel exceeds i32"))?;
    let num_key_heads_i32 = i32::try_from(p.num_key_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key heads exceeds i32"))?;
    let num_value_heads_i32 = i32::try_from(p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value heads exceeds i32"))?;
    let rows_i32 = i32::try_from(p.seq_len * p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention rows exceeds i32"))?;
    let key_dim_i32 = i32::try_from(p.key_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key_dim exceeds i32"))?;
    let value_dim_i32 = i32::try_from(p.value_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value_dim exceeds i32"))?;

    {
        let stage_started = linear_attention_debug_stage_start();
        let (preact_ptr, _preact_guard) = preact.device_ptr_mut(&backend.stream);
        let (qkv_conv_ptr, _qkv_conv_guard) = qkv_conv.device_ptr_mut(&backend.stream);
        let (qkv_ptr, _qkv_guard) = qkv.device_ptr(&backend.stream);
        let (conv_ptr, _conv_guard) = conv1d_weight.device_ptr(&backend.stream);
        // conv_tail = carried boundary window (nullptr → default zero-tap path, byte-identical).
        let conv_tail = carry_conv.map(|s| s.device_ptr(&backend.stream));
        let conv_tail_ptr = conv_tail.as_ref().map_or(0u64, |(ptr, _)| *ptr);
        let conv_tail_len_i32 = carry_conv
            .map(|_| i32::try_from(conv_tail_len))
            .transpose()
            .map_err(|_| {
                AutogradError::TapeInvariant("linear_attention conv_tail_len exceeds i32")
            })?
            .unwrap_or(0);
        let total_u64 = u64::try_from(qkv_len)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention qkv_len exceeds u64"))?;
        launch_1d(
            &backend.stream,
            backend
                .kernels
                .function("linear_attention_conv1d_silu_forward_f32_to_bf16")?,
            qkv_len,
            |mut builder| {
                builder
                    .arg(&preact_ptr)
                    .arg(&qkv_conv_ptr)
                    .arg(&qkv_ptr)
                    .arg(&conv_ptr)
                    .arg(&total_u64)
                    .arg(&qkv_dim_i32)
                    .arg(&seq_len_i32)
                    .arg(&conv_kernel_i32)
                    .arg(&conv_tail_ptr)
                    .arg(&conv_tail_len_i32);
                builder
            },
        )?;
        linear_attention_debug_stage_done(backend, "conv1d_silu", stage_started)?;
    }
    {
        let stage_started = linear_attention_debug_stage_start();
        let (qkv_ptr, _qkv_guard) = qkv_conv.device_ptr(&backend.stream);
        let (b_ptr, _b_guard) = b_bf16.device_ptr(&backend.stream);
        let (a_ptr, _a_guard) = a_bf16.device_ptr(&backend.stream);
        let (dt_ptr, _dt_guard) = dt_bf16.device_ptr(&backend.stream);
        let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
        let (q_ptr, _q_guard) = q.device_ptr_mut(&backend.stream);
        let (k_ptr, _k_guard) = k.device_ptr_mut(&backend.stream);
        let (v_ptr, _v_guard) = v.device_ptr_mut(&backend.stream);
        let (g_ptr, _g_guard) = g.device_ptr_mut(&backend.stream);
        let (beta_ptr, _beta_guard) = beta.device_ptr_mut(&backend.stream);
        check_cuda_ffi(
            // SAFETY: reads qkv_conv/b/a/dt/a_log, writes q/k/v/g/beta — all live guarded slices
            // sized above (qkv_len / head_len / num_value_heads / q_len / v_len) to match the dims.
            unsafe {
                ffi::gated_delta_rule_prefill_chunk_prepare_cuda(
                    qkv_ptr as *const ffi::Half,
                    b_ptr as *const ffi::Half,
                    a_ptr as *const ffi::Half,
                    dt_ptr as *const ffi::Half,
                    a_log_ptr as *const f32,
                    q_ptr as *mut ffi::Half,
                    k_ptr as *mut ffi::Half,
                    v_ptr as *mut ffi::Half,
                    g_ptr as *mut f32,
                    beta_ptr as *mut f32,
                    num_key_heads_i32,
                    num_value_heads_i32,
                    qkv_dim_i32,
                    seq_len_i32,
                    backend.stream.cu_stream(),
                )
            },
            "gated_delta_rule_prefill_chunk_prepare_cuda",
        )?;
        linear_attention_debug_stage_done(backend, "gdr_prepare", stage_started)?;
    }
    if use_chunkwise {
        {
            let stage_started = linear_attention_debug_stage_start();
            let (g_ptr, _g_guard) = g.device_ptr(&backend.stream);
            let (gc_ptr, _gc_guard) = g_cumsum.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: g and g_cumsum are live guarded head_len (seq_len*num_value_heads)
                // f32 slices — exactly the extent the cumsum kernel scans.
                unsafe {
                    ffi::gated_delta_rule_prefill_chunk_cumsum_cuda(
                        g_ptr as *const f32,
                        gc_ptr as *mut f32,
                        seq_len_i32,
                        num_value_heads_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gated_delta_rule_prefill_chunk_cumsum_cuda",
            )?;
            linear_attention_debug_stage_done(backend, "gdr_cumsum", stage_started)?;
        }
        {
            let stage_started = linear_attention_debug_stage_start();
            let (k_ptr, _k_guard) = k.device_ptr(&backend.stream);
            let (gc_ptr, _gc_guard) = g_cumsum.device_ptr(&backend.stream);
            let (beta_ptr, _beta_guard) = beta.device_ptr(&backend.stream);
            let (a_tril_ptr, _a_tril_guard) = a_tril.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: k/g_cumsum/beta are live guarded inputs (q_len/head_len); a_tril is
                // allocated a_len = seq_len*num_value_heads*64, the kernel's per-chunk tril extent.
                unsafe {
                    ffi::gated_delta_rule_prefill_chunk_a_cuda(
                        k_ptr as *const ffi::Half,
                        gc_ptr as *const f32,
                        beta_ptr as *const f32,
                        a_tril_ptr as *mut f32,
                        seq_len_i32,
                        num_value_heads_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gated_delta_rule_prefill_chunk_a_cuda",
            )?;
            linear_attention_debug_stage_done(backend, "gdr_a", stage_started)?;
        }
        {
            let stage_started = linear_attention_debug_stage_start();
            let (a_tril_ptr, _a_tril_guard) = a_tril.device_ptr(&backend.stream);
            let (a_inv_ptr, _a_inv_guard) = a_inv.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: a_tril and a_inv are both live guarded a_len slices; the solve kernel
                // reads/writes only that per-chunk 64x64 tril extent.
                unsafe {
                    ffi::gated_delta_rule_prefill_chunk_solve_cuda(
                        a_tril_ptr as *const f32,
                        a_inv_ptr as *mut ffi::Half,
                        seq_len_i32,
                        num_value_heads_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gated_delta_rule_prefill_chunk_solve_cuda",
            )?;
            linear_attention_debug_stage_done(backend, "gdr_solve", stage_started)?;
        }
        {
            let stage_started = linear_attention_debug_stage_start();
            let (k_ptr, _k_guard) = k.device_ptr(&backend.stream);
            let (v_ptr, _v_guard) = v.device_ptr(&backend.stream);
            let (beta_ptr, _beta_guard) = beta.device_ptr(&backend.stream);
            let (w_ptr, _w_guard) = w.device_ptr_mut(&backend.stream);
            let (u_ptr, _u_guard) = u.device_ptr_mut(&backend.stream);
            let (a_inv_ptr, _a_inv_guard) = a_inv.device_ptr(&backend.stream);
            let (gc_ptr, _gc_guard) = g_cumsum.device_ptr(&backend.stream);
            check_cuda_ffi(
                // SAFETY: k/w share q_len, v/u share v_len, beta/g_cumsum head_len, a_inv a_len —
                // all live guarded slices allocated above at the extents the dims imply.
                unsafe {
                    ffi::gated_delta_rule_prefill_chunk_recompute_cuda(
                        k_ptr as *const ffi::Half,
                        v_ptr as *const ffi::Half,
                        beta_ptr as *const f32,
                        w_ptr as *mut ffi::Half,
                        u_ptr as *mut ffi::Half,
                        a_inv_ptr as *const ffi::Half,
                        gc_ptr as *const f32,
                        seq_len_i32,
                        num_value_heads_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gated_delta_rule_prefill_chunk_recompute_cuda",
            )?;
            linear_attention_debug_stage_done(backend, "gdr_recompute", stage_started)?;
        }
        {
            let stage_started = linear_attention_debug_stage_start();
            let (k_ptr, _k_guard) = k.device_ptr(&backend.stream);
            let (w_ptr, _w_guard) = w.device_ptr(&backend.stream);
            let (u_ptr, _u_guard) = u.device_ptr(&backend.stream);
            let (gc_ptr, _gc_guard) = g_cumsum.device_ptr(&backend.stream);
            let (initial_ptr, _initial_guard) = initial_state.device_ptr(&backend.stream);
            let (chunk_ptr, _chunk_guard) = chunk_state.device_ptr_mut(&backend.stream);
            let (vnew_ptr, _vnew_guard) = v_new.device_ptr_mut(&backend.stream);
            let (final_ptr, _final_guard) = final_state.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: k/w/u/g_cumsum/initial_state are live guarded inputs; chunk_state holds
                // num_chunks*state_len, v_new v_len, final_state state_len — writes stay in bounds.
                unsafe {
                    ffi::gated_delta_rule_prefill_chunk_state_cuda(
                        k_ptr as *const ffi::Half,
                        w_ptr as *const ffi::Half,
                        u_ptr as *const ffi::Half,
                        gc_ptr as *const f32,
                        initial_ptr as *const f32,
                        chunk_ptr as *mut f32,
                        vnew_ptr as *mut ffi::Half,
                        final_ptr as *mut f32,
                        seq_len_i32,
                        num_value_heads_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gated_delta_rule_prefill_chunk_state_cuda",
            )?;
            linear_attention_debug_stage_done(backend, "gdr_state", stage_started)?;
        }
        {
            let stage_started = linear_attention_debug_stage_start();
            let (q_ptr, _q_guard) = q.device_ptr(&backend.stream);
            let (k_ptr, _k_guard) = k.device_ptr(&backend.stream);
            let (vnew_ptr, _vnew_guard) = v_new.device_ptr(&backend.stream);
            let (chunk_ptr, _chunk_guard) = chunk_state.device_ptr(&backend.stream);
            let (gc_ptr, _gc_guard) = g_cumsum.device_ptr(&backend.stream);
            let (raw_ptr, _raw_guard) = raw_output.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: q/k/v_new/chunk_state/g_cumsum are live guarded inputs; raw_output is
                // allocated v_len (seq_len*num_value_heads*value_dim), the kernel's write extent.
                unsafe {
                    ffi::gated_delta_rule_prefill_chunk_o_cuda(
                        q_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        vnew_ptr as *const ffi::Half,
                        chunk_ptr as *const f32,
                        gc_ptr as *const f32,
                        raw_ptr as *mut ffi::Half,
                        seq_len_i32,
                        num_value_heads_i32,
                        (p.key_dim as f32).sqrt().recip(),
                        backend.stream.cu_stream(),
                    )
                },
                "gated_delta_rule_prefill_chunk_o_cuda",
            )?;
            linear_attention_debug_stage_done(backend, "gdr_o", stage_started)?;
        }
    } else {
        let stage_started = linear_attention_debug_stage_start();
        let (qkv_ptr, _qkv_guard) = qkv_conv.device_ptr(&backend.stream);
        let (b_ptr, _b_guard) = b_bf16.device_ptr(&backend.stream);
        let (a_ptr, _a_guard) = a_bf16.device_ptr(&backend.stream);
        let (dt_ptr, _dt_guard) = dt_bf16.device_ptr(&backend.stream);
        let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
        let (state_ptr, _state_guard) = final_state.device_ptr_mut(&backend.stream);
        let (chunk_ptr, _chunk_guard) = chunk_state.device_ptr_mut(&backend.stream);
        let (raw_ptr, _raw_guard) = raw_output.device_ptr_mut(&backend.stream);
        let state_len_i32 = i32::try_from(state_len)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention state_len exceeds i32"))?;
        for chunk_idx in 0..num_chunks {
            let base = chunk_idx * 64;
            let chunk_len = (p.seq_len - base).min(64);
            let chunk_len_i32 = i32::try_from(chunk_len).map_err(|_| {
                AutogradError::TapeInvariant("linear_attention chunk_len exceeds i32")
            })?;
            // SAFETY: chunk_state has num_chunks*state_len f32s; chunk_idx < num_chunks.
            let dst_addr = unsafe { (chunk_ptr as *mut f32).add(chunk_idx * state_len) } as u64;
            let src_addr = state_ptr;
            launch_1d(
                &backend.stream,
                backend.kernels.function("linear_attention_copy_f32")?,
                state_len,
                |mut builder| {
                    builder.arg(&dst_addr).arg(&src_addr).arg(&state_len_i32);
                    builder
                },
            )?;
            check_cuda_ffi(
                // SAFETY: base = chunk_idx*64 with chunk_len = min(seq_len-base, 64), so the
                // offset qkv_conv/b/a/raw_output views and final_state (state_len) stay inside
                // their live guarded slices.
                unsafe {
                    ffi::gated_delta_rule_prefill_recurrent_cuda(
                        (qkv_ptr as *const ffi::Half).add(base * qkv_dim),
                        (b_ptr as *const ffi::Half).add(base * p.num_value_heads),
                        (a_ptr as *const ffi::Half).add(base * p.num_value_heads),
                        dt_ptr as *const ffi::Half,
                        a_log_ptr as *const f32,
                        state_ptr as *mut f32,
                        (raw_ptr as *mut ffi::Half).add(base * p.num_value_heads * p.value_dim),
                        num_key_heads_i32,
                        num_value_heads_i32,
                        key_dim_i32,
                        value_dim_i32,
                        chunk_len_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gated_delta_rule_prefill_recurrent_cuda",
            )?;
        }
        linear_attention_debug_stage_done(backend, "gdr_recurrent", stage_started)?;
    }
    {
        let stage_started = linear_attention_debug_stage_start();
        let (out_ptr, _out_guard) = output.device_ptr_mut(&backend.stream);
        let (raw_ptr, _raw_guard) = raw_output.device_ptr(&backend.stream);
        let (z_ptr, _z_guard) = z.device_ptr(&backend.stream);
        let (norm_ptr, _norm_guard) = norm_weight.device_ptr(&backend.stream);
        launch_rows(
            &backend.stream,
            backend
                .kernels
                .function("linear_attention_rms_gated_forward_f32_from_bf16")?,
            p.seq_len * p.num_value_heads,
            256,
            (256 * std::mem::size_of::<f32>()) as u32,
            |mut builder| {
                builder
                    .arg(&out_ptr)
                    .arg(&raw_ptr)
                    .arg(&z_ptr)
                    .arg(&norm_ptr)
                    .arg(&rows_i32)
                    .arg(&value_dim_i32)
                    .arg(&p.eps);
                builder
            },
        )?;
        linear_attention_debug_stage_done(backend, "rms_gated", stage_started)?;
    }

    Ok(LinearAttentionDeviceForwardResult {
        output: DeviceHandle::Cuda(CudaStorage::new(output)),
        preact: DeviceHandle::Cuda(CudaStorage::new(preact)),
        qkv_conv: DeviceHandle::CudaBf16(CudaBf16Storage::new(qkv_conv)),
        q: DeviceHandle::CudaBf16(CudaBf16Storage::new(q)),
        k: DeviceHandle::CudaBf16(CudaBf16Storage::new(k)),
        v: DeviceHandle::CudaBf16(CudaBf16Storage::new(v)),
        g: DeviceHandle::Cuda(CudaStorage::new(g)),
        g_cumsum: DeviceHandle::Cuda(CudaStorage::new(g_cumsum)),
        beta: DeviceHandle::Cuda(CudaStorage::new(beta)),
        a_inv: DeviceHandle::CudaBf16(CudaBf16Storage::new(a_inv)),
        chunk_state: DeviceHandle::Cuda(CudaStorage::new(chunk_state)),
        raw_output: DeviceHandle::CudaBf16(CudaBf16Storage::new(raw_output)),
    })
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_linear_attention_backward_device(
    backend: &CudaBackend,
    args: LinearAttentionDeviceBackwardArgs<'_>,
) -> Result<Option<LinearAttentionDeviceBackwardResult>> {
    let p = args.params;
    if !cuda_la_device_supported(p) {
        return Ok(None);
    }
    if p.batch == 1 {
        return cuda_linear_attention_backward_device_row(backend, args).map(Some);
    }

    // batch > 1: per-row dispatch to the proven batch==1 path. Upstream, inputs
    // and every saved ctx tensor are batch-leading contiguous rows; weights pass
    // through whole. Per-token grads concatenate; weight grads sum across rows.
    let qkv_dim = p.num_key_heads * p.key_dim * 2 + p.num_value_heads * p.value_dim;
    let conv_len = qkv_dim * p.conv_kernel;
    let row_params = LinearAttentionDeviceParams { batch: 1, ..p };
    let rows = (0..p.batch)
        .map(|row| {
            let slice = |src| cuda_row_slice(backend, src, row, p.batch);
            let initial_conv_window = args.initial_conv_window.map(slice).transpose()?;
            cuda_linear_attention_backward_device_row(
                backend,
                LinearAttentionDeviceBackwardArgs {
                    params: row_params,
                    upstream: &slice(args.upstream)?,
                    qkv: &slice(args.qkv)?,
                    z: &slice(args.z)?,
                    b_proj: &slice(args.b_proj)?,
                    a_proj: &slice(args.a_proj)?,
                    preact: &slice(args.preact)?,
                    qkv_conv: &slice(args.qkv_conv)?,
                    g: &slice(args.g)?,
                    beta: &slice(args.beta)?,
                    chunk_state: &slice(args.chunk_state)?,
                    conv1d_weight: args.conv1d_weight,
                    dt_bias: args.dt_bias,
                    a_log: args.a_log,
                    norm_weight: args.norm_weight,
                    initial_conv_window: initial_conv_window.as_ref(),
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let concat = |field: fn(&LinearAttentionDeviceBackwardResult) -> &DeviceHandle| {
        let parts: Vec<&DeviceHandle> = rows.iter().map(field).collect();
        cuda_concat_rows(backend, &parts)
    };
    let sum = |field: fn(&LinearAttentionDeviceBackwardResult) -> &DeviceHandle, len: usize| {
        rows[1..]
            .iter()
            .try_fold(field(&rows[0]).clone(), |acc, row| {
                backend.add(&acc, field(row), &[len])
            })
    };
    Ok(Some(LinearAttentionDeviceBackwardResult {
        dqkv: concat(|r| &r.dqkv)?,
        dz: concat(|r| &r.dz)?,
        db: concat(|r| &r.db)?,
        da: concat(|r| &r.da)?,
        dconv: sum(|r| &r.dconv, conv_len)?,
        ddt: sum(|r| &r.ddt, p.num_value_heads)?,
        da_log: sum(|r| &r.da_log, p.num_value_heads)?,
        dnorm: sum(|r| &r.dnorm, p.value_dim)?,
    }))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_linear_attention_backward_device_row(
    backend: &CudaBackend,
    args: LinearAttentionDeviceBackwardArgs<'_>,
) -> Result<LinearAttentionDeviceBackwardResult> {
    let p = args.params;
    debug_assert_eq!(p.batch, 1);
    let q_dim = p.num_key_heads * p.key_dim;
    let qkv_dim = q_dim * 2 + p.num_value_heads * p.value_dim;
    let qkv_len = p.batch * p.seq_len * qkv_dim;
    let z_len = p.batch * p.seq_len * p.num_value_heads * p.value_dim;
    let head_len = p.batch * p.seq_len * p.num_value_heads;
    let conv_len = qkv_dim * p.conv_kernel;
    let rows = p.batch * p.num_value_heads;
    let state_elems = p.key_dim * p.value_dim;
    let state_len = rows * state_elems;
    let num_chunks = p.seq_len.div_ceil(64);
    let chunk_state_len = num_chunks * p.num_value_heads * state_elems;

    let upstream = backend.cuda_slice(args.upstream, "linear_attention_backward upstream")?;
    let qkv = backend.cuda_slice(args.qkv, "linear_attention_backward qkv")?;
    let z = backend.cuda_slice(args.z, "linear_attention_backward z")?;
    let a_proj = backend.cuda_slice(args.a_proj, "linear_attention_backward a_proj")?;
    let conv1d_weight = backend.cuda_slice(
        args.conv1d_weight,
        "linear_attention_backward conv1d_weight",
    )?;
    let dt_bias = backend.cuda_slice(args.dt_bias, "linear_attention_backward dt_bias")?;
    let a_log = backend.cuda_slice(args.a_log, "linear_attention_backward a_log")?;
    let norm_weight =
        backend.cuda_slice(args.norm_weight, "linear_attention_backward norm_weight")?;
    let preact = backend.cuda_slice(args.preact, "linear_attention_backward preact")?;
    let qkv_conv = backend.cuda_bf16_slice(args.qkv_conv, "linear_attention_backward qkv_conv")?;
    let beta = backend.cuda_slice(args.beta, "linear_attention_backward beta")?;
    let g = backend.cuda_slice(args.g, "linear_attention_backward g")?;
    let chunk_state =
        backend.cuda_slice(args.chunk_state, "linear_attention_backward chunk_state")?;

    for (label, got, expected) in [
        ("upstream", upstream.len(), z_len),
        ("qkv", qkv.len(), qkv_len),
        ("z", z.len(), z_len),
        ("a_proj", a_proj.len(), head_len),
        ("conv1d_weight", conv1d_weight.len(), conv_len),
        ("dt_bias", dt_bias.len(), p.num_value_heads),
        ("a_log", a_log.len(), p.num_value_heads),
        ("norm_weight", norm_weight.len(), p.value_dim),
        ("preact", preact.len(), qkv_len),
        ("qkv_conv", qkv_conv.len(), qkv_len),
        ("beta", beta.len(), head_len),
        ("g", g.len(), head_len),
        ("chunk_state", chunk_state.len(), chunk_state_len),
    ] {
        if got != expected {
            return Err(AutogradError::TapeInvariant(Box::leak(
                format!(
                    "cuda linear_attention_backward_device {label} len mismatch: got={got} expected={expected}"
                )
                .into_boxed_str(),
            )));
        }
    }

    let mut dqkv_conv = backend
        .stream
        .alloc_zeros::<f32>(qkv_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la dqkv_conv)"))?;
    let mut dz = backend
        .stream
        .alloc_zeros::<f32>(z_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la dz)"))?;
    let mut db = backend
        .stream
        .alloc_zeros::<f32>(head_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la db)"))?;
    let mut da = backend
        .stream
        .alloc_zeros::<f32>(head_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la da)"))?;
    let mut ddt = backend
        .stream
        .alloc_zeros::<f32>(p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la ddt)"))?;
    let mut da_log = backend
        .stream
        .alloc_zeros::<f32>(p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la da_log)"))?;
    let mut dnorm = backend
        .stream
        .alloc_zeros::<f32>(p.value_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la dnorm)"))?;
    let batch_i32 = i32::try_from(p.batch)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention batch exceeds i32"))?;
    let seq_len_i32 = i32::try_from(p.seq_len)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention seq_len exceeds i32"))?;
    let num_key_heads_i32 = i32::try_from(p.num_key_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key heads exceeds i32"))?;
    let num_value_heads_i32 = i32::try_from(p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value heads exceeds i32"))?;
    let key_dim_i32 = i32::try_from(p.key_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key_dim exceeds i32"))?;
    let value_dim_i32 = i32::try_from(p.value_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value_dim exceeds i32"))?;
    let qkv_dim_i32 = i32::try_from(qkv_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention qkv_dim exceeds i32"))?;
    let total_u64 = u64::try_from(qkv_len)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention qkv_len exceeds u64"))?;
    let conv_kernel_i32 = i32::try_from(p.conv_kernel)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention conv_kernel exceeds i32"))?;
    let carry_len = num_chunks * rows * state_elems;
    let staged_elems = carry_len.saturating_mul(3).saturating_add(
        num_chunks
            .saturating_mul(rows)
            .saturating_mul(p.key_dim)
            .saturating_mul(p.key_dim),
    );
    let staged_bytes = staged_elems.saturating_mul(std::mem::size_of::<f32>());
    let free_bytes = backend.mem_get_info().map_or(0, |(free, _)| free);
    let use_mono = linear_attention_mono_backward_forced() || staged_bytes > free_bytes / 2;

    if use_mono {
        let mut grad_state_scratch = backend
            .stream
            .alloc_zeros::<f32>(state_len)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la grad_state)"))?;
        let mut state_recompute_scratch =
            backend.stream.alloc_zeros::<f32>(state_len).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (la state_recompute)")
            })?;
        let mut chunk_history_scratch = backend
            .stream
            .alloc_zeros::<f32>(rows * 64 * state_elems)
            .map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (la chunk_history)")
            })?;
        let mut chunk_kv_scratch = backend
            .stream
            .alloc_zeros::<f32>(rows * 64 * p.value_dim)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la chunk_kv)"))?;

        let (dqkv_conv_ptr, _dqkv_conv_guard) = dqkv_conv.device_ptr_mut(&backend.stream);
        let (dz_ptr, _dz_guard) = dz.device_ptr_mut(&backend.stream);
        let (db_ptr, _db_guard) = db.device_ptr_mut(&backend.stream);
        let (da_ptr, _da_guard) = da.device_ptr_mut(&backend.stream);
        let (ddt_ptr, _ddt_guard) = ddt.device_ptr_mut(&backend.stream);
        let (da_log_ptr, _da_log_guard) = da_log.device_ptr_mut(&backend.stream);
        let (dnorm_ptr, _dnorm_guard) = dnorm.device_ptr_mut(&backend.stream);
        let (grad_state_ptr, _grad_state_guard) =
            grad_state_scratch.device_ptr_mut(&backend.stream);
        let (state_recompute_ptr, _state_recompute_guard) =
            state_recompute_scratch.device_ptr_mut(&backend.stream);
        let (chunk_history_ptr, _chunk_history_guard) =
            chunk_history_scratch.device_ptr_mut(&backend.stream);
        let (chunk_kv_ptr, _chunk_kv_guard) = chunk_kv_scratch.device_ptr_mut(&backend.stream);
        let (upstream_ptr, _upstream_guard) = upstream.device_ptr(&backend.stream);
        let (z_ptr, _z_guard) = z.device_ptr(&backend.stream);
        let (a_proj_ptr, _a_proj_guard) = a_proj.device_ptr(&backend.stream);
        let (dt_ptr, _dt_guard) = dt_bias.device_ptr(&backend.stream);
        let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
        let (norm_ptr, _norm_guard) = norm_weight.device_ptr(&backend.stream);
        let (preact_ptr, _preact_guard) = preact.device_ptr(&backend.stream);
        let (qkv_conv_saved_ptr, _qkv_conv_saved_guard) = qkv_conv.device_ptr(&backend.stream);
        let (beta_ptr, _beta_guard) = beta.device_ptr(&backend.stream);
        let (g_ptr, _g_guard) = g.device_ptr(&backend.stream);
        let (chunk_state_ptr, _chunk_state_guard) = chunk_state.device_ptr(&backend.stream);

        launch_rows(
            &backend.stream,
            backend
                .kernels
                .function("linear_attention_chunked_scan_backward_f32")?,
            rows,
            256,
            0,
            |mut builder| {
                builder
                    .arg(&dqkv_conv_ptr)
                    .arg(&dz_ptr)
                    .arg(&db_ptr)
                    .arg(&da_ptr)
                    .arg(&ddt_ptr)
                    .arg(&da_log_ptr)
                    .arg(&dnorm_ptr)
                    .arg(&grad_state_ptr)
                    .arg(&state_recompute_ptr)
                    .arg(&chunk_history_ptr)
                    .arg(&chunk_kv_ptr)
                    .arg(&upstream_ptr)
                    .arg(&z_ptr)
                    .arg(&a_proj_ptr)
                    .arg(&dt_ptr)
                    .arg(&a_log_ptr)
                    .arg(&norm_ptr)
                    .arg(&preact_ptr)
                    .arg(&qkv_conv_saved_ptr)
                    .arg(&beta_ptr)
                    .arg(&g_ptr)
                    .arg(&chunk_state_ptr)
                    .arg(&batch_i32)
                    .arg(&seq_len_i32)
                    .arg(&num_key_heads_i32)
                    .arg(&num_value_heads_i32)
                    .arg(&key_dim_i32)
                    .arg(&value_dim_i32)
                    .arg(&qkv_dim_i32)
                    .arg(&p.eps);
                builder
            },
        )?;
    } else {
        let rows_i32 = i32::try_from(rows)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention rows exceeds i32"))?;
        let num_chunks_i32 = i32::try_from(num_chunks)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention num_chunks exceeds i32"))?;
        let wave = num_chunks.min(LA_BWD_CHUNK_WAVE);
        let wave_i32 = i32::try_from(wave)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention wave exceeds i32"))?;
        let grid = wave * rows;
        let mut g_in_scratch = backend
            .stream
            .alloc_zeros::<f32>(carry_len)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la g_in)"))?;

        if num_chunks > 1 {
            let mut m_scratch = backend
                .stream
                .alloc_zeros::<f32>(num_chunks * rows * p.key_dim * p.key_dim)
                .map_err(|_| {
                    AutogradError::TapeInvariant("cuda alloc_zeros failed (la transfer_m)")
                })?;
            let mut b_scratch = backend.stream.alloc_zeros::<f32>(carry_len).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (la transfer_b)")
            })?;
            let mut state_scratch = backend.stream.alloc_zeros::<f32>(carry_len).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (la transfer_state)")
            })?;

            {
                let (m_ptr, _m_guard) = m_scratch.device_ptr_mut(&backend.stream);
                let (b_ptr, _b_guard) = b_scratch.device_ptr_mut(&backend.stream);
                let (state_ptr, _state_guard) = state_scratch.device_ptr_mut(&backend.stream);
                let (upstream_ptr, _upstream_guard) = upstream.device_ptr(&backend.stream);
                let (z_ptr, _z_guard) = z.device_ptr(&backend.stream);
                let (norm_ptr, _norm_guard) = norm_weight.device_ptr(&backend.stream);
                let (qkv_conv_saved_ptr, _qkv_conv_saved_guard) =
                    qkv_conv.device_ptr(&backend.stream);
                let (beta_ptr, _beta_guard) = beta.device_ptr(&backend.stream);
                let (g_ptr, _g_guard) = g.device_ptr(&backend.stream);
                let (chunk_state_ptr, _chunk_state_guard) = chunk_state.device_ptr(&backend.stream);
                launch_rows(
                    &backend.stream,
                    backend
                        .kernels
                        .function("linear_attention_chunk_transfer_f32")?,
                    num_chunks * rows,
                    256,
                    0,
                    |mut builder| {
                        builder
                            .arg(&m_ptr)
                            .arg(&b_ptr)
                            .arg(&state_ptr)
                            .arg(&upstream_ptr)
                            .arg(&z_ptr)
                            .arg(&norm_ptr)
                            .arg(&qkv_conv_saved_ptr)
                            .arg(&beta_ptr)
                            .arg(&g_ptr)
                            .arg(&chunk_state_ptr)
                            .arg(&batch_i32)
                            .arg(&seq_len_i32)
                            .arg(&num_key_heads_i32)
                            .arg(&num_value_heads_i32)
                            .arg(&key_dim_i32)
                            .arg(&value_dim_i32)
                            .arg(&qkv_dim_i32)
                            .arg(&p.eps);
                        builder
                    },
                )?;
            }
            {
                let (g_in_ptr, _g_in_guard) = g_in_scratch.device_ptr_mut(&backend.stream);
                let (m_ptr, _m_guard) = m_scratch.device_ptr(&backend.stream);
                let (b_ptr, _b_guard) = b_scratch.device_ptr(&backend.stream);
                launch_rows(
                    &backend.stream,
                    backend
                        .kernels
                        .function("linear_attention_chunk_carry_f32")?,
                    rows,
                    256,
                    0,
                    |mut builder| {
                        builder
                            .arg(&g_in_ptr)
                            .arg(&m_ptr)
                            .arg(&b_ptr)
                            .arg(&rows_i32)
                            .arg(&num_chunks_i32)
                            .arg(&key_dim_i32)
                            .arg(&value_dim_i32);
                        builder
                    },
                )?;
            }
        }

        let mut grad_state_scratch = backend
            .stream
            .alloc_zeros::<f32>(grid * state_elems)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la grad_state)"))?;
        let mut chunk_history_scratch = backend
            .stream
            .alloc_zeros::<f32>(grid * 64 * state_elems)
            .map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (la chunk_history)")
            })?;
        let mut chunk_kv_scratch = backend
            .stream
            .alloc_zeros::<f32>(grid * 64 * p.value_dim)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la chunk_kv)"))?;

        let (dqkv_conv_ptr, _dqkv_conv_guard) = dqkv_conv.device_ptr_mut(&backend.stream);
        let (dz_ptr, _dz_guard) = dz.device_ptr_mut(&backend.stream);
        let (db_ptr, _db_guard) = db.device_ptr_mut(&backend.stream);
        let (da_ptr, _da_guard) = da.device_ptr_mut(&backend.stream);
        let (ddt_ptr, _ddt_guard) = ddt.device_ptr_mut(&backend.stream);
        let (da_log_ptr, _da_log_guard) = da_log.device_ptr_mut(&backend.stream);
        let (dnorm_ptr, _dnorm_guard) = dnorm.device_ptr_mut(&backend.stream);
        let (grad_state_ptr, _grad_state_guard) =
            grad_state_scratch.device_ptr_mut(&backend.stream);
        let (chunk_history_ptr, _chunk_history_guard) =
            chunk_history_scratch.device_ptr_mut(&backend.stream);
        let (chunk_kv_ptr, _chunk_kv_guard) = chunk_kv_scratch.device_ptr_mut(&backend.stream);
        let (upstream_ptr, _upstream_guard) = upstream.device_ptr(&backend.stream);
        let (z_ptr, _z_guard) = z.device_ptr(&backend.stream);
        let (a_proj_ptr, _a_proj_guard) = a_proj.device_ptr(&backend.stream);
        let (dt_ptr, _dt_guard) = dt_bias.device_ptr(&backend.stream);
        let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
        let (norm_ptr, _norm_guard) = norm_weight.device_ptr(&backend.stream);
        let (qkv_conv_saved_ptr, _qkv_conv_saved_guard) = qkv_conv.device_ptr(&backend.stream);
        let (beta_ptr, _beta_guard) = beta.device_ptr(&backend.stream);
        let (g_ptr, _g_guard) = g.device_ptr(&backend.stream);
        let (chunk_state_ptr, _chunk_state_guard) = chunk_state.device_ptr(&backend.stream);
        let (g_in_ptr, _g_in_guard) = g_in_scratch.device_ptr(&backend.stream);

        launch_rows(
            &backend.stream,
            backend
                .kernels
                .function("linear_attention_chunk_grad_f32")?,
            grid,
            256,
            0,
            |mut builder| {
                builder
                    .arg(&dqkv_conv_ptr)
                    .arg(&dz_ptr)
                    .arg(&db_ptr)
                    .arg(&da_ptr)
                    .arg(&ddt_ptr)
                    .arg(&da_log_ptr)
                    .arg(&dnorm_ptr)
                    .arg(&grad_state_ptr)
                    .arg(&chunk_history_ptr)
                    .arg(&chunk_kv_ptr)
                    .arg(&upstream_ptr)
                    .arg(&z_ptr)
                    .arg(&a_proj_ptr)
                    .arg(&dt_ptr)
                    .arg(&a_log_ptr)
                    .arg(&norm_ptr)
                    .arg(&qkv_conv_saved_ptr)
                    .arg(&beta_ptr)
                    .arg(&g_ptr)
                    .arg(&chunk_state_ptr)
                    .arg(&g_in_ptr)
                    .arg(&batch_i32)
                    .arg(&seq_len_i32)
                    .arg(&num_key_heads_i32)
                    .arg(&num_value_heads_i32)
                    .arg(&key_dim_i32)
                    .arg(&value_dim_i32)
                    .arg(&qkv_dim_i32)
                    .arg(&wave_i32)
                    .arg(&p.eps);
                builder
            },
        )?;
    }

    let mut dqkv = backend
        .stream
        .alloc_zeros::<f32>(qkv_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la dqkv)"))?;
    let mut dconv = backend
        .stream
        .alloc_zeros::<f32>(conv_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la dconv)"))?;
    {
        let (dqkv_ptr, _dqkv_guard) = dqkv.device_ptr_mut(&backend.stream);
        let (dconv_ptr, _dconv_guard) = dconv.device_ptr_mut(&backend.stream);
        let (dqkv_conv_ptr, _dqkv_conv_guard) = dqkv_conv.device_ptr(&backend.stream);
        let (preact_ptr, _preact_guard) = preact.device_ptr(&backend.stream);
        let (qkv_ptr, _qkv_guard) = qkv.device_ptr(&backend.stream);
        let (conv_ptr, _conv_guard) = conv1d_weight.device_ptr(&backend.stream);
        // conv_tail: carried boundary window (nullptr → zero-pad default). Its boundary
        // taps' grad_weight is real; grad_input stays off (carry frozen).
        let conv_tail = args
            .initial_conv_window
            .map(|h| backend.cuda_slice(h, "linear_attention_backward conv_tail"))
            .transpose()?;
        let conv_tail_dev = conv_tail.map(|s| s.device_ptr(&backend.stream));
        let conv_tail_ptr = conv_tail_dev.as_ref().map_or(0u64, |(ptr, _)| *ptr);
        let conv_tail_len_i32 = conv_tail
            .map(|_| i32::try_from(p.conv_kernel - 1))
            .transpose()
            .map_err(|_| {
                AutogradError::TapeInvariant("linear_attention conv_tail_len exceeds i32")
            })?
            .unwrap_or(0);
        launch_1d(
            &backend.stream,
            backend
                .kernels
                .function("linear_attention_conv1d_silu_backward_f32")?,
            qkv_len,
            |mut builder| {
                builder
                    .arg(&dqkv_ptr)
                    .arg(&dconv_ptr)
                    .arg(&dqkv_conv_ptr)
                    .arg(&preact_ptr)
                    .arg(&qkv_ptr)
                    .arg(&conv_ptr)
                    .arg(&total_u64)
                    .arg(&qkv_dim_i32)
                    .arg(&seq_len_i32)
                    .arg(&conv_kernel_i32)
                    .arg(&conv_tail_ptr)
                    .arg(&conv_tail_len_i32);
                builder
            },
        )?;
    }

    Ok(LinearAttentionDeviceBackwardResult {
        dqkv: DeviceHandle::Cuda(CudaStorage::new(dqkv)),
        dz: DeviceHandle::Cuda(CudaStorage::new(dz)),
        db: DeviceHandle::Cuda(CudaStorage::new(db)),
        da: DeviceHandle::Cuda(CudaStorage::new(da)),
        dconv: DeviceHandle::Cuda(CudaStorage::new(dconv)),
        ddt: DeviceHandle::Cuda(CudaStorage::new(ddt)),
        da_log: DeviceHandle::Cuda(CudaStorage::new(da_log)),
        dnorm: DeviceHandle::Cuda(CudaStorage::new(dnorm)),
    })
}

/// GPU assist for the host-fallback backward — reachable only for shapes the
/// device path declines (non-128 head dims); kept for those, not a hot path.
#[cfg(not(feature = "no-cuda"))]
fn cuda_linear_attention_scan_backward(
    backend: &CudaBackend,
    args: LinearAttentionScanBackwardArgs<'_>,
) -> Result<Option<LinearAttentionScanBackwardGrads>> {
    let p = args.params;
    const MAX_DIM: usize = 256;
    if p.key_dim > MAX_DIM || p.value_dim > MAX_DIM {
        return Ok(None);
    }

    let q_dim = p.num_key_heads * p.key_dim;
    let qkv_dim = q_dim * 2 + p.num_value_heads * p.value_dim;
    let z_dim = p.num_value_heads * p.value_dim;
    let qkv_len = p.batch * p.seq_len * qkv_dim;
    let z_len = p.batch * p.seq_len * z_dim;
    let head_len = p.batch * p.seq_len * p.num_value_heads;
    let state_len = p.batch * p.num_value_heads * p.key_dim * p.value_dim;
    let state_history_len = p.batch * p.seq_len * p.num_value_heads * p.key_dim * p.value_dim;

    for (label, got, expected) in [
        ("upstream", args.upstream.len(), z_len),
        ("z", args.z.len(), z_len),
        ("a_proj", args.a_proj.len(), head_len),
        ("dt_bias", args.dt_bias.len(), p.num_value_heads),
        ("a_log", args.a_log.len(), p.num_value_heads),
        ("norm_weight", args.norm_weight.len(), p.value_dim),
        ("preact", args.preact.len(), qkv_len),
        ("beta", args.beta.len(), head_len),
        ("exp_g", args.exp_g.len(), head_len),
        ("kv_mem", args.kv_mem.len(), z_len),
        ("state_history", args.state_history.len(), state_history_len),
        ("final_state", args.final_state.len(), state_len),
    ] {
        if got != expected {
            return Err(AutogradError::TapeInvariant(Box::leak(
                format!(
                    "cuda linear_attention_scan_backward {label} len mismatch: got={got} expected={expected}"
                )
                .into_boxed_str(),
            )));
        }
    }

    let d_upstream = backend.upload_slice(args.upstream, &[z_len])?;
    let d_z = backend.upload_slice(args.z, &[z_len])?;
    let d_a_proj = backend.upload_slice(args.a_proj, &[head_len])?;
    let d_dt_bias = backend.upload_slice(args.dt_bias, &[p.num_value_heads])?;
    let d_a_log = backend.upload_slice(args.a_log, &[p.num_value_heads])?;
    let d_norm_weight = backend.upload_slice(args.norm_weight, &[p.value_dim])?;
    let d_preact = backend.upload_slice(args.preact, &[qkv_len])?;
    let d_beta = backend.upload_slice(args.beta, &[head_len])?;
    let d_exp_g = backend.upload_slice(args.exp_g, &[head_len])?;
    let d_kv_mem = backend.upload_slice(args.kv_mem, &[z_len])?;
    let d_state_history = backend.upload_slice(args.state_history, &[state_history_len])?;
    let d_final_state = backend.upload_slice(args.final_state, &[state_len])?;

    let mut d_dqkv = backend.stream.alloc_zeros::<f32>(qkv_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention dqkv)")
    })?;
    let mut d_dz = backend.stream.alloc_zeros::<f32>(z_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention dz)")
    })?;
    let mut d_db = backend.stream.alloc_zeros::<f32>(head_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention db)")
    })?;
    let mut d_da = backend.stream.alloc_zeros::<f32>(head_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention da)")
    })?;
    let mut d_ddt = backend
        .stream
        .alloc_zeros::<f32>(p.num_value_heads)
        .map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention ddt)")
        })?;
    let mut d_da_log = backend
        .stream
        .alloc_zeros::<f32>(p.num_value_heads)
        .map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention da_log)")
        })?;
    let mut d_dnorm = backend
        .stream
        .alloc_zeros::<f32>(p.value_dim)
        .map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention dnorm)")
        })?;
    let mut d_grad_state = backend.stream.alloc_zeros::<f32>(state_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention grad_state)")
    })?;

    let rows = p.batch * p.num_value_heads;
    let batch_i32 = i32::try_from(p.batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda linear_attention batch exceeds i32"))?;
    let seq_len_i32 = i32::try_from(p.seq_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda linear_attention seq_len exceeds i32"))?;
    let key_heads_i32 = i32::try_from(p.num_key_heads).map_err(|_| {
        AutogradError::TapeInvariant("cuda linear_attention num_key_heads exceeds i32")
    })?;
    let value_heads_i32 = i32::try_from(p.num_value_heads).map_err(|_| {
        AutogradError::TapeInvariant("cuda linear_attention num_value_heads exceeds i32")
    })?;
    let key_dim_i32 = i32::try_from(p.key_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda linear_attention key_dim exceeds i32"))?;
    let value_dim_i32 = i32::try_from(p.value_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda linear_attention value_dim exceeds i32"))?;
    let qkv_dim_i32 = i32::try_from(qkv_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda linear_attention qkv_dim exceeds i32"))?;

    launch_rows(
        &backend.stream,
        backend
            .kernels
            .function("linear_attention_scan_backward_f32")?,
        rows,
        256,
        0,
        |mut builder| {
            builder
                .arg(&mut d_dqkv)
                .arg(&mut d_dz)
                .arg(&mut d_db)
                .arg(&mut d_da)
                .arg(&mut d_ddt)
                .arg(&mut d_da_log)
                .arg(&mut d_dnorm)
                .arg(&mut d_grad_state)
                .arg(&d_upstream)
                .arg(&d_z)
                .arg(&d_a_proj)
                .arg(&d_dt_bias)
                .arg(&d_a_log)
                .arg(&d_norm_weight)
                .arg(&d_preact)
                .arg(&d_beta)
                .arg(&d_exp_g)
                .arg(&d_kv_mem)
                .arg(&d_state_history)
                .arg(&d_final_state)
                .arg(&batch_i32)
                .arg(&seq_len_i32)
                .arg(&key_heads_i32)
                .arg(&value_heads_i32)
                .arg(&key_dim_i32)
                .arg(&value_dim_i32)
                .arg(&qkv_dim_i32)
                .arg(&p.eps);
            builder
        },
    )?;

    Ok(Some(LinearAttentionScanBackwardGrads {
        dqkv: cuda_readback_slice(backend, &d_dqkv, qkv_len, "linear_attention dqkv")?,
        dz: cuda_readback_slice(backend, &d_dz, z_len, "linear_attention dz")?,
        db: cuda_readback_slice(backend, &d_db, head_len, "linear_attention db")?,
        da: cuda_readback_slice(backend, &d_da, head_len, "linear_attention da")?,
        ddt: cuda_readback_slice(backend, &d_ddt, p.num_value_heads, "linear_attention ddt")?,
        da_log: cuda_readback_slice(
            backend,
            &d_da_log,
            p.num_value_heads,
            "linear_attention da_log",
        )?,
        dnorm: cuda_readback_slice(backend, &d_dnorm, p.value_dim, "linear_attention dnorm")?,
    }))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_readback_slice(
    backend: &CudaBackend,
    slice: &CudaSlice<f32>,
    len: usize,
    label: &'static str,
) -> Result<Vec<f32>> {
    if slice.len() != len {
        return Err(AutogradError::TapeInvariant(Box::leak(
            format!(
                "cuda readback len mismatch ({label}): got={} expected={len}",
                slice.len()
            )
            .into_boxed_str(),
        )));
    }
    let mut host = vec![0.0f32; len];
    backend
        .stream
        .memcpy_dtoh(slice, &mut host)
        .map_err(|err| {
            AutogradError::TapeInvariant(Box::leak(
                format!("cuda dtoh copy failed ({label}): {err:?}").into_boxed_str(),
            ))
        })?;
    backend.stream.synchronize().map_err(|err| {
        AutogradError::TapeInvariant(Box::leak(
            format!("cuda synchronize failed ({label}): {err:?}").into_boxed_str(),
        ))
    })?;
    Ok(host)
}

#[cfg(not(feature = "no-cuda"))]
fn shape_size(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_alloc_failed(op: &'static str, shape: Vec<usize>) -> AutogradError {
    let bytes = shape_size(&shape).saturating_mul(std::mem::size_of::<f32>());
    AutogradError::CudaAllocFailed { op, shape, bytes }
}

// Alloc failure: report driver code + live free/total to tell fragmentation
// from a sticky async fault (fails with GB free).
#[cfg(not(feature = "no-cuda"))]
fn cuda_alloc_failed_rich(
    backend: &CudaBackend,
    op: &'static str,
    bytes: usize,
    err: &dyn std::fmt::Debug,
) -> AutogradError {
    leak_err(format!(
        "alloc {op} failed: bytes={bytes} err={err:?} free_total={:?}",
        backend.mem_get_info().ok()
    ))
}

#[cfg(not(feature = "no-cuda"))]
fn leak_err(msg: String) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(msg.into_boxed_str()))
}

// Compute both matmul gradients via two cuBLAS SGEMM calls with an OP_T on
// whichever operand must be transposed; avoids the host-side physical
// transpose the old CPU fallback did and keeps the math on-device.
//
// Row-major conventions in the header comment (swap-and-OP_N forward trick)
// carry through: we reuse the same "pass B first, then A" ordering. For
// `grad_a = dC @ B^T` we pass `(B, dC, transa=OP_T, transb=OP_N)`; for
// `grad_b = A^T @ dC` we pass `(dC, A, transa=OP_N, transb=OP_T)`. See the
// file-level comment + derivation in the companion commit for the full
// dimension/ld walk-through. PENDING REMOTE CUDA VERIFICATION.
#[cfg(not(feature = "no-cuda"))]
fn cuda_matmul_backward(
    backend: &CudaBackend,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    grad_out: &[f32],
    grad_out_shape: &[usize],
    need_grad_a: bool,
    need_grad_b: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let expected_out = matmul_output_shape(a_shape, b_shape)?;
    if grad_out_shape != expected_out.as_slice() {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_out,
            got: grad_out_shape.to_vec(),
        });
    }

    if !need_grad_a && !need_grad_b {
        return Ok((Vec::new(), Vec::new()));
    }

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => {
            let m = a_shape[0];
            let k = a_shape[1];
            let n = b_shape[1];

            // Upload inputs once each and reuse for both SGEMMs.
            let d_a = backend.upload_slice(a, a_shape)?;
            let d_b = backend.upload_slice(b, b_shape)?;
            let d_g = backend.upload_slice(grad_out, grad_out_shape)?;

            let grad_a_host = if need_grad_a {
                // grad_a[M,K] = grad_out[M,N] @ B^T[N,K]
                // cuBLAS: first_arg=B(OP_T), second_arg=dC(OP_N); m=K,n=M,k=N.
                // lda = N (B cm[N,K]), ldb = N (dC cm[N,M]), ldc = K.
                let mut c = backend
                    .stream
                    .alloc_zeros::<f32>(m * k)
                    .map_err(|_| cuda_alloc_failed("matmul_backward grad_a", vec![m, k]))?;
                let cfg = GemmConfig::<f32> {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: k as i32,
                    n: m as i32,
                    k: n as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: n as i32,
                    beta: 0.0,
                    ldc: k as i32,
                };
                // Safety: dims validated; device buffers outlive the call.
                unsafe {
                    backend.blas.gemm(cfg, &d_b, &d_g, &mut c).map_err(|_| {
                        AutogradError::TapeInvariant("cuBLAS sgemm failed (grad_a)")
                    })?;
                }
                cuda_download(backend, &c, m * k)?
            } else {
                Vec::new()
            };

            let grad_b_host = if need_grad_b {
                // grad_b[K,N] = A^T[K,M] @ grad_out[M,N]
                // cuBLAS: first_arg=dC(OP_N), second_arg=A(OP_T); m=N,n=K,k=M.
                // lda = N (dC cm[N,M]), ldb = K (A cm[K,M]), ldc = N.
                let mut c = backend
                    .stream
                    .alloc_zeros::<f32>(k * n)
                    .map_err(|_| cuda_alloc_failed("matmul_backward grad_b", vec![k, n]))?;
                let cfg = GemmConfig::<f32> {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: n as i32,
                    n: k as i32,
                    k: m as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: k as i32,
                    beta: 0.0,
                    ldc: n as i32,
                };
                // Safety: dims validated; device buffers outlive the call.
                unsafe {
                    backend.blas.gemm(cfg, &d_g, &d_a, &mut c).map_err(|_| {
                        AutogradError::TapeInvariant("cuBLAS sgemm failed (grad_b)")
                    })?;
                }
                cuda_download(backend, &c, k * n)?
            } else {
                Vec::new()
            };

            Ok((grad_a_host, grad_b_host))
        }
        (3, 3) => {
            let batch = a_shape[0];
            let m = a_shape[1];
            let k = a_shape[2];
            let n = b_shape[2];
            if b_shape[0] != batch || grad_out_shape[0] != batch {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![batch],
                    got: vec![b_shape[0].min(grad_out_shape[0])],
                });
            }

            let d_a = backend.upload_slice(a, a_shape)?;
            let d_b = backend.upload_slice(b, b_shape)?;
            let d_g = backend.upload_slice(grad_out, grad_out_shape)?;

            let grad_a_host = if need_grad_a {
                let mut c = backend
                    .stream
                    .alloc_zeros::<f32>(batch * m * k)
                    .map_err(|_| {
                        cuda_alloc_failed("matmul_backward batched grad_a", vec![batch, m, k])
                    })?;
                let gemm = GemmConfig::<f32> {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: k as i32,
                    n: m as i32,
                    k: n as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: n as i32,
                    beta: 0.0,
                    ldc: k as i32,
                };
                let cfg = StridedBatchedConfig::<f32> {
                    gemm,
                    batch_size: batch as i32,
                    stride_a: (k * n) as i64,
                    stride_b: (m * n) as i64,
                    stride_c: (m * k) as i64,
                };
                // Safety: dims validated; buffers outlive the call.
                unsafe {
                    backend
                        .blas
                        .gemm_strided_batched(cfg, &d_b, &d_g, &mut c)
                        .map_err(|_| {
                            AutogradError::TapeInvariant(
                                "cuBLAS sgemm_strided_batched failed (grad_a)",
                            )
                        })?;
                }
                cuda_download(backend, &c, batch * m * k)?
            } else {
                Vec::new()
            };

            let grad_b_host = if need_grad_b {
                let mut c = backend
                    .stream
                    .alloc_zeros::<f32>(batch * k * n)
                    .map_err(|_| {
                        cuda_alloc_failed("matmul_backward batched grad_b", vec![batch, k, n])
                    })?;
                let gemm = GemmConfig::<f32> {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: n as i32,
                    n: k as i32,
                    k: m as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: k as i32,
                    beta: 0.0,
                    ldc: n as i32,
                };
                let cfg = StridedBatchedConfig::<f32> {
                    gemm,
                    batch_size: batch as i32,
                    stride_a: (m * n) as i64,
                    stride_b: (m * k) as i64,
                    stride_c: (k * n) as i64,
                };
                // Safety: dims validated; buffers outlive the call.
                unsafe {
                    backend
                        .blas
                        .gemm_strided_batched(cfg, &d_g, &d_a, &mut c)
                        .map_err(|_| {
                            AutogradError::TapeInvariant(
                                "cuBLAS sgemm_strided_batched failed (grad_b)",
                            )
                        })?;
                }
                cuda_download(backend, &c, batch * k * n)?
            } else {
                Vec::new()
            };

            Ok((grad_a_host, grad_b_host))
        }
        _ => Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2 or rank-3",
            got: a_shape.len().max(b_shape.len()),
        }),
    }
}

// Device-resident sibling of `cuda_matmul_backward`. Same cuBLAS dispatch
// (two SGEMMs with OP_T on the transposed operand) but consumes existing
// `CudaSlice<f32>` handles via `cuda_slice` and emits the gradients as
// fresh `CudaSlice<f32>` buffers wrapped in `DeviceHandle::Cuda`. No
// `synchronize()` — the caller's terminal `eval` does the single host
// fence per training step (contract).
#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_matmul_backward_device(
    backend: &CudaBackend,
    a: &DeviceHandle,
    a_shape: &[usize],
    b: &DeviceHandle,
    b_shape: &[usize],
    grad_out: &DeviceHandle,
    grad_out_shape: &[usize],
    need_grad_a: bool,
    need_grad_b: bool,
) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
    let expected_out = matmul_output_shape(a_shape, b_shape)?;
    if grad_out_shape != expected_out.as_slice() {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_out,
            got: grad_out_shape.to_vec(),
        });
    }

    if !need_grad_a && !need_grad_b {
        return Ok((None, None));
    }

    let d_a_op = backend.f32_operand(a, "matmul_backward_device")?;
    let d_a = d_a_op.get();
    let d_b = backend.cuda_slice(b, "matmul_backward_device")?;
    let d_g_op = backend.f32_operand(grad_out, "matmul_backward_device")?;
    let d_g = d_g_op.get();

    if d_a.len() != shape_size(a_shape)
        || d_b.len() != shape_size(b_shape)
        || d_g.len() != shape_size(grad_out_shape)
    {
        return Err(AutogradError::TapeInvariant(
            "cuda matmul_backward_device handle size does not match shape",
        ));
    }

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => {
            let m = a_shape[0];
            let k = a_shape[1];
            let n = b_shape[1];

            let grad_a_handle = if need_grad_a {
                // grad_a[M,K] = grad_out[M,N] @ B^T[N,K]
                let mut c = backend
                    .stream
                    .alloc_zeros::<f32>(m * k)
                    .map_err(|_| cuda_alloc_failed("matmul_backward_device grad_a", vec![m, k]))?;
                let cfg = GemmConfig::<f32> {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: k as i32,
                    n: m as i32,
                    k: n as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: n as i32,
                    beta: 0.0,
                    ldc: k as i32,
                };
                // Safety: dims validated; device buffers outlive the call.
                unsafe {
                    backend.blas.gemm(cfg, d_b, d_g, &mut c).map_err(|_| {
                        AutogradError::TapeInvariant(
                            "cuBLAS sgemm failed (matmul_backward_device grad_a)",
                        )
                    })?;
                }
                Some(DeviceHandle::Cuda(CudaStorage::new(c)))
            } else {
                None
            };

            let grad_b_handle = if need_grad_b {
                // grad_b[K,N] = A^T[K,M] @ grad_out[M,N]
                let mut c = backend
                    .stream
                    .alloc_zeros::<f32>(k * n)
                    .map_err(|_| cuda_alloc_failed("matmul_backward_device grad_b", vec![k, n]))?;
                let cfg = GemmConfig::<f32> {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: n as i32,
                    n: k as i32,
                    k: m as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: k as i32,
                    beta: 0.0,
                    ldc: n as i32,
                };
                // Safety: dims validated; device buffers outlive the call.
                unsafe {
                    backend.blas.gemm(cfg, d_g, d_a, &mut c).map_err(|_| {
                        AutogradError::TapeInvariant(
                            "cuBLAS sgemm failed (matmul_backward_device grad_b)",
                        )
                    })?;
                }
                Some(DeviceHandle::Cuda(CudaStorage::new(c)))
            } else {
                None
            };

            Ok((grad_a_handle, grad_b_handle))
        }
        (3, 3) => {
            let batch = a_shape[0];
            let m = a_shape[1];
            let k = a_shape[2];
            let n = b_shape[2];
            if b_shape[0] != batch || grad_out_shape[0] != batch {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![batch],
                    got: vec![b_shape[0].min(grad_out_shape[0])],
                });
            }

            let grad_a_handle = if need_grad_a {
                let mut c = backend
                    .stream
                    .alloc_zeros::<f32>(batch * m * k)
                    .map_err(|_| {
                        cuda_alloc_failed(
                            "matmul_backward_device batched grad_a",
                            vec![batch, m, k],
                        )
                    })?;
                let gemm = GemmConfig::<f32> {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: k as i32,
                    n: m as i32,
                    k: n as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: n as i32,
                    beta: 0.0,
                    ldc: k as i32,
                };
                let cfg = StridedBatchedConfig::<f32> {
                    gemm,
                    batch_size: batch as i32,
                    stride_a: (k * n) as i64,
                    stride_b: (m * n) as i64,
                    stride_c: (m * k) as i64,
                };
                // Safety: dims validated; buffers outlive the call.
                unsafe {
                    backend
                        .blas
                        .gemm_strided_batched(cfg, d_b, d_g, &mut c)
                        .map_err(|_| {
                            AutogradError::TapeInvariant(
                                "cuBLAS sgemm_strided_batched failed (matmul_backward_device grad_a)",
                            )
                        })?;
                }
                Some(DeviceHandle::Cuda(CudaStorage::new(c)))
            } else {
                None
            };

            let grad_b_handle = if need_grad_b {
                let mut c = backend
                    .stream
                    .alloc_zeros::<f32>(batch * k * n)
                    .map_err(|_| {
                        cuda_alloc_failed(
                            "matmul_backward_device batched grad_b",
                            vec![batch, k, n],
                        )
                    })?;
                let gemm = GemmConfig::<f32> {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: n as i32,
                    n: k as i32,
                    k: m as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: k as i32,
                    beta: 0.0,
                    ldc: n as i32,
                };
                let cfg = StridedBatchedConfig::<f32> {
                    gemm,
                    batch_size: batch as i32,
                    stride_a: (m * n) as i64,
                    stride_b: (m * k) as i64,
                    stride_c: (k * n) as i64,
                };
                // Safety: dims validated; buffers outlive the call.
                unsafe {
                    backend
                        .blas
                        .gemm_strided_batched(cfg, d_g, d_a, &mut c)
                        .map_err(|_| {
                            AutogradError::TapeInvariant(
                                "cuBLAS sgemm_strided_batched failed (matmul_backward_device grad_b)",
                            )
                        })?;
                }
                Some(DeviceHandle::Cuda(CudaStorage::new(c)))
            } else {
                None
            };

            Ok((grad_a_handle, grad_b_handle))
        }
        _ => Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2 or rank-3",
            got: a_shape.len().max(b_shape.len()),
        }),
    }
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_matmul_bt_input_grad_device(
    backend: &CudaBackend,
    b: &DeviceHandle,
    b_shape: &[usize],
    grad_out: &DeviceHandle,
    grad_out_shape: &[usize],
    input_shape: &[usize],
) -> Result<DeviceHandle> {
    let expected_out = matmul_bt_output_shape(input_shape, b_shape)?;
    if grad_out_shape != expected_out.as_slice() {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_out,
            got: grad_out_shape.to_vec(),
        });
    }

    let d_g_op = backend.f32_operand(grad_out, "matmul_bt_input_grad_device")?;
    let d_g = d_g_op.get();
    if d_g.len() != shape_size(grad_out_shape) {
        return Err(AutogradError::TapeInvariant(
            "cuda matmul_bt_input_grad_device grad handle size does not match shape",
        ));
    }

    let (grad_a, grad_a_shape) = match b {
        DeviceHandle::Cuda(storage) => {
            let d_b = backend.cuda_storage_slice(storage)?;
            if d_b.len() != shape_size(b_shape) {
                return Err(AutogradError::TapeInvariant(
                    "cuda matmul_bt_input_grad_device handle size does not match shape",
                ));
            }
            backend.matmul_device(d_g, grad_out_shape, d_b, b_shape)?
        }
        DeviceHandle::CudaBf16(storage) => {
            let d_b = backend.cuda_bf16_storage_slice(storage)?;
            if d_b.len() != shape_size(b_shape) {
                return Err(AutogradError::TapeInvariant(
                    "cuda bf16 matmul_bt_input_grad_device handle size does not match shape",
                ));
            }
            backend.matmul_device_f32_bf16(d_g, grad_out_shape, d_b, b_shape)?
        }
        DeviceHandle::CudaFp8BlockScaled(storage) => {
            let (weight, _, rows, cols, _, _) = backend.cuda_fp8_block_scaled_storage(storage)?;
            if b_shape != [rows, cols] {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![rows, cols],
                    got: b_shape.to_vec(),
                });
            }
            if weight.len() != shape_size(b_shape) {
                return Err(AutogradError::TapeInvariant(
                    "cuda fp8 matmul_bt_input_grad_device handle size does not match shape",
                ));
            }
            backend.matmul_device_f32_fp8_block_scaled(d_g, grad_out_shape, storage)?
        }
        DeviceHandle::Cpu(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda matmul_bt_input_grad_device requires cuda handles",
            ));
        }
        #[cfg(feature = "metal")]
        DeviceHandle::Metal(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda matmul_bt_input_grad_device cannot use a metal handle",
            ));
        }
    };
    if grad_a_shape != input_shape {
        return Err(AutogradError::ShapeMismatch {
            expected: input_shape.to_vec(),
            got: grad_a_shape,
        });
    }
    Ok(DeviceHandle::Cuda(CudaStorage::new(grad_a)))
}

// Device-resident sibling of `cpu_matmul_bt_backward`.
#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_matmul_bt_backward_device(
    backend: &CudaBackend,
    a: &DeviceHandle,
    a_shape: &[usize],
    b: &DeviceHandle,
    b_shape: &[usize],
    grad_out: &DeviceHandle,
    grad_out_shape: &[usize],
    need_grad_a: bool,
    need_grad_b: bool,
) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
    let expected_out = matmul_bt_output_shape(a_shape, b_shape)?;
    if grad_out_shape != expected_out.as_slice() {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_out,
            got: grad_out_shape.to_vec(),
        });
    }
    if !need_grad_a && !need_grad_b {
        return Ok((None, None));
    }

    let d_a_op = backend.f32_operand(a, "matmul_bt_backward_device")?;
    let d_a = d_a_op.get();
    let d_g_op = backend.f32_operand(grad_out, "matmul_bt_backward_device")?;
    let d_g = d_g_op.get();
    if d_a.len() != shape_size(a_shape) || d_g.len() != shape_size(grad_out_shape) {
        return Err(AutogradError::TapeInvariant(
            "cuda matmul_bt_backward_device handle size does not match shape",
        ));
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];

    let grad_a = if need_grad_a {
        let (c, out_shape) = match b {
            DeviceHandle::Cuda(storage) => {
                let d_b = backend.cuda_storage_slice(storage)?;
                if d_b.len() != shape_size(b_shape) {
                    return Err(AutogradError::TapeInvariant(
                        "cuda matmul_bt_backward_device handle size does not match shape",
                    ));
                }
                backend.matmul_device(d_g, grad_out_shape, d_b, b_shape)?
            }
            DeviceHandle::CudaBf16(storage) => {
                let d_b = backend.cuda_bf16_storage_slice(storage)?;
                if d_b.len() != shape_size(b_shape) {
                    return Err(AutogradError::TapeInvariant(
                        "cuda bf16 matmul_bt_backward_device handle size does not match shape",
                    ));
                }
                backend.matmul_device_f32_bf16(d_g, grad_out_shape, d_b, b_shape)?
            }
            DeviceHandle::CudaFp8BlockScaled(storage) => {
                let (weight, _, rows, cols, _, _) =
                    backend.cuda_fp8_block_scaled_storage(storage)?;
                if b_shape != [rows, cols] {
                    return Err(AutogradError::ShapeMismatch {
                        expected: vec![rows, cols],
                        got: b_shape.to_vec(),
                    });
                }
                if weight.len() != shape_size(b_shape) {
                    return Err(AutogradError::TapeInvariant(
                        "cuda fp8 matmul_bt_backward_device handle size does not match shape",
                    ));
                }
                backend.matmul_device_f32_fp8_block_scaled(d_g, grad_out_shape, storage)?
            }
            DeviceHandle::Cpu(_) => {
                return Err(AutogradError::TapeInvariant(
                    "cuda matmul_bt_backward_device requires cuda handles",
                ));
            }
            #[cfg(feature = "metal")]
            DeviceHandle::Metal(_) => {
                return Err(AutogradError::TapeInvariant(
                    "cuda matmul_bt_backward_device cannot use a metal handle",
                ));
            }
        };
        if out_shape != a_shape {
            return Err(AutogradError::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: out_shape,
            });
        }
        Some(DeviceHandle::Cuda(CudaStorage::new(c)))
    } else {
        None
    };

    let grad_b = if need_grad_b {
        let d_b = backend.cuda_slice(b, "matmul_bt_backward_device")?;
        if d_b.len() != shape_size(b_shape) {
            return Err(AutogradError::TapeInvariant(
                "cuda matmul_bt_backward_device handle size does not match shape",
            ));
        }
        // grad_b[N,K] = grad_out^T[N,M] @ A[M,K]. The output's row-major
        // buffer is cuBLAS's column-major [K,N], so compute A^T[K,M] @
        // grad_out[M,N] directly into that column-major view.
        let mut c = backend
            .stream
            .alloc_zeros::<f32>(n * k)
            .map_err(|_| cuda_alloc_failed("matmul_bt_backward_device grad_b", vec![n, k]))?;
        let cfg = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: k as i32,
            n: n as i32,
            k: m as i32,
            alpha: 1.0,
            lda: k as i32,
            ldb: n as i32,
            beta: 0.0,
            ldc: k as i32,
        };
        // Safety: dims validated; device buffers outlive the call.
        unsafe {
            backend.blas.gemm(cfg, d_a, d_g, &mut c).map_err(|_| {
                AutogradError::TapeInvariant(
                    "cuBLAS sgemm failed (matmul_bt_backward_device grad_b)",
                )
            })?;
        }
        Some(DeviceHandle::Cuda(CudaStorage::new(c)))
    } else {
        None
    };

    Ok((grad_a, grad_b))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_causal_sdpa_recompute_backward_device(
    backend: &CudaBackend,
    args: CausalSdpaDeviceBackwardArgs<'_>,
) -> Result<CausalSdpaDeviceGradTriplet> {
    if args.shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: args.shape.len(),
        });
    }
    if !args.need_grad_q && !args.need_grad_k && !args.need_grad_v {
        return Ok((None, None, None));
    }

    let batch = args.shape[0];
    let heads = args.shape[1];
    let seq_len = args.shape[2];
    let head_dim = args.shape[3];
    let total = shape_size(args.shape);
    if seq_len == 0 || head_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero sequence and head_dim",
            got: args.shape.len(),
        });
    }

    if head_dim > 256 {
        let q = backend.readback(args.q)?;
        let k = backend.readback(args.k)?;
        let v = backend.readback(args.v)?;
        let upstream = backend.readback(args.upstream)?;
        let (grad_q, grad_k, grad_v) = cpu_causal_sdpa_recompute_backward(
            &q,
            &k,
            &v,
            &upstream,
            args.shape,
            args.need_grad_q,
            args.need_grad_k,
            args.need_grad_v,
        )?;
        return Ok((
            match grad_q {
                Some(grad) => Some(DeviceHandle::Cuda(CudaStorage::new(
                    backend.upload_slice(&grad, args.shape)?,
                ))),
                None => None,
            },
            match grad_k {
                Some(grad) => Some(DeviceHandle::Cuda(CudaStorage::new(
                    backend.upload_slice(&grad, args.shape)?,
                ))),
                None => None,
            },
            match grad_v {
                Some(grad) => Some(DeviceHandle::Cuda(CudaStorage::new(
                    backend.upload_slice(&grad, args.shape)?,
                ))),
                None => None,
            },
        ));
    }

    let d_q_op = backend.f32_operand(args.q, "causal_sdpa_recompute_backward_device")?;
    let d_k_op = backend.f32_operand(args.k, "causal_sdpa_recompute_backward_device")?;
    let d_v_op = backend.f32_operand(args.v, "causal_sdpa_recompute_backward_device")?;
    let d_up_op = backend.f32_operand(args.upstream, "causal_sdpa_recompute_backward_device")?;
    let (d_q, d_k, d_v, d_up) = (d_q_op.get(), d_k_op.get(), d_v_op.get(), d_up_op.get());
    if d_q.len() != total || d_k.len() != total || d_v.len() != total || d_up.len() != total {
        return Err(AutogradError::TapeInvariant(
            "cuda causal_sdpa_recompute_backward_device handle size does not match shape",
        ));
    }

    let out_len_q = if args.need_grad_q { total } else { 1 };
    let out_len_k = if args.need_grad_k { total } else { 1 };
    let out_len_v = if args.need_grad_v { total } else { 1 };
    let mut grad_q = backend
        .stream
        .alloc_zeros::<f32>(out_len_q)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sdpa grad_q)"))?;
    let mut grad_k = backend
        .stream
        .alloc_zeros::<f32>(out_len_k)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sdpa grad_k)"))?;
    let mut grad_v = backend
        .stream
        .alloc_zeros::<f32>(out_len_v)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sdpa grad_v)"))?;

    let rows = batch
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(seq_len))
        .ok_or(AutogradError::TapeInvariant("cuda sdpa row count overflow"))?;
    let rows_i = i32::try_from(rows)
        .map_err(|_| AutogradError::TapeInvariant("cuda sdpa rows exceeds i32"))?;
    let seq_i = i32::try_from(seq_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda sdpa seq_len exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda sdpa head_dim exceeds i32"))?;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let need_q_i = if args.need_grad_q { 1 } else { 0 };
    let need_k_i = if args.need_grad_k { 1 } else { 0 };
    let need_v_i = if args.need_grad_v { 1 } else { 0 };

    const BLOCK: u32 = 32;
    const SHARED: u32 = 0;
    launch_rows(
        &backend.stream,
        backend
            .kernels
            .function("causal_sdpa_recompute_backward_f32")?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder
                .arg(d_q)
                .arg(d_k)
                .arg(d_v)
                .arg(d_up)
                .arg(&mut grad_q)
                .arg(&mut grad_k)
                .arg(&mut grad_v)
                .arg(&rows_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&scale)
                .arg(&need_q_i)
                .arg(&need_k_i)
                .arg(&need_v_i);
            builder
        },
    )?;

    Ok((
        args.need_grad_q
            .then(|| DeviceHandle::Cuda(CudaStorage::new(grad_q))),
        args.need_grad_k
            .then(|| DeviceHandle::Cuda(CudaStorage::new(grad_k))),
        args.need_grad_v
            .then(|| DeviceHandle::Cuda(CudaStorage::new(grad_v))),
    ))
}

// Device-resident backward for `mul_scalar(x, k)`. Reads
// `upstream[i] * k` via `mul_scalar_backward_f32` (functionally identical
// to the forward `mul_scalar_f32`, but kept as a separately-registered
// kernel so the audit trail in nsys traces matches the autograd op name).
// Returned handle is unevaluated — terminal `eval` is the caller's.
#[cfg(not(feature = "no-cuda"))]
fn cuda_mul_scalar_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    scale: f32,
    shape: &[usize],
) -> Result<DeviceHandle> {
    let d_up = backend.cuda_slice(upstream, "mul_scalar_backward_device")?;
    let size = shape_size(shape);
    if d_up.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_up.len(),
            shape: shape.to_vec(),
            size,
        });
    }

    let mut d_out = backend.stream.alloc_zeros::<f32>(size).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (mul_scalar_backward_device)")
    })?;
    let n = i32::try_from(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda mul_scalar_backward length exceeds i32"))?;
    launch_1d(
        &backend.stream,
        backend.kernels.function("mul_scalar_backward_f32")?,
        size,
        |mut builder| {
            builder.arg(&mut d_out).arg(d_up).arg(&scale).arg(&n);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

// Device-resident backward for `mean(x)`. The upstream is a rank-0
// device scalar; the kernel reads it once per thread (block-broadcast
// from L1 after the first warp) and writes `upstream * (1/N)` across
// `elem_count` slots. Returned handle is unevaluated.
#[cfg(not(feature = "no-cuda"))]
fn cuda_mean_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    output_shape: &[usize],
    elem_count: usize,
) -> Result<DeviceHandle> {
    let d_up = backend.cuda_slice(upstream, "mean_backward_device")?;
    if d_up.len() != 1 {
        return Err(AutogradError::ShapeMismatch {
            expected: Vec::new(),
            got: vec![d_up.len()],
        });
    }
    let expected = shape_size(output_shape);
    if expected != elem_count {
        return Err(AutogradError::DataLengthMismatch {
            len: elem_count,
            shape: output_shape.to_vec(),
            size: expected,
        });
    }

    let inv_n: f32 = if elem_count == 0 {
        0.0
    } else {
        1.0 / elem_count as f32
    };
    let mut d_out = backend.stream.alloc_zeros::<f32>(elem_count).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (mean_backward_device)")
    })?;
    let n = i32::try_from(elem_count)
        .map_err(|_| AutogradError::TapeInvariant("cuda mean_backward length exceeds i32"))?;
    launch_1d(
        &backend.stream,
        backend.kernels.function("mean_backward_f32")?,
        elem_count,
        |mut builder| {
            builder.arg(&mut d_out).arg(d_up).arg(&inv_n).arg(&n);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_sum_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    output_shape: &[usize],
) -> Result<DeviceHandle> {
    let d_up = backend.cuda_slice(upstream, "sum_backward_device")?;
    if d_up.len() != 1 {
        return Err(AutogradError::ShapeMismatch {
            expected: Vec::new(),
            got: vec![d_up.len()],
        });
    }
    let elem_count = shape_size(output_shape);
    let mut d_out = backend.stream.alloc_zeros::<f32>(elem_count).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (sum_backward_device)")
    })?;
    let n = i32::try_from(elem_count)
        .map_err(|_| AutogradError::TapeInvariant("cuda sum_backward length exceeds i32"))?;
    let scale = 1.0_f32;
    launch_1d(
        &backend.stream,
        backend.kernels.function("mean_backward_f32")?,
        elem_count,
        |mut builder| {
            builder.arg(&mut d_out).arg(d_up).arg(&scale).arg(&n);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

// Device-resident gradient accumulation. Allocates a fresh output buffer
// and writes `dest[i] + src[i]` via the `add_into_f32` NVRTC kernel. The
// returned handle is unevaluated — terminal `eval` is the caller's.
#[cfg(not(feature = "no-cuda"))]
fn cuda_add_into_device(
    backend: &CudaBackend,
    dest: &DeviceHandle,
    src: &DeviceHandle,
    shape: &[usize],
) -> Result<DeviceHandle> {
    let size = shape_size(shape);
    // Dest dtype decides the lane: bf16 activation-grad chains stay bf16,
    // f32 (param-grad) accumulators stay f32 with bf16 sources widened.
    if let DeviceHandle::CudaBf16(storage) = dest {
        let d_dest = backend.cuda_bf16_storage_slice(storage)?;
        let d_src_op = backend.bf16_operand(src, "add_into_device")?;
        let d_src = d_src_op.get();
        if d_dest.len() != size || d_src.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: d_dest.len().min(d_src.len()),
                shape: shape.to_vec(),
                size,
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(size)
            .map_err(|e| cuda_alloc_failed_rich(backend, "add_into_device", size * 2, &e))?;
        let n = size as u64;
        let func = backend
            .kernels
            .function_for("add_into_f32", TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, size, |mut builder| {
            builder.arg(&mut d_out).arg(d_dest).arg(d_src).arg(&n);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }
    let d_dest = backend.cuda_slice(dest, "add_into_device")?;
    let d_src_op = backend.f32_operand(src, "add_into_device")?;
    let d_src = d_src_op.get();
    if d_dest.len() != size || d_src.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_dest.len().min(d_src.len()),
            shape: shape.to_vec(),
            size,
        });
    }

    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(size)
        .map_err(|e| cuda_alloc_failed_rich(backend, "add_into_device", size * 4, &e))?;
    let n = size as u64;
    launch_1d(
        &backend.stream,
        backend.kernels.function("add_into_f32")?,
        size,
        |mut builder| {
            builder.arg(&mut d_out).arg(d_dest).arg(d_src).arg(&n);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_accumulate_into_device(
    backend: &CudaBackend,
    dest: &DeviceHandle,
    src: &DeviceHandle,
    shape: &[usize],
) -> Result<DeviceHandle> {
    let size = shape_size(shape);
    if let DeviceHandle::CudaBf16(storage) = dest {
        let d_dest = backend.cuda_bf16_storage_slice(storage)?;
        let d_src_op = backend.bf16_operand(src, "accumulate_into_device")?;
        let d_src = d_src_op.get();
        if d_dest.len() != size || d_src.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: d_dest.len().min(d_src.len()),
                shape: shape.to_vec(),
                size,
            });
        }
        let n = size as u64;
        let func = backend
            .kernels
            .function_for("accumulate_into_f32", TapeDtype::Bf16)?;
        let (dest_ptr, _dest_guard) = d_dest.device_ptr(&backend.stream);
        launch_1d(&backend.stream, &func, size, |mut builder| {
            builder.arg(&dest_ptr).arg(d_src).arg(&n);
            builder
        })?;
        return Ok(dest.clone());
    }
    let d_dest = backend.cuda_slice(dest, "accumulate_into_device")?;
    let d_src_op = backend.f32_operand(src, "accumulate_into_device")?;
    let d_src = d_src_op.get();
    if d_dest.len() != size || d_src.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_dest.len().min(d_src.len()),
            shape: shape.to_vec(),
            size,
        });
    }
    let n = size as u64;
    let (dest_ptr, _dest_guard) = d_dest.device_ptr(&backend.stream);
    launch_1d(
        &backend.stream,
        backend.kernels.function("accumulate_into_f32")?,
        size,
        |mut builder| {
            builder.arg(&dest_ptr).arg(d_src).arg(&n);
            builder
        },
    )?;
    Ok(dest.clone())
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_unary_1d(backend: &CudaBackend, a: &[f32], kernel_name: &'static str) -> Result<Vec<f32>> {
    let n_usize = a.len();
    let d_in = backend
        .stream
        .clone_htod(a)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(n_usize)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
    let n = n_usize as u64;
    launch_1d(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        n_usize,
        |mut builder| {
            builder.arg(&mut d_out).arg(&d_in).arg(&n);
            builder
        },
    )?;
    cuda_download(backend, &d_out, n_usize)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_scalar_1d(
    backend: &CudaBackend,
    a: &[f32],
    s: f32,
    kernel_name: &'static str,
) -> Result<Vec<f32>> {
    let n_usize = a.len();
    let d_in = backend
        .stream
        .clone_htod(a)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(n_usize)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
    let n = n_usize as u64;
    launch_1d(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        n_usize,
        |mut builder| {
            builder.arg(&mut d_out).arg(&d_in).arg(&s).arg(&n);
            builder
        },
    )?;
    cuda_download(backend, &d_out, n_usize)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_binary_1d(
    backend: &CudaBackend,
    a: &[f32],
    b: &[f32],
    kernel_name: &'static str,
) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![a.len()],
            got: vec![b.len()],
        });
    }
    let n_usize = a.len();
    let d_a = backend
        .stream
        .clone_htod(a)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let d_b = backend
        .stream
        .clone_htod(b)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(n_usize)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
    let n = n_usize as u64;
    launch_1d(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        n_usize,
        |mut builder| {
            builder.arg(&mut d_out).arg(&d_a).arg(&d_b).arg(&n);
            builder
        },
    )?;
    cuda_download(backend, &d_out, n_usize)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_unary_1d_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    shape: &[usize],
    kernel_name: &'static str,
    op_label: &'static str,
) -> Result<DeviceHandle> {
    let size = shape_size(shape);
    if backend.tape_bf16() {
        let d_in_op = backend.bf16_operand(x, op_label)?;
        let d_in = d_in_op.get();
        if d_in.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: d_in.len(),
                shape: shape.to_vec(),
                size,
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(size)
            .map_err(|e| cuda_alloc_failed_rich(backend, op_label, size * 2, &e))?;
        let n = size as u64;
        let func = backend.kernels.function_for(kernel_name, TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, size, |mut builder| {
            builder.arg(&mut d_out).arg(d_in).arg(&n);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }
    let d_in = backend.cuda_slice(x, op_label)?;
    if d_in.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_in.len(),
            shape: shape.to_vec(),
            size,
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(size)
        .map_err(|e| cuda_alloc_failed_rich(backend, op_label, size * 4, &e))?;
    let n = size as u64;
    launch_1d(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        size,
        |mut builder| {
            builder.arg(&mut d_out).arg(d_in).arg(&n);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_scalar_1d_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    s: f32,
    shape: &[usize],
    kernel_name: &'static str,
    op_label: &'static str,
) -> Result<DeviceHandle> {
    let size = shape_size(shape);
    if backend.tape_bf16() {
        let d_in_op = backend.bf16_operand(x, op_label)?;
        let d_in = d_in_op.get();
        if d_in.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: d_in.len(),
                shape: shape.to_vec(),
                size,
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(size)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
        let n = size as u64;
        let func = backend.kernels.function_for(kernel_name, TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, size, |mut builder| {
            builder.arg(&mut d_out).arg(d_in).arg(&s).arg(&n);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }
    let d_in = backend.cuda_slice(x, op_label)?;
    if d_in.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_in.len(),
            shape: shape.to_vec(),
            size,
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
    let n = size as u64;
    launch_1d(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        size,
        |mut builder| {
            builder.arg(&mut d_out).arg(d_in).arg(&s).arg(&n);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_binary_1d_device(
    backend: &CudaBackend,
    a: &DeviceHandle,
    b: &DeviceHandle,
    shape: &[usize],
    kernel_name: &'static str,
    op_label: &'static str,
) -> Result<DeviceHandle> {
    let size = shape_size(shape);
    if backend.tape_bf16() {
        let d_a_op = backend.bf16_operand(a, op_label)?;
        let d_b_op = backend.bf16_operand(b, op_label)?;
        let d_a = d_a_op.get();
        let d_b = d_b_op.get();
        if d_a.len() != size || d_b.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: d_a.len().min(d_b.len()),
                shape: shape.to_vec(),
                size,
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(size)
            .map_err(|e| cuda_alloc_failed_rich(backend, op_label, size * 2, &e))?;
        let n = size as u64;
        let func = backend.kernels.function_for(kernel_name, TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, size, |mut builder| {
            builder.arg(&mut d_out).arg(d_a).arg(d_b).arg(&n);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }
    let d_a = backend.cuda_slice(a, op_label)?;
    let d_b = backend.cuda_slice(b, op_label)?;
    if d_a.len() != size || d_b.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_a.len().min(d_b.len()),
            shape: shape.to_vec(),
            size,
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(size)
        .map_err(|e| cuda_alloc_failed_rich(backend, op_label, size * 4, &e))?;
    let n = size as u64;
    launch_1d(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        size,
        |mut builder| {
            builder.arg(&mut d_out).arg(d_a).arg(d_b).arg(&n);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_rms_norm(
    backend: &CudaBackend,
    x: &[f32],
    weight: &[f32],
    shape: &[usize],
    eps: f32,
) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected: usize = shape.iter().product();
    if x.len() != expected {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![x.len()],
        });
    }
    if weight.len() != last_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![last_dim],
            got: vec![weight.len()],
        });
    }
    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rms_norm cols exceeds i32"))?;
    let d_x = backend.upload_slice(x, shape)?;
    let d_w = backend
        .stream
        .clone_htod(weight)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(expected)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
    launch_rows(
        &backend.stream,
        backend.kernels.function("rms_norm_f32")?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_x)
                .arg(&d_w)
                .arg(&cols)
                .arg(&eps);
            builder
        },
    )?;
    cuda_download(backend, &d_out, expected)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_rms_norm_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    weight: &[f32],
    shape: &[usize],
    eps: f32,
) -> Result<DeviceHandle> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected = shape_size(shape);
    if weight.len() != last_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![last_dim],
            got: vec![weight.len()],
        });
    }
    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rms_norm cols exceeds i32"))?;
    let d_w = backend
        .stream
        .clone_htod(weight)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;

    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
    if let DeviceHandle::CudaBf16(storage) = x {
        let d_x = backend.cuda_bf16_storage_slice(storage)?;
        if d_x.len() != expected {
            return Err(AutogradError::DataLengthMismatch {
                len: d_x.len(),
                shape: shape.to_vec(),
                size: expected,
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(expected)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
        let func = backend
            .kernels
            .function_for("rms_norm_f32", TapeDtype::Bf16)?;
        launch_rows(
            &backend.stream,
            &func,
            rows,
            BLOCK,
            SHARED,
            |mut builder| {
                builder
                    .arg(&mut d_out)
                    .arg(d_x)
                    .arg(&d_w)
                    .arg(&cols)
                    .arg(&eps);
                builder
            },
        )?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }
    let d_x = backend.cuda_slice(x, "rms_norm")?;
    if d_x.len() != expected {
        return Err(AutogradError::DataLengthMismatch {
            len: d_x.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(expected)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    launch_rows(
        &backend.stream,
        backend.kernels.function("rms_norm_f32")?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_x)
                .arg(&d_w)
                .arg(&cols)
                .arg(&eps);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_embedding(
    backend: &CudaBackend,
    weight: &[f32],
    vocab: usize,
    dim: usize,
    ids: &[i32],
) -> Result<Vec<f32>> {
    if weight.len() != vocab * dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![vocab * dim],
            got: vec![weight.len()],
        });
    }
    let n_ids = ids.len();
    let out_len = n_ids * dim;
    let d_w = backend
        .stream
        .clone_htod(weight)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let d_ids = backend
        .stream
        .clone_htod(ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(out_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    let n_i32 = i32::try_from(n_ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding n_ids exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding vocab exceeds i32"))?;
    let dim_i32 = i32::try_from(dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding dim exceeds i32"))?;

    const BLOCK: u32 = 256;
    launch_rows(
        &backend.stream,
        backend.kernels.function("embedding_f32")?,
        n_ids,
        BLOCK,
        0,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_w)
                .arg(&d_ids)
                .arg(&n_i32)
                .arg(&vocab_i32)
                .arg(&dim_i32);
            builder
        },
    )?;
    cuda_download(backend, &d_out, out_len)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_embedding_device(
    backend: &CudaBackend,
    table: &DeviceHandle,
    table_shape: &[usize],
    ids: &[i32],
) -> Result<DeviceHandle> {
    if table_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: table_shape.len(),
        });
    }
    let vocab = table_shape[0];
    let dim = table_shape[1];
    let n_ids = ids.len();
    let out_len = n_ids * dim;
    let d_ids = backend
        .stream
        .clone_htod(ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (embedding ids)"))?;

    let n_i32 = i32::try_from(n_ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding n_ids exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding vocab exceeds i32"))?;
    let dim_i32 = i32::try_from(dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding dim exceeds i32"))?;

    const BLOCK: u32 = 256;
    // The table stays a parameter (f32 or bf16 weights); only the gathered
    // output activation follows the tape dtype.
    let (kernel_name, d_w_f32, d_w_bf16) = match table {
        DeviceHandle::Cuda(storage) => {
            let d_w = backend.cuda_storage_slice(storage)?;
            if d_w.len() != vocab * dim {
                return Err(AutogradError::DataLengthMismatch {
                    len: d_w.len(),
                    shape: table_shape.to_vec(),
                    size: vocab * dim,
                });
            }
            ("embedding_f32", Some(d_w), None)
        }
        DeviceHandle::CudaBf16(storage) => {
            let d_w = backend.cuda_bf16_storage_slice(storage)?;
            if d_w.len() != vocab * dim {
                return Err(AutogradError::DataLengthMismatch {
                    len: d_w.len(),
                    shape: table_shape.to_vec(),
                    size: vocab * dim,
                });
            }
            ("embedding_bf16_to_f32", None, Some(d_w))
        }
        DeviceHandle::CudaFp8BlockScaled(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding a fp8 block-scaled device handle",
            ));
        }
        DeviceHandle::Cpu(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding a cpu device handle",
            ));
        }
        #[cfg(feature = "metal")]
        DeviceHandle::Metal(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding a metal device handle",
            ));
        }
    };

    if backend.tape_bf16() {
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(out_len)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (embedding)"))?;
        let func = backend.kernels.function_for(kernel_name, TapeDtype::Bf16)?;
        launch_rows(&backend.stream, &func, n_ids, BLOCK, 0, |mut builder| {
            match (&d_w_f32, &d_w_bf16) {
                (Some(d_w), _) => builder.arg(&mut d_out).arg(*d_w),
                (_, Some(d_w)) => builder.arg(&mut d_out).arg(*d_w),
                _ => unreachable!("embedding table matched above"),
            };
            builder
                .arg(&d_ids)
                .arg(&n_i32)
                .arg(&vocab_i32)
                .arg(&dim_i32);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }

    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(out_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (embedding)"))?;
    launch_rows(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        n_ids,
        BLOCK,
        0,
        |mut builder| {
            match (&d_w_f32, &d_w_bf16) {
                (Some(d_w), _) => builder.arg(&mut d_out).arg(*d_w),
                (_, Some(d_w)) => builder.arg(&mut d_out).arg(*d_w),
                _ => unreachable!("embedding table matched above"),
            };
            builder
                .arg(&d_ids)
                .arg(&n_i32)
                .arg(&vocab_i32)
                .arg(&dim_i32);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_embedding_from_f32_ids_device(
    backend: &CudaBackend,
    table: &DeviceHandle,
    table_shape: &[usize],
    ids: &DeviceHandle,
    n_ids: usize,
) -> Result<DeviceHandle> {
    if table_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: table_shape.len(),
        });
    }
    let vocab = table_shape[0];
    let dim = table_shape[1];
    let d_ids = backend.cuda_slice(ids, "embedding_from_f32_ids")?;
    if d_ids.len() != n_ids {
        return Err(AutogradError::DataLengthMismatch {
            len: d_ids.len(),
            shape: vec![n_ids],
            size: n_ids,
        });
    }
    let out_len = n_ids * dim;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(out_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (embedding f32 ids)"))?;

    let n_i32 = i32::try_from(n_ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding n_ids exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding vocab exceeds i32"))?;
    let dim_i32 = i32::try_from(dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding dim exceeds i32"))?;

    const BLOCK: u32 = 256;
    match table {
        DeviceHandle::Cuda(storage) => {
            let d_w = backend.cuda_storage_slice(storage)?;
            if d_w.len() != vocab * dim {
                return Err(AutogradError::DataLengthMismatch {
                    len: d_w.len(),
                    shape: table_shape.to_vec(),
                    size: vocab * dim,
                });
            }
            launch_rows(
                &backend.stream,
                backend.kernels.function("embedding_f32_ids_f32")?,
                n_ids,
                BLOCK,
                0,
                |mut builder| {
                    builder
                        .arg(&mut d_out)
                        .arg(d_w)
                        .arg(d_ids)
                        .arg(&n_i32)
                        .arg(&vocab_i32)
                        .arg(&dim_i32);
                    builder
                },
            )?;
        }
        DeviceHandle::CudaBf16(storage) => {
            let d_w = backend.cuda_bf16_storage_slice(storage)?;
            if d_w.len() != vocab * dim {
                return Err(AutogradError::DataLengthMismatch {
                    len: d_w.len(),
                    shape: table_shape.to_vec(),
                    size: vocab * dim,
                });
            }
            launch_rows(
                &backend.stream,
                backend.kernels.function("embedding_bf16_ids_f32")?,
                n_ids,
                BLOCK,
                0,
                |mut builder| {
                    builder
                        .arg(&mut d_out)
                        .arg(d_w)
                        .arg(d_ids)
                        .arg(&n_i32)
                        .arg(&vocab_i32)
                        .arg(&dim_i32);
                    builder
                },
            )?;
        }
        DeviceHandle::CudaFp8BlockScaled(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding_from_f32_ids a fp8 block-scaled device handle",
            ));
        }
        DeviceHandle::Cpu(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding_from_f32_ids a cpu device handle",
            ));
        }
        #[cfg(feature = "metal")]
        DeviceHandle::Metal(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding_from_f32_ids a metal device handle",
            ));
        }
    }

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_argmax_last_dim(
    backend: &CudaBackend,
    x: &DeviceHandle,
    shape: &[usize],
) -> Result<DeviceHandle> {
    let vocab = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if vocab == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-empty last dim",
            got: 0,
        });
    }
    let total = shape_size(shape);
    if !total.is_multiple_of(vocab) {
        return Err(AutogradError::DataLengthMismatch {
            len: total,
            shape: shape.to_vec(),
            size: total,
        });
    }
    let rows = total / vocab;
    let d_x = backend.cuda_slice(x, "argmax_last_dim")?;
    if d_x.len() != total {
        return Err(AutogradError::DataLengthMismatch {
            len: d_x.len(),
            shape: shape.to_vec(),
            size: total,
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(rows)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (argmax)"))?;
    let rows_i = i32::try_from(rows)
        .map_err(|_| AutogradError::TapeInvariant("cuda argmax rows exceeds i32"))?;
    let vocab_i = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda argmax vocab exceeds i32"))?;
    const BLOCK: u32 = 256;
    let shared = BLOCK * (std::mem::size_of::<f32>() as u32 + std::mem::size_of::<i32>() as u32);
    launch_rows(
        &backend.stream,
        backend.kernels.function("argmax_last_dim_f32")?,
        rows,
        BLOCK,
        shared,
        |mut builder| {
            builder.arg(&mut d_out).arg(d_x).arg(&rows_i).arg(&vocab_i);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_write_scalar_at(
    backend: &CudaBackend,
    dest: &DeviceHandle,
    src: &DeviceHandle,
    len: usize,
    index: usize,
) -> Result<DeviceHandle> {
    if index >= len {
        return Err(AutogradError::IndexOutOfBounds { index, upper: len });
    }
    let d_dest = backend.cuda_slice(dest, "write_scalar_at")?;
    let d_src = backend.cuda_slice(src, "write_scalar_at")?;
    if d_dest.len() != len || d_src.is_empty() {
        return Err(AutogradError::DataLengthMismatch {
            len: d_dest.len(),
            shape: vec![len],
            size: len,
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (write scalar)"))?;
    let len_i = i32::try_from(len)
        .map_err(|_| AutogradError::TapeInvariant("cuda write scalar len exceeds i32"))?;
    let index_i = i32::try_from(index)
        .map_err(|_| AutogradError::TapeInvariant("cuda write scalar index exceeds i32"))?;
    launch_1d(
        &backend.stream,
        backend.kernels.function("write_scalar_at_f32")?,
        len,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_dest)
                .arg(d_src)
                .arg(&len_i)
                .arg(&index_i);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

// Device-resident full reduction `out[0] = sum(x[0..size])`. Multi-pass
// block reduce via `sum_partial_f32`: pass 1 reduces `size` elements into
// `ceil(size/BLOCK)` partials; each subsequent pass reduces the partials the
// same way until one element remains. Returns a 1-element device handle with
// NO host transfer and NO `synchronize()` — the launches are enqueued on the
// backend stream and the caller's terminal eval forces them. This is the
// device-resident sibling of the host-reduce path `sum_all` takes.
#[cfg(not(feature = "no-cuda"))]
fn cuda_sum_all_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    size: usize,
) -> Result<DeviceHandle> {
    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;

    // size == 0 → empty sum is 0.0; size == 1 → already a scalar, copy as-is
    // through one trivial reduce so the returned buffer is freshly owned.
    if size == 0 {
        let d_out = backend
            .stream
            .alloc_zeros::<f32>(1)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sum_all empty)"))?;
        return Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)));
    }

    // First pass reads the borrowed input slice; later passes read the
    // previous pass's owned partial buffer. `function()` is re-fetched inline
    // per launch (a cheap HashMap lookup) so each `&CudaFunction` borrow is
    // scoped to a single `launch_rows` call — mirrors `cuda_sum_squares`.
    let in_slice = backend.cuda_slice(x, "sum_all")?;
    let mut n = size;
    let mut blocks = n.div_ceil(BLOCK as usize);
    let n_i32 = i32::try_from(n)
        .map_err(|_| AutogradError::TapeInvariant("cuda sum_all size exceeds i32"))?;
    let mut current = backend
        .stream
        .alloc_zeros::<f32>(blocks)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sum_all)"))?;
    launch_rows(
        &backend.stream,
        backend.kernels.function("sum_partial_f32")?,
        blocks,
        BLOCK,
        SHARED,
        |mut builder| {
            builder.arg(&mut current).arg(in_slice).arg(&n_i32);
            builder
        },
    )?;

    // Recursively reduce the partials until a single scalar remains.
    while blocks > 1 {
        n = blocks;
        blocks = n.div_ceil(BLOCK as usize);
        let pass_n = i32::try_from(n)
            .map_err(|_| AutogradError::TapeInvariant("cuda sum_all partials exceed i32"))?;
        let mut next = backend
            .stream
            .alloc_zeros::<f32>(blocks)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sum_all pass)"))?;
        launch_rows(
            &backend.stream,
            backend.kernels.function("sum_partial_f32")?,
            blocks,
            BLOCK,
            SHARED,
            |mut builder| {
                builder.arg(&mut next).arg(&current).arg(&pass_n);
                builder
            },
        )?;
        current = next;
    }

    Ok(DeviceHandle::Cuda(CudaStorage::new(current)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_sum_squares(backend: &CudaBackend, x: &DeviceHandle, shape: &[usize]) -> Result<f64> {
    let size = shape_size(shape);
    let d_x = backend.cuda_slice(x, "sum_squares")?;
    if d_x.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_x.len(),
            shape: shape.to_vec(),
            size,
        });
    }
    if size == 0 {
        return Ok(0.0);
    }

    const BLOCK: u32 = 256;
    let blocks = size.div_ceil(BLOCK as usize);
    let mut d_partial = backend
        .stream
        .alloc_zeros::<f64>(blocks)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sum_squares)"))?;
    let n_i32 = i32::try_from(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda sum_squares size exceeds i32"))?;
    launch_rows(
        &backend.stream,
        backend.kernels.function("sum_squares_partial_f32")?,
        blocks,
        BLOCK,
        BLOCK * std::mem::size_of::<f64>() as u32,
        |mut builder| {
            builder.arg(&mut d_partial).arg(d_x).arg(&n_i32);
            builder
        },
    )?;

    let mut partial = vec![0.0_f64; blocks];
    backend
        .stream
        .memcpy_dtoh(&d_partial, &mut partial)
        .map_err(|_| AutogradError::TapeInvariant("cuda dtoh copy failed (sum_squares)"))?;
    backend
        .stream
        .synchronize()
        .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed (sum_squares)"))?;
    Ok(partial.into_iter().sum())
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_clip_grad_norm_device(
    backend: &CudaBackend,
    grads: &[(DeviceHandle, Vec<usize>)],
    max_norm: f32,
) -> Result<DeviceGradClipResult> {
    if !(max_norm > 0.0 && max_norm.is_finite()) {
        return Ok(DeviceGradClipResult {
            pre_clip_norm: 0.0,
            clipped_grads: None,
        });
    }
    if grads.is_empty() {
        return Ok(DeviceGradClipResult {
            pre_clip_norm: 0.0,
            clipped_grads: None,
        });
    }

    const BLOCK: u32 = 256;
    const ITEMS_PER_THREAD: usize = 8;
    const CHUNK_ELEMS: usize = BLOCK as usize * ITEMS_PER_THREAD;

    let mut grad_ptrs = Vec::with_capacity(grads.len());
    let mut grad_sizes = Vec::with_capacity(grads.len());
    let mut chunk_offsets = Vec::with_capacity(grads.len() + 1);
    let mut input_guards = Vec::with_capacity(grads.len());
    let mut total_chunks = 0usize;
    chunk_offsets.push(0_i32);

    for (handle, shape) in grads {
        let size = shape_size(shape);
        let d_grad = backend.cuda_slice(handle, "clip_grad_norm_device")?;
        if d_grad.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: d_grad.len(),
                shape: shape.clone(),
                size,
            });
        }
        let size_i32 = i32::try_from(size)
            .map_err(|_| AutogradError::TapeInvariant("cuda grad_clip tensor size exceeds i32"))?;
        let chunks = size.div_ceil(CHUNK_ELEMS);
        total_chunks = total_chunks
            .checked_add(chunks)
            .ok_or(AutogradError::TapeInvariant(
                "cuda grad_clip total chunk count overflow",
            ))?;
        let total_chunks_i32 = i32::try_from(total_chunks)
            .map_err(|_| AutogradError::TapeInvariant("cuda grad_clip chunks exceed i32"))?;
        let (ptr, guard) = d_grad.device_ptr(&backend.stream);
        grad_ptrs.push(ptr);
        input_guards.push(guard);
        grad_sizes.push(size_i32);
        chunk_offsets.push(total_chunks_i32);
    }

    if total_chunks == 0 {
        return Ok(DeviceGradClipResult {
            pre_clip_norm: 0.0,
            clipped_grads: None,
        });
    }

    let d_grad_ptrs = backend
        .stream
        .clone_htod(&grad_ptrs)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (grad_clip ptrs)"))?;
    let d_grad_sizes = backend
        .stream
        .clone_htod(&grad_sizes)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (grad_clip sizes)"))?;
    let d_chunk_offsets = backend
        .stream
        .clone_htod(&chunk_offsets)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (grad_clip offsets)"))?;
    let mut d_partial = backend
        .stream
        .alloc_zeros::<f64>(total_chunks)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (grad_clip partial)"))?;
    let num_grads_i32 = i32::try_from(grads.len())
        .map_err(|_| AutogradError::TapeInvariant("cuda grad_clip grad count exceeds i32"))?;
    let chunk_elems_i32 = i32::try_from(CHUNK_ELEMS)
        .map_err(|_| AutogradError::TapeInvariant("cuda grad_clip chunk size exceeds i32"))?;

    launch_rows(
        &backend.stream,
        backend.kernels.function("grad_clip_sumsq_f32")?,
        total_chunks,
        BLOCK,
        BLOCK * std::mem::size_of::<f64>() as u32,
        |mut builder| {
            builder
                .arg(&mut d_partial)
                .arg(&d_grad_ptrs)
                .arg(&d_grad_sizes)
                .arg(&d_chunk_offsets)
                .arg(&num_grads_i32)
                .arg(&chunk_elems_i32);
            builder
        },
    )?;

    let mut partial = vec![0.0_f64; total_chunks];
    backend
        .stream
        .memcpy_dtoh(&d_partial, &mut partial)
        .map_err(|_| AutogradError::TapeInvariant("cuda dtoh copy failed (grad_clip partial)"))?;
    backend
        .stream
        .synchronize()
        .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed (grad_clip norm)"))?;
    let total_sq_norm = partial.into_iter().sum::<f64>();
    let pre_clip_norm = total_sq_norm.sqrt();
    if pre_clip_norm <= f64::from(max_norm) || pre_clip_norm == 0.0 {
        return Ok(DeviceGradClipResult {
            pre_clip_norm,
            clipped_grads: None,
        });
    }

    let scale = (f64::from(max_norm) / pre_clip_norm) as f32;
    let mut out_slices = grads
        .iter()
        .map(|(_, shape)| {
            backend
                .stream
                .alloc_zeros::<f32>(shape_size(shape))
                .map_err(|_| {
                    AutogradError::TapeInvariant("cuda alloc_zeros failed (grad_clip scaled)")
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let (out_ptrs, out_guards): (Vec<_>, Vec<_>) = out_slices
        .iter_mut()
        .map(|out| out.device_ptr_mut(&backend.stream))
        .unzip();
    let mut d_out_ptrs = backend
        .stream
        .clone_htod(&out_ptrs)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (grad_clip out ptrs)"))?;

    launch_rows(
        &backend.stream,
        backend.kernels.function("grad_clip_scale_f32")?,
        total_chunks,
        BLOCK,
        0,
        |mut builder| {
            builder
                .arg(&mut d_out_ptrs)
                .arg(&d_grad_ptrs)
                .arg(&d_grad_sizes)
                .arg(&d_chunk_offsets)
                .arg(&scale)
                .arg(&num_grads_i32)
                .arg(&chunk_elems_i32);
            builder
        },
    )?;

    drop(out_guards);
    drop(input_guards);

    Ok(DeviceGradClipResult {
        pre_clip_norm,
        clipped_grads: Some(
            out_slices
                .into_iter()
                .map(|slice| DeviceHandle::Cuda(CudaStorage::new(slice)))
                .collect(),
        ),
    })
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_reduce_last_axis(
    backend: &CudaBackend,
    x: &[f32],
    shape: &[usize],
    kernel_name: &'static str,
) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected: usize = shape.iter().product();
    if x.len() != expected {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![x.len()],
        });
    }
    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda reduce cols exceeds i32"))?;
    let d_in = backend.upload_slice(x, shape)?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(rows)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
    launch_rows(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder.arg(&mut d_out).arg(&d_in).arg(&cols);
            builder
        },
    )?;
    cuda_download(backend, &d_out, rows)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_rope(
    backend: &CudaBackend,
    x: &[f32],
    x_shape: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> Result<Vec<f32>> {
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }
    let batch = x_shape[0];
    let heads = x_shape[1];
    let seq = x_shape[2];
    let head_dim = x_shape[3];
    if !head_dim.is_multiple_of(2) {
        return Err(AutogradError::InvalidRank {
            expected: "even head dim",
            got: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    let total = batch * heads * seq * head_dim;
    if x.len() != total {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![total],
            got: vec![x.len()],
        });
    }
    // Partial rotary (cos rows = rotary_dim/2 ≤ head_dim/2): rotate the leading
    // segment, pass the tail through — mirrors `cpu_rope_forward`.
    let rot_half = cos.len().checked_div(seq).unwrap_or(0);
    if cos.len() != sin.len() || rot_half == 0 || rot_half > half_dim || cos.len() != seq * rot_half
    {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![seq * half_dim],
            got: vec![cos.len().min(sin.len())],
        });
    }

    let d_x = backend.upload_slice(x, x_shape)?;
    let d_cos = backend
        .stream
        .clone_htod(cos)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let d_sin = backend
        .stream
        .clone_htod(sin)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope batch exceeds i32"))?;
    let heads_i = i32::try_from(heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope heads exceeds i32"))?;
    let seq_i = i32::try_from(seq)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope seq exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope head_dim exceeds i32"))?;
    let rot_half_i = i32::try_from(rot_half)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope rot_half exceeds i32"))?;

    let rows = batch * heads * seq;
    let block = std::cmp::min(half_dim, 256) as u32;
    let block = block.max(1);
    launch_rows(
        &backend.stream,
        backend.kernels.function("rope_f32")?,
        rows,
        block,
        0,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_x)
                .arg(&d_cos)
                .arg(&d_sin)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&rot_half_i);
            builder
        },
    )?;
    cuda_download(backend, &d_out, total)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_rope_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    x_shape: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> Result<DeviceHandle> {
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }
    let batch = x_shape[0];
    let heads = x_shape[1];
    let seq = x_shape[2];
    let head_dim = x_shape[3];
    if !head_dim.is_multiple_of(2) {
        return Err(AutogradError::InvalidRank {
            expected: "even head dim",
            got: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    let total = batch * heads * seq * head_dim;
    // Partial rotary: see `cuda_rope_forward`.
    let rot_half = cos.len().checked_div(seq).unwrap_or(0);
    if cos.len() != sin.len() || rot_half == 0 || rot_half > half_dim || cos.len() != seq * rot_half
    {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![seq * half_dim],
            got: vec![cos.len().min(sin.len())],
        });
    }

    let d_cos = backend
        .stream
        .clone_htod(cos)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (rope cos)"))?;
    let d_sin = backend
        .stream
        .clone_htod(sin)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (rope sin)"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope batch exceeds i32"))?;
    let heads_i = i32::try_from(heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope heads exceeds i32"))?;
    let seq_i = i32::try_from(seq)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope seq exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope head_dim exceeds i32"))?;
    let rot_half_i = i32::try_from(rot_half)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope rot_half exceeds i32"))?;

    let rows = batch * heads * seq;
    let block = std::cmp::min(half_dim, 256) as u32;
    let block = block.max(1);
    if let DeviceHandle::CudaBf16(storage) = x {
        let d_x = backend.cuda_bf16_storage_slice(storage)?;
        if d_x.len() != total {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![total],
                got: vec![d_x.len()],
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(total)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (rope)"))?;
        let func = backend.kernels.function_for("rope_f32", TapeDtype::Bf16)?;
        launch_rows(&backend.stream, &func, rows, block, 0, |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_x)
                .arg(&d_cos)
                .arg(&d_sin)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&rot_half_i);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }
    let d_x = backend.cuda_slice(x, "rope")?;
    if d_x.len() != total {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![total],
            got: vec![d_x.len()],
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (rope)"))?;
    launch_rows(
        &backend.stream,
        backend.kernels.function("rope_f32")?,
        rows,
        block,
        0,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_x)
                .arg(&d_cos)
                .arg(&d_sin)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&rot_half_i);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_gather_last_dim(
    backend: &CudaBackend,
    src: &[f32],
    src_shape: &[usize],
    ids: &[i32],
) -> Result<Vec<f32>> {
    if src_shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }
    let vocab = *src_shape.last().expect("non-empty shape above");
    let prefix: usize = src_shape[..src_shape.len() - 1]
        .iter()
        .product::<usize>()
        .max(1);
    let expected: usize = src_shape.iter().product();
    if src.len() != expected {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![src.len()],
        });
    }
    if ids.len() != prefix {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix,
            got: ids.len(),
        });
    }
    let d_src = backend.upload_slice(src, src_shape)?;
    let d_ids = backend
        .stream
        .clone_htod(ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(prefix)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    let n_i32 = i32::try_from(prefix)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather n exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather vocab exceeds i32"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("gather_last_dim_f32")?,
        prefix,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_src)
                .arg(&d_ids)
                .arg(&n_i32)
                .arg(&vocab_i32);
            builder
        },
    )?;
    cuda_download(backend, &d_out, prefix)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_scatter_add_rows(
    backend: &CudaBackend,
    upstream: &[f32],
    prefix_rows: usize,
    feature_dim: usize,
    indices: &[i32],
    vocab: usize,
) -> Result<Vec<f32>> {
    let expected_upstream = prefix_rows * feature_dim;
    if upstream.len() != expected_upstream {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected_upstream],
            got: vec![upstream.len()],
        });
    }
    if indices.len() != prefix_rows {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix_rows,
            got: indices.len(),
        });
    }
    let out_len = vocab * feature_dim;
    // Zero-initialize the accumulator on-device — the kernel only adds.
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(out_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
    if prefix_rows == 0 || feature_dim == 0 {
        return cuda_download(backend, &d_out, out_len);
    }
    let d_upstream = backend
        .stream
        .clone_htod(upstream)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let d_idx = backend
        .stream
        .clone_htod(indices)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;

    let prefix_i32 = i32::try_from(prefix_rows)
        .map_err(|_| AutogradError::TapeInvariant("cuda scatter_add prefix_rows exceeds i32"))?;
    let feature_i32 = i32::try_from(feature_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda scatter_add feature_dim exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda scatter_add vocab exceeds i32"))?;

    let block = std::cmp::min(feature_dim, 256) as u32;
    let block = block.max(1);
    launch_rows(
        &backend.stream,
        backend.kernels.function("scatter_add_rows_f32")?,
        prefix_rows,
        block,
        0,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_upstream)
                .arg(&d_idx)
                .arg(&prefix_i32)
                .arg(&feature_i32)
                .arg(&vocab_i32);
            builder
        },
    )?;
    cuda_download(backend, &d_out, out_len)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_add_broadcast(
    backend: &CudaBackend,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<Vec<f32>> {
    validate_broadcast(a_shape, b_shape)?;
    let total: usize = if a_shape.is_empty() {
        1
    } else {
        a_shape.iter().product()
    };
    let b_size: usize = if b_shape.is_empty() {
        1
    } else {
        b_shape.iter().product()
    };
    if a.len() != total {
        return Err(AutogradError::DataLengthMismatch {
            len: a.len(),
            shape: a_shape.to_vec(),
            size: total,
        });
    }
    if b.len() != b_size {
        return Err(AutogradError::DataLengthMismatch {
            len: b.len(),
            shape: b_shape.to_vec(),
            size: b_size,
        });
    }

    // Build right-aligned b-strides of length `out_rank`: 0 on broadcast axes
    // (axis missing from b_shape or b_shape dim == 1), contiguous otherwise.
    let out_rank = a_shape.len();
    let rank_offset = out_rank - b_shape.len();
    let mut b_strides = vec![0_i32; out_rank];
    let mut stride: i32 = 1;
    for i in (0..b_shape.len()).rev() {
        let dim = b_shape[i];
        if dim == 1 {
            b_strides[rank_offset + i] = 0;
        } else {
            b_strides[rank_offset + i] = stride;
        }
        // Advance stride regardless so the row-major layout over the b buffer
        // is consistent — broadcast axes still occupy 1 slot in b.
        stride = stride.saturating_mul(dim as i32);
    }

    let out_shape_i32: Vec<i32> = a_shape.iter().map(|&d| d as i32).collect();

    let d_a = backend.upload_slice(a, a_shape)?;
    let d_b = backend
        .stream
        .clone_htod(b)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let d_out_shape = backend
        .stream
        .clone_htod(&out_shape_i32)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let d_b_strides = backend
        .stream
        .clone_htod(&b_strides)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    let out_rank_i32 = i32::try_from(out_rank)
        .map_err(|_| AutogradError::TapeInvariant("cuda add_broadcast rank exceeds i32"))?;
    let total_i32 = i32::try_from(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda add_broadcast total exceeds i32"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("add_broadcast_f32")?,
        total,
        |mut builder| {
            builder
                .arg(&d_a)
                .arg(&d_b)
                .arg(&mut d_out)
                .arg(&d_out_shape)
                .arg(&d_b_strides)
                .arg(&out_rank_i32)
                .arg(&total_i32);
            builder
        },
    )?;
    cuda_download(backend, &d_out, total)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_add_broadcast_device(
    backend: &CudaBackend,
    a: &DeviceHandle,
    a_shape: &[usize],
    b: &DeviceHandle,
    b_shape: &[usize],
) -> Result<DeviceHandle> {
    validate_broadcast(a_shape, b_shape)?;
    let total = shape_size(a_shape);
    let b_size = shape_size(b_shape);

    let out_rank = a_shape.len();
    let rank_offset = out_rank - b_shape.len();
    let mut b_strides = vec![0_i32; out_rank];
    let mut stride: i32 = 1;
    for i in (0..b_shape.len()).rev() {
        let dim = b_shape[i];
        if dim == 1 {
            b_strides[rank_offset + i] = 0;
        } else {
            b_strides[rank_offset + i] = stride;
        }
        stride = stride.saturating_mul(dim as i32);
    }

    let out_shape_i32: Vec<i32> = a_shape.iter().map(|&d| d as i32).collect();
    let d_out_shape = backend
        .stream
        .clone_htod(&out_shape_i32)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let d_b_strides = backend
        .stream
        .clone_htod(&b_strides)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;

    let out_rank_i32 = i32::try_from(out_rank)
        .map_err(|_| AutogradError::TapeInvariant("cuda add_broadcast rank exceeds i32"))?;
    let total_i32 = i32::try_from(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda add_broadcast total exceeds i32"))?;

    if backend.tape_bf16() {
        let d_a_op = backend.bf16_operand(a, "add_broadcast")?;
        let d_b_op = backend.bf16_operand(b, "add_broadcast")?;
        let d_a = d_a_op.get();
        let d_b = d_b_op.get();
        if d_a.len() != total || d_b.len() != b_size {
            return Err(AutogradError::DataLengthMismatch {
                len: d_a.len().min(d_b.len()),
                shape: a_shape.to_vec(),
                size: total,
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(total)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
        let func = backend
            .kernels
            .function_for("add_broadcast_f32", TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, total, |mut builder| {
            builder
                .arg(d_a)
                .arg(d_b)
                .arg(&mut d_out)
                .arg(&d_out_shape)
                .arg(&d_b_strides)
                .arg(&out_rank_i32)
                .arg(&total_i32);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }

    let d_a = backend.cuda_slice(a, "add_broadcast")?;
    let d_b = backend.cuda_slice(b, "add_broadcast")?;
    if d_a.len() != total {
        return Err(AutogradError::DataLengthMismatch {
            len: d_a.len(),
            shape: a_shape.to_vec(),
            size: total,
        });
    }
    if d_b.len() != b_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_b.len(),
            shape: b_shape.to_vec(),
            size: b_size,
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("add_broadcast_f32")?,
        total,
        |mut builder| {
            builder
                .arg(d_a)
                .arg(d_b)
                .arg(&mut d_out)
                .arg(&d_out_shape)
                .arg(&d_b_strides)
                .arg(&out_rank_i32)
                .arg(&total_i32);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_broadcast_expand_device(
    backend: &CudaBackend,
    src: &DeviceHandle,
    src_shape: &[usize],
    target_shape: &[usize],
) -> Result<DeviceHandle> {
    // Reuse `add_broadcast_f32` with a zeroed `a`: out = 0 + src_broadcast. The
    // zero carrier is a scratch buffer freed on return — never a tape tensor.
    validate_broadcast(target_shape, src_shape)?;
    let total = shape_size(target_shape);
    let src_size = shape_size(src_shape);

    let out_rank = target_shape.len();
    let rank_offset = out_rank - src_shape.len();
    let mut src_strides = vec![0_i32; out_rank];
    let mut stride: i32 = 1;
    for i in (0..src_shape.len()).rev() {
        let dim = src_shape[i];
        if dim == 1 {
            src_strides[rank_offset + i] = 0;
        } else {
            src_strides[rank_offset + i] = stride;
        }
        stride = stride.saturating_mul(dim as i32);
    }

    let out_shape_i32: Vec<i32> = target_shape.iter().map(|&d| d as i32).collect();
    let d_out_shape = backend
        .stream
        .clone_htod(&out_shape_i32)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let d_src_strides = backend
        .stream
        .clone_htod(&src_strides)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;

    let out_rank_i32 = i32::try_from(out_rank)
        .map_err(|_| AutogradError::TapeInvariant("cuda broadcast_expand rank exceeds i32"))?;
    let total_i32 = i32::try_from(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda broadcast_expand total exceeds i32"))?;

    if let DeviceHandle::CudaBf16(storage) = src {
        let d_src = backend.cuda_bf16_storage_slice(storage)?;
        if d_src.len() != src_size {
            return Err(AutogradError::DataLengthMismatch {
                len: d_src.len(),
                shape: src_shape.to_vec(),
                size: src_size,
            });
        }
        // SAFETY: the kernel writes every element.
        let mut d_out = unsafe { backend.stream.alloc::<u16>(total) }
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc failed"))?;
        let func = backend
            .kernels
            .function_for("broadcast_copy_f32", TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, total, |mut builder| {
            builder
                .arg(d_src)
                .arg(&mut d_out)
                .arg(&d_out_shape)
                .arg(&d_src_strides)
                .arg(&out_rank_i32)
                .arg(&total_i32);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }

    let d_src = backend.cuda_slice(src, "broadcast_expand")?;
    if d_src.len() != src_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_src.len(),
            shape: src_shape.to_vec(),
            size: src_size,
        });
    }
    // SAFETY: the kernel writes every element.
    let mut d_out = unsafe { backend.stream.alloc::<f32>(total) }
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc failed"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("broadcast_copy_f32")?,
        total,
        |mut builder| {
            builder
                .arg(d_src)
                .arg(&mut d_out)
                .arg(&d_out_shape)
                .arg(&d_src_strides)
                .arg(&out_rank_i32)
                .arg(&total_i32);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_transpose_axes_swap_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    old_shape: &[usize],
    axis1: usize,
    axis2: usize,
) -> Result<(DeviceHandle, Vec<usize>)> {
    let rank = old_shape.len();
    if axis1 >= rank {
        return Err(AutogradError::AxisOutOfBounds { axis: axis1, rank });
    }
    if axis2 >= rank {
        return Err(AutogradError::AxisOutOfBounds { axis: axis2, rank });
    }
    let total = shape_size(old_shape);
    // Pure movement: the lane follows the input handle's dtype.
    let x_bf16 = matches!(x, DeviceHandle::CudaBf16(_));
    let x_len = match x {
        DeviceHandle::CudaBf16(storage) => backend.cuda_bf16_storage_slice(storage)?.len(),
        _ => backend.cuda_slice(x, "transpose_axes_swap")?.len(),
    };
    if x_len != total {
        return Err(AutogradError::DataLengthMismatch {
            len: x_len,
            shape: old_shape.to_vec(),
            size: total,
        });
    }
    if axis1 == axis2 {
        return Ok((x.clone(), old_shape.to_vec()));
    }

    let mut new_shape = old_shape.to_vec();
    new_shape.swap(axis1, axis2);
    let old_shape_i32: Vec<i32> = old_shape.iter().map(|&d| d as i32).collect();
    let new_shape_i32: Vec<i32> = new_shape.iter().map(|&d| d as i32).collect();
    let d_old_shape = backend
        .stream
        .clone_htod(&old_shape_i32)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (transpose shape)"))?;
    let d_new_shape = backend
        .stream
        .clone_htod(&new_shape_i32)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (transpose shape)"))?;
    let rank_i = i32::try_from(rank)
        .map_err(|_| AutogradError::TapeInvariant("cuda transpose rank exceeds i32"))?;
    let axis1_i = i32::try_from(axis1)
        .map_err(|_| AutogradError::TapeInvariant("cuda transpose axis exceeds i32"))?;
    let axis2_i = i32::try_from(axis2)
        .map_err(|_| AutogradError::TapeInvariant("cuda transpose axis exceeds i32"))?;
    let total_i = i32::try_from(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda transpose total exceeds i32"))?;

    if x_bf16 {
        let d_x = backend.cuda_bf16_slice(x, "transpose_axes_swap")?;
        let bytes = total.saturating_mul(std::mem::size_of::<u16>());
        let mut d_out = backend.stream.alloc_zeros::<u16>(total).map_err(|_| {
            AutogradError::CudaAllocFailed {
                op: "transpose",
                shape: new_shape.clone(),
                bytes,
            }
        })?;
        let func = backend
            .kernels
            .function_for("transpose_axes_swap_f32", TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, total, |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_x)
                .arg(&d_old_shape)
                .arg(&d_new_shape)
                .arg(&rank_i)
                .arg(&axis1_i)
                .arg(&axis2_i)
                .arg(&total_i);
            builder
        })?;
        return Ok((
            DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)),
            new_shape,
        ));
    }

    let d_x = backend.cuda_slice(x, "transpose_axes_swap")?;
    let bytes = total.saturating_mul(std::mem::size_of::<f32>());
    let mut d_out =
        backend
            .stream
            .alloc_zeros::<f32>(total)
            .map_err(|_| AutogradError::CudaAllocFailed {
                op: "transpose",
                shape: new_shape.clone(),
                bytes,
            })?;
    launch_1d(
        &backend.stream,
        backend.kernels.function("transpose_axes_swap_f32")?,
        total,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_x)
                .arg(&d_old_shape)
                .arg(&d_new_shape)
                .arg(&rank_i)
                .arg(&axis1_i)
                .arg(&axis2_i)
                .arg(&total_i);
            builder
        },
    )?;
    Ok((DeviceHandle::Cuda(CudaStorage::new(d_out)), new_shape))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_slice_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    old_shape: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> Result<DeviceHandle> {
    let rank = old_shape.len();
    if starts.len() != rank {
        return Err(AutogradError::InvalidIndicesLen {
            expected: rank,
            got: starts.len(),
        });
    }
    if ends.len() != rank {
        return Err(AutogradError::InvalidIndicesLen {
            expected: rank,
            got: ends.len(),
        });
    }
    for ((&start, &end), &dim) in starts.iter().zip(ends.iter()).zip(old_shape.iter()) {
        if start > end {
            return Err(AutogradError::TapeInvariant(
                "slice start must be <= end for every axis",
            ));
        }
        if end > dim {
            return Err(AutogradError::IndexOutOfBounds {
                index: end,
                upper: dim,
            });
        }
        if start > dim {
            return Err(AutogradError::IndexOutOfBounds {
                index: start,
                upper: dim,
            });
        }
    }

    let old_total = shape_size(old_shape);
    let x_bf16 = matches!(x, DeviceHandle::CudaBf16(_));
    let x_len = match x {
        DeviceHandle::CudaBf16(storage) => backend.cuda_bf16_storage_slice(storage)?.len(),
        _ => backend.cuda_slice(x, "slice")?.len(),
    };
    if x_len != old_total {
        return Err(AutogradError::DataLengthMismatch {
            len: x_len,
            shape: old_shape.to_vec(),
            size: old_total,
        });
    }
    let new_shape: Vec<usize> = starts
        .iter()
        .zip(ends.iter())
        .map(|(&start, &end)| end - start)
        .collect();
    let total = shape_size(&new_shape);

    let old_shape_i32: Vec<i32> = old_shape.iter().map(|&d| d as i32).collect();
    let starts_i32: Vec<i32> = starts.iter().map(|&d| d as i32).collect();
    let new_shape_i32: Vec<i32> = new_shape.iter().map(|&d| d as i32).collect();
    let d_old_shape = backend
        .stream
        .clone_htod(&old_shape_i32)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (slice shape)"))?;
    let d_starts = backend
        .stream
        .clone_htod(&starts_i32)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (slice starts)"))?;
    let d_new_shape = backend
        .stream
        .clone_htod(&new_shape_i32)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (slice shape)"))?;
    let rank_i = i32::try_from(rank)
        .map_err(|_| AutogradError::TapeInvariant("cuda slice rank exceeds i32"))?;
    let total_u64 = total as u64;

    if x_bf16 {
        let d_x = backend.cuda_bf16_slice(x, "slice")?;
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(total)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (slice)"))?;
        let func = backend.kernels.function_for("slice_f32", TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, total, |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_x)
                .arg(&d_old_shape)
                .arg(&d_starts)
                .arg(&d_new_shape)
                .arg(&rank_i)
                .arg(&total_u64);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }

    let d_x = backend.cuda_slice(x, "slice")?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (slice)"))?;
    launch_1d(
        &backend.stream,
        backend.kernels.function("slice_f32")?,
        total,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_x)
                .arg(&d_old_shape)
                .arg(&d_starts)
                .arg(&d_new_shape)
                .arg(&rank_i)
                .arg(&total_u64);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_concat_axis2_device(
    backend: &CudaBackend,
    a: &DeviceHandle,
    a_shape: &[usize],
    b: &DeviceHandle,
    b_shape: &[usize],
) -> Result<(DeviceHandle, Vec<usize>)> {
    if a_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: a_shape.len(),
        });
    }
    if b_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: b_shape.len(),
        });
    }
    if a_shape[0] != b_shape[0] || a_shape[1] != b_shape[1] || a_shape[3] != b_shape[3] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![a_shape[0], a_shape[1], a_shape[3]],
            got: vec![b_shape[0], b_shape[1], b_shape[3]],
        });
    }
    let a_total = shape_size(a_shape);
    let b_total = shape_size(b_shape);
    let out_shape = vec![a_shape[0], a_shape[1], a_shape[2] + b_shape[2], a_shape[3]];
    let total = shape_size(&out_shape);
    let batch_i = i32::try_from(a_shape[0])
        .map_err(|_| AutogradError::TapeInvariant("cuda concat_axis2 batch exceeds i32"))?;
    let heads_i = i32::try_from(a_shape[1])
        .map_err(|_| AutogradError::TapeInvariant("cuda concat_axis2 heads exceeds i32"))?;
    let a_seq_i = i32::try_from(a_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda concat_axis2 a_seq exceeds i32"))?;
    let b_seq_i = i32::try_from(b_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda concat_axis2 b_seq exceeds i32"))?;
    let dim_i = i32::try_from(a_shape[3])
        .map_err(|_| AutogradError::TapeInvariant("cuda concat_axis2 dim exceeds i32"))?;
    let total_i = i32::try_from(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda concat_axis2 total exceeds i32"))?;

    // bf16 lane only when both inputs already are bf16; a mixed pair widens
    // the bf16 side exactly instead of quantizing inside a movement op.
    if matches!(a, DeviceHandle::CudaBf16(_)) && matches!(b, DeviceHandle::CudaBf16(_)) {
        let d_a = backend.cuda_bf16_slice(a, "concat_axis2")?;
        let d_b = backend.cuda_bf16_slice(b, "concat_axis2")?;
        if d_a.len() != a_total || d_b.len() != b_total {
            return Err(AutogradError::DataLengthMismatch {
                len: d_a.len().min(d_b.len()),
                shape: a_shape.to_vec(),
                size: a_total,
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(total)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (concat_axis2)"))?;
        let func = backend
            .kernels
            .function_for("concat_axis2_f32", TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, total, |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_a)
                .arg(d_b)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&a_seq_i)
                .arg(&b_seq_i)
                .arg(&dim_i)
                .arg(&total_i);
            builder
        })?;
        return Ok((
            DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)),
            out_shape,
        ));
    }

    let d_a_op = backend.f32_operand(a, "concat_axis2")?;
    let d_b_op = backend.f32_operand(b, "concat_axis2")?;
    let d_a = d_a_op.get();
    let d_b = d_b_op.get();
    if d_a.len() != a_total {
        return Err(AutogradError::DataLengthMismatch {
            len: d_a.len(),
            shape: a_shape.to_vec(),
            size: a_total,
        });
    }
    if d_b.len() != b_total {
        return Err(AutogradError::DataLengthMismatch {
            len: d_b.len(),
            shape: b_shape.to_vec(),
            size: b_total,
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (concat_axis2)"))?;
    launch_1d(
        &backend.stream,
        backend.kernels.function("concat_axis2_f32")?,
        total,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_a)
                .arg(d_b)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&a_seq_i)
                .arg(&b_seq_i)
                .arg(&dim_i)
                .arg(&total_i);
            builder
        },
    )?;
    Ok((DeviceHandle::Cuda(CudaStorage::new(d_out)), out_shape))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_kv_cache_write_axis2(
    backend: &CudaBackend,
    dst: &DeviceHandle,
    dst_shape: &[usize],
    src: &DeviceHandle,
    src_shape: &[usize],
    seq_offset: usize,
) -> Result<DeviceHandle> {
    if dst_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: dst_shape.len(),
        });
    }
    if src_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: src_shape.len(),
        });
    }
    if dst_shape[0] != src_shape[0] || dst_shape[1] != src_shape[1] || dst_shape[3] != src_shape[3]
    {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![dst_shape[0], dst_shape[1], dst_shape[3]],
            got: vec![src_shape[0], src_shape[1], src_shape[3]],
        });
    }
    if seq_offset + src_shape[2] > dst_shape[2] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![dst_shape[2]],
            got: vec![seq_offset + src_shape[2]],
        });
    }
    let dst_total = shape_size(dst_shape);
    let src_total = shape_size(src_shape);
    // Cache lane stays f32: the sdpa consumers are untemplated, so a bf16
    // src is widened exactly at this boundary.
    let d_dst = backend.cuda_slice(dst, "kv_cache_write_axis2")?;
    let d_src_op = backend.f32_operand(src, "kv_cache_write_axis2")?;
    let d_src = d_src_op.get();
    if d_dst.len() != dst_total {
        return Err(AutogradError::DataLengthMismatch {
            len: d_dst.len(),
            shape: dst_shape.to_vec(),
            size: dst_total,
        });
    }
    if d_src.len() != src_total {
        return Err(AutogradError::DataLengthMismatch {
            len: d_src.len(),
            shape: src_shape.to_vec(),
            size: src_total,
        });
    }

    let batch_i = i32::try_from(dst_shape[0])
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write batch exceeds i32"))?;
    let heads_i = i32::try_from(dst_shape[1])
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write heads exceeds i32"))?;
    let max_seq_i = i32::try_from(dst_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write max_seq exceeds i32"))?;
    let src_seq_i = i32::try_from(src_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write src_seq exceeds i32"))?;
    let dim_i = i32::try_from(dst_shape[3])
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write dim exceeds i32"))?;
    let offset_i = i32::try_from(seq_offset)
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write offset exceeds i32"))?;
    let total_i = i32::try_from(src_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write total exceeds i32"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("kv_cache_write_axis2_f32")?,
        src_total,
        |mut builder| {
            builder
                .arg(d_dst)
                .arg(d_src)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&max_seq_i)
                .arg(&src_seq_i)
                .arg(&dim_i)
                .arg(&offset_i)
                .arg(&total_i);
            builder
        },
    )?;
    Ok(dst.clone())
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_causal_sdpa_decode_gqa(
    backend: &CudaBackend,
    q: &DeviceHandle,
    q_shape: &[usize],
    k: &DeviceHandle,
    k_shape: &[usize],
    v: &DeviceHandle,
    v_shape: &[usize],
    q_start: usize,
) -> Result<(DeviceHandle, Vec<usize>)> {
    validate_decode_gqa_shapes(q_shape, k_shape, v_shape, q_start)?;
    if k_shape[2] > 32 {
        return Err(AutogradError::TapeInvariant(
            "cuda causal_sdpa_decode_gqa supports kv_len <= 32",
        ));
    }

    let d_q = backend.cuda_slice(q, "causal_sdpa_decode_gqa")?;
    let d_k = backend.cuda_slice(k, "causal_sdpa_decode_gqa")?;
    let d_v = backend.cuda_slice(v, "causal_sdpa_decode_gqa")?;
    let q_size = shape_size(q_shape);
    let k_size = shape_size(k_shape);
    let v_size = shape_size(v_shape);
    if d_q.len() != q_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_q.len(),
            shape: q_shape.to_vec(),
            size: q_size,
        });
    }
    if d_k.len() != k_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_k.len(),
            shape: k_shape.to_vec(),
            size: k_size,
        });
    }
    if d_v.len() != v_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_v.len(),
            shape: v_shape.to_vec(),
            size: v_size,
        });
    }

    let out_shape = vec![q_shape[0], q_shape[1], 1, q_shape[3]];
    let out_total = shape_size(&out_shape);
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(out_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (decode sdpa)"))?;

    let batch_i = i32::try_from(q_shape[0])
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa batch exceeds i32"))?;
    let query_heads_i = i32::try_from(q_shape[1])
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa heads exceeds i32"))?;
    let kv_heads_i = i32::try_from(k_shape[1])
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa kv_heads exceeds i32"))?;
    let kv_len_i = i32::try_from(k_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa kv_len exceeds i32"))?;
    let head_dim_i = i32::try_from(q_shape[3])
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa head_dim exceeds i32"))?;
    let q_start_i = i32::try_from(q_start)
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa q_start exceeds i32"))?;
    let scale = 1.0_f32 / (q_shape[3] as f32).sqrt();
    let rows = q_shape[0] * q_shape[1];
    const BLOCK: u32 = 256;
    let shared = BLOCK * std::mem::size_of::<f32>() as u32 + 32 * std::mem::size_of::<f32>() as u32;
    launch_rows(
        &backend.stream,
        backend.kernels.function("causal_sdpa_decode_gqa_f32")?,
        rows,
        BLOCK,
        shared,
        |mut builder| {
            builder
                .arg(d_q)
                .arg(d_k)
                .arg(d_v)
                .arg(&mut d_out)
                .arg(&batch_i)
                .arg(&query_heads_i)
                .arg(&kv_heads_i)
                .arg(&kv_len_i)
                .arg(&head_dim_i)
                .arg(&q_start_i)
                .arg(&scale);
            builder
        },
    )?;
    Ok((DeviceHandle::Cuda(CudaStorage::new(d_out)), out_shape))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_causal_sdpa_decode_gqa_cache(
    backend: &CudaBackend,
    q: &DeviceHandle,
    q_shape: &[usize],
    k: &DeviceHandle,
    k_shape: &[usize],
    v: &DeviceHandle,
    v_shape: &[usize],
    kv_len: usize,
    q_start: usize,
) -> Result<(DeviceHandle, Vec<usize>)> {
    validate_decode_gqa_cache_shapes(q_shape, k_shape, v_shape, kv_len, q_start)?;

    let d_q = backend.cuda_slice(q, "causal_sdpa_decode_gqa_cache")?;
    let d_k = backend.cuda_slice(k, "causal_sdpa_decode_gqa_cache")?;
    let d_v = backend.cuda_slice(v, "causal_sdpa_decode_gqa_cache")?;
    let q_size = shape_size(q_shape);
    let k_size = shape_size(k_shape);
    let v_size = shape_size(v_shape);
    if d_q.len() != q_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_q.len(),
            shape: q_shape.to_vec(),
            size: q_size,
        });
    }
    if d_k.len() != k_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_k.len(),
            shape: k_shape.to_vec(),
            size: k_size,
        });
    }
    if d_v.len() != v_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_v.len(),
            shape: v_shape.to_vec(),
            size: v_size,
        });
    }

    let out_shape = vec![q_shape[0], q_shape[1], 1, q_shape[3]];
    let out_total = shape_size(&out_shape);
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(out_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (cache decode sdpa)"))?;

    let batch_i = i32::try_from(q_shape[0])
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode batch exceeds i32"))?;
    let query_heads_i = i32::try_from(q_shape[1])
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode heads exceeds i32"))?;
    let kv_heads_i = i32::try_from(k_shape[1])
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode kv_heads exceeds i32"))?;
    let max_seq_i = i32::try_from(k_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode max_seq exceeds i32"))?;
    let kv_len_i = i32::try_from(kv_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode kv_len exceeds i32"))?;
    let head_dim_i = i32::try_from(q_shape[3])
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode head_dim exceeds i32"))?;
    let q_start_i = i32::try_from(q_start)
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode q_start exceeds i32"))?;
    let scale = 1.0_f32 / (q_shape[3] as f32).sqrt();
    let rows = q_shape[0] * q_shape[1];
    let head_dim = q_shape[3];

    // Online-softmax fast path for Qwen3.5-style head_dim=256 (default knob:
    // --autograd-decode-attn-legacy forces the original two-pass kernel).
    // The online kernel uses HEAD_DIM threads per block (vs 256 in the legacy),
    // one-pass running-max softmax (vs two-pass with shared-mem scores buffer),
    // and warp-level reductions throughout.
    let use_online = head_dim == 256 && !force_legacy_decode_attn();
    if use_online {
        const BLOCK_ONLINE: u32 = 256; // = HEAD_DIM
        let n_warps = BLOCK_ONLINE / 32;
        let shared_online = n_warps * std::mem::size_of::<f32>() as u32;
        launch_rows(
            &backend.stream,
            backend
                .kernels
                .function("causal_sdpa_decode_gqa_cache_online_f32_hd256")?,
            rows,
            BLOCK_ONLINE,
            shared_online,
            |mut builder| {
                builder
                    .arg(d_q)
                    .arg(d_k)
                    .arg(d_v)
                    .arg(&mut d_out)
                    .arg(&batch_i)
                    .arg(&query_heads_i)
                    .arg(&kv_heads_i)
                    .arg(&max_seq_i)
                    .arg(&kv_len_i)
                    .arg(&head_dim_i)
                    .arg(&q_start_i)
                    .arg(&scale);
                builder
            },
        )?;
        return Ok((DeviceHandle::Cuda(CudaStorage::new(d_out)), out_shape));
    }

    // Legacy two-pass kernel — kept for head_dim != 256 and as the
    // legacy escape hatch.
    const BLOCK: u32 = 256;
    let visible = q_start.saturating_add(1).min(kv_len);
    let shared = BLOCK * std::mem::size_of::<f32>() as u32
        + u32::try_from(visible)
            .map_err(|_| AutogradError::TapeInvariant("cuda cache decode visible exceeds u32"))?
            * std::mem::size_of::<f32>() as u32;
    launch_rows(
        &backend.stream,
        backend
            .kernels
            .function("causal_sdpa_decode_gqa_cache_f32")?,
        rows,
        BLOCK,
        shared,
        |mut builder| {
            builder
                .arg(d_q)
                .arg(d_k)
                .arg(d_v)
                .arg(&mut d_out)
                .arg(&batch_i)
                .arg(&query_heads_i)
                .arg(&kv_heads_i)
                .arg(&max_seq_i)
                .arg(&kv_len_i)
                .arg(&head_dim_i)
                .arg(&q_start_i)
                .arg(&scale);
            builder
        },
    )?;
    Ok((DeviceHandle::Cuda(CudaStorage::new(d_out)), out_shape))
}

#[cfg(not(feature = "no-cuda"))]
fn force_legacy_decode_attn() -> bool {
    crate::runtime_flags::decode_attn_legacy()
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_qwen_decode_prepare_q(
    backend: &CudaBackend,
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
    validate_qwen_decode_prepare_q_shapes(
        q_full_shape,
        q_norm_weight_shape,
        cos_shape,
        sin_shape,
        query_heads,
        head_dim,
        gated,
    )?;
    let q_full_size = shape_size(q_full_shape);
    let d_q_full = backend.cuda_slice(q_full, "qwen_decode_prepare_q")?;
    let d_weight = backend.cuda_slice(q_norm_weight, "qwen_decode_prepare_q")?;
    let d_cos = backend.cuda_slice(cos, "qwen_decode_prepare_q")?;
    let d_sin = backend.cuda_slice(sin, "qwen_decode_prepare_q")?;
    if d_q_full.len() != q_full_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_q_full.len(),
            shape: q_full_shape.to_vec(),
            size: q_full_size,
        });
    }
    if d_weight.len() != head_dim {
        return Err(AutogradError::DataLengthMismatch {
            len: d_weight.len(),
            shape: q_norm_weight_shape.to_vec(),
            size: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    if d_cos.len() != half_dim || d_sin.len() != half_dim {
        return Err(AutogradError::DataLengthMismatch {
            len: d_cos.len().max(d_sin.len()),
            shape: vec![1, half_dim],
            size: half_dim,
        });
    }

    let batch = q_full_shape[0];
    let out_shape = vec![batch, query_heads, 1, head_dim];
    let out_total = shape_size(&out_shape);
    let mut d_q = backend
        .stream
        .alloc_zeros::<f32>(out_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (qwen prep q)"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep q batch exceeds i32"))?;
    let query_heads_i = i32::try_from(query_heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep q heads exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep q head_dim exceeds i32"))?;
    let q_full_stride_i = i32::try_from(q_full_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep q stride exceeds i32"))?;
    let rows = batch * query_heads;
    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;

    let gate_handle = if gated {
        let mut d_gate = backend.stream.alloc_zeros::<f32>(out_total).map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (qwen prep gate)")
        })?;
        launch_rows(
            &backend.stream,
            backend
                .kernels
                .function("qwen_decode_prepare_q_gated_f32")?,
            rows,
            BLOCK,
            SHARED,
            |mut builder| {
                builder
                    .arg(&mut d_q)
                    .arg(&mut d_gate)
                    .arg(d_q_full)
                    .arg(d_weight)
                    .arg(d_cos)
                    .arg(d_sin)
                    .arg(&batch_i)
                    .arg(&query_heads_i)
                    .arg(&head_dim_i)
                    .arg(&q_full_stride_i)
                    .arg(&eps);
                builder
            },
        )?;
        Some(DeviceHandle::Cuda(CudaStorage::new(d_gate)))
    } else {
        launch_rows(
            &backend.stream,
            backend.kernels.function("qwen_decode_prepare_q_f32")?,
            rows,
            BLOCK,
            SHARED,
            |mut builder| {
                builder
                    .arg(&mut d_q)
                    .arg(d_q_full)
                    .arg(d_weight)
                    .arg(d_cos)
                    .arg(d_sin)
                    .arg(&batch_i)
                    .arg(&query_heads_i)
                    .arg(&head_dim_i)
                    .arg(&q_full_stride_i)
                    .arg(&eps);
                builder
            },
        )?;
        None
    };

    Ok((
        DeviceHandle::Cuda(CudaStorage::new(d_q)),
        gate_handle,
        out_shape,
    ))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_qwen_decode_prepare_kv(
    backend: &CudaBackend,
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
    validate_qwen_decode_prepare_kv_shapes(
        k_full_shape,
        v_full_shape,
        k_norm_weight_shape,
        cos_shape,
        sin_shape,
        kv_heads,
        head_dim,
    )?;
    let k_full_size = shape_size(k_full_shape);
    let v_full_size = shape_size(v_full_shape);
    let d_k_full = backend.cuda_slice(k_full, "qwen_decode_prepare_kv")?;
    let d_v_full = backend.cuda_slice(v_full, "qwen_decode_prepare_kv")?;
    let d_weight = backend.cuda_slice(k_norm_weight, "qwen_decode_prepare_kv")?;
    let d_cos = backend.cuda_slice(cos, "qwen_decode_prepare_kv")?;
    let d_sin = backend.cuda_slice(sin, "qwen_decode_prepare_kv")?;
    if d_k_full.len() != k_full_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_k_full.len(),
            shape: k_full_shape.to_vec(),
            size: k_full_size,
        });
    }
    if d_v_full.len() != v_full_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_v_full.len(),
            shape: v_full_shape.to_vec(),
            size: v_full_size,
        });
    }
    if d_weight.len() != head_dim {
        return Err(AutogradError::DataLengthMismatch {
            len: d_weight.len(),
            shape: k_norm_weight_shape.to_vec(),
            size: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    if d_cos.len() != half_dim || d_sin.len() != half_dim {
        return Err(AutogradError::DataLengthMismatch {
            len: d_cos.len().max(d_sin.len()),
            shape: vec![1, half_dim],
            size: half_dim,
        });
    }

    let batch = k_full_shape[0];
    let out_shape = vec![batch, kv_heads, 1, head_dim];
    let out_total = shape_size(&out_shape);
    let mut d_k = backend
        .stream
        .alloc_zeros::<f32>(out_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (qwen prep k)"))?;
    let mut d_v = backend
        .stream
        .alloc_zeros::<f32>(out_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (qwen prep v)"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep kv batch exceeds i32"))?;
    let kv_heads_i = i32::try_from(kv_heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep kv heads exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep kv head_dim exceeds i32"))?;
    let kv_full_stride_i = i32::try_from(k_full_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep kv stride exceeds i32"))?;
    let rows = batch * kv_heads;
    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;

    launch_rows(
        &backend.stream,
        backend.kernels.function("qwen_decode_prepare_kv_f32")?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder
                .arg(&mut d_k)
                .arg(&mut d_v)
                .arg(d_k_full)
                .arg(d_v_full)
                .arg(d_weight)
                .arg(d_cos)
                .arg(d_sin)
                .arg(&batch_i)
                .arg(&kv_heads_i)
                .arg(&head_dim_i)
                .arg(&kv_full_stride_i)
                .arg(&eps);
            builder
        },
    )?;

    Ok((
        DeviceHandle::Cuda(CudaStorage::new(d_k)),
        DeviceHandle::Cuda(CudaStorage::new(d_v)),
        out_shape,
    ))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_slice_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    input_shape: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> Result<DeviceHandle> {
    validate_slice_shape(input_shape, starts, ends)?;
    let dest = if matches!(upstream, DeviceHandle::CudaBf16(_)) {
        let d_grad = backend
            .stream
            .alloc_zeros::<u16>(shape_size(input_shape))
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (slice_bwd)"))?;
        DeviceHandle::CudaBf16(CudaBf16Storage::new(d_grad))
    } else {
        let d_grad = backend
            .stream
            .alloc_zeros::<f32>(shape_size(input_shape))
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (slice_bwd)"))?;
        DeviceHandle::Cuda(CudaStorage::new(d_grad))
    };
    cuda_write_slice_device(backend, &dest, upstream, input_shape, starts, ends)
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_write_slice_device(
    backend: &CudaBackend,
    dest: &DeviceHandle,
    upstream: &DeviceHandle,
    input_shape: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> Result<DeviceHandle> {
    let upstream_shape = validate_slice_shape(input_shape, starts, ends)?;
    let upstream_size = shape_size(&upstream_shape);
    let input_size = shape_size(input_shape);
    let rank = input_shape.len();
    let input_shape_i32: Vec<i32> = input_shape.iter().map(|&d| d as i32).collect();
    let starts_i32: Vec<i32> = starts.iter().map(|&d| d as i32).collect();
    let upstream_shape_i32: Vec<i32> = upstream_shape.iter().map(|&d| d as i32).collect();
    let d_input_shape = backend
        .stream
        .clone_htod(&input_shape_i32)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (slice_bwd shape)"))?;
    let d_starts = backend
        .stream
        .clone_htod(&starts_i32)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (slice_bwd starts)"))?;
    let d_upstream_shape = backend
        .stream
        .clone_htod(&upstream_shape_i32)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (slice_bwd shape)"))?;
    let rank_i = i32::try_from(rank)
        .map_err(|_| AutogradError::TapeInvariant("cuda slice_bwd rank exceeds i32"))?;
    let upstream_size_u64 = upstream_size as u64;

    // Lane follows the destination's dtype; the upstream is harmonized to it.
    if let DeviceHandle::CudaBf16(storage) = dest {
        let d_dest = backend.cuda_bf16_storage_slice(storage)?;
        let d_up_op = backend.bf16_operand(upstream, "slice_backward_device")?;
        let d_up = d_up_op.get();
        if d_dest.len() != input_size || d_up.len() != upstream_size {
            return Err(AutogradError::DataLengthMismatch {
                len: d_dest.len().min(d_up.len()),
                shape: input_shape.to_vec(),
                size: input_size,
            });
        }
        let (dest_ptr, _dest_guard) = d_dest.device_ptr(&backend.stream);
        let func = backend
            .kernels
            .function_for("slice_backward_f32", TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, upstream_size, |mut builder| {
            builder
                .arg(&dest_ptr)
                .arg(d_up)
                .arg(&d_input_shape)
                .arg(&d_starts)
                .arg(&d_upstream_shape)
                .arg(&rank_i)
                .arg(&upstream_size_u64);
            builder
        })?;
        return Ok(dest.clone());
    }

    let d_dest = backend.cuda_slice(dest, "write_slice_device")?;
    let d_up_op = backend.f32_operand(upstream, "slice_backward_device")?;
    let d_up = d_up_op.get();
    if d_dest.len() != input_size || d_up.len() != upstream_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_dest.len().min(d_up.len()),
            shape: input_shape.to_vec(),
            size: input_size,
        });
    }
    let (dest_ptr, _dest_guard) = d_dest.device_ptr(&backend.stream);
    launch_1d(
        &backend.stream,
        backend.kernels.function("slice_backward_f32")?,
        upstream_size,
        |mut builder| {
            builder
                .arg(&dest_ptr)
                .arg(d_up)
                .arg(&d_input_shape)
                .arg(&d_starts)
                .arg(&d_upstream_shape)
                .arg(&rank_i)
                .arg(&upstream_size_u64);
            builder
        },
    )?;
    Ok(dest.clone())
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
        let src = backend
            .stream
            .clone_htod(&bf16_bits)
            .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (bf16 test)"))?;
        let (src_ptr, _src_guard) = src.device_ptr(&backend.stream);

        let staging = backend.copy_bf16_device_ptr_to_local(src_ptr, bf16_bits.len())?;
        let copied = backend
            .stream
            .clone_dtoh(&staging)
            .map_err(|_| AutogradError::TapeInvariant("cuda dtoh copy failed (bf16 test)"))?;
        backend
            .stream
            .synchronize()
            .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed (bf16 test)"))?;
        assert_eq!(copied, bf16_bits);

        let handle =
            backend.import_bf16_device_ptr_as_f32(src_ptr, bf16_bits.len(), &[bf16_bits.len()])?;
        let widened = backend.readback(&handle)?;
        let expected: Vec<f32> = bf16_bits
            .iter()
            .map(|&bits| f32::from_bits((bits as u32) << 16))
            .collect();
        assert_eq!(widened, expected);
        Ok(())
    }
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_download(
    backend: &CudaBackend,
    d_out: &cudarc::driver::CudaSlice<f32>,
    len: usize,
) -> Result<Vec<f32>> {
    let mut host = vec![0.0_f32; len];
    backend
        .stream
        .memcpy_dtoh(d_out, &mut host)
        .map_err(|_| AutogradError::TapeInvariant("cuda dtoh copy failed"))?;
    backend
        .stream
        .synchronize()
        .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))?;
    Ok(host)
}

// Device-resident embedding backward. Allocates a
// zero-filled `[vocab, hidden]` grad on-device and atomicAdd-scatters the
// per-token-position upstream slice into `grad_table[ids[row], :]`. Only the
// int32 `indices` array crosses PCIe; the `[n_ids, hidden]` upstream stays
// on-device. No `synchronize()` — terminal eval is the caller's.
#[cfg(not(feature = "no-cuda"))]
fn cuda_embedding_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    indices: &[i32],
    vocab_size: usize,
    hidden_dim: usize,
) -> Result<DeviceHandle> {
    let n_ids = indices.len();
    let expected_upstream = n_ids * hidden_dim;
    // grad_table is always f32 (param-grad boundary); only the upstream
    // activation grad may arrive bf16.
    let up_bf16 = matches!(upstream, DeviceHandle::CudaBf16(_));
    let up_len = match upstream {
        DeviceHandle::CudaBf16(storage) => backend.cuda_bf16_storage_slice(storage)?.len(),
        _ => backend
            .cuda_slice(upstream, "embedding_backward_device")?
            .len(),
    };
    if up_len != expected_upstream {
        return Err(AutogradError::DataLengthMismatch {
            len: up_len,
            shape: vec![n_ids, hidden_dim],
            size: expected_upstream,
        });
    }

    let out_len = vocab_size * hidden_dim;
    // alloc_zeros gives the required zero-init contract — the kernel only adds.
    let mut d_grad = backend.stream.alloc_zeros::<f32>(out_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (embedding_backward_device)")
    })?;

    if n_ids == 0 || hidden_dim == 0 {
        return Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)));
    }

    let d_ids = backend.stream.clone_htod(indices).map_err(|_| {
        AutogradError::TapeInvariant("cuda htod copy failed (embedding_backward ids)")
    })?;

    let n_ids_i32 = i32::try_from(n_ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding_backward n_ids exceeds i32"))?;
    let hidden_i32 = i32::try_from(hidden_dim).map_err(|_| {
        AutogradError::TapeInvariant("cuda embedding_backward hidden_dim exceeds i32")
    })?;
    let vocab_i32 = i32::try_from(vocab_size).map_err(|_| {
        AutogradError::TapeInvariant("cuda embedding_backward vocab_size exceeds i32")
    })?;

    // One thread per token position (block=256 via launch_1d). Inner
    // per-thread loop strides `hidden_dim` columns with atomicAdd. With
    // n_ids = B*S = 1024 on the canonical bench shape, this dispatches
    // 4 blocks × 256 threads — atomicAdd traffic dominates, so block-size
    // selection beyond "warp-aligned" is in the noise.
    if up_bf16 {
        let d_up = backend.cuda_bf16_slice(upstream, "embedding_backward_device")?;
        let func = backend
            .kernels
            .function_for("embedding_backward_f32", TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, n_ids, |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_ids)
                .arg(&n_ids_i32)
                .arg(&hidden_i32)
                .arg(&vocab_i32);
            builder
        })?;
        return Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)));
    }
    let d_up = backend.cuda_slice(upstream, "embedding_backward_device")?;
    launch_1d(
        &backend.stream,
        backend.kernels.function("embedding_backward_f32")?,
        n_ids,
        |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_ids)
                .arg(&n_ids_i32)
                .arg(&hidden_i32)
                .arg(&vocab_i32);
            builder
        },
    )?;

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
}

// Device-resident add_broadcast backward. Reduces the
// upstream tensor along broadcast axes into a `[b_shape]` grad. One block
// per output element; threads cooperatively reduce over the cartesian
// product of contracted axes via shared memory.
#[cfg(not(feature = "no-cuda"))]
fn cuda_add_broadcast_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<DeviceHandle> {
    validate_broadcast(a_shape, b_shape)?;
    let out_rank = a_shape.len();
    if out_rank > 8 {
        return Err(AutogradError::InvalidRank {
            expected: "<= 8",
            got: out_rank,
        });
    }
    let a_total: usize = if a_shape.is_empty() {
        1
    } else {
        a_shape.iter().product()
    };
    let b_total: usize = if b_shape.is_empty() {
        1
    } else {
        b_shape.iter().product()
    };
    let up_bf16 = matches!(upstream, DeviceHandle::CudaBf16(_));
    let up_len = match upstream {
        DeviceHandle::CudaBf16(storage) => backend.cuda_bf16_storage_slice(storage)?.len(),
        _ => backend
            .cuda_slice(upstream, "add_broadcast_backward_device")?
            .len(),
    };
    if up_len != a_total {
        return Err(AutogradError::DataLengthMismatch {
            len: up_len,
            shape: a_shape.to_vec(),
            size: a_total,
        });
    }

    if a_total == 0 || b_total == 0 || out_rank == 0 {
        return if up_bf16 {
            let zeros = backend
                .stream
                .alloc_zeros::<u16>(b_total.max(1))
                .map_err(|_| {
                    AutogradError::TapeInvariant(
                        "cuda alloc_zeros failed (add_broadcast_backward_device)",
                    )
                })?;
            Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(zeros)))
        } else {
            let zeros = backend
                .stream
                .alloc_zeros::<f32>(b_total.max(1))
                .map_err(|_| {
                    AutogradError::TapeInvariant(
                        "cuda alloc_zeros failed (add_broadcast_backward_device)",
                    )
                })?;
            Ok(DeviceHandle::Cuda(CudaStorage::new(zeros)))
        };
    }

    // Build right-aligned b-strides (length=out_rank, 0 on contracted axes;
    // contiguous row-major stride within b on matching axes). Mirrors the
    // forward `cuda_add_broadcast` helper.
    let rank_offset = out_rank - b_shape.len();
    let mut b_strides = vec![0_i32; out_rank];
    let mut stride_b: i32 = 1;
    for i in (0..b_shape.len()).rev() {
        let dim = b_shape[i];
        if dim == 1 {
            b_strides[rank_offset + i] = 0;
        } else {
            b_strides[rank_offset + i] = stride_b;
        }
        stride_b = stride_b.saturating_mul(dim as i32);
    }
    // Row-major contiguous strides in upstream (a-shape layout).
    let mut out_strides = vec![0_i32; out_rank];
    let mut stride_a: i32 = 1;
    for i in (0..out_rank).rev() {
        out_strides[i] = stride_a;
        stride_a = stride_a.saturating_mul(a_shape[i] as i32);
    }
    // contract_total = product of out_shape[d] over axes where b_strides[d]==0.
    let contract_total: i64 = (0..out_rank)
        .filter(|&d| b_strides[d] == 0)
        .map(|d| a_shape[d] as i64)
        .product();
    let contract_total_i32 = i32::try_from(contract_total).map_err(|_| {
        AutogradError::TapeInvariant("cuda add_broadcast_backward contract_total exceeds i32")
    })?;

    let out_shape_i32: Vec<i32> = a_shape.iter().map(|&d| d as i32).collect();

    let d_out_shape = backend.stream.clone_htod(&out_shape_i32).map_err(|_| {
        AutogradError::TapeInvariant("cuda htod copy failed (add_broadcast_bwd out_shape)")
    })?;
    let d_b_strides = backend.stream.clone_htod(&b_strides).map_err(|_| {
        AutogradError::TapeInvariant("cuda htod copy failed (add_broadcast_bwd b_strides)")
    })?;
    let d_out_strides = backend.stream.clone_htod(&out_strides).map_err(|_| {
        AutogradError::TapeInvariant("cuda htod copy failed (add_broadcast_bwd out_strides)")
    })?;

    let out_rank_i32 = i32::try_from(out_rank).map_err(|_| {
        AutogradError::TapeInvariant("cuda add_broadcast_backward out_rank exceeds i32")
    })?;
    let b_total_i32 = i32::try_from(b_total).map_err(|_| {
        AutogradError::TapeInvariant("cuda add_broadcast_backward b_total exceeds i32")
    })?;

    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
    if up_bf16 {
        let d_up_op = backend.bf16_operand(upstream, "add_broadcast_backward_device")?;
        let d_up = d_up_op.get();
        let mut d_grad = backend.stream.alloc_zeros::<u16>(b_total).map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (add_broadcast_backward_device)")
        })?;
        let func = backend
            .kernels
            .function_for("add_broadcast_backward_f32", TapeDtype::Bf16)?;
        launch_rows(
            &backend.stream,
            &func,
            b_total,
            BLOCK,
            SHARED,
            |mut builder| {
                builder
                    .arg(&mut d_grad)
                    .arg(d_up)
                    .arg(&d_out_shape)
                    .arg(&d_b_strides)
                    .arg(&d_out_strides)
                    .arg(&out_rank_i32)
                    .arg(&b_total_i32)
                    .arg(&contract_total_i32);
                builder
            },
        )?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_grad)));
    }

    let d_up = backend.cuda_slice(upstream, "add_broadcast_backward_device")?;
    let mut d_grad = backend.stream.alloc_zeros::<f32>(b_total).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (add_broadcast_backward_device)")
    })?;
    launch_rows(
        &backend.stream,
        backend.kernels.function("add_broadcast_backward_f32")?,
        b_total,
        BLOCK,
        SHARED,
        |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_out_shape)
                .arg(&d_b_strides)
                .arg(&d_out_strides)
                .arg(&out_rank_i32)
                .arg(&b_total_i32)
                .arg(&contract_total_i32);
            builder
        },
    )?;

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_elementwise_backward_with_saved(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    saved: &DeviceHandle,
    shape: &[usize],
    kernel_name: &'static str,
    op_label: &'static str,
) -> Result<DeviceHandle> {
    let size = shape_size(shape);
    // Adjoint of the forward's actual precision: under bf16 tape the forward
    // consumed bf16 operands, so backward re-quantizes the same way.
    if backend.tape_bf16() {
        let d_up_op = backend.bf16_operand(upstream, op_label)?;
        let d_saved_op = backend.bf16_operand(saved, op_label)?;
        let d_up = d_up_op.get();
        let d_saved = d_saved_op.get();
        if d_up.len() != size || d_saved.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: d_up.len().min(d_saved.len()),
                shape: shape.to_vec(),
                size,
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(size)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
        let n = size as u64;
        let func = backend.kernels.function_for(kernel_name, TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, size, |mut builder| {
            builder.arg(&mut d_out).arg(d_up).arg(d_saved).arg(&n);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }
    let d_up = backend.cuda_slice(upstream, op_label)?;
    let d_saved = backend.cuda_slice(saved, op_label)?;
    if d_up.len() != size || d_saved.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_up.len().min(d_saved.len()),
            shape: shape.to_vec(),
            size,
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
    let n = size as u64;
    launch_1d(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        size,
        |mut builder| {
            builder.arg(&mut d_out).arg(d_up).arg(d_saved).arg(&n);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_silu_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    x: &DeviceHandle,
    shape: &[usize],
) -> Result<DeviceHandle> {
    cuda_elementwise_backward_with_saved(
        backend,
        upstream,
        x,
        shape,
        "silu_backward_f32",
        "silu_backward_device",
    )
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_gelu_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    x: &DeviceHandle,
    shape: &[usize],
) -> Result<DeviceHandle> {
    cuda_elementwise_backward_with_saved(
        backend,
        upstream,
        x,
        shape,
        "gelu_backward_f32",
        "gelu_backward_device",
    )
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_sigmoid_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    y: &DeviceHandle,
    shape: &[usize],
) -> Result<DeviceHandle> {
    cuda_elementwise_backward_with_saved(
        backend,
        upstream,
        y,
        shape,
        "sigmoid_backward_f32",
        "sigmoid_backward_device",
    )
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_exp_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    y: &DeviceHandle,
    shape: &[usize],
) -> Result<DeviceHandle> {
    cuda_elementwise_backward_with_saved(
        backend,
        upstream,
        y,
        shape,
        "exp_backward_f32",
        "exp_backward_device",
    )
}

#[cfg(not(feature = "no-cuda"))]
fn cuda_mul_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    a: &DeviceHandle,
    b: &DeviceHandle,
    shape: &[usize],
    need_grad_a: bool,
    need_grad_b: bool,
) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
    if !need_grad_a && !need_grad_b {
        return Ok((None, None));
    }
    let d_up_op = backend.f32_operand(upstream, "mul_backward_device")?;
    let d_a_op = backend.f32_operand_tape_quantized(a, "mul_backward_device")?;
    let d_b_op = backend.f32_operand_tape_quantized(b, "mul_backward_device")?;
    let d_up = d_up_op.get();
    let d_a = d_a_op.get();
    let d_b = d_b_op.get();
    let size = shape_size(shape);
    if d_up.len() != size || d_a.len() != size || d_b.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_up.len().min(d_a.len()).min(d_b.len()),
            shape: shape.to_vec(),
            size,
        });
    }
    let n = size as u64;

    let grad_a = if need_grad_a {
        let mut d_out = backend.stream.alloc_zeros::<f32>(size).map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (mul_backward grad_a)")
        })?;
        launch_1d(
            &backend.stream,
            backend.kernels.function("mul_backward_lhs_f32")?,
            size,
            |mut builder| {
                builder.arg(&mut d_out).arg(d_up).arg(d_b).arg(&n);
                builder
            },
        )?;
        Some(DeviceHandle::Cuda(CudaStorage::new(d_out)))
    } else {
        None
    };
    let grad_b = if need_grad_b {
        let mut d_out = backend.stream.alloc_zeros::<f32>(size).map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (mul_backward grad_b)")
        })?;
        launch_1d(
            &backend.stream,
            backend.kernels.function("mul_backward_rhs_f32")?,
            size,
            |mut builder| {
                builder.arg(&mut d_out).arg(d_up).arg(d_a).arg(&n);
                builder
            },
        )?;
        Some(DeviceHandle::Cuda(CudaStorage::new(d_out)))
    } else {
        None
    };
    Ok((grad_a, grad_b))
}

// Device-resident backward for `rms_norm`. Three kernels:
//   1. `rms_norm_inv_rms_f32` — one block per row, reduces sum_sq and
//      emits `inv_rms[rows]` to a device scratch buffer.
//   2. `rms_norm_backward_x_f32` — one block per row, consumes the saved
//      `inv_rms` and reduces `dot` (one shared-mem reduction).
//   3. `rms_norm_backward_w_f32` — one block per column, accumulates
//      `upstream * x * inv_rms` across rows and reduces to grad_w.
// Returned handles are unevaluated; the terminal `eval` belongs to the
// caller.
#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
fn cuda_rms_norm_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    x: &DeviceHandle,
    weight: &DeviceHandle,
    shape: &[usize],
    eps: f32,
    need_grad_x: bool,
    need_grad_w: bool,
) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
    if !need_grad_x && !need_grad_w {
        return Ok((None, None));
    }
    let hidden = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if hidden == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let total = shape_size(shape);
    let rows = total / hidden;

    // Lane follows the saved x's dtype so the adjoint reads what forward saw;
    // grad_w stays f32 (param-grad boundary) in both lanes.
    if let DeviceHandle::CudaBf16(storage) = x {
        let d_x = backend.cuda_bf16_storage_slice(storage)?;
        let d_up_op = backend.bf16_operand(upstream, "rms_norm_backward_device")?;
        let d_up = d_up_op.get();
        let d_w = backend.cuda_slice(weight, "rms_norm_backward_device")?;
        if d_up.len() != total || d_x.len() != total || d_w.len() != hidden {
            return Err(AutogradError::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![d_up.len()],
            });
        }
        let mut d_inv = backend
            .stream
            .alloc_zeros::<f32>(rows.max(1))
            .map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward inv_rms)")
            })?;
        let cols_i = i32::try_from(hidden)
            .map_err(|_| AutogradError::TapeInvariant("cuda rms_norm_backward cols exceeds i32"))?;
        const BLOCK: u32 = 256;
        const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
        if rows > 0 {
            let func = backend
                .kernels
                .function_for("rms_norm_inv_rms_f32", TapeDtype::Bf16)?;
            launch_rows(
                &backend.stream,
                &func,
                rows,
                BLOCK,
                SHARED,
                |mut builder| {
                    builder.arg(&mut d_inv).arg(d_x).arg(&cols_i).arg(&eps);
                    builder
                },
            )?;
        }
        let grad_x = if need_grad_x {
            let mut d_grad = backend.stream.alloc_zeros::<u16>(total).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward grad_x)")
            })?;
            if rows > 0 {
                let func = backend
                    .kernels
                    .function_for("rms_norm_backward_x_f32", TapeDtype::Bf16)?;
                launch_rows(
                    &backend.stream,
                    &func,
                    rows,
                    BLOCK,
                    SHARED,
                    |mut builder| {
                        builder
                            .arg(&mut d_grad)
                            .arg(d_up)
                            .arg(d_x)
                            .arg(d_w)
                            .arg(&d_inv)
                            .arg(&cols_i);
                        builder
                    },
                )?;
            }
            Some(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_grad)))
        } else {
            None
        };
        let grad_w = if need_grad_w {
            let mut d_grad = backend.stream.alloc_zeros::<f32>(hidden).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward grad_w)")
            })?;
            if rows > 0 && hidden > 0 {
                let rows_i = i32::try_from(rows).map_err(|_| {
                    AutogradError::TapeInvariant("cuda rms_norm_backward rows exceeds i32")
                })?;
                let func = backend
                    .kernels
                    .function_for("rms_norm_backward_w_f32", TapeDtype::Bf16)?;
                launch_rows(
                    &backend.stream,
                    &func,
                    hidden,
                    BLOCK,
                    SHARED,
                    |mut builder| {
                        builder
                            .arg(&mut d_grad)
                            .arg(d_up)
                            .arg(d_x)
                            .arg(&d_inv)
                            .arg(&rows_i)
                            .arg(&cols_i);
                        builder
                    },
                )?;
            }
            Some(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
        } else {
            None
        };
        return Ok((grad_x, grad_w));
    }

    let d_up_op = backend.f32_operand(upstream, "rms_norm_backward_device")?;
    let d_x_op = backend.f32_operand(x, "rms_norm_backward_device")?;
    let d_up = d_up_op.get();
    let d_x = d_x_op.get();
    let d_w = backend.cuda_slice(weight, "rms_norm_backward_device")?;
    if d_up.len() != total || d_x.len() != total || d_w.len() != hidden {
        return Err(AutogradError::ShapeMismatch {
            expected: shape.to_vec(),
            got: vec![d_up.len()],
        });
    }

    // Inv_rms scratch buffer.
    let mut d_inv = backend
        .stream
        .alloc_zeros::<f32>(rows.max(1))
        .map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward inv_rms)")
        })?;
    if rows > 0 {
        let cols_i = i32::try_from(hidden)
            .map_err(|_| AutogradError::TapeInvariant("cuda rms_norm_backward cols exceeds i32"))?;
        const BLOCK: u32 = 256;
        const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
        launch_rows(
            &backend.stream,
            backend.kernels.function("rms_norm_inv_rms_f32")?,
            rows,
            BLOCK,
            SHARED,
            |mut builder| {
                builder.arg(&mut d_inv).arg(d_x).arg(&cols_i).arg(&eps);
                builder
            },
        )?;
    }

    let grad_x = if need_grad_x {
        let mut d_grad = backend.stream.alloc_zeros::<f32>(total).map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward grad_x)")
        })?;
        if rows > 0 {
            let cols_i = i32::try_from(hidden).map_err(|_| {
                AutogradError::TapeInvariant("cuda rms_norm_backward cols exceeds i32")
            })?;
            const BLOCK: u32 = 256;
            const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
            launch_rows(
                &backend.stream,
                backend.kernels.function("rms_norm_backward_x_f32")?,
                rows,
                BLOCK,
                SHARED,
                |mut builder| {
                    builder
                        .arg(&mut d_grad)
                        .arg(d_up)
                        .arg(d_x)
                        .arg(d_w)
                        .arg(&d_inv)
                        .arg(&cols_i);
                    builder
                },
            )?;
        }
        Some(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
    } else {
        None
    };

    let grad_w = if need_grad_w {
        let mut d_grad = backend.stream.alloc_zeros::<f32>(hidden).map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (rms_norm_backward grad_w)")
        })?;
        if rows > 0 && hidden > 0 {
            let rows_i = i32::try_from(rows).map_err(|_| {
                AutogradError::TapeInvariant("cuda rms_norm_backward rows exceeds i32")
            })?;
            let cols_i = i32::try_from(hidden).map_err(|_| {
                AutogradError::TapeInvariant("cuda rms_norm_backward cols exceeds i32")
            })?;
            const BLOCK: u32 = 256;
            const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;
            launch_rows(
                &backend.stream,
                backend.kernels.function("rms_norm_backward_w_f32")?,
                hidden,
                BLOCK,
                SHARED,
                |mut builder| {
                    builder
                        .arg(&mut d_grad)
                        .arg(d_up)
                        .arg(d_x)
                        .arg(&d_inv)
                        .arg(&rows_i)
                        .arg(&cols_i);
                    builder
                },
            )?;
        }
        Some(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
    } else {
        None
    };

    Ok((grad_x, grad_w))
}

// Device-resident backward for `rope`. Same launch shape as
// `cuda_rope` (one block per (batch, head, token); block=min(half_dim,256)).
// Only difference vs the forward kernel is the inlined `sin -> -sin` sign
// flip — `cpu_rope_backward` does the equivalent via a host
// `neg_forward(sin) → cpu_rope_forward` chain. cos/sin upload fresh every
// call (tiny: `[seq, head_dim/2]` per call).
#[cfg(not(feature = "no-cuda"))]
fn cuda_rope_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    x_shape: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> Result<DeviceHandle> {
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }
    let batch = x_shape[0];
    let heads = x_shape[1];
    let seq = x_shape[2];
    let head_dim = x_shape[3];
    if !head_dim.is_multiple_of(2) {
        return Err(AutogradError::InvalidRank {
            expected: "even head dim",
            got: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    let total = batch * heads * seq * head_dim;
    // Partial rotary: see `cuda_rope_forward`.
    let rot_half = cos.len().checked_div(seq).unwrap_or(0);
    if cos.len() != sin.len() || rot_half == 0 || rot_half > half_dim || cos.len() != seq * rot_half
    {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![seq * half_dim],
            got: vec![cos.len().min(sin.len())],
        });
    }

    let d_cos = backend
        .stream
        .clone_htod(cos)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (rope_backward cos)"))?;
    let d_sin = backend
        .stream
        .clone_htod(sin)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (rope_backward sin)"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward batch exceeds i32"))?;
    let heads_i = i32::try_from(heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward heads exceeds i32"))?;
    let seq_i = i32::try_from(seq)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward seq exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward head_dim exceeds i32"))?;
    let rot_half_i = i32::try_from(rot_half)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward rot_half exceeds i32"))?;

    let rows = batch * heads * seq;
    let block = std::cmp::min(half_dim, 256) as u32;
    let block = block.max(1);
    if let DeviceHandle::CudaBf16(storage) = upstream {
        let d_up = backend.cuda_bf16_storage_slice(storage)?;
        if d_up.len() != total {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![total],
                got: vec![d_up.len()],
            });
        }
        let mut d_grad = backend
            .stream
            .alloc_zeros::<u16>(total)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (rope_backward)"))?;
        let func = backend
            .kernels
            .function_for("rope_backward_f32", TapeDtype::Bf16)?;
        launch_rows(&backend.stream, &func, rows, block, 0, |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_cos)
                .arg(&d_sin)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&rot_half_i);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_grad)));
    }
    let d_up = backend.cuda_slice(upstream, "rope_backward_device")?;
    if d_up.len() != total {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![total],
            got: vec![d_up.len()],
        });
    }
    let mut d_grad = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (rope_backward)"))?;
    launch_rows(
        &backend.stream,
        backend.kernels.function("rope_backward_f32")?,
        rows,
        block,
        0,
        |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_cos)
                .arg(&d_sin)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&rot_half_i);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
}

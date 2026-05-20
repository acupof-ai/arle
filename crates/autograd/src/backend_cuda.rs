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
        CudaStorage, matmul_bt_output_shape, matmul_output_shape, validate_broadcast,
        validate_decode_gqa_shapes,
    },
};
use crate::{
    Result,
    backend::{Backend, Device, DeviceGradClipResult, DeviceHandle},
};
#[cfg(not(feature = "no-cuda"))]
#[path = "backend_cuda/kernels.rs"]
mod kernels;

#[cfg(not(feature = "no-cuda"))]
use self::kernels::{KernelCache, launch_1d, launch_rows};
#[cfg(not(feature = "no-cuda"))]
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
#[cfg(not(feature = "no-cuda"))]
use cudarc::cublas::sys::cublasOperation_t;
#[cfg(not(feature = "no-cuda"))]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, PushKernelArg};
#[cfg(not(feature = "no-cuda"))]
use std::sync::Arc;

/// cuBLAS-backed matmul plus NVRTC-compiled point kernels. Holds an
/// `Arc<CudaStream>` + `CudaBlas` so the context lives as long as the backend;
/// safe to share across threads.
#[derive(Debug)]
pub struct CudaBackend {
    #[cfg(not(feature = "no-cuda"))]
    stream: Arc<CudaStream>,
    #[cfg(not(feature = "no-cuda"))]
    blas: Arc<CudaBlas>,
    #[cfg(not(feature = "no-cuda"))]
    kernels: KernelCache,
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
                stream,
                blas: Arc::new(blas),
                kernels,
            })
        }
    }

    #[cfg(not(feature = "no-cuda"))]
    fn upload_slice(&self, host: &[f32], shape: &[usize]) -> Result<CudaSlice<f32>> {
        let size = shape_size(shape);
        if host.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: shape.to_vec(),
                size,
            });
        }

        self.stream
            .clone_htod(host)
            .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))
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
    fn cuda_slice<'a>(
        &self,
        handle: &'a DeviceHandle,
        op: &'static str,
    ) -> Result<&'a CudaSlice<f32>> {
        match handle {
            DeviceHandle::Cuda(storage) => self.cuda_storage_slice(storage),
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
    fn validate_cuda_handle_kind(&self, handle: &DeviceHandle) -> Result<()> {
        match handle {
            DeviceHandle::Cpu(_) | DeviceHandle::Cuda(_) => Ok(()),
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
                    .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

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
                    .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

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
}

impl Backend for CudaBackend {
    fn device(&self) -> Device {
        Device::Cuda
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
                .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
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
                #[cfg(feature = "metal")]
                DeviceHandle::Metal(_) => Err(AutogradError::TapeInvariant(
                    "cuda backend cannot read back a metal device handle",
                )),
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
            let a = self.cuda_slice(a, "matmul")?;
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
            let d_a = self.cuda_slice(a, "matmul_bt")?;
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
                .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

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
            let a = self.cuda_slice(a, "add")?;
            let b = self.cuda_slice(b, "add")?;
            let size = shape_size(shape);
            if a.len() != size || b.len() != size {
                return Err(AutogradError::ShapeMismatch {
                    expected: shape.to_vec(),
                    got: vec![a.len().min(b.len())],
                });
            }

            let mut out = self
                .stream
                .alloc_zeros::<f32>(size)
                .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
            let n = i32::try_from(size)
                .map_err(|_| AutogradError::TapeInvariant("cuda add length exceeds i32"))?;
            launch_1d(
                &self.stream,
                self.kernels.function("add_f32")?,
                size,
                |mut builder| {
                    builder.arg(&mut out).arg(a).arg(b).arg(&n);
                    builder
                },
            )?;
            Ok(DeviceHandle::Cuda(CudaStorage::new(out)))
        }
    }

    fn sum_all(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = (x, shape);
            todo!("GPU required: cuda sum_all is unavailable under feature no-cuda")
        }

        // CUDA stays eager for M5.3b.1 — no device-resident lazy graph here.
        // We still keep the result on-device so the returned handle composes
        // with future device-resident consumers (e.g. tape.backward()'s
        // ensure_host of the loss). Strategy: download → reduce on host →
        // upload the scalar back. This is one HtoD round-trip and a tiny
        // sum, dwarfed by the matmul that produced `x`. PENDING REMOTE CUDA
        // VERIFICATION.
        #[cfg(not(feature = "no-cuda"))]
        {
            let slice = self.cuda_slice(x, "sum_all")?;
            let size = shape_size(shape);
            if slice.len() != size {
                return Err(AutogradError::DataLengthMismatch {
                    len: slice.len(),
                    shape: shape.to_vec(),
                    size,
                });
            }
            let mut host = vec![0.0_f32; size];
            self.stream
                .memcpy_dtoh(slice, &mut host)
                .map_err(|_| AutogradError::TapeInvariant("cuda dtoh copy failed"))?;
            self.stream
                .synchronize()
                .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))?;
            let total: f32 = host.iter().sum();
            let scalar = self
                .stream
                .clone_htod(&[total])
                .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
            Ok(DeviceHandle::Cuda(CudaStorage::new(scalar)))
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

    /// Device-resident matmul backward — foundation for the post-G3
    /// device-resident gradient tape. Mirrors the cuBLAS dispatch of the
    /// host-buffer `matmul_backward` (`grad_a = dC @ B^T`,
    /// `grad_b = A^T @ dC` via two SGEMMs with `OP_T` on the transposed
    /// operand) but consumes existing device handles and returns
    /// unevaluated `CudaSlice<f32>` outputs — no host roundtrip on either
    /// side. The terminal `backend.eval(...)` in `AdamW::step_device`
    /// performs the single host fence per training step (M5.3b.11
    /// batched-eval contract). See
    /// `docs/research/2026-05-17-cuda-training-architectural-correction.md`.
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

    /// Device-resident gradient accumulation. Allocates a fresh
    /// `CudaSlice<f32>` for the sum (so the previous `dest` handle remains
    /// valid for any tape consumers still holding it) and launches the
    /// `add_into_f32` 1D NVRTC kernel. Returns the unevaluated handle for
    /// the batched terminal `eval`. See
    /// `docs/research/2026-05-17-cuda-training-architectural-correction.md`.
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

    /// P3: device-resident backward for `mul_scalar`. Pure elementwise
    /// `grad_x[i] = upstream[i] * k` via a 1D NVRTC kernel; returns an
    /// unevaluated handle per the M5.3b.11 batched-eval contract.
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

    /// P3: device-resident backward for `mean`. Scalar `upstream_grad`
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

    /// M5.3b: device-resident row-wise softmax over the last axis. The
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

    /// M5.3b: device-resident row-wise log-softmax over the last axis.
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

    /// M5.3b: device-resident gather along the last axis. Reuses the
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

    /// Wave 1 (post-M5.3b nsys attribution): device-resident backward for
    /// `log_softmax_last_axis`. Consumes the saved forward output
    /// directly from its `DeviceHandle` (no DtoH) and the upstream gradient
    /// directly from device — kills the `1 015 MB` log_softmax-grad readback
    /// nsys identified as the single largest transfer per training step
    /// (see `docs/research/2026-05-17-cuda-training-step-nsys-attribution.md`).
    /// Returns an unevaluated `CudaSlice<f32>` handle per the M5.3b.11
    /// batched-eval contract — `Tape::backward`'s terminal eval (or the
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

    /// Wave 1: device-resident backward for `gather_last_dim`. Produces a
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
            let d_x = self.cuda_slice(x, "reshape")?;
            let expected = shape_size(new_shape);
            if d_x.len() != expected {
                return Err(AutogradError::DataLengthMismatch {
                    len: d_x.len(),
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

    /// Wave 2 Commit A: device-resident embedding backward.
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

    /// Wave 2 Commit A: device-resident add_broadcast backward.
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

    /// Wave 2.0: device-grad fused AdamW. Same kernel as `adamw_step`
    /// (`adamw_step_f32` from G3) but the gradient is sourced directly from
    /// the caller's `DeviceHandle::Cuda` — **no `clone_htod`**. This kills
    /// the per-param-per-grad-accum-step DtoH that Wave 2 Commit A
    /// inadvertently introduced when `embedding_backward` /
    /// `add_broadcast_backward` started producing device-resident grads.
    /// See `docs/experience/wins/2026-05-17-bench-pretrain-wave2a-embedding-addbcast.md`.
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

    /// Wave 2.1: device-resident backward for `silu(x)`. Single 1D NVRTC
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

    /// Wave 2.1: device-resident backward for `gelu(x)` (erf form).
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

    /// Wave 2.1: device-resident backward for `sigmoid(x)`. Consumes the
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

    /// Wave 2.1: device-resident backward for `exp(x)`. Consumes the saved
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

    /// Wave 2.1: device-resident backward for `mul(a, b)`. Two 1D NVRTC
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

    /// Wave 2.1: device-resident backward for `rms_norm`. Three NVRTC
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

    /// Wave 2.1: device-resident backward for `rope`. Single NVRTC kernel
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

    // Per the Backend::adamw_step eval contract (M5.3b.11): return
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

    // Eval contract (M5.3b.11): return unevaluated; caller batches the
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
// M5.3b.11 contract). Reused for both `softmax_last_axis_f32` and
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
    let d_in = backend.cuda_slice(x, "softmax_last_axis")?;
    if d_in.len() != expected {
        return Err(AutogradError::DataLengthMismatch {
            len: d_in.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }

    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda softmax cols exceeds i32"))?;
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

// Wave 1 (post-M5.3b nsys attribution): device-resident log_softmax
// backward. `upstream` and `log_softmax_output` arrive as borrowed CUDA
// slices via the `Backend::log_softmax_last_axis_backward` contract; the
// fresh grad allocation stays device-resident and is returned unevaluated
// for the tape's terminal eval (mirrors the M5.3b forward helper pattern).
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
    let d_up = backend.cuda_slice(upstream, "log_softmax_last_axis_backward")?;
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

    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda log_softmax_backward cols exceeds i32"))?;
    let mut d_grad = backend
        .stream
        .alloc_zeros::<f32>(expected)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (log_softmax_bwd)"))?;

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
    let d_up = backend.cuda_slice(upstream, "softmax_last_axis_backward")?;
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

    let rows = expected / last_dim;
    let cols = i32::try_from(last_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda softmax_backward cols exceeds i32"))?;
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

// Wave 1: device-resident backward for gather_last_dim. Allocates a
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

#[cfg(not(feature = "no-cuda"))]
fn shape_size(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
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
                    .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
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
                    .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
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
                    .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
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
                    .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
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
// fence per training step (M5.3b.11 contract). Foundation for the
// post-G3 device-resident gradient tape; the dispatch wiring lands in a
// follow-up subagent.
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

    let d_a = backend.cuda_slice(a, "matmul_backward_device")?;
    let d_b = backend.cuda_slice(b, "matmul_backward_device")?;
    let d_g = backend.cuda_slice(grad_out, "matmul_backward_device")?;

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
                    .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
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
                    .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
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
                    .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
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
                    .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
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

    let d_a = backend.cuda_slice(a, "matmul_bt_backward_device")?;
    let d_b = backend.cuda_slice(b, "matmul_bt_backward_device")?;
    let d_g = backend.cuda_slice(grad_out, "matmul_bt_backward_device")?;
    if d_a.len() != shape_size(a_shape)
        || d_b.len() != shape_size(b_shape)
        || d_g.len() != shape_size(grad_out_shape)
    {
        return Err(AutogradError::TapeInvariant(
            "cuda matmul_bt_backward_device handle size does not match shape",
        ));
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];

    let grad_a = if need_grad_a {
        let (c, out_shape) = backend.matmul_device(d_g, grad_out_shape, d_b, b_shape)?;
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
        // grad_b[N,K] = grad_out^T[N,M] @ A[M,K]. The output's row-major
        // buffer is cuBLAS's column-major [K,N], so compute A^T[K,M] @
        // grad_out[M,N] directly into that column-major view.
        let mut c = backend
            .stream
            .alloc_zeros::<f32>(n * k)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
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

// P3: device-resident backward for `mul_scalar(x, k)`. Reads
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

// P3: device-resident backward for `mean(x)`. The upstream is a rank-0
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

// Device-resident gradient accumulation. Allocates a fresh output buffer
// and writes `dest[i] + src[i]` via the `add_into_f32` NVRTC kernel. The
// returned handle is unevaluated — terminal `eval` is the caller's.
// Foundation for the post-G3 device-resident gradient tape; dispatch wires
// in a follow-up subagent.
#[cfg(not(feature = "no-cuda"))]
fn cuda_add_into_device(
    backend: &CudaBackend,
    dest: &DeviceHandle,
    src: &DeviceHandle,
    shape: &[usize],
) -> Result<DeviceHandle> {
    let d_dest = backend.cuda_slice(dest, "add_into_device")?;
    let d_src = backend.cuda_slice(src, "add_into_device")?;
    let size = shape_size(shape);
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
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (add_into_device)"))?;
    let n = i32::try_from(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda add_into length exceeds i32"))?;
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
    let n = i32::try_from(n_usize)
        .map_err(|_| AutogradError::TapeInvariant("cuda unary length exceeds i32"))?;
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
    let n = i32::try_from(n_usize)
        .map_err(|_| AutogradError::TapeInvariant("cuda scalar length exceeds i32"))?;
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
    let n = i32::try_from(n_usize)
        .map_err(|_| AutogradError::TapeInvariant("cuda binary length exceeds i32"))?;
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
    let d_in = backend.cuda_slice(x, op_label)?;
    let size = shape_size(shape);
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
    let n = i32::try_from(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda unary length exceeds i32"))?;
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
    let d_in = backend.cuda_slice(x, op_label)?;
    let size = shape_size(shape);
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
    let n = i32::try_from(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda scalar length exceeds i32"))?;
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
    let d_a = backend.cuda_slice(a, op_label)?;
    let d_b = backend.cuda_slice(b, op_label)?;
    let size = shape_size(shape);
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
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
    let n = i32::try_from(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda binary length exceeds i32"))?;
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
    let d_x = backend.cuda_slice(x, "rms_norm")?;
    if d_x.len() != expected {
        return Err(AutogradError::DataLengthMismatch {
            len: d_x.len(),
            shape: shape.to_vec(),
            size: expected,
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
    let d_w = backend.cuda_slice(table, "embedding")?;
    if d_w.len() != vocab * dim {
        return Err(AutogradError::DataLengthMismatch {
            len: d_w.len(),
            shape: table_shape.to_vec(),
            size: vocab * dim,
        });
    }
    let n_ids = ids.len();
    let out_len = n_ids * dim;
    let d_ids = backend
        .stream
        .clone_htod(ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (embedding ids)"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(out_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (embedding)"))?;

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
                .arg(d_w)
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
    let d_w = backend.cuda_slice(table, "embedding_from_f32_ids")?;
    let d_ids = backend.cuda_slice(ids, "embedding_from_f32_ids")?;
    if d_w.len() != vocab * dim {
        return Err(AutogradError::DataLengthMismatch {
            len: d_w.len(),
            shape: table_shape.to_vec(),
            size: vocab * dim,
        });
    }
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
    let mut out_slices = Vec::with_capacity(grads.len());
    for (_, shape) in grads {
        let size = shape_size(shape);
        out_slices.push(backend.stream.alloc_zeros::<f32>(size).map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (grad_clip scaled)")
        })?);
    }

    let mut out_ptrs = Vec::with_capacity(out_slices.len());
    let mut out_guards = Vec::with_capacity(out_slices.len());
    for out in &mut out_slices {
        let (ptr, guard) = out.device_ptr_mut(&backend.stream);
        out_ptrs.push(ptr);
        out_guards.push(guard);
    }
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
    let cache_len = seq * half_dim;
    if cos.len() != cache_len || sin.len() != cache_len {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![cache_len],
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
                .arg(&head_dim_i);
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
    let d_x = backend.cuda_slice(x, "rope")?;
    if d_x.len() != total {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![total],
            got: vec![d_x.len()],
        });
    }
    let cache_len = seq * half_dim;
    if cos.len() != cache_len || sin.len() != cache_len {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![cache_len],
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
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (rope)"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope batch exceeds i32"))?;
    let heads_i = i32::try_from(heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope heads exceeds i32"))?;
    let seq_i = i32::try_from(seq)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope seq exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope head_dim exceeds i32"))?;

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
                .arg(d_x)
                .arg(&d_cos)
                .arg(&d_sin)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&head_dim_i);
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
    let d_x = backend.cuda_slice(x, "transpose_axes_swap")?;
    if d_x.len() != total {
        return Err(AutogradError::DataLengthMismatch {
            len: d_x.len(),
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
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (transpose)"))?;
    let rank_i = i32::try_from(rank)
        .map_err(|_| AutogradError::TapeInvariant("cuda transpose rank exceeds i32"))?;
    let axis1_i = i32::try_from(axis1)
        .map_err(|_| AutogradError::TapeInvariant("cuda transpose axis exceeds i32"))?;
    let axis2_i = i32::try_from(axis2)
        .map_err(|_| AutogradError::TapeInvariant("cuda transpose axis exceeds i32"))?;
    let total_i = i32::try_from(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda transpose total exceeds i32"))?;

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
    let d_x = backend.cuda_slice(x, "slice")?;
    if d_x.len() != old_total {
        return Err(AutogradError::DataLengthMismatch {
            len: d_x.len(),
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
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (slice)"))?;
    let rank_i = i32::try_from(rank)
        .map_err(|_| AutogradError::TapeInvariant("cuda slice rank exceeds i32"))?;
    let total_i = i32::try_from(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda slice total exceeds i32"))?;

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
                .arg(&total_i);
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
    let d_a = backend.cuda_slice(a, "concat_axis2")?;
    let d_b = backend.cuda_slice(b, "concat_axis2")?;
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

    let out_shape = vec![a_shape[0], a_shape[1], a_shape[2] + b_shape[2], a_shape[3]];
    let total = shape_size(&out_shape);
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (concat_axis2)"))?;
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
fn cuda_slice_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    input_shape: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> Result<DeviceHandle> {
    let rank = input_shape.len();
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
    for ((&start, &end), &dim) in starts.iter().zip(ends.iter()).zip(input_shape.iter()) {
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

    let upstream_shape: Vec<usize> = starts
        .iter()
        .zip(ends.iter())
        .map(|(&start, &end)| end - start)
        .collect();
    let upstream_size = shape_size(&upstream_shape);
    let input_size = shape_size(input_shape);
    let d_up = backend.cuda_slice(upstream, "slice_backward_device")?;
    if d_up.len() != upstream_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_up.len(),
            shape: upstream_shape.clone(),
            size: upstream_size,
        });
    }

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
    let mut d_grad = backend
        .stream
        .alloc_zeros::<f32>(input_size)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (slice_bwd)"))?;
    let rank_i = i32::try_from(rank)
        .map_err(|_| AutogradError::TapeInvariant("cuda slice_bwd rank exceeds i32"))?;
    let upstream_size_i = i32::try_from(upstream_size)
        .map_err(|_| AutogradError::TapeInvariant("cuda slice_bwd total exceeds i32"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("slice_backward_f32")?,
        upstream_size,
        |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_input_shape)
                .arg(&d_starts)
                .arg(&d_upstream_shape)
                .arg(&rank_i)
                .arg(&upstream_size_i);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
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

// Wave 2 Commit A: device-resident embedding backward. Allocates a
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
    let d_up = backend.cuda_slice(upstream, "embedding_backward_device")?;
    if d_up.len() != expected_upstream {
        return Err(AutogradError::DataLengthMismatch {
            len: d_up.len(),
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

// Wave 2 Commit A: device-resident add_broadcast backward. Reduces the
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
    let d_up = backend.cuda_slice(upstream, "add_broadcast_backward_device")?;
    if d_up.len() != a_total {
        return Err(AutogradError::DataLengthMismatch {
            len: d_up.len(),
            shape: a_shape.to_vec(),
            size: a_total,
        });
    }

    let mut d_grad = backend
        .stream
        .alloc_zeros::<f32>(b_total.max(1))
        .map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (add_broadcast_backward_device)")
        })?;

    if a_total == 0 || b_total == 0 || out_rank == 0 {
        return Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)));
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

// Wave 2.1: shared helper for the 4 elementwise activation/exp backward
// ops. All consume one extra device buffer (`saved`, either x or y) plus
// `upstream`; output is the same `shape`. Returned handle is unevaluated
// per the M5.3b.11 batched-eval contract.
#[cfg(not(feature = "no-cuda"))]
fn cuda_elementwise_backward_with_saved(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    saved: &DeviceHandle,
    shape: &[usize],
    kernel_name: &'static str,
    op_label: &'static str,
) -> Result<DeviceHandle> {
    let d_up = backend.cuda_slice(upstream, op_label)?;
    let d_saved = backend.cuda_slice(saved, op_label)?;
    let size = shape_size(shape);
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
    let n = i32::try_from(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda activation_backward length exceeds i32"))?;
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

// Wave 2.1: device-resident backward for `mul(a, b)`. Two independent 1D
// NVRTC kernels, each gated by `need_grad_*` so the unused side is never
// launched (mirrors `matmul_backward_device`'s short-circuit). Returned
// handles are unevaluated.
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
    let d_up = backend.cuda_slice(upstream, "mul_backward_device")?;
    let d_a = backend.cuda_slice(a, "mul_backward_device")?;
    let d_b = backend.cuda_slice(b, "mul_backward_device")?;
    let size = shape_size(shape);
    if d_up.len() != size || d_a.len() != size || d_b.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_up.len().min(d_a.len()).min(d_b.len()),
            shape: shape.to_vec(),
            size,
        });
    }
    let n = i32::try_from(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda mul_backward length exceeds i32"))?;

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

// Wave 2.1: device-resident backward for `rms_norm`. Three kernels:
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

    let d_up = backend.cuda_slice(upstream, "rms_norm_backward_device")?;
    let d_x = backend.cuda_slice(x, "rms_norm_backward_device")?;
    let d_w = backend.cuda_slice(weight, "rms_norm_backward_device")?;
    if d_up.len() != total || d_x.len() != total || d_w.len() != hidden {
        return Err(AutogradError::ShapeMismatch {
            expected: shape.to_vec(),
            got: vec![d_up.len()],
        });
    }

    // Phase 1: inv_rms scratch buffer.
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

// Wave 2.1: device-resident backward for `rope`. Same launch shape as
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
    let d_up = backend.cuda_slice(upstream, "rope_backward_device")?;
    if d_up.len() != total {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![total],
            got: vec![d_up.len()],
        });
    }
    let cache_len = seq * half_dim;
    if cos.len() != cache_len || sin.len() != cache_len {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![cache_len],
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
    let mut d_grad = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (rope_backward)"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward batch exceeds i32"))?;
    let heads_i = i32::try_from(heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward heads exceeds i32"))?;
    let seq_i = i32::try_from(seq)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward seq exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward head_dim exceeds i32"))?;

    let rows = batch * heads * seq;
    let block = std::cmp::min(half_dim, 256) as u32;
    let block = block.max(1);
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
                .arg(&head_dim_i);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
}

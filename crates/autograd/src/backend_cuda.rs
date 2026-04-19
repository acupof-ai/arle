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
    backend::{CudaStorage, matmul_output_shape},
};
use crate::{
    Result,
    backend::{Backend, Device, DeviceHandle},
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
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
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
                |builder| {
                    builder.arg(&mut out).arg(a).arg(b).arg(&n);
                },
            )?;
            Ok(DeviceHandle::Cuda(CudaStorage::new(out)))
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
        |builder| {
            builder.arg(&mut d_out).arg(&d_in).arg(&cols);
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

#[cfg(not(feature = "no-cuda"))]
fn shape_size(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
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
        |builder| {
            builder.arg(&mut d_out).arg(&d_in).arg(&n);
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
        |builder| {
            builder.arg(&mut d_out).arg(&d_in).arg(&s).arg(&n);
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
        |builder| {
            builder.arg(&mut d_out).arg(&d_a).arg(&d_b).arg(&n);
        },
    )?;
    cuda_download(backend, &d_out, n_usize)
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
        |builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_x)
                .arg(&d_w)
                .arg(&cols)
                .arg(&eps);
        },
    )?;
    cuda_download(backend, &d_out, expected)
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
        |builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_w)
                .arg(&d_ids)
                .arg(&n_i32)
                .arg(&vocab_i32)
                .arg(&dim_i32);
        },
    )?;
    cuda_download(backend, &d_out, out_len)
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
        |builder| {
            builder.arg(&mut d_out).arg(&d_in).arg(&cols);
        },
    )?;
    cuda_download(backend, &d_out, rows)
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

//! CUDA backend via `cudarc::cublas` SGEMM. Host `Vec<f32>` stays
//! authoritative: upload per call, SGEMM on device, download back.
//!
//! PENDING REMOTE CUDA VERIFICATION — user validates on GPU box.
//! Type-checks on Mac under `--no-default-features --features cuda,no-cuda`;
//! actual execution paths unreachable without a device are marked with
//! `todo!("GPU required: ...")` so a CPU-only binary fails loudly.
//!
//! Row-major dispatch uses the standard cuBLAS swap-and-transpose trick:
//! for row-major C[M,N] = A[M,K] @ B[K,N], call SGEMM with args swapped
//! (A=B_data, B=A_data) and m=N, n=M, k=K so cuBLAS's column-major view
//! of the output buffer matches the row-major layout we want on host.
//! Batched (rank-3) uses `sgemm_strided_batched` with the same swap.

use crate::{AutogradError, Result, backend::Backend, backend::Device};
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;

/// cuBLAS-backed matmul. Holds an `Arc<CudaStream>` + `CudaBlas` handle so the
/// context lives as long as the backend; safe to share across threads.
#[derive(Debug)]
pub struct CudaBackend {
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlas>,
}

impl CudaBackend {
    /// Create a backend bound to the CUDA device at `ordinal`.
    ///
    /// # Errors
    /// Returns an error if the device cannot be opened or cuBLAS cannot be
    /// initialised. On non-CUDA targets this code path is unreachable — the
    /// Mac build uses the `no-cuda` feature and surfaces a `todo!()` if run.
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).map_err(|_| {
            AutogradError::TapeInvariant("CudaContext::new failed (is a GPU present?)")
        })?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|_| AutogradError::TapeInvariant("CudaBlas::new failed"))?;
        Ok(Self {
            stream,
            blas: Arc::new(blas),
        })
    }
}

impl Backend for CudaBackend {
    fn device(&self) -> Device {
        Device::Cuda
    }

    fn matmul_forward(
        &self,
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        match (a_shape.len(), b_shape.len()) {
            (2, 2) => {
                let m = a_shape[0];
                let k = a_shape[1];
                if b_shape[0] != k {
                    return Err(AutogradError::ShapeMismatch {
                        expected: vec![k],
                        got: vec![b_shape[0]],
                    });
                }
                let n = b_shape[1];

                let a_dev = self
                    .stream
                    .clone_htod(a)
                    .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed for A"))?;
                let b_dev = self
                    .stream
                    .clone_htod(b)
                    .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed for B"))?;
                let mut c_dev = self
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
                        .gemm(cfg, &b_dev, &a_dev, &mut c_dev)
                        .map_err(|_| AutogradError::TapeInvariant("cuBLAS sgemm failed"))?;
                }

                let mut host = vec![0.0f32; m * n];
                self.stream
                    .memcpy_dtoh(&c_dev, &mut host)
                    .map_err(|_| AutogradError::TapeInvariant("cuda dtoh copy failed"))?;
                self.stream
                    .synchronize()
                    .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))?;

                Ok((host, vec![m, n]))
            }
            (3, 3) => {
                let batch = a_shape[0];
                if b_shape[0] != batch {
                    return Err(AutogradError::ShapeMismatch {
                        expected: vec![batch],
                        got: vec![b_shape[0]],
                    });
                }
                let m = a_shape[1];
                let k = a_shape[2];
                if b_shape[1] != k {
                    return Err(AutogradError::ShapeMismatch {
                        expected: vec![k],
                        got: vec![b_shape[1]],
                    });
                }
                let n = b_shape[2];

                let a_dev = self
                    .stream
                    .clone_htod(a)
                    .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed for A"))?;
                let b_dev = self
                    .stream
                    .clone_htod(b)
                    .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed for B"))?;
                let mut c_dev = self
                    .stream
                    .alloc_zeros::<f32>(batch * m * n)
                    .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

                let gemm_cfg = GemmConfig::<f32> {
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
                    gemm: gemm_cfg,
                    batch_size: batch as i32,
                    stride_a: (k * n) as i64,
                    stride_b: (m * k) as i64,
                    stride_c: (m * n) as i64,
                };

                // Safety: shapes validated above; device buffers outlive the call.
                unsafe {
                    self.blas
                        .gemm_strided_batched(cfg, &b_dev, &a_dev, &mut c_dev)
                        .map_err(|_| {
                            AutogradError::TapeInvariant("cuBLAS sgemm_strided_batched failed")
                        })?;
                }

                let mut host = vec![0.0f32; batch * m * n];
                self.stream
                    .memcpy_dtoh(&c_dev, &mut host)
                    .map_err(|_| AutogradError::TapeInvariant("cuda dtoh copy failed"))?;
                self.stream
                    .synchronize()
                    .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))?;

                Ok((host, vec![batch, m, n]))
            }
            _ => Err(AutogradError::InvalidRank {
                expected: "both operands must be rank-2 or rank-3",
                got: a_shape.len().max(b_shape.len()),
            }),
        }
    }
}

use super::*;

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
pub(super) fn cuda_matmul_backward(
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
pub(super) fn cuda_matmul_backward_device(
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
pub(super) fn cuda_matmul_bt_input_grad_device(
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
pub(super) fn cuda_matmul_bt_backward_device(
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

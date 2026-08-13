use super::*;

impl CudaBackend {
    #[cfg(not(feature = "no-cuda"))]
    pub(super) fn cublaslt_bf16_gemm_n(rows: usize) -> usize {
        if rows == 0 {
            0
        } else {
            rows.max(CUBLASLT_BF16_GEMMEX_MIN_N)
        }
    }

    #[cfg(not(feature = "no-cuda"))]
    pub(super) fn checked_bf16_len(rows: usize, cols: usize, op: &'static str) -> Result<usize> {
        rows.checked_mul(cols)
            .ok_or(AutogradError::TapeInvariant(op))
    }

    #[cfg(not(feature = "no-cuda"))]
    pub(super) fn maybe_pad_bf16_gemm_n(
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
    pub(super) fn matmul_device(
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
    pub(super) fn matmul_bt_device_f32_bf16(
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
    pub(super) fn matmul_device_f32_bf16(
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
    pub(super) fn fp8_block_scaled_as_bf16(
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
    pub(super) fn matmul_bt_device_f32_fp8_block_scaled(
        &self,
        a: &CudaSlice<f32>,
        a_shape: &[usize],
        storage: &CudaFp8BlockScaledStorage,
    ) -> Result<(CudaSlice<f32>, Vec<usize>)> {
        // Native FP8 DeepGEMM for the frozen-weight forward projection `a @ bᵀ`,
        // copying serve's `try_fp8_deepgemm_dense_batch` (quant_linear.rs:373):
        // quantize the bf16 activation to fp8 and run the fp8 tensor-core GEMM,
        // skipping the per-GEMM weight dequant the bf16 fallback pays. The
        // fallback stays for the flag-off / non-conforming / non-Hopper path.
        if crate::runtime_flags::fp8_native_gemm()
            && let Some(out) = self.matmul_bt_device_fp8_deepgemm(a, a_shape, storage)?
        {
            return Ok(out);
        }
        let (b_bf16, b_shape) = self.fp8_block_scaled_as_bf16(storage)?;
        self.matmul_bt_device_f32_bf16(a, a_shape, &b_bf16, &b_shape)
    }

    /// Native FP8 DeepGEMM path for `a @ bᵀ` with a frozen block-scaled weight.
    /// Returns `None` when the shape/hardware does not meet DeepGEMM's contract
    /// (128×128 blocks, `rows%8==0`, `cols%128==0`, SM≥9) so the caller falls
    /// back to the bf16 dequant GEMM.
    #[cfg(not(feature = "no-cuda"))]
    pub(super) fn matmul_bt_device_fp8_deepgemm(
        &self,
        a: &CudaSlice<f32>,
        a_shape: &[usize],
        storage: &CudaFp8BlockScaledStorage,
    ) -> Result<Option<(CudaSlice<f32>, Vec<usize>)>> {
        use cuda_kernels::tensor::cache_ptr_on;
        let (weight, scales, rows, cols, block_m, block_k) =
            self.cuda_fp8_block_scaled_storage(storage)?;
        if a_shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: "fp8 matmul_bt lhs must be rank-2",
                got: a_shape.len(),
            });
        }
        let m = a_shape[0];
        let (n, k) = (rows, cols);
        let shape_ok = block_m == 128
            && block_k == 128
            && n.is_multiple_of(8)
            && k.is_multiple_of(128)
            && self.fp8_deepgemm_sm_ok();
        if !shape_ok || m == 0 {
            return Ok(None);
        }

        let a_bf16 = self.local_f32_as_bf16(a, a.len())?; // reuses the bf16-GEMM cast
        let scale_stride_m = m.div_ceil(4) * 4; // TMA 4-row alignment of the activation scales
        let scale_cols = k.div_ceil(128);
        let input_fp8 = self
            .stream
            .alloc_zeros::<u8>(m * k)
            .map_err(|_| cuda_alloc_failed("fp8 deepgemm input", vec![m, k]))?;
        let input_scales = self
            .stream
            .alloc_zeros::<f32>(scale_stride_m * scale_cols)
            .map_err(|_| {
                cuda_alloc_failed("fp8 deepgemm scales", vec![scale_stride_m, scale_cols])
            })?;
        let out_bf16 = self
            .stream
            .alloc_zeros::<u16>(m * n)
            .map_err(|_| cuda_alloc_failed("fp8 deepgemm out", vec![m, n]))?;
        // Single-group quantize metadata: one dense "expert" covering all m rows.
        let active_experts = self
            .stream
            .clone_htod(&[0i32])
            .map_err(|_| AutogradError::TapeInvariant("fp8 deepgemm active_experts H2D failed"))?;
        let active_offsets = self
            .stream
            .clone_htod(&[0i32])
            .map_err(|_| AutogradError::TapeInvariant("fp8 deepgemm active_offsets H2D failed"))?;
        let m_i32 = i32::try_from(m)
            .map_err(|_| AutogradError::TapeInvariant("fp8 deepgemm m exceeds i32"))?;
        let active_counts = self
            .stream
            .clone_htod(&[m_i32])
            .map_err(|_| AutogradError::TapeInvariant("fp8 deepgemm active_counts H2D failed"))?;

        let s = &self.stream;
        // SAFETY: every ptr is a live device allocation sized to the dims passed.
        unsafe {
            cuda_kernels::moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                cache_ptr_on(&a_bf16, s).cast::<cuda_kernels::bf16>(),
                cache_ptr_on(&input_fp8, s),
                cache_ptr_on(&input_scales, s),
                cache_ptr_on(&active_experts, s),
                cache_ptr_on(&active_offsets, s),
                cache_ptr_on(&active_counts, s),
                1,
                m,
                k,
                scale_stride_m,
                s.cu_stream(),
            )
            .map_err(|_| AutogradError::TapeInvariant("fp8 deepgemm pack_quantize failed"))?;
            cuda_kernels::moe::dsv4_deepgemm_fp8_gemm_nt(
                cache_ptr_on(&input_fp8, s),
                cache_ptr_on(&input_scales, s),
                cache_ptr_on(weight, s),
                cache_ptr_on(scales, s),
                cache_ptr_on(&out_bf16, s).cast::<cuda_kernels::bf16>(),
                m,
                n,
                k,
                scale_stride_m,
                s.cu_stream(),
            )
            .map_err(|_| AutogradError::TapeInvariant("fp8 deepgemm gemm_nt failed"))?;
        }
        let out_f32 = self.import_local_bf16_as_f32(&out_bf16, m * n)?;
        Ok(Some((out_f32, vec![m, n])))
    }

    /// DeepGEMM FP8 needs Hopper+ (sm_90 / sm_100). Major ≥ 9.
    #[cfg(not(feature = "no-cuda"))]
    pub(super) fn fp8_deepgemm_sm_ok(&self) -> bool {
        use cudarc::driver::sys::CUdevice_attribute as Attr;
        self.stream
            .context()
            .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .is_ok_and(|v| v >= 9)
    }

    #[cfg(not(feature = "no-cuda"))]
    pub(super) fn matmul_device_f32_fp8_block_scaled(
        &self,
        a: &CudaSlice<f32>,
        a_shape: &[usize],
        storage: &CudaFp8BlockScaledStorage,
    ) -> Result<(CudaSlice<f32>, Vec<usize>)> {
        let (b_bf16, b_shape) = self.fp8_block_scaled_as_bf16(storage)?;
        self.matmul_device_f32_bf16(a, a_shape, &b_bf16, &b_shape)
    }
}

pub(super) fn cuda_matmul_bt(
    backend: &CudaBackend,
    a: &DeviceHandle,
    a_shape: &[usize],
    b: &DeviceHandle,
    b_shape: &[usize],
) -> Result<(DeviceHandle, Vec<usize>)> {
    let out_shape = matmul_bt_output_shape(a_shape, b_shape)?;
    let d_a_op = backend.f32_operand(a, "matmul_bt")?;
    let d_a = d_a_op.get();
    if let DeviceHandle::CudaBf16(storage) = b {
        let d_b = backend.cuda_bf16_storage_slice(storage)?;
        if d_a.len() != shape_size(a_shape) || d_b.len() != shape_size(b_shape) {
            return Err(AutogradError::TapeInvariant(
                "cuda backend bf16 matmul_bt handle size does not match shape",
            ));
        }
        let (c, out_shape) = backend.matmul_bt_device_f32_bf16(d_a, a_shape, d_b, b_shape)?;
        return Ok((DeviceHandle::Cuda(CudaStorage::new(c)), out_shape));
    }
    if let DeviceHandle::CudaFp8BlockScaled(storage) = b {
        let (weight, _, rows, cols, _, _) = backend.cuda_fp8_block_scaled_storage(storage)?;
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
            backend.matmul_bt_device_f32_fp8_block_scaled(d_a, a_shape, storage)?;
        return Ok((DeviceHandle::Cuda(CudaStorage::new(c)), out_shape));
    }

    let d_b = backend.cuda_slice(b, "matmul_bt")?;
    if d_a.len() != shape_size(a_shape) || d_b.len() != shape_size(b_shape) {
        return Err(AutogradError::TapeInvariant(
            "cuda backend matmul_bt handle size does not match shape",
        ));
    }
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    let mut c = backend
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
        backend
            .blas
            .gemm(cfg, d_b, d_a, &mut c)
            .map_err(|_| AutogradError::TapeInvariant("cuBLAS sgemm failed (matmul_bt)"))?;
    }
    Ok((DeviceHandle::Cuda(CudaStorage::new(c)), out_shape))
}

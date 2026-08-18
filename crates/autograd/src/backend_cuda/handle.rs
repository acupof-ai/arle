use super::*;

#[cfg(not(feature = "no-cuda"))]
pub(super) fn check_cuda_ffi(status: CUresult, label: &'static str) -> Result<()> {
    if status == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(AutogradError::TapeInvariant(Box::leak(
            format!("{label} failed with CUDA status {status:?}").into_boxed_str(),
        )))
    }
}

/// cuBLAS-backed matmul plus NVRTC-compiled point kernels. Holds an
/// `Arc<CudaStream>` + `CudaBlas` so the context lives as long as the backend;
/// safe to share across threads.
#[cfg(not(feature = "no-cuda"))]
pub(super) enum F32Operand<'a> {
    Borrowed(&'a CudaSlice<f32>),
    Imported(CudaSlice<f32>),
}

#[cfg(not(feature = "no-cuda"))]
impl F32Operand<'_> {
    pub(super) fn get(&self) -> &CudaSlice<f32> {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Imported(slice) => slice,
        }
    }
}

#[cfg(not(feature = "no-cuda"))]
pub(super) enum Bf16Operand<'a> {
    Borrowed(&'a CudaSlice<u16>),
    Quantized(CudaSlice<u16>),
}

#[cfg(not(feature = "no-cuda"))]
impl Bf16Operand<'_> {
    pub(super) fn get(&self) -> &CudaSlice<u16> {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Quantized(slice) => slice,
        }
    }
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_row_len(len: usize, rows: usize) -> Result<usize> {
    if rows == 0 || !len.is_multiple_of(rows) {
        return Err(AutogradError::TapeInvariant(
            "linear_attention batched dispatch len not divisible by batch",
        ));
    }
    Ok(len / rows)
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_copy_range<T: DeviceRepr>(
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
pub(super) fn cuda_concat_axis(
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
pub(super) fn cuda_concat_parts<T: DeviceRepr>(
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
pub(super) fn cuda_row_slice(
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
pub(super) fn cuda_concat_rows(
    backend: &CudaBackend,
    rows: &[&DeviceHandle],
) -> Result<DeviceHandle> {
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

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_readback_slice(
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
pub(super) fn shape_size(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_alloc_failed(op: &'static str, shape: Vec<usize>) -> AutogradError {
    let bytes = shape_size(&shape).saturating_mul(std::mem::size_of::<f32>());
    AutogradError::CudaAllocFailed { op, shape, bytes }
}

// Alloc failure: report driver code + live free/total to tell fragmentation
// from a sticky async fault (fails with GB free).
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_alloc_failed_rich(
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
pub(super) fn leak_err(msg: String) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(msg.into_boxed_str()))
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_download(
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

impl CudaBackend {
    #[cfg(not(feature = "no-cuda"))]
    #[track_caller]
    pub(super) fn upload_slice(&self, host: &[f32], shape: &[usize]) -> Result<CudaSlice<f32>> {
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
    pub(super) fn cuda_storage_slice<'a>(
        &self,
        storage: &'a CudaStorage,
    ) -> Result<&'a CudaSlice<f32>> {
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
    pub(super) fn upload_bf16_bits_slice(
        &self,
        host: &[u16],
        shape: &[usize],
    ) -> Result<CudaSlice<u16>> {
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
    pub(super) fn cuda_bf16_storage_slice<'a>(
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
    pub(super) fn upload_fp8_bytes_slice(
        &self,
        host: &[u8],
        shape: &[usize],
    ) -> Result<CudaSlice<u8>> {
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
    pub(super) fn cuda_fp8_block_scaled_storage<'a>(
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
    pub(super) fn copy_bf16_device_ptr_to_local(
        &self,
        src_device_ptr: u64,
        src_stream: u64,
        len: usize,
    ) -> Result<CudaSlice<u16>> {
        if len == 0 {
            return self.stream.alloc_zeros::<u16>(0).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (bf16 bridge)")
            });
        }

        let byte_count =
            len.checked_mul(std::mem::size_of::<u16>())
                .ok_or(AutogradError::TapeInvariant(
                    "bf16 bridge byte count overflow",
                ))?;

        if src_stream == 0 {
            // No source stream to order against: sync copy, then drain the
            // context so the foreign owner's `cuMemFreeAsync` cannot race it.
            let mut staging = self.stream.alloc_zeros::<u16>(len).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (bf16 bridge)")
            })?;
            let (dst_ptr, _dst_guard) = staging.device_ptr_mut(&self.stream);
            // SAFETY: dst spans byte_count bytes (`_dst_guard` held); the caller
            // guarantees src_device_ptr covers byte_count bytes until the sync.
            check_cuda_ffi(
                unsafe {
                    cuMemcpyDtoD_v2(
                        dst_ptr as CUdeviceptr,
                        src_device_ptr as CUdeviceptr,
                        byte_count,
                    )
                },
                "bf16 bridge D2D",
            )?;
            drop(_dst_guard);
            self.stream
                .context()
                .synchronize()
                .map_err(|_| AutogradError::TapeInvariant("cuda D2D bridge sync failed"))?;
            return Ok(staging);
        }

        // Stream-ordered path: the copy runs on the source stream (after the
        // producer's kernels), and this backend's stream waits on a completion
        // event. The source's later `cuMemFreeAsync` on the same stream is
        // ordered after the copy, so the foreign owner may free as soon as this
        // returns. Staging is uninitialized because the copy overwrites every
        // element; `alloc_zeros` would memset on this stream and race the copy.
        // SAFETY: `alloc` is unsafe in cudarc; the slice is fully written by the
        // D2D copy before any read.
        let mut staging = unsafe { self.stream.alloc::<u16>(len) }
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc failed (bf16 bridge)"))?;
        let (dst_ptr, _dst_guard) = staging.device_ptr_mut(&self.stream);
        // The allocation is stream-ordered on self.stream; the copy runs on
        // src_stream. Record an event after the alloc and make src_stream wait
        // for it, else the copy can race the allocation.
        let alloc_event = self
            .stream
            .record_event(None)
            .map_err(|_| AutogradError::TapeInvariant("bf16 bridge alloc event create failed"))?;
        // SAFETY: event and src_stream are valid in this (primary) context.
        check_cuda_ffi(
            unsafe { cuStreamWaitEvent(src_stream as CUstream, alloc_event.cu_event(), 0) },
            "bf16 bridge src stream wait alloc",
        )?;
        // SAFETY: dst spans byte_count bytes (`_dst_guard` held); the caller
        // guarantees src_device_ptr covers byte_count bytes and src_stream is a
        // valid stream in this context whose queued work produced the source.
        check_cuda_ffi(
            unsafe {
                cuMemcpyDtoDAsync_v2(
                    dst_ptr as CUdeviceptr,
                    src_device_ptr as CUdeviceptr,
                    byte_count,
                    src_stream as CUstream,
                )
            },
            "bf16 bridge D2D async",
        )?;
        // Record the copy's completion on the source stream, then make this
        // backend's stream wait for it. The event drops after the wait is
        // enqueued; the driver defers destruction until the wait completes.
        let event = self
            .stream
            .context()
            .new_event(None)
            .map_err(|_| AutogradError::TapeInvariant("bf16 bridge event create failed"))?;
        // SAFETY: event and src_stream are valid in this (primary) context.
        check_cuda_ffi(
            unsafe { cuEventRecord(event.cu_event(), src_stream as CUstream) },
            "bf16 bridge event record",
        )?;
        self.stream
            .wait(&event)
            .map_err(|_| AutogradError::TapeInvariant("bf16 bridge stream wait failed"))?;
        drop(_dst_guard);
        Ok(staging)
    }

    /// First `len` elements of `buf` (identity when unpadded — no copy).
    #[cfg(not(feature = "no-cuda"))]
    pub(super) fn f32_prefix(&self, buf: CudaSlice<f32>, len: usize) -> Result<CudaSlice<f32>> {
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
    pub(super) fn import_local_bf16_as_f32(
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
    pub(super) fn local_f32_as_bf16(
        &self,
        src: &CudaSlice<f32>,
        len: usize,
    ) -> Result<CudaSlice<u16>> {
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
    pub(super) fn cuda_slice<'a>(
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
            DeviceHandle::CudaFp4E2M1Group(_) => Err(AutogradError::TapeInvariant(match op {
                "matmul" => "cuda backend cannot matmul an nvfp4 handle on this f32-only path",
                "matmul_bt" => {
                    "cuda backend cannot use an nvfp4 handle as lhs on this matmul_bt path"
                }
                _ => "cuda backend cannot operate on an nvfp4 handle on this f32-only path",
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
    pub(super) fn cuda_bf16_slice<'a>(
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
    pub(super) fn tape_bf16(&self) -> bool {
        self.tape_dtype.load(Ordering::Relaxed) == TapeDtype::Bf16 as u8
    }

    /// f32 view of a handle: borrows f32 storage, imports (exact widen) bf16.
    #[cfg(not(feature = "no-cuda"))]
    pub(super) fn f32_operand<'a>(
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
    pub(super) fn f32_operand_tape_quantized<'a>(
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
    pub(super) fn bf16_operand<'a>(
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
    pub(super) fn validate_cuda_handle_kind(&self, handle: &DeviceHandle) -> Result<()> {
        match handle {
            DeviceHandle::Cpu(_)
            | DeviceHandle::Cuda(_)
            | DeviceHandle::CudaBf16(_)
            | DeviceHandle::CudaFp8BlockScaled(_)
            | DeviceHandle::CudaFp4E2M1Group(_) => Ok(()),
            #[cfg(feature = "metal")]
            DeviceHandle::Metal(_) => Err(AutogradError::TapeInvariant(
                "cuda backend cannot evaluate a metal device handle",
            )),
        }
    }
}

pub(super) fn cuda_quantize_frozen_to_bf16(
    backend: &CudaBackend,
    handle: &DeviceHandle,
    shape: &[usize],
) -> Result<DeviceHandle> {
    // Idempotent: non-f32-Cuda handles pass through unchanged.
    let DeviceHandle::Cuda(_) = handle else {
        return Ok(handle.clone());
    };
    let src = backend.cuda_slice(handle, "quantize_frozen_to_bf16")?;
    let bits = backend.local_f32_as_bf16(src, shape_size(shape))?;
    Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(bits)))
}

pub(super) fn cuda_upload_fp8_block_scaled(
    backend: &CudaBackend,
    weight: &[u8],
    scales: &[f32],
    shape: &[usize],
    block_m: usize,
    block_k: usize,
) -> Result<DeviceHandle> {
    validate_fp8_block_scaled(weight, scales, shape, block_m, block_k)?;
    Ok(DeviceHandle::CudaFp8BlockScaled(
        CudaFp8BlockScaledStorage::new(
            backend.upload_fp8_bytes_slice(weight, shape)?,
            backend.upload_slice(scales, &[scales.len()])?,
            shape[0],
            shape[1],
            block_m,
            block_k,
        ),
    ))
}

pub(super) fn cuda_upload_fp4_e2m1_group(
    backend: &CudaBackend,
    weight: &[u8],
    scales: &[u8],
    global_scale: f32,
    shape: &[usize],
    group_size: usize,
    scale_cols: usize,
) -> Result<DeviceHandle> {
    if shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: shape.len(),
        });
    }
    let (rows, cols) = (shape[0], shape[1]);
    if group_size == 0 || scale_cols == 0 {
        return Err(AutogradError::TapeInvariant(
            "nvfp4 group_size/scale_cols must be non-zero",
        ));
    }
    if cols % 2 != 0 {
        return Err(AutogradError::TapeInvariant(
            "nvfp4 cols must be even (two weights per byte)",
        ));
    }
    // The dequant kernel indexes scales[row * scale_cols + col / group_size]
    // without a bound check, so this relation is the kernel's precondition.
    if scale_cols * group_size != cols {
        return Err(AutogradError::TapeInvariant(
            "nvfp4 scale_cols * group_size must equal cols",
        ));
    }
    // Packed: two 4-bit weights per byte, so the buffer is half the logical size.
    let packed_len = rows * (cols / 2);
    if weight.len() != packed_len {
        return Err(AutogradError::DataLengthMismatch {
            len: weight.len(),
            shape: shape.to_vec(),
            size: packed_len,
        });
    }
    if scales.len() != rows * scale_cols {
        return Err(AutogradError::DataLengthMismatch {
            len: scales.len(),
            shape: vec![rows, scale_cols],
            size: rows * scale_cols,
        });
    }
    Ok(DeviceHandle::CudaFp4E2M1Group(
        CudaFp4E2M1GroupStorage::new(
            backend.upload_fp8_bytes_slice(weight, &[packed_len])?,
            backend.upload_fp8_bytes_slice(scales, &[scales.len()])?,
            backend.upload_slice(&[global_scale], &[1])?,
            rows,
            cols,
            group_size,
            scale_cols,
        ),
    ))
}

pub(super) fn cuda_import_bf16_device_ptr_as_f32(
    backend: &CudaBackend,
    src_device_ptr: u64,
    src_stream: u64,
    len: usize,
    shape: &[usize],
) -> Result<DeviceHandle> {
    if shape_size(shape) != len {
        return Err(AutogradError::DataLengthMismatch {
            len,
            shape: shape.to_vec(),
            size: shape_size(shape),
        });
    }
    let staging = backend.copy_bf16_device_ptr_to_local(src_device_ptr, src_stream, len)?;
    let f32_slice = backend.import_local_bf16_as_f32(&staging, len)?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(f32_slice)))
}

pub(super) fn cuda_import_fp8_block_scaled_device_ptr(
    backend: &CudaBackend,
    weight_device_ptr: u64,
    scale_device_ptr: u64,
    shape: &[usize],
    block_m: usize,
    block_k: usize,
) -> Result<DeviceHandle> {
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
    // (`weight.context() == backend.stream.context()`) passes (both engines
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
        backend
            .stream
            .upgrade_device_ptr::<u8>(weight_device_ptr as CUdeviceptr, weight_len)
    };
    // SAFETY: same caller contract as the weight slice — scale_device_ptr is resident
    // for scale_len f32s and never freed by the borrowed handle.
    let scale_slice: CudaSlice<f32> = unsafe {
        backend
            .stream
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

pub(super) fn cuda_import_bf16_device_ptr(
    backend: &CudaBackend,
    device_ptr: u64,
    shape: &[usize],
) -> Result<DeviceHandle> {
    if shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: shape.len(),
        });
    }
    let rows = shape[0];
    let cols = shape[1];
    let len = rows
        .checked_mul(cols)
        .ok_or(AutogradError::TapeInvariant("bf16 import len overflow"))?;

    // NON-OWNING view over the foreign engine's resident BF16 base bytes.
    // `upgrade_device_ptr` wraps the raw `CUdeviceptr` in a `CudaSlice`
    // bound to THIS backend's stream/primary context. The
    // `CudaBf16Storage::new_borrowed` handle leaks the slice on drop
    // instead of freeing — the infer engine owns the bytes.
    //
    // Safety: the caller (post-LoRA-merge frozen-base refresh) guarantees
    // `device_ptr` is a valid resident allocation on this ordinal of at
    // least `len` u16 elements, kept resident for the handle's lifetime.
    let slice: CudaSlice<u16> = unsafe {
        backend
            .stream
            .upgrade_device_ptr::<u16>(device_ptr as CUdeviceptr, len)
    };
    Ok(DeviceHandle::CudaBf16(
        crate::backend::CudaBf16Storage::new_borrowed(slice),
    ))
}

pub(super) fn cuda_readback(backend: &CudaBackend, handle: &DeviceHandle) -> Result<Vec<f32>> {
    match handle {
        DeviceHandle::Cpu(data) => Ok(data.clone()),
        DeviceHandle::Cuda(storage) => {
            let slice = backend.cuda_storage_slice(storage)?;
            let mut host = vec![0.0f32; slice.len()];
            backend
                .stream
                .memcpy_dtoh(slice, &mut host)
                .map_err(|_| AutogradError::TapeInvariant("cuda dtoh copy failed"))?;
            // cudarc 0.18 routes memcpy_dtoh through cuMemcpyDtoHAsync_v2
            // (async DMA); callers do not always eval() first, so this
            // single host fence is required. PENDING REMOTE CUDA VERIFICATION.
            backend
                .stream
                .synchronize()
                .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))?;
            Ok(host)
        }
        DeviceHandle::CudaBf16(storage) => {
            let slice = backend.cuda_bf16_storage_slice(storage)?;
            let mut host = vec![0u16; slice.len()];
            backend
                .stream
                .memcpy_dtoh(slice, &mut host)
                .map_err(|_| AutogradError::TapeInvariant("cuda bf16 dtoh copy failed"))?;
            backend
                .stream
                .synchronize()
                .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))?;
            Ok(host
                .into_iter()
                .map(crate::backend::bf16_bits_to_f32)
                .collect())
        }
        DeviceHandle::CudaFp8BlockScaled(storage) => {
            let (weight, scales, rows, cols, block_m, block_k) =
                backend.cuda_fp8_block_scaled_storage(storage)?;
            let mut host_weight = vec![0u8; weight.len()];
            let mut host_scales = vec![0.0f32; scales.len()];
            backend
                .stream
                .memcpy_dtoh(weight, &mut host_weight)
                .map_err(|_| AutogradError::TapeInvariant("cuda fp8 dtoh copy failed"))?;
            backend
                .stream
                .memcpy_dtoh(scales, &mut host_scales)
                .map_err(|_| AutogradError::TapeInvariant("cuda fp8 scales dtoh copy failed"))?;
            backend
                .stream
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
        DeviceHandle::CudaFp4E2M1Group(storage) => {
            let weight = storage.weight();
            let scales = storage.scales();
            let global = storage.global_scale();
            let mut host_weight = vec![0u8; weight.len()];
            let mut host_scales = vec![0u8; scales.len()];
            let mut host_global = vec![0.0f32; global.len()];
            backend
                .stream
                .memcpy_dtoh(weight, &mut host_weight)
                .map_err(|_| AutogradError::TapeInvariant("cuda nvfp4 dtoh copy failed"))?;
            backend
                .stream
                .memcpy_dtoh(scales, &mut host_scales)
                .map_err(|_| AutogradError::TapeInvariant("cuda nvfp4 scales dtoh copy failed"))?;
            backend
                .stream
                .memcpy_dtoh(global, &mut host_global)
                .map_err(|_| {
                    AutogradError::TapeInvariant("cuda nvfp4 global scale dtoh copy failed")
                })?;
            backend
                .stream
                .synchronize()
                .map_err(|_| AutogradError::TapeInvariant("cuda synchronize failed"))?;
            let global_scale = *host_global.first().ok_or(AutogradError::TapeInvariant(
                "cuda nvfp4 global scale empty",
            ))?;
            crate::backend::dequantize_fp4_e2m1_group_host(
                &host_weight,
                &host_scales,
                global_scale,
                &[storage.rows(), storage.cols()],
                storage.group_size(),
                storage.scale_cols(),
            )
        }
        #[cfg(feature = "metal")]
        DeviceHandle::Metal(_) => Err(AutogradError::TapeInvariant(
            "cuda backend cannot read back a metal device handle",
        )),
    }
}

pub(super) fn cuda_readback_into(
    backend: &CudaBackend,
    handle: &DeviceHandle,
    dst: &mut [f32],
) -> Result<()> {
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
            let slice = backend.cuda_storage_slice(storage)?;
            if slice.len() != dst.len() {
                return Err(AutogradError::DataLengthMismatch {
                    len: slice.len(),
                    shape: vec![dst.len()],
                    size: dst.len(),
                });
            }
            backend.stream.memcpy_dtoh(slice, dst).map_err(|_| {
                AutogradError::TapeInvariant("cuda dtoh copy failed (readback_into)")
            })?;
            backend.stream.synchronize().map_err(|_| {
                AutogradError::TapeInvariant("cuda synchronize failed (readback_into)")
            })
        }
        _ => {
            let src = backend.readback(handle)?;
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

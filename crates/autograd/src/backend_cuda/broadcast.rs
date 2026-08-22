use super::*;

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_add_broadcast(
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, total)
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
pub(super) fn cuda_add_broadcast_device(
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
        let mut d_out = alloc_zeros_retry::<u16>(backend, total)
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, total)
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
pub(super) fn cuda_broadcast_expand_device(
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
pub(super) fn cuda_add_broadcast_backward_device(
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
        let mut d_grad = alloc_zeros_retry::<u16>(backend, b_total).map_err(|_| {
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
    let mut d_grad = alloc_zeros_retry::<f32>(backend, b_total).map_err(|_| {
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

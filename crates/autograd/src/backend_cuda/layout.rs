use super::*;

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_write_scalar_at(
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
pub(super) fn cuda_transpose_axes_swap_device(
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
pub(super) fn cuda_slice_device(
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
pub(super) fn cuda_concat_axis2_device(
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
pub(super) fn cuda_slice_backward_device(
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
pub(super) fn cuda_write_slice_device(
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

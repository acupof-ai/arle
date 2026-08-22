use super::*;

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_softmax_like(
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, expected)
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

// Device-resident sibling of `cuda_softmax_like`; no `synchronize()` — the
// caller owns the terminal eval. Serves both softmax and log_softmax via
// `kernel_name`.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_softmax_like_device(
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
        let mut d_out = alloc_zeros_retry::<u16>(backend, expected)
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

    let mut d_out = alloc_zeros_retry::<f32>(backend, expected)
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

// Device-resident log_softmax backward; returned unevaluated for the tape's
// terminal eval. Same reduce shape as `softmax_last_axis_f32`.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_log_softmax_last_axis_backward(
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
        let mut d_grad = alloc_zeros_retry::<u16>(backend, expected).map_err(|_| {
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

    let mut d_grad = alloc_zeros_retry::<f32>(backend, expected).map_err(|e| {
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
pub(super) fn cuda_softmax_last_axis_backward(
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
        let mut d_grad = alloc_zeros_retry::<u16>(backend, expected)
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

    let mut d_grad = alloc_zeros_retry::<f32>(backend, expected)
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

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_argmax_last_dim(
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, rows)
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

// Device-resident full reduction; no host transfer, no `synchronize()` —
// the caller's terminal eval forces the passes. Sibling of the host-reduce
// path `sum_all` takes.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_sum_all_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    size: usize,
) -> Result<DeviceHandle> {
    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;

    // size == 0 → empty sum is 0.0; size == 1 → already a scalar, copy as-is
    // through one trivial reduce so the returned buffer is freshly owned.
    if size == 0 {
        let d_out = alloc_zeros_retry::<f32>(backend, 1)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sum_all empty)"))?;
        return Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)));
    }

    // First pass reads the borrowed input slice; later passes read the
    // previous pass's owned partial buffer. `function()` is re-fetched per
    // launch (a cheap HashMap lookup) so each `&CudaFunction` borrow is
    // scoped to a single `launch_rows` call — mirrors `cuda_sum_squares`.
    let in_slice = backend.cuda_slice(x, "sum_all")?;
    let mut n = size;
    let mut blocks = n.div_ceil(BLOCK as usize);
    let n_i32 = i32::try_from(n)
        .map_err(|_| AutogradError::TapeInvariant("cuda sum_all size exceeds i32"))?;
    let mut current = alloc_zeros_retry::<f32>(backend, blocks)
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

    while blocks > 1 {
        n = blocks;
        blocks = n.div_ceil(BLOCK as usize);
        let pass_n = i32::try_from(n)
            .map_err(|_| AutogradError::TapeInvariant("cuda sum_all partials exceed i32"))?;
        let mut next = alloc_zeros_retry::<f32>(backend, blocks)
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
pub(super) fn cuda_reduce_last_axis(
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, rows)
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

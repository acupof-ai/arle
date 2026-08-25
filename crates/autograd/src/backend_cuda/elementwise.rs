use super::*;

// Separate kernel registration (functionally identical to the forward
// `mul_scalar_f32`) so the nsys audit trail matches the autograd op name.
// Returned handle is unevaluated — terminal `eval` is the caller's.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_mul_scalar_backward_device(
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

    let mut d_out = alloc_zeros_retry::<f32>(backend, size).map_err(|_| {
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

// Upstream is a rank-0 device scalar. Returned handle is unevaluated.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_mean_backward_device(
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, elem_count).map_err(|_| {
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
pub(super) fn cuda_sum_backward_device(
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, elem_count).map_err(|e| {
        let (free, total) = backend.mem_get_info().unwrap_or((0, 0));
        eprintln!(
            "[autograd] alloc_zeros {elem_count} x f32 failed (sum_backward_device, \
             shape {output_shape:?}): {e}; free={}MiB total={}MiB",
            free >> 20,
            total >> 20
        );
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

// Returned handle is unevaluated — terminal `eval` is the caller's.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_add_into_device(
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
        let mut d_out = alloc_zeros_retry::<u16>(backend, size)
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

    let mut d_out = alloc_zeros_retry::<f32>(backend, size)
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
pub(super) fn cuda_accumulate_into_device(
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
pub(super) fn cuda_unary_1d(
    backend: &CudaBackend,
    a: &[f32],
    kernel_name: &'static str,
) -> Result<Vec<f32>> {
    let n_usize = a.len();
    let d_in = backend
        .stream
        .clone_htod(a)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = alloc_zeros_retry::<f32>(backend, n_usize)
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
pub(super) fn cuda_scalar_1d(
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, n_usize)
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
pub(super) fn cuda_binary_1d(
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, n_usize)
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
pub(super) fn cuda_unary_1d_device(
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
        let mut d_out = alloc_zeros_retry::<u16>(backend, size)
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, size)
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
pub(super) fn cuda_scalar_1d_device(
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
        let mut d_out = alloc_zeros_retry::<u16>(backend, size)
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, size)
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
pub(super) fn cuda_binary_1d_device(
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
        let mut d_out = alloc_zeros_retry::<u16>(backend, size)
            .map_err(|e| cuda_alloc_failed_rich(backend, op_label, size * 2, &e))?;
        let n = size as u64;
        let func = backend.kernels.function_for(kernel_name, TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, size, |mut builder| {
            builder.arg(&mut d_out).arg(d_a).arg(d_b).arg(&n);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }
    let d_a_op = backend.f32_operand(a, op_label)?;
    let d_b_op = backend.f32_operand(b, op_label)?;
    let d_a = d_a_op.get();
    let d_b = d_b_op.get();
    if d_a.len() != size || d_b.len() != size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_a.len().min(d_b.len()),
            shape: shape.to_vec(),
            size,
        });
    }
    let mut d_out = alloc_zeros_retry::<f32>(backend, size)
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
pub(super) fn cuda_elementwise_backward_with_saved(
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
        let mut d_out = alloc_zeros_retry::<u16>(backend, size)
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
    let mut d_out = alloc_zeros_retry::<f32>(backend, size)
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
pub(super) fn cuda_silu_backward_device(
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
pub(super) fn cuda_sigmoid_backward_device(
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
pub(super) fn cuda_abs_backward_device(
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
        "abs_backward_f32",
        "abs_backward_device",
    )
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_mul_backward_device(
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
        let mut d_out = alloc_zeros_retry::<f32>(backend, size).map_err(|_| {
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
        let mut d_out = alloc_zeros_retry::<f32>(backend, size).map_err(|_| {
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

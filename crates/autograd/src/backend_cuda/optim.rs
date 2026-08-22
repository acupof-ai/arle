use super::*;

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_adamw_step(
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
            // `PushKernelArg<&CudaSlice<T>>` passes the raw CUdeviceptr; the
            // kernel's mutable `float*` params update the buffers in place.
            // Deliberately no `CudaSlice::clone()` — that is a device copy.
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

    // Backend::adamw_step eval contract: return unevaluated handles. These
    // are Arc clones of the same in-place buffers, not fresh allocations.
    Ok((param.clone(), m.clone(), v.clone()))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_adamw_step_device(
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

    // No `clone_htod(grad)`: the grad already lives on-device; pass the
    // existing `&CudaSlice<f32>` straight into the kernel.
    let n = i32::try_from(size)
        .map_err(|_| AutogradError::TapeInvariant("cuda adamw length exceeds i32"))?;
    launch_1d(
        &backend.stream,
        backend.kernels.function("adamw_step_f32")?,
        size,
        |mut builder| {
            // In-place update: see `cuda_adamw_step` above. Borrowed slices
            // avoid `CudaSlice::clone()`, a DtoD allocation+copy in cudarc,
            // not an Arc ref-count bump.
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

    // Eval contract: return unevaluated; the caller batches the terminal
    // `stream.synchronize()` for the whole optimizer step.
    Ok((param.clone(), m.clone(), v.clone()))
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_sum_squares(
    backend: &CudaBackend,
    x: &DeviceHandle,
    shape: &[usize],
) -> Result<f64> {
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
    let mut d_partial = alloc_zeros_retry::<f64>(backend, blocks)
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
pub(super) fn cuda_clip_grad_norm_device(
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
    let mut d_partial = alloc_zeros_retry::<f64>(backend, total_chunks)
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
    let mut out_slices = grads
        .iter()
        .map(|(_, shape)| {
            backend
                .stream
                .alloc_zeros::<f32>(shape_size(shape))
                .map_err(|_| {
                    AutogradError::TapeInvariant("cuda alloc_zeros failed (grad_clip scaled)")
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let (out_ptrs, out_guards): (Vec<_>, Vec<_>) = out_slices
        .iter_mut()
        .map(|out| out.device_ptr_mut(&backend.stream))
        .unzip();
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

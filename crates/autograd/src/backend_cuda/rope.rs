use super::*;

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_rope(
    backend: &CudaBackend,
    x: &[f32],
    x_shape: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> Result<Vec<f32>> {
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }
    let batch = x_shape[0];
    let heads = x_shape[1];
    let seq = x_shape[2];
    let head_dim = x_shape[3];
    if !head_dim.is_multiple_of(2) {
        return Err(AutogradError::InvalidRank {
            expected: "even head dim",
            got: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    let total = batch * heads * seq * head_dim;
    if x.len() != total {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![total],
            got: vec![x.len()],
        });
    }
    // Partial rotary (cos rows = rotary_dim/2 ≤ head_dim/2): rotate the leading
    // segment, pass the tail through — mirrors `cpu_rope_forward`.
    let rot_half = cos.len().checked_div(seq).unwrap_or(0);
    if cos.len() != sin.len() || rot_half == 0 || rot_half > half_dim || cos.len() != seq * rot_half
    {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![seq * half_dim],
            got: vec![cos.len().min(sin.len())],
        });
    }

    let d_x = backend.upload_slice(x, x_shape)?;
    let d_cos = backend
        .stream
        .clone_htod(cos)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let d_sin = backend
        .stream
        .clone_htod(sin)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope batch exceeds i32"))?;
    let heads_i = i32::try_from(heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope heads exceeds i32"))?;
    let seq_i = i32::try_from(seq)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope seq exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope head_dim exceeds i32"))?;
    let rot_half_i = i32::try_from(rot_half)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope rot_half exceeds i32"))?;

    let rows = batch * heads * seq;
    let block = std::cmp::min(half_dim, 256) as u32;
    let block = block.max(1);
    launch_rows(
        &backend.stream,
        backend.kernels.function("rope_f32")?,
        rows,
        block,
        0,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_x)
                .arg(&d_cos)
                .arg(&d_sin)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&rot_half_i);
            builder
        },
    )?;
    cuda_download(backend, &d_out, total)
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_rope_device(
    backend: &CudaBackend,
    x: &DeviceHandle,
    x_shape: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> Result<DeviceHandle> {
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }
    let batch = x_shape[0];
    let heads = x_shape[1];
    let seq = x_shape[2];
    let head_dim = x_shape[3];
    if !head_dim.is_multiple_of(2) {
        return Err(AutogradError::InvalidRank {
            expected: "even head dim",
            got: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    let total = batch * heads * seq * head_dim;
    // Partial rotary: see `cuda_rope_forward`.
    let rot_half = cos.len().checked_div(seq).unwrap_or(0);
    if cos.len() != sin.len() || rot_half == 0 || rot_half > half_dim || cos.len() != seq * rot_half
    {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![seq * half_dim],
            got: vec![cos.len().min(sin.len())],
        });
    }

    let d_cos = backend
        .stream
        .clone_htod(cos)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (rope cos)"))?;
    let d_sin = backend
        .stream
        .clone_htod(sin)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (rope sin)"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope batch exceeds i32"))?;
    let heads_i = i32::try_from(heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope heads exceeds i32"))?;
    let seq_i = i32::try_from(seq)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope seq exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope head_dim exceeds i32"))?;
    let rot_half_i = i32::try_from(rot_half)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope rot_half exceeds i32"))?;

    let rows = batch * heads * seq;
    let block = std::cmp::min(half_dim, 256) as u32;
    let block = block.max(1);
    if let DeviceHandle::CudaBf16(storage) = x {
        let d_x = backend.cuda_bf16_storage_slice(storage)?;
        if d_x.len() != total {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![total],
                got: vec![d_x.len()],
            });
        }
        let mut d_out = backend
            .stream
            .alloc_zeros::<u16>(total)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (rope)"))?;
        let func = backend.kernels.function_for("rope_f32", TapeDtype::Bf16)?;
        launch_rows(&backend.stream, &func, rows, block, 0, |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_x)
                .arg(&d_cos)
                .arg(&d_sin)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&rot_half_i);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }
    let d_x = backend.cuda_slice(x, "rope")?;
    if d_x.len() != total {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![total],
            got: vec![d_x.len()],
        });
    }
    let mut d_out = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (rope)"))?;
    launch_rows(
        &backend.stream,
        backend.kernels.function("rope_f32")?,
        rows,
        block,
        0,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_x)
                .arg(&d_cos)
                .arg(&d_sin)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&rot_half_i);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

// Device-resident backward for `rope`. Same launch shape as
// `cuda_rope` (one block per (batch, head, token); block=min(half_dim,256)).
// Only difference vs the forward kernel is the inlined `sin -> -sin` sign
// flip — `cpu_rope_backward` does the equivalent via a host
// `neg_forward(sin) → cpu_rope_forward` chain. cos/sin upload fresh every
// call (tiny: `[seq, head_dim/2]` per call).
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_rope_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    x_shape: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> Result<DeviceHandle> {
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }
    let batch = x_shape[0];
    let heads = x_shape[1];
    let seq = x_shape[2];
    let head_dim = x_shape[3];
    if !head_dim.is_multiple_of(2) {
        return Err(AutogradError::InvalidRank {
            expected: "even head dim",
            got: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    let total = batch * heads * seq * head_dim;
    // Partial rotary: see `cuda_rope_forward`.
    let rot_half = cos.len().checked_div(seq).unwrap_or(0);
    if cos.len() != sin.len() || rot_half == 0 || rot_half > half_dim || cos.len() != seq * rot_half
    {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![seq * half_dim],
            got: vec![cos.len().min(sin.len())],
        });
    }

    let d_cos = backend
        .stream
        .clone_htod(cos)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (rope_backward cos)"))?;
    let d_sin = backend
        .stream
        .clone_htod(sin)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (rope_backward sin)"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward batch exceeds i32"))?;
    let heads_i = i32::try_from(heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward heads exceeds i32"))?;
    let seq_i = i32::try_from(seq)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward seq exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward head_dim exceeds i32"))?;
    let rot_half_i = i32::try_from(rot_half)
        .map_err(|_| AutogradError::TapeInvariant("cuda rope_backward rot_half exceeds i32"))?;

    let rows = batch * heads * seq;
    let block = std::cmp::min(half_dim, 256) as u32;
    let block = block.max(1);
    if let DeviceHandle::CudaBf16(storage) = upstream {
        let d_up = backend.cuda_bf16_storage_slice(storage)?;
        if d_up.len() != total {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![total],
                got: vec![d_up.len()],
            });
        }
        let mut d_grad = backend
            .stream
            .alloc_zeros::<u16>(total)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (rope_backward)"))?;
        let func = backend
            .kernels
            .function_for("rope_backward_f32", TapeDtype::Bf16)?;
        launch_rows(&backend.stream, &func, rows, block, 0, |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_cos)
                .arg(&d_sin)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&rot_half_i);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_grad)));
    }
    let d_up = backend.cuda_slice(upstream, "rope_backward_device")?;
    if d_up.len() != total {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![total],
            got: vec![d_up.len()],
        });
    }
    let mut d_grad = backend
        .stream
        .alloc_zeros::<f32>(total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (rope_backward)"))?;
    launch_rows(
        &backend.stream,
        backend.kernels.function("rope_backward_f32")?,
        rows,
        block,
        0,
        |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_cos)
                .arg(&d_sin)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&rot_half_i);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
}

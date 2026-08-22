use super::*;

// Sibling of `cuda_gather_last_dim` over a borrowed device slice; no
// `synchronize()` — the caller owns the terminal eval.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_gather_last_dim_device(
    backend: &CudaBackend,
    src: &DeviceHandle,
    src_shape: &[usize],
    ids: &[i32],
) -> Result<DeviceHandle> {
    if src_shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }
    let vocab = *src_shape.last().expect("non-empty shape above");
    let prefix: usize = src_shape[..src_shape.len() - 1]
        .iter()
        .product::<usize>()
        .max(1);
    let expected: usize = src_shape.iter().product();
    let d_src = backend.cuda_slice(src, "gather_last_dim")?;
    if d_src.len() != expected {
        return Err(AutogradError::DataLengthMismatch {
            len: d_src.len(),
            shape: src_shape.to_vec(),
            size: expected,
        });
    }
    if ids.len() != prefix {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix,
            got: ids.len(),
        });
    }
    let d_ids = backend
        .stream
        .clone_htod(ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = alloc_zeros_retry::<f32>(backend, prefix)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    let n_i32 = i32::try_from(prefix)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather n exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather vocab exceeds i32"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("gather_last_dim_f32")?,
        prefix,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(d_src)
                .arg(&d_ids)
                .arg(&n_i32)
                .arg(&vocab_i32);
            builder
        },
    )?;

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

// Device-resident backward for gather_last_dim; no `synchronize()` —
// terminal eval is the caller's.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_gather_last_dim_backward(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    indices: &[i32],
    src_shape: &[usize],
) -> Result<DeviceHandle> {
    if src_shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }
    let vocab = *src_shape.last().expect("non-empty shape above");
    let prefix: usize = src_shape[..src_shape.len() - 1]
        .iter()
        .product::<usize>()
        .max(1);
    let total = prefix * vocab;
    let d_up = backend.cuda_slice(upstream, "gather_last_dim_backward")?;
    if d_up.len() != prefix {
        return Err(AutogradError::DataLengthMismatch {
            len: d_up.len(),
            shape: src_shape[..src_shape.len() - 1].to_vec(),
            size: prefix,
        });
    }
    if indices.len() != prefix {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix,
            got: indices.len(),
        });
    }
    let d_ids = backend
        .stream
        .clone_htod(indices)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (gather_bwd ids)"))?;
    // alloc_zeros gives the zero-fill for free — the kernel only writes the
    // single (row, ids[row]) slot per prefix row.
    let mut d_grad = alloc_zeros_retry::<f32>(backend, total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (gather_bwd grad)"))?;

    let prefix_i32 = i32::try_from(prefix)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather_bwd prefix exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather_bwd vocab exceeds i32"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("gather_last_dim_backward_f32")?,
        prefix,
        |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_ids)
                .arg(&prefix_i32)
                .arg(&vocab_i32);
            builder
        },
    )?;

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_gather_last_dim(
    backend: &CudaBackend,
    src: &[f32],
    src_shape: &[usize],
    ids: &[i32],
) -> Result<Vec<f32>> {
    if src_shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }
    let vocab = *src_shape.last().expect("non-empty shape above");
    let prefix: usize = src_shape[..src_shape.len() - 1]
        .iter()
        .product::<usize>()
        .max(1);
    let expected: usize = src_shape.iter().product();
    if src.len() != expected {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![src.len()],
        });
    }
    if ids.len() != prefix {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix,
            got: ids.len(),
        });
    }
    let d_src = backend.upload_slice(src, src_shape)?;
    let d_ids = backend
        .stream
        .clone_htod(ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = alloc_zeros_retry::<f32>(backend, prefix)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    let n_i32 = i32::try_from(prefix)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather n exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda gather vocab exceeds i32"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("gather_last_dim_f32")?,
        prefix,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_src)
                .arg(&d_ids)
                .arg(&n_i32)
                .arg(&vocab_i32);
            builder
        },
    )?;
    cuda_download(backend, &d_out, prefix)
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_scatter_add_rows(
    backend: &CudaBackend,
    upstream: &[f32],
    prefix_rows: usize,
    feature_dim: usize,
    indices: &[i32],
    vocab: usize,
) -> Result<Vec<f32>> {
    let expected_upstream = prefix_rows * feature_dim;
    if upstream.len() != expected_upstream {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected_upstream],
            got: vec![upstream.len()],
        });
    }
    if indices.len() != prefix_rows {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix_rows,
            got: indices.len(),
        });
    }
    let out_len = vocab * feature_dim;
    // Zero-initialize: the kernel only adds into the accumulator.
    let mut d_out = alloc_zeros_retry::<f32>(backend, out_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;
    if prefix_rows == 0 || feature_dim == 0 {
        return cuda_download(backend, &d_out, out_len);
    }
    let d_upstream = backend
        .stream
        .clone_htod(upstream)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let d_idx = backend
        .stream
        .clone_htod(indices)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;

    let prefix_i32 = i32::try_from(prefix_rows)
        .map_err(|_| AutogradError::TapeInvariant("cuda scatter_add prefix_rows exceeds i32"))?;
    let feature_i32 = i32::try_from(feature_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda scatter_add feature_dim exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda scatter_add vocab exceeds i32"))?;

    let block = std::cmp::min(feature_dim, 256) as u32;
    let block = block.max(1);
    launch_rows(
        &backend.stream,
        backend.kernels.function("scatter_add_rows_f32")?,
        prefix_rows,
        block,
        0,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_upstream)
                .arg(&d_idx)
                .arg(&prefix_i32)
                .arg(&feature_i32)
                .arg(&vocab_i32);
            builder
        },
    )?;
    cuda_download(backend, &d_out, out_len)
}

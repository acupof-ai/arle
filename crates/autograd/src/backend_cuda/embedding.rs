use super::*;

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_embedding(
    backend: &CudaBackend,
    weight: &[f32],
    vocab: usize,
    dim: usize,
    ids: &[i32],
) -> Result<Vec<f32>> {
    if weight.len() != vocab * dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![vocab * dim],
            got: vec![weight.len()],
        });
    }
    let n_ids = ids.len();
    let out_len = n_ids * dim;
    let d_w = backend
        .stream
        .clone_htod(weight)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let d_ids = backend
        .stream
        .clone_htod(ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed"))?;
    let mut d_out = alloc_zeros_retry::<f32>(backend, out_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed"))?;

    let n_i32 = i32::try_from(n_ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding n_ids exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding vocab exceeds i32"))?;
    let dim_i32 = i32::try_from(dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding dim exceeds i32"))?;

    const BLOCK: u32 = 256;
    launch_rows(
        &backend.stream,
        backend.kernels.function("embedding_f32")?,
        n_ids,
        BLOCK,
        0,
        |mut builder| {
            builder
                .arg(&mut d_out)
                .arg(&d_w)
                .arg(&d_ids)
                .arg(&n_i32)
                .arg(&vocab_i32)
                .arg(&dim_i32);
            builder
        },
    )?;
    cuda_download(backend, &d_out, out_len)
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_embedding_device(
    backend: &CudaBackend,
    table: &DeviceHandle,
    table_shape: &[usize],
    ids: &[i32],
) -> Result<DeviceHandle> {
    if table_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: table_shape.len(),
        });
    }
    let vocab = table_shape[0];
    let dim = table_shape[1];
    let n_ids = ids.len();
    let out_len = n_ids * dim;
    let d_ids = backend
        .stream
        .clone_htod(ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda htod copy failed (embedding ids)"))?;

    let n_i32 = i32::try_from(n_ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding n_ids exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding vocab exceeds i32"))?;
    let dim_i32 = i32::try_from(dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding dim exceeds i32"))?;

    const BLOCK: u32 = 256;
    // The table stays a parameter (f32 or bf16 weights); only the gathered
    // output activation follows the tape dtype.
    let (kernel_name, d_w_f32, d_w_bf16) = match table {
        DeviceHandle::Cuda(storage) => {
            let d_w = backend.cuda_storage_slice(storage)?;
            if d_w.len() != vocab * dim {
                return Err(AutogradError::DataLengthMismatch {
                    len: d_w.len(),
                    shape: table_shape.to_vec(),
                    size: vocab * dim,
                });
            }
            ("embedding_f32", Some(d_w), None)
        }
        DeviceHandle::CudaBf16(storage) => {
            let d_w = backend.cuda_bf16_storage_slice(storage)?;
            if d_w.len() != vocab * dim {
                return Err(AutogradError::DataLengthMismatch {
                    len: d_w.len(),
                    shape: table_shape.to_vec(),
                    size: vocab * dim,
                });
            }
            ("embedding_bf16_to_f32", None, Some(d_w))
        }
        DeviceHandle::CudaFp8BlockScaled(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding a fp8 block-scaled device handle",
            ));
        }
        DeviceHandle::CudaFp4E2M1Group(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding an nvfp4 device handle",
            ));
        }
        DeviceHandle::Cpu(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding a cpu device handle",
            ));
        }
        #[cfg(feature = "metal")]
        DeviceHandle::Metal(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding a metal device handle",
            ));
        }
    };

    if backend.tape_bf16() {
        let mut d_out = alloc_zeros_retry::<u16>(backend, out_len)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (embedding)"))?;
        let func = backend.kernels.function_for(kernel_name, TapeDtype::Bf16)?;
        launch_rows(&backend.stream, &func, n_ids, BLOCK, 0, |mut builder| {
            match (&d_w_f32, &d_w_bf16) {
                (Some(d_w), _) => builder.arg(&mut d_out).arg(*d_w),
                (_, Some(d_w)) => builder.arg(&mut d_out).arg(*d_w),
                _ => unreachable!("embedding table matched above"),
            };
            builder
                .arg(&d_ids)
                .arg(&n_i32)
                .arg(&vocab_i32)
                .arg(&dim_i32);
            builder
        })?;
        return Ok(DeviceHandle::CudaBf16(CudaBf16Storage::new(d_out)));
    }

    let mut d_out = alloc_zeros_retry::<f32>(backend, out_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (embedding)"))?;
    launch_rows(
        &backend.stream,
        backend.kernels.function(kernel_name)?,
        n_ids,
        BLOCK,
        0,
        |mut builder| {
            match (&d_w_f32, &d_w_bf16) {
                (Some(d_w), _) => builder.arg(&mut d_out).arg(*d_w),
                (_, Some(d_w)) => builder.arg(&mut d_out).arg(*d_w),
                _ => unreachable!("embedding table matched above"),
            };
            builder
                .arg(&d_ids)
                .arg(&n_i32)
                .arg(&vocab_i32)
                .arg(&dim_i32);
            builder
        },
    )?;
    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_embedding_from_f32_ids_device(
    backend: &CudaBackend,
    table: &DeviceHandle,
    table_shape: &[usize],
    ids: &DeviceHandle,
    n_ids: usize,
) -> Result<DeviceHandle> {
    if table_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: table_shape.len(),
        });
    }
    let vocab = table_shape[0];
    let dim = table_shape[1];
    let d_ids = backend.cuda_slice(ids, "embedding_from_f32_ids")?;
    if d_ids.len() != n_ids {
        return Err(AutogradError::DataLengthMismatch {
            len: d_ids.len(),
            shape: vec![n_ids],
            size: n_ids,
        });
    }
    let out_len = n_ids * dim;
    let mut d_out = alloc_zeros_retry::<f32>(backend, out_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (embedding f32 ids)"))?;

    let n_i32 = i32::try_from(n_ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding n_ids exceeds i32"))?;
    let vocab_i32 = i32::try_from(vocab)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding vocab exceeds i32"))?;
    let dim_i32 = i32::try_from(dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding dim exceeds i32"))?;

    const BLOCK: u32 = 256;
    match table {
        DeviceHandle::Cuda(storage) => {
            let d_w = backend.cuda_storage_slice(storage)?;
            if d_w.len() != vocab * dim {
                return Err(AutogradError::DataLengthMismatch {
                    len: d_w.len(),
                    shape: table_shape.to_vec(),
                    size: vocab * dim,
                });
            }
            launch_rows(
                &backend.stream,
                backend.kernels.function("embedding_f32_ids_f32")?,
                n_ids,
                BLOCK,
                0,
                |mut builder| {
                    builder
                        .arg(&mut d_out)
                        .arg(d_w)
                        .arg(d_ids)
                        .arg(&n_i32)
                        .arg(&vocab_i32)
                        .arg(&dim_i32);
                    builder
                },
            )?;
        }
        DeviceHandle::CudaBf16(storage) => {
            let d_w = backend.cuda_bf16_storage_slice(storage)?;
            if d_w.len() != vocab * dim {
                return Err(AutogradError::DataLengthMismatch {
                    len: d_w.len(),
                    shape: table_shape.to_vec(),
                    size: vocab * dim,
                });
            }
            launch_rows(
                &backend.stream,
                backend.kernels.function("embedding_bf16_ids_f32")?,
                n_ids,
                BLOCK,
                0,
                |mut builder| {
                    builder
                        .arg(&mut d_out)
                        .arg(d_w)
                        .arg(d_ids)
                        .arg(&n_i32)
                        .arg(&vocab_i32)
                        .arg(&dim_i32);
                    builder
                },
            )?;
        }
        DeviceHandle::CudaFp8BlockScaled(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding_from_f32_ids a fp8 block-scaled device handle",
            ));
        }
        DeviceHandle::CudaFp4E2M1Group(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding_from_f32_ids an nvfp4 device handle",
            ));
        }
        DeviceHandle::Cpu(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding_from_f32_ids a cpu device handle",
            ));
        }
        #[cfg(feature = "metal")]
        DeviceHandle::Metal(_) => {
            return Err(AutogradError::TapeInvariant(
                "cuda backend cannot embedding_from_f32_ids a metal device handle",
            ));
        }
    }

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_out)))
}

// Device-resident embedding backward. Allocates a
// zero-filled `[vocab, hidden]` grad on-device and atomicAdd-scatters the
// per-token-position upstream slice into `grad_table[ids[row], :]`. Only the
// int32 `indices` array crosses PCIe; the `[n_ids, hidden]` upstream stays
// on-device. No `synchronize()` — terminal eval is the caller's.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_embedding_backward_device(
    backend: &CudaBackend,
    upstream: &DeviceHandle,
    indices: &[i32],
    vocab_size: usize,
    hidden_dim: usize,
) -> Result<DeviceHandle> {
    let n_ids = indices.len();
    let expected_upstream = n_ids * hidden_dim;
    // grad_table is always f32 (param-grad boundary); only the upstream
    // activation grad may arrive bf16.
    let up_bf16 = matches!(upstream, DeviceHandle::CudaBf16(_));
    let up_len = match upstream {
        DeviceHandle::CudaBf16(storage) => backend.cuda_bf16_storage_slice(storage)?.len(),
        _ => backend
            .cuda_slice(upstream, "embedding_backward_device")?
            .len(),
    };
    if up_len != expected_upstream {
        return Err(AutogradError::DataLengthMismatch {
            len: up_len,
            shape: vec![n_ids, hidden_dim],
            size: expected_upstream,
        });
    }

    let out_len = vocab_size * hidden_dim;
    // alloc_zeros gives the required zero-init contract — the kernel only adds.
    let mut d_grad = alloc_zeros_retry::<f32>(backend, out_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (embedding_backward_device)")
    })?;

    if n_ids == 0 || hidden_dim == 0 {
        return Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)));
    }

    let d_ids = backend.stream.clone_htod(indices).map_err(|_| {
        AutogradError::TapeInvariant("cuda htod copy failed (embedding_backward ids)")
    })?;

    let n_ids_i32 = i32::try_from(n_ids)
        .map_err(|_| AutogradError::TapeInvariant("cuda embedding_backward n_ids exceeds i32"))?;
    let hidden_i32 = i32::try_from(hidden_dim).map_err(|_| {
        AutogradError::TapeInvariant("cuda embedding_backward hidden_dim exceeds i32")
    })?;
    let vocab_i32 = i32::try_from(vocab_size).map_err(|_| {
        AutogradError::TapeInvariant("cuda embedding_backward vocab_size exceeds i32")
    })?;

    // One thread per token position (block=256 via launch_1d). Inner
    // per-thread loop strides `hidden_dim` columns with atomicAdd. With
    // n_ids = B*S = 1024 on the canonical bench shape, this dispatches
    // 4 blocks × 256 threads — atomicAdd traffic dominates, so block-size
    // selection beyond "warp-aligned" is in the noise.
    if up_bf16 {
        let d_up = backend.cuda_bf16_slice(upstream, "embedding_backward_device")?;
        let func = backend
            .kernels
            .function_for("embedding_backward_f32", TapeDtype::Bf16)?;
        launch_1d(&backend.stream, &func, n_ids, |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_ids)
                .arg(&n_ids_i32)
                .arg(&hidden_i32)
                .arg(&vocab_i32);
            builder
        })?;
        return Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)));
    }
    let d_up = backend.cuda_slice(upstream, "embedding_backward_device")?;
    launch_1d(
        &backend.stream,
        backend.kernels.function("embedding_backward_f32")?,
        n_ids,
        |mut builder| {
            builder
                .arg(&mut d_grad)
                .arg(d_up)
                .arg(&d_ids)
                .arg(&n_ids_i32)
                .arg(&hidden_i32)
                .arg(&vocab_i32);
            builder
        },
    )?;

    Ok(DeviceHandle::Cuda(CudaStorage::new(d_grad)))
}

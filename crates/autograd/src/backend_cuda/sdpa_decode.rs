use super::*;

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_kv_cache_write_axis2(
    backend: &CudaBackend,
    dst: &DeviceHandle,
    dst_shape: &[usize],
    src: &DeviceHandle,
    src_shape: &[usize],
    seq_offset: usize,
) -> Result<DeviceHandle> {
    if dst_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: dst_shape.len(),
        });
    }
    if src_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: src_shape.len(),
        });
    }
    if dst_shape[0] != src_shape[0] || dst_shape[1] != src_shape[1] || dst_shape[3] != src_shape[3]
    {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![dst_shape[0], dst_shape[1], dst_shape[3]],
            got: vec![src_shape[0], src_shape[1], src_shape[3]],
        });
    }
    if seq_offset + src_shape[2] > dst_shape[2] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![dst_shape[2]],
            got: vec![seq_offset + src_shape[2]],
        });
    }
    let dst_total = shape_size(dst_shape);
    let src_total = shape_size(src_shape);
    // Cache lane stays f32: the sdpa consumers are untemplated, so a bf16
    // src is widened exactly at this boundary.
    let d_dst = backend.cuda_slice(dst, "kv_cache_write_axis2")?;
    let d_src_op = backend.f32_operand(src, "kv_cache_write_axis2")?;
    let d_src = d_src_op.get();
    if d_dst.len() != dst_total {
        return Err(AutogradError::DataLengthMismatch {
            len: d_dst.len(),
            shape: dst_shape.to_vec(),
            size: dst_total,
        });
    }
    if d_src.len() != src_total {
        return Err(AutogradError::DataLengthMismatch {
            len: d_src.len(),
            shape: src_shape.to_vec(),
            size: src_total,
        });
    }

    let batch_i = i32::try_from(dst_shape[0])
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write batch exceeds i32"))?;
    let heads_i = i32::try_from(dst_shape[1])
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write heads exceeds i32"))?;
    let max_seq_i = i32::try_from(dst_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write max_seq exceeds i32"))?;
    let src_seq_i = i32::try_from(src_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write src_seq exceeds i32"))?;
    let dim_i = i32::try_from(dst_shape[3])
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write dim exceeds i32"))?;
    let offset_i = i32::try_from(seq_offset)
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write offset exceeds i32"))?;
    let total_i = i32::try_from(src_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda kv write total exceeds i32"))?;

    launch_1d(
        &backend.stream,
        backend.kernels.function("kv_cache_write_axis2_f32")?,
        src_total,
        |mut builder| {
            builder
                .arg(d_dst)
                .arg(d_src)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&max_seq_i)
                .arg(&src_seq_i)
                .arg(&dim_i)
                .arg(&offset_i)
                .arg(&total_i);
            builder
        },
    )?;
    Ok(dst.clone())
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_causal_sdpa_decode_gqa(
    backend: &CudaBackend,
    q: &DeviceHandle,
    q_shape: &[usize],
    k: &DeviceHandle,
    k_shape: &[usize],
    v: &DeviceHandle,
    v_shape: &[usize],
    q_start: usize,
) -> Result<(DeviceHandle, Vec<usize>)> {
    validate_decode_gqa_shapes(q_shape, k_shape, v_shape, q_start)?;
    if k_shape[2] > 32 {
        return Err(AutogradError::TapeInvariant(
            "cuda causal_sdpa_decode_gqa supports kv_len <= 32",
        ));
    }

    let d_q = backend.cuda_slice(q, "causal_sdpa_decode_gqa")?;
    let d_k = backend.cuda_slice(k, "causal_sdpa_decode_gqa")?;
    let d_v = backend.cuda_slice(v, "causal_sdpa_decode_gqa")?;
    let q_size = shape_size(q_shape);
    let k_size = shape_size(k_shape);
    let v_size = shape_size(v_shape);
    if d_q.len() != q_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_q.len(),
            shape: q_shape.to_vec(),
            size: q_size,
        });
    }
    if d_k.len() != k_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_k.len(),
            shape: k_shape.to_vec(),
            size: k_size,
        });
    }
    if d_v.len() != v_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_v.len(),
            shape: v_shape.to_vec(),
            size: v_size,
        });
    }

    let out_shape = vec![q_shape[0], q_shape[1], 1, q_shape[3]];
    let out_total = shape_size(&out_shape);
    let mut d_out = alloc_zeros_retry::<f32>(backend, out_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (decode sdpa)"))?;

    let batch_i = i32::try_from(q_shape[0])
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa batch exceeds i32"))?;
    let query_heads_i = i32::try_from(q_shape[1])
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa heads exceeds i32"))?;
    let kv_heads_i = i32::try_from(k_shape[1])
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa kv_heads exceeds i32"))?;
    let kv_len_i = i32::try_from(k_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa kv_len exceeds i32"))?;
    let head_dim_i = i32::try_from(q_shape[3])
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa head_dim exceeds i32"))?;
    let q_start_i = i32::try_from(q_start)
        .map_err(|_| AutogradError::TapeInvariant("cuda decode sdpa q_start exceeds i32"))?;
    let scale = 1.0_f32 / (q_shape[3] as f32).sqrt();
    let rows = q_shape[0] * q_shape[1];
    const BLOCK: u32 = 256;
    let shared = BLOCK * std::mem::size_of::<f32>() as u32 + 32 * std::mem::size_of::<f32>() as u32;
    launch_rows(
        &backend.stream,
        backend.kernels.function("causal_sdpa_decode_gqa_f32")?,
        rows,
        BLOCK,
        shared,
        |mut builder| {
            builder
                .arg(d_q)
                .arg(d_k)
                .arg(d_v)
                .arg(&mut d_out)
                .arg(&batch_i)
                .arg(&query_heads_i)
                .arg(&kv_heads_i)
                .arg(&kv_len_i)
                .arg(&head_dim_i)
                .arg(&q_start_i)
                .arg(&scale);
            builder
        },
    )?;
    Ok((DeviceHandle::Cuda(CudaStorage::new(d_out)), out_shape))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_causal_sdpa_decode_gqa_cache(
    backend: &CudaBackend,
    q: &DeviceHandle,
    q_shape: &[usize],
    k: &DeviceHandle,
    k_shape: &[usize],
    v: &DeviceHandle,
    v_shape: &[usize],
    kv_len: usize,
    q_start: usize,
) -> Result<(DeviceHandle, Vec<usize>)> {
    validate_decode_gqa_cache_shapes(q_shape, k_shape, v_shape, kv_len, q_start)?;

    let d_q = backend.cuda_slice(q, "causal_sdpa_decode_gqa_cache")?;
    let d_k = backend.cuda_slice(k, "causal_sdpa_decode_gqa_cache")?;
    let d_v = backend.cuda_slice(v, "causal_sdpa_decode_gqa_cache")?;
    let q_size = shape_size(q_shape);
    let k_size = shape_size(k_shape);
    let v_size = shape_size(v_shape);
    if d_q.len() != q_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_q.len(),
            shape: q_shape.to_vec(),
            size: q_size,
        });
    }
    if d_k.len() != k_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_k.len(),
            shape: k_shape.to_vec(),
            size: k_size,
        });
    }
    if d_v.len() != v_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_v.len(),
            shape: v_shape.to_vec(),
            size: v_size,
        });
    }

    let out_shape = vec![q_shape[0], q_shape[1], 1, q_shape[3]];
    let out_total = shape_size(&out_shape);
    let mut d_out = alloc_zeros_retry::<f32>(backend, out_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (cache decode sdpa)"))?;

    let batch_i = i32::try_from(q_shape[0])
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode batch exceeds i32"))?;
    let query_heads_i = i32::try_from(q_shape[1])
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode heads exceeds i32"))?;
    let kv_heads_i = i32::try_from(k_shape[1])
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode kv_heads exceeds i32"))?;
    let max_seq_i = i32::try_from(k_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode max_seq exceeds i32"))?;
    let kv_len_i = i32::try_from(kv_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode kv_len exceeds i32"))?;
    let head_dim_i = i32::try_from(q_shape[3])
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode head_dim exceeds i32"))?;
    let q_start_i = i32::try_from(q_start)
        .map_err(|_| AutogradError::TapeInvariant("cuda cache decode q_start exceeds i32"))?;
    let scale = 1.0_f32 / (q_shape[3] as f32).sqrt();
    let rows = q_shape[0] * q_shape[1];
    let head_dim = q_shape[3];

    // --autograd-decode-attn-legacy forces the two-pass kernel below.
    let use_online = head_dim == 256 && !crate::runtime_flags::decode_attn_legacy();
    if use_online {
        const BLOCK_ONLINE: u32 = 256; // = HEAD_DIM
        let n_warps = BLOCK_ONLINE / 32;
        let shared_online = n_warps * std::mem::size_of::<f32>() as u32;
        launch_rows(
            &backend.stream,
            backend
                .kernels
                .function("causal_sdpa_decode_gqa_cache_online_f32_hd256")?,
            rows,
            BLOCK_ONLINE,
            shared_online,
            |mut builder| {
                builder
                    .arg(d_q)
                    .arg(d_k)
                    .arg(d_v)
                    .arg(&mut d_out)
                    .arg(&batch_i)
                    .arg(&query_heads_i)
                    .arg(&kv_heads_i)
                    .arg(&max_seq_i)
                    .arg(&kv_len_i)
                    .arg(&head_dim_i)
                    .arg(&q_start_i)
                    .arg(&scale);
                builder
            },
        )?;
        return Ok((DeviceHandle::Cuda(CudaStorage::new(d_out)), out_shape));
    }

    // Kept for head_dim != 256 and as the legacy escape hatch.
    const BLOCK: u32 = 256;
    let visible = q_start.saturating_add(1).min(kv_len);
    let shared = BLOCK * std::mem::size_of::<f32>() as u32
        + u32::try_from(visible)
            .map_err(|_| AutogradError::TapeInvariant("cuda cache decode visible exceeds u32"))?
            * std::mem::size_of::<f32>() as u32;
    launch_rows(
        &backend.stream,
        backend
            .kernels
            .function("causal_sdpa_decode_gqa_cache_f32")?,
        rows,
        BLOCK,
        shared,
        |mut builder| {
            builder
                .arg(d_q)
                .arg(d_k)
                .arg(d_v)
                .arg(&mut d_out)
                .arg(&batch_i)
                .arg(&query_heads_i)
                .arg(&kv_heads_i)
                .arg(&max_seq_i)
                .arg(&kv_len_i)
                .arg(&head_dim_i)
                .arg(&q_start_i)
                .arg(&scale);
            builder
        },
    )?;
    Ok((DeviceHandle::Cuda(CudaStorage::new(d_out)), out_shape))
}

#[cfg(not(feature = "no-cuda"))]
#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_qwen_decode_prepare_q(
    backend: &CudaBackend,
    q_full: &DeviceHandle,
    q_full_shape: &[usize],
    q_norm_weight: &DeviceHandle,
    q_norm_weight_shape: &[usize],
    cos: &DeviceHandle,
    cos_shape: &[usize],
    sin: &DeviceHandle,
    sin_shape: &[usize],
    query_heads: usize,
    head_dim: usize,
    gated: bool,
    eps: f32,
) -> Result<(DeviceHandle, Option<DeviceHandle>, Vec<usize>)> {
    validate_qwen_decode_prepare_q_shapes(
        q_full_shape,
        q_norm_weight_shape,
        cos_shape,
        sin_shape,
        query_heads,
        head_dim,
        gated,
    )?;
    let q_full_size = shape_size(q_full_shape);
    let d_q_full = backend.cuda_slice(q_full, "qwen_decode_prepare_q")?;
    let d_weight = backend.cuda_slice(q_norm_weight, "qwen_decode_prepare_q")?;
    let d_cos = backend.cuda_slice(cos, "qwen_decode_prepare_q")?;
    let d_sin = backend.cuda_slice(sin, "qwen_decode_prepare_q")?;
    if d_q_full.len() != q_full_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_q_full.len(),
            shape: q_full_shape.to_vec(),
            size: q_full_size,
        });
    }
    if d_weight.len() != head_dim {
        return Err(AutogradError::DataLengthMismatch {
            len: d_weight.len(),
            shape: q_norm_weight_shape.to_vec(),
            size: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    if d_cos.len() != half_dim || d_sin.len() != half_dim {
        return Err(AutogradError::DataLengthMismatch {
            len: d_cos.len().max(d_sin.len()),
            shape: vec![1, half_dim],
            size: half_dim,
        });
    }

    let batch = q_full_shape[0];
    let out_shape = vec![batch, query_heads, 1, head_dim];
    let out_total = shape_size(&out_shape);
    let mut d_q = alloc_zeros_retry::<f32>(backend, out_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (qwen prep q)"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep q batch exceeds i32"))?;
    let query_heads_i = i32::try_from(query_heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep q heads exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep q head_dim exceeds i32"))?;
    let q_full_stride_i = i32::try_from(q_full_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep q stride exceeds i32"))?;
    let rows = batch * query_heads;
    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;

    let gate_handle = if gated {
        let mut d_gate = alloc_zeros_retry::<f32>(backend, out_total).map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (qwen prep gate)")
        })?;
        launch_rows(
            &backend.stream,
            backend
                .kernels
                .function("qwen_decode_prepare_q_gated_f32")?,
            rows,
            BLOCK,
            SHARED,
            |mut builder| {
                builder
                    .arg(&mut d_q)
                    .arg(&mut d_gate)
                    .arg(d_q_full)
                    .arg(d_weight)
                    .arg(d_cos)
                    .arg(d_sin)
                    .arg(&batch_i)
                    .arg(&query_heads_i)
                    .arg(&head_dim_i)
                    .arg(&q_full_stride_i)
                    .arg(&eps);
                builder
            },
        )?;
        Some(DeviceHandle::Cuda(CudaStorage::new(d_gate)))
    } else {
        launch_rows(
            &backend.stream,
            backend.kernels.function("qwen_decode_prepare_q_f32")?,
            rows,
            BLOCK,
            SHARED,
            |mut builder| {
                builder
                    .arg(&mut d_q)
                    .arg(d_q_full)
                    .arg(d_weight)
                    .arg(d_cos)
                    .arg(d_sin)
                    .arg(&batch_i)
                    .arg(&query_heads_i)
                    .arg(&head_dim_i)
                    .arg(&q_full_stride_i)
                    .arg(&eps);
                builder
            },
        )?;
        None
    };

    Ok((
        DeviceHandle::Cuda(CudaStorage::new(d_q)),
        gate_handle,
        out_shape,
    ))
}

#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_qwen_decode_prepare_kv(
    backend: &CudaBackend,
    k_full: &DeviceHandle,
    k_full_shape: &[usize],
    v_full: &DeviceHandle,
    v_full_shape: &[usize],
    k_norm_weight: &DeviceHandle,
    k_norm_weight_shape: &[usize],
    cos: &DeviceHandle,
    cos_shape: &[usize],
    sin: &DeviceHandle,
    sin_shape: &[usize],
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(DeviceHandle, DeviceHandle, Vec<usize>)> {
    validate_qwen_decode_prepare_kv_shapes(
        k_full_shape,
        v_full_shape,
        k_norm_weight_shape,
        cos_shape,
        sin_shape,
        kv_heads,
        head_dim,
    )?;
    let k_full_size = shape_size(k_full_shape);
    let v_full_size = shape_size(v_full_shape);
    let d_k_full = backend.cuda_slice(k_full, "qwen_decode_prepare_kv")?;
    let d_v_full = backend.cuda_slice(v_full, "qwen_decode_prepare_kv")?;
    let d_weight = backend.cuda_slice(k_norm_weight, "qwen_decode_prepare_kv")?;
    let d_cos = backend.cuda_slice(cos, "qwen_decode_prepare_kv")?;
    let d_sin = backend.cuda_slice(sin, "qwen_decode_prepare_kv")?;
    if d_k_full.len() != k_full_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_k_full.len(),
            shape: k_full_shape.to_vec(),
            size: k_full_size,
        });
    }
    if d_v_full.len() != v_full_size {
        return Err(AutogradError::DataLengthMismatch {
            len: d_v_full.len(),
            shape: v_full_shape.to_vec(),
            size: v_full_size,
        });
    }
    if d_weight.len() != head_dim {
        return Err(AutogradError::DataLengthMismatch {
            len: d_weight.len(),
            shape: k_norm_weight_shape.to_vec(),
            size: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    if d_cos.len() != half_dim || d_sin.len() != half_dim {
        return Err(AutogradError::DataLengthMismatch {
            len: d_cos.len().max(d_sin.len()),
            shape: vec![1, half_dim],
            size: half_dim,
        });
    }

    let batch = k_full_shape[0];
    let out_shape = vec![batch, kv_heads, 1, head_dim];
    let out_total = shape_size(&out_shape);
    let mut d_k = alloc_zeros_retry::<f32>(backend, out_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (qwen prep k)"))?;
    let mut d_v = alloc_zeros_retry::<f32>(backend, out_total)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (qwen prep v)"))?;

    let batch_i = i32::try_from(batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep kv batch exceeds i32"))?;
    let kv_heads_i = i32::try_from(kv_heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep kv heads exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep kv head_dim exceeds i32"))?;
    let kv_full_stride_i = i32::try_from(k_full_shape[2])
        .map_err(|_| AutogradError::TapeInvariant("cuda qwen prep kv stride exceeds i32"))?;
    let rows = batch * kv_heads;
    const BLOCK: u32 = 256;
    const SHARED: u32 = BLOCK * std::mem::size_of::<f32>() as u32;

    launch_rows(
        &backend.stream,
        backend.kernels.function("qwen_decode_prepare_kv_f32")?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder
                .arg(&mut d_k)
                .arg(&mut d_v)
                .arg(d_k_full)
                .arg(d_v_full)
                .arg(d_weight)
                .arg(d_cos)
                .arg(d_sin)
                .arg(&batch_i)
                .arg(&kv_heads_i)
                .arg(&head_dim_i)
                .arg(&kv_full_stride_i)
                .arg(&eps);
            builder
        },
    )?;

    Ok((
        DeviceHandle::Cuda(CudaStorage::new(d_k)),
        DeviceHandle::Cuda(CudaStorage::new(d_v)),
        out_shape,
    ))
}

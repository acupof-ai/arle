use super::*;

// Fused causal prefill SDPA — the production inference kernel
// (`nonpaged_prefill_attention_cuda`: bf16, online softmax, GQA native)
// adopted for the training forward. Layout bridge: training q `[1, h, s, d]`
// transposes to the kernel's token-major `[s, h, d]`; k/v `[1, h_kv, kv, d]`
// contiguous already match its head-major cache view (`max_seq_len = kv`).
#[cfg(not(feature = "no-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_causal_sdpa_prefill_device(
    backend: &CudaBackend,
    q: &DeviceHandle,
    q_shape: &[usize],
    k: &DeviceHandle,
    k_shape: &[usize],
    v: &DeviceHandle,
    v_shape: &[usize],
    q_start: usize,
) -> Result<Option<(DeviceHandle, Vec<usize>)>> {
    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape != k_shape {
        return Ok(None);
    }
    let (batch, heads, seq, dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (kv_heads, kv_len) = (k_shape[1], k_shape[2]);
    // Kernel envelope: head_dim ∈ {128, 256}, grid.y = seq ≤ 65535, exact
    // causal window, whole GQA groups. Outside it => compose from primitives.
    if batch != 1
        || k_shape[0] != 1
        || k_shape[3] != dim
        || !(dim == 128 || dim == 256)
        || seq > 65_535
        || kv_heads == 0
        || heads % kv_heads != 0
        || q_start + seq != kv_len
    {
        return Ok(None);
    }

    let (q_t, _) = backend.transpose_axes_swap(q, q_shape, 1, 2)?; // [1, s, h, d]
    let q_slice = backend.cuda_slice(&q_t, "sdpa_prefill q")?;
    let k_slice = backend.cuda_slice(k, "sdpa_prefill k")?;
    let v_slice = backend.cuda_slice(v, "sdpa_prefill v")?;
    let q_bf16 = backend.local_f32_as_bf16(q_slice, q_slice.len())?;
    let k_bf16 = backend.local_f32_as_bf16(k_slice, k_slice.len())?;
    let v_bf16 = backend.local_f32_as_bf16(v_slice, v_slice.len())?;
    let out_len = seq * heads * dim;
    let mut out_bf16 = alloc_zeros_retry::<u16>(backend, out_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sdpa prefill out)"))?;

    {
        let (q_ptr, _q_guard) = q_bf16.device_ptr(&backend.stream);
        let (k_ptr, _k_guard) = k_bf16.device_ptr(&backend.stream);
        let (v_ptr, _v_guard) = v_bf16.device_ptr(&backend.stream);
        let (out_ptr, _out_guard) = out_bf16.device_ptr_mut(&backend.stream);
        // q/k/v are live guarded bf16 copies of tape tensors whose shapes passed
        // the envelope check; out is allocated seq*heads*dim; the dims passed
        // mirror those shapes (`max_seq_len = kv_len` for the contiguous view).
        cuda_kernels::attention::nonpaged_prefill_attention_raw(
            &backend.stream,
            q_ptr,
            k_ptr,
            v_ptr,
            out_ptr,
            heads,
            kv_heads,
            dim,
            seq,
            kv_len,
            kv_len,
            (dim as f32).sqrt().recip(),
        )
        .map_err(|e| leak_err(format!("nonpaged_prefill_attention_cuda: {e}")))?;
    }

    // Trace-gated sync: pin an async kernel fault here, not at the next alloc.
    if std::env::var("ARLE_SDPA_TRACE").is_ok() {
        backend.stream.synchronize().map_err(|e| {
            leak_err(format!(
                "sdpa fused async fault: seq={seq} dim={dim} err={e:?}"
            ))
        })?;
    }

    let out_f32 = backend.import_local_bf16_as_f32(&out_bf16, out_len)?;
    let out_handle = DeviceHandle::Cuda(CudaStorage::new(out_f32));
    let (out, out_shape) = backend.transpose_axes_swap(&out_handle, &[1, seq, heads, dim], 1, 2)?; // [1, h, s, d]
    Ok(Some((out, out_shape)))
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_causal_sdpa_recompute_backward_device(
    backend: &CudaBackend,
    args: CausalSdpaDeviceBackwardArgs<'_>,
) -> Result<CausalSdpaDeviceGradTriplet> {
    if args.shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: args.shape.len(),
        });
    }
    if !args.need_grad_q && !args.need_grad_k && !args.need_grad_v {
        return Ok((None, None, None));
    }

    let batch = args.shape[0];
    let heads = args.shape[1];
    let seq_len = args.shape[2];
    let head_dim = args.shape[3];
    let total = shape_size(args.shape);
    if seq_len == 0 || head_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero sequence and head_dim",
            got: args.shape.len(),
        });
    }

    if head_dim > 256 {
        let q = backend.readback(args.q)?;
        let k = backend.readback(args.k)?;
        let v = backend.readback(args.v)?;
        let upstream = backend.readback(args.upstream)?;
        let (grad_q, grad_k, grad_v) = cpu_causal_sdpa_recompute_backward(
            &q,
            &k,
            &v,
            &upstream,
            args.shape,
            args.need_grad_q,
            args.need_grad_k,
            args.need_grad_v,
        )?;
        return Ok((
            match grad_q {
                Some(grad) => Some(DeviceHandle::Cuda(CudaStorage::new(
                    backend.upload_slice(&grad, args.shape)?,
                ))),
                None => None,
            },
            match grad_k {
                Some(grad) => Some(DeviceHandle::Cuda(CudaStorage::new(
                    backend.upload_slice(&grad, args.shape)?,
                ))),
                None => None,
            },
            match grad_v {
                Some(grad) => Some(DeviceHandle::Cuda(CudaStorage::new(
                    backend.upload_slice(&grad, args.shape)?,
                ))),
                None => None,
            },
        ));
    }

    let d_q_op = backend.f32_operand(args.q, "causal_sdpa_recompute_backward_device")?;
    let d_k_op = backend.f32_operand(args.k, "causal_sdpa_recompute_backward_device")?;
    let d_v_op = backend.f32_operand(args.v, "causal_sdpa_recompute_backward_device")?;
    let d_up_op = backend.f32_operand(args.upstream, "causal_sdpa_recompute_backward_device")?;
    let (d_q, d_k, d_v, d_up) = (d_q_op.get(), d_k_op.get(), d_v_op.get(), d_up_op.get());
    if d_q.len() != total || d_k.len() != total || d_v.len() != total || d_up.len() != total {
        return Err(AutogradError::TapeInvariant(
            "cuda causal_sdpa_recompute_backward_device handle size does not match shape",
        ));
    }

    let out_len_q = if args.need_grad_q { total } else { 1 };
    let out_len_k = if args.need_grad_k { total } else { 1 };
    let out_len_v = if args.need_grad_v { total } else { 1 };
    let mut grad_q = alloc_zeros_retry::<f32>(backend, out_len_q)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sdpa grad_q)"))?;
    let mut grad_k = alloc_zeros_retry::<f32>(backend, out_len_k)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sdpa grad_k)"))?;
    let mut grad_v = alloc_zeros_retry::<f32>(backend, out_len_v)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (sdpa grad_v)"))?;

    let rows = batch
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(seq_len))
        .ok_or(AutogradError::TapeInvariant("cuda sdpa row count overflow"))?;
    let rows_i = i32::try_from(rows)
        .map_err(|_| AutogradError::TapeInvariant("cuda sdpa rows exceeds i32"))?;
    let seq_i = i32::try_from(seq_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda sdpa seq_len exceeds i32"))?;
    let head_dim_i = i32::try_from(head_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda sdpa head_dim exceeds i32"))?;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let need_q_i = if args.need_grad_q { 1 } else { 0 };
    let need_k_i = if args.need_grad_k { 1 } else { 0 };
    let need_v_i = if args.need_grad_v { 1 } else { 0 };

    const BLOCK: u32 = 32;
    const SHARED: u32 = 0;
    launch_rows(
        &backend.stream,
        backend
            .kernels
            .function("causal_sdpa_recompute_backward_f32")?,
        rows,
        BLOCK,
        SHARED,
        |mut builder| {
            builder
                .arg(d_q)
                .arg(d_k)
                .arg(d_v)
                .arg(d_up)
                .arg(&mut grad_q)
                .arg(&mut grad_k)
                .arg(&mut grad_v)
                .arg(&rows_i)
                .arg(&seq_i)
                .arg(&head_dim_i)
                .arg(&scale)
                .arg(&need_q_i)
                .arg(&need_k_i)
                .arg(&need_v_i);
            builder
        },
    )?;

    Ok((
        args.need_grad_q
            .then(|| DeviceHandle::Cuda(CudaStorage::new(grad_q))),
        args.need_grad_k
            .then(|| DeviceHandle::Cuda(CudaStorage::new(grad_k))),
        args.need_grad_v
            .then(|| DeviceHandle::Cuda(CudaStorage::new(grad_v))),
    ))
}

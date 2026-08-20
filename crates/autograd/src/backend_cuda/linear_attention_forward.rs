use super::*;

#[cfg(not(feature = "no-cuda"))]
pub(super) fn linear_attention_debug_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("ARLE_LINEAR_ATTENTION_DEBUG_TIMING").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        )
    })
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn linear_attention_debug_stage_start() -> Option<std::time::Instant> {
    linear_attention_debug_timing_enabled().then(std::time::Instant::now)
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn linear_attention_debug_stage_done(
    backend: &CudaBackend,
    label: &'static str,
    started: Option<std::time::Instant>,
) -> Result<()> {
    if let Some(started) = started {
        backend.stream.synchronize().map_err(|err| {
            crate::AutogradError::TapeInvariant(Box::leak(
                format!("cuda synchronize failed (linear_attention {label}): {err:?}")
                    .into_boxed_str(),
            ))
        })?;
        eprintln!(
            "cuda linear_attention_forward stage={label} seconds={:.6}",
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// Shape gate for the device LA path. Only the head DIM is baked into the
/// kernels (GDR_KEY_DIM/VAL_DIM=128; num_value_heads stays a runtime param);
/// batch>1 rides per-row dispatch because the chunked kernels' chunk_state
/// carries no batch stride.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_la_device_supported(p: LinearAttentionDeviceParams) -> bool {
    p.key_dim == 128 && p.value_dim == 128 && p.conv_kernel > 0 && p.conv_kernel <= 5 && p.batch > 0
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_linear_attention_forward_device(
    backend: &CudaBackend,
    args: LinearAttentionDeviceForwardArgs<'_>,
) -> Result<Option<LinearAttentionDeviceForwardResult>> {
    let p = args.params;
    if !cuda_la_device_supported(p) {
        return Ok(None);
    }
    if p.batch == 1 {
        return cuda_linear_attention_forward_device_row(backend, args).map(Some);
    }

    // batch > 1: per-row dispatch to the proven batch==1 path. Every input and
    // result tensor is batch-leading, so row slicing and reassembly are
    // contiguous-range D2D copies; weights pass through whole.
    let row_params = LinearAttentionDeviceParams { batch: 1, ..p };
    let rows = (0..p.batch)
        .map(|row| {
            let slice = |src| cuda_row_slice(backend, src, row, p.batch);
            // Carry is batch-leading like the other inputs — slice per row too.
            let initial_state = args.initial_state.map(slice).transpose()?;
            let initial_conv_window = args.initial_conv_window.map(slice).transpose()?;
            cuda_linear_attention_forward_device_row(
                backend,
                LinearAttentionDeviceForwardArgs {
                    params: row_params,
                    qkv: &slice(args.qkv)?,
                    z: &slice(args.z)?,
                    b_proj: &slice(args.b_proj)?,
                    a_proj: &slice(args.a_proj)?,
                    conv1d_weight: args.conv1d_weight,
                    dt_bias: args.dt_bias,
                    a_log: args.a_log,
                    norm_weight: args.norm_weight,
                    initial_state: initial_state.as_ref(),
                    initial_conv_window: initial_conv_window.as_ref(),
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let concat = |field: fn(&LinearAttentionDeviceForwardResult) -> &DeviceHandle| {
        let parts: Vec<&DeviceHandle> = rows.iter().map(field).collect();
        cuda_concat_rows(backend, &parts)
    };
    Ok(Some(LinearAttentionDeviceForwardResult {
        output: concat(|r| &r.output)?,
        preact: concat(|r| &r.preact)?,
        qkv_conv: concat(|r| &r.qkv_conv)?,
        q: concat(|r| &r.q)?,
        k: concat(|r| &r.k)?,
        v: concat(|r| &r.v)?,
        g: concat(|r| &r.g)?,
        g_cumsum: concat(|r| &r.g_cumsum)?,
        beta: concat(|r| &r.beta)?,
        a_inv: concat(|r| &r.a_inv)?,
        chunk_state: concat(|r| &r.chunk_state)?,
        raw_output: concat(|r| &r.raw_output)?,
        flashqla: rows[0].flashqla,
    }))
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_linear_attention_boundary_device(
    backend: &CudaBackend,
    args: LinearAttentionDeviceBoundaryArgs<'_>,
) -> Result<Option<DeviceHandle>> {
    let p = args.params;
    if !cuda_la_device_supported(p) {
        return Ok(None);
    }
    if p.batch == 1 {
        return cuda_linear_attention_boundary_device_row(backend, args).map(Some);
    }

    let row_params = LinearAttentionDeviceParams { batch: 1, ..p };
    let rows = (0..p.batch)
        .map(|row| {
            let slice = |src| cuda_row_slice(backend, src, row, p.batch);
            let qkv = slice(args.qkv)?;
            let b_proj = slice(args.b_proj)?;
            let a_proj = slice(args.a_proj)?;
            let initial_state = args.initial_state.map(slice).transpose()?;
            let initial_conv_window = args.initial_conv_window.map(slice).transpose()?;
            cuda_linear_attention_boundary_device_row(
                backend,
                LinearAttentionDeviceBoundaryArgs {
                    params: row_params,
                    qkv: &qkv,
                    b_proj: &b_proj,
                    a_proj: &a_proj,
                    conv1d_weight: args.conv1d_weight,
                    dt_bias: args.dt_bias,
                    a_log: args.a_log,
                    initial_state: initial_state.as_ref(),
                    initial_conv_window: initial_conv_window.as_ref(),
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let parts = rows.iter().collect::<Vec<_>>();
    cuda_concat_rows(backend, &parts).map(Some)
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_linear_attention_boundary_device_row(
    backend: &CudaBackend,
    args: LinearAttentionDeviceBoundaryArgs<'_>,
) -> Result<DeviceHandle> {
    let p = args.params;
    debug_assert_eq!(p.batch, 1);
    let q_dim = p.num_key_heads * p.key_dim;
    let qkv_dim = q_dim * 2 + p.num_value_heads * p.value_dim;
    let qkv_len = p.seq_len * qkv_dim;
    let head_len = p.seq_len * p.num_value_heads;
    let state_len = p.num_value_heads * p.key_dim * p.value_dim;
    let conv_tail_len = p.conv_kernel - 1;
    let qkv = backend.cuda_slice(args.qkv, "linear_attention_boundary qkv")?;
    let b_proj = backend.cuda_slice(args.b_proj, "linear_attention_boundary b_proj")?;
    let a_proj = backend.cuda_slice(args.a_proj, "linear_attention_boundary a_proj")?;
    let conv1d_weight = backend.cuda_slice(
        args.conv1d_weight,
        "linear_attention_boundary conv1d_weight",
    )?;
    let dt_bias = backend.cuda_slice(args.dt_bias, "linear_attention_boundary dt_bias")?;
    let a_log = backend.cuda_slice(args.a_log, "linear_attention_boundary a_log")?;
    let carry_state = args
        .initial_state
        .map(|h| backend.cuda_slice(h, "linear_attention_boundary initial_state"))
        .transpose()?;
    let carry_conv = args
        .initial_conv_window
        .map(|h| backend.cuda_slice(h, "linear_attention_boundary initial_conv_window"))
        .transpose()?;

    for (got, expected) in [
        (qkv.len(), qkv_len),
        (b_proj.len(), head_len),
        (a_proj.len(), head_len),
        (conv1d_weight.len(), qkv_dim * p.conv_kernel),
        (dt_bias.len(), p.num_value_heads),
        (a_log.len(), p.num_value_heads),
    ] {
        if got != expected {
            return Err(AutogradError::TapeInvariant(
                "linear_attention_boundary input length mismatch",
            ));
        }
    }
    if carry_state.is_some_and(|x| x.len() != state_len)
        || carry_conv.is_some_and(|x| x.len() != conv_tail_len * qkv_dim)
    {
        return Err(AutogradError::TapeInvariant(
            "linear_attention_boundary carry length mismatch",
        ));
    }

    let b_bf16 = backend.local_f32_as_bf16(b_proj, head_len)?;
    let a_bf16 = backend.local_f32_as_bf16(a_proj, head_len)?;
    let dt_bf16 = backend.local_f32_as_bf16(dt_bias, p.num_value_heads)?;
    let chunk_rows = p.seq_len.min(64);
    let mut qkv_chunk = backend
        .stream
        .alloc_zeros::<u16>(chunk_rows * qkv_dim)
        .map_err(|_| cuda_alloc_failed("la boundary qkv", vec![chunk_rows, qkv_dim]))?;
    let mut raw_chunk = backend
        .stream
        .alloc_zeros::<u16>(chunk_rows * p.num_value_heads * p.value_dim)
        .map_err(|_| {
            cuda_alloc_failed(
                "la boundary raw",
                vec![chunk_rows, p.num_value_heads, p.value_dim],
            )
        })?;
    let mut state = backend
        .stream
        .alloc_zeros::<f32>(state_len)
        .map_err(|_| cuda_alloc_failed("la boundary state", vec![state_len]))?;
    if let Some(initial) = carry_state {
        backend
            .stream
            .memcpy_dtod(initial, &mut state)
            .map_err(|_| AutogradError::TapeInvariant("la boundary state seed failed"))?;
    }

    let qkv_dim_i32 = i32::try_from(qkv_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention qkv_dim exceeds i32"))?;
    let conv_kernel_i32 = i32::try_from(p.conv_kernel)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention conv_kernel exceeds i32"))?;
    let num_key_heads_i32 = i32::try_from(p.num_key_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key heads exceeds i32"))?;
    let num_value_heads_i32 = i32::try_from(p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value heads exceeds i32"))?;
    let key_dim_i32 = i32::try_from(p.key_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key_dim exceeds i32"))?;
    let value_dim_i32 = i32::try_from(p.value_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value_dim exceeds i32"))?;
    let conv_tail = carry_conv.map(|x| x.device_ptr(&backend.stream));
    let conv_tail_ptr = conv_tail.as_ref().map_or(0u64, |(ptr, _)| *ptr);
    let conv_tail_len_i32 = if carry_conv.is_some() {
        i32::try_from(conv_tail_len)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention conv tail exceeds i32"))?
    } else {
        0
    };

    for start in (0..p.seq_len).step_by(64) {
        let rows = (p.seq_len - start).min(64);
        let head_start = start * p.num_value_heads;
        let total = rows * qkv_dim;
        let total_u64 = total as u64;
        let start_i32 = i32::try_from(start)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention start exceeds i32"))?;
        {
            let (out_ptr, _out_guard) = qkv_chunk.device_ptr_mut(&backend.stream);
            let (qkv_ptr, _qkv_guard) = qkv.device_ptr(&backend.stream);
            let (conv_ptr, _conv_guard) = conv1d_weight.device_ptr(&backend.stream);
            launch_1d(
                &backend.stream,
                backend
                    .kernels
                    .function("linear_attention_conv1d_silu_boundary_f32_to_bf16")?,
                total,
                |mut builder| {
                    builder
                        .arg(&out_ptr)
                        .arg(&qkv_ptr)
                        .arg(&conv_ptr)
                        .arg(&total_u64)
                        .arg(&start_i32)
                        .arg(&qkv_dim_i32)
                        .arg(&conv_kernel_i32)
                        .arg(&conv_tail_ptr)
                        .arg(&conv_tail_len_i32);
                    builder
                },
            )?;
        }
        let rows_i32 = i32::try_from(rows)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention rows exceeds i32"))?;
        let (qkv_ptr, _qkv_guard) = qkv_chunk.device_ptr(&backend.stream);
        let (b_ptr, _b_guard) = b_bf16.device_ptr(&backend.stream);
        let (a_ptr, _a_guard) = a_bf16.device_ptr(&backend.stream);
        let (dt_ptr, _dt_guard) = dt_bf16.device_ptr(&backend.stream);
        let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
        let (state_ptr, _state_guard) = state.device_ptr_mut(&backend.stream);
        let (raw_ptr, _raw_guard) = raw_chunk.device_ptr_mut(&backend.stream);
        check_cuda_ffi(
            // SAFETY: all pointers cover the dimensions passed below.
            unsafe {
                ffi::gated_delta_rule_prefill_recurrent_cuda(
                    qkv_ptr as *const ffi::Half,
                    (b_ptr as *const ffi::Half).add(head_start),
                    (a_ptr as *const ffi::Half).add(head_start),
                    dt_ptr as *const ffi::Half,
                    a_log_ptr as *const f32,
                    state_ptr as *mut f32,
                    raw_ptr as *mut ffi::Half,
                    num_key_heads_i32,
                    num_value_heads_i32,
                    key_dim_i32,
                    value_dim_i32,
                    rows_i32,
                    backend.stream.cu_stream(),
                )
            },
            "gated_delta_rule_prefill_recurrent_cuda",
        )?;
    }
    Ok(DeviceHandle::Cuda(CudaStorage::new(state)))
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_linear_attention_forward_device_row(
    backend: &CudaBackend,
    args: LinearAttentionDeviceForwardArgs<'_>,
) -> Result<LinearAttentionDeviceForwardResult> {
    let p = args.params;
    debug_assert_eq!(p.batch, 1);
    let q_dim = p.num_key_heads * p.key_dim;
    let qkv_dim = q_dim * 2 + p.num_value_heads * p.value_dim;
    let qkv_len = p.batch * p.seq_len * qkv_dim;
    let z_len = p.batch * p.seq_len * p.num_value_heads * p.value_dim;
    let head_len = p.batch * p.seq_len * p.num_value_heads;
    let conv_len = qkv_dim * p.conv_kernel;
    let num_chunks = p.seq_len.div_ceil(64);
    // The old seq<=32 clamp guarded a WGMMA deadlock in the chunk kernel 778fef873 replaced.
    let use_chunkwise = crate::runtime_flags::gdr_chunkwise_prefill();
    // FlashQLA keeps q/k on the key-head axis; the recurrent prepare broadcasts each
    // key head across its value group, so that route needs the wide extent.
    let qk_heads = if use_chunkwise {
        p.num_key_heads
    } else {
        p.num_value_heads
    };
    let q_len = p.seq_len * qk_heads * p.key_dim;
    let v_len = p.seq_len * p.num_value_heads * p.value_dim;
    let a_len = p.seq_len * p.num_value_heads * 64;
    let state_len = p.num_value_heads * p.key_dim * p.value_dim;
    // FlashQLA recomputes the per-chunk states in the backward, so only the
    // carry (chunk 0) survives on the tape.
    let chunk_state_len = if use_chunkwise {
        state_len
    } else {
        num_chunks * state_len
    };

    let qkv = backend.cuda_slice(args.qkv, "linear_attention_forward qkv")?;
    let z = backend.cuda_slice(args.z, "linear_attention_forward z")?;
    let b_proj = backend.cuda_slice(args.b_proj, "linear_attention_forward b_proj")?;
    let a_proj = backend.cuda_slice(args.a_proj, "linear_attention_forward a_proj")?;
    let conv1d_weight =
        backend.cuda_slice(args.conv1d_weight, "linear_attention_forward conv1d_weight")?;
    let dt_bias = backend.cuda_slice(args.dt_bias, "linear_attention_forward dt_bias")?;
    let a_log = backend.cuda_slice(args.a_log, "linear_attention_forward a_log")?;
    let norm_weight =
        backend.cuda_slice(args.norm_weight, "linear_attention_forward norm_weight")?;

    // OPD carry (None = default zero-seed). initial_state seeds final_state (→ chunk_state[0]);
    // conv_tail feeds the conv1d boundary taps. tail_len = conv_kernel-1 rows of qkv_dim channels.
    let conv_tail_len = p.conv_kernel - 1;
    let carry_state = args
        .initial_state
        .map(|h| backend.cuda_slice(h, "linear_attention_forward initial_state"))
        .transpose()?;
    let carry_conv = args
        .initial_conv_window
        .map(|h| backend.cuda_slice(h, "linear_attention_forward initial_conv_window"))
        .transpose()?;

    for (label, got, expected) in [
        ("qkv", Some(qkv.len()), qkv_len),
        ("z", Some(z.len()), z_len),
        ("b_proj", Some(b_proj.len()), head_len),
        ("a_proj", Some(a_proj.len()), head_len),
        ("conv1d_weight", Some(conv1d_weight.len()), conv_len),
        ("dt_bias", Some(dt_bias.len()), p.num_value_heads),
        ("a_log", Some(a_log.len()), p.num_value_heads),
        ("norm_weight", Some(norm_weight.len()), p.value_dim),
        ("initial_state", carry_state.map(|s| s.len()), state_len),
        (
            "initial_conv_window",
            carry_conv.map(|s| s.len()),
            conv_tail_len * qkv_dim,
        ),
    ] {
        if let Some(got) = got
            && got != expected
        {
            return Err(AutogradError::TapeInvariant(Box::leak(
                format!(
                    "cuda linear_attention_forward_device {label} len mismatch: got={got} expected={expected}"
                )
                .into_boxed_str(),
            )));
        }
    }

    let b_bf16 = backend.local_f32_as_bf16(b_proj, head_len)?;
    let a_bf16 = backend.local_f32_as_bf16(a_proj, head_len)?;
    let dt_bf16 = backend.local_f32_as_bf16(dt_bias, p.num_value_heads)?;

    let mut preact = backend
        .stream
        .alloc_zeros::<f32>(qkv_len)
        .map_err(|e| cuda_alloc_failed_rich(backend, "la preact", qkv_len * 4, &e))?;
    let mut qkv_conv = backend
        .stream
        .alloc_zeros::<u16>(qkv_len)
        .map_err(|e| cuda_alloc_failed_rich(backend, "la qkv_conv", qkv_len * 2, &e))?;
    let mut q = backend
        .stream
        .alloc_zeros::<u16>(q_len)
        .map_err(|e| cuda_alloc_failed_rich(backend, "la q", q_len * 2, &e))?;
    let mut k = backend
        .stream
        .alloc_zeros::<u16>(q_len)
        .map_err(|e| cuda_alloc_failed_rich(backend, "la k", q_len * 2, &e))?;
    let mut v = backend
        .stream
        .alloc_zeros::<u16>(v_len)
        .map_err(|e| cuda_alloc_failed_rich(backend, "la v", v_len * 2, &e))?;
    let mut g = backend
        .stream
        .alloc_zeros::<f32>(head_len)
        .map_err(|e| cuda_alloc_failed_rich(backend, "la g", head_len * 4, &e))?;
    let mut g_cumsum = backend
        .stream
        .alloc_zeros::<f32>(head_len)
        .map_err(|e| cuda_alloc_failed_rich(backend, "la g_cumsum", head_len * 4, &e))?;
    let mut beta = backend
        .stream
        .alloc_zeros::<f32>(head_len)
        .map_err(|e| cuda_alloc_failed_rich(backend, "la beta", head_len * 4, &e))?;
    let mut a_inv = backend
        .stream
        .alloc_zeros::<u16>(a_len)
        .map_err(|e| cuda_alloc_failed_rich(backend, "la a_inv", a_len * 2, &e))?;
    let mut chunk_state = backend
        .stream
        .alloc_zeros::<f32>(chunk_state_len)
        .map_err(|e| cuda_alloc_failed_rich(backend, "la chunk_state", chunk_state_len * 4, &e))?;
    let mut final_state = backend
        .stream
        .alloc_zeros::<f32>(state_len)
        .map_err(|e| cuda_alloc_failed_rich(backend, "la final_state", state_len * 4, &e))?;
    // Seed carry so chunk_state[0] = carry. FlashQLA reads it as h0 directly; the
    // recurrent branch runs final_state → chunk_state[0].
    if let Some(state) = carry_state {
        let dst = if use_chunkwise {
            &mut chunk_state
        } else {
            &mut final_state
        };
        backend
            .stream
            .memcpy_dtod(state, dst)
            .map_err(|_| AutogradError::TapeInvariant("cuda D2D copy failed (la carry seed)"))?;
    }
    let mut raw_output = backend
        .stream
        .alloc_zeros::<u16>(v_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la raw_output)"))?;
    let mut output = backend
        .stream
        .alloc_zeros::<f32>(z_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la output)"))?;

    let seq_len_i32 = i32::try_from(p.seq_len)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention seq_len exceeds i32"))?;
    let qkv_dim_i32 = i32::try_from(qkv_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention qkv_dim exceeds i32"))?;
    let conv_kernel_i32 = i32::try_from(p.conv_kernel)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention conv_kernel exceeds i32"))?;
    let num_key_heads_i32 = i32::try_from(p.num_key_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key heads exceeds i32"))?;
    let num_value_heads_i32 = i32::try_from(p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value heads exceeds i32"))?;
    let rows_i32 = i32::try_from(p.seq_len * p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention rows exceeds i32"))?;
    let key_dim_i32 = i32::try_from(p.key_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key_dim exceeds i32"))?;
    let value_dim_i32 = i32::try_from(p.value_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value_dim exceeds i32"))?;

    launch_conv1d_silu(
        backend,
        &mut preact,
        &mut qkv_conv,
        &qkv,
        &conv1d_weight,
        carry_conv,
        conv_tail_len,
        ConvSiluDims {
            qkv_len,
            qkv_dim: qkv_dim_i32,
            seq_len: seq_len_i32,
            conv_kernel: conv_kernel_i32,
        },
    )?;
    if use_chunkwise {
        let fq = flashqla_gdr_symbols(p.num_value_heads, p.num_key_heads)?;
        {
            let stage_started = linear_attention_debug_stage_start();
            let (qkv_ptr, _qkv_guard) = qkv_conv.device_ptr(&backend.stream);
            let (b_ptr, _b_guard) = b_bf16.device_ptr(&backend.stream);
            let (a_ptr, _a_guard) = a_bf16.device_ptr(&backend.stream);
            let (dt_ptr, _dt_guard) = dt_bf16.device_ptr(&backend.stream);
            let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
            let (q_ptr, _q_guard) = q.device_ptr_mut(&backend.stream);
            let (k_ptr, _k_guard) = k.device_ptr_mut(&backend.stream);
            let (v_ptr, _v_guard) = v.device_ptr_mut(&backend.stream);
            let (g_ptr, _g_guard) = g.device_ptr_mut(&backend.stream);
            let (beta_ptr, _beta_guard) = beta.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: q/k are written [S, Hg, key_dim], exactly q_len on this route;
                // v/g/beta match v_len/head_len.
                unsafe {
                    ffi::gdr_fq_prep_cuda(
                        qkv_ptr as *const ffi::Half,
                        b_ptr as *const ffi::Half,
                        a_ptr as *const ffi::Half,
                        dt_ptr as *const ffi::Half,
                        a_log_ptr as *const f32,
                        q_ptr as *mut ffi::Half,
                        k_ptr as *mut ffi::Half,
                        v_ptr as *mut ffi::Half,
                        g_ptr as *mut f32,
                        beta_ptr as *mut f32,
                        num_key_heads_i32,
                        num_value_heads_i32,
                        key_dim_i32,
                        value_dim_i32,
                        seq_len_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gdr_fq_prep_cuda",
            )?;
            linear_attention_debug_stage_done(backend, "gdr_fq_prep", stage_started)?;
        }
        {
            let stage_started = linear_attention_debug_stage_start();
            let (g_ptr, _g_guard) = g.device_ptr(&backend.stream);
            let (gc_ptr, _gc_guard) = g_cumsum.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: g and g_cumsum are live guarded head_len f32 slices.
                unsafe {
                    (fq.cumsum)(
                        g_ptr as *const f32,
                        gc_ptr as *mut f32,
                        seq_len_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gdr_fq_cumsum",
            )?;
            linear_attention_debug_stage_done(backend, "gdr_fq_cumsum", stage_started)?;
        }
        {
            let stage_started = linear_attention_debug_stage_start();
            let (k_ptr, _k_guard) = k.device_ptr(&backend.stream);
            let (beta_ptr, _beta_guard) = beta.device_ptr(&backend.stream);
            let (a_inv_ptr, _a_inv_guard) = a_inv.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: k/beta are live guarded inputs; a_inv holds a_len = S*H*64 bf16.
                unsafe {
                    (fq.kkt)(
                        k_ptr as *const ffi::Half,
                        beta_ptr as *const f32,
                        a_inv_ptr as *mut ffi::Half,
                        seq_len_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gdr_fq_kkt",
            )?;
            linear_attention_debug_stage_done(backend, "gdr_fq_kkt", stage_started)?;
        }
        {
            let stage_started = linear_attention_debug_stage_start();
            let (q_ptr, _q_guard) = q.device_ptr(&backend.stream);
            let (k_ptr, _k_guard) = k.device_ptr(&backend.stream);
            let (v_ptr, _v_guard) = v.device_ptr(&backend.stream);
            let (a_inv_ptr, _a_inv_guard) = a_inv.device_ptr(&backend.stream);
            let (gc_ptr, _gc_guard) = g_cumsum.device_ptr(&backend.stream);
            let (beta_ptr, _beta_guard) = beta.device_ptr(&backend.stream);
            let (h0_ptr, _h0_guard) = chunk_state.device_ptr(&backend.stream);
            let (raw_ptr, _raw_guard) = raw_output.device_ptr_mut(&backend.stream);
            let (final_ptr, _final_guard) = final_state.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: chunk_state is state_len f32 here (the h0 carry) and final_state the
                // same extent; raw_output is v_len, the kernel's o write extent.
                unsafe {
                    (fq.fwd)(
                        q_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        v_ptr as *const ffi::Half,
                        a_inv_ptr as *const ffi::Half,
                        gc_ptr as *const f32,
                        beta_ptr as *const f32,
                        h0_ptr as *const f32,
                        raw_ptr as *mut ffi::Half,
                        final_ptr as *mut f32,
                        seq_len_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gdr_fq_fwd",
            )?;
            linear_attention_debug_stage_done(backend, "gdr_fq_fwd", stage_started)?;
        }
    } else {
        {
            let stage_started = linear_attention_debug_stage_start();
            let (qkv_ptr, _qkv_guard) = qkv_conv.device_ptr(&backend.stream);
            let (b_ptr, _b_guard) = b_bf16.device_ptr(&backend.stream);
            let (a_ptr, _a_guard) = a_bf16.device_ptr(&backend.stream);
            let (dt_ptr, _dt_guard) = dt_bf16.device_ptr(&backend.stream);
            let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
            let (q_ptr, _q_guard) = q.device_ptr_mut(&backend.stream);
            let (k_ptr, _k_guard) = k.device_ptr_mut(&backend.stream);
            let (v_ptr, _v_guard) = v.device_ptr_mut(&backend.stream);
            let (g_ptr, _g_guard) = g.device_ptr_mut(&backend.stream);
            let (beta_ptr, _beta_guard) = beta.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: reads qkv_conv/b/a/dt/a_log, writes q/k/v/g/beta — all live guarded
                // slices sized above (qkv_len / head_len / num_value_heads / q_len / v_len).
                unsafe {
                    ffi::gated_delta_rule_prefill_chunk_prepare_cuda(
                        qkv_ptr as *const ffi::Half,
                        b_ptr as *const ffi::Half,
                        a_ptr as *const ffi::Half,
                        dt_ptr as *const ffi::Half,
                        a_log_ptr as *const f32,
                        q_ptr as *mut ffi::Half,
                        k_ptr as *mut ffi::Half,
                        v_ptr as *mut ffi::Half,
                        g_ptr as *mut f32,
                        beta_ptr as *mut f32,
                        num_key_heads_i32,
                        num_value_heads_i32,
                        qkv_dim_i32,
                        seq_len_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gated_delta_rule_prefill_chunk_prepare_cuda",
            )?;
            linear_attention_debug_stage_done(backend, "gdr_prepare", stage_started)?;
        }
        let stage_started = linear_attention_debug_stage_start();
        let (qkv_ptr, _qkv_guard) = qkv_conv.device_ptr(&backend.stream);
        let (b_ptr, _b_guard) = b_bf16.device_ptr(&backend.stream);
        let (a_ptr, _a_guard) = a_bf16.device_ptr(&backend.stream);
        let (dt_ptr, _dt_guard) = dt_bf16.device_ptr(&backend.stream);
        let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
        let (state_ptr, _state_guard) = final_state.device_ptr_mut(&backend.stream);
        let (chunk_ptr, _chunk_guard) = chunk_state.device_ptr_mut(&backend.stream);
        let (raw_ptr, _raw_guard) = raw_output.device_ptr_mut(&backend.stream);
        let state_len_i32 = i32::try_from(state_len)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention state_len exceeds i32"))?;
        for chunk_idx in 0..num_chunks {
            let base = chunk_idx * 64;
            let chunk_len = (p.seq_len - base).min(64);
            let chunk_len_i32 = i32::try_from(chunk_len).map_err(|_| {
                AutogradError::TapeInvariant("linear_attention chunk_len exceeds i32")
            })?;
            // SAFETY: chunk_state has num_chunks*state_len f32s; chunk_idx < num_chunks.
            let dst_addr = unsafe { (chunk_ptr as *mut f32).add(chunk_idx * state_len) } as u64;
            let src_addr = state_ptr;
            launch_1d(
                &backend.stream,
                backend.kernels.function("linear_attention_copy_f32")?,
                state_len,
                |mut builder| {
                    builder.arg(&dst_addr).arg(&src_addr).arg(&state_len_i32);
                    builder
                },
            )?;
            check_cuda_ffi(
                // SAFETY: base = chunk_idx*64 with chunk_len = min(seq_len-base, 64), so the
                // offset qkv_conv/b/a/raw_output views and final_state (state_len) stay inside
                // their live guarded slices.
                unsafe {
                    ffi::gated_delta_rule_prefill_recurrent_cuda(
                        (qkv_ptr as *const ffi::Half).add(base * qkv_dim),
                        (b_ptr as *const ffi::Half).add(base * p.num_value_heads),
                        (a_ptr as *const ffi::Half).add(base * p.num_value_heads),
                        dt_ptr as *const ffi::Half,
                        a_log_ptr as *const f32,
                        state_ptr as *mut f32,
                        (raw_ptr as *mut ffi::Half).add(base * p.num_value_heads * p.value_dim),
                        num_key_heads_i32,
                        num_value_heads_i32,
                        key_dim_i32,
                        value_dim_i32,
                        chunk_len_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gated_delta_rule_prefill_recurrent_cuda",
            )?;
        }
        linear_attention_debug_stage_done(backend, "gdr_recurrent", stage_started)?;
    }
    {
        let stage_started = linear_attention_debug_stage_start();
        let (out_ptr, _out_guard) = output.device_ptr_mut(&backend.stream);
        let (raw_ptr, _raw_guard) = raw_output.device_ptr(&backend.stream);
        let (z_ptr, _z_guard) = z.device_ptr(&backend.stream);
        let (norm_ptr, _norm_guard) = norm_weight.device_ptr(&backend.stream);
        launch_rows(
            &backend.stream,
            backend
                .kernels
                .function("linear_attention_rms_gated_forward_f32_from_bf16")?,
            p.seq_len * p.num_value_heads,
            256,
            (256 * std::mem::size_of::<f32>()) as u32,
            |mut builder| {
                builder
                    .arg(&out_ptr)
                    .arg(&raw_ptr)
                    .arg(&z_ptr)
                    .arg(&norm_ptr)
                    .arg(&rows_i32)
                    .arg(&value_dim_i32)
                    .arg(&p.eps);
                builder
            },
        )?;
        linear_attention_debug_stage_done(backend, "rms_gated", stage_started)?;
    }

    Ok(LinearAttentionDeviceForwardResult {
        output: DeviceHandle::Cuda(CudaStorage::new(output)),
        preact: DeviceHandle::Cuda(CudaStorage::new(preact)),
        qkv_conv: DeviceHandle::CudaBf16(CudaBf16Storage::new(qkv_conv)),
        q: DeviceHandle::CudaBf16(CudaBf16Storage::new(q)),
        k: DeviceHandle::CudaBf16(CudaBf16Storage::new(k)),
        v: DeviceHandle::CudaBf16(CudaBf16Storage::new(v)),
        g: DeviceHandle::Cuda(CudaStorage::new(g)),
        g_cumsum: DeviceHandle::Cuda(CudaStorage::new(g_cumsum)),
        beta: DeviceHandle::Cuda(CudaStorage::new(beta)),
        a_inv: DeviceHandle::CudaBf16(CudaBf16Storage::new(a_inv)),
        chunk_state: DeviceHandle::Cuda(CudaStorage::new(chunk_state)),
        raw_output: DeviceHandle::CudaBf16(CudaBf16Storage::new(raw_output)),
        flashqla: use_chunkwise,
    })
}

pub(super) struct ConvSiluDims {
    pub qkv_len: usize,
    pub qkv_dim: i32,
    pub seq_len: i32,
    pub conv_kernel: i32,
}

/// Causal depthwise conv + SiLU, writing the f32 pre-activation and its bf16
/// SiLU together. The backward calls this to regenerate both instead of reading
/// them off the tape: at local seq 131,072 the pair is 6,720 MiB held from the
/// forward, against one depthwise-conv launch to rebuild.
pub(super) fn launch_conv1d_silu(
    backend: &CudaBackend,
    preact: &mut CudaSlice<f32>,
    qkv_conv: &mut CudaSlice<u16>,
    qkv: &CudaSlice<f32>,
    conv1d_weight: &CudaSlice<f32>,
    carry_conv: Option<&CudaSlice<f32>>,
    conv_tail_len: usize,
    dims: ConvSiluDims,
) -> Result<()> {
    let stage_started = linear_attention_debug_stage_start();
    let (preact_ptr, _preact_guard) = preact.device_ptr_mut(&backend.stream);
    let (qkv_conv_ptr, _qkv_conv_guard) = qkv_conv.device_ptr_mut(&backend.stream);
    let (qkv_ptr, _qkv_guard) = qkv.device_ptr(&backend.stream);
    let (conv_ptr, _conv_guard) = conv1d_weight.device_ptr(&backend.stream);
    // conv_tail = carried boundary window (nullptr → default zero-tap path, byte-identical).
    let conv_tail = carry_conv.map(|s| s.device_ptr(&backend.stream));
    let conv_tail_ptr = conv_tail.as_ref().map_or(0u64, |(ptr, _)| *ptr);
    let conv_tail_len_i32 = carry_conv
        .map(|_| i32::try_from(conv_tail_len))
        .transpose()
        .map_err(|_| AutogradError::TapeInvariant("linear_attention conv_tail_len exceeds i32"))?
        .unwrap_or(0);
    let total_u64 = u64::try_from(dims.qkv_len)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention qkv_len exceeds u64"))?;
    let (qkv_dim_i32, seq_len_i32, conv_kernel_i32) =
        (dims.qkv_dim, dims.seq_len, dims.conv_kernel);
    launch_1d(
        &backend.stream,
        backend
            .kernels
            .function("linear_attention_conv1d_silu_forward_f32_to_bf16")?,
        dims.qkv_len,
        |mut builder| {
            builder
                .arg(&preact_ptr)
                .arg(&qkv_conv_ptr)
                .arg(&qkv_ptr)
                .arg(&conv_ptr)
                .arg(&total_u64)
                .arg(&qkv_dim_i32)
                .arg(&seq_len_i32)
                .arg(&conv_kernel_i32)
                .arg(&conv_tail_ptr)
                .arg(&conv_tail_len_i32);
            builder
        },
    )?;
    linear_attention_debug_stage_done(backend, "conv1d_silu", stage_started)
}

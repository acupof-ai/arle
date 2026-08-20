use super::*;

#[cfg(not(feature = "no-cuda"))]

/// (H, Hg) is an AOT instantiation parameter; under context parallelism each
/// rank owns H/cp value heads and Hg/cp key heads, so the built set is one
/// geometry per (model, cp_size). The table is generated from kernels.toml.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn flashqla_gdr_symbols(h: usize, hg: usize) -> Result<&'static ffi::FlashqlaGdrSyms> {
    ffi::FLASHQLA_GDR_TABLE
        .iter()
        .find(|g| g.q_heads as usize == h && g.kv_heads as usize == hg)
        .ok_or_else(|| {
            let have = ffi::FLASHQLA_GDR_TABLE
                .iter()
                .map(|g| format!("{}/{}", g.q_heads, g.kv_heads))
                .collect::<Vec<_>>()
                .join(", ");
            AutogradError::TapeInvariant(Box::leak(
                format!("flashqla GDN head geometry H={h}/Hg={hg} not built (have {have})")
                    .into_boxed_str(),
            ))
        })
}

/// A/B escape hatch: force the legacy monolithic chunked-scan backward (one
/// block per batch x value_head) instead of the staged chunk-parallel path.
#[cfg(not(feature = "no-cuda"))]

/// Max concurrent chunk lanes in the stage-3 grad kernel. Bounds the per-block
/// history slab at `wave x rows x 64 x state_elems` f32 (1.6 GiB at 48 heads,
/// 128x128 state) independent of seq_len; 8 lanes x 48 rows = 384 blocks fills
/// H20's ~624-resident-block budget without oversubscribing it.
#[cfg(not(feature = "no-cuda"))]
pub(super) const LA_BWD_CHUNK_WAVE: usize = 8;

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_linear_attention_backward_device(
    backend: &CudaBackend,
    args: LinearAttentionDeviceBackwardArgs<'_>,
) -> Result<Option<LinearAttentionDeviceBackwardResult>> {
    let p = args.params;
    if !cuda_la_device_supported(p) {
        return Ok(None);
    }
    if p.batch == 1 {
        return cuda_linear_attention_backward_device_row(backend, args).map(Some);
    }

    // batch > 1: per-row dispatch to the proven batch==1 path. Upstream, inputs
    // and every saved ctx tensor are batch-leading contiguous rows; weights pass
    // through whole. Per-token grads concatenate; weight grads sum across rows.
    let qkv_dim = p.num_key_heads * p.key_dim * 2 + p.num_value_heads * p.value_dim;
    let conv_len = qkv_dim * p.conv_kernel;
    let row_params = LinearAttentionDeviceParams { batch: 1, ..p };
    let rows = (0..p.batch)
        .map(|row| {
            let slice = |src| cuda_row_slice(backend, src, row, p.batch);
            let initial_conv_window = args.initial_conv_window.map(slice).transpose()?;
            let raw_output = args.raw_output.map(slice).transpose()?;
            let g = args.g.map(slice).transpose()?;
            let beta = args.beta.map(slice).transpose()?;
            cuda_linear_attention_backward_device_row(
                backend,
                LinearAttentionDeviceBackwardArgs {
                    params: row_params,
                    upstream: &slice(args.upstream)?,
                    qkv: &slice(args.qkv)?,
                    z: &slice(args.z)?,
                    b_proj: &slice(args.b_proj)?,
                    a_proj: &slice(args.a_proj)?,
                    preact: &slice(args.preact)?,
                    qkv_conv: &slice(args.qkv_conv)?,
                    g: g.as_ref(),
                    beta: beta.as_ref(),
                    chunk_state: &slice(args.chunk_state)?,
                    raw_output: raw_output.as_ref(),
                    conv1d_weight: args.conv1d_weight,
                    dt_bias: args.dt_bias,
                    a_log: args.a_log,
                    norm_weight: args.norm_weight,
                    initial_conv_window: initial_conv_window.as_ref(),
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let concat = |field: fn(&LinearAttentionDeviceBackwardResult) -> &DeviceHandle| {
        let parts: Vec<&DeviceHandle> = rows.iter().map(field).collect();
        cuda_concat_rows(backend, &parts)
    };
    let sum = |field: fn(&LinearAttentionDeviceBackwardResult) -> &DeviceHandle, len: usize| {
        rows[1..]
            .iter()
            .try_fold(field(&rows[0]).clone(), |acc, row| {
                backend.add(&acc, field(row), &[len])
            })
    };
    Ok(Some(LinearAttentionDeviceBackwardResult {
        dqkv: concat(|r| &r.dqkv)?,
        dz: concat(|r| &r.dz)?,
        db: concat(|r| &r.db)?,
        da: concat(|r| &r.da)?,
        dconv: sum(|r| &r.dconv, conv_len)?,
        ddt: sum(|r| &r.ddt, p.num_value_heads)?,
        da_log: sum(|r| &r.da_log, p.num_value_heads)?,
        dnorm: sum(|r| &r.dnorm, p.value_dim)?,
    }))
}

/// Only FlashQLA re-derives g/beta; the recurrent routes read them off the tape.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn taped_g_beta<'a>(
    beta: Option<&'a CudaSlice<f32>>,
    g: Option<&'a CudaSlice<f32>>,
) -> Result<(&'a CudaSlice<f32>, &'a CudaSlice<f32>)> {
    beta.zip(g).ok_or(AutogradError::TapeInvariant(
        "linear_attention backward missing taped g/beta",
    ))
}

#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_linear_attention_backward_device_row(
    backend: &CudaBackend,
    args: LinearAttentionDeviceBackwardArgs<'_>,
) -> Result<LinearAttentionDeviceBackwardResult> {
    let p = args.params;
    debug_assert_eq!(p.batch, 1);
    let q_dim = p.num_key_heads * p.key_dim;
    let qkv_dim = q_dim * 2 + p.num_value_heads * p.value_dim;
    let qkv_len = p.batch * p.seq_len * qkv_dim;
    let z_len = p.batch * p.seq_len * p.num_value_heads * p.value_dim;
    let head_len = p.batch * p.seq_len * p.num_value_heads;
    let conv_len = qkv_dim * p.conv_kernel;
    let rows = p.batch * p.num_value_heads;
    let state_elems = p.key_dim * p.value_dim;
    let state_len = rows * state_elems;
    let num_chunks = p.seq_len.div_ceil(64);
    // FlashQLA saves only the carry; the per-chunk states are recomputed below.
    let chunk_state_len = if args.raw_output.is_some() {
        state_len
    } else {
        num_chunks * p.num_value_heads * state_elems
    };

    let upstream = backend.cuda_slice(args.upstream, "linear_attention_backward upstream")?;
    let qkv = backend.cuda_slice(args.qkv, "linear_attention_backward qkv")?;
    let z = backend.cuda_slice(args.z, "linear_attention_backward z")?;
    let a_proj = backend.cuda_slice(args.a_proj, "linear_attention_backward a_proj")?;
    let conv1d_weight = backend.cuda_slice(
        args.conv1d_weight,
        "linear_attention_backward conv1d_weight",
    )?;
    let dt_bias = backend.cuda_slice(args.dt_bias, "linear_attention_backward dt_bias")?;
    let a_log = backend.cuda_slice(args.a_log, "linear_attention_backward a_log")?;
    let norm_weight =
        backend.cuda_slice(args.norm_weight, "linear_attention_backward norm_weight")?;
    let preact = backend.cuda_slice(args.preact, "linear_attention_backward preact")?;
    let qkv_conv = backend.cuda_bf16_slice(args.qkv_conv, "linear_attention_backward qkv_conv")?;
    let beta = args
        .beta
        .map(|h| backend.cuda_slice(h, "linear_attention_backward beta"))
        .transpose()?;
    let g = args
        .g
        .map(|h| backend.cuda_slice(h, "linear_attention_backward g"))
        .transpose()?;
    let chunk_state =
        backend.cuda_slice(args.chunk_state, "linear_attention_backward chunk_state")?;

    for (label, got, expected) in [
        ("upstream", Some(upstream.len()), z_len),
        ("qkv", Some(qkv.len()), qkv_len),
        ("z", Some(z.len()), z_len),
        ("a_proj", Some(a_proj.len()), head_len),
        ("conv1d_weight", Some(conv1d_weight.len()), conv_len),
        ("dt_bias", Some(dt_bias.len()), p.num_value_heads),
        ("a_log", Some(a_log.len()), p.num_value_heads),
        ("norm_weight", Some(norm_weight.len()), p.value_dim),
        ("preact", Some(preact.len()), qkv_len),
        ("qkv_conv", Some(qkv_conv.len()), qkv_len),
        ("beta", beta.map(|s| s.len()), head_len),
        ("g", g.map(|s| s.len()), head_len),
        ("chunk_state", Some(chunk_state.len()), chunk_state_len),
    ] {
        if let Some(got) = got
            && got != expected
        {
            return Err(AutogradError::TapeInvariant(Box::leak(
                format!(
                    "cuda linear_attention_backward_device {label} len mismatch: got={got} expected={expected}"
                )
                .into_boxed_str(),
            )));
        }
    }

    let mut dqkv_conv = backend
        .stream
        .alloc_zeros::<f32>(qkv_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la dqkv_conv)"))?;
    let mut dz = backend
        .stream
        .alloc_zeros::<f32>(z_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la dz)"))?;
    let mut db = backend
        .stream
        .alloc_zeros::<f32>(head_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la db)"))?;
    let mut da = backend
        .stream
        .alloc_zeros::<f32>(head_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la da)"))?;
    let mut ddt = backend
        .stream
        .alloc_zeros::<f32>(p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la ddt)"))?;
    let mut da_log = backend
        .stream
        .alloc_zeros::<f32>(p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la da_log)"))?;
    let mut dnorm = backend
        .stream
        .alloc_zeros::<f32>(p.value_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la dnorm)"))?;
    let batch_i32 = i32::try_from(p.batch)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention batch exceeds i32"))?;
    let seq_len_i32 = i32::try_from(p.seq_len)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention seq_len exceeds i32"))?;
    let num_key_heads_i32 = i32::try_from(p.num_key_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key heads exceeds i32"))?;
    let num_value_heads_i32 = i32::try_from(p.num_value_heads)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value heads exceeds i32"))?;
    let key_dim_i32 = i32::try_from(p.key_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention key_dim exceeds i32"))?;
    let value_dim_i32 = i32::try_from(p.value_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention value_dim exceeds i32"))?;
    let qkv_dim_i32 = i32::try_from(qkv_dim)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention qkv_dim exceeds i32"))?;
    let total_u64 = u64::try_from(qkv_len)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention qkv_len exceeds u64"))?;
    let conv_kernel_i32 = i32::try_from(p.conv_kernel)
        .map_err(|_| AutogradError::TapeInvariant("linear_attention conv_kernel exceeds i32"))?;
    let carry_len = num_chunks * rows * state_elems;
    let staged_elems = carry_len.saturating_mul(3).saturating_add(
        num_chunks
            .saturating_mul(rows)
            .saturating_mul(p.key_dim)
            .saturating_mul(p.key_dim),
    );
    let staged_bytes = staged_elems.saturating_mul(std::mem::size_of::<f32>());
    let use_mono = args.raw_output.is_none()
        && (crate::runtime_flags::la_backward_mono()
            || staged_bytes > backend.mem_get_info().map_or(0, |(free, _)| free) / 2);

    if args.raw_output.is_some() {
        let fq = flashqla_gdr_symbols(p.num_value_heads, p.num_key_heads)?;
        let raw_output = args
            .raw_output
            .ok_or(AutogradError::TapeInvariant(
                "flashqla linear_attention backward missing raw_output",
            ))
            .and_then(|h| backend.cuda_bf16_slice(h, "linear_attention_backward raw_output"))?;
        let b_proj = backend.cuda_slice(args.b_proj, "linear_attention_backward b_proj")?;
        let b_bf16 = backend.local_f32_as_bf16(b_proj, head_len)?;
        let a_bf16 = backend.local_f32_as_bf16(a_proj, head_len)?;
        let dt_bf16 = backend.local_f32_as_bf16(dt_bias, p.num_value_heads)?;

        // FlashQLA keeps q/k on the key-head axis; only dq/dk carry the value-head one.
        let q_len = p.seq_len * p.num_key_heads * p.key_dim;
        let v_len = p.seq_len * p.num_value_heads * p.value_dim;
        let a_len = p.seq_len * p.num_value_heads * 64;
        let alloc_u16 = |len: usize, what: &'static str| {
            backend
                .stream
                .alloc_zeros::<u16>(len)
                .map_err(|_| AutogradError::TapeInvariant(what))
        };
        let alloc_f32 = |len: usize, what: &'static str| {
            backend
                .stream
                .alloc_zeros::<f32>(len)
                .map_err(|_| AutogradError::TapeInvariant(what))
        };
        let mut q = alloc_u16(q_len, "cuda alloc_zeros failed (fq q)")?;
        let mut k = alloc_u16(q_len, "cuda alloc_zeros failed (fq k)")?;
        let mut v = alloc_u16(v_len, "cuda alloc_zeros failed (fq v)")?;
        let mut g_re = alloc_f32(head_len, "cuda alloc_zeros failed (fq g)")?;
        let mut beta_re = alloc_f32(head_len, "cuda alloc_zeros failed (fq beta)")?;
        let mut g_cumsum = alloc_f32(head_len, "cuda alloc_zeros failed (fq g_cumsum)")?;
        let mut a_inv = alloc_u16(a_len, "cuda alloc_zeros failed (fq a_inv)")?;
        let mut h_states = alloc_u16(
            num_chunks * p.num_value_heads * state_elems,
            "cuda alloc_zeros failed (fq h)",
        )?;
        let mut d_raw = alloc_u16(v_len, "cuda alloc_zeros failed (fq d_raw)")?;
        let mut dq = alloc_u16(v_len, "cuda alloc_zeros failed (fq dq)")?;
        let mut dk = alloc_u16(v_len, "cuda alloc_zeros failed (fq dk)")?;
        let mut dv = alloc_u16(v_len, "cuda alloc_zeros failed (fq dv)")?;
        let mut dg_cumsum = alloc_f32(head_len, "cuda alloc_zeros failed (fq dg)")?;
        let mut dbeta = alloc_f32(head_len, "cuda alloc_zeros failed (fq dbeta)")?;
        // No consumer chains a final-state gradient in yet: dht stays zero, dh0 is dropped.
        let dht = alloc_f32(state_len, "cuda alloc_zeros failed (fq dht)")?;
        let mut dh0 = alloc_f32(state_len, "cuda alloc_zeros failed (fq dh0)")?;

        {
            let (qkv_conv_ptr, _qkv_conv_guard) = qkv_conv.device_ptr(&backend.stream);
            let (b_ptr, _b_guard) = b_bf16.device_ptr(&backend.stream);
            let (a_ptr, _a_guard) = a_bf16.device_ptr(&backend.stream);
            let (dt_ptr, _dt_guard) = dt_bf16.device_ptr(&backend.stream);
            let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
            let (q_ptr, _q_guard) = q.device_ptr_mut(&backend.stream);
            let (k_ptr, _k_guard) = k.device_ptr_mut(&backend.stream);
            let (v_ptr, _v_guard) = v.device_ptr_mut(&backend.stream);
            let (g_ptr, _g_guard) = g_re.device_ptr_mut(&backend.stream);
            let (beta_ptr, _beta_guard) = beta_re.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: same shapes as the forward's prep — q/k [S,Hg,key_dim] = q_len,
                // v v_len, g/beta head_len.
                unsafe {
                    ffi::gdr_fq_prep_cuda(
                        qkv_conv_ptr as *const ffi::Half,
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
        }
        {
            let (g_ptr, _g_guard) = g_re.device_ptr(&backend.stream);
            let (gc_ptr, _gc_guard) = g_cumsum.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: both are live guarded head_len f32 slices.
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
        }
        {
            let (k_ptr, _k_guard) = k.device_ptr(&backend.stream);
            let (beta_ptr, _beta_guard) = beta_re.device_ptr(&backend.stream);
            let (a_inv_ptr, _a_inv_guard) = a_inv.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: a_inv holds a_len = S*H*64 bf16, the kernel's write extent.
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
        }
        {
            let (k_ptr, _k_guard) = k.device_ptr(&backend.stream);
            let (v_ptr, _v_guard) = v.device_ptr(&backend.stream);
            let (a_inv_ptr, _a_inv_guard) = a_inv.device_ptr(&backend.stream);
            let (gc_ptr, _gc_guard) = g_cumsum.device_ptr(&backend.stream);
            let (beta_ptr, _beta_guard) = beta_re.device_ptr(&backend.stream);
            let (h0_ptr, _h0_guard) = chunk_state.device_ptr(&backend.stream);
            let (h_ptr, _h_guard) = h_states.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: chunk_state is the state_len f32 carry (h0); h_states holds
                // num_chunks*H*key_dim*value_dim bf16, the kernel's per-chunk write extent.
                unsafe {
                    (fq.prepare_h)(
                        k_ptr as *const ffi::Half,
                        v_ptr as *const ffi::Half,
                        a_inv_ptr as *const ffi::Half,
                        gc_ptr as *const f32,
                        beta_ptr as *const f32,
                        h0_ptr as *const f32,
                        h_ptr as *mut ffi::Half,
                        seq_len_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gdr_fq_prepare_h",
            )?;
        }
        {
            let (d_raw_ptr, _d_raw_guard) = d_raw.device_ptr_mut(&backend.stream);
            let (dz_ptr, _dz_guard) = dz.device_ptr_mut(&backend.stream);
            let (dnorm_ptr, _dnorm_guard) = dnorm.device_ptr_mut(&backend.stream);
            let (raw_ptr, _raw_guard) = raw_output.device_ptr(&backend.stream);
            let (z_ptr, _z_guard) = z.device_ptr(&backend.stream);
            let (upstream_ptr, _upstream_guard) = upstream.device_ptr(&backend.stream);
            let (norm_ptr, _norm_guard) = norm_weight.device_ptr(&backend.stream);
            let rows_i32 = i32::try_from(p.seq_len * p.num_value_heads)
                .map_err(|_| AutogradError::TapeInvariant("linear_attention rows exceeds i32"))?;
            launch_rows(
                &backend.stream,
                backend
                    .kernels
                    .function("linear_attention_rms_gated_backward_f32_to_bf16")?,
                p.seq_len * p.num_value_heads,
                256,
                (256 * std::mem::size_of::<f32>()) as u32,
                |mut builder| {
                    builder
                        .arg(&d_raw_ptr)
                        .arg(&dz_ptr)
                        .arg(&dnorm_ptr)
                        .arg(&raw_ptr)
                        .arg(&z_ptr)
                        .arg(&upstream_ptr)
                        .arg(&norm_ptr)
                        .arg(&rows_i32)
                        .arg(&value_dim_i32)
                        .arg(&p.eps);
                    builder
                },
            )?;
        }
        {
            let (do_ptr, _do_guard) = d_raw.device_ptr(&backend.stream);
            let (dht_ptr, _dht_guard) = dht.device_ptr(&backend.stream);
            let (q_ptr, _q_guard) = q.device_ptr(&backend.stream);
            let (k_ptr, _k_guard) = k.device_ptr(&backend.stream);
            let (v_ptr, _v_guard) = v.device_ptr(&backend.stream);
            let (a_inv_ptr, _a_inv_guard) = a_inv.device_ptr(&backend.stream);
            let (gc_ptr, _gc_guard) = g_cumsum.device_ptr(&backend.stream);
            let (beta_ptr, _beta_guard) = beta_re.device_ptr(&backend.stream);
            let (h_ptr, _h_guard) = h_states.device_ptr(&backend.stream);
            let (dq_ptr, _dq_guard) = dq.device_ptr_mut(&backend.stream);
            let (dk_ptr, _dk_guard) = dk.device_ptr_mut(&backend.stream);
            let (dv_ptr, _dv_guard) = dv.device_ptr_mut(&backend.stream);
            let (dg_ptr, _dg_guard) = dg_cumsum.device_ptr_mut(&backend.stream);
            let (dbeta_ptr, _dbeta_guard) = dbeta.device_ptr_mut(&backend.stream);
            let (dh0_ptr, _dh0_guard) = dh0.device_ptr_mut(&backend.stream);
            check_cuda_ffi(
                // SAFETY: every input is a live guarded slice at the extent its shape implies;
                // dq/dk/dv are v_len bf16, dg/dbeta head_len f32, dh0 state_len f32.
                unsafe {
                    (fq.bwd)(
                        do_ptr as *const ffi::Half,
                        dht_ptr as *const f32,
                        q_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        v_ptr as *const ffi::Half,
                        a_inv_ptr as *const ffi::Half,
                        gc_ptr as *const f32,
                        beta_ptr as *const f32,
                        h_ptr as *const ffi::Half,
                        dq_ptr as *mut ffi::Half,
                        dk_ptr as *mut ffi::Half,
                        dv_ptr as *mut ffi::Half,
                        dg_ptr as *mut f32,
                        dbeta_ptr as *mut f32,
                        dh0_ptr as *mut f32,
                        seq_len_i32,
                        backend.stream.cu_stream(),
                    )
                },
                "gdr_fq_bwd",
            )?;
        }
        {
            let (dqkv_conv_ptr, _dqkv_conv_guard) = dqkv_conv.device_ptr_mut(&backend.stream);
            let (db_ptr, _db_guard) = db.device_ptr_mut(&backend.stream);
            let (da_ptr, _da_guard) = da.device_ptr_mut(&backend.stream);
            let (ddt_ptr, _ddt_guard) = ddt.device_ptr_mut(&backend.stream);
            let (da_log_ptr, _da_log_guard) = da_log.device_ptr_mut(&backend.stream);
            let (dq_ptr, _dq_guard) = dq.device_ptr(&backend.stream);
            let (dk_ptr, _dk_guard) = dk.device_ptr(&backend.stream);
            let (dv_ptr, _dv_guard) = dv.device_ptr(&backend.stream);
            let (dg_ptr, _dg_guard) = dg_cumsum.device_ptr(&backend.stream);
            let (dbeta_ptr, _dbeta_guard) = dbeta.device_ptr(&backend.stream);
            let (qkv_conv_ptr, _qkv_conv_guard) = qkv_conv.device_ptr(&backend.stream);
            let (a_proj_ptr, _a_proj_guard) = a_proj.device_ptr(&backend.stream);
            let (dt_ptr, _dt_guard) = dt_bias.device_ptr(&backend.stream);
            let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
            let (beta_ptr, _beta_guard) = beta_re.device_ptr(&backend.stream);
            launch_rows(
                &backend.stream,
                backend
                    .kernels
                    .function("linear_attention_gdr_prepare_backward_f32")?,
                num_chunks * rows,
                256,
                0,
                |mut builder| {
                    builder
                        .arg(&dqkv_conv_ptr)
                        .arg(&db_ptr)
                        .arg(&da_ptr)
                        .arg(&ddt_ptr)
                        .arg(&da_log_ptr)
                        .arg(&dq_ptr)
                        .arg(&dk_ptr)
                        .arg(&dv_ptr)
                        .arg(&dg_ptr)
                        .arg(&dbeta_ptr)
                        .arg(&qkv_conv_ptr)
                        .arg(&a_proj_ptr)
                        .arg(&dt_ptr)
                        .arg(&a_log_ptr)
                        .arg(&beta_ptr)
                        .arg(&batch_i32)
                        .arg(&seq_len_i32)
                        .arg(&num_key_heads_i32)
                        .arg(&num_value_heads_i32)
                        .arg(&key_dim_i32)
                        .arg(&value_dim_i32)
                        .arg(&qkv_dim_i32);
                    builder
                },
            )?;
        }
    } else if use_mono {
        let (beta, g) = taped_g_beta(beta, g)?;
        let mut grad_state_scratch = backend
            .stream
            .alloc_zeros::<f32>(state_len)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la grad_state)"))?;
        let mut state_recompute_scratch =
            backend.stream.alloc_zeros::<f32>(state_len).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (la state_recompute)")
            })?;
        let mut chunk_history_scratch = backend
            .stream
            .alloc_zeros::<f32>(rows * 64 * state_elems)
            .map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (la chunk_history)")
            })?;
        let mut chunk_kv_scratch = backend
            .stream
            .alloc_zeros::<f32>(rows * 64 * p.value_dim)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la chunk_kv)"))?;

        let (dqkv_conv_ptr, _dqkv_conv_guard) = dqkv_conv.device_ptr_mut(&backend.stream);
        let (dz_ptr, _dz_guard) = dz.device_ptr_mut(&backend.stream);
        let (db_ptr, _db_guard) = db.device_ptr_mut(&backend.stream);
        let (da_ptr, _da_guard) = da.device_ptr_mut(&backend.stream);
        let (ddt_ptr, _ddt_guard) = ddt.device_ptr_mut(&backend.stream);
        let (da_log_ptr, _da_log_guard) = da_log.device_ptr_mut(&backend.stream);
        let (dnorm_ptr, _dnorm_guard) = dnorm.device_ptr_mut(&backend.stream);
        let (grad_state_ptr, _grad_state_guard) =
            grad_state_scratch.device_ptr_mut(&backend.stream);
        let (state_recompute_ptr, _state_recompute_guard) =
            state_recompute_scratch.device_ptr_mut(&backend.stream);
        let (chunk_history_ptr, _chunk_history_guard) =
            chunk_history_scratch.device_ptr_mut(&backend.stream);
        let (chunk_kv_ptr, _chunk_kv_guard) = chunk_kv_scratch.device_ptr_mut(&backend.stream);
        let (upstream_ptr, _upstream_guard) = upstream.device_ptr(&backend.stream);
        let (z_ptr, _z_guard) = z.device_ptr(&backend.stream);
        let (a_proj_ptr, _a_proj_guard) = a_proj.device_ptr(&backend.stream);
        let (dt_ptr, _dt_guard) = dt_bias.device_ptr(&backend.stream);
        let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
        let (norm_ptr, _norm_guard) = norm_weight.device_ptr(&backend.stream);
        let (preact_ptr, _preact_guard) = preact.device_ptr(&backend.stream);
        let (qkv_conv_saved_ptr, _qkv_conv_saved_guard) = qkv_conv.device_ptr(&backend.stream);
        let (beta_ptr, _beta_guard) = beta.device_ptr(&backend.stream);
        let (g_ptr, _g_guard) = g.device_ptr(&backend.stream);
        let (chunk_state_ptr, _chunk_state_guard) = chunk_state.device_ptr(&backend.stream);

        launch_rows(
            &backend.stream,
            backend
                .kernels
                .function("linear_attention_chunked_scan_backward_f32")?,
            rows,
            256,
            0,
            |mut builder| {
                builder
                    .arg(&dqkv_conv_ptr)
                    .arg(&dz_ptr)
                    .arg(&db_ptr)
                    .arg(&da_ptr)
                    .arg(&ddt_ptr)
                    .arg(&da_log_ptr)
                    .arg(&dnorm_ptr)
                    .arg(&grad_state_ptr)
                    .arg(&state_recompute_ptr)
                    .arg(&chunk_history_ptr)
                    .arg(&chunk_kv_ptr)
                    .arg(&upstream_ptr)
                    .arg(&z_ptr)
                    .arg(&a_proj_ptr)
                    .arg(&dt_ptr)
                    .arg(&a_log_ptr)
                    .arg(&norm_ptr)
                    .arg(&preact_ptr)
                    .arg(&qkv_conv_saved_ptr)
                    .arg(&beta_ptr)
                    .arg(&g_ptr)
                    .arg(&chunk_state_ptr)
                    .arg(&batch_i32)
                    .arg(&seq_len_i32)
                    .arg(&num_key_heads_i32)
                    .arg(&num_value_heads_i32)
                    .arg(&key_dim_i32)
                    .arg(&value_dim_i32)
                    .arg(&qkv_dim_i32)
                    .arg(&p.eps);
                builder
            },
        )?;
    } else {
        let (beta, g) = taped_g_beta(beta, g)?;
        let rows_i32 = i32::try_from(rows)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention rows exceeds i32"))?;
        let num_chunks_i32 = i32::try_from(num_chunks)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention num_chunks exceeds i32"))?;
        let wave = num_chunks.min(LA_BWD_CHUNK_WAVE);
        let wave_i32 = i32::try_from(wave)
            .map_err(|_| AutogradError::TapeInvariant("linear_attention wave exceeds i32"))?;
        let grid = wave * rows;
        let mut g_in_scratch = backend
            .stream
            .alloc_zeros::<f32>(carry_len)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la g_in)"))?;

        if num_chunks > 1 {
            let mut m_scratch = backend
                .stream
                .alloc_zeros::<f32>(num_chunks * rows * p.key_dim * p.key_dim)
                .map_err(|_| {
                    AutogradError::TapeInvariant("cuda alloc_zeros failed (la transfer_m)")
                })?;
            let mut b_scratch = backend.stream.alloc_zeros::<f32>(carry_len).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (la transfer_b)")
            })?;
            let mut state_scratch = backend.stream.alloc_zeros::<f32>(carry_len).map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (la transfer_state)")
            })?;

            {
                let (m_ptr, _m_guard) = m_scratch.device_ptr_mut(&backend.stream);
                let (b_ptr, _b_guard) = b_scratch.device_ptr_mut(&backend.stream);
                let (state_ptr, _state_guard) = state_scratch.device_ptr_mut(&backend.stream);
                let (upstream_ptr, _upstream_guard) = upstream.device_ptr(&backend.stream);
                let (z_ptr, _z_guard) = z.device_ptr(&backend.stream);
                let (norm_ptr, _norm_guard) = norm_weight.device_ptr(&backend.stream);
                let (qkv_conv_saved_ptr, _qkv_conv_saved_guard) =
                    qkv_conv.device_ptr(&backend.stream);
                let (beta_ptr, _beta_guard) = beta.device_ptr(&backend.stream);
                let (g_ptr, _g_guard) = g.device_ptr(&backend.stream);
                let (chunk_state_ptr, _chunk_state_guard) = chunk_state.device_ptr(&backend.stream);
                launch_rows(
                    &backend.stream,
                    backend
                        .kernels
                        .function("linear_attention_chunk_transfer_f32")?,
                    num_chunks * rows,
                    256,
                    0,
                    |mut builder| {
                        builder
                            .arg(&m_ptr)
                            .arg(&b_ptr)
                            .arg(&state_ptr)
                            .arg(&upstream_ptr)
                            .arg(&z_ptr)
                            .arg(&norm_ptr)
                            .arg(&qkv_conv_saved_ptr)
                            .arg(&beta_ptr)
                            .arg(&g_ptr)
                            .arg(&chunk_state_ptr)
                            .arg(&batch_i32)
                            .arg(&seq_len_i32)
                            .arg(&num_key_heads_i32)
                            .arg(&num_value_heads_i32)
                            .arg(&key_dim_i32)
                            .arg(&value_dim_i32)
                            .arg(&qkv_dim_i32)
                            .arg(&p.eps);
                        builder
                    },
                )?;
            }
            {
                let (g_in_ptr, _g_in_guard) = g_in_scratch.device_ptr_mut(&backend.stream);
                let (m_ptr, _m_guard) = m_scratch.device_ptr(&backend.stream);
                let (b_ptr, _b_guard) = b_scratch.device_ptr(&backend.stream);
                launch_rows(
                    &backend.stream,
                    backend
                        .kernels
                        .function("linear_attention_chunk_carry_f32")?,
                    rows,
                    256,
                    0,
                    |mut builder| {
                        builder
                            .arg(&g_in_ptr)
                            .arg(&m_ptr)
                            .arg(&b_ptr)
                            .arg(&rows_i32)
                            .arg(&num_chunks_i32)
                            .arg(&key_dim_i32)
                            .arg(&value_dim_i32);
                        builder
                    },
                )?;
            }
        }

        let mut grad_state_scratch = backend
            .stream
            .alloc_zeros::<f32>(grid * state_elems)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la grad_state)"))?;
        let mut chunk_history_scratch = backend
            .stream
            .alloc_zeros::<f32>(grid * 64 * state_elems)
            .map_err(|_| {
                AutogradError::TapeInvariant("cuda alloc_zeros failed (la chunk_history)")
            })?;
        let mut chunk_kv_scratch = backend
            .stream
            .alloc_zeros::<f32>(grid * 64 * p.value_dim)
            .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la chunk_kv)"))?;

        let (dqkv_conv_ptr, _dqkv_conv_guard) = dqkv_conv.device_ptr_mut(&backend.stream);
        let (dz_ptr, _dz_guard) = dz.device_ptr_mut(&backend.stream);
        let (db_ptr, _db_guard) = db.device_ptr_mut(&backend.stream);
        let (da_ptr, _da_guard) = da.device_ptr_mut(&backend.stream);
        let (ddt_ptr, _ddt_guard) = ddt.device_ptr_mut(&backend.stream);
        let (da_log_ptr, _da_log_guard) = da_log.device_ptr_mut(&backend.stream);
        let (dnorm_ptr, _dnorm_guard) = dnorm.device_ptr_mut(&backend.stream);
        let (grad_state_ptr, _grad_state_guard) =
            grad_state_scratch.device_ptr_mut(&backend.stream);
        let (chunk_history_ptr, _chunk_history_guard) =
            chunk_history_scratch.device_ptr_mut(&backend.stream);
        let (chunk_kv_ptr, _chunk_kv_guard) = chunk_kv_scratch.device_ptr_mut(&backend.stream);
        let (upstream_ptr, _upstream_guard) = upstream.device_ptr(&backend.stream);
        let (z_ptr, _z_guard) = z.device_ptr(&backend.stream);
        let (a_proj_ptr, _a_proj_guard) = a_proj.device_ptr(&backend.stream);
        let (dt_ptr, _dt_guard) = dt_bias.device_ptr(&backend.stream);
        let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&backend.stream);
        let (norm_ptr, _norm_guard) = norm_weight.device_ptr(&backend.stream);
        let (qkv_conv_saved_ptr, _qkv_conv_saved_guard) = qkv_conv.device_ptr(&backend.stream);
        let (beta_ptr, _beta_guard) = beta.device_ptr(&backend.stream);
        let (g_ptr, _g_guard) = g.device_ptr(&backend.stream);
        let (chunk_state_ptr, _chunk_state_guard) = chunk_state.device_ptr(&backend.stream);
        let (g_in_ptr, _g_in_guard) = g_in_scratch.device_ptr(&backend.stream);

        launch_rows(
            &backend.stream,
            backend
                .kernels
                .function("linear_attention_chunk_grad_f32")?,
            grid,
            256,
            0,
            |mut builder| {
                builder
                    .arg(&dqkv_conv_ptr)
                    .arg(&dz_ptr)
                    .arg(&db_ptr)
                    .arg(&da_ptr)
                    .arg(&ddt_ptr)
                    .arg(&da_log_ptr)
                    .arg(&dnorm_ptr)
                    .arg(&grad_state_ptr)
                    .arg(&chunk_history_ptr)
                    .arg(&chunk_kv_ptr)
                    .arg(&upstream_ptr)
                    .arg(&z_ptr)
                    .arg(&a_proj_ptr)
                    .arg(&dt_ptr)
                    .arg(&a_log_ptr)
                    .arg(&norm_ptr)
                    .arg(&qkv_conv_saved_ptr)
                    .arg(&beta_ptr)
                    .arg(&g_ptr)
                    .arg(&chunk_state_ptr)
                    .arg(&g_in_ptr)
                    .arg(&batch_i32)
                    .arg(&seq_len_i32)
                    .arg(&num_key_heads_i32)
                    .arg(&num_value_heads_i32)
                    .arg(&key_dim_i32)
                    .arg(&value_dim_i32)
                    .arg(&qkv_dim_i32)
                    .arg(&wave_i32)
                    .arg(&p.eps);
                builder
            },
        )?;
    }

    let mut dqkv = backend
        .stream
        .alloc_zeros::<f32>(qkv_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la dqkv)"))?;
    let mut dconv = backend
        .stream
        .alloc_zeros::<f32>(conv_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda alloc_zeros failed (la dconv)"))?;
    {
        let (dqkv_ptr, _dqkv_guard) = dqkv.device_ptr_mut(&backend.stream);
        let (dconv_ptr, _dconv_guard) = dconv.device_ptr_mut(&backend.stream);
        let (dqkv_conv_ptr, _dqkv_conv_guard) = dqkv_conv.device_ptr(&backend.stream);
        let (preact_ptr, _preact_guard) = preact.device_ptr(&backend.stream);
        let (qkv_ptr, _qkv_guard) = qkv.device_ptr(&backend.stream);
        let (conv_ptr, _conv_guard) = conv1d_weight.device_ptr(&backend.stream);
        // conv_tail: carried boundary window (nullptr → zero-pad default). Its boundary
        // taps' grad_weight is real; grad_input stays off (carry frozen).
        let conv_tail = args
            .initial_conv_window
            .map(|h| backend.cuda_slice(h, "linear_attention_backward conv_tail"))
            .transpose()?;
        let conv_tail_dev = conv_tail.map(|s| s.device_ptr(&backend.stream));
        let conv_tail_ptr = conv_tail_dev.as_ref().map_or(0u64, |(ptr, _)| *ptr);
        let conv_tail_len_i32 = conv_tail
            .map(|_| i32::try_from(p.conv_kernel - 1))
            .transpose()
            .map_err(|_| {
                AutogradError::TapeInvariant("linear_attention conv_tail_len exceeds i32")
            })?
            .unwrap_or(0);
        launch_1d(
            &backend.stream,
            backend
                .kernels
                .function("linear_attention_conv1d_silu_backward_f32")?,
            qkv_len,
            |mut builder| {
                builder
                    .arg(&dqkv_ptr)
                    .arg(&dconv_ptr)
                    .arg(&dqkv_conv_ptr)
                    .arg(&preact_ptr)
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
    }

    Ok(LinearAttentionDeviceBackwardResult {
        dqkv: DeviceHandle::Cuda(CudaStorage::new(dqkv)),
        dz: DeviceHandle::Cuda(CudaStorage::new(dz)),
        db: DeviceHandle::Cuda(CudaStorage::new(db)),
        da: DeviceHandle::Cuda(CudaStorage::new(da)),
        dconv: DeviceHandle::Cuda(CudaStorage::new(dconv)),
        ddt: DeviceHandle::Cuda(CudaStorage::new(ddt)),
        da_log: DeviceHandle::Cuda(CudaStorage::new(da_log)),
        dnorm: DeviceHandle::Cuda(CudaStorage::new(dnorm)),
    })
}

/// GPU assist for the host-fallback backward — reachable only for shapes the
/// device path declines (non-128 head dims); kept for those, not a hot path.
#[cfg(not(feature = "no-cuda"))]
pub(super) fn cuda_linear_attention_scan_backward(
    backend: &CudaBackend,
    args: LinearAttentionScanBackwardArgs<'_>,
) -> Result<Option<LinearAttentionScanBackwardGrads>> {
    let p = args.params;
    const MAX_DIM: usize = 256;
    if p.key_dim > MAX_DIM || p.value_dim > MAX_DIM {
        return Ok(None);
    }

    let q_dim = p.num_key_heads * p.key_dim;
    let qkv_dim = q_dim * 2 + p.num_value_heads * p.value_dim;
    let z_dim = p.num_value_heads * p.value_dim;
    let qkv_len = p.batch * p.seq_len * qkv_dim;
    let z_len = p.batch * p.seq_len * z_dim;
    let head_len = p.batch * p.seq_len * p.num_value_heads;
    let state_len = p.batch * p.num_value_heads * p.key_dim * p.value_dim;
    let state_history_len = p.batch * p.seq_len * p.num_value_heads * p.key_dim * p.value_dim;

    for (label, got, expected) in [
        ("upstream", args.upstream.len(), z_len),
        ("z", args.z.len(), z_len),
        ("a_proj", args.a_proj.len(), head_len),
        ("dt_bias", args.dt_bias.len(), p.num_value_heads),
        ("a_log", args.a_log.len(), p.num_value_heads),
        ("norm_weight", args.norm_weight.len(), p.value_dim),
        ("preact", args.preact.len(), qkv_len),
        ("beta", args.beta.len(), head_len),
        ("exp_g", args.exp_g.len(), head_len),
        ("kv_mem", args.kv_mem.len(), z_len),
        ("state_history", args.state_history.len(), state_history_len),
        ("final_state", args.final_state.len(), state_len),
    ] {
        if got != expected {
            return Err(AutogradError::TapeInvariant(Box::leak(
                format!(
                    "cuda linear_attention_scan_backward {label} len mismatch: got={got} expected={expected}"
                )
                .into_boxed_str(),
            )));
        }
    }

    let d_upstream = backend.upload_slice(args.upstream, &[z_len])?;
    let d_z = backend.upload_slice(args.z, &[z_len])?;
    let d_a_proj = backend.upload_slice(args.a_proj, &[head_len])?;
    let d_dt_bias = backend.upload_slice(args.dt_bias, &[p.num_value_heads])?;
    let d_a_log = backend.upload_slice(args.a_log, &[p.num_value_heads])?;
    let d_norm_weight = backend.upload_slice(args.norm_weight, &[p.value_dim])?;
    let d_preact = backend.upload_slice(args.preact, &[qkv_len])?;
    let d_beta = backend.upload_slice(args.beta, &[head_len])?;
    let d_exp_g = backend.upload_slice(args.exp_g, &[head_len])?;
    let d_kv_mem = backend.upload_slice(args.kv_mem, &[z_len])?;
    let d_state_history = backend.upload_slice(args.state_history, &[state_history_len])?;
    let d_final_state = backend.upload_slice(args.final_state, &[state_len])?;

    let mut d_dqkv = backend.stream.alloc_zeros::<f32>(qkv_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention dqkv)")
    })?;
    let mut d_dz = backend.stream.alloc_zeros::<f32>(z_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention dz)")
    })?;
    let mut d_db = backend.stream.alloc_zeros::<f32>(head_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention db)")
    })?;
    let mut d_da = backend.stream.alloc_zeros::<f32>(head_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention da)")
    })?;
    let mut d_ddt = backend
        .stream
        .alloc_zeros::<f32>(p.num_value_heads)
        .map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention ddt)")
        })?;
    let mut d_da_log = backend
        .stream
        .alloc_zeros::<f32>(p.num_value_heads)
        .map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention da_log)")
        })?;
    let mut d_dnorm = backend
        .stream
        .alloc_zeros::<f32>(p.value_dim)
        .map_err(|_| {
            AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention dnorm)")
        })?;
    let mut d_grad_state = backend.stream.alloc_zeros::<f32>(state_len).map_err(|_| {
        AutogradError::TapeInvariant("cuda alloc_zeros failed (linear_attention grad_state)")
    })?;

    let rows = p.batch * p.num_value_heads;
    let batch_i32 = i32::try_from(p.batch)
        .map_err(|_| AutogradError::TapeInvariant("cuda linear_attention batch exceeds i32"))?;
    let seq_len_i32 = i32::try_from(p.seq_len)
        .map_err(|_| AutogradError::TapeInvariant("cuda linear_attention seq_len exceeds i32"))?;
    let key_heads_i32 = i32::try_from(p.num_key_heads).map_err(|_| {
        AutogradError::TapeInvariant("cuda linear_attention num_key_heads exceeds i32")
    })?;
    let value_heads_i32 = i32::try_from(p.num_value_heads).map_err(|_| {
        AutogradError::TapeInvariant("cuda linear_attention num_value_heads exceeds i32")
    })?;
    let key_dim_i32 = i32::try_from(p.key_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda linear_attention key_dim exceeds i32"))?;
    let value_dim_i32 = i32::try_from(p.value_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda linear_attention value_dim exceeds i32"))?;
    let qkv_dim_i32 = i32::try_from(qkv_dim)
        .map_err(|_| AutogradError::TapeInvariant("cuda linear_attention qkv_dim exceeds i32"))?;

    launch_rows(
        &backend.stream,
        backend
            .kernels
            .function("linear_attention_scan_backward_f32")?,
        rows,
        256,
        0,
        |mut builder| {
            builder
                .arg(&mut d_dqkv)
                .arg(&mut d_dz)
                .arg(&mut d_db)
                .arg(&mut d_da)
                .arg(&mut d_ddt)
                .arg(&mut d_da_log)
                .arg(&mut d_dnorm)
                .arg(&mut d_grad_state)
                .arg(&d_upstream)
                .arg(&d_z)
                .arg(&d_a_proj)
                .arg(&d_dt_bias)
                .arg(&d_a_log)
                .arg(&d_norm_weight)
                .arg(&d_preact)
                .arg(&d_beta)
                .arg(&d_exp_g)
                .arg(&d_kv_mem)
                .arg(&d_state_history)
                .arg(&d_final_state)
                .arg(&batch_i32)
                .arg(&seq_len_i32)
                .arg(&key_heads_i32)
                .arg(&value_heads_i32)
                .arg(&key_dim_i32)
                .arg(&value_dim_i32)
                .arg(&qkv_dim_i32)
                .arg(&p.eps);
            builder
        },
    )?;

    Ok(Some(LinearAttentionScanBackwardGrads {
        dqkv: cuda_readback_slice(backend, &d_dqkv, qkv_len, "linear_attention dqkv")?,
        dz: cuda_readback_slice(backend, &d_dz, z_len, "linear_attention dz")?,
        db: cuda_readback_slice(backend, &d_db, head_len, "linear_attention db")?,
        da: cuda_readback_slice(backend, &d_da, head_len, "linear_attention da")?,
        ddt: cuda_readback_slice(backend, &d_ddt, p.num_value_heads, "linear_attention ddt")?,
        da_log: cuda_readback_slice(
            backend,
            &d_da_log,
            p.num_value_heads,
            "linear_attention da_log",
        )?,
        dnorm: cuda_readback_slice(backend, &d_dnorm, p.value_dim, "linear_attention dnorm")?,
    }))
}

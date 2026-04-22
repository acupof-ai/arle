use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::model::qwen35::prefill_buffers::GdrChunkwiseScratch35;
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates};

/// GDR (gated delta rule) shared weights: `dt_bias` + `a_log` are reused across every GDR layer.
pub struct GdrWeights<'a> {
    pub dt_bias: &'a DeviceVec,
    pub a_log: &'a CudaSlice<f32>,
}

/// GDR head configuration: linear-attention head counts and per-head dims.
pub struct GdrHeadConfig {
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_dim: usize,
    pub val_dim: usize,
}

fn validate_packed_prefill_seq_indptr(
    op_name: &str,
    batch_size: usize,
    seq_indptr: &[i32],
) -> Result<()> {
    ensure!(
        batch_size > 0,
        "{op_name} packed prefill launch requires at least one request"
    );
    ensure!(
        seq_indptr.len() == batch_size + 1,
        "{op_name} packed prefill launch seq_indptr len {} must equal batch size {} + 1",
        seq_indptr.len(),
        batch_size
    );
    ensure!(
        seq_indptr.first().copied() == Some(0),
        "{op_name} packed prefill launch seq_indptr must start at 0"
    );
    for window in seq_indptr.windows(2) {
        ensure!(
            window[1] > window[0],
            "{op_name} packed prefill launch seq_indptr must be strictly increasing"
        );
    }
    Ok(())
}

/// Host-side launch metadata for packed multi-request conv1d prefill.
///
/// Packed activations stay contiguous on device. Per-request conv state remains
/// request-private, so the launch ABI is a host pointer array plus packed
/// `seq_indptr`.
#[allow(dead_code)]
pub(crate) struct Conv1dPrefillBatchLaunch<'a> {
    pub conv_state_ptrs: &'a [u64],
    /// Prefix sums over packed token rows. Length must be `batch_size + 1`.
    pub seq_indptr: &'a [i32],
}

#[allow(dead_code)]
impl<'a> Conv1dPrefillBatchLaunch<'a> {
    fn batch_size(&self) -> usize {
        self.conv_state_ptrs.len()
    }

    fn validate(&self) -> Result<()> {
        validate_packed_prefill_seq_indptr("conv1d", self.batch_size(), self.seq_indptr)
    }
}

/// Host-side launch metadata for packed multi-request GDR prefill.
///
/// The packed `qkv`/`b_proj`/`a_proj`/`output` tensors stay contiguous on the
/// device. Per-request recurrent state and chunkwise scratch remain
/// request-private, so launch metadata is a set of host pointer arrays plus a
/// packed `seq_indptr`.
#[allow(dead_code)]
pub(crate) struct GdrPrefillBatchLaunch<'a> {
    pub state_ptrs: &'a [u64],
    pub q_ptrs: &'a [u64],
    pub k_ptrs: &'a [u64],
    pub v_ptrs: &'a [u64],
    pub g_cumsum_ptrs: &'a [u64],
    pub beta_ptrs: &'a [u64],
    pub a_tril_ptrs: &'a [u64],
    pub a_inv_ptrs: &'a [u64],
    pub w_ptrs: &'a [u64],
    pub u_ptrs: &'a [u64],
    pub chunk_state_ptrs: &'a [u64],
    pub v_new_ptrs: &'a [u64],
    /// Prefix sums over packed token rows. Length must be `batch_size + 1`.
    pub seq_indptr: &'a [i32],
}

#[allow(dead_code)]
impl<'a> GdrPrefillBatchLaunch<'a> {
    fn batch_size(&self) -> usize {
        self.state_ptrs.len()
    }

    fn validate(&self) -> Result<()> {
        let batch_size = self.batch_size();
        ensure!(
            self.q_ptrs.len() == batch_size
                && self.k_ptrs.len() == batch_size
                && self.v_ptrs.len() == batch_size
                && self.g_cumsum_ptrs.len() == batch_size
                && self.beta_ptrs.len() == batch_size
                && self.a_tril_ptrs.len() == batch_size
                && self.a_inv_ptrs.len() == batch_size
                && self.w_ptrs.len() == batch_size
                && self.u_ptrs.len() == batch_size
                && self.chunk_state_ptrs.len() == batch_size
                && self.v_new_ptrs.len() == batch_size,
            "gdr packed prefill launch pointer arrays must all match batch size {}",
            batch_size
        );
        validate_packed_prefill_seq_indptr("gdr", batch_size, self.seq_indptr)
    }
}

/// Gated delta rule recurrent decode (single step, seq_len=1).
/// Fused CUDA kernel: L2-norm q/k, compute g/beta, decay + rank-1 state update, output.
/// ~15μs/layer on RTX 5070 Ti vs ~33μs for the 7-stage chunk-wise pipeline.
pub(crate) fn gated_delta_rule_decode_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    weights: &GdrWeights<'_>,
    state: &mut CudaSlice<f32>,
    output: &mut HiddenStates,
    heads: &GdrHeadConfig,
) -> Result<()> {
    debug_assert_eq!(qkv.seq_len, 1);
    debug_assert_eq!(b_proj.seq_len, 1);
    debug_assert_eq!(a_proj.seq_len, 1);
    debug_assert_eq!(output.seq_len, 1);

    let (qkv_ptr, _gq) = qkv.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_proj.data.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_proj.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = weights.dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = weights.a_log.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = state.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::gated_delta_rule_decode_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_ptr as *const ffi::Half,
            alog_ptr as *const f32,
            s_ptr as *mut f32,
            o_ptr as *mut ffi::Half,
            heads.num_key_heads as i32,
            heads.num_value_heads as i32,
            heads.key_dim as i32,
            heads.val_dim as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Batched conv1d decode: process B requests' conv1d in one kernel launch.
///
/// Per-request conv states are accessed via device pointer array — no gather/scatter.
/// Specialized for seq_len=1 (decode step).
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_decode_batch_into(
    ctx: &DeviceContext,
    x_batch: &HiddenStates,
    conv_weight: &DeviceVec,
    conv_state_ptrs: &mut CudaSlice<u64>, // device array of pointers to per-request conv states
    out_batch: &mut HiddenStates,
    kernel_size: usize,
    batch_size: usize,
) {
    let num_channels = x_batch.hidden_dim;
    debug_assert_eq!(out_batch.hidden_dim, num_channels);
    debug_assert!(batch_size <= x_batch.seq_len);
    debug_assert_eq!(conv_weight.len, num_channels * kernel_size);
    assert!(
        (2..=4).contains(&kernel_size),
        "conv1d_decode_batch kernel requires 2 <= kernel_size <= 4, got {kernel_size}"
    );

    let (x_ptr, _gx) = x_batch.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = conv_weight.data.device_ptr(&ctx.stream);
    let (sp_ptr, _gsp) = conv_state_ptrs.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = out_batch.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::conv1d_decode_batch_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            sp_ptr as *mut *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            num_channels as i32,
            kernel_size as i32,
            batch_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()
        .expect("conv1d_decode_batch_cuda failed");
    }
}

/// Batched GDR decode: process B requests' recurrent state update in one kernel launch.
///
/// Per-request recurrent states are accessed via device pointer array.
pub(crate) fn gdr_decode_batch_into(
    ctx: &DeviceContext,
    qkv_batch: &HiddenStates,
    b_proj_batch: &HiddenStates,
    a_proj_batch: &HiddenStates,
    weights: &GdrWeights<'_>,
    state_ptrs: &mut CudaSlice<u64>, // device array of pointers to per-request states (f32)
    output_batch: &mut HiddenStates,
    heads: &GdrHeadConfig,
    batch_size: usize,
) -> Result<()> {
    let (qkv_ptr, _gq) = qkv_batch.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_proj_batch.data.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_proj_batch.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = weights.dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = weights.a_log.device_ptr(&ctx.stream);
    let (sp_ptr, _gsp) = state_ptrs.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = output_batch.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::gdr_decode_batch_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_ptr as *const ffi::Half,
            alog_ptr as *const f32,
            sp_ptr as *mut *mut f32,
            o_ptr as *mut ffi::Half,
            heads.num_key_heads as i32,
            heads.num_value_heads as i32,
            heads.key_dim as i32,
            heads.val_dim as i32,
            batch_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Single-request causal depthwise conv1d prefill over one contiguous sequence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_prefill_batch_into(
    ctx: &DeviceContext,
    x_seq: &HiddenStates,
    conv_weight: &DeviceVec,
    conv_state: &mut DeviceVec,
    out_seq: &mut HiddenStates,
    kernel_size: usize,
) {
    let num_channels = x_seq.hidden_dim;
    assert_eq!(out_seq.hidden_dim, num_channels);
    assert_eq!(out_seq.seq_len, x_seq.seq_len);
    assert_eq!(conv_weight.len, num_channels * kernel_size);
    assert_eq!(conv_state.len, num_channels * (kernel_size - 1));

    let (x_ptr, _gx) = x_seq.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = conv_weight.data.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = conv_state.data.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = out_seq.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::conv1d_prefill_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            s_ptr as *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            num_channels as i32,
            x_seq.seq_len as i32,
            kernel_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()
        .expect("conv1d_prefill_cuda failed");
    }
}

/// Packed multi-request conv1d prefill.
///
/// This mirrors the packed GDR surface: packed activations stay contiguous,
/// while per-request conv state remains request-private and is addressed
/// through host pointer arrays plus `seq_indptr`.
#[allow(dead_code)]
pub(crate) fn conv1d_prefill_packed_batch_into(
    ctx: &DeviceContext,
    x_batch: &HiddenStates,
    conv_weight: &DeviceVec,
    launch: &Conv1dPrefillBatchLaunch<'_>,
    out_batch: &mut HiddenStates,
    kernel_size: usize,
) -> Result<()> {
    launch.validate()?;

    let num_channels = x_batch.hidden_dim;
    ensure!(
        x_batch.seq_len == out_batch.seq_len && out_batch.hidden_dim == num_channels,
        "conv1d packed prefill tensors must share the same packed shape"
    );
    ensure!(
        conv_weight.len == num_channels * kernel_size,
        "conv1d packed prefill weight len {} must equal num_channels {} * kernel_size {}",
        conv_weight.len,
        num_channels,
        kernel_size
    );
    ensure!(
        launch.seq_indptr.last().copied() == Some(out_batch.seq_len as i32),
        "conv1d packed prefill seq_indptr last {} must equal packed seq_len {}",
        launch.seq_indptr.last().copied().unwrap_or_default(),
        out_batch.seq_len
    );

    let (x_ptr, _gx) = x_batch.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = conv_weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out_batch.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::conv1d_prefill_packed_batch_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            launch.conv_state_ptrs.as_ptr(),
            launch.seq_indptr.as_ptr(),
            o_ptr as *mut ffi::Half,
            num_channels as i32,
            kernel_size as i32,
            launch.batch_size() as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

fn gated_delta_rule_prefill_chunk_prepare_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    weights: &GdrWeights<'_>,
    q_out: &mut HiddenStates,
    k_out: &mut HiddenStates,
    v_out: &mut HiddenStates,
    g_out: &mut CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    heads: &GdrHeadConfig,
) -> Result<()> {
    let (qkv_ptr, _gqkv) = qkv.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_proj.data.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_proj.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = weights.dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = weights.a_log.device_ptr(&ctx.stream);
    let (q_out_ptr, _gqo) = q_out.data.device_ptr_mut(&ctx.stream);
    let (k_out_ptr, _gko) = k_out.data.device_ptr_mut(&ctx.stream);
    let (v_out_ptr, _gvo) = v_out.data.device_ptr_mut(&ctx.stream);
    let (g_out_ptr, _ggo) = g_out.device_ptr_mut(&ctx.stream);
    let (beta_out_ptr, _gbetao) = beta_out.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_prepare_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_ptr as *const ffi::Half,
            alog_ptr as *const f32,
            q_out_ptr as *mut ffi::Half,
            k_out_ptr as *mut ffi::Half,
            v_out_ptr as *mut ffi::Half,
            g_out_ptr as *mut f32,
            beta_out_ptr as *mut f32,
            heads.num_key_heads as i32,
            heads.num_value_heads as i32,
            qkv.hidden_dim as i32,
            qkv.seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_cumsum_inplace(
    ctx: &DeviceContext,
    g_cumsum: &mut CudaSlice<f32>,
    seq_len: usize,
    num_value_heads: usize,
) -> Result<()> {
    let (g_ptr, _gg) = g_cumsum.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_cumsum_cuda(
            g_ptr as *const f32,
            g_ptr as *mut f32,
            seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_a_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    g_cumsum: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    a_tril: &mut CudaSlice<f32>,
    num_value_heads: usize,
) -> Result<()> {
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (beta_ptr, _gb) = beta.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_tril.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_a_cuda(
            k_ptr as *const ffi::Half,
            g_ptr as *const f32,
            beta_ptr as *const f32,
            a_ptr as *mut f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_solve_into(
    ctx: &DeviceContext,
    a_tril: &CudaSlice<f32>,
    a_inv: &mut CudaSlice<half::bf16>,
    seq_len: usize,
    num_value_heads: usize,
) -> Result<()> {
    let (a_ptr, _ga) = a_tril.device_ptr(&ctx.stream);
    let (ai_ptr, _gai) = a_inv.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_solve_cuda(
            a_ptr as *const f32,
            ai_ptr as *mut ffi::Half,
            seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_recompute_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    v: &HiddenStates,
    beta: &CudaSlice<f32>,
    w: &mut HiddenStates,
    u: &mut HiddenStates,
    a_inv: &CudaSlice<half::bf16>,
    g_cumsum: &CudaSlice<f32>,
    num_value_heads: usize,
) -> Result<()> {
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (beta_ptr, _gb) = beta.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = w.data.device_ptr_mut(&ctx.stream);
    let (u_ptr, _gu) = u.data.device_ptr_mut(&ctx.stream);
    let (ai_ptr, _gai) = a_inv.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_recompute_cuda(
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            beta_ptr as *const f32,
            w_ptr as *mut ffi::Half,
            u_ptr as *mut ffi::Half,
            ai_ptr as *const ffi::Half,
            g_ptr as *const f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_state_stage_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    w: &HiddenStates,
    u: &HiddenStates,
    g_cumsum: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    chunk_state: &mut CudaSlice<f32>,
    v_new: &mut HiddenStates,
    num_value_heads: usize,
) -> Result<()> {
    assert_eq!(k.hidden_dim, w.hidden_dim);
    assert_eq!(u.hidden_dim, v_new.hidden_dim);
    assert_eq!(k.seq_len, w.seq_len);
    assert_eq!(k.seq_len, u.seq_len);
    assert_eq!(k.seq_len, v_new.seq_len);

    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = w.data.device_ptr(&ctx.stream);
    let (u_ptr, _gu) = u.data.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = state.device_ptr_mut(&ctx.stream);
    let (cs_ptr, _gcs) = chunk_state.device_ptr_mut(&ctx.stream);
    let (vn_ptr, _gvn) = v_new.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_state_cuda(
            k_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            u_ptr as *const ffi::Half,
            g_ptr as *const f32,
            s_ptr as *const f32,
            cs_ptr as *mut f32,
            vn_ptr as *mut ffi::Half,
            s_ptr as *mut f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_o_stage_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v_new: &HiddenStates,
    chunk_state: &CudaSlice<f32>,
    g_cumsum: &CudaSlice<f32>,
    output: &mut HiddenStates,
    num_value_heads: usize,
    scale: f32,
) -> Result<()> {
    assert_eq!(q.hidden_dim, k.hidden_dim);
    assert_eq!(v_new.hidden_dim, output.hidden_dim);
    assert_eq!(q.seq_len, k.seq_len);
    assert_eq!(q.seq_len, v_new.seq_len);
    assert_eq!(q.seq_len, output.seq_len);

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (vn_ptr, _gvn) = v_new.data.device_ptr(&ctx.stream);
    let (cs_ptr, _gcs) = chunk_state.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_o_cuda(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            vn_ptr as *const ffi::Half,
            cs_ptr as *const f32,
            g_ptr as *const f32,
            o_ptr as *mut ffi::Half,
            q.seq_len as i32,
            num_value_heads as i32,
            scale,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Chunk-wise GDR prefill operator contract for Qwen3.5.
///
/// The chunk-wise path is an explicit multi-stage operator with pre-allocated
/// scratch instead of one opaque kernel launch.
pub fn gated_delta_rule_prefill_chunkwise_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    weights: &GdrWeights<'_>,
    state: &mut CudaSlice<f32>,
    scratch: &mut GdrChunkwiseScratch35,
    output: &mut HiddenStates,
    heads: &GdrHeadConfig,
) -> Result<()> {
    let num_value_heads = heads.num_value_heads;
    let key_dim = heads.key_dim;
    let val_dim = heads.val_dim;

    assert_eq!(scratch.q_expanded.seq_len, qkv.seq_len);
    assert_eq!(scratch.k_expanded.seq_len, qkv.seq_len);
    assert_eq!(scratch.v_raw.seq_len, qkv.seq_len);
    assert_eq!(scratch.w.seq_len, qkv.seq_len);
    assert_eq!(scratch.u.seq_len, qkv.seq_len);
    assert_eq!(scratch.v_new.seq_len, qkv.seq_len);
    assert_eq!(scratch.q_expanded.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.k_expanded.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.v_raw.hidden_dim, num_value_heads * val_dim);
    assert_eq!(scratch.w.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.u.hidden_dim, num_value_heads * val_dim);
    assert_eq!(scratch.v_new.hidden_dim, num_value_heads * val_dim);

    let expected_gate_len = qkv.seq_len * num_value_heads;
    let expected_chunk_a_len = qkv.seq_len * num_value_heads * GdrChunkwiseScratch35::CHUNK_SIZE;
    let expected_chunk_ai_len = expected_chunk_a_len;
    let expected_chunk_state_len =
        GdrChunkwiseScratch35::num_chunks(qkv.seq_len) * num_value_heads * val_dim * key_dim;
    assert_eq!(scratch.g_cumsum.len(), expected_gate_len);
    assert_eq!(scratch.beta.len(), expected_gate_len);
    assert_eq!(scratch.a_tril.len(), expected_chunk_a_len);
    assert_eq!(scratch.a_inv.len(), expected_chunk_ai_len);
    assert_eq!(scratch.chunk_state.len(), expected_chunk_state_len);

    gated_delta_rule_prefill_chunk_prepare_into(
        ctx,
        qkv,
        b_proj,
        a_proj,
        weights,
        &mut scratch.q_expanded,
        &mut scratch.k_expanded,
        &mut scratch.v_raw,
        &mut scratch.g_cumsum,
        &mut scratch.beta,
        heads,
    )?;
    gated_delta_rule_prefill_chunk_cumsum_inplace(
        ctx,
        &mut scratch.g_cumsum,
        qkv.seq_len,
        num_value_heads,
    )?;
    gated_delta_rule_prefill_chunk_a_into(
        ctx,
        &scratch.k_expanded,
        &scratch.g_cumsum,
        &scratch.beta,
        &mut scratch.a_tril,
        num_value_heads,
    )?;
    gated_delta_rule_prefill_chunk_solve_into(
        ctx,
        &scratch.a_tril,
        &mut scratch.a_inv,
        qkv.seq_len,
        num_value_heads,
    )?;
    gated_delta_rule_prefill_chunk_recompute_into(
        ctx,
        &scratch.k_expanded,
        &scratch.v_raw,
        &scratch.beta,
        &mut scratch.w,
        &mut scratch.u,
        &scratch.a_inv,
        &scratch.g_cumsum,
        num_value_heads,
    )?;
    gated_delta_rule_prefill_chunk_state_stage_into(
        ctx,
        &scratch.k_expanded,
        &scratch.w,
        &scratch.u,
        &scratch.g_cumsum,
        state,
        &mut scratch.chunk_state,
        &mut scratch.v_new,
        num_value_heads,
    )?;
    gated_delta_rule_prefill_chunk_o_stage_into(
        ctx,
        &scratch.q_expanded,
        &scratch.k_expanded,
        &scratch.v_new,
        &scratch.chunk_state,
        &scratch.g_cumsum,
        output,
        num_value_heads,
        1.0 / (key_dim as f32).sqrt(),
    )
}

/// Packed multi-request chunkwise GDR prefill.
///
/// This is the minimum stable ABI for future Qwen3.5 packed paged-prefill
/// wiring: packed activations stay contiguous, while per-request recurrent
/// state and scratch remain request-private and are addressed through host
/// pointer arrays plus `seq_indptr`.
#[allow(dead_code)]
pub(crate) fn gated_delta_rule_prefill_chunkwise_batch_into(
    ctx: &DeviceContext,
    qkv_batch: &HiddenStates,
    b_proj_batch: &HiddenStates,
    a_proj_batch: &HiddenStates,
    weights: &GdrWeights<'_>,
    launch: &GdrPrefillBatchLaunch<'_>,
    output_batch: &mut HiddenStates,
    heads: &GdrHeadConfig,
) -> Result<()> {
    launch.validate()?;

    ensure!(
        qkv_batch.seq_len == output_batch.seq_len
            && b_proj_batch.seq_len == output_batch.seq_len
            && a_proj_batch.seq_len == output_batch.seq_len,
        "gdr packed prefill tensors must share the same packed seq_len"
    );
    ensure!(
        launch.seq_indptr.last().copied() == Some(output_batch.seq_len as i32),
        "gdr packed prefill seq_indptr last {} must equal packed seq_len {}",
        launch.seq_indptr.last().copied().unwrap_or_default(),
        output_batch.seq_len
    );

    let (qkv_ptr, _gqkv) = qkv_batch.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_proj_batch.data.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_proj_batch.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = weights.dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = weights.a_log.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output_batch.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::gated_delta_rule_prefill_chunkwise_batch_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_ptr as *const ffi::Half,
            alog_ptr as *const f32,
            launch.state_ptrs.as_ptr(),
            launch.q_ptrs.as_ptr(),
            launch.k_ptrs.as_ptr(),
            launch.v_ptrs.as_ptr(),
            launch.g_cumsum_ptrs.as_ptr(),
            launch.beta_ptrs.as_ptr(),
            launch.a_tril_ptrs.as_ptr(),
            launch.a_inv_ptrs.as_ptr(),
            launch.w_ptrs.as_ptr(),
            launch.u_ptrs.as_ptr(),
            launch.chunk_state_ptrs.as_ptr(),
            launch.v_new_ptrs.as_ptr(),
            launch.seq_indptr.as_ptr(),
            o_ptr as *mut ffi::Half,
            heads.num_key_heads as i32,
            heads.num_value_heads as i32,
            heads.key_dim as i32,
            heads.val_dim as i32,
            launch.batch_size() as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

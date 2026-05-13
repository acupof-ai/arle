use anyhow::{Result, anyhow};
use cudarc::driver::{DevicePtr, DevicePtrMut};

use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates};

/// Batched element-wise add: out = a + b (same shape HiddenStates)
pub fn add_batch(ctx: &DeviceContext, a: &HiddenStates, b: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
    add_batch_into(ctx, a, b, &mut out)?;
    Ok(out)
}

/// Batched element-wise add into pre-allocated output buffer (zero allocation).
pub(crate) fn add_batch_into(
    ctx: &DeviceContext,
    a: &HiddenStates,
    b: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(a.hidden_dim, b.hidden_dim);
    assert_eq!(a.seq_len, b.seq_len);
    assert_eq!(out.hidden_dim, a.hidden_dim);
    assert_eq!(out.seq_len, a.seq_len);

    let n = a.hidden_dim * a.seq_len;
    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::add_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Batched SiLU+mul: `out[i] = silu(gate[i]) * up[i]`
pub fn silu_mul_batch(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, gate.hidden_dim, gate.seq_len)?;
    silu_mul_batch_into(ctx, gate, up, &mut out)?;
    Ok(out)
}

/// Batched SiLU+mul into pre-allocated output buffer (zero allocation).
pub(crate) fn silu_mul_batch_into(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(gate.hidden_dim, up.hidden_dim);
    assert_eq!(gate.seq_len, up.seq_len);
    assert_eq!(out.hidden_dim, gate.hidden_dim);
    assert_eq!(out.seq_len, gate.seq_len);

    let n = gate.hidden_dim * gate.seq_len;
    let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (u_ptr, _gu) = up.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::silu_mul_cuda(
            g_ptr as *const ffi::Half,
            u_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// DeepSeek V4 SwiGLU: `silu(min(gate, limit)) * clamp(up, -limit, limit)`.
pub(crate) fn dsv4_swiglu_clamped_batch_into(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
    out: &mut HiddenStates,
    limit: f32,
) -> Result<()> {
    assert_eq!(gate.hidden_dim, up.hidden_dim);
    assert_eq!(gate.seq_len, up.seq_len);
    assert_eq!(out.hidden_dim, gate.hidden_dim);
    assert_eq!(out.seq_len, gate.seq_len);

    let n = gate.hidden_dim * gate.seq_len;
    let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (u_ptr, _gu) = up.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::dsv4_swiglu_clamped_cuda(
            g_ptr as *const ffi::Half,
            u_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            limit,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Add one BF16 row into a token row of a HiddenStates batch:
/// `out[token_idx, i] += scale * row[0, i]`.
pub(crate) fn add_scaled_row_into(
    ctx: &DeviceContext,
    row: &HiddenStates,
    out: &mut HiddenStates,
    token_idx: usize,
    scale: f32,
) -> Result<()> {
    assert_eq!(row.hidden_dim, out.hidden_dim);
    assert_eq!(row.seq_len, 1);
    assert!(token_idx < out.seq_len);

    let (row_ptr, _gr) = row.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::add_scaled_row_cuda(
            row_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            out.hidden_dim as i32,
            token_idx as i32,
            scale,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Add one BF16 row into a segment of a token row:
/// `out[token_idx, segment_offset + i] += scale * row[0, i]`.
#[allow(dead_code)] // used by DeepSeek V4 layer-HC wiring once that tranche lands
pub(crate) fn add_scaled_row_segment_into(
    ctx: &DeviceContext,
    row: &HiddenStates,
    out: &mut HiddenStates,
    token_idx: usize,
    segment_offset: usize,
    scale: f32,
) -> Result<()> {
    assert_eq!(row.seq_len, 1);
    assert!(token_idx < out.seq_len);
    assert!(segment_offset + row.hidden_dim <= out.hidden_dim);

    let (row_ptr, _gr) = row.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::add_scaled_row_segment_cuda(
            row_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            row.hidden_dim as i32,
            out.hidden_dim as i32,
            token_idx as i32,
            segment_offset as i32,
            scale,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Batched SiLU+mul from a fused gate-up buffer.
///
/// `gate_up` stores each token row as `[gate, up]`, with
/// `gate_up.hidden_dim == 2 * out.hidden_dim`.
pub(crate) fn silu_mul_split_batch_into(
    ctx: &DeviceContext,
    gate_up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(gate_up.hidden_dim, out.hidden_dim * 2);
    assert_eq!(gate_up.seq_len, out.seq_len);

    let (gate_up_ptr, _ggu) = gate_up.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::silu_mul_fused_cuda(
            gate_up_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            gate_up.seq_len as i32,
            out.hidden_dim as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Extract a single token's vector from a HiddenStates batch (GPU copy)
pub(crate) fn extract_vec(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
) -> Result<DeviceVec> {
    let offset = token_idx * batch.hidden_dim;
    let len = batch.hidden_dim;
    let mut out = DeviceVec::zeros(ctx, len)?;

    let src_view = batch.data.slice(offset..offset + len);
    ctx.stream
        .memcpy_dtod(&src_view, &mut out.data)
        .map_err(|e| anyhow!("Device copy failed: {}", e))?;

    Ok(out)
}

/// Extract into a pre-allocated DeviceVec (zero-alloc D2D copy).
pub(crate) fn extract_vec_into(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    let offset = token_idx * batch.hidden_dim;
    let src_view = batch.data.slice(offset..offset + batch.hidden_dim);
    ctx.stream
        .memcpy_dtod(&src_view, &mut out.data)
        .map_err(|e| anyhow!("Device copy failed: {}", e))?;
    Ok(())
}

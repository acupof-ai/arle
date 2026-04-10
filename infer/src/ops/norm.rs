use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// RMSNorm into pre-allocated output buffer
pub fn rms_norm_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(x.len, out.len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// RMSNorm (allocating)
pub(crate) fn rms_norm(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
) -> Result<DeviceVec> {
    let mut out = DeviceVec::zeros(ctx, x.len)?;
    rms_norm_into(ctx, x, weight, eps, &mut out)?;
    Ok(out)
}

/// Fused add + RMSNorm: hidden += residual; out = rms_norm(hidden, weight)
/// Saves one global read of hidden compared to separate add + rms_norm.
pub fn fused_add_rms_norm_into(
    ctx: &DeviceContext,
    hidden: &mut DeviceVec,
    residual: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(hidden.len, residual.len);
    assert_eq!(hidden.len, out.len);
    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::fused_add_rms_norm_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden.len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Batched fused add + RMSNorm: hidden[b] += residual[b]; normed[b] = rms_norm(hidden[b], weight)
/// Saves one global read of hidden compared to separate add_batch_into + rms_norm_batch_into.
pub(crate) fn fused_add_rms_norm_batch_into(
    ctx: &DeviceContext,
    hidden: &mut HiddenStates,
    residual: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    normed: &mut HiddenStates,
) {
    assert_eq!(hidden.hidden_dim, residual.hidden_dim);
    assert_eq!(hidden.seq_len, residual.seq_len);
    assert_eq!(weight.len, hidden.hidden_dim);
    assert_eq!(normed.hidden_dim, hidden.hidden_dim);
    assert_eq!(normed.seq_len, hidden.seq_len);
    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = normed.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::fused_add_rms_norm_batched_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden.hidden_dim as i32,
            hidden.seq_len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()
        .expect("fused_add_rms_norm_batched_cuda failed");
    }
}

/// Batched RMSNorm into pre-allocated output buffer (zero allocation).
/// fp32-residual-shadow variant of batched RMSNorm.
/// Reads from a raw fp32 device buffer (shape `[seq_len, hidden_dim]`) and
/// writes bf16 `out` suitable for feeding downstream GEMMs. Used by Qwen3
/// prefill when the residual stream is maintained in fp32.
pub(crate) fn rms_norm_batch_f32_in_into(
    ctx: &DeviceContext,
    x_f32: &CudaSlice<f32>,
    weight: &DeviceVec,
    out: &mut HiddenStates,
    seq_len: usize,
    eps: f32,
) -> Result<()> {
    assert_eq!(x_f32.len(), out.hidden_dim * seq_len);
    let (x_ptr, _gx) = x_f32.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_batched_f32_in_cuda(
            x_ptr as *const f32,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            out.hidden_dim as i32,
            seq_len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Accumulate `b` (bf16) into `a` (fp32): `a[i] += f32(b[i])`.
pub(crate) fn add_bf16_into_f32(
    ctx: &DeviceContext,
    a_f32: &mut CudaSlice<f32>,
    b_bf16: &HiddenStates,
) -> Result<()> {
    let n = b_bf16.hidden_dim * b_bf16.seq_len;
    assert!(a_f32.len() >= n);
    let (a_ptr, _ga) = a_f32.device_ptr_mut(&ctx.stream);
    let (b_ptr, _gb) = b_bf16.data.device_ptr(&ctx.stream);
    unsafe {
        ffi::add_bf16_into_f32_cuda(
            a_ptr as *mut f32,
            b_ptr as *const ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

pub(crate) fn cast_bf16_to_f32(
    ctx: &DeviceContext,
    src: &HiddenStates,
    dst: &mut CudaSlice<f32>,
) -> Result<()> {
    let n = src.hidden_dim * src.seq_len;
    assert!(dst.len() >= n);
    let (s_ptr, _gs) = src.data.device_ptr(&ctx.stream);
    let (d_ptr, _gd) = dst.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::cast_bf16_to_f32_cuda(
            s_ptr as *const ffi::Half,
            d_ptr as *mut f32,
            n as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

pub(crate) fn cast_f32_to_bf16(
    ctx: &DeviceContext,
    src: &CudaSlice<f32>,
    dst: &mut HiddenStates,
) -> Result<()> {
    let n = dst.hidden_dim * dst.seq_len;
    assert!(src.len() >= n);
    let (s_ptr, _gs) = src.device_ptr(&ctx.stream);
    let (d_ptr, _gd) = dst.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::cast_f32_to_bf16_cuda(
            s_ptr as *const f32,
            d_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

pub(crate) fn rms_norm_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) {
    assert_eq!(weight.len, x.hidden_dim);
    assert_eq!(out.hidden_dim, x.hidden_dim);
    assert_eq!(out.seq_len, x.seq_len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_batched_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.hidden_dim as i32,
            x.seq_len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()
        .expect("rms_norm_batched_cuda failed");
    }
}

/// Batched (1+weight) RMSNorm over HiddenStates — one kernel launch for all tokens.
pub fn rms_norm_batch_offset_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(weight.len, x.hidden_dim);
    assert_eq!(out.hidden_dim, x.hidden_dim);
    assert_eq!(out.seq_len, x.seq_len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_batched_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.hidden_dim as i32,
            x.seq_len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// (1+weight) RMSNorm into pre-allocated output buffer (Gemma/Qwen3.5 style)
pub fn rms_norm_offset_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(x.len, out.len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Fused add + (1+weight) RMSNorm: hidden += residual; out = rms_norm_offset(hidden, weight)
pub fn fused_add_rms_norm_offset_into(
    ctx: &DeviceContext,
    hidden: &mut DeviceVec,
    residual: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(hidden.len, residual.len);
    assert_eq!(hidden.len, out.len);
    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::fused_add_rms_norm_offset_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden.len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Per-head RMSNorm with F32 weight + SiLU gate multiplication.
/// x: [num_heads * head_dim], weight: [head_dim] f32, gate: [num_heads * head_dim]
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gated_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &CudaSlice<f32>,
    gate: &DeviceVec,
    out: &mut DeviceVec,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<()> {
    assert_eq!(x.len, num_heads * head_dim);
    assert_eq!(gate.len, x.len);
    assert_eq!(out.len, x.len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_gated_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const f32,
            g_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            num_heads as i32,
            head_dim as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Batched per-head RMSNorm with F32 weight + SiLU gate multiplication.
/// HiddenStates are flattened as (seq_len * num_heads) contiguous head slices.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_norm_gated_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &CudaSlice<f32>,
    gate: &HiddenStates,
    out: &mut HiddenStates,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let total_heads = x.seq_len * num_heads;
    assert_eq!(x.hidden_dim, num_heads * head_dim);
    assert_eq!(gate.hidden_dim, x.hidden_dim);
    assert_eq!(gate.seq_len, x.seq_len);
    assert_eq!(out.hidden_dim, x.hidden_dim);
    assert_eq!(out.seq_len, x.seq_len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_gated_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const f32,
            g_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            total_heads as i32,
            head_dim as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()
        .expect("rms_norm_gated_cuda failed");
    }
}

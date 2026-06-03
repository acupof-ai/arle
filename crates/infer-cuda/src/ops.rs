//! CUDA op wrappers for the clean dense-BF16 Qwen3 forward (HOT axis).
//!
//! Thin, crate-private wrappers over `cuda-kernels` FFI entry points: embedding,
//! RMSNorm, GEMM/GEMV, elementwise add, SwiGLU, row copy, argmax, host uploads,
//! and the RoPE cache precompute. Pure relocation from `model.rs` — identical
//! numerics, identical FFI call sites.

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

pub(crate) fn precompute_rope(
    ctx: &DeviceContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    scaling: Option<&qwen3_spec::RopeScalingConfig>,
) -> Result<(DeviceVec, DeviceVec)> {
    let half_dim = head_dim / 2;
    let inv_freq = qwen3_spec::compute_scaled_inv_freq(head_dim, theta, scaling);
    let total = max_seq_len * head_dim;
    let mut cos_host = vec![bf16::ZERO; total];
    let mut sin_host = vec![bf16::ZERO; total];

    for pos in 0..max_seq_len {
        let base = pos * head_dim;
        for (i, freq) in inv_freq.iter().enumerate().take(half_dim) {
            let freq = pos as f32 * *freq;
            let cos_val = bf16::from_f32(freq.cos());
            let sin_val = bf16::from_f32(freq.sin());
            cos_host[base + i] = cos_val;
            cos_host[base + i + half_dim] = cos_val;
            sin_host[base + i] = sin_val;
            sin_host[base + i + half_dim] = sin_val;
        }
    }

    Ok((
        DeviceVec::from_host(ctx, &cos_host)?,
        DeviceVec::from_host(ctx, &sin_host)?,
    ))
}

pub(crate) fn upload_i32(ctx: &DeviceContext, values: &[i32]) -> Result<CudaSlice<i32>> {
    ctx.stream
        .clone_htod(values)
        .map_err(|e| anyhow!("H2D i32 upload failed: {e}"))
}

pub(crate) fn embedding_batch(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids: &CudaSlice<i32>,
    out: &mut HiddenStates,
) -> Result<()> {
    let (embed_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (token_ptr, _gt) = token_ids.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::embedding_batched_cuda(
            embed_ptr as *const ffi::Half,
            token_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            embed.cols as i32,
            out.seq_len as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

pub(crate) fn rms_norm_batch(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
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
        .result()?;
    }
    Ok(())
}

pub(crate) fn rms_norm_vec(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
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

pub(crate) fn gemm_batch(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        weight.cols == x.hidden_dim,
        "gemm input dim mismatch: weight cols {}, x hidden_dim {}",
        weight.cols,
        x.hidden_dim
    );
    ensure!(
        weight.rows == out.hidden_dim && x.seq_len == out.seq_len,
        "gemm output shape mismatch: weight rows {}, out hidden_dim {}, x seq {}, out seq {}",
        weight.rows,
        out.hidden_dim,
        x.seq_len,
        out.seq_len
    );
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::gemm_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            weight.rows as i32,
            x.seq_len as i32,
            weight.cols as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

pub(crate) fn gemv(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &DeviceVec,
    out: &mut DeviceVec,
) -> Result<()> {
    ensure!(
        weight.cols == x.len && weight.rows == out.len,
        "gemv shape mismatch: weight [{}x{}], x {}, out {}",
        weight.rows,
        weight.cols,
        x.len,
        out.len
    );
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::gemv_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            weight.rows as i32,
            weight.cols as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

pub(crate) fn add_batch(
    ctx: &DeviceContext,
    a: &HiddenStates,
    b: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        a.hidden_dim == b.hidden_dim
            && a.hidden_dim == out.hidden_dim
            && a.seq_len == b.seq_len
            && a.seq_len == out.seq_len,
        "add_batch shape mismatch"
    );
    let n = a.hidden_dim * a.seq_len;
    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::add_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

pub(crate) fn silu_mul(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        gate.hidden_dim == up.hidden_dim
            && gate.hidden_dim == out.hidden_dim
            && gate.seq_len == up.seq_len
            && gate.seq_len == out.seq_len,
        "silu_mul shape mismatch"
    );
    let n = gate.hidden_dim * gate.seq_len;
    let (gate_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (up_ptr, _gu) = up.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::silu_mul_cuda(
            gate_ptr as *const ffi::Half,
            up_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

pub(crate) fn copy_row_to_vec(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    ensure!(
        out.len == batch.hidden_dim,
        "copy_row_to_vec output len {} != hidden_dim {}",
        out.len,
        batch.hidden_dim
    );
    let offset = token_idx * batch.hidden_dim;
    let src = batch.data.slice(offset..offset + batch.hidden_dim);
    ctx.stream
        .memcpy_dtod(&src, &mut out.data)
        .map_err(|e| anyhow!("D2D copy last hidden row failed: {e}"))
}

pub(crate) fn argmax(ctx: &DeviceContext, logits: &DeviceVec) -> Result<u32> {
    let mut out = ctx
        .stream
        .alloc_zeros::<i32>(1)
        .map_err(|e| anyhow!("argmax output alloc failed: {e}"))?;
    {
        let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::argmax_cuda(
                logits_ptr as *const ffi::Half,
                out_ptr as *mut i32,
                logits.len as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    ctx.sync()?;
    let token = ctx
        .stream
        .clone_dtoh(&out)
        .map_err(|e| anyhow!("D2H argmax token failed: {e}"))?;
    Ok(token[0] as u32)
}

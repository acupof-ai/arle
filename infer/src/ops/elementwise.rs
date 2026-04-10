use anyhow::{Result, anyhow};
use cudarc::driver::{DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

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

/// Element-wise add for 1D vectors: out = a + b (in-place on `a`).
/// `a` is modified in-place. `b` must have the same length.
pub(crate) fn vec_add_inplace(ctx: &DeviceContext, a: &mut DeviceVec, b: &DeviceVec) -> Result<()> {
    assert_eq!(a.len, b.len, "vec_add_inplace: length mismatch");
    let n = a.len;
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
    // We need to copy `a` to a temp or use a fused add-into kernel.
    // Use add_cuda with out = a (overwrites a).
    let (a_ptr, _ga) = a.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::add_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

/// Element-wise add for 1D vectors on batched hidden states: out = a + bias
/// where bias has length `a.hidden_dim` and is broadcast across the batch.
pub(crate) fn add_bias_batch_into(
    ctx: &DeviceContext,
    a: &mut HiddenStates,
    bias: &DeviceVec,
) -> Result<()> {
    assert_eq!(
        a.hidden_dim, bias.len,
        "add_bias_batch: hidden_dim {} != bias len {}",
        a.hidden_dim, bias.len
    );
    let n = a.hidden_dim * a.seq_len;
    let (a_ptr, _ga) = a.data.device_ptr_mut(&ctx.stream);
    let (b_ptr, _gb) = bias.data.device_ptr(&ctx.stream);

    // For batched broadcast bias add, we need a kernel that adds bias[j] to
    // a[i * hidden_dim + j] for all i in [0, seq_len). The simple add_cuda
    // only works when both arrays have the same length.
    // For batch_size=1 (decode), hidden_dim == n, so add_cuda works.
    // For batch_size>1 (prefill/batch_decode), we need a broadcast kernel.
    // For now, only support batch_size=1 (decode path).
    if a.seq_len == 1 {
        let result = unsafe {
            ffi::add_cuda(
                a_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                a_ptr as *mut ffi::Half,
                n as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    } else {
        // For batched path: call add_cuda per token (safe, serialized on stream).
        for i in 0..a.seq_len {
            let offset = i * a.hidden_dim;
            let result = unsafe {
                ffi::add_cuda(
                    (a_ptr as usize + offset * 2) as *const ffi::Half,
                    b_ptr as *const ffi::Half,
                    (a_ptr as usize + offset * 2) as *mut ffi::Half,
                    a.hidden_dim as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result()?;
        }
    }
    Ok(())
}

/// Batched SiLU+mul: out[i] = silu(gate[i]) * up[i]
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
        ffi::silu_mul_triton_aot_cuda(
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

/// Split merged QKV buffer [B, q_dim + 2*kv_dim] into separate Q, K, V buffers.
/// Single kernel launch — no per-token D2D copies.
pub(crate) fn split_qkv_batch(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    q: &mut HiddenStates,
    k: &mut HiddenStates,
    v: &mut HiddenStates,
) -> Result<()> {
    let batch_size = qkv.seq_len;
    let q_dim = q.hidden_dim;
    let kv_dim = k.hidden_dim;
    debug_assert_eq!(qkv.hidden_dim, q_dim + 2 * kv_dim);

    let (qkv_ptr, _g0) = qkv.data.device_ptr(&ctx.stream);
    let (q_ptr, _g1) = q.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _g2) = k.data.device_ptr_mut(&ctx.stream);
    let (v_ptr, _g3) = v.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::split_qkv_cuda(
            qkv_ptr as *const ffi::Half,
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            v_ptr as *mut ffi::Half,
            batch_size as i32,
            q_dim as i32,
            kv_dim as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Qwen3.5 (Qwen3-Next) gated attention output split.
///
/// `q_full` has per-head concat layout
/// `[q_h0 | gate_h0 | q_h1 | gate_h1 | ... | q_h{H-1} | gate_h{H-1}]`
/// (each head owns `2 * head_dim` contiguous elements).
///
/// De-interleaves into `q [B, H*D]` and `gate [B, H*D]` — both
/// contiguous per head, ready for attention and sigmoid-gate respectively.
pub(crate) fn split_q_gate_batch(
    ctx: &DeviceContext,
    q_full: &HiddenStates,
    q: &mut HiddenStates,
    gate: &mut HiddenStates,
    num_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let batch_size = q_full.seq_len;
    let qh_dim = num_heads * head_dim;
    debug_assert_eq!(q_full.hidden_dim, 2 * qh_dim);
    debug_assert_eq!(q.hidden_dim, qh_dim);
    debug_assert_eq!(gate.hidden_dim, qh_dim);

    let (qf_ptr, _g0) = q_full.data.device_ptr(&ctx.stream);
    let (q_ptr, _g1) = q.data.device_ptr_mut(&ctx.stream);
    let (gate_ptr, _g2) = gate.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::split_q_gate_batch_cuda(
            qf_ptr as *const ffi::Half,
            q_ptr as *mut ffi::Half,
            gate_ptr as *mut ffi::Half,
            batch_size as i32,
            num_heads as i32,
            head_dim as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// In-place: `attn_out[i] *= sigmoid(gate[i])`. Shapes must match.
pub(crate) fn sigmoid_gate_mul_batch(
    ctx: &DeviceContext,
    attn_out: &mut HiddenStates,
    gate: &HiddenStates,
    num_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let batch_size = attn_out.seq_len;
    debug_assert_eq!(attn_out.hidden_dim, num_heads * head_dim);
    debug_assert_eq!(gate.hidden_dim, attn_out.hidden_dim);
    debug_assert_eq!(gate.seq_len, attn_out.seq_len);

    let (out_ptr, _g0) = attn_out.data.device_ptr_mut(&ctx.stream);
    let (gate_ptr, _g1) = gate.data.device_ptr(&ctx.stream);

    unsafe {
        ffi::sigmoid_gate_mul_batch_cuda(
            out_ptr as *mut ffi::Half,
            gate_ptr as *const ffi::Half,
            batch_size as i32,
            num_heads as i32,
            head_dim as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Fused SiLU-mul from merged gate+up buffer [B, 2*inter_dim].
/// out = silu(gate_up[..inter_dim]) * gate_up[inter_dim..]
pub(crate) fn silu_mul_fused_batch_into(
    ctx: &DeviceContext,
    gate_up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    let batch_size = gate_up.seq_len;
    let inter_dim = out.hidden_dim;
    debug_assert_eq!(gate_up.hidden_dim, 2 * inter_dim);

    let (gu_ptr, _g0) = gate_up.data.device_ptr(&ctx.stream);
    let (out_ptr, _g1) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::silu_mul_fused_cuda(
            gu_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            batch_size as i32,
            inter_dim as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

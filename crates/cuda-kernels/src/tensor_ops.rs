//! Embedding / norm / elementwise launch helpers.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi::{self, Half};
use crate::tensor::DeviceContext;

// Safe wrappers over the embedding/norm/elementwise FFI, per the `moe.rs` /
// `quant_linear.rs` pattern: typed buffers, checked i32 casts, pointer guards
// held through submission, one FFI symbol per launcher.

fn extent(a: usize, b: usize, what: &'static str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow!("{what} shape overflow: {a}x{b}"))
}

/// Batched embedding gather: out[s, :] = embed[token_ids[s], :].
pub fn embedding_batched(
    ctx: &DeviceContext,
    embed: &impl DevicePtr<bf16>,
    token_ids: &impl DevicePtr<i32>,
    out: &mut impl DevicePtrMut<bf16>,
    hidden_size: usize,
    seq_len: usize,
) -> Result<()> {
    ensure!(
        embed.len() >= hidden_size
            && token_ids.len() >= seq_len
            && out.len() >= extent(seq_len, hidden_size, "embedding_batched out")?,
        "embedding_batched buffers do not cover [seq,hidden]=[{seq_len},{hidden_size}]: embed={} token_ids={} out={}",
        embed.len(),
        token_ids.len(),
        out.len()
    );
    let (embed_ptr, _ge) = embed.device_ptr(&ctx.stream);
    let (token_ptr, _gt) = token_ids.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; token values index rows of the caller's
    // vocab-sized table.
    unsafe {
        ffi::embedding_batched_cuda(
            embed_ptr as *const Half,
            token_ptr as *const i32,
            out_ptr as *mut Half,
            i32::try_from(hidden_size)?,
            i32::try_from(seq_len)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("embedding_batched_cuda failed at [seq,hidden]=[{seq_len},{hidden_size}]: {e}")
        })
    }
}

/// Single-vector RMSNorm: out[n] = x[n] / rms(x) * weight[n].
pub fn rms_norm(
    ctx: &DeviceContext,
    x: &impl DevicePtr<bf16>,
    weight: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    n: usize,
    eps: f32,
) -> Result<()> {
    ensure!(
        x.len() >= n && weight.len() >= n && out.len() >= n,
        "rms_norm buffers do not cover n={n}: x={} weight={} out={}",
        x.len(),
        weight.len(),
        out.len()
    );
    let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::rms_norm_cuda(
            x_ptr as *const Half,
            w_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(n)?,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("rms_norm_cuda failed at n={n}: {e}"))
    }
}

/// Batched RMSNorm over `[seq_len, hidden_dim]` rows, reading `x` from element
/// `x_offset` (a row slice of a wider fused buffer; 0 for the whole buffer).
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_batched(
    ctx: &DeviceContext,
    x: &impl DevicePtr<bf16>,
    x_offset: usize,
    weight: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    hidden_dim: usize,
    seq_len: usize,
    eps: f32,
) -> Result<()> {
    let span = extent(seq_len, hidden_dim, "rms_norm_batched x")?;
    ensure!(
        x.len() >= x_offset.saturating_add(span) && weight.len() >= hidden_dim && out.len() >= span,
        "rms_norm_batched buffers do not cover [seq,hidden]=[{seq_len},{hidden_dim}]+offset {x_offset}: x={} weight={} out={}",
        x.len(),
        weight.len(),
        out.len()
    );
    let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: x_offset + seq*hidden bounded within x above.
    unsafe {
        ffi::rms_norm_batched_cuda(
            (x_ptr as *const Half).add(x_offset),
            w_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(hidden_dim)?,
            i32::try_from(seq_len)?,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("rms_norm_batched_cuda failed at [seq,hidden]=[{seq_len},{hidden_dim}]: {e}")
        })
    }
}

/// Batched offset RMSNorm (weights stored as `weight - 1`, kernel applies
/// `1 + weight`) over `[seq_len, hidden_dim]` rows.
pub fn rms_norm_batched_offset(
    ctx: &DeviceContext,
    x: &impl DevicePtr<bf16>,
    weight: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    hidden_dim: usize,
    seq_len: usize,
    eps: f32,
) -> Result<()> {
    let span = extent(seq_len, hidden_dim, "rms_norm_batched_offset x")?;
    ensure!(
        x.len() >= span && weight.len() >= hidden_dim && out.len() >= span,
        "rms_norm_batched_offset buffers do not cover [seq,hidden]=[{seq_len},{hidden_dim}]: x={} weight={} out={}",
        x.len(),
        weight.len(),
        out.len()
    );
    let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::rms_norm_batched_offset_cuda(
            x_ptr as *const Half,
            w_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(hidden_dim)?,
            i32::try_from(seq_len)?,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "rms_norm_batched_offset_cuda failed at [seq,hidden]=[{seq_len},{hidden_dim}]: {e}"
            )
        })
    }
}

/// Single-vector offset RMSNorm (`1 + weight`).
pub fn rms_norm_offset(
    ctx: &DeviceContext,
    x: &impl DevicePtr<bf16>,
    weight: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    n: usize,
    eps: f32,
) -> Result<()> {
    ensure!(
        x.len() >= n && weight.len() >= n && out.len() >= n,
        "rms_norm_offset buffers do not cover n={n}: x={} weight={} out={}",
        x.len(),
        weight.len(),
        out.len()
    );
    let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::rms_norm_offset_cuda(
            x_ptr as *const Half,
            w_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(n)?,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("rms_norm_offset_cuda failed at n={n}: {e}"))
    }
}

/// Gated per-head RMSNorm over `num_heads` slices of `head_dim`:
/// out = rms_norm(x) * weight * silu(gate). `weight` is a per-`[head_dim]`
/// f32 broadcast.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gated(
    ctx: &DeviceContext,
    x: &impl DevicePtr<bf16>,
    weight: &impl DevicePtr<f32>,
    gate: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<()> {
    let span = extent(num_heads, head_dim, "rms_norm_gated x")?;
    ensure!(
        x.len() >= span && weight.len() >= head_dim && gate.len() >= span && out.len() >= span,
        "rms_norm_gated buffers do not cover [heads,head_dim]=[{num_heads},{head_dim}]: x={} weight={} gate={} out={}",
        x.len(),
        weight.len(),
        gate.len(),
        out.len()
    );
    let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.device_ptr(&ctx.stream);
    let (gate_ptr, _gg) = gate.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::rms_norm_gated_cuda(
            x_ptr as *const Half,
            w_ptr as *const f32,
            gate_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(num_heads)?,
            i32::try_from(head_dim)?,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("rms_norm_gated_cuda failed at [heads,head_dim]=[{num_heads},{head_dim}]: {e}")
        })
    }
}

/// Elementwise add: out[n] = a[n] + b[n].
pub fn add(
    ctx: &DeviceContext,
    a: &impl DevicePtr<bf16>,
    b: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    n: usize,
) -> Result<()> {
    ensure!(
        a.len() >= n && b.len() >= n && out.len() >= n,
        "add buffers do not cover n={n}: a={} b={} out={}",
        a.len(),
        b.len(),
        out.len()
    );
    let (a_ptr, _ga) = a.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::add_cuda(
            a_ptr as *const Half,
            b_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(n)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("add_cuda failed at n={n}: {e}"))
    }
}

/// In-place scaled row add: out[token_idx, :] += scale * row[:hidden_dim].
pub fn add_scaled_row(
    ctx: &DeviceContext,
    row: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    hidden_dim: usize,
    token_idx: usize,
    scale: f32,
) -> Result<()> {
    ensure!(
        row.len() >= hidden_dim
            && out.len() >= extent(token_idx + 1, hidden_dim, "add_scaled_row out")?,
        "add_scaled_row buffers do not cover hidden_dim={hidden_dim} token_idx={token_idx}: row={} out={}",
        row.len(),
        out.len()
    );
    let (row_ptr, _gr) = row.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::add_scaled_row_cuda(
            row_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(hidden_dim)?,
            i32::try_from(token_idx)?,
            scale,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "add_scaled_row_cuda failed at hidden_dim={hidden_dim} token_idx={token_idx}: {e}"
            )
        })
    }
}

/// SwiGLU over split buffers: out[n] = silu(gate[n]) * up[n].
pub fn silu_mul(
    ctx: &DeviceContext,
    gate: &impl DevicePtr<bf16>,
    up: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    n: usize,
) -> Result<()> {
    ensure!(
        gate.len() >= n && up.len() >= n && out.len() >= n,
        "silu_mul buffers do not cover n={n}: gate={} up={} out={}",
        gate.len(),
        up.len(),
        out.len()
    );
    let (gate_ptr, _gg) = gate.device_ptr(&ctx.stream);
    let (up_ptr, _gu) = up.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::silu_mul_cuda(
            gate_ptr as *const Half,
            up_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(n)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("silu_mul_cuda failed at n={n}: {e}"))
    }
}

/// SwiGLU over a row-fused `[batch, 2*inter_dim]` gate_up buffer (gate = first
/// half of each row): out[batch, inter_dim] = silu(gate) * up.
pub fn silu_mul_fused(
    ctx: &DeviceContext,
    gate_up: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    batch_size: usize,
    inter_dim: usize,
) -> Result<()> {
    let out_span = extent(batch_size, inter_dim, "silu_mul_fused out")?;
    ensure!(
        gate_up.len() >= extent(out_span, 2, "silu_mul_fused gate_up")? && out.len() >= out_span,
        "silu_mul_fused buffers do not cover [batch,inter]=[{batch_size},{inter_dim}]: gate_up={} out={}",
        gate_up.len(),
        out.len()
    );
    let (gu_ptr, _gg) = gate_up.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; both buffers belong to `ctx.stream`.
    unsafe {
        ffi::silu_mul_fused_cuda(
            gu_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(batch_size)?,
            i32::try_from(inter_dim)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("silu_mul_fused_cuda failed at [batch,inter]=[{batch_size},{inter_dim}]: {e}")
        })
    }
}

/// Split a row-fused `[batch, first_dim + second_dim]` buffer into two buffers.
pub fn split2(
    ctx: &DeviceContext,
    fused: &impl DevicePtr<bf16>,
    first: &mut impl DevicePtrMut<bf16>,
    second: &mut impl DevicePtrMut<bf16>,
    batch_size: usize,
    first_dim: usize,
    second_dim: usize,
) -> Result<()> {
    let fused_dim = first_dim
        .checked_add(second_dim)
        .ok_or_else(|| anyhow!("split2 dim overflow: {first_dim}+{second_dim}"))?;
    ensure!(
        fused.len() >= extent(batch_size, fused_dim, "split2 fused")?
            && first.len() >= extent(batch_size, first_dim, "split2 first")?
            && second.len() >= extent(batch_size, second_dim, "split2 second")?,
        "split2 buffers do not cover [batch,first,second]=[{batch_size},{first_dim},{second_dim}]: fused={} first={} second={}",
        fused.len(),
        first.len(),
        second.len()
    );
    let (fused_ptr, _gf) = fused.device_ptr(&ctx.stream);
    let (first_ptr, _g1) = first.device_ptr_mut(&ctx.stream);
    let (second_ptr, _g2) = second.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::split2_cuda(
            fused_ptr as *const Half,
            first_ptr as *mut Half,
            second_ptr as *mut Half,
            i32::try_from(batch_size)?,
            i32::try_from(first_dim)?,
            i32::try_from(second_dim)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "split2_cuda failed at [batch,first,second]=[{batch_size},{first_dim},{second_dim}]: {e}"
            )
        })
    }
}

/// Split a row-fused `[batch, q_dim + 2*kv_dim]` qkv buffer into q/k/v buffers.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv(
    ctx: &DeviceContext,
    qkv: &impl DevicePtr<bf16>,
    q: &mut impl DevicePtrMut<bf16>,
    k: &mut impl DevicePtrMut<bf16>,
    v: &mut impl DevicePtrMut<bf16>,
    batch_size: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<()> {
    let qkv_dim = q_dim
        .checked_add(2 * kv_dim)
        .ok_or_else(|| anyhow!("split_qkv dim overflow: {q_dim}+2*{kv_dim}"))?;
    ensure!(
        qkv.len() >= extent(batch_size, qkv_dim, "split_qkv qkv")?
            && q.len() >= extent(batch_size, q_dim, "split_qkv q")?
            && k.len() >= extent(batch_size, kv_dim, "split_qkv k")?
            && v.len() >= extent(batch_size, kv_dim, "split_qkv v")?,
        "split_qkv buffers do not cover [batch,q,kv]=[{batch_size},{q_dim},{kv_dim}]: qkv={} q={} k={} v={}",
        qkv.len(),
        q.len(),
        k.len(),
        v.len()
    );
    let (qkv_ptr, _gf) = qkv.device_ptr(&ctx.stream);
    let (q_ptr, _g1) = q.device_ptr_mut(&ctx.stream);
    let (k_ptr, _g2) = k.device_ptr_mut(&ctx.stream);
    let (v_ptr, _g3) = v.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::split_qkv_cuda(
            qkv_ptr as *const Half,
            q_ptr as *mut Half,
            k_ptr as *mut Half,
            v_ptr as *mut Half,
            i32::try_from(batch_size)?,
            i32::try_from(q_dim)?,
            i32::try_from(kv_dim)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("split_qkv_cuda failed at [batch,q,kv]=[{batch_size},{q_dim},{kv_dim}]: {e}")
        })
    }
}

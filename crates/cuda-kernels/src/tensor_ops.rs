//! Embedding / norm / elementwise launch helpers.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut};
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

/// Expand token embeddings `[num_tokens, hidden_size]` into the initial wide
/// DSv4 hyper-connection stream `[num_tokens, hidden_size * hc_mult]` (each
/// lane seeded from the embedding).
pub fn dsv4_mhc_expand(
    ctx: &DeviceContext,
    embeddings: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    num_tokens: usize,
    hidden_size: usize,
    hc_mult: usize,
) -> Result<()> {
    let row = extent(hidden_size, hc_mult, "dsv4_mhc_expand row")?;
    ensure!(
        embeddings.len() >= extent(num_tokens, hidden_size, "dsv4_mhc_expand embeddings")?
            && out.len() >= extent(num_tokens, row, "dsv4_mhc_expand out")?,
        "dsv4_mhc_expand buffers do not cover [tokens,hidden,hc]=[{num_tokens},{hidden_size},{hc_mult}]: embeddings={} out={}",
        embeddings.len(),
        out.len()
    );
    let (emb_ptr, _ge) = embeddings.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; both buffers belong to `ctx.stream`.
    unsafe {
        ffi::dsv4_mhc_expand_cuda(
            emb_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(num_tokens)?,
            i32::try_from(hidden_size)?,
            i32::try_from(hc_mult)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "dsv4_mhc_expand_cuda failed at [tokens,hidden,hc]=[{num_tokens},{hidden_size},{hc_mult}]: {e}"
            )
        })
    }
}

/// Sinkhorn-normalized DSv4 hyper-connection mixer: per token, project the
/// wide residual stream + its `mixes` row through `base`/`scale` into the
/// per-lane `pre`/`post`/`comb` f32 mixing weights (`scale` carries the three
/// per-family scalars).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_mhc_params(
    ctx: &DeviceContext,
    residual: &impl DevicePtr<bf16>,
    mixes: &impl DevicePtr<bf16>,
    base: &impl DevicePtr<bf16>,
    scale: &impl DevicePtr<bf16>,
    pre: &mut impl DevicePtrMut<f32>,
    post: &mut impl DevicePtrMut<f32>,
    comb: &mut impl DevicePtrMut<f32>,
    num_tokens: usize,
    residual_hidden_dim: usize,
    mix_dim: usize,
    hc_mult: usize,
    eps: f32,
    sinkhorn_iters: usize,
) -> Result<()> {
    let lanes = extent(num_tokens, hc_mult, "dsv4_mhc_params lanes")?;
    ensure!(
        residual.len() >= extent(num_tokens, residual_hidden_dim, "dsv4_mhc_params residual")?
            && mixes.len() >= extent(num_tokens, mix_dim, "dsv4_mhc_params mixes")?
            && base.len() >= mix_dim
            && scale.len() >= 3
            && pre.len() >= lanes
            && post.len() >= lanes
            && comb.len() >= extent(lanes, hc_mult, "dsv4_mhc_params comb")?,
        "dsv4_mhc_params buffers do not cover [tokens,stream,mix,hc]=[{num_tokens},{residual_hidden_dim},{mix_dim},{hc_mult}]: residual={} mixes={} base={} scale={} pre={} post={} comb={}",
        residual.len(),
        mixes.len(),
        base.len(),
        scale.len(),
        pre.len(),
        post.len(),
        comb.len()
    );
    let (res_ptr, _gr) = residual.device_ptr(&ctx.stream);
    let (mix_ptr, _gm) = mixes.device_ptr(&ctx.stream);
    let (base_ptr, _gb) = base.device_ptr(&ctx.stream);
    let (scale_ptr, _gs) = scale.device_ptr(&ctx.stream);
    let (pre_ptr, _gp) = pre.device_ptr_mut(&ctx.stream);
    let (post_ptr, _gpo) = post.device_ptr_mut(&ctx.stream);
    let (comb_ptr, _gc) = comb.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::dsv4_mhc_params_cuda(
            res_ptr as *const Half,
            mix_ptr as *const Half,
            base_ptr as *const Half,
            scale_ptr as *const Half,
            pre_ptr as *mut f32,
            post_ptr as *mut f32,
            comb_ptr as *mut f32,
            i32::try_from(num_tokens)?,
            i32::try_from(residual_hidden_dim)?,
            i32::try_from(mix_dim)?,
            i32::try_from(hc_mult)?,
            eps,
            i32::try_from(sinkhorn_iters)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "dsv4_mhc_params_cuda failed at [tokens,stream,mix,hc]=[{num_tokens},{residual_hidden_dim},{mix_dim},{hc_mult}]: {e}"
            )
        })
    }
}

/// Collapse the wide DSv4 stream to one `hidden_size` row per token using the
/// pre-mix lane weights.
pub fn dsv4_mhc_pre(
    ctx: &DeviceContext,
    residual: &impl DevicePtr<bf16>,
    pre: &impl DevicePtr<f32>,
    out: &mut impl DevicePtrMut<bf16>,
    num_tokens: usize,
    hidden_size: usize,
    hc_mult: usize,
) -> Result<()> {
    let row = extent(hidden_size, hc_mult, "dsv4_mhc_pre row")?;
    ensure!(
        residual.len() >= extent(num_tokens, row, "dsv4_mhc_pre residual")?
            && pre.len() >= extent(num_tokens, hc_mult, "dsv4_mhc_pre pre")?
            && out.len() >= extent(num_tokens, hidden_size, "dsv4_mhc_pre out")?,
        "dsv4_mhc_pre buffers do not cover [tokens,hidden,hc]=[{num_tokens},{hidden_size},{hc_mult}]: residual={} pre={} out={}",
        residual.len(),
        pre.len(),
        out.len()
    );
    let (res_ptr, _gr) = residual.device_ptr(&ctx.stream);
    let (pre_ptr, _gp) = pre.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::dsv4_mhc_pre_cuda(
            res_ptr as *const Half,
            pre_ptr as *const f32,
            out_ptr as *mut Half,
            i32::try_from(num_tokens)?,
            i32::try_from(hidden_size)?,
            i32::try_from(hc_mult)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "dsv4_mhc_pre_cuda failed at [tokens,hidden,hc]=[{num_tokens},{hidden_size},{hc_mult}]: {e}"
            )
        })
    }
}

/// Fused [`dsv4_mhc_pre`] + RMSNorm: mix the stream into one lane and
/// normalize in a single launch.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_mhc_pre_rms_norm(
    ctx: &DeviceContext,
    residual: &impl DevicePtr<bf16>,
    pre: &impl DevicePtr<f32>,
    weight: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    num_tokens: usize,
    hidden_size: usize,
    hc_mult: usize,
    eps: f32,
) -> Result<()> {
    let row = extent(hidden_size, hc_mult, "dsv4_mhc_pre_rms_norm row")?;
    ensure!(
        residual.len() >= extent(num_tokens, row, "dsv4_mhc_pre_rms_norm residual")?
            && pre.len() >= extent(num_tokens, hc_mult, "dsv4_mhc_pre_rms_norm pre")?
            && weight.len() >= hidden_size
            && out.len() >= extent(num_tokens, hidden_size, "dsv4_mhc_pre_rms_norm out")?,
        "dsv4_mhc_pre_rms_norm buffers do not cover [tokens,hidden,hc]=[{num_tokens},{hidden_size},{hc_mult}]: residual={} pre={} weight={} out={}",
        residual.len(),
        pre.len(),
        weight.len(),
        out.len()
    );
    let (res_ptr, _gr) = residual.device_ptr(&ctx.stream);
    let (pre_ptr, _gp) = pre.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::dsv4_mhc_pre_rms_norm_cuda(
            res_ptr as *const Half,
            pre_ptr as *const f32,
            w_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(num_tokens)?,
            i32::try_from(hidden_size)?,
            i32::try_from(hc_mult)?,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "dsv4_mhc_pre_rms_norm_cuda failed at [tokens,hidden,hc]=[{num_tokens},{hidden_size},{hc_mult}]: {e}"
            )
        })
    }
}

/// Scatter a sub-block output back across the DSv4 stream lanes and re-mix the
/// residual with the post/comb weights.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_mhc_post(
    ctx: &DeviceContext,
    new_x: &impl DevicePtr<bf16>,
    residual: &impl DevicePtr<bf16>,
    post: &impl DevicePtr<f32>,
    comb: &impl DevicePtr<f32>,
    out: &mut impl DevicePtrMut<bf16>,
    num_tokens: usize,
    hidden_size: usize,
    hc_mult: usize,
) -> Result<()> {
    let row = extent(hidden_size, hc_mult, "dsv4_mhc_post row")?;
    let lanes = extent(num_tokens, hc_mult, "dsv4_mhc_post lanes")?;
    ensure!(
        new_x.len() >= extent(num_tokens, hidden_size, "dsv4_mhc_post new_x")?
            && residual.len() >= extent(num_tokens, row, "dsv4_mhc_post residual")?
            && post.len() >= lanes
            && comb.len() >= extent(lanes, hc_mult, "dsv4_mhc_post comb")?
            && out.len() >= extent(num_tokens, row, "dsv4_mhc_post out")?,
        "dsv4_mhc_post buffers do not cover [tokens,hidden,hc]=[{num_tokens},{hidden_size},{hc_mult}]: new_x={} residual={} post={} comb={} out={}",
        new_x.len(),
        residual.len(),
        post.len(),
        comb.len(),
        out.len()
    );
    let (new_ptr, _gn) = new_x.device_ptr(&ctx.stream);
    let (res_ptr, _gr) = residual.device_ptr(&ctx.stream);
    let (post_ptr, _gp) = post.device_ptr(&ctx.stream);
    let (comb_ptr, _gc) = comb.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::dsv4_mhc_post_cuda(
            new_ptr as *const Half,
            res_ptr as *const Half,
            post_ptr as *const f32,
            comb_ptr as *const f32,
            out_ptr as *mut Half,
            i32::try_from(num_tokens)?,
            i32::try_from(hidden_size)?,
            i32::try_from(hc_mult)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "dsv4_mhc_post_cuda failed at [tokens,hidden,hc]=[{num_tokens},{hidden_size},{hc_mult}]: {e}"
            )
        })
    }
}

/// Fold one wide stream row into a single `hidden_size` vector via the head
/// hyper-connection mixer (single-token variant of pre-mixing).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_mhc_head_pre(
    ctx: &DeviceContext,
    residual_row: &impl DevicePtr<bf16>,
    mixes: &impl DevicePtr<bf16>,
    base: &impl DevicePtr<bf16>,
    scale: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<bf16>,
    residual_hidden_dim: usize,
    hidden_size: usize,
    hc_mult: usize,
    eps: f32,
) -> Result<()> {
    ensure!(
        residual_row.len() >= residual_hidden_dim
            && mixes.len() >= hc_mult
            && base.len() >= hc_mult
            && scale.len() >= 1
            && out.len() >= hidden_size,
        "dsv4_mhc_head_pre buffers do not cover [stream,hidden,hc]=[{residual_hidden_dim},{hidden_size},{hc_mult}]: residual_row={} mixes={} base={} scale={} out={}",
        residual_row.len(),
        mixes.len(),
        base.len(),
        scale.len(),
        out.len()
    );
    let (row_ptr, _gr) = residual_row.device_ptr(&ctx.stream);
    let (mix_ptr, _gm) = mixes.device_ptr(&ctx.stream);
    let (base_ptr, _gb) = base.device_ptr(&ctx.stream);
    let (scale_ptr, _gs) = scale.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::dsv4_mhc_head_pre_cuda(
            row_ptr as *const Half,
            mix_ptr as *const Half,
            base_ptr as *const Half,
            scale_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(residual_hidden_dim)?,
            i32::try_from(hidden_size)?,
            i32::try_from(hc_mult)?,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "dsv4_mhc_head_pre_cuda failed at [stream,hidden,hc]=[{residual_hidden_dim},{hidden_size},{hc_mult}]: {e}"
            )
        })
    }
}

/// Per-token lane mean of a wide DSv4 stream, written into a strided column
/// window of `out`: for each token, average the `hc_mult` lanes of `lanes`
/// (read from element `lanes_offset`) into `out[token * out_stride +
/// out_col_offset ..][..hidden_size]`.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_mhc_lane_mean(
    ctx: &DeviceContext,
    lanes: &impl DevicePtr<bf16>,
    lanes_offset: usize,
    out: &mut impl DevicePtrMut<bf16>,
    num_tokens: usize,
    hidden_size: usize,
    hc_mult: usize,
    out_stride: usize,
    out_col_offset: usize,
) -> Result<()> {
    let row = extent(hidden_size, hc_mult, "dsv4_mhc_lane_mean row")?;
    let span = extent(num_tokens, row, "dsv4_mhc_lane_mean lanes")?;
    ensure!(
        lanes.len() >= lanes_offset.saturating_add(span)
            && out_col_offset.saturating_add(hidden_size) <= out_stride
            && out.len() >= extent(num_tokens, out_stride, "dsv4_mhc_lane_mean out")?,
        "dsv4_mhc_lane_mean buffers do not cover [tokens,hidden,hc]=[{num_tokens},{hidden_size},{hc_mult}]+offset {lanes_offset} stride {out_stride} col {out_col_offset}: lanes={} out={}",
        lanes.len(),
        out.len()
    );
    let (lanes_ptr, _gl) = lanes.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lanes_offset + tokens*hidden*hc bounded within lanes above; the
    // strided out window is bounded by out_stride.
    unsafe {
        ffi::dsv4_mhc_lane_mean_cuda(
            (lanes_ptr as *const Half).add(lanes_offset),
            out_ptr as *mut Half,
            i32::try_from(num_tokens)?,
            i32::try_from(hidden_size)?,
            i32::try_from(hc_mult)?,
            i32::try_from(out_stride)?,
            i32::try_from(out_col_offset)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "dsv4_mhc_lane_mean_cuda failed at [tokens,hidden,hc]=[{num_tokens},{hidden_size},{hc_mult}]: {e}"
            )
        })
    }
}

/// DSv4 MTP stream seed for one token row: out_stream[lane, :] =
/// e_proj[:] + h_proj[lane, :] for each of the `hc_mult` lanes, reading and
/// writing at the given element offsets (row slices of batched buffers).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_mtp_add_eproj_hproj(
    ctx: &DeviceContext,
    e_proj: &impl DevicePtr<bf16>,
    e_offset: usize,
    h_proj: &impl DevicePtr<bf16>,
    h_offset: usize,
    out_stream: &mut impl DevicePtrMut<bf16>,
    out_offset: usize,
    hidden_size: usize,
    hc_mult: usize,
) -> Result<()> {
    let row = extent(hidden_size, hc_mult, "dsv4_mtp_add_eproj_hproj row")?;
    ensure!(
        e_proj.len() >= e_offset.saturating_add(hidden_size)
            && h_proj.len() >= h_offset.saturating_add(row)
            && out_stream.len() >= out_offset.saturating_add(row),
        "dsv4_mtp_add_eproj_hproj buffers do not cover [hidden,hc]=[{hidden_size},{hc_mult}] at offsets [{e_offset},{h_offset},{out_offset}]: e_proj={} h_proj={} out={}",
        e_proj.len(),
        h_proj.len(),
        out_stream.len()
    );
    let (e_ptr, _ge) = e_proj.device_ptr(&ctx.stream);
    let (h_ptr, _gh) = h_proj.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out_stream.device_ptr_mut(&ctx.stream);
    // SAFETY: every offset + its row span is bounded within its buffer above.
    unsafe {
        ffi::dsv4_mtp_add_eproj_hproj_cuda(
            (e_ptr as *const Half).add(e_offset),
            (h_ptr as *const Half).add(h_offset),
            (out_ptr as *mut Half).add(out_offset),
            i32::try_from(hidden_size)?,
            i32::try_from(hc_mult)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "dsv4_mtp_add_eproj_hproj_cuda failed at [hidden,hc]=[{hidden_size},{hc_mult}]: {e}"
            )
        })
    }
}

/// Elementwise bf16 → f32 convert: dst[n] = f32(src[n]).
pub fn bf16_to_f32(
    ctx: &DeviceContext,
    src: &impl DevicePtr<bf16>,
    dst: &mut impl DevicePtrMut<f32>,
    n: usize,
) -> Result<()> {
    ensure!(
        src.len() >= n && dst.len() >= n,
        "bf16_to_f32 buffers do not cover n={n}: src={} dst={}",
        src.len(),
        dst.len()
    );
    let (src_ptr, _gs) = src.device_ptr(&ctx.stream);
    let (dst_ptr, _gd) = dst.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; both buffers belong to `ctx.stream`.
    unsafe {
        ffi::arle_bf16_to_f32_cuda(
            src_ptr as *const Half,
            dst_ptr as *mut f32,
            i32::try_from(n)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("arle_bf16_to_f32_cuda failed at n={n}: {e}"))
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

/// Dense BF16 GEMM over raw device addresses: `Y[n,m] = X[n,k] @ W[m,k]^T`
/// (`gemm_cuda`'s M is the weight-row dim). Raw-pointer variant for callers
/// that offset into a larger allocation (per-head / per-group blocks); typed
/// callers use [`crate::quant_linear::gemm_bf16`].
pub fn gemm_bf16_raw(
    stream: &CudaStream,
    w_ptr: u64,
    x_ptr: u64,
    y_ptr: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<()> {
    // SAFETY: the caller holds the originating allocations live on `stream` and
    // asserts they cover [m,k] / [n,k] / [n,m] bf16 at the given offsets.
    unsafe {
        ffi::gemm_cuda(
            w_ptr as *const Half,
            x_ptr as *const Half,
            y_ptr as *mut Half,
            m,
            n,
            k,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("gemm_cuda failed at [m,n,k]=[{m},{n},{k}]: {e}"))
    }
}

/// [`gemm_bf16_raw`] with an FP32 output row (`gemm_bf16_f32_cuda`): BF16
/// inputs, `Y[n,m]` f32.
pub fn gemm_bf16_f32_raw(
    stream: &CudaStream,
    w_ptr: u64,
    x_ptr: u64,
    y_ptr: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<()> {
    // SAFETY: the caller holds the originating allocations live on `stream` and
    // asserts they cover [m,k] / [n,k] bf16 and [n,m] f32 at the given offsets.
    unsafe {
        ffi::gemm_bf16_f32_cuda(
            w_ptr as *const Half,
            x_ptr as *const Half,
            y_ptr as *mut f32,
            m,
            n,
            k,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("gemm_bf16_f32_cuda failed at [m,n,k]=[{m},{n},{k}]: {e}"))
    }
}

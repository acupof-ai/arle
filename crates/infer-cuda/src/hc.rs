//! DSv4 hyper-connections (`hc_mult > 1`).
//!
//! The residual is a `hidden_size * hc_mult`-wide STREAM (`hc_mult` lanes per
//! token). Each attention/FFN sub-block is wrapped: a learned mixer projects the
//! stream into per-lane pre/post/combination weights (sinkhorn-normalized), the
//! pre weights collapse the stream to one `hidden_size` row for the sub-block,
//! and the post/comb weights scatter the sub-block output back across the lanes
//! and re-mix the residual. Reuses the shared `dsv4_mhc_*` kernels; the math is

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates};
use cuda_kernels::tensor_ops;
use cudarc::driver::CudaSlice;
use deepseek_spec::DeepSeekV4Config;

use crate::dsv4::Dsv4HyperConnection;

pub(crate) struct MhcParams {
    /// `[seq_len * hc_mult]` pre-mix lane weights (stream → hidden).
    pub pre: CudaSlice<f32>,
    /// `[seq_len * hc_mult]` post-mix lane weights (new hidden → stream lanes).
    pub post: CudaSlice<f32>,
    /// `[seq_len * hc_mult * hc_mult]` residual re-combination weights.
    pub comb: CudaSlice<f32>,
}

pub(crate) fn initial_stream_from_embeddings(
    ctx: &DeviceContext,
    embeddings: &HiddenStates,
    hidden_size: usize,
    hc_mult: usize,
    stream: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        embeddings.hidden_dim == hidden_size,
        "DSv4 HC expand embedding dim {} != hidden_size {hidden_size}",
        embeddings.hidden_dim
    );
    ensure!(hc_mult > 0, "DSv4 hc_mult must be non-zero");
    ensure!(
        stream.hidden_dim == hidden_size * hc_mult && stream.seq_len == embeddings.seq_len,
        "DSv4 HC stream shape {}x{} != {}x{}",
        stream.hidden_dim,
        stream.seq_len,
        hidden_size * hc_mult,
        embeddings.seq_len
    );
    tensor_ops::dsv4_mhc_expand(
        ctx,
        &embeddings.data,
        &mut stream.data,
        embeddings.seq_len,
        hidden_size,
        hc_mult,
    )
}

pub(crate) fn gen_mhc_params(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    hc: &Dsv4HyperConnection,
    stream: &HiddenStates,
) -> Result<MhcParams> {
    let hc_mult = config.hc_mult;
    ensure!(hc_mult > 0, "DSv4 MHC requires non-zero hc_mult");
    let mix_dim = (2 + hc_mult) * hc_mult;
    ensure!(
        hc.mix_fn.cols == stream.hidden_dim && hc.mix_fn.rows >= mix_dim,
        "DSv4 HC mix shape {}x{} cannot produce {mix_dim} weights from stream dim {}",
        hc.mix_fn.rows,
        hc.mix_fn.cols,
        stream.hidden_dim
    );
    ensure!(
        hc.base.len >= mix_dim && hc.scale.len >= 3,
        "DSv4 HC base/scale too short: base={} scale={} need base>={mix_dim} scale>=3",
        hc.base.len,
        hc.scale.len
    );

    // SAFETY: dsv4_linear writes the full mix buffer.
    let mut mixes = unsafe { HiddenStates::uninit(ctx, hc.mix_fn.rows, stream.seq_len)? };
    crate::attention::dsv4_linear(ctx, &hc.mix_fn, stream, &mut mixes)?;

    // SAFETY: uninit device scratch; fully written before first read.
    let mut pre = unsafe {
        ctx.stream
            .alloc::<f32>(stream.seq_len * hc_mult)
            .map_err(|e| anyhow!("DSv4 HC pre alloc failed: {e}"))?
    };
    // SAFETY: uninit device scratch; fully written before first read.
    let mut post = unsafe {
        ctx.stream
            .alloc::<f32>(stream.seq_len * hc_mult)
            .map_err(|e| anyhow!("DSv4 HC post alloc failed: {e}"))?
    };
    // SAFETY: uninit device scratch; fully written before first read.
    let mut comb = unsafe {
        ctx.stream
            .alloc::<f32>(stream.seq_len * hc_mult * hc_mult)
            .map_err(|e| anyhow!("DSv4 HC comb alloc failed: {e}"))?
    };

    tensor_ops::dsv4_mhc_params(
        ctx,
        &stream.data,
        &mixes.data,
        &hc.base.data,
        &hc.scale.data,
        &mut pre,
        &mut post,
        &mut comb,
        stream.seq_len,
        stream.hidden_dim,
        mixes.hidden_dim,
        hc_mult,
        config.hc_eps,
        config.hc_sinkhorn_iters,
    )?;
    Ok(MhcParams { pre, post, comb })
}

// Kept as the unfused primitive; current DSv4 decode uses `mhc_pre_rms_norm`.
#[allow(dead_code)]
pub(crate) fn hc_pre(
    ctx: &DeviceContext,
    stream: &HiddenStates,
    pre: &CudaSlice<f32>,
    hidden_size: usize,
    hc_mult: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        stream.hidden_dim == hidden_size * hc_mult,
        "DSv4 HC pre stream dim {} != hidden_size {hidden_size} * hc_mult {hc_mult}",
        stream.hidden_dim
    );
    ensure!(
        pre.len() >= stream.seq_len * hc_mult,
        "DSv4 HC pre len {} < seq {} * hc_mult {hc_mult}",
        pre.len(),
        stream.seq_len
    );
    ensure!(
        out.hidden_dim == hidden_size && out.seq_len == stream.seq_len,
        "DSv4 HC pre out shape {}x{} != {hidden_size}x{}",
        out.hidden_dim,
        out.seq_len,
        stream.seq_len
    );
    tensor_ops::dsv4_mhc_pre(
        ctx,
        &stream.data,
        pre,
        &mut out.data,
        stream.seq_len,
        hidden_size,
        hc_mult,
    )
}

/// Fused [`hc_pre`] + rms-norm: mix the stream into one lane and normalize in
/// a single kernel (one boundary, no intermediate tensor). Drop-in for the
/// `hc_pre(...) ; rms_norm_batch(...)` pair on every layer's attn/ffn prologue.
pub(crate) fn mhc_pre_rms_norm(
    ctx: &DeviceContext,
    stream: &HiddenStates,
    pre: &CudaSlice<f32>,
    norm_weight: &cuda_kernels::prelude::DeviceVec,
    eps: f32,
    hidden_size: usize,
    hc_mult: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        stream.hidden_dim == hidden_size * hc_mult,
        "DSv4 HC pre+norm stream dim {} != hidden_size {hidden_size} * hc_mult {hc_mult}",
        stream.hidden_dim
    );
    ensure!(
        pre.len() >= stream.seq_len * hc_mult,
        "DSv4 HC pre+norm pre len {} < seq {} * hc_mult {hc_mult}",
        pre.len(),
        stream.seq_len
    );
    ensure!(
        norm_weight.len == hidden_size,
        "DSv4 HC pre+norm weight len {} != hidden {hidden_size}",
        norm_weight.len
    );
    ensure!(
        out.hidden_dim == hidden_size && out.seq_len == stream.seq_len,
        "DSv4 HC pre+norm out shape {}x{} != {hidden_size}x{}",
        out.hidden_dim,
        out.seq_len,
        stream.seq_len
    );
    tensor_ops::dsv4_mhc_pre_rms_norm(
        ctx,
        &stream.data,
        pre,
        &norm_weight.data,
        &mut out.data,
        stream.seq_len,
        hidden_size,
        hc_mult,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_post(
    ctx: &DeviceContext,
    new_x: &HiddenStates,
    residual: &HiddenStates,
    post: &CudaSlice<f32>,
    comb: &CudaSlice<f32>,
    hidden_size: usize,
    hc_mult: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        new_x.hidden_dim == hidden_size && residual.hidden_dim == hidden_size * hc_mult,
        "DSv4 HC post dim mismatch: new_x={} residual={} hidden_size={hidden_size} hc_mult={hc_mult}",
        new_x.hidden_dim,
        residual.hidden_dim
    );
    ensure!(
        new_x.seq_len == residual.seq_len && out.seq_len == residual.seq_len,
        "DSv4 HC post seq mismatch: new_x={} residual={} out={}",
        new_x.seq_len,
        residual.seq_len,
        out.seq_len
    );
    ensure!(
        post.len() >= residual.seq_len * hc_mult
            && comb.len() >= residual.seq_len * hc_mult * hc_mult,
        "DSv4 HC post weights too small: post={} comb={}",
        post.len(),
        comb.len()
    );
    ensure!(
        out.hidden_dim == hidden_size * hc_mult,
        "DSv4 HC post out dim {} != hidden_size {hidden_size} * hc_mult {hc_mult}",
        out.hidden_dim
    );
    tensor_ops::dsv4_mhc_post(
        ctx,
        &new_x.data,
        &residual.data,
        post,
        comb,
        &mut out.data,
        residual.seq_len,
        hidden_size,
        hc_mult,
    )
}

pub(crate) fn head_hidden_from_stream(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    head_hc: &Dsv4HyperConnection,
    stream: &HiddenStates,
    token_idx: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    let hidden_size = config.hidden_size;
    let hc_mult = config.hc_mult;
    ensure!(
        token_idx < stream.seq_len,
        "DSv4 head token {token_idx} out of range for stream seq {}",
        stream.seq_len
    );
    ensure!(
        stream.hidden_dim == hidden_size * hc_mult,
        "DSv4 head stream dim {} != hidden_size {hidden_size} * hc_mult {hc_mult}",
        stream.hidden_dim
    );
    ensure!(
        head_hc.mix_fn.cols == stream.hidden_dim && head_hc.mix_fn.rows >= hc_mult,
        "DSv4 head HC mix shape {}x{} cannot produce {hc_mult} pre weights from stream dim {}",
        head_hc.mix_fn.rows,
        head_hc.mix_fn.cols,
        stream.hidden_dim
    );
    ensure!(
        head_hc.base.len >= hc_mult && head_hc.scale.len >= 1,
        "DSv4 head HC base/scale too short: base={} scale={}",
        head_hc.base.len,
        head_hc.scale.len
    );
    ensure!(
        out.len == hidden_size,
        "DSv4 head out len {} != hidden_size {hidden_size}",
        out.len
    );

    // SAFETY: copy_row_to_hidden writes the full one-token stream row.
    let mut stream_row = unsafe { HiddenStates::uninit(ctx, stream.hidden_dim, 1)? };
    // SAFETY: dsv4_linear writes the full head-HC mix buffer.
    let mut mixes = unsafe { HiddenStates::uninit(ctx, head_hc.mix_fn.rows, 1)? };
    head_hidden_from_stream_into(
        ctx,
        config,
        head_hc,
        stream,
        token_idx,
        &mut stream_row,
        &mut mixes,
        out,
    )
}

/// Graph-safe [`head_hidden_from_stream`] variant using caller-owned scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn head_hidden_from_stream_into(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    head_hc: &Dsv4HyperConnection,
    stream: &HiddenStates,
    token_idx: usize,
    stream_row: &mut HiddenStates,
    mixes: &mut HiddenStates,
    out: &mut DeviceVec,
) -> Result<()> {
    let hidden_size = config.hidden_size;
    let hc_mult = config.hc_mult;
    ensure!(
        token_idx < stream.seq_len,
        "DSv4 head token {token_idx} out of range for stream seq {}",
        stream.seq_len
    );
    ensure!(
        stream.hidden_dim == hidden_size * hc_mult,
        "DSv4 head stream dim {} != hidden_size {hidden_size} * hc_mult {hc_mult}",
        stream.hidden_dim
    );
    ensure!(
        head_hc.mix_fn.cols == stream.hidden_dim && head_hc.mix_fn.rows >= hc_mult,
        "DSv4 head HC mix shape {}x{} cannot produce {hc_mult} pre weights from stream dim {}",
        head_hc.mix_fn.rows,
        head_hc.mix_fn.cols,
        stream.hidden_dim
    );
    ensure!(
        head_hc.base.len >= hc_mult && head_hc.scale.len >= 1,
        "DSv4 head HC base/scale too short: base={} scale={}",
        head_hc.base.len,
        head_hc.scale.len
    );
    ensure!(
        stream_row.hidden_dim == stream.hidden_dim && stream_row.seq_len == 1,
        "DSv4 head stream scratch shape {}x{} != {}x1",
        stream_row.hidden_dim,
        stream_row.seq_len,
        stream.hidden_dim
    );
    ensure!(
        mixes.hidden_dim == head_hc.mix_fn.rows && mixes.seq_len == 1,
        "DSv4 head mix scratch shape {}x{} != {}x1",
        mixes.hidden_dim,
        mixes.seq_len,
        head_hc.mix_fn.rows
    );
    ensure!(
        out.len == hidden_size,
        "DSv4 head out len {} != hidden_size {hidden_size}",
        out.len
    );

    crate::ops::copy_row_to_hidden(ctx, stream, token_idx, stream_row)?;
    crate::attention::dsv4_linear(ctx, &head_hc.mix_fn, stream_row, mixes)?;

    tensor_ops::dsv4_mhc_head_pre(
        ctx,
        &stream_row.data,
        &mixes.data,
        &head_hc.base.data,
        &head_hc.scale.data,
        &mut out.data,
        stream.hidden_dim,
        hidden_size,
        hc_mult,
        config.hc_eps,
    )
}

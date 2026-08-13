//! DSv4 hyper-connections (`hc_mult > 1`).
//!
//! The residual is a `hidden_size * hc_mult`-wide STREAM (`hc_mult` lanes per
//! token). Each attention/FFN sub-block is wrapped: a learned mixer projects the
//! stream into per-lane pre/post/combination weights (sinkhorn-normalized), the
//! pre weights collapse the stream to one `hidden_size` row for the sub-block,
//! and the post/comb weights scatter the sub-block output back across the lanes
//! and re-mix the residual. Reuses the shared `dsv4_mhc_*` kernels; the math is
//! the legacy `weights.rs` `gen_mhc_params` / `hc_pre_from_stream` /
//! `hc_post_to_stream` / `head_hidden_from_stream`, reorganized into one module.

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::DeepSeekV4Config;

use crate::dsv4::Dsv4HyperConnection;

/// Per-token sinkhorn-normalized mixing weights for one HC sub-block.
pub(crate) struct MhcParams {
    /// `[seq_len * hc_mult]` pre-mix lane weights (stream → hidden).
    pub pre: CudaSlice<f32>,
    /// `[seq_len * hc_mult]` post-mix lane weights (new hidden → stream lanes).
    pub post: CudaSlice<f32>,
    /// `[seq_len * hc_mult * hc_mult]` residual re-combination weights.
    pub comb: CudaSlice<f32>,
}

/// Expand token embeddings `[hidden_size, seq]` into the initial wide HC stream
/// `[hidden_size * hc_mult, seq]` (each lane seeded from the embedding).
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
    let (emb_ptr, _ge) = embeddings.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = stream.data.device_ptr_mut(&ctx.stream);
    // SAFETY: both buffers valid on ctx.stream; shapes checked above.
    unsafe {
        ffi::dsv4_mhc_expand_cuda(
            emb_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            embeddings.seq_len as i32,
            hidden_size as i32,
            hc_mult as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Project the wide stream through `hc.mix_fn`, then run the sinkhorn mixer
/// (`dsv4_mhc_params_cuda`) to produce the per-lane pre/post/comb weights.
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

    {
        let (stream_ptr, _gs) = stream.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _gm) = mixes.data.device_ptr(&ctx.stream);
        let (base_ptr, _gb) = hc.base.data.device_ptr(&ctx.stream);
        let (scale_ptr, _gsc) = hc.scale.data.device_ptr(&ctx.stream);
        let (pre_ptr, _gp) = pre.device_ptr_mut(&ctx.stream);
        let (post_ptr, _gpo) = post.device_ptr_mut(&ctx.stream);
        let (comb_ptr, _gc) = comb.device_ptr_mut(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; mix shape + lengths checked above.
        unsafe {
            ffi::dsv4_mhc_params_cuda(
                stream_ptr as *const ffi::Half,
                mixes_ptr as *const ffi::Half,
                base_ptr as *const ffi::Half,
                scale_ptr as *const ffi::Half,
                pre_ptr as *mut f32,
                post_ptr as *mut f32,
                comb_ptr as *mut f32,
                stream.seq_len as i32,
                stream.hidden_dim as i32,
                mixes.hidden_dim as i32,
                hc_mult as i32,
                config.hc_eps,
                config.hc_sinkhorn_iters as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    Ok(MhcParams { pre, post, comb })
}

/// Collapse the wide stream to one `hidden_size` row per token using the pre-mix
/// lane weights (input to the sub-block).
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
    let (stream_ptr, _gs) = stream.data.device_ptr(&ctx.stream);
    let (pre_ptr, _gp) = pre.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    // SAFETY: buffers valid on ctx.stream; shapes checked above.
    unsafe {
        ffi::dsv4_mhc_pre_cuda(
            stream_ptr as *const ffi::Half,
            pre_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            stream.seq_len as i32,
            hidden_size as i32,
            hc_mult as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
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
    let (stream_ptr, _gs) = stream.data.device_ptr(&ctx.stream);
    let (pre_ptr, _gp) = pre.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = norm_weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    // SAFETY: buffers valid on ctx.stream; shapes checked above.
    unsafe {
        ffi::dsv4_mhc_pre_rms_norm_cuda(
            stream_ptr as *const ffi::Half,
            pre_ptr as *const f32,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            stream.seq_len as i32,
            hidden_size as i32,
            hc_mult as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Scatter the sub-block output back across the lanes and re-mix the residual
/// stream (output of the sub-block).
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
    let (new_ptr, _gn) = new_x.data.device_ptr(&ctx.stream);
    let (res_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (post_ptr, _gp) = post.device_ptr(&ctx.stream);
    let (comb_ptr, _gc) = comb.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    // SAFETY: buffers valid on ctx.stream; shapes checked above.
    unsafe {
        ffi::dsv4_mhc_post_cuda(
            new_ptr as *const ffi::Half,
            res_ptr as *const ffi::Half,
            post_ptr as *const f32,
            comb_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            residual.seq_len as i32,
            hidden_size as i32,
            hc_mult as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Fold the `token_idx` row of the wide stream into one `hidden_size` vector via
/// the head HC mixer (`mix_fn` row gemv → `dsv4_mhc_head_pre_cuda`).
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

    // Extract the single stream row, project it through mix_fn (batch=1).
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

    let (row_ptr, _gr) = stream_row.data.device_ptr(&ctx.stream);
    let (mixes_ptr, _gm) = mixes.data.device_ptr(&ctx.stream);
    let (base_ptr, _gb) = head_hc.base.data.device_ptr(&ctx.stream);
    let (scale_ptr, _gsc) = head_hc.scale.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    // SAFETY: buffers valid on ctx.stream; shapes checked above.
    unsafe {
        ffi::dsv4_mhc_head_pre_cuda(
            row_ptr as *const ffi::Half,
            mixes_ptr as *const ffi::Half,
            base_ptr as *const ffi::Half,
            scale_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            stream.hidden_dim as i32,
            hidden_size as i32,
            hc_mult as i32,
            config.hc_eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

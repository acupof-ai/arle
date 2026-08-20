//! Sampling and speculative draft/verify launch helpers.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi::{self, Half};
use crate::tensor::DeviceContext;

// Safe wrappers over the sampling FFI, per the `moe.rs` / `quant_linear.rs`
// pattern: typed buffers, checked i32 casts, pointer guards held through
// submission, one FFI symbol per launcher. Row selection stays with the
// caller: pass a sliced view for a row-offset launch.

fn extent(a: usize, b: usize, what: &'static str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow!("{what} shape overflow: {a}x{b}"))
}

/// DSpark logits filter (temperature softmax → top-k → top-p → min-p), as the
/// filter/draft kernels consume it.
#[derive(Clone, Copy, Debug)]
pub struct DsparkFilter {
    pub inv_temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
}

/// Single-row argmax: out[0] = argmax(logits[0..n]).
pub fn argmax(
    ctx: &DeviceContext,
    logits: &impl DevicePtr<bf16>,
    out: &mut impl DevicePtrMut<i32>,
    n: usize,
) -> Result<()> {
    ensure!(
        logits.len() >= n && !out.is_empty(),
        "argmax buffers do not cover n={n}: logits={} out={}",
        logits.len(),
        out.len()
    );
    let (logits_ptr, _gl) = logits.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above.
    unsafe {
        ffi::argmax_cuda(
            logits_ptr as *const Half,
            out_ptr as *mut i32,
            i32::try_from(n)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("argmax_cuda failed at n={n}: {e}"))
    }
}

/// Batched argmax: token_ids[r] = argmax(logits[r, :]) over `[batch, vocab]`.
pub fn argmax_batch(
    ctx: &DeviceContext,
    logits: &impl DevicePtr<bf16>,
    token_ids: &mut impl DevicePtrMut<i32>,
    batch: usize,
    vocab: usize,
) -> Result<()> {
    ensure!(
        logits.len() >= extent(batch, vocab, "argmax_batch logits")? && token_ids.len() >= batch,
        "argmax_batch buffers do not cover [batch,vocab]=[{batch},{vocab}]: logits={} ids={}",
        logits.len(),
        token_ids.len()
    );
    let (logits_ptr, _gl) = logits.device_ptr(&ctx.stream);
    let (ids_ptr, _gi) = token_ids.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above.
    unsafe {
        ffi::argmax_batch_cuda(
            logits_ptr as *const Half,
            ids_ptr as *mut i32,
            i32::try_from(batch)?,
            i32::try_from(vocab)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("argmax_batch_cuda failed at [batch,vocab]=[{batch},{vocab}]: {e}"))
    }
}

/// Batched engine-sampler filter: `probs[row] :=` filtered, renormalized
/// dist of `logits[row]` over `[rows, vocab]`.
pub fn dspark_filter_probs(
    ctx: &DeviceContext,
    logits: &impl DevicePtr<bf16>,
    probs: &mut impl DevicePtrMut<f32>,
    rows: usize,
    vocab: usize,
    filter: DsparkFilter,
) -> Result<()> {
    let span = extent(rows, vocab, "dspark_filter_probs")?;
    ensure!(
        logits.len() >= span && probs.len() >= span,
        "dspark_filter_probs buffers do not cover [rows,vocab]=[{rows},{vocab}]: logits={} probs={}",
        logits.len(),
        probs.len()
    );
    let (logits_ptr, _gl) = logits.device_ptr(&ctx.stream);
    let (probs_ptr, _gp) = probs.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above.
    unsafe {
        ffi::dspark_filter_probs_cuda(
            logits_ptr as *const Half,
            probs_ptr as *mut f32,
            i32::try_from(rows)?,
            i32::try_from(vocab)?,
            filter.inv_temperature,
            filter.top_k,
            filter.top_p,
            filter.min_p,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("dspark_filter_probs_cuda failed at [rows,vocab]=[{rows},{vocab}]: {e}")
        })
    }
}

/// Filter one logits row into `q_row` AND multinomial-sample from it; only
/// the token id in `token_out[0]` returns to the host (draft markov step).
pub fn dspark_draft_sample(
    ctx: &DeviceContext,
    logits: &impl DevicePtr<bf16>,
    q_row: &mut impl DevicePtrMut<f32>,
    token_out: &mut impl DevicePtrMut<i32>,
    vocab: usize,
    filter: DsparkFilter,
    random_val: f32,
) -> Result<()> {
    ensure!(
        logits.len() >= vocab && q_row.len() >= vocab && !token_out.is_empty(),
        "dspark_draft_sample buffers do not cover vocab={vocab}: logits={} q={} tok={}",
        logits.len(),
        q_row.len(),
        token_out.len()
    );
    let (logits_ptr, _gl) = logits.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q_row.device_ptr_mut(&ctx.stream);
    let (tok_ptr, _gt) = token_out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above.
    unsafe {
        ffi::dspark_draft_sample_cuda(
            logits_ptr as *const Half,
            q_ptr as *mut f32,
            tok_ptr as *mut i32,
            i32::try_from(vocab)?,
            filter.inv_temperature,
            filter.top_k,
            filter.top_p,
            filter.min_p,
            random_val,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dspark_draft_sample_cuda failed at vocab={vocab}: {e}"))
    }
}

/// Chain rejection sampling over pre-filtered `q [depth, vocab]` /
/// `p [depth+1, vocab]` dists with host-supplied uniforms;
/// `out = [accepted_len, residual-or-bonus token]`.
#[allow(clippy::too_many_arguments)]
pub fn dspark_chain_accept(
    ctx: &DeviceContext,
    q: &impl DevicePtr<f32>,
    p: &impl DevicePtr<f32>,
    draft_tokens: &impl DevicePtr<i32>,
    u_accept: &impl DevicePtr<f32>,
    u_residual: &impl DevicePtr<f32>,
    out: &mut impl DevicePtrMut<i32>,
    depth: usize,
    vocab: usize,
) -> Result<()> {
    ensure!(
        q.len() >= extent(depth, vocab, "dspark_chain_accept q")?
            && p.len() >= extent(depth + 1, vocab, "dspark_chain_accept p")?
            && draft_tokens.len() >= depth
            && u_accept.len() >= depth
            && u_residual.len() > depth
            && out.len() >= 2,
        "dspark_chain_accept buffers do not cover [depth,vocab]=[{depth},{vocab}]: q={} p={} draft={} u_acc={} u_res={} out={}",
        q.len(),
        p.len(),
        draft_tokens.len(),
        u_accept.len(),
        u_residual.len(),
        out.len()
    );
    let (q_ptr, _gq) = q.device_ptr(&ctx.stream);
    let (p_ptr, _gp) = p.device_ptr(&ctx.stream);
    let (d_ptr, _gd) = draft_tokens.device_ptr(&ctx.stream);
    let (ua_ptr, _gua) = u_accept.device_ptr(&ctx.stream);
    let (ur_ptr, _gur) = u_residual.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; the q rows were pre-filtered by
    // `dspark_draft_sample`, the p rows by `dspark_filter_probs`.
    unsafe {
        ffi::dspark_chain_accept_cuda(
            q_ptr as *const f32,
            p_ptr as *const f32,
            d_ptr as *const i32,
            ua_ptr as *const f32,
            ur_ptr as *const f32,
            out_ptr as *mut i32,
            i32::try_from(depth)?,
            i32::try_from(vocab)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("dspark_chain_accept_cuda failed at [depth,vocab]=[{depth},{vocab}]: {e}")
        })
    }
}

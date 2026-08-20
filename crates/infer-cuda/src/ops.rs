//! CUDA op wrappers for the dense-BF16 Qwen3 forward (HOT axis).
//!
//! Thin crate-private wrappers over `cuda-kernels` FFI: embedding, RMSNorm,
//! GEMM/GEMV, add, SwiGLU, row copy, argmax, host uploads, RoPE precompute.

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::{ffi, tensor_ops};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::sync::OnceLock;

#[path = "ops/quant_linear.rs"]
mod quant_linear;
pub(crate) use quant_linear::qwen_fp8_dense_operator_stats;

fn qwen_gemm_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("ARLE_QWEN35_PROFILE").is_some()
            || std::env::var_os("ARLE_QWEN35_QUANT_PROFILE").is_some()
    })
}

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

/// Upload a host `&[u64]` to device (e.g. an array of device pointers for a

/// Compile the NVFP4 prefill arm's DeepGEMM kernel at load and reserve its E4M3
/// scratch, instead of letting the first prefill chunk do both in-request.
pub(crate) fn warm_fp4_deepgemm_dense(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    seq_len: usize,
) -> Result<bool> {
    quant_linear::warm_fp4_deepgemm_dense(ctx, weight, seq_len)
}
/// pointer-array batched kernel). Mirrors [`upload_i32`].
pub(crate) fn upload_u64(ctx: &DeviceContext, values: &[u64]) -> Result<CudaSlice<u64>> {
    ctx.stream
        .clone_htod(values)
        .map_err(|e| anyhow!("H2D u64 upload failed: {e}"))
}

pub(crate) fn upload_f32(ctx: &DeviceContext, values: &[f32]) -> Result<CudaSlice<f32>> {
    ctx.stream
        .clone_htod(values)
        .map_err(|e| anyhow!("H2D f32 upload failed: {e}"))
}

pub(crate) fn embedding_batch(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids: &CudaSlice<i32>,
    out: &mut HiddenStates,
) -> Result<()> {
    tensor_ops::embedding_batched(
        ctx,
        &embed.data,
        token_ids,
        &mut out.data,
        embed.cols,
        out.seq_len,
    )
}

pub(crate) fn rms_norm_batch(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    tensor_ops::rms_norm_batched(
        ctx,
        &x.data,
        0,
        &weight.data,
        &mut out.data,
        x.hidden_dim,
        x.seq_len,
        eps,
    )
}

pub(crate) fn rms_norm_vec(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    tensor_ops::rms_norm(ctx, &x.data, &weight.data, &mut out.data, x.len, eps)
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
    if qwen_gemm_profile_enabled()
        && std::env::var("INFER_TP_RANK")
            .map(|rank| rank == "0")
            .unwrap_or(true)
    {
        eprintln!(
            "[qwen-gemm-profile] format={} M={} N={} K={}",
            weight.weight_format(),
            weight.rows,
            x.seq_len,
            weight.cols
        );
    }
    if weight.weight_format.is_quantized() {
        return quant_linear::gemm_batch(ctx, weight, x, out);
    }
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    // SAFETY: ptrs from live device allocations sized to the dims passed.
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

pub(crate) fn warm_fp8_deepgemm_dense(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    seq_len: usize,
) -> Result<bool> {
    quant_linear::warm_fp8_deepgemm_dense(ctx, weight, seq_len)
}

/// Whether the NVFP4 DeepGEMM prefill arm can serve this weight at some M — the
/// loader's test for building its `sfb`.
pub(crate) fn fp4_deepgemm_available(ctx: &DeviceContext, weight: &DeviceMatrix) -> bool {
    quant_linear::fp4_deepgemm_available(ctx, weight)
}

/// Whether the per-channel FP8 DeepGEMM prefill arm can serve this weight at
/// some M — the loader's test for setting `fp8_deepgemm_prefill`.
pub(crate) fn fp8_deepgemm_per_channel_available(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
) -> bool {
    quant_linear::fp8_deepgemm_per_channel_available(ctx, weight)
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
    if weight.weight_format.is_quantized() {
        return quant_linear::gemv(ctx, weight, x, out);
    }
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    // Routed as an N=1 cuBLASLt GEMM: ~2× the hand-written kernel's lm_head
    // bandwidth, and capture-safe (fixed workspace, warm-cached algo).
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        ffi::gemm_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            weight.rows as i32,
            1,
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
    tensor_ops::add(ctx, &a.data, &b.data, &mut out.data, n)
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
    tensor_ops::silu_mul(ctx, &gate.data, &up.data, &mut out.data, n)
}

/// Split a row-fused `[seq, first + second]` buffer into two buffers (leading
/// `first.hidden_dim` of each row → `first`, remainder → `second`).
pub(crate) fn split2(
    ctx: &DeviceContext,
    fused: &HiddenStates,
    first: &mut HiddenStates,
    second: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        fused.hidden_dim == first.hidden_dim + second.hidden_dim
            && fused.seq_len == first.seq_len
            && fused.seq_len == second.seq_len,
        "split2 shape mismatch: fused [{}, {}] vs parts [{}, {}] / [{}, {}]",
        fused.hidden_dim,
        fused.seq_len,
        first.hidden_dim,
        first.seq_len,
        second.hidden_dim,
        second.seq_len
    );
    tensor_ops::split2(
        ctx,
        &fused.data,
        &mut first.data,
        &mut second.data,
        fused.seq_len,
        first.hidden_dim,
        second.hidden_dim,
    )
}

/// Split a row-fused `[seq, q + 2*kv]` qkv buffer into `q`/`k`/`v` buffers.
pub(crate) fn split_qkv(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    q: &mut HiddenStates,
    k: &mut HiddenStates,
    v: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        k.hidden_dim == v.hidden_dim
            && qkv.hidden_dim == q.hidden_dim + 2 * k.hidden_dim
            && qkv.seq_len == q.seq_len
            && qkv.seq_len == k.seq_len
            && qkv.seq_len == v.seq_len,
        "split_qkv shape mismatch: qkv [{}, {}] vs q [{}, {}] k [{}, {}] v [{}, {}]",
        qkv.hidden_dim,
        qkv.seq_len,
        q.hidden_dim,
        q.seq_len,
        k.hidden_dim,
        k.seq_len,
        v.hidden_dim,
        v.seq_len
    );
    tensor_ops::split_qkv(
        ctx,
        &qkv.data,
        &mut q.data,
        &mut k.data,
        &mut v.data,
        qkv.seq_len,
        q.hidden_dim,
        k.hidden_dim,
    )
}

/// SwiGLU over a row-fused `[seq, 2*inter]` gate_up buffer (gate = first half
/// of each row, up = second half): `out[seq, inter] = silu(gate) * up`. The
/// fused layout is what a single GEMM over a rows-concatenated `[2*inter, K]`
/// weight produces, at any seq_len.
pub(crate) fn silu_mul_fused(
    ctx: &DeviceContext,
    gate_up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        gate_up.hidden_dim == 2 * out.hidden_dim && gate_up.seq_len == out.seq_len,
        "silu_mul_fused shape mismatch: gate_up [{}, {}] vs out [{}, {}]",
        gate_up.hidden_dim,
        gate_up.seq_len,
        out.hidden_dim,
        out.seq_len
    );
    tensor_ops::silu_mul_fused(
        ctx,
        &gate_up.data,
        &mut out.data,
        gate_up.seq_len,
        out.hidden_dim,
    )
}

/// On-device LoRA delta GEMM: `out = B · A` where `B` is `[rows, rank]` and
/// `A` is `[rank, cols]`, producing `out` as a flat `[rows, cols]` **row-major**
/// buffer (the same layout as a dense `DeviceMatrix.data`).
///
/// `gemm_cuda` computes `Y[M,N] col-major = W[M,K] row-major · X[K,N] col-major`.
/// Mapping `M=cols, N=rows, K=rank` makes `Y` col-major `[cols, rows]` byte-
/// identical to `out` row-major `[rows, cols]`, with `Y[c,r] = Σ_k W[c,k]·X[k,r]`.
/// To get `Σ_k B[r,k]·A[k,c]` the caller must pass:
///
/// - `a_t` = transpose of `A`, i.e. `[cols, rank]` row-major (`W[c,k]=A[k,c]`);
/// - `b`   = `B` `[rows, rank]` row-major as-is (col-major `X[k,r]=B[r,k]`).
///
/// `out` must hold at least `rows*cols` elements (only the first `rows*cols` are
/// written). Generic over the buffer view so a reused (over-sized) scratch slice
/// works without a copy.
pub(crate) fn lora_device_gemm<A, B, O>(
    ctx: &DeviceContext,
    a_t: &A,
    b: &B,
    out: &mut O,
    rows: usize,
    cols: usize,
    rank: usize,
) -> Result<()>
where
    A: DevicePtr<bf16>,
    B: DevicePtr<bf16>,
    O: DevicePtrMut<bf16>,
{
    let (w_ptr, _gw) = a_t.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = b.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        ffi::gemm_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            cols as i32,
            rows as i32,
            rank as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// In-place full-buffer scaled add over the first `n` elements:
/// `out[i] = base[i] + scale·delta[i]`. `base` (len == `n`) is first copied into
/// `out` (len == `n`) device→device, then the `add_scaled_row` kernel folds in
/// `scale·delta` over the buffer treated as a single row of `hidden_dim = n`.
/// Used for the device LoRA merge `W = base + scale·(B·A)`. `delta` is generic
/// over the buffer view so a reused (over-sized) scratch slice works; only its
/// first `n` elements are read.
#[cfg(test)]
pub(crate) fn lora_scaled_add_into<B, D, O>(
    ctx: &DeviceContext,
    base: &B,
    delta: &D,
    out: &mut O,
    n: usize,
    scale: f32,
) -> Result<()>
where
    B: DevicePtr<bf16>,
    D: DevicePtr<bf16>,
    O: DevicePtrMut<bf16>,
{
    // Callers slice `base`/`out` to exactly `n` elements (row-fused weights
    // pass a row window of the resident matrix).
    ctx.stream
        .memcpy_dtod(base, out)
        .map_err(|e| anyhow!("lora_scaled_add_into: base D2D copy failed: {e}"))?;
    tensor_ops::add_scaled_row(ctx, delta, out, n, 0, scale)
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

/// Copy one token row of a batch into a single-token `HiddenStates` (`seq_len ==
/// 1`, same `hidden_dim`). The HC head needs the last stream row as a batch-of-1
/// so the mix_fn GEMM can run on it.
pub(crate) fn copy_row_to_hidden(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        out.hidden_dim == batch.hidden_dim && out.seq_len == 1,
        "copy_row_to_hidden out shape {}x{} != {}x1",
        out.hidden_dim,
        out.seq_len,
        batch.hidden_dim
    );
    let offset = token_idx * batch.hidden_dim;
    let src = batch.data.slice(offset..offset + batch.hidden_dim);
    ctx.stream
        .memcpy_dtod(&src, &mut out.data)
        .map_err(|e| anyhow!("D2D copy stream row failed: {e}"))
}

pub(crate) fn argmax(ctx: &DeviceContext, logits: &DeviceVec) -> Result<u32> {
    let mut out = ctx
        .stream
        .alloc_zeros::<i32>(1)
        .map_err(|e| anyhow!("argmax output alloc failed: {e}"))?;
    argmax_into(ctx, logits, &mut out)
}

/// [`argmax`] into a caller-provided persistent 1-element device scratch —
/// the steady-state decode sampler (Qwen3.5/3.6 workspace `argmax_out` slot)
/// uses this so greedy decode performs ZERO device allocations per token.
/// Syncs the stream (the token must reach the host), so it must stay OUTSIDE
/// any CUDA-graph capture.
pub(crate) fn argmax_into(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    out: &mut CudaSlice<i32>,
) -> Result<u32> {
    ensure!(
        out.len() == 1,
        "argmax scratch must be one i32, got {}",
        out.len()
    );
    cuda_kernels::sampling::argmax(ctx, &logits.data, out, logits.len)?;
    ctx.sync()?;
    let token = ctx
        .stream
        .clone_dtoh(out)
        .map_err(|e| anyhow!("D2H argmax token failed: {e}"))?;
    Ok(token[0] as u32)
}

/// On-device argmax over ONE row of a `[seq, vocab]` bf16 logits buffer, into a
/// caller-provided 1-element scratch. The spec-decode verify needs the argmax of
/// each verify row WITHOUT a full `[seq, vocab]` D2H + host scan — this offsets
/// the device pointer to `row * vocab` and argmaxes `vocab` elements on-device
/// (only the 1-int result crosses to the host), the same fused kernel the
/// steady-state decode sampler uses.
pub(crate) fn argmax_row_into(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    row: usize,
    vocab: usize,
    out: &mut CudaSlice<i32>,
) -> Result<u32> {
    ensure!(
        out.len() == 1,
        "argmax scratch must be one i32, got {}",
        out.len()
    );
    ensure!(
        (row + 1) * vocab <= logits.len,
        "argmax_row_into row {row} (vocab {vocab}) exceeds logits len {}",
        logits.len
    );
    // Row slice bounds-checked above; only the 1-int result crosses to the host.
    let row_view = logits.data.slice(row * vocab..(row + 1) * vocab);
    cuda_kernels::sampling::argmax(ctx, &row_view, out, vocab)?;
    ctx.sync()?;
    let token = ctx
        .stream
        .clone_dtoh(out)
        .map_err(|e| anyhow!("D2H argmax row token failed: {e}"))?;
    Ok(token[0] as u32)
}

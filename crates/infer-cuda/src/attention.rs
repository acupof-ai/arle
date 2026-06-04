//! Paged attention kernel-call paths for the dense-BF16 Qwen3 forward (HOT axis).
//!
//! Prep kernels fuse Q/K RMSNorm + RoPE + KV-cache write; the TileLang kernels
//! run the HD128/kv8 paged attention.

use anyhow::{Result, bail, ensure};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, PagedKVPool};
use cuda_kernels::tensor::WeightFormat;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config};

use crate::dsv4::{Dsv4Attention, Dsv4Compressor, Dsv4Indexer};
use crate::loader::PageMeta;

#[allow(clippy::too_many_arguments)]
pub(crate) fn paged_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    if meta.seq_len == 1 {
        decode_attention(
            ctx,
            layer_idx,
            pool,
            q_batch,
            k_batch,
            v_batch,
            q_norm,
            k_norm,
            cos_cache,
            sin_cache,
            rms_eps,
            meta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            out,
        )
    } else {
        prefill_attention(
            ctx,
            layer_idx,
            pool,
            q_batch,
            k_batch,
            v_batch,
            q_norm,
            k_norm,
            cos_cache,
            sin_cache,
            rms_eps,
            meta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            out,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn prefill_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr_mut(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
        let (indices_ptr, _gi) = meta.kv_indices.device_ptr(&ctx.stream);
        let (offsets_ptr, _goff) = meta.page_table_offsets.device_ptr(&ctx.stream);
        let (start_ptr, _gstart) = meta.start_positions.device_ptr(&ctx.stream);
        let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
        let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);

        unsafe {
            ffi::prefill_attention_paged_prep_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                indices_ptr as *const i32,
                offsets_ptr as *const i32,
                pool.page_size as i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                meta.seq_len as i32,
                start_ptr as *const i32,
                rms_eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    run_tilelang_paged(
        ctx,
        false,
        layer_idx,
        pool,
        q_batch,
        meta,
        num_q_heads,
        num_kv_heads,
        head_dim,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
        let (pos_ptr, _gp) = meta.positions.device_ptr(&ctx.stream);
        let (indices_ptr, _gi) = meta.kv_indices.device_ptr(&ctx.stream);
        let (indptr_ptr, _gind) = meta.kv_indptr.device_ptr(&ctx.stream);
        let (last_ptr, _glp) = meta.kv_last_page_len.device_ptr(&ctx.stream);
        let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
        let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);
        let stride_page = pool.kv_dim * pool.page_size;

        unsafe {
            ffi::decode_prep_paged_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                pos_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                indices_ptr as *const i32,
                indptr_ptr as *const i32,
                last_ptr as *const i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                stride_page as i32,
                1,
                rms_eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    run_tilelang_paged(
        ctx,
        true,
        layer_idx,
        pool,
        q_batch,
        meta,
        num_q_heads,
        num_kv_heads,
        head_dim,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_tilelang_paged(
    ctx: &DeviceContext,
    decode: bool,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &HiddenStates,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(head_dim == 128, "only HD128 TileLang kernels are wired");
    ensure!(num_kv_heads == 8, "only kv8 TileLang kernels are wired");

    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (qo_ptr, _gqo) = meta.q_indptr.device_ptr(&ctx.stream);
    let (kv_indptr_ptr, _gki) = meta.kv_indptr.device_ptr(&ctx.stream);
    let (kv_indices_ptr, _gkx) = meta.kv_indices.device_ptr(&ctx.stream);
    let (last_ptr, _glp) = meta.kv_last_page_len.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
    let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    // Set R6_ATTN_DEBUG=1 to dump the scalar args + device arrays fed to the
    // TileLang paged kernel.
    if std::env::var("R6_ATTN_DEBUG").is_ok() {
        eprintln!(
            "[r6-attn] decode={decode} layer={layer_idx} q_heads={num_q_heads} kv_heads={num_kv_heads} head_dim={head_dim} seq_len={} num_pages(meta)={} max_total_pages={} page_size={} kv_dim={} sm_scale={sm_scale}",
            meta.seq_len, meta.num_pages, pool.max_total_pages, pool.page_size, pool.kv_dim
        );
        for (name, slice) in [
            ("q_indptr", &meta.q_indptr),
            ("kv_indptr", &meta.kv_indptr),
            ("kv_indices", &meta.kv_indices),
            ("kv_last_page_len", &meta.kv_last_page_len),
        ] {
            match ctx.stream.clone_dtoh(slice) {
                Ok(v) => eprintln!("[r6-attn]   {name} = {v:?}"),
                Err(e) => eprintln!("[r6-attn]   {name} dtoh err: {e}"),
            }
        }
    }

    // TileLang arg order (load-bearing): `num_pages` (arg 12) = pool capacity
    // (`pool.max_total_pages`, the k_pool/v_pool first-dim extent); `total_pages`
    // (arg 13) = page-table length (`meta.num_pages`). Swapping them gives wrong
    // pool strides + an OOB kv_indices walk that hangs the kernel (Xid 43).
    unsafe {
        match (decode, num_q_heads) {
            (false, 16) => ffi::tilelang_batch_prefill_paged_hd128_q16_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 32) => ffi::tilelang_batch_prefill_paged_hd128_q32_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 40) => ffi::tilelang_batch_prefill_paged_hd128_q40_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 64) => ffi::tilelang_batch_prefill_paged_hd128_q64_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 16) => ffi::tilelang_batch_decode_paged_hd128_q16_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 32) => ffi::tilelang_batch_decode_paged_hd128_q32_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 40) => ffi::tilelang_batch_decode_paged_hd128_q40_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 64) => ffi::tilelang_batch_decode_paged_hd128_q64_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            _ => bail!("unsupported HD128 q/kv head config q{num_q_heads}_kv{num_kv_heads}"),
        }
    }
    Ok(())
}

// ============================================================================
// DSv4-Flash MLA attention core
// ============================================================================
//
// The MLA attention is a genuinely new subsystem next to the dense-BF16 paged
// path above (it is NOT a GEMM swap): a low-rank Q/KV projection (`wq_a → q_norm
// → wq_b` for Q; `wkv → kv_norm` for the single compressed KV latent), partial
// RoPE on the trailing `rope_dim` columns, a windowed attention with a per-head
// sink logit + (on CSA/HCA layers) a compressed-key stream, and a low-rank O
// projection (`wo_a → wo_b`).
//
// All three modes run through the bf16 correctness core (the perf-optimized
// FlashMLA sparse path stays gated — `Dsv4MlaKvArena::alloc_fp8_arena`):
//   - SlidingWindow (`compress_ratio == 0`): Q/K prep RoPE + `dsv4_swa_attention`
//     over the bf16 SW ring cache, with the output inverse-RoPE fused.
//   - CompressedSparse (`0 < ratio < 16`): a compressor produces compressed keys,
//     an indexer + `dsv4_csa_select_cuda` picks the top-k blocks, then
//     `dsv4_hybrid_attention_cuda` (mode 1) attends over SW window + selected
//     compressed blocks.
//   - HybridCompressed (`ratio >= 16`): compressor + `dsv4_hybrid_attention_cuda`
//     (mode 2) attending over SW window + ALL compressed blocks (no selector).
//
// Shared kernels: `dsv4_{fp8,fp4}_gemv_batch_cuda` / `gemm_cuda` (LoRA matmuls),
// `dsv4_prepare_qk_cuda`, `dsv4_swa_attention_cuda`, `dsv4_compressor_update_cuda`,
// `dsv4_csa_select_cuda`, `dsv4_hybrid_attention_cuda`.

/// Run one DSv4 FP8/FP4 block-scaled LoRA matmul: `out[N, T] = W[N, K] · x[K, T]`.
///
/// The MLA LoRA weights (`wq_a/wq_b/wkv/wo_a/wo_b`) load as
/// [`WeightFormat::Dsv4Fp8BlockScaled`] / [`WeightFormat::Dsv4Fp4BlockScaled`]
/// (raw quant bytes in `qweight`, E8M0 block scales in `dsv4_scales`), so the
/// dense bf16 [`gemm_batch`] cannot run them — this dispatches the shared
/// `dsv4_*_gemv_batch_cuda` kernels instead. `batch_size` is the token count.
pub(crate) fn mla_linear(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        weight.cols == x.hidden_dim,
        "mla_linear input dim mismatch: weight cols {}, x hidden_dim {}",
        weight.cols,
        x.hidden_dim
    );
    ensure!(
        weight.rows == out.hidden_dim && x.seq_len == out.seq_len,
        "mla_linear output shape mismatch: weight rows {}, out hidden_dim {}, x seq {}, out seq {}",
        weight.rows,
        out.hidden_dim,
        x.seq_len,
        out.seq_len
    );
    let qw = weight
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 MLA matrix missing raw quant bytes (qweight)"))?;
    let scales = weight
        .dsv4_scales
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 MLA matrix missing block scales (dsv4_scales)"))?;
    let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    // SAFETY: all buffers are valid on ctx.stream; shapes are checked above and
    // the scale-row/col extents come from the matrix the loader built.
    unsafe {
        let res = match weight.weight_format {
            WeightFormat::Dsv4Fp8BlockScaled => ffi::dsv4_fp8_gemv_batch_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const u8,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                x.seq_len as i32,
                weight.rows as i32,
                weight.cols as i32,
                weight.dsv4_scale_rows as i32,
                weight.dsv4_scale_cols as i32,
                stream,
            ),
            WeightFormat::Dsv4Fp4BlockScaled => ffi::dsv4_fp4_gemv_batch_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const u8,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                x.seq_len as i32,
                weight.rows as i32,
                weight.cols as i32,
                weight.dsv4_scale_rows as i32,
                weight.dsv4_scale_cols as i32,
                stream,
            ),
            other => bail!("mla_linear: expected DSv4 FP8/FP4 block-scaled weight, got {other:?}"),
        };
        res.result()?;
    }
    Ok(())
}

/// Run one DSv4 linear `out = W · x` dispatching on the weight's on-disk format:
/// bf16 dense → [`crate::ops::gemm_batch`]; FP8/FP4 block-scaled → [`mla_linear`].
/// DSv4 checkpoints ship the compressor / indexer / HC-mix matrices in either
/// precision, so callers route every non-router linear through here.
pub(crate) fn dsv4_linear(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    match weight.weight_format {
        WeightFormat::DenseBf16 => crate::ops::gemm_batch(ctx, weight, x, out),
        WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled => {
            mla_linear(ctx, weight, x, out)
        }
        other => bail!("dsv4_linear: unsupported weight format {other:?}"),
    }
}

/// RMSNorm a `HiddenStates` in place into a fresh buffer (the MLA Q/KV LoRA
/// norms `q_norm` / `kv_norm`). Thin wrapper over the shared batched RMSNorm.
fn mla_rms_norm(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, x.hidden_dim, x.seq_len)?;
    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
        // SAFETY: buffers valid on ctx.stream; out matches x shape.
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
    }
    Ok(out)
}

/// One DSv4 MLA attention block (SlidingWindow / CompressedSparse /
/// HybridCompressed, dispatched on `mode` / `compress_ratio`).
///
/// `hidden` is the post-attn-LN input `[hidden_size, token_count]`;
/// `sw_window_cache` is this layer's bf16 sliding-window ring cache
/// (`sliding_window * head_dim` elements), read+written in place by the windowed
/// kernel. `start_pos` is the absolute position of `hidden`'s first token (0 for
/// a fresh prefill). Writes `[hidden_size, token_count]` into `out` (the O-LoRA
/// output, pre-TP-all-reduce — the model layer-loop owns the row-parallel sum).
///
/// The compressed pool is recomputed per call from `hidden` over the absolute
/// `[0, start_pos + token_count)` range (the uncached correctness path); a
/// continuous-batching executor will instead retain it per slot. The
/// FlashMLA-FP8 decode launch stays gated (perf path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_attention(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    layer_idx: usize,
    hidden: &HiddenStates,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    start_pos: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        hidden.hidden_dim == config.hidden_size,
        "DSv4 MLA hidden dim {} != hidden_size {}",
        hidden.hidden_dim,
        config.hidden_size
    );

    let head_dim = config.head_dim;
    let token_count = hidden.seq_len;
    let local_width = attention.wq_b.rows;
    ensure!(
        head_dim > 0 && local_width.is_multiple_of(head_dim),
        "DSv4 MLA local q width {local_width} is not a multiple of head_dim {head_dim}"
    );
    let local_heads = local_width / head_dim;
    ensure!(local_heads > 0, "DSv4 MLA requires at least one local head");
    ensure!(
        attention.wkv.rows == head_dim,
        "DSv4 MLA wkv rows {} != head_dim {head_dim}",
        attention.wkv.rows
    );
    ensure!(
        attention.wo_a.cols == local_width,
        "DSv4 MLA wo_a cols {} != local attention width {local_width}",
        attention.wo_a.cols
    );
    ensure!(
        attention.wo_b.rows == out.hidden_dim && out.seq_len == token_count,
        "DSv4 MLA output shape mismatch: wo_b rows {} out {}x{} expected {}x{}",
        attention.wo_b.rows,
        out.hidden_dim,
        out.seq_len,
        attention.wo_b.rows,
        token_count
    );
    ensure!(
        config.sliding_window > 0,
        "DSv4 MLA requires a non-zero sliding_window"
    );
    ensure!(
        config.qk_rope_head_dim <= head_dim,
        "DSv4 MLA rope dim {} exceeds head_dim {head_dim}",
        config.qk_rope_head_dim
    );
    ensure!(
        sw_window_cache.len() == config.sliding_window * head_dim,
        "DSv4 MLA SW window cache len {} != sliding_window*head_dim {}",
        sw_window_cache.len(),
        config.sliding_window * head_dim
    );
    ensure!(
        attention.attn_sink.len >= local_heads,
        "DSv4 MLA attn_sink len {} cannot cover local heads {local_heads}",
        attention.attn_sink.len
    );

    let rope = &config.rope_parameters;
    // Q / SW-K / output RoPE is ALWAYS the main rope_theta with NO YaRN, for
    // every layer regardless of compress_ratio (only the COMPRESSED keys use
    // compress_rope_theta — applied inside `compressor_forward`). This matches
    // the validated long-context fix (errors/2026-05-29-dsv4-longctx-rope...).
    let rope_base = config.rope_theta;
    let original_seq_len = 0i32;
    let start_pos_i32 = i32::try_from(start_pos)
        .map_err(|_| anyhow::anyhow!("DSv4 MLA start_pos {start_pos} overflows i32"))?;

    // ── 1. Q-LoRA: wq_a (down) → q_norm RMSNorm → wq_b (up to per-head Q).
    let mut c_q = HiddenStates::zeros(ctx, attention.wq_a.rows, token_count)?;
    dsv4_linear(ctx, &attention.wq_a, hidden, &mut c_q)?;
    let c_q_normed = mla_rms_norm(ctx, &c_q, &attention.q_norm, config.rms_norm_eps)?;
    let mut q_raw = HiddenStates::zeros(ctx, local_width, token_count)?;
    dsv4_linear(ctx, &attention.wq_b, &c_q_normed, &mut q_raw)?;

    // ── 2. KV latent: wkv (down to the single compressed latent) → kv_norm.
    let mut kv_raw = HiddenStates::zeros(ctx, head_dim, token_count)?;
    dsv4_linear(ctx, &attention.wkv, hidden, &mut kv_raw)?;
    let kv_normed = mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, config.rms_norm_eps)?;

    // ── 3. Partial RoPE on the trailing rope_dim cols of Q (per head) and K.
    let mut q_prepared = HiddenStates::zeros(ctx, local_width, token_count)?;
    let mut k_prepared = HiddenStates::zeros(ctx, head_dim, token_count)?;
    {
        let (q_raw_ptr, _qr) = q_raw.data.device_ptr(&ctx.stream);
        let (k_raw_ptr, _kr) = kv_normed.data.device_ptr(&ctx.stream);
        let (q_out_ptr, _qo) = q_prepared.data.device_ptr_mut(&ctx.stream);
        let (k_out_ptr, _ko) = k_prepared.data.device_ptr_mut(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; head/dim args checked above.
        unsafe {
            ffi::dsv4_prepare_qk_cuda(
                q_raw_ptr as *const ffi::Half,
                k_raw_ptr as *const ffi::Half,
                q_out_ptr as *mut ffi::Half,
                k_out_ptr as *mut ffi::Half,
                token_count as i32,
                local_heads as i32,
                head_dim as i32,
                config.qk_rope_head_dim as i32,
                start_pos_i32,
                config.rms_norm_eps,
                rope_base,
                original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut local_attn = HiddenStates::zeros(ctx, local_width, token_count)?;

    if mode == DeepSeekV4AttentionMode::SlidingWindow {
        // ── 4a. SW: windowed attention + per-head sink + output inverse-RoPE.
        // The kernel reads the pre-roped q/k, attends over the bf16 SW ring cache
        // (which it also updates), adds the sink, and un-rotates the rope tail of
        // the OUTPUT (sign = -1) before returning.
        let (q_ptr, _qg) = q_prepared.data.device_ptr(&ctx.stream);
        let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
        let (window_ptr, _wg) = sw_window_cache.device_ptr_mut(&ctx.stream);
        let (sink_ptr, _sg) = attention.attn_sink.data.device_ptr(&ctx.stream);
        let (out_ptr, _og) = local_attn.data.device_ptr_mut(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; window sized above; sink_offset
        // is 0 (per-rank local heads index the per-rank attn_sink slice directly;
        // EP/TP head sharding is a multi-rank follow-up).
        unsafe {
            ffi::dsv4_swa_attention_cuda(
                q_ptr as *const ffi::Half,
                k_ptr as *const ffi::Half,
                window_ptr as *mut ffi::Half,
                sink_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                token_count as i32,
                local_heads as i32,
                head_dim as i32,
                config.sliding_window as i32,
                start_pos_i32,
                0,
                sm_scale,
                config.qk_rope_head_dim as i32,
                rope_base,
                original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
                1,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    } else {
        // ── 4b. CSA / HCA: compressor → (CSA) indexer top-k select → hybrid
        // windowed+compressed attention.
        let compressor = attention.compressor.as_ref().ok_or_else(|| {
            anyhow::anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no compressor weights")
        })?;
        let overlap = compress_ratio < 16;
        let compressed = compressor_forward(
            ctx,
            config,
            compressor,
            hidden,
            head_dim,
            compress_ratio,
            overlap,
            start_pos,
            true,
        )?;

        let selected = if mode == DeepSeekV4AttentionMode::CompressedSparse {
            let indexer = attention.indexer.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "DSv4 layer {layer_idx} is CompressedSparse but has no indexer weights"
                )
            })?;
            // Indexer keys: a second compressor over index_head_dim keys (no APE
            // gate on the keys — `apply_rope = true`, head_dim = index_head_dim).
            let index_keys = compressor_forward(
                ctx,
                config,
                &indexer.compressor,
                hidden,
                config.index_head_dim,
                compress_ratio,
                true,
                start_pos,
                false,
            )?;
            Some(csa_select(
                ctx,
                config,
                indexer,
                hidden,
                &c_q_normed,
                &index_keys,
                start_pos,
                compress_ratio,
            )?)
        } else {
            None
        };

        let compressed_count = compressed.seq_len;
        let mode_int = match mode {
            DeepSeekV4AttentionMode::CompressedSparse => 1,
            DeepSeekV4AttentionMode::HybridCompressed => 2,
            DeepSeekV4AttentionMode::SlidingWindow => unreachable!(),
        };
        let (q_ptr, _qg) = q_prepared.data.device_ptr(&ctx.stream);
        let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
        let (window_ptr, _wg) = sw_window_cache.device_ptr_mut(&ctx.stream);
        let (sink_ptr, _sg) = attention.attn_sink.data.device_ptr(&ctx.stream);
        let (out_ptr, _og) = local_attn.data.device_ptr_mut(&ctx.stream);
        let (comp_ptr, _cg, _cguard) = if compressed_count > 0 {
            let (p, g) = compressed.data.device_ptr(&ctx.stream);
            (p as *const ffi::Half, true, Some(g))
        } else {
            (std::ptr::null(), false, None)
        };
        let (sel_ptr, _sguard) = match selected.as_ref() {
            Some(sel) => {
                let (p, g) = sel.device_ptr(&ctx.stream);
                (p as *const i32, Some(g))
            }
            None => (std::ptr::null(), None),
        };
        // SAFETY: all buffers valid on ctx.stream; compressed/selected may be null
        // (the kernel branches on compressed_count / mode). write_window_cache=1
        // updates the bf16 SW ring inline.
        unsafe {
            ffi::dsv4_hybrid_attention_cuda(
                q_ptr as *const ffi::Half,
                k_ptr as *const ffi::Half,
                window_ptr as *mut ffi::Half,
                comp_ptr,
                sel_ptr,
                sink_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                token_count as i32,
                local_heads as i32,
                head_dim as i32,
                config.sliding_window as i32,
                start_pos_i32,
                0,
                sm_scale,
                config.qk_rope_head_dim as i32,
                rope_base,
                original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
                mode_int,
                compress_ratio as i32,
                compressed_count as i32,
                config.index_topk as i32,
                1,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }

    // ── 5. O-LoRA: wo_a (per o-group, down to the output latent) → wo_b (up
    // back to hidden). Row-parallel: the all-reduce-sum is the model's concern.
    let mut latent = HiddenStates::zeros(ctx, attention.wo_a.rows, token_count)?;
    dsv4_linear(ctx, &attention.wo_a, &local_attn, &mut latent)?;
    dsv4_linear(ctx, &attention.wo_b, &latent, out)?;
    Ok(())
}

/// Run one compressor sub-block over `hidden`, producing the bf16 compressed-key
/// pool `[head_dim, compressed_rows]` for the absolute `[0, start_pos +
/// token_count)` range (uncached: a fresh pending/overlap buffer per call).
///
/// `wkv`/`wgate` project the hidden into the per-block KV / gating-score streams
/// (`width = 2*head_dim` when `overlap`, else `head_dim`); `dsv4_compressor_update_cuda`
/// folds them through `ape` + RMSNorm(`norm`) + compress-rope into one row per
/// `compress_ratio` tokens. `apply_rope = false` skips the rope tail (indexer
/// keys).
#[allow(clippy::too_many_arguments)]
fn compressor_forward(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    compressor: &Dsv4Compressor,
    hidden: &HiddenStates,
    head_dim: usize,
    ratio: usize,
    overlap: bool,
    start_pos: usize,
    apply_rope: bool,
) -> Result<HiddenStates> {
    ensure!(ratio > 0, "DSv4 compressor ratio must be non-zero");
    // Uncached full-prefix contract: `hidden` must span the whole sequence from
    // position 0, so the fresh (zeroed) pending/overlap scratch is the correct
    // empty prior. Incremental decode of CSA/HCA layers (start_pos > 0, one new
    // token) needs the prior pending tokens + compressed rows retained per slot —
    // that's the continuous-batching follow-up; the bf16 SW ring caches already
    // carry the windowed contribution, but the compressed pool is recomputed here.
    ensure!(
        start_pos == 0,
        "DSv4 CSA/HCA compressor is the uncached full-prefix path (start_pos must be 0); \
         incremental compressed-pool retention is the continuous-batching follow-up"
    );
    let width = if overlap { 2 * head_dim } else { head_dim };
    ensure!(
        compressor.wkv.rows == width && compressor.wgate.rows == width,
        "DSv4 compressor rows mismatch: wkv={} wgate={} expected width={width}",
        compressor.wkv.rows,
        compressor.wgate.rows
    );
    let token_count = hidden.seq_len;
    let total = token_count;
    let compressed_rows = total / ratio;
    // Capacity (rows) the kernel may write into; one full block per `ratio` tokens.
    let capacity_rows = total.max(1).div_ceil(ratio);

    let mut kv_raw = HiddenStates::zeros(ctx, width, token_count)?;
    dsv4_linear(ctx, &compressor.wkv, hidden, &mut kv_raw)?;
    let mut score_raw = HiddenStates::zeros(ctx, width, token_count)?;
    dsv4_linear(ctx, &compressor.wgate, hidden, &mut score_raw)?;

    // Fresh pending / overlap scratch (uncached full-prefix path: empty prior).
    let pending_kv = ctx
        .stream
        .alloc_zeros::<half::bf16>(ratio * width)
        .map_err(|e| anyhow::anyhow!("DSv4 compressor pending kv alloc failed: {e}"))?;
    let pending_score = ctx
        .stream
        .alloc_zeros::<half::bf16>(ratio * width)
        .map_err(|e| anyhow::anyhow!("DSv4 compressor pending score alloc failed: {e}"))?;
    let prev_overlap_kv = ctx
        .stream
        .alloc_zeros::<half::bf16>(ratio * head_dim)
        .map_err(|e| anyhow::anyhow!("DSv4 compressor prev kv alloc failed: {e}"))?;
    let prev_overlap_score = ctx
        .stream
        .alloc_zeros::<half::bf16>(ratio * head_dim)
        .map_err(|e| anyhow::anyhow!("DSv4 compressor prev score alloc failed: {e}"))?;
    let compressed = HiddenStates::zeros(ctx, head_dim, capacity_rows.max(1))?;
    if compressed_rows == 0 {
        // No completed blocks yet — return the empty pool (seq_len 0).
        return Ok(HiddenStates {
            data: compressed.data,
            hidden_dim: head_dim,
            seq_len: 0,
        });
    }

    let rope = &config.rope_parameters;
    // Compressed keys use compress_rope_theta with NO YaRN (original_seq_len = 0).
    let (rope_dim, rope_base) = if apply_rope {
        (config.qk_rope_head_dim, config.compress_rope_theta)
    } else {
        (0, config.compress_rope_theta)
    };
    let mut pending_kv = pending_kv;
    let mut pending_score = pending_score;
    let mut prev_overlap_kv = prev_overlap_kv;
    let mut prev_overlap_score = prev_overlap_score;
    let mut compressed = compressed;
    {
        let (kv_ptr, _kg) = kv_raw.data.device_ptr(&ctx.stream);
        let (score_ptr, _scg) = score_raw.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ag) = compressor.ape.data.device_ptr(&ctx.stream);
        let (norm_ptr, _ng) = compressor.norm.data.device_ptr(&ctx.stream);
        let (pkv_ptr, _pkg) = pending_kv.device_ptr_mut(&ctx.stream);
        let (psc_ptr, _psg) = pending_score.device_ptr_mut(&ctx.stream);
        let (prkv_ptr, _prkg) = prev_overlap_kv.device_ptr_mut(&ctx.stream);
        let (prsc_ptr, _prsg) = prev_overlap_score.device_ptr_mut(&ctx.stream);
        let (comp_ptr, _cg) = compressed.data.device_ptr_mut(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; capacity sized for compressed_rows.
        // Uncached full-prefix: start_pos=0, pending_len=0, compressed_base=0,
        // has_prev_overlap=0 (the fresh zeroed scratch is the correct empty prior).
        unsafe {
            ffi::dsv4_compressor_update_cuda(
                kv_ptr as *const ffi::Half,
                score_ptr as *const ffi::Half,
                ape_ptr as *const ffi::Half,
                norm_ptr as *const ffi::Half,
                pkv_ptr as *mut ffi::Half,
                psc_ptr as *mut ffi::Half,
                prkv_ptr as *mut ffi::Half,
                prsc_ptr as *mut ffi::Half,
                comp_ptr as *mut ffi::Half,
                token_count as i32,
                0,
                0,
                0,
                head_dim as i32,
                ratio as i32,
                width as i32,
                i32::from(overlap),
                0,
                config.rms_norm_eps,
                rope_dim as i32,
                rope_base,
                0,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    Ok(HiddenStates {
        data: compressed.data,
        hidden_dim: head_dim,
        seq_len: compressed_rows,
    })
}

/// CSA top-k block selection: project the index query (`wq_b`) + per-head gating
/// (`weights_proj`), then `dsv4_csa_select_cuda` scores each compressed-key block
/// and writes the top-`index_topk` block ids per token into `[seq * index_topk]`.
#[allow(clippy::too_many_arguments)]
fn csa_select(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    indexer: &Dsv4Indexer,
    hidden: &HiddenStates,
    c_q_normed: &HiddenStates,
    keys: &HiddenStates,
    start_pos: usize,
    ratio: usize,
) -> Result<CudaSlice<i32>> {
    let mut q_i = HiddenStates::zeros(ctx, indexer.wq_b.rows, c_q_normed.seq_len)?;
    dsv4_linear(ctx, &indexer.wq_b, c_q_normed, &mut q_i)?;
    let mut weights = HiddenStates::zeros(ctx, indexer.weights_proj.rows, hidden.seq_len)?;
    dsv4_linear(ctx, &indexer.weights_proj, hidden, &mut weights)?;

    ensure!(
        q_i.hidden_dim.is_multiple_of(config.index_head_dim),
        "DSv4 indexer q width {} is not divisible by index_head_dim {}",
        q_i.hidden_dim,
        config.index_head_dim
    );
    let local_index_heads = q_i.hidden_dim / config.index_head_dim;
    ensure!(
        weights.hidden_dim == local_index_heads,
        "DSv4 indexer weights width {} != local index heads {local_index_heads}",
        weights.hidden_dim
    );

    let key_count = keys.seq_len;
    let mut selected = ctx
        .stream
        .alloc_zeros::<i32>(hidden.seq_len * config.index_topk)
        .map_err(|e| anyhow::anyhow!("DSv4 CSA selected alloc failed: {e}"))?;
    let score_scale =
        (config.index_head_dim as f32).powf(-0.5) * (config.index_n_heads as f32).powf(-0.5);
    {
        let (q_ptr, _qg) = q_i.data.device_ptr(&ctx.stream);
        let (w_ptr, _wg) = weights.data.device_ptr(&ctx.stream);
        let (keys_ptr, _kg) = keys.data.device_ptr(&ctx.stream);
        let (sel_ptr, _sg) = selected.device_ptr_mut(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; selected sized seq*index_topk.
        unsafe {
            ffi::dsv4_csa_select_cuda(
                q_ptr as *const ffi::Half,
                w_ptr as *const ffi::Half,
                keys_ptr as *const ffi::Half,
                sel_ptr as *mut i32,
                hidden.seq_len as i32,
                q_i.hidden_dim as i32,
                local_index_heads as i32,
                config.index_head_dim as i32,
                key_count as i32,
                ratio as i32,
                config.index_topk as i32,
                score_scale,
                start_pos as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    Ok(selected)
}

//! Paged attention kernel-call paths for the clean dense-BF16 Qwen3 forward (HOT axis).
//!
//! Holds the prefill/decode prep + TileLang paged-attention dispatch. The prep
//! kernels fuse Q/K RMSNorm + RoPE + KV-cache write; the TileLang kernels run the
//! HD128 / kv8 paged attention. Pure relocation from `model.rs` — identical
//! numerics, identical FFI call sites.

use anyhow::{Result, bail, ensure};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates, PagedKVPool};
use cudarc::driver::{DevicePtr, DevicePtrMut};

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

    // Ground-truth arg dump (gated): set R6_ATTN_DEBUG=1 to print the exact
    // scalar args + device-array contents fed to the TileLang paged kernel.
    // Used to localize the prefill-kernel spin that static arg comparison vs the
    // legacy call could not explain.
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

    // NOTE on the two TileLang symbolic-shape args (mirrors the legacy
    // `infer/src/ops/attention.rs` call, which is the contract of record):
    //   - `num_pages`  (arg 12) = the K/V *pool capacity* = `pool.max_total_pages`.
    //                  It is the first-dim extent of the k_pool/v_pool tensors,
    //                  so it must be the whole pool, NOT this request's page count.
    //   - `total_pages`(arg 13) = the *page-table length* = the number of valid
    //                  entries in `kv_indices` (= `meta.num_pages` here, since
    //                  `PageMeta::for_slot` sizes kv_indices to exactly that).
    // Swapping these two passes a tiny capacity (→ wrong pool strides) and an
    // oversized page-table walk (→ OOB read past kv_indices) — an illegal access
    // that hangs the kernel (Xid 43). Keep capacity first, page-table length second.
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

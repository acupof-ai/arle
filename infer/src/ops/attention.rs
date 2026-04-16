//! Attention ops: prefill (FlashInfer) and decode (Triton AOT / custom CUDA).
//!
//! Three paged decode attention paths (selected by KV pool format):
//!   - **BF16**: FlashInfer native `BatchDecodeWithPagedKVCacheRun`
//!   - **INT8**: Custom split-KV kernel with fused INT8 dequant (`decode_attention_int8`)
//!   - **FP8**: Custom split-KV kernel with FP8→FP32 cast (`decode_attention_fp8`)
//!
//! Prefill uses FlashInfer batch-forward with layout dispatch:
//!   - HD128: `flashinfer_batch_forward_with_layout` (Qwen3, GLM4)
//!   - HD256: `flashinfer_batch_forward_hd256` (Qwen3.5 full-attention layers)
//!
//! Single-token decode uses Triton AOT kernel: fused QK-norm + RoPE + split-KV
//! attention + online softmax + merge in one kernel launch.

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use infer_cuda_kernels::ffi;
use infer_cuda_kernels::flashinfer::FlashInferWorkspace;
use infer_cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates, PagedKVPool};

// ============================================================================
// Parameter structs — group related config/weight params for high-arity ops.
// ============================================================================

/// QK normalization weights + RoPE caches, shared across layers.
pub(crate) struct NormRopeParams<'a> {
    pub q_norm: &'a DeviceVec,
    pub k_norm: &'a DeviceVec,
    pub cos_cache: &'a DeviceVec,
    pub sin_cache: &'a DeviceVec,
    pub rms_eps: f32,
}

/// Head configuration for attention.
pub(crate) struct HeadConfig {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Paged KV metadata for batched decode.
pub(crate) struct PagedKVMeta<'a> {
    pub kv_pool: &'a PagedKVPool,
    pub layer_idx: usize,
    pub kv_indices: &'a CudaSlice<i32>,
    pub kv_indptr: &'a CudaSlice<i32>,
    pub kv_last_page_len: &'a CudaSlice<i32>,
    pub page_size: usize,
}

/// Batched prefill attention with FlashAttention-2.
///
/// Pipeline:
///   1. QK norm + RoPE (CUDA kernel, in-place on q_batch/k_batch)
///   2. KV cache write (CUDA kernel)
///   3. FlashAttention-2 (Triton kernel — fused QK + causal softmax + V)
///
/// No O(n²) scratch buffers needed — FlashAttention uses online softmax.
pub(crate) fn prefill_attention_batch(
    ctx: &DeviceContext,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    nrp: &NormRopeParams,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut HiddenStates,
    heads: &HeadConfig,
    start_pos: usize,
) -> Result<()> {
    let seq_len = q_batch.seq_len;
    let num_q_heads = heads.num_q_heads;
    let num_kv_heads = heads.num_kv_heads;
    let head_dim = heads.head_dim;
    let rms_eps = nrp.rms_eps;
    let _q_dim = num_q_heads * head_dim;
    assert!(num_kv_heads > 0, "num_kv_heads must be > 0");
    let _gqa_ratio = num_q_heads / num_kv_heads;

    // Derive max_seq_len from KV cache buffer size.
    // Buffer layout: [num_kv_heads * max_seq_len * head_dim] u16 elements.
    let kv_elements = k_cache.len;
    let max_seq_len = kv_elements / (num_kv_heads * head_dim);

    {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr_mut(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = nrp.q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = nrp.k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = nrp.cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = nrp.sin_cache.data.device_ptr(&ctx.stream);
        let (kc_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
        let (vc_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);
        let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

        unsafe {
            // Steps 1-2: QK norm + RoPE, KV cache write
            ffi::prefill_attention_prep_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                kc_ptr as *mut ffi::Half,
                vc_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                seq_len as i32,
                start_pos as i32,
                max_seq_len as i32,
                rms_eps,
                ctx.stream.cu_stream(),
            )
            .result()?;

            // Step 3: FlashInfer single prefill — reads normed Q and KV cache
            let kv_len = start_pos + seq_len;
            let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
            let ret = ffi::flashinfer_single_prefill(
                q_ptr as *mut ffi::Half,
                kc_ptr as *mut ffi::Half,
                vc_ptr as *mut ffi::Half,
                o_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                seq_len as i32,
                kv_len as i32,
                max_seq_len as i32,
                sm_scale,
                std::ptr::null_mut(),
                ctx.stream.cu_stream(),
            );
            if ret != 0 {
                return Err(anyhow!(
                    "flashinfer_single_prefill failed: CUDA error {}",
                    ret
                ));
            }
        }
    }

    Ok(())
}

pub(crate) unsafe fn prefill_attention_batch_dual_write(
    ctx: &DeviceContext,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    nrp: &NormRopeParams,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    heads: &HeadConfig,
    start_pos: usize,
    page_table: &[i32],
    page_size: i32,
    k_pool: *mut ffi::Half,
    v_pool: *mut ffi::Half,
) -> Result<()> {
    let seq_len = q_batch.seq_len;
    let num_q_heads = heads.num_q_heads;
    let num_kv_heads = heads.num_kv_heads;
    let head_dim = heads.head_dim;
    let rms_eps = nrp.rms_eps;
    let _q_dim = num_q_heads * head_dim;
    assert!(num_kv_heads > 0, "num_kv_heads must be > 0");
    let _gqa_ratio = num_q_heads / num_kv_heads;

    // Derive max_seq_len from KV cache buffer size.
    // Buffer layout: [num_kv_heads * max_seq_len * head_dim] u16 elements.
    let kv_elements = k_cache.len;
    let max_seq_len = kv_elements / (num_kv_heads * head_dim);

    let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _gk) = k_batch.data.device_ptr_mut(&ctx.stream);
    let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = nrp.q_norm.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = nrp.k_norm.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = nrp.cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = nrp.sin_cache.data.device_ptr(&ctx.stream);
    let (kc_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
    let (vc_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::prefill_attention_prep_dual_write_cuda(
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            v_ptr as *const ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            kc_ptr as *mut ffi::Half,
            vc_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            seq_len as i32,
            start_pos as i32,
            max_seq_len as i32,
            rms_eps,
            ctx.stream.cu_stream(),
            page_table.as_ptr(),
            page_size,
            k_pool,
            v_pool,
        )
        .result()?;
    }

    Ok(())
}

/// FlashAttention-2 prefill for HEAD_DIM=256 with precomputed Q and KV cache.
/// Q / output layout: HiddenStates [q_dim, seq_len] in column-major token-major storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flash_attention_prefill_hd256_into(
    ctx: &DeviceContext,
    q_batch: &HiddenStates,
    k_cache: &DeviceVec,
    v_cache: &DeviceVec,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    start_pos: usize,
) -> Result<()> {
    let seq_len = q_batch.seq_len;
    let q_dim = q_batch.hidden_dim;
    let head_dim = q_dim / num_q_heads;
    assert_eq!(head_dim, 256, "HD256 kernel requires head_dim=256");
    assert_eq!(q_dim, output.hidden_dim, "output hidden_dim mismatch");
    assert_eq!(seq_len, output.seq_len, "output seq_len mismatch");
    assert!(num_kv_heads > 0, "num_kv_heads must be > 0");

    // Derive max_seq_len from KV cache buffer size.
    let max_seq_len = k_cache.len / (num_kv_heads * head_dim);

    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (kc_ptr, _gkc) = k_cache.data.device_ptr(&ctx.stream);
    let (vc_ptr, _gvc) = v_cache.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    let kv_len = start_pos + seq_len;
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let ret = unsafe {
        ffi::flashinfer_single_prefill_hd256(
            q_ptr as *mut ffi::Half,
            kc_ptr as *mut ffi::Half,
            vc_ptr as *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            seq_len as i32,
            kv_len as i32,
            max_seq_len as i32,
            sm_scale,
            std::ptr::null_mut(),
            ctx.stream.cu_stream(),
        )
    };
    if ret != 0 {
        return Err(anyhow!(
            "flashinfer_single_prefill_hd256 failed: CUDA error {}",
            ret
        ));
    }

    Ok(())
}

/// Qwen3.5 full-attention prefill: prep Q/K/cache, run HD256 FlashAttention-2, then apply gate.
pub(crate) fn prefill_attention_hd256_batch(
    ctx: &DeviceContext,
    q_full_batch: &HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    nrp: &NormRopeParams,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    start_pos: usize,
    rotary_dim: usize,
) -> Result<()> {
    let q_dim = num_q_heads * 256;
    let mut q_prepped = HiddenStates::zeros(ctx, q_dim, q_full_batch.seq_len)?;
    // Allocate temporary GPU scalar for start_pos
    let start_pos_buf: CudaSlice<i32> = ctx
        .stream
        .clone_htod(&[start_pos as i32])
        .map_err(|e| anyhow::anyhow!("start_pos H2D failed: {e}"))?;
    prefill_attention_hd256_batch_with_scratch(
        ctx,
        q_full_batch,
        k_batch,
        v_batch,
        nrp,
        k_cache,
        v_cache,
        output,
        &mut q_prepped,
        num_q_heads,
        num_kv_heads,
        start_pos,
        &start_pos_buf,
        rotary_dim,
    )
}

/// Same as `prefill_attention_hd256_batch` but uses pre-allocated scratch buffers.
/// `start_pos_buf` is a GPU-resident `i32` for CUDA Graph safety.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_attention_hd256_batch_with_scratch(
    ctx: &DeviceContext,
    q_full_batch: &HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    nrp: &NormRopeParams,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut HiddenStates,
    q_prepped: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    start_pos: usize,
    start_pos_buf: &CudaSlice<i32>,
    rotary_dim: usize,
) -> Result<()> {
    let seq_len = q_full_batch.seq_len;
    let q_dim = num_q_heads * 256;
    let kv_dim = num_kv_heads * 256;
    let rms_eps = nrp.rms_eps;

    assert_eq!(q_full_batch.hidden_dim, q_dim * 2);
    assert_eq!(k_batch.hidden_dim, kv_dim);
    assert_eq!(v_batch.hidden_dim, kv_dim);
    assert_eq!(k_batch.seq_len, seq_len);
    assert_eq!(v_batch.seq_len, seq_len);
    assert_eq!(output.hidden_dim, q_dim);
    assert_eq!(output.seq_len, seq_len);
    assert_eq!(q_prepped.hidden_dim, q_dim);

    // Derive max_seq_len from the K cache buffer size.
    let head_dim = 256;
    let max_seq_len = k_cache.len / (num_kv_heads * head_dim);

    unsafe {
        let (qf_ptr, _gqf) = q_full_batch.data.device_ptr(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = nrp.q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = nrp.k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gcos) = nrp.cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gsin) = nrp.sin_cache.data.device_ptr(&ctx.stream);
        let (qp_ptr, _gqp) = q_prepped.data.device_ptr_mut(&ctx.stream);
        let (kc_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
        let (vc_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);
        let (sp_ptr, _gsp) = start_pos_buf.device_ptr(&ctx.stream);

        ffi::prefill_attention_hd256_prep_cuda(
            qf_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            qp_ptr as *mut ffi::Half,
            kc_ptr as *mut ffi::Half,
            vc_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            seq_len as i32,
            sp_ptr as *const i32,
            rotary_dim as i32,
            rms_eps,
            max_seq_len as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    flash_attention_prefill_hd256_into(
        ctx,
        q_prepped,
        k_cache,
        v_cache,
        output,
        num_q_heads,
        num_kv_heads,
        start_pos,
    )?;

    unsafe {
        let (qf_ptr, _gqf) = q_full_batch.data.device_ptr(&ctx.stream);
        let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
        ffi::attention_gate_batch_hd256_cuda(
            qf_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            num_q_heads as i32,
            seq_len as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Batched fused GQA decode attention (CUDA, split-KV, HEAD_DIM=128).
///
/// Processes B requests in two kernel launches (split-KV + reduce) instead of
/// 2*B launches from the per-request loop. Each request's KV cache is accessed
/// via device pointer arrays.
///
/// Q/K/V are already in contiguous batch buffers [B, dim]. Output is written
/// directly to `output` batch buffer [B, q_dim]. No D2D copies needed.
///
/// `positions`: [B] i32 on GPU — current_pos per request
/// `seq_lens`: [B] i32 on GPU — seq_len per request (= pos + 1)
/// `k_cache_ptrs`/`v_cache_ptrs`: [B] device pointers on GPU
/// `partial_out/m/l`: pre-allocated FP32 scratch [B * num_qheads * NUM_KV_SPLITS * ...]
#[allow(clippy::too_many_arguments)]
pub fn fused_attention_decode_batched_into(
    ctx: &DeviceContext,
    q_batch: &HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache_base: &DeviceVec,
    sin_cache_base: &DeviceVec,
    positions: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    k_cache_ptrs: &CudaSlice<u64>,
    v_cache_ptrs: &CudaSlice<u64>,
    output: &mut HiddenStates,
    partial_out: &mut CudaSlice<f32>,
    partial_m: &mut CudaSlice<f32>,
    partial_l: &mut CudaSlice<f32>,
    num_qheads: usize,
    num_kvheads: usize,
    head_dim: usize,
    max_seq_len: usize,
    rms_eps: f32,
) -> Result<()> {
    let batch_size = q_batch.seq_len;
    let gqa_ratio = num_qheads / num_kvheads;

    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k_batch.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
    let (q_norm_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (k_norm_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gcos) = cos_cache_base.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gsin) = sin_cache_base.data.device_ptr(&ctx.stream);
    let (pos_ptr, _gp) = positions.device_ptr(&ctx.stream);
    let (sl_ptr, _gsl) = seq_lens.device_ptr(&ctx.stream);
    let (kc_ptrs_ptr, _gkcp) = k_cache_ptrs.device_ptr(&ctx.stream);
    let (vc_ptrs_ptr, _gvcp) = v_cache_ptrs.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (partial_out_ptr, _gpo) = partial_out.device_ptr_mut(&ctx.stream);
    let (partial_m_ptr, _gpm) = partial_m.device_ptr_mut(&ctx.stream);
    let (partial_l_ptr, _gpl) = partial_l.device_ptr_mut(&ctx.stream);

    // Phase 1: split-KV attention (writes partials)
    unsafe {
        ffi::fused_gqa_attention_decode_batched(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            pos_ptr as *const i32,
            sl_ptr as *const i32,
            kc_ptrs_ptr as *const *const ffi::Half,
            vc_ptrs_ptr as *const *const ffi::Half,
            partial_out_ptr as *mut f32,
            partial_m_ptr as *mut f32,
            partial_l_ptr as *mut f32,
            num_qheads as i32,
            num_kvheads as i32,
            gqa_ratio as i32,
            head_dim as i32,
            max_seq_len as i32,
            batch_size as i32,
            rms_eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    // Phase 2: reduce partials → final bf16 output
    unsafe {
        ffi::attention_decode_reduce_batched(
            partial_out_ptr as *const f32,
            partial_m_ptr as *const f32,
            partial_l_ptr as *const f32,
            o_ptr as *mut ffi::Half,
            num_qheads as i32,
            head_dim as i32,
            batch_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Fused GQA Attention for decode (Triton AOT, split-KV, HEAD_DIM=128).
/// Reads pos/seq_len from decode_meta — CUDA Graph safe.
/// cos_cache_base/sin_cache_base: full RoPE buffers [max_seq_len * head_dim].
/// decode_meta: [token_id, current_pos, seq_len] on GPU.
/// partial_out/m/l: pre-allocated FP32 scratch for split-KV intermediates.
#[allow(clippy::too_many_arguments)]
pub fn fused_attention_decode_into(
    ctx: &DeviceContext,
    q_full: &DeviceVec,
    k_full: &DeviceVec,
    v_full: &DeviceVec,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache_base: &DeviceVec,
    sin_cache_base: &DeviceVec,
    decode_meta: &CudaSlice<i32>,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut DeviceVec,
    partial_out: &mut CudaSlice<f32>,
    partial_m: &mut CudaSlice<f32>,
    partial_l: &mut CudaSlice<f32>,
    num_qheads: usize,
    num_kvheads: usize,
) -> Result<()> {
    // Derive max_seq_len from KV cache buffer size before borrowing.
    let actual_head_dim = q_full.len / num_qheads;
    let max_seq_len = k_cache.len / (num_kvheads * actual_head_dim);

    let (q_ptr, _gq) = q_full.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k_full.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_full.data.device_ptr(&ctx.stream);
    let (q_norm_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (k_norm_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gcos) = cos_cache_base.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gsin) = sin_cache_base.data.device_ptr(&ctx.stream);
    let (meta_ptr, _gm) = decode_meta.device_ptr(&ctx.stream);
    let (k_cache_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
    let (v_cache_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (partial_out_ptr, _gpo) = partial_out.device_ptr_mut(&ctx.stream);
    let (partial_m_ptr, _gpm) = partial_m.device_ptr_mut(&ctx.stream);
    let (partial_l_ptr, _gpl) = partial_l.device_ptr_mut(&ctx.stream);

    // Phase 1: split-KV attention (writes partials)
    let result = unsafe {
        ffi::fused_gqa_attention_decode(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            meta_ptr as *const i32,
            k_cache_ptr as *mut ffi::Half,
            v_cache_ptr as *mut ffi::Half,
            partial_out_ptr as *mut f32,
            partial_m_ptr as *mut f32,
            partial_l_ptr as *mut f32,
            num_qheads as i32,
            num_kvheads as i32,
            (num_qheads / num_kvheads) as i32,
            max_seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    // Phase 2: reduce partials → final bf16 output
    let result = unsafe {
        ffi::attention_decode_reduce(
            partial_out_ptr as *mut f32,
            partial_m_ptr as *mut f32,
            partial_l_ptr as *mut f32,
            out_ptr as *mut ffi::Half,
            num_qheads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Batched decode prep for paged KV cache: QK-norm + RoPE (in-place on Q) + paged KV write.
///
/// After this call:
/// - `q_batch` contains RMSNorm'd + RoPE'd Q values (in-place, layout [B, num_qo_heads * head_dim])
/// - K (normed + roped) and V (raw) are written to the paged KV cache at the correct positions
///
/// `positions`: [B] i32 on GPU — current_pos per request
/// `page_table_gpu`: flattened page indices on GPU
/// `page_indptr_gpu`: [B+1] i32 on GPU — cumulative page counts
/// `last_page_len_gpu`: [B] i32 on GPU — tokens in last page (including the new token)
pub(crate) fn decode_prep_paged(
    ctx: &DeviceContext,
    q_batch: &mut HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    nrp: &NormRopeParams,
    positions: &CudaSlice<i32>,
    paged: &PagedKVMeta,
    num_qo_heads: usize,
    num_kv_heads: usize,
) -> Result<()> {
    let batch_size = q_batch.seq_len;
    let stride_page = paged.kv_pool.kv_dim * paged.page_size;
    let rms_eps = nrp.rms_eps;
    let page_size = paged.page_size;

    let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _gk) = k_batch.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = nrp.q_norm.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = nrp.k_norm.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = nrp.cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = nrp.sin_cache.data.device_ptr(&ctx.stream);
    let (pos_ptr, _gp) = positions.device_ptr(&ctx.stream);
    let (pt_ptr, _gpt) = paged.kv_indices.device_ptr(&ctx.stream);
    let (pi_ptr, _gpi) = paged.kv_indptr.device_ptr(&ctx.stream);
    let (lp_ptr, _glp) = paged.kv_last_page_len.device_ptr(&ctx.stream);

    let k_pool_ptr = paged.kv_pool.k_ptr(paged.layer_idx, &ctx.stream);
    let v_pool_ptr = paged.kv_pool.v_ptr(paged.layer_idx, &ctx.stream);

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
            pt_ptr as *const i32,
            pi_ptr as *const i32,
            lp_ptr as *const i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            page_size as i32,
            stride_page as i32,
            batch_size as i32,
            rms_eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Fused QKV variant of [`decode_prep_paged`]: reads Q/K/V from merged buffer
/// `qkv_batch` [B, q_dim + 2*kv_dim] and writes RoPE'd Q to `q_out`.
/// Eliminates the separate `split_qkv` kernel launch (saves 36 launches/step).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_prep_paged_fused_qkv(
    ctx: &DeviceContext,
    qkv_batch: &HiddenStates,
    q_out: &mut HiddenStates,
    nrp: &NormRopeParams,
    positions: &CudaSlice<i32>,
    paged: &PagedKVMeta,
    num_qo_heads: usize,
    num_kv_heads: usize,
) -> Result<()> {
    let batch_size = qkv_batch.seq_len;
    let q_dim = num_qo_heads * 128; // HEAD_DIM = 128
    let kv_dim = num_kv_heads * 128;
    let qkv_stride = q_dim + 2 * kv_dim;
    let stride_page = paged.kv_pool.kv_dim * paged.page_size;
    let rms_eps = nrp.rms_eps;
    let page_size = paged.page_size;

    let (qkv_ptr, _g0) = qkv_batch.data.device_ptr(&ctx.stream);
    let (qo_ptr, _g1) = q_out.data.device_ptr_mut(&ctx.stream);
    let (qn_ptr, _g2) = nrp.q_norm.data.device_ptr(&ctx.stream);
    let (kn_ptr, _g3) = nrp.k_norm.data.device_ptr(&ctx.stream);
    let (cos_ptr, _g4) = nrp.cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _g5) = nrp.sin_cache.data.device_ptr(&ctx.stream);
    let (pos_ptr, _g6) = positions.device_ptr(&ctx.stream);
    let (pt_ptr, _g7) = paged.kv_indices.device_ptr(&ctx.stream);
    let (pi_ptr, _g8) = paged.kv_indptr.device_ptr(&ctx.stream);
    let (lp_ptr, _g9) = paged.kv_last_page_len.device_ptr(&ctx.stream);

    let k_pool_ptr = paged.kv_pool.k_ptr(paged.layer_idx, &ctx.stream);
    let v_pool_ptr = paged.kv_pool.v_ptr(paged.layer_idx, &ctx.stream);

    unsafe {
        ffi::decode_prep_paged_fused_qkv_cuda(
            qkv_ptr as *const ffi::Half,
            qo_ptr as *mut ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            pos_ptr as *const i32,
            k_pool_ptr as *mut ffi::Half,
            v_pool_ptr as *mut ffi::Half,
            pt_ptr as *const i32,
            pi_ptr as *const i32,
            lp_ptr as *const i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            page_size as i32,
            stride_page as i32,
            batch_size as i32,
            rms_eps,
            qkv_stride as i32,
            q_dim as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// FlashInfer run step only (GPU kernel). Call once per layer after a single plan call.
#[allow(clippy::too_many_arguments)]
pub fn flashinfer_run_layer(
    ctx: &DeviceContext,
    q_batch: &HiddenStates,
    kv_pool: &PagedKVPool,
    layer_idx: usize,
    kv_indptr_gpu: &CudaSlice<i32>,
    kv_indices_gpu: &CudaSlice<i32>,
    kv_last_page_len_gpu: &CudaSlice<i32>,
    output: &mut HiddenStates,
    workspace: &mut FlashInferWorkspace,
    num_qo_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    head_dim: usize,
) -> Result<()> {
    let batch_size = q_batch.seq_len;
    let sm_scale = 1.0 / (head_dim as f32).sqrt();

    let (fw_ptr, _gfw) = workspace.float_workspace.device_ptr_mut(&ctx.stream);
    let (iw_ptr, _giw) = workspace.int_workspace.device_ptr_mut(&ctx.stream);
    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (ind_ptr, _gind) = kv_indptr_gpu.device_ptr(&ctx.stream);
    let (idx_ptr, _gidx) = kv_indices_gpu.device_ptr(&ctx.stream);
    let (lp_ptr, _glp) = kv_last_page_len_gpu.device_ptr(&ctx.stream);
    let (lse_ptr, _glse) = workspace.lse.device_ptr_mut(&ctx.stream);

    let k_pool_ptr = kv_pool.k_ptr(layer_idx, &ctx.stream);
    let v_pool_ptr = kv_pool.v_ptr(layer_idx, &ctx.stream);

    let ret = unsafe {
        ffi::flashinfer_batch_decode_run(
            fw_ptr as *mut u8,
            iw_ptr as *mut u8,
            workspace.plan_info as *const u8,
            q_ptr as *const ffi::Half,
            k_pool_ptr as *const ffi::Half,
            v_pool_ptr as *const ffi::Half,
            ind_ptr as *const i32,
            idx_ptr as *const i32,
            lp_ptr as *const i32,
            o_ptr as *mut ffi::Half,
            lse_ptr as *mut f32,
            batch_size as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            page_size as i32,
            head_dim as i32,
            sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    if ret != 0 {
        return Err(anyhow!(
            "flashinfer_batch_decode_run failed with CUDA error {}",
            ret
        ));
    }
    Ok(())
}

// ============================================================================
// HD256 variants for Qwen3.5 full attention (head_dim=256, partial RoPE, gate)
// ============================================================================

/// HD256 batched decode prep: QK-norm (1+w offset) + partial RoPE + paged KV write.
///
/// - `q_full_batch` [B, num_q_heads * 256 * 2]: Q with interleaved gate
/// - `q_out_batch` [B, num_q_heads * 256]: output Q (normed + roped, no gate)
/// - Writes K (normed + roped) and V (raw) to paged pool.
pub(crate) fn decode_prep_paged_hd256(
    ctx: &DeviceContext,
    q_full_batch: &HiddenStates,
    q_out_batch: &mut HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    nrp: &NormRopeParams,
    positions: &CudaSlice<i32>,
    paged: &PagedKVMeta,
    num_qo_heads: usize,
    num_kv_heads: usize,
    rotary_dim: usize,
) -> Result<()> {
    let batch_size = q_full_batch.seq_len;
    let stride_page = paged.kv_pool.kv_dim * paged.page_size;
    let rms_eps = nrp.rms_eps;
    let page_size = paged.page_size;

    let (qf_ptr, _g0) = q_full_batch.data.device_ptr(&ctx.stream);
    let (qo_ptr, _g1) = q_out_batch.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _g2) = k_batch.data.device_ptr(&ctx.stream);
    let (v_ptr, _g3) = v_batch.data.device_ptr(&ctx.stream);
    let (qn_ptr, _g4) = nrp.q_norm.data.device_ptr(&ctx.stream);
    let (kn_ptr, _g5) = nrp.k_norm.data.device_ptr(&ctx.stream);
    let (cos_ptr, _g6) = nrp.cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _g7) = nrp.sin_cache.data.device_ptr(&ctx.stream);
    let (pos_ptr, _g8) = positions.device_ptr(&ctx.stream);
    let (pt_ptr, _g9) = paged.kv_indices.device_ptr(&ctx.stream);
    let (pi_ptr, _g10) = paged.kv_indptr.device_ptr(&ctx.stream);
    let (lp_ptr, _g11) = paged.kv_last_page_len.device_ptr(&ctx.stream);

    let k_pool_ptr = paged.kv_pool.k_ptr(paged.layer_idx, &ctx.stream);
    let v_pool_ptr = paged.kv_pool.v_ptr(paged.layer_idx, &ctx.stream);

    unsafe {
        ffi::decode_prep_paged_hd256_cuda(
            qf_ptr as *const ffi::Half,
            qo_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            pos_ptr as *const i32,
            k_pool_ptr as *mut ffi::Half,
            v_pool_ptr as *mut ffi::Half,
            pt_ptr as *const i32,
            pi_ptr as *const i32,
            lp_ptr as *const i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            page_size as i32,
            stride_page as i32,
            batch_size as i32,
            rotary_dim as i32,
            rms_eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Apply sigmoid gate from Q's gate portion to attention output.
/// `q_full_batch` has gate at [head * 2 * 256 + 256 .. head * 2 * 256 + 512].
pub(crate) fn attention_gate_paged_hd256(
    ctx: &DeviceContext,
    q_full_batch: &HiddenStates,
    attn_output: &mut HiddenStates,
    num_q_heads: usize,
) {
    let batch_size = attn_output.seq_len;
    let (qf_ptr, _g0) = q_full_batch.data.device_ptr(&ctx.stream);
    let (ao_ptr, _g1) = attn_output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::attention_gate_paged_hd256_cuda(
            qf_ptr as *const ffi::Half,
            ao_ptr as *mut ffi::Half,
            num_q_heads as i32,
            batch_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()
        .expect("attention_gate_paged_hd256_cuda failed");
    }
}

/// FlashInfer HD256 run-layer: uses the pre-computed plan to run attention for one layer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flashinfer_run_layer_hd256(
    ctx: &DeviceContext,
    q_batch: &HiddenStates,
    kv_pool: &PagedKVPool,
    layer_idx: usize,
    kv_indptr_gpu: &CudaSlice<i32>,
    kv_indices_gpu: &CudaSlice<i32>,
    kv_last_page_len_gpu: &CudaSlice<i32>,
    output: &mut HiddenStates,
    workspace: &mut FlashInferWorkspace,
    num_qo_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    head_dim: usize,
) -> Result<()> {
    let batch_size = q_batch.seq_len;
    let sm_scale = 1.0 / (head_dim as f32).sqrt();

    let (fw_ptr, _gfw) = workspace.float_workspace.device_ptr_mut(&ctx.stream);
    let (iw_ptr, _giw) = workspace.int_workspace.device_ptr_mut(&ctx.stream);
    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (ind_ptr, _gind) = kv_indptr_gpu.device_ptr(&ctx.stream);
    let (idx_ptr, _gidx) = kv_indices_gpu.device_ptr(&ctx.stream);
    let (lp_ptr, _glp) = kv_last_page_len_gpu.device_ptr(&ctx.stream);
    let (lse_ptr, _glse) = workspace.lse.device_ptr_mut(&ctx.stream);

    let k_pool_ptr = kv_pool.k_ptr(layer_idx, &ctx.stream);
    let v_pool_ptr = kv_pool.v_ptr(layer_idx, &ctx.stream);

    let ret = unsafe {
        ffi::flashinfer_batch_decode_hd256_run(
            fw_ptr as *mut u8,
            iw_ptr as *mut u8,
            workspace.plan_info as *const u8,
            q_ptr as *const ffi::Half,
            k_pool_ptr as *const ffi::Half,
            v_pool_ptr as *const ffi::Half,
            ind_ptr as *const i32,
            idx_ptr as *const i32,
            lp_ptr as *const i32,
            o_ptr as *mut ffi::Half,
            lse_ptr as *mut f32,
            batch_size as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            page_size as i32,
            head_dim as i32,
            sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    if ret != 0 {
        return Err(anyhow!(
            "flashinfer_batch_decode_hd256_run failed with CUDA error {}",
            ret
        ));
    }
    Ok(())
}

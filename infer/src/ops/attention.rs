//! Attention ops: TileLang paged prefill/decode plus custom CUDA prep/quantized decode.
//!
//! Three paged decode attention paths (selected by KV pool format):
//!   - **BF16**: TileLang AOT paged attention
//!   - **INT8**: Custom split-KV kernel with fused INT8 dequant (`decode_attention_int8`)
//!   - **FP8**: Custom split-KV kernel with FP8→FP32 cast (`decode_attention_fp8`)

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates, PagedKVPool};
use cuda_kernels::tilelang::TileLangWorkspace;

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

/// TileLang paged-decode head configuration (HD128, HD256, tensor-core prefill).
pub struct TileLangHeadConfig {
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub page_size: usize,
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

#[allow(clippy::too_many_arguments)]
fn nonpaged_prefill_attention_into(
    ctx: &DeviceContext,
    q_batch: &HiddenStates,
    k_cache: &DeviceVec,
    v_cache: &DeviceVec,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
) -> Result<()> {
    let seq_len = q_batch.seq_len;
    let q_dim = num_q_heads * head_dim;
    ensure!(num_kv_heads > 0, "num_kv_heads must be > 0");
    ensure!(
        head_dim == 128 || head_dim == 256,
        "non-paged prefill supports head_dim 128 or 256, got {head_dim}"
    );
    ensure!(
        q_batch.hidden_dim == q_dim,
        "q hidden_dim mismatch: got {}, expected {q_dim}",
        q_batch.hidden_dim
    );
    ensure!(
        output.hidden_dim == q_dim && output.seq_len == seq_len,
        "output shape mismatch for non-paged prefill"
    );
    let max_seq_len = k_cache.len / (num_kv_heads * head_dim);
    let kv_len = start_pos + seq_len;
    ensure!(
        kv_len <= max_seq_len,
        "non-paged prefill kv_len {kv_len} exceeds max_seq_len {max_seq_len}"
    );

    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (kc_ptr, _gkc) = k_cache.data.device_ptr(&ctx.stream);
    let (vc_ptr, _gvc) = v_cache.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    unsafe {
        ffi::nonpaged_prefill_attention_cuda(
            q_ptr as *const ffi::Half,
            kc_ptr as *const ffi::Half,
            vc_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            seq_len as i32,
            kv_len as i32,
            max_seq_len as i32,
            sm_scale,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Legacy non-paged HD128 prefill entrypoint.
///
/// The serving path uses TileLang paged prefill. This native CUDA fallback keeps
/// direct/no-pool model entrypoints working without external attention runtimes.
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
    ensure!(num_kv_heads > 0, "num_kv_heads must be > 0");

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

        unsafe {
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
        }
    }

    nonpaged_prefill_attention_into(
        ctx,
        q_batch,
        k_cache,
        v_cache,
        output,
        num_q_heads,
        num_kv_heads,
        head_dim,
        start_pos,
    )
}

/// Legacy non-paged HD256 prefill entrypoint.
#[allow(clippy::too_many_arguments)]
pub(crate) fn nonpaged_prefill_hd256_into(
    ctx: &DeviceContext,
    q_batch: &HiddenStates,
    k_cache: &DeviceVec,
    v_cache: &DeviceVec,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    start_pos: usize,
) -> Result<()> {
    nonpaged_prefill_attention_into(
        ctx,
        q_batch,
        k_cache,
        v_cache,
        output,
        num_q_heads,
        num_kv_heads,
        256,
        start_pos,
    )
}

/// Legacy Qwen3.5 full-attention prefill without a paged KV pool.
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

    nonpaged_prefill_hd256_into(
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

// ============================================================================
// Paged-KV prefill (Phase 2 — consumes Phase 1 FFI)
//
// Callers pass in a paged KV pool + per-slot page indices (GPU-resident) +
// per-forward metadata reused across layers.
// Unlike the contiguous prefill path this writes K/V directly into pool pages
// via page-table indirection — no migrate_kv_range_to_paged step needed
// afterward.
//
// ============================================================================

/// One packed prefill sequence inside a paged-prefill forward.
///
/// `token_offset` and `page_table_offset` are offsets into the packed token and
/// page-table buffers for this forward. Callers must pack both contiguously in
/// request order — `PagedPrefillForward` validates that contract when it builds
/// the TileLang indptr metadata.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PagedPrefillSequence {
    pub token_offset: usize,
    pub seq_len: usize,
    pub start_pos: usize,
    pub page_table_offset: usize,
    pub num_pages: usize,
}

/// Paged-KV prefill metadata for one layer of a packed varlen batch.
pub(crate) struct PagedPrefillMeta<'a> {
    pub pool: &'a PagedKVPool,
    pub layer_idx: usize,
    /// Concatenated page-table rows for every packed sequence in batch order.
    pub page_indices: &'a CudaSlice<i32>,
    pub sequences: &'a [PagedPrefillSequence],
    pub page_size: usize,
}

fn paged_prefill_last_page_len(kv_len: usize, page_size: usize) -> i32 {
    if kv_len == 0 {
        return 0;
    }
    ((kv_len - 1) % page_size + 1) as i32
}

/// Per-forward scratch that holds uploaded indptr/last-page-len device
/// buffers. Built once before the per-layer loop and passed by `&mut` to each
/// layer's attention call.
///
/// TileLang consumes these device buffers directly and does not require a
/// CPU-side attention plan.
pub(crate) struct PagedPrefillForward {
    pub qo_indptr_dev: CudaSlice<i32>,
    pub kv_indptr_dev: CudaSlice<i32>,
    pub kv_last_page_len_dev: CudaSlice<i32>,
    pub batch_size: usize,
    pub total_qo_rows: usize,
    pub page_size: usize,
}

impl PagedPrefillForward {
    /// Upload indptrs ONCE for the whole TileLang forward. HD128 flavour.
    pub(crate) fn new_hd128(
        ctx: &DeviceContext,
        sequences: &[PagedPrefillSequence],
        page_size: usize,
    ) -> Result<Self> {
        Self::new_inner(ctx, sequences, page_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        ctx: &DeviceContext,
        sequences: &[PagedPrefillSequence],
        page_size: usize,
    ) -> Result<Self> {
        ensure!(
            !sequences.is_empty(),
            "paged prefill forward requires at least one sequence"
        );

        let mut total_qo_rows = 0usize;
        let mut total_pages = 0usize;
        let mut qo_indptr = Vec::with_capacity(sequences.len() + 1);
        let mut kv_indptr = Vec::with_capacity(sequences.len() + 1);
        let mut kv_last_page_len = Vec::with_capacity(sequences.len());
        qo_indptr.push(0);
        kv_indptr.push(0);

        for seq in sequences {
            ensure!(seq.seq_len > 0, "paged prefill sequence must not be empty");
            ensure!(
                seq.token_offset == total_qo_rows,
                "paged prefill token packing gap/overlap: expected offset {}, got {}",
                total_qo_rows,
                seq.token_offset
            );
            ensure!(
                seq.page_table_offset == total_pages,
                "paged prefill page-table packing gap/overlap: expected offset {}, got {}",
                total_pages,
                seq.page_table_offset
            );

            let kv_len = seq.start_pos + seq.seq_len;
            let num_pages = kv_len.div_ceil(page_size);
            ensure!(
                seq.num_pages == num_pages,
                "paged prefill sequence page count mismatch: expected {}, got {}",
                num_pages,
                seq.num_pages
            );

            total_qo_rows += seq.seq_len;
            total_pages += seq.num_pages;
            qo_indptr.push(total_qo_rows as i32);
            kv_indptr.push(total_pages as i32);
            kv_last_page_len.push(paged_prefill_last_page_len(kv_len, page_size));
        }

        let qo_indptr_dev: CudaSlice<i32> = ctx
            .stream
            .clone_htod(&qo_indptr)
            .map_err(|e| anyhow!("qo_indptr H2D failed: {e}"))?;
        let kv_indptr_dev: CudaSlice<i32> = ctx
            .stream
            .clone_htod(&kv_indptr)
            .map_err(|e| anyhow!("kv_indptr H2D failed: {e}"))?;
        let kv_last_page_len_dev: CudaSlice<i32> = ctx
            .stream
            .clone_htod(&kv_last_page_len)
            .map_err(|e| anyhow!("kv_last_page_len H2D failed: {e}"))?;

        Ok(Self {
            qo_indptr_dev,
            kv_indptr_dev,
            kv_last_page_len_dev,
            batch_size: sequences.len(),
            total_qo_rows,
            page_size,
        })
    }
}

/// Qwen3-style HD128 paged prefill — per-layer kernels only.
///
/// Structural contract: the caller MUST have built a `PagedPrefillForward`
/// via `PagedPrefillForward::new_hd128` BEFORE the per-layer loop and
/// passes it by `&mut` into each layer. That struct holds the pre-uploaded
/// qo/kv indptrs. This function only runs:
///  1. QK norm + RoPE + paged K/V write (per-layer, touches per-layer K/V
///     pool pointers).
///  2. TileLang paged-prefill HD128 `_run`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_attention_paged_batch(
    ctx: &DeviceContext,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    nrp: &NormRopeParams,
    meta: &PagedPrefillMeta,
    fwd: &mut PagedPrefillForward,
    output: &mut HiddenStates,
    heads: &HeadConfig,
) -> Result<()> {
    let seq_len = q_batch.seq_len;
    let num_q_heads = heads.num_q_heads;
    let num_kv_heads = heads.num_kv_heads;
    let head_dim = heads.head_dim;
    assert!(num_kv_heads > 0, "num_kv_heads must be > 0");
    assert_eq!(head_dim, 128, "prefill_attention_paged_batch is HD128 only");
    assert_eq!(seq_len, fwd.total_qo_rows, "fwd.total_qo_rows mismatch");
    assert_eq!(meta.page_size, fwd.page_size, "fwd.page_size mismatch");
    assert_eq!(
        meta.sequences.len(),
        fwd.batch_size,
        "fwd.batch_size mismatch"
    );
    let page_size = meta.page_size;

    let tilelang_kernel = {
        ensure!(
            page_size == 16,
            "TileLang prefill HD128 kernel requires page_size=16, got {page_size}"
        );
        match (num_q_heads, num_kv_heads) {
            (16, 8) => ffi::tilelang_batch_prefill_paged_hd128_q16_kv8_run_cuda,
            (32, 8) => ffi::tilelang_batch_prefill_paged_hd128_q32_kv8_run_cuda,
            (40, 8) => ffi::tilelang_batch_prefill_paged_hd128_q40_kv8_run_cuda,
            (64, 8) => ffi::tilelang_batch_prefill_paged_hd128_q64_kv8_run_cuda,
            other => {
                return Err(anyhow!(
                    "TileLang: no specialized prefill HD128 kernel for \
                     (num_q_heads, num_kv_heads) = {other:?}; supported configs \
                     are (16,8), (32,8), (40,8), (64,8). Extend SUPPORTED_HEADS \
                     in tools/tilelang/batch_prefill_paged_hd128.py, \
                     TILELANG_PREFILL_HD128_HEAD_CONFIGS in cuda-kernels/build.rs, \
                     and the FFI macro + this match in lockstep, then rebuild."
                ));
            }
        }
    };

    // Step 1: QK norm + RoPE + paged K/V write. The prep kernel is still
    // single-sequence, so we launch it once per packed sequence before the
    // single batched paged-prefill run below.
    unsafe {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr_mut(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = nrp.q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = nrp.k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = nrp.cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = nrp.sin_cache.data.device_ptr(&ctx.stream);
        let (pt_ptr, _gpt) = meta.page_indices.device_ptr(&ctx.stream);
        let kp_ptr = meta.pool.k_ptr(meta.layer_idx, &ctx.stream);
        let vp_ptr = meta.pool.v_ptr(meta.layer_idx, &ctx.stream);

        let q_stride = q_batch.hidden_dim;
        let kv_stride = k_batch.hidden_dim;
        let half_size = std::mem::size_of::<ffi::Half>();
        let i32_size = std::mem::size_of::<i32>();

        for seq in meta.sequences {
            let q_ptr_offset =
                (q_ptr as usize + seq.token_offset * q_stride * half_size) as *mut ffi::Half;
            let k_ptr_offset =
                (k_ptr as usize + seq.token_offset * kv_stride * half_size) as *mut ffi::Half;
            let v_ptr_offset =
                (v_ptr as usize + seq.token_offset * kv_stride * half_size) as *const ffi::Half;
            let pt_ptr_offset = (pt_ptr as usize + seq.page_table_offset * i32_size) as *const i32;

            ffi::prefill_attention_paged_prep_cuda(
                q_ptr_offset,
                k_ptr_offset,
                v_ptr_offset,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                pt_ptr_offset,
                page_size as i32,
                kp_ptr as *mut ffi::Half,
                vp_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                seq.seq_len as i32,
                seq.start_pos as i32,
                nrp.rms_eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }

    // Step 2: run TileLang paged prefill.
    let (q_u64, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (o_u64, _go) = output.data.device_ptr_mut(&ctx.stream);
    let kp_u64 = meta.pool.k_ptr(meta.layer_idx, &ctx.stream);
    let vp_u64 = meta.pool.v_ptr(meta.layer_idx, &ctx.stream);
    let (qoi_u64, _gqoi) = fwd.qo_indptr_dev.device_ptr(&ctx.stream);
    let (kvi_u64, _gkvi) = fwd.kv_indptr_dev.device_ptr(&ctx.stream);
    let (kvidx_u64, _gkvidx) = meta.page_indices.device_ptr(&ctx.stream);
    let (kvlpl_u64, _gkvlpl) = fwd.kv_last_page_len_dev.device_ptr(&ctx.stream);

    {
        let max_qlen = meta.sequences.iter().map(|s| s.seq_len).max().unwrap_or(0) as i32;
        let sm_scale = 1.0_f32 / (head_dim as f32).sqrt();
        // TileLang 0.1.9 auto-promotes T.symbolic shape vars into kernel
        // arguments; the wrapper needs concrete values for the K/V pool
        // capacity (`num_pages`) and the per-batch page-table length
        // (`total_pages`). Both come from the runtime metadata.
        let num_pages = meta.pool.max_total_pages as i32;
        let total_pages = meta.sequences.iter().map(|s| s.num_pages).sum::<usize>() as i32;
        unsafe {
            tilelang_kernel(
                q_u64 as *mut ffi::Half,
                qoi_u64 as *const i32,
                kp_u64 as *mut ffi::Half,
                vp_u64 as *mut ffi::Half,
                kvi_u64 as *const i32,
                kvidx_u64 as *const i32,
                kvlpl_u64 as *const i32,
                o_u64 as *mut ffi::Half,
                fwd.batch_size as i32,
                fwd.total_qo_rows as i32,
                max_qlen,
                num_pages,
                total_pages,
                num_q_heads as i32,
                num_kv_heads as i32,
                page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }

    Ok(())
}

/// HD256 paged-prefill TileLang run step. The Qwen3.5 callers do their HD256 prep
/// kernel inline (it's distinct from the HD128 prep contract used by
/// `prefill_attention_paged_batch`), so this helper covers the run step
/// only — qwen35/prefill.rs keeps prep + this run alongside each other.
///
/// Dispatches to the AOT-specialized TileLang HD256 cubin family
/// `tilelang_batch_prefill_paged_hd256_q{Q}_kv{KV}_run_cuda`. The kernel
/// signature and varlen Q / paged-KV semantics are identical to the HD128
/// twin — only the baked `head_dim` differs.
///
/// `max_qlen` and `total_pages` feed TileLang 0.1.9 symbolic shape arguments.
///
/// Caller responsibility:
///   - Pre-prepped Q lives in `q_ptr` (HD256 RoPE/QK-norm/KV-write done).
///   - `qo_indptr_ptr` / `kv_indptr_ptr` / `kv_indices_ptr` / `kv_last_page_len_ptr`
///     are GPU-resident, indexable by `batch_size` requests.
///   - `k_pool_ptr` / `v_pool_ptr` are the per-layer paged-KV base pointers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_attention_paged_run_hd256(
    ctx: &DeviceContext,
    q_ptr: u64,
    qo_indptr_ptr: u64,
    k_pool_ptr: u64,
    v_pool_ptr: u64,
    kv_indptr_ptr: u64,
    kv_indices_ptr: u64,
    kv_last_page_len_ptr: u64,
    output_ptr: u64,
    kv_pool: &cuda_kernels::TokenKVPool,
    batch_size: usize,
    total_q_tokens: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    max_qlen: i32,
    total_pages: i32,
) -> Result<()> {
    {
        ensure!(
            page_size == 16,
            "TileLang prefill HD256 kernel requires page_size=16, got {page_size}"
        );
        let tilelang_kernel = match (num_q_heads, num_kv_heads) {
            (8, 2) => ffi::tilelang_batch_prefill_paged_hd256_q8_kv2_run_cuda,
            (16, 2) => ffi::tilelang_batch_prefill_paged_hd256_q16_kv2_run_cuda,
            (16, 4) => ffi::tilelang_batch_prefill_paged_hd256_q16_kv4_run_cuda,
            other => {
                return Err(anyhow!(
                    "TileLang: no specialized prefill HD256 kernel for \
                     (num_q_heads, num_kv_heads) = {other:?}; supported configs \
                     are (8,2), (16,2), (16,4). Extend SUPPORTED_HEADS \
                     in tools/tilelang/batch_prefill_paged_hd256.py, \
                     TILELANG_PREFILL_HD256_HEAD_CONFIGS in cuda-kernels/build.rs, \
                     and the FFI macro + this match in lockstep, then rebuild."
                ));
            }
        };
        let head_dim = 256;
        let sm_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let num_pages = kv_pool.max_total_pages as i32;
        unsafe {
            tilelang_kernel(
                q_ptr as *mut ffi::Half,
                qo_indptr_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                kv_last_page_len_ptr as *const i32,
                output_ptr as *mut ffi::Half,
                batch_size as i32,
                total_q_tokens as i32,
                max_qlen,
                num_pages,
                total_pages,
                num_q_heads as i32,
                num_kv_heads as i32,
                page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| {
                anyhow!(
                    "tilelang_batch_prefill_paged_hd256 (q{num_q_heads}_kv{num_kv_heads}) failed: {e}"
                )
            })?;
        }
        Ok(())
    }
}

/// Batched fused GQA decode attention (CUDA, split-KV, HEAD_DIM=128).
///
/// Processes B requests in two kernel launches (split-KV + reduce) instead of
/// 2*B launches from the per-request loop. Each request's KV cache is accessed
/// via device pointer arrays.
///
/// Q/K/V are already in contiguous batch buffers `[B, dim]`. Output is written
/// directly to `output` batch buffer `[B, q_dim]`. No D2D copies needed.
///
/// `positions`: `[B]` i32 on GPU — current_pos per request
/// `seq_lens`: `[B]` i32 on GPU — seq_len per request (= pos + 1)
/// `k_cache_ptrs`/`v_cache_ptrs`: `[B]` device pointers on GPU
/// `partial_out/m/l`: pre-allocated FP32 scratch `[B * num_qheads * NUM_KV_SPLITS * ...]`
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

/// Fused GQA Attention for legacy decode (custom CUDA split-KV, HEAD_DIM=128).
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

/// TileLang tensor-core-shaped batched paged-attention run step.
///
/// `batch_size` is the number of requests (qo_indptr length minus 1), which
/// in mixed decode+prefill batches differs from `q_batch.seq_len` (the total
/// Q row count).
///
/// `max_qlen` and `total_pages` feed TileLang 0.1.9 symbolic shape arguments.
#[allow(clippy::too_many_arguments)]
pub fn tilelang_tc_run_layer(
    ctx: &DeviceContext,
    q_batch: &HiddenStates,
    qo_indptr_gpu: &CudaSlice<i32>,
    kv_pool: &PagedKVPool,
    layer_idx: usize,
    kv_indptr_gpu: &CudaSlice<i32>,
    kv_indices_gpu: &CudaSlice<i32>,
    kv_last_page_len_gpu: &CudaSlice<i32>,
    output: &mut HiddenStates,
    workspace: &mut TileLangWorkspace,
    heads: &TileLangHeadConfig,
    batch_size: i32,
    max_qlen: i32,
    total_pages: i32,
) -> Result<()> {
    let sm_scale = 1.0 / (heads.head_dim as f32).sqrt();

    let tilelang_kernel = {
        ensure!(
            heads.head_dim == 128,
            "TileLang TC decode alias requires head_dim=128, got {}",
            heads.head_dim
        );
        ensure!(
            heads.page_size == 16,
            "TileLang TC decode alias requires page_size=16, got {}",
            heads.page_size
        );
        match (heads.num_qo_heads, heads.num_kv_heads) {
            (16, 8) => ffi::tilelang_batch_prefill_paged_hd128_q16_kv8_run_cuda,
            (32, 8) => ffi::tilelang_batch_prefill_paged_hd128_q32_kv8_run_cuda,
            (40, 8) => ffi::tilelang_batch_prefill_paged_hd128_q40_kv8_run_cuda,
            (64, 8) => ffi::tilelang_batch_prefill_paged_hd128_q64_kv8_run_cuda,
            other => {
                return Err(anyhow!(
                    "TileLang: no specialized TC-decode HD128 kernel for \
                     (num_qo_heads, num_kv_heads) = {other:?}; supported configs \
                     are (16,8), (32,8), (40,8), (64,8). Extend SUPPORTED_HEADS \
                     in tools/tilelang/batch_prefill_paged_hd128.py, \
                     TILELANG_PREFILL_HD128_HEAD_CONFIGS in cuda-kernels/build.rs, \
                     and the FFI macro + this match in lockstep, then rebuild."
                ));
            }
        }
    };

    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (qoi_ptr, _gqoi) = qo_indptr_gpu.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (ind_ptr, _gind) = kv_indptr_gpu.device_ptr(&ctx.stream);
    let (idx_ptr, _gidx) = kv_indices_gpu.device_ptr(&ctx.stream);
    let (lp_ptr, _glp) = kv_last_page_len_gpu.device_ptr(&ctx.stream);

    let k_pool_ptr = kv_pool.k_ptr(layer_idx, &ctx.stream);
    let v_pool_ptr = kv_pool.v_ptr(layer_idx, &ctx.stream);

    {
        let _ = workspace; // TileLang is plan-less; workspace is unused here.
        let num_pages = kv_pool.max_total_pages as i32;
        unsafe {
            tilelang_kernel(
                q_ptr as *mut ffi::Half,
                qoi_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                ind_ptr as *const i32,
                idx_ptr as *const i32,
                lp_ptr as *const i32,
                o_ptr as *mut ffi::Half,
                batch_size,
                q_batch.seq_len as i32,
                max_qlen,
                num_pages,
                total_pages,
                heads.num_qo_heads as i32,
                heads.num_kv_heads as i32,
                heads.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| {
                anyhow!(
                    "tilelang_batch_prefill_paged_hd128 (TC decode alias, q{}_kv{}) failed: {e}",
                    heads.num_qo_heads,
                    heads.num_kv_heads
                )
            })?;
        }
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

/// TileLang HD256 batched paged-decode run step for Qwen3.5 full-attention layers.
///
/// Dispatches to the AOT-specialized TileLang HD256 cubin family
/// `tilelang_batch_decode_paged_hd256_q{Q}_kv{KV}_run_cuda`. The kernel
/// signature matches the HD256 prefill twin — only the cubin internals
/// differ (decode uses qlen=1 per request, no causal mask).
///
/// `batch_size` is the number of requests (qo_indptr length minus 1).
/// For decode, `total_q_tokens == batch_size` (one Q row per request).
///
/// `max_qlen` and `total_pages` feed TileLang 0.1.9 symbolic shape arguments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tilelang_run_layer_hd256(
    ctx: &DeviceContext,
    q_batch: &HiddenStates,
    kv_pool: &PagedKVPool,
    layer_idx: usize,
    qo_indptr_gpu: &CudaSlice<i32>,
    kv_indptr_gpu: &CudaSlice<i32>,
    kv_indices_gpu: &CudaSlice<i32>,
    kv_last_page_len_gpu: &CudaSlice<i32>,
    output: &mut HiddenStates,
    workspace: &mut TileLangWorkspace,
    heads: &TileLangHeadConfig,
    batch_size: i32,
    max_qlen: i32,
    total_pages: i32,
) -> Result<()> {
    let sm_scale = 1.0 / (heads.head_dim as f32).sqrt();

    let tilelang_kernel = {
        ensure!(
            heads.head_dim == 256,
            "TileLang decode HD256 kernel requires head_dim=256, got {}",
            heads.head_dim
        );
        ensure!(
            heads.page_size == 16,
            "TileLang decode HD256 kernel requires page_size=16, got {}",
            heads.page_size
        );
        match (heads.num_qo_heads, heads.num_kv_heads) {
            (8, 2) => ffi::tilelang_batch_decode_paged_hd256_q8_kv2_run_cuda,
            (16, 2) => ffi::tilelang_batch_decode_paged_hd256_q16_kv2_run_cuda,
            (16, 4) => ffi::tilelang_batch_decode_paged_hd256_q16_kv4_run_cuda,
            other => {
                return Err(anyhow!(
                    "TileLang: no specialized decode HD256 kernel for \
                     (num_qo_heads, num_kv_heads) = {other:?}; supported configs \
                     are (8,2), (16,2), (16,4). Extend SUPPORTED_HEADS \
                     in tools/tilelang/batch_decode_paged_hd256.py, \
                     TILELANG_DECODE_HD256_HEAD_CONFIGS in cuda-kernels/build.rs, \
                     and the FFI macro + this match in lockstep, then rebuild."
                ));
            }
        }
    };

    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (qoi_ptr, _gqoi) = qo_indptr_gpu.device_ptr(&ctx.stream);
    let (ind_ptr, _gind) = kv_indptr_gpu.device_ptr(&ctx.stream);
    let (idx_ptr, _gidx) = kv_indices_gpu.device_ptr(&ctx.stream);
    let (lp_ptr, _glp) = kv_last_page_len_gpu.device_ptr(&ctx.stream);

    let k_pool_ptr = kv_pool.k_ptr(layer_idx, &ctx.stream);
    let v_pool_ptr = kv_pool.v_ptr(layer_idx, &ctx.stream);

    {
        let _ = workspace; // TileLang is plan-less; workspace is unused here.
        // Decode: qlen=1 per request, so total_q_tokens == batch_size.
        let total_q_tokens = batch_size;
        let num_pages = kv_pool.max_total_pages as i32;
        unsafe {
            tilelang_kernel(
                q_ptr as *mut ffi::Half,
                qoi_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                ind_ptr as *const i32,
                idx_ptr as *const i32,
                lp_ptr as *const i32,
                o_ptr as *mut ffi::Half,
                batch_size,
                total_q_tokens,
                max_qlen,
                num_pages,
                total_pages,
                heads.num_qo_heads as i32,
                heads.num_kv_heads as i32,
                heads.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| {
                anyhow!(
                    "tilelang_batch_decode_paged_hd256 (q{}_kv{}) failed: {e}",
                    heads.num_qo_heads,
                    heads.num_kv_heads
                )
            })?;
        }
    }

    Ok(())
}

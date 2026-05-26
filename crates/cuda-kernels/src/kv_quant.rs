//! KV cache quantization ops: bf16 ↔ INT8/FP8 per-head per-token symmetric.
//!
//! Also includes fused-dequant decode attention for quantized KV formats
//! that TileLang BF16 attention doesn't support natively (INT8+scale, INT4+scale, etc.).

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

const MAX_TOKEN_ROWS_PER_PAGED_KV_LAUNCH: usize = 65_535;

/// Quantize bf16 KV data → INT8 + f32 scales for tokens `[start_pos..start_pos+token_count)`.
///
/// `kv_bf16`:  bf16 working buffer, HND layout `[num_kv_heads, max_seq_len, head_dim]`
/// `kv_int8`:  INT8 storage, same layout
/// `scales`:   f32 per-head per-token, layout `[num_kv_heads, max_seq_len]`
#[allow(clippy::too_many_arguments)]
pub fn quantize_kv(
    ctx: &DeviceContext,
    kv_bf16: &DeviceVec,
    kv_int8: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    start_pos: usize,
    token_count: usize,
) -> Result<()> {
    if token_count == 0 {
        return Ok(());
    }

    let (bf16_ptr, _g1) = kv_bf16.data.device_ptr(&ctx.stream);
    let (int8_ptr, _g2) = kv_int8.device_ptr_mut(&ctx.stream);
    let (scales_ptr, _g3) = scales.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::quantize_kv_bf16_to_int8_cuda(
            bf16_ptr as *const ffi::Half,
            int8_ptr as *mut i8,
            scales_ptr as *mut f32,
            num_kv_heads as i32,
            head_dim as i32,
            max_seq_len as i32,
            start_pos as i32,
            token_count as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Dequantize INT8 KV data → bf16 for tokens `[0..token_count)`.
///
/// Writes to the bf16 working buffer so attention kernels can read it.
pub fn dequantize_kv(
    ctx: &DeviceContext,
    kv_int8: &CudaSlice<i8>,
    scales: &CudaSlice<f32>,
    kv_bf16: &mut DeviceVec,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    token_count: usize,
) -> Result<()> {
    if token_count == 0 {
        return Ok(());
    }

    let (int8_ptr, _g1) = kv_int8.device_ptr(&ctx.stream);
    let (scales_ptr, _g2) = scales.device_ptr(&ctx.stream);
    let (bf16_ptr, _g3) = kv_bf16.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::dequantize_kv_int8_to_bf16_cuda(
            int8_ptr as *const i8,
            scales_ptr as *const f32,
            bf16_ptr as *mut ffi::Half,
            num_kv_heads as i32,
            head_dim as i32,
            max_seq_len as i32,
            token_count as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

// ─── Paged pool INT8 quantization ops (NHD layout) ───

/// Dequantize all tokens from INT8 paged pool → bf16 working buffer.
///
/// Raw pointers (u64) are used because the pool's INT8/scales/work buffers may
/// be different types (`CudaSlice<i8>`, `CudaSlice<f32>`, `CudaSlice<u16>`).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn dequantize_paged_kv(
    ctx: &DeviceContext,
    kv_int8_ptr: u64,
    kv_scales_ptr: u64,
    kv_bf16_ptr: u64,
    token_indices_gpu: &CudaSlice<i32>,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    total_tokens: usize,
) -> Result<()> {
    if total_tokens == 0 {
        return Ok(());
    }

    let (ti_ptr, _gti) = token_indices_gpu.device_ptr(&ctx.stream);

    unsafe {
        ffi::dequantize_paged_kv_cuda(
            kv_int8_ptr as *const i8,
            kv_scales_ptr as *const f32,
            kv_bf16_ptr as *mut ffi::Half,
            ti_ptr as *const i32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            total_tokens as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

// ─── FP8 E4M3 paged pool ops ───

/// Quantize 1 new token per request: bf16 working → FP8 E4M3 paged pool.
/// No separate scale — FP8 E4M3 is self-contained.
#[allow(clippy::too_many_arguments)]
pub fn quantize_paged_kv_fp8(
    ctx: &DeviceContext,
    kv_bf16_ptr: u64,
    kv_fp8_ptr: u64,
    scales_ptr: u64,
    new_token_indices_gpu: &CudaSlice<i32>,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    batch_size: usize,
) -> Result<()> {
    if batch_size == 0 {
        return Ok(());
    }
    let mut offset = 0usize;
    while offset < batch_size {
        let chunk_tokens = (batch_size - offset).min(MAX_TOKEN_ROWS_PER_PAGED_KV_LAUNCH);
        let rows = new_token_indices_gpu.slice(offset..offset + chunk_tokens);
        let (nti_ptr, _g) = rows.device_ptr(&ctx.stream);
        unsafe {
            ffi::quantize_paged_kv_fp8_cuda(
                kv_bf16_ptr as *const ffi::Half,
                kv_fp8_ptr as *mut u8,
                scales_ptr as *mut f32,
                nti_ptr as *const i32,
                num_kv_heads as i32,
                head_dim as i32,
                kv_dim as i32,
                chunk_tokens as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        offset += chunk_tokens;
    }
    Ok(())
}

/// **KIVI per-channel K quantize.** Reads a pre-computed
/// `[num_kv_heads, head_dim]` f32 scale table (populated via
/// [`compute_k_per_channel_absmax`] + [`finalize_k_per_channel_scales`])
/// and quantizes bf16 K → FP8 E4M3 without per-(token, head) absmax
/// reduction. See
/// `docs/plans/2026-05-26-fp8-kv-per-channel-k-fix.md` for the rationale.
#[allow(clippy::too_many_arguments)]
pub fn quantize_paged_kv_fp8_per_channel(
    ctx: &DeviceContext,
    kv_bf16_ptr: u64,
    kv_fp8_ptr: u64,
    k_static_scales_ptr: u64,
    new_token_indices_gpu: &CudaSlice<i32>,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    batch_size: usize,
) -> Result<()> {
    if batch_size == 0 {
        return Ok(());
    }
    let mut offset = 0usize;
    while offset < batch_size {
        let chunk_tokens = (batch_size - offset).min(MAX_TOKEN_ROWS_PER_PAGED_KV_LAUNCH);
        let rows = new_token_indices_gpu.slice(offset..offset + chunk_tokens);
        let (nti_ptr, _g) = rows.device_ptr(&ctx.stream);
        unsafe {
            ffi::quantize_paged_kv_fp8_per_channel_cuda(
                kv_bf16_ptr as *const ffi::Half,
                kv_fp8_ptr as *mut u8,
                k_static_scales_ptr as *const f32,
                nti_ptr as *const i32,
                num_kv_heads as i32,
                head_dim as i32,
                kv_dim as i32,
                chunk_tokens as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        offset += chunk_tokens;
    }
    Ok(())
}

/// **KIVI calibration step 1**: accumulate per-(kv_head, head_dim) absmax
/// over a batch of K rows from the bf16 HND-paged work buffer. Stores
/// raw absmax (not divided by 448) so multiple batches can be
/// accumulated; call [`finalize_k_per_channel_scales`] once when done.
#[allow(clippy::too_many_arguments)]
pub fn compute_k_per_channel_absmax(
    ctx: &DeviceContext,
    kv_bf16_ptr: u64,
    k_static_scales_ptr: u64,
    token_rows_gpu: &CudaSlice<i32>,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    batch_size: usize,
) -> Result<()> {
    if batch_size == 0 {
        return Ok(());
    }
    let mut offset = 0usize;
    while offset < batch_size {
        let chunk_tokens = (batch_size - offset).min(MAX_TOKEN_ROWS_PER_PAGED_KV_LAUNCH);
        let rows = token_rows_gpu.slice(offset..offset + chunk_tokens);
        let (nti_ptr, _g) = rows.device_ptr(&ctx.stream);
        unsafe {
            ffi::compute_k_per_channel_absmax_cuda(
                kv_bf16_ptr as *const ffi::Half,
                k_static_scales_ptr as *mut f32,
                nti_ptr as *const i32,
                num_kv_heads as i32,
                head_dim as i32,
                kv_dim as i32,
                chunk_tokens as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        offset += chunk_tokens;
    }
    Ok(())
}

/// **KIVI calibration step 2**: divide accumulated absmax by 448 (FP8
/// E4M3 max) to produce the final per-channel scale. Idempotent only if
/// called exactly once after all absmax-accumulation batches.
pub fn finalize_k_per_channel_scales(
    ctx: &DeviceContext,
    k_static_scales_ptr: u64,
    num_channels: usize,
) -> Result<()> {
    if num_channels == 0 {
        return Ok(());
    }
    unsafe {
        ffi::finalize_k_per_channel_scales_cuda(
            k_static_scales_ptr as *mut f32,
            num_channels as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Quantize + scatter contiguous bf16 KV → FP8 paged pool (for prefill→pool migration).
#[allow(clippy::too_many_arguments)]
pub fn quantize_scatter_kv_fp8(
    ctx: &DeviceContext,
    kv_cont: &DeviceVec,
    kv_fp8_ptr: u64,
    scales_ptr: u64,
    page_indices_gpu: &CudaSlice<i32>,
    max_seq_len: usize,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
) -> Result<()> {
    if seq_len == 0 {
        return Ok(());
    }
    let (cont_ptr, _g1) = kv_cont.data.device_ptr(&ctx.stream);
    let (pi_ptr, _g2) = page_indices_gpu.device_ptr(&ctx.stream);
    unsafe {
        ffi::quantize_scatter_kv_fp8_cuda(
            cont_ptr as *const ffi::Half,
            kv_fp8_ptr as *mut u8,
            scales_ptr as *mut f32,
            pi_ptr as *const i32,
            max_seq_len as i32,
            seq_len as i32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Quantize + scatter a contiguous bf16 KV range `[start_pos, start_pos + token_count)`.
#[allow(clippy::too_many_arguments)]
pub fn quantize_scatter_kv_fp8_range(
    ctx: &DeviceContext,
    kv_cont: &DeviceVec,
    kv_fp8_ptr: u64,
    scales_ptr: u64,
    page_indices_gpu: &CudaSlice<i32>,
    start_pos: usize,
    max_seq_len: usize,
    token_count: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
) -> Result<()> {
    if token_count == 0 {
        return Ok(());
    }
    let (cont_ptr, _g1) = kv_cont.data.device_ptr(&ctx.stream);
    let (pi_ptr, _g2) = page_indices_gpu.device_ptr(&ctx.stream);
    unsafe {
        ffi::quantize_scatter_kv_fp8_range_cuda(
            cont_ptr as *const ffi::Half,
            kv_fp8_ptr as *mut u8,
            scales_ptr as *mut f32,
            pi_ptr as *const i32,
            start_pos as i32,
            max_seq_len as i32,
            token_count as i32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Dequantize durable FP8 NHD token rows into the BF16 HND paged work buffer.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_paged_kv_fp8_to_hnd(
    ctx: &DeviceContext,
    kv_fp8_ptr: u64,
    scales_ptr: u64,
    kv_bf16_hnd_ptr: u64,
    token_rows_gpu: &CudaSlice<i32>,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    total_tokens: usize,
) -> Result<()> {
    if total_tokens == 0 {
        return Ok(());
    }
    let mut offset = 0usize;
    while offset < total_tokens {
        let chunk_tokens = (total_tokens - offset).min(MAX_TOKEN_ROWS_PER_PAGED_KV_LAUNCH);
        let rows = token_rows_gpu.slice(offset..offset + chunk_tokens);
        let (rows_ptr, _g) = rows.device_ptr(&ctx.stream);
        unsafe {
            ffi::dequantize_paged_kv_fp8_to_hnd_cuda(
                kv_fp8_ptr as *const u8,
                scales_ptr as *const f32,
                kv_bf16_hnd_ptr as *mut ffi::Half,
                rows_ptr as *const i32,
                num_kv_heads as i32,
                head_dim as i32,
                kv_dim as i32,
                chunk_tokens as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        offset += chunk_tokens;
    }
    Ok(())
}

/// Dequantize durable INT8 NHD token rows into the BF16 HND paged work buffer.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_paged_kv_int8_to_hnd(
    ctx: &DeviceContext,
    kv_int8_ptr: u64,
    scales_ptr: u64,
    kv_bf16_hnd_ptr: u64,
    token_rows_gpu: &CudaSlice<i32>,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    total_tokens: usize,
) -> Result<()> {
    if total_tokens == 0 {
        return Ok(());
    }
    let mut offset = 0usize;
    while offset < total_tokens {
        let chunk_tokens = (total_tokens - offset).min(MAX_TOKEN_ROWS_PER_PAGED_KV_LAUNCH);
        let rows = token_rows_gpu.slice(offset..offset + chunk_tokens);
        let (rows_ptr, _g) = rows.device_ptr(&ctx.stream);
        unsafe {
            ffi::dequantize_paged_kv_int8_to_hnd_cuda(
                kv_int8_ptr as *const i8,
                scales_ptr as *const f32,
                kv_bf16_hnd_ptr as *mut ffi::Half,
                rows_ptr as *const i32,
                num_kv_heads as i32,
                head_dim as i32,
                kv_dim as i32,
                chunk_tokens as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        offset += chunk_tokens;
    }
    Ok(())
}

// ─── Fused-dequant decode attention (INT8+scale) ───

/// Workspace size for split-KV fused INT8 decode attention.
pub fn decode_attention_int8_workspace_bytes(
    batch_size: usize,
    num_qo_heads: usize,
    head_dim: usize,
    num_splits: usize,
) -> usize {
    unsafe {
        ffi::decode_attention_int8_workspace_bytes(
            batch_size as i32,
            num_qo_heads as i32,
            head_dim as i32,
            num_splits as i32,
        )
    }
}

/// Decode attention with fused INT8 dequantization (split-KV optimized).
///
/// Reads quantized INT8 K/V + f32 scales directly from the paged pool,
/// dequants in registers, computes attention. Split-KV across multiple
/// blocks per query head for GPU saturation.
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_int8(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k_data_ptr: u64,
    v_data_ptr: u64,
    k_scales_ptr: u64,
    v_scales_ptr: u64,
    kv_indices: &CudaSlice<i32>,
    kv_meta: &CudaSlice<i32>,
    o: &mut HiddenStates,
    batch_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    sm_scale: f32,
    workspace: &CudaSlice<u8>,
    workspace_bytes: usize,
) -> Result<()> {
    if batch_size == 0 {
        return Ok(());
    }
    let (q_ptr, _g1) = q.data.device_ptr(&ctx.stream);
    let (ki_ptr, _g2) = kv_indices.device_ptr(&ctx.stream);
    let (ip_ptr, _g3) = kv_meta.device_ptr(&ctx.stream);
    let (o_ptr, _g4) = o.data.device_ptr_mut(&ctx.stream);
    let (ws_ptr, _g5) = workspace.device_ptr(&ctx.stream);
    unsafe {
        ffi::decode_attention_int8_cuda(
            q_ptr as *const ffi::Half,
            k_data_ptr as *const i8,
            v_data_ptr as *const i8,
            k_scales_ptr as *const f32,
            v_scales_ptr as *const f32,
            ki_ptr as *const i32,
            ip_ptr as *const i32,
            o_ptr as *mut ffi::Half,
            batch_size as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            sm_scale,
            ctx.stream.cu_stream(),
            ws_ptr as *mut u8,
            workspace_bytes,
        )
        .result()?;
    }
    Ok(())
}

/// Decode attention with fused FP8 E4M3 dequantization (split-KV).
///
/// Same architecture as INT8 variant with per-token/per-head FP8 scales.
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_fp8(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k_data_ptr: u64,
    v_data_ptr: u64,
    k_scales_ptr: u64,
    v_scales_ptr: u64,
    kv_indices: &CudaSlice<i32>,
    kv_meta: &CudaSlice<i32>,
    o: &mut HiddenStates,
    batch_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    sm_scale: f32,
    workspace: &CudaSlice<u8>,
    workspace_bytes: usize,
) -> Result<()> {
    if batch_size == 0 {
        return Ok(());
    }
    let (q_ptr, _g1) = q.data.device_ptr(&ctx.stream);
    let (ki_ptr, _g2) = kv_indices.device_ptr(&ctx.stream);
    let (ip_ptr, _g3) = kv_meta.device_ptr(&ctx.stream);
    let (o_ptr, _g4) = o.data.device_ptr_mut(&ctx.stream);
    let (ws_ptr, _g5) = workspace.device_ptr(&ctx.stream);
    unsafe {
        ffi::decode_attention_fp8_cuda(
            q_ptr as *const ffi::Half,
            k_data_ptr as *const u8,
            v_data_ptr as *const u8,
            k_scales_ptr as *const f32,
            v_scales_ptr as *const f32,
            ki_ptr as *const i32,
            ip_ptr as *const i32,
            o_ptr as *mut ffi::Half,
            batch_size as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            sm_scale,
            ctx.stream.cu_stream(),
            ws_ptr as *mut u8,
            workspace_bytes,
        )
        .result()?;
    }
    Ok(())
}

/// **KIVI per-channel K decode attention.** Same shape as
/// [`decode_attention_fp8`] but K reads a `[num_kv_heads, head_dim]` static
/// scale table instead of per-(row, head) scales. V keeps per-(row, head)
/// scales.
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_fp8_per_channel_k(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k_data_ptr: u64,
    v_data_ptr: u64,
    k_static_scales_ptr: u64,
    v_scales_ptr: u64,
    kv_indices: &CudaSlice<i32>,
    kv_meta: &CudaSlice<i32>,
    o: &mut HiddenStates,
    batch_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    sm_scale: f32,
    workspace: &CudaSlice<u8>,
    workspace_bytes: usize,
) -> Result<()> {
    if batch_size == 0 {
        return Ok(());
    }
    let (q_ptr, _g1) = q.data.device_ptr(&ctx.stream);
    let (ki_ptr, _g2) = kv_indices.device_ptr(&ctx.stream);
    let (ip_ptr, _g3) = kv_meta.device_ptr(&ctx.stream);
    let (o_ptr, _g4) = o.data.device_ptr_mut(&ctx.stream);
    let (ws_ptr, _g5) = workspace.device_ptr(&ctx.stream);
    unsafe {
        ffi::decode_attention_fp8_per_channel_k_cuda(
            q_ptr as *const ffi::Half,
            k_data_ptr as *const u8,
            v_data_ptr as *const u8,
            k_static_scales_ptr as *const f32,
            v_scales_ptr as *const f32,
            ki_ptr as *const i32,
            ip_ptr as *const i32,
            o_ptr as *mut ffi::Half,
            batch_size as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            sm_scale,
            ctx.stream.cu_stream(),
            ws_ptr as *mut u8,
            workspace_bytes,
        )
        .result()?;
    }
    Ok(())
}

// ─── Varlen-Q + quantized paged KV attention (mixed batch path) ───
//
// Generalization of `decode_attention_fp8` to mixed prefill+decode batches.
// Reads FP8 KV directly from the pool (no bf16 shadow); enables lifting the
// K2 gate in `infer/src/model/qwen3/forward.rs::supports_mixed_batch` once
// the kernel is wired into `decode_batch_with_prefill`.
//
// HD128 + page_size=16 only for now — same coverage envelope as the qlen=1
// variant. FP8 and INT8 share the split-KV kernel; `int8_kv` controls how the
// durable bytes are interpreted.
pub const VARLEN_QUANTIZED_MAX_SPLITS: usize = 16;

pub fn decode_attention_varlen_fp8_workspace_bytes(
    total_q_tokens: usize,
    num_q_heads: usize,
    head_dim: usize,
    num_splits: usize,
) -> usize {
    unsafe {
        ffi::decode_attention_varlen_fp8_workspace_bytes(
            total_q_tokens as i32,
            num_q_heads as i32,
            head_dim as i32,
            num_splits as i32,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_attention_varlen_fp8(
    ctx: &DeviceContext,
    q_packed: &HiddenStates,
    qo_indptr: &CudaSlice<i32>,
    k_pool_ptr: u64,
    v_pool_ptr: u64,
    k_scales_ptr: Option<u64>,
    v_scales_ptr: Option<u64>,
    kv_indptr: &CudaSlice<i32>,
    kv_indices: &CudaSlice<i32>,
    last_page_len: &CudaSlice<i32>,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    batch_size: usize,
    total_q_tokens: usize,
    max_kv_len: usize,
    int8_kv: bool,
    causal: bool,
    sm_scale: f32,
    workspace: &CudaSlice<u8>,
    workspace_bytes: usize,
) -> Result<()> {
    if batch_size == 0 || total_q_tokens == 0 {
        return Ok(());
    }

    let (q_ptr, _gq) = q_packed.data.device_ptr(&ctx.stream);
    let (qoi_ptr, _gqoi) = qo_indptr.device_ptr(&ctx.stream);
    let (kvi_ptr, _gkvi) = kv_indptr.device_ptr(&ctx.stream);
    let (kvx_ptr, _gkvx) = kv_indices.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = last_page_len.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (ws_ptr, _gws) = workspace.device_ptr(&ctx.stream);

    unsafe {
        ffi::decode_attention_varlen_fp8_cuda(
            q_ptr as *const ffi::Half,
            qoi_ptr as *const i32,
            k_pool_ptr as *const u8,
            v_pool_ptr as *const u8,
            k_scales_ptr.unwrap_or(0) as *const f32,
            v_scales_ptr.unwrap_or(0) as *const f32,
            kvi_ptr as *const i32,
            kvx_ptr as *const i32,
            lpl_ptr as *const i32,
            o_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            page_size as i32,
            batch_size as i32,
            total_q_tokens as i32,
            max_kv_len as i32,
            int8_kv,
            causal,
            sm_scale,
            ctx.stream.cu_stream(),
            ws_ptr as *mut u8,
            workspace_bytes,
        )
        .result()?;
    }
    Ok(())
}

/// Quantize 1 new token per request from bf16 working buffer → INT8 paged pool.
#[allow(clippy::too_many_arguments)]
pub fn quantize_paged_kv_single(
    ctx: &DeviceContext,
    kv_bf16_ptr: u64,
    kv_int8_ptr: u64,
    kv_scales_ptr: u64,
    new_token_indices_gpu: &CudaSlice<i32>,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    batch_size: usize,
) -> Result<()> {
    if batch_size == 0 {
        return Ok(());
    }
    let mut offset = 0usize;
    while offset < batch_size {
        let chunk_tokens = (batch_size - offset).min(MAX_TOKEN_ROWS_PER_PAGED_KV_LAUNCH);
        let rows = new_token_indices_gpu.slice(offset..offset + chunk_tokens);
        let (nti_ptr, _gnti) = rows.device_ptr(&ctx.stream);
        unsafe {
            ffi::quantize_paged_kv_single_cuda(
                kv_bf16_ptr as *const ffi::Half,
                kv_int8_ptr as *mut i8,
                kv_scales_ptr as *mut f32,
                nti_ptr as *const i32,
                num_kv_heads as i32,
                head_dim as i32,
                kv_dim as i32,
                chunk_tokens as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        offset += chunk_tokens;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn fp8_reference(byte: u8) -> f32 {
        match byte {
            0x00 => 0.0,
            0x38 => 1.0,
            0xb8 => -1.0,
            0x40 => 2.0,
            other => panic!("unexpected fp8 test byte: {other:#04x}"),
        }
    }

    fn hnd_offset(
        token_row: usize,
        kv_head: usize,
        d: usize,
        head_dim: usize,
        kv_dim: usize,
    ) -> usize {
        const PAGE_SIZE: usize = 16;
        let page_idx = token_row / PAGE_SIZE;
        let offset_in_page = token_row % PAGE_SIZE;
        page_idx * PAGE_SIZE * kv_dim
            + kv_head * PAGE_SIZE * head_dim
            + offset_in_page * head_dim
            + d
    }

    fn run_hnd_refill_case(ctx: &DeviceContext, head_dim: usize) {
        let num_kv_heads = 3usize;
        let total_tokens = 17usize;
        let kv_dim = num_kv_heads * head_dim;
        let elem_count = total_tokens * kv_dim;
        let hnd_elem_count = total_tokens.div_ceil(16) * 16 * kv_dim;

        let fp8_pattern = [0x00u8, 0x38, 0xb8, 0x40];
        let fp8_host: Vec<u8> = (0..elem_count)
            .map(|idx| fp8_pattern[idx % fp8_pattern.len()])
            .collect();
        let int8_host: Vec<i8> = (0..elem_count).map(|idx| (idx as i8 % 9) - 4).collect();
        let scale_host: Vec<f32> = (0..total_tokens * num_kv_heads)
            .map(|idx| 0.25 + (idx % 5) as f32 * 0.125)
            .collect();
        let token_rows_host: Vec<i32> = (0..total_tokens).map(|idx| idx as i32).collect();

        let fp8_gpu = ctx.stream.clone_htod(&fp8_host).expect("fp8 H2D");
        let int8_gpu = ctx.stream.clone_htod(&int8_host).expect("int8 H2D");
        let scales_gpu = ctx.stream.clone_htod(&scale_host).expect("scales H2D");
        let token_rows_gpu = ctx.stream.clone_htod(&token_rows_host).expect("rows H2D");
        let mut fp8_out = ctx
            .stream
            .alloc_zeros::<u16>(hnd_elem_count)
            .expect("fp8 out");
        let mut int8_out = ctx
            .stream
            .alloc_zeros::<u16>(hnd_elem_count)
            .expect("int8 out");

        {
            let (fp8_ptr, _fp8_guard) = fp8_gpu.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales_gpu.device_ptr(&ctx.stream);
            let (out_ptr, _out_guard) = fp8_out.device_ptr_mut(&ctx.stream);
            dequantize_paged_kv_fp8_to_hnd(
                ctx,
                fp8_ptr,
                scales_ptr,
                out_ptr,
                &token_rows_gpu,
                num_kv_heads,
                head_dim,
                kv_dim,
                total_tokens,
            )
            .expect("fp8 hnd refill");
        }
        {
            let (int8_ptr, _int8_guard) = int8_gpu.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales_gpu.device_ptr(&ctx.stream);
            let (out_ptr, _out_guard) = int8_out.device_ptr_mut(&ctx.stream);
            dequantize_paged_kv_int8_to_hnd(
                ctx,
                int8_ptr,
                scales_ptr,
                out_ptr,
                &token_rows_gpu,
                num_kv_heads,
                head_dim,
                kv_dim,
                total_tokens,
            )
            .expect("int8 hnd refill");
        }

        ctx.sync().expect("sync refill kernels");
        let fp8_got = ctx.stream.clone_dtoh(&fp8_out).expect("fp8 D2H");
        let int8_got = ctx.stream.clone_dtoh(&int8_out).expect("int8 D2H");

        for token_row in 0..total_tokens {
            for kv_head in 0..num_kv_heads {
                let scale = scale_host[token_row * num_kv_heads + kv_head];
                for d in 0..head_dim {
                    let src = token_row * kv_dim + kv_head * head_dim + d;
                    let dst = hnd_offset(token_row, kv_head, d, head_dim, kv_dim);
                    let expected_fp8 =
                        bf16::from_f32(fp8_reference(fp8_host[src]) * scale).to_bits();
                    let expected_int8 = bf16::from_f32(int8_host[src] as f32 * scale).to_bits();
                    assert_eq!(
                        fp8_got[dst], expected_fp8,
                        "fp8 mismatch head_dim={head_dim} token={token_row} head={kv_head} d={d}"
                    );
                    assert_eq!(
                        int8_got[dst], expected_int8,
                        "int8 mismatch head_dim={head_dim} token={token_row} head={kv_head} d={d}"
                    );
                }
            }
        }
    }

    fn run_fp8_scatter_roundtrip_case(ctx: &DeviceContext, head_dim: usize) {
        let num_kv_heads = 3usize;
        let max_seq_len = 19usize;
        let total_tokens = 17usize;
        let kv_dim = num_kv_heads * head_dim;
        let hnd_elem_count = total_tokens.div_ceil(16) * 16 * kv_dim;
        let pattern = [1.0f32, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.125];

        let mut cont_host = vec![bf16::ZERO; num_kv_heads * max_seq_len * head_dim];
        for kv_head in 0..num_kv_heads {
            for token_row in 0..total_tokens {
                for d in 0..head_dim {
                    let value = pattern[(token_row + kv_head + d) % pattern.len()];
                    let src = kv_head * max_seq_len * head_dim + token_row * head_dim + d;
                    cont_host[src] = bf16::from_f32(value);
                }
            }
        }

        let kv_cont = DeviceVec::from_host(ctx, &cont_host).expect("scatter cont H2D");
        let mut kv_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(total_tokens * kv_dim)
            .expect("scatter fp8 alloc");
        let mut scales = ctx
            .stream
            .alloc_zeros::<f32>(total_tokens * num_kv_heads)
            .expect("scatter scales alloc");
        let token_rows_host: Vec<i32> = (0..total_tokens).map(|idx| idx as i32).collect();
        let token_rows_gpu = ctx.stream.clone_htod(&token_rows_host).expect("rows H2D");

        {
            let (fp8_ptr, _fp8_guard) = kv_fp8.device_ptr_mut(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr_mut(&ctx.stream);
            quantize_scatter_kv_fp8_range(
                ctx,
                &kv_cont,
                fp8_ptr,
                scales_ptr,
                &token_rows_gpu,
                0,
                max_seq_len,
                total_tokens,
                num_kv_heads,
                head_dim,
                kv_dim,
            )
            .expect("fp8 scatter quantize");
        }

        let mut hnd_out = ctx
            .stream
            .alloc_zeros::<u16>(hnd_elem_count)
            .expect("scatter hnd alloc");
        {
            let (fp8_ptr, _fp8_guard) = kv_fp8.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr(&ctx.stream);
            let (out_ptr, _out_guard) = hnd_out.device_ptr_mut(&ctx.stream);
            dequantize_paged_kv_fp8_to_hnd(
                ctx,
                fp8_ptr,
                scales_ptr,
                out_ptr,
                &token_rows_gpu,
                num_kv_heads,
                head_dim,
                kv_dim,
                total_tokens,
            )
            .expect("fp8 scatter hnd refill");
        }

        ctx.sync().expect("sync scatter roundtrip");
        let got = ctx.stream.clone_dtoh(&hnd_out).expect("scatter D2H");
        let got_scales = ctx.stream.clone_dtoh(&scales).expect("scales D2H");
        let expected_scale = 1.0f32 / 448.0f32;

        for token_row in 0..total_tokens {
            for kv_head in 0..num_kv_heads {
                let scale = got_scales[token_row * num_kv_heads + kv_head];
                assert!(
                    (scale - expected_scale).abs() < 1.0e-7,
                    "scale mismatch head_dim={head_dim} token={token_row} head={kv_head}: {scale}"
                );
                for d in 0..head_dim {
                    let src = kv_head * max_seq_len * head_dim + token_row * head_dim + d;
                    let dst = hnd_offset(token_row, kv_head, d, head_dim, kv_dim);
                    let expected = cont_host[src].to_f32();
                    let actual = bf16::from_bits(got[dst]).to_f32();
                    assert!(
                        (actual - expected).abs() <= 0.002,
                        "roundtrip mismatch head_dim={head_dim} token={token_row} head={kv_head} d={d}: got {actual}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn hnd_refill_quantized_kv_matches_reference_values() {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        run_hnd_refill_case(&ctx, 8);
        run_hnd_refill_case(&ctx, 7);
    }

    #[test]
    fn fp8_scatter_quantized_kv_roundtrips_representable_values() {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        run_fp8_scatter_roundtrip_case(&ctx, 8);
        run_fp8_scatter_roundtrip_case(&ctx, 7);
    }

    /// Diagnostic for the 2026-05-26 FP8 KV catastrophic step-1 divergence.
    /// The existing `fp8_scatter_quantized_kv_roundtrips_representable_values`
    /// test uses tiny representable values (|val| ≤ 1.0), which fit FP8 E4M3
    /// without loss. The production Qwen3-4B forward produces K/V values
    /// closer to N(0, 1) with occasional ±5 outliers, where FP8 quantization
    /// has measurable relative error. This test exercises the realistic
    /// shape (num_kv_heads=8, head_dim=128) at multi-page coverage so a
    /// shape-specific kernel bug would surface here, while a pure
    /// precision-limit issue would print bounded error numbers without
    /// failing.
    #[test]
    fn fp8_scatter_qwen3_production_layout_diagnostic() {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let num_kv_heads = 8usize;
        let head_dim = 128usize;
        let total_tokens = 64usize;
        let max_seq_len = 128usize;
        let kv_dim = num_kv_heads * head_dim;
        let hnd_elem_count = total_tokens.div_ceil(16) * 16 * kv_dim;

        // Deterministic pseudo-Gaussian via xorshift; values in roughly
        // ±3.5 range with one ±6 outlier per (token, head) to stress the
        // per-(token, head) scale path the same way real attention does.
        let mut rng_state: u64 = 0x9E3779B97F4A7C15;
        let mut next_f32 = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let v = ((rng_state >> 32) as u32) as f32 / u32::MAX as f32;
            (v - 0.5) * 4.0
        };

        let mut cont_host = vec![bf16::ZERO; num_kv_heads * max_seq_len * head_dim];
        for kv_head in 0..num_kv_heads {
            for token_row in 0..total_tokens {
                for d in 0..head_dim {
                    let mut value = next_f32();
                    if d == 0 {
                        // Outlier per (token, head): forces the scale to a
                        // larger value than the bulk of dims would need.
                        value = if (token_row + kv_head) % 2 == 0 {
                            6.0
                        } else {
                            -5.5
                        };
                    }
                    let src = kv_head * max_seq_len * head_dim + token_row * head_dim + d;
                    cont_host[src] = bf16::from_f32(value);
                }
            }
        }

        let kv_cont = DeviceVec::from_host(&ctx, &cont_host).expect("prod cont H2D");
        let mut kv_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(total_tokens * kv_dim)
            .expect("prod fp8 alloc");
        let mut scales = ctx
            .stream
            .alloc_zeros::<f32>(total_tokens * num_kv_heads)
            .expect("prod scales alloc");
        let token_rows_host: Vec<i32> = (0..total_tokens).map(|idx| idx as i32).collect();
        let token_rows_gpu = ctx
            .stream
            .clone_htod(&token_rows_host)
            .expect("prod rows H2D");

        {
            let (fp8_ptr, _g1) = kv_fp8.device_ptr_mut(&ctx.stream);
            let (scales_ptr, _g2) = scales.device_ptr_mut(&ctx.stream);
            quantize_scatter_kv_fp8_range(
                &ctx,
                &kv_cont,
                fp8_ptr,
                scales_ptr,
                &token_rows_gpu,
                0,
                max_seq_len,
                total_tokens,
                num_kv_heads,
                head_dim,
                kv_dim,
            )
            .expect("prod fp8 quantize");
        }

        let mut hnd_out = ctx
            .stream
            .alloc_zeros::<u16>(hnd_elem_count)
            .expect("prod hnd alloc");
        {
            let (fp8_ptr, _g1) = kv_fp8.device_ptr(&ctx.stream);
            let (scales_ptr, _g2) = scales.device_ptr(&ctx.stream);
            let (out_ptr, _g3) = hnd_out.device_ptr_mut(&ctx.stream);
            dequantize_paged_kv_fp8_to_hnd(
                &ctx,
                fp8_ptr,
                scales_ptr,
                out_ptr,
                &token_rows_gpu,
                num_kv_heads,
                head_dim,
                kv_dim,
                total_tokens,
            )
            .expect("prod hnd refill");
        }

        ctx.sync().expect("prod sync");
        let got = ctx.stream.clone_dtoh(&hnd_out).expect("prod D2H");
        let got_scales = ctx.stream.clone_dtoh(&scales).expect("prod scales D2H");

        // Compute per-(token, head) error stats. The expected relative L∞
        // error for FP8 E4M3 with per-(token, head) scaling is ~1/2^3 ≈
        // 12.5% (3 mantissa bits at the mantissa-only range, plus tail). If
        // the kernel itself is bug-free, error stays in that envelope. If
        // we see error > 50% rel or arbitrary garbage, the kernel has a
        // shape/indexing bug specific to this layout.
        let mut max_abs_err = 0.0f32;
        let mut sum_abs_err = 0.0f64;
        let mut max_rel_err = 0.0f32;
        let mut n_elems = 0usize;
        let mut scale_min = f32::INFINITY;
        let mut scale_max = 0.0f32;
        for token_row in 0..total_tokens {
            for kv_head in 0..num_kv_heads {
                let s = got_scales[token_row * num_kv_heads + kv_head];
                scale_min = scale_min.min(s);
                scale_max = scale_max.max(s);
                for d in 0..head_dim {
                    let src = kv_head * max_seq_len * head_dim + token_row * head_dim + d;
                    let dst = hnd_offset(token_row, kv_head, d, head_dim, kv_dim);
                    let expected = cont_host[src].to_f32();
                    let actual = bf16::from_bits(got[dst]).to_f32();
                    let abs_err = (actual - expected).abs();
                    let rel_err = if expected.abs() > 1.0e-6 {
                        abs_err / expected.abs()
                    } else {
                        0.0
                    };
                    max_abs_err = max_abs_err.max(abs_err);
                    sum_abs_err += abs_err as f64;
                    max_rel_err = max_rel_err.max(rel_err);
                    n_elems += 1;
                }
            }
        }
        let mean_abs_err = sum_abs_err / n_elems as f64;

        eprintln!(
            "fp8_scatter_qwen3_production_diagnostic: \
             num_kv_heads={num_kv_heads} head_dim={head_dim} total_tokens={total_tokens} \
             max_abs_err={max_abs_err:.6} mean_abs_err={mean_abs_err:.6} \
             max_rel_err={max_rel_err:.6} scale_range=[{scale_min:.6}, {scale_max:.6}]"
        );

        // Sanity floor: FP8 E4M3 with per-(token, head) scaling on inputs
        // bounded by 6.0 must produce max_abs_err well under 1.0. Beyond
        // that indicates a kernel structural bug, not a precision-limit
        // issue. (The kernel comment notes max_rel_err can spike past 12%
        // for dims whose value is small relative to the per-head outlier
        // — that's expected, so we don't gate on rel_err.)
        assert!(
            max_abs_err < 1.0,
            "FP8 KV scatter roundtrip max_abs_err={max_abs_err:.6} exceeds 1.0 \
             — kernel is producing garbage at production layout"
        );
    }

    /// End-to-end isolation for the 2026-05-26 FP8 KV step-1 divergence.
    /// Wires the production kernel pair (`quantize_paged_kv_fp8` writes
    /// → `decode_attention_fp8` reads) over a deterministic Qwen3-4B-shaped
    /// (num_q_heads=4 GQA 2:1, head_dim=128, 32 KV tokens / 2 pages)
    /// workload, then compares the GPU attention output against a
    /// dequantize-then-host-compute reference. The earlier prefill-logit
    /// parity test (`infer/tests/kv_fp8_prefill_logit_parity.rs`) proved
    /// prefill bit-identical between BF16 and FP8 modes, so any production
    /// FP8 divergence must come from this kernel pair or its dispatch
    /// wiring. A clean result here pins the bug to scheduler-side
    /// kv_indices / kv_meta / scale-pointer plumbing at runtime. A failing
    /// result identifies the actual quant→read break.
    #[test]
    fn fp8_kernel_pair_decode_attention_diagnostic() {
        let ctx = DeviceContext::new().expect("ctx");
        // Production Qwen3-4B layout: num_q_heads=32 num_kv_heads=8 (GQA 4:1).
        // The 4:2 ratio used previously matches GQA semantics but a wider q
        // group could expose head-mapping bugs that smaller ratios miss.
        let num_q_heads = 32usize;
        let num_kv_heads = 8usize;
        let head_dim = 128usize;
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_q_heads * head_dim;
        let kv_seq_len = 32usize;
        const PAGE_SIZE: usize = 16;
        let num_pages = kv_seq_len.div_ceil(PAGE_SIZE);
        let batch_size = 1usize;
        let total_pool_rows = num_pages * PAGE_SIZE;

        // ── 1. Deterministic Q, K, V (BF16) in the value range Qwen3-4B
        //       attention sees post-QK-norm post-RoPE: roughly ±2 with
        //       occasional ±5 outliers on the first dim of each token.
        let mut rng_state: u64 = 0xA5A5_5A5A_DEAD_BEEF;
        let mut next_f32 = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let v = ((rng_state >> 32) as u32) as f32 / u32::MAX as f32;
            (v - 0.5) * 4.0
        };
        let mut q_host = vec![bf16::ZERO; q_dim];
        for d in 0..q_dim {
            q_host[d] = bf16::from_f32(next_f32());
        }
        let mut k_host = vec![bf16::ZERO; kv_seq_len * kv_dim];
        let mut v_host = vec![bf16::ZERO; kv_seq_len * kv_dim];
        for t in 0..kv_seq_len {
            for h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let mut kv = next_f32();
                    let mut vv = next_f32();
                    if d == 0 {
                        kv = if (t + h) % 2 == 0 { 5.0 } else { -4.5 };
                        vv = if (t + h) % 3 == 0 { 5.5 } else { -4.0 };
                    }
                    k_host[t * kv_dim + h * head_dim + d] = bf16::from_f32(kv);
                    v_host[t * kv_dim + h * head_dim + d] = bf16::from_f32(vv);
                }
            }
        }

        // ── 2. Stage K, V into the HND-paged work buffer layout that
        //       `quantize_paged_kv_fp8_kernel` reads (it expects
        //       `[page, head, token, dim]`). Pool rows are dense in [0,
        //       total_pool_rows).
        let mut k_work_host = vec![bf16::ZERO; total_pool_rows * kv_dim];
        let mut v_work_host = vec![bf16::ZERO; total_pool_rows * kv_dim];
        for t in 0..kv_seq_len {
            let page = t / PAGE_SIZE;
            let in_page = t % PAGE_SIZE;
            for h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let src = t * kv_dim + h * head_dim + d;
                    let dst = page * PAGE_SIZE * kv_dim
                        + h * PAGE_SIZE * head_dim
                        + in_page * head_dim
                        + d;
                    k_work_host[dst] = k_host[src];
                    v_work_host[dst] = v_host[src];
                }
            }
        }

        let k_work = DeviceVec::from_host(&ctx, &k_work_host).expect("k_work H2D");
        let v_work = DeviceVec::from_host(&ctx, &v_work_host).expect("v_work H2D");

        let mut k_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(total_pool_rows * kv_dim)
            .expect("k_fp8 alloc");
        let mut v_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(total_pool_rows * kv_dim)
            .expect("v_fp8 alloc");
        let mut k_scales = ctx
            .stream
            .alloc_zeros::<f32>(total_pool_rows * num_kv_heads)
            .expect("k_scales alloc");
        let mut v_scales = ctx
            .stream
            .alloc_zeros::<f32>(total_pool_rows * num_kv_heads)
            .expect("v_scales alloc");

        // Quantize all `kv_seq_len` tokens via the production kernel.
        let token_rows_host: Vec<i32> = (0..kv_seq_len).map(|i| i as i32).collect();
        let token_rows_gpu = ctx.stream.clone_htod(&token_rows_host).expect("rows H2D");
        {
            let (k_fp8_ptr, _g1) = k_fp8.device_ptr_mut(&ctx.stream);
            let (k_scl_ptr, _g2) = k_scales.device_ptr_mut(&ctx.stream);
            let (k_src_ptr, _g3) = k_work.data.device_ptr(&ctx.stream);
            quantize_paged_kv_fp8(
                &ctx,
                k_src_ptr,
                k_fp8_ptr,
                k_scl_ptr,
                &token_rows_gpu,
                num_kv_heads,
                head_dim,
                kv_dim,
                kv_seq_len,
            )
            .expect("k quant");
        }
        {
            let (v_fp8_ptr, _g1) = v_fp8.device_ptr_mut(&ctx.stream);
            let (v_scl_ptr, _g2) = v_scales.device_ptr_mut(&ctx.stream);
            let (v_src_ptr, _g3) = v_work.data.device_ptr(&ctx.stream);
            quantize_paged_kv_fp8(
                &ctx,
                v_src_ptr,
                v_fp8_ptr,
                v_scl_ptr,
                &token_rows_gpu,
                num_kv_heads,
                head_dim,
                kv_dim,
                kv_seq_len,
            )
            .expect("v quant");
        }

        // ── 3. Pack the kv_meta / kv_indices the way the decode-attention
        //       FP8 kernel expects: kv_meta = [kv_indptr (batch+1) ||
        //       last_page_len (batch)]; kv_indices = page list.
        let last_page_tokens = kv_seq_len - (num_pages - 1) * PAGE_SIZE;
        let kv_meta_host: Vec<i32> = vec![0, num_pages as i32, last_page_tokens as i32];
        let kv_meta_gpu = ctx.stream.clone_htod(&kv_meta_host).expect("kv_meta H2D");
        let kv_indices_host: Vec<i32> = (0..num_pages as i32).collect();
        let kv_indices_gpu = ctx.stream.clone_htod(&kv_indices_host).expect("kv_idx H2D");

        // ── 4. Upload Q + allocate output.
        let q = {
            let q_data = ctx.stream.clone_htod(&q_host).expect("q H2D");
            HiddenStates {
                data: q_data,
                hidden_dim: q_dim,
                seq_len: batch_size,
            }
        };
        let mut o = HiddenStates::zeros(&ctx, q_dim, batch_size).expect("o alloc");

        // ── 5. Workspace + dispatch.
        let num_splits = 4usize;
        let workspace_bytes =
            decode_attention_int8_workspace_bytes(batch_size, num_q_heads, head_dim, num_splits);
        let workspace = ctx
            .stream
            .alloc_zeros::<u8>(workspace_bytes.max(1))
            .expect("ws alloc");

        let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
        let (k_fp8_ptr, _gk) = k_fp8.device_ptr(&ctx.stream);
        let (v_fp8_ptr, _gv) = v_fp8.device_ptr(&ctx.stream);
        let (k_scl_ptr, _gks) = k_scales.device_ptr(&ctx.stream);
        let (v_scl_ptr, _gvs) = v_scales.device_ptr(&ctx.stream);
        decode_attention_fp8(
            &ctx,
            &q,
            k_fp8_ptr,
            v_fp8_ptr,
            k_scl_ptr,
            v_scl_ptr,
            &kv_indices_gpu,
            &kv_meta_gpu,
            &mut o,
            batch_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            sm_scale,
            &workspace,
            workspace_bytes,
        )
        .expect("decode_attn_fp8");

        ctx.sync().expect("sync");
        let got_bits = ctx.stream.clone_dtoh(&o.data).expect("o D2H");
        let got: Vec<f32> = got_bits.iter().map(|b| b.to_f32()).collect();

        // ── 6. Host reference: dequantize K, V back to f32 (using the
        //       per-(token, head) scales the production kernel wrote), then
        //       compute attention for the single Q row.
        let k_fp8_host = ctx.stream.clone_dtoh(&k_fp8).expect("k_fp8 D2H");
        let v_fp8_host = ctx.stream.clone_dtoh(&v_fp8).expect("v_fp8 D2H");
        let k_scl_host = ctx.stream.clone_dtoh(&k_scales).expect("k_scl D2H");
        let v_scl_host = ctx.stream.clone_dtoh(&v_scales).expect("v_scl D2H");

        let mut max_abs_err = 0.0f32;
        let mut sum_abs_err = 0.0f64;
        let group_q_per_kv = num_q_heads / num_kv_heads;
        let mut q_f32 = vec![0.0f32; q_dim];
        for i in 0..q_dim {
            q_f32[i] = q_host[i].to_f32();
        }

        // Pre-dequantize K, V for the active rows.
        let mut k_deq = vec![0.0f32; kv_seq_len * kv_dim];
        let mut v_deq = vec![0.0f32; kv_seq_len * kv_dim];
        for t in 0..kv_seq_len {
            for h in 0..num_kv_heads {
                let ks = k_scl_host[t * num_kv_heads + h];
                let vs = v_scl_host[t * num_kv_heads + h];
                for d in 0..head_dim {
                    let off = t * kv_dim + h * head_dim + d;
                    // FP8 E4M3 decode: cast i8 bits via `__nv_fp8_e4m3` is
                    // the production path; here we re-quantize on host
                    // using the same formula `val/scale → fp8 → val*scale`
                    // to match what GPU does.
                    let raw = k_fp8_host[off];
                    let k_dq = fp8_e4m3_to_f32(raw) * ks;
                    let raw_v = v_fp8_host[off];
                    let v_dq = fp8_e4m3_to_f32(raw_v) * vs;
                    k_deq[off] = k_dq;
                    v_deq[off] = v_dq;
                }
            }
        }

        for hq in 0..num_q_heads {
            let hk = hq / group_q_per_kv;
            let mut scores = vec![0.0f32; kv_seq_len];
            let mut m = f32::NEG_INFINITY;
            for t in 0..kv_seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_f32[hq * head_dim + d] * k_deq[t * kv_dim + hk * head_dim + d];
                }
                scores[t] = dot * sm_scale;
                if scores[t] > m {
                    m = scores[t];
                }
            }
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum_exp += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum_exp;
            }
            for d in 0..head_dim {
                let mut o_d = 0.0f32;
                for t in 0..kv_seq_len {
                    o_d += scores[t] * v_deq[t * kv_dim + hk * head_dim + d];
                }
                let actual = got[hq * head_dim + d];
                let err = (actual - o_d).abs();
                max_abs_err = max_abs_err.max(err);
                sum_abs_err += err as f64;
            }
        }
        let mean_abs_err = sum_abs_err / (num_q_heads * head_dim) as f64;

        eprintln!(
            "fp8_kernel_pair_decode_attention_diagnostic: \
             num_q_heads={num_q_heads} num_kv_heads={num_kv_heads} head_dim={head_dim} \
             kv_seq_len={kv_seq_len} max_abs_err={max_abs_err:.6} \
             mean_abs_err={mean_abs_err:.6}"
        );

        // The GPU output is BF16 (truncated from f32). Host reference uses
        // the same dequantized values the GPU sees. Acceptable error
        // envelope: BF16 truncation (~1e-3 per element) + accumulation
        // noise (~5e-3 for kv_seq_len=32). Anything > 0.5 indicates the
        // kernel-pair is producing systematically wrong output.
        assert!(
            max_abs_err < 0.5,
            "FP8 kernel-pair decode attention diverges from host reference \
             (max_abs_err={max_abs_err:.6}). Bug is in the production kernel \
             pair (quantize_paged_kv_fp8 → decode_attention_fp8). Otherwise \
             the audit's step-1 divergence is in scheduler-side dispatch \
             wiring (kv_indices / kv_meta / scale-pointer plumbing) outside \
             this kernel pair."
        );
    }

    /// Host-side FP8 E4M3 decode for the diagnostic above. Bit-exact for
    /// the 256-value table the GPU's `__nv_fp8_e4m3` cast produces.
    fn fp8_e4m3_to_f32(byte: u8) -> f32 {
        // FP8 E4M3 layout: 1 sign | 4 exponent | 3 mantissa, bias=7, no
        // infinities (S.1111.111 is NaN). Values: ±[0, 1, ..., 448].
        let sign = ((byte >> 7) & 0x1) as u32;
        let exp = ((byte >> 3) & 0x0F) as u32;
        let mant = (byte & 0x07) as u32;
        if exp == 0 {
            // Subnormal: 2^-6 * (mant / 8)
            let mag = (mant as f32) * (1.0f32 / 8.0f32) * 2.0f32.powi(-6);
            if sign == 1 { -mag } else { mag }
        } else if exp == 0xF && mant == 0x7 {
            // NaN sentinel.
            f32::NAN
        } else {
            let m = 1.0f32 + (mant as f32) / 8.0f32;
            let e = exp as i32 - 7;
            let mag = m * 2.0f32.powi(e);
            if sign == 1 { -mag } else { mag }
        }
    }

    /// Same diagnostic as above but exercises `quantize_paged_kv_fp8`, the
    /// kernel actually called by `finalize_paged_prefill_kv_layer` and the
    /// per-decode-step write path (NOT `quantize_scatter_kv_fp8`, which only
    /// runs for non-paged-prefill formats like TurboQuant). The source
    /// layout assumption differs: `quantize_paged_kv_fp8_kernel` reads HND-
    /// paged `[page, head, token, dim]` from the work buffer. A wrong
    /// stride or per-(token, head) scale plumbing bug here would be
    /// invisible to the scatter diagnostic above and explain the 2026-05-26
    /// audit's token-1 catastrophic divergence (where FP8 = ~0.4% match
    /// while the scatter kernel proves clean).
    #[test]
    fn fp8_paged_quantize_qwen3_production_layout_diagnostic() {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let num_kv_heads = 8usize;
        let head_dim = 128usize;
        let total_tokens = 64usize;
        let kv_dim = num_kv_heads * head_dim;
        const PAGE_SIZE: usize = 16;
        let num_pages = total_tokens.div_ceil(PAGE_SIZE);
        // HND-paged work buffer: [page, head, token, dim]
        let work_elem_count = num_pages * PAGE_SIZE * kv_dim;
        let hnd_elem_count = work_elem_count;

        let mut rng_state: u64 = 0x9E3779B97F4A7C15;
        let mut next_f32 = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let v = ((rng_state >> 32) as u32) as f32 / u32::MAX as f32;
            (v - 0.5) * 4.0
        };

        // Write known values into the HND-paged work buffer at the layout
        // `quantize_paged_kv_fp8_kernel` reads from. Tokens beyond
        // `total_tokens` stay zero — they're "padding" rows in the last
        // partial page.
        let mut work_host = vec![bf16::ZERO; work_elem_count];
        let mut expected_per_token_head = vec![vec![0.0f32; head_dim]; total_tokens * num_kv_heads];
        for token_row in 0..total_tokens {
            let page_idx = token_row / PAGE_SIZE;
            let offset_in_page = token_row % PAGE_SIZE;
            for kv_head in 0..num_kv_heads {
                for d in 0..head_dim {
                    let mut value = next_f32();
                    if d == 0 {
                        value = if (token_row + kv_head) % 2 == 0 {
                            6.0
                        } else {
                            -5.5
                        };
                    }
                    let src_offset = page_idx * PAGE_SIZE * kv_dim
                        + kv_head * PAGE_SIZE * head_dim
                        + offset_in_page * head_dim
                        + d;
                    work_host[src_offset] = bf16::from_f32(value);
                    expected_per_token_head[token_row * num_kv_heads + kv_head][d] = value;
                }
            }
        }

        let kv_work = DeviceVec::from_host(&ctx, &work_host).expect("paged work H2D");
        let mut kv_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(total_tokens * kv_dim)
            .expect("paged fp8 alloc");
        let mut scales = ctx
            .stream
            .alloc_zeros::<f32>(total_tokens * num_kv_heads)
            .expect("paged scales alloc");
        let token_rows_host: Vec<i32> = (0..total_tokens).map(|idx| idx as i32).collect();
        let token_rows_gpu = ctx
            .stream
            .clone_htod(&token_rows_host)
            .expect("paged rows H2D");

        {
            let (fp8_ptr, _g1) = kv_fp8.device_ptr_mut(&ctx.stream);
            let (scales_ptr, _g2) = scales.device_ptr_mut(&ctx.stream);
            let (work_ptr, _g3) = kv_work.data.device_ptr(&ctx.stream);
            quantize_paged_kv_fp8(
                &ctx,
                work_ptr,
                fp8_ptr,
                scales_ptr,
                &token_rows_gpu,
                num_kv_heads,
                head_dim,
                kv_dim,
                total_tokens,
            )
            .expect("paged fp8 quantize");
        }

        let mut hnd_out = ctx
            .stream
            .alloc_zeros::<u16>(hnd_elem_count)
            .expect("paged hnd alloc");
        {
            let (fp8_ptr, _g1) = kv_fp8.device_ptr(&ctx.stream);
            let (scales_ptr, _g2) = scales.device_ptr(&ctx.stream);
            let (out_ptr, _g3) = hnd_out.device_ptr_mut(&ctx.stream);
            dequantize_paged_kv_fp8_to_hnd(
                &ctx,
                fp8_ptr,
                scales_ptr,
                out_ptr,
                &token_rows_gpu,
                num_kv_heads,
                head_dim,
                kv_dim,
                total_tokens,
            )
            .expect("paged hnd refill");
        }

        ctx.sync().expect("paged sync");
        let got = ctx.stream.clone_dtoh(&hnd_out).expect("paged D2H");
        let got_scales = ctx.stream.clone_dtoh(&scales).expect("paged scales D2H");

        let mut max_abs_err = 0.0f32;
        let mut sum_abs_err = 0.0f64;
        let mut max_rel_err = 0.0f32;
        let mut n_elems = 0usize;
        let mut scale_min = f32::INFINITY;
        let mut scale_max = 0.0f32;
        for token_row in 0..total_tokens {
            for kv_head in 0..num_kv_heads {
                let s = got_scales[token_row * num_kv_heads + kv_head];
                scale_min = scale_min.min(s);
                scale_max = scale_max.max(s);
                for d in 0..head_dim {
                    let dst = hnd_offset(token_row, kv_head, d, head_dim, kv_dim);
                    let expected = expected_per_token_head[token_row * num_kv_heads + kv_head][d];
                    let actual = bf16::from_bits(got[dst]).to_f32();
                    let abs_err = (actual - expected).abs();
                    let rel_err = if expected.abs() > 1.0e-6 {
                        abs_err / expected.abs()
                    } else {
                        0.0
                    };
                    max_abs_err = max_abs_err.max(abs_err);
                    sum_abs_err += abs_err as f64;
                    max_rel_err = max_rel_err.max(rel_err);
                    n_elems += 1;
                }
            }
        }
        let mean_abs_err = sum_abs_err / n_elems as f64;

        eprintln!(
            "fp8_paged_quantize_qwen3_production_diagnostic: \
             num_kv_heads={num_kv_heads} head_dim={head_dim} total_tokens={total_tokens} \
             max_abs_err={max_abs_err:.6} mean_abs_err={mean_abs_err:.6} \
             max_rel_err={max_rel_err:.6} scale_range=[{scale_min:.6}, {scale_max:.6}]"
        );

        assert!(
            max_abs_err < 1.0,
            "FP8 paged quantize roundtrip max_abs_err={max_abs_err:.6} exceeds 1.0 \
             — kernel is producing garbage when called from the prefill-finalize / \
             decode-write path. Scatter path tested cleanly, so the divergence is \
             specific to the HND-paged source layout."
        );
    }
}

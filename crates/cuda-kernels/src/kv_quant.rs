//! KV cache quantization ops: bf16 ↔ INT8/FP8 per-head per-token symmetric.
//!
//! Also includes fused-dequant decode attention for quantized KV formats
//! that FlashInfer doesn't support natively (INT8+scale, INT4+scale, etc.).

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

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
    let (nti_ptr, _g) = new_token_indices_gpu.device_ptr(&ctx.stream);
    unsafe {
        ffi::quantize_paged_kv_fp8_cuda(
            kv_bf16_ptr as *const ffi::Half,
            kv_fp8_ptr as *mut u8,
            scales_ptr as *mut f32,
            nti_ptr as *const i32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            batch_size as i32,
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
    let (rows_ptr, _g) = token_rows_gpu.device_ptr(&ctx.stream);
    unsafe {
        ffi::dequantize_paged_kv_fp8_to_hnd_cuda(
            kv_fp8_ptr as *const u8,
            scales_ptr as *const f32,
            kv_bf16_hnd_ptr as *mut ffi::Half,
            rows_ptr as *const i32,
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

// ─── Varlen-Q + FP8 paged KV attention (mixed batch path) ───
//
// Generalization of `decode_attention_fp8` to mixed prefill+decode batches.
// Reads FP8 KV directly from the pool (no bf16 shadow); enables lifting the
// K2 gate in `infer/src/model/qwen3/forward.rs::supports_mixed_batch` once
// the kernel is wired into `decode_batch_with_prefill`.
//
// HD128 + page_size=16 only for now — same coverage envelope as the qlen=1
// variant. INT8 follow-up uses the same shape with an extra scales pointer.
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_varlen_fp8(
    ctx: &DeviceContext,
    q_packed: &HiddenStates,
    qo_indptr: &CudaSlice<i32>,
    k_pool_ptr: u64,
    v_pool_ptr: u64,
    kv_indptr: &CudaSlice<i32>,
    kv_indices: &CudaSlice<i32>,
    last_page_len: &CudaSlice<i32>,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    batch_size: usize,
    total_q_tokens: usize,
    causal: bool,
    sm_scale: f32,
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

    unsafe {
        ffi::decode_attention_varlen_fp8_cuda(
            q_ptr as *const ffi::Half,
            qoi_ptr as *const i32,
            k_pool_ptr as *const u8,
            v_pool_ptr as *const u8,
            kvi_ptr as *const i32,
            kvx_ptr as *const i32,
            lpl_ptr as *const i32,
            o_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            page_size as i32,
            batch_size as i32,
            total_q_tokens as i32,
            causal,
            sm_scale,
            ctx.stream.cu_stream(),
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

    let (nti_ptr, _gnti) = new_token_indices_gpu.device_ptr(&ctx.stream);

    unsafe {
        ffi::quantize_paged_kv_single_cuda(
            kv_bf16_ptr as *const ffi::Half,
            kv_int8_ptr as *mut i8,
            kv_scales_ptr as *mut f32,
            nti_ptr as *const i32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            batch_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

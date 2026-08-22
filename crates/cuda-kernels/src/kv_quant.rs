//! KV cache quantization ops: bf16 ↔ INT8/FP8 per-head per-token symmetric.
//!
//! Also includes fused-dequant decode attention for quantized KV formats
//! that TileLang BF16 attention doesn't support natively (INT8+scale, FP8+scale).

use anyhow::{Result, bail};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::kv_types::KVFormat;
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

    // SAFETY: all three pointers come from live device buffers pinned by the
    // `_g*` guards, each in HND layout over `max_seq_len`; the kernel touches
    // only rows `[start_pos, start_pos + token_count)`, stream-ordered.
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

    // SAFETY: all three pointers come from live device buffers pinned by the
    // `_g*` guards, each in HND layout over `max_seq_len`; the kernel touches
    // only rows `[0, token_count)`, stream-ordered on `ctx.stream`.
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

// ─── FP8 E4M3 paged pool ops ───

/// Quantize 1 new token per request: bf16 working → FP8 E4M3 or INT8 paged pool.
/// FP8 uses self-contained E4M3 (scale = absmax/448); INT8 uses symmetric
/// per-(token, kv_head) scaling (scale = absmax/127).
#[allow(clippy::too_many_arguments)]
pub fn quantize_paged_kv_per_token(
    ctx: &DeviceContext,
    kv_bf16_ptr: u64,
    kv_ptr: u64,
    scales_ptr: u64,
    new_token_indices_gpu: &CudaSlice<i32>,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    batch_size: usize,
    format: KVFormat,
) -> Result<()> {
    if batch_size == 0 {
        return Ok(());
    }
    let mut offset = 0usize;
    while offset < batch_size {
        let chunk_tokens = (batch_size - offset).min(MAX_TOKEN_ROWS_PER_PAGED_KV_LAUNCH);
        let rows = new_token_indices_gpu.slice(offset..offset + chunk_tokens);
        let (nti_ptr, _g) = rows.device_ptr(&ctx.stream);
        // SAFETY: raw u64 args are the pool's live bf16-work/quant/scale device
        // buffers; the chunked `rows` slice is pinned by `_g`. Writes are
        // limited to the `chunk_tokens` pool rows it names, stream-ordered.
        unsafe {
            match format {
                KVFormat::FP8E4M3 => ffi::quantize_paged_kv_fp8_cuda(
                    kv_bf16_ptr as *const ffi::Half,
                    kv_ptr as *mut u8,
                    scales_ptr as *mut f32,
                    nti_ptr as *const i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    kv_dim as i32,
                    chunk_tokens as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?,
                KVFormat::INT8 => ffi::quantize_paged_kv_single_cuda(
                    kv_bf16_ptr as *const ffi::Half,
                    kv_ptr as *mut i8,
                    scales_ptr as *mut f32,
                    nti_ptr as *const i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    kv_dim as i32,
                    chunk_tokens as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?,
                other => bail!("quantize_paged_kv_per_token: unsupported format {other:?}"),
            }
        }
        offset += chunk_tokens;
    }
    Ok(())
}

// ─── Fused-dequant decode attention (INT8+scale) ───

// ─── Native paged attention for quantized KV ───
//
// FA3-style split-KV over the 1-byte pools directly — no dequant temp. The
// kernel is decode-shaped (one q token per batch row) and consumes the same
// rectangular page table + cu_seqlens_q / seqused_k metadata as the FA3 lane.

pub fn paged_attention_quantized_fa3_workspace_bytes(
    total_q_tokens: usize,
    num_q_heads: usize,
    head_dim: usize,
    num_splits: usize,
) -> usize {
    // SAFETY: pure host-side size computation — no pointers, no device work.
    unsafe {
        ffi::paged_attention_quantized_fa3_workspace_bytes(
            total_q_tokens as i32,
            num_q_heads as i32,
            head_dim as i32,
            num_splits as i32,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_quantized_fa3(
    ctx: &DeviceContext,
    q_packed: &HiddenStates,
    k_pool_ptr: u64,
    v_pool_ptr: u64,
    k_scales_ptr: u64,
    v_scales_ptr: u64,
    page_table: u64,
    cu_seqlens_q: u64,
    seqused_k: u64,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    page_table_stride: usize,
    batch: usize,
    total_q: usize,
    sm_scale: f32,
    kv_format: KVFormat,
    num_splits: usize,
    workspace: &CudaSlice<u8>,
    workspace_bytes: usize,
) -> Result<()> {
    if batch == 0 || total_q == 0 {
        return Ok(());
    }

    let is_fp8 = matches!(kv_format, KVFormat::FP8E4M3);
    let (q_ptr, _gq) = q_packed.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (ws_ptr, _gws) = workspace.device_ptr(&ctx.stream);

    // SAFETY: packed-Q/output/workspace pointers come from live buffers pinned
    // by `_g*`; raw u64 args (page table, indptr, seqused, K/V pool, scales)
    // are the caller's live device pointers. The kernel reads only pages the
    // table names, bounded by `seqused_k`, and writes `total_q` output rows,
    // stream-ordered.
    unsafe {
        ffi::paged_attention_quantized_fa3_cuda(
            q_ptr as *const ffi::Half,
            k_pool_ptr as *const u8,
            v_pool_ptr as *const u8,
            k_scales_ptr as *const f32,
            v_scales_ptr as *const f32,
            page_table as *const i32,
            cu_seqlens_q as *const i32,
            seqused_k as *const i32,
            o_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            page_table_stride as i32,
            batch as i32,
            total_q as i32,
            sm_scale,
            is_fp8,
            num_splits as i32,
            ctx.stream.cu_stream(),
            ws_ptr as *mut u8,
            workspace_bytes,
        )
        .result()?;
    }
    Ok(())
}

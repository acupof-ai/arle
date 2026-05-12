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

    #[test]
    fn hnd_refill_quantized_kv_matches_reference_values() {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        run_hnd_refill_case(&ctx, 8);
        run_hnd_refill_case(&ctx, 7);
    }
}

//! KV cache quantization ops: bf16 ↔ INT8/FP8 per-head per-token symmetric.
//!
//! Also includes fused-dequant decode attention for quantized KV formats
//! that TileLang BF16 attention doesn't support natively (INT8+scale, INT4+scale, etc.).

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

/// Quantize 1 new token per request: bf16 working → FP8 E4M3 paged pool.
/// Thin wrapper around `quantize_paged_kv_per_token` with `KVFormat::FP8E4M3`.
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
    quantize_paged_kv_per_token(
        ctx,
        kv_bf16_ptr,
        kv_fp8_ptr,
        scales_ptr,
        new_token_indices_gpu,
        num_kv_heads,
        head_dim,
        kv_dim,
        batch_size,
        KVFormat::FP8E4M3,
    )
}

/// **KIVI per-channel K quantize.** Reads a pre-computed
/// `[num_kv_heads, head_dim]` f32 scale table (populated via
/// [`compute_k_per_channel_absmax`] + [`finalize_k_per_channel_scales`])
/// and quantizes bf16 K → FP8 E4M3 without per-(token, head) absmax
/// reduction.
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
        // SAFETY: raw u64 args are the pool's live bf16-work/FP8 buffers plus
        // the `[num_kv_heads, head_dim]` static scale table; chunked `rows`
        // pinned by `_g`. Writes bounded by `chunk_tokens`, stream-ordered.
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
        // SAFETY: raw u64 args are the pool's live bf16-work buffer and the
        // `[num_kv_heads, head_dim]` absmax accumulator (safe to accumulate
        // across chunks); chunked `rows` pinned by `_g`. Reads bounded by
        // `chunk_tokens`, stream-ordered.
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
    // SAFETY: `k_static_scales_ptr` is the caller's live `[num_channels]` f32
    // scale table on `ctx.stream`; in-place divide bounded by `num_channels`,
    // stream-ordered.
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

/// INT8 KIVI calibration step 2: divide accumulated absmax by 127 (INT8
/// symmetric max) to produce the final per-channel scale. Same idempotent
/// guarantee as [`finalize_k_per_channel_scales`].
pub fn finalize_k_per_channel_scales_int8(
    ctx: &DeviceContext,
    k_static_scales_ptr: u64,
    num_channels: usize,
) -> Result<()> {
    if num_channels == 0 {
        return Ok(());
    }
    // SAFETY: same contract as `finalize_k_per_channel_scales` — live
    // `[num_channels]` f32 table, in-place divide, stream-ordered.
    unsafe {
        ffi::finalize_k_per_channel_scales_int8_cuda(
            k_static_scales_ptr as *mut f32,
            num_channels as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// INT8 KIVI per-channel K quantize for the per-decode-step append path.
/// Mirrors [`quantize_paged_kv_fp8_per_channel`] but writes INT8 (rounded
/// then clamped to ±127) using a pre-computed `[num_kv_heads, head_dim]`
/// scale table.
#[allow(clippy::too_many_arguments)]
pub fn quantize_paged_kv_int8_per_channel(
    ctx: &DeviceContext,
    kv_bf16_ptr: u64,
    kv_int8_ptr: u64,
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
        // SAFETY: raw u64 args are the pool's live bf16-work/INT8 buffers plus
        // the static per-channel scale table; chunked `rows` pinned by `_g`.
        // Writes bounded by `chunk_tokens`, stream-ordered.
        unsafe {
            ffi::quantize_paged_kv_int8_per_channel_cuda(
                kv_bf16_ptr as *const ffi::Half,
                kv_int8_ptr as *mut i8,
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
        // SAFETY: raw u64 args are the pool's live FP8/scale planes and the
        // bf16 HND work buffer; chunked `rows` pinned by `_g`. Reads/writes
        // bounded by the `chunk_tokens` rows it names, stream-ordered.
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
        // SAFETY: same contract as `dequantize_paged_kv_fp8_to_hnd`, reading
        // the INT8 plane instead; chunked `rows` pinned by `_g`, writes bounded
        // by `chunk_tokens`, stream-ordered.
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

/// Per-channel-K variant of [`dequantize_paged_kv_fp8_to_hnd`] for the KIVI
/// prefix refill: K reads the static `[num_kv_heads, head_dim]` channel scale
/// table instead of per-(token, head) scales. The per-token sibling must NOT
/// be used for per-channel K — per-channel K quantize never writes per-token
/// K scales, so it would refill the prefix as zeros.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_paged_kv_fp8_per_channel_k_to_hnd(
    ctx: &DeviceContext,
    kv_fp8_ptr: u64,
    k_static_scales_ptr: u64,
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
        // SAFETY: same contract as `dequantize_paged_kv_fp8_to_hnd`, but the
        // scale arg is the static `[num_kv_heads, head_dim]` channel table;
        // chunked `rows` pinned by `_g`, writes bounded by `chunk_tokens`.
        unsafe {
            ffi::dequantize_paged_kv_fp8_per_channel_k_to_hnd_cuda(
                kv_fp8_ptr as *const u8,
                k_static_scales_ptr as *const f32,
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

/// Per-channel-K variant of [`dequantize_paged_kv_int8_to_hnd`] for the KIVI
/// prefix refill (see [`dequantize_paged_kv_fp8_per_channel_k_to_hnd`] for why
/// the per-token sibling is wrong for per-channel K).
#[allow(clippy::too_many_arguments)]
pub fn dequantize_paged_kv_int8_per_channel_k_to_hnd(
    ctx: &DeviceContext,
    kv_int8_ptr: u64,
    k_static_scales_ptr: u64,
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
        // SAFETY: same contract as the FP8 per-channel-K sibling above, reading
        // the INT8 plane; chunked `rows` pinned by `_g`, writes bounded by
        // `chunk_tokens`, stream-ordered.
        unsafe {
            ffi::dequantize_paged_kv_int8_per_channel_k_to_hnd_cuda(
                kv_int8_ptr as *const i8,
                k_static_scales_ptr as *const f32,
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
    // SAFETY: pure host-side size computation — no pointers, no device work.
    unsafe {
        ffi::decode_attention_int8_workspace_bytes(
            batch_size as i32,
            num_qo_heads as i32,
            head_dim as i32,
            num_splits as i32,
        )
    }
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
    // SAFETY: Q/indices/meta/output/workspace pointers come from live buffers
    // pinned by `_g*`; raw u64 args are the pool's FP8 K/V planes, the static
    // K channel-scale table and per-(row, head) V scales. Reads only pages
    // named by `kv_indices`/`kv_meta`, writes `batch_size * num_qo_heads`
    // output rows; workspace caller-sized via the `_workspace_bytes` query.
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

/// INT8 KIVI decode attention: mirrors [`decode_attention_fp8_per_channel_k`]
/// but reads INT8 K/V from the pool with cp.async pipelining (same as the
/// per-(row, head) INT8 sibling).
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_int8_per_channel_k(
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
    // SAFETY: same contract as `decode_attention_fp8_per_channel_k`, reading
    // INT8 K/V planes instead; guards `_g*` pin the live slices, workspace
    // caller-sized, stream-ordered.
    unsafe {
        ffi::decode_attention_int8_per_channel_k_cuda(
            q_ptr as *const ffi::Half,
            k_data_ptr as *const i8,
            v_data_ptr as *const i8,
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

    fn run_per_channel_k_fp8_refill_case(ctx: &DeviceContext, head_dim: usize) {
        let num_kv_heads = 3usize;
        let total_tokens = 17usize;
        let kv_dim = num_kv_heads * head_dim;
        let elem_count = total_tokens * kv_dim;
        let hnd_elem_count = total_tokens.div_ceil(16) * 16 * kv_dim;

        let fp8_pattern = [0x00u8, 0x38, 0xb8, 0x40];
        let fp8_host: Vec<u8> = (0..elem_count)
            .map(|idx| fp8_pattern[idx % fp8_pattern.len()])
            .collect();
        // Per-channel scale table: [num_kv_heads * head_dim].
        let scale_host: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|idx| 0.25 + (idx % 5) as f32 * 0.125)
            .collect();
        let token_rows_host: Vec<i32> = (0..total_tokens).map(|idx| idx as i32).collect();

        let fp8_gpu = ctx.stream.clone_htod(&fp8_host).expect("fp8 H2D");
        let scales_gpu = ctx.stream.clone_htod(&scale_host).expect("scales H2D");
        let token_rows_gpu = ctx.stream.clone_htod(&token_rows_host).expect("rows H2D");
        let mut fp8_out = ctx
            .stream
            .alloc_zeros::<u16>(hnd_elem_count)
            .expect("fp8 out");

        {
            let (fp8_ptr, _fp8_guard) = fp8_gpu.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales_gpu.device_ptr(&ctx.stream);
            let (out_ptr, _out_guard) = fp8_out.device_ptr_mut(&ctx.stream);
            dequantize_paged_kv_fp8_per_channel_k_to_hnd(
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
            .expect("fp8 per-channel-k hnd refill");
        }

        ctx.sync().expect("sync per-channel-k fp8 refill");
        let got = ctx.stream.clone_dtoh(&fp8_out).expect("fp8 D2H");

        for token_row in 0..total_tokens {
            for kv_head in 0..num_kv_heads {
                for d in 0..head_dim {
                    let scale = scale_host[kv_head * head_dim + d];
                    let src = token_row * kv_dim + kv_head * head_dim + d;
                    let dst = hnd_offset(token_row, kv_head, d, head_dim, kv_dim);
                    let expected = bf16::from_f32(fp8_reference(fp8_host[src]) * scale).to_bits();
                    assert_eq!(
                        got[dst], expected,
                        "fp8 per-channel-k mismatch head_dim={head_dim} token={token_row} head={kv_head} d={d}"
                    );
                }
            }
        }
    }

    fn run_per_channel_k_int8_refill_case(ctx: &DeviceContext, head_dim: usize) {
        let num_kv_heads = 3usize;
        let total_tokens = 17usize;
        let kv_dim = num_kv_heads * head_dim;
        let elem_count = total_tokens * kv_dim;
        let hnd_elem_count = total_tokens.div_ceil(16) * 16 * kv_dim;

        let int8_host: Vec<i8> = (0..elem_count).map(|idx| (idx as i8 % 9) - 4).collect();
        // Per-channel scale table: [num_kv_heads * head_dim].
        let scale_host: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|idx| 0.25 + (idx % 5) as f32 * 0.125)
            .collect();
        let token_rows_host: Vec<i32> = (0..total_tokens).map(|idx| idx as i32).collect();

        let int8_gpu = ctx.stream.clone_htod(&int8_host).expect("int8 H2D");
        let scales_gpu = ctx.stream.clone_htod(&scale_host).expect("scales H2D");
        let token_rows_gpu = ctx.stream.clone_htod(&token_rows_host).expect("rows H2D");
        let mut int8_out = ctx
            .stream
            .alloc_zeros::<u16>(hnd_elem_count)
            .expect("int8 out");

        {
            let (int8_ptr, _int8_guard) = int8_gpu.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales_gpu.device_ptr(&ctx.stream);
            let (out_ptr, _out_guard) = int8_out.device_ptr_mut(&ctx.stream);
            dequantize_paged_kv_int8_per_channel_k_to_hnd(
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
            .expect("int8 per-channel-k hnd refill");
        }

        ctx.sync().expect("sync per-channel-k int8 refill");
        let got = ctx.stream.clone_dtoh(&int8_out).expect("int8 D2H");

        for token_row in 0..total_tokens {
            for kv_head in 0..num_kv_heads {
                for d in 0..head_dim {
                    let scale = scale_host[kv_head * head_dim + d];
                    let src = token_row * kv_dim + kv_head * head_dim + d;
                    let dst = hnd_offset(token_row, kv_head, d, head_dim, kv_dim);
                    let expected = bf16::from_f32(int8_host[src] as f32 * scale).to_bits();
                    assert_eq!(
                        got[dst], expected,
                        "int8 per-channel-k mismatch head_dim={head_dim} token={token_row} head={kv_head} d={d}"
                    );
                }
            }
        }
    }

    #[test]
    fn hnd_refill_fp8_per_channel_k_matches_scale_table() {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        run_per_channel_k_fp8_refill_case(&ctx, 8);
        run_per_channel_k_fp8_refill_case(&ctx, 7);
    }

    #[test]
    fn hnd_refill_int8_per_channel_k_matches_scale_table() {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        run_per_channel_k_int8_refill_case(&ctx, 8);
        run_per_channel_k_int8_refill_case(&ctx, 7);
    }

    /// Same diagnostic as above but exercises `quantize_paged_kv_fp8`, the
    /// kernel actually called by `finalize_paged_prefill_kv_layer` and the
    /// per-decode-step write path. The source
    /// layout assumption differs: `quantize_paged_kv_fp8_kernel` reads HND-
    /// paged `[page, head, token, dim]` from the work buffer. A wrong
    /// stride or per-(token, head) scale plumbing bug here would be
    /// invisible to the scatter diagnostic above and explain the 2026-05-26
    /// audit's token-1 catastrophic divergence (where FP8 = ~0.4% match
    /// while the scatter kernel proves clean).
    #[test]
    #[allow(clippy::needless_range_loop)]
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

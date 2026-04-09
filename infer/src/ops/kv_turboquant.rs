//! TurboQuant KV cache quantization ops.
//!
//! Rotation-based near-optimal quantization: random orthogonal rotation +
//! Lloyd-Max scalar quantization on the known Beta distribution.
//!
//! Compresses KV from bf16 (16-bit) to 2-4 bits per coordinate + f16 norm,
//! achieving 5x+ memory reduction at quality-neutral levels.

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::model::turboquant_state::TurboQuantLayerState;
use crate::tensor::{DeviceContext, DeviceVec};

/// Quantize bf16 KV → TurboQuant packed indices + f16 norms (contiguous batch).
///
/// - `kv_bf16`: source bf16 data, `[batch_size, kv_dim]` where `kv_dim = num_kv_heads * head_dim`
/// - `packed_out`: destination packed buffer, `[batch_size, num_kv_heads * packed_per_head]`
/// - `norms_out`: destination norms (f16), `[batch_size, num_kv_heads]`
/// - `state`: pre-computed rotation + codebook for this layer
#[allow(clippy::too_many_arguments)]
pub(crate) fn turboquant_quantize(
    ctx: &DeviceContext,
    kv_bf16: &DeviceVec,
    packed_out: &mut CudaSlice<u8>,
    norms_out: &mut CudaSlice<u16>,
    state: &TurboQuantLayerState,
    layer_idx: usize,
    num_kv_heads: usize,
    head_dim: usize,
    batch_size: usize,
) -> Result<()> {
    if batch_size == 0 {
        return Ok(());
    }

    let kv_dim = num_kv_heads * head_dim;
    let rotation = &state.rotations[layer_idx];
    let codebook = &state.codebook;

    let (bf16_ptr, _g1) = kv_bf16.data.device_ptr(&ctx.stream);
    let (packed_ptr, _g2) = packed_out.device_ptr_mut(&ctx.stream);
    let (norms_ptr, _g3) = norms_out.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _g4) = rotation.matrix.device_ptr(&ctx.stream);
    let (bound_ptr, _g5) = codebook.boundaries.device_ptr(&ctx.stream);

    unsafe {
        ffi::turboquant_quantize_kv_cuda(
            bf16_ptr as *const ffi::Half,
            packed_ptr as *mut u8,
            norms_ptr as *mut ffi::Half,
            pi_ptr as *const f32,
            bound_ptr as *const f32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            state.packed_per_head as i32,
            codebook.num_levels as i32,
            codebook.bits as i32,
            batch_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Dequantize TurboQuant packed → bf16 KV (contiguous tokens).
///
/// - `packed_in`: `[token_count, num_kv_heads * packed_per_head]`
/// - `norms_in`: `[token_count, num_kv_heads]` f16
/// - `kv_bf16`: output, `[token_count, kv_dim]`
#[allow(clippy::too_many_arguments)]
pub(crate) fn turboquant_dequantize(
    ctx: &DeviceContext,
    packed_in: &CudaSlice<u8>,
    norms_in: &CudaSlice<u16>,
    kv_bf16: &mut DeviceVec,
    state: &TurboQuantLayerState,
    layer_idx: usize,
    num_kv_heads: usize,
    head_dim: usize,
    token_count: usize,
) -> Result<()> {
    if token_count == 0 {
        return Ok(());
    }

    let kv_dim = num_kv_heads * head_dim;
    let rotation = &state.rotations[layer_idx];
    let codebook = &state.codebook;

    let (packed_ptr, _g1) = packed_in.device_ptr(&ctx.stream);
    let (norms_ptr, _g2) = norms_in.device_ptr(&ctx.stream);
    let (bf16_ptr, _g3) = kv_bf16.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _g4) = rotation.matrix.device_ptr(&ctx.stream);
    let (cent_ptr, _g5) = codebook.centroids.device_ptr(&ctx.stream);

    unsafe {
        ffi::turboquant_dequantize_kv_cuda(
            packed_ptr as *const u8,
            norms_ptr as *const ffi::Half,
            bf16_ptr as *mut ffi::Half,
            pi_ptr as *const f32,
            cent_ptr as *const f32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            state.packed_per_head as i32,
            codebook.num_levels as i32,
            codebook.bits as i32,
            token_count as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Quantize 1 new token per request from bf16 working buffer → TQ paged pool.
///
/// - `kv_bf16_ptr`: working buffer pointer (u64), `[batch_size, kv_dim]`
/// - `pool_data`: paged pool, `[max_tokens, num_kv_heads * packed_per_head]`
/// - `pool_norms`: `[max_tokens, num_kv_heads]` f16
/// - `pool_indices`: `[batch_size]` — physical pool index for each request's new token
#[allow(clippy::too_many_arguments)]
pub(crate) fn turboquant_quantize_paged_single(
    ctx: &DeviceContext,
    kv_bf16_ptr: u64,
    pool_data: &CudaSlice<u8>,
    pool_norms: &CudaSlice<u16>,
    pool_indices: &CudaSlice<i32>,
    state: &TurboQuantLayerState,
    layer_idx: usize,
    num_kv_heads: usize,
    head_dim: usize,
    batch_size: usize,
) -> Result<()> {
    if batch_size == 0 {
        return Ok(());
    }

    let kv_dim = num_kv_heads * head_dim;
    let rotation = &state.rotations[layer_idx];
    let codebook = &state.codebook;

    let (data_ptr, _g2) = pool_data.device_ptr(&ctx.stream);
    let (norms_ptr, _g3) = pool_norms.device_ptr(&ctx.stream);
    let (idx_ptr, _g4) = pool_indices.device_ptr(&ctx.stream);
    let (pi_ptr, _g5) = rotation.matrix.device_ptr(&ctx.stream);
    let (bound_ptr, _g6) = codebook.boundaries.device_ptr(&ctx.stream);

    unsafe {
        ffi::turboquant_quantize_single_cuda(
            kv_bf16_ptr as *const ffi::Half,
            data_ptr as *mut u8,
            norms_ptr as *mut ffi::Half,
            idx_ptr as *const i32,
            pi_ptr as *const f32,
            bound_ptr as *const f32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            state.packed_per_head as i32,
            codebook.num_levels as i32,
            codebook.bits as i32,
            batch_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Dequantize TQ paged pool → bf16 via scatter-read (token_indices maps logical → physical).
///
/// - `pool_data`: `[max_tokens, num_kv_heads * packed_per_head]`
/// - `pool_norms`: `[max_tokens, num_kv_heads]` f16
/// - `kv_bf16`: output, `[total_tokens, kv_dim]`
/// - `token_indices`: `[total_tokens]` — physical pool index per logical token
#[allow(clippy::too_many_arguments)]
pub(crate) fn turboquant_dequantize_paged(
    ctx: &DeviceContext,
    pool_data: &CudaSlice<u8>,
    pool_norms: &CudaSlice<u16>,
    kv_bf16: &mut DeviceVec,
    token_indices: &CudaSlice<i32>,
    state: &TurboQuantLayerState,
    layer_idx: usize,
    num_kv_heads: usize,
    head_dim: usize,
    total_tokens: usize,
) -> Result<()> {
    if total_tokens == 0 {
        return Ok(());
    }

    let kv_dim = num_kv_heads * head_dim;
    let rotation = &state.rotations[layer_idx];
    let codebook = &state.codebook;

    let (data_ptr, _g1) = pool_data.device_ptr(&ctx.stream);
    let (norms_ptr, _g2) = pool_norms.device_ptr(&ctx.stream);
    let (bf16_ptr, _g3) = kv_bf16.data.device_ptr_mut(&ctx.stream);
    let (idx_ptr, _g4) = token_indices.device_ptr(&ctx.stream);
    let (pi_ptr, _g5) = rotation.matrix.device_ptr(&ctx.stream);
    let (cent_ptr, _g6) = codebook.centroids.device_ptr(&ctx.stream);

    unsafe {
        ffi::turboquant_dequantize_paged_cuda(
            data_ptr as *const u8,
            norms_ptr as *const ffi::Half,
            bf16_ptr as *mut ffi::Half,
            idx_ptr as *const i32,
            pi_ptr as *const f32,
            cent_ptr as *const f32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            state.packed_per_head as i32,
            codebook.num_levels as i32,
            codebook.bits as i32,
            total_tokens as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Dequantize TQ pool → bf16 working buffer, preserving NHD paged layout.
///
/// For each pool index in `pool_indices`, reads packed data from `pool_data`,
/// dequantizes, and writes bf16 to `work_bf16` at the **same** physical pool
/// index position. This allows FlashInfer to read from the working buffer
/// using its existing page table (Phase 1: separate dequant + FlashInfer).
#[allow(clippy::too_many_arguments)]
pub(crate) fn turboquant_dequantize_inplace(
    ctx: &DeviceContext,
    pool_data: &CudaSlice<u8>,
    pool_norms: &CudaSlice<u16>,
    work_bf16_ptr: u64,
    pool_indices: &CudaSlice<i32>,
    state: &TurboQuantLayerState,
    layer_idx: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_indices: usize,
) -> Result<()> {
    if num_indices == 0 {
        return Ok(());
    }

    let kv_dim = num_kv_heads * head_dim;
    let rotation = &state.rotations[layer_idx];
    let codebook = &state.codebook;

    let (data_ptr, _g1) = pool_data.device_ptr(&ctx.stream);
    let (norms_ptr, _g2) = pool_norms.device_ptr(&ctx.stream);
    let (idx_ptr, _g3) = pool_indices.device_ptr(&ctx.stream);
    let (pi_ptr, _g4) = rotation.matrix.device_ptr(&ctx.stream);
    let (cent_ptr, _g5) = codebook.centroids.device_ptr(&ctx.stream);

    unsafe {
        ffi::turboquant_dequantize_inplace_cuda(
            data_ptr as *const u8,
            norms_ptr as *const ffi::Half,
            work_bf16_ptr as *mut ffi::Half,
            idx_ptr as *const i32,
            pi_ptr as *const f32,
            cent_ptr as *const f32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_dim as i32,
            state.packed_per_head as i32,
            codebook.num_levels as i32,
            codebook.bits as i32,
            num_indices as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

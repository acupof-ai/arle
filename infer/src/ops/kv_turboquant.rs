//! TurboQuant KV cache quantization ops.
//!
//! Rotation-based near-optimal quantization: random rotation + Lloyd-Max scalar
//! quantization on the known Beta distribution.
//!
//! Two rotation backends:
//! - **Full** (O(D²)): random orthogonal matmul — reference quality
//! - **Hadamard** (O(D log D)): sign flip + FWHT — 18x fewer FMAs, production default
//!
//! Compresses KV from bf16 (16-bit) to 2-4 bits per coordinate + f16 norm,
//! achieving 5x+ memory reduction at quality-neutral levels.

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr};

use crate::backend::cuda::ffi;
use crate::backend::cuda::tensor::DeviceContext;
use crate::model::turboquant_state::{RotationMode, TurboQuantLayerState};

/// Quantize 1 new token per request from bf16 working buffer → TQ paged pool.
///
/// Dispatches to Full or Hadamard kernel based on `state.mode`.
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
    let (bound_ptr, _g6) = codebook.boundaries.device_ptr(&ctx.stream);

    unsafe {
        match state.mode {
            RotationMode::Full => {
                let (pi_ptr, _g5) = rotation.full_matrix_ptr().device_ptr(&ctx.stream);
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
            RotationMode::Hadamard => {
                let (signs_ptr, _g5) = rotation.hadamard_signs_ptr().device_ptr(&ctx.stream);
                ffi::turboquant_fast_quantize_single_cuda(
                    kv_bf16_ptr as *const ffi::Half,
                    data_ptr as *mut u8,
                    norms_ptr as *mut ffi::Half,
                    idx_ptr as *const i32,
                    signs_ptr as *const i8,
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
        }
    }

    Ok(())
}

/// Dequantize TQ pool → bf16 working buffer, preserving NHD paged layout.
///
/// For each pool index in `pool_indices`, reads packed data from `pool_data`,
/// dequantizes, and writes bf16 to `work_bf16` at the **same** physical pool
/// index position. This allows FlashInfer to read from the working buffer
/// using its existing page table.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
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
    let (cent_ptr, _g5) = codebook.centroids.device_ptr(&ctx.stream);

    unsafe {
        match state.mode {
            RotationMode::Full => {
                let (pi_ptr, _g4) = rotation.full_matrix_ptr().device_ptr(&ctx.stream);
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
            RotationMode::Hadamard => {
                let (signs_ptr, _g4) = rotation.hadamard_signs_ptr().device_ptr(&ctx.stream);
                ffi::turboquant_fast_dequantize_inplace_cuda(
                    data_ptr as *const u8,
                    norms_ptr as *const ffi::Half,
                    work_bf16_ptr as *mut ffi::Half,
                    idx_ptr as *const i32,
                    signs_ptr as *const i8,
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
        }
    }

    Ok(())
}

/// Rotate query vectors: sign flip + FWHT.
/// Prepares Q for fused TQ decode attention (avoids K dequant).
///
/// - `q`: query vectors `[num_heads_total, head_dim]` bf16
/// - `q_rot`: output rotated queries (same shape)
/// - `signs`: Hadamard signs `[head_dim]` i8 from K rotation state
#[allow(clippy::too_many_arguments)]
pub(crate) fn turboquant_rotate_query(
    ctx: &DeviceContext,
    q_ptr: u64,
    q_rot_ptr: u64,
    state: &TurboQuantLayerState,
    layer_idx: usize,
    num_heads_total: usize,
    head_dim: usize,
) -> Result<()> {
    if num_heads_total == 0 {
        return Ok(());
    }
    let rotation = &state.rotations[layer_idx];
    let (signs_ptr, _g) = rotation.hadamard_signs_ptr().device_ptr(&ctx.stream);

    unsafe {
        ffi::tq_rotate_query_cuda(
            q_ptr as *const ffi::Half,
            q_rot_ptr as *mut ffi::Half,
            signs_ptr as *const i8,
            num_heads_total as i32,
            head_dim as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Fused TQ decode attention: score from packed K centroids, dequant V in-kernel.
///
/// Q must be pre-rotated via `turboquant_rotate_query`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn turboquant_fused_decode_attention(
    ctx: &DeviceContext,
    q_rot_ptr: u64,
    k_packed: &CudaSlice<u8>,
    k_norms: &CudaSlice<u16>,
    v_packed: &CudaSlice<u8>,
    v_norms: &CudaSlice<u16>,
    kv_indices: &CudaSlice<i32>,
    kv_indptr: &CudaSlice<i32>,
    output_ptr: u64,
    k_state: &TurboQuantLayerState,
    v_state: &TurboQuantLayerState,
    batch_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
) -> Result<()> {
    let (kp_ptr, _g1) = k_packed.device_ptr(&ctx.stream);
    let (kn_ptr, _g2) = k_norms.device_ptr(&ctx.stream);
    let (vp_ptr, _g3) = v_packed.device_ptr(&ctx.stream);
    let (vn_ptr, _g4) = v_norms.device_ptr(&ctx.stream);
    let (ki_ptr, _g5) = kv_indices.device_ptr(&ctx.stream);
    let (kip_ptr, _g6) = kv_indptr.device_ptr(&ctx.stream);
    let (ck_ptr, _g7) = k_state.codebook.centroids.device_ptr(&ctx.stream);
    let (cv_ptr, _g8) = v_state.codebook.centroids.device_ptr(&ctx.stream);

    unsafe {
        ffi::tq_decode_attention_cuda(
            q_rot_ptr as *const ffi::Half,
            kp_ptr as *const u8,
            kn_ptr as *const ffi::Half,
            vp_ptr as *const u8,
            vn_ptr as *const ffi::Half,
            ki_ptr as *const i32,
            kip_ptr as *const i32,
            output_ptr as *mut ffi::Half,
            ck_ptr as *const f32,
            cv_ptr as *const f32,
            batch_size as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            k_state.packed_per_head as i32,
            k_state.codebook.num_levels as i32,
            k_state.codebook.bits as i32,
            sm_scale,
            head_dim as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

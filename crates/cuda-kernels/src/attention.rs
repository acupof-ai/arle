//! DSv4-Flash attention-side packing/dispatch wrappers.
//!
//! Today this module holds a single op — `dsv4_fp8_kv_pack` — the MODEL1
//! FP8 KV pack kernel that feeds FlashMLA's sparse-FP8 decode path
//! (`sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`).
//! Phase D-3' of
//! [`docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`].
//!
//! Runtime wire-up (D-4) is a separate dispatch; this wrapper exposes the
//! FFI through the established `DeviceContext`-driven idiom that the rest
//! of the kernel crate uses.

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec};

/// Pack `n_tokens` worth of (NoPE bf16, RoPE bf16) tensors into the MODEL1
/// FP8 block-paged layout that FlashMLA's sparse-FP8 decode consumes
/// (584 bytes/token; see `csrc/attention/dsv4_fp8_kv_pack.cu` for the
/// byte layout + e8m0 scale encoding).
///
/// - `nope`: bf16 `[n_tokens, 448]` (NoPE dims, host-allocated DeviceVec).
/// - `rope`: bf16 `[n_tokens, 64]`  (RoPE dims, host-allocated DeviceVec).
/// - `packed_kv`: u64 device pointer into the FP8 KV pool. Caller sizes
///   the pool to `num_blocks * page_block_size * 584` bytes.
/// - `token_block_id`: i32 `[n_tokens]` — destination block index per token.
/// - `token_in_block_row`: i32 `[n_tokens]` — 0..page_block_size-1 per token.
/// - `page_block_size`: upstream's `page_block_size` (64 for DSv4-Flash MODEL1).
///
/// No-op when `n_tokens == 0`.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_fp8_kv_pack(
    ctx: &DeviceContext,
    nope: &DeviceVec,
    rope: &DeviceVec,
    packed_kv_ptr: u64,
    token_block_id: &CudaSlice<i32>,
    token_in_block_row: &CudaSlice<i32>,
    n_tokens: usize,
    page_block_size: usize,
) -> Result<()> {
    if n_tokens == 0 {
        return Ok(());
    }

    let (nope_ptr, _gn) = nope.data.device_ptr(&ctx.stream);
    let (rope_ptr, _gr) = rope.data.device_ptr(&ctx.stream);
    let (tbid_ptr, _gt) = token_block_id.device_ptr(&ctx.stream);
    let (tibr_ptr, _gi) = token_in_block_row.device_ptr(&ctx.stream);

    unsafe {
        ffi::arle_dsv4_fp8_kv_pack_cuda(
            nope_ptr as *const ffi::Half,
            rope_ptr as *const ffi::Half,
            packed_kv_ptr as *mut u8,
            tbid_ptr as *const i32,
            tibr_ptr as *const i32,
            n_tokens as i32,
            page_block_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Pack variant that accepts raw u64 pointers for NoPE/RoPE (used when the
/// caller has already lifted the bf16 buffers out of `DeviceVec` for
/// fused-launch scheduling). The packed_kv pointer is u64 too.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_fp8_kv_pack_raw(
    ctx: &DeviceContext,
    nope_ptr: u64,
    rope_ptr: u64,
    packed_kv_ptr: u64,
    token_block_id: &CudaSlice<i32>,
    token_in_block_row: &CudaSlice<i32>,
    n_tokens: usize,
    page_block_size: usize,
) -> Result<()> {
    if n_tokens == 0 {
        return Ok(());
    }

    let (tbid_ptr, _gt) = token_block_id.device_ptr(&ctx.stream);
    let (tibr_ptr, _gi) = token_in_block_row.device_ptr(&ctx.stream);

    unsafe {
        ffi::arle_dsv4_fp8_kv_pack_cuda(
            nope_ptr as *const ffi::Half,
            rope_ptr as *const ffi::Half,
            packed_kv_ptr as *mut u8,
            tbid_ptr as *const i32,
            tibr_ptr as *const i32,
            n_tokens as i32,
            page_block_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Strided-input pack variant — accepts raw u64 NoPE/RoPE pointers plus an
/// explicit element stride per axis. Phase D-4 runtime hooks feed this
/// from `k_prepared` (interleaved [NoPE 448 | RoPE 64] bf16, head_dim=512
/// stride) without an intermediate deinterleave; see
/// `docs/experience/wins/2026-05-28-dsv4-flashmla-decode-d4-plumbing.md`
/// Finding 1.
///
/// `stride_nope_elems` / `stride_rope_elems` are per-token bf16 element
/// strides. Must be ≥ 448 / 64 respectively. For `k_prepared`-shaped
/// input pass `(nope_ptr = k_prepared, rope_ptr = k_prepared + 448*2 B,
/// stride_*_elems = 512)`.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_fp8_kv_pack_strided_raw(
    ctx: &DeviceContext,
    nope_ptr: u64,
    rope_ptr: u64,
    packed_kv_ptr: u64,
    token_block_id: &CudaSlice<i32>,
    token_in_block_row: &CudaSlice<i32>,
    n_tokens: usize,
    page_block_size: usize,
    stride_nope_elems: usize,
    stride_rope_elems: usize,
) -> Result<()> {
    if n_tokens == 0 {
        return Ok(());
    }

    let (tbid_ptr, _gt) = token_block_id.device_ptr(&ctx.stream);
    let (tibr_ptr, _gi) = token_in_block_row.device_ptr(&ctx.stream);

    unsafe {
        ffi::arle_dsv4_fp8_kv_pack_strided_cuda(
            nope_ptr as *const ffi::Half,
            rope_ptr as *const ffi::Half,
            packed_kv_ptr as *mut u8,
            tbid_ptr as *const i32,
            tibr_ptr as *const i32,
            n_tokens as i32,
            page_block_size as i32,
            stride_nope_elems as i32,
            stride_rope_elems as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// Phase D-4 step 1 — DSv4 FlashMLA sparse-decode indices builder.
///
/// Builds the unified per-decode-token indices row in block-paged coords
/// (s_q = 1). See `csrc/attention/dsv4_flashmla_decode_build_indices.cu`
/// for the row layout (SW slots | compressed selections | -1 padding).
///
/// Parameters:
/// - `indices_ptr`: u64 device pointer to `int32[topk_unified]` output.
/// - `selected_ptr`: u64 device pointer to `int32[max_compressed_keys]`
///   for CSA mode (mode_int=1); pass `0` for HCA mode (mode_int=2).
/// - `sw_blocks`: SW sub-pool block count (`ceil(sliding_window / page_block_size)`).
/// - `sliding_window`: bf16 SW ring length.
/// - `start_pos`: absolute position of the decode token.
/// - `max_compressed_keys`: `index_topk` (CSA) or padded compressed count (HCA).
/// - `compress_ratio`: causality-gate ratio.
/// - `mode_int`: 1 = CSA, 2 = HCA.
/// - `page_block_size`: 64 for DSv4-Flash MODEL1.
///
/// `topk_unified = sliding_window + max_compressed_keys` must be %128==0
/// (FlashMLA invariant); the kernel returns `cudaErrorInvalidValue` otherwise.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_flashmla_decode_build_indices_raw(
    ctx: &DeviceContext,
    indices_ptr: u64,
    selected_ptr: u64,
    sw_blocks: usize,
    sliding_window: usize,
    start_pos: usize,
    max_compressed_keys: usize,
    compress_ratio: usize,
    mode_int: i32,
    page_block_size: usize,
) -> Result<()> {
    unsafe {
        ffi::arle_dsv4_flashmla_decode_build_indices_cuda(
            indices_ptr as *mut i32,
            selected_ptr as *const i32,
            sw_blocks as i32,
            sliding_window as i32,
            start_pos as i32,
            max_compressed_keys as i32,
            compress_ratio as i32,
            mode_int,
            page_block_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

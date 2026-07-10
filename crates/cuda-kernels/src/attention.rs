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

    // SAFETY: NoPE/RoPE/index pointers come from live device buffers pinned by
    // the `_g*` guards, each holding `n_tokens` rows; `packed_kv_ptr` is the
    // caller's FP8 pool sized per the doc contract (`num_blocks *
    // page_block_size * 584` B). Writes land only at `[block_id, row]` slots
    // named per token, stream-ordered on `ctx.stream`.
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

    // SAFETY: same kernel contract as `dsv4_fp8_kv_pack`; `nope_ptr`/`rope_ptr`
    // are caller-lifted device addresses covering `n_tokens` bf16 rows on
    // `ctx.stream`, index slices are pinned by the `_g*` guards.
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
///
/// `page_table` is the OPTIONAL Stage-B device page-table lookup: `None` (the
/// default for every current caller) keeps the Stage-A band path byte-for-byte;
/// `Some(table)` reinterprets `token_block_id[t]` as a slot-LOGICAL page routed
/// through `table[logical]` (identity table = byte-identical to band).
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
    page_table: Option<&CudaSlice<i32>>,
) -> Result<()> {
    if n_tokens == 0 {
        return Ok(());
    }

    let (tbid_ptr, _gt) = token_block_id.device_ptr(&ctx.stream);
    let (tibr_ptr, _gi) = token_in_block_row.device_ptr(&ctx.stream);
    let (pt_ptr, num_logical_pages, _gp) = match page_table {
        Some(table) => {
            let (ptr, guard) = table.device_ptr(&ctx.stream);
            (ptr as *const i32, table.len() as i32, Some(guard))
        }
        None => (std::ptr::null(), 0, None),
    };

    // SAFETY: `nope_ptr`/`rope_ptr` are caller device addresses valid for
    // `n_tokens` rows at the given element strides (≥ 448 / 64 per the doc
    // contract); index slices are pinned by `_g*`; `pt_ptr` is null exactly
    // when no Stage-B table is supplied (the kernel then uses band addressing,
    // bounds-checked by `num_logical_pages`). Stream-ordered on `ctx.stream`.
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
            pt_ptr,
            num_logical_pages,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}

/// V32 (GLM-5.2, 512 NoPE / 656 B/tok) strided FP8 KV pack. Same call shape as
/// [`dsv4_fp8_kv_pack_strided_raw`] but launches the SEPARATE V32 kernel that
/// writes the inline layout `[512 NoPE fp8][4 F32 scales @512][128 rope bf16]`
/// (stride 656; F32 per-128-block scales = amax/448, no pow-2 rounding — NOT
/// the MODEL1 trailing-e8m0 format). Caller passes `nope = k_prepared,
/// rope = k_prepared + 512, stride_nope_elems = stride_rope_elems = head_dim = 576`.
pub fn dsv4_v32_fp8_kv_pack_strided_raw(
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

    // SAFETY: `nope_ptr`/`rope_ptr` are caller device addresses valid for
    // `n_tokens` rows at the given strides; `packed_kv_ptr` is the V32 pool
    // (656 B/token layout per the doc contract); index slices pinned by `_g*`.
    // Stream-ordered on `ctx.stream`.
    unsafe {
        ffi::arle_dsv4_v32_fp8_kv_pack_strided_cuda(
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

/// Fill the single `[block_id,row]` scratch pair for FlashMLA decode SW pack
/// from a device-resident `start_pos` scalar.
pub fn dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_raw(
    ctx: &DeviceContext,
    token_block_id_ptr: u64,
    token_in_block_row_ptr: u64,
    start_pos_ptr: u64,
    sliding_window: usize,
    page_block_size: usize,
) -> Result<()> {
    // SAFETY: the three raw pointers are caller device addresses — one i32
    // scratch pair to write plus a device-resident `start_pos` scalar to read —
    // all live on `ctx.stream`; the kernel writes exactly one `[block_id, row]`
    // pair, stream-ordered.
    unsafe {
        ffi::arle_dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_cuda(
            token_block_id_ptr as *mut i32,
            token_in_block_row_ptr as *mut i32,
            start_pos_ptr as *const i32,
            sliding_window as i32,
            page_block_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Fill the `[block_id,row]` scratch arrays for a batch of FlashMLA decode SW
/// packs. `token_block_id[row]` is absolute in the shared FP8 KV arena.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_fp8_kv_fill_sw_slots_from_start_pos_raw(
    ctx: &DeviceContext,
    token_block_id_ptr: u64,
    token_in_block_row_ptr: u64,
    start_pos_ptr: u64,
    slot_layer_block_offsets_ptr: u64,
    n_tokens: usize,
    sliding_window: usize,
    page_block_size: usize,
) -> Result<()> {
    // SAFETY: all raw pointers are caller device addresses valid for
    // `n_tokens` i32 entries (scratch outputs, per-row `start_pos`, per-slot
    // block offsets) on `ctx.stream`; writes are bounded by `n_tokens`,
    // stream-ordered.
    unsafe {
        ffi::arle_dsv4_fp8_kv_fill_sw_slots_from_start_pos_cuda(
            token_block_id_ptr as *mut i32,
            token_in_block_row_ptr as *mut i32,
            start_pos_ptr as *const i32,
            slot_layer_block_offsets_ptr as *const i32,
            n_tokens as i32,
            sliding_window as i32,
            page_block_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Pack the just-completed compressor row, if this decode step completes one.
/// The kernel reads `start_pos_ptr` on device, so a captured CUDA Graph can
/// replay across decode positions without host-computed row constants.
pub fn dsv4_fp8_kv_pack_completed_compressor_row_start_pos_raw(
    ctx: &DeviceContext,
    compressed_ptr: u64,
    packed_kv_ptr: u64,
    start_pos_ptr: u64,
    ratio: usize,
    sw_blocks: usize,
    page_block_size: usize,
    stride_elems: usize,
    page_table: Option<&CudaSlice<i32>>,
) -> Result<()> {
    let (pt_ptr, num_logical_pages, _gp) = match page_table {
        Some(table) => {
            let (ptr, guard) = table.device_ptr(&ctx.stream);
            (ptr as *const i32, table.len() as i32, Some(guard))
        }
        None => (std::ptr::null(), 0, None),
    };
    // SAFETY: `compressed_ptr`/`packed_kv_ptr`/`start_pos_ptr` are caller
    // device addresses (compressor row source, FP8 pool, device i32 scalar) on
    // `ctx.stream`; the kernel early-outs unless this step completes a
    // compressor row, `pt_ptr` is null exactly when no Stage-B table is
    // supplied. Stream-ordered (graph-replay safe: row derived on device).
    unsafe {
        ffi::arle_dsv4_fp8_kv_pack_completed_compressor_row_start_pos_cuda(
            compressed_ptr as *const ffi::Half,
            packed_kv_ptr as *mut u8,
            start_pos_ptr as *const i32,
            ratio as i32,
            sw_blocks as i32,
            page_block_size as i32,
            stride_elems as i32,
            pt_ptr,
            num_logical_pages,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Batched (b=N) MODEL1 SW one-token pack: ONE launch over `n` decode rows. The
/// pointer-array form of [`dsv4_fp8_kv_pack_strided_raw`] for the batched decode
/// lane — `nope_arr`/`rope_arr` hold N device pointers (each = a row's
/// `k_prepared` NoPE / RoPE base), `page_table_arr` holds N per-slot device
/// page-table pointers, `start_pos` is the contiguous `[N]` decode positions,
/// `packed_kv_ptr` is the single shared pool base. Ring slot + page-table route
/// are computed on device from `start_pos[row]` (graph-replay safe). n=1 with the
/// singleton arrays is byte-identical to the per-row fill + strided pack.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_fp8_kv_pack_strided_batched_raw(
    ctx: &DeviceContext,
    nope_arr: &CudaSlice<u64>,
    rope_arr: &CudaSlice<u64>,
    packed_kv_ptr: u64,
    start_pos: &CudaSlice<i32>,
    n: usize,
    page_block_size: usize,
    sliding_window: usize,
    stride_nope_elems: usize,
    stride_rope_elems: usize,
    page_table_arr: &CudaSlice<u64>,
    num_logical_pages: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let (nope_a, _gn) = nope_arr.device_ptr(&ctx.stream);
    let (rope_a, _gr) = rope_arr.device_ptr(&ctx.stream);
    let (start_a, _gs) = start_pos.device_ptr(&ctx.stream);
    let (pt_a, _gp) = page_table_arr.device_ptr(&ctx.stream);
    // SAFETY: the pointer arrays (`nope_arr`/`rope_arr`/`page_table_arr`) and
    // `start_pos` are live `[n]` CudaSlices pinned by `_g*`; each embedded
    // device pointer is a live per-row `k_prepared` / page-table base per the
    // batched-lane contract. Ring slot + route are computed on device from
    // `start_pos[row]`; writes go to the shared pool only. Stream-ordered.
    unsafe {
        ffi::arle_dsv4_fp8_kv_pack_strided_batched_cuda(
            nope_a as *const *const ffi::Half,
            rope_a as *const *const ffi::Half,
            packed_kv_ptr as *mut u8,
            start_a as *const i32,
            n as i32,
            page_block_size as i32,
            sliding_window as i32,
            stride_nope_elems as i32,
            stride_rope_elems as i32,
            pt_a as *const *const i32,
            num_logical_pages as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Batched (b=N) MODEL1 compressed-delta pack: ONE launch over `n` decode rows.
/// The pointer-array form of
/// [`dsv4_fp8_kv_pack_completed_compressor_row_start_pos_raw`] — `compressed_arr`
/// holds N device pointers (a row's compressor `compressed` base, or `0`/null for
/// rows with no compressor → kernel no-op). Each row early-outs on
/// `(pos+1)%ratio != 0`. n=1 byte-identical to the per-row compressed pack.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_fp8_kv_pack_completed_compressor_row_batched_raw(
    ctx: &DeviceContext,
    compressed_arr: &CudaSlice<u64>,
    packed_kv_ptr: u64,
    start_pos: &CudaSlice<i32>,
    n: usize,
    ratio: usize,
    sw_blocks: usize,
    page_block_size: usize,
    stride_elems: usize,
    page_table_arr: &CudaSlice<u64>,
    num_logical_pages: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let (comp_a, _gc) = compressed_arr.device_ptr(&ctx.stream);
    let (start_a, _gs) = start_pos.device_ptr(&ctx.stream);
    let (pt_a, _gp) = page_table_arr.device_ptr(&ctx.stream);
    // SAFETY: `compressed_arr`/`start_pos`/`page_table_arr` are live `[n]`
    // CudaSlices pinned by `_g*`; embedded per-row compressor pointers may be
    // null (kernel no-ops that row) and each row early-outs unless it completes
    // a compressor row. Writes go to the shared FP8 pool only, stream-ordered.
    unsafe {
        ffi::arle_dsv4_fp8_kv_pack_completed_compressor_row_batched_cuda(
            comp_a as *const *const ffi::Half,
            packed_kv_ptr as *mut u8,
            start_a as *const i32,
            n as i32,
            ratio as i32,
            sw_blocks as i32,
            page_block_size as i32,
            stride_elems as i32,
            pt_a as *const *const i32,
            num_logical_pages as i32,
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
///
/// `page_table` is the OPTIONAL Stage-B logical→physical page table: `None` (the
/// default for every current caller) keeps the Stage-A slot-relative path
/// byte-for-byte; `Some(table)` routes each emitted index's logical page through
/// the table (identity table = the Stage-A index, index-for-index).
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
    page_table: Option<&CudaSlice<i32>>,
    total_blocks: usize,
) -> Result<()> {
    let (pt_ptr, num_logical_pages, _gp) = match page_table {
        Some(table) => {
            let (ptr, guard) = table.device_ptr(&ctx.stream);
            (ptr as *const i32, table.len() as i32, Some(guard))
        }
        None => (std::ptr::null(), 0, None),
    };
    // SAFETY: `indices_ptr` is the caller's `int32[topk_unified]` output and
    // `selected_ptr` the CSA selection row (0/null in HCA mode, which the
    // kernel never dereferences); `pt_ptr` is null exactly when no Stage-B
    // table is supplied, else bounds-checked by `num_logical_pages`.
    // Stream-ordered on `ctx.stream`.
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
            pt_ptr,
            num_logical_pages,
            total_blocks as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Same as [`dsv4_flashmla_decode_build_indices_raw`], but reads
/// `start_pos` from a single `int32` device scalar. This is the graph-ready
/// entrypoint for decode paths that stamp per-step metadata on the stream.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_flashmla_decode_build_indices_start_pos_ptr_raw(
    ctx: &DeviceContext,
    indices_ptr: u64,
    selected_ptr: u64,
    sw_blocks: usize,
    sliding_window: usize,
    start_pos_ptr: u64,
    max_compressed_keys: usize,
    compress_ratio: usize,
    mode_int: i32,
    page_block_size: usize,
    page_table: Option<&CudaSlice<i32>>,
    total_blocks: usize,
) -> Result<()> {
    let (pt_ptr, num_logical_pages, _gp) = match page_table {
        Some(table) => {
            let (ptr, guard) = table.device_ptr(&ctx.stream);
            (ptr as *const i32, table.len() as i32, Some(guard))
        }
        None => (std::ptr::null(), 0, None),
    };
    // SAFETY: same contract as the non-`_start_pos_ptr` variant above, except
    // `start_pos_ptr` is a device i32 scalar read in-kernel (graph-replay
    // safe). All pointers live on `ctx.stream`; `pt_ptr` null ⇔ no Stage-B
    // table. Stream-ordered.
    unsafe {
        ffi::arle_dsv4_flashmla_decode_build_indices_start_pos_ptr_cuda(
            indices_ptr as *mut i32,
            selected_ptr as *const i32,
            sw_blocks as i32,
            sliding_window as i32,
            start_pos_ptr as *const i32,
            max_compressed_keys as i32,
            compress_ratio as i32,
            mode_int,
            page_block_size as i32,
            pt_ptr,
            num_logical_pages,
            total_blocks as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Batched `b = N` variant for DSv4 FlashMLA sparse-decode indices.
///
/// Emits `indices[b, topk_unified]` and `topk_length[b]` in one launch, reading
/// per-row absolute positions from `start_pos[b]`. Non-masked slot ids are
/// shifted by `slot_layer_block_offsets[row] * page_block_size`, matching
/// FlashMLA's absolute slot addressing over one contiguous shared FP8 KV base.
/// Active scheduler slots are not required to be row-contiguous.
///
/// `page_table` is the OPTIONAL Stage-B per-row logical→physical page table
/// (`[max_batch, page_table_row_width]`): `None` (Stage-A band path) keeps the
/// kernel byte-for-byte; `Some(table)` routes each row's indices to POOL-absolute
/// and SKIPS the `slot_layer_block_offsets` band shift (identity per-row table =
/// the Stage-A index after the band shift). `page_table_row_width` is the FIXED
/// row stride the host writes at (`table.len() / max_batch`, = per-slot
/// `total_blocks`), independent of the active row count `b ≤ max_batch` — the
/// kernel reads `page_table + row * row_width` and bounds-checks logical pages
/// against it, so it MUST match the host stride (not `table.len()/b`, which only
/// equals it when `b == max_batch`).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_flashmla_decode_build_indices_batched_raw(
    ctx: &DeviceContext,
    indices_ptr: u64,
    start_pos_ptr: u64,
    slot_layer_block_offsets_ptr: u64,
    selected_ptr: u64,
    topk_length_ptr: u64,
    b: usize,
    sw_blocks: usize,
    sliding_window: usize,
    max_compressed_keys: usize,
    compress_ratio: usize,
    mode_int: i32,
    page_block_size: usize,
    total_blocks: usize,
    page_table: Option<&CudaSlice<i32>>,
    page_table_row_width: usize,
) -> Result<()> {
    let (pt_ptr, num_logical_pages, _gp) = match page_table {
        Some(table) => {
            // The shared page-table buffer is allocated for max_total_blocks (widest
            // layer). Narrower layers use a smaller row_width and write only
            // n*row_width entries into it — so divisibility of the full buffer by
            // the current layer's row_width is NOT required. The correct bound is
            // that the active portion (b * row_width) fits in the buffer.
            anyhow::ensure!(
                page_table_row_width > 0,
                "batched page table row width must be > 0"
            );
            anyhow::ensure!(
                b * page_table_row_width <= table.len(),
                "batched page table: b {b} × row width {page_table_row_width} exceeds len {}",
                table.len()
            );
            let (ptr, guard) = table.device_ptr(&ctx.stream);
            (ptr as *const i32, page_table_row_width as i32, Some(guard))
        }
        None => (std::ptr::null(), 0, None),
    };
    // SAFETY: raw pointers are caller device addresses sized for `b` rows
    // (`indices[b, topk_unified]`, `start_pos[b]`, offsets, `topk_length[b]`;
    // `selected_ptr` null ⇔ HCA mode). The page table, when present, was
    // bounds-checked above (`b * row_width <= len`) and is pinned by `_gp`;
    // the kernel bounds-checks logical pages per row. Stream-ordered on
    // `ctx.stream`.
    unsafe {
        ffi::arle_dsv4_flashmla_decode_build_indices_batched_cuda(
            indices_ptr as *mut i32,
            start_pos_ptr as *const i32,
            slot_layer_block_offsets_ptr as *const i32,
            selected_ptr as *const i32,
            topk_length_ptr as *mut i32,
            b as i32,
            sw_blocks as i32,
            sliding_window as i32,
            max_compressed_keys as i32,
            compress_ratio as i32,
            mode_int,
            page_block_size as i32,
            total_blocks as i32,
            pt_ptr,
            num_logical_pages,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

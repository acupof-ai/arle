//! DSv4-Flash attention-side packing/dispatch wrappers.
//!
//! This module holds the DSv4-Flash MODEL1 (and V32) FP8 KV pack kernels plus
//! FlashMLA sparse-decode index builders that feed FlashMLA's sparse-FP8
//! decode path
//! (`sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`).
//!
//! Runtime wire-up (D-4) is a separate dispatch; this wrapper exposes the
//! FFI through the established `DeviceContext`-driven idiom that the rest
//! of the kernel crate uses.

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

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

/// Build FlashMLA decode indices, reading `start_pos` from a single `int32`
/// device scalar. This is the graph-ready entrypoint for decode paths that
/// stamp per-step metadata on the stream.
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

// ---------------------------------------------------------------------------
// Qwen3.5/3.6 non-paged prefill attention family: typed launchers over the
// HD256 prep/gate/attention FFI. Raw u64 device addresses — consumers apply
// pointer offsets (pool head shifts, per-row columns) before the call and keep
// every owning buffer alive and stream-ordered on `stream`.
// ---------------------------------------------------------------------------

fn attn_i32(v: usize, what: &'static str) -> Result<i32> {
    i32::try_from(v).map_err(|_| anyhow!("{what} {v} exceeds i32"))
}

/// Non-paged causal attention (bf16, online softmax, GQA native). `q`/`out`
/// token-major `[seq, q_heads, d]`; `k_cache`/`v_cache` head-major
/// `[kv_heads, max_seq_len, d]`.
#[allow(clippy::too_many_arguments)]
pub fn nonpaged_prefill_attention_raw(
    stream: &CudaStream,
    q_ptr: u64,
    k_cache_ptr: u64,
    v_cache_ptr: u64,
    out_ptr: u64,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    max_seq_len: usize,
    sm_scale: f32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::nonpaged_prefill_attention_cuda(
            q_ptr as *const ffi::Half,
            k_cache_ptr as *const ffi::Half,
            v_cache_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            attn_i32(num_q_heads, "nonpaged q_heads")?,
            attn_i32(num_kv_heads, "nonpaged kv_heads")?,
            attn_i32(head_dim, "nonpaged head_dim")?,
            attn_i32(seq_len, "nonpaged seq_len")?,
            attn_i32(kv_len, "nonpaged kv_len")?,
            attn_i32(max_seq_len, "nonpaged max_seq_len")?,
            sm_scale,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("nonpaged_prefill_attention_cuda failed at seq={seq_len} kv={kv_len}: {e}")
        })
    }
}

/// Device-position variant of [`nonpaged_prefill_attention_raw`]:
/// `start_pos_dev_ptr` is one device i32 read in-kernel (graph-replay safe).
#[allow(clippy::too_many_arguments)]
pub fn nonpaged_prefill_attention_devpos_raw(
    stream: &CudaStream,
    q_ptr: u64,
    k_cache_ptr: u64,
    v_cache_ptr: u64,
    out_ptr: u64,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    start_pos_dev_ptr: u64,
    max_seq_len: usize,
    sm_scale: f32,
) -> Result<()> {
    // SAFETY: same contract as `nonpaged_prefill_attention_raw`;
    // `start_pos_dev_ptr` is a live device i32 scalar.
    unsafe {
        ffi::nonpaged_prefill_attention_devpos_cuda(
            q_ptr as *const ffi::Half,
            k_cache_ptr as *const ffi::Half,
            v_cache_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            attn_i32(num_q_heads, "devpos q_heads")?,
            attn_i32(num_kv_heads, "devpos kv_heads")?,
            attn_i32(head_dim, "devpos head_dim")?,
            attn_i32(seq_len, "devpos seq_len")?,
            start_pos_dev_ptr as *const i32,
            attn_i32(max_seq_len, "devpos max_seq_len")?,
            sm_scale,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("nonpaged_prefill_attention_devpos_cuda failed at seq={seq_len}: {e}"))
    }
}

/// Hand-written FA2-style forward for sm_70 (V100): drop-in replacement for
/// [`nonpaged_prefill_attention_raw`] where FA3 (sm_80+) is unavailable.
#[allow(clippy::too_many_arguments)]
pub fn fa2_sm70_attention_raw(
    stream: &CudaStream,
    q_ptr: u64,
    k_cache_ptr: u64,
    v_cache_ptr: u64,
    out_ptr: u64,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    max_seq_len: usize,
    sm_scale: f32,
) -> Result<()> {
    // SAFETY: same layout contract as `nonpaged_prefill_attention_raw`.
    unsafe {
        ffi::arle_fa2_sm70_attention_cuda(
            q_ptr as *const ffi::Half,
            k_cache_ptr as *const ffi::Half,
            v_cache_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            attn_i32(num_q_heads, "fa2 q_heads")?,
            attn_i32(num_kv_heads, "fa2 kv_heads")?,
            attn_i32(head_dim, "fa2 head_dim")?,
            attn_i32(seq_len, "fa2 seq_len")?,
            attn_i32(kv_len, "fa2 kv_len")?,
            attn_i32(max_seq_len, "fa2 max_seq_len")?,
            sm_scale,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("arle_fa2_sm70_attention_cuda failed at seq={seq_len} kv={kv_len}: {e}")
        })
    }
}

/// Qwen3.6 HD256 non-paged prefill prep: q/k-norm + partial RoPE, q into
/// `q_out`, K/V appended to the contiguous head-major caches at
/// `*start_pos_dev_ptr`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_hd256_prep_raw(
    stream: &CudaStream,
    q_full_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    q_norm_ptr: u64,
    k_norm_ptr: u64,
    cos_ptr: u64,
    sin_ptr: u64,
    q_out_ptr: u64,
    k_cache_ptr: u64,
    v_cache_ptr: u64,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    start_pos_dev_ptr: u64,
    rotary_dim: usize,
    rms_eps: f32,
    max_seq_len: usize,
) -> Result<()> {
    // SAFETY: caller passes live device addresses for the prep layout; the
    // caches are sized `max_seq_len * kv_dim`.
    unsafe {
        ffi::prefill_attention_hd256_prep_cuda(
            q_full_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            q_out_ptr as *mut ffi::Half,
            k_cache_ptr as *mut ffi::Half,
            v_cache_ptr as *mut ffi::Half,
            attn_i32(num_q_heads, "hd256 prep q_heads")?,
            attn_i32(num_kv_heads, "hd256 prep kv_heads")?,
            attn_i32(head_dim, "hd256 prep head_dim")?,
            attn_i32(seq_len, "hd256 prep seq_len")?,
            start_pos_dev_ptr as *const i32,
            attn_i32(rotary_dim, "hd256 prep rotary_dim")?,
            rms_eps,
            attn_i32(max_seq_len, "hd256 prep max_seq_len")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("prefill_attention_hd256_prep_cuda failed at seq={seq_len}: {e}"))
    }
}

/// Qwen3.6 HD256 paged prefill prep for ONE request: `page_table_ptr` is the
/// request's slice of `kv_indices`, `start_pos_dev_ptr` its scalar entry in
/// the batch start-position table.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_prep_hd256_raw(
    stream: &CudaStream,
    q_full_ptr: u64,
    q_out_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    q_norm_ptr: u64,
    k_norm_ptr: u64,
    cos_ptr: u64,
    sin_ptr: u64,
    page_table_ptr: u64,
    page_size: usize,
    k_pool_ptr: u64,
    v_pool_ptr: u64,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    start_pos_dev_ptr: u64,
    rotary_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses (row-offset q/k/v columns,
    // per-request page-table slice, pool bases); per-row offsets come from the
    // meta's own prefix sums, so each launch stays inside its row.
    unsafe {
        ffi::prefill_attention_paged_prep_hd256_cuda(
            q_full_ptr as *const ffi::Half,
            q_out_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            page_table_ptr as *const i32,
            attn_i32(page_size, "paged prep page_size")?,
            k_pool_ptr as *mut ffi::Half,
            v_pool_ptr as *mut ffi::Half,
            attn_i32(num_q_heads, "paged prep q_heads")?,
            attn_i32(num_kv_heads, "paged prep kv_heads")?,
            attn_i32(seq_len, "paged prep seq_len")?,
            start_pos_dev_ptr as *const i32,
            attn_i32(rotary_dim, "paged prep rotary_dim")?,
            rms_eps,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("prefill_attention_paged_prep_hd256_cuda failed at seq={seq_len}: {e}")
        })
    }
}

/// Qwen3.6 HD256 paged decode prep, one q row per batch element.
/// `write_kv`: 0 = skip the K/V pool write (2D non-owner shard); 1 = write.
#[allow(clippy::too_many_arguments)]
pub fn decode_prep_paged_hd256_raw(
    stream: &CudaStream,
    q_full_ptr: u64,
    q_out_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    q_norm_ptr: u64,
    k_norm_ptr: u64,
    cos_ptr: u64,
    sin_ptr: u64,
    positions_ptr: u64,
    k_pool_ptr: u64,
    v_pool_ptr: u64,
    page_table_ptr: u64,
    page_indptr_ptr: u64,
    last_page_len_ptr: u64,
    num_qo_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    stride_page: usize,
    batch_size: usize,
    rotary_dim: usize,
    rms_eps: f32,
    write_kv: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses for the decode-prep layout;
    // the pool base may be head-offset (B2 subset), tail pages allocated.
    unsafe {
        ffi::decode_prep_paged_hd256_cuda(
            q_full_ptr as *const ffi::Half,
            q_out_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            positions_ptr as *const i32,
            k_pool_ptr as *mut ffi::Half,
            v_pool_ptr as *mut ffi::Half,
            page_table_ptr as *const i32,
            page_indptr_ptr as *const i32,
            last_page_len_ptr as *const i32,
            attn_i32(num_qo_heads, "decode prep qo_heads")?,
            attn_i32(num_kv_heads, "decode prep kv_heads")?,
            attn_i32(page_size, "decode prep page_size")?,
            attn_i32(stride_page, "decode prep stride_page")?,
            attn_i32(batch_size, "decode prep batch")?,
            attn_i32(rotary_dim, "decode prep rotary_dim")?,
            rms_eps,
            write_kv,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("decode_prep_paged_hd256_cuda failed at batch={batch_size}: {e}"))
    }
}

/// Paged attention v1 (HD128/HD256 BF16 pool), resolved per
/// `(head_dim, q_heads, kv_heads, phase)` from the generated kernel table;
/// an unregistered head geometry is an error, never a silent fallback.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v1_raw(
    stream: &CudaStream,
    phase: ffi::AttnPhase,
    q_ptr: u64,
    q_indptr_ptr: u64,
    k_pool_ptr: u64,
    v_pool_ptr: u64,
    kv_indptr_ptr: u64,
    kv_indices_ptr: u64,
    last_page_len_ptr: u64,
    out_ptr: u64,
    batch: usize,
    total_q: usize,
    max_q: usize,
    max_total_pages: usize,
    num_pages: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    sm_scale: f32,
) -> Result<()> {
    let kernel = ffi::resolve_paged_attn_v1(
        head_dim as u32,
        num_q_heads as u32,
        num_kv_heads as u32,
        phase,
    )
    .ok_or_else(|| {
        anyhow!(
            "no paged_attn_v1 {phase:?} kernel for hd{head_dim} q{num_q_heads}_kv{num_kv_heads}"
        )
    })?;
    // SAFETY: caller passes live device addresses (q rows, page tables, pool
    // bases, output) sized to the dims passed, stream-ordered on `stream`.
    unsafe {
        kernel(
            q_ptr as *mut ffi::Half,
            q_indptr_ptr as *const i32,
            k_pool_ptr as *mut ffi::Half,
            v_pool_ptr as *mut ffi::Half,
            kv_indptr_ptr as *const i32,
            kv_indices_ptr as *const i32,
            last_page_len_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            attn_i32(batch, "paged attn batch")?,
            attn_i32(total_q, "paged attn total_q")?,
            attn_i32(max_q, "paged attn max_q")?,
            attn_i32(max_total_pages, "paged attn max_total_pages")?,
            attn_i32(num_pages, "paged attn num_pages")?,
            attn_i32(num_q_heads, "paged attn q_heads")?,
            attn_i32(num_kv_heads, "paged attn kv_heads")?,
            attn_i32(page_size, "paged attn page_size")?,
            sm_scale,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("paged_attn_v1 {phase:?} failed at total_q={total_q}: {e}"))
    }
}

/// Sliding-window ring variant of [`prefill_attention_hd256_prep_raw`]: the
/// K/V cache write row wraps as `pos % ring_modulus`. One launch must write
/// `<= ring_modulus` rows (caller-checked — the kernel cannot).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_hd256_prep_ring_raw(
    stream: &CudaStream,
    q_full_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    q_norm_ptr: u64,
    k_norm_ptr: u64,
    cos_ptr: u64,
    sin_ptr: u64,
    q_out_ptr: u64,
    k_cache_ptr: u64,
    v_cache_ptr: u64,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    start_pos_dev_ptr: u64,
    rotary_dim: usize,
    rms_eps: f32,
    ring_modulus: usize,
) -> Result<()> {
    // SAFETY: caller passes live device addresses; ring caches sized
    // `ring_modulus * kv_dim`, `start_pos_dev_ptr` an ABSOLUTE device i32.
    unsafe {
        ffi::prefill_attention_hd256_prep_ring_cuda(
            q_full_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            q_out_ptr as *mut ffi::Half,
            k_cache_ptr as *mut ffi::Half,
            v_cache_ptr as *mut ffi::Half,
            attn_i32(num_q_heads, "ring prep q_heads")?,
            attn_i32(num_kv_heads, "ring prep kv_heads")?,
            attn_i32(head_dim, "ring prep head_dim")?,
            attn_i32(seq_len, "ring prep seq_len")?,
            start_pos_dev_ptr as *const i32,
            attn_i32(rotary_dim, "ring prep rotary_dim")?,
            rms_eps,
            attn_i32(ring_modulus, "ring prep ring_modulus")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("prefill_attention_hd256_prep_ring_cuda failed at seq={seq_len}: {e}"))
    }
}

/// Ragged-window ring attention: one launch for `seq_len` rows with per-row
/// device-resident key windows `[ring_base_dev[t], +kv_len_dev[t])`, walked
/// non-causally. Caller guarantees `kv_len_dev[t] <= ring_modulus`.
#[allow(clippy::too_many_arguments)]
pub fn nonpaged_prefill_attention_ring_varlen_raw(
    stream: &CudaStream,
    q_ptr: u64,
    k_cache_ptr: u64,
    v_cache_ptr: u64,
    out_ptr: u64,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    ring_base_dev_ptr: u64,
    kv_len_dev_ptr: u64,
    ring_modulus: usize,
    sm_scale: f32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses; the window tables hold
    // `seq_len` i32 each, stream-ordered on `stream`.
    unsafe {
        ffi::nonpaged_prefill_attention_ring_varlen_cuda(
            q_ptr as *const ffi::Half,
            k_cache_ptr as *const ffi::Half,
            v_cache_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            attn_i32(num_q_heads, "ring varlen q_heads")?,
            attn_i32(num_kv_heads, "ring varlen kv_heads")?,
            attn_i32(head_dim, "ring varlen head_dim")?,
            attn_i32(seq_len, "ring varlen seq_len")?,
            ring_base_dev_ptr as *const i32,
            kv_len_dev_ptr as *const i32,
            attn_i32(ring_modulus, "ring varlen ring_modulus")?,
            sm_scale,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("nonpaged_prefill_attention_ring_varlen_cuda failed at seq={seq_len}: {e}")
        })
    }
}

/// Slot-batched [`nonpaged_prefill_attention_ring_varlen_raw`]: `k_slots_ptr` /
/// `v_slots_ptr` are device arrays of `slots` ring-cache base pointers, and the
/// window tables are slot-major (`slots * seq_len` i32 each).
#[allow(clippy::too_many_arguments)]
pub fn nonpaged_prefill_attention_ring_varlen_batched_raw(
    stream: &CudaStream,
    q_ptr: u64,
    k_slots_ptr: u64,
    v_slots_ptr: u64,
    out_ptr: u64,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    slots: usize,
    ring_base_dev_ptr: u64,
    kv_len_dev_ptr: u64,
    ring_modulus: usize,
    sm_scale: f32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses; the slot arrays hold `slots`
    // live ring-cache bases, window tables slot-major, stream-ordered.
    unsafe {
        ffi::nonpaged_prefill_attention_ring_varlen_batched_cuda(
            q_ptr as *const ffi::Half,
            k_slots_ptr as *const *const std::ffi::c_void,
            v_slots_ptr as *const *const std::ffi::c_void,
            out_ptr as *mut ffi::Half,
            attn_i32(num_q_heads, "ring varlen batched q_heads")?,
            attn_i32(num_kv_heads, "ring varlen batched kv_heads")?,
            attn_i32(head_dim, "ring varlen batched head_dim")?,
            attn_i32(seq_len, "ring varlen batched seq_len")?,
            attn_i32(slots, "ring varlen batched slots")?,
            ring_base_dev_ptr as *const i32,
            kv_len_dev_ptr as *const i32,
            attn_i32(ring_modulus, "ring varlen batched ring_modulus")?,
            sm_scale,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "nonpaged_prefill_attention_ring_varlen_batched_cuda failed at slots={slots}: {e}"
            )
        })
    }
}

/// 2D ring-prefill dense prep (Qwen3.6 HD256): q/k-norm + partial RoPE into
/// dense head-major buffers with NO pool write (the scatter owns the pool
/// write).
#[allow(clippy::too_many_arguments)]
pub fn ring_prefill_dense_prep_hd256_raw(
    stream: &CudaStream,
    q_full_ptr: u64,
    k_in_ptr: u64,
    v_in_ptr: u64,
    q_norm_ptr: u64,
    k_norm_ptr: u64,
    cos_ptr: u64,
    sin_ptr: u64,
    q_out_ptr: u64,
    k_out_ptr: u64,
    v_out_ptr: u64,
    num_qo_heads: usize,
    num_kv_heads: usize,
    rows: usize,
    start_pos: usize,
    rotary_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::ring_prefill_dense_prep_hd256_cuda(
            q_full_ptr as *const ffi::Half,
            k_in_ptr as *const ffi::Half,
            v_in_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            q_out_ptr as *mut ffi::Half,
            k_out_ptr as *mut ffi::Half,
            v_out_ptr as *mut ffi::Half,
            attn_i32(num_qo_heads, "ring dense prep qo_heads")?,
            attn_i32(num_kv_heads, "ring dense prep kv_heads")?,
            attn_i32(rows, "ring dense prep rows")?,
            attn_i32(start_pos, "ring dense prep start_pos")?,
            attn_i32(rotary_dim, "ring dense prep rotary_dim")?,
            rms_eps,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("ring_prefill_dense_prep_hd256_cuda failed at rows={rows}: {e}"))
    }
}

/// 2D ring-prefill block-cyclic scatter: write the ring block's tokens whose
/// global page is owned by this shard into the sharded HND pool.
#[allow(clippy::too_many_arguments)]
pub fn ring_prefill_scatter_sharded_hd256_raw(
    stream: &CudaStream,
    k_dense_ptr: u64,
    v_dense_ptr: u64,
    local_page_table_ptr: u64,
    local_page_count: usize,
    page_size: usize,
    kv_heads: usize,
    blk_start: usize,
    blk_len: usize,
    cp_rank: usize,
    cp_size: usize,
    stride_page: usize,
    k_pool_ptr: u64,
    v_pool_ptr: u64,
) -> Result<()> {
    // SAFETY: dense buffers hold the block's prepped K/V (head-major, stride =
    // blk_len); the page table is this shard's local table; stream-ordered.
    unsafe {
        ffi::ring_prefill_scatter_sharded_hd256_cuda(
            k_dense_ptr as *const ffi::Half,
            v_dense_ptr as *const ffi::Half,
            local_page_table_ptr as *const i32,
            attn_i32(local_page_count, "ring scatter page_count")?,
            attn_i32(page_size, "ring scatter page_size")?,
            attn_i32(kv_heads, "ring scatter kv_heads")?,
            attn_i32(blk_start, "ring scatter blk_start")?,
            attn_i32(blk_len, "ring scatter blk_len")?,
            attn_i32(cp_rank, "ring scatter cp_rank")?,
            attn_i32(cp_size, "ring scatter cp_size")?,
            attn_i32(stride_page, "ring scatter stride_page")?,
            k_pool_ptr as *mut ffi::Half,
            v_pool_ptr as *mut ffi::Half,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("ring_prefill_scatter_sharded_hd256_cuda failed at blk_len={blk_len}: {e}")
        })
    }
}

/// 2D ring-prefill finalize: `out = O / L` (flash-2 normalized), transposing
/// the accumulator's head-major layout into row-major bf16.
pub fn ring_prefill_finalize_bf16_hd256_raw(
    stream: &CudaStream,
    acc_l_ptr: u64,
    acc_o_ptr: u64,
    out_ptr: u64,
    q_heads: usize,
    rows: usize,
) -> Result<()> {
    // SAFETY: acc buffers are `[q_heads*rows]` / `[q_heads*rows*d]` f32; out is
    // `[rows, q_heads*d]` bf16, stream-ordered on `stream`.
    unsafe {
        ffi::ring_prefill_finalize_bf16_hd256_cuda(
            acc_l_ptr as *const f32,
            acc_o_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            attn_i32(q_heads, "ring finalize q_heads")?,
            attn_i32(rows, "ring finalize rows")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("ring_prefill_finalize_bf16_hd256_cuda failed at rows={rows}: {e}"))
    }
}

/// Non-paged HD256 sigmoid gate: `attn_out *= sigmoid(gate)` with the gate
/// rows read from the full q projection.
pub fn attention_gate_batch_hd256_raw(
    stream: &CudaStream,
    q_full_ptr: u64,
    attn_out_ptr: u64,
    num_q_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Result<()> {
    // SAFETY: q_full/attn_out are live device buffers in the full-attn prep
    // layout.
    unsafe {
        ffi::attention_gate_batch_hd256_cuda(
            q_full_ptr as *const ffi::Half,
            attn_out_ptr as *mut ffi::Half,
            attn_i32(num_q_heads, "gate q_heads")?,
            attn_i32(head_dim, "gate head_dim")?,
            attn_i32(seq_len, "gate seq_len")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("attention_gate_batch_hd256_cuda failed at seq={seq_len}: {e}"))
    }
}

/// Paged variant of [`attention_gate_batch_hd256_raw`]: iterates
/// `rows * num_q_heads` over the ragged batch.
pub fn attention_gate_paged_hd256_raw(
    stream: &CudaStream,
    q_full_ptr: u64,
    attn_out_ptr: u64,
    num_q_heads: usize,
    rows: usize,
) -> Result<()> {
    // SAFETY: q_full/attn_out are live device buffers in the full-attn prep
    // layout.
    unsafe {
        ffi::attention_gate_paged_hd256_cuda(
            q_full_ptr as *const ffi::Half,
            attn_out_ptr as *mut ffi::Half,
            attn_i32(num_q_heads, "paged gate q_heads")?,
            attn_i32(rows, "paged gate rows")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("attention_gate_paged_hd256_cuda failed at rows={rows}: {e}"))
    }
}

/// 2D cross-cp flash-decoding merge: combine `cp_size` rank-major (lse, out)
/// sections into the full attention output (see the FFI decl for the packed
/// section layout).
#[allow(clippy::too_many_arguments)]
pub fn cross_cp_merge_bf16_hd256_raw(
    stream: &CudaStream,
    packed_ptr: u64,
    lse_stride_f32: usize,
    out_off_bf16: usize,
    out_stride_bf16: usize,
    out_ptr: u64,
    cp_size: usize,
    rows: usize,
    head_dim: usize,
) -> Result<()> {
    // SAFETY: `packed_ptr` is the gathered `[cp, section]` buffer, `out_ptr`
    // the live output; the caller fences the gather before this launch.
    unsafe {
        ffi::cross_cp_merge_bf16_hd256_cuda(
            packed_ptr as *const ffi::Half,
            attn_i32(lse_stride_f32, "cp merge lse_stride")?,
            attn_i32(out_off_bf16, "cp merge out_off")?,
            attn_i32(out_stride_bf16, "cp merge out_stride")?,
            out_ptr as *mut ffi::Half,
            attn_i32(cp_size, "cp merge cp_size")?,
            attn_i32(rows, "cp merge rows")?,
            attn_i32(head_dim, "cp merge head_dim")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("cross_cp_merge_bf16_hd256_cuda failed at rows={rows}: {e}"))
    }
}

/// FA3 hd256 bf16 forward. The args struct is the ABI — the caller builds it
/// per the field docs on [`ffi::ArleFa3FwdHd256Args`] and keeps every pointed-to
/// buffer alive and stream-ordered through submission.
pub fn fa3_fwd_hd256_bf16(stream: &CudaStream, args: &ffi::ArleFa3FwdHd256Args) -> Result<()> {
    // SAFETY: caller upholds the args-struct pointer contract.
    unsafe {
        ffi::arle_fa3_fwd_hd256_bf16_cuda(args, stream.cu_stream())
            .result()
            .map_err(|e| {
                anyhow!(
                    "arle_fa3_fwd_hd256_bf16_cuda failed at batch={} total_q={}: {e}",
                    args.batch,
                    args.total_q
                )
            })
    }
}

/// FA3 hd256 quantized-KV forward (Path A dequant shim); same contract as
/// [`fa3_fwd_hd256_bf16`] over [`ffi::ArleFa3FwdHd256QuantArgs`].
pub fn fa3_fwd_hd256_quant(
    stream: &CudaStream,
    args: &ffi::ArleFa3FwdHd256QuantArgs,
) -> Result<()> {
    // SAFETY: caller upholds the args-struct pointer contract.
    unsafe {
        ffi::arle_fa3_fwd_hd256_quant_cuda(args, stream.cu_stream())
            .result()
            .map_err(|e| {
                anyhow!(
                    "arle_fa3_fwd_hd256_quant_cuda failed at batch={} total_q={}: {e}",
                    args.base.batch,
                    args.base.total_q
                )
            })
    }
}

// ---------------------------------------------------------------------------
// DSv4 MLA/DSA/compressor attention family: typed launchers over the FlashMLA
// shim, DSA indexer, compressor/window-cache, QK prep, output inverse-RoPE, TP
// repack/slice, and top-k transform FFI. Pointer args are raw u64
// device addresses (0 = null where the ABI allows it); scalar args mirror the
// FFI types so consumers pass identical values. Callers keep every owning
// buffer alive and stream-ordered on `stream`.
// ---------------------------------------------------------------------------

/// FlashMLA SM90 sparse bf16 prefill forward (vendored shim; d_qk ∈ {512,576},
/// d_v = 512).
#[allow(clippy::too_many_arguments)]
pub fn flashmla_sm90_sparse_prefill_fwd_raw(
    stream: &CudaStream,
    q_ptr: u64,
    kv_ptr: u64,
    indices_ptr: u64,
    attn_sink_ptr: u64,
    topk_length_ptr: u64,
    out_ptr: u64,
    max_logits_ptr: u64,
    lse_ptr: u64,
    s_q: i32,
    s_kv: i32,
    h_q: i32,
    h_kv: i32,
    d_qk: i32,
    d_v: i32,
    topk: i32,
    sm_scale: f32,
    stride_q_s_q: i32,
    stride_q_h_q: i32,
    stride_kv_s_kv: i32,
    stride_kv_h_kv: i32,
    stride_indices_s_q: i32,
    stride_indices_h_kv: i32,
    num_sm: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims/strides
    // below, stream-ordered on `stream`.
    unsafe {
        ffi::arle_flashmla_sm90_sparse_prefill_fwd(
            q_ptr as *const ffi::Half,
            kv_ptr as *const ffi::Half,
            indices_ptr as *const i32,
            attn_sink_ptr as *const f32,
            topk_length_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            max_logits_ptr as *mut f32,
            lse_ptr as *mut f32,
            s_q,
            s_kv,
            h_q,
            h_kv,
            d_qk,
            d_v,
            topk,
            sm_scale,
            stride_q_s_q,
            stride_q_h_q,
            stride_kv_s_kv,
            stride_kv_h_kv,
            stride_indices_s_q,
            stride_indices_h_kv,
            num_sm,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("arle_flashmla_sm90_sparse_prefill_fwd failed at s_q={s_q}: {e}"))
    }
}

/// FlashMLA SM90 sparse FP8 decode forward (splitkv + combine; KV is the
/// FP8-packed pool per the model-specific byte layout).
#[allow(clippy::too_many_arguments)]
pub fn flashmla_sm90_sparse_decode_fwd_raw(
    stream: &CudaStream,
    q_ptr: u64,
    kv_ptr: u64,
    indices_ptr: u64,
    topk_length_ptr: u64,
    attn_sink_ptr: u64,
    out_ptr: u64,
    lse_ptr: u64,
    lse_accum_ptr: u64,
    o_accum_ptr: u64,
    tile_scheduler_metadata_ptr: u64,
    num_splits_ptr: u64,
    b: i32,
    s_q: i32,
    h_q: i32,
    h_kv: i32,
    d_qk: i32,
    d_v: i32,
    num_blocks: i32,
    page_block_size: i32,
    topk: i32,
    num_sm_parts: i32,
    model_type_int: i32,
    sm_scale: f32,
    stride_q_b: i32,
    stride_q_s_q: i32,
    stride_q_h_q: i32,
    stride_kv_block_bytes: i32,
    stride_kv_row_bytes: i32,
    stride_indices_b: i32,
    stride_indices_s_q: i32,
    stride_lse_b: i32,
    stride_lse_s_q: i32,
    stride_o_b: i32,
    stride_o_s_q: i32,
    stride_o_h_q: i32,
    stride_lse_accum_split: i32,
    stride_lse_accum_s_q: i32,
    stride_o_accum_split: i32,
    stride_o_accum_s_q: i32,
    stride_o_accum_h_q: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims/strides
    // below (KV FP8-packed per `model_type_int`), stream-ordered on `stream`.
    unsafe {
        ffi::arle_flashmla_sm90_sparse_decode_fwd(
            q_ptr as *const ffi::Half,
            kv_ptr as *const ffi::Half,
            indices_ptr as *const i32,
            topk_length_ptr as *const i32,
            attn_sink_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            lse_ptr as *mut f32,
            lse_accum_ptr as *mut f32,
            o_accum_ptr as *mut f32,
            tile_scheduler_metadata_ptr as *const i32,
            num_splits_ptr as *const i32,
            b,
            s_q,
            h_q,
            h_kv,
            d_qk,
            d_v,
            num_blocks,
            page_block_size,
            topk,
            num_sm_parts,
            model_type_int,
            sm_scale,
            stride_q_b,
            stride_q_s_q,
            stride_q_h_q,
            stride_kv_block_bytes,
            stride_kv_row_bytes,
            stride_indices_b,
            stride_indices_s_q,
            stride_lse_b,
            stride_lse_s_q,
            stride_o_b,
            stride_o_s_q,
            stride_o_h_q,
            stride_lse_accum_split,
            stride_lse_accum_s_q,
            stride_o_accum_split,
            stride_o_accum_s_q,
            stride_o_accum_h_q,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("arle_flashmla_sm90_sparse_decode_fwd failed at b={b}: {e}"))
    }
}

/// Host-side decode scheduler tuning meta for a (`h_q`, `s_q`,
/// `model_type_int`) tuple. Returns `(num_sm_parts,
/// fixed_overhead_num_blocks, block_size_topk)`.
pub fn flashmla_sm90_sparse_decode_get_meta(
    h_q: i32,
    s_q: i32,
    model_type_int: i32,
) -> Result<(i32, i32, i32)> {
    let mut num_sm_parts = 0_i32;
    let mut fixed_overhead_num_blocks = 0_i32;
    let mut block_size_topk = 0_i32;
    // SAFETY: host-only computation writing the three out scalars.
    unsafe {
        ffi::arle_flashmla_sm90_sparse_decode_get_meta(
            h_q,
            s_q,
            model_type_int,
            &mut num_sm_parts,
            &mut fixed_overhead_num_blocks,
            &mut block_size_topk,
        )
        .result()
        .map_err(|e| {
            anyhow!("arle_flashmla_sm90_sparse_decode_get_meta failed at h_q={h_q}: {e}")
        })?;
    }
    Ok((num_sm_parts, fixed_overhead_num_blocks, block_size_topk))
}

/// Populate the decode `tile_scheduler_metadata` + `num_splits` device arrays
/// from per-batch effective topk lengths. `extra_topk_length_ptr` may be 0.
#[allow(clippy::too_many_arguments)]
pub fn flashmla_sm90_sparse_decode_sched_meta_raw(
    stream: &CudaStream,
    b: i32,
    s_q: i32,
    block_size_topk: i32,
    fixed_overhead_num_blocks: i32,
    topk: i32,
    extra_topk: i32,
    topk_length_ptr: u64,
    extra_topk_length_ptr: u64,
    tile_scheduler_metadata_ptr: u64,
    num_splits_ptr: u64,
    num_sm_parts: i32,
) -> Result<()> {
    // SAFETY: caller passes live device arrays sized per the FFI doc
    // (`num_sm_parts * DecodingSchedMetaSize/4` and `b + 1` i32);
    // `extra_topk_length_ptr` is null exactly when 0. Stream-ordered.
    unsafe {
        ffi::arle_flashmla_sm90_sparse_decode_sched_meta(
            b,
            s_q,
            block_size_topk,
            fixed_overhead_num_blocks,
            topk,
            extra_topk,
            topk_length_ptr as *const i32,
            extra_topk_length_ptr as *const i32,
            tile_scheduler_metadata_ptr as *mut i32,
            num_splits_ptr as *mut i32,
            num_sm_parts,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("arle_flashmla_sm90_sparse_decode_sched_meta failed at b={b}: {e}"))
    }
}

/// Chain-verify sparse indices for FlashMLA prefill (top-1 chain verifier).
/// `selected_ptr` may be 0 (no CSA topk arm).
#[allow(clippy::too_many_arguments)]
pub fn flashmla_chain_verify_build_indices_raw(
    stream: &CudaStream,
    indices_ptr: u64,
    topk_length_ptr: u64,
    positions_ptr: u64,
    ancestors_ptr: u64,
    max_anc: i32,
    selected_ptr: u64,
    s_q: i32,
    start_pos: i32,
    sw_window: i32,
    index_topk: i32,
    max_compressed: i32,
    topk_unified: i32,
    compressed_count: i32,
    compress_ratio: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below;
    // `selected_ptr` is null exactly when 0. Stream-ordered on `stream`.
    unsafe {
        ffi::arle_flashmla_chain_verify_build_indices(
            indices_ptr as *mut i32,
            topk_length_ptr as *mut i32,
            positions_ptr as *const i32,
            ancestors_ptr as *const i32,
            max_anc,
            selected_ptr as *const i32,
            s_q,
            start_pos,
            sw_window,
            index_topk,
            max_compressed,
            topk_unified,
            compressed_count,
            compress_ratio,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("arle_flashmla_chain_verify_build_indices failed at s_q={s_q}: {e}"))
    }
}

/// CSA per-token unified indices + topk_length (layout of
/// [`flashmla_csa_pack_kv_raw`]). `selected_ptr` may be 0 (SWA reuse: fills -1).
#[allow(clippy::too_many_arguments)]
pub fn flashmla_csa_build_indices_raw(
    stream: &CudaStream,
    indices_ptr: u64,
    topk_length_ptr: u64,
    selected_ptr: u64,
    s_q: i32,
    start_pos: i32,
    sw_window: i32,
    index_topk: i32,
    compressed_count: i32,
    compress_ratio: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below;
    // `selected_ptr` is null exactly when 0. Stream-ordered on `stream`.
    unsafe {
        ffi::arle_flashmla_csa_build_indices(
            indices_ptr as *mut i32,
            topk_length_ptr as *mut i32,
            selected_ptr as *const i32,
            s_q,
            start_pos,
            sw_window,
            index_topk,
            compressed_count,
            compress_ratio,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("arle_flashmla_csa_build_indices failed at s_q={s_q}: {e}"))
    }
}

/// HCA per-token unified indices (no selector; all compressed pages causally
/// gated by `compress_ratio`).
#[allow(clippy::too_many_arguments)]
pub fn flashmla_hca_build_indices_raw(
    stream: &CudaStream,
    indices_ptr: u64,
    topk_length_ptr: u64,
    s_q: i32,
    start_pos: i32,
    sw_window: i32,
    max_compressed_keys: i32,
    compressed_count: i32,
    compress_ratio: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized per the FFI doc
    // (`s_q * (sw_window + max_compressed_keys)` i32). Stream-ordered.
    unsafe {
        ffi::arle_flashmla_hca_build_indices(
            indices_ptr as *mut i32,
            topk_length_ptr as *mut i32,
            s_q,
            start_pos,
            sw_window,
            max_compressed_keys,
            compressed_count,
            compress_ratio,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("arle_flashmla_hca_build_indices failed at s_q={s_q}: {e}"))
    }
}

/// Pack SW window cache + current-chunk K + compressed pool into one
/// contiguous KV pool for FlashMLA SM90 sparse prefill. `compressed_ptr` may
/// be 0 (no compressed rows).
#[allow(clippy::too_many_arguments)]
pub fn flashmla_csa_pack_kv_raw(
    stream: &CudaStream,
    kv_unified_ptr: u64,
    window_cache_ptr: u64,
    k_prepared_ptr: u64,
    compressed_ptr: u64,
    start_pos: i32,
    sw_window: i32,
    n_tokens: i32,
    compressed_count: i32,
    d_qk: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below;
    // `compressed_ptr` is null exactly when 0. Stream-ordered on `stream`.
    unsafe {
        ffi::arle_flashmla_csa_pack_kv(
            kv_unified_ptr as *mut ffi::Half,
            window_cache_ptr as *const ffi::Half,
            k_prepared_ptr as *const ffi::Half,
            compressed_ptr as *const ffi::Half,
            start_pos,
            sw_window,
            n_tokens,
            compressed_count,
            d_qk,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("arle_flashmla_csa_pack_kv failed at n_tokens={n_tokens}: {e}"))
    }
}

/// DSA fused Q-indexer RoPE + Hadamard + FP8 quant over `batch_size` rows.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_dsa_fused_q_indexer_rope_hadamard_quant_raw(
    stream: &CudaStream,
    q_input_ptr: u64,
    q_fp8_ptr: u64,
    weight_ptr: u64,
    weights_out_ptr: u64,
    weight_scale: f32,
    freqs_cis_ptr: u64,
    positions_ptr: u64,
    batch_size: i32,
    num_heads: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_dsa_fused_q_indexer_rope_hadamard_quant_cuda(
            q_input_ptr as *const ffi::Half,
            q_fp8_ptr as *mut u8,
            weight_ptr as *const ffi::Half,
            weights_out_ptr as *mut f32,
            weight_scale,
            freqs_cis_ptr as *const f32,
            positions_ptr as *const i32,
            batch_size,
            num_heads,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "dsv4_dsa_fused_q_indexer_rope_hadamard_quant_cuda failed at batch={batch_size}: {e}"
            )
        })
    }
}

/// DSA Hadamard-128 rotate over `rows` bf16 index-key rows.
pub fn dsv4_dsa_hadamard128_bf16_raw(
    stream: &CudaStream,
    input_ptr: u64,
    output_ptr: u64,
    rows: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses holding `rows` index-key
    // rows, stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_dsa_hadamard128_bf16_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut ffi::Half,
            rows,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_dsa_hadamard128_bf16_cuda failed at rows={rows}: {e}"))
    }
}

/// DSA fused FP8 quant + store of rotated index keys into the paged key cache.
pub fn dsv4_dsa_fused_store_index_k_cache_raw(
    stream: &CudaStream,
    key_ptr: u64,
    index_k_with_scale_ptr: u64,
    out_cache_loc_ptr: u64,
    num_tokens: i32,
    page_size: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses (rotated keys, cache band,
    // `num_tokens` i64 cache locs), stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_dsa_fused_store_index_k_cache_cuda(
            key_ptr as *const ffi::Half,
            index_k_with_scale_ptr as *mut u8,
            out_cache_loc_ptr as *const i64,
            num_tokens,
            page_size,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("dsv4_dsa_fused_store_index_k_cache_cuda failed at tokens={num_tokens}: {e}")
        })
    }
}

/// Batched (grid.y=slot) DSA Hadamard rotate: ONE launch over `n` slots' rows.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_dsa_hadamard128_batched_raw(
    stream: &CudaStream,
    keys_src_arr_ptr: u64,
    src_ring_row_arr_ptr: u64,
    rotated_dst_arr_ptr: u64,
    dst_row_arr_ptr: u64,
    newly_packed_arr_ptr: u64,
    n: i32,
    max_rows: i32,
) -> Result<()> {
    // SAFETY: caller passes live `[n]` device pointer/offset arrays whose
    // embedded per-slot pointers are live, stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_dsa_hadamard128_batched_cuda(
            keys_src_arr_ptr as *const *const ffi::Half,
            src_ring_row_arr_ptr as *const i32,
            rotated_dst_arr_ptr as *const *mut ffi::Half,
            dst_row_arr_ptr as *const i32,
            newly_packed_arr_ptr as *const i32,
            n,
            max_rows,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_dsa_hadamard128_batched_cuda failed at n={n}: {e}"))
    }
}

/// Batched (grid.y=slot) DSA FP8 fused-store: ONE launch over `n` slots.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_dsa_fused_store_index_k_cache_batched_raw(
    stream: &CudaStream,
    key_arr_ptr: u64,
    cache_arr_ptr: u64,
    out_cache_loc_arr_ptr: u64,
    newly_packed_arr_ptr: u64,
    n: i32,
    max_tokens: i32,
    page_size: i32,
) -> Result<()> {
    // SAFETY: caller passes live `[n]` device pointer arrays whose embedded
    // per-slot pointers are live, stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_dsa_fused_store_index_k_cache_batched_cuda(
            key_arr_ptr as *const *const ffi::Half,
            cache_arr_ptr as *const *mut u8,
            out_cache_loc_arr_ptr as *const *const i64,
            newly_packed_arr_ptr as *const i32,
            n,
            max_tokens,
            page_size,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("dsv4_dsa_fused_store_index_k_cache_batched_cuda failed at n={n}: {e}")
        })
    }
}

/// Fill per-tile DSA `context_lens` / `positions` on device from a
/// device-resident `start_pos` scalar (graph-replay safe).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_dsa_fill_context_lens_positions_start_pos_raw(
    stream: &CudaStream,
    context_lens_ptr: u64,
    positions_ptr: u64,
    start_pos_ptr: u64,
    token_offset: i32,
    batch_size: i32,
    key_count: i32,
    ratio: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses (`batch_size` i32 each plus
    // a device i32 scalar), stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_dsa_fill_context_lens_positions_start_pos_cuda(
            context_lens_ptr as *mut i32,
            positions_ptr as *mut i32,
            start_pos_ptr as *const i32,
            token_offset,
            batch_size,
            key_count,
            ratio,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "dsv4_dsa_fill_context_lens_positions_start_pos_cuda failed at batch={batch_size}: {e}"
            )
        })
    }
}

/// Compressor sub-block update over `num_tokens` (host `start_pos` variant).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_compressor_update_raw(
    stream: &CudaStream,
    kv_raw_ptr: u64,
    score_raw_ptr: u64,
    ape_ptr: u64,
    norm_ptr: u64,
    pending_kv_ptr: u64,
    pending_score_ptr: u64,
    prev_overlap_kv_ptr: u64,
    prev_overlap_score_ptr: u64,
    compressed_ptr: u64,
    num_tokens: i32,
    start_pos: i32,
    pending_len: i32,
    compressed_base: i32,
    head_dim: i32,
    ratio: i32,
    width: i32,
    overlap: i32,
    has_prev_overlap: i32,
    overlap_page_stride: i32,
    eps: f32,
    rope_dim: i32,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses matching the checked ratio /
    // width / token count, stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_compressor_update_cuda(
            kv_raw_ptr as *const ffi::Half,
            score_raw_ptr as *const ffi::Half,
            ape_ptr as *const ffi::Half,
            norm_ptr as *const ffi::Half,
            pending_kv_ptr as *mut ffi::Half,
            pending_score_ptr as *mut ffi::Half,
            prev_overlap_kv_ptr as *mut ffi::Half,
            prev_overlap_score_ptr as *mut ffi::Half,
            compressed_ptr as *mut ffi::Half,
            num_tokens,
            start_pos,
            pending_len,
            compressed_base,
            head_dim,
            ratio,
            width,
            overlap,
            has_prev_overlap,
            overlap_page_stride,
            eps,
            rope_dim,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_compressor_update_cuda failed at tokens={num_tokens}: {e}"))
    }
}

/// Device-position compressor update (graph-replay safe).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_compressor_update_start_pos_ptr_raw(
    stream: &CudaStream,
    kv_raw_ptr: u64,
    score_raw_ptr: u64,
    ape_ptr: u64,
    norm_ptr: u64,
    pending_kv_ptr: u64,
    pending_score_ptr: u64,
    prev_overlap_kv_ptr: u64,
    prev_overlap_score_ptr: u64,
    compressed_ptr: u64,
    num_tokens: i32,
    start_pos_ptr: u64,
    head_dim: i32,
    ratio: i32,
    width: i32,
    overlap: i32,
    overlap_page_stride: i32,
    eps: f32,
    rope_dim: i32,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: same contract as [`dsv4_compressor_update_raw`];
    // `start_pos_ptr` is a live device i32 scalar.
    unsafe {
        ffi::dsv4_compressor_update_start_pos_ptr_cuda(
            kv_raw_ptr as *const ffi::Half,
            score_raw_ptr as *const ffi::Half,
            ape_ptr as *const ffi::Half,
            norm_ptr as *const ffi::Half,
            pending_kv_ptr as *mut ffi::Half,
            pending_score_ptr as *mut ffi::Half,
            prev_overlap_kv_ptr as *mut ffi::Half,
            prev_overlap_score_ptr as *mut ffi::Half,
            compressed_ptr as *mut ffi::Half,
            num_tokens,
            start_pos_ptr as *const i32,
            head_dim,
            ratio,
            width,
            overlap,
            overlap_page_stride,
            eps,
            rope_dim,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("dsv4_compressor_update_start_pos_ptr_cuda failed at tokens={num_tokens}: {e}")
        })
    }
}

/// Batched decode compressor update: ONE launch over `n` rows' per-slot ring
/// state (host-gathered device pointer arrays).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_compressor_update_batched_start_pos_ptr_raw(
    stream: &CudaStream,
    kv_raw_ptr: u64,
    score_raw_ptr: u64,
    ape_ptr: u64,
    norm_ptr: u64,
    pending_kv_arr_ptr: u64,
    pending_score_arr_ptr: u64,
    prev_overlap_kv_arr_ptr: u64,
    prev_overlap_score_arr_ptr: u64,
    compressed_arr_ptr: u64,
    n: i32,
    num_tokens: i32,
    start_pos_arr_ptr: u64,
    head_dim: i32,
    ratio: i32,
    width: i32,
    overlap: i32,
    overlap_page_stride: i32,
    eps: f32,
    rope_dim: i32,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: caller passes live `[n]` device pointer arrays whose embedded
    // per-row pointers are live; kv/score are the batched prepass outputs.
    // Stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_compressor_update_batched_start_pos_ptr_cuda(
            kv_raw_ptr as *const ffi::Half,
            score_raw_ptr as *const ffi::Half,
            ape_ptr as *const ffi::Half,
            norm_ptr as *const ffi::Half,
            pending_kv_arr_ptr as *const *mut ffi::Half,
            pending_score_arr_ptr as *const *mut ffi::Half,
            prev_overlap_kv_arr_ptr as *const *mut ffi::Half,
            prev_overlap_score_arr_ptr as *const *mut ffi::Half,
            compressed_arr_ptr as *const *mut ffi::Half,
            n,
            num_tokens,
            start_pos_arr_ptr as *const i32,
            head_dim,
            ratio,
            width,
            overlap,
            overlap_page_stride,
            eps,
            rope_dim,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("dsv4_compressor_update_batched_start_pos_ptr_cuda failed at n={n}: {e}")
        })
    }
}

/// FP32-accumulating compressor prefill probe (writes both f32 carry and bf16
/// mirror buffers).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_compressor_fp32_prefill_probe_raw(
    stream: &CudaStream,
    kv_raw_ptr: u64,
    score_raw_ptr: u64,
    ape_ptr: u64,
    norm_ptr: u64,
    pending_kv_ptr: u64,
    pending_score_ptr: u64,
    prev_overlap_kv_ptr: u64,
    prev_overlap_score_ptr: u64,
    prev_overlap_kv_bf16_ptr: u64,
    prev_overlap_score_bf16_ptr: u64,
    pending_kv_bf16_ptr: u64,
    pending_score_bf16_ptr: u64,
    compressed_ptr: u64,
    num_tokens: i32,
    start_pos: i32,
    pending_len: i32,
    compressed_base: i32,
    head_dim: i32,
    ratio: i32,
    width: i32,
    overlap: i32,
    has_prev_overlap: i32,
    overlap_page_stride: i32,
    eps: f32,
    rope_dim: i32,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses matching the checked ratio /
    // width / token count (f32 raws + f32/bf16 carry pairs), stream-ordered.
    unsafe {
        ffi::dsv4_compressor_fp32_prefill_probe_cuda(
            kv_raw_ptr as *const f32,
            score_raw_ptr as *const f32,
            ape_ptr as *const f32,
            norm_ptr as *const ffi::Half,
            pending_kv_ptr as *mut f32,
            pending_score_ptr as *mut f32,
            prev_overlap_kv_ptr as *mut f32,
            prev_overlap_score_ptr as *mut f32,
            prev_overlap_kv_bf16_ptr as *mut ffi::Half,
            prev_overlap_score_bf16_ptr as *mut ffi::Half,
            pending_kv_bf16_ptr as *mut ffi::Half,
            pending_score_bf16_ptr as *mut ffi::Half,
            compressed_ptr as *mut ffi::Half,
            num_tokens,
            start_pos,
            pending_len,
            compressed_base,
            head_dim,
            ratio,
            width,
            overlap,
            has_prev_overlap,
            overlap_page_stride,
            eps,
            rope_dim,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("dsv4_compressor_fp32_prefill_probe_cuda failed at tokens={num_tokens}: {e}")
        })
    }
}

/// bf16 → f32 upcast of a compressor state's four carry buffers before an
/// FP32 probe whose bf16 carry advanced.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_compressor_fp32_carry_reseed_raw(
    stream: &CudaStream,
    pending_kv_bf16_ptr: u64,
    pending_score_bf16_ptr: u64,
    prev_kv_bf16_ptr: u64,
    prev_score_bf16_ptr: u64,
    pending_kv_ptr: u64,
    pending_score_ptr: u64,
    prev_kv_ptr: u64,
    prev_score_ptr: u64,
    pending_elems: i32,
    prev_elems: i32,
) -> Result<()> {
    // SAFETY: caller passes bf16/f32 carry buffers allocated with identical
    // element counts, stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_compressor_fp32_carry_reseed_cuda(
            pending_kv_bf16_ptr as *const ffi::Half,
            pending_score_bf16_ptr as *const ffi::Half,
            prev_kv_bf16_ptr as *const ffi::Half,
            prev_score_bf16_ptr as *const ffi::Half,
            pending_kv_ptr as *mut f32,
            pending_score_ptr as *mut f32,
            prev_kv_ptr as *mut f32,
            prev_score_ptr as *mut f32,
            pending_elems,
            prev_elems,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_compressor_fp32_carry_reseed_cuda failed: {e}"))
    }
}

/// Write `num_tokens` new keys into the bf16 SW ring cache (host `start_pos`).
pub fn dsv4_update_window_cache_raw(
    stream: &CudaStream,
    k_new_ptr: u64,
    window_cache_ptr: u64,
    num_tokens: i32,
    start_pos: i32,
    sliding_window: i32,
    head_dim: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses (`num_tokens` key rows, ring
    // sized `sliding_window * head_dim`), stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_update_window_cache_cuda(
            k_new_ptr as *const ffi::Half,
            window_cache_ptr as *mut ffi::Half,
            num_tokens,
            start_pos,
            sliding_window,
            head_dim,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_update_window_cache_cuda failed at tokens={num_tokens}: {e}"))
    }
}

/// Device-position SW ring write (graph-replay safe).
pub fn dsv4_update_window_cache_start_pos_ptr_raw(
    stream: &CudaStream,
    k_new_ptr: u64,
    window_cache_ptr: u64,
    num_tokens: i32,
    start_pos_ptr: u64,
    sliding_window: i32,
    head_dim: i32,
) -> Result<()> {
    // SAFETY: same contract as [`dsv4_update_window_cache_raw`];
    // `start_pos_ptr` is a live device i32 scalar.
    unsafe {
        ffi::dsv4_update_window_cache_start_pos_ptr_cuda(
            k_new_ptr as *const ffi::Half,
            window_cache_ptr as *mut ffi::Half,
            num_tokens,
            start_pos_ptr as *const i32,
            sliding_window,
            head_dim,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "dsv4_update_window_cache_start_pos_ptr_cuda failed at tokens={num_tokens}: {e}"
            )
        })
    }
}

/// Pointer-array batched SW ring write: ONE launch over `n` non-contiguous
/// rows.
pub fn dsv4_update_window_cache_batched_ptr_raw(
    stream: &CudaStream,
    k_arr_ptr: u64,
    cache_arr_ptr: u64,
    n: i32,
    start_pos_ptr: u64,
    sliding_window: i32,
    head_dim: i32,
) -> Result<()> {
    // SAFETY: caller passes live `[n]` device pointer arrays whose embedded
    // per-row pointers are live; `start_pos_ptr` is `[n]` i32. Stream-ordered.
    unsafe {
        ffi::dsv4_update_window_cache_batched_ptr_cuda(
            k_arr_ptr as *const *const ffi::Half,
            cache_arr_ptr as *const *mut ffi::Half,
            n,
            start_pos_ptr as *const i32,
            sliding_window,
            head_dim,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_update_window_cache_batched_ptr_cuda failed at n={n}: {e}"))
    }
}

/// DSv4 QK prep: q/k RMSNorm + YaRN partial RoPE (host `start_pos`).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_prepare_qk_raw(
    stream: &CudaStream,
    q_raw_ptr: u64,
    k_raw_ptr: u64,
    q_out_ptr: u64,
    k_out_ptr: u64,
    num_tokens: i32,
    local_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    start_pos: i32,
    rms_eps: f32,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_prepare_qk_cuda(
            q_raw_ptr as *const ffi::Half,
            k_raw_ptr as *const ffi::Half,
            q_out_ptr as *mut ffi::Half,
            k_out_ptr as *mut ffi::Half,
            num_tokens,
            local_heads,
            head_dim,
            rope_dim,
            start_pos,
            rms_eps,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_prepare_qk_cuda failed at tokens={num_tokens}: {e}"))
    }
}

/// Device-position [`dsv4_prepare_qk_raw`] (graph-replay safe).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_prepare_qk_start_pos_ptr_raw(
    stream: &CudaStream,
    q_raw_ptr: u64,
    k_raw_ptr: u64,
    q_out_ptr: u64,
    k_out_ptr: u64,
    num_tokens: i32,
    local_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    start_pos_ptr: u64,
    rms_eps: f32,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: same contract as [`dsv4_prepare_qk_raw`]; `start_pos_ptr` is a
    // live device i32 scalar.
    unsafe {
        ffi::dsv4_prepare_qk_start_pos_ptr_cuda(
            q_raw_ptr as *const ffi::Half,
            k_raw_ptr as *const ffi::Half,
            q_out_ptr as *mut ffi::Half,
            k_out_ptr as *mut ffi::Half,
            num_tokens,
            local_heads,
            head_dim,
            rope_dim,
            start_pos_ptr as *const i32,
            rms_eps,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("dsv4_prepare_qk_start_pos_ptr_cuda failed at tokens={num_tokens}: {e}")
        })
    }
}

/// Per-token device-position QK prep: `start_pos_ptr` is a `[num_tokens]`
/// device i32 array of absolute positions.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_prepare_qk_fused_batch_start_pos_raw(
    stream: &CudaStream,
    q_raw_ptr: u64,
    k_raw_ptr: u64,
    q_out_ptr: u64,
    k_out_ptr: u64,
    num_tokens: i32,
    local_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    start_pos_ptr: u64,
    rms_eps: f32,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: same contract as [`dsv4_prepare_qk_raw`]; `start_pos_ptr` is a
    // live `[num_tokens]` device i32 array.
    unsafe {
        ffi::dsv4_prepare_qk_fused_batch_start_pos_cuda(
            q_raw_ptr as *const ffi::Half,
            k_raw_ptr as *const ffi::Half,
            q_out_ptr as *mut ffi::Half,
            k_out_ptr as *mut ffi::Half,
            num_tokens,
            local_heads,
            head_dim,
            rope_dim,
            start_pos_ptr as *const i32,
            rms_eps,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("dsv4_prepare_qk_fused_batch_start_pos_cuda failed at tokens={num_tokens}: {e}")
        })
    }
}

/// Dense non-causal DSpark draft attention over the head-shared MLA latent:
/// `q` `[block_size, local_heads, head_dim]`, `latent_kv` `[kv_len, head_dim]`
/// (K==V, broadcast over heads), `out` `[block_size, local_heads, head_dim]`.
/// The value tail (`rope_dim`) is inverse-RoPE'd at `base_start_pos + token`
/// with the same RoPE params the caller's forward q/latent prep used.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_dspark_draft_attention_raw(
    stream: &CudaStream,
    q_ptr: u64,
    latent_kv_ptr: u64,
    out_ptr: u64,
    kv_len: i32,
    block_size: i32,
    local_heads: i32,
    head_dim: i32,
    nope_dim: i32,
    rope_dim: i32,
    base_start_pos: i32,
    sm_scale: f32,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: the caller holds q/latent/out live on `stream` at the shapes above.
    unsafe {
        ffi::dsv4_dspark_draft_attention_cuda(
            q_ptr as *const ffi::Half,
            latent_kv_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            kv_len,
            block_size,
            local_heads,
            head_dim,
            nope_dim,
            rope_dim,
            base_start_pos,
            sm_scale,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_dspark_draft_attention_cuda failed at kv_len={kv_len}: {e}"))
    }
}

/// In-place attention-output inverse-RoPE for the FlashMLA paths (host
/// `start_pos`). NEVER on the legacy hybrid path (double-apply).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_output_inverse_rope_raw(
    stream: &CudaStream,
    out_ptr: u64,
    token_count: i32,
    local_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    start_pos: i32,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: `out_ptr` is a live `[token_count, local_heads, head_dim]` bf16
    // buffer, stream-ordered on `stream`.
    unsafe {
        ffi::arle_dsv4_output_inverse_rope_cuda(
            out_ptr as *mut ffi::Half,
            token_count,
            local_heads,
            head_dim,
            rope_dim,
            start_pos,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("arle_dsv4_output_inverse_rope_cuda failed at tokens={token_count}: {e}")
        })
    }
}

/// Device-position [`dsv4_output_inverse_rope_raw`] (graph-replay safe).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_output_inverse_rope_start_pos_ptr_raw(
    stream: &CudaStream,
    out_ptr: u64,
    token_count: i32,
    local_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    start_pos_ptr: u64,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: same contract as [`dsv4_output_inverse_rope_raw`];
    // `start_pos_ptr` is a live device i32 scalar.
    unsafe {
        ffi::arle_dsv4_output_inverse_rope_start_pos_ptr_cuda(
            out_ptr as *mut ffi::Half,
            token_count,
            local_heads,
            head_dim,
            rope_dim,
            start_pos_ptr as *const i32,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "arle_dsv4_output_inverse_rope_start_pos_ptr_cuda failed at tokens={token_count}: {e}"
            )
        })
    }
}

/// Per-token device-position inverse-RoPE: `start_pos_ptr` is a
/// `[token_count]` device i32 array of absolute positions.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_output_inverse_rope_batch_start_pos_raw(
    stream: &CudaStream,
    out_ptr: u64,
    token_count: i32,
    local_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    start_pos_ptr: u64,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: same contract as [`dsv4_output_inverse_rope_raw`];
    // `start_pos_ptr` is a live `[token_count]` device i32 array.
    unsafe {
        ffi::arle_dsv4_output_inverse_rope_batch_start_pos_cuda(
            out_ptr as *mut ffi::Half,
            token_count,
            local_heads,
            head_dim,
            rope_dim,
            start_pos_ptr as *const i32,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!(
                "arle_dsv4_output_inverse_rope_batch_start_pos_cuda failed at tokens={token_count}: {e}"
            )
        })
    }
}

/// Pointer-array batched inverse-RoPE: ONE launch over `n` non-contiguous
/// output rows.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_output_inverse_rope_batched_ptr_raw(
    stream: &CudaStream,
    out_arr_ptr: u64,
    n: i32,
    local_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    start_pos_ptr: u64,
    rope_base: f32,
    original_seq_len: i32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // SAFETY: caller passes a live `[n]` device pointer array whose embedded
    // per-row pointers are live; `start_pos_ptr` is `[n]` i32. Stream-ordered.
    unsafe {
        ffi::arle_dsv4_output_inverse_rope_batched_ptr_cuda(
            out_arr_ptr as *const *mut ffi::Half,
            n,
            local_heads,
            head_dim,
            rope_dim,
            start_pos_ptr as *const i32,
            rope_base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("arle_dsv4_output_inverse_rope_batched_ptr_cuda failed at n={n}: {e}"))
    }
}

/// Repack AllGather rank-major recv buffer into FlashMLA's h_q-major Q layout.
pub fn dsv4_tp_q_repack_raw(
    stream: &CudaStream,
    gathered_ptr: u64,
    packed_ptr: u64,
    tp_world: i32,
    s_q: i32,
    h_local: i32,
    d: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_tp_q_repack_cuda(
            gathered_ptr as *const ffi::Half,
            packed_ptr as *mut ffi::Half,
            tp_world,
            s_q,
            h_local,
            d,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_tp_q_repack_cuda failed at s_q={s_q}: {e}"))
    }
}

/// Slice a local column block out of a `[s_q, global_width]` row-major buffer.
pub fn dsv4_tp_out_slice_raw(
    stream: &CudaStream,
    full_out_ptr: u64,
    local_ptr: u64,
    s_q: i32,
    global_width: i32,
    local_width: i32,
    head_offset: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_tp_out_slice_cuda(
            full_out_ptr as *const ffi::Half,
            local_ptr as *mut ffi::Half,
            s_q,
            global_width,
            local_width,
            head_offset,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_tp_out_slice_cuda failed at s_q={s_q}: {e}"))
    }
}

/// Gather O-LoRA group `group`'s strided column slice into a contiguous
/// `[cols_per_group, num_tokens]` buffer.
pub fn dsv4_oproj_group_gather_raw(
    stream: &CudaStream,
    src_ptr: u64,
    dst_ptr: u64,
    num_tokens: i32,
    groups: i32,
    cols_per_group: i32,
    group: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_oproj_group_gather_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            num_tokens,
            groups,
            cols_per_group,
            group,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_oproj_group_gather_cuda failed at group={group}: {e}"))
    }
}

/// Scatter O-LoRA group `group`'s contiguous output back into the strided
/// latent layout.
pub fn dsv4_oproj_group_scatter_raw(
    stream: &CudaStream,
    src_ptr: u64,
    dst_ptr: u64,
    num_tokens: i32,
    groups: i32,
    rows_per_group: i32,
    group: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_oproj_group_scatter_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            num_tokens,
            groups,
            rows_per_group,
            group,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dsv4_oproj_group_scatter_cuda failed at group={group}: {e}"))
    }
}

/// DSA top-k transform: read indexer logits, emit page-routed `page_indices`
/// plus raw top-k `raw_indices`.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_deepseek_v4_topk_transform_raw(
    stream: &CudaStream,
    scores_ptr: u64,
    seq_lens_ptr: u64,
    page_table_ptr: u64,
    page_indices_ptr: u64,
    raw_indices_ptr: u64,
    score_stride: i64,
    page_table_stride: i64,
    output_stride: i64,
    batch_size: i32,
    topk: i32,
    page_size: i32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the strides/dims
    // below, stream-ordered on `stream`.
    unsafe {
        ffi::dsv4_deepseek_v4_topk_transform_cuda(
            scores_ptr as *const f32,
            seq_lens_ptr as *const i32,
            page_table_ptr as *const i32,
            page_indices_ptr as *mut i32,
            raw_indices_ptr as *mut i32,
            score_stride,
            page_table_stride,
            output_stride,
            batch_size,
            topk,
            page_size,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("dsv4_deepseek_v4_topk_transform_cuda failed at batch={batch_size}: {e}")
        })
    }
}

#[cfg(test)]
mod fa2_sm70_tests {
    use crate::ffi::attention::arle_fa2_sm70_attention_cuda;
    use crate::tensor::{DeviceContext, cache_ptr};
    use cudarc::driver::sys::CUstream;
    use half::bf16;

    // Host reference: causal multi-head attention, BF16 I/O, FP32 math.
    // Q layout [seq, q_heads, d]; K/V cache layout [kv_heads, max_seq, d].
    fn reference_attention(
        q: &[bf16],
        k: &[bf16],
        v: &[bf16],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        max_seq_len: usize,
        sm_scale: f32,
    ) -> Vec<bf16> {
        let gqa = num_q_heads / num_kv_heads;
        let mut out = vec![bf16::from_f32(0.0); seq_len * num_q_heads * head_dim];
        for qh in 0..num_q_heads {
            let kh = qh / gqa;
            for q_pos in 0..seq_len {
                let q_base = q_pos * num_q_heads * head_dim + qh * head_dim;
                let k_base = kh * max_seq_len * head_dim;

                let mut logits = vec![0.0f32; q_pos + 1];
                for k_pos in 0..=q_pos {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d].to_f32() * k[k_base + k_pos * head_dim + d].to_f32();
                    }
                    logits[k_pos] = dot * sm_scale;
                }
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logits.iter().map(|l| (l - max).exp()).sum();

                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for k_pos in 0..=q_pos {
                        acc +=
                            (logits[k_pos] - max).exp() * v[k_base + k_pos * head_dim + d].to_f32();
                    }
                    out[q_base + d] = bf16::from_f32(acc / exp_sum);
                }
            }
        }
        out
    }

    #[test]
    fn fa2_sm70_matches_host_reference() {
        let ctx = DeviceContext::new().expect("cuda context");

        const SEQ_LEN: i32 = 4;
        const NUM_Q_HEADS: i32 = 2;
        const NUM_KV_HEADS: i32 = 1;
        const HEAD_DIM: i32 = 256;
        let sm_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

        let q_len = (SEQ_LEN * NUM_Q_HEADS * HEAD_DIM) as usize;
        let kv_len = (NUM_KV_HEADS * SEQ_LEN * HEAD_DIM) as usize;

        // Deterministic, signed, non-trivial input pattern.
        let mk = |len, mod_, off, scale| {
            (0..len)
                .map(|i| bf16::from_f32(((i as f32 % mod_) - off) * scale))
                .collect::<Vec<_>>()
        };
        let q_host = mk(q_len, 11.0, 5.0, 0.01);
        let k_host = mk(kv_len, 7.0, 3.0, 0.012);
        let v_host = mk(kv_len, 13.0, 6.0, 0.008);

        let q_dev = ctx.stream.clone_htod(&q_host).expect("q h2d");
        let k_dev = ctx.stream.clone_htod(&k_host).expect("k h2d");
        let v_dev = ctx.stream.clone_htod(&v_host).expect("v h2d");
        let o_dev = ctx.stream.alloc_zeros::<bf16>(q_len).expect("o alloc");

        let q_p = cache_ptr(&q_dev, &ctx).as_ptr() as *const u16;
        let k_p = cache_ptr(&k_dev, &ctx).as_ptr() as *const u16;
        let v_p = cache_ptr(&v_dev, &ctx).as_ptr() as *const u16;
        let o_p = cache_ptr(&o_dev, &ctx).as_mut_ptr() as *mut u16;
        let stream = ctx.stream.cu_stream() as CUstream;

        // SAFETY: buffers live for the call; shapes match the kernel contract.
        unsafe {
            arle_fa2_sm70_attention_cuda(
                q_p,
                k_p,
                v_p,
                o_p,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                SEQ_LEN,
                SEQ_LEN,
                SEQ_LEN,
                sm_scale,
                stream,
            )
            .result()
            .expect("fa2 sm70 kernel");
        }
        ctx.sync().expect("sync");

        let o_host = ctx.stream.clone_dtoh(&o_dev).expect("o d2h");
        let reference = reference_attention(
            &q_host,
            &k_host,
            &v_host,
            NUM_Q_HEADS as usize,
            NUM_KV_HEADS as usize,
            HEAD_DIM as usize,
            SEQ_LEN as usize,
            SEQ_LEN as usize,
            sm_scale,
        );

        let mut max_err = 0.0f32;
        let mut sum_err = 0.0f32;
        for (a, b) in o_host.iter().zip(reference.iter()) {
            let e = (a.to_f32() - b.to_f32()).abs();
            max_err = max_err.max(e);
            sum_err += e;
        }
        let mean_err = sum_err / (o_host.len() as f32);
        assert!(
            max_err < 0.05 && mean_err < 0.01,
            "FA2 sm_70 vs host reference mismatch: max_err={max_err} mean_err={mean_err}"
        );
    }
}

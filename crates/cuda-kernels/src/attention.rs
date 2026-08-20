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

/// Qwen3 dense paged prefill prep: q/k-norm + RoPE in place on `q`/`k`, K/V
/// scattered into the paged pools through the per-request page-table offsets.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_prep_raw(
    stream: &CudaStream,
    q_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    q_norm_ptr: u64,
    k_norm_ptr: u64,
    cos_ptr: u64,
    sin_ptr: u64,
    page_table_ptr: u64,
    page_table_offset_ptr: u64,
    page_size: usize,
    k_pool_ptr: u64,
    v_pool_ptr: u64,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    start_pos_dev_ptr: u64,
    rms_eps: f32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses (batch rows, per-request
    // page-table slice + offsets, pool bases), stream-ordered on `stream`.
    unsafe {
        ffi::prefill_attention_paged_prep_cuda(
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            v_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            page_table_ptr as *const i32,
            page_table_offset_ptr as *const i32,
            attn_i32(page_size, "qwen paged prep page_size")?,
            k_pool_ptr as *mut ffi::Half,
            v_pool_ptr as *mut ffi::Half,
            attn_i32(num_q_heads, "qwen paged prep q_heads")?,
            attn_i32(num_kv_heads, "qwen paged prep kv_heads")?,
            attn_i32(head_dim, "qwen paged prep head_dim")?,
            attn_i32(seq_len, "qwen paged prep seq_len")?,
            start_pos_dev_ptr as *const i32,
            rms_eps,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("prefill_attention_paged_prep_cuda failed at seq={seq_len}: {e}"))
    }
}

/// Qwen3 dense paged decode prep, one q row per batch element: q/k-norm +
/// RoPE in place on `q`, K/V appended to the paged pools at each row's last
/// page.
#[allow(clippy::too_many_arguments)]
pub fn decode_prep_paged_raw(
    stream: &CudaStream,
    q_ptr: u64,
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
    rms_eps: f32,
) -> Result<()> {
    // SAFETY: caller passes live device addresses for the decode-prep layout;
    // tail pages allocated, stream-ordered on `stream`.
    unsafe {
        ffi::decode_prep_paged_cuda(
            q_ptr as *mut ffi::Half,
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
            attn_i32(num_qo_heads, "qwen decode prep qo_heads")?,
            attn_i32(num_kv_heads, "qwen decode prep kv_heads")?,
            attn_i32(page_size, "qwen decode prep page_size")?,
            attn_i32(stride_page, "qwen decode prep stride_page")?,
            attn_i32(batch_size, "qwen decode prep batch")?,
            rms_eps,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("decode_prep_paged_cuda failed at batch={batch_size}: {e}"))
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

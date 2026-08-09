use super::{CUresult, CUstream, Half};

unsafe extern "C" {
    pub fn prefill_attention_paged_prep_cuda(
        q_batch: *mut Half,
        k_batch: *mut Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        page_table: *const i32,
        page_table_offset_ptr: *const i32,
        page_size: i32,
        k_pool: *mut Half,
        v_pool: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        start_pos_ptr: *const i32,
        rms_eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn prefill_attention_hd256_prep_cuda(
        q_full_batch: *const Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        q_batch_out: *mut Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        start_pos_ptr: *const i32,
        rotary_dim: i32,
        rms_eps: f32,
        max_seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Sliding-window ring variant of [`prefill_attention_hd256_prep_cuda`]:
    /// the K/V cache write row wraps as `pos % ring_modulus` (== the per-head
    /// stride == window+block). `start_pos_ptr` is ABSOLUTE (RoPE reads it
    /// unshifted). One launch must write `<= ring_modulus` rows.
    pub fn prefill_attention_hd256_prep_ring_cuda(
        q_full_batch: *const Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        q_batch_out: *mut Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        start_pos_ptr: *const i32,
        rotary_dim: i32,
        rms_eps: f32,
        ring_modulus: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn attention_gate_batch_hd256_cuda(
        q_full_batch: *const Half,
        attn_out: *mut Half,
        num_q_heads: i32,
        head_dim: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn fused_gqa_attention_decode_batched(
        q_batch: *const Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        positions: *const i32,
        seq_lens: *const i32,
        k_cache_ptrs: *const *const Half,
        v_cache_ptrs: *const *const Half,
        partial_out: *mut f32,
        partial_m: *mut f32,
        partial_l: *mut f32,
        num_qheads: i32,
        num_kvheads: i32,
        gqa_ratio: i32,
        head_dim: i32,
        rotary_dim: i32,
        max_seq_len: i32,
        batch_size: i32,
        rms_eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn attention_decode_reduce_batched(
        partial_out: *const f32,
        partial_m: *const f32,
        partial_l: *const f32,
        output: *mut Half,
        num_qheads: i32,
        head_dim: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn nonpaged_prefill_attention_cuda(
        q: *const Half,
        k_cache: *const Half,
        v_cache: *const Half,
        out: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Sliding-window ring variant of [`nonpaged_prefill_attention_cuda`]:
    /// physical row = `(ring_base + logical) % ring_modulus` (== per-head stride
    /// == window+block). Pass the K/V base at the head-0 origin (no pre-offset),
    /// `ring_base` = absolute position of logical key 0 (= lo), `seq_len` = 1.
    pub fn nonpaged_prefill_attention_ring_cuda(
        q: *const Half,
        k_cache: *const Half,
        v_cache: *const Half,
        out: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        kv_len: i32,
        ring_base: i32,
        ring_modulus: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Ragged-window ring variant of [`nonpaged_prefill_attention_ring_cuda`]:
    /// one launch for `seq_len` rows with per-row key ranges. `ring_base_dev[t]`
    /// / `kv_len_dev[t]` (device i32, `seq_len` each) are row t's window
    /// `[base, base+len)`, walked non-causally. Ranges are device-resident, so
    /// the caller must guarantee `kv_len_dev[t] <= ring_modulus`.
    pub fn nonpaged_prefill_attention_ring_varlen_cuda(
        q: *const Half,
        k_cache: *const Half,
        v_cache: *const Half,
        out: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        ring_base_dev: *const i32,
        kv_len_dev: *const i32,
        ring_modulus: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Slot-batched [`nonpaged_prefill_attention_ring_varlen_cuda`]: one launch
    /// for every draft slot. `k_slots` / `v_slots` are device arrays of `slots`
    /// ring-cache base pointers (one per slot, separately allocated), and the
    /// window table is slot-major — `ring_base_dev` / `kv_len_dev` are each
    /// `slots * seq_len` i32, row `s * seq_len + t`.
    pub fn nonpaged_prefill_attention_ring_varlen_batched_cuda(
        q: *const Half,
        k_slots: *const *const std::ffi::c_void,
        v_slots: *const *const std::ffi::c_void,
        out: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        slots: i32,
        ring_base_dev: *const i32,
        kv_len_dev: *const i32,
        ring_modulus: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Device-position variant of [`nonpaged_prefill_attention_cuda`]:
    /// `start_pos_dev` is one device-resident i32 (kv_len = *start_pos_dev +
    /// token + 1 inside the kernel), making the launch CUDA-graph
    /// capture-safe — the Qwen3.5/3.6 decode graph stages the position
    /// pre-replay instead of baking a host scalar.
    pub fn nonpaged_prefill_attention_devpos_cuda(
        q: *const Half,
        k_cache: *const Half,
        v_cache: *const Half,
        out: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        start_pos_dev: *const i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Context-parallel ring attention: fuse one q-tile × one KV block into the
    /// running flash-2 (M, L, O) accumulator on-device (no full-seq buffer, no
    /// readback between ring steps). Functional: reads `*_in`, writes `*_out`
    /// (`[num_q_tiles, q_rows]` M/L, `[num_q_tiles, q_rows, head_dim]` O, f32) —
    /// caller inits the first block's `*_in` M to -inf, L/O to 0. Absolute causal
    /// mask via per-row `q_pos`/`k_pos` (f32 device arrays, exact for seq < 2^24):
    /// q row r attends k col c iff `k_pos[c] <= q_pos[r]` — so a zigzag shard's
    /// non-contiguous rows/cols mask correctly. Contiguous is `pos = base..base+n`.
    pub fn ring_block_attention_fwd_merge_cuda(
        q: *const Half,
        k_blk: *const Half,
        v_blk: *const Half,
        acc_m_in: *const f32,
        acc_l_in: *const f32,
        acc_o_in: *const f32,
        acc_m_out: *mut f32,
        acc_l_out: *mut f32,
        acc_o_out: *mut f32,
        q_pos: *const f32,
        k_pos: *const f32,
        num_q_tiles: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        q_rows: i32,
        blk_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Finalize the ring accumulator after all blocks: `out = O / L` (f32) and
    /// `lse = M + ln(L)` (f32, one per row). `total_rows = num_q_tiles * q_rows`.
    /// `out` is f32 — the ring output stays on the f32 autograd tape.
    pub fn ring_block_attention_finalize_cuda(
        acc_m: *const f32,
        acc_l: *const f32,
        acc_o: *const f32,
        out: *mut f32,
        lse: *mut f32,
        total_rows: i32,
        head_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Per-block ring-attention backward (flash-2 adjoint): reconstruct
    /// `P = exp(S·scale − lse)` from the saved final `lse`, accumulate `grad_q`
    /// in place (row-unique, sequential-safe) and `grad_k_blk`/`grad_v_blk` via
    /// atomicAdd (GQA q-heads + q-rows fan into one kv row). f32 grad buffers.
    pub fn ring_block_attention_bwd_cuda(
        q: *const Half,
        k_blk: *const Half,
        v_blk: *const Half,
        out: *const f32,
        lse: *const f32,
        d_out: *const Half,
        grad_q: *mut f32,
        grad_k_blk: *mut f32,
        grad_v_blk: *mut f32,
        q_pos: *const f32,
        k_pos: *const f32,
        num_q_tiles: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        q_rows: i32,
        blk_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// FA3 pair-route ring merge (`csrc/attention/ring_fa3_merge.cu`): fold one
    /// pair's NORMALIZED FA3 (o, lse) — block stats (m = lse, l = 1) — into the
    /// running (M, L, O) accumulators IN PLACE (caller passes fresh copies).
    /// `acc_*` full head-major `[tiles, seq(, d)]`; `lse_pair`/`o_pair` compact
    /// `[heads, run_len(, d)]`; run rows `[run_start, run_start+run_len)` of
    /// tiles `[tile_base, tile_base+num_heads)`.
    pub fn ring_fa3_merge_pair_cuda(
        acc_m: *mut f32,
        acc_l: *mut f32,
        acc_o: *mut f32,
        lse_pair: *const f32,
        o_pair: *const Half,
        num_heads: i32,
        tile_base: i32,
        seq_len: i32,
        run_start: i32,
        run_len: i32,
        head_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Gather one run's rows of the full `[tiles, seq]` lse into the compact
    /// `[heads, run_len]` layout FA3's bwd expects (it takes no lse stride).
    pub fn ring_fa3_gather_lse_cuda(
        dst: *mut f32,
        lse: *const f32,
        num_heads: i32,
        tile_base: i32,
        seq_len: i32,
        run_start: i32,
        run_len: i32,
        stream: CUstream,
    ) -> CUresult;

    /// `dst[run rows] += bf16(src)`: accumulate a pair's compact
    /// `[heads, run_len, d]` bf16 grad into the full head-major f32 buffer.
    pub fn ring_fa3_accum_grad_bf16_cuda(
        dst: *mut f32,
        src: *const Half,
        num_heads: i32,
        tile_base: i32,
        seq_len: i32,
        run_start: i32,
        run_len: i32,
        head_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Hand-written FA2-style forward attention for sm_70 (V100). BF16 I/O,
    /// FP16 (half2) internal math, FP32 accumulation. Tiled online softmax
    /// (Br=8 Q tokens, Bc=16 KV tiles), causal chunked-prefill semantics.
    /// Drop-in replacement for [`nonpaged_prefill_attention_cuda`] on sm_70
    /// where FA3 (sm_80+) is unavailable. Same strides and layout.
    pub fn arle_fa2_sm70_attention_cuda(
        q: *const Half,
        k_cache: *const Half,
        v_cache: *const Half,
        out: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// DSpark draft dense MLA-latent attention (T4.1). Non-causal: every one of
    /// `block_size` query rows attends the whole `[kv_len]` latent range (draft
    /// context ++ noise block). MLA has NO separate V — K and V are the SAME
    /// compressed latent, shared across all query heads: `latent_kv` row =
    /// `head_dim` = NoPE(`nope_dim`) ++ RoPE(`rope_dim`).
    ///
    /// Math contract:
    ///   for each block row r, head h:
    ///     score[j] = sm_scale * dot(q[r,h,0..head_dim], latent_kv[j,0..head_dim])
    ///               // over the FULL head_dim (NoPE + RoPE); RoPE applied upstream
    ///     w[0..kv_len] = online_softmax(score[0..kv_len])   // non-causal, all keys
    ///     out[r,h,0..head_dim] = Σ_j w[j] * latent_kv[j,0..head_dim]
    ///               // weighted sum of the FULL head_dim latent (Flag #1: the
    ///               // value is the whole latent, feeding mla_oproj as
    ///               // local_heads*head_dim, identical to the main model's local_attn)
    ///
    /// Layout: `q` [block_size, local_heads, head_dim] token-major; `latent_kv`
    /// [kv_len, head_dim] kv-major (one head-shared latent, broadcast over
    /// `local_heads`); `out` [block_size, local_heads, head_dim] token-major.
    /// `sm_scale` is the caller's `1/sqrt(head_dim)`. bf16 in/out. `nope_dim` is
    /// unused (Flag #1: full head_dim value); the value's trailing `rope_dim`
    /// dims are inverse-RoPE'd at `abs_pos = base_start_pos + token` (query
    /// position) with `rope_base`/`original_seq_len`/YaRN `factor`/`beta_*` —
    /// the same params the caller's forward q/latent RoPE used (cr==0, no YaRN).
    pub fn dsv4_dspark_draft_attention_cuda(
        q: *const Half,
        latent_kv: *const Half,
        out: *mut Half,
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
        stream: CUstream,
    ) -> CUresult;

    pub fn decode_prep_paged_cuda(
        q_batch: *mut Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        positions: *const i32,
        k_pool: *mut Half,
        v_pool: *mut Half,
        page_table: *const i32,
        page_indptr: *const i32,
        last_page_len: *const i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        stride_page: i32,
        batch_size: i32,
        rms_eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn paged_kv_append_last_token_indices_cuda(
        kv_indices: *mut i32,
        kv_indptr: *const i32,
        last_token_indices: *const i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn paged_kv_append_new_page_indices_cuda(
        kv_indices: *mut i32,
        prev_kv_indptr: *const i32,
        next_kv_indptr: *const i32,
        append_indptr: *const i32,
        appended_page_indices: *const i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    // Paged HD256 prep/gate helpers for the Qwen3.6 KV-recall read-swap. Removed
    // as "dead" in c7eb88cd (no callers in main at the time); restored here because
    // the read-swap (qwen35.rs full_attention_paged) reintroduces the usage.
    pub fn prefill_attention_paged_prep_hd256_cuda(
        q_full_batch: *const Half,
        q_out_batch: *mut Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        page_table: *const i32,
        page_size: i32,
        k_pool: *mut Half,
        v_pool: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        start_pos_ptr: *const i32,
        rotary_dim: i32,
        rms_eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn decode_prep_paged_hd256_cuda(
        q_full_batch: *const Half,
        q_out_batch: *mut Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        positions: *const i32,
        k_pool: *mut Half,
        v_pool: *mut Half,
        page_table: *const i32,
        page_indptr: *const i32,
        last_page_len: *const i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        stride_page: i32,
        batch_size: i32,
        rotary_dim: i32,
        rms_eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn attention_gate_paged_hd256_cuda(
        q_full_batch: *const Half,
        attn_out: *mut Half,
        num_q_heads: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn decode_attention_int8_workspace_bytes(
        batch_size: i32,
        num_qo_heads: i32,
        head_dim: i32,
        num_splits: i32,
    ) -> usize;

    /// KIVI per-channel K decode attention. `k_static_scales` shape is
    /// `[num_kv_heads, head_dim]` f32 (one scale per channel per KV head,
    /// shared across tokens). `v_scales` keeps per-(row, head) layout
    /// `[max_total_tokens, num_kv_heads]`.
    pub fn decode_attention_fp8_per_channel_k_cuda(
        q: *const Half,
        k_data: *const u8,
        v_data: *const u8,
        k_static_scales: *const f32,
        v_scales: *const f32,
        kv_indices: *const i32,
        kv_indptr: *const i32,
        o: *mut Half,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        sm_scale: f32,
        stream: CUstream,
        workspace: *mut u8,
        workspace_bytes: usize,
    ) -> CUresult;

    /// INT8 KIVI per-channel K decode attention. Mirrors
    /// `decode_attention_fp8_per_channel_k_cuda` but reads INT8 K/V (with
    /// the cp.async-pipelined tiling from `decode_attention_int8_cuda`).
    pub fn decode_attention_int8_per_channel_k_cuda(
        q: *const Half,
        k_data: *const i8,
        v_data: *const i8,
        k_static_scales: *const f32,
        v_scales: *const f32,
        kv_indices: *const i32,
        kv_indptr: *const i32,
        o: *mut Half,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        sm_scale: f32,
        stream: CUstream,
        workspace: *mut u8,
        workspace_bytes: usize,
    ) -> CUresult;

    /// INT4 KIVI two-level K decode attention. K dequant uses
    /// `static[kv_head, dim] * dynamic[row, kv_head]` (per-channel × per-
    /// (token, kv_head)). V uses per-(row, kv_head) scale.
    pub fn decode_attention_int4_per_channel_k_cuda(
        q: *const Half,
        k_data_packed: *const u8,
        v_data_packed: *const u8,
        k_static_scales: *const f32,
        k_dynamic_scales: *const f32,
        v_scales: *const f32,
        kv_indices: *const i32,
        kv_indptr: *const i32,
        o: *mut Half,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        sm_scale: f32,
        stream: CUstream,
        workspace: *mut u8,
        workspace_bytes: usize,
    ) -> CUresult;

    /// Variable-length Q + paged FP8 E4M3 KV attention.
    ///
    /// Mirrors the TileLang TC decode shape but reads FP8 KV directly (no bf16
    /// shadow). Used by the mixed prefill+decode path when KV format is FP8.
    /// HD128 + page_size=16 only for now.
    ///
    /// Q packing: `[total_q_tokens, num_q_heads * HEAD_DIM]` in bf16, where
    /// `total_q_tokens = qo_indptr[batch_size]`. Output has the same shape.
    /// `causal=true` enables the causal mask for prefill rows
    /// (qlen > 1); decode rows (qlen=1) ignore the mask.
    pub fn decode_attention_varlen_fp8_workspace_bytes(
        total_q_tokens: i32,
        num_q_heads: i32,
        head_dim: i32,
        num_splits: i32,
    ) -> usize;

    pub fn decode_attention_varlen_fp8_cuda(
        q_packed: *const Half,
        qo_indptr: *const i32,
        k_pool: *const u8, // FP8 E4M3
        v_pool: *const u8, // FP8 E4M3
        k_scales: *const f32,
        v_scales: *const f32,
        kv_indptr: *const i32,
        kv_indices: *const i32,
        last_page_len: *const i32,
        output: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        batch_size: i32,
        total_q_tokens: i32,
        max_kv_len: i32,
        int8_kv: bool,
        causal: bool,
        sm_scale: f32,
        stream: CUstream,
        workspace: *mut u8,
        workspace_bytes: usize,
    ) -> CUresult;

}

// TileLang AOT paged-attention FFI — generated from `crates/cuda-kernels/kernels.toml`.
// build.rs emits OUT_DIR/ffi_tilelang_generated.rs: the 25 `unsafe extern "C"`
// paged-attn symbols (prefill/decode/split-partial/split-merge/fp8), their
// fn-pointer type aliases, `AttnPhase`, and `resolve_*()` dispatch tables. This
// replaces the hand-written `tilelang_*_decl!` macros + per-config invocations.
// `CUresult`, `CUstream`, `Half` are in scope from the `use super::{...}` above.
include!(concat!(env!("OUT_DIR"), "/ffi_tilelang_generated.rs"));

// DSv4-Flash (MODEL1) FP8 KV pack.
//
// Packs ARLE's bf16 DSv4 KV (NoPE 448 + RoPE 64) into the MODEL1 FP8
// block-paged layout consumed by upstream FlashMLA's
// `sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`.
// 584 bytes/token per the upstream contract (see
// `crates/cuda-kernels/csrc/attention/dsv4_fp8_kv_pack.cu` for the byte
// layout + e8m0 scale encoding).
//
// Phase D-3' of the FlashMLA decode integration plan. Sibling FFI
// for the kernel-side decode dispatch lives in `ffi/misc.rs` next to
// `arle_flashmla_sm90_sparse_decode_fwd`; runtime wire-up is a separate
// downstream item.
unsafe extern "C" {
    /// Pack `n_tokens` worth of (NoPE bf16, RoPE bf16) into the MODEL1 FP8
    /// block-paged layout. `page_block_size` is the upstream
    /// `page_block_size` (64 for DSv4-Flash). `token_block_id[i]` is the
    /// destination block for token `i`; `token_in_block_row[i]` is the
    /// 0..page_block_size-1 row within that block.
    pub fn arle_dsv4_fp8_kv_pack_cuda(
        nope: *const Half,
        rope: *const Half,
        packed_kv: *mut u8,
        token_block_id: *const i32,
        token_in_block_row: *const i32,
        n_tokens: i32,
        page_block_size: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Strided variant — same packing contract as `arle_dsv4_fp8_kv_pack_cuda`
    /// but the NoPE and RoPE buffers carry an explicit per-token element
    /// stride. Used by the Phase D-4 decode hooks to feed
    /// `k_prepared`-shaped `[n_tokens, head_dim=512]` interleaved input
    /// without an intermediate deinterleave: caller passes
    ///   `nope = k_prepared,           stride_nope_elems = 512`
    ///   `rope = k_prepared + 448,     stride_rope_elems = 512`
    /// Strides must be ≥ HEAD_DIM_NOPE (448) / HEAD_DIM_ROPE (64) respectively.
    /// See Finding 1 in `docs/experience/wins/2026-05-28-dsv4-flashmla-decode-d4-plumbing.md`.
    ///
    /// `page_table`/`num_logical_pages` are the OPTIONAL Stage-B device page-table
    /// lookup (`null`/0 = Stage-A band path, unchanged). When set, `token_block_id[t]`
    /// is a slot-LOGICAL page routed through `page_table[logical]`; an identity table
    /// is byte-identical to the band path.
    pub fn arle_dsv4_fp8_kv_pack_strided_cuda(
        nope: *const Half,
        rope: *const Half,
        packed_kv: *mut u8,
        token_block_id: *const i32,
        token_in_block_row: *const i32,
        n_tokens: i32,
        page_block_size: i32,
        stride_nope_elems: i32,
        stride_rope_elems: i32,
        page_table: *const i32,
        num_logical_pages: i32,
        stream: CUstream,
    ) -> CUresult;

    /// V32 (512 NoPE / 656 B/tok) variant of `arle_dsv4_fp8_kv_pack_strided_cuda`
    /// for GLM-5.2. Identical signature, DIFFERENT inline layout (NOT the MODEL1
    /// trailing-e8m0 format): per token `[512 NoPE fp8][4 F32 scales @512][128
    /// rope bf16]`, stride 656. Each F32 scale covers one 128-elem NoPE block
    /// (= amax/448, NO power-of-two rounding) — matches the vendored V32 decode
    /// (`config.h` NUM_SCALES=4, QUANT_TILE_SIZE=128). Caller passes
    /// `nope = k_prepared, rope = k_prepared + 512,
    /// stride_nope_elems = stride_rope_elems = head_dim = 576`. Strides must be
    /// ≥ 512 (NoPE) / 64 (RoPE).
    pub fn arle_dsv4_v32_fp8_kv_pack_strided_cuda(
        nope: *const Half,
        rope: *const Half,
        packed_kv: *mut u8,
        token_block_id: *const i32,
        token_in_block_row: *const i32,
        n_tokens: i32,
        page_block_size: i32,
        stride_nope_elems: i32,
        stride_rope_elems: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Fill one `[block_id,row]` pair for FlashMLA decode's SW FP8 pack from
    /// a device-resident `start_pos` scalar. The following
    /// `arle_dsv4_fp8_kv_pack_strided_cuda(..., n_tokens=1, ...)` consumes
    /// these scratch values.
    pub fn arle_dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_cuda(
        token_block_id: *mut i32,
        token_in_block_row: *mut i32,
        start_pos: *const i32,
        sliding_window: i32,
        page_block_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn arle_dsv4_fp8_kv_fill_sw_slots_from_start_pos_cuda(
        token_block_id: *mut i32,
        token_in_block_row: *mut i32,
        start_pos: *const i32,
        slot_layer_block_offsets: *const i32,
        n_tokens: i32,
        sliding_window: i32,
        page_block_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn arle_dsv4_fp8_kv_pack_completed_compressor_row_start_pos_cuda(
        compressed: *const Half,
        packed_kv: *mut u8,
        start_pos: *const i32,
        ratio: i32,
        sw_blocks: i32,
        page_block_size: i32,
        stride_elems: i32,
        page_table: *const i32,
        num_logical_pages: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Batched (b=N) MODEL1 SW one-token pack: ONE launch over `n` decode rows,
    /// each writing its CURRENT token into its own SW ring slot. `nope_arr` /
    /// `rope_arr` are N device pointers (each = that row's `k_prepared` NoPE /
    /// RoPE base); `page_table_arr` is N per-slot device page-table pointers
    /// (each `num_logical_pages` long, `null` = Stage-A band); `packed_kv` is the
    /// single shared pool base. n=1 is byte-identical to the per-row fill +
    /// `arle_dsv4_fp8_kv_pack_strided_cuda(n_tokens=1)`.
    pub fn arle_dsv4_fp8_kv_pack_strided_batched_cuda(
        nope_arr: *const *const Half,
        rope_arr: *const *const Half,
        packed_kv: *mut u8,
        start_pos: *const i32,
        n: i32,
        page_block_size: i32,
        sliding_window: i32,
        stride_nope_elems: i32,
        stride_rope_elems: i32,
        page_table_arr: *const *const i32,
        num_logical_pages: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Batched (b=N) MODEL1 compressed-delta pack: ONE launch over `n` decode
    /// rows. `compressed_arr[row]` is that row's compressor `compressed` base
    /// (`null` for rows with no compressor → no-op); each row early-outs on
    /// `(pos+1)%ratio != 0`. `page_table_arr` is N per-slot device page-table
    /// pointers. n=1 is byte-identical to
    /// `arle_dsv4_fp8_kv_pack_completed_compressor_row_start_pos_cuda`.
    pub fn arle_dsv4_fp8_kv_pack_completed_compressor_row_batched_cuda(
        compressed_arr: *const *const Half,
        packed_kv: *mut u8,
        start_pos: *const i32,
        n: i32,
        ratio: i32,
        sw_blocks: i32,
        page_block_size: i32,
        stride_elems: i32,
        page_table_arr: *const *const i32,
        num_logical_pages: i32,
        stream: CUstream,
    ) -> CUresult;
}

// DSv4 FlashMLA sparse-decode indices builder (block-paged coords).
//
// Builds the unified per-decode-token indices buffer (s_q=1) in the
// block-paged coord space of the FP8 KV pool (Phase D-3').
//
// Sibling kernel of the prefill-side `arle_flashmla_csa_build_indices` /
// `arle_flashmla_hca_build_indices`; mode_int selects between them.
// See `csrc/attention/dsv4_flashmla_decode_build_indices.cu` for the
// row-segment layout (SW slots | compressed selections | -1 padding).
//
// Phase D-4 step 1 of the FlashMLA decode integration.
unsafe extern "C" {
    /// Build the unified decode indices row (`s_q=1`).
    ///
    /// - `indices`: out, `int32 [topk_unified]` where
    ///   `topk_unified = sliding_window + max_compressed_keys` (must be %128 == 0).
    /// - `selected`: `int32 [max_compressed_keys]` for CSA (mode_int=1),
    ///   nullptr for HCA (mode_int=2).
    /// - `sw_blocks`: SW sub-pool block count
    ///   (`ceil(sliding_window / page_block_size)`).
    /// - `start_pos`: absolute position of the decode token.
    /// - `max_compressed_keys`: `index_topk` (CSA) or padded
    ///   `compressed_count` (HCA).
    /// - `compress_ratio`: causality-gate ratio for compressed selections.
    /// - `mode_int`: 1 = CSA, 2 = HCA.
    /// - `page_block_size`: 64 for DSv4-Flash MODEL1.
    /// - `page_table`/`num_logical_pages`: OPTIONAL Stage-B logical→physical page
    ///   table (`null`/0 = Stage-A slot-relative path, unchanged). When set, each
    ///   emitted index's logical page is routed to physical, yielding POOL-absolute
    ///   indices (an identity table reproduces the Stage-A index for-for-index).
    pub fn arle_dsv4_flashmla_decode_build_indices_cuda(
        indices: *mut i32,
        selected: *const i32,
        sw_blocks: i32,
        sliding_window: i32,
        start_pos: i32,
        max_compressed_keys: i32,
        compress_ratio: i32,
        mode_int: i32,
        page_block_size: i32,
        page_table: *const i32,
        num_logical_pages: i32,
        total_blocks: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Build the unified decode indices row using a device-resident
    /// `start_pos` scalar. This keeps the kernel replay-safe for CUDA graph
    /// decode once ARLE stamps per-step scalars into graph-visible device
    /// metadata.
    pub fn arle_dsv4_flashmla_decode_build_indices_start_pos_ptr_cuda(
        indices: *mut i32,
        selected: *const i32,
        sw_blocks: i32,
        sliding_window: i32,
        start_pos: *const i32,
        max_compressed_keys: i32,
        compress_ratio: i32,
        mode_int: i32,
        page_block_size: i32,
        page_table: *const i32,
        num_logical_pages: i32,
        total_blocks: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Batched `b = N` indices builder for DSv4 FlashMLA sparse decode.
    ///
    /// `indices` is `[b, topk_unified]`, `start_pos` is `[b]`,
    /// `slot_layer_block_offsets` is `[b]` in blocks from the shared KV arena
    /// base, `selected` is `[b, max_compressed_keys]` for CSA or null for HCA,
    /// and `topk_length` is `[b]`.
    ///
    /// `page_table`/`num_logical_pages`: OPTIONAL Stage-B per-row logical→physical
    /// page table (`[b, num_logical_pages]`; `null`/0 = Stage-A band path,
    /// unchanged). When set the routed indices are POOL-absolute, so the
    /// `slot_layer_block_offsets` band shift is SKIPPED (an identity per-row table
    /// = the Stage-A index after the band shift).
    pub fn arle_dsv4_flashmla_decode_build_indices_batched_cuda(
        indices: *mut i32,
        start_pos: *const i32,
        slot_layer_block_offsets: *const i32,
        selected: *const i32,
        topk_length: *mut i32,
        b: i32,
        sw_blocks: i32,
        sliding_window: i32,
        max_compressed_keys: i32,
        compress_ratio: i32,
        mode_int: i32,
        page_block_size: i32,
        total_blocks: i32,
        page_table: *const i32,
        num_logical_pages: i32,
        stream: CUstream,
    ) -> CUresult;
}

// FA3 hopper fwd + bwd shim (hdim256/bf16/sm_90a) — vendored Dao-AILab
// flash-attention @ fc8cbad6, torch-free C ABI in
// `csrc/attention/arle_fa3_shim.cu`. Built whenever the target includes sm_90;
// without it the stub returns `cudaErrorNotSupported` and the marker returns
// 0 (assert it returns 1 before enabling the runtime path — flashmla stub
// lesson).

/// Mirror of `ArleFa3FwdHd256Args` in `csrc/attention/arle_fa3_shim.cu`.
/// All strides in elements; the last dim (head_dim) must be contiguous. The
/// separate row/head strides express both token-major q/o (`[S, h, d]`:
/// row = h*d, head = d) and the qwen35 head-major slot caches
/// (`[h_k, max_seq, d]`: row = d, head = max_seq*d) without relayout.
/// ONE call per layer whatever the batch: q/o are packed `[total_q, h, d]`
/// addressed through `cu_seqlens_q`, and each row's KV extent comes from
/// `seqused_k` against a rectangular page table strided by
/// `page_table_batch_stride`. `num_splits > 1` opts into the upstream split-KV
/// + PackGQA + combine path.
#[repr(C)]
pub struct ArleFa3FwdHd256Args {
    pub q: *const Half,
    pub k: *const Half,
    pub v: *const Half,
    pub o: *mut Half,
    /// fp32 scratch, `num_heads * total_q` elements.
    pub softmax_lse: *mut f32,
    /// fp32 split scratch, `num_splits * num_heads * total_q * head_dim`
    /// elements; null when `num_splits <= 1`.
    pub out_accum: *mut f32,
    /// fp32 split LSE scratch, `num_splits * num_heads * total_q` elements;
    /// null when `num_splits <= 1`.
    pub softmax_lse_accum: *mut f32,
    /// Device i32 scratch for the scheduler metadata + semaphore. Size it
    /// `round_up(batch, 4) * 4 + 1`.
    pub tile_count_semaphore: *mut i32,
    /// Elements available at `tile_count_semaphore`.
    pub metadata_capacity: i32,
    /// device i32 `[batch + 1]` prefix sum over query rows.
    pub cu_seqlens_q: *const i32,
    /// device i32 `[batch]` per-row KV extent in tokens.
    pub seqused_k: *const i32,
    pub batch: i32,
    /// `cu_seqlens_q[batch]`.
    pub total_q: i32,
    /// Longest row's query length.
    pub seqlen_q: i32,
    /// Longest row's KV length.
    pub seqlen_k: i32,
    pub num_heads: i32,
    pub num_heads_k: i32,
    /// Must be 256.
    pub head_dim: i32,
    pub q_row_stride: i64,
    pub k_row_stride: i64,
    pub v_row_stride: i64,
    pub o_row_stride: i64,
    pub q_head_stride: i64,
    pub k_head_stride: i64,
    pub v_head_stride: i64,
    pub o_head_stride: i64,
    pub softmax_scale: f32,
    pub is_causal: i32,
    /// 1 = direct fwd; >1 = split-KV decode fwd + combine (max 256).
    pub num_splits: i32,
    /// Rectangular page table `[batch, page_table_batch_stride]`; null =
    /// contiguous KV. When set, `k`/`v` are the pool base and the strides
    /// describe one page, which expresses the HND pool
    /// `[page, h_k, page_size, d]` without a relayout.
    pub page_table: *const i32,
    pub page_table_batch_stride: i64,
    pub page_size: i32,
    /// Pages in the POOL — the extent of the page dimension.
    pub num_pages: i32,
    /// FA3's 4th K/V dim is the page dim when paged, so this is its stride.
    pub k_page_stride: i64,
    pub v_page_stride: i64,
}

/// Mirror of `ArleFa3BwdHd256Args` in `csrc/attention/arle_fa3_shim.cu`.
/// Backward substrate: NON-varlen, NON-paged, batch=1 contiguous `[S, h, d]`
/// bf16 views; no dropout/softcap, deterministic=false. All scratch is
/// caller-provided; sizes use the hdim256 sm90 bwd tiles kBlockM=64 /
/// kBlockN=80: `sq_r = round_up(seqlen_q, 64)`, `sk_r = round_up(seqlen_k,
/// 80)`. dq_accum/dq_semaphore are zeroed by the bwd preprocess kernel;
/// dk/dv_accum are memset to zero by the shim (upstream at::zeros semantics).
#[repr(C)]
pub struct ArleFa3BwdHd256Args {
    pub q: *const Half,
    pub k: *const Half,
    pub v: *const Half,
    /// Forward output, needed to recompute dP_sum.
    pub o: *const Half,
    pub dout: *const Half,
    /// fp32 `num_heads * seqlen_q` from the forward.
    pub softmax_lse: *const f32,
    pub dq: *mut Half,
    pub dk: *mut Half,
    pub dv: *mut Half,
    /// fp32 scratch, `num_heads * sq_r` elements.
    pub softmax_d: *mut f32,
    /// fp32 scratch, `num_heads * sq_r` elements.
    pub softmax_lse_log2: *mut f32,
    /// fp32 scratch, `num_heads * sq_r * 256` elements.
    pub dq_accum: *mut f32,
    /// fp32 scratch, `num_heads_k * sk_r * 256` elements; GQA only, else null.
    pub dk_accum: *mut f32,
    /// fp32 scratch, `num_heads_k * sk_r * 256` elements; GQA only, else null.
    pub dv_accum: *mut f32,
    /// i32 scratch, `ceil(seqlen_q / 64) * num_heads` elements.
    pub dq_semaphore: *mut i32,
    pub softmax_d_capacity: i64,
    pub softmax_lse_log2_capacity: i64,
    pub dq_accum_capacity: i64,
    pub dk_accum_capacity: i64,
    pub dv_accum_capacity: i64,
    pub dq_semaphore_capacity: i64,
    pub seqlen_q: i32,
    pub seqlen_k: i32,
    pub num_heads: i32,
    pub num_heads_k: i32,
    /// Must be 256.
    pub head_dim: i32,
    pub q_row_stride: i64,
    pub k_row_stride: i64,
    pub v_row_stride: i64,
    pub o_row_stride: i64,
    pub do_row_stride: i64,
    pub dq_row_stride: i64,
    pub dk_row_stride: i64,
    pub dv_row_stride: i64,
    pub q_head_stride: i64,
    pub k_head_stride: i64,
    pub v_head_stride: i64,
    pub o_head_stride: i64,
    pub do_head_stride: i64,
    pub dq_head_stride: i64,
    pub dk_head_stride: i64,
    pub dv_head_stride: i64,
    pub softmax_scale: f32,
    pub is_causal: i32,
}

unsafe extern "C" {
    pub fn arle_fa3_fwd_hd256_bf16_cuda(
        args: *const ArleFa3FwdHd256Args,
        stream: CUstream,
    ) -> CUresult;

    pub fn arle_fa3_bwd_hd256_bf16_cuda(
        args: *const ArleFa3BwdHd256Args,
        stream: CUstream,
    ) -> CUresult;

    /// 1 = real FA3 shim linked; 0 = stub build.
    pub fn arle_fa3_real_kernel_marker_cuda() -> i32;
}

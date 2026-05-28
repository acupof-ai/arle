use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub fn prefill_attention_prep_cuda(
        q_batch: *mut Half,
        k_batch: *mut Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        start_pos: i32,
        max_seq_len: i32,
        rms_eps: f32,
        stream: CUstream,
    ) -> CUresult;

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
        seq_len: i32,
        start_pos_ptr: *const i32,
        rotary_dim: i32,
        rms_eps: f32,
        max_seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

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

    pub fn attention_gate_batch_hd256_cuda(
        q_full_batch: *const Half,
        attn_out: *mut Half,
        num_q_heads: i32,
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

    pub fn fused_gqa_attention_decode(
        q_full: *const Half,
        k_full: *const Half,
        v_full: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache_base: *const Half,
        sin_cache_base: *const Half,
        decode_meta: *const i32,
        k_cache: *mut Half,
        v_cache: *mut Half,
        partial_out: *mut f32,
        partial_m: *mut f32,
        partial_l: *mut f32,
        num_qheads: i32,
        num_kvheads: i32,
        gqa_ratio: i32,
        max_seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn attention_decode_reduce(
        partial_out: *mut f32,
        partial_m: *mut f32,
        partial_l: *mut f32,
        output: *mut Half,
        num_qheads: i32,
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

    /// Fused QKV variant: reads Q/K/V from merged buffer, writes RoPE'd Q to
    /// separate output. Eliminates the split_qkv kernel launch.
    pub fn decode_prep_paged_fused_qkv_cuda(
        qkv_batch: *const Half,
        q_out: *mut Half,
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
        qkv_stride: i32,
        q_dim: i32,
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
    /// See docs/plans/2026-05-27-int8-kv-kivi-per-channel.md.
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

// One AOT-specialized symbol per (num_q_heads, num_kv_heads). The matching
// Rust dispatch table lives in `infer/src/ops/attention.rs`. Adding a new
// HD128 head config requires extending all three:
//   - SUPPORTED_HEADS in tools/tilelang/batch_prefill_paged_hd128.py
//   - TILELANG_PREFILL_HD128_HEAD_CONFIGS in cuda-kernels/build.rs
//   - the macro invocation below + the dispatch arm in attention.rs
macro_rules! tilelang_prefill_hd128_decl {
    ($($name:ident),+ $(,)?) => {
        unsafe extern "C" {
            $(
                #[allow(dead_code)]
                pub fn $name(
                    q: *mut Half,
                    q_indptr: *const i32,
                    k_pool: *mut Half,
                    v_pool: *mut Half,
                    kv_indptr: *const i32,
                    kv_indices: *const i32,
                    kv_last_page_len: *const i32,
                    o: *mut Half,
                    batch_size: i32,
                    total_q_tokens: i32,
                    max_qlen: i32,
                    num_pages: i32,
                    total_pages: i32,
                    num_q_heads: i32,
                    num_kv_heads: i32,
                    page_size: i32,
                    sm_scale: f32,
                    stream: CUstream,
                ) -> CUresult;
            )+
        }
    };
}

tilelang_prefill_hd128_decl!(
    tilelang_batch_prefill_paged_hd128_q16_kv8_run_cuda,
    tilelang_batch_prefill_paged_hd128_q32_kv8_run_cuda,
    tilelang_batch_prefill_paged_hd128_q40_kv8_run_cuda,
    tilelang_batch_prefill_paged_hd128_q64_kv8_run_cuda,
);

// HD256 prefill — same FFI shape as HD128 (the kernels share the wrapper
// fill rules in tools/tilelang/gen_tilelang_aot.py); only the cubin's baked
// `head_dim` differs. Adding a new Qwen3.5 head config requires extending
// all three:
//   - SUPPORTED_HEADS in tools/tilelang/batch_prefill_paged_hd256.py
//   - TILELANG_PREFILL_HD256_HEAD_CONFIGS in cuda-kernels/build.rs
//   - the macro invocation below + the dispatch arm in attention.rs
macro_rules! tilelang_prefill_hd256_decl {
    ($($name:ident),+ $(,)?) => {
        unsafe extern "C" {
            $(
                #[allow(dead_code)]
                pub fn $name(
                    q: *mut Half,
                    q_indptr: *const i32,
                    k_pool: *mut Half,
                    v_pool: *mut Half,
                    kv_indptr: *const i32,
                    kv_indices: *const i32,
                    kv_last_page_len: *const i32,
                    o: *mut Half,
                    batch_size: i32,
                    total_q_tokens: i32,
                    max_qlen: i32,
                    num_pages: i32,
                    total_pages: i32,
                    num_q_heads: i32,
                    num_kv_heads: i32,
                    page_size: i32,
                    sm_scale: f32,
                    stream: CUstream,
                ) -> CUresult;
            )+
        }
    };
}

tilelang_prefill_hd256_decl!(
    tilelang_batch_prefill_paged_hd256_q8_kv2_run_cuda,
    tilelang_batch_prefill_paged_hd256_q16_kv2_run_cuda,
    tilelang_batch_prefill_paged_hd256_q16_kv4_run_cuda,
);

// HD256 decode — same FFI shape as the HD256/HD128 prefill macros (the
// kernels share `gen_tilelang_aot.py`'s wrapper fill rules); only the cubin
// internals differ (qlen=1 grid, no causal mask). A separate macro keeps
// symbol scoping clean. Adding a new Qwen3.5 head config requires extending
// all three:
//   - SUPPORTED_HEADS in tools/tilelang/batch_decode_paged_hd256.py
//   - TILELANG_DECODE_HD256_HEAD_CONFIGS in cuda-kernels/build.rs
//   - the macro invocation below + the dispatch arm in attention.rs
macro_rules! tilelang_decode_hd256_decl {
    ($($name:ident),+ $(,)?) => {
        unsafe extern "C" {
            $(
                #[allow(dead_code)]
                pub fn $name(
                    q: *mut Half,
                    q_indptr: *const i32,
                    k_pool: *mut Half,
                    v_pool: *mut Half,
                    kv_indptr: *const i32,
                    kv_indices: *const i32,
                    kv_last_page_len: *const i32,
                    o: *mut Half,
                    batch_size: i32,
                    total_q_tokens: i32,
                    max_qlen: i32,
                    num_pages: i32,
                    total_pages: i32,
                    num_q_heads: i32,
                    num_kv_heads: i32,
                    page_size: i32,
                    sm_scale: f32,
                    stream: CUstream,
                ) -> CUresult;
            )+
        }
    };
}

tilelang_decode_hd256_decl!(
    tilelang_batch_decode_paged_hd256_q8_kv2_run_cuda,
    tilelang_batch_decode_paged_hd256_q16_kv2_run_cuda,
    tilelang_batch_decode_paged_hd256_q16_kv4_run_cuda,
);

// HD128 decode — same FFI shape as the HD256 decode macro above; the
// kernels share `gen_tilelang_aot.py`'s wrapper fill rules. Adding a new
// HD128 head config requires extending all three:
//   - SUPPORTED_HEADS in tools/tilelang/batch_decode_paged_hd128.py
//   - TILELANG_DECODE_HD128_HEAD_CONFIGS in cuda-kernels/build.rs
//   - the macro invocation below + the dispatch arm in attention.rs
macro_rules! tilelang_decode_hd128_decl {
    ($($name:ident),+ $(,)?) => {
        unsafe extern "C" {
            $(
                #[allow(dead_code)]
                pub fn $name(
                    q: *mut Half,
                    q_indptr: *const i32,
                    k_pool: *mut Half,
                    v_pool: *mut Half,
                    kv_indptr: *const i32,
                    kv_indices: *const i32,
                    kv_last_page_len: *const i32,
                    o: *mut Half,
                    batch_size: i32,
                    total_q_tokens: i32,
                    max_qlen: i32,
                    num_pages: i32,
                    total_pages: i32,
                    num_q_heads: i32,
                    num_kv_heads: i32,
                    page_size: i32,
                    sm_scale: f32,
                    stream: CUstream,
                ) -> CUresult;
            )+
        }
    };
}

tilelang_decode_hd128_decl!(
    tilelang_batch_decode_paged_hd128_q16_kv8_run_cuda,
    tilelang_batch_decode_paged_hd128_q32_kv8_run_cuda,
    tilelang_batch_decode_paged_hd128_q40_kv8_run_cuda,
    tilelang_batch_decode_paged_hd128_q64_kv8_run_cuda,
);

// HD64 prefill — DSV4-mini-class substrate (head_dim=64, single KV head).
// Same FFI shape as HD128 prefill (the kernels share `gen_tilelang_aot.py`'s
// wrapper fill rules); only the cubin's baked `head_dim` differs. Master
// §8.2 P1.0 will wire these symbols into a model. Adding a new HD64 head
// config requires extending all three:
//   - SUPPORTED_HEADS in tools/tilelang/batch_prefill_paged_hd64.py
//   - TILELANG_PREFILL_HD64_HEAD_CONFIGS in cuda-kernels/build.rs
//   - the macro invocation below.
macro_rules! tilelang_prefill_hd64_decl {
    ($($name:ident),+ $(,)?) => {
        unsafe extern "C" {
            $(
                #[allow(dead_code)]
                pub fn $name(
                    q: *mut Half,
                    q_indptr: *const i32,
                    k_pool: *mut Half,
                    v_pool: *mut Half,
                    kv_indptr: *const i32,
                    kv_indices: *const i32,
                    kv_last_page_len: *const i32,
                    o: *mut Half,
                    batch_size: i32,
                    total_q_tokens: i32,
                    max_qlen: i32,
                    num_pages: i32,
                    total_pages: i32,
                    num_q_heads: i32,
                    num_kv_heads: i32,
                    page_size: i32,
                    sm_scale: f32,
                    stream: CUstream,
                ) -> CUresult;
            )+
        }
    };
}

tilelang_prefill_hd64_decl!(tilelang_batch_prefill_paged_hd64_q16_kv1_run_cuda,);

// HD64 decode — DSV4-mini-class substrate. Same FFI shape as the HD128
// decode macro above. Adding a new HD64 head config requires extending
// all three:
//   - SUPPORTED_HEADS in tools/tilelang/batch_decode_paged_hd64.py
//   - TILELANG_DECODE_HD64_HEAD_CONFIGS in cuda-kernels/build.rs
//   - the macro invocation below.
macro_rules! tilelang_decode_hd64_decl {
    ($($name:ident),+ $(,)?) => {
        unsafe extern "C" {
            $(
                #[allow(dead_code)]
                pub fn $name(
                    q: *mut Half,
                    q_indptr: *const i32,
                    k_pool: *mut Half,
                    v_pool: *mut Half,
                    kv_indptr: *const i32,
                    kv_indices: *const i32,
                    kv_last_page_len: *const i32,
                    o: *mut Half,
                    batch_size: i32,
                    total_q_tokens: i32,
                    max_qlen: i32,
                    num_pages: i32,
                    total_pages: i32,
                    num_q_heads: i32,
                    num_kv_heads: i32,
                    page_size: i32,
                    sm_scale: f32,
                    stream: CUstream,
                ) -> CUresult;
            )+
        }
    };
}

tilelang_decode_hd64_decl!(tilelang_batch_decode_paged_hd64_q16_kv1_run_cuda,);

// M_b.1 — HD128 BF16 split-KV decode. Partial and merge phases use separate
// TileLang AOT cubins and an explicit float workspace supplied by
// TileLangWorkspace. Keep symbols in lockstep with build.rs and
// tools/tilelang/batch_decode_paged_hd128.py.
macro_rules! tilelang_decode_hd128_split_partial_decl {
    ($($name:ident),+ $(,)?) => {
        unsafe extern "C" {
            $(
                #[allow(dead_code)]
                pub fn $name(
                    q: *mut Half,
                    q_indptr: *const i32,
                    k_pool: *mut Half,
                    v_pool: *mut Half,
                    kv_indptr: *const i32,
                    kv_indices: *const i32,
                    kv_last_page_len: *const i32,
                    partial_out: *mut f32,
                    partial_m: *mut f32,
                    partial_l: *mut f32,
                    batch_size: i32,
                    total_q_tokens: i32,
                    max_qlen: i32,
                    num_pages: i32,
                    total_pages: i32,
                    num_q_heads: i32,
                    num_kv_heads: i32,
                    page_size: i32,
                    sm_scale: f32,
                    num_splits: i32,
                    stream: CUstream,
                ) -> CUresult;
            )+
        }
    };
}

tilelang_decode_hd128_split_partial_decl!(
    tilelang_batch_decode_paged_hd128_split_partial_q16_kv8_run_cuda,
    tilelang_batch_decode_paged_hd128_split_partial_q32_kv8_run_cuda,
    tilelang_batch_decode_paged_hd128_split_partial_q40_kv8_run_cuda,
    tilelang_batch_decode_paged_hd128_split_partial_q64_kv8_run_cuda,
);

macro_rules! tilelang_decode_hd128_split_merge_decl {
    ($($name:ident),+ $(,)?) => {
        unsafe extern "C" {
            $(
                #[allow(dead_code)]
                pub fn $name(
                    partial_out: *const f32,
                    partial_m: *const f32,
                    partial_l: *const f32,
                    o: *mut Half,
                    batch_size: i32,
                    total_q_tokens: i32,
                    max_qlen: i32,
                    num_pages: i32,
                    total_pages: i32,
                    num_q_heads: i32,
                    num_kv_heads: i32,
                    page_size: i32,
                    sm_scale: f32,
                    num_splits: i32,
                    stream: CUstream,
                ) -> CUresult;
            )+
        }
    };
}

tilelang_decode_hd128_split_merge_decl!(
    tilelang_batch_decode_paged_hd128_split_merge_q16_kv8_run_cuda,
    tilelang_batch_decode_paged_hd128_split_merge_q32_kv8_run_cuda,
    tilelang_batch_decode_paged_hd128_split_merge_q40_kv8_run_cuda,
    tilelang_batch_decode_paged_hd128_split_merge_q64_kv8_run_cuda,
);

// M_b.2 — HD128 FP8 KV decode. Different ABI from the BF16 decl: K/V pools
// come in as `*const u8` (FP8 E4M3 bytes) rather than `*mut Half`, and an
// extra pair of `*const f32` scale pointers feed per-token / per-kv-head
// dequant. Keep the macro / spec / build.rs / `.py` `SUPPORTED_HEADS` lists
// in lockstep — see the BF16 macro comment block above for the contract.
macro_rules! tilelang_decode_hd128_fp8_decl {
    ($($name:ident),+ $(,)?) => {
        unsafe extern "C" {
            $(
                #[allow(dead_code)]
                pub fn $name(
                    q: *mut Half,
                    q_indptr: *const i32,
                    k_pool: *const u8,
                    v_pool: *const u8,
                    k_scales: *const f32,
                    v_scales: *const f32,
                    kv_indptr: *const i32,
                    kv_indices: *const i32,
                    kv_last_page_len: *const i32,
                    o: *mut Half,
                    batch_size: i32,
                    total_q_tokens: i32,
                    max_qlen: i32,
                    num_pages: i32,
                    total_pages: i32,
                    num_q_heads: i32,
                    num_kv_heads: i32,
                    page_size: i32,
                    sm_scale: f32,
                    stream: CUstream,
                ) -> CUresult;
            )+
        }
    };
}

tilelang_decode_hd128_fp8_decl!(tilelang_batch_decode_paged_hd128_fp8_q32_kv8_run_cuda,);

// ============================================================================
// DSv4-Flash (MODEL1) FP8 KV pack.
//
// Packs ARLE's bf16 DSv4 KV (NoPE 448 + RoPE 64) into the MODEL1 FP8
// block-paged layout consumed by upstream FlashMLA's
// `sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`.
// 584 bytes/token per the upstream contract (see
// `crates/cuda-kernels/csrc/attention/dsv4_fp8_kv_pack.cu` for the byte
// layout + e8m0 scale encoding).
//
// Phase D-3' of the FlashMLA decode integration plan
// (`docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`). Sibling FFI
// for the kernel-side decode dispatch lives in `ffi/misc.rs` next to
// `arle_flashmla_sm90_sparse_decode_fwd`; runtime wire-up is a separate
// downstream item.
// ============================================================================
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
        stream: CUstream,
    ) -> CUresult;
}

// ============================================================================
// DSv4 FlashMLA sparse-decode indices builder (block-paged coords).
//
// Builds the unified per-decode-token indices buffer (s_q=1) in the
// block-paged coord space of the FP8 KV pool described in
// `docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md` Phase D-3'.
//
// Sibling kernel of the prefill-side `arle_flashmla_csa_build_indices` /
// `arle_flashmla_hca_build_indices`; mode_int selects between them.
// See `csrc/attention/dsv4_flashmla_decode_build_indices.cu` for the
// row-segment layout (SW slots | compressed selections | -1 padding).
//
// Phase D-4 step 1 of the FlashMLA decode integration.
// ============================================================================
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
        stream: CUstream,
    ) -> CUresult;
}

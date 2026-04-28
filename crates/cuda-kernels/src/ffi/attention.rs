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
        page_size: i32,
        k_pool: *mut Half,
        v_pool: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        start_pos: i32,
        rms_eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn flashinfer_single_prefill(
        q: *mut Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        output: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        tmp_buffer: *mut u8,
        stream: CUstream,
    ) -> i32;

    pub fn flashinfer_single_prefill_hd256(
        q: *mut Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        output: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        tmp_buffer: *mut Half,
        stream: CUstream,
    ) -> i32;

    pub fn flash_attention_prefill_hd256_cuda(
        Q: *const Half,
        K_cache: *const Half,
        V_cache: *const Half,
        Output: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        gqa_ratio: i32,
        seq_len: i32,
        start_pos_ptr: *const i32,
        max_seq_len: i32,
        q_dim: i32,
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

    pub fn flashinfer_batch_decode_plan(
        float_workspace: *mut u8,
        float_workspace_bytes: usize,
        int_workspace: *mut u8,
        page_locked_workspace: *mut u8,
        int_workspace_bytes: usize,
        indptr_h: *const i32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        plan_info_out: *mut u8,
        stream: CUstream,
    ) -> i32;

    pub fn flashinfer_batch_decode_run(
        float_workspace: *mut u8,
        int_workspace: *mut u8,
        plan_info: *const u8,
        q: *const Half,
        k_data: *const Half,
        v_data: *const Half,
        kv_indptr: *const i32,
        kv_indices: *const i32,
        kv_last_page_len: *const i32,
        o: *mut Half,
        lse: *mut f32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn flashinfer_append_last_token_indices_cuda(
        kv_indices: *mut i32,
        kv_indptr: *const i32,
        last_token_indices: *const i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn flashinfer_batch_decode_hd256_plan(
        float_workspace: *mut u8,
        float_workspace_bytes: usize,
        int_workspace: *mut u8,
        page_locked_workspace: *mut u8,
        int_workspace_bytes: usize,
        indptr_h: *const i32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        plan_info_out: *mut u8,
        stream: CUstream,
    ) -> i32;

    pub fn flashinfer_batch_decode_hd256_run(
        float_workspace: *mut u8,
        int_workspace: *mut u8,
        plan_info: *const u8,
        q: *const Half,
        k_data: *const Half,
        v_data: *const Half,
        kv_indptr: *const i32,
        kv_indices: *const i32,
        kv_last_page_len: *const i32,
        o: *mut Half,
        lse: *mut f32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

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

    pub fn flashinfer_tc_decode_plan(
        float_workspace: *mut u8,
        float_workspace_bytes: usize,
        int_workspace: *mut u8,
        page_locked_workspace: *mut u8,
        int_workspace_bytes: usize,
        qo_indptr_h: *const i32,
        kv_indptr_h: *const i32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        plan_info_out: *mut u8,
        stream: CUstream,
    ) -> i32;

    pub fn flashinfer_tc_decode_run(
        float_workspace: *mut u8,
        int_workspace: *mut u8,
        plan_info: *const u8,
        q: *const Half,
        q_indptr: *const i32,
        k_data: *const Half,
        v_data: *const Half,
        kv_indptr: *const i32,
        kv_indices: *const i32,
        kv_last_page_len: *const i32,
        o: *mut Half,
        lse: *mut f32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn flashinfer_batch_prefill_paged_hd128_plan(
        float_workspace: *mut u8,
        float_workspace_bytes: usize,
        int_workspace: *mut u8,
        page_locked_workspace: *mut u8,
        int_workspace_bytes: usize,
        qo_indptr_h: *const i32,
        kv_indptr_h: *const i32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        plan_info_out: *mut u8,
        stream: CUstream,
    ) -> i32;

    pub fn flashinfer_batch_prefill_paged_hd128_run(
        float_workspace: *mut u8,
        int_workspace: *mut u8,
        plan_info: *const u8,
        q: *mut Half,
        q_indptr: *const i32,
        k_data: *mut Half,
        v_data: *mut Half,
        kv_indptr: *const i32,
        kv_indices: *const i32,
        kv_last_page_len: *const i32,
        o: *mut Half,
        lse: *mut f32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn flashinfer_batch_prefill_paged_hd256_plan(
        float_workspace: *mut u8,
        float_workspace_bytes: usize,
        int_workspace: *mut u8,
        page_locked_workspace: *mut u8,
        int_workspace_bytes: usize,
        qo_indptr_h: *const i32,
        kv_indptr_h: *const i32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        plan_info_out: *mut u8,
        stream: CUstream,
    ) -> i32;

    pub fn flashinfer_batch_prefill_paged_hd256_run(
        float_workspace: *mut u8,
        int_workspace: *mut u8,
        plan_info: *const u8,
        q: *mut Half,
        q_indptr: *const i32,
        k_data: *mut Half,
        v_data: *mut Half,
        kv_indptr: *const i32,
        kv_indices: *const i32,
        kv_last_page_len: *const i32,
        o: *mut Half,
        lse: *mut f32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        page_size: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn decode_attention_int8_workspace_bytes(
        batch_size: i32,
        num_qo_heads: i32,
        head_dim: i32,
        num_splits: i32,
    ) -> usize;

    pub fn decode_attention_int8_cuda(
        q: *const Half,
        k_data: *const i8,
        v_data: *const i8,
        k_scales: *const f32,
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

    pub fn decode_attention_fp8_cuda(
        q: *const Half,
        k_data: *const u8, // FP8 E4M3
        v_data: *const u8, // FP8 E4M3
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
}

// One AOT-specialized symbol per (num_q_heads, num_kv_heads). The matching
// Rust dispatch table lives in `infer/src/ops/attention.rs`. Adding a new
// Qwen3 head config requires extending all three:
//   - SUPPORTED_HEADS in tools/tilelang/batch_prefill_paged_hd128.py
//   - TILELANG_PREFILL_HD128_HEAD_CONFIGS in cuda-kernels/build.rs
//   - the macro invocation below + the dispatch arm in attention.rs
#[cfg(feature = "tilelang-attn")]
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

#[cfg(feature = "tilelang-attn")]
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
#[cfg(feature = "tilelang-attn")]
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

#[cfg(feature = "tilelang-attn")]
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
#[cfg(feature = "tilelang-decode-hd256")]
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

#[cfg(feature = "tilelang-decode-hd256")]
tilelang_decode_hd256_decl!(
    tilelang_batch_decode_paged_hd256_q8_kv2_run_cuda,
    tilelang_batch_decode_paged_hd256_q16_kv2_run_cuda,
    tilelang_batch_decode_paged_hd256_q16_kv4_run_cuda,
);

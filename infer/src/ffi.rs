use cudarc::driver::sys::{CUresult, CUstream};

// Half type (16-bit float) - same layout as CUDA half
pub(crate) type Half = u16;

// CUDA kernels - all use half precision
unsafe extern "C" {
    pub(crate) fn rms_norm_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn rms_norm_batched_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn add_cuda(
        a: *const Half,
        b: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn fused_add_rms_norm_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn fused_add_rms_norm_batched_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn silu_mul_triton_aot_cuda(
        gate: *const Half,
        up: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn embedding_batched_cuda(
        embed: *const Half,
        token_ids: *const i32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn argmax_cuda(x: *const Half, out: *mut i32, n: i32, stream: CUstream) -> CUresult;

    pub(crate) fn argmax_logprob_cuda(
        x: *const Half,
        out_idx: *mut i32,
        out_logprob: *mut f32,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn argmax_batch_logprob_cuda(
        logits: *const Half,
        token_ids: *mut i32,
        logprobs: *mut f32,
        batch_size: i32,
        vocab_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn argmax_batch_cuda(
        logits: *const Half,
        token_ids: *mut i32,
        batch_size: i32,
        vocab_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gpu_sample_cuda(
        logits: *const Half,
        probs_scratch: *mut f32,
        output: *mut i32,
        vocab_size: i32,
        inv_temperature: f32,
        top_k: i32,
        top_p: f32,
        random_val: f32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gemv_cuda(
        A: *const Half,
        x: *const Half,
        y: *mut Half,
        M: i32,
        K: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gemm_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gemm_graphsafe_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn fused_mlp_cuda(
        x: *const Half,
        gate_proj: *const Half,
        up_proj: *const Half,
        down_proj: *const Half,
        act: *mut Half,
        out: *mut Half,
        hidden_size: i32,
        intermediate_size: i32,
        stream: CUstream,
    ) -> CUresult;

    // Embedding lookup reading token_id from decode_meta[0] (CUDA Graph safe)
    pub(crate) fn embedding_decode_cuda(
        embed: *const Half,
        decode_meta: *const i32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn cublas_init();

    // Prefill attention preparation: QK norm + RoPE + KV cache write (steps 1-2).
    // Step 3 (attention) is handled by flash_attention_prefill_cuda (Triton).
    pub(crate) fn prefill_attention_prep_cuda(
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

    // FlashAttention-2 prefill (Triton AOT): fused QK + softmax + V for all query tokens.
    // Q/Output are col-major [q_dim, seq_len]. K/V cache are per-head [max_seq, HEAD_DIM].
    pub(crate) fn flash_attention_prefill_cuda(
        Q: *const Half,
        K_cache: *const Half,
        V_cache: *const Half,
        Output: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        gqa_ratio: i32,
        seq_len: i32,
        start_pos: i32,
        max_seq_len: i32,
        q_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    // FlashInfer single-request prefill: replaces Triton FA2 for HEAD_DIM=128.
    // Q/Output: [seq_len, num_q_heads * head_dim] NHD interleaved row-major.
    // K/V cache: [num_kv_heads, max_seq_len, head_dim] HND layout.
    pub(crate) fn flashinfer_single_prefill(
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

    // FlashAttention-2 prefill (Triton AOT) for HEAD_DIM=256.
    // Q/Output are col-major [q_dim, seq_len]. K/V cache are per-head [max_seq, HEAD_DIM].
    pub(crate) fn flash_attention_prefill_hd256_cuda(
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

    // Qwen3.5 full-attention prefill prep: Q/K norm + partial RoPE + KV cache write.
    pub(crate) fn prefill_attention_hd256_prep_cuda(
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

    // Apply sigmoid(gate) from interleaved q_full onto attention output in-place.
    pub(crate) fn attention_gate_batch_hd256_cuda(
        q_full_batch: *const Half,
        attn_out: *mut Half,
        num_q_heads: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    // Batched fused GQA decode attention (CUDA, split-KV, HEAD_DIM=128)
    // Processes B requests in one launch using per-request KV cache pointers.
    // Grid: (num_qheads, NUM_KV_SPLITS=4, batch_size).
    pub(crate) fn fused_gqa_attention_decode_batched(
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

    // Batched attention reduce: merge split-KV partials for B requests.
    // Grid: (num_qheads, batch_size).
    pub(crate) fn attention_decode_reduce_batched(
        partial_out: *const f32,
        partial_m: *const f32,
        partial_l: *const f32,
        output: *mut Half,
        num_qheads: i32,
        head_dim: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    // Fused GQA Attention — decode variant (Triton AOT, split-KV, HEAD_DIM=128)
    // Reads pos/seq_len from decode_meta; scale and rms_eps computed inside kernel.
    // Writes partial results to partial_out/m/l (FP32). Call attention_decode_reduce after.
    pub(crate) fn fused_gqa_attention_decode(
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

    // Attention reduce: merge split-KV partials into final bf16 output.
    pub(crate) fn attention_decode_reduce(
        partial_out: *mut f32,
        partial_m: *mut f32,
        partial_l: *mut f32,
        output: *mut Half,
        num_qheads: i32,
        stream: CUstream,
    ) -> CUresult;

    // ========================================================================
    // Qwen3.5 kernels
    // ========================================================================

    // Batched (1+weight) RMSNorm — one block per token
    pub(crate) fn rms_norm_batched_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    // (1+weight) RMSNorm — Qwen3.5 / Gemma style
    pub(crate) fn rms_norm_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    // Fused add + (1+weight) RMSNorm
    pub(crate) fn fused_add_rms_norm_offset_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    // Per-head RMSNorm with F32 weight + SiLU gate
    pub(crate) fn rms_norm_gated_cuda(
        x: *const Half,
        weight: *const f32,
        gate: *const Half,
        out: *mut Half,
        num_heads: i32,
        head_dim: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    // Gated delta rule recurrent decode (single step)
    pub(crate) fn gated_delta_rule_decode_cuda(
        qkv: *const Half,
        b_proj: *const Half,
        a_proj: *const Half,
        dt_bias: *const Half,
        A_log: *const f32,
        state: *mut f32,
        output: *mut Half,
        num_key_heads: i32,
        num_value_heads: i32,
        key_dim: i32,
        val_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    // Batched conv1d decode (B requests, seq_len=1)
    pub(crate) fn conv1d_decode_batch_cuda(
        x_batch: *const Half,
        conv_weight: *const Half,
        conv_state_ptrs: *mut *mut Half,
        out_batch: *mut Half,
        num_channels: i32,
        kernel_size: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    // Batched gated delta rule decode (B requests)
    pub(crate) fn gdr_decode_batch_cuda(
        qkv_batch: *const Half,
        b_proj_batch: *const Half,
        a_proj_batch: *const Half,
        dt_bias: *const Half,
        A_log: *const f32,
        state_ptrs: *mut *mut f32,
        output_batch: *mut Half,
        num_key_heads: i32,
        num_value_heads: i32,
        key_dim: i32,
        val_dim: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    // Causal depthwise conv1d prefill (parallel over sequence)
    pub(crate) fn conv1d_prefill_cuda(
        x_seq: *const Half,
        conv_weight: *const Half,
        conv_state: *mut Half,
        out_seq: *mut Half,
        num_channels: i32,
        seq_len: i32,
        kernel_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gated_delta_rule_prefill_chunk_prepare_cuda(
        qkv: *const Half,
        b_proj: *const Half,
        a_proj: *const Half,
        dt_bias: *const Half,
        a_log: *const f32,
        q_out: *mut Half,
        k_out: *mut Half,
        v_out: *mut Half,
        g_out: *mut f32,
        beta_out: *mut f32,
        num_key_heads: i32,
        num_value_heads: i32,
        qkv_dim: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gated_delta_rule_prefill_chunk_cumsum_cuda(
        g_in: *const f32,
        g_out: *mut f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gated_delta_rule_prefill_chunk_a_cuda(
        k: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        a_tril: *mut f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gated_delta_rule_prefill_chunk_solve_cuda(
        a_tril: *const f32,
        a_inv: *mut Half,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gated_delta_rule_prefill_chunk_recompute_cuda(
        k: *const Half,
        v: *const Half,
        beta: *const f32,
        w: *mut Half,
        u: *mut Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    // Chunk-wise GDR prefill stage 1 (Triton AOT): recurrent chunk-state update.
    // Expected future inputs:
    //   k / w: [seq_len, num_value_heads, 128] bf16
    //   u / v_new: [seq_len, num_value_heads, 128] bf16
    //   g_cumsum: [seq_len, num_value_heads] fp32
    //   initial_state / final_state: [num_value_heads, 128, 128] fp32 in [H, K, V] (V contiguous)
    //   chunk_state: [num_chunks, num_value_heads, 128, 128] fp32
    pub(crate) fn gated_delta_rule_prefill_chunk_state_cuda(
        k: *const Half,
        w: *const Half,
        u: *const Half,
        g_cumsum: *const f32,
        initial_state: *const f32,
        chunk_state: *mut f32,
        v_new: *mut Half,
        final_state: *mut f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    // Chunk-wise GDR prefill stage 2 (Triton AOT): chunk output accumulation.
    // Expected future inputs:
    //   q / k / v_new: [seq_len, num_value_heads, 128] bf16
    //   chunk_state: [num_chunks, num_value_heads, 128, 128] fp32
    //   g_cumsum: [seq_len, num_value_heads] fp32
    //   output: [seq_len, num_value_heads * 128] bf16
    pub(crate) fn gated_delta_rule_prefill_chunk_o_cuda(
        q: *const Half,
        k: *const Half,
        v_new: *const Half,
        chunk_state: *const f32,
        g_cumsum: *const f32,
        output: *mut Half,
        seq_len: i32,
        num_value_heads: i32,
        scale: f32,
        stream: CUstream,
    ) -> CUresult;

    // ─── Batched decode prep for paged KV: QK-norm + RoPE + paged KV write ───

    pub(crate) fn decode_prep_paged_cuda(
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

    // ─── Contiguous KV → paged KV copy ───

    pub(crate) fn kv_cache_to_paged_cuda(
        k_contiguous: *const Half,
        v_contiguous: *const Half,
        k_paged: *mut Half,
        v_paged: *mut Half,
        page_indices: *const i32,
        max_seq_len: i32,
        seq_len: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        stride_page: i32,
        stream: CUstream,
    ) -> CUresult;

    // ─── Paged KV cache append ───

    pub(crate) fn paged_kv_append_cuda(
        k_batch: *const Half,
        v_batch: *const Half,
        k_data: *mut Half,
        v_data: *mut Half,
        page_indices: *const i32,
        indptr: *const i32,
        positions: *const i32,
        batch_size: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    // ─── Scatter-write prefill K/V to token pool ───

    pub(crate) fn scatter_write_kv_cuda(
        k_batch: *const Half,
        v_batch: *const Half,
        k_pool: *mut Half,
        v_pool: *mut Half,
        token_indices: *const i32,
        seq_len: i32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    // ─── FlashInfer batch decode with paged KV cache ───

    pub(crate) fn flashinfer_batch_decode_plan(
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

    pub(crate) fn flashinfer_batch_decode_run(
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

    // ─── FlashInfer batch decode HD256 (Qwen3.5 full attention) ───

    pub(crate) fn flashinfer_batch_decode_hd256_plan(
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

    pub(crate) fn flashinfer_batch_decode_hd256_run(
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

    // ─── HD256 paged prep: QK-norm (1+w) + partial RoPE + paged KV write ───

    pub(crate) fn decode_prep_paged_hd256_cuda(
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

    pub(crate) fn attention_gate_paged_hd256_cuda(
        q_full_batch: *const Half,
        attn_out: *mut Half,
        num_q_heads: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    // Tensor-core decode: uses prefill kernel for decode (flat ITL at long contexts).
    // Plan step (CPU-side scheduling for BatchPrefillWithPagedKVCache).
    pub(crate) fn flashinfer_tc_decode_plan(
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

    // Tensor-core decode: run step (GPU kernel).
    pub(crate) fn flashinfer_tc_decode_run(
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

    // Split merged QKV buffer into separate Q, K, V buffers.
    pub(crate) fn split_qkv_cuda(
        qkv: *const Half,
        q: *mut Half,
        k: *mut Half,
        v: *mut Half,
        batch_size: i32,
        q_dim: i32,
        kv_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    // Fused silu_mul from merged gate+up buffer.
    pub(crate) fn silu_mul_fused_cuda(
        gate_up: *const Half,
        out: *mut Half,
        batch_size: i32,
        inter_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    // ─── KV cache quantization (INT8) ───

    // Quantize bf16 KV data → INT8 + f32 scales for tokens [start_pos..start_pos+token_count).
    // HND layout: [num_kv_heads, max_seq_len, head_dim].
    pub(crate) fn quantize_kv_bf16_to_int8_cuda(
        kv_bf16: *const Half,
        kv_int8: *mut i8,
        scales: *mut f32,
        num_kv_heads: i32,
        head_dim: i32,
        max_seq_len: i32,
        start_pos: i32,
        token_count: i32,
        stream: CUstream,
    ) -> CUresult;

    // Dequantize INT8 KV data → bf16 for tokens [0..token_count).
    pub(crate) fn dequantize_kv_int8_to_bf16_cuda(
        kv_int8: *const i8,
        scales: *const f32,
        kv_bf16: *mut Half,
        num_kv_heads: i32,
        head_dim: i32,
        max_seq_len: i32,
        token_count: i32,
        stream: CUstream,
    ) -> CUresult;

    // ─── KV cache quantization (INT8 paged / NHD layout) ───

    // Dequantize INT8 paged KV → bf16 working buffer for all tokens in the batch.
    // NHD layout: pool_idx * kv_dim + kv_head * head_dim + d
    pub(crate) fn dequantize_paged_kv_cuda(
        kv_int8: *const i8,
        scales: *const f32,
        kv_bf16: *mut Half,
        token_indices: *const i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        total_tokens: i32,
        stream: CUstream,
    ) -> CUresult;

    // Quantize 1 new token per request from bf16 working → INT8 paged pool.
    pub(crate) fn quantize_paged_kv_single_cuda(
        kv_bf16: *const Half,
        kv_int8: *mut i8,
        scales: *mut f32,
        new_token_indices: *const i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    // ─── FP8 E4M3 KV quantization (paged NHD layout) ───

    // Quantize 1 new token per request: bf16 working → FP8 paged pool.
    pub(crate) fn quantize_paged_kv_fp8_cuda(
        kv_bf16: *const Half,
        kv_fp8: *mut u8,
        new_token_indices: *const i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    // Quantize + scatter contiguous bf16 KV → FP8 paged pool (for migration).
    pub(crate) fn quantize_scatter_kv_fp8_cuda(
        kv_cont: *const Half,
        kv_fp8: *mut u8,
        page_indices: *const i32,
        max_seq_len: i32,
        seq_len: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    // ─── Fused-dequant decode attention (INT8 + FP8) ───

    pub(crate) fn decode_attention_int8_workspace_bytes(
        batch_size: i32,
        num_qo_heads: i32,
        head_dim: i32,
        num_splits: i32,
    ) -> usize;

    pub(crate) fn decode_attention_int8_cuda(
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

    // ─── KV cache contiguous → paged migration (INT8) ───

    // Copy INT8 KV data + scales from contiguous HND to paged NHD layout.
    pub(crate) fn kv_cache_to_paged_int8_cuda(
        k_cont: *const i8,
        v_cont: *const i8,
        k_scales_cont: *const f32,
        v_scales_cont: *const f32,
        k_paged: *mut i8,
        v_paged: *mut i8,
        k_scales_paged: *mut f32,
        v_scales_paged: *mut f32,
        token_indices: *const i32,
        max_seq_len: i32,
        seq_len: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    // FP8 E4M3 fused-dequant decode attention (no scales).
    pub(crate) fn decode_attention_fp8_cuda(
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

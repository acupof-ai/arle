use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub fn gated_delta_rule_decode_cuda(
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

    pub fn gated_delta_rule_prefill_recurrent_cuda(
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
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn conv1d_decode_batch_cuda(
        x_batch: *const Half,
        conv_weight: *const Half,
        conv_state_ptrs: *mut *mut Half,
        out_batch: *mut Half,
        num_channels: i32,
        kernel_size: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_decode_batch_cuda(
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

    /// Varlen partial-accept replay: slot `s` reads `qkv`/writes `output` at its
    /// own `s * max_len` row block and takes b/a from its own capture, running
    /// `row_len[s]` rows into `state_ptrs[s]`. One launch per layer for a whole
    /// speculative batch.
    pub fn gated_delta_rule_prefill_recurrent_varlen_cuda(
        qkv: *const Half,
        b_ptrs: *const *const Half,
        a_ptrs: *const *const Half,
        dt_bias: *const Half,
        A_log: *const f32,
        state_ptrs: *const *mut f32,
        row_len: *const i32,
        output: *mut Half,
        num_key_heads: i32,
        num_value_heads: i32,
        key_dim: i32,
        val_dim: i32,
        max_len: i32,
        batch: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Varlen twin of [`conv1d_prefill_cuda`]: slot `s` reads `x_ptrs[s]` for
    /// `row_len[s]` rows and writes `out_seq + s * max_len` rows.
    pub fn conv1d_prefill_varlen_cuda(
        x_ptrs: *const *const Half,
        conv_weight: *const Half,
        state_ptrs: *const *mut Half,
        row_len: *const i32,
        out_seq: *mut Half,
        num_channels: i32,
        max_len: i32,
        kernel_size: i32,
        batch: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn conv1d_prefill_cuda(
        x_seq: *const Half,
        conv_weight: *const Half,
        conv_state: *mut Half,
        out_seq: *mut Half,
        num_channels: i32,
        seq_len: i32,
        kernel_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gated_delta_rule_prefill_chunk_prepare_cuda(
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

    pub fn gated_delta_rule_prefill_chunk_cumsum_cuda(
        g_in: *const f32,
        g_out: *mut f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gated_delta_rule_prefill_chunk_a_cuda(
        k: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        a_tril: *mut f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gated_delta_rule_prefill_chunk_solve_cuda(
        a_tril: *const f32,
        a_inv: *mut Half,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gated_delta_rule_prefill_chunk_recompute_cuda(
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

    pub fn gated_delta_rule_prefill_chunk_state_cuda(
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

    pub fn gated_delta_rule_prefill_chunk_o_cuda(
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

}

// ============================================================================
// FlashQLA chunked GDR fwd (board #3) — TileLang AOT, Hopper-only, fixed
// Qwen3.6 shard H=32/Hg=16/DK=DV=128/chunk=64. Non-sm90 builds link
// CUDA_ERROR_NOT_SUPPORTED stubs; `gdr_fq_prep_cuda` is native CUDA C
// (csrc/misc/gated_delta_rule.cu) and links everywhere.
//
// Pipeline per prefill chunk (batch=1, token-major dense tensors):
//   prep:   qkv_conv [S, qkv_dim] -> q/k [S,16,128] l2norm'd bf16,
//           v [S,32,128] bf16, g/beta [S,32] f32
//   cumsum: g -> g_cumsum (chunk-local, 64)
//   kkt:    (k, beta) -> a_inv [S,32,64] bf16
//   fwd:    (q,k,v,a_inv,g_cumsum,beta,h0) -> o [S,32,128] bf16, ht
//           h0/ht may BOTH point at the slot state [32,128,128] f32
//           (each CTA reads its h0 slice fully before writing ht).
// ============================================================================
#[allow(dead_code)]
unsafe extern "C" {
    pub fn gdr_fq_prep_cuda(
        qkv: *const Half,
        b_proj: *const Half,
        a_proj: *const Half,
        dt_bias: *const Half,
        A_log: *const f32,
        q_out: *mut Half,
        k_out: *mut Half,
        v_out: *mut Half,
        g_out: *mut f32,
        beta_out: *mut f32,
        num_key_heads: i32,
        num_value_heads: i32,
        key_dim: i32,
        val_dim: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_cumsum_cuda(
        g_in: *const f32,
        g_out: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_kkt_cuda(
        k: *const Half,
        beta: *const f32,
        a_inv: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_fwd_cuda(
        q: *const Half,
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h0: *const f32,
        o: *mut Half,
        ht: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;
}

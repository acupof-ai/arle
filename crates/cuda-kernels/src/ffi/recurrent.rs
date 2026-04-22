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

    pub fn conv1d_prefill_packed_batch_cuda(
        x_batch: *const Half,
        conv_weight: *const Half,
        conv_state_ptrs: *const u64,
        seq_indptr: *const i32,
        out_batch: *mut Half,
        num_channels: i32,
        kernel_size: i32,
        batch_size: i32,
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

    pub fn gated_delta_rule_prefill_chunkwise_batch_cuda(
        qkv_batch: *const Half,
        b_proj_batch: *const Half,
        a_proj_batch: *const Half,
        dt_bias: *const Half,
        a_log: *const f32,
        state_ptrs: *const u64,
        q_ptrs: *const u64,
        k_ptrs: *const u64,
        v_ptrs: *const u64,
        g_cumsum_ptrs: *const u64,
        beta_ptrs: *const u64,
        a_tril_ptrs: *const u64,
        a_inv_ptrs: *const u64,
        w_ptrs: *const u64,
        u_ptrs: *const u64,
        chunk_state_ptrs: *const u64,
        v_new_ptrs: *const u64,
        seq_indptr: *const i32,
        output_batch: *mut Half,
        num_key_heads: i32,
        num_value_heads: i32,
        key_dim: i32,
        val_dim: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;
}

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

    /// Varlen replay: slot `s` reads qkv / writes output at its `s * max_len`
    /// block, running `row_len[s]` rows into `state_ptrs[s]`.
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

    /// Varlen twin: slot `s` reads `x_ptrs[s]`, writes at `s * max_len`.
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

    /// `count` equal-sized D2D copies in one launch. `bytes` must be a multiple
    /// of 16. Replaces a `cuMemcpyDtoDAsync` loop whose cost is the ~11 µs of
    /// host driver time per call, not the bandwidth.
    pub fn batched_copy_uniform_cuda(
        dst_ptrs: *const *mut std::ffi::c_void,
        src_ptrs: *const *const std::ffi::c_void,
        words_each: *const i32,
        bytes: usize,
        max_words: usize,
        count: i32,
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

}

// ============================================================================
// FlashQLA chunked GDR fwd+bwd — TileLang AOT, Hopper-only, DK=DV=128/chunk=64,
// one symbol set per (H, Hg) instantiation (kernels.toml): unsuffixed =
// H=32/Hg=16, `_h48` = H=48/Hg=16, `_h<H>g<Hg>` = the context-parallel shards.
// `FLASHQLA_GDR_TABLE` below is generated from the same rows. Non-sm90 builds link
// CUDA_ERROR_NOT_SUPPORTED stubs; `gdr_fq_prep_cuda` is native CUDA C
// (csrc/recurrent/gated_delta_rule.cu), head-count parameterized, links
// everywhere.
//
// Pipeline per prefill chunk (batch=1, token-major dense tensors):
//   prep:   qkv_conv [S, qkv_dim] -> q/k [S,Hg,128] l2norm'd bf16,
//           v [S,H,128] bf16, g/beta [S,H] f32
//   cumsum: g -> g_cumsum (chunk-local, 64)
//   kkt:    (k, beta) -> a_inv [S,H,64] bf16
//   fwd:    (q,k,v,a_inv,g_cumsum,beta,h0) -> o [S,H,128] bf16, ht
//           h0/ht may BOTH point at the slot state [H,128,128] f32
//           (each CTA reads its h0 slice fully before writing ht).
//
// Backward (training only; the fwd runs with store_h off):
//   prepare_h: (k,v,a_inv,g_cumsum,beta,h0) -> h [num_chunks,H,128,128] bf16
//   bwd:       (dout,dht,q,k,v,a_inv,g_cumsum,beta,h)
//              -> dq/dk/dv [S,H,128] bf16, dg/dbeta [S,H] f32, dh0
//              dq/dk carry the VALUE-head axis: with Hg<H the caller sums the
//              head group. dg is w.r.t. g_cumsum, not the per-token g.
// ============================================================================

pub type FqCumsumFn = unsafe extern "C" fn(*const f32, *mut f32, i32, CUstream) -> CUresult;
pub type FqKktFn =
    unsafe extern "C" fn(*const Half, *const f32, *mut Half, i32, CUstream) -> CUresult;
#[allow(clippy::type_complexity)]
pub type FqFwdFn = unsafe extern "C" fn(
    *const Half,
    *const Half,
    *const Half,
    *const Half,
    *const f32,
    *const f32,
    *const f32,
    *mut Half,
    *mut f32,
    i32,
    CUstream,
) -> CUresult;
#[allow(clippy::type_complexity)]
pub type FqPrepareHFn = unsafe extern "C" fn(
    *const Half,
    *const Half,
    *const Half,
    *const f32,
    *const f32,
    *const f32,
    *mut Half,
    i32,
    CUstream,
) -> CUresult;
#[allow(clippy::type_complexity)]
pub type FqBwdFn = unsafe extern "C" fn(
    *const Half,
    *const f32,
    *const Half,
    *const Half,
    *const Half,
    *const Half,
    *const f32,
    *const f32,
    *const Half,
    *mut Half,
    *mut Half,
    *mut Half,
    *mut f32,
    *mut f32,
    *mut f32,
    i32,
    CUstream,
) -> CUresult;
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

    pub fn gdr_fq_prepare_h_cuda(
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h0: *const f32,
        h: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_bwd_cuda(
        dout: *const Half,
        dht: *const f32,
        q: *const Half,
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h: *const Half,
        dq: *mut Half,
        dk: *mut Half,
        dv: *mut Half,
        dg: *mut f32,
        dbeta: *mut f32,
        dh0: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_cumsum_h48_cuda(
        g_in: *const f32,
        g_out: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_kkt_h48_cuda(
        k: *const Half,
        beta: *const f32,
        a_inv: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_fwd_h48_cuda(
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

    pub fn gdr_fq_prepare_h_h48_cuda(
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h0: *const f32,
        h: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_bwd_h48_cuda(
        dout: *const Half,
        dht: *const f32,
        q: *const Half,
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h: *const Half,
        dq: *mut Half,
        dk: *mut Half,
        dv: *mut Half,
        dg: *mut f32,
        dbeta: *mut f32,
        dh0: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_cumsum_h24g8_cuda(
        g_in: *const f32,
        g_out: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_kkt_h24g8_cuda(
        k: *const Half,
        beta: *const f32,
        a_inv: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_fwd_h24g8_cuda(
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

    pub fn gdr_fq_prepare_h_h24g8_cuda(
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h0: *const f32,
        h: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_bwd_h24g8_cuda(
        dout: *const Half,
        dht: *const f32,
        q: *const Half,
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h: *const Half,
        dq: *mut Half,
        dk: *mut Half,
        dv: *mut Half,
        dg: *mut f32,
        dbeta: *mut f32,
        dh0: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_cumsum_h12g4_cuda(
        g_in: *const f32,
        g_out: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_kkt_h12g4_cuda(
        k: *const Half,
        beta: *const f32,
        a_inv: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_fwd_h12g4_cuda(
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

    pub fn gdr_fq_prepare_h_h12g4_cuda(
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h0: *const f32,
        h: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_bwd_h12g4_cuda(
        dout: *const Half,
        dht: *const f32,
        q: *const Half,
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h: *const Half,
        dq: *mut Half,
        dk: *mut Half,
        dv: *mut Half,
        dg: *mut f32,
        dbeta: *mut f32,
        dh0: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_cumsum_h16g8_cuda(
        g_in: *const f32,
        g_out: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_kkt_h16g8_cuda(
        k: *const Half,
        beta: *const f32,
        a_inv: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_fwd_h16g8_cuda(
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

    pub fn gdr_fq_prepare_h_h16g8_cuda(
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h0: *const f32,
        h: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_bwd_h16g8_cuda(
        dout: *const Half,
        dht: *const f32,
        q: *const Half,
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h: *const Half,
        dq: *mut Half,
        dk: *mut Half,
        dv: *mut Half,
        dg: *mut f32,
        dbeta: *mut f32,
        dh0: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_cumsum_h16g16_cuda(
        g_in: *const f32,
        g_out: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_kkt_h16g16_cuda(
        k: *const Half,
        beta: *const f32,
        a_inv: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_fwd_h16g16_cuda(
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

    pub fn gdr_fq_prepare_h_h16g16_cuda(
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h0: *const f32,
        h: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gdr_fq_bwd_h16g16_cuda(
        dout: *const Half,
        dht: *const f32,
        q: *const Half,
        k: *const Half,
        v: *const Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        h: *const Half,
        dq: *mut Half,
        dk: *mut Half,
        dv: *mut Half,
        dg: *mut f32,
        dbeta: *mut f32,
        dh0: *mut f32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;
}

// Geometry -> symbol table, generated from the kernels.toml flashqla rows.
include!(concat!(env!("OUT_DIR"), "/flashqla_gdr_generated.rs"));

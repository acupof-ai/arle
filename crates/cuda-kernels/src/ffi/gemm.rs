use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub fn gemv_cuda(
        A: *const Half,
        x: *const Half,
        y: *mut Half,
        M: i32,
        K: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gemm_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gemm_graphsafe_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn fused_mlp_cuda(
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

    pub fn marlin_gemm_cuda(
        a: *const Half, // [M, K] bf16 activations
        b: *const u8,   // Marlin-packed int4 weights
        c: *mut Half,   // [M, N] bf16 output
        s: *const Half, // [K/group_size, N] fp16 scales
        prob_m: i32,
        prob_n: i32,
        prob_k: i32,
        workspace: *mut i32, // lock buffer
        groupsize: i32,
        dev: i32,
        stream: CUstream,
        thread_k: i32,
        thread_n: i32,
        sms: i32,
        max_par: i32,
    ) -> i32;

    pub fn gptq_marlin_repack_cuda(
        b_q_weight: *const u32,
        out: *mut u32,
        size_k: i32,
        size_n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn marlin_workspace_size(prob_n: i32, sms: i32) -> usize;

    pub fn w8a16_gemv_cuda(
        weight: *const i8,
        scales: *const Half,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn w4a16_gemv_cuda(
        weight: *const u8,
        scales: *const Half,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn w2a16_gemv_cuda(
        weight: *const u8,
        scales: *const Half,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn w8a16_gemv_batch_cuda(
        weight: *const i8,
        scales: *const Half,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn w4a16_gemv_batch_cuda(
        weight: *const u8,
        scales: *const Half,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn w2a16_gemv_batch_cuda(
        weight: *const u8,
        scales: *const Half,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q6k_gemv_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q6k_gemv_batch_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q6k_dequant_chunk_cuda(
        weight: *const u8,
        out_bf16: *mut Half,
        n: i32,
        k: i32,
        k_start: i32,
        k_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q3k_gemv_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q3k_gemv_batch_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q3k_dequant_chunk_cuda(
        weight: *const u8,
        out_bf16: *mut Half,
        n: i32,
        k: i32,
        k_start: i32,
        k_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q4k_gemv_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q4k_gemv_batch_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q4k_dequant_chunk_cuda(
        weight: *const u8,
        out_bf16: *mut Half,
        n: i32,
        k: i32,
        k_start: i32,
        k_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q5k_gemv_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q5k_gemv_batch_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q5k_dequant_chunk_cuda(
        weight: *const u8,
        out_bf16: *mut Half,
        n: i32,
        k: i32,
        k_start: i32,
        k_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn turboquant_weight_gemv_cuda(
        packed: *const u8,
        scales: *const Half, // f16
        signs: *const i8,
        centroids: *const f32,
        x: *const Half, // bf16
        y: *mut Half,   // bf16
        N: i32,
        K: i32,
        group_size: i32,
        packed_cols: i32,
        num_groups: i32,
        bits: i32,
        stream: CUstream,
    );

    pub fn turboquant_weight_dequant_cuda(
        packed: *const u8,
        scales: *const Half,
        signs: *const i8,
        centroids: *const f32,
        out: *mut Half, // bf16
        N: i32,
        K: i32,
        group_size: i32,
        packed_cols: i32,
        num_groups: i32,
        bits: i32,
        stream: CUstream,
    );
}

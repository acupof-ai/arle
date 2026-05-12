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

    pub fn gemm_w4a8_marlin_cuda(
        a: *const i8,    // [M, K] row-major INT8 activations
        b: *const u8,    // W4A8 Marlin-packed INT4 weights
        c: *mut i32,     // [max_par * 64, N] INT32 reduce buffer
        d: *mut Half,    // [M, N] FP16 output
        s1: *const f32,  // [M] activation scales
        s2: *const f32,  // [N] per-channel weight scales
        s3: *const Half, // [K/group_size, N] per-group scales
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

    pub fn gemm_w4_fp8_marlin_cuda(
        a: *const u8,    // [M, K] row-major FP8 e4m3 activations
        b: *const u8,    // PF8.2-preprocessed Marlin-packed INT4 weights
        c_tmp: *mut f32, // FP32 global-reduce buffer
        d: *mut Half,    // [M, N] BF16 output
        s1: *const f32,  // [M] activation scales
        s2: *const Half, // [K/group_size, N] BF16 W4 group scales
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

    pub fn quantize_bf16_rows_to_int8_cuda(
        input: *const Half,
        output: *mut i8,
        scales: *mut f32,
        rows: i32,
        cols: i32,
        stream: CUstream,
    ) -> CUresult;

    /// PF8.1 — BF16 → FP8 e4m3 per-row activation quant.
    /// `output` is `*mut u8` (FP8 e4m3 is a 1-byte type).
    /// `scales` stores per-row absmax / 448.0 (e4m3 finite max).
    pub fn quantize_bf16_rows_to_fp8_e4m3_cuda(
        input: *const Half,
        output: *mut u8,
        scales: *mut f32,
        rows: i32,
        cols: i32,
        stream: CUstream,
    ) -> CUresult;

    /// PF8.2 — Subtraction-merge zero-point=8 into packed INT4 weight tensor.
    /// Offline weight-prep step for W4+FP8 marlin GEMM; eliminates per-element
    /// zero-point subtract at runtime.
    /// `numel` must be a multiple of 32 (kernel processes 32 INT32 per block).
    /// Returns `cudaErrorInvalidValue` if alignment violated.
    pub fn marlin_int4_fp8_preprocess_without_zp_cuda(
        qweight: *const i32,
        output: *mut i32,
        numel: i32,
        stream: CUstream,
    ) -> CUresult;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DeviceContext;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    #[test]
    fn fp8_row_quantization_scales_match_absmax() {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let rows = 2usize;
        let cols = 513usize;
        let mut input_host = vec![bf16::ZERO; rows * cols];
        for col in 0..cols {
            let value = if col == 257 {
                -2.0
            } else {
                ((col % 17) as f32 - 8.0) * 0.03125
            };
            input_host[cols + col] = bf16::from_f32(value);
        }

        let input = ctx.stream.clone_htod(&input_host).expect("input H2D");
        let mut output = ctx
            .stream
            .alloc_zeros::<u8>(rows * cols)
            .expect("output alloc");
        let mut scales = ctx.stream.alloc_zeros::<f32>(rows).expect("scales alloc");
        {
            let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
            let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr_mut(&ctx.stream);

            unsafe {
                quantize_bf16_rows_to_fp8_e4m3_cuda(
                    input_ptr as *const Half,
                    output_ptr as *mut u8,
                    scales_ptr as *mut f32,
                    rows as i32,
                    cols as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("fp8 row quantize");
            }
        }
        ctx.sync().expect("sync fp8 row quantize");

        let got_scales = ctx.stream.clone_dtoh(&scales).expect("scales D2H");
        assert_eq!(got_scales[0], 1.0);
        assert!(
            (got_scales[1] - (2.0 / 448.0)).abs() < 1.0e-7,
            "nonzero row scale mismatch: got {}, expected {}",
            got_scales[1],
            2.0 / 448.0
        );

        let got_output = ctx.stream.clone_dtoh(&output).expect("output D2H");
        assert!(
            got_output[..cols].iter().all(|&byte| byte == 0),
            "zero row should quantize to all-zero fp8 bytes"
        );
    }
}

use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub fn add_bf16_into_f32_cuda(
        out: *mut f32,
        r#in: *const Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn add_cuda(
        a: *const Half,
        b: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn add_assign_cuda(a: *mut Half, b: *const Half, n: i32, stream: CUstream) -> CUresult;

    pub fn silu_mul_cuda(
        gate: *const Half,
        up: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_swiglu_clamped_cuda(
        gate: *const Half,
        up: *const Half,
        out: *mut Half,
        n: i32,
        limit: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn add_scaled_row_cuda(
        row: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        token_idx: i32,
        scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn add_scaled_row_segment_cuda(
        row: *const Half,
        out: *mut Half,
        row_len: i32,
        out_hidden_dim: i32,
        token_idx: i32,
        segment_offset: i32,
        scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn split_qkv_cuda(
        qkv: *const Half,
        q: *mut Half,
        k: *mut Half,
        v: *mut Half,
        batch_size: i32,
        q_dim: i32,
        kv_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn split2_cuda(
        fused: *const Half,
        first: *mut Half,
        second: *mut Half,
        batch_size: i32,
        first_dim: i32,
        second_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn silu_mul_fused_cuda(
        gate_up: *const Half,
        out: *mut Half,
        batch_size: i32,
        inter_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    // bf16 -> fp8 (e4m3) per-tensor cast with identity scale. Feeds the q8kv8
    // sparse prefill kernel, which expects fp8 Q/KV. See
    // csrc/elementwise/bf16_to_fp8.cu.
    pub fn arle_bf16_to_fp8_e4m3_cuda(
        input: *const Half,
        output: *mut u8,
        n: i64,
        stream: CUstream,
    ) -> CUresult;
}

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

    pub fn silu_mul_cuda(
        gate: *const Half,
        up: *const Half,
        out: *mut Half,
        n: i32,
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

    pub fn silu_mul_fused_cuda(
        gate_up: *const Half,
        out: *mut Half,
        batch_size: i32,
        inter_dim: i32,
        stream: CUstream,
    ) -> CUresult;
}

use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub fn rms_norm_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn rms_norm_batched_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn rms_norm_batched_f32_in_cuda(
        x: *const f32,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn fused_add_rms_norm_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn fused_add_rms_norm_batched_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn rms_norm_batched_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn rms_norm_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn fused_add_rms_norm_offset_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn rms_norm_gated_cuda(
        x: *const Half,
        weight: *const f32,
        gate: *const Half,
        out: *mut Half,
        num_heads: i32,
        head_dim: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;
}

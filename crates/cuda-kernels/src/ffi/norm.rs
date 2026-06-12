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

    /// flashinfer fused residual-add + (1+weight) RMSNorm (opt-in
    /// `ARLE_QWEN35_FUSED_ADDNORM`). Vendored `FusedAddRMSNormKernel`
    /// (csrc/vendor/flashinfer/norm.cuh) launched with `weight_bias = 1.0`
    /// to match ARLE's `rms_norm_offset` Gemma-style gain. Mutates BOTH
    /// buffers in place: `input` ← rms(input+residual), `residual` ←
    /// input+residual (the running pre-norm sum). `[batch, d]` row-major bf16
    /// (one token = `d` contiguous features; row stride = `stride_*`).
    pub fn arle_fused_add_rmsnorm_offset_bf16_cuda(
        input: *mut Half,
        residual: *mut Half,
        weight: *const Half,
        batch: u32,
        d: u32,
        stride_input: u32,
        stride_residual: u32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;
}

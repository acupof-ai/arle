use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {

    pub fn quantize_kv_bf16_to_int8_cuda(
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

    pub fn dequantize_kv_int8_to_bf16_cuda(
        kv_int8: *const i8,
        scales: *const f32,
        kv_bf16: *mut Half,
        num_kv_heads: i32,
        head_dim: i32,
        max_seq_len: i32,
        token_count: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn quantize_paged_kv_single_cuda(
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

    pub fn quantize_paged_kv_fp8_cuda(
        kv_bf16: *const Half,
        kv_fp8: *mut u8,
        scales: *mut f32,
        new_token_indices: *const i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn quantize_scatter_kv_fp8_range_cuda(
        kv_cont: *const Half,
        kv_fp8: *mut u8,
        scales: *mut f32,
        page_indices: *const i32,
        start_pos: i32,
        max_seq_len: i32,
        token_count: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        stream: CUstream,
    ) -> CUresult;

}

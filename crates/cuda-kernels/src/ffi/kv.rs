use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub fn kv_cache_to_paged_cuda(
        k_contiguous: *const Half,
        v_contiguous: *const Half,
        k_paged: *mut Half,
        v_paged: *mut Half,
        page_indices: *const i32,
        max_seq_len: i32,
        seq_len: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        stride_page: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kv_cache_to_paged_range_cuda(
        k_contiguous: *const Half,
        v_contiguous: *const Half,
        k_paged: *mut Half,
        v_paged: *mut Half,
        token_indices: *const i32,
        start_pos: i32,
        max_seq_len: i32,
        token_count: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kv_cache_to_paged_range_hnd_cuda(
        k_contiguous: *const Half,
        v_contiguous: *const Half,
        k_paged: *mut Half,
        v_paged: *mut Half,
        page_indices: *const i32,
        start_pos: i32,
        max_seq_len: i32,
        token_count: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        stride_page: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn paged_kv_append_cuda(
        k_batch: *const Half,
        v_batch: *const Half,
        k_data: *mut Half,
        v_data: *mut Half,
        page_indices: *const i32,
        indptr: *const i32,
        positions: *const i32,
        batch_size: i32,
        num_kv_heads: i32,
        page_size: i32,
        head_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn scatter_write_kv_cuda(
        k_batch: *const Half,
        v_batch: *const Half,
        k_pool: *mut Half,
        v_pool: *mut Half,
        token_indices: *const i32,
        seq_len: i32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: CUstream,
    ) -> CUresult;

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

    #[allow(dead_code)]
    pub fn dequantize_paged_kv_cuda(
        kv_int8: *const i8,
        scales: *const f32,
        kv_bf16: *mut Half,
        token_indices: *const i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        total_tokens: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dequantize_paged_kv_int8_to_hnd_cuda(
        kv_int8: *const i8,
        scales: *const f32,
        kv_bf16_hnd: *mut Half,
        token_rows: *const i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        total_tokens: i32,
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

    pub fn quantize_scatter_kv_fp8_cuda(
        kv_cont: *const Half,
        kv_fp8: *mut u8,
        scales: *mut f32,
        page_indices: *const i32,
        max_seq_len: i32,
        seq_len: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
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

    pub fn dequantize_paged_kv_fp8_to_hnd_cuda(
        kv_fp8: *const u8,
        scales: *const f32,
        kv_bf16_hnd: *mut Half,
        token_rows: *const i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        total_tokens: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kv_cache_to_paged_int8_cuda(
        k_cont: *const i8,
        v_cont: *const i8,
        k_scales_cont: *const f32,
        v_scales_cont: *const f32,
        k_paged: *mut i8,
        v_paged: *mut i8,
        k_scales_paged: *mut f32,
        v_scales_paged: *mut f32,
        token_indices: *const i32,
        max_seq_len: i32,
        seq_len: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kv_cache_to_paged_int8_range_cuda(
        k_cont: *const i8,
        v_cont: *const i8,
        k_scales_cont: *const f32,
        v_scales_cont: *const f32,
        k_paged: *mut i8,
        v_paged: *mut i8,
        k_scales_paged: *mut f32,
        v_scales_paged: *mut f32,
        token_indices: *const i32,
        start_pos: i32,
        max_seq_len: i32,
        token_count: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        stream: CUstream,
    ) -> CUresult;
}

use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub(crate) fn cast_bf16_to_f32_cuda(
        r#in: *const Half,
        out: *mut f32,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn cast_f32_to_bf16_cuda(
        r#in: *const f32,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn bf16_to_fp16_cuda(
        input: *const Half,
        output: *mut u16, // __half is u16
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn fp16_to_bf16_cuda(
        input: *const u16, // __half
        output: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn turboquant_lloyd_max(
        centroids: *mut f32,
        boundaries: *mut f32,
        num_levels: i32,
        head_dim: i32,
        max_iters: i32,
    );

    pub(crate) fn turboquant_generate_rotation(Pi: *mut f32, head_dim: i32, seed: u64);

    pub(crate) fn turboquant_quantize_kv_cuda(
        kv_bf16: *const Half,
        packed_out: *mut u8,
        norms_out: *mut Half,
        Pi: *const f32,
        boundaries: *const f32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        packed_per_head: i32,
        num_levels: i32,
        bits: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn turboquant_dequantize_kv_cuda(
        packed_in: *const u8,
        norms_in: *const Half,
        kv_bf16: *mut Half,
        Pi: *const f32,
        centroids: *const f32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        packed_per_head: i32,
        num_levels: i32,
        bits: i32,
        token_count: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn turboquant_quantize_single_cuda(
        kv_bf16: *const Half,
        pool_data: *mut u8,
        pool_norms: *mut Half,
        pool_indices: *const i32,
        Pi: *const f32,
        boundaries: *const f32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        packed_per_head: i32,
        num_levels: i32,
        bits: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn turboquant_dequantize_paged_cuda(
        pool_data: *const u8,
        pool_norms: *const Half,
        kv_bf16: *mut Half,
        token_indices: *const i32,
        Pi: *const f32,
        centroids: *const f32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        packed_per_head: i32,
        num_levels: i32,
        bits: i32,
        total_tokens: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn turboquant_dequantize_inplace_cuda(
        pool_data: *const u8,
        pool_norms: *const Half,
        work_bf16: *mut Half,
        pool_indices: *const i32,
        Pi: *const f32,
        centroids: *const f32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        packed_per_head: i32,
        num_levels: i32,
        bits: i32,
        num_indices: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn turboquant_generate_signs(signs: *mut i8, head_dim: i32, seed: u64);

    pub(crate) fn turboquant_fast_quantize_kv_cuda(
        kv_bf16: *const Half,
        packed_out: *mut u8,
        norms_out: *mut Half,
        signs: *const i8,
        boundaries: *const f32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        packed_per_head: i32,
        num_levels: i32,
        bits: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn turboquant_fast_dequantize_kv_cuda(
        packed_in: *const u8,
        norms_in: *const Half,
        kv_bf16: *mut Half,
        signs: *const i8,
        centroids: *const f32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        packed_per_head: i32,
        num_levels: i32,
        bits: i32,
        token_count: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn turboquant_fast_dequantize_inplace_cuda(
        pool_data: *const u8,
        pool_norms: *const Half,
        work_bf16: *mut Half,
        pool_indices: *const i32,
        signs: *const i8,
        centroids: *const f32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        packed_per_head: i32,
        num_levels: i32,
        bits: i32,
        num_indices: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn turboquant_fast_quantize_single_cuda(
        kv_bf16: *const Half,
        pool_data: *mut u8,
        pool_norms: *mut Half,
        pool_indices: *const i32,
        signs: *const i8,
        boundaries: *const f32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        packed_per_head: i32,
        num_levels: i32,
        bits: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn tq_rotate_query_cuda(
        Q: *const Half,
        Q_rot: *mut Half,
        signs: *const i8,
        num_heads_total: i32,
        head_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn tq_decode_attention_cuda(
        Q_rot: *const Half,
        K_packed: *const u8,
        K_norms: *const Half,
        V_packed: *const u8,
        V_norms: *const Half,
        kv_indices: *const i32,
        kv_indptr: *const i32,
        O: *mut Half,
        centroids_k: *const f32,
        centroids_v: *const f32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        packed_per_head: i32,
        num_levels: i32,
        bits: i32,
        sm_scale: f32,
        head_dim: i32,
        stream: CUstream,
    ) -> CUresult;
}

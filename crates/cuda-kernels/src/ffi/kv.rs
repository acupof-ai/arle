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

    #[allow(dead_code)]
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

    pub fn dequantize_paged_kv_int8_per_channel_k_to_hnd_cuda(
        kv_int8: *const i8,
        k_static_scales: *const f32,
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

    /// KIVI per-channel K quantize: consumes a pre-computed
    /// `[num_kv_heads, head_dim]` f32 scale table, no per-(token, head)
    /// reduction. See `csrc/kv/kv_quant.cu::quantize_paged_kv_fp8_per_channel_cuda`.
    pub fn quantize_paged_kv_fp8_per_channel_cuda(
        kv_bf16: *const Half,
        kv_fp8: *mut u8,
        k_static_scales: *const f32,
        new_token_indices: *const i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Calibration: accumulate per-(kv_head, head_dim) absmax over a batch
    /// of K rows from the bf16 HND-paged work buffer into
    /// `k_static_scales`. Stores **raw absmax**, not divided by 448. Caller
    /// must invoke `finalize_k_per_channel_scales_cuda` once after all
    /// calibration batches are accumulated.
    pub fn compute_k_per_channel_absmax_cuda(
        kv_bf16: *const Half,
        k_static_scales: *mut f32,
        token_rows: *const i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Divide accumulated absmax in `k_static_scales` by 448.0 (FP8 E4M3
    /// max representable) to yield the final per-channel scale. Idempotent
    /// only if called exactly once per layer per calibration session.
    pub fn finalize_k_per_channel_scales_cuda(
        k_static_scales: *mut f32,
        num_channels: i32,
        stream: CUstream,
    ) -> CUresult;

    /// INT8 KIVI sibling of `finalize_k_per_channel_scales_cuda`: divides
    /// accumulated absmax by 127.0 (INT8 max representable, symmetric). Same
    /// 1e-30 floor for untouched channels.
    pub fn finalize_k_per_channel_scales_int8_cuda(
        k_static_scales: *mut f32,
        num_channels: i32,
        stream: CUstream,
    ) -> CUresult;

    /// INT8 KIVI per-channel K quantize: mirrors
    /// `quantize_paged_kv_fp8_per_channel_cuda` but writes INT8 (round +
    /// clamp to [-127, 127]) instead of FP8.
    pub fn quantize_paged_kv_int8_per_channel_cuda(
        kv_bf16: *const Half,
        kv_int8: *mut i8,
        k_static_scales: *const f32,
        new_token_indices: *const i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    /// INT4 KIVI per-channel K finalize: divides absmax by 7 (symmetric
    /// int4 max), same 1e-30 floor as INT8/FP8 siblings.

    /// INT4 KIVI two-level K quantize: per-channel STATIC × per-(token, head)
    /// DYNAMIC. 4-bit packed (2 nibbles/byte). `kv_int4_packed` must have
    /// `max_total_tokens * kv_dim / 2` bytes. `k_dynamic_scales` is
    /// `[max_total_tokens, num_kv_heads]` f32, written by this kernel.

    /// INT4 V quantize (per-(row, head) absmax, /7). 4-bit packed output.
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

    pub fn dequantize_paged_kv_fp8_per_channel_k_to_hnd_cuda(
        kv_fp8: *const u8,
        k_static_scales: *const f32,
        kv_bf16_hnd: *mut Half,
        token_rows: *const i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_dim: i32,
        total_tokens: i32,
        stream: CUstream,
    ) -> CUresult;

}

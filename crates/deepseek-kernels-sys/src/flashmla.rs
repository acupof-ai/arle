//! FlashMLA SM90 sparse prefill + sparse FP8 decode C ABI
//! (`csrc/attention/arle_flashmla_{shim,decode_shim}.cu`). Builds without the
//! FlashMLA tree link `arle_flashmla_decode_stubs.cu`, which returns
//! `cudaErrorNotSupported` and does not define the real-kernel marker.

use crate::{CUresult, CUstream, Half};

unsafe extern "C" {
    // FlashMLA SM90 sparse prefill (vendored sgl-project/FlashMLA @ df022eb).
    // Bypasses FlashMLA's PyTorch wrapper and calls `sm90::run_fwd_kernel`
    // directly. q/kv must be bf16 device pointers; the kernel supports
    // d_qk ∈ {512, 576} and d_v = 512 — matches DSv4-Flash MLA (head_dim 512
    // NoPE + optional 64-dim RoPE tail). See arle_flashmla_shim.cu.
    pub fn arle_flashmla_sm90_sparse_prefill_fwd(
        q: *const Half,
        kv: *const Half,
        indices: *const i32,
        attn_sink: *const f32,
        topk_length: *const i32,
        out: *mut Half,
        max_logits: *mut f32,
        lse: *mut f32,
        s_q: i32,
        s_kv: i32,
        h_q: i32,
        h_kv: i32,
        d_qk: i32,
        d_v: i32,
        topk: i32,
        sm_scale: f32,
        stride_q_s_q: i32,
        stride_q_h_q: i32,
        stride_kv_s_kv: i32,
        stride_kv_h_kv: i32,
        stride_indices_s_q: i32,
        stride_indices_h_kv: i32,
        num_sm: i32,
        stream: CUstream,
    ) -> CUresult;

    // FlashMLA SM90 sparse FP8 decode (vendored sgl-project/FlashMLA @ df022eb).
    //
    // Wraps `sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`
    // + `smxx::decode::run_flash_mla_combine_kernel` in a single call. KV
    // must be FP8-packed bytes per the model-specific contract; the bf16
    // typing of the `kv` argument is only because upstream's params struct
    // declares it that way. See `arle_flashmla_decode_shim.cu` for the
    // full byte layout (MODEL1 = 584 bytes/token, V32 = 656 bytes/token).
    //
    // **ARLE's current decode KV pool is bf16, not FP8 — this FFI will
    // return `cudaErrorInvalidValue` until a separate FP8-packing kernel
    // converts the bf16 sliding-window + compressed buffers into the
    // expected layout. Dormant until that packing lands.**
    pub fn arle_flashmla_sm90_sparse_decode_fwd(
        q: *const Half,
        kv: *const Half,
        indices: *const i32,
        topk_length: *const i32,
        attn_sink: *const f32,
        out: *mut Half,
        lse: *mut f32,
        lse_accum: *mut f32,
        o_accum: *mut f32,
        tile_scheduler_metadata: *const i32,
        num_splits: *const i32,
        b: i32,
        s_q: i32,
        h_q: i32,
        h_kv: i32,
        d_qk: i32,
        d_v: i32,
        num_blocks: i32,
        page_block_size: i32,
        topk: i32,
        num_sm_parts: i32,
        model_type_int: i32,
        sm_scale: f32,
        stride_q_b: i32,
        stride_q_s_q: i32,
        stride_q_h_q: i32,
        stride_kv_block_bytes: i32,
        stride_kv_row_bytes: i32,
        stride_indices_b: i32,
        stride_indices_s_q: i32,
        stride_lse_b: i32,
        stride_lse_s_q: i32,
        stride_o_b: i32,
        stride_o_s_q: i32,
        stride_o_h_q: i32,
        stride_lse_accum_split: i32,
        stride_lse_accum_s_q: i32,
        stride_o_accum_split: i32,
        stride_o_accum_s_q: i32,
        stride_o_accum_h_q: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Returns the FP8-packed bytes/token for a (`d_qk`, `model_type_int`)
    /// pair, or -1 if unsupported. `model_type_int`: 0 = V32 (d_qk=576),
    /// 1 = MODEL1 (d_qk=512).
    pub fn arle_flashmla_sm90_sparse_decode_bytes_per_token(d_qk: i32, model_type_int: i32) -> i32;

    /// Compute the decode scheduler tuning meta (`num_sm_parts`,
    /// `fixed_overhead_num_blocks`, `block_size_topk`) on the host for a
    /// (`h_q`, `s_q`, `model_type_int`) tuple. Caller uses
    /// `num_sm_parts` to size the GPU-side tile-scheduler-metadata buffer
    /// before calling `arle_flashmla_sm90_sparse_decode_sched_meta`.
    pub fn arle_flashmla_sm90_sparse_decode_get_meta(
        h_q: i32,
        s_q: i32,
        model_type_int: i32,
        out_num_sm_parts: *mut i32,
        out_fixed_overhead_num_blocks: *mut i32,
        out_block_size_topk: *mut i32,
    ) -> CUresult;

    /// Populate the `tile_scheduler_metadata` + `num_splits` arrays from
    /// per-batch effective topk lengths. Both arrays must be device buffers
    /// of the right size:
    ///   `tile_scheduler_metadata`: `num_sm_parts * DecodingSchedMetaSize/4` i32
    ///   `num_splits`: `b + 1` i32
    pub fn arle_flashmla_sm90_sparse_decode_sched_meta(
        b: i32,
        s_q: i32,
        block_size_topk: i32,
        fixed_overhead_num_blocks: i32,
        topk: i32,
        extra_topk: i32,
        topk_length: *const i32,
        extra_topk_length: *const i32,
        tile_scheduler_metadata: *mut i32,
        num_splits: *mut i32,
        num_sm_parts: i32,
        stream: CUstream,
    ) -> CUresult;
}

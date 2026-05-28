#[allow(dead_code)]
unsafe extern "C" {
    pub fn cublas_init();
    pub fn autotune_all_cached_gemms_cuda(stream: super::CUstream);

    pub fn dsv4_mhc_expand_cuda(
        embeddings: *const super::Half,
        out: *mut super::Half,
        num_tokens: i32,
        hidden_size: i32,
        hc_mult: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_mhc_params_cuda(
        residual: *const super::Half,
        mixes: *const super::Half,
        base: *const super::Half,
        scale: *const super::Half,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        num_tokens: i32,
        residual_hidden_dim: i32,
        mix_dim: i32,
        hc_mult: i32,
        eps: f32,
        sinkhorn_iters: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_mhc_pre_cuda(
        residual: *const super::Half,
        pre: *const f32,
        out: *mut super::Half,
        num_tokens: i32,
        hidden_size: i32,
        hc_mult: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_mhc_post_cuda(
        new_x: *const super::Half,
        residual: *const super::Half,
        post: *const f32,
        comb: *const f32,
        out: *mut super::Half,
        num_tokens: i32,
        hidden_size: i32,
        hc_mult: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_mhc_head_pre_cuda(
        residual_row: *const super::Half,
        mixes: *const super::Half,
        base: *const super::Half,
        scale: *const super::Half,
        out: *mut super::Half,
        residual_hidden_dim: i32,
        hidden_size: i32,
        hc_mult: i32,
        eps: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_prepare_qk_cuda(
        q_raw: *const super::Half,
        k_raw: *const super::Half,
        q_out: *mut super::Half,
        k_out: *mut super::Half,
        num_tokens: i32,
        local_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        start_pos: i32,
        rms_eps: f32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_prepare_qk_fused_cuda(
        q_raw: *const super::Half,
        k_raw: *const super::Half,
        q_out: *mut super::Half,
        k_out: *mut super::Half,
        num_tokens: i32,
        local_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        start_pos: i32,
        rms_eps: f32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_swa_attention_cuda(
        q: *const super::Half,
        k_new: *const super::Half,
        window_cache: *mut super::Half,
        attn_sink: *const super::Half,
        out: *mut super::Half,
        num_tokens: i32,
        local_heads: i32,
        head_dim: i32,
        sliding_window: i32,
        start_pos: i32,
        sink_offset: i32,
        scale_value: f32,
        rope_dim: i32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        write_window_cache: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_update_window_cache_cuda(
        k_new: *const super::Half,
        window_cache: *mut super::Half,
        num_tokens: i32,
        start_pos: i32,
        sliding_window: i32,
        head_dim: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_compressor_update_cuda(
        kv_raw: *const super::Half,
        score_raw: *const super::Half,
        ape: *const super::Half,
        norm: *const super::Half,
        pending_kv: *mut super::Half,
        pending_score: *mut super::Half,
        prev_overlap_kv: *mut super::Half,
        prev_overlap_score: *mut super::Half,
        compressed: *mut super::Half,
        num_tokens: i32,
        start_pos: i32,
        pending_len: i32,
        compressed_base: i32,
        head_dim: i32,
        ratio: i32,
        width: i32,
        overlap: i32,
        has_prev_overlap: i32,
        eps: f32,
        rope_dim: i32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_hybrid_attention_cuda(
        q: *const super::Half,
        k_new: *const super::Half,
        window_cache: *mut super::Half,
        compressed: *const super::Half,
        selected: *const i32,
        attn_sink: *const super::Half,
        out: *mut super::Half,
        num_tokens: i32,
        local_heads: i32,
        head_dim: i32,
        sliding_window: i32,
        start_pos: i32,
        sink_offset: i32,
        scale_value: f32,
        rope_dim: i32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        mode: i32,
        compress_ratio: i32,
        compressed_count: i32,
        selected_topk: i32,
        write_window_cache: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    // FlashMLA SM90 sparse prefill (vendored sgl-project/FlashMLA @ df022eb).
    // Bypasses FlashMLA's PyTorch wrapper and calls `sm90::run_fwd_kernel`
    // directly. q/kv must be bf16 device pointers; the kernel supports
    // d_qk ∈ {512, 576} and d_v = 512 — matches DSv4-Flash MLA (head_dim 512
    // NoPE + optional 64-dim RoPE tail). See arle_flashmla_shim.cu.
    pub fn arle_flashmla_sm90_sparse_prefill_fwd(
        q: *const super::Half,
        kv: *const super::Half,
        indices: *const i32,
        attn_sink: *const f32,
        topk_length: *const i32,
        out: *mut super::Half,
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
        stream: super::CUstream,
    ) -> super::CUresult;

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
    // expected layout. Tracked under `ARLE_DSV4_FLASHMLA_DECODE` (default
    // OFF) and the project plan
    // `docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`.**
    pub fn arle_flashmla_sm90_sparse_decode_fwd(
        q: *const super::Half,
        kv: *const super::Half,
        indices: *const i32,
        topk_length: *const i32,
        attn_sink: *const f32,
        out: *mut super::Half,
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
        stream: super::CUstream,
    ) -> super::CUresult;

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
    ) -> super::CUresult;

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
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_csa_select_cuda(
        q: *const super::Half,
        weights: *const super::Half,
        keys: *const super::Half,
        selected: *mut i32,
        num_tokens: i32,
        q_width: i32,
        local_heads: i32,
        index_dim: i32,
        key_count: i32,
        ratio: i32,
        topk: i32,
        score_scale: f32,
        start_pos: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    // ------------------------------------------------------------------
    // V2 FlashMLA support: bf16→f32 convert, TP repack/slice, CSA prep.
    // See:
    //   crates/cuda-kernels/csrc/misc/arle_dtype_convert.cu
    //   crates/cuda-kernels/csrc/misc/dsv4_tp_attention_repack.cu
    //   crates/cuda-kernels/csrc/misc/arle_flashmla_csa_prep.cu
    // ------------------------------------------------------------------

    /// bf16 → f32 device-side convert. One-shot at model load (e.g. DSv4
    /// attn_sink f32 mirror for FlashMLA's float[h_q] contract).
    pub fn arle_bf16_to_f32_cuda(
        src: *const super::Half,
        dst: *mut f32,
        n: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Repack AllGather recv buffer (rank-major) into FlashMLA's expected
    /// h_q-major Q layout. gathered: bf16 [tp_world, s_q, h_local, d];
    /// packed: bf16 [s_q, tp_world*h_local, d] with rank w at heads
    /// [w*h_local, (w+1)*h_local).
    pub fn dsv4_tp_q_repack_cuda(
        gathered: *const super::Half,
        packed: *mut super::Half,
        tp_world: i32,
        s_q: i32,
        h_local: i32,
        d: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Slice this rank's local-heads slab out of FlashMLA's [s_q, h_global, d]
    /// output into the per-rank local_attn buffer [s_q, h_local, d].
    pub fn dsv4_tp_out_slice_cuda(
        full_out: *const super::Half,
        local: *mut super::Half,
        s_q: i32,
        global_width: i32,
        local_width: i32,
        head_offset: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Pack ARLE's rolling sliding-window cache + current-chunk K + compressed
    /// pool into a single contiguous KV pool for FlashMLA SM90 sparse prefill.
    pub fn arle_flashmla_csa_pack_kv(
        kv_unified: *mut super::Half,
        window_cache: *const super::Half,
        k_prepared: *const super::Half,
        compressed: *const super::Half,
        start_pos: i32,
        sw_window: i32,
        n_tokens: i32,
        compressed_count: i32,
        d_qk: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Build per-token unified indices + topk_length matching the layout
    /// produced by `arle_flashmla_csa_pack_kv`. `compress_ratio` enables
    /// the compress-block causality gate (block_end > abs_pos → -1).
    pub fn arle_flashmla_csa_build_indices(
        indices: *mut i32,
        topk_length: *mut i32,
        selected: *const i32,
        s_q: i32,
        start_pos: i32,
        sw_window: i32,
        index_topk: i32,
        compressed_count: i32,
        compress_ratio: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// HCA (HybridCompressed) per-token unified indices. No selector;
    /// attends to all compressed pages causally gated by `compress_ratio`.
    /// `max_compressed_keys` is the pool capacity for compressed slots in
    /// each row — caller must allocate `s_q * (sw_window + max_compressed_keys)`
    /// int32 with `(sw_window + max_compressed_keys) % 128 == 0`.
    pub fn arle_flashmla_hca_build_indices(
        indices: *mut i32,
        topk_length: *mut i32,
        s_q: i32,
        start_pos: i32,
        sw_window: i32,
        max_compressed_keys: i32,
        compressed_count: i32,
        compress_ratio: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Fill the [s_q_actual..s_q_padded) rows of the indices buffer with -1
    /// and the corresponding topk_length entries with 0, for FlashMLA s_q
    /// padding (V2.3). Use this after a build_indices call that wrote rows
    /// [0..s_q_actual). No-op when s_q_padded <= s_q_actual.
    pub fn arle_flashmla_fill_pad_rows(
        indices: *mut i32,
        topk_length: *mut i32,
        s_q_actual: i32,
        s_q_padded: i32,
        topk_unified: i32,
        stream: super::CUstream,
    ) -> super::CUresult;
}

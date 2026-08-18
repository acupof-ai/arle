#[allow(dead_code)]
unsafe extern "C" {
    pub fn cublas_init();

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

    pub fn dsv4_mhc_pre_rms_norm_cuda(
        residual: *const super::Half,
        pre: *const f32,
        weight: *const super::Half,
        out: *mut super::Half,
        num_tokens: i32,
        hidden_size: i32,
        hc_mult: i32,
        eps: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Fused [`dsv4_mhc_params_cuda`] + [`dsv4_mhc_pre_rms_norm_cuda`]: one
    /// launch per token computes pre/post/comb AND the pre-mixed rms-normed
    /// row (`pre` consumed from shared memory). Requires the wide-stream
    /// layout `residual_hidden_dim == hidden_size * hc_mult`.
    pub fn dsv4_mhc_params_pre_rms_norm_cuda(
        residual: *const super::Half,
        mixes: *const super::Half,
        base: *const super::Half,
        scale: *const super::Half,
        weight: *const super::Half,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        out: *mut super::Half,
        num_tokens: i32,
        hidden_size: i32,
        mix_dim: i32,
        hc_mult: i32,
        params_eps: f32,
        sinkhorn_iters: i32,
        norm_eps: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_mhc_params_bench_cuda(
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
        block_dim: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_mhc_pre_rms_norm_bench_cuda(
        residual: *const super::Half,
        pre: *const f32,
        weight: *const super::Half,
        out: *mut super::Half,
        num_tokens: i32,
        hidden_size: i32,
        hc_mult: i32,
        eps: f32,
        block_dim: i32,
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

    pub fn dsv4_mhc_lane_mean_cuda(
        stream: *const super::Half,
        out: *mut super::Half,
        num_tokens: i32,
        hidden_size: i32,
        hc_mult: i32,
        out_stride: i32,
        tap_offset: i32,
        cuda_stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_mtp_add_eproj_hproj_cuda(
        e_proj: *const super::Half,
        h_proj: *const super::Half,
        out_stream: *mut super::Half,
        hidden_size: i32,
        hc_mult: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_oproj_group_gather_cuda(
        src: *const super::Half,
        dst: *mut super::Half,
        num_tokens: i32,
        groups: i32,
        cols_per_group: i32,
        group: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_oproj_group_scatter_cuda(
        src: *const super::Half,
        dst: *mut super::Half,
        num_tokens: i32,
        groups: i32,
        rows_per_group: i32,
        group: i32,
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

    pub fn dsv4_prepare_qk_start_pos_ptr_cuda(
        q_raw: *const super::Half,
        k_raw: *const super::Half,
        q_out: *mut super::Half,
        k_out: *mut super::Half,
        num_tokens: i32,
        local_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        start_pos_ptr: *const i32,
        rms_eps: f32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_prepare_qk_fused_start_pos_ptr_cuda(
        q_raw: *const super::Half,
        k_raw: *const super::Half,
        q_out: *mut super::Half,
        k_out: *mut super::Half,
        num_tokens: i32,
        local_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        start_pos_ptr: *const i32,
        rms_eps: f32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_prepare_qk_fused_batch_start_pos_cuda(
        q_raw: *const super::Half,
        k_raw: *const super::Half,
        q_out: *mut super::Half,
        k_out: *mut super::Half,
        num_tokens: i32,
        local_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        start_pos: *const i32,
        rms_eps: f32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
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

    pub fn dsv4_update_window_cache_start_pos_ptr_cuda(
        k_new: *const super::Half,
        window_cache: *mut super::Half,
        num_tokens: i32,
        start_pos: *const i32,
        sliding_window: i32,
        head_dim: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Pointer-array batched SW-window write: ONE launch over `n` rows whose
    /// `k_prepared` (`k_arr[r]`) and SW ring cache (`cache_arr[r]`) buffers are
    /// NOT contiguous. Each row writes its single new key into its own ring at
    /// slot `start_pos[r] % sliding_window`. Replaces n single-row
    /// [`dsv4_update_window_cache_start_pos_ptr_cuda`] calls (byte-identical).
    pub fn dsv4_update_window_cache_batched_ptr_cuda(
        k_arr: *const *const super::Half,
        cache_arr: *const *mut super::Half,
        n: i32,
        start_pos: *const i32,
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
        // Elements-per-page stride for `prev_overlap_kv/score` pool
        // addressing. `0` = per-slot single-register buffer; `ratio*head_dim`
        // = shared, page-addressable pool.
        overlap_page_stride: i32,
        eps: f32,
        rope_dim: i32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_compressor_fp32_prefill_probe_cuda(
        kv_raw: *const f32,
        score_raw: *const f32,
        ape: *const f32,
        norm: *const super::Half,
        pending_kv: *mut f32,
        pending_score: *mut f32,
        prev_overlap_kv: *mut f32,
        prev_overlap_score: *mut f32,
        prev_overlap_kv_bf16: *mut super::Half,
        prev_overlap_score_bf16: *mut super::Half,
        pending_kv_bf16: *mut super::Half,
        pending_score_bf16: *mut super::Half,
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
        overlap_page_stride: i32,
        eps: f32,
        rope_dim: i32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// FP32-carry reseed: bf16 → f32 upcast of a compressor state's four carry
    /// buffers (pending kv/score = `pending_elems` each, prev_overlap kv/score
    /// = `prev_elems` each). Run before an FP32 probe whose bf16 carry advanced
    /// since the last probe (decode lane / prefix restore / reset).
    pub fn dsv4_compressor_fp32_carry_reseed_cuda(
        pending_kv_bf16: *const super::Half,
        pending_score_bf16: *const super::Half,
        prev_kv_bf16: *const super::Half,
        prev_score_bf16: *const super::Half,
        pending_kv: *mut f32,
        pending_score: *mut f32,
        prev_kv: *mut f32,
        prev_score: *mut f32,
        pending_elems: i32,
        prev_elems: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_compressor_update_start_pos_ptr_cuda(
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
        start_pos: *const i32,
        head_dim: i32,
        ratio: i32,
        width: i32,
        overlap: i32,
        overlap_page_stride: i32,
        eps: f32,
        rope_dim: i32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Batched decode compressor update: ONE `<<<n, BLOCK>>>` launch replacing n
    /// per-row [`dsv4_compressor_update_start_pos_ptr_cuda`] calls. Each row `r`
    /// reads its per-slot ring-state buffers from the host-gathered device
    /// pointer arrays (`*_arr[r]`) and its decode position from `start_pos_arr[r]`.
    /// `kv_raw`/`score_raw` are the batched m=N prepass outputs `[width, n]`
    /// (token-major). `ape`/`norm` are the SHARED compressor weights. Math is
    /// byte-identical to n single-row launches.
    pub fn dsv4_compressor_update_batched_start_pos_ptr_cuda(
        kv_raw: *const super::Half,
        score_raw: *const super::Half,
        ape: *const super::Half,
        norm: *const super::Half,
        pending_kv_arr: *const *mut super::Half,
        pending_score_arr: *const *mut super::Half,
        prev_overlap_kv_arr: *const *mut super::Half,
        prev_overlap_score_arr: *const *mut super::Half,
        compressed_arr: *const *mut super::Half,
        n: i32,
        num_tokens: i32,
        start_pos_arr: *const i32,
        head_dim: i32,
        ratio: i32,
        width: i32,
        overlap: i32,
        // Uniform across all n rows (one launch is always one (layer,
        // compress_ratio) class).
        overlap_page_stride: i32,
        eps: f32,
        rope_dim: i32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// In-place attention-output inverse-rope for the FlashMLA decode/prefill
    /// paths. The FlashMLA SM90 sparse decode/prefill kernels do NOT un-rotate
    /// the output rope tail, so callers must apply this explicitly. `out` is bf16
    /// (u16 bits), layout [token_count, local_heads, head_dim] with head_dim
    /// contiguous and rope
    /// tail = the last `rope_dim` cols; abs_pos = start_pos + token. NEVER call
    /// this on the legacy hybrid path (double-apply).
    pub fn arle_dsv4_output_inverse_rope_cuda(
        out: *mut super::Half,
        token_count: i32,
        local_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        start_pos: i32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Same as [`arle_dsv4_output_inverse_rope_cuda`], but reads the base
    /// absolute decode position from a stable device pointer for CUDA graph
    /// replay.
    pub fn arle_dsv4_output_inverse_rope_start_pos_ptr_cuda(
        out: *mut super::Half,
        token_count: i32,
        local_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        start_pos_ptr: *const i32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn arle_dsv4_output_inverse_rope_batch_start_pos_cuda(
        out: *mut super::Half,
        token_count: i32,
        local_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        start_pos: *const i32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Pointer-array batched output inverse-RoPE: ONE launch over `n` rows whose
    /// `local_attn` buffers are NOT contiguous. `out_arr[r]` is row `r`'s
    /// `[local_width, 1]` buffer base, `start_pos[r]` its absolute decode
    /// position. Replaces n single-row
    /// [`arle_dsv4_output_inverse_rope_start_pos_ptr_cuda`] calls; per-row math
    /// byte-identical.
    pub fn arle_dsv4_output_inverse_rope_batched_ptr_cuda(
        out_arr: *const *mut super::Half,
        n: i32,
        local_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        start_pos: *const i32,
        rope_base: f32,
        original_seq_len: i32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
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

    // SGLang sparse_mla_q8kv8_prefill_sm90: native FP8 (q8 x kv8) sparse MLA
    // prefill. q/kv are fp8 device pointers (per-tensor scale 1.0); out is
    // bf16. d_qk ∈ {512, 576}, d_v = 512, h_kv = 1. See
    // csrc/attention/arle_q8kv8_prefill_shim.cu.
    pub fn arle_q8kv8_sparse_prefill_fwd(
        q: *const u8,
        kv: *const u8,
        indices: *const i32,
        q_scale: *const f32,
        kv_scale: *const f32,
        attn_sink: *const f32,
        topk_length: *const i32,
        out: *mut super::Half,
        max_logits: *mut f32,
        lse: *mut f32,
        s_q: i32,
        s_kv: i32,
        h_q: i32,
        d_qk: i32,
        topk: i32,
        sm_scale: f32,
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
    // expected layout. Tracked under `--dsv4-flashmla-decode` (default
    // OFF).**
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

    pub fn dsv4_dsa_fused_q_indexer_rope_hadamard_quant_cuda(
        q_input: *const super::Half,
        q_fp8: *mut u8,
        weight: *const super::Half,
        weights_out: *mut f32,
        weight_scale: f32,
        freqs_cis: *const f32,
        positions: *const i32,
        batch_size: i32,
        num_heads: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_dsa_hadamard128_bf16_cuda(
        input: *const super::Half,
        output: *mut super::Half,
        rows: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_dsa_fused_store_index_k_cache_cuda(
        key: *const super::Half,
        index_k_with_scale: *mut u8,
        out_cache_loc: *const i64,
        num_tokens: i32,
        page_size: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Batched (grid.y=slot) Hadamard rotate: ONE `<<<dim3(blocks_per_slot, n)>>>`
    /// launch replacing n per-row [`dsv4_dsa_hadamard128_bf16_cuda`] calls. Each
    /// slot's base ptr / row offsets / row count come from the host-gathered
    /// device pointer/offset arrays (`*_arr[slot]`); `max_rows` sizes the x-grid.
    /// Math byte-identical to n single-row launches.
    pub fn dsv4_dsa_hadamard128_batched_cuda(
        keys_src_arr: *const *const super::Half,
        src_ring_row_arr: *const i32,
        rotated_dst_arr: *const *mut super::Half,
        dst_row_arr: *const i32,
        newly_packed_arr: *const i32,
        n: i32,
        max_rows: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    /// Batched (grid.y=slot) FP8 fused-store: ONE `<<<dim3(blocks_per_slot, n)>>>`
    /// launch replacing n per-row [`dsv4_dsa_fused_store_index_k_cache_cuda`]
    /// calls. Each slot's rotated-key base / cache band base / slot-local
    /// cache-loc array come from the host-gathered device arrays (the cache-loc
    /// array is `*const *const i64` — an array OF i64-ptrs). `max_tokens` sizes
    /// the x-grid. Math byte-identical to n single-row launches.
    pub fn dsv4_dsa_fused_store_index_k_cache_batched_cuda(
        key_arr: *const *const super::Half,
        cache_arr: *const *mut u8,
        out_cache_loc_arr: *const *const i64,
        newly_packed_arr: *const i32,
        n: i32,
        max_tokens: i32,
        page_size: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_dsa_fill_context_lens_positions_start_pos_cuda(
        context_lens: *mut i32,
        positions: *mut i32,
        start_pos: *const i32,
        token_offset: i32,
        batch_size: i32,
        key_count: i32,
        ratio: i32,
        stream: super::CUstream,
    ) -> super::CUresult;

    pub fn dsv4_deepseek_v4_topk_transform_cuda(
        scores: *const f32,
        seq_lens: *const i32,
        page_table: *const i32,
        page_indices: *mut i32,
        raw_indices: *mut i32,
        score_stride: i64,
        page_table_stride: i64,
        output_stride: i64,
        batch_size: i32,
        topk: i32,
        page_size: i32,
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

    /// Chain-verify sparse indices for FlashMLA prefill. `positions` is the
    /// per-row absolute position and `ancestors` is `[s_q, max_anc]` prefix chunk
    /// rows with `-1` padding. This is the top-1 chain verifier, not a complete
    /// top-k tree verifier.
    pub fn arle_flashmla_chain_verify_build_indices(
        indices: *mut i32,
        topk_length: *mut i32,
        positions: *const i32,
        ancestors: *const i32,
        max_anc: i32,
        selected: *const i32,
        s_q: i32,
        start_pos: i32,
        sw_window: i32,
        index_topk: i32,
        max_compressed: i32,
        topk_unified: i32,
        compressed_count: i32,
        compress_ratio: i32,
        stream: super::CUstream,
    ) -> super::CUresult;
}

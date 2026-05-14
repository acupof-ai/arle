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

    pub fn dsv4_swa_attention_cuda(
        q: *const super::Half,
        k_new: *const super::Half,
        window_cache: *const super::Half,
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
        window_cache: *const super::Half,
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
}

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
}

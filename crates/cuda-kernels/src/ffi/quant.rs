use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub fn cast_bf16_to_f32_cuda(
        r#in: *const Half,
        out: *mut f32,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn cast_f32_to_bf16_cuda(
        r#in: *const f32,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn turboquant_lloyd_max(
        centroids: *mut f32,
        boundaries: *mut f32,
        num_levels: i32,
        head_dim: i32,
        max_iters: i32,
    );

    pub fn turboquant_generate_rotation(Pi: *mut f32, head_dim: i32, seed: u64);

    pub fn turboquant_generate_signs(signs: *mut i8, head_dim: i32, seed: u64);

}

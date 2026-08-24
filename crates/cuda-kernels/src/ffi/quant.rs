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

}

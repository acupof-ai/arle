#[allow(dead_code)]
unsafe extern "C" {
    pub fn cublas_init();
    pub fn autotune_all_cached_gemms_cuda(stream: super::CUstream);
}

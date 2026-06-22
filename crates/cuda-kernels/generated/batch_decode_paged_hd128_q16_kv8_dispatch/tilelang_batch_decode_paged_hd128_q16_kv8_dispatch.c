#include <cuda.h>
#include <stdint.h>

CUresult tilelang_batch_decode_paged_hd128_q16_kv8_run_sm100_cuda(uint16_t *q, const int32_t *q_indptr, uint16_t *k_pool, uint16_t *v_pool, const int32_t *kv_indptr, const int32_t *kv_indices, const int32_t *kv_last_page_len, uint16_t *o, int32_t batch_size, int32_t total_q_tokens, int32_t max_qlen, int32_t num_pages, int32_t total_pages, int32_t num_q_heads, int32_t num_kv_heads, int32_t page_size, float sm_scale, CUstream stream);
CUresult tilelang_batch_decode_paged_hd128_q16_kv8_run_sm120_cuda(uint16_t *q, const int32_t *q_indptr, uint16_t *k_pool, uint16_t *v_pool, const int32_t *kv_indptr, const int32_t *kv_indices, const int32_t *kv_last_page_len, uint16_t *o, int32_t batch_size, int32_t total_q_tokens, int32_t max_qlen, int32_t num_pages, int32_t total_pages, int32_t num_q_heads, int32_t num_kv_heads, int32_t page_size, float sm_scale, CUstream stream);

static __thread int g_sm_pack = -1;

static int load_sm_pack(void) {
    int major = 0, minor = 0;
    CUdevice dev = 0;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) return -1;
    if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS) return -1;
    if (cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS) return -1;
    return major * 10 + minor;
}

CUresult tilelang_batch_decode_paged_hd128_q16_kv8_run_cuda(uint16_t *q, const int32_t *q_indptr, uint16_t *k_pool, uint16_t *v_pool, const int32_t *kv_indptr, const int32_t *kv_indices, const int32_t *kv_last_page_len, uint16_t *o, int32_t batch_size, int32_t total_q_tokens, int32_t max_qlen, int32_t num_pages, int32_t total_pages, int32_t num_q_heads, int32_t num_kv_heads, int32_t page_size, float sm_scale, CUstream stream) {
    int sm = g_sm_pack;
    if (sm < 0) {
        sm = load_sm_pack();
        if (sm < 0) return CUDA_ERROR_NOT_SUPPORTED;
        g_sm_pack = sm;
    }
    switch (sm) {
        case 100: return tilelang_batch_decode_paged_hd128_q16_kv8_run_sm100_cuda(q, q_indptr, k_pool, v_pool, kv_indptr, kv_indices, kv_last_page_len, o, batch_size, total_q_tokens, max_qlen, num_pages, total_pages, num_q_heads, num_kv_heads, page_size, sm_scale, stream);
        case 120: return tilelang_batch_decode_paged_hd128_q16_kv8_run_sm120_cuda(q, q_indptr, k_pool, v_pool, kv_indptr, kv_indices, kv_last_page_len, o, batch_size, total_q_tokens, max_qlen, num_pages, total_pages, num_q_heads, num_kv_heads, page_size, sm_scale, stream);
        default: return CUDA_ERROR_NOT_SUPPORTED;
    }
}

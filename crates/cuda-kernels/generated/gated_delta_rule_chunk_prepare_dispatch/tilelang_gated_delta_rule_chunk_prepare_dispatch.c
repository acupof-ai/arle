#include <cuda.h>
#include <stdint.h>

CUresult gated_delta_rule_prefill_chunk_prepare_sm100_cuda(const uint16_t* qkv, const uint16_t* b_proj, const uint16_t* a_proj, const uint16_t* dt_bias, const float* a_log, uint16_t* q_out, uint16_t* k_out, uint16_t* v_out, float* g_out, float* beta_out, int32_t num_key_heads, int32_t num_value_heads, int32_t qkv_dim, int32_t seq_len, CUstream stream);
CUresult gated_delta_rule_prefill_chunk_prepare_sm120_cuda(const uint16_t* qkv, const uint16_t* b_proj, const uint16_t* a_proj, const uint16_t* dt_bias, const float* a_log, uint16_t* q_out, uint16_t* k_out, uint16_t* v_out, float* g_out, float* beta_out, int32_t num_key_heads, int32_t num_value_heads, int32_t qkv_dim, int32_t seq_len, CUstream stream);

static __thread int g_sm_pack = -1;

static int load_sm_pack(void) {
    int major = 0, minor = 0;
    CUdevice dev = 0;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) return -1;
    if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS) return -1;
    if (cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS) return -1;
    return major * 10 + minor;
}

CUresult gated_delta_rule_prefill_chunk_prepare_cuda(const uint16_t* qkv, const uint16_t* b_proj, const uint16_t* a_proj, const uint16_t* dt_bias, const float* a_log, uint16_t* q_out, uint16_t* k_out, uint16_t* v_out, float* g_out, float* beta_out, int32_t num_key_heads, int32_t num_value_heads, int32_t qkv_dim, int32_t seq_len, CUstream stream) {
    int sm = g_sm_pack;
    if (sm < 0) {
        sm = load_sm_pack();
        if (sm < 0) return CUDA_ERROR_NOT_SUPPORTED;
        g_sm_pack = sm;
    }
    switch (sm) {
        case 100: return gated_delta_rule_prefill_chunk_prepare_sm100_cuda(qkv, b_proj, a_proj, dt_bias, a_log, q_out, k_out, v_out, g_out, beta_out, num_key_heads, num_value_heads, qkv_dim, seq_len, stream);
        case 120: return gated_delta_rule_prefill_chunk_prepare_sm120_cuda(qkv, b_proj, a_proj, dt_bias, a_log, q_out, k_out, v_out, g_out, beta_out, num_key_heads, num_value_heads, qkv_dim, seq_len, stream);
        default: return CUDA_ERROR_NOT_SUPPORTED;
    }
}

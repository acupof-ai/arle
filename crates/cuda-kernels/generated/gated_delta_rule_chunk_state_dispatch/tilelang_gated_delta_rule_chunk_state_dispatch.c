#include <cuda.h>
#include <stdint.h>

CUresult gated_delta_rule_prefill_chunk_state_sm100_cuda(const uint16_t* k, const uint16_t* w, const uint16_t* u, const float* g_cumsum, const float* initial_state, float* chunk_state, uint16_t* v_new, float* final_state, int32_t seq_len, int32_t num_value_heads, CUstream stream);
CUresult gated_delta_rule_prefill_chunk_state_sm120_cuda(const uint16_t* k, const uint16_t* w, const uint16_t* u, const float* g_cumsum, const float* initial_state, float* chunk_state, uint16_t* v_new, float* final_state, int32_t seq_len, int32_t num_value_heads, CUstream stream);

static __thread int g_sm_pack = -1;

static int load_sm_pack(void) {
    int major = 0, minor = 0;
    CUdevice dev = 0;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) return -1;
    if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS) return -1;
    if (cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS) return -1;
    return major * 10 + minor;
}

CUresult gated_delta_rule_prefill_chunk_state_cuda(const uint16_t* k, const uint16_t* w, const uint16_t* u, const float* g_cumsum, const float* initial_state, float* chunk_state, uint16_t* v_new, float* final_state, int32_t seq_len, int32_t num_value_heads, CUstream stream) {
    int sm = g_sm_pack;
    if (sm < 0) {
        sm = load_sm_pack();
        if (sm < 0) return CUDA_ERROR_NOT_SUPPORTED;
        g_sm_pack = sm;
    }
    switch (sm) {
        case 100: return gated_delta_rule_prefill_chunk_state_sm100_cuda(k, w, u, g_cumsum, initial_state, chunk_state, v_new, final_state, seq_len, num_value_heads, stream);
        case 120: return gated_delta_rule_prefill_chunk_state_sm120_cuda(k, w, u, g_cumsum, initial_state, chunk_state, v_new, final_state, seq_len, num_value_heads, stream);
        default: return CUDA_ERROR_NOT_SUPPORTED;
    }
}

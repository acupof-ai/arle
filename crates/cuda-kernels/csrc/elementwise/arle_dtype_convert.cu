// ARLE dtype-convert helpers — small device kernels used at model load
// (e.g. attn_sink f32 mirror) and other rare bf16↔f32 hops. Not a hot
// path; one-shot, one-block-per-N kernel is sufficient.
//
// First user: DSv4 FlashMLA SM90 sparse prefill needs `float[h_q]` for
// `attn_sink`; ARLE loads it as bf16. Build the f32 mirror once at
// model load via arle_bf16_to_f32_cuda.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

__global__ void arle_bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

} // namespace

extern "C" cudaError_t arle_bf16_to_f32_cuda(
    const __nv_bfloat16* src,
    float* dst,
    int n,
    cudaStream_t stream
) {
    if (n <= 0) return cudaSuccess;
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    arle_bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(src, dst, n);
    return cudaGetLastError();
}

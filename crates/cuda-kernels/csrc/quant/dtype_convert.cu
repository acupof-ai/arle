// BF16 ↔ FP16 conversion kernels for Marlin integration.
// Marlin kernel uses FP16 (half), our engine uses BF16 (__nv_bfloat16).
// These kernels convert between them with minimal overhead.
// For decode (batch=1, hidden_dim=2560): ~1μs per conversion — negligible.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define CONVERT_BLOCK 256

__global__ void bf16_to_fp16_kernel(
    const __nv_bfloat16* __restrict__ in,
    __half* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * CONVERT_BLOCK + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__bfloat162float(in[idx]));
    }
}

__global__ void fp16_to_bf16_kernel(
    const __half* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * CONVERT_BLOCK + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(__half2float(in[idx]));
    }
}

extern "C" {

cudaError_t bf16_to_fp16_cuda(
    const __nv_bfloat16* in, __half* out, int n, cudaStream_t stream)
{
    int grid = (n + CONVERT_BLOCK - 1) / CONVERT_BLOCK;
    bf16_to_fp16_kernel<<<grid, CONVERT_BLOCK, 0, stream>>>(in, out, n);
    return cudaGetLastError();
}

cudaError_t fp16_to_bf16_cuda(
    const __half* in, __nv_bfloat16* out, int n, cudaStream_t stream)
{
    int grid = (n + CONVERT_BLOCK - 1) / CONVERT_BLOCK;
    fp16_to_bf16_kernel<<<grid, CONVERT_BLOCK, 0, stream>>>(in, out, n);
    return cudaGetLastError();
}

}  // extern "C"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void quantize_bf16_rows_to_int8_kernel(
    const __nv_bfloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int cols) {
  extern __shared__ float smem[];
  int row = blockIdx.x;
  if (row >= rows) return;

  const __nv_bfloat16* in_row = input + (size_t)row * cols;
  int8_t* out_row = output + (size_t)row * cols;

  float local_max = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(__bfloat162float(in_row[col])));
  }
  smem[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  float scale = smem[0] > 0.0f ? smem[0] / 127.0f : 1.0f;
  if (threadIdx.x == 0) {
    scales[row] = scale;
  }

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float qf = nearbyintf(__bfloat162float(in_row[col]) / scale);
    qf = fminf(127.0f, fmaxf(-128.0f, qf));
    out_row[col] = static_cast<int8_t>(qf);
  }
}

extern "C" cudaError_t quantize_bf16_rows_to_int8_cuda(
    const __nv_bfloat16* input,
    int8_t* output,
    float* scales,
    int rows,
    int cols,
    cudaStream_t stream) {
  constexpr int threads = 256;
  dim3 grid(rows);
  dim3 block(threads);
  size_t smem = threads * sizeof(float);
  quantize_bf16_rows_to_int8_kernel<<<grid, block, smem, stream>>>(
      input, output, scales, rows, cols);
  return cudaGetLastError();
}

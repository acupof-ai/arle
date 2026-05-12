#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float warp_reduce_max_abs(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

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
  local_max = warp_reduce_max_abs(local_max);

  int lane_id = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int num_warps = (blockDim.x + 31) >> 5;
  if (lane_id == 0) {
    smem[warp_id] = local_max;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_max = lane_id < num_warps ? smem[lane_id] : 0.0f;
    block_max = warp_reduce_max_abs(block_max);
    if (lane_id == 0) {
      smem[0] = block_max;
    }
  }
  __syncthreads();

  float scale = smem[0] > 0.0f ? smem[0] / 127.0f : 1.0f;
  if (threadIdx.x == 0) {
    scales[row] = scale;
  }

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    int q = __float2int_rn(__bfloat162float(in_row[col]) / scale);
    q = max(-128, min(127, q));
    out_row[col] = static_cast<int8_t>(q);
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
  size_t smem = ((threads + 31) / 32) * sizeof(float);
  quantize_bf16_rows_to_int8_kernel<<<grid, block, smem, stream>>>(
      input, output, scales, rows, cols);
  return cudaGetLastError();
}

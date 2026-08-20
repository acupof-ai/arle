// Rank-1 column scale over a bf16 [rows, cols] buffer: data[r, c] *= scales[c].
//
// A per-channel weight scale commutes with the contraction, so a GEMM may run
// against the raw quantized bytes and take the scale afterwards. Marlin does
// this inside its kernel (scales the FP32 accumulator after the K loop);
// DeepGEMM's dense NT epilogue emits bf16 and exposes no value hook, so its
// caller applies the scale here instead.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int COLUMN_SCALE_THREADS = 256;
// gridDim.y hardware limit; rows above it are covered by the outer stride loop.
constexpr int COLUMN_SCALE_MAX_GRID_Y = 65535;

__global__ void scale_columns_bf16_kernel(
    __nv_bfloat16* __restrict__ data,
    const float* __restrict__ scales,
    int rows,
    int cols) {
  const int col_stride = static_cast<int>(gridDim.x * blockDim.x);
  const int row_stride = static_cast<int>(gridDim.y);
  for (int row = static_cast<int>(blockIdx.y); row < rows; row += row_stride) {
    const int64_t base = static_cast<int64_t>(row) * cols;
    for (int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         col < cols; col += col_stride) {
      const float value = __bfloat162float(data[base + col]) * scales[col];
      data[base + col] = __float2bfloat16(value);
    }
  }
}

}  // namespace

extern "C" CUresult scale_columns_bf16_cuda(
    __nv_bfloat16* data,
    const float* scales,
    int rows,
    int cols,
    CUstream stream) {
  if (rows < 0 || cols <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows == 0) return CUDA_SUCCESS;
  if (data == nullptr || scales == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int grid_x = (cols + COLUMN_SCALE_THREADS - 1) / COLUMN_SCALE_THREADS;
  const int grid_y = rows < COLUMN_SCALE_MAX_GRID_Y ? rows : COLUMN_SCALE_MAX_GRID_Y;
  scale_columns_bf16_kernel<<<dim3(grid_x, grid_y), COLUMN_SCALE_THREADS, 0,
                              (cudaStream_t)stream>>>(data, scales, rows, cols);
  return (CUresult)cudaGetLastError();
}

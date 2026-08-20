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
  // Eight bf16 per thread: one 16-byte load/store instead of eight 2-byte ones,
  // and the scale vector is read as two float4. The caller guarantees n % 8 == 0
  // (`fp8_per_channel_deepgemm_shape`), and `data` is a row-major [rows, cols]
  // allocation, so both are 16-byte aligned.
  const int vec_cols = cols >> 3;
  const int col_stride = static_cast<int>(gridDim.x * blockDim.x);
  const int row_stride = static_cast<int>(gridDim.y);
  for (int row = static_cast<int>(blockIdx.y); row < rows; row += row_stride) {
    uint4* __restrict__ row_ptr =
        reinterpret_cast<uint4*>(data + static_cast<int64_t>(row) * cols);
    for (int v = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); v < vec_cols;
         v += col_stride) {
      uint4 packed = row_ptr[v];
      const float4 s0 = reinterpret_cast<const float4*>(scales)[v * 2];
      const float4 s1 = reinterpret_cast<const float4*>(scales)[v * 2 + 1];
      __nv_bfloat162* lanes = reinterpret_cast<__nv_bfloat162*>(&packed);
      lanes[0] = __floats2bfloat162_rn(__bfloat162float(lanes[0].x) * s0.x,
                                       __bfloat162float(lanes[0].y) * s0.y);
      lanes[1] = __floats2bfloat162_rn(__bfloat162float(lanes[1].x) * s0.z,
                                       __bfloat162float(lanes[1].y) * s0.w);
      lanes[2] = __floats2bfloat162_rn(__bfloat162float(lanes[2].x) * s1.x,
                                       __bfloat162float(lanes[2].y) * s1.y);
      lanes[3] = __floats2bfloat162_rn(__bfloat162float(lanes[3].x) * s1.z,
                                       __bfloat162float(lanes[3].y) * s1.w);
      row_ptr[v] = packed;
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
  if (rows < 0 || cols <= 0 || (cols & 7) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows == 0) return CUDA_SUCCESS;
  if (data == nullptr || scales == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int vec_cols = cols >> 3;
  const int grid_x = (vec_cols + COLUMN_SCALE_THREADS - 1) / COLUMN_SCALE_THREADS;
  const int grid_y = rows < COLUMN_SCALE_MAX_GRID_Y ? rows : COLUMN_SCALE_MAX_GRID_Y;
  scale_columns_bf16_kernel<<<dim3(grid_x, grid_y), COLUMN_SCALE_THREADS, 0,
                              (cudaStream_t)stream>>>(data, scales, rows, cols);
  return (CUresult)cudaGetLastError();
}

// NVFP4 (E2M1 float4 + E8M0 block scales) → W4AFP8 (signed INT4 + BF16
// interleaved scales) load-time conversion for the SGLang CUTLASS kernel.
//
// The 0731 checkpoint ships routed experts as E2M1 packed 2-per-byte with
// F8_E8M0 per-1×32-block scales. The SGLang W4A8 MMA expects signed INT4
// two's complement with BF16 per-1×128-group interleaved scales. This kernel
// converts one expert's weights on GPU at load time — the checkpoint on disk
// is untouched.
//
// Weight: same shape [N, K//2] in and out (2 nibbles per byte, low = even K).
// Scales: [N, K//32] E8M0 → [K//512, N*4] BF16 interleaved.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "../../common.cuh"

namespace {

__device__ __forceinline__ float decode_e8m0(uint8_t bits) {
  uint32_t raw = static_cast<uint32_t>(bits) << 23;
  return __uint_as_float(raw);
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

// Grid: (N, K/128). Block: 128 threads (one per K element in the group).
__global__ void nvfp4_to_w4afp8_kernel(
    const int8_t* __restrict__ src_weight,   // [N, K//2] packed E2M1
    const uint8_t* __restrict__ src_scales,  // [N, K//32] E8M0
    int8_t* __restrict__ dst_weight,         // [N, K//2] packed INT4
    __nv_bfloat16* __restrict__ dst_scales,  // [K//512, N*4] BF16 interleaved
    int N,
    int K) {
  const int n = blockIdx.x;
  const int g = blockIdx.y;  // 128-group index
  const int tid = threadIdx.x;

  const int stored_k = K / 2;       // packed bytes per row
  const int scale_cols = K / 32;    // E8M0 scales per row

  // Decode E2M1: 128 values = 64 packed bytes starting at g*64.
  const int src_byte_offset = n * stored_k + g * 64;
  const uint8_t packed = reinterpret_cast<const uint8_t*>(src_weight)[src_byte_offset + tid / 2];
  const uint8_t nibble = (tid & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
  const float w = arle_decode_fp4_e2m1(nibble);

  // E8M0 scale: one per 32 logical elements → index g*4 + tid/32.
  const float scale = decode_e8m0(src_scales[n * scale_cols + g * 4 + tid / 32]);
  const float value = w * scale;

  // Block-wide amax (4 warps → shared → warp 0 reduces).
  __shared__ float smem[4];
  float local_max = fabsf(value);
  local_max = warp_reduce_max(local_max);
  const int warp_id = tid / 32;
  const int lane = tid % 32;
  if (lane == 0) smem[warp_id] = local_max;
  __syncthreads();
  if (warp_id == 0) {
    float v = (lane < 4) ? smem[lane] : 0.0f;
    v = warp_reduce_max(v);
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  const float block_max = smem[0];

  // Per-group BF16 scale: amax / 8 (signed INT4 range [-8, 7]).
  const float group_scale = fmaxf(block_max, 1e-12f) / 8.0f;

  // Quantize to signed INT4.
  int q = __float2int_rn(value / group_scale);
  q = q < -8 ? -8 : (q > 7 ? 7 : q);
  const uint8_t uq = static_cast<uint8_t>(q) & 0x0f;

  // Pack: even tid → low nibble, odd tid → high nibble. Race-free via
  // separate shared arrays, combined by threads 0-63.
  __shared__ uint8_t pack_lo[64];
  __shared__ uint8_t pack_hi[64];
  if ((tid & 1) == 0) {
    pack_lo[tid / 2] = uq;
  } else {
    pack_hi[tid / 2] = uq;
  }
  __syncthreads();
  if (tid < 64) {
    reinterpret_cast<uint8_t*>(dst_weight)[src_byte_offset + tid] =
        pack_lo[tid] | (pack_hi[tid] << 4);
  }

  // Write BF16 scale to interleaved [K//512, N*4]: group g → chunk g/4, pos g%4.
  if (tid == 0) {
    dst_scales[(g / 4) * N * 4 + n * 4 + (g % 4)] =
        __float2bfloat16(group_scale);
  }
}

}  // namespace

extern "C" cudaError_t nvfp4_to_w4afp8(
    const int8_t* src_weight,
    const uint8_t* src_scales,
    int8_t* dst_weight,
    uint8_t* dst_scales,  // BF16 bytes
    int N,
    int K,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0 || K % 512 != 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(N, K / 128);
  dim3 block(128);
  nvfp4_to_w4afp8_kernel<<<grid, block, 0, stream>>>(
      src_weight,
      src_scales,
      dst_weight,
      reinterpret_cast<__nv_bfloat16*>(dst_scales),
      N,
      K);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  return cudaStreamSynchronize(stream);
}

// Paged KV cache transfer kernels.
//
// Adapted from SGLang's kvcacheio transfer kernel structure
// (sgl-kernel/csrc/kvcacheio/transfer.cu, Apache-2.0). This version keeps the
// same warp-per-item, 64-bit non-temporal copy strategy, but exposes a small C
// ABI for ARLE's Rust/cudarc runtime instead of PyTorch tensors.

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kDefaultWarpsPerBlock = 8;
constexpr int kMaxBlocks = 65535;

__device__ __forceinline__ void transfer_item_warp(
    int32_t lane_id,
    const void* src_addr,
    void* dst_addr,
    int64_t item_size_bytes) {
  const uint64_t* __restrict__ src64 =
      reinterpret_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst64 = reinterpret_cast<uint64_t*>(dst_addr);
  const int64_t chunks64 = item_size_bytes / static_cast<int64_t>(sizeof(uint64_t));

  for (int64_t j = lane_id; j < chunks64; j += kWarpSize) {
    uint64_t tmp;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src64 + j) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst64 + j), "l"(tmp) : "memory");
#else
    tmp = src64[j];
    dst64[j] = tmp;
#endif
  }

  const int64_t tail_start = chunks64 * static_cast<int64_t>(sizeof(uint64_t));
  const int64_t tail_bytes = item_size_bytes - tail_start;
  if (tail_bytes > 0) {
    const uint8_t* __restrict__ src8 =
        reinterpret_cast<const uint8_t*>(src_addr) + tail_start;
    uint8_t* __restrict__ dst8 = reinterpret_cast<uint8_t*>(dst_addr) + tail_start;
    for (int64_t j = lane_id; j < tail_bytes; j += kWarpSize) {
      dst8[j] = src8[j];
    }
  }
}

__global__ void transfer_kv_pages_layer_table_kernel(
    const uintptr_t* __restrict__ src_k_layers,
    const uintptr_t* __restrict__ dst_k_layers,
    const uintptr_t* __restrict__ src_v_layers,
    const uintptr_t* __restrict__ dst_v_layers,
    const int32_t* __restrict__ src_pages,
    const int32_t* __restrict__ dst_pages,
    int32_t num_pages,
    int32_t start_layer,
    int32_t num_layers,
    int64_t bytes_per_page,
    int64_t items_per_warp) {
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t lane_id = tid & (kWarpSize - 1);
  const int32_t warp_id = tid / kWarpSize;

  for (int64_t i = 0; i < items_per_warp; ++i) {
    const int64_t item_id = static_cast<int64_t>(warp_id) * items_per_warp + i;
    if (item_id >= num_pages) {
      break;
    }

    const int64_t src_offset = static_cast<int64_t>(src_pages[item_id]) * bytes_per_page;
    const int64_t dst_offset = static_cast<int64_t>(dst_pages[item_id]) * bytes_per_page;

    for (int32_t rel_layer = 0; rel_layer < num_layers; ++rel_layer) {
      const int32_t layer = start_layer + rel_layer;

      const char* src_k = reinterpret_cast<const char*>(src_k_layers[layer]) + src_offset;
      char* dst_k = reinterpret_cast<char*>(dst_k_layers[layer]) + dst_offset;
      transfer_item_warp(lane_id, src_k, dst_k, bytes_per_page);

      if (src_v_layers != nullptr && dst_v_layers != nullptr) {
        const char* src_v = reinterpret_cast<const char*>(src_v_layers[layer]) + src_offset;
        char* dst_v = reinterpret_cast<char*>(dst_v_layers[layer]) + dst_offset;
        transfer_item_warp(lane_id, src_v, dst_v, bytes_per_page);
      }
    }
  }
}

int64_t div_up_i64(int64_t x, int64_t y) { return (x + y - 1) / y; }

}  // namespace

extern "C" cudaError_t transfer_kv_pages_layer_table_cuda(
    const uintptr_t* src_k_layers,
    const uintptr_t* dst_k_layers,
    const uintptr_t* src_v_layers,
    const uintptr_t* dst_v_layers,
    const int32_t* src_pages,
    const int32_t* dst_pages,
    int32_t num_pages,
    int32_t start_layer,
    int32_t num_layers,
    int64_t bytes_per_page,
    int32_t num_warps_per_block,
    cudaStream_t stream) {
  if (num_pages <= 0 || num_layers <= 0 || bytes_per_page <= 0) {
    return cudaSuccess;
  }
  if (src_k_layers == nullptr || dst_k_layers == nullptr ||
      src_pages == nullptr || dst_pages == nullptr) {
    return cudaErrorInvalidValue;
  }

  const int32_t warps_per_block =
      num_warps_per_block > 0 ? num_warps_per_block : kDefaultWarpsPerBlock;
  if (warps_per_block <= 0 || warps_per_block > 32) {
    return cudaErrorInvalidValue;
  }

  const int64_t target_warps =
      static_cast<int64_t>(kMaxBlocks) * static_cast<int64_t>(warps_per_block);
  const int64_t items_per_warp = div_up_i64(num_pages, target_warps);
  const int64_t warps_needed = div_up_i64(num_pages, items_per_warp);
  const int32_t blocks =
      static_cast<int32_t>(div_up_i64(warps_needed, warps_per_block));
  const int32_t threads = warps_per_block * kWarpSize;

  transfer_kv_pages_layer_table_kernel<<<blocks, threads, 0, stream>>>(
      src_k_layers,
      dst_k_layers,
      src_v_layers,
      dst_v_layers,
      src_pages,
      dst_pages,
      num_pages,
      start_layer,
      num_layers,
      bytes_per_page,
      items_per_warp);
  return cudaGetLastError();
}

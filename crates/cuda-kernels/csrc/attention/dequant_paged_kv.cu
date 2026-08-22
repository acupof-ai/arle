// Dequantize ARLE's 1-byte quantized KV planes (FP8 e4m3 or INT8, KIVI
// per-(token, kv_head) f32 scales) to bf16.
//
// Entry point:
//   - dequantize_paged_kv_compact_cuda: page-table compaction for the FA3
//     quant shim — each (batch row b, logical page j) of the rectangular
//     page table lands in compact slot b*stride+j of the bf16 output pool
//     (HND per-page layout, the view FA3 strides), and `compact_table` is
//     written as the identity over those slots. No host reads and a fixed
//     launch shape per (batch, stride), so it replays under CUDA graph
//     capture.
//
// Conversion idioms mirror paged_attention_quantized_fa3.cu: software
// casts, valid on every arch this file is compiled for.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstring>

namespace {

__device__ __forceinline__ float dequant_value(const unsigned char* data,
                                               int64_t idx, bool is_fp8) {
    return is_fp8
               ? static_cast<float>(
                     reinterpret_cast<const __nv_fp8_e4m3*>(data)[idx])
               : static_cast<float>(reinterpret_cast<const int8_t*>(data)[idx]);
}

// Store 16 bf16 values (one uint4 pair) at `out + base`. The caller guarantees
// 16-element alignment (head_dim % 16 == 0, chunk index multiple of 16).
__device__ __forceinline__ void store_bf16_16(__nv_bfloat16* out, int64_t base,
                                              const float vals[16]) {
    __nv_bfloat162 packed[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        packed[k] = __floats2bfloat162_rn(vals[2 * k], vals[2 * k + 1]);
    }
    uint4 lo;
    uint4 hi;
    std::memcpy(&lo, &packed[0], sizeof(uint4));
    std::memcpy(&hi, &packed[4], sizeof(uint4));
    auto* dst = reinterpret_cast<uint4*>(out + base);
    dst[0] = lo;
    dst[1] = hi;
}

// One block per compact slot. Dequants one page (page_size * H * D elements,
// 16-element chunks) NHD durable -> HND FA3 view and writes the identity
// page-table entry.
__global__ void dequant_compact_kernel(
        const unsigned char* __restrict__ data,
        const float* __restrict__ scales,
        __nv_bfloat16* __restrict__ out,
        const int* __restrict__ page_table,
        int* __restrict__ compact_table,
        int num_kv_heads,
        int head_dim,
        int page_size,
        bool is_fp8) {
    const int slot = blockIdx.x;  // table is [batch, stride] row-major
    const int phys = page_table[slot];
    compact_table[slot] = slot;
    if (phys < 0) return;
    const int64_t page_elems =
        static_cast<int64_t>(page_size) * num_kv_heads * head_dim;
    const int64_t in_page_base =
        (static_cast<int64_t>(phys) * num_kv_heads * page_size) * head_dim;
    const int64_t out_base = static_cast<int64_t>(slot) * page_elems;
    for (int64_t e = static_cast<int64_t>(threadIdx.x) * 16; e < page_elems;
         e += static_cast<int64_t>(blockDim.x) * 16) {
        const int d = static_cast<int>(e % head_dim);
        const int64_t row = e / head_dim;  // HND-flat: (head, offset) row-major
        const int h = static_cast<int>(row / page_size);
        const int o = static_cast<int>(row % page_size);
        const float scale =
            scales[(static_cast<int64_t>(phys) * page_size + o) * num_kv_heads +
                   h];
        // NHD durable pool: [page, offset, head, dim] (token-major) — the
        // layout the quantize kernels write and the decode kernels read.
        const int64_t in_idx = in_page_base +
                               (static_cast<int64_t>(o) * num_kv_heads + h) *
                                   head_dim +
                               d;
        float vals[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            vals[k] = dequant_value(data, in_idx + k, is_fp8) * scale;
        }
        store_bf16_16(out, out_base + e, vals);
    }
}

}  // namespace

extern "C" {

cudaError_t dequantize_paged_kv_compact_cuda(const void* data,
                                             const float* scales,
                                             void* out,
                                             const int* page_table,
                                             int* compact_table,
                                             int batch,
                                             int page_table_stride,
                                             int num_kv_heads,
                                             int head_dim,
                                             int page_size,
                                             int is_fp8,
                                             cudaStream_t stream) {
    if (data == nullptr || scales == nullptr || out == nullptr ||
        page_table == nullptr || compact_table == nullptr || batch <= 0 ||
        page_table_stride <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        page_size <= 0 || (head_dim % 16) != 0) {
        return cudaErrorInvalidValue;
    }
    const int slots = batch * page_table_stride;
    constexpr int kThreads = 256;
    dequant_compact_kernel<<<slots, kThreads, 0, stream>>>(
        static_cast<const unsigned char*>(data), scales,
        static_cast<__nv_bfloat16*>(out), page_table, compact_table,
        num_kv_heads, head_dim, page_size, is_fp8 != 0);
    return cudaGetLastError();
}

}  // extern "C"

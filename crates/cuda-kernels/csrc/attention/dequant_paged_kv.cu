// Convert ARLE's 1-byte quantized KV planes (FP8 e4m3 or INT8, per-(token,
// kv_head) f32 scales) into an FA3 operand: bf16 (short context) or e4m3 with
// one descale per (batch row, kv_head) (long context, fp8 tensor-core rate).
//
// Entry points:
//   - dequantize_paged_kv_compact_cuda: the bf16 form of the compaction below.
//   - requantize_paged_kv_compact_e4m3_cuda: page-table compaction for the FA3
//     quant shim — each (batch row b, logical page j) of the rectangular page
//     table lands in compact slot b*stride+j of the e4m3 output pool (HND
//     per-page layout, the view FA3 strides), `compact_table` is written as
//     the identity over those slots, and `descale[b, h]` = the row's largest
//     per-token scale (the per-token quantiser puts one element at full
//     range, so this is the tight per-row bound). No host reads and a fixed
//     launch shape per (batch, stride), so it replays under CUDA graph
//     capture.
//   - quantize_q_e4m3_cuda: bf16 Q rows -> e4m3 with one descale per
//     (batch row, kv_head) over the GQA group.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cfloat>
#include <cstring>

namespace {

constexpr float kE4m3Max = 448.0f;
constexpr float kInt8Max = 127.0f;

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

__device__ __forceinline__ unsigned char to_e4m3(float v) {
    return static_cast<unsigned char>(
        __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float block_max(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nwarps = blockDim.x >> 5;
    v = (threadIdx.x < nwarps) ? red[threadIdx.x] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    }
    __syncthreads();
    return v;
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

// One block per (batch row, kv_head): max of the per-token scale over the
// row's valid tokens (`seqused_k[b]`, or `seqlen_k` when null).
__global__ void kv_descale_kernel(
        const float* __restrict__ scales,
        const int* __restrict__ page_table,
        const int* __restrict__ seqused_k,
        int seqlen_k,
        int page_table_stride,
        int num_kv_heads,
        int page_size,
        float value_max,
        float* __restrict__ descale) {
    __shared__ float red[32];
    const int b = blockIdx.x, h = blockIdx.y;
    const int len = seqused_k ? seqused_k[b] : seqlen_k;
    float m = 0.0f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        const int phys = page_table[b * page_table_stride + t / page_size];
        if (phys < 0) continue;
        m = fmaxf(m, scales[(static_cast<int64_t>(phys) * page_size + t % page_size) * num_kv_heads + h]);
    }
    m = block_max(m, red);
    if (threadIdx.x == 0) {
        // |value| <= value_max * scale_t <= value_max * m; the e4m3 operand is
        // value / descale with descale = value_max * m / 448.
        descale[b * num_kv_heads + h] = fmaxf(m * value_max / kE4m3Max, FLT_MIN);
    }
}

// One block per compact slot. Requants one page (page_size * H * D elements,
// 16-element chunks) NHD durable -> HND FA3 view and writes the identity
// page-table entry.
__global__ void requant_compact_kernel(
        const unsigned char* __restrict__ data,
        const float* __restrict__ scales,
        const float* __restrict__ descale,
        unsigned char* __restrict__ out,
        const int* __restrict__ page_table,
        int* __restrict__ compact_table,
        int page_table_stride,
        int num_kv_heads,
        int head_dim,
        int page_size,
        bool is_fp8) {
    const int slot = blockIdx.x;  // table is [batch, stride] row-major
    const int b = slot / page_table_stride;
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
            scales[(static_cast<int64_t>(phys) * page_size + o) * num_kv_heads + h] /
            descale[b * num_kv_heads + h];
        // NHD durable pool: [page, offset, head, dim] (token-major).
        const int64_t in_idx = in_page_base +
                               (static_cast<int64_t>(o) * num_kv_heads + h) * head_dim + d;
        uint4 packed;
        unsigned char* bytes = reinterpret_cast<unsigned char*>(&packed);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            bytes[k] = to_e4m3(dequant_value(data, in_idx + k, is_fp8) * scale);
        }
        *reinterpret_cast<uint4*>(out + out_base + e) = packed;
    }
}

// One block per (batch row, kv_head): absmax of the row's Q over the GQA
// group's heads. Rows are `[cu_seqlens_q[b], cu_seqlens_q[b+1])`, or all
// `total_q` rows when the batch is one contiguous row.
__global__ void q_descale_kernel(
        const __nv_bfloat16* __restrict__ q,
        const int* __restrict__ cu_seqlens_q,
        int total_q,
        int64_t q_row_stride,
        int64_t q_head_stride,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        float* __restrict__ descale) {
    __shared__ float red[32];
    const int b = blockIdx.x, hk = blockIdx.y;
    const int g = num_heads / num_kv_heads;
    const int r0 = cu_seqlens_q ? cu_seqlens_q[b] : 0;
    const int r1 = cu_seqlens_q ? cu_seqlens_q[b + 1] : total_q;
    const int64_t n = static_cast<int64_t>(r1 - r0) * g * head_dim;
    float m = 0.0f;
    for (int64_t e = threadIdx.x; e < n; e += blockDim.x) {
        const int d = static_cast<int>(e % head_dim);
        const int64_t rh = e / head_dim;
        const int hh = static_cast<int>(rh % g);
        const int64_t r = r0 + rh / g;
        m = fmaxf(m, fabsf(__bfloat162float(
                        q[r * q_row_stride + (hk * g + hh) * q_head_stride + d])));
    }
    m = block_max(m, red);
    if (threadIdx.x == 0) descale[b * num_kv_heads + hk] = fmaxf(m / kE4m3Max, FLT_MIN);
}

// Elementwise Q -> e4m3 into a contiguous [total_q, num_heads, head_dim] temp.
__global__ void q_quant_kernel(
        const __nv_bfloat16* __restrict__ q,
        const int* __restrict__ cu_seqlens_q,
        const float* __restrict__ descale,
        unsigned char* __restrict__ out,
        int total_q,
        int batch,
        int64_t q_row_stride,
        int64_t q_head_stride,
        int num_heads,
        int num_kv_heads,
        int head_dim) {
    const int64_t n = static_cast<int64_t>(total_q) * num_heads * head_dim;
    const int g = num_heads / num_kv_heads;
    for (int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < n;
         e += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int d = static_cast<int>(e % head_dim);
        const int64_t rh = e / head_dim;
        const int h = static_cast<int>(rh % num_heads);
        const int64_t r = rh / num_heads;
        int b = 0;
        if (cu_seqlens_q) {
            int lo = 0, hi = batch - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (cu_seqlens_q[mid] <= r) lo = mid; else hi = mid - 1;
            }
            b = lo;
        }
        const float v = __bfloat162float(q[r * q_row_stride + h * q_head_stride + d]) /
                        descale[b * num_kv_heads + h / g];
        out[e] = to_e4m3(v);
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

cudaError_t requantize_paged_kv_compact_e4m3_cuda(const void* data,
                                                  const float* scales,
                                                  void* out,
                                                  float* descale,
                                                  const int* page_table,
                                                  int* compact_table,
                                                  const int* seqused_k,
                                                  int seqlen_k,
                                                  int batch,
                                                  int page_table_stride,
                                                  int num_kv_heads,
                                                  int head_dim,
                                                  int page_size,
                                                  int is_fp8,
                                                  cudaStream_t stream) {
    if (data == nullptr || scales == nullptr || out == nullptr ||
        descale == nullptr || page_table == nullptr || compact_table == nullptr ||
        batch <= 0 || page_table_stride <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || page_size <= 0 || (head_dim % 16) != 0 ||
        (seqused_k == nullptr && seqlen_k <= 0)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    kv_descale_kernel<<<dim3(batch, num_kv_heads), kThreads, 0, stream>>>(
        scales, page_table, seqused_k, seqlen_k, page_table_stride, num_kv_heads,
        page_size, is_fp8 ? kE4m3Max : kInt8Max, descale);
    const int slots = batch * page_table_stride;
    requant_compact_kernel<<<slots, kThreads, 0, stream>>>(
        static_cast<const unsigned char*>(data), scales, descale,
        static_cast<unsigned char*>(out), page_table, compact_table,
        page_table_stride, num_kv_heads, head_dim, page_size, is_fp8 != 0);
    return cudaGetLastError();
}

cudaError_t quantize_q_e4m3_cuda(const void* q_bf16,
                                 void* q_e4m3,
                                 float* descale,
                                 const int* cu_seqlens_q,
                                 int total_q,
                                 int batch,
                                 int64_t q_row_stride,
                                 int64_t q_head_stride,
                                 int num_heads,
                                 int num_kv_heads,
                                 int head_dim,
                                 cudaStream_t stream) {
    if (q_bf16 == nullptr || q_e4m3 == nullptr || descale == nullptr ||
        total_q <= 0 || batch <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
        num_heads % num_kv_heads != 0 || head_dim <= 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const auto* q = static_cast<const __nv_bfloat16*>(q_bf16);
    q_descale_kernel<<<dim3(batch, num_kv_heads), kThreads, 0, stream>>>(
        q, cu_seqlens_q, total_q, q_row_stride, q_head_stride, num_heads,
        num_kv_heads, head_dim, descale);
    const int64_t n = static_cast<int64_t>(total_q) * num_heads * head_dim;
    const int blocks = static_cast<int>(std::min<int64_t>((n + kThreads - 1) / kThreads, 65535));
    q_quant_kernel<<<blocks, kThreads, 0, stream>>>(
        q, cu_seqlens_q, descale, static_cast<unsigned char*>(q_e4m3), total_q,
        batch, q_row_stride, q_head_stride, num_heads, num_kv_heads, head_dim);
    return cudaGetLastError();
}

}  // extern "C"

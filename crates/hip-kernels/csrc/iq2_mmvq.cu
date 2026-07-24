// IQ2_XXS / Q2_K mat-vec kernels + q8_1 activation quantization for the
// AIPC DSv4-Flash 2-bit lane (#76/#77).
//
// Adapted from ggml-org/llama.cpp @ d2462f8f (MIT, vendored at
// vendor/llama.cpp/): mul_mat_vec_q (ggml-cuda/mmvq.cu),
// vec_dot_iq2_xxs_q8_1 / vec_dot_q2_K_q8_1 (ggml-cuda/vecdotq.cuh),
// quantize_q8_1 (ggml-cuda/quantize.cu), warp reduces + ggml_cuda_dp4a
// (ggml-cuda/common.cuh), HIP __vsub4/__vcmpne4 emulation
// (ggml-cuda/vendors/hip.h). All ggml runtime plumbing (ggml_tensor,
// contexts, pools, type dispatch, MoE channels, fusion) stripped — raw
// pointers + explicit dims, B=1 decode shape, the two quant types
// specialized directly. Block structs and codebook tables come from the
// vendored ggml-common.h (see the DECL/IMPL macros below — same pull-in as
// ggml-cuda/common.cuh:12-23).
//
// Single-source: valid for nvcc as-is and for hipcc with
// crates/hip-kernels/csrc/arle_hip_shim.h force-included.

#if !defined(__HIP_PLATFORM_AMD__)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

#include <cstdint>
#include <limits>

#if defined(__HIP_PLATFORM_AMD__)
#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#endif
#include "ggml-common.h"

// Verified against vendored ggml-common.h: block_q8_1 (l.249-259) =
// 2*half + 32 = 36; block_q2_K (l.286-299) = 16 + 64 + 2*half = 84;
// block_iq2_xxs (l.368-375) = half + (256/8)*uint16 = 66. QK_K=256 (l.89),
// QK8_1=32 (l.248). lib.rs row-bytes helpers pin the same numbers.
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 layout drifted from ggml-common.h");
static_assert(sizeof(block_q2_K) == 84, "block_q2_K layout drifted from ggml-common.h");
static_assert(sizeof(block_iq2_xxs) == 66, "block_iq2_xxs layout drifted from ggml-common.h");

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

// Wave32 pinned: build.rs only targets gfx11/gfx12 (--offload-arch, default
// gfx1151); CDNA wave64 is out of the AIPC lane.
#define WARP_SIZE 32

#if defined(__HIP_PLATFORM_AMD__)
// NVIDIA SIMD-in-word intrinsics absent on HIP — emulation from
// llama.cpp ggml-cuda/vendors/hip.h.
typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t &>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t &>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
#else
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        if (tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        if (tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    return reinterpret_cast<int &>(c);
#endif
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    return __vsubss4(a, b);
}

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const uint8x4_t &va = reinterpret_cast<const uint8x4_t &>(a);
    const uint8x4_t &vb = reinterpret_cast<const uint8x4_t &>(b);
    unsigned int c;
    uint8x4_t &vc = reinterpret_cast<uint8x4_t &>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    }
    return c;
}
#endif // __HIP_PLATFORM_AMD__

// ggml-cuda/common.cuh ggml_cuda_dp4a, arch dispatch reduced to the lanes
// this crate builds: gfx11/gfx12 sudot4 on HIP, __dp4a on sm_61+.
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(__HIP_PLATFORM_AMD__)
#if defined(__GFX11__) || defined(__GFX12__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t &>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t &>(b);
    return c + va[0]*vb[0] + va[1]*vb[1] + va[2]*vb[2] + va[3]*vb[3];
#endif
#else
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t *a8 = (const int8_t *) &a;
    const int8_t *b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
#endif
}

// ggml-cuda/common.cuh warp reduces (float, partial-width).
template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

// ggml-cuda/vecdotq.cuh helpers.
static __device__ __forceinline__ int get_int_b2(const void *x, const int &i32) {
    const uint16_t *x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void *x, const int &i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

static __device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    // v is a 7 bit int, with the 8th sign being encodable as popcnt
    // with xor we can "correct" the bit instead of having to mask
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    // broadcast over uint to allow for 0x08040201 / 0x80402010 as selectors
    return s * 0x01010101;
}

#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_IQ2_XXS_Q8_1_MMVQ 2

// ggml-cuda/vecdotq.cuh vec_dot_q2_K_q8_1_impl_mmvq + vec_dot_q2_K_q8_1.
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int &v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 &dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int &kbx, const int &iqs) {

    const block_q2_K *bq2_K = (const block_q2_K *) vbq + kbx;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t *scales = bq2_K->scales + scale_offset;

    const int v = get_int_b4(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

// ggml-cuda/vecdotq.cuh vec_dot_iq2_xxs_q8_1.
static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int &kbx, const int &iqs) {

    const block_iq2_xxs *bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t *aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2 *) iq2xxs_grid)[aux8[k0/2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
    sumi = sumi * ls / 8;           // (sumi * scale + sumi / 2) / 4
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// ============================================================================
// q8_1 activation quantization — ggml-cuda/quantize.cu quantize_q8_1,
// reduced to contiguous [nrows, ncols] (no ggml strides/padding; the
// launcher rejects ncols % QK8_1 != 0 so ne0 == ne00).
// ============================================================================

#define ARLE_QUANTIZE_BLOCK_SIZE 256

__launch_bounds__(ARLE_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void arle_quantize_q8_1_kernel(
    const float * __restrict__ x, void * __restrict__ vy, const int64_t ncols) {
    const int64_t i0 = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i0 >= ncols) {
        return;
    }
    const int64_t row = blockIdx.y;
    const int64_t i_cont = row*ncols + i0;

    block_q8_1 *y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = x[i_cont];
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

// ============================================================================
// mul_mat_vec_q — ggml-cuda/mmvq.cu, specialized to ncols_dst=1 (B=1 decode),
// no MoE ids, no fusion, single channel/sample, contiguous row-major weights.
// Launch geometry from mmvq.cu calc_nwarps/calc_rows_per_block:
//   HIP RDNA2/RDNA3.5 table -> nwarps=1; CUDA GENERIC table -> nwarps=4;
//   rows_per_cuda_block == 1 on every table for ncols_dst=1.
// ============================================================================

#if defined(__HIP_PLATFORM_AMD__)
static constexpr int ARLE_MMVQ_NWARPS = 1;
#else
static constexpr int ARLE_MMVQ_NWARPS = 4;
#endif

typedef float (*arle_vec_dot_q_t)(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int &kbx, const int &iqs);

template <arle_vec_dot_q_t vec_dot_q, int qi, int vdr>
__launch_bounds__(ARLE_MMVQ_NWARPS*WARP_SIZE, 1)
static __global__ void arle_mmvq_kernel(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x) {
    constexpr int qk = QK_K;
    constexpr int nwarps = ARLE_MMVQ_NWARPS;
    constexpr int warp_size = WARP_SIZE;

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int row = blockIdx.x; // rows_per_cuda_block == 1: grid.x is exact
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const block_q8_1 *y = (const block_q8_1 *) vy;
    const int kbx_offset = row * blocks_per_row_x;

    float tmp = 0.0f;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

        tmp += vec_dot_q(vx, &y[kby], kbx_offset + kbx, kqs);
    }

    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][warp_size];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int l = 0; l < nwarps - 1; ++l) {
        tmp += tmp_shared[l][threadIdx.x];
    }
    tmp = warp_reduce_sum<warp_size>(tmp);

    if (threadIdx.x == 0) { // threadIdx.x < rows_per_cuda_block
        dst[row] = tmp;
    }
}

// ============================================================================
// C-ABI launchers (csrc convention: stream last, status returned).
// ============================================================================

extern "C" {

cudaError_t arle_quantize_row_q8_1_cuda(
    const float *x, void *y_q8_1, int64_t ncols, int64_t nrows, cudaStream_t stream) {
    // Vendored quantize_row_q8_1_cuda enforces ne0 % QK8_1 == 0 host-side
    // (GGML_ASSERT, quantize.cu:380); here that becomes a returned status.
    if (x == nullptr || y_q8_1 == nullptr || ncols <= 0 || nrows <= 0 || ncols % QK8_1 != 0) {
        return cudaErrorInvalidValue;
    }
    const int64_t block_num_x = (ncols + ARLE_QUANTIZE_BLOCK_SIZE - 1) / ARLE_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, nrows, 1);
    const dim3 block_size(ARLE_QUANTIZE_BLOCK_SIZE, 1, 1);
    arle_quantize_q8_1_kernel<<<num_blocks, block_size, 0, stream>>>(x, y_q8_1, ncols);
    return cudaGetLastError();
}

cudaError_t arle_mmvq_iq2_xxs_cuda(
    const void *w_blocks, const void *x_q8_1, float *y, int ncols, int nrows, cudaStream_t stream) {
    if (w_blocks == nullptr || x_q8_1 == nullptr || y == nullptr || ncols <= 0 || nrows <= 0 ||
        ncols % QK_K != 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims(WARP_SIZE, ARLE_MMVQ_NWARPS, 1);
    arle_mmvq_kernel<vec_dot_iq2_xxs_q8_1, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ>
        <<<block_nums, block_dims, 0, stream>>>(w_blocks, x_q8_1, y, ncols);
    return cudaGetLastError();
}

cudaError_t arle_mmvq_q2_k_cuda(
    const void *w_blocks, const void *x_q8_1, float *y, int ncols, int nrows, cudaStream_t stream) {
    if (w_blocks == nullptr || x_q8_1 == nullptr || y == nullptr || ncols <= 0 || nrows <= 0 ||
        ncols % QK_K != 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims(WARP_SIZE, ARLE_MMVQ_NWARPS, 1);
    arle_mmvq_kernel<vec_dot_q2_K_q8_1, QI2_K, VDR_Q2_K_Q8_1_MMVQ>
        <<<block_nums, block_dims, 0, stream>>>(w_blocks, x_q8_1, y, ncols);
    return cudaGetLastError();
}

} // extern "C"

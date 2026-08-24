// Unified W2/W4/W8 A16 dequant-on-the-fly GEMV kernel.
//
// Nibble extraction uses parallel bitmask on uint32 (like llama.cpp/vLLM),
// NOT per-element shift/mask or pointer aliasing on register variables.
//
// W8: signed int8, no zero-point. Direct cast to float.
// W4: unsigned nibbles, zero-point=8. Parallel extract via 0x0F0F0F0F mask.
// W2: unsigned 2-bit, zero-point=2. Extract via 0x03030303 mask.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "../common.cuh"

#define GEMV_THREADS 256
#define GEMV_ROWS 4
#define DSV4_BATCH_TILE 32
#define QWEN_GEMV_BATCH_TILE 8

// Single-row W8/W4/W2 A16 GEMV is the B=1 special case of the batched twin
// (w{8,4,2}a16_gemv_batch_kernel below): at batch_idx=0 the batch kernel's
// `input + batch_idx*K` / `output[batch_idx*N + row]` collapse to `input[k]` /
// `output[row]`, byte-identical arithmetic and accumulation order. The single-
// row launchers therefore call the batch kernel with grid.y=1, B=1 (#144
// pattern) instead of carrying a near-duplicate kernel.

__device__ __forceinline__ float dsv4_decode_e8m0(uint8_t bits) {
    uint32_t raw = static_cast<uint32_t>(bits) << 23;
    return __uint_as_float(raw);
}

__device__ __forceinline__ float dsv4_decode_fp8_e4m3(uint8_t bits) {
    if ((bits & 0x7f) == 0) return 0.0f;
    if ((bits & 0x7f) == 0x7f) {
        return (bits & 0x80) ? -448.0f : 448.0f;
    }
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return static_cast<float>(value);
}

__device__ __forceinline__ float dsv4_decode_fp4_e2m1(uint8_t bits) {
    return arle_decode_fp4_e2m1(bits);
}

__device__ __forceinline__ float dsv4_block_scale(
    const uint8_t* __restrict__ scales,
    int row,
    int col,
    int N,
    int K,
    int scale_rows,
    int scale_cols)
{
    const int block_h = (N + scale_rows - 1) / scale_rows;
    const int block_w = (K + scale_cols - 1) / scale_cols;
    const int sr_raw = row / block_h;
    const int sc_raw = col / block_w;
    const int sr = sr_raw < scale_rows ? sr_raw : (scale_rows - 1);
    const int sc = sc_raw < scale_cols ? sc_raw : (scale_cols - 1);
    return dsv4_decode_e8m0(scales[sr * scale_cols + sc]);
}

__device__ __forceinline__ float fp8_f32_block_scale(
    const float* __restrict__ scales,
    int row,
    int col,
    int scale_rows,
    int scale_cols,
    int block_m,
    int block_k)
{
    const int sr_raw = row / block_m;
    const int sc_raw = col / block_k;
    const int sr = sr_raw < scale_rows ? sr_raw : (scale_rows - 1);
    const int sc = sc_raw < scale_cols ? sc_raw : (scale_cols - 1);
    return scales[sr * scale_cols + sc];
}

// Software dequant of an FP8 E4M3 2D-block-scaled weight matrix [N, K]
// (row-major) into a dense BF16 matrix [N, K]. The block-scale layout matches
// fp8_f32_block_scale / the autograd reference (fp8_block_scaled.cu):
//   scale[(row/block_m)*scale_cols + (col/block_k)].
// One thread per (row, col) element. Used by the infer-cuda dense FP8 GEMM
// fallback on pre-Hopper GPUs (sm < 9.0), where DeepGEMM's wgmma path is
// unavailable: dequant once → reuse the existing BF16 cuBLAS GEMM (gemm_cuda).
__global__ void dequantize_fp8_block_scaled_to_bf16_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ output,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int block_m,
    int block_k)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)N * K;
    if (idx >= total) return;
    const int row = (int)(idx / K);
    const int col = (int)(idx % K);
    const float scale =
        fp8_f32_block_scale(scales, row, col, scale_rows, scale_cols, block_m, block_k);
    const float w = dsv4_decode_fp8_e4m3(weight[idx]) * scale;
    output[idx] = __float2bfloat16(w);
}

extern "C" cudaError_t dequantize_fp8_block_scaled_to_bf16_cuda(
    const uint8_t* weight,
    const float* scales,
    __nv_bfloat16* output,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int block_m,
    int block_k,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || scale_rows <= 0 || scale_cols <= 0 || block_m <= 0 ||
        block_k <= 0) {
        return cudaErrorInvalidValue;
    }
    const long total = (long)N * K;
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    dequantize_fp8_block_scaled_to_bf16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        weight, scales, output, N, K, scale_rows, scale_cols, block_m, block_k);
    return cudaGetLastError();
}

// Inverse of the dequant above: BF16 → FP8 E4M3 + per-block f32 scales
// (amax/448). One CUDA block per weight block. LoRA merge-requant path.
__global__ void quantize_bf16_to_fp8_block_scaled_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ weight,
    float* __restrict__ scales,
    int N,
    int K,
    int scale_cols,
    int block_m,
    int block_k)
{
    const int sr = blockIdx.y;
    const int sc = blockIdx.x;
    const int row0 = sr * block_m;
    const int col0 = sc * block_k;
    const int rows = min(block_m, N - row0);
    const int cols = min(block_k, K - col0);

    __shared__ float red[256];
    float amax = 0.0f;
    for (int r = row0; r < row0 + rows; ++r) {
        for (int c = col0 + threadIdx.x; c < col0 + cols; c += blockDim.x) {
            amax = fmaxf(amax, fabsf(__bfloat162float(input[(long)r * K + c])));
        }
    }
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    // Below this the reciprocal below overflows to inf and every element
    // would saturate to 448 instead of quantizing to ~0.
    const float scale = red[0] > 1e-30f ? red[0] / 448.0f : 1.0f;
    if (threadIdx.x == 0) scales[sr * scale_cols + sc] = scale;

    const float inv = 1.0f / scale;
    for (int r = row0; r < row0 + rows; ++r) {
        for (int c = col0 + threadIdx.x; c < col0 + cols; c += blockDim.x) {
            const float w = __bfloat162float(input[(long)r * K + c]) * inv;
            weight[(long)r * K + c] = __nv_cvt_float_to_fp8(w, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

extern "C" cudaError_t quantize_bf16_to_fp8_block_scaled_cuda(
    const __nv_bfloat16* input,
    uint8_t* weight,
    float* scales,
    int N,
    int K,
    int block_m,
    int block_k,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || block_m <= 0 || block_k <= 0) {
        return cudaErrorInvalidValue;
    }
    const int scale_rows = (N + block_m - 1) / block_m;
    const int scale_cols = (K + block_k - 1) / block_k;
    dim3 grid((unsigned int)scale_cols, (unsigned int)scale_rows, 1);
    quantize_bf16_to_fp8_block_scaled_kernel<<<grid, 256, 0, stream>>>(
        input, weight, scales, N, K, scale_cols, block_m, block_k);
    return cudaGetLastError();
}

// W8A16 dequant: INT8 weight [N, K] × per-row per-column-group BF16 scale
// [N, K/group_size] → BF16 [N, K]. Prefill path — dequant once (N*K), then one
// cuBLAS BF16 GEMM over all M rows, instead of re-reading weight per token via
// the scalar GEMV (the latter is why W8A16 TTFT was ~6x FP8's).
__global__ void dequantize_w8a16_to_bf16_kernel(
    const int8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    __nv_bfloat16* __restrict__ output,
    int N,
    int K,
    int group_size)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)N * K;
    if (idx >= total) return;
    const int row = (int)(idx / K);
    const int col = (int)(idx % K);
    const int num_groups = K / group_size;
    const float scale = __bfloat162float(scales[row * num_groups + col / group_size]);
    output[idx] = __float2bfloat16((float)weight[idx] * scale);
}

extern "C" cudaError_t dequantize_w8a16_to_bf16_cuda(
    const int8_t* weight,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int N,
    int K,
    int group_size,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || group_size <= 0 || K % group_size != 0) {
        return cudaErrorInvalidValue;
    }
    const long total = (long)N * K;
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    dequantize_w8a16_to_bf16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        weight, scales, output, N, K, group_size);
    return cudaGetLastError();
}

// W4A16 dequant: 4-bit weight [N, K/2] (2 int4 per byte, low nibble first) ×
// per-row per-group BF16 scale [N, K/group_size] → BF16 [N, K].
// Same purpose as the W8A16 dequant above: dequant once, then one cuBLAS BF16
// GEMM over all M rows (tensor cores on sm_80+, FP16-cast on sm_70).
__global__ void dequantize_w4a16_to_bf16_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    __nv_bfloat16* __restrict__ output,
    int N,
    int K,
    int group_size)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)N * K;
    if (idx >= total) return;
    const int row = (int)(idx / K);
    const int col = (int)(idx % K);
    const int num_groups = K / group_size;
    const float scale = __bfloat162float(scales[row * num_groups + col / group_size]);
    // 2 int4 per byte: low nibble = even col, high nibble = odd col.
    const uint8_t byte = weight[row * (K / 2) + col / 2];
    const int int4 = (col & 1) ? (byte >> 4) : (byte & 0x0F);
    output[idx] = __float2bfloat16((float)(int4 - 8) * scale);
}

extern "C" cudaError_t dequantize_w4a16_to_bf16_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int N,
    int K,
    int group_size,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || group_size <= 0 || K % group_size != 0 || (K & 1) != 0) {
        return cudaErrorInvalidValue;
    }
    const long total = (long)N * K;
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    dequantize_w4a16_to_bf16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        weight, scales, output, N, K, group_size);
    return cudaGetLastError();
}

// NVFP4 -> E4M3, for the FP8 tensor-core prefill path.
//
// sm_90 has no FP4 tensor core, so a real GEMM has to widen the nibbles first.
// The twin above widens to BF16 and hands cuBLAS a 148 TFLOPS ceiling (84
// measured through Marlin); widening to E4M3 instead lets DeepGEMM contract
// them at 274, and costs one dequant pass (~3.4% of the GEMM it feeds).
//
// The group scale cannot ride along inside the E4M3 value. This checkpoint's
// `weight_scale` uses the full E4M3 range (measured max 448) and an E2M1 value
// reaches 6, so the product reaches 2688 against E4M3's 448 ceiling. A per-
// 128x128-block power of two is divided out here and handed back to DeepGEMM as
// its `sfb`, which applies it to the fp32 accumulator. A power of two keeps the
// division exact, so the only loss is the E2M1 x E4M3 product needing 4 mantissa
// bits where E4M3 stores 3 — about a quarter of the nonzero weights round, at
// half an ulp. That is the same order as the E4M3 activation rounding the FP8
// checkpoint already runs with, and it is why this path is gated on the needle
// ladder rather than argued from the scale algebra.

// ---------------------------------------------------------------------------
// Marlin-sourced weight materialisation, shared by both DeepGEMM prefill arms.
//
// The arms used to contract the pre-repack checkpoint bytes, so those had to
// stay resident beside the Marlin layout — the same weight stored twice, 10.0
// GB on Qwen3.8-27B-NVFP4. The repack (gptq_marlin_repack.cuh) is a pure
// permutation, so reading it back is exact and the source can be freed at load.
//
// One Marlin word interleaves several k with two n, so a direct element gather
// fragments one side of the traffic. Both kernels instead stage a 64n x 128k
// super-tile (8 Marlin k-tiles) in shared memory and run full 128-B
// transactions on both sides; the k-tile stride is padded by one word to keep
// the 8 tiles a thread block reads on distinct banks.
// ---------------------------------------------------------------------------
#define MARLIN_MAT_THREADS 256
#define MARLIN_MAT_KTILES 8

// Marlin's S0E5M3 scale byte is the high half of `f16(v) << 1` (sign always 0).
__device__ __forceinline__ float marlin_s0e5m3_to_f32(uint8_t bits) {
    return __half2float(__ushort_as_half((unsigned short)bits << 7));
}

// Where `repack_for_marlin_fp4` step 2 put the group scale of output row `n`,
// group `g`: transpose to [K/16, N], an 8x8 transpose inside each 64-run, then
// [0,2,1,3] inside each 4-run. N % 64 keeps every 64-run inside one group row.
__device__ __forceinline__ int marlin_fp4_scale_tail(int n, int g, int N) {
    const int nn = n & 63;
    const int x = (nn & 7) * 8 + (nn >> 3);
    // {0,2,1,3}[i] is i's low two bits swapped. As a local array with a dynamic
    // index nvcc puts it in local memory, which is a load per call.
    const int f = (x & ~3) | ((x & 1) << 1) | ((x & 3) >> 1);
    return (g * (N >> 6) + (n >> 6)) * 64 + f;
}

// FP4 E2M1's largest magnitude, and E4M3's largest finite value: the block
// power of two has to map one onto the other.
#define DSV4_FP4_E2M1_MAX 6.0f
#define DSV4_FP8_E4M3_MAX 448.0f
#define FP4_BLOCK_SCALE_THREADS 256

// One block per 128x128 weight tile: reduce |group_scale| * global over the
// tile's 128 rows x (128/16) scale columns, then round the ratio that would
// saturate E4M3 up to a power of two. Blocks past the last row of scales
// write 1.0f — DeepGEMM reads one sfb row past the last n-block when
// BLOCK_K % BLOCK_N != 0, and a finite 1.0f there keeps the masked lanes clean.
__global__ void fp4_marlin_scale_block_pow2_kernel(
    const uint8_t* __restrict__ tail,
    const float* __restrict__ global_scales,
    float inv_lift,
    float* __restrict__ block_pow2,
    int N,
    int scale_cols,
    int k_blocks)
{
    __shared__ float red[FP4_BLOCK_SCALE_THREADS];
    const int kb = blockIdx.x;
    const int nb = blockIdx.y;
    const int row0 = nb * 128;
    const int col0 = kb * 8;
    const int row_end = min(row0 + 128, N);
    const int col_end = min(col0 + 8, scale_cols);

    float mx = 0.0f;
    const int span = (col_end > col0) ? (col_end - col0) : 0;
    if (span > 0) {
        const long count = (long)(row_end - row0) * span;
        for (long i = threadIdx.x; i < count; i += FP4_BLOCK_SCALE_THREADS) {
            const int r = row0 + (int)(i / span);
            const int c = col0 + (int)(i % span);
            mx = fmaxf(mx, marlin_s0e5m3_to_f32(tail[marlin_fp4_scale_tail(r, c, N)]));
        }
    }
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = FP4_BLOCK_SCALE_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x != 0) return;

    // `inv_lift` undoes the per-tensor power of two the repack multiplied into
    // the stored scale; exact, so the peak is the one the raw scales gave.
    const float peak = red[0] * inv_lift * DSV4_FP4_E2M1_MAX * fabsf(global_scales[0]);
    float p = 1.0f;
    if (peak > 0.0f && isfinite(peak)) {
        p = exp2f(ceilf(log2f(peak / DSV4_FP8_E4M3_MAX)));
        // exp2f of a large magnitude, or an all-zero tile, must not poison sfb.
        if (!isfinite(p) || p <= 0.0f) p = 1.0f;
    }
    block_pow2[(long)nb * k_blocks + kb] = p;
}

extern "C" cudaError_t fp4_marlin_scale_block_pow2_cuda(
    const uint8_t* marlin_packed,
    const float* global_scales,
    float inv_lift,
    float* block_pow2,
    int N,
    int K,
    int group_size,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || group_size != 16 || (N & 63) != 0 || (K & 127) != 0) {
        return cudaErrorInvalidValue;
    }
    const int n_blocks = (N + 127) / 128;
    const int k_blocks = K / 128;
    fp4_marlin_scale_block_pow2_kernel<<<dim3(k_blocks, n_blocks + 1), FP4_BLOCK_SCALE_THREADS, 0,
                                        stream>>>(
        marlin_packed + (size_t)N * K / 2, global_scales, inv_lift, block_pow2, N, K / 16,
        k_blocks);
    return cudaGetLastError();
}

// Marlin FP4 tiles + their S0E5M3 scale tail x one F32 global scale, divided by
// the tile power of two above -> dense E4M3 [N, K].
__global__ void dequantize_fp4_marlin_to_fp8_kernel(
    const uint32_t* __restrict__ marlin,
    const uint8_t* __restrict__ tail,
    const float* __restrict__ global_scales,
    const float* __restrict__ block_pow2,
    float inv_lift,
    uint8_t* __restrict__ output,
    int N,
    int K,
    int k_blocks)
{
    __shared__ uint32_t sh[MARLIN_MAT_KTILES * 129];
    const int n_tiles = N >> 6;
    const int k_super = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int tid = threadIdx.x;

    const int tile0 = (k_super * MARLIN_MAT_KTILES * n_tiles + n_tile) * 128;
    for (int i = tid; i < MARLIN_MAT_KTILES * 128; i += MARLIN_MAT_THREADS) {
        sh[(i >> 7) * 129 + (i & 127)] = marlin[tile0 + (i >> 7) * n_tiles * 128 + (i & 127)];
    }
    __syncthreads();

    const float gscale = global_scales[0] * inv_lift;
    // A 64n x 128k super-tile sits inside one 128x128 sfb block, and block_pow2
    // is a power of two, so this is one exact reciprocal for the whole block.
    const float inv_block = 1.0f / block_pow2[(long)(n_tile >> 1) * k_blocks + k_super];

    for (int c = tid; c < 64 * MARLIN_MAT_KTILES; c += MARLIN_MAT_THREADS) {
        const int nn = c >> 3;
        const int kt = c & (MARLIN_MAT_KTILES - 1);
        const int n = (n_tile << 6) + nn;
        const int g = k_super * MARLIN_MAT_KTILES + kt;
        const int hi = (nn & 15) >> 3;
        const uint32_t* w = &sh[kt * 129 + (nn & 7) * 16 + (nn >> 4)];
        const float scale =
            marlin_s0e5m3_to_f32(tail[marlin_fp4_scale_tail(n, g, N)]) * gscale * inv_block;
        uint32_t packed[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            const int slot = ((kk & 1) << 2) | (hi << 1) | (kk >> 3);
            const uint32_t word = w[((kk & 7) >> 1) * 4];
            const float v = dsv4_decode_fp4_e2m1((uint8_t)((word >> (slot * 4)) & 0xf)) * scale;
            const uint32_t byte = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
            packed[kk >> 2] |= byte << ((kk & 3) * 8);
        }
        *reinterpret_cast<uint4*>(&output[(long)n * K + (g << 4)]) =
            make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
}

// Marlin FP4 tiles + their S0E5M3 scale tail x one F32 global scale -> dense
// BF16 [N, K]. The FP8 twin above divides out a per-128x128 power of two
// because DeepGEMM takes it back as sfb; a BF16 consumer wants the true value,
// so there is no block_pow2 here.
__global__ void dequantize_fp4_marlin_to_bf16_kernel(
    const uint32_t* __restrict__ marlin,
    const uint8_t* __restrict__ tail,
    const float* __restrict__ global_scales,
    float inv_lift,
    __nv_bfloat16* __restrict__ output,
    int N,
    int K)
{
    __shared__ uint32_t sh[MARLIN_MAT_KTILES * 129];
    const int n_tiles = N >> 6;
    const int k_super = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int tid = threadIdx.x;

    const int tile0 = (k_super * MARLIN_MAT_KTILES * n_tiles + n_tile) * 128;
    for (int i = tid; i < MARLIN_MAT_KTILES * 128; i += MARLIN_MAT_THREADS) {
        sh[(i >> 7) * 129 + (i & 127)] = marlin[tile0 + (i >> 7) * n_tiles * 128 + (i & 127)];
    }
    __syncthreads();

    const float gscale = global_scales[0] * inv_lift;

    for (int c = tid; c < 64 * MARLIN_MAT_KTILES; c += MARLIN_MAT_THREADS) {
        const int nn = c >> 3;
        const int kt = c & (MARLIN_MAT_KTILES - 1);
        const int n = (n_tile << 6) + nn;
        const int g = k_super * MARLIN_MAT_KTILES + kt;
        const int hi = (nn & 15) >> 3;
        const uint32_t* w = &sh[kt * 129 + (nn & 7) * 16 + (nn >> 4)];
        const float scale =
            marlin_s0e5m3_to_f32(tail[marlin_fp4_scale_tail(n, g, N)]) * gscale;
        __nv_bfloat16 vals[16];
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            const int slot = ((kk & 1) << 2) | (hi << 1) | (kk >> 3);
            const uint32_t word = w[((kk & 7) >> 1) * 4];
            vals[kk] = __float2bfloat16(
                dsv4_decode_fp4_e2m1((uint8_t)((word >> (slot * 4)) & 0xf)) * scale);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            reinterpret_cast<uint4*>(&output[(long)n * K + (g << 4)])[j] =
                reinterpret_cast<const uint4*>(vals)[j];
        }
    }
}

extern "C" cudaError_t dequantize_fp4_marlin_to_bf16_cuda(
    const uint8_t* marlin_packed,
    const float* global_scales,
    float inv_lift,
    __nv_bfloat16* output,
    int N,
    int K,
    int group_size,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || group_size != 16 || (N & 63) != 0 || (K & 127) != 0) {
        return cudaErrorInvalidValue;
    }
    dequantize_fp4_marlin_to_bf16_kernel<<<dim3(K / 128, N / 64), MARLIN_MAT_THREADS, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(marlin_packed), marlin_packed + (size_t)N * K / 2,
        global_scales, inv_lift, output, N, K);
    return cudaGetLastError();
}

extern "C" cudaError_t dequantize_fp4_marlin_to_fp8_cuda(
    const uint8_t* marlin_packed,
    const float* global_scales,
    const float* block_pow2,
    float inv_lift,
    uint8_t* output,
    int N,
    int K,
    int group_size,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || group_size != 16 || (N & 63) != 0 || (K & 127) != 0) {
        return cudaErrorInvalidValue;
    }
    dequantize_fp4_marlin_to_fp8_kernel<<<dim3(K / 128, N / 64), MARLIN_MAT_THREADS, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(marlin_packed), marlin_packed + (size_t)N * K / 2,
        global_scales, block_pow2, inv_lift, output, N, K, K / 128);
    return cudaGetLastError();
}

// Marlin FP8 tiles -> the plain [N, K] E4M3 bytes DeepGEMM's dense NT entry
// takes as B. The repack applies no value transform, so this reproduces the
// checkpoint's own bytes; one word holds k {2j, 2j+8, 2j+1, 2j+9} of a single
// row in bytes {0,1,2,3}, which is why four strided loads make one 16-B store.
__global__ void marlin_fp8_to_e4m3_kernel(
    const uint32_t* __restrict__ marlin,
    uint8_t* __restrict__ output,
    int N,
    int K)
{
    __shared__ uint32_t sh[MARLIN_MAT_KTILES * 257];
    const int n_tiles = N >> 6;
    const int k_super = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int tid = threadIdx.x;

    const int tile0 = (k_super * MARLIN_MAT_KTILES * n_tiles + n_tile) * 256;
    for (int i = tid; i < MARLIN_MAT_KTILES * 256; i += MARLIN_MAT_THREADS) {
        sh[(i >> 8) * 257 + (i & 255)] = marlin[tile0 + (i >> 8) * n_tiles * 256 + (i & 255)];
    }
    __syncthreads();

    for (int c = tid; c < 64 * MARLIN_MAT_KTILES; c += MARLIN_MAT_THREADS) {
        const int nn = c >> 3;
        const int kt = c & (MARLIN_MAT_KTILES - 1);
        const uint32_t* w = &sh[kt * 257 + (nn & 7) * 32 + (nn >> 4) * 2 + ((nn & 15) >> 3)];
        uint4 v;
        v.x = __byte_perm(w[0], w[8], 0x6420);
        v.y = __byte_perm(w[16], w[24], 0x6420);
        v.z = __byte_perm(w[0], w[8], 0x7531);
        v.w = __byte_perm(w[16], w[24], 0x7531);
        const long row = (long)((n_tile << 6) + nn);
        *reinterpret_cast<uint4*>(
            &output[row * K + ((k_super * MARLIN_MAT_KTILES + kt) << 4)]) = v;
    }
}

extern "C" cudaError_t marlin_fp8_to_e4m3_cuda(
    const uint8_t* marlin_packed,
    uint8_t* output,
    int N,
    int K,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || (N & 63) != 0 || (K & 127) != 0) {
        return cudaErrorInvalidValue;
    }
    marlin_fp8_to_e4m3_kernel<<<dim3(K / 128, N / 64), MARLIN_MAT_THREADS, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(marlin_packed), output, N, K);
    return cudaGetLastError();
}

__device__ __forceinline__ float fp8_f32_dot16(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ x)
{
    const auto* w4 = reinterpret_cast<const __nv_fp8x4_e4m3*>(weight);
    const float4 wf0 = static_cast<float4>(w4[0]);
    const float4 wf1 = static_cast<float4>(w4[1]);
    const float4 wf2 = static_cast<float4>(w4[2]);
    const float4 wf3 = static_cast<float4>(w4[3]);
    const uint4 x0 = *reinterpret_cast<const uint4*>(x);
    const uint4 x1 = *reinterpret_cast<const uint4*>(x + 8);
    const auto* xb0 = reinterpret_cast<const __nv_bfloat16*>(&x0);
    const auto* xb1 = reinterpret_cast<const __nv_bfloat16*>(&x1);
    return wf0.x * __bfloat162float(xb0[0])
        + wf0.y * __bfloat162float(xb0[1])
        + wf0.z * __bfloat162float(xb0[2])
        + wf0.w * __bfloat162float(xb0[3])
        + wf1.x * __bfloat162float(xb0[4])
        + wf1.y * __bfloat162float(xb0[5])
        + wf1.z * __bfloat162float(xb0[6])
        + wf1.w * __bfloat162float(xb0[7])
        + wf2.x * __bfloat162float(xb1[0])
        + wf2.y * __bfloat162float(xb1[1])
        + wf2.z * __bfloat162float(xb1[2])
        + wf2.w * __bfloat162float(xb1[3])
        + wf3.x * __bfloat162float(xb1[4])
        + wf3.y * __bfloat162float(xb1[5])
        + wf3.z * __bfloat162float(xb1[6])
        + wf3.w * __bfloat162float(xb1[7]);
}

// Dot 16 ALREADY-DECODED fp8 weights (4 float4, decoded once by the caller)
// against one bf16 x16. The batched tiled GEMV decodes each weight chunk ONCE
// and calls this per batch column, so the fp8->fp32 conversion is not repeated
// per column (unlike fp8_f32_dot16). Same arithmetic + accumulation order as
// fp8_f32_dot16, so numerics are identical.
__device__ __forceinline__ float dot16_with_decoded(
    const float4& wf0,
    const float4& wf1,
    const float4& wf2,
    const float4& wf3,
    const __nv_bfloat16* __restrict__ x)
{
    const uint4 x0 = *reinterpret_cast<const uint4*>(x);
    const uint4 x1 = *reinterpret_cast<const uint4*>(x + 8);
    const auto* xb0 = reinterpret_cast<const __nv_bfloat16*>(&x0);
    const auto* xb1 = reinterpret_cast<const __nv_bfloat16*>(&x1);
    return wf0.x * __bfloat162float(xb0[0])
        + wf0.y * __bfloat162float(xb0[1])
        + wf0.z * __bfloat162float(xb0[2])
        + wf0.w * __bfloat162float(xb0[3])
        + wf1.x * __bfloat162float(xb0[4])
        + wf1.y * __bfloat162float(xb0[5])
        + wf1.z * __bfloat162float(xb0[6])
        + wf1.w * __bfloat162float(xb0[7])
        + wf2.x * __bfloat162float(xb1[0])
        + wf2.y * __bfloat162float(xb1[1])
        + wf2.z * __bfloat162float(xb1[2])
        + wf2.w * __bfloat162float(xb1[3])
        + wf3.x * __bfloat162float(xb1[4])
        + wf3.y * __bfloat162float(xb1[5])
        + wf3.z * __bfloat162float(xb1[6])
        + wf3.w * __bfloat162float(xb1[7]);
}

// One row's FP4 x BF16 dot product for this thread's strided slice of K.
// uint4 loads move 16 B (32 weights) per transaction; the byte-at-a-time form
// this replaced fetched a 128 B cacheline per 1 B used. Shared by the dense and
// the two grouped (MoE) FP4 GEMV kernels.
// Decode packed E2M1 nibbles into bf16 through PRMT byte tables -- no memory,
// no branches, no warp divergence.
//
// Both bytes of the bf16 are 8-entry functions of the magnitude bits n&7:
//   low  byte = {00,00,80,c0,00,40,80,c0}[n&7]
//   high byte = {00,3f,3f,3f,40,40,40,40}[n&7] | (n&8 ? 0x80 : 0)
// One __byte_perm is four such lookups, so a 32-bit word (8 weights) decodes in
// 15 integer instructions. The shift/mask form this replaced rebuilt the bf16
// exponent field arithmetically at ~13 ops per nibble, which put the kernel at
// 92% ALU-pipe utilisation with the FMA pipe at 18%.
//
// n=8 must decode to 0x8000 (negative zero), so the sign OR needs no exception
// for the zero codes. Verified bit-exact against the 16-entry reference table
// for all 2^32 packed words.
constexpr uint32_t FP4_E2M1_LO_LUT03 = 0xC0800000u;  // low bytes, n&7 = 0..3
constexpr uint32_t FP4_E2M1_LO_LUT47 = 0xC0804000u;  // low bytes, n&7 = 4..7
constexpr uint32_t FP4_E2M1_HI_LUT03 = 0x3F3F3F00u;  // high magnitude, n&7 = 0..3
constexpr uint32_t FP4_E2M1_HI_LUT47 = 0x40404040u;  // high magnitude, n&7 = 4..7

// 8 packed nibbles -> 4 bf16x2, in nibble order.
__device__ __forceinline__ void fp4_e2m1_word_to_bf16x2(uint32_t p, __nv_bfloat162 out[4]) {
    const uint32_t idx = p & 0x77777777u;
    const uint32_t idx_hi = idx >> 16;
    // p<<4 moves each even nibble's sign bit to a byte's bit 7, where the odd
    // nibble's already sits, so one PRMT gathers four signs byte-aligned.
    const uint32_t sgn = p << 4;

    const uint32_t lo0 = __byte_perm(FP4_E2M1_LO_LUT03, FP4_E2M1_LO_LUT47, idx);
    const uint32_t mag0 = __byte_perm(FP4_E2M1_HI_LUT03, FP4_E2M1_HI_LUT47, idx);
    const uint32_t hi0 = mag0 | (__byte_perm(sgn, p, 0x5140u) & 0x80808080u);
    const uint32_t w0 = __byte_perm(lo0, hi0, 0x5140u);
    const uint32_t w1 = __byte_perm(lo0, hi0, 0x7362u);

    const uint32_t lo1 = __byte_perm(FP4_E2M1_LO_LUT03, FP4_E2M1_LO_LUT47, idx_hi);
    const uint32_t mag1 = __byte_perm(FP4_E2M1_HI_LUT03, FP4_E2M1_HI_LUT47, idx_hi);
    const uint32_t hi1 = mag1 | (__byte_perm(sgn, p, 0x7362u) & 0x80808080u);
    const uint32_t w2 = __byte_perm(lo1, hi1, 0x5140u);
    const uint32_t w3 = __byte_perm(lo1, hi1, 0x7362u);

    out[0] = *reinterpret_cast<const __nv_bfloat162*>(&w0);
    out[1] = *reinterpret_cast<const __nv_bfloat162*>(&w1);
    out[2] = *reinterpret_cast<const __nv_bfloat162*>(&w2);
    out[3] = *reinterpret_cast<const __nv_bfloat162*>(&w3);
}

__device__ __forceinline__ __nv_bfloat162 fp4_e2m1_pair_to_bf16x2(uint32_t lo, uint32_t hi) {
    const uint32_t p = (lo & 0xfu) | ((hi & 0xfu) << 4);
    const uint32_t idx = p & 0x77u;
    const uint32_t l = __byte_perm(FP4_E2M1_LO_LUT03, FP4_E2M1_LO_LUT47, idx);
    const uint32_t mag = __byte_perm(FP4_E2M1_HI_LUT03, FP4_E2M1_HI_LUT47, idx);
    const uint32_t h = mag | (__byte_perm(p << 4, p, 0x5140u) & 0x80808080u);
    const uint32_t bits = __byte_perm(l, h, 0x5140u);
    return *reinterpret_cast<const __nv_bfloat162*>(&bits);
}

// One row's FP4 x BF16 dot product for this thread's strided slice of K.
// uint4 loads move 16 B (32 weights) per transaction; the byte-at-a-time form
// this replaced fetched a 128 B cacheline per 1 B used. Shared by the dense and
// the two grouped (MoE) FP4 GEMV kernels.
// One row's FP4 x BF16 dot product for this thread's strided slice of K.
// Shared by the dense and the two grouped (MoE) FP4 GEMV kernels.
__device__ __forceinline__ float fp4_e2m1_row_dot(
    const uint8_t* __restrict__ weight_row,
    const uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x,
    float g_scale,
    int scale_base,
    int K,
    int group_size,
    int tid_in_row,
    int threads_per_row)
{
    const int bytes_per_row = K / 2;
    float sum = 0.0f;

    const int vec_k_span = 32;
    const int k_vec_end = (K / vec_k_span) * vec_k_span;
    const bool vec_ok = (bytes_per_row & 15) == 0;
    if (vec_ok) {
        // group_size is a power of two for every NVFP4 export; shifting avoids
        // the runtime signed-idiv sequence nvcc emits for a variable divisor.
        const int group_shift = __ffs(group_size) - 1;
        const int groups_per_chunk = vec_k_span >> group_shift;
        for (int k = tid_in_row * vec_k_span; k < k_vec_end;
             k += threads_per_row * vec_k_span) {
            const uint4 packed = __ldg(reinterpret_cast<const uint4*>(weight_row + (k >> 1)));
            const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};

            // A 32-weight chunk spans groups_per_chunk groups (2 at the usual
            // group_size=16), so the scale is loaded once per group, not once
            // per 8-weight word as it was.
            const int group_base = k >> group_shift;
            float chunk_scale[2];
            #pragma unroll
            for (int g = 0; g < 2; g++) {
                const int gi = g < groups_per_chunk ? g : groups_per_chunk - 1;
                chunk_scale[g] =
                    dsv4_decode_fp8_e4m3(__ldg(&scales[scale_base + group_base + gi])) * g_scale;
            }

            #pragma unroll
            for (int w = 0; w < 4; w++) {
                const uint32_t p = words[w];
                const int kk = k + w * 8;
                const int gsel = (groups_per_chunk > 1) ? ((w * 8) >> group_shift) : 0;
                const float s = chunk_scale[gsel < 2 ? gsel : 1];

                const uint4 xv = __ldg(reinterpret_cast<const uint4*>(&x[kk]));
                const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);

                __nv_bfloat162 wv[4];
                fp4_e2m1_word_to_bf16x2(p, wv);

                float acc = 0.0f;
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    const float2 prod = __bfloat1622float2(__hmul2(wv[b], xp[b]));
                    acc += prod.x + prod.y;
                }
                sum += acc * s;
            }
        }
    }

    const int tail_pair_start = vec_ok ? (k_vec_end >> 1) : 0;
    for (int pair = tail_pair_start + tid_in_row; pair < bytes_per_row;
         pair += threads_per_row) {
        const int k0 = pair << 1;
        const uint8_t packed = __ldg(&weight_row[pair]);
        const float s =
            dsv4_decode_fp8_e4m3(__ldg(&scales[scale_base + k0 / group_size])) * g_scale;
        const __nv_bfloat162 wv = fp4_e2m1_pair_to_bf16x2(packed, packed >> 4);
        const float2 wf = __bfloat1622float2(wv);
        sum += wf.x * s * __bfloat162float(x[k0]);
        sum += wf.y * s * __bfloat162float(x[k0 + 1]);
    }
    return sum;
}


// Single-output batch GEMV (grid.y = B, one column per block): fp8 weights,
// bf16 acts, `ScaleFn` picks the DSv4 e8m0 or the f32-block scale (#144 merged
// the two byte-identical copies — dsv4_fp8_gemv_batch_kernel /
// fp8_f32_block_gemv_batch_kernel).
template <class ScaleFn>
__global__ void fp8_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B,
    int N,
    int K,
    ScaleFn scale_fn)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_idx = blockIdx.y;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N || batch_idx >= B) return;

    const __nv_bfloat16* x = input + batch_idx * K;
    float sum = 0.0f;
    // uint4 fast path: weight rows are 16B-aligned when K%16==0, and a 16-chunk
    // carries one scale when the scale block is 16-aligned in K.
    if ((K % 16) == 0 && scale_fn.k_block_16_aligned()) {
        const int kv = K / 16;
        const uint8_t* weight_row = weight + (int64_t)row * K;
        for (int v = tid_in_row; v < kv; v += threads_per_row) {
            const int k = v * 16;
            const float scale = scale_fn(row, k);
            sum += scale * fp8_f32_dot16(weight_row + k, x + k);
        }
    } else {
        for (int k = tid_in_row; k < K; k += threads_per_row) {
            const float w = dsv4_decode_fp8_e4m3(weight[row * K + k]) * scale_fn(row, k);
            sum += w * __bfloat162float(x[k]);
        }
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        output[batch_idx * N + row] = __float2bfloat16(total);
    }
}

// Per-16-chunk scale functors: the ONLY axis on which the DSv4 e8m0 and the
// f32-block-scaled tiled fp8 GEMVs differ (#144). Row-invariant work (sr) is
// loop-invariant and hoisted, so the hot path matches the hand-written pair this
// replaces. `operator()(row, k)` returns the scale for the block covering k;
// `k_block_16_aligned()` gates the uint4 fast path (a 16-chunk stays in one
// scale block).
struct Fp8E8m0BlockScale {
    const uint8_t* scales;
    int scale_rows;
    int scale_cols;
    int block_h;
    int block_w;
    __device__ __forceinline__ float operator()(int row, int k) const {
        int sr = row / block_h;
        if (sr >= scale_rows) sr = scale_rows - 1;
        int sc = k / block_w;
        if (sc >= scale_cols) sc = scale_cols - 1;
        return dsv4_decode_e8m0(scales[sr * scale_cols + sc]);
    }
    __device__ __forceinline__ bool k_block_16_aligned() const { return (block_w % 16) == 0; }
};

struct Fp8F32BlockScale {
    const float* scales;
    int scale_rows;
    int scale_cols;
    int block_m;
    int block_k;
    __device__ __forceinline__ float operator()(int row, int k) const {
        return fp8_f32_block_scale(scales, row, k, scale_rows, scale_cols, block_m, block_k);
    }
    __device__ __forceinline__ bool k_block_16_aligned() const { return (block_k % 16) == 0; }
};

// Weight-amortizing batch GEMV for the fp8 decode path (default for B>1): each
// 16-byte weight chunk is decoded ONCE and MAC'd against every batch column in
// the tile. TILE is a COMPILE-TIME param so sums[TILE] uses exactly TILE
// registers — the launcher picks the smallest tile covering B (the Qwen sibling
// measured fixed-tile at 2.15x/3.59x/6.50x vs templated 1.04x/1.07x/1.14x for
// B=2/4/8). `ScaleFn` is the DSv4 e8m0 or the f32-block scale (#144 merged the
// two byte-identical copies).
template <int TILE, class ScaleFn>
__global__ void fp8_gemv_batch_tiled_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B,
    int N,
    int K,
    ScaleFn scale_fn)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_base = blockIdx.y * TILE;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N) return;

    const uint8_t* weight_row = weight + (int64_t)row * K;
    const int tile_batches_raw = B - batch_base;
    const int tile_batches = tile_batches_raw < TILE ? tile_batches_raw : TILE;

    float sums[TILE];
#pragma unroll
    for (int b = 0; b < TILE; ++b) sums[b] = 0.0f;

    // uint4 fast path: a 16-chunk carries one scale (guaranteed by K%16==0 and
    // the scale block being 16-aligned in K). Decode the 16 fp8 weights ONCE,
    // then MAC against every batch column in the tile (no per-column re-decode).
    if ((K % 16) == 0 && scale_fn.k_block_16_aligned()) {
        const int kv = K / 16;
        for (int v = tid_in_row; v < kv; v += threads_per_row) {
            const int k = v * 16;
            const float scale = scale_fn(row, k);
            const auto* w4 = reinterpret_cast<const __nv_fp8x4_e4m3*>(weight_row + k);
            const float4 wf0 = static_cast<float4>(w4[0]);
            const float4 wf1 = static_cast<float4>(w4[1]);
            const float4 wf2 = static_cast<float4>(w4[2]);
            const float4 wf3 = static_cast<float4>(w4[3]);
#pragma unroll
            for (int b = 0; b < TILE; ++b) {
                if (b < tile_batches) {
                    const __nv_bfloat16* x_b = input + (int64_t)(batch_base + b) * K;
                    sums[b] += scale * dot16_with_decoded(wf0, wf1, wf2, wf3, x_b + k);
                }
            }
        }
    } else {
        for (int k = tid_in_row; k < K; k += threads_per_row) {
            const float w = dsv4_decode_fp8_e4m3(weight_row[k]) * scale_fn(row, k);
#pragma unroll
            for (int b = 0; b < TILE; ++b) {
                if (b < tile_batches) {
                    sums[b] += w * __bfloat162float(input[(int64_t)(batch_base + b) * K + k]);
                }
            }
        }
    }

    __shared__ float smem[GEMV_ROWS * 8 * TILE];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
#pragma unroll
    for (int b = 0; b < TILE; ++b) {
        sums[b] = warp_reduce_sum(sums[b]);
        if (lane_id == 0) {
            smem[(row_in_block * warps_per_row + warp_in_row) * TILE + b] = sums[b];
        }
    }
    __syncthreads();
    if (tid_in_row == 0) {
#pragma unroll
        for (int b = 0; b < TILE; ++b) {
            if (b >= tile_batches) continue;
            float total = 0.0f;
            for (int w = 0; w < warps_per_row; ++w) {
                total += smem[(row_in_block * warps_per_row + w) * TILE + b];
            }
            output[(int64_t)(batch_base + b) * N + row] = __float2bfloat16(total);
        }
    }
}

__global__ void dsv4_fp4_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B,
    int N,
    int K,
    int scale_rows,
    int scale_cols)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_idx = blockIdx.y;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N) return;

    const int bytes_per_row = K / 2;
    const int block_h = (N + scale_rows - 1) / scale_rows;
    const int block_w = (K + scale_cols - 1) / scale_cols;
    const int sr_raw = row / block_h;
    const int sr = sr_raw < scale_rows ? sr_raw : (scale_rows - 1);
    const int scale_row_offset = sr * scale_cols;
    const __nv_bfloat16* x = input + batch_idx * K;
    float sum = 0.0f;
    for (int pair = tid_in_row; pair < bytes_per_row; pair += threads_per_row) {
        const int k0 = pair << 1;
        const int k1 = k0 + 1;
        const uint8_t packed = weight[row * bytes_per_row + pair];
        const uint8_t lo = packed & 0x0f;
        const uint8_t hi = (packed >> 4) & 0x0f;
        const int sc0_raw = k0 / block_w;
        const int sc0 = sc0_raw < scale_cols ? sc0_raw : (scale_cols - 1);
        const int sc1_raw = k1 / block_w;
        const int sc1 = sc1_raw < scale_cols ? sc1_raw : (scale_cols - 1);
        const float w0 = dsv4_decode_fp4_e2m1(lo)
            * dsv4_decode_e8m0(scales[scale_row_offset + sc0]);
        const float w1 = dsv4_decode_fp4_e2m1(hi)
            * dsv4_decode_e8m0(scales[scale_row_offset + sc1]);
        sum += w0 * __bfloat162float(x[k0]);
        sum += w1 * __bfloat162float(x[k1]);
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        output[batch_idx * N + row] = __float2bfloat16(total);
    }
}

__global__ void dsv4_fp4_gemv_batch_tiled_kernel(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B,
    int N,
    int K,
    int scale_rows,
    int scale_cols)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_base = blockIdx.y * DSV4_BATCH_TILE;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N) return;

    const int bytes_per_row = K / 2;
    const int tile_batches_raw = B - batch_base;
    const int tile_batches = tile_batches_raw < DSV4_BATCH_TILE ? tile_batches_raw : DSV4_BATCH_TILE;

    if (tile_batches <= 4) {
        float sums4[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) sums4[b] = 0.0f;

        for (int pair = tid_in_row; pair < bytes_per_row; pair += threads_per_row) {
            const int k0 = pair << 1;
            const int k1 = k0 + 1;
            const uint8_t packed = weight[row * bytes_per_row + pair];
            const uint8_t lo = packed & 0x0f;
            const uint8_t hi = (packed >> 4) & 0x0f;
            const float w0 = dsv4_decode_fp4_e2m1(lo)
                * dsv4_block_scale(scales, row, k0, N, K, scale_rows, scale_cols);
            const float w1 = dsv4_decode_fp4_e2m1(hi)
                * dsv4_block_scale(scales, row, k1, N, K, scale_rows, scale_cols);
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                if (b < tile_batches) {
                    const int batch_idx = batch_base + b;
                    const __nv_bfloat16* x = input + batch_idx * K;
                    sums4[b] += w0 * __bfloat162float(x[k0]);
                    sums4[b] += w1 * __bfloat162float(x[k1]);
                }
            }
        }

        __shared__ float smem4[GEMV_ROWS * 8 * 4];
        int warps_per_row = threads_per_row / WARP_SIZE;
        int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            sums4[b] = warp_reduce_sum(sums4[b]);
            if (lane_id == 0) {
                smem4[(row_in_block * warps_per_row + warp_in_row) * 4 + b] = sums4[b];
            }
        }
        __syncthreads();
        if (tid_in_row == 0) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                if (b >= tile_batches) continue;
                const int batch_idx = batch_base + b;
                float total = 0.0f;
                for (int w = 0; w < warps_per_row; ++w) {
                    total += smem4[(row_in_block * warps_per_row + w) * 4 + b];
                }
                output[batch_idx * N + row] = __float2bfloat16(total);
            }
        }
        return;
    }

    float sums[DSV4_BATCH_TILE];
#pragma unroll
    for (int b = 0; b < DSV4_BATCH_TILE; ++b) sums[b] = 0.0f;

    for (int pair = tid_in_row; pair < bytes_per_row; pair += threads_per_row) {
        const int k0 = pair << 1;
        const int k1 = k0 + 1;
        const uint8_t packed = weight[row * bytes_per_row + pair];
        const uint8_t lo = packed & 0x0f;
        const uint8_t hi = (packed >> 4) & 0x0f;
        const float w0 = dsv4_decode_fp4_e2m1(lo)
            * dsv4_block_scale(scales, row, k0, N, K, scale_rows, scale_cols);
        const float w1 = dsv4_decode_fp4_e2m1(hi)
            * dsv4_block_scale(scales, row, k1, N, K, scale_rows, scale_cols);
#pragma unroll
        for (int b = 0; b < DSV4_BATCH_TILE; ++b) {
            int batch_idx = batch_base + b;
            if (batch_idx < B) {
                const __nv_bfloat16* x = input + batch_idx * K;
                sums[b] += w0 * __bfloat162float(x[k0]);
                sums[b] += w1 * __bfloat162float(x[k1]);
            }
        }
    }

    __shared__ float smem[GEMV_ROWS * 8 * DSV4_BATCH_TILE];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
#pragma unroll
    for (int b = 0; b < DSV4_BATCH_TILE; ++b) {
        sums[b] = warp_reduce_sum(sums[b]);
        if (lane_id == 0) {
            smem[(row_in_block * warps_per_row + warp_in_row) * DSV4_BATCH_TILE + b] = sums[b];
        }
    }
    __syncthreads();
    if (tid_in_row == 0) {
#pragma unroll
        for (int b = 0; b < DSV4_BATCH_TILE; ++b) {
            int batch_idx = batch_base + b;
            if (batch_idx >= B) continue;
            float total = 0.0f;
            for (int w = 0; w < warps_per_row; ++w) {
                total += smem[(row_in_block * warps_per_row + w) * DSV4_BATCH_TILE + b];
            }
            output[batch_idx * N + row] = __float2bfloat16(total);
        }
    }
}

// NVFP4 GEMV. Each thread reads 16 B (uint4) = 32 packed FP4 weights per
// iteration; the byte-at-a-time form issued 16x the memory transactions for the
// same bytes and ran at ~1% of HBM bandwidth. K % 32 tails fall to the scalar
// loop below.
__global__ void fp4_e2m1_group_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales,
    const float* __restrict__ global_scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B,
    int N,
    int K,
    int group_size,
    int scale_cols)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_idx = blockIdx.y;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N || batch_idx >= B) return;

    const __nv_bfloat16* x = input + batch_idx * K;
    const uint8_t* weight_row = weight + (int64_t)row * (K / 2);
    float sum = fp4_e2m1_row_dot(
        weight_row, scales, x, global_scales[0], row * scale_cols,
        K, group_size, tid_in_row, threads_per_row);

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        output[batch_idx * N + row] = __float2bfloat16(total);
    }
}

__global__ void fp8_f32_block_grouped_gemv_batch_kernel(
    const uint64_t* __restrict__ weight_ptrs,
    const uint64_t* __restrict__ scale_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int max_count,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int block_m,
    int block_k)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_idx = blockIdx.y;
    int compact_expert_idx = blockIdx.z;
    int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N || batch_idx >= max_count || batch_idx >= counts[compact_expert_idx]) return;

    const auto* weight = reinterpret_cast<const uint8_t*>(weight_ptrs[expert_idx]);
    const auto* scales = reinterpret_cast<const float*>(scale_ptrs[expert_idx]);
    const int route = offsets[compact_expert_idx] + batch_idx;
    const __nv_bfloat16* x = input + route * K;
    float sum = 0.0f;
    for (int k = tid_in_row; k < K; k += threads_per_row) {
        const float w = dsv4_decode_fp8_e4m3(weight[row * K + k])
            * fp8_f32_block_scale(scales, row, k, scale_rows, scale_cols, block_m, block_k);
        sum += w * __bfloat162float(x[k]);
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        output[route * N + row] = __float2bfloat16(total);
    }
}

__global__ void fp8_f32_block_grouped_gemv_pair_batch_kernel(
    const uint64_t* __restrict__ weight_a_ptrs,
    const uint64_t* __restrict__ scale_a_ptrs,
    const uint64_t* __restrict__ weight_b_ptrs,
    const uint64_t* __restrict__ scale_b_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output_a,
    __nv_bfloat16* __restrict__ output_b,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int max_count,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int block_m,
    int block_k)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_idx = blockIdx.y;
    int compact_expert_idx = blockIdx.z;
    int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N || batch_idx >= max_count || batch_idx >= counts[compact_expert_idx]) return;

    const auto* weight_a = reinterpret_cast<const uint8_t*>(weight_a_ptrs[expert_idx]);
    const auto* scales_a = reinterpret_cast<const float*>(scale_a_ptrs[expert_idx]);
    const auto* weight_b = reinterpret_cast<const uint8_t*>(weight_b_ptrs[expert_idx]);
    const auto* scales_b = reinterpret_cast<const float*>(scale_b_ptrs[expert_idx]);
    const int route = offsets[compact_expert_idx] + batch_idx;
    const __nv_bfloat16* x = input + route * K;
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (int k = tid_in_row; k < K; k += threads_per_row) {
        const float xv = __bfloat162float(x[k]);
        const float wa = dsv4_decode_fp8_e4m3(weight_a[row * K + k])
            * fp8_f32_block_scale(scales_a, row, k, scale_rows, scale_cols, block_m, block_k);
        const float wb = dsv4_decode_fp8_e4m3(weight_b[row * K + k])
            * fp8_f32_block_scale(scales_b, row, k, scale_rows, scale_cols, block_m, block_k);
        sum_a += wa * xv;
        sum_b += wb * xv;
    }

    sum_a = warp_reduce_sum(sum_a);
    sum_b = warp_reduce_sum(sum_b);
    __shared__ float smem_a[GEMV_ROWS * 8];
    __shared__ float smem_b[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) {
        smem_a[row_in_block * warps_per_row + warp_in_row] = sum_a;
        smem_b[row_in_block * warps_per_row + warp_in_row] = sum_b;
    }
    __syncthreads();
    if (tid_in_row == 0) {
        float total_a = 0.0f;
        float total_b = 0.0f;
        for (int w = 0; w < warps_per_row; w++) {
            total_a += smem_a[row_in_block * warps_per_row + w];
            total_b += smem_b[row_in_block * warps_per_row + w];
        }
        output_a[route * N + row] = __float2bfloat16(total_a);
        output_b[route * N + row] = __float2bfloat16(total_b);
    }
}

__global__ void fp4_e2m1_grouped_gemv_batch_kernel(
    const uint64_t* __restrict__ weight_ptrs,
    const uint64_t* __restrict__ scale_ptrs,
    const uint64_t* __restrict__ global_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int max_count,
    int N,
    int K,
    int group_size,
    int scale_cols)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_idx = blockIdx.y;
    int compact_expert_idx = blockIdx.z;
    int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N || batch_idx >= max_count || batch_idx >= counts[compact_expert_idx]) return;

    const auto* weight = reinterpret_cast<const uint8_t*>(weight_ptrs[expert_idx]);
    const auto* scales = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_idx]);
    const auto* global_scales = reinterpret_cast<const float*>(global_ptrs[expert_idx]);
    const int route = offsets[compact_expert_idx] + batch_idx;
    const __nv_bfloat16* x = input + route * K;
    float sum = fp4_e2m1_row_dot(
        weight + (int64_t)row * (K / 2), scales, x, global_scales[0],
        row * scale_cols, K, group_size, tid_in_row, threads_per_row);

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        output[route * N + row] = __float2bfloat16(total);
    }
}

__global__ void fp4_e2m1_grouped_gemv_pair_batch_kernel(
    const uint64_t* __restrict__ weight_a_ptrs,
    const uint64_t* __restrict__ scale_a_ptrs,
    const uint64_t* __restrict__ global_a_ptrs,
    const uint64_t* __restrict__ weight_b_ptrs,
    const uint64_t* __restrict__ scale_b_ptrs,
    const uint64_t* __restrict__ global_b_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output_a,
    __nv_bfloat16* __restrict__ output_b,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int max_count,
    int N,
    int K,
    int group_size,
    int scale_cols)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_idx = blockIdx.y;
    int compact_expert_idx = blockIdx.z;
    int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N || batch_idx >= max_count || batch_idx >= counts[compact_expert_idx]) return;

    const auto* weight_a = reinterpret_cast<const uint8_t*>(weight_a_ptrs[expert_idx]);
    const auto* scales_a = reinterpret_cast<const uint8_t*>(scale_a_ptrs[expert_idx]);
    const auto* global_a = reinterpret_cast<const float*>(global_a_ptrs[expert_idx]);
    const auto* weight_b = reinterpret_cast<const uint8_t*>(weight_b_ptrs[expert_idx]);
    const auto* scales_b = reinterpret_cast<const uint8_t*>(scale_b_ptrs[expert_idx]);
    const auto* global_b = reinterpret_cast<const float*>(global_b_ptrs[expert_idx]);
    const int route = offsets[compact_expert_idx] + batch_idx;
    const __nv_bfloat16* x = input + route * K;
    const int64_t row_off = (int64_t)row * (K / 2);
    const int scale_base = row * scale_cols;
    // x is re-read for the b half; it is K*2 B against the row's K/2 B of
    // weights and stays hot in L1 across the two calls.
    float sum_a = fp4_e2m1_row_dot(weight_a + row_off, scales_a, x, global_a[0],
                                   scale_base, K, group_size,
                                   tid_in_row, threads_per_row);
    float sum_b = fp4_e2m1_row_dot(weight_b + row_off, scales_b, x, global_b[0],
                                   scale_base, K, group_size,
                                   tid_in_row, threads_per_row);

    sum_a = warp_reduce_sum(sum_a);
    sum_b = warp_reduce_sum(sum_b);
    __shared__ float smem_a[GEMV_ROWS * 8];
    __shared__ float smem_b[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) {
        smem_a[row_in_block * warps_per_row + warp_in_row] = sum_a;
        smem_b[row_in_block * warps_per_row + warp_in_row] = sum_b;
    }
    __syncthreads();
    if (tid_in_row == 0) {
        float total_a = 0.0f;
        float total_b = 0.0f;
        for (int w = 0; w < warps_per_row; w++) {
            total_a += smem_a[row_in_block * warps_per_row + w];
            total_b += smem_b[row_in_block * warps_per_row + w];
        }
        output_a[route * N + row] = __float2bfloat16(total_a);
        output_b[route * N + row] = __float2bfloat16(total_b);
    }
}

__device__ __forceinline__ int dsv4_route_local_expert(
    const int32_t* __restrict__ route_meta,
    int route,
    int local_expert_start,
    int experts_per_rank)
{
    if (route_meta == nullptr) {
        return route % experts_per_rank;
    }
    const int expert = route_meta[route * 3 + 1];
    const int local = expert - local_expert_start;
    return (local >= 0 && local < experts_per_rank) ? local : -1;
}

__device__ __forceinline__ float dsv4_route_weight(
    const int32_t* __restrict__ route_meta,
    int route)
{
    if (route_meta == nullptr) {
        return 1.0f;
    }
    return __int_as_float(route_meta[route * 3 + 2]);
}

__global__ void dsv4_fp8_route_gemv_batch_kernel(
    const uint64_t* __restrict__ weight_ptrs,
    const uint64_t* __restrict__ scale_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ route_meta,
    int local_expert_start,
    int experts_per_rank,
    int num_routes,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int apply_route_weight)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int route = blockIdx.y;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N || route >= num_routes) return;

    const int expert_idx =
        dsv4_route_local_expert(route_meta, route, local_expert_start, experts_per_rank);
    if (expert_idx < 0) return;

    const auto* weight = reinterpret_cast<const uint8_t*>(weight_ptrs[expert_idx]);
    const auto* scales = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_idx]);
    const __nv_bfloat16* x = input + route * K;

    const int block_h = (N + scale_rows - 1) / scale_rows;
    const int block_w = (K + scale_cols - 1) / scale_cols;
    const int sr_raw = row / block_h;
    const int sr = sr_raw < scale_rows ? sr_raw : (scale_rows - 1);
    const int scale_row_offset = sr * scale_cols;
    float sum = 0.0f;
    for (int k = tid_in_row; k < K; k += threads_per_row) {
        const int sc_raw = k / block_w;
        const int sc = sc_raw < scale_cols ? sc_raw : (scale_cols - 1);
        const float w = dsv4_decode_fp8_e4m3(weight[row * K + k])
            * dsv4_decode_e8m0(scales[scale_row_offset + sc]);
        sum += w * __bfloat162float(x[k]);
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        if (apply_route_weight) total *= dsv4_route_weight(route_meta, route);
        output[route * N + row] = __float2bfloat16(total);
    }
}

__global__ void dsv4_fp4_route_gemv_batch_kernel(
    const uint64_t* __restrict__ weight_ptrs,
    const uint64_t* __restrict__ scale_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ route_meta,
    int local_expert_start,
    int experts_per_rank,
    int num_routes,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int apply_route_weight)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int route = blockIdx.y;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N || route >= num_routes) return;

    const int expert_idx =
        dsv4_route_local_expert(route_meta, route, local_expert_start, experts_per_rank);
    if (expert_idx < 0) return;

    const auto* weight = reinterpret_cast<const uint8_t*>(weight_ptrs[expert_idx]);
    const auto* scales = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_idx]);
    const __nv_bfloat16* x = input + route * K;
    const int bytes_per_row = K / 2;
    float sum = 0.0f;
    for (int k = tid_in_row; k < K; k += threads_per_row) {
        const uint8_t packed = weight[row * bytes_per_row + (k >> 1)];
        const uint8_t nibble = (k & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
        const float w = dsv4_decode_fp4_e2m1(nibble)
            * dsv4_block_scale(scales, row, k, N, K, scale_rows, scale_cols);
        sum += w * __bfloat162float(x[k]);
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        if (apply_route_weight) total *= dsv4_route_weight(route_meta, route);
        output[route * N + row] = __float2bfloat16(total);
    }
}

// Batched W8A16 GEMM: B_TILE inputs share one weight read (vs B separate GEMVs
// re-reading weight B times). Each thread holds B_TILE accumulators; weight is
// loaded once per K-step and multiplied against B_TILE input vectors. Reads
// INT8 directly (4 weights per uint32, no nibble unpack) — this is the
// multi-request decode path (B>=2).
#define W8A16_GEMM_BTILE 4
__global__ void w8a16_gemm_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int group_size)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_base = blockIdx.y * W8A16_GEMM_BTILE;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;

    if (row >= N) return;

    float sum[W8A16_GEMM_BTILE];
    #pragma unroll
    for (int b = 0; b < W8A16_GEMM_BTILE; b++) sum[b] = 0.0f;

    int num_groups = K / group_size;
    int valid_b = min(W8A16_GEMM_BTILE, B - batch_base);

    for (int k = tid_in_row * 4; k < K; k += threads_per_row * 4) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * K + k]);
        float w0 = (float)static_cast<int8_t>(packed) * scale_f;
        float w1 = (float)static_cast<int8_t>(packed >> 8) * scale_f;
        float w2 = (float)static_cast<int8_t>(packed >> 16) * scale_f;
        float w3 = (float)static_cast<int8_t>(packed >> 24) * scale_f;

        #pragma unroll
        for (int b = 0; b < W8A16_GEMM_BTILE; b++) {
            if (b >= valid_b) break;
            const __nv_bfloat16* xb = input + (batch_base + b) * K;
            sum[b] += w0 * __bfloat162float(xb[k]);
            sum[b] += w1 * __bfloat162float(xb[k + 1]);
            sum[b] += w2 * __bfloat162float(xb[k + 2]);
            sum[b] += w3 * __bfloat162float(xb[k + 3]);
        }
    }

    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    #pragma unroll
    for (int b = 0; b < W8A16_GEMM_BTILE; b++) {
        if (b < valid_b) sum[b] = warp_reduce_sum(sum[b]);
    }

    __shared__ float smem_out[GEMV_ROWS * W8A16_GEMM_BTILE * 8];
    if (lane_id == 0) {
        #pragma unroll
        for (int b = 0; b < W8A16_GEMM_BTILE; b++) {
            if (b < valid_b)
                smem_out[(row_in_block * W8A16_GEMM_BTILE + b) * warps_per_row + warp_in_row] = sum[b];
        }
    }
    __syncthreads();
    if (tid_in_row == 0) {
        #pragma unroll
        for (int b = 0; b < W8A16_GEMM_BTILE; b++) {
            if (b >= valid_b) break;
            float total = 0.0f;
            for (int w = 0; w < warps_per_row; w++)
                total += smem_out[(row_in_block * W8A16_GEMM_BTILE + b) * warps_per_row + w];
            output[(batch_base + b) * N + row] = __float2bfloat16(total);
        }
    }
}

// Batched W8A16 GEMV: [B, K] × [N, K]^T → [B, N]
__global__ void w8a16_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int group_size)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_idx = blockIdx.y;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;

    if (row >= N) return;
    const __nv_bfloat16* x = input + batch_idx * K;
    float sum = 0.0f;
    int num_groups = K / group_size;

    for (int k = tid_in_row * 4; k < K; k += threads_per_row * 4) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * K + k]);
        int8_t v0 = static_cast<int8_t>(packed);
        int8_t v1 = static_cast<int8_t>(packed >> 8);
        int8_t v2 = static_cast<int8_t>(packed >> 16);
        int8_t v3 = static_cast<int8_t>(packed >> 24);
        sum += static_cast<float>(v0) * scale_f * __bfloat162float(x[k]);
        sum += static_cast<float>(v1) * scale_f * __bfloat162float(x[k + 1]);
        sum += static_cast<float>(v2) * scale_f * __bfloat162float(x[k + 2]);
        sum += static_cast<float>(v3) * scale_f * __bfloat162float(x[k + 3]);
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        output[batch_idx * N + row] = __float2bfloat16(total);
    }
}

// Batched W4A16 GEMM: B_TILE inputs share one weight read (vs B separate GEMVs
// re-reading weight B times). Each thread holds B_TILE accumulators; weight is
// loaded once per K-step and multiplied against B_TILE input vectors. Reads
// INT8 directly (4 weights per uint32, no nibble unpack) — this is the
// multi-request decode path (B>=2).
#define W4A16_GEMM_BTILE 4

// Batched W4A16 GEMV: [B, K] × [N, K/2]^T → [B, N]
// Same nibble extraction as single W4A16, with batch dimension in grid.y.
//
// sm_70 (V100) variant: shared-mem caches the input vector so GEMV_ROWS rows
// share one HBM read of input (~75% less input HBM traffic). Compute structure
// (8 int4/iter, uint32 loads) is unchanged to keep register pressure identical.
#if __CUDA_ARCH__ == 700
__global__ void w4a16_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int group_size)
{
    extern __shared__ __nv_bfloat16 smem_input[];

    int batch_idx = blockIdx.y;
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;

    // Cooperatively load input into shared memory (all rows in block share it)
    const __nv_bfloat16* x = input + batch_idx * K;
    for (int i = threadIdx.x; i < K; i += GEMV_THREADS)
        smem_input[i] = x[i];
    __syncthreads();

    if (row >= N) return;

    float sum = 0.0f;
    int num_groups = K / group_size;
    int bytes_per_row = K / 2;

    for (int k = tid_in_row * 8; k < K; k += threads_per_row * 8) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * bytes_per_row + k / 2]);

        uint32_t lo4 = packed & 0x0F0F0F0Fu;
        uint32_t hi4 = (packed >> 4) & 0x0F0F0F0Fu;

        int lo0 = static_cast<int>(lo4 & 0xFF) - 8;
        int hi0 = static_cast<int>(hi4 & 0xFF) - 8;
        int lo1 = static_cast<int>((lo4 >> 8) & 0xFF) - 8;
        int hi1 = static_cast<int>((hi4 >> 8) & 0xFF) - 8;
        int lo2 = static_cast<int>((lo4 >> 16) & 0xFF) - 8;
        int hi2 = static_cast<int>((hi4 >> 16) & 0xFF) - 8;
        int lo3 = static_cast<int>((lo4 >> 24) & 0xFF) - 8;
        int hi3 = static_cast<int>((hi4 >> 24) & 0xFF) - 8;

        sum += static_cast<float>(lo0) * scale_f * __bfloat162float(smem_input[k]);
        sum += static_cast<float>(hi0) * scale_f * __bfloat162float(smem_input[k + 1]);
        sum += static_cast<float>(lo1) * scale_f * __bfloat162float(smem_input[k + 2]);
        sum += static_cast<float>(hi1) * scale_f * __bfloat162float(smem_input[k + 3]);
        sum += static_cast<float>(lo2) * scale_f * __bfloat162float(smem_input[k + 4]);
        sum += static_cast<float>(hi2) * scale_f * __bfloat162float(smem_input[k + 5]);
        sum += static_cast<float>(lo3) * scale_f * __bfloat162float(smem_input[k + 6]);
        sum += static_cast<float>(hi3) * scale_f * __bfloat162float(smem_input[k + 7]);
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        output[batch_idx * N + row] = __float2bfloat16(total);
    }
}
#else
__global__ void w4a16_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int group_size)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_idx = blockIdx.y;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;

    if (row >= N) return;
    const __nv_bfloat16* x = input + batch_idx * K;
    float sum = 0.0f;
    int num_groups = K / group_size;
    int bytes_per_row = K / 2;

    for (int k = tid_in_row * 8; k < K; k += threads_per_row * 8) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * bytes_per_row + k / 2]);

        uint32_t lo4 = packed & 0x0F0F0F0Fu;
        uint32_t hi4 = (packed >> 4) & 0x0F0F0F0Fu;

        int lo0 = static_cast<int>(lo4 & 0xFF) - 8;
        int hi0 = static_cast<int>(hi4 & 0xFF) - 8;
        int lo1 = static_cast<int>((lo4 >> 8) & 0xFF) - 8;
        int hi1 = static_cast<int>((hi4 >> 8) & 0xFF) - 8;
        int lo2 = static_cast<int>((lo4 >> 16) & 0xFF) - 8;
        int hi2 = static_cast<int>((hi4 >> 16) & 0xFF) - 8;
        int lo3 = static_cast<int>((lo4 >> 24) & 0xFF) - 8;
        int hi3 = static_cast<int>((hi4 >> 24) & 0xFF) - 8;

        sum += static_cast<float>(lo0) * scale_f * __bfloat162float(x[k]);
        sum += static_cast<float>(hi0) * scale_f * __bfloat162float(x[k + 1]);
        sum += static_cast<float>(lo1) * scale_f * __bfloat162float(x[k + 2]);
        sum += static_cast<float>(hi1) * scale_f * __bfloat162float(x[k + 3]);
        sum += static_cast<float>(lo2) * scale_f * __bfloat162float(x[k + 4]);
        sum += static_cast<float>(hi2) * scale_f * __bfloat162float(x[k + 5]);
        sum += static_cast<float>(lo3) * scale_f * __bfloat162float(x[k + 6]);
        sum += static_cast<float>(hi3) * scale_f * __bfloat162float(x[k + 7]);
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        output[batch_idx * N + row] = __float2bfloat16(total);
    }
}
#endif

// Grouped W4A16 GEMV (MoE): one weight/scale ptr per expert, routed tokens.
// W4A16 nibble-extraction (zero-point 8) fused with the grouped dispatch
// (offsets/counts/expert_indices) so each expert processes its routed rows.
// Marlin-style grouped GEMV for MoE: weight shared across all tokens routed
// to the same expert. Uses uint4 loads and half2 FP16 arithmetic.
#define GROUPED_BTILE 16

__global__ void w4a16_grouped_gemv_batch_kernel(
    const uint64_t* __restrict__ weight_ptrs,
    const uint64_t* __restrict__ scale_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int max_count,
    int N,
    int K,
    int group_size,
    uint32_t xor_mask)
{
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / threads_per_row;
    int batch_base = blockIdx.y * GROUPED_BTILE;
    int compact_expert_idx = blockIdx.z;
    int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    int tid_in_row = threadIdx.x % threads_per_row;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (row >= N) return;
    int valid_b = min(GROUPED_BTILE, counts[compact_expert_idx] - batch_base);
    if (valid_b <= 0) return;

    const auto* weight = reinterpret_cast<const uint8_t*>(weight_ptrs[expert_idx]);
    const auto* scales = reinterpret_cast<const __nv_bfloat16*>(scale_ptrs[expert_idx]);
    int num_groups = K / group_size;
    int bytes_per_row = K / 2;

    const uint32_t MASK4 = 0x0f0f0f0fu;
    const uint32_t SUB   = 0x64086408u;
    const half2 SUB_H2   = *reinterpret_cast<const half2*>(&SUB);

    half2 sum_h2[GROUPED_BTILE];
    #pragma unroll
    for (int b = 0; b < GROUPED_BTILE; b++)
        sum_h2[b] = __float2half2_rn(0.0f);

    const uint8_t* weight_row = weight + (int64_t)row * bytes_per_row;

    for (int k = tid_in_row * 32; k < K; k += threads_per_row * 32) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        half2 scale_h2 = __half2half2(__float2half(scale_f));

        uint4 packed = __ldg(reinterpret_cast<const uint4*>(weight_row + k / 2));
        uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};

        #pragma unroll
        for (int w = 0; w < 4; w++) {
            uint32_t p = words[w];
            int kk = k + w * 8;

            uint32_t lo_all = (p & MASK4) ^ xor_mask;
            uint32_t hi_all = ((p >> 4) & MASK4) ^ xor_mask;

            uint32_t lo01 = (0x6400u | (lo_all & 0xffu)) |
                            ((0x6400u | ((lo_all >> 8) & 0xffu)) << 16);
            uint32_t lo23 = (0x6400u | ((lo_all >> 16) & 0xffu)) |
                            ((0x6400u | ((lo_all >> 24) & 0xffu)) << 16);
            uint32_t hi01 = (0x6400u | (hi_all & 0xffu)) |
                            ((0x6400u | ((hi_all >> 8) & 0xffu)) << 16);
            uint32_t hi23 = (0x6400u | ((hi_all >> 16) & 0xffu)) |
                            ((0x6400u | ((hi_all >> 24) & 0xffu)) << 16);

            half2 w0 = __hsub2(*reinterpret_cast<half2*>(&lo01), SUB_H2);
            half2 w1 = __hsub2(*reinterpret_cast<half2*>(&hi01), SUB_H2);
            half2 w2 = __hsub2(*reinterpret_cast<half2*>(&lo23), SUB_H2);
            half2 w3 = __hsub2(*reinterpret_cast<half2*>(&hi23), SUB_H2);

            half2 wx  = __halves2half2(w0.x, w1.x);
            half2 wy  = __halves2half2(w0.y, w1.y);
            half2 wx2 = __halves2half2(w2.x, w3.x);
            half2 wy2 = __halves2half2(w2.y, w3.y);

            half2 wsx  = __hmul2(wx, scale_h2);
            half2 wsy  = __hmul2(wy, scale_h2);
            half2 wsx2 = __hmul2(wx2, scale_h2);
            half2 wsy2 = __hmul2(wy2, scale_h2);

            #pragma unroll
            for (int b = 0; b < GROUPED_BTILE; b++) {
                if (b >= valid_b) break;
                int route = offsets[compact_expert_idx] + batch_base + b;
                const __nv_bfloat16* xb = input + route * K;

                uint4 xpacked = __ldg(reinterpret_cast<const uint4*>(&xb[kk]));
                const __nv_bfloat16* xbv = reinterpret_cast<const __nv_bfloat16*>(&xpacked);
                half2 x01 = __halves2half2(__float2half(__bfloat162float(xbv[0])),
                                           __float2half(__bfloat162float(xbv[1])));
                half2 x23 = __halves2half2(__float2half(__bfloat162float(xbv[2])),
                                           __float2half(__bfloat162float(xbv[3])));
                half2 x45 = __halves2half2(__float2half(__bfloat162float(xbv[4])),
                                           __float2half(__bfloat162float(xbv[5])));
                half2 x67 = __halves2half2(__float2half(__bfloat162float(xbv[6])),
                                           __float2half(__bfloat162float(xbv[7])));

                sum_h2[b] = __hfma2(wsx,  x01, sum_h2[b]);
                sum_h2[b] = __hfma2(wsy,  x23, sum_h2[b]);
                sum_h2[b] = __hfma2(wsx2, x45, sum_h2[b]);
                sum_h2[b] = __hfma2(wsy2, x67, sum_h2[b]);
            }
        }
    }

    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = tid_in_row / WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;

    if (warps_per_row == 1) {
        #pragma unroll
        for (int b = 0; b < GROUPED_BTILE; b++) {
            if (b >= valid_b) break;
            half sum_h = __hadd(sum_h2[b].x, sum_h2[b].y);
            float sum = __half2float(sum_h);
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) {
                int route = offsets[compact_expert_idx] + batch_base + b;
                output[route * N + row] = __float2bfloat16(sum);
            }
        }
    } else {
        __shared__ float warp_sums[GEMV_ROWS * GROUPED_BTILE * 8];
        #pragma unroll
        for (int b = 0; b < GROUPED_BTILE; b++) {
            if (b >= valid_b) break;
            half sum_h = __hadd(sum_h2[b].x, sum_h2[b].y);
            float sum = __half2float(sum_h);
            sum = warp_reduce_sum(sum);
            int idx = row_in_block * GROUPED_BTILE * warps_per_row + b * warps_per_row + warp_in_row;
            if (lane_id == 0) warp_sums[idx] = sum;
        }
        __syncthreads();
        if (warp_in_row == 0) {
            #pragma unroll
            for (int b = 0; b < GROUPED_BTILE; b++) {
                if (b >= valid_b) break;
                float total = 0.0f;
                #pragma unroll
                for (int w = 0; w < warps_per_row; w++) {
                    int idx = row_in_block * GROUPED_BTILE * warps_per_row + b * warps_per_row + w;
                    total += warp_sums[idx];
                }
                if (lane_id == 0) {
                    int route = offsets[compact_expert_idx] + batch_base + b;
                    output[route * N + row] = __float2bfloat16(total);
                }
            }
        }
    }
}

// Marlin-style grouped pair GEMV for MoE gate+up: two weight matrices share
// the same input. Weight shared across all tokens routed to the same expert.
// DSv4 clamped SwiGLU (matches dsv4_swiglu_clamped_one / fp8d_swiglu_clamped):
//   gate = min(gate, limit); up = clamp(up, -limit, limit);
//   out  = (gate / (1 + exp(-gate))) * up
static __device__ __forceinline__ float w4a16_swiglu_clamped(float gate, float up,
                                                              float limit) {
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
    return (gate / (1.0f + __expf(-gate))) * up;
}

__global__ void w4a16_grouped_gemv_pair_batch_kernel(
    const uint64_t* __restrict__ weight_a_ptrs,
    const uint64_t* __restrict__ scale_a_ptrs,
    const uint64_t* __restrict__ weight_b_ptrs,
    const uint64_t* __restrict__ scale_b_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output_a,
    __nv_bfloat16* __restrict__ output_b,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int max_count,
    int N,
    int K,
    int group_size,
    uint32_t xor_mask,
    bool fuse_swiglu,
    float swiglu_limit)
{
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / threads_per_row;
    int batch_base = blockIdx.y * GROUPED_BTILE;
    int compact_expert_idx = blockIdx.z;
    int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    int tid_in_row = threadIdx.x % threads_per_row;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (row >= N) return;
    int valid_b = min(GROUPED_BTILE, counts[compact_expert_idx] - batch_base);
    if (valid_b <= 0) return;

    const auto* weight_a = reinterpret_cast<const uint8_t*>(weight_a_ptrs[expert_idx]);
    const auto* scales_a = reinterpret_cast<const __nv_bfloat16*>(scale_a_ptrs[expert_idx]);
    const auto* weight_b = reinterpret_cast<const uint8_t*>(weight_b_ptrs[expert_idx]);
    const auto* scales_b = reinterpret_cast<const __nv_bfloat16*>(scale_b_ptrs[expert_idx]);
    int num_groups = K / group_size;
    int bytes_per_row = K / 2;

    const uint32_t MASK4 = 0x0f0f0f0fu;
    const uint32_t SUB   = 0x64086408u;
    const half2 SUB_H2   = *reinterpret_cast<const half2*>(&SUB);

    half2 sum_a[GROUPED_BTILE];
    half2 sum_b[GROUPED_BTILE];
    #pragma unroll
    for (int b = 0; b < GROUPED_BTILE; b++) {
        sum_a[b] = __float2half2_rn(0.0f);
        sum_b[b] = __float2half2_rn(0.0f);
    }

    const uint8_t* wrow_a = weight_a + (int64_t)row * bytes_per_row;
    const uint8_t* wrow_b = weight_b + (int64_t)row * bytes_per_row;

    for (int k = tid_in_row * 32; k < K; k += threads_per_row * 32) {
        float scale_a_f = __bfloat162float(scales_a[row * num_groups + k / group_size]);
        float scale_b_f = __bfloat162float(scales_b[row * num_groups + k / group_size]);
        half2 scale_a_h2 = __half2half2(__float2half(scale_a_f));
        half2 scale_b_h2 = __half2half2(__float2half(scale_b_f));

        uint4 packed_a = __ldg(reinterpret_cast<const uint4*>(wrow_a + k / 2));
        uint4 packed_b = __ldg(reinterpret_cast<const uint4*>(wrow_b + k / 2));
        uint32_t wa[4] = {packed_a.x, packed_a.y, packed_a.z, packed_a.w};
        uint32_t wb[4] = {packed_b.x, packed_b.y, packed_b.z, packed_b.w};

        #pragma unroll
        for (int w = 0; w < 4; w++) {
            uint32_t pa = wa[w];
            uint32_t pb = wb[w];
            int kk = k + w * 8;

            // Dequant A
            uint32_t lo_a = (pa & MASK4) ^ xor_mask;
            uint32_t hi_a = ((pa >> 4) & MASK4) ^ xor_mask;
            uint32_t lo01_a = (0x6400u | (lo_a & 0xffu)) | ((0x6400u | ((lo_a >> 8) & 0xffu)) << 16);
            uint32_t lo23_a = (0x6400u | ((lo_a >> 16) & 0xffu)) | ((0x6400u | ((lo_a >> 24) & 0xffu)) << 16);
            uint32_t hi01_a = (0x6400u | (hi_a & 0xffu)) | ((0x6400u | ((hi_a >> 8) & 0xffu)) << 16);
            uint32_t hi23_a = (0x6400u | ((hi_a >> 16) & 0xffu)) | ((0x6400u | ((hi_a >> 24) & 0xffu)) << 16);
            half2 w0a = __hsub2(*reinterpret_cast<half2*>(&lo01_a), SUB_H2);
            half2 w1a = __hsub2(*reinterpret_cast<half2*>(&hi01_a), SUB_H2);
            half2 w2a = __hsub2(*reinterpret_cast<half2*>(&lo23_a), SUB_H2);
            half2 w3a = __hsub2(*reinterpret_cast<half2*>(&hi23_a), SUB_H2);
            half2 wsx_a = __hmul2(__halves2half2(w0a.x, w1a.x), scale_a_h2);
            half2 wsy_a = __hmul2(__halves2half2(w0a.y, w1a.y), scale_a_h2);
            half2 wsx2_a = __hmul2(__halves2half2(w2a.x, w3a.x), scale_a_h2);
            half2 wsy2_a = __hmul2(__halves2half2(w2a.y, w3a.y), scale_a_h2);

            // Dequant B
            uint32_t lo_b = (pb & MASK4) ^ xor_mask;
            uint32_t hi_b = ((pb >> 4) & MASK4) ^ xor_mask;
            uint32_t lo01_b = (0x6400u | (lo_b & 0xffu)) | ((0x6400u | ((lo_b >> 8) & 0xffu)) << 16);
            uint32_t lo23_b = (0x6400u | ((lo_b >> 16) & 0xffu)) | ((0x6400u | ((lo_b >> 24) & 0xffu)) << 16);
            uint32_t hi01_b = (0x6400u | (hi_b & 0xffu)) | ((0x6400u | ((hi_b >> 8) & 0xffu)) << 16);
            uint32_t hi23_b = (0x6400u | ((hi_b >> 16) & 0xffu)) | ((0x6400u | ((hi_b >> 24) & 0xffu)) << 16);
            half2 w0b = __hsub2(*reinterpret_cast<half2*>(&lo01_b), SUB_H2);
            half2 w1b = __hsub2(*reinterpret_cast<half2*>(&hi01_b), SUB_H2);
            half2 w2b = __hsub2(*reinterpret_cast<half2*>(&lo23_b), SUB_H2);
            half2 w3b = __hsub2(*reinterpret_cast<half2*>(&hi23_b), SUB_H2);
            half2 wsx_b = __hmul2(__halves2half2(w0b.x, w1b.x), scale_b_h2);
            half2 wsy_b = __hmul2(__halves2half2(w0b.y, w1b.y), scale_b_h2);
            half2 wsx2_b = __hmul2(__halves2half2(w2b.x, w3b.x), scale_b_h2);
            half2 wsy2_b = __hmul2(__halves2half2(w2b.y, w3b.y), scale_b_h2);

            #pragma unroll
            for (int b = 0; b < GROUPED_BTILE; b++) {
                if (b >= valid_b) break;
                int route = offsets[compact_expert_idx] + batch_base + b;
                const __nv_bfloat16* xb = input + route * K;

                uint4 xpacked = __ldg(reinterpret_cast<const uint4*>(&xb[kk]));
                const __nv_bfloat16* xbv = reinterpret_cast<const __nv_bfloat16*>(&xpacked);
                half2 x01 = __halves2half2(__float2half(__bfloat162float(xbv[0])),
                                           __float2half(__bfloat162float(xbv[1])));
                half2 x23 = __halves2half2(__float2half(__bfloat162float(xbv[2])),
                                           __float2half(__bfloat162float(xbv[3])));
                half2 x45 = __halves2half2(__float2half(__bfloat162float(xbv[4])),
                                           __float2half(__bfloat162float(xbv[5])));
                half2 x67 = __halves2half2(__float2half(__bfloat162float(xbv[6])),
                                           __float2half(__bfloat162float(xbv[7])));

                sum_a[b] = __hfma2(wsx_a,  x01, sum_a[b]);
                sum_a[b] = __hfma2(wsy_a,  x23, sum_a[b]);
                sum_a[b] = __hfma2(wsx2_a, x45, sum_a[b]);
                sum_a[b] = __hfma2(wsy2_a, x67, sum_a[b]);
                sum_b[b] = __hfma2(wsx_b,  x01, sum_b[b]);
                sum_b[b] = __hfma2(wsy_b,  x23, sum_b[b]);
                sum_b[b] = __hfma2(wsx2_b, x45, sum_b[b]);
                sum_b[b] = __hfma2(wsy2_b, x67, sum_b[b]);
            }
        }
    }

    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = tid_in_row / WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;

    if (warps_per_row == 1) {
        #pragma unroll
        for (int b = 0; b < GROUPED_BTILE; b++) {
            if (b >= valid_b) break;
            float sa = __half2float(__hadd(sum_a[b].x, sum_a[b].y));
            float sb = __half2float(__hadd(sum_b[b].x, sum_b[b].y));
            sa = warp_reduce_sum(sa);
            sb = warp_reduce_sum(sb);
            if (lane_id == 0) {
                int route = offsets[compact_expert_idx] + batch_base + b;
                if (fuse_swiglu) {
                    output_a[route * N + row] =
                        __float2bfloat16(w4a16_swiglu_clamped(sa, sb, swiglu_limit));
                } else {
                    output_a[route * N + row] = __float2bfloat16(sa);
                    output_b[route * N + row] = __float2bfloat16(sb);
                }
            }
        }
    } else {
        __shared__ float warp_sa[GEMV_ROWS * GROUPED_BTILE * 8];
        __shared__ float warp_sb[GEMV_ROWS * GROUPED_BTILE * 8];
        #pragma unroll
        for (int b = 0; b < GROUPED_BTILE; b++) {
            if (b >= valid_b) break;
            float sa = __half2float(__hadd(sum_a[b].x, sum_a[b].y));
            float sb = __half2float(__hadd(sum_b[b].x, sum_b[b].y));
            sa = warp_reduce_sum(sa);
            sb = warp_reduce_sum(sb);
            int idx = row_in_block * GROUPED_BTILE * warps_per_row + b * warps_per_row + warp_in_row;
            if (lane_id == 0) {
                warp_sa[idx] = sa;
                warp_sb[idx] = sb;
            }
        }
        __syncthreads();
        if (warp_in_row == 0) {
            #pragma unroll
            for (int b = 0; b < GROUPED_BTILE; b++) {
                if (b >= valid_b) break;
                float ta = 0.0f, tb = 0.0f;
                #pragma unroll
                for (int w = 0; w < warps_per_row; w++) {
                    int idx = row_in_block * GROUPED_BTILE * warps_per_row + b * warps_per_row + w;
                    ta += warp_sa[idx];
                    tb += warp_sb[idx];
                }
                if (lane_id == 0) {
                    int route = offsets[compact_expert_idx] + batch_base + b;
                    if (fuse_swiglu) {
                        output_a[route * N + row] =
                            __float2bfloat16(w4a16_swiglu_clamped(ta, tb, swiglu_limit));
                    } else {
                        output_a[route * N + row] = __float2bfloat16(ta);
                        output_b[route * N + row] = __float2bfloat16(tb);
                    }
                }
            }
        }
    }
}

// ============================================================================
// W4AFP8 custom M=1 decode kernels — mirror dsv4_fp8_decode_moe.cu structure.
// Compact (work scales with real routed rows), warp-per-row ownership (no
// shared-mem reduction), fused gate+up+SwiGLU. W4AFP8 dequant: two's-complement
// nibble → (n-8) via the 0x6400 half trick, × BF16 per-128-group scale.
// ============================================================================

#define W4D_WARP_SIZE 32
#define W4D_WARPS 8
#define W4D_THREADS (W4D_WARPS * W4D_WARP_SIZE)
#define W4D_ACT_TILE 8    // activation rows per chunk (grid.y)
#define W4D_VEC 32        // INT4 elements per uint4 load (2 per byte)
#define W4D_MIN_BLOCKS_PER_SM 4

static __device__ __forceinline__ float w4d_warp_reduce_sum(float val) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, off);
    return val;
}

// DSv4 clamped SwiGLU (matches w4a16_swiglu_clamped / fp8d_swiglu_clamped).
static __device__ __forceinline__ float w4d_swiglu_clamped(float gate, float up,
                                                            float limit) {
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
    return (gate / (1.0f + __expf(-gate))) * up;
}

// Dequant 8 INT4 nibbles (one uint32) to scaled floats, dot with 8 BF16 acts.
// Float accumulation (matches the FP8 decode kernel's accuracy).
static __device__ __forceinline__ float
w4d_dot8(float acc, uint32_t packed, uint32_t xor_mask, half2 scale_h2,
         const __nv_bfloat16* __restrict__ xp) {
    const uint32_t MASK4 = 0x0f0f0f0fu;
    const uint32_t SUB   = 0x64086408u;
    const half2 SUB_H2   = *reinterpret_cast<const half2*>(&SUB);

    uint32_t lo = (packed & MASK4) ^ xor_mask;
    uint32_t hi = ((packed >> 4) & MASK4) ^ xor_mask;
    uint32_t lo01 = (0x6400u | (lo & 0xffu)) | ((0x6400u | ((lo >> 8) & 0xffu)) << 16);
    uint32_t lo23 = (0x6400u | ((lo >> 16) & 0xffu)) | ((0x6400u | ((lo >> 24) & 0xffu)) << 16);
    uint32_t hi01 = (0x6400u | (hi & 0xffu)) | ((0x6400u | ((hi >> 8) & 0xffu)) << 16);
    uint32_t hi23 = (0x6400u | ((hi >> 16) & 0xffu)) | ((0x6400u | ((hi >> 24) & 0xffu)) << 16);
    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&lo01), SUB_H2);
    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&hi01), SUB_H2);
    half2 w2 = __hsub2(*reinterpret_cast<half2*>(&lo23), SUB_H2);
    half2 w3 = __hsub2(*reinterpret_cast<half2*>(&hi23), SUB_H2);
    half2 s0 = __hmul2(__halves2half2(w0.x, w1.x), scale_h2);
    half2 s1 = __hmul2(__halves2half2(w0.y, w1.y), scale_h2);
    half2 s2 = __hmul2(__halves2half2(w2.x, w3.x), scale_h2);
    half2 s3 = __hmul2(__halves2half2(w2.y, w3.y), scale_h2);

    float2 f0 = __half22float2(s0);
    float2 f1 = __half22float2(s1);
    float2 f2 = __half22float2(s2);
    float2 f3 = __half22float2(s3);

    acc = __fmaf_rn(f0.x, __bfloat162float(xp[0]), acc);
    acc = __fmaf_rn(f0.y, __bfloat162float(xp[1]), acc);
    acc = __fmaf_rn(f1.x, __bfloat162float(xp[2]), acc);
    acc = __fmaf_rn(f1.y, __bfloat162float(xp[3]), acc);
    acc = __fmaf_rn(f2.x, __bfloat162float(xp[4]), acc);
    acc = __fmaf_rn(f2.y, __bfloat162float(xp[5]), acc);
    acc = __fmaf_rn(f3.x, __bfloat162float(xp[6]), acc);
    acc = __fmaf_rn(f3.y, __bfloat162float(xp[7]), acc);
    return acc;
}

// Fused gate+up+SwiGLU decode grouped GEMV: each warp owns one row of N, reads
// those gate/up INT4 rows once, writes act = silu(gate·x) * (up·x) directly.
// Grid: (N/W4D_WARPS, max_count/W4D_ACT_TILE, num_experts). N = intermediate,
// K = hidden. Scales: BF16 [N, K/group_size] per expert.
__global__ __launch_bounds__(W4D_THREADS, W4D_MIN_BLOCKS_PER_SM)
void w4afp8_grouped_swiglu_decode_kernel(
    const uint64_t* __restrict__ weight_gate_ptrs,
    const uint64_t* __restrict__ scale_gate_ptrs,
    const uint64_t* __restrict__ weight_up_ptrs,
    const uint64_t* __restrict__ scale_up_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ act,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int N, int K, int group_size,
    uint32_t xor_mask,
    float limit)
{
    const int compact_expert_idx = blockIdx.z;
    const int expert_M = counts[compact_expert_idx];
    const int chunk_base = blockIdx.y * W4D_ACT_TILE;
    if (chunk_base >= expert_M) return;
    const int warp = threadIdx.x / W4D_WARP_SIZE;
    const int lane = threadIdx.x % W4D_WARP_SIZE;
    const int row = blockIdx.x * W4D_WARPS + warp;
    if (row >= N) return;
    const int tile_raw = expert_M - chunk_base;
    const int tile = tile_raw < W4D_ACT_TILE ? tile_raw : W4D_ACT_TILE;
    const int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    const int route_base = offsets[compact_expert_idx] + chunk_base;

    const auto* wg = reinterpret_cast<const uint8_t*>(weight_gate_ptrs[expert_idx]);
    const auto* wu = reinterpret_cast<const uint8_t*>(weight_up_ptrs[expert_idx]);
    const auto* sg = reinterpret_cast<const __nv_bfloat16*>(scale_gate_ptrs[expert_idx]);
    const auto* su = reinterpret_cast<const __nv_bfloat16*>(scale_up_ptrs[expert_idx]);
    const int num_groups = K / group_size;
    const int bytes_per_row = K / 2;

    float acc_g[W4D_ACT_TILE];
    float acc_u[W4D_ACT_TILE];
#pragma unroll
    for (int b = 0; b < W4D_ACT_TILE; ++b) {
        acc_g[b] = 0.0f;
        acc_u[b] = 0.0f;
    }

    const int kv = K / W4D_VEC;
    for (int v = lane; v < kv; v += W4D_WARP_SIZE) {
        const int k = v * W4D_VEC;
        const int g = k / group_size;
        half2 scale_g_h2 = __half2half2(__float2half(__bfloat162float(sg[row * num_groups + g])));
        half2 scale_u_h2 = __half2half2(__float2half(__bfloat162float(su[row * num_groups + g])));

        const uint8_t* wg_row = wg + (int64_t)row * bytes_per_row;
        const uint8_t* wu_row = wu + (int64_t)row * bytes_per_row;
        uint4 wg4 = *reinterpret_cast<const uint4*>(wg_row + k / 2);
        uint4 wu4 = *reinterpret_cast<const uint4*>(wu_row + k / 2);
        uint32_t wga[4] = {wg4.x, wg4.y, wg4.z, wg4.w};
        uint32_t wua[4] = {wu4.x, wu4.y, wu4.z, wu4.w};

#pragma unroll
        for (int b = 0; b < W4D_ACT_TILE; ++b) {
            if (b < tile) {
                const __nv_bfloat16* xp = input + (int64_t)(route_base + b) * K + k;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    acc_g[b] = w4d_dot8(acc_g[b], wga[w], xor_mask, scale_g_h2, xp + w * 8);
                    acc_u[b] = w4d_dot8(acc_u[b], wua[w], xor_mask, scale_u_h2, xp + w * 8);
                }
            }
        }
    }

#pragma unroll
    for (int b = 0; b < W4D_ACT_TILE; ++b) {
        if (b < tile) {
            acc_g[b] = w4d_warp_reduce_sum(acc_g[b]);
            acc_u[b] = w4d_warp_reduce_sum(acc_u[b]);
            if (lane == 0) {
                act[(int64_t)(route_base + b) * N + row] =
                    __float2bfloat16(w4d_swiglu_clamped(acc_g[b], acc_u[b], limit));
            }
        }
    }
}

extern "C" cudaError_t w4afp8_grouped_swiglu_decode_cuda(
    const uint64_t* weight_gate_ptrs,
    const uint64_t* scale_gate_ptrs,
    const uint64_t* weight_up_ptrs,
    const uint64_t* scale_up_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* act,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N, int K, int group_size,
    uint32_t xor_mask,
    float limit,
    cudaStream_t stream)
{
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0) return cudaSuccess;
    if (K % W4D_VEC != 0) return cudaErrorInvalidValue;
    dim3 block(W4D_THREADS);
    dim3 grid((N + W4D_WARPS - 1) / W4D_WARPS,
              (max_count + W4D_ACT_TILE - 1) / W4D_ACT_TILE,
              num_experts);
    w4afp8_grouped_swiglu_decode_kernel<<<grid, block, 0, stream>>>(
        weight_gate_ptrs, scale_gate_ptrs, weight_up_ptrs, scale_up_ptrs,
        input, act, offsets, counts, expert_indices,
        N, K, group_size, xor_mask, limit);
    return cudaGetLastError();
}

// Q6_K (GGUF) native packed GEMV + dequant.
//
// One superblock = 256 K-dim elements = 210 bytes:
//   ql:[128]  | qh:[64]  | scales:[16 × i8]  | d:f16(2)
//
// Element layout mirrors llama.cpp `dequantize_row_q6_K`. Each half of 128
// elements interleaves four 32-element quadrants drawn from:
//   q0 at y[l+  0] = (ql[l+ 0] & 0xF) | ((qh[l]>>0 & 3)<<4)
//   q1 at y[l+ 32] = (ql[l+32] & 0xF) | ((qh[l]>>2 & 3)<<4)
//   q2 at y[l+ 64] = (ql[l+ 0] >> 4)  | ((qh[l]>>4 & 3)<<4)
//   q3 at y[l+ 96] = (ql[l+32] >> 4)  | ((qh[l]>>6 & 3)<<4)
// Signed weight = (6bit - 32). Scale: scales[is + quadrant*2], is = l/16.
// Second half uses ql+=64, qh+=32, sc+=8.
#define Q6K_SB_SIZE 256
#define Q6K_SB_BYTES 210
#define Q6K_GEMV_ROWS 8
#define Q6K_GEMV_THREADS 256  // = Q6K_GEMV_ROWS * 32

__global__ void q6k_gemv_kernel(
    const uint8_t* __restrict__ weight,       // [N, (K/256) * 210]
    const __nv_bfloat16* __restrict__ input,  // [K]
    __nv_bfloat16* __restrict__ output,       // [N]
    int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q6K_GEMV_ROWS + warp_id;
    if (row >= N) return;

    const int num_sb    = K / Q6K_SB_SIZE;
    const int row_bytes = num_sb * Q6K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q6K_SB_BYTES;
        const uint8_t* ql_all = sb_p + 0;    // 128 bytes
        const uint8_t* qh_all = sb_p + 128;  // 64 bytes
        const int8_t*  sc_all = (const int8_t*)(sb_p + 192); // 16 bytes signed
        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 208))[0];
        const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));

        const int k_base = sb * Q6K_SB_SIZE;
        const int l = lane;           // 0..32 — position within a 32-element quadrant
        const int is = l / 16;        // 0 or 1

        // Process both halves × four quadrants per lane = 8 elements/superblock.
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql = ql_all + half * 64;
            const uint8_t* qh = qh_all + half * 32;
            const int8_t*  sc = sc_all + half * 8;
            const int k_half_base = k_base + half * 128;
            const uint8_t qh_l = qh[l];
            const uint8_t ql_0 = ql[l];
            const uint8_t ql_1 = ql[l + 32];

            // Quadrant 0: y[l+0]
            {
                const int low4 = ql_0 & 0x0F;
                const int high2 = (qh_l >> 0) & 0x03;
                const int q = (low4 | (high2 << 4)) - 32;
                const float w = d * (float)sc[is + 0] * (float)q;
                sum += w * __bfloat162float(input[k_half_base + l + 0]);
            }
            // Quadrant 1: y[l+32]
            {
                const int low4 = ql_1 & 0x0F;
                const int high2 = (qh_l >> 2) & 0x03;
                const int q = (low4 | (high2 << 4)) - 32;
                const float w = d * (float)sc[is + 2] * (float)q;
                sum += w * __bfloat162float(input[k_half_base + l + 32]);
            }
            // Quadrant 2: y[l+64]
            {
                const int low4 = ql_0 >> 4;
                const int high2 = (qh_l >> 4) & 0x03;
                const int q = (low4 | (high2 << 4)) - 32;
                const float w = d * (float)sc[is + 4] * (float)q;
                sum += w * __bfloat162float(input[k_half_base + l + 64]);
            }
            // Quadrant 3: y[l+96]
            {
                const int low4 = ql_1 >> 4;
                const int high2 = (qh_l >> 6) & 0x03;
                const int q = (low4 | (high2 << 4)) - 32;
                const float w = d * (float)sc[is + 6] * (float)q;
                sum += w * __bfloat162float(input[k_half_base + l + 96]);
            }
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[row] = __float2bfloat16(sum);
}

__global__ void q6k_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q6K_GEMV_ROWS + warp_id;
    const int batch   = blockIdx.y;
    if (row >= N || batch >= B) return;

    const int num_sb    = K / Q6K_SB_SIZE;
    const int row_bytes = num_sb * Q6K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;
    const __nv_bfloat16* x = input + batch * K;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q6K_SB_BYTES;
        const uint8_t* ql_all = sb_p + 0;
        const uint8_t* qh_all = sb_p + 128;
        const int8_t*  sc_all = (const int8_t*)(sb_p + 192);
        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 208))[0];
        const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));

        const int k_base = sb * Q6K_SB_SIZE;
        const int l = lane;
        const int is = l / 16;

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql = ql_all + half * 64;
            const uint8_t* qh = qh_all + half * 32;
            const int8_t*  sc = sc_all + half * 8;
            const int k_half_base = k_base + half * 128;
            const uint8_t qh_l = qh[l];
            const uint8_t ql_0 = ql[l];
            const uint8_t ql_1 = ql[l + 32];

            {
                const int q = ((ql_0 & 0x0F) | (((qh_l >> 0) & 0x03) << 4)) - 32;
                sum += d * (float)sc[is + 0] * (float)q
                       * __bfloat162float(x[k_half_base + l + 0]);
            }
            {
                const int q = ((ql_1 & 0x0F) | (((qh_l >> 2) & 0x03) << 4)) - 32;
                sum += d * (float)sc[is + 2] * (float)q
                       * __bfloat162float(x[k_half_base + l + 32]);
            }
            {
                const int q = ((ql_0 >> 4) | (((qh_l >> 4) & 0x03) << 4)) - 32;
                sum += d * (float)sc[is + 4] * (float)q
                       * __bfloat162float(x[k_half_base + l + 64]);
            }
            {
                const int q = ((ql_1 >> 4) | (((qh_l >> 6) & 0x03) << 4)) - 32;
                sum += d * (float)sc[is + 6] * (float)q
                       * __bfloat162float(x[k_half_base + l + 96]);
            }
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[batch * N + row] = __float2bfloat16(sum);
}

// Dequantize chunk kernel: each block handles ONE (row, superblock) and 256
// threads write the 256 dequanted elements of that superblock to the BF16 tile.
__global__ void q6k_dequant_chunk_kernel(
    const uint8_t* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int N, int K, int k_start, int k_len)
{
    const int row = blockIdx.x;
    const int sb_in_chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= N) return;

    const int num_sb_total = K / Q6K_SB_SIZE;
    const int sb_global    = (k_start / Q6K_SB_SIZE) + sb_in_chunk;
    const int row_bytes    = num_sb_total * Q6K_SB_BYTES;
    const uint8_t* sb_p    = weight + row * row_bytes + sb_global * Q6K_SB_BYTES;

    __shared__ float s_d;
    __shared__ int8_t s_scales[16];

    if (tid == 0) {
        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 208))[0];
        s_d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
    }
    if (tid < 16) {
        s_scales[tid] = ((const int8_t*)(sb_p + 192))[tid];
    }
    __syncthreads();

    // tid 0..255 → half, quadrant, l
    const int half = tid / 128;          // 0,1
    const int j_local = tid % 128;
    const int quad = j_local / 32;       // 0..4
    const int l = j_local % 32;
    const int is = l / 16;

    const uint8_t* ql = sb_p + half * 64;                  // ql[half*64..(half+1)*64]
    const uint8_t* qh = sb_p + 128 + half * 32;
    const int sc_base = half * 8;

    uint8_t low4, high2;
    switch (quad) {
        case 0: low4 = ql[l] & 0x0F;        high2 = (qh[l] >> 0) & 0x03; break;
        case 1: low4 = ql[l + 32] & 0x0F;   high2 = (qh[l] >> 2) & 0x03; break;
        case 2: low4 = ql[l] >> 4;          high2 = (qh[l] >> 4) & 0x03; break;
        default: low4 = ql[l + 32] >> 4;    high2 = (qh[l] >> 6) & 0x03; break;
    }
    const int q = (int)(low4 | (high2 << 4)) - 32;
    const int8_t sc = s_scales[sc_base + is + quad * 2];
    const float w = s_d * (float)sc * (float)q;

    const int out_k = sb_in_chunk * Q6K_SB_SIZE + half * 128 + quad * 32 + l;
    out[row * k_len + out_k] = __float2bfloat16(w);
}

// Q3_K (GGUF) native packed GEMV + dequant.
//
// One superblock = 256 K-dim elements = 110 bytes:
//   hmask:[32]  | qs:[64, 2-bit]  | scales:[12, 6-bit signed]  | d:f16(2)
//
// Element dequant:
//   q2  = (qs[k/4]    >> ((k%4)*2)) & 0x3
//   hbit= (hmask[k/8] >> (k%8))     & 0x1
//   q3  = q2 | (hbit << 2)
//   scale[i=k/16] = (scales_lo[i] | scales_hi[i] << 4) - 8 (signed, -8..55)
//   w   = d * scale * (q3 - 4)
//
// Scales decode (12 bytes → 16 sub-block scales, signed i8, one per 16 elements).
//
// Each scale is a 6-bit UNSIGNED value in 0..63. Low 4 bits come from the
// low/high nibble of scales_raw[0..8] (i<8 → low nibble of raw[i], i≥8 →
// high nibble of raw[i-8]). High 2 bits come from scales_raw[8+(i&3)] shifted
// right 2*(i/4) then masked with 0x3.
//
// Signed scale = unsigned6 - 32. Range: -32..31.
//
// NOTE: must combine the 6 bits BEFORE subtracting 32. Subtracting first and
// then OR'ing bit 4 into a negative i8 loses the bit to sign extension.
// (matches dequant_q3_k in gguf.rs after fix for the same bug.)
__device__ __forceinline__ void q3k_decode_scales(
    const uint8_t* __restrict__ scales_raw,  // 12 bytes
    int8_t scales[16])
{
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t low4 = (i < 8)
            ? (scales_raw[i] & 0x0F)
            : ((scales_raw[i - 8] >> 4) & 0x0F);
        const uint8_t high2 = (scales_raw[8 + (i & 3)] >> (2 * (i / 4))) & 0x03;
        const uint8_t u6 = low4 | (high2 << 4);
        scales[i] = (int8_t)((int)u6 - 32);
    }
}

// Q4_K (GGUF Q4_K_M / Q4_K_S) native packed GEMV + dequant.
//
// One superblock = 256 K-dim elements = 144 bytes:
//   d:f16(2) | dmin:f16(2) | scales_packed(12) | qs(128)
//
// scales_packed encodes 8 sub-block scales and 8 sub-block mins as 6-bit values:
//   first 4:  lower 6 bits of bytes[0..4]
//   last  4:  upper 2 bits of bytes[0..4] ORed with low 4 bits of bytes[8..12]
// mins follow the same pattern over bytes[4..8] / bytes[8..12] high nibbles.
//
// Dequant:  w = d * sub_scale[j] * nibble - dmin * sub_min[j]    (llama.cpp)
//
// Packed row stride = (K / 256) * 144 bytes.
//
// Block layout: 256 threads, 8 rows per block, 32 threads (1 warp) per row.
// Each warp processes one row's superblocks sequentially. Within a superblock,
// the 32 lanes cover 1 sub-block (32 elements) per iteration for 8 iterations,
// yielding 256 elements/superblock with every lane active.
#define Q4K_GEMV_ROWS 8
#define Q4K_GEMV_THREADS 256  // = Q4K_GEMV_ROWS * 32
#define Q4K_SB_SIZE 256
#define Q4K_SB_BYTES 144

// Decode 8 6-bit scales + 8 6-bit mins from the 12 scale bytes.
// Matches dequant_q4_k in gguf.rs and llama.cpp's get_scale_min_k4 layout.
__device__ __forceinline__ void q4k_decode_scales(
    const uint8_t* __restrict__ scales_raw,
    uint8_t sc[8],
    uint8_t mn[8])
{
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sc[i] = scales_raw[i] & 0x3F;
        mn[i] = scales_raw[i + 4] & 0x3F;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sc[4 + i] = (scales_raw[8 + i] & 0x0F) | ((scales_raw[i]     >> 6) << 4);
        mn[4 + i] = (scales_raw[8 + i] >> 4)   | ((scales_raw[i + 4] >> 6) << 4);
    }
}

// Element layout for Q4_K — MUST match llama.cpp `dequantize_row_q4_K`:
//   for iter in 0..4:
//     for l in 0..32:  y[iter*64 + l    ] = sc[2*iter+0] * (qs[iter*32+l] & 0x0F) - mn[2*iter+0]
//     for l in 0..32:  y[iter*64 + l+32] = sc[2*iter+1] * (qs[iter*32+l] >>  4) - mn[2*iter+1]
// NOT the naive "2 elements per ql byte" interpretation!
__global__ void q4k_gemv_kernel(
    const uint8_t* __restrict__ weight,        // [N, (K/256) * 144]
    const __nv_bfloat16* __restrict__ input,   // [K]
    __nv_bfloat16* __restrict__ output,        // [N]
    int N, int K)
{
    const int warp_id   = threadIdx.x / WARP_SIZE;    // 0..7  → row_in_block
    const int lane      = threadIdx.x % WARP_SIZE;    // 0..31
    const int row       = blockIdx.x * Q4K_GEMV_ROWS + warp_id;
    if (row >= N) return;

    const int num_sb      = K / Q4K_SB_SIZE;
    const int row_bytes   = num_sb * Q4K_SB_BYTES;
    const uint8_t* row_p  = weight + row * row_bytes;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q4K_SB_BYTES;

        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        const float d     = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        const float dmin  = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));

        uint8_t sc[8], mn[8];
        q4k_decode_scales(sb_p + 4, sc, mn);

        const uint8_t* qs = sb_p + 16;  // 128 bytes
        const int k_base  = sb * Q4K_SB_SIZE;

        // 4 outer iterations of 64 elements, 2 sub-blocks each.
        // Each lane processes 2 elements per iter (one lo nibble + one hi nibble
        // of the SAME ql byte) — so 8 elements/superblock/lane, 256/superblock total.
        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const int j_lo = iter * 2;
            const int j_hi = j_lo + 1;
            const float d1 = d * (float)sc[j_lo];
            const float m1 = dmin * (float)mn[j_lo];
            const float d2 = d * (float)sc[j_hi];
            const float m2 = dmin * (float)mn[j_hi];
            const uint8_t byte = qs[iter * 32 + lane];
            const float q_lo = (float)(byte & 0x0F);
            const float q_hi = (float)(byte >> 4);
            const float w_lo = q_lo * d1 - m1;
            const float w_hi = q_hi * d2 - m2;
            const int k_lo = k_base + j_lo * 32 + lane;
            const int k_hi = k_base + j_hi * 32 + lane;
            sum += w_lo * __bfloat162float(input[k_lo]);
            sum += w_hi * __bfloat162float(input[k_hi]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[row] = __float2bfloat16(sum);
}

// Batched variant: [B, K] × [N, packed]^T → [B, N]. Batch in grid.y.
__global__ void q4k_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K)
{
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int row      = blockIdx.x * Q4K_GEMV_ROWS + warp_id;
    const int batch    = blockIdx.y;
    if (row >= N || batch >= B) return;

    const int num_sb     = K / Q4K_SB_SIZE;
    const int row_bytes  = num_sb * Q4K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;
    const __nv_bfloat16* x = input + batch * K;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q4K_SB_BYTES;

        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        const float d    = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));

        uint8_t sc[8], mn[8];
        q4k_decode_scales(sb_p + 4, sc, mn);

        const uint8_t* qs = sb_p + 16;
        const int k_base  = sb * Q4K_SB_SIZE;

        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const int j_lo = iter * 2;
            const int j_hi = j_lo + 1;
            const float d1 = d * (float)sc[j_lo];
            const float m1 = dmin * (float)mn[j_lo];
            const float d2 = d * (float)sc[j_hi];
            const float m2 = dmin * (float)mn[j_hi];
            const uint8_t byte = qs[iter * 32 + lane];
            const float q_lo = (float)(byte & 0x0F);
            const float q_hi = (float)(byte >> 4);
            sum += (q_lo * d1 - m1) * __bfloat162float(x[k_base + j_lo * 32 + lane]);
            sum += (q_hi * d2 - m2) * __bfloat162float(x[k_base + j_hi * 32 + lane]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[batch * N + row] = __float2bfloat16(sum);
}

// Dequantize a K-dim chunk [k_start..k_start+k_len) of a Q4_K weight matrix into BF16.
// Grid:  (N, k_len / 256), Block: 256 threads — but element-to-thread mapping follows
// llama.cpp's canonical iter/half/l layout, NOT the naive "tid is element index"
// interpretation. 256 threads cover one superblock:
//   thread t → iter = (t >> 6) & 3, half = (t >> 5) & 1, l = t & 31
//   writes y[iter*64 + half*32 + l]
__global__ void q4k_dequant_chunk_kernel(
    const uint8_t* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int N, int K, int k_start, int k_len)
{
    const int row = blockIdx.x;
    const int sb_in_chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= N) return;

    const int num_sb_total = K / Q4K_SB_SIZE;
    const int sb_global    = (k_start / Q4K_SB_SIZE) + sb_in_chunk;
    const int row_bytes    = num_sb_total * Q4K_SB_BYTES;
    const uint8_t* sb_p    = weight + row * row_bytes + sb_global * Q4K_SB_BYTES;

    __shared__ float s_d;
    __shared__ float s_dmin;
    __shared__ uint8_t s_sc[8];
    __shared__ uint8_t s_mn[8];

    if (tid == 0) {
        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        s_d    = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        s_dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));
        q4k_decode_scales(sb_p + 4, s_sc, s_mn);
    }
    __syncthreads();

    const uint8_t* qs = sb_p + 16;
    const int iter = (tid >> 6) & 3;  // 0..4
    const int half = (tid >> 5) & 1;  // 0..2
    const int l    = tid & 31;
    const int sub  = iter * 2 + half;  // 0..8
    const uint8_t byte = qs[iter * 32 + l];
    const int q = half ? (byte >> 4) : (byte & 0x0F);
    const float w = (float)q * (s_d * (float)s_sc[sub]) - (s_dmin * (float)s_mn[sub]);

    const int out_k = sb_in_chunk * Q4K_SB_SIZE + sub * 32 + l;
    out[row * k_len + out_k] = __float2bfloat16(w);
}

// Q5_K (GGUF Q5_K_M / Q5_K_S) native packed GEMV + dequant.
//
// One superblock = 256 K-dim elements = 176 bytes:
//   d:f16(2) | dmin:f16(2) | scales_packed(12) | qh(32) | qs(128)
//
// Q5_K shares Q4_K's scale/min packing and element order. `qs` stores low
// nibbles, while `qh[l]` contributes one high bit for each of the 8 sub-blocks.
#define Q5K_GEMV_ROWS 8
#define Q5K_GEMV_THREADS 256
#define Q5K_SB_SIZE 256
#define Q5K_SB_BYTES 176

__global__ void q5k_gemv_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q5K_GEMV_ROWS + warp_id;
    if (row >= N) return;

    const int num_sb = K / Q5K_SB_SIZE;
    const int row_bytes = num_sb * Q5K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q5K_SB_BYTES;
        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        const float d    = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));

        uint8_t sc[8], mn[8];
        q4k_decode_scales(sb_p + 4, sc, mn);

        const uint8_t* qh = sb_p + 16;
        const uint8_t* qs = sb_p + 48;
        const int k_base = sb * Q5K_SB_SIZE;

        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const int j_lo = iter * 2;
            const int j_hi = j_lo + 1;
            const float d1 = d * (float)sc[j_lo];
            const float m1 = dmin * (float)mn[j_lo];
            const float d2 = d * (float)sc[j_hi];
            const float m2 = dmin * (float)mn[j_hi];
            const uint8_t byte = qs[iter * 32 + lane];
            const int q_lo = (int)(byte & 0x0F) | (((int)(qh[lane] >> j_lo) & 1) << 4);
            const int q_hi = (int)(byte >> 4) | (((int)(qh[lane] >> j_hi) & 1) << 4);
            sum += ((float)q_lo * d1 - m1) * __bfloat162float(input[k_base + j_lo * 32 + lane]);
            sum += ((float)q_hi * d2 - m2) * __bfloat162float(input[k_base + j_hi * 32 + lane]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[row] = __float2bfloat16(sum);
}

__global__ void q5k_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q5K_GEMV_ROWS + warp_id;
    const int batch   = blockIdx.y;
    if (row >= N || batch >= B) return;

    const int num_sb = K / Q5K_SB_SIZE;
    const int row_bytes = num_sb * Q5K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;
    const __nv_bfloat16* x = input + batch * K;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q5K_SB_BYTES;
        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        const float d    = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));

        uint8_t sc[8], mn[8];
        q4k_decode_scales(sb_p + 4, sc, mn);

        const uint8_t* qh = sb_p + 16;
        const uint8_t* qs = sb_p + 48;
        const int k_base = sb * Q5K_SB_SIZE;

        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const int j_lo = iter * 2;
            const int j_hi = j_lo + 1;
            const float d1 = d * (float)sc[j_lo];
            const float m1 = dmin * (float)mn[j_lo];
            const float d2 = d * (float)sc[j_hi];
            const float m2 = dmin * (float)mn[j_hi];
            const uint8_t byte = qs[iter * 32 + lane];
            const int q_lo = (int)(byte & 0x0F) | (((int)(qh[lane] >> j_lo) & 1) << 4);
            const int q_hi = (int)(byte >> 4) | (((int)(qh[lane] >> j_hi) & 1) << 4);
            sum += ((float)q_lo * d1 - m1) * __bfloat162float(x[k_base + j_lo * 32 + lane]);
            sum += ((float)q_hi * d2 - m2) * __bfloat162float(x[k_base + j_hi * 32 + lane]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[batch * N + row] = __float2bfloat16(sum);
}

__global__ void q5k_dequant_chunk_kernel(
    const uint8_t* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int N, int K, int k_start, int k_len)
{
    const int row = blockIdx.x;
    const int sb_in_chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= N) return;

    const int num_sb_total = K / Q5K_SB_SIZE;
    const int sb_global = (k_start / Q5K_SB_SIZE) + sb_in_chunk;
    const int row_bytes = num_sb_total * Q5K_SB_BYTES;
    const uint8_t* sb_p = weight + row * row_bytes + sb_global * Q5K_SB_BYTES;

    __shared__ float s_d;
    __shared__ float s_dmin;
    __shared__ uint8_t s_sc[8];
    __shared__ uint8_t s_mn[8];

    if (tid == 0) {
        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        s_d    = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        s_dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));
        q4k_decode_scales(sb_p + 4, s_sc, s_mn);
    }
    __syncthreads();

    const uint8_t* qh = sb_p + 16;
    const uint8_t* qs = sb_p + 48;
    const int iter = (tid >> 6) & 3;
    const int half = (tid >> 5) & 1;
    const int l = tid & 31;
    const int sub = iter * 2 + half;
    const uint8_t byte = qs[iter * 32 + l];
    const int low = half ? (byte >> 4) : (byte & 0x0F);
    const int q = low | ((((int)qh[l] >> sub) & 1) << 4);
    const float w = (float)q * (s_d * (float)s_sc[sub]) - (s_dmin * (float)s_mn[sub]);

    const int out_k = sb_in_chunk * Q5K_SB_SIZE + sub * 32 + l;
    out[row * k_len + out_k] = __float2bfloat16(w);
}

__device__ __forceinline__ float q3k_value(const uint8_t* __restrict__ sb_p, int k_local)
{
    int8_t scales[16];
    q3k_decode_scales(sb_p + 96, scales);
    const unsigned short d_u16 = ((const unsigned short*)(sb_p + 108))[0];
    const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
    const uint8_t* hmask = sb_p;
    const uint8_t* qs = sb_p + 32;
    const int q2 = (qs[k_local >> 2] >> ((k_local & 3) << 1)) & 0x3;
    const int hbit = (hmask[k_local >> 3] >> (k_local & 7)) & 0x1;
    const int q3 = q2 | (hbit << 2);
    return d * (float)scales[k_local >> 4] * ((float)q3 - 4.0f);
}

__device__ __forceinline__ float q4k_value(const uint8_t* __restrict__ sb_p, int k_local)
{
    uint8_t sc[8], mn[8];
    q4k_decode_scales(sb_p + 4, sc, mn);
    const unsigned short d_u16 = ((const unsigned short*)sb_p)[0];
    const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
    const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));
    const int iter = k_local >> 6;
    const int half = (k_local >> 5) & 1;
    const int l = k_local & 31;
    const int sub = iter * 2 + half;
    const uint8_t byte = sb_p[16 + iter * 32 + l];
    const int q = half ? (byte >> 4) : (byte & 0x0F);
    return (float)q * (d * (float)sc[sub]) - (dmin * (float)mn[sub]);
}

__device__ __forceinline__ float q5k_value(const uint8_t* __restrict__ sb_p, int k_local)
{
    uint8_t sc[8], mn[8];
    q4k_decode_scales(sb_p + 4, sc, mn);
    const unsigned short d_u16 = ((const unsigned short*)sb_p)[0];
    const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
    const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));
    const int iter = k_local >> 6;
    const int half = (k_local >> 5) & 1;
    const int l = k_local & 31;
    const int sub = iter * 2 + half;
    const uint8_t byte = sb_p[48 + iter * 32 + l];
    const int low = half ? (byte >> 4) : (byte & 0x0F);
    const int q = low | ((((int)sb_p[16 + l] >> sub) & 1) << 4);
    return (float)q * (d * (float)sc[sub]) - (dmin * (float)mn[sub]);
}

__device__ __forceinline__ float q6k_value(const uint8_t* __restrict__ sb_p, int k_local)
{
    const int half = k_local >> 7;
    const int j_local = k_local & 127;
    const int quad = j_local >> 5;
    const int l = j_local & 31;
    const int is = l >> 4;
    const uint8_t* ql = sb_p + half * 64;
    const uint8_t* qh = sb_p + 128 + half * 32;
    uint8_t low4, high2;
    switch (quad) {
        case 0: low4 = ql[l] & 0x0F;      high2 = (qh[l] >> 0) & 0x03; break;
        case 1: low4 = ql[l + 32] & 0x0F; high2 = (qh[l] >> 2) & 0x03; break;
        case 2: low4 = ql[l] >> 4;        high2 = (qh[l] >> 4) & 0x03; break;
        default: low4 = ql[l + 32] >> 4;  high2 = (qh[l] >> 6) & 0x03; break;
    }
    const int q = (int)(low4 | (high2 << 4)) - 32;
    const int8_t sc = ((const int8_t*)(sb_p + 192))[half * 8 + is + quad * 2];
    const unsigned short d_u16 = ((const unsigned short*)(sb_p + 208))[0];
    const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
    return d * (float)sc * (float)q;
}

__global__ void qxk_embedding_batched_kernel(
    const uint8_t* __restrict__ weight,
    const int* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ out,
    int hidden_dim,
    int batch_size,
    int format,
    int block_bytes)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden_dim * batch_size;
    if (tid >= total) return;
    const int b = tid / hidden_dim;
    const int k = tid - b * hidden_dim;
    const int row = __ldg(&token_ids[b]);
    const int num_sb = hidden_dim / 256;
    const uint8_t* row_p = weight + row * num_sb * block_bytes;
    const uint8_t* sb_p = row_p + (k >> 8) * block_bytes;
    float value;
    switch (format) {
        case 3: value = q3k_value(sb_p, k & 255); break;
        case 4: value = q4k_value(sb_p, k & 255); break;
        case 5: value = q5k_value(sb_p, k & 255); break;
        default: value = q6k_value(sb_p, k & 255); break;
    }
    out[tid] = __float2bfloat16(value);
}

__global__ void qxk_embedding_decode_kernel(
    const uint8_t* __restrict__ weight,
    const int* __restrict__ token_id,
    __nv_bfloat16* __restrict__ out,
    int hidden_dim,
    int format,
    int block_bytes)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_dim) return;
    const int row = __ldg(&token_id[0]);
    const int num_sb = hidden_dim / 256;
    const uint8_t* row_p = weight + row * num_sb * block_bytes;
    const uint8_t* sb_p = row_p + (tid >> 8) * block_bytes;
    float value;
    switch (format) {
        case 3: value = q3k_value(sb_p, tid & 255); break;
        case 4: value = q4k_value(sb_p, tid & 255); break;
        case 5: value = q5k_value(sb_p, tid & 255); break;
        default: value = q6k_value(sb_p, tid & 255); break;
    }
    out[tid] = __float2bfloat16(value);
}

// C API
extern "C" {

cudaError_t gemv_fp8_block_scaled_batch_cuda(
    const uint8_t* weight, const float* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int scale_rows, int scale_cols, int block_m, int block_k,
    cudaStream_t stream);

cudaError_t gemv_fp4_e2m1_group_batch_cuda(
    const uint8_t* weight, const uint8_t* scales, const float* global_scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, int scale_cols, cudaStream_t stream);

cudaError_t w8a16_gemv_cuda(
    const int8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, 1);
    dim3 block(GEMV_THREADS);
    w8a16_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(weight), scales, input, output, 1, N, K, group_size);
    return cudaGetLastError();
}

cudaError_t w4a16_gemv_cuda(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, 1);
    dim3 block(GEMV_THREADS);
    // sm_70 path caches input in dynamic shared memory; other SMs ignore it.
    size_t smem = (size_t)K * sizeof(__nv_bfloat16);
    w4a16_gemv_batch_kernel<<<grid, block, smem, stream>>>(
        weight, scales, input, output, 1, N, K, group_size);
    return cudaGetLastError();
}

cudaError_t w8a16_gemv_batch_cuda(
    const int8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, cudaStream_t stream)
{
    dim3 block(GEMV_THREADS);
    // B>=2 uses the batched GEMM (B_TILE inputs share one weight read) instead
    // of B independent GEMVs that each re-read the full weight — that re-read is
    // why the plain per-batch GEMV collapsed at c>1.
    if (B >= 2) {
        dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, (B + W8A16_GEMM_BTILE - 1) / W8A16_GEMM_BTILE);
        w8a16_gemm_batch_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(weight), scales, input, output, B, N, K, group_size);
        return cudaGetLastError();
    }
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    w8a16_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(weight), scales, input, output, B, N, K, group_size);
    return cudaGetLastError();
}

// Marlin-style W4A16 GEMV (uint4 loads, 1 warp/row) — V100-optimized.
extern "C" cudaError_t w4a16_gemv_batch_cuda_marlin(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, cudaStream_t stream);

cudaError_t w4a16_gemv_batch_cuda(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, cudaStream_t stream)
{
    // Marlin GEMV for all batch sizes — uint4 loads and FP16 FMA.
    return w4a16_gemv_batch_cuda_marlin(weight, scales, input, output, B, N, K, group_size, stream);
}

cudaError_t moe_w4a16_grouped_gemv_batch_cuda(
    const uint64_t* weight_ptrs,
    const uint64_t* scale_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N,
    int K,
    int group_size,
    uint32_t xor_mask,
    cudaStream_t stream)
{
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0 ||
        (K & 1) != 0 || group_size <= 0 || (K % group_size) != 0) {
        return cudaSuccess;
    }
    dim3 block(GEMV_THREADS);
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS,
              (max_count + GROUPED_BTILE - 1) / GROUPED_BTILE,
              num_experts);
    w4a16_grouped_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight_ptrs, scale_ptrs, input, output, offsets, counts, expert_indices,
        max_count, N, K, group_size, xor_mask);
    return cudaGetLastError();
}

cudaError_t moe_w4a16_grouped_gemv_pair_batch_cuda(
    const uint64_t* weight_a_ptrs,
    const uint64_t* scale_a_ptrs,
    const uint64_t* weight_b_ptrs,
    const uint64_t* scale_b_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* output_a,
    __nv_bfloat16* output_b,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N,
    int K,
    int group_size,
    uint32_t xor_mask,
    bool fuse_swiglu,
    float swiglu_limit,
    cudaStream_t stream)
{
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0 ||
        (K & 1) != 0 || group_size <= 0 || (K % group_size) != 0) {
        return cudaSuccess;
    }
    dim3 block(GEMV_THREADS);
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS,
              (max_count + GROUPED_BTILE - 1) / GROUPED_BTILE,
              num_experts);
    w4a16_grouped_gemv_pair_batch_kernel<<<grid, block, 0, stream>>>(
        weight_a_ptrs, scale_a_ptrs, weight_b_ptrs, scale_b_ptrs, input,
        output_a, output_b, offsets, counts, expert_indices, max_count, N, K,
        group_size, xor_mask, fuse_swiglu, swiglu_limit);
    return cudaGetLastError();
}

cudaError_t dsv4_fp8_gemv_batch_cuda(
    const uint8_t* weight, const uint8_t* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int scale_rows, int scale_cols, cudaStream_t stream)
{
    if (B <= 0 || N <= 0 || K <= 0 || scale_rows <= 0 || scale_cols <= 0) {
        return cudaErrorInvalidValue;
    }
    if (B > 1) {
        dim3 block(GEMV_THREADS);
        // TILE = smallest of {2,4,8,16,32} covering min(B, 32): accumulator
        // register pressure tracks the actual batch, grid.y tiles the rest.
        const int block_h = (N + scale_rows - 1) / scale_rows;
        const int block_w = (K + scale_cols - 1) / scale_cols;
        Fp8E8m0BlockScale scale_fn{scales, scale_rows, scale_cols, block_h, block_w};
        auto launch = [&](auto kern, int tile) {
            dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, (B + tile - 1) / tile);
            kern<<<grid, block, 0, stream>>>(weight, input, output, B, N, K, scale_fn);
        };
        if (B <= 2) launch(fp8_gemv_batch_tiled_kernel<2, Fp8E8m0BlockScale>, 2);
        else if (B <= 4) launch(fp8_gemv_batch_tiled_kernel<4, Fp8E8m0BlockScale>, 4);
        else if (B <= 8) launch(fp8_gemv_batch_tiled_kernel<8, Fp8E8m0BlockScale>, 8);
        else if (B <= 16) launch(fp8_gemv_batch_tiled_kernel<16, Fp8E8m0BlockScale>, 16);
        else launch(fp8_gemv_batch_tiled_kernel<DSV4_BATCH_TILE, Fp8E8m0BlockScale>, DSV4_BATCH_TILE);
        return cudaGetLastError();
    }
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    dim3 block(GEMV_THREADS);
    const int block_h = (N + scale_rows - 1) / scale_rows;
    const int block_w = (K + scale_cols - 1) / scale_cols;
    Fp8E8m0BlockScale scale_fn{scales, scale_rows, scale_cols, block_h, block_w};
    fp8_gemv_batch_kernel<<<grid, block, 0, stream>>>(weight, input, output, B, N, K, scale_fn);
    return cudaGetLastError();
}

cudaError_t dsv4_fp4_gemv_batch_cuda(
    const uint8_t* weight, const uint8_t* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int scale_rows, int scale_cols, cudaStream_t stream)
{
    if (B <= 0 || N <= 0 || K <= 0 || (K & 1) != 0 || scale_rows <= 0 || scale_cols <= 0) {
        return cudaErrorInvalidValue;
    }
    if (B > 1) {
        dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, (B + DSV4_BATCH_TILE - 1) / DSV4_BATCH_TILE);
        dim3 block(GEMV_THREADS);
        dsv4_fp4_gemv_batch_tiled_kernel<<<grid, block, 0, stream>>>(
            weight, scales, input, output, B, N, K, scale_rows, scale_cols);
        return cudaGetLastError();
    }
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    dim3 block(GEMV_THREADS);
    dsv4_fp4_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, B, N, K, scale_rows, scale_cols);
    return cudaGetLastError();
}

cudaError_t gemv_fp8_block_scaled_cuda(
    const uint8_t* weight, const float* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, int scale_rows, int scale_cols, int block_m, int block_k,
    cudaStream_t stream)
{
    return gemv_fp8_block_scaled_batch_cuda(
        weight, scales, input, output, 1, N, K, scale_rows, scale_cols, block_m, block_k, stream);
}

cudaError_t gemv_fp8_block_scaled_batch_cuda(
    const uint8_t* weight, const float* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int scale_rows, int scale_cols, int block_m, int block_k,
    cudaStream_t stream)
{
    if (B <= 0 || N <= 0 || K <= 0 || scale_rows <= 0 || scale_cols <= 0 ||
        block_m <= 0 || block_k <= 0) {
        return cudaErrorInvalidValue;
    }
    // B>1 (spec-decode verify / batched decode): always tile the batch so each
    // weight row streams from HBM ONCE per tile (TILE==B) instead of once per
    // column — measured strictly faster than the per-column grid.y=B kernel for
    // B>1 (H20: verify 82.7→60.7ms; the lever that makes MTP spec-decode a net
    // win). B==1 keeps the proven single-column kernel byte-for-byte unchanged.
    if (B > 1) {
        dim3 block(GEMV_THREADS);
        // Instantiate TILE == B for the small spec-verify depths (grid.y=1 →
        // weight read ONCE, register pressure == B). B>8 falls back to the
        // fixed-8 tile with grid.y batching.
        Fp8F32BlockScale scale_fn{scales, scale_rows, scale_cols, block_m, block_k};
        auto launch = [&](auto kern, int tile) {
            dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, (B + tile - 1) / tile);
            kern<<<grid, block, 0, stream>>>(weight, input, output, B, N, K, scale_fn);
        };
        switch (B) {
            case 2: launch(fp8_gemv_batch_tiled_kernel<2, Fp8F32BlockScale>, 2); break;
            case 3: launch(fp8_gemv_batch_tiled_kernel<3, Fp8F32BlockScale>, 3); break;
            case 4: launch(fp8_gemv_batch_tiled_kernel<4, Fp8F32BlockScale>, 4); break;
            case 5: launch(fp8_gemv_batch_tiled_kernel<5, Fp8F32BlockScale>, 5); break;
            case 6: launch(fp8_gemv_batch_tiled_kernel<6, Fp8F32BlockScale>, 6); break;
            case 7: launch(fp8_gemv_batch_tiled_kernel<7, Fp8F32BlockScale>, 7); break;
            case 8: launch(fp8_gemv_batch_tiled_kernel<8, Fp8F32BlockScale>, 8); break;
            default: launch(fp8_gemv_batch_tiled_kernel<QWEN_GEMV_BATCH_TILE, Fp8F32BlockScale>, QWEN_GEMV_BATCH_TILE); break;
        }
        return cudaGetLastError();
    }
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    dim3 block(GEMV_THREADS);
    Fp8F32BlockScale scale_fn{scales, scale_rows, scale_cols, block_m, block_k};
    fp8_gemv_batch_kernel<<<grid, block, 0, stream>>>(weight, input, output, B, N, K, scale_fn);
    return cudaGetLastError();
}

cudaError_t gemv_fp4_e2m1_group_cuda(
    const uint8_t* weight, const uint8_t* scales, const float* global_scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, int group_size, int scale_cols, cudaStream_t stream)
{
    return gemv_fp4_e2m1_group_batch_cuda(
        weight, scales, global_scales, input, output, 1, N, K, group_size, scale_cols, stream);
}

cudaError_t gemv_fp4_e2m1_group_batch_cuda(
    const uint8_t* weight, const uint8_t* scales, const float* global_scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, int scale_cols, cudaStream_t stream)
{
    if (B <= 0 || N <= 0 || K <= 0 || (K & 1) != 0 || group_size <= 0 ||
        scale_cols <= 0 || (K % group_size) != 0 || scale_cols != K / group_size) {
        return cudaErrorInvalidValue;
    }
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    dim3 block(GEMV_THREADS);
    fp4_e2m1_group_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight, scales, global_scales, input, output, B, N, K, group_size, scale_cols);
    return cudaGetLastError();
}

cudaError_t moe_fp8_block_scaled_grouped_gemv_batch_cuda(
    const uint64_t* weight_ptrs,
    const uint64_t* scale_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int block_m,
    int block_k,
    cudaStream_t stream)
{
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0 ||
        scale_rows <= 0 || scale_cols <= 0 || block_m <= 0 || block_k <= 0) {
        return cudaSuccess;
    }
    dim3 block(GEMV_THREADS);
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, max_count, num_experts);
    fp8_f32_block_grouped_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight_ptrs, scale_ptrs, input, output, offsets, counts, expert_indices,
        max_count, N, K, scale_rows, scale_cols, block_m, block_k);
    return cudaGetLastError();
}

cudaError_t moe_fp8_block_scaled_grouped_gemv_pair_batch_cuda(
    const uint64_t* weight_a_ptrs,
    const uint64_t* scale_a_ptrs,
    const uint64_t* weight_b_ptrs,
    const uint64_t* scale_b_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* output_a,
    __nv_bfloat16* output_b,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int block_m,
    int block_k,
    cudaStream_t stream)
{
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0 ||
        scale_rows <= 0 || scale_cols <= 0 || block_m <= 0 || block_k <= 0) {
        return cudaSuccess;
    }
    dim3 block(GEMV_THREADS);
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, max_count, num_experts);
    fp8_f32_block_grouped_gemv_pair_batch_kernel<<<grid, block, 0, stream>>>(
        weight_a_ptrs, scale_a_ptrs, weight_b_ptrs, scale_b_ptrs, input,
        output_a, output_b, offsets, counts, expert_indices, max_count, N, K,
        scale_rows, scale_cols, block_m, block_k);
    return cudaGetLastError();
}

cudaError_t moe_fp4_e2m1_grouped_gemv_batch_cuda(
    const uint64_t* weight_ptrs,
    const uint64_t* scale_ptrs,
    const uint64_t* global_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N,
    int K,
    int group_size,
    int scale_cols,
    cudaStream_t stream)
{
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0 ||
        (K & 1) != 0 || group_size <= 0 || scale_cols <= 0 || (K % group_size) != 0 ||
        scale_cols != K / group_size) {
        return cudaSuccess;
    }
    dim3 block(GEMV_THREADS);
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, max_count, num_experts);
    fp4_e2m1_grouped_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight_ptrs, scale_ptrs, global_ptrs, input, output, offsets, counts,
        expert_indices, max_count, N, K, group_size, scale_cols);
    return cudaGetLastError();
}

cudaError_t moe_fp4_e2m1_grouped_gemv_pair_batch_cuda(
    const uint64_t* weight_a_ptrs,
    const uint64_t* scale_a_ptrs,
    const uint64_t* global_a_ptrs,
    const uint64_t* weight_b_ptrs,
    const uint64_t* scale_b_ptrs,
    const uint64_t* global_b_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* output_a,
    __nv_bfloat16* output_b,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N,
    int K,
    int group_size,
    int scale_cols,
    cudaStream_t stream)
{
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0 ||
        (K & 1) != 0 || group_size <= 0 || scale_cols <= 0 || (K % group_size) != 0 ||
        scale_cols != K / group_size) {
        return cudaSuccess;
    }
    dim3 block(GEMV_THREADS);
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, max_count, num_experts);
    fp4_e2m1_grouped_gemv_pair_batch_kernel<<<grid, block, 0, stream>>>(
        weight_a_ptrs, scale_a_ptrs, global_a_ptrs, weight_b_ptrs, scale_b_ptrs,
        global_b_ptrs, input, output_a, output_b, offsets, counts, expert_indices,
        max_count, N, K, group_size, scale_cols);
    return cudaGetLastError();
}

cudaError_t q6k_gemv_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q6K_GEMV_ROWS - 1) / Q6K_GEMV_ROWS);
    dim3 block(Q6K_GEMV_THREADS);
    q6k_gemv_kernel<<<grid, block, 0, stream>>>(weight, input, output, N, K);
    return cudaGetLastError();
}

cudaError_t dsv4_fp8_route_gemv_batch_cuda(
    const uint64_t* weight_ptrs,
    const uint64_t* scale_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const int32_t* route_meta,
    int local_expert_start,
    int experts_per_rank,
    int num_routes,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int apply_route_weight,
    cudaStream_t stream) {
    if (num_routes <= 0 || N <= 0 || K <= 0 || local_expert_start < 0 ||
        experts_per_rank <= 0 || scale_rows <= 0 || scale_cols <= 0) {
        return cudaSuccess;
    }
    dim3 block(GEMV_THREADS);
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, num_routes);
    dsv4_fp8_route_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight_ptrs, scale_ptrs, input, output, route_meta, local_expert_start,
        experts_per_rank, num_routes, N, K, scale_rows, scale_cols,
        apply_route_weight);
    return cudaGetLastError();
}

cudaError_t dsv4_fp4_route_gemv_batch_cuda(
    const uint64_t* weight_ptrs,
    const uint64_t* scale_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const int32_t* route_meta,
    int local_expert_start,
    int experts_per_rank,
    int num_routes,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int apply_route_weight,
    cudaStream_t stream) {
    if (num_routes <= 0 || N <= 0 || K <= 0 || (K & 1) != 0 ||
        local_expert_start < 0 || experts_per_rank <= 0 || scale_rows <= 0 ||
        scale_cols <= 0) {
        return cudaSuccess;
    }
    dim3 block(GEMV_THREADS);
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, num_routes);
    dsv4_fp4_route_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight_ptrs, scale_ptrs, input, output, route_meta, local_expert_start,
        experts_per_rank, num_routes, N, K, scale_rows, scale_cols,
        apply_route_weight);
    return cudaGetLastError();
}

cudaError_t q6k_gemv_batch_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q6K_GEMV_ROWS - 1) / Q6K_GEMV_ROWS, B);
    dim3 block(Q6K_GEMV_THREADS);
    q6k_gemv_batch_kernel<<<grid, block, 0, stream>>>(weight, input, output, B, N, K);
    return cudaGetLastError();
}

cudaError_t q6k_dequant_chunk_cuda(
    const uint8_t* weight, __nv_bfloat16* out,
    int N, int K, int k_start, int k_len, cudaStream_t stream)
{
    if ((k_start % Q6K_SB_SIZE) != 0 || (k_len % Q6K_SB_SIZE) != 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(N, k_len / Q6K_SB_SIZE);
    dim3 block(Q6K_SB_SIZE);
    q6k_dequant_chunk_kernel<<<grid, block, 0, stream>>>(
        weight, out, N, K, k_start, k_len);
    return cudaGetLastError();
}

cudaError_t q4k_gemv_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q4K_GEMV_ROWS - 1) / Q4K_GEMV_ROWS);
    dim3 block(Q4K_GEMV_THREADS);
    q4k_gemv_kernel<<<grid, block, 0, stream>>>(weight, input, output, N, K);
    return cudaGetLastError();
}

cudaError_t q4k_gemv_batch_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q4K_GEMV_ROWS - 1) / Q4K_GEMV_ROWS, B);
    dim3 block(Q4K_GEMV_THREADS);
    q4k_gemv_batch_kernel<<<grid, block, 0, stream>>>(weight, input, output, B, N, K);
    return cudaGetLastError();
}

cudaError_t q4k_dequant_chunk_cuda(
    const uint8_t* weight, __nv_bfloat16* out,
    int N, int K, int k_start, int k_len, cudaStream_t stream)
{
    // Safety: chunk must align to superblock boundaries.
    if ((k_start % Q4K_SB_SIZE) != 0 || (k_len % Q4K_SB_SIZE) != 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(N, k_len / Q4K_SB_SIZE);
    dim3 block(Q4K_SB_SIZE);
    q4k_dequant_chunk_kernel<<<grid, block, 0, stream>>>(
        weight, out, N, K, k_start, k_len);
    return cudaGetLastError();
}

cudaError_t q5k_gemv_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q5K_GEMV_ROWS - 1) / Q5K_GEMV_ROWS);
    dim3 block(Q5K_GEMV_THREADS);
    q5k_gemv_kernel<<<grid, block, 0, stream>>>(weight, input, output, N, K);
    return cudaGetLastError();
}

cudaError_t q5k_gemv_batch_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q5K_GEMV_ROWS - 1) / Q5K_GEMV_ROWS, B);
    dim3 block(Q5K_GEMV_THREADS);
    q5k_gemv_batch_kernel<<<grid, block, 0, stream>>>(weight, input, output, B, N, K);
    return cudaGetLastError();
}

cudaError_t q5k_dequant_chunk_cuda(
    const uint8_t* weight, __nv_bfloat16* out,
    int N, int K, int k_start, int k_len, cudaStream_t stream)
{
    if ((k_start % Q5K_SB_SIZE) != 0 || (k_len % Q5K_SB_SIZE) != 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(N, k_len / Q5K_SB_SIZE);
    dim3 block(Q5K_SB_SIZE);
    q5k_dequant_chunk_kernel<<<grid, block, 0, stream>>>(
        weight, out, N, K, k_start, k_len);
    return cudaGetLastError();
}

static cudaError_t qxk_embedding_batched_cuda(
    const uint8_t* weight,
    const int* token_ids,
    __nv_bfloat16* out,
    int hidden_dim,
    int batch_size,
    int format,
    int block_bytes,
    cudaStream_t stream)
{
    if ((hidden_dim % 256) != 0) {
        return cudaErrorInvalidValue;
    }
    const int total = hidden_dim * batch_size;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    qxk_embedding_batched_kernel<<<grid, block, 0, stream>>>(
        weight, token_ids, out, hidden_dim, batch_size, format, block_bytes);
    return cudaGetLastError();
}

static cudaError_t qxk_embedding_decode_cuda(
    const uint8_t* weight,
    const int* token_id,
    __nv_bfloat16* out,
    int hidden_dim,
    int format,
    int block_bytes,
    cudaStream_t stream)
{
    if ((hidden_dim % 256) != 0) {
        return cudaErrorInvalidValue;
    }
    const int block = 256;
    const int grid = (hidden_dim + block - 1) / block;
    qxk_embedding_decode_kernel<<<grid, block, 0, stream>>>(
        weight, token_id, out, hidden_dim, format, block_bytes);
    return cudaGetLastError();
}

cudaError_t q4k_embedding_batched_cuda(
    const uint8_t* weight, const int* token_ids, __nv_bfloat16* out,
    int hidden_dim, int batch_size, cudaStream_t stream)
{
    return qxk_embedding_batched_cuda(
        weight, token_ids, out, hidden_dim, batch_size, 4, Q4K_SB_BYTES, stream);
}

cudaError_t q5k_embedding_batched_cuda(
    const uint8_t* weight, const int* token_ids, __nv_bfloat16* out,
    int hidden_dim, int batch_size, cudaStream_t stream)
{
    return qxk_embedding_batched_cuda(
        weight, token_ids, out, hidden_dim, batch_size, 5, Q5K_SB_BYTES, stream);
}

cudaError_t q6k_embedding_batched_cuda(
    const uint8_t* weight, const int* token_ids, __nv_bfloat16* out,
    int hidden_dim, int batch_size, cudaStream_t stream)
{
    return qxk_embedding_batched_cuda(
        weight, token_ids, out, hidden_dim, batch_size, 6, Q6K_SB_BYTES, stream);
}

cudaError_t q4k_embedding_decode_cuda(
    const uint8_t* weight, const int* token_id, __nv_bfloat16* out,
    int hidden_dim, cudaStream_t stream)
{
    return qxk_embedding_decode_cuda(weight, token_id, out, hidden_dim, 4, Q4K_SB_BYTES, stream);
}

cudaError_t q5k_embedding_decode_cuda(
    const uint8_t* weight, const int* token_id, __nv_bfloat16* out,
    int hidden_dim, cudaStream_t stream)
{
    return qxk_embedding_decode_cuda(weight, token_id, out, hidden_dim, 5, Q5K_SB_BYTES, stream);
}

cudaError_t q6k_embedding_decode_cuda(
    const uint8_t* weight, const int* token_id, __nv_bfloat16* out,
    int hidden_dim, cudaStream_t stream)
{
    return qxk_embedding_decode_cuda(weight, token_id, out, hidden_dim, 6, Q6K_SB_BYTES, stream);
}

}  // extern "C"

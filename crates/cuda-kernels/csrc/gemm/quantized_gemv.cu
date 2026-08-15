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

#define WARP_SIZE 32
#define GEMV_THREADS 256
#define GEMV_ROWS 4
#define DSV4_BATCH_TILE 32
#define QWEN_GEMV_BATCH_TILE 8

__device__ __constant__ float DSV4_FP4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

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
    return DSV4_FP4_E2M1_LUT[bits & 0x0f];
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
    int scale_rows,
    int scale_cols,
    int block_m,
    int block_k)
{
    const int sr = blockIdx.y;
    const int sc = blockIdx.x;
    if (sr >= scale_rows || sc >= scale_cols) return;
    const int row0 = sr * block_m;
    const int col0 = sc * block_k;
    const int rows = min(block_m, N - row0);
    const int cols = min(block_k, K - col0);
    const int elems = rows * cols;

    __shared__ float red[256];
    float amax = 0.0f;
    for (int i = threadIdx.x; i < elems; i += blockDim.x) {
        const int r = row0 + i / cols;
        const int c = col0 + i % cols;
        amax = fmaxf(amax, fabsf(__bfloat162float(input[(long)r * K + c])));
    }
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float scale = red[0] > 0.0f ? red[0] / 448.0f : 1.0f;
    if (threadIdx.x == 0) scales[sr * scale_cols + sc] = scale;
    __syncthreads();

    const float inv = 1.0f / scale;
    for (int i = threadIdx.x; i < elems; i += blockDim.x) {
        const int r = row0 + i / cols;
        const int c = col0 + i % cols;
        const float w = __bfloat162float(input[(long)r * K + c]) * inv;
        weight[(long)r * K + c] = __nv_cvt_float_to_fp8(w, __NV_SATFINITE, __NV_E4M3);
    }
}

extern "C" cudaError_t quantize_bf16_to_fp8_block_scaled_cuda(
    const __nv_bfloat16* input,
    uint8_t* weight,
    float* scales,
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
    // scale grid must tile the matrix exactly (ceil-div)
    if (scale_rows != (N + block_m - 1) / block_m || scale_cols != (K + block_k - 1) / block_k) {
        return cudaErrorInvalidValue;
    }
    dim3 grid((unsigned int)scale_cols, (unsigned int)scale_rows, 1);
    quantize_bf16_to_fp8_block_scaled_kernel<<<grid, 256, 0, stream>>>(
        input, weight, scales, N, K, scale_rows, scale_cols, block_m, block_k);
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

// W4A16 dequant directly to FP16 (skip BF16 intermediate). On sm_70 this
// avoids the BF16→FP16 cast that `gemm_cuda` would otherwise do, halving the
// dequant+cast memory traffic. Output is FP16; caller feeds it to a FP16
// cublasGemmEx directly.
//
// Each thread processes 8 consecutive int4 values (4 bytes) to amortize the
// scale load and reduce thread count.
__global__ void dequantize_w4a16_to_fp16_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    __half* __restrict__ output,
    int N,
    int K,
    int group_size)
{
    // Each thread handles 8 consecutive columns.
    const int cols_per_thread = 8;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)N * (K / cols_per_thread);
    if (idx >= total) return;
    const int row = (int)(idx / (K / cols_per_thread));
    const int col_base = (int)(idx % (K / cols_per_thread)) * cols_per_thread;
    const int num_groups = K / group_size;

    // Load 4 bytes = 8 int4 values.
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(
        &weight[row * (K / 2) + col_base / 2]);

    // Extract 8 nibbles.
    uint8_t b0 = packed & 0xFF;
    uint8_t b1 = (packed >> 8) & 0xFF;
    uint8_t b2 = (packed >> 16) & 0xFF;
    uint8_t b3 = (packed >> 24) & 0xFF;

    float vals[8];
    vals[0] = (float)((int)(b0 & 0x0F) - 8);
    vals[1] = (float)((int)(b0 >> 4) - 8);
    vals[2] = (float)((int)(b1 & 0x0F) - 8);
    vals[3] = (float)((int)(b1 >> 4) - 8);
    vals[4] = (float)((int)(b2 & 0x0F) - 8);
    vals[5] = (float)((int)(b2 >> 4) - 8);
    vals[6] = (float)((int)(b3 & 0x0F) - 8);
    vals[7] = (float)((int)(b3 >> 4) - 8);

    // Apply scales (one scale per group_size columns).
    #pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        int group = (col_base + i) / group_size;
        float scale = __bfloat162float(scales[row * num_groups + group]);
        output[row * K + col_base + i] = __float2half(vals[i] * scale);
    }
}

extern "C" cudaError_t dequantize_w4a16_to_fp16_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* scales,
    __half* output,
    int N,
    int K,
    int group_size,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || group_size <= 0 || K % group_size != 0 || (K & 1) != 0) {
        return cudaErrorInvalidValue;
    }
    const int cols_per_thread = 8;
    const long total = (long)N * (K / cols_per_thread);
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    dequantize_w4a16_to_fp16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        weight, scales, output, N, K, group_size);
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

__device__ __forceinline__ float fp4_e2m1_group_scale(
    const uint8_t* __restrict__ scales,
    const float* __restrict__ global_scales,
    int row,
    int col,
    int scale_cols,
    int group_size)
{
    const int group_raw = col / group_size;
    const int group = group_raw < scale_cols ? group_raw : (scale_cols - 1);
    return dsv4_decode_fp8_e4m3(scales[row * scale_cols + group]) * global_scales[0];
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

// Probe-only weight-read diagnostic (fp8_smallm_gemm_probe): same block
// geometry + access pattern as fp8_gemv_batch_kernel, x-work
// removed. mode 0 sums raw uint4 words (pure bandwidth); mode 1 adds the
// fp8->f32 decode (isolates decode ALU). H20 attribution (2026-07-10):
// mode0 2.9-3.7 TB/s, mode1 2.8-3.0, full GEMV 1.78 — the per-row x
// load+convert tail is the whole gap. Two fix attempts measured SLOWER and
// were killed (smem-staged x: LDS wavefronts cost the same as L1; x-in-regs
// 4-row tile: 62 vs 50 us) — keep this probe for the next attempt's A/B.
__global__ void fp8_wread_probe_kernel(
    const uint8_t* __restrict__ weight,
    float* __restrict__ output,
    int N, int K, int mode)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N) return;

    const uint8_t* weight_row = weight + (int64_t)row * K;
    const int kv = K / 16;
    float sum = 0.0f;
    if (mode == 0) {
        unsigned acc = 0;
        for (int v = tid_in_row; v < kv; v += threads_per_row) {
            const uint4 w = *reinterpret_cast<const uint4*>(weight_row + v * 16);
            acc += w.x + w.y + w.z + w.w;
        }
        sum = (float)acc;
    } else {
        for (int v = tid_in_row; v < kv; v += threads_per_row) {
            const auto* w4 = reinterpret_cast<const __nv_fp8x4_e4m3*>(weight_row + v * 16);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 wf = static_cast<float4>(w4[i]);
                sum += wf.x + wf.y + wf.z + wf.w;
            }
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
        output[row] = total;
    }
}

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

    const int bytes_per_row = K / 2;
    const __nv_bfloat16* x = input + batch_idx * K;
    float sum = 0.0f;
    for (int pair = tid_in_row; pair < bytes_per_row; pair += threads_per_row) {
        const int k0 = pair << 1;
        const int k1 = k0 + 1;
        const uint8_t packed = weight[row * bytes_per_row + pair];
        const uint8_t lo = packed & 0x0f;
        const uint8_t hi = (packed >> 4) & 0x0f;
        const float w0 = dsv4_decode_fp4_e2m1(lo)
            * fp4_e2m1_group_scale(scales, global_scales, row, k0, scale_cols, group_size);
        const float w1 = dsv4_decode_fp4_e2m1(hi)
            * fp4_e2m1_group_scale(scales, global_scales, row, k1, scale_cols, group_size);
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
    const int bytes_per_row = K / 2;
    float sum = 0.0f;
    for (int pair = tid_in_row; pair < bytes_per_row; pair += threads_per_row) {
        const int k0 = pair << 1;
        const int k1 = k0 + 1;
        const uint8_t packed = weight[row * bytes_per_row + pair];
        const uint8_t lo = packed & 0x0f;
        const uint8_t hi = (packed >> 4) & 0x0f;
        const float w0 = dsv4_decode_fp4_e2m1(lo)
            * fp4_e2m1_group_scale(scales, global_scales, row, k0, scale_cols, group_size);
        const float w1 = dsv4_decode_fp4_e2m1(hi)
            * fp4_e2m1_group_scale(scales, global_scales, row, k1, scale_cols, group_size);
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
    const int bytes_per_row = K / 2;
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (int pair = tid_in_row; pair < bytes_per_row; pair += threads_per_row) {
        const int k0 = pair << 1;
        const int k1 = k0 + 1;
        const uint8_t packed_a = weight_a[row * bytes_per_row + pair];
        const uint8_t packed_b = weight_b[row * bytes_per_row + pair];
        const uint8_t lo_a = packed_a & 0x0f;
        const uint8_t hi_a = (packed_a >> 4) & 0x0f;
        const uint8_t lo_b = packed_b & 0x0f;
        const uint8_t hi_b = (packed_b >> 4) & 0x0f;
        const float xv0 = __bfloat162float(x[k0]);
        const float xv1 = __bfloat162float(x[k1]);
        const float scale_a0 = fp4_e2m1_group_scale(scales_a, global_a, row, k0, scale_cols, group_size);
        const float scale_a1 = fp4_e2m1_group_scale(scales_a, global_a, row, k1, scale_cols, group_size);
        const float scale_b0 = fp4_e2m1_group_scale(scales_b, global_b, row, k0, scale_cols, group_size);
        const float scale_b1 = fp4_e2m1_group_scale(scales_b, global_b, row, k1, scale_cols, group_size);
        sum_a += dsv4_decode_fp4_e2m1(lo_a) * scale_a0 * xv0;
        sum_a += dsv4_decode_fp4_e2m1(hi_a) * scale_a1 * xv1;
        sum_b += dsv4_decode_fp4_e2m1(lo_b) * scale_b0 * xv0;
        sum_b += dsv4_decode_fp4_e2m1(hi_b) * scale_b1 * xv1;
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

__global__ void q8_embedding_batched_kernel(
    const int8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const int* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ out,
    int hidden_dim,
    int batch_size,
    int group_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden_dim * batch_size;
    if (idx >= total) return;

    const int batch = idx / hidden_dim;
    const int col = idx - batch * hidden_dim;
    const int row = token_ids[batch];
    const int num_groups = hidden_dim / group_size;
    const float scale = __bfloat162float(scales[row * num_groups + col / group_size]);
    const int8_t q = weight[row * hidden_dim + col];
    out[idx] = __float2bfloat16(static_cast<float>(q) * scale);
}

__global__ void q8_embedding_decode_kernel(
    const int8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const int* __restrict__ token_id,
    __nv_bfloat16* __restrict__ out,
    int hidden_dim,
    int group_size)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden_dim) return;

    const int row = token_id[0];
    const int num_groups = hidden_dim / group_size;
    const float scale = __bfloat162float(scales[row * num_groups + col / group_size]);
    const int8_t q = weight[row * hidden_dim + col];
    out[col] = __float2bfloat16(static_cast<float>(q) * scale);
}

// Batched W4A16 GEMM: B_TILE inputs share one weight read (vs B separate GEMVs
// re-reading weight B times). Each thread holds B_TILE accumulators; weight is
// loaded once per K-step and multiplied against B_TILE input vectors. Reads
// INT8 directly (4 weights per uint32, no nibble unpack) — this is the
// multi-request decode path (B>=2).
#define W4A16_GEMM_BTILE 4
__global__ void w4a16_gemm_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int group_size)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_base = blockIdx.y * W4A16_GEMM_BTILE;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;

    if (row >= N) return;

    float sum[W4A16_GEMM_BTILE];
    #pragma unroll
    for (int b = 0; b < W4A16_GEMM_BTILE; b++) sum[b] = 0.0f;

    int num_groups = K / group_size;
    int bytes_per_row = K / 2;
    int valid_b = min(W4A16_GEMM_BTILE, B - batch_base);

    for (int k = tid_in_row * 8; k < K; k += threads_per_row * 8) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * bytes_per_row + k / 2]);

        uint32_t lo4 = packed & 0x0F0F0F0Fu;
        uint32_t hi4 = (packed >> 4) & 0x0F0F0F0Fu;

        float w0 = (float)((int)(lo4 & 0xFF) - 8) * scale_f;
        float w1 = (float)((int)(hi4 & 0xFF) - 8) * scale_f;
        float w2 = (float)((int)((lo4 >> 8) & 0xFF) - 8) * scale_f;
        float w3 = (float)((int)((hi4 >> 8) & 0xFF) - 8) * scale_f;
        float w4 = (float)((int)((lo4 >> 16) & 0xFF) - 8) * scale_f;
        float w5 = (float)((int)((hi4 >> 16) & 0xFF) - 8) * scale_f;
        float w6 = (float)((int)((lo4 >> 24) & 0xFF) - 8) * scale_f;
        float w7 = (float)((int)((hi4 >> 24) & 0xFF) - 8) * scale_f;

        #pragma unroll
        for (int b = 0; b < W4A16_GEMM_BTILE; b++) {
            if (b >= valid_b) break;
            const __nv_bfloat16* xb = input + (batch_base + b) * K;
            sum[b] += w0 * __bfloat162float(xb[k]);
            sum[b] += w1 * __bfloat162float(xb[k+1]);
            sum[b] += w2 * __bfloat162float(xb[k+2]);
            sum[b] += w3 * __bfloat162float(xb[k+3]);
            sum[b] += w4 * __bfloat162float(xb[k+4]);
            sum[b] += w5 * __bfloat162float(xb[k+5]);
            sum[b] += w6 * __bfloat162float(xb[k+6]);
            sum[b] += w7 * __bfloat162float(xb[k+7]);
        }
    }

    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    #pragma unroll
    for (int b = 0; b < W4A16_GEMM_BTILE; b++) {
        if (b < valid_b) sum[b] = warp_reduce_sum(sum[b]);
    }

    __shared__ float smem_out[GEMV_ROWS * W4A16_GEMM_BTILE * 8];
    if (lane_id == 0) {
        #pragma unroll
        for (int b = 0; b < W4A16_GEMM_BTILE; b++) {
            if (b < valid_b)
                smem_out[(row_in_block * W4A16_GEMM_BTILE + b) * warps_per_row + warp_in_row] = sum[b];
        }
    }
    __syncthreads();
    if (tid_in_row == 0) {
        #pragma unroll
        for (int b = 0; b < W4A16_GEMM_BTILE; b++) {
            if (b >= valid_b) break;
            float total = 0.0f;
            for (int w = 0; w < warps_per_row; w++)
                total += smem_out[(row_in_block * W4A16_GEMM_BTILE + b) * warps_per_row + w];
            output[(batch_base + b) * N + row] = __float2bfloat16(total);
        }
    }
}

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
    int group_size)
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

            uint32_t lo_all = p & MASK4;
            uint32_t hi_all = (p >> 4) & MASK4;

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
    int group_size)
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
            uint32_t lo_a = pa & MASK4;
            uint32_t hi_a = (pa >> 4) & MASK4;
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
            uint32_t lo_b = pb & MASK4;
            uint32_t hi_b = (pb >> 4) & MASK4;
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
                output_a[route * N + row] = __float2bfloat16(sa);
                output_b[route * N + row] = __float2bfloat16(sb);
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
                    output_a[route * N + row] = __float2bfloat16(ta);
                    output_b[route * N + row] = __float2bfloat16(tb);
                }
            }
        }
    }
}

// Batched W2A16 GEMV: [B, K] × [N, K/4]^T → [B, N]
// Same 2-bit extraction as single W2A16, with batch dimension in grid.y.
__global__ void w2a16_gemv_batch_kernel(
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
    int bytes_per_row = K / 4;

    for (int k = tid_in_row * 16; k < K; k += threads_per_row * 16) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * bytes_per_row + k / 4]);

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int val = static_cast<int>((packed >> (i * 2)) & 0x3) - 2;
            sum += static_cast<float>(val) * scale_f * __bfloat162float(x[k + i]);
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
#define Q3K_SB_SIZE 256
#define Q3K_SB_BYTES 110
#define Q3K_GEMV_ROWS 8
#define Q3K_GEMV_THREADS 256  // = Q3K_GEMV_ROWS * 32

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

__global__ void q3k_gemv_kernel(
    const uint8_t* __restrict__ weight,       // [N, (K/256) * 110]
    const __nv_bfloat16* __restrict__ input,  // [K]
    __nv_bfloat16* __restrict__ output,       // [N]
    int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q3K_GEMV_ROWS + warp_id;
    if (row >= N) return;

    const int num_sb    = K / Q3K_SB_SIZE;
    const int row_bytes = num_sb * Q3K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q3K_SB_BYTES;
        const uint8_t* hmask = sb_p + 0;
        const uint8_t* qs    = sb_p + 32;
        const uint8_t* sc_raw = sb_p + 96;

        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 108))[0];
        const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));

        int8_t scales[16];
        q3k_decode_scales(sc_raw, scales);

        const int k_base = sb * Q3K_SB_SIZE;

        // Each lane handles 8 elements per superblock, stride 32 → adjacent lanes
        // touch adjacent K indices → coalesced input loads.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k_local = i * 32 + lane;  // 0..255
            const int q2 = (qs[k_local >> 2] >> ((k_local & 3) << 1)) & 0x3;
            const int hbit = (hmask[k_local >> 3] >> (k_local & 7)) & 0x1;
            const int q3 = q2 | (hbit << 2);
            const int sub_idx = k_local >> 4;  // /16
            const float scale = d * (float)scales[sub_idx];
            const float w = scale * ((float)q3 - 4.0f);
            sum += w * __bfloat162float(input[k_base + k_local]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[row] = __float2bfloat16(sum);
}

__global__ void q3k_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q3K_GEMV_ROWS + warp_id;
    const int batch   = blockIdx.y;
    if (row >= N || batch >= B) return;

    const int num_sb    = K / Q3K_SB_SIZE;
    const int row_bytes = num_sb * Q3K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;
    const __nv_bfloat16* x = input + batch * K;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q3K_SB_BYTES;
        const uint8_t* hmask = sb_p + 0;
        const uint8_t* qs    = sb_p + 32;
        const uint8_t* sc_raw = sb_p + 96;
        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 108))[0];
        const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));

        int8_t scales[16];
        q3k_decode_scales(sc_raw, scales);
        const int k_base = sb * Q3K_SB_SIZE;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k_local = i * 32 + lane;
            const int q2 = (qs[k_local >> 2] >> ((k_local & 3) << 1)) & 0x3;
            const int hbit = (hmask[k_local >> 3] >> (k_local & 7)) & 0x1;
            const int q3 = q2 | (hbit << 2);
            const int sub_idx = k_local >> 4;
            const float scale = d * (float)scales[sub_idx];
            const float w = scale * ((float)q3 - 4.0f);
            sum += w * __bfloat162float(x[k_base + k_local]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[batch * N + row] = __float2bfloat16(sum);
}

// Dequant chunk kernel: writes a BF16 tile [N, k_len] starting at k_start.
// Grid: (N, k_len / 256).  Block: 256 threads — one per element in the superblock.
__global__ void q3k_dequant_chunk_kernel(
    const uint8_t* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int N, int K, int k_start, int k_len)
{
    const int row = blockIdx.x;
    const int sb_in_chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= N) return;

    const int num_sb_total = K / Q3K_SB_SIZE;
    const int sb_global    = (k_start / Q3K_SB_SIZE) + sb_in_chunk;
    const int row_bytes    = num_sb_total * Q3K_SB_BYTES;
    const uint8_t* sb_p    = weight + row * row_bytes + sb_global * Q3K_SB_BYTES;

    __shared__ float s_d;
    __shared__ int8_t s_scales[16];

    if (tid == 0) {
        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 108))[0];
        s_d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        q3k_decode_scales(sb_p + 96, s_scales);
    }
    __syncthreads();

    const uint8_t* hmask = sb_p + 0;
    const uint8_t* qs    = sb_p + 32;

    const int k_local = tid;
    const int q2 = (qs[k_local >> 2] >> ((k_local & 3) << 1)) & 0x3;
    const int hbit = (hmask[k_local >> 3] >> (k_local & 7)) & 0x1;
    const int q3 = q2 | (hbit << 2);
    const int sub_idx = k_local >> 4;
    const float scale = s_d * (float)s_scales[sub_idx];
    const float w = scale * ((float)q3 - 4.0f);

    const int out_k = sb_in_chunk * Q3K_SB_SIZE + k_local;
    out[row * k_len + out_k] = __float2bfloat16(w);
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

cudaError_t w2a16_gemv_cuda(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, 1);
    dim3 block(GEMV_THREADS);
    w2a16_gemv_batch_kernel<<<grid, block, 0, stream>>>(
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

extern "C" cudaError_t w4a16_gemm_wmma_cuda(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int M, int N, int K, int group_size, cudaStream_t stream);

cudaError_t w4a16_gemv_batch_cuda(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, cudaStream_t stream)
{
    // Marlin GEMV for all batch sizes. The WMMA tensor-core path is under
    // development (correctness issues); fall back to the known-good marlin
    // GEMV which uses uint4 loads and FP16 FMA.
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
        max_count, N, K, group_size);
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
        group_size);
    return cudaGetLastError();
}

cudaError_t w2a16_gemv_batch_cuda(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    dim3 block(GEMV_THREADS);
    w2a16_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, B, N, K, group_size);
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

cudaError_t gemv_fp8_wread_probe_cuda(
    const uint8_t* weight, float* output, int N, int K, int mode,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || (K % 16) != 0) return cudaErrorInvalidValue;
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS);
    dim3 block(GEMV_THREADS);
    fp8_wread_probe_kernel<<<grid, block, 0, stream>>>(weight, output, N, K, mode);
    return cudaGetLastError();
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
        scale_cols <= 0 || (K % group_size) != 0) {
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
        (K & 1) != 0 || group_size <= 0 || scale_cols <= 0 || (K % group_size) != 0) {
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
        (K & 1) != 0 || group_size <= 0 || scale_cols <= 0 || (K % group_size) != 0) {
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

cudaError_t q3k_gemv_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q3K_GEMV_ROWS - 1) / Q3K_GEMV_ROWS);
    dim3 block(Q3K_GEMV_THREADS);
    q3k_gemv_kernel<<<grid, block, 0, stream>>>(weight, input, output, N, K);
    return cudaGetLastError();
}

cudaError_t q3k_gemv_batch_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q3K_GEMV_ROWS - 1) / Q3K_GEMV_ROWS, B);
    dim3 block(Q3K_GEMV_THREADS);
    q3k_gemv_batch_kernel<<<grid, block, 0, stream>>>(weight, input, output, B, N, K);
    return cudaGetLastError();
}

cudaError_t q3k_dequant_chunk_cuda(
    const uint8_t* weight, __nv_bfloat16* out,
    int N, int K, int k_start, int k_len, cudaStream_t stream)
{
    if ((k_start % Q3K_SB_SIZE) != 0 || (k_len % Q3K_SB_SIZE) != 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(N, k_len / Q3K_SB_SIZE);
    dim3 block(Q3K_SB_SIZE);
    q3k_dequant_chunk_kernel<<<grid, block, 0, stream>>>(
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

cudaError_t q8_embedding_batched_cuda(
    const int8_t* weight, const __nv_bfloat16* scales, const int* token_ids,
    __nv_bfloat16* out, int hidden_dim, int batch_size, int group_size,
    cudaStream_t stream)
{
    if (hidden_dim <= 0 || group_size <= 0 || (hidden_dim % group_size) != 0) {
        return cudaErrorInvalidValue;
    }
    const int total = hidden_dim * batch_size;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    q8_embedding_batched_kernel<<<grid, block, 0, stream>>>(
        weight, scales, token_ids, out, hidden_dim, batch_size, group_size);
    return cudaGetLastError();
}

cudaError_t q8_embedding_decode_cuda(
    const int8_t* weight, const __nv_bfloat16* scales, const int* token_id,
    __nv_bfloat16* out, int hidden_dim, int group_size, cudaStream_t stream)
{
    if (hidden_dim <= 0 || group_size <= 0 || (hidden_dim % group_size) != 0) {
        return cudaErrorInvalidValue;
    }
    const int block = 256;
    const int grid = (hidden_dim + block - 1) / block;
    q8_embedding_decode_kernel<<<grid, block, 0, stream>>>(
        weight, scales, token_id, out, hidden_dim, group_size);
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

cudaError_t q3k_embedding_batched_cuda(
    const uint8_t* weight, const int* token_ids, __nv_bfloat16* out,
    int hidden_dim, int batch_size, cudaStream_t stream)
{
    return qxk_embedding_batched_cuda(
        weight, token_ids, out, hidden_dim, batch_size, 3, Q3K_SB_BYTES, stream);
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

cudaError_t q3k_embedding_decode_cuda(
    const uint8_t* weight, const int* token_id, __nv_bfloat16* out,
    int hidden_dim, cudaStream_t stream)
{
    return qxk_embedding_decode_cuda(weight, token_id, out, hidden_dim, 3, Q3K_SB_BYTES, stream);
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

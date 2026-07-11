// Self-contained CUDA micro-benchmark for the Qwen FP8 block-scaled decode GEMV.
//
// A/B kernel variants for PERFORMANCE and verify CORRECTNESS on any sm_80+ GPU
// WITHOUT needing the 27B model. Designed for a Colab GPU (L4 / A100 / H100).
//
// Build + run (single nvcc command, no Rust, no external libs beyond CUDA):
//   nvcc -O3 -arch=sm_89 gemv_bench.cu -o gemv_bench && ./gemv_bench 4000
//
// -arch: sm_89 for L4, sm_80 for A100, sm_90 for H100/H20.
// arg1 = the GPU's roofline HBM bandwidth in GB/s
//        (L4 ~= 300, A100 ~= 2039, H100 ~= 3350, H20 ~= 4000).
//
// The optimized kernel (fp8_f32_block_gemv_batch_kernel) and its device helpers
// are extracted VERBATIM from
//   crates/cuda-kernels/csrc/gemm/quantized_gemv.cu
// The NAIVE reference kernel (gemv_reference_kernel) uses the SAME decode + scale
// math so cosine(out, ref) ~= 1.0 is the correctness gate (PASS if > 0.9999).

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>  // atof, std::exit
#include <cmath>
#include <vector>
#include <algorithm>

// ============================================================================
// VERBATIM from quantized_gemv.cu — #defines + device helpers + kernel + launch
// ============================================================================

#define WARP_SIZE 32
#ifndef GEMV_THREADS
#define GEMV_THREADS 256
#endif
#ifndef GEMV_ROWS
#define GEMV_ROWS 4
#endif

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

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

// Dot 16 fp8 weights against an ALREADY-LOADED bf16 x16 (passed as two uint4
// registers). Identical math to fp8_f32_dot16, but the activation is loaded by
// the CALLER once and reused across N-blocked output rows — so the activation
// L1TEX load (2 × 128-bit per chunk) is issued ONCE per chunk instead of once
// per output row. Same accumulation order → numerically identical.
__device__ __forceinline__ float dot16_xreg(
    const uint8_t* __restrict__ weight,
    const uint4& x0,
    const uint4& x1)
{
    const auto* w4 = reinterpret_cast<const __nv_fp8x4_e4m3*>(weight);
    const float4 wf0 = static_cast<float4>(w4[0]);
    const float4 wf1 = static_cast<float4>(w4[1]);
    const float4 wf2 = static_cast<float4>(w4[2]);
    const float4 wf3 = static_cast<float4>(w4[3]);
    const auto* xb0 = reinterpret_cast<const __nv_bfloat16*>(&x0);
    const auto* xb1 = reinterpret_cast<const __nv_bfloat16*>(&x1);
    return wf0.x * __bfloat162float(xb0[0]) + wf0.y * __bfloat162float(xb0[1])
        + wf0.z * __bfloat162float(xb0[2]) + wf0.w * __bfloat162float(xb0[3])
        + wf1.x * __bfloat162float(xb0[4]) + wf1.y * __bfloat162float(xb0[5])
        + wf1.z * __bfloat162float(xb0[6]) + wf1.w * __bfloat162float(xb0[7])
        + wf2.x * __bfloat162float(xb1[0]) + wf2.y * __bfloat162float(xb1[1])
        + wf2.z * __bfloat162float(xb1[2]) + wf2.w * __bfloat162float(xb1[3])
        + wf3.x * __bfloat162float(xb1[4]) + wf3.y * __bfloat162float(xb1[5])
        + wf3.z * __bfloat162float(xb1[6]) + wf3.w * __bfloat162float(xb1[7]);
}

// N register-blocked GEMV: each 64-thread group computes R consecutive output
// rows, loading the activation chunk ONCE into registers (x0/x1) and reusing it
// across all R weight rows. Attacks the measured L1TEX-load bottleneck (ncu:
// L1/TEX 78%, DRAM 35%, top stall = L1TEX scoreboard) by cutting activation
// loads R×. blocks = ceil(N/(GEMV_ROWS*R)) — still ≫ #SM at R≤8, so occupancy
// holds (unlike the N/32-block MMA tile). B=1 decode shape (grid.y = B).
template <int R>
__global__ void fp8_f32_block_gemv_nblock_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int scale_rows, int scale_cols, int block_m, int block_k)
{
    const int threads_per_group = GEMV_THREADS / GEMV_ROWS;  // 64
    const int group_in_block = threadIdx.x / threads_per_group;  // 0..GEMV_ROWS-1
    const int tid_in_group = threadIdx.x % threads_per_group;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warps_per_group = threads_per_group / WARP_SIZE;  // 2
    const int warp_in_group = tid_in_group / WARP_SIZE;
    const int row_base = (blockIdx.x * GEMV_ROWS + group_in_block) * R;
    const int batch_idx = blockIdx.y;
    if (row_base >= N || batch_idx >= B) return;

    const __nv_bfloat16* x = input + (int64_t)batch_idx * K;
    float sums[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sums[r] = 0.0f;

    const int kv = K / 16;  // Qwen shapes: K % 16 == 0
    for (int v = tid_in_group; v < kv; v += threads_per_group) {
        const int k = v * 16;
        const uint4 x0 = *reinterpret_cast<const uint4*>(x + k);
        const uint4 x1 = *reinterpret_cast<const uint4*>(x + k + 8);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row_base + r;
            if (row < N) {
                const float scale = fp8_f32_block_scale(
                    scales, row, k, scale_rows, scale_cols, block_m, block_k);
                sums[r] += scale * dot16_xreg(weight + (int64_t)row * K + k, x0, x1);
            }
        }
    }

    // Reduce each row's partial sum across the 64-thread group (warp + smem).
    __shared__ float smem[GEMV_ROWS * 2 * R];  // [group][warp_in_group][r]
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = warp_reduce_sum(sums[r]);
        if (lane_id == 0)
            smem[(group_in_block * warps_per_group + warp_in_group) * R + r] = s;
    }
    __syncthreads();
    if (tid_in_group == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row_base + r;
            if (row >= N) continue;
            float total = 0.0f;
            for (int w = 0; w < warps_per_group; ++w)
                total += smem[(group_in_block * warps_per_group + w) * R + r];
            output[(int64_t)batch_idx * N + row] = __float2bfloat16(total);
        }
    }
}

// dot16 of PRE-DECODED weight (4 float4, decoded once) against a bf16 x16.
// Lets the batched kernel decode each weight chunk ONCE and MAC it across all
// batch columns — the weight HBM read + fp8 decode are amortized over B.
__device__ __forceinline__ float dot16_dec(
    const float4& wf0, const float4& wf1, const float4& wf2, const float4& wf3,
    const __nv_bfloat16* __restrict__ x)
{
    const uint4 x0 = *reinterpret_cast<const uint4*>(x);
    const uint4 x1 = *reinterpret_cast<const uint4*>(x + 8);
    const auto* xb0 = reinterpret_cast<const __nv_bfloat16*>(&x0);
    const auto* xb1 = reinterpret_cast<const __nv_bfloat16*>(&x1);
    return wf0.x * __bfloat162float(xb0[0]) + wf0.y * __bfloat162float(xb0[1])
        + wf0.z * __bfloat162float(xb0[2]) + wf0.w * __bfloat162float(xb0[3])
        + wf1.x * __bfloat162float(xb0[4]) + wf1.y * __bfloat162float(xb0[5])
        + wf1.z * __bfloat162float(xb0[6]) + wf1.w * __bfloat162float(xb0[7])
        + wf2.x * __bfloat162float(xb1[0]) + wf2.y * __bfloat162float(xb1[1])
        + wf2.z * __bfloat162float(xb1[2]) + wf2.w * __bfloat162float(xb1[3])
        + wf3.x * __bfloat162float(xb1[4]) + wf3.y * __bfloat162float(xb1[5])
        + wf3.z * __bfloat162float(xb1[6]) + wf3.w * __bfloat162float(xb1[7]);
}

// Weight-AMORTIZING batched GEMV (the MTP-verify lever): grid.y = 1, each thread
// reads+decodes each weight chunk ONCE and MACs it across all B columns. Weight
// HBM read is shared over B (vs the default grid.y=B kernel that re-reads weight
// B times). Mirrors the production fp8_gemv_batch_tiled_kernel (Fp8F32BlockScale).
template <int TILE>
__global__ void fp8_f32_block_gemv_amort_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int scale_rows, int scale_cols, int block_m, int block_k)
{
    const int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    const int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    const int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N) return;
    const int bn = B < TILE ? B : TILE;

    float sums[TILE];
#pragma unroll
    for (int b = 0; b < TILE; ++b) sums[b] = 0.0f;

    const int kv = K / 16;
    const uint8_t* weight_row = weight + (int64_t)row * K;
    for (int v = tid_in_row; v < kv; v += threads_per_row) {
        const int k = v * 16;
        const float scale =
            fp8_f32_block_scale(scales, row, k, scale_rows, scale_cols, block_m, block_k);
        const auto* w4 = reinterpret_cast<const __nv_fp8x4_e4m3*>(weight_row + k);
        const float4 wf0 = static_cast<float4>(w4[0]);
        const float4 wf1 = static_cast<float4>(w4[1]);
        const float4 wf2 = static_cast<float4>(w4[2]);
        const float4 wf3 = static_cast<float4>(w4[3]);
#pragma unroll
        for (int b = 0; b < TILE; ++b)
            if (b < bn)
                sums[b] += scale * dot16_dec(wf0, wf1, wf2, wf3, input + (int64_t)b * K + k);
    }

    __shared__ float smem[GEMV_ROWS * 2 * TILE];
    const int warps_per_row = threads_per_row / WARP_SIZE;
    const int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
#pragma unroll
    for (int b = 0; b < TILE; ++b) {
        float s = warp_reduce_sum(sums[b]);
        if (lane_id == 0)
            smem[(row_in_block * warps_per_row + warp_in_row) * TILE + b] = s;
    }
    __syncthreads();
    if (tid_in_row == 0) {
#pragma unroll
        for (int b = 0; b < TILE; ++b) {
            if (b >= bn) continue;
            float total = 0.0f;
            for (int w = 0; w < warps_per_row; ++w)
                total += smem[(row_in_block * warps_per_row + w) * TILE + b];
            output[(int64_t)b * N + row] = __float2bfloat16(total);
        }
    }
}

// Practical HBM-ceiling probe: read exactly the N*K weight bytes (coalesced
// uint4), XOR-reduce (defeat DCE), no activation / no fp8 decode / no FMA. Its
// GB/s = the best this shape+duration can do on this GPU — the ceiling any real
// GEMV is bounded by. (cosine will FAIL; only the GB/s line is meaningful.)
__global__ void weight_stream_kernel(
    const uint8_t* __restrict__ weight, __nv_bfloat16* __restrict__ output, int N, int K)
{
    const int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    const int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    const int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    if (row >= N) return;
    const uint8_t* wr = weight + (int64_t)row * K;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (int k = tid_in_row * 16; k < K; k += threads_per_row * 16) {
        const uint4 w = *reinterpret_cast<const uint4*>(wr + k);
        acc.x ^= w.x; acc.y ^= w.y; acc.z ^= w.z; acc.w ^= w.w;
    }
    unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) r ^= __shfl_xor_sync(0xffffffff, r, o);
    if (tid_in_row == 0) output[(int64_t)row] = __float2bfloat16((float)(r & 0xFF));
}

// dot16 from an ALREADY-LOADED raw fp8x16 (one uint4 = 16 fp8 bytes). Lets the
// caller issue the weight DRAM load AHEAD of the convert+FMA (software prefetch),
// so the cold-weight Long-Scoreboard latency (ncu: 6.55 cyc/issue, the dominant
// stall) is hidden behind the current chunk's compute. Math identical to
// fp8_f32_dot16.
__device__ __forceinline__ float dot16_from_raw(
    const uint4& wraw, const __nv_bfloat16* __restrict__ x)
{
    const auto* w4 = reinterpret_cast<const __nv_fp8x4_e4m3*>(&wraw);
    const float4 wf0 = static_cast<float4>(w4[0]);
    const float4 wf1 = static_cast<float4>(w4[1]);
    const float4 wf2 = static_cast<float4>(w4[2]);
    const float4 wf3 = static_cast<float4>(w4[3]);
    const uint4 x0 = *reinterpret_cast<const uint4*>(x);
    const uint4 x1 = *reinterpret_cast<const uint4*>(x + 8);
    const auto* xb0 = reinterpret_cast<const __nv_bfloat16*>(&x0);
    const auto* xb1 = reinterpret_cast<const __nv_bfloat16*>(&x1);
    return wf0.x * __bfloat162float(xb0[0]) + wf0.y * __bfloat162float(xb0[1])
        + wf0.z * __bfloat162float(xb0[2]) + wf0.w * __bfloat162float(xb0[3])
        + wf1.x * __bfloat162float(xb0[4]) + wf1.y * __bfloat162float(xb0[5])
        + wf1.z * __bfloat162float(xb0[6]) + wf1.w * __bfloat162float(xb0[7])
        + wf2.x * __bfloat162float(xb1[0]) + wf2.y * __bfloat162float(xb1[1])
        + wf2.z * __bfloat162float(xb1[2]) + wf2.w * __bfloat162float(xb1[3])
        + wf3.x * __bfloat162float(xb1[4]) + wf3.y * __bfloat162float(xb1[5])
        + wf3.z * __bfloat162float(xb1[6]) + wf3.w * __bfloat162float(xb1[7]);
}

// Weight-prefetch GEMV: SAME geometry/occupancy as the scalar kernel, but a
// depth-2 software pipeline keeps the NEXT weight chunk's DRAM load in flight
// (named registers w_cur/w_next — no array → no local-memory spill) while the
// current chunk converts+FMAs. Hides the cold-weight Long-Scoreboard latency
// (ncu: 6.55 cyc/issue, the dominant stall) to fill DRAM BW. Activation (L1-hit,
// Short-Scoreboard negligible) left as a normal load.
__global__ void fp8_f32_block_gemv_prefetch_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int scale_rows, int scale_cols, int block_m, int block_k)
{
    const int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    const int batch_idx = blockIdx.y;
    const int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    const int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N || batch_idx >= B) return;

    const __nv_bfloat16* x = input + (int64_t)batch_idx * K;
    const uint8_t* weight_row = weight + (int64_t)row * K;
    const int kv = K / 16;
    float sum = 0.0f;

    if (tid_in_row < kv) {
        int k_cur = tid_in_row * 16;
        uint4 w_cur = *reinterpret_cast<const uint4*>(weight_row + k_cur);
        for (int v = tid_in_row + threads_per_row;; v += threads_per_row) {
            const bool has_next = v < kv;
            uint4 w_next;
            int k_next = v * 16;
            if (has_next)  // issue next load AHEAD of consuming w_cur
                w_next = *reinterpret_cast<const uint4*>(weight_row + k_next);
            const float scale =
                fp8_f32_block_scale(scales, row, k_cur, scale_rows, scale_cols, block_m, block_k);
            sum += scale * dot16_from_raw(w_cur, x + k_cur);
            if (!has_next) break;
            w_cur = w_next;
            k_cur = k_next;
        }
    }

    sum = warp_reduce_sum(sum);
    __shared__ float rsmem[GEMV_ROWS * 8];
    const int warps_per_row = threads_per_row / WARP_SIZE;
    const int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) rsmem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; ++w)
            total += rsmem[row_in_block * warps_per_row + w];
        output[(int64_t)batch_idx * N + row] = __float2bfloat16(total);
    }
}

// Shared-memory activation-staged GEMV: SAME block geometry as the scalar
// kernel (GEMV_ROWS rows/block, N/GEMV_ROWS blocks → occupancy unchanged), but
// the activation K-vector is loaded into dynamic smem ONCE per block (256
// threads cooperatively) and all GEMV_ROWS rows read it from smem. The activation
// is ~2/3 of the scalar kernel's L1TEX load instructions (ncu: L1/TEX 78% =
// bottleneck) and smem accesses use a SEPARATE pipe — so this cuts L1TEX traffic
// ~2× without losing the occupancy that N-blocking sacrificed. Weight still
// streams cold from HBM (the irreducible N*K read). K%16==0 required.
__global__ void fp8_f32_block_gemv_smem_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int scale_rows, int scale_cols, int block_m, int block_k)
{
    extern __shared__ __nv_bfloat16 xs[];  // K bf16 activations for this batch col
    const int batch_idx = blockIdx.y;
    if (batch_idx >= B) return;
    const __nv_bfloat16* x = input + (int64_t)batch_idx * K;
    // Cooperative coalesced fill (once per block).
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    const int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    const int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row_in_block = threadIdx.x / threads_per_row;
    float sum = 0.0f;
    if (row < N) {
        const int kv = K / 16;
        const uint8_t* weight_row = weight + (int64_t)row * K;
        for (int v = tid_in_row; v < kv; v += threads_per_row) {
            const int k = v * 16;
            const float scale =
                fp8_f32_block_scale(scales, row, k, scale_rows, scale_cols, block_m, block_k);
            sum += scale * fp8_f32_dot16(weight_row + k, xs + k);  // activation from smem
        }
    }
    sum = warp_reduce_sum(sum);
    __shared__ float rsmem[GEMV_ROWS * 8];
    const int warps_per_row = threads_per_row / WARP_SIZE;
    const int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) rsmem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0 && row < N) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; ++w)
            total += rsmem[row_in_block * warps_per_row + w];
        output[(int64_t)batch_idx * N + row] = __float2bfloat16(total);
    }
}

__global__ void fp8_f32_block_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int block_m,
    int block_k)
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
    // Qwen FP8 decode uses K/block_k aligned to 16, so one block scale covers
    // each vector dot and the scalar fallback stays available for odd shapes.
    if ((K % 16) == 0 && (block_k % 16) == 0) {
        const int kv = K / 16;
        const uint8_t* weight_row = weight + (int64_t)row * K;
#ifdef ILP_UNROLL
        // 4 independent accumulators -> 4 dot16 loads in flight before any
        // dependent add, hiding HBM latency on the short K-walk.
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        const int step = threads_per_row;
        int v = tid_in_row;
        for (; v + 3 * step < kv; v += 4 * step) {
            const int k0 = v * 16, k1 = (v + step) * 16;
            const int k2 = (v + 2 * step) * 16, k3 = (v + 3 * step) * 16;
            s0 += fp8_f32_block_scale(scales, row, k0, scale_rows, scale_cols, block_m, block_k)
                  * fp8_f32_dot16(weight_row + k0, x + k0);
            s1 += fp8_f32_block_scale(scales, row, k1, scale_rows, scale_cols, block_m, block_k)
                  * fp8_f32_dot16(weight_row + k1, x + k1);
            s2 += fp8_f32_block_scale(scales, row, k2, scale_rows, scale_cols, block_m, block_k)
                  * fp8_f32_dot16(weight_row + k2, x + k2);
            s3 += fp8_f32_block_scale(scales, row, k3, scale_rows, scale_cols, block_m, block_k)
                  * fp8_f32_dot16(weight_row + k3, x + k3);
        }
        for (; v < kv; v += step) {
            const int k = v * 16;
            s0 += fp8_f32_block_scale(scales, row, k, scale_rows, scale_cols, block_m, block_k)
                  * fp8_f32_dot16(weight_row + k, x + k);
        }
        sum = (s0 + s1) + (s2 + s3);
#else
        for (int v = tid_in_row; v < kv; v += threads_per_row) {
            const int k = v * 16;
            const float scale =
                fp8_f32_block_scale(scales, row, k, scale_rows, scale_cols, block_m, block_k);
            sum += scale * fp8_f32_dot16(weight_row + k, x + k);
        }
#endif
    } else {
        for (int k = tid_in_row; k < K; k += threads_per_row) {
            const float w = dsv4_decode_fp8_e4m3(weight[row * K + k])
                * fp8_f32_block_scale(scales, row, k, scale_rows, scale_cols, block_m, block_k);
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

extern "C" cudaError_t gemv_fp8_block_scaled_batch_cuda(
    const uint8_t* weight, const float* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int scale_rows, int scale_cols, int block_m, int block_k,
    cudaStream_t stream)
{
    if (B <= 0 || N <= 0 || K <= 0 || scale_rows <= 0 || scale_cols <= 0 ||
        block_m <= 0 || block_k <= 0) {
        return cudaErrorInvalidValue;
    }
    // GEMV_NBLOCK=R routes to the N register-blocked kernel (R output rows per
    // 64-thread group, activation reused R× in registers). Default (unset/0) =
    // the proven scalar warp-per-row kernel, byte-for-byte. K%16==0 required.
    static int nblock = -1;
    static int use_smem = -1;
    if (nblock < 0) {
        const char* e = getenv("GEMV_NBLOCK");
        nblock = (e && e[0]) ? atoi(e) : 0;
        const char* es = getenv("GEMV_SMEM");
        use_smem = (es && es[0] && es[0] != '0') ? 1 : 0;
    }
    static int use_prefetch = -1;
    if (use_prefetch < 0) {
        const char* ep = getenv("GEMV_PREFETCH");
        use_prefetch = (ep && ep[0] && ep[0] != '0') ? 1 : 0;
    }
    if (use_prefetch && (K % 16) == 0) {
        dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
        dim3 block(GEMV_THREADS);
        fp8_f32_block_gemv_prefetch_kernel<<<grid, block, 0, stream>>>(
            weight, scales, input, output, B, N, K, scale_rows, scale_cols, block_m, block_k);
        return cudaGetLastError();
    }
    if (getenv("GEMV_AMORT") && (K % 16) == 0) {  // weight-amortizing batched GEMV
        // grid.y=1: weight read ONCE, MAC'd across all B columns. TILE is
        // instantiated == B (small spec-decode depth) so sums[TILE] uses exactly
        // B registers — NOT a fixed-16 array that craters occupancy at small B.
        dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, 1);
        dim3 block(GEMV_THREADS);
        switch (B) {
            case 1:  fp8_f32_block_gemv_amort_kernel<1><<<grid,block,0,stream>>>(weight,scales,input,output,B,N,K,scale_rows,scale_cols,block_m,block_k); break;
            case 2:  fp8_f32_block_gemv_amort_kernel<2><<<grid,block,0,stream>>>(weight,scales,input,output,B,N,K,scale_rows,scale_cols,block_m,block_k); break;
            case 3:  fp8_f32_block_gemv_amort_kernel<3><<<grid,block,0,stream>>>(weight,scales,input,output,B,N,K,scale_rows,scale_cols,block_m,block_k); break;
            case 4:  fp8_f32_block_gemv_amort_kernel<4><<<grid,block,0,stream>>>(weight,scales,input,output,B,N,K,scale_rows,scale_cols,block_m,block_k); break;
            case 5:  fp8_f32_block_gemv_amort_kernel<5><<<grid,block,0,stream>>>(weight,scales,input,output,B,N,K,scale_rows,scale_cols,block_m,block_k); break;
            case 6:  fp8_f32_block_gemv_amort_kernel<6><<<grid,block,0,stream>>>(weight,scales,input,output,B,N,K,scale_rows,scale_cols,block_m,block_k); break;
            case 8:  fp8_f32_block_gemv_amort_kernel<8><<<grid,block,0,stream>>>(weight,scales,input,output,B,N,K,scale_rows,scale_cols,block_m,block_k); break;
            default: fp8_f32_block_gemv_amort_kernel<16><<<grid,block,0,stream>>>(weight,scales,input,output,B,N,K,scale_rows,scale_cols,block_m,block_k); break;
        }
        return cudaGetLastError();
    }
    if (getenv("GEMV_STREAM") && (K % 16) == 0) {  // HBM-ceiling probe (cosine FAILs)
        dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
        dim3 block(GEMV_THREADS);
        weight_stream_kernel<<<grid, block, 0, stream>>>(weight, output, N, K);
        return cudaGetLastError();
    }
    if (use_smem && (K % 16) == 0) {
        const size_t smem_bytes = (size_t)K * sizeof(__nv_bfloat16);
        cudaFuncSetAttribute(fp8_f32_block_gemv_smem_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
        dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
        dim3 block(GEMV_THREADS);
        fp8_f32_block_gemv_smem_kernel<<<grid, block, smem_bytes, stream>>>(
            weight, scales, input, output, B, N, K, scale_rows, scale_cols, block_m, block_k);
        return cudaGetLastError();
    }
    if (nblock > 0 && (K % 16) == 0) {
        dim3 block(GEMV_THREADS);
        auto launch = [&](auto kern, int R) {
            dim3 grid((N + GEMV_ROWS * R - 1) / (GEMV_ROWS * R), B);
            kern<<<grid, block, 0, stream>>>(
                weight, scales, input, output, B, N, K, scale_rows, scale_cols,
                block_m, block_k);
        };
        switch (nblock) {
            case 2: launch(fp8_f32_block_gemv_nblock_kernel<2>, 2); break;
            case 4: launch(fp8_f32_block_gemv_nblock_kernel<4>, 4); break;
            case 8: launch(fp8_f32_block_gemv_nblock_kernel<8>, 8); break;
            case 16: launch(fp8_f32_block_gemv_nblock_kernel<16>, 16); break;
            default: launch(fp8_f32_block_gemv_nblock_kernel<4>, 4); break;
        }
        return cudaGetLastError();
    }
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    dim3 block(GEMV_THREADS);
    fp8_f32_block_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, B, N, K, scale_rows, scale_cols, block_m, block_k);
    return cudaGetLastError();
}

// ============================================================================
// NAIVE reference GPU kernel — correctness ORACLE.
// One thread per output row, simple k-loop, SAME decode + scale math, fp32
// accumulate, write bf16. B is assumed 1 (the bench drives B=1).
// ============================================================================
__global__ void gemv_reference_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    int block_m,
    int block_k)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float w = dsv4_decode_fp8_e4m3(weight[(int64_t)row * K + k])
            * fp8_f32_block_scale(scales, row, k, scale_rows, scale_cols, block_m, block_k);
        sum += w * __bfloat162float(input[k]);
    }
    output[row] = __float2bfloat16(sum);
}

// ============================================================================
// Host harness
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,        \
                         __FILE__, __LINE__, cudaGetErrorString(_err));        \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

struct Shape {
    const char* name;
    int N;
    int K;
};

int main(int argc, char** argv) {
    const double roofline_GBs = (argc > 1) ? atof(argv[1]) : 4000.0;  // default H20

    // L2-flush buffer. In real decode each weight is read COLD from HBM once per
    // token, but re-running ONE weight 200x leaves it L2-resident (Blackwell L2
    // caches even the 89MB weights -> measures L2 BW, not HBM, giving >100%
    // "roofline"). memset a >L2 buffer before each timed iter to evict the weight
    // and force a cold HBM read = the real decode access pattern.
    const size_t flush_bytes = 512ull << 20;  // 512 MB > any GPU L2
    unsigned char* d_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));

    // Qwen3.6-27B decode GEMV shapes, block_m = block_k = 128.
    // B (batch / spec-verify width) is argv[2] (default 1 = no-spec decode);
    // sweep e.g. `gemv_bench 4000 1`, `... 4000 2`, `... 4000 5` to see how the
    // GEMV scales with the spec-decode verify batch.
    const int B = (argc > 2) ? std::max(1, atoi(argv[2])) : 1;
    const int block_m = 128;
    const int block_k = 128;
    const Shape shapes[] = {
        {"qkv",    6144,  5120},
        {"o_proj", 5120,  6144},
        {"gate",   17408, 5120},
        {"down",   5120,  17408},
    };
    const int num_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    const int warmup_iters = 50;
    const int timed_iters = 200;

    std::printf("Roofline = %.1f GB/s\n", roofline_GBs);
    std::printf("B=%d block_m=%d block_k=%d  warmup=%d timed=%d\n\n",
                B, block_m, block_k, warmup_iters, timed_iters);

    double total_median_ms = 0.0;
    double sum_pct_roofline = 0.0;
    int passes = 0;

    for (int s = 0; s < num_shapes; ++s) {
        const int N = shapes[s].N;
        const int K = shapes[s].K;
        const int scale_rows = (N + 127) / 128;
        const int scale_cols = (K + 127) / 128;

        const size_t weight_elems = (size_t)N * K;          // FP8 weight bytes
        const size_t scale_elems = (size_t)scale_rows * scale_cols;

        // ---------------- Host init ----------------
        std::vector<uint8_t> h_weight(weight_elems);
        std::vector<float> h_scales(scale_elems);
        std::vector<__nv_bfloat16> h_input((size_t)K * B);

        // Deterministic LCG byte pattern mapped to a SAFE FP8 E4M3 value:
        //   b in [1, 120], then keep |code & 0x7f| in [1, 0x7e] (avoid 0/NaN/Inf),
        //   allow sign bit. NaN/Inf E4M3 code is 0x7f (mag bits) -> excluded.
        uint64_t rng = 0x9E3779B97F4A7C15ull ^ (uint64_t)(s + 1);
        for (size_t i = 0; i < weight_elems; ++i) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            uint8_t b = (uint8_t)(rng % 120 + 1);  // [1, 120]
            uint8_t mag = b & 0x7f;
            if (mag == 0) mag = 1;
            if (mag >= 0x7f) mag = 0x7e;            // clear NaN/Inf code 0x7f
            uint8_t sign = (uint8_t)((rng >> 7) & 0x80);  // pseudo-random sign
            h_weight[i] = (uint8_t)(sign | mag);
        }

        // Scales ~1.0 with +/-10% variation.
        for (size_t i = 0; i < scale_elems; ++i) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            double frac = (double)((rng >> 11) & 0xFFFFF) / (double)0xFFFFF;  // [0,1)
            h_scales[i] = (float)(1.0 + (frac - 0.5) * 0.2);  // [0.9, 1.1)
        }

        // Input bf16 pseudo-random in [-1, 1], B independent columns.
        for (size_t k = 0; k < (size_t)K * B; ++k) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            double frac = (double)((rng >> 11) & 0xFFFFF) / (double)0xFFFFF;  // [0,1)
            h_input[k] = __float2bfloat16((float)(frac * 2.0 - 1.0));
        }

        // ---------------- Device alloc + H2D ----------------
        uint8_t* d_weight = nullptr;
        float* d_scales = nullptr;
        __nv_bfloat16* d_input = nullptr;
        __nv_bfloat16* d_out = nullptr;
        __nv_bfloat16* d_ref = nullptr;
        CUDA_CHECK(cudaMalloc(&d_weight, weight_elems * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_scales, scale_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_input, (size_t)K * B * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * B * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref, (size_t)N * B * sizeof(__nv_bfloat16)));

        CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(),
                              weight_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(),
                              scale_elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                              (size_t)K * B * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        // ---------------- Reference (oracle) ----------------
        {
            const int threads = 128;
            const int blocks = (N + threads - 1) / threads;
            // Oracle per batch column (batch-major output [B, N], matching the
            // optimized kernel's output[batch_idx*N + row]).
            for (int b = 0; b < B; ++b) {
                gemv_reference_kernel<<<blocks, threads>>>(
                    d_weight, d_scales, d_input + (size_t)b * K, d_ref + (size_t)b * N,
                    N, K, scale_rows, scale_cols, block_m, block_k);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ---------------- Optimized kernel ----------------
        CUDA_CHECK(gemv_fp8_block_scaled_batch_cuda(
            d_weight, d_scales, d_input, d_out, B, N, K, scale_rows, scale_cols,
            block_m, block_k, /*stream=*/0));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---------------- D2H + correctness ----------------
        std::vector<__nv_bfloat16> h_out((size_t)N * B), h_ref((size_t)N * B);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, (size_t)N * B * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, (size_t)N * B * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));

        double dot = 0.0, na = 0.0, nb = 0.0, max_err = 0.0;
        for (int i = 0; i < N * B; ++i) {
            double a = (double)__bfloat162float(h_out[i]);
            double b = (double)__bfloat162float(h_ref[i]);
            dot += a * b;
            na += a * a;
            nb += b * b;
            double e = std::fabs(a - b);
            if (e > max_err) max_err = e;
        }
        double cosine = (na > 0.0 && nb > 0.0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
        bool pass = cosine > 0.9999;
        if (pass) ++passes;

        // ---------------- Timing (median of timed_iters) ----------------
        for (int i = 0; i < warmup_iters; ++i) {
            CUDA_CHECK(gemv_fp8_block_scaled_batch_cuda(
                d_weight, d_scales, d_input, d_out, B, N, K, scale_rows, scale_cols,
                block_m, block_k, /*stream=*/0));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        std::vector<float> times_ms(timed_iters);
        for (int i = 0; i < timed_iters; ++i) {
            // Evict the weight from L2 so the GEMV reads it cold from HBM (real
            // decode pattern). memset is BEFORE the start event -> not timed, but
            // serializes ahead of the GEMV on stream 0.
            CUDA_CHECK(cudaMemsetAsync(d_flush, (int)(i & 0xFF), flush_bytes, 0));
            CUDA_CHECK(cudaEventRecord(start, 0));
            CUDA_CHECK(gemv_fp8_block_scaled_batch_cuda(
                d_weight, d_scales, d_input, d_out, B, N, K, scale_rows, scale_cols,
                block_m, block_k, /*stream=*/0));
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            times_ms[i] = ms;
        }
        std::sort(times_ms.begin(), times_ms.end());
        double median_ms = times_ms[timed_iters / 2];

        // FP8 weight bytes = N*K. bandwidth = bytes / time.
        double bandwidth_GBs = ((double)N * K) / (median_ms * 1e-3) / 1e9;
        double pct_roofline = 100.0 * bandwidth_GBs / roofline_GBs;

        total_median_ms += median_ms;
        sum_pct_roofline += pct_roofline;

        std::printf(
            "SHAPE %-7s N=%-6d K=%-6d  %s  cosine=%.5f max_err=%.3g  "
            "median_ms=%.4f  GB/s=%.0f  %%roofline=%.1f\n",
            shapes[s].name, N, K, pass ? "PASS" : "FAIL",
            cosine, max_err, median_ms, bandwidth_GBs, pct_roofline);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_weight));
        CUDA_CHECK(cudaFree(d_scales));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_ref));
    }

    std::printf(
        "\nTOTAL  sum_median_ms=%.4f (per-step GEMV proxy)  avg_%%roofline=%.1f  "
        "correctness=%d/%d PASS\n",
        total_median_ms, sum_pct_roofline / num_shapes, passes, num_shapes);

    return (passes == num_shapes) ? 0 : 2;
}

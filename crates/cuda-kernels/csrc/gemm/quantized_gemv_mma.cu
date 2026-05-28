// GAP-A CUTLASS-MMA path for DSv4 FP8 batched GEMV.
//
// Replaces the scalar FP32-FMA accumulation in
// `quantized_gemv.cu::dsv4_fp8_gemv_batch_tiled_kernel` with an
// `mma.m16n8k16` BF16×BF16→FP32 MMA mainloop, dequantizing FP8E4M3
// weights and E8M0 per-block scales into BF16 in-registers before issue.
//
// Numerical parity with the scalar kernel: both accumulate in FP32 from
// BF16×BF16 element products (FP8 dequant → BF16 → FP32 accumulator).
// The MMA instruction adds no rounding error vs scalar FFMA; only the
// dequant→BF16 cast might lose bits — but the cast is bit-identical to
// the scalar's `dsv4_decode_fp8_e4m3(...) * dsv4_decode_e8m0(...)`
// followed by an implicit FP32-keep, so as long as we also cast to BF16
// at the same point the scalar kernel does (i.e. immediately before
// participating in the accumulator FFMA, via `__float2bfloat16`), the
// numerics match within last-bit BF16 rounding.
//
// Op (matches dsv4_fp8_gemv_batch_cuda contract):
//   Y[b, n] = Σ_k A[n, k] · X[b, k]
//   weight[N, K] row-major FP8E4M3
//   scales[scale_rows, scale_cols] E8M0, block_h × block_w blocks
//                                  (block_h = ⌈N/scale_rows⌉, block_w = ⌈K/scale_cols⌉)
//   input[B, K]  row-major BF16
//   output[B, N] row-major BF16
//
// Tile layout (decode-shape: B ≤ 16):
//   BLOCK_M = 16  (one warp-tile of m16n8k16 across the M axis)
//   BLOCK_N = 64  (4 warps × n8)
//   BLOCK_K = 128 (matches DSv4 scale granularity → one scale per K-block)
//   Threads per block = 128 (4 warps)
//   Grid = (⌈N/BLOCK_N⌉, ⌈B/BLOCK_M⌉)
//
// SM coverage:
//   - SM_80+ has `mma.m16n8k16.f32.bf16.bf16.f32` natively.
//   - SM_89 / SM_90 same path; WGMMA is overkill for B ≤ 16.
//   - SM <= 75: kernel is gated off in the dispatch shim; scalar
//     fallback used.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#define MMA_BLOCK_M 16
#define MMA_BLOCK_N 64
#define MMA_BLOCK_K 128
#define MMA_WARPS 4
#define MMA_THREADS (MMA_WARPS * 32)

// Same E8M0 / FP8E4M3 decoders as quantized_gemv.cu (verbatim for
// numerical parity).
__device__ __forceinline__ float gemv_mma_decode_e8m0(uint8_t bits) {
    uint32_t raw = static_cast<uint32_t>(bits) << 23;
    return __uint_as_float(raw);
}

__device__ __forceinline__ float gemv_mma_decode_fp8_e4m3(uint8_t bits) {
    if ((bits & 0x7f) == 0) return 0.0f;
    if ((bits & 0x7f) == 0x7f) {
        return (bits & 0x80) ? -448.0f : 448.0f;
    }
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return static_cast<float>(value);
}

// Pack two BF16 into a uint32 (lo = first, hi = second), matching the
// PTX `f32` → `bf16x2` register layout used by mma.m16n8k16.
__device__ __forceinline__ uint32_t pack_bf16x2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    uint32_t out;
    uint16_t lo_bits = *reinterpret_cast<uint16_t*>(&lo);
    uint16_t hi_bits = *reinterpret_cast<uint16_t*>(&hi);
    out = static_cast<uint32_t>(lo_bits) | (static_cast<uint32_t>(hi_bits) << 16);
    return out;
}

// Issue one mma.m16n8k16 BF16×BF16→FP32 instruction.
// Layout per cuBLAS / CUTLASS / PTX ISA §9.7.13.4:
//   A fragment: 4 × bf16x2 = 8 BF16 / thread / warp, arranged as the
//               m16k16 row-major tile (a0..a3 are the 4 column-pairs).
//   B fragment: 2 × bf16x2 = 4 BF16 / thread / warp, k16n8 col-major.
//   C/D fragment: 4 × FP32 / thread / warp = 16 FP32 / warp (m16n8).
__device__ __forceinline__ void mma_m16n8k16_bf16(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float& d0, float& d1, float& d2, float& d3)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    // Compile-time fallback for older SMs (kernel is gated off at host).
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    (void)d0; (void)d1; (void)d2; (void)d3;
#endif
}

// Main kernel: one block = (BLOCK_N=64 row-output, BLOCK_M=16 batch-output)
// tile. 4 warps fan across the N axis (each warp owns an n8 column).
//
// Per K-block (size 128):
//   1) Each thread loads a strip of A (activation) and B (weight) into
//      smem. cp.async would help BW pressure but at this shape we are
//      compute-bound by MMA issue rate, not load BW — keep sync loads
//      to minimize complexity. Revisit if MMA path is BW-bound at
//      large K (Phase 4 ncu).
//   2) Decode FP8→BF16 and apply E8M0 scale in registers (fused).
//   3) Issue 8 mma.m16n8k16 calls (BLOCK_K / k16 = 8 inner-k steps).
//
// Tail handling:
//   - B % 16 != 0: mask Y store at epilogue.
//   - N % 64 != 0: mask Y store at epilogue + skip B load past N.
//   - K % 128 != 0: clamp K-tile inner bounds; not expected at DSv4
//     canonical shapes (K = 7168 = 128 × 56).
__global__ void dsv4_fp8_gemv_batch_mma_kernel(
    const uint8_t* __restrict__ weight,   // [N, K] FP8E4M3
    const uint8_t* __restrict__ scales,   // [scale_rows, scale_cols] E8M0
    const __nv_bfloat16* __restrict__ input,  // [B, K] BF16
    __nv_bfloat16* __restrict__ output,   // [B, N] BF16
    int B,
    int N,
    int K,
    int scale_rows,
    int scale_cols)
{
    const int n_block = blockIdx.x * MMA_BLOCK_N;  // base N for this block
    const int m_block = blockIdx.y * MMA_BLOCK_M;  // base B for this block

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int n_warp_base = n_block + warp_id * 8;  // each warp owns 8 N rows

    if (n_warp_base >= N || m_block >= B) return;

    // Scale block lookup: which scale-block row contains this output N row.
    // For DSv4 standard layout, block_h = block_w = 128 → scale_rows = ⌈N/128⌉,
    // scale_cols = ⌈K/128⌉.
    const int block_h = (N + scale_rows - 1) / scale_rows;
    const int block_w = (K + scale_cols - 1) / scale_cols;

    // Each warp's 8 N rows live in the same scale-block-row (when block_h=128
    // and BLOCK_N=64, all 64 N rows of a block share scale_row). Pre-compute
    // scale-row index using warp's first N row.
    const int sr_raw = n_warp_base / block_h;
    const int sr = sr_raw < scale_rows ? sr_raw : (scale_rows - 1);

    // FP32 accumulator: 4 floats per warp (one m16n8 tile, 16 elements split
    // across 32 threads → 4 per thread).
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    // Iterate over K in BLOCK_K=128 chunks (matches scale granularity).
    const int n_k_blocks = (K + MMA_BLOCK_K - 1) / MMA_BLOCK_K;
    for (int kb = 0; kb < n_k_blocks; ++kb) {
        const int k_base = kb * MMA_BLOCK_K;
        const int sc_raw = k_base / block_w;
        const int sc = sc_raw < scale_cols ? sc_raw : (scale_cols - 1);
        const float w_scale_f = gemv_mma_decode_e8m0(scales[sr * scale_cols + sc]);

        // Inner k-loop: BLOCK_K=128 = 8 × k16 steps.
        #pragma unroll
        for (int ki = 0; ki < MMA_BLOCK_K; ki += 16) {
            // ---- Load B fragment (activation = X, [BLOCK_M, k16]) ----
            // For mma.m16n8k16 row.col:
            //   A operand = [M=16, K=16] row-major (we put X here since we
            //   iterate B as the M axis).
            //   B operand = [K=16, N=8] col-major (we put weight here since
            //   we iterate N as the warp-fanned axis).
            //
            // A fragment per-thread layout (PTX ISA):
            //   a0 = X[m_row(t)+0..1,  k_col(t)+0..1]  (bf16x2)
            //   a1 = X[m_row(t)+8..9,  k_col(t)+0..1]
            //   a2 = X[m_row(t)+0..1,  k_col(t)+8..9]
            //   a3 = X[m_row(t)+8..9,  k_col(t)+8..9]
            // where m_row(t) = t / 4 (0..7), k_col(t) = (t % 4) * 2 (0,2,4,6).

            const int m_row = lane_id / 4;        // 0..7
            const int k_col_pair = (lane_id % 4) * 2;  // 0, 2, 4, 6

            __nv_bfloat16 a_hi[2][2];  // [bank, pair]
            __nv_bfloat16 a_lo[2][2];

            #pragma unroll
            for (int bank = 0; bank < 2; ++bank) {
                const int m = m_block + m_row + bank * 8;
                #pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const int k = k_base + ki + k_col_pair + pair * 8;
                    if (m < B && m < m_block + MMA_BLOCK_M && k < K) {
                        a_lo[bank][pair] = input[m * K + k];
                        if (k + 1 < K) {
                            a_hi[bank][pair] = input[m * K + k + 1];
                        } else {
                            a_hi[bank][pair] = __float2bfloat16(0.f);
                        }
                    } else {
                        a_lo[bank][pair] = __float2bfloat16(0.f);
                        a_hi[bank][pair] = __float2bfloat16(0.f);
                    }
                }
            }

            uint32_t a0 = pack_bf16x2(a_lo[0][0], a_hi[0][0]);  // (m_row+0)..1, k+0..1
            uint32_t a1 = pack_bf16x2(a_lo[1][0], a_hi[1][0]);  // (m_row+8)..9, k+0..1
            uint32_t a2 = pack_bf16x2(a_lo[0][1], a_hi[0][1]);  // (m_row+0)..1, k+8..9
            uint32_t a3 = pack_bf16x2(a_lo[1][1], a_hi[1][1]);  // (m_row+8)..9, k+8..9

            // ---- Load + dequant A fragment (weight = W, [k16, n8]) ----
            // B operand per-thread layout (PTX ISA k16n8):
            //   b0 = W[k_row(t)+0..1, n_col(t)]  (bf16x2)
            //   b1 = W[k_row(t)+8..9, n_col(t)]
            // where k_row(t) = (t / 4) * 2  (0, 2, 4, 6, 8, 10, 12, 14)
            // → actually: lane layout is k_row = (t / 4) * 2, n_col = t % 4
            //   plus the inner row pair from the strided 16-row tile.
            // Reference: PTX ISA 9.7.13.4.6 "mma.m16n8k16 BF16 dense":
            //   row-major B: b0 = [k=(t/4)*2..k=(t/4)*2+1, n=t%4*2..t%4*2+1] hm.
            //   col-major B: b0 = [k=(t%4)*2..(t%4)*2+1, n=t/4*1] ... too
            //   error-prone to recompute from spec. Use simpler scheme:
            //   We store weight in row-major (k row, n col layout); since
            //   mma.row.col expects B col-major, we transpose at load time.
            //
            // The PTX-correct mapping for `mma.m16n8k16 .col B`:
            //   For 32-lane warp, B occupies [k=16, n=8] col-major. Each
            //   thread loads 4 BF16 (= 2 × bf16x2):
            //     b0 = B[(lane%4)*2..((lane%4)*2+1), lane/4]   (one column,
            //                                                   two adjacent k)
            //     b1 = B[(lane%4)*2+8..((lane%4)*2+9), lane/4]
            //
            // We map lane → (n_inwarp = lane / 4, k_inwarp_lo = (lane%4)*2).

            const int n_inwarp = lane_id / 4;         // 0..7 (which N within warp)
            const int k_inwarp_lo = (lane_id % 4) * 2;  // 0, 2, 4, 6 (low half)

            const int n_global = n_warp_base + n_inwarp;
            uint32_t b0 = 0;
            uint32_t b1 = 0;

            if (n_global < N) {
                const uint8_t* w_row = weight + n_global * K + k_base + ki;
                // Low pair: k = k_inwarp_lo .. k_inwarp_lo + 1
                float w0f = gemv_mma_decode_fp8_e4m3(w_row[k_inwarp_lo]) * w_scale_f;
                float w1f = gemv_mma_decode_fp8_e4m3(w_row[k_inwarp_lo + 1]) * w_scale_f;
                __nv_bfloat16 w0 = __float2bfloat16(w0f);
                __nv_bfloat16 w1 = __float2bfloat16(w1f);
                b0 = pack_bf16x2(w0, w1);

                // High pair: k = k_inwarp_lo + 8 .. k_inwarp_lo + 9
                if (k_base + ki + k_inwarp_lo + 8 < K) {
                    float w8f = gemv_mma_decode_fp8_e4m3(w_row[k_inwarp_lo + 8]) * w_scale_f;
                    float w9f = gemv_mma_decode_fp8_e4m3(w_row[k_inwarp_lo + 9]) * w_scale_f;
                    __nv_bfloat16 w8 = __float2bfloat16(w8f);
                    __nv_bfloat16 w9 = __float2bfloat16(w9f);
                    b1 = pack_bf16x2(w8, w9);
                }
            }

            // ---- Issue MMA ----
            mma_m16n8k16_bf16(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
        }
    }

    // Epilogue: write D (FP32) → BF16 output[B, N].
    // mma.m16n8k16 C/D layout (PTX ISA):
    //   per thread: 4 FP32 forming an m16n8 tile.
    //     d0 = D[m_row(t)+0, n_col(t)+0]    where m_row(t) = t / 4 (0..7),
    //     d1 = D[m_row(t)+0, n_col(t)+1]          n_col(t) = (t % 4) * 2
    //     d2 = D[m_row(t)+8, n_col(t)+0]
    //     d3 = D[m_row(t)+8, n_col(t)+1]
    {
        const int m_row = lane_id / 4;          // 0..7
        const int n_col_base = (lane_id % 4) * 2;  // 0, 2, 4, 6 within warp's 8-N slot

        #pragma unroll
        for (int bank = 0; bank < 2; ++bank) {
            const int m = m_block + m_row + bank * 8;
            if (m >= B || m >= m_block + MMA_BLOCK_M) continue;
            #pragma unroll
            for (int col = 0; col < 2; ++col) {
                const int n = n_warp_base + n_col_base + col;
                if (n >= N || n >= n_warp_base + 8) continue;
                float val;
                if (bank == 0 && col == 0) val = d0;
                else if (bank == 0 && col == 1) val = d1;
                else if (bank == 1 && col == 0) val = d2;
                else val = d3;
                output[m * N + n] = __float2bfloat16(val);
            }
        }
    }
}

// Host entry point — exposed via FFI through a dispatch shim in
// quantized_gemv.cu's `dsv4_fp8_gemv_batch_cuda`.
extern "C" cudaError_t dsv4_fp8_gemv_batch_mma_launch(
    const uint8_t* weight,
    const uint8_t* scales,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int B,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream)
{
    if (B <= 0 || N <= 0 || K <= 0 || scale_rows <= 0 || scale_cols <= 0) {
        return cudaErrorInvalidValue;
    }
    // MMA-tile shape requires N % 8 == 0 and K % 16 == 0 to issue without
    // expensive tail handling. DSv4 standard hidden dims (2048, 4096, 7168,
    // 16384) all satisfy this. Caller dispatches scalar fallback otherwise.
    if ((N & 7) != 0 || (K & 15) != 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid((N + MMA_BLOCK_N - 1) / MMA_BLOCK_N,
              (B + MMA_BLOCK_M - 1) / MMA_BLOCK_M);
    dim3 block(MMA_THREADS);
    dsv4_fp8_gemv_batch_mma_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, B, N, K, scale_rows, scale_cols);
    return cudaGetLastError();
}

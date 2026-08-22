extern "C" __device__ __forceinline__ float arle_fp8_e4m3_to_f32(unsigned char bits) {
    const float sign = (bits & 0x80) ? -1.0f : 1.0f;
    const int exp = (bits >> 3) & 0x0f;
    const int mant = bits & 0x07;
    if (exp == 0) {
        if (mant == 0) return sign * 0.0f;
        return sign * ldexpf(static_cast<float>(mant) * 0.125f, -6);
    }
    if (exp == 0x0f && mant == 0x07) {
        return nanf("");
    }
    return sign * ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
}

// Dequantize an FP8 E4M3 block-scaled weight to BF16 bits. GEMMs then ride the
// tensor-core cuBLAS BF16 path; this stays memory-bound (~0.1 ms per 27B weight).
extern "C" __global__ void fp8_block_scaled_to_bf16(
    const unsigned char* __restrict__ weight, // [rows, cols] FP8 E4M3
    const float* __restrict__ scales,         // [ceil(rows/BM), ceil(cols/BK)]
    unsigned short* __restrict__ out,         // [rows, cols] BF16 bits
    int total,
    int cols,
    int block_m,
    int block_k,
    int scale_cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int row = idx / cols;
    const int col = idx - row * cols;
    const float scale = scales[(row / block_m) * scale_cols + (col / block_k)];
    const float w = arle_fp8_e4m3_to_f32(weight[idx]) * scale;

    const unsigned int bits = __float_as_uint(w);
    const unsigned int lsb = (bits >> 16) & 1u;
    out[idx] = static_cast<unsigned short>((bits + 0x7fffu + lsb) >> 16);
}

__device__ __constant__ float ARLE_FP4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Dequantize an NVFP4 weight to BF16 bits: packed E2M1 [rows, cols/2] (low
// nibble = even col) x per-row per-group FP8 E4M3 scale [rows, cols/group_size]
// x one F32 global scale. Same role as the FP8 kernel above — sm_90 has no FP4
// tensor cores, so the frozen 4-bit base is dequantized once per step and the
// GEMM rides cuBLAS BF16.
extern "C" __global__ void fp4_e2m1_group_to_bf16(
    const unsigned char* __restrict__ weight,       // [rows, cols/2] packed E2M1
    const unsigned char* __restrict__ scales,       // [rows, scale_cols] E4M3
    const float* __restrict__ global_scale,         // [1]
    unsigned short* __restrict__ out,               // [rows, cols] BF16 bits
    int total,
    int cols,
    int group_size,
    int scale_cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int row = idx / cols;
    const int col = idx - row * cols;
    // scale_cols == cols / group_size is validated at upload; no clamp needed.
    const float scale =
        arle_fp8_e4m3_to_f32(scales[row * scale_cols + col / group_size]) * global_scale[0];
    const unsigned char byte = weight[row * (cols / 2) + (col >> 1)];
    const unsigned char nib = (col & 1) ? (byte >> 4) : (byte & 0x0f);
    const float w = ARLE_FP4_E2M1_LUT[nib] * scale;

    const unsigned int bits = __float_as_uint(w);
    const unsigned int lsb = (bits >> 16) & 1u;
    out[idx] = static_cast<unsigned short>((bits + 0x7fffu + lsb) >> 16);
}

// Dequantize an NVFP4 weight that already carries the Marlin tensor-core layout.
// The serving engine repacks its base once and releases the group-layout bytes,
// so a student sharing that base reads the Marlin buffer directly instead of
// keeping a second copy.
//
// Layout (device_matrix.rs `repack_for_marlin_fp4` + the vendored
// `gptq_marlin_repack_kernel`, num_bits=4): tiles of k=16 x n=64 walk k-major,
// 128 u32 each. Inside a tile, u32 `w` belongs to th_id=w/4, warp=w%4, and its
// nibble `i` holds source element `pack_idx[i]` of that thread's 8, which maps
// to k = tc_row + tc_offsets[j%4] and n = warp*16 + th_id/4 + (j<4 ? 0 : 8).
// The S0E5M3 group scales sit in the tail of the same allocation, permuted 8x8
// inside each 64-run then swapped [0,2,1,3] per quad. `global_scale` already
// carries the repack's 2^119 dequant bias and its scale_factor divisor, so
// dividing it back out here reproduces exactly what the Marlin GEMM computes —
// including the group scales the repack flushed to zero.
extern "C" __global__ void marlin_fp4_to_bf16(
    const unsigned int* __restrict__ marlin,   // [k*n/8] u32, then the scale tail
    const unsigned char* __restrict__ scales,  // [k/16 * n] S0E5M3 bytes
    float global_scale,                        // bf16-rounded, includes 2^119
    unsigned short* __restrict__ out,          // [n, k] BF16 bits
    int words,                                 // k*n/8
    int size_k,
    int size_n)
{
    const int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (w_idx >= words) return;

    const int n_tiles = size_n / 64;
    const int tile = w_idx / 128;
    const int w = w_idx - tile * 128;
    const int k_tile = tile / n_tiles;
    const int n_tile = tile - k_tile * n_tiles;

    const int th_id = w >> 2;
    const int warp_id = w & 3;
    const int tc_col = th_id >> 2;
    const int tc_row = (th_id & 3) * 2;
    const int cur_n = warp_id * 16 + tc_col;

    const int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    const int tc_offsets[4] = {0, 1, 8, 9};
    const unsigned int packed = marlin[w_idx];
    const float gscale = ldexpf(global_scale, -119);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int j = pack_idx[i];
        const int k = k_tile * 16 + tc_row + tc_offsets[j & 3];
        const int n = n_tile * 64 + cur_n + ((j < 4) ? 0 : 8);
        const int group = k >> 4;
        // Inverse of the repack's scale permutation, in the order the repack
        // applied it: sperm[b*64 + o] = sflat[b*64 + (o%8)*8 + o/8] (an 8x8
        // transpose, self-inverse), and only THEN each quad swapped 1<->2. So
        // transpose first, then follow the swap — doing it the other way round
        // misplaces 3 of every 4 scales.
        const int flat = group * size_n + n;
        const int base = flat & ~63;
        const int o = flat - base;
        const int t = (o % 8) * 8 + (o / 8);
        const int src = base + ((t & 3) == 1 ? t + 1 : ((t & 3) == 2 ? t - 1 : t));
        // S0E5M3 byte -> f16 (byte << 7) decoded inline: NVRTC compiles these
        // kernels without cuda_fp16.h. Then drop the repack's 2^7 lift.
        const unsigned int sb = scales[src];
        const int sexp = (sb >> 3) & 0x1f;
        const float smant = static_cast<float>(sb & 0x7) * 0.125f;
        const float s = (sexp == 0 ? ldexpf(smant, -14) : ldexpf(1.0f + smant, sexp - 15))
            * 0.0078125f;
        const float value = ARLE_FP4_E2M1_LUT[(packed >> (i * 4)) & 0xfu] * s * gscale;

        const unsigned int bits = __float_as_uint(value);
        const unsigned int lsb = (bits >> 16) & 1u;
        out[static_cast<long long>(n) * size_k + k] =
            static_cast<unsigned short>((bits + 0x7fffu + lsb) >> 16);
    }
}

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
    int group = col / group_size;
    if (group > scale_cols - 1) group = scale_cols - 1;

    const float scale =
        arle_fp8_e4m3_to_f32(scales[row * scale_cols + group]) * global_scale[0];
    const unsigned char byte = weight[row * (cols / 2) + (col >> 1)];
    const unsigned char nib = (col & 1) ? (byte >> 4) : (byte & 0x0f);
    const float w = ARLE_FP4_E2M1_LUT[nib] * scale;

    const unsigned int bits = __float_as_uint(w);
    const unsigned int lsb = (bits >> 16) & 1u;
    out[idx] = static_cast<unsigned short>((bits + 0x7fffu + lsb) >> 16);
}

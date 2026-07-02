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

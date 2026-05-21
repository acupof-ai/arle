extern "C" __global__ void bf16_bits_to_f32(
    const unsigned short* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    unsigned int bits = static_cast<unsigned int>(input[idx]) << 16;
    output[idx] = __uint_as_float(bits);
}

extern "C" __global__ void f32_to_bf16_bits(
    const float* __restrict__ input,
    unsigned short* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    unsigned int bits = __float_as_uint(input[idx]);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounding_bias = 0x7fffu + lsb;
    output[idx] = static_cast<unsigned short>((bits + rounding_bias) >> 16);
}

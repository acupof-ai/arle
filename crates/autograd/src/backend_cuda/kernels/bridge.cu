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

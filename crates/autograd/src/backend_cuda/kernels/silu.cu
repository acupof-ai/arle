extern "C" __global__ void silu_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float s = 1.0f / (1.0f + __expf(-v));
        out[i] = v * s;
    }
}

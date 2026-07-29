extern "C" __global__ void add_into_f32(
    float* out,
    const float* dest,
    const float* src,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = dest[i] + src[i];
    }
}

extern "C" __global__ void accumulate_into_f32(
    float* dest,
    const float* src,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        dest[i] += src[i];
    }
}

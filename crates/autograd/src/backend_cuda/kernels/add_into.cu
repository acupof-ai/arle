extern "C" __global__ void add_into_f32(
    T* out,
    const T* dest,
    const T* src,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<T>(static_cast<float>(dest[i]) + static_cast<float>(src[i]));
    }
}

extern "C" __global__ void accumulate_into_f32(
    T* dest,
    const T* src,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        dest[i] = static_cast<T>(static_cast<float>(dest[i]) + static_cast<float>(src[i]));
    }
}

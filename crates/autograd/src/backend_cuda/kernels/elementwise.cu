extern "C" __global__ void add_f32(
    T* out,
    const T* a,
    const T* b,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<T>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
    }
}

extern "C" __global__ void mul_f32(
    T* out,
    const T* a,
    const T* b,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<T>(static_cast<float>(a[i]) * static_cast<float>(b[i]));
    }
}

extern "C" __global__ void mul_scalar_f32(
    T* out,
    const T* a,
    float s,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<T>(static_cast<float>(a[i]) * s);
    }
}

extern "C" __global__ void sigmoid_f32(
    T* out,
    const T* a,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<T>(1.0f / (1.0f + __expf(-static_cast<float>(a[i]))));
    }
}

extern "C" __global__ void gelu_f32(
    T* out,
    const T* a,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = static_cast<float>(a[i]);
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + (0.044715f * x3));
        out[i] = static_cast<T>(0.5f * x * (1.0f + tanhf(inner)));
    }
}

extern "C" __global__ void exp_f32(
    T* out,
    const T* a,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<T>(expf(static_cast<float>(a[i])));
    }
}

extern "C" __global__ void neg_f32(
    T* out,
    const T* a,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<T>(-static_cast<float>(a[i]));
    }
}

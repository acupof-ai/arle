extern "C" __global__ void silu_backward_f32(
    T* __restrict__ grad_input,
    const T* __restrict__ upstream,
    const T* __restrict__ x,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = static_cast<float>(x[i]);
        float s = 1.0f / (1.0f + __expf(-v));
        float deriv = s + (v * s * (1.0f - s));
        grad_input[i] = static_cast<T>(static_cast<float>(upstream[i]) * deriv);
    }
}

extern "C" __global__ void gelu_backward_f32(
    T* __restrict__ grad_input,
    const T* __restrict__ upstream,
    const T* __restrict__ x,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = static_cast<float>(x[i]);
        float erf_term = erff(v * 0.70710677f);
        float exp_term = __expf(-0.5f * v * v);
        float deriv = 0.5f * (1.0f + erf_term) + (v * 0.3989423f * exp_term);
        grad_input[i] = static_cast<T>(static_cast<float>(upstream[i]) * deriv);
    }
}

extern "C" __global__ void sigmoid_backward_f32(
    T* __restrict__ grad_input,
    const T* __restrict__ upstream,
    const T* __restrict__ y,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float yv = static_cast<float>(y[i]);
        grad_input[i] = static_cast<T>(static_cast<float>(upstream[i]) * yv * (1.0f - yv));
    }
}

// sign(0) = 0 — the L1 subgradient, so a zero residual contributes nothing.
extern "C" __global__ void abs_backward_f32(
    T* __restrict__ grad_input,
    const T* __restrict__ upstream,
    const T* __restrict__ x,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float xv = static_cast<float>(x[i]);
        float sign = (xv > 0.0f) ? 1.0f : ((xv < 0.0f) ? -1.0f : 0.0f);
        grad_input[i] = static_cast<T>(static_cast<float>(upstream[i]) * sign);
    }
}

extern "C" __global__ void exp_backward_f32(
    T* __restrict__ grad_input,
    const T* __restrict__ upstream,
    const T* __restrict__ y,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_input[i] =
            static_cast<T>(static_cast<float>(upstream[i]) * static_cast<float>(y[i]));
    }
}

extern "C" __global__ void silu_backward_f32(
    float* __restrict__ grad_input,
    const float* __restrict__ upstream,
    const float* __restrict__ x,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float s = 1.0f / (1.0f + __expf(-v));
        float deriv = s + (v * s * (1.0f - s));
        grad_input[i] = upstream[i] * deriv;
    }
}

extern "C" __global__ void gelu_backward_f32(
    float* __restrict__ grad_input,
    const float* __restrict__ upstream,
    const float* __restrict__ x,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float erf_term = erff(v * 0.70710677f);
        float exp_term = __expf(-0.5f * v * v);
        float deriv = 0.5f * (1.0f + erf_term) + (v * 0.3989423f * exp_term);
        grad_input[i] = upstream[i] * deriv;
    }
}

extern "C" __global__ void sigmoid_backward_f32(
    float* __restrict__ grad_input,
    const float* __restrict__ upstream,
    const float* __restrict__ y,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float yv = y[i];
        grad_input[i] = upstream[i] * yv * (1.0f - yv);
    }
}

extern "C" __global__ void exp_backward_f32(
    float* __restrict__ grad_input,
    const float* __restrict__ upstream,
    const float* __restrict__ y,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_input[i] = upstream[i] * y[i];
    }
}

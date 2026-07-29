extern "C" __global__ void mul_backward_lhs_f32(
    float* __restrict__ grad_a,
    const float* __restrict__ upstream,
    const float* __restrict__ b,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_a[i] = upstream[i] * b[i];
    }
}

extern "C" __global__ void mul_backward_rhs_f32(
    float* __restrict__ grad_b,
    const float* __restrict__ upstream,
    const float* __restrict__ a,
    unsigned long long n
) {
    unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_b[i] = upstream[i] * a[i];
    }
}

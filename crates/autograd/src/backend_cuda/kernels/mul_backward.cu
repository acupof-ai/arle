// Wave 2.1: elementwise mul backward.
//   grad_a[i] = upstream[i] * b[i]
//   grad_b[i] = upstream[i] * a[i]
//
// Two separate kernel symbols so the dispatch site can short-circuit one
// side when `need_grad_a` / `need_grad_b` is false (mirrors
// `matmul_backward_device`). Block=256 via the shared `launch_1d` helper.

extern "C" __global__ void mul_backward_lhs_f32(
    float* __restrict__ grad_a,
    const float* __restrict__ upstream,
    const float* __restrict__ b,
    int n
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        grad_a[i] = upstream[i] * b[i];
    }
}

extern "C" __global__ void mul_backward_rhs_f32(
    float* __restrict__ grad_b,
    const float* __restrict__ upstream,
    const float* __restrict__ a,
    int n
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        grad_b[i] = upstream[i] * a[i];
    }
}

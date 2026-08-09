// Device-resident mean backward: keeps the CE-loss backward chain
// (mul_scalar -> gather_last_dim -> log_softmax_last_axis -> matmul) on-device —
// no DtoH on the single upstream scalar. inv_n = 1/N is precomputed host-side
// so the kernel stays a pure scalar broadcast.
extern "C" __global__ void mean_backward_f32(
    float* d_input,
    const float* upstream_scalar,
    float inv_n,
    int n
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        // All threads read the same scalar; L1 cache makes this free.
        float g = upstream_scalar[0];
        d_input[i] = g * inv_n;
    }
}

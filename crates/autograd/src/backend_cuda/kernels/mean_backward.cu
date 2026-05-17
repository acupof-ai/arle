// P3: device-resident backward for `mean(x)`. The forward reduces
// `product(input_shape) = N` elements to a rank-0 scalar; the backward
// broadcasts a single `upstream / N` value across `N` slots of the
// `d_input` buffer.
//
// `upstream` is a device pointer to a single fp32 scalar (the rank-0
// gradient produced by the downstream chain — typically `mul_scalar`'s
// device backward). `inv_n` is `1.0f / N` pre-computed host-side so the
// kernel itself stays a pure scalar broadcast.
//
// Wires the CE-loss backward chain so the upstream into
// `mul_scalar_backward` (and from there `gather_last_dim_backward`,
// `log_softmax_last_axis_backward`, `matmul_backward_device`) STAYS
// device-resident — no DtoH on the single scalar.
extern "C" __global__ void mean_backward_f32(
    float* d_input,
    const float* upstream_scalar,
    float inv_n,
    int n
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        // Block-broadcasted read: every thread fetches the same scalar.
        // The L1 hit on subsequent iterations of the grid keeps this
        // free.
        float g = upstream_scalar[0];
        d_input[i] = g * inv_n;
    }
}

// Fresh output buffer (not in-place) so the upstream handle remains valid
// for any parallel consumer on the tape.
//
// Unblocks the CE-loss backward chain from the first host op
// (`d_loss → mul_scalar_backward → mean_backward`) down through
// `gather_last_dim_backward`, `log_softmax_last_axis_backward`, and
// `matmul_backward_device` — all of which already have device overrides but
// were demoted to host by this upstream poison source.
extern "C" __global__ void mul_scalar_backward_f32(
    float* out,
    const float* upstream,
    float k,
    int n
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        out[i] = upstream[i] * k;
    }
}

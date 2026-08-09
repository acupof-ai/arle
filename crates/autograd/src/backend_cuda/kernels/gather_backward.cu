// Output is pre-zeroed by the caller; per-row indices are independent so no
// atomics are needed. Out-of-range ids are skipped to match the CPU and
// scatter_add fallback paths.

extern "C" __global__ void gather_last_dim_backward_f32(
    float* __restrict__ grad_input,
    const float* __restrict__ upstream,
    const int* __restrict__ ids,
    int prefix_rows,
    int vocab
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= prefix_rows) {
        return;
    }
    int id = ids[row];
    if (id < 0 || id >= vocab) {
        return;
    }
    grad_input[row * vocab + id] = upstream[row];
}

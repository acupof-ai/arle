// Reuses the saved forward log_softmax output as the softmax probability via
// `__expf` instead of recomputing softmax from the input — same identity as
// `cpu_log_softmax_backward`.
//
// Launch matches `softmax_last_axis_f32` (grid = (rows, 1, 1), blockDim.x = 256)
// so the kernel-cache `launch_rows` helper with `SHARED = BLOCK * sizeof(float)`
// reuses cleanly.

extern "C" __global__ void log_softmax_last_axis_backward_f32(
    T* __restrict__ grad_input,
    const T* __restrict__ upstream,
    const T* __restrict__ log_softmax_output,
    int cols
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block = blockDim.x;
    const T* row_up = upstream + row * cols;
    const T* row_out = log_softmax_output + row * cols;
    T* row_grad = grad_input + row * cols;

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block) {
        local_sum += static_cast<float>(row_up[i]);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) smem[tid] += smem[tid + step];
        __syncthreads();
    }
    float sum_grad = smem[0];

    for (int i = tid; i < cols; i += block) {
        row_grad[i] = static_cast<T>(static_cast<float>(row_up[i]) - __expf(static_cast<float>(row_out[i])) * sum_grad);
    }
}

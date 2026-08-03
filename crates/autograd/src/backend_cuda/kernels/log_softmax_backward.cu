// Wave 1 (post-M5.3b nsys attribution): row-wise log_softmax backward.
// Caller passes the saved forward output (log_softmax(x)) so we reuse it
// as the softmax probability via `__expf` rather than recomputing softmax
// from the input — same identity as `cpu_log_softmax_backward`:
//
//   grad_input[i, j] = upstream[i, j]
//                    - __expf(log_softmax_output[i, j]) * sum_j(upstream[i, j])
//
// Each block handles one row; threads cooperate on the per-row sum via
// shared-memory tree reduction. Grid is launched with (rows, 1, 1) and
// blockDim.x = 256 — matches `softmax_last_axis_f32` so the kernel-cache
// `launch_rows` helper with `SHARED = BLOCK * sizeof(float)` reuses cleanly.

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

    // Phase 1: per-row sum of upstream.
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

    // Phase 2: per-element grad = upstream - exp(log_softmax_output) * sum_grad.
    for (int i = tid; i < cols; i += block) {
        row_grad[i] = static_cast<T>(static_cast<float>(row_up[i]) - __expf(static_cast<float>(row_out[i])) * sum_grad);
    }
}

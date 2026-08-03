// Row-wise RMSNorm over the last axis. For each row:
//   mean_sq = sum(x[i]^2) / cols
//   rms = sqrt(mean_sq + eps)
//   out[i] = (x[i] / rms) * weight[i]
//
// One block per row; threads cooperate on the sum-of-squares reduction via
// shared memory. Grid: (rows, 1, 1). Block: (256, 1, 1). Shared: block * f32.

extern "C" __global__ void rms_norm_f32(
    T* __restrict__ out,
    const T* __restrict__ x,
    const float* __restrict__ weight,
    int cols,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block = blockDim.x;
    const T* row_x = x + row * cols;
    T* row_out = out + row * cols;

    // Phase 1: per-thread local sum of squares.
    float local_sq = 0.0f;
    for (int i = tid; i < cols; i += block) {
        float v = static_cast<float>(row_x[i]);
        local_sq += v * v;
    }
    smem[tid] = local_sq;
    __syncthreads();

    // Phase 2: tree reduction.
    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf((smem[0] / (float)cols) + eps);

    // Phase 3: normalize and scale by weight.
    for (int i = tid; i < cols; i += block) {
        row_out[i] = static_cast<T>(static_cast<float>(row_x[i]) * inv_rms * weight[i]);
    }
}

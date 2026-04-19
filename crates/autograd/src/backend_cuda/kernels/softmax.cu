// Numerically-stable row-wise softmax / log_softmax over the last axis.
// Each block handles one row; threads cooperate on the max + sum reductions.
// Caller passes `cols` (last-dim size) and `rows` (product of leading dims);
// grid is launched with (rows, 1, 1) and block_dim 256.

extern "C" __global__ void softmax_last_axis_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    int cols
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block = blockDim.x;
    const float* row_x = x + row * cols;
    float* row_out = out + row * cols;

    // Phase 1: row max.
    float local_max = __int_as_float(0xFF800000);
    for (int i = tid; i < cols; i += block) {
        float v = row_x[i];
        if (v > local_max) local_max = v;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) {
            float other = smem[tid + step];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }
    float row_max = smem[0];

    // Phase 2: sum of exp(x - max).
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block) {
        local_sum += __expf(row_x[i] - row_max);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) smem[tid] += smem[tid + step];
        __syncthreads();
    }
    float row_sum = smem[0];
    float inv = 1.0f / row_sum;

    // Phase 3: normalized output.
    for (int i = tid; i < cols; i += block) {
        row_out[i] = __expf(row_x[i] - row_max) * inv;
    }
}

extern "C" __global__ void log_softmax_last_axis_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    int cols
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block = blockDim.x;
    const float* row_x = x + row * cols;
    float* row_out = out + row * cols;

    float local_max = __int_as_float(0xFF800000);
    for (int i = tid; i < cols; i += block) {
        float v = row_x[i];
        if (v > local_max) local_max = v;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) {
            float other = smem[tid + step];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }
    float row_max = smem[0];

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block) {
        local_sum += __expf(row_x[i] - row_max);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) smem[tid] += smem[tid + step];
        __syncthreads();
    }
    float log_denom = logf(smem[0]);

    for (int i = tid; i < cols; i += block) {
        row_out[i] = (row_x[i] - row_max) - log_denom;
    }
}

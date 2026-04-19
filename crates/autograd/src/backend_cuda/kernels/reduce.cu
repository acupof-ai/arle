// Row-wise reductions over the last axis. For each row:
//   sum_last_axis_f32:  out[row] = sum(x[row, :])
//   mean_last_axis_f32: out[row] = sum(x[row, :]) / cols
//
// One block per row; threads cooperate via shared memory. Grid=(rows, 1, 1),
// block=(256, 1, 1), shared=block * f32. Output is a contiguous rank-(n-1)
// tensor of length `rows`.

extern "C" __global__ void sum_last_axis_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    int cols
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block = blockDim.x;
    const float* row_x = x + row * cols;

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block) {
        local_sum += row_x[i];
    }
    smem[tid] = local_sum;
    __syncthreads();

    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[row] = smem[0];
    }
}

extern "C" __global__ void mean_last_axis_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    int cols
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block = blockDim.x;
    const float* row_x = x + row * cols;

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block) {
        local_sum += row_x[i];
    }
    smem[tid] = local_sum;
    __syncthreads();

    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[row] = smem[0] / (float)cols;
    }
}

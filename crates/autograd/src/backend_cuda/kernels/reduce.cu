// Row-wise reductions over the last axis. For each row:
//   sum_last_axis_f32:  out[row] = sum(x[row, :])
//   mean_last_axis_f32: out[row] = sum(x[row, :]) / cols
//
// One block per row; threads cooperate via shared memory. Grid=(rows, 1, 1),
// block=(256, 1, 1), shared=block * f32. Output is a contiguous rank-(n-1)
// tensor of length `rows`.

extern "C" __global__ void sum_squares_partial_f32(
    double* __restrict__ partial,
    const float* __restrict__ x,
    int n
) {
    extern __shared__ double smem64[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double local = 0.0;
    if (idx < n) {
        double value = (double)x[idx];
        local = value * value;
    }
    smem64[tid] = local;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem64[tid] += smem64[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = smem64[0];
    }
}

extern "C" __global__ void grad_clip_sumsq_f32(
    double* __restrict__ partial,
    const unsigned long long* __restrict__ grad_ptrs,
    const int* __restrict__ grad_sizes,
    const int* __restrict__ chunk_offsets,
    int num_grads,
    int chunk_elems
) {
    extern __shared__ double smem64[];
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int lo = 0;
    int hi = num_grads;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (chunk_offsets[mid] <= block_idx) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    int grad_idx = lo;
    int local_chunk = block_idx - chunk_offsets[grad_idx];
    int size = grad_sizes[grad_idx];
    const float* grad = (const float*)grad_ptrs[grad_idx];
    int start = local_chunk * chunk_elems;
    int end = start + chunk_elems;
    if (end > size) end = size;

    double local = 0.0;
    for (int idx = start + tid; idx < end; idx += blockDim.x) {
        double value = (double)grad[idx];
        local += value * value;
    }
    smem64[tid] = local;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem64[tid] += smem64[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[block_idx] = smem64[0];
    }
}

extern "C" __global__ void grad_clip_scale_f32(
    unsigned long long* __restrict__ out_ptrs,
    const unsigned long long* __restrict__ grad_ptrs,
    const int* __restrict__ grad_sizes,
    const int* __restrict__ chunk_offsets,
    float scale,
    int num_grads,
    int chunk_elems
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int lo = 0;
    int hi = num_grads;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (chunk_offsets[mid] <= block_idx) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    int grad_idx = lo;
    int local_chunk = block_idx - chunk_offsets[grad_idx];
    int size = grad_sizes[grad_idx];
    const float* grad = (const float*)grad_ptrs[grad_idx];
    float* out = (float*)out_ptrs[grad_idx];
    int start = local_chunk * chunk_elems;
    int end = start + chunk_elems;
    if (end > size) end = size;

    for (int idx = start + tid; idx < end; idx += blockDim.x) {
        out[idx] = grad[idx] * scale;
    }
}

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

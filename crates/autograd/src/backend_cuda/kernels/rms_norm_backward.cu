// Wave 2.1: row-wise RMSNorm backward. Mirrors `cpu_rmsnorm_backward`
// exactly so the parity gate stays bit-identical modulo `__expf` /
// reduction-order ULP.
//
// `__syncthreads()` discipline: full block-wide barriers around every
// shared-mem read; tree reduction is the canonical block / 2 form. `eps`
// is consumed only by the first kernel so the forward and backward agree
// bit-for-bit.

extern "C" __global__ void rms_norm_inv_rms_f32(
    float* __restrict__ inv_rms,
    const T* __restrict__ x,
    int cols,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block = blockDim.x;
    const T* row_x = x + row * cols;

    float local_sq = 0.0f;
    for (int i = tid; i < cols; i += block) {
        float v = static_cast<float>(row_x[i]);
        local_sq += v * v;
    }
    smem[tid] = local_sq;
    __syncthreads();
    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        inv_rms[row] = rsqrtf((smem[0] / (float)cols) + eps);
    }
}

extern "C" __global__ void rms_norm_backward_x_f32(
    T* __restrict__ grad_x,
    const T* __restrict__ upstream,
    const T* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ inv_rms,
    int cols
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block = blockDim.x;
    const T* row_x = x + row * cols;
    const T* row_up = upstream + row * cols;
    T* row_grad = grad_x + row * cols;

    float local_dot = 0.0f;
    for (int i = tid; i < cols; i += block) {
        local_dot += static_cast<float>(row_up[i]) * weight[i] * static_cast<float>(row_x[i]);
    }
    smem[tid] = local_dot;
    __syncthreads();
    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    float inv = inv_rms[row];
    float correction = inv * inv * smem[0] / (float)cols;

    for (int i = tid; i < cols; i += block) {
        row_grad[i] = static_cast<T>((inv * static_cast<float>(row_up[i]) * weight[i]) - (static_cast<float>(row_x[i]) * inv * correction));
    }
}

extern "C" __global__ void rms_norm_backward_w_f32(
    float* __restrict__ grad_w,
    const T* __restrict__ upstream,
    const T* __restrict__ x,
    const float* __restrict__ inv_rms,
    int rows,
    int cols
) {
    extern __shared__ float smem[];
    int col = blockIdx.x;
    int tid = threadIdx.x;
    int block = blockDim.x;

    float local_sum = 0.0f;
    for (int r = tid; r < rows; r += block) {
        local_sum += static_cast<float>(upstream[r * cols + col]) * static_cast<float>(x[r * cols + col]) * inv_rms[r];
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
        grad_w[col] = smem[0];
    }
}

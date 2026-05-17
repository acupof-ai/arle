// Wave 2.1: row-wise RMSNorm backward. Mirrors `cpu_rmsnorm_backward`
// exactly so the parity gate stays bit-identical modulo `__expf` /
// reduction-order ULP.
//
// Forward identity (matches `rms_norm_f32`):
//   sum_sq    = sum_j(x[r,j]^2)
//   inv_rms   = 1 / sqrt(sum_sq / hidden + eps)
//   y[r,j]    = x[r,j] * inv_rms * w[j]
//
// Backward (matches `cpu_rmsnorm_backward`):
//   inv       = inv_rms[r]
//   dot       = sum_j(upstream[r,j] * w[j] * x[r,j])
//   correction= inv * inv * dot / hidden
//   grad_x[r,j] = (inv * upstream[r,j] * w[j]) - (x[r,j] * inv * correction)
//   grad_w[j]   = sum_r(upstream[r,j] * x[r,j] * inv[r])
//
// Three kernels (driver chains them via lazy `launch_*` — terminal `eval`
// is the caller's):
//
// 1. `rms_norm_inv_rms_f32` — one block per row, reduces sum_sq and emits
//    the per-row inv_rms into a `[rows]` scratch buffer.
// 2. `rms_norm_backward_x_f32` — one block per row, consumes the saved
//    `inv_rms[r]` and reduces `dot` (one shared-mem reduction). Writes
//    grad_x[r,:].
// 3. `rms_norm_backward_w_f32` — one block per column, accumulates
//    `upstream[r,col] * x[r,col] * inv_rms[r]` across rows and reduces.
//    Writes grad_w[col].
//
// `__syncthreads()` discipline: full block-wide barriers around every
// shared-mem read; tree reduction is the canonical block / 2 form. `eps`
// is consumed only by the first kernel so the forward and backward agree
// bit-for-bit.

extern "C" __global__ void rms_norm_inv_rms_f32(
    float* __restrict__ inv_rms,
    const float* __restrict__ x,
    int cols,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block = blockDim.x;
    const float* row_x = x + row * cols;

    float local_sq = 0.0f;
    for (int i = tid; i < cols; i += block) {
        float v = row_x[i];
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
    float* __restrict__ grad_x,
    const float* __restrict__ upstream,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ inv_rms,
    int cols
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block = blockDim.x;
    const float* row_x = x + row * cols;
    const float* row_up = upstream + row * cols;
    float* row_grad = grad_x + row * cols;

    // Phase 1: reduce dot = sum_j(upstream * weight * x).
    float local_dot = 0.0f;
    for (int i = tid; i < cols; i += block) {
        local_dot += row_up[i] * weight[i] * row_x[i];
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

    // Phase 2: write grad_x elementwise.
    for (int i = tid; i < cols; i += block) {
        row_grad[i] = (inv * row_up[i] * weight[i]) - (row_x[i] * inv * correction);
    }
}

extern "C" __global__ void rms_norm_backward_w_f32(
    float* __restrict__ grad_w,
    const float* __restrict__ upstream,
    const float* __restrict__ x,
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
        local_sum += upstream[r * cols + col] * x[r * cols + col] * inv_rms[r];
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

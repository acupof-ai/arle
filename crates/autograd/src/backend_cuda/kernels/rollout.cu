extern "C" __global__ void argmax_last_dim_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    int rows,
    int vocab
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int block = blockDim.x;
    const float* row_x = x + (long long)row * vocab;

    float best_val = -3.4028234663852886e38F;
    int best_idx = 0;
    for (int i = tid; i < vocab; i += block) {
        float value = row_x[i];
        if (value > best_val || (value == best_val && i < best_idx)) {
            best_val = value;
            best_idx = i;
        }
    }

    extern __shared__ float smem[];
    float* vals = smem;
    int* idxs = reinterpret_cast<int*>(vals + blockDim.x);
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = vals[tid + stride];
            int other_idx = idxs[tid + stride];
            if (other_val > vals[tid] || (other_val == vals[tid] && other_idx < idxs[tid])) {
                vals[tid] = other_val;
                idxs[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = static_cast<float>(idxs[0]);
    }
}

extern "C" __global__ void embedding_f32_ids_f32(
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ ids,
    int n_ids,
    int vocab,
    int dim
) {
    int row = blockIdx.x;
    if (row >= n_ids) return;
    int tid = threadIdx.x;
    int block = blockDim.x;
    int id = static_cast<int>(ids[row]);
    float* row_out = out + (long long)row * dim;
    if (id < 0 || id >= vocab) {
        for (int i = tid; i < dim; i += block) {
            row_out[i] = 0.0f;
        }
        return;
    }
    const float* row_w = weight + (long long)id * dim;
    for (int i = tid; i < dim; i += block) {
        row_out[i] = row_w[i];
    }
}

extern "C" __global__ void write_scalar_at_f32(
    float* __restrict__ out,
    const float* __restrict__ dest,
    const float* __restrict__ src,
    int len,
    int index
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    out[i] = (i == index) ? src[0] : dest[i];
}

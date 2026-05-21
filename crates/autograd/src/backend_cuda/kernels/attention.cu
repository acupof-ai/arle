extern "C" __global__ void causal_sdpa_decode_gqa_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int batch,
    int query_heads,
    int kv_heads,
    int kv_len,
    int head_dim,
    int q_start,
    float scale
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int b = row / query_heads;
    int qh = row - b * query_heads;
    if (b >= batch) {
        return;
    }

    int kv_repeat = query_heads / kv_heads;
    int kvh = qh / kv_repeat;
    int visible = q_start + 1;
    if (visible > kv_len) {
        visible = kv_len;
    }
    if (visible <= 0 || visible > 32) {
        return;
    }

    extern __shared__ float smem[];
    float* reduce = smem;
    float* scores = smem + blockDim.x;

    int q_base = ((b * query_heads + qh) * head_dim);
    int kv_base = ((b * kv_heads + kvh) * kv_len) * head_dim;

    for (int pos = 0; pos < visible; ++pos) {
        float partial = 0.0f;
        int k_base = kv_base + pos * head_dim;
        for (int dim = tid; dim < head_dim; dim += blockDim.x) {
            partial += q[q_base + dim] * k[k_base + dim];
        }
        reduce[tid] = partial;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            scores[pos] = reduce[0] * scale;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float max_score = scores[0];
        for (int pos = 1; pos < visible; ++pos) {
            max_score = fmaxf(max_score, scores[pos]);
        }
        float denom = 0.0f;
        for (int pos = 0; pos < visible; ++pos) {
            float weight = expf(scores[pos] - max_score);
            scores[pos] = weight;
            denom += weight;
        }
        float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (int pos = 0; pos < visible; ++pos) {
            scores[pos] *= inv_denom;
        }
    }
    __syncthreads();

    int out_base = ((b * query_heads + qh) * head_dim);
    for (int dim = tid; dim < head_dim; dim += blockDim.x) {
        float acc = 0.0f;
        for (int pos = 0; pos < visible; ++pos) {
            int v_base = kv_base + pos * head_dim;
            acc += scores[pos] * v[v_base + dim];
        }
        out[out_base + dim] = acc;
    }
}

extern "C" __global__ void qwen_decode_prepare_q_f32(
    float* __restrict__ q_out,
    const float* __restrict__ q_full,
    const float* __restrict__ q_norm_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int batch,
    int query_heads,
    int head_dim,
    int q_full_stride,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int b = row / query_heads;
    int h = row - b * query_heads;
    if (b >= batch) {
        return;
    }

    int half_dim = head_dim >> 1;
    int q_full_base = b * q_full_stride + h * head_dim;
    int out_base = row * head_dim;

    float local_sq = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float x = q_full[q_full_base + d];
        local_sq += x * x;
    }
    smem[tid] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf((smem[0] / (float)head_dim) + eps);

    for (int i = tid; i < half_dim; i += blockDim.x) {
        float x0 = q_full[q_full_base + i] * inv_rms * (1.0f + q_norm_weight[i]);
        float x1 = q_full[q_full_base + i + half_dim] * inv_rms * (1.0f + q_norm_weight[i + half_dim]);
        float c = cos_table[i];
        float s = sin_table[i];
        q_out[out_base + i] = x0 * c - x1 * s;
        q_out[out_base + i + half_dim] = x1 * c + x0 * s;
    }
}

extern "C" __global__ void qwen_decode_prepare_q_gated_f32(
    float* __restrict__ q_out,
    float* __restrict__ gate_out,
    const float* __restrict__ q_full,
    const float* __restrict__ q_norm_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int batch,
    int query_heads,
    int head_dim,
    int q_full_stride,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int b = row / query_heads;
    int h = row - b * query_heads;
    if (b >= batch) {
        return;
    }

    int half_dim = head_dim >> 1;
    int head_stride = head_dim * 2;
    int q_full_base = b * q_full_stride + h * head_stride;
    int out_base = row * head_dim;

    float local_sq = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float x = q_full[q_full_base + d];
        local_sq += x * x;
        gate_out[out_base + d] = q_full[q_full_base + head_dim + d];
    }
    smem[tid] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf((smem[0] / (float)head_dim) + eps);

    for (int i = tid; i < half_dim; i += blockDim.x) {
        float x0 = q_full[q_full_base + i] * inv_rms * (1.0f + q_norm_weight[i]);
        float x1 = q_full[q_full_base + i + half_dim] * inv_rms * (1.0f + q_norm_weight[i + half_dim]);
        float c = cos_table[i];
        float s = sin_table[i];
        q_out[out_base + i] = x0 * c - x1 * s;
        q_out[out_base + i + half_dim] = x1 * c + x0 * s;
    }
}

extern "C" __global__ void qwen_decode_prepare_kv_f32(
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ k_full,
    const float* __restrict__ v_full,
    const float* __restrict__ k_norm_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int batch,
    int kv_heads,
    int head_dim,
    int kv_full_stride,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int b = row / kv_heads;
    int h = row - b * kv_heads;
    if (b >= batch) {
        return;
    }

    int half_dim = head_dim >> 1;
    int full_base = b * kv_full_stride + h * head_dim;
    int out_base = row * head_dim;

    float local_sq = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float x = k_full[full_base + d];
        local_sq += x * x;
        v_out[out_base + d] = v_full[full_base + d];
    }
    smem[tid] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf((smem[0] / (float)head_dim) + eps);

    for (int i = tid; i < half_dim; i += blockDim.x) {
        float x0 = k_full[full_base + i] * inv_rms * (1.0f + k_norm_weight[i]);
        float x1 = k_full[full_base + i + half_dim] * inv_rms * (1.0f + k_norm_weight[i + half_dim]);
        float c = cos_table[i];
        float s = sin_table[i];
        k_out[out_base + i] = x0 * c - x1 * s;
        k_out[out_base + i + half_dim] = x1 * c + x0 * s;
    }
}

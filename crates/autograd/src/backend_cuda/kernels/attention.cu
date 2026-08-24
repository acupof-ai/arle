__inline__ __device__ float arle_warp_sum_f32(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

extern "C" __global__ void causal_sdpa_recompute_backward_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ upstream,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    int rows,
    int seq_len,
    int head_dim,
    float scale,
    int need_grad_q,
    int need_grad_k,
    int need_grad_v
) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }

    int merged_head = row / seq_len;
    int q_pos = row - merged_head * seq_len;
    int visible = q_pos + 1;
    int q_base = (merged_head * seq_len + q_pos) * head_dim;
    int kv_base = merged_head * seq_len * head_dim;
    float dq_acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dq_acc[i] = 0.0f;
    }

    float max_score = -3.4028234663852886e38f;
    for (int pos = 0; pos < visible; ++pos) {
        int k_base = kv_base + pos * head_dim;
        float partial = 0.0f;
        for (int dim = lane; dim < head_dim; dim += 32) {
            partial += q[q_base + dim] * k[k_base + dim];
        }
        float score = arle_warp_sum_f32(partial) * scale;
        if (score > max_score) {
            max_score = score;
        }
    }

    float denom = 0.0f;
    for (int pos = 0; pos < visible; ++pos) {
        int k_base = kv_base + pos * head_dim;
        float partial = 0.0f;
        for (int dim = lane; dim < head_dim; dim += 32) {
            partial += q[q_base + dim] * k[k_base + dim];
        }
        float score = arle_warp_sum_f32(partial) * scale;
        denom += expf(score - max_score);
    }
    float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;

    float weighted_dprob_sum = 0.0f;
    for (int pos = 0; pos < visible; ++pos) {
        int k_base = kv_base + pos * head_dim;
        int v_base = kv_base + pos * head_dim;
        float qk_partial = 0.0f;
        float dv_partial = 0.0f;
        for (int dim = lane; dim < head_dim; dim += 32) {
            qk_partial += q[q_base + dim] * k[k_base + dim];
            dv_partial += upstream[q_base + dim] * v[v_base + dim];
        }
        float score = arle_warp_sum_f32(qk_partial) * scale;
        float prob = expf(score - max_score) * inv_denom;
        if (need_grad_v) {
            for (int dim = lane; dim < head_dim; dim += 32) {
                atomicAdd(&grad_v[v_base + dim], prob * upstream[q_base + dim]);
            }
        }
        float dprob = arle_warp_sum_f32(dv_partial);
        weighted_dprob_sum += prob * dprob;
    }

    for (int pos = 0; pos < visible; ++pos) {
        int k_base = kv_base + pos * head_dim;
        int v_base = kv_base + pos * head_dim;
        float qk_partial = 0.0f;
        float dv_partial = 0.0f;
        for (int dim = lane; dim < head_dim; dim += 32) {
            qk_partial += q[q_base + dim] * k[k_base + dim];
            dv_partial += upstream[q_base + dim] * v[v_base + dim];
        }
        float score = arle_warp_sum_f32(qk_partial) * scale;
        float prob = expf(score - max_score) * inv_denom;
        float dprob = arle_warp_sum_f32(dv_partial);
        float d_score = prob * (dprob - weighted_dprob_sum) * scale;
        int local = 0;
        for (int dim = lane; dim < head_dim; dim += 32, ++local) {
            if (need_grad_q) {
                dq_acc[local] += d_score * k[k_base + dim];
            }
            if (need_grad_k) {
                atomicAdd(&grad_k[k_base + dim], d_score * q[q_base + dim]);
            }
        }
    }
    if (need_grad_q) {
        int local = 0;
        for (int dim = lane; dim < head_dim; dim += 32, ++local) {
            grad_q[q_base + dim] = dq_acc[local];
        }
    }
}

extern "C" __global__ void causal_sdpa_decode_gqa_cache_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int batch,
    int query_heads,
    int kv_heads,
    int max_seq,
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
    if (visible <= 0) {
        return;
    }

    extern __shared__ float smem[];
    float* reduce = smem;
    float* scores = smem + blockDim.x;

    int q_base = ((b * query_heads + qh) * head_dim);
    int kv_base = ((b * kv_heads + kvh) * max_seq) * head_dim;

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

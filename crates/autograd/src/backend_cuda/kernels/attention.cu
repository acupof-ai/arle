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

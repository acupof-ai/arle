// Qwen3.5 linear-attention backward scan spike.
//
// One CUDA block owns one (batch, value_head) stream and walks sequence in
// reverse. The recurrence dependency is along time, so the cheap parallelism is
// inside each step: threads cooperate across key_dim/value_dim reductions and
// the key_dim x value_dim state-gradient matrix. This intentionally keeps the
// existing Rust forward recompute and conv1d backward in place; the spike only
// replaces the `scan_state_history` host loop.

static constexpr int LINEAR_ATTENTION_MAX_DIM = 256;
static constexpr int LINEAR_ATTENTION_BLOCK = 256;

__device__ __forceinline__ float la_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float la_silu(float x) {
    return x * la_sigmoid(x);
}

__device__ __forceinline__ float la_silu_grad(float x) {
    float sig = la_sigmoid(x);
    return sig * (1.0f + x * (1.0f - sig));
}

__device__ __forceinline__ float la_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ int la_idx3(int batch, int seq, int dim, int seq_len, int width) {
    return (batch * seq_len + seq) * width + dim;
}

__device__ __forceinline__ int la_idx4(
    int batch,
    int seq,
    int head,
    int dim,
    int seq_len,
    int heads,
    int width
) {
    return (((batch * seq_len + seq) * heads + head) * width) + dim;
}

__device__ __forceinline__ int la_state_base(
    int batch,
    int head,
    int heads,
    int key_dim,
    int value_dim
) {
    return ((batch * heads + head) * key_dim) * value_dim;
}

__device__ __forceinline__ int la_state_time_base(
    int batch,
    int seq,
    int head,
    int seq_len,
    int heads,
    int key_dim,
    int value_dim
) {
    return (((batch * seq_len + seq) * heads + head) * key_dim) * value_dim;
}

extern "C" __global__ void linear_attention_scan_backward_f32(
    float* __restrict__ dqkv,
    float* __restrict__ dz,
    float* __restrict__ db,
    float* __restrict__ da,
    float* __restrict__ ddt,
    float* __restrict__ da_log,
    float* __restrict__ dnorm,
    float* __restrict__ grad_state_scratch,
    const float* __restrict__ upstream,
    const float* __restrict__ z,
    const float* __restrict__ a_proj,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const float* __restrict__ norm_weight,
    const float* __restrict__ preact,
    const float* __restrict__ beta,
    const float* __restrict__ exp_g,
    const float* __restrict__ kv_mem,
    const float* __restrict__ state_history,
    const float* __restrict__ final_state,
    int batch,
    int seq_len,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int value_dim,
    int qkv_dim,
    float eps
) {
    int row = blockIdx.x;
    int batch_idx = row / num_value_heads;
    int value_head = row - (batch_idx * num_value_heads);
    if (batch_idx >= batch || key_dim > LINEAR_ATTENTION_MAX_DIM ||
        value_dim > LINEAR_ATTENTION_MAX_DIM) {
        return;
    }

    int tid = threadIdx.x;
    int q_dim = num_key_heads * key_dim;
    int v_offset = q_dim + q_dim;
    int key_head = value_head * num_key_heads / num_value_heads;
    int state_elems = key_dim * value_dim;
    float* grad_state = grad_state_scratch + row * state_elems;

    __shared__ float q_raw[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float k_raw[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float q_vec[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float k_vec[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float dq_vec[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float dk_vec[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float v_raw[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float delta[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float d_delta[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float dkv_mem[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float core_out[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float dcore[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float reduce[LINEAR_ATTENTION_BLOCK];
    __shared__ float scalar0;
    __shared__ float scalar1;

    for (int i = tid; i < state_elems; i += blockDim.x) {
        grad_state[i] = 0.0f;
    }
    __syncthreads();

    float exp_a = expf(a_log[value_head]);
    float q_scale = rsqrtf((float)key_dim);

    for (int seq_idx = seq_len - 1; seq_idx >= 0; --seq_idx) {
        for (int i = tid; i < key_dim; i += blockDim.x) {
            q_raw[i] = la_silu(preact[la_idx3(
                batch_idx, seq_idx, key_head * key_dim + i, seq_len, qkv_dim)]);
            k_raw[i] = la_silu(preact[la_idx3(
                batch_idx, seq_idx, q_dim + key_head * key_dim + i, seq_len, qkv_dim)]);
        }
        for (int i = tid; i < value_dim; i += blockDim.x) {
            v_raw[i] = la_silu(preact[la_idx3(
                batch_idx, seq_idx, v_offset + value_head * value_dim + i, seq_len, qkv_dim)]);
        }
        __syncthreads();

        float local_q_sq = 0.0f;
        float local_k_sq = 0.0f;
        for (int i = tid; i < key_dim; i += blockDim.x) {
            local_q_sq += q_raw[i] * q_raw[i];
            local_k_sq += k_raw[i] * k_raw[i];
        }
        reduce[tid] = local_q_sq;
        __syncthreads();
        for (int step = blockDim.x / 2; step > 0; step >>= 1) {
            if (tid < step) {
                reduce[tid] += reduce[tid + step];
            }
            __syncthreads();
        }
        if (tid == 0) {
            scalar0 = sqrtf(reduce[0] + 1.0e-12f);
        }
        __syncthreads();
        float q_norm = scalar0;

        reduce[tid] = local_k_sq;
        __syncthreads();
        for (int step = blockDim.x / 2; step > 0; step >>= 1) {
            if (tid < step) {
                reduce[tid] += reduce[tid + step];
            }
            __syncthreads();
        }
        if (tid == 0) {
            scalar0 = sqrtf(reduce[0] + 1.0e-12f);
        }
        __syncthreads();
        float k_norm = scalar0;

        for (int i = tid; i < key_dim; i += blockDim.x) {
            q_vec[i] = q_scale * q_raw[i] / q_norm;
            k_vec[i] = k_raw[i] / k_norm;
        }
        __syncthreads();

        int state_base = seq_idx == seq_len - 1
            ? la_state_base(batch_idx, value_head, num_value_heads, key_dim, value_dim)
            : la_state_time_base(
                  batch_idx, seq_idx, value_head, seq_len, num_value_heads, key_dim, value_dim);
        const float* state = seq_idx == seq_len - 1
            ? final_state + state_base
            : state_history + state_base;
        int prev_base = seq_idx == 0
            ? 0
            : la_state_time_base(
                  batch_idx, seq_idx - 1, value_head, seq_len, num_value_heads, key_dim, value_dim);
        const float* prev_state = state_history + prev_base;

        for (int v = tid; v < value_dim; v += blockDim.x) {
            float accum = 0.0f;
            for (int k = 0; k < key_dim; ++k) {
                accum += state[k * value_dim + v] * q_vec[k];
            }
            core_out[v] = accum;
        }
        __syncthreads();

        float local_core_sq = 0.0f;
        for (int v = tid; v < value_dim; v += blockDim.x) {
            local_core_sq += core_out[v] * core_out[v];
        }
        reduce[tid] = local_core_sq;
        __syncthreads();
        for (int step = blockDim.x / 2; step > 0; step >>= 1) {
            if (tid < step) {
                reduce[tid] += reduce[tid + step];
            }
            __syncthreads();
        }
        if (tid == 0) {
            scalar0 = rsqrtf((reduce[0] / (float)value_dim) + eps);
        }
        __syncthreads();
        float inv_rms = scalar0;

        float local_dot_beta = 0.0f;
        for (int v = tid; v < value_dim; v += blockDim.x) {
            int out_idx = la_idx4(
                batch_idx, seq_idx, value_head, v, seq_len, num_value_heads, value_dim);
            float normed = core_out[v] * inv_rms * norm_weight[v];
            float gate = z[out_idx];
            float gate_silu = la_silu(gate);
            float dcore_v = upstream[out_idx] * gate_silu;
            dcore[v] = dcore_v;
            dz[out_idx] = upstream[out_idx] * normed * la_silu_grad(gate);
            atomicAdd(&dnorm[v], dcore_v * core_out[v] * inv_rms);
            local_dot_beta += dcore_v * core_out[v] * norm_weight[v];
        }
        reduce[tid] = local_dot_beta;
        __syncthreads();
        for (int step = blockDim.x / 2; step > 0; step >>= 1) {
            if (tid < step) {
                reduce[tid] += reduce[tid + step];
            }
            __syncthreads();
        }
        if (tid == 0) {
            scalar0 = reduce[0];
        }
        __syncthreads();
        float dot_beta = scalar0;
        float coeff = inv_rms * inv_rms * inv_rms / (float)value_dim;
        for (int v = tid; v < value_dim; v += blockDim.x) {
            dcore[v] = dcore[v] * norm_weight[v] * inv_rms - core_out[v] * coeff * dot_beta;
        }
        __syncthreads();

        for (int k = tid; k < key_dim; k += blockDim.x) {
            float accum = 0.0f;
            for (int v = 0; v < value_dim; ++v) {
                accum += state[k * value_dim + v] * dcore[v];
            }
            dq_vec[k] = accum;
        }
        for (int idx = tid; idx < state_elems; idx += blockDim.x) {
            int k = idx / value_dim;
            int v = idx - k * value_dim;
            grad_state[idx] += q_vec[k] * dcore[v];
        }
        __syncthreads();

        float beta_value = beta[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
        float exp_g_value =
            exp_g[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
        for (int v = tid; v < value_dim; v += blockDim.x) {
            float kv = kv_mem[la_idx4(
                batch_idx, seq_idx, value_head, v, seq_len, num_value_heads, value_dim)];
            delta[v] = (v_raw[v] - kv) * beta_value;
        }
        __syncthreads();

        for (int v = tid; v < value_dim; v += blockDim.x) {
            float accum = 0.0f;
            for (int k = 0; k < key_dim; ++k) {
                accum += grad_state[k * value_dim + v] * k_vec[k];
            }
            d_delta[v] = accum;
        }
        for (int k = tid; k < key_dim; k += blockDim.x) {
            float accum = 0.0f;
            for (int v = 0; v < value_dim; ++v) {
                accum += grad_state[k * value_dim + v] * delta[v];
            }
            dk_vec[k] = accum;
        }
        __syncthreads();

        float local_dbeta = 0.0f;
        for (int v = tid; v < value_dim; v += blockDim.x) {
            float kv = kv_mem[la_idx4(
                batch_idx, seq_idx, value_head, v, seq_len, num_value_heads, value_dim)];
            local_dbeta += d_delta[v] * (v_raw[v] - kv);
            dkv_mem[v] = -d_delta[v] * beta_value;
        }
        reduce[tid] = local_dbeta;
        __syncthreads();
        for (int step = blockDim.x / 2; step > 0; step >>= 1) {
            if (tid < step) {
                reduce[tid] += reduce[tid + step];
            }
            __syncthreads();
        }
        if (tid == 0) {
            scalar0 = reduce[0];
        }
        __syncthreads();
        float dbeta_scalar = scalar0;

        for (int idx = tid; idx < state_elems; idx += blockDim.x) {
            int k = idx / value_dim;
            int v = idx - k * value_dim;
            grad_state[idx] += k_vec[k] * dkv_mem[v];
        }
        __syncthreads();
        for (int k = tid; k < key_dim; k += blockDim.x) {
            float accum = 0.0f;
            for (int v = 0; v < value_dim; ++v) {
                float s_decay = state[k * value_dim + v] - k_vec[k] * delta[v];
                accum += s_decay * dkv_mem[v];
            }
            dk_vec[k] += accum;
        }
        __syncthreads();

        float local_dexp_g = 0.0f;
        for (int idx = tid; idx < state_elems; idx += blockDim.x) {
            float prev = seq_idx == 0 ? 0.0f : prev_state[idx];
            local_dexp_g += prev * grad_state[idx];
        }
        reduce[tid] = local_dexp_g;
        __syncthreads();
        for (int step = blockDim.x / 2; step > 0; step >>= 1) {
            if (tid < step) {
                reduce[tid] += reduce[tid + step];
            }
            __syncthreads();
        }
        if (tid == 0) {
            scalar0 = reduce[0];
        }
        __syncthreads();
        float dexp_g = scalar0;

        if (tid == 0) {
            float a_value = a_proj[la_idx3(
                batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
            float softplus_input = a_value + dt_bias[value_head];
            float softplus_value = la_softplus(softplus_input);
            float softplus_grad = la_sigmoid(softplus_input);
            float dg = dexp_g * exp_g_value;
            float common = dg * (-exp_a);
            da[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)] =
                common * softplus_grad;
            atomicAdd(&ddt[value_head], common * softplus_grad);
            atomicAdd(&da_log[value_head], common * softplus_value);
            db[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)] =
                dbeta_scalar * beta_value * (1.0f - beta_value);
        }

        float local_q_dot = 0.0f;
        float local_k_dot = 0.0f;
        for (int i = tid; i < key_dim; i += blockDim.x) {
            local_q_dot += q_raw[i] * dq_vec[i];
            local_k_dot += k_raw[i] * dk_vec[i];
        }
        reduce[tid] = local_q_dot;
        __syncthreads();
        for (int step = blockDim.x / 2; step > 0; step >>= 1) {
            if (tid < step) {
                reduce[tid] += reduce[tid + step];
            }
            __syncthreads();
        }
        if (tid == 0) {
            scalar0 = reduce[0];
        }
        __syncthreads();
        float q_dot = scalar0;

        reduce[tid] = local_k_dot;
        __syncthreads();
        for (int step = blockDim.x / 2; step > 0; step >>= 1) {
            if (tid < step) {
                reduce[tid] += reduce[tid + step];
            }
            __syncthreads();
        }
        if (tid == 0) {
            scalar1 = reduce[0];
        }
        __syncthreads();
        float k_dot = scalar1;
        float q_norm_cubed = q_norm * q_norm * q_norm;
        float k_norm_cubed = k_norm * k_norm * k_norm;

        for (int i = tid; i < key_dim; i += blockDim.x) {
            float dq_raw =
                q_scale * (dq_vec[i] / q_norm - q_raw[i] * q_dot / q_norm_cubed);
            float dk_raw = dk_vec[i] / k_norm - k_raw[i] * k_dot / k_norm_cubed;
            atomicAdd(
                &dqkv[la_idx3(batch_idx, seq_idx, key_head * key_dim + i, seq_len, qkv_dim)],
                dq_raw);
            atomicAdd(
                &dqkv[la_idx3(
                    batch_idx, seq_idx, q_dim + key_head * key_dim + i, seq_len, qkv_dim)],
                dk_raw);
        }
        for (int v = tid; v < value_dim; v += blockDim.x) {
            float dv_raw = d_delta[v] * beta_value;
            atomicAdd(
                &dqkv[la_idx3(
                    batch_idx, seq_idx, v_offset + value_head * value_dim + v, seq_len, qkv_dim)],
                dv_raw);
        }
        __syncthreads();

        for (int idx = tid; idx < state_elems; idx += blockDim.x) {
            grad_state[idx] = grad_state[idx] * exp_g_value;
        }
        __syncthreads();
    }
}

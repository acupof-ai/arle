// Flattened offsets are 64-bit; dimensions remain 32-bit.

static constexpr int LINEAR_ATTENTION_MAX_DIM = 256;
static constexpr int LINEAR_ATTENTION_BLOCK = 256;

__device__ __forceinline__ float la_bf16_to_float(unsigned short bits16) {
    unsigned int bits32 = static_cast<unsigned int>(bits16) << 16;
    return __uint_as_float(bits32);
}

__device__ __forceinline__ unsigned short la_float_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounding_bias = 0x7fffu + lsb;
    return static_cast<unsigned short>((bits + rounding_bias) >> 16);
}

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

__device__ __forceinline__ unsigned long long la_idx3(
    int batch,
    int seq,
    int dim,
    int seq_len,
    int width
) {
    return (static_cast<unsigned long long>(batch) * seq_len + seq) * width + dim;
}

__device__ __forceinline__ unsigned long long la_idx4(
    int batch,
    int seq,
    int head,
    int dim,
    int seq_len,
    int heads,
    int width
) {
    return (((static_cast<unsigned long long>(batch) * seq_len + seq) * heads + head) * width) + dim;
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

// 64-bit: batch*seq*heads*key_dim*value_dim overflows int32 past ~2^31 elems
// (e.g. 4x3153x16x128x128 = 3.31e9 state_history floats).
__device__ __forceinline__ long long la_state_time_base(
    int batch,
    int seq,
    int head,
    int seq_len,
    int heads,
    int key_dim,
    int value_dim
) {
    return ((((long long)batch * seq_len + seq) * heads + head) * key_dim) * value_dim;
}

// Block-wide sum. Every thread gets the total and `reduce` is reusable on return.
__device__ __forceinline__ float la_block_sum(
    float* reduce,
    int tid,
    int block,
    float value
) {
    reduce[tid] = value;
    __syncthreads();
    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) {
            reduce[tid] += reduce[tid + step];
        }
        __syncthreads();
    }
    float total = reduce[0];
    __syncthreads();
    return total;
}

// The pre-norm GDN core row and its gradient: bf16 in global memory on the
// FlashQLA route, f32 in shared memory inside the fused chunk kernel.
struct LaCoreRowBf16 {
    const unsigned short* __restrict__ x;
    unsigned short* __restrict__ d;
    __device__ __forceinline__ float load(int i) const { return la_bf16_to_float(x[i]); }
    __device__ __forceinline__ void store(int i, float v) const { d[i] = la_float_to_bf16(v); }
};

struct LaCoreRowF32 {
    const float* x;
    float* d;
    __device__ __forceinline__ float load(int i) const { return x[i]; }
    __device__ __forceinline__ void store(int i, float v) const { d[i] = v; }
};

// Backward of out = silu(z) * weight * x * rsqrt(mean(x^2) + eps) for one
// (token, value head) row. dz/dnorm land in global memory; dx goes back
// through `row`. Pointers are already offset to the row.
template <typename Row>
__device__ __forceinline__ void la_rms_gated_backward_row(
    Row row,
    float* reduce,
    float* __restrict__ dz,
    float* __restrict__ dnorm,
    const float* __restrict__ z,
    const float* __restrict__ upstream,
    const float* __restrict__ norm_weight,
    int tid,
    int block,
    int value_dim,
    float eps
) {
    float local_sq = 0.0f;
    for (int i = tid; i < value_dim; i += block) {
        float x = row.load(i);
        local_sq += x * x;
    }
    float inv_rms =
        rsqrtf((la_block_sum(reduce, tid, block, local_sq) / (float)value_dim) + eps);

    float local_dot = 0.0f;
    for (int i = tid; i < value_dim; i += block) {
        float x = row.load(i);
        float gate = z[i];
        float up = upstream[i];
        float dcore = up * la_silu(gate);
        // Association matches the fused kernels' original: they are the verified
        // path, so the never-run FlashQLA kernel is the one that conforms.
        dz[i] = up * (x * inv_rms * norm_weight[i]) * la_silu_grad(gate);
        atomicAdd(&dnorm[i], dcore * x * inv_rms);
        local_dot += dcore * x * norm_weight[i];
    }
    float dot_beta = la_block_sum(reduce, tid, block, local_dot);

    float coeff = inv_rms * inv_rms * inv_rms / (float)value_dim;
    for (int i = tid; i < value_dim; i += block) {
        float x = row.load(i);
        float dcore = upstream[i] * la_silu(z[i]);
        row.store(i, dcore * norm_weight[i] * inv_rms - x * coeff * dot_beta);
    }
}

// d(x / ||x||) pulled back onto x.
__device__ __forceinline__ float la_l2_norm_grad(
    float d_normed,
    float raw,
    float dot,
    float norm,
    float norm_cubed
) {
    return d_normed / norm - raw * dot / norm_cubed;
}

// Gate-parameter tail: the per-token decay/beta grads onto a_proj, dt_bias,
// A_log and b_proj. Single-thread; `head_idx` indexes the [b, s, head] planes.
__device__ __forceinline__ void la_gate_param_backward(
    float* __restrict__ da,
    float* __restrict__ ddt,
    float* __restrict__ da_log,
    float* __restrict__ db,
    const float* __restrict__ a_proj,
    const float* __restrict__ dt_bias,
    unsigned long long head_idx,
    int value_head,
    float exp_a,
    float dg,
    float dbeta_value,
    float beta_value
) {
    float softplus_input = a_proj[head_idx] + dt_bias[value_head];
    float softplus_value = la_softplus(softplus_input);
    float softplus_grad = la_sigmoid(softplus_input);
    float common = dg * (-exp_a);
    da[head_idx] = common * softplus_grad;
    atomicAdd(&ddt[value_head], common * softplus_grad);
    atomicAdd(&da_log[value_head], common * softplus_value);
    db[head_idx] = dbeta_value * beta_value * (1.0f - beta_value);
}

extern "C" __global__ void linear_attention_conv1d_silu_forward_f32_to_bf16(
    float* __restrict__ preact,
    unsigned short* __restrict__ out_bf16,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    unsigned long long total,
    int channels,
    int seq_len,
    int kernel_size,
    const float* __restrict__ conv_tail,
    int tail_len
) {
    unsigned long long idx =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int c = idx % channels;
    int t = idx / channels;
    float sum = 0.0f;
    for (int tap = 0; tap < kernel_size; ++tap) {
        int src_t = t + tap + 1 - kernel_size;
        if (src_t < 0) {
            // Carried boundary window feeds the left taps; nullptr keeps the zero-pad default.
            if (conv_tail != nullptr) {
                sum += conv_tail[(src_t + tail_len) * channels + c] * weight[c * kernel_size + tap];
            }
            continue;
        }
        sum += x[static_cast<unsigned long long>(src_t) * channels + c] * weight[c * kernel_size + tap];
    }
    preact[idx] = sum;
    out_bf16[idx] = la_float_to_bf16(la_silu(sum));
}

extern "C" __global__ void linear_attention_conv1d_silu_boundary_f32_to_bf16(
    unsigned short* __restrict__ out_bf16,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    unsigned long long total,
    int start_seq,
    int channels,
    int kernel_size,
    const float* __restrict__ conv_tail,
    int tail_len
) {
    unsigned long long idx =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int c = idx % channels;
    int t = start_seq + static_cast<int>(idx / channels);
    float sum = 0.0f;
    for (int tap = 0; tap < kernel_size; ++tap) {
        int src_t = t + tap + 1 - kernel_size;
        if (src_t < 0) {
            if (conv_tail != nullptr) {
                sum += conv_tail[(src_t + tail_len) * channels + c] *
                    weight[c * kernel_size + tap];
            }
        } else {
            sum += x[static_cast<unsigned long long>(src_t) * channels + c] *
                weight[c * kernel_size + tap];
        }
    }
    out_bf16[idx] = la_float_to_bf16(la_silu(sum));
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

        long long state_base = seq_idx == seq_len - 1
            ? (long long)la_state_base(batch_idx, value_head, num_value_heads, key_dim, value_dim)
            : la_state_time_base(
                  batch_idx, seq_idx, value_head, seq_len, num_value_heads, key_dim, value_dim);
        const float* state = seq_idx == seq_len - 1
            ? final_state + state_base
            : state_history + state_base;
        long long prev_base = seq_idx == 0
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
            unsigned long long out_idx = la_idx4(
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
            la_gate_param_backward(
                da, ddt, da_log, db, a_proj, dt_bias,
                la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads),
                value_head, exp_a, dexp_g * exp_g_value, dbeta_scalar, beta_value);
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
                q_scale * la_l2_norm_grad(dq_vec[i], q_raw[i], q_dot, q_norm, q_norm_cubed);
            float dk_raw = la_l2_norm_grad(dk_vec[i], k_raw[i], k_dot, k_norm, k_norm_cubed);
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

extern "C" __global__ void linear_attention_copy_f32(
    unsigned long long dst_addr,
    unsigned long long src_addr,
    int len
) {
    float* __restrict__ dst = reinterpret_cast<float*>(dst_addr);
    const float* __restrict__ src = reinterpret_cast<const float*>(src_addr);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        dst[idx] = src[idx];
    }
}

extern "C" __global__ void linear_attention_rms_gated_forward_f32_from_bf16(
    float* __restrict__ out,
    const unsigned short* __restrict__ x_bf16,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    int rows,
    int value_dim,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x;
    int block = blockDim.x;
    const unsigned short* row_x = x_bf16 + row * value_dim;
    const float* row_z = z + row * value_dim;
    float* row_out = out + row * value_dim;

    float local_sq = 0.0f;
    for (int i = tid; i < value_dim; i += block) {
        float x = la_bf16_to_float(row_x[i]);
        local_sq += x * x;
    }
    float inv_rms =
        rsqrtf((la_block_sum(smem, tid, block, local_sq) / (float)value_dim) + eps);

    for (int i = tid; i < value_dim; i += block) {
        float gate = la_silu(row_z[i]);
        float x = la_bf16_to_float(row_x[i]);
        row_out[i] = x * inv_rms * weight[i] * gate;
    }
}

// Backward of the gated RMS-norm head, split out so the FlashQLA route can
// hand the GDN core a plain dO. One block per (token, value head) row.
extern "C" __global__ void linear_attention_rms_gated_backward_f32_to_bf16(
    unsigned short* __restrict__ d_raw,
    float* __restrict__ dz,
    float* __restrict__ dnorm,
    const unsigned short* __restrict__ raw_output,
    const float* __restrict__ z,
    const float* __restrict__ upstream,
    const float* __restrict__ norm_weight,
    int rows,
    int value_dim,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long base = (long long)row * value_dim;
    LaCoreRowBf16 core{raw_output + base, d_raw + base};
    la_rms_gated_backward_row(
        core, smem, dz + base, dnorm, z + base, upstream + base, norm_weight,
        threadIdx.x, blockDim.x, value_dim, eps);
}

// Backward of gdr_fq_prep + the chunk-local cumsum: takes the FlashQLA core
// grads and lands them on qkv_conv / a_proj / b_proj / dt_bias / A_log.
// grid = num_chunks * batch * value heads, chunk-major.
// dq/dk arrive on the VALUE-head axis, so the Hg<H group sums via atomicAdd —
// exact, because the l2-norm Jacobian is linear in the incoming grad.
extern "C" __global__ void linear_attention_gdr_prepare_backward_f32(
    float* __restrict__ dqkv_conv,
    float* __restrict__ db,
    float* __restrict__ da,
    float* __restrict__ ddt,
    float* __restrict__ da_log,
    const unsigned short* __restrict__ dq,
    const unsigned short* __restrict__ dk,
    const unsigned short* __restrict__ dv,
    const float* __restrict__ dg_cumsum,
    const float* __restrict__ dbeta,
    const unsigned short* __restrict__ qkv_conv,
    const float* __restrict__ a_proj,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const float* __restrict__ beta,
    int batch,
    int seq_len,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int value_dim,
    int qkv_dim
) {
    constexpr int BT = 64;
    __shared__ float dg_tok[BT];
    __shared__ float reduce[LINEAR_ATTENTION_BLOCK];

    int rows = batch * num_value_heads;
    int row = blockIdx.x % rows;
    int batch_idx = row / num_value_heads;
    int value_head = row - batch_idx * num_value_heads;
    if (batch_idx >= batch || key_dim > LINEAR_ATTENTION_MAX_DIM ||
        value_dim > LINEAR_ATTENTION_MAX_DIM) {
        return;
    }
    int base = (blockIdx.x / rows) * BT;
    int limit = seq_len - base;
    if (limit > BT) {
        limit = BT;
    }
    if (limit <= 0) {
        return;
    }

    int tid = threadIdx.x;
    int block = blockDim.x;
    int q_dim = num_key_heads * key_dim;
    int v_offset = q_dim + q_dim;
    int key_head = value_head * num_key_heads / num_value_heads;
    float exp_a = expf(a_log[value_head]);

    // dg is w.r.t. the chunk-local cumsum: reverse-inclusive-sum it back per token.
    for (int t = tid; t < BT; t += block) {
        dg_tok[t] = t < limit
            ? dg_cumsum[la_idx3(batch_idx, base + t, value_head, seq_len,
                                num_value_heads)]
            : 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        for (int t = limit - 2; t >= 0; --t) {
            dg_tok[t] += dg_tok[t + 1];
        }
    }
    __syncthreads();

    for (int local_t = 0; local_t < limit; ++local_t) {
        int seq_idx = base + local_t;
        long long v_row =
            ((long long)seq_idx * num_value_heads + value_head) * value_dim;

        float local_qq = 0.0f;
        float local_kk = 0.0f;
        float local_qd = 0.0f;
        float local_kd = 0.0f;
        for (int i = tid; i < key_dim; i += block) {
            float qr = la_bf16_to_float(qkv_conv[la_idx3(
                batch_idx, seq_idx, key_head * key_dim + i, seq_len, qkv_dim)]);
            float kr = la_bf16_to_float(qkv_conv[la_idx3(
                batch_idx, seq_idx, q_dim + key_head * key_dim + i, seq_len,
                qkv_dim)]);
            local_qq += qr * qr;
            local_kk += kr * kr;
            local_qd += qr * la_bf16_to_float(dq[v_row + i]);
            local_kd += kr * la_bf16_to_float(dk[v_row + i]);
        }

        float q_norm = sqrtf(la_block_sum(reduce, tid, block, local_qq) + 1.0e-12f);
        float k_norm = sqrtf(la_block_sum(reduce, tid, block, local_kk) + 1.0e-12f);
        float q_dot = la_block_sum(reduce, tid, block, local_qd);
        float k_dot = la_block_sum(reduce, tid, block, local_kd);

        float q_norm_cubed = q_norm * q_norm * q_norm;
        float k_norm_cubed = k_norm * k_norm * k_norm;
        for (int i = tid; i < key_dim; i += block) {
            float qr = la_bf16_to_float(qkv_conv[la_idx3(
                batch_idx, seq_idx, key_head * key_dim + i, seq_len, qkv_dim)]);
            float kr = la_bf16_to_float(qkv_conv[la_idx3(
                batch_idx, seq_idx, q_dim + key_head * key_dim + i, seq_len,
                qkv_dim)]);
            float dqn = la_bf16_to_float(dq[v_row + i]);
            float dkn = la_bf16_to_float(dk[v_row + i]);
            atomicAdd(
                &dqkv_conv[la_idx3(batch_idx, seq_idx, key_head * key_dim + i,
                                   seq_len, qkv_dim)],
                la_l2_norm_grad(dqn, qr, q_dot, q_norm, q_norm_cubed));
            atomicAdd(
                &dqkv_conv[la_idx3(batch_idx, seq_idx,
                                   q_dim + key_head * key_dim + i, seq_len,
                                   qkv_dim)],
                la_l2_norm_grad(dkn, kr, k_dot, k_norm, k_norm_cubed));
        }
        for (int i = tid; i < value_dim; i += block) {
            atomicAdd(
                &dqkv_conv[la_idx3(batch_idx, seq_idx,
                                   v_offset + value_head * value_dim + i,
                                   seq_len, qkv_dim)],
                la_bf16_to_float(dv[v_row + i]));
        }

        if (tid == 0) {
            unsigned long long head_idx = la_idx3(
                batch_idx, seq_idx, value_head, seq_len, num_value_heads);
            la_gate_param_backward(
                da, ddt, da_log, db, a_proj, dt_bias, head_idx, value_head, exp_a,
                dg_tok[local_t], dbeta[head_idx], beta[head_idx]);
        }
        __syncthreads();
    }
}

extern "C" __global__ void linear_attention_chunked_scan_backward_f32(
    float* __restrict__ dqkv_conv,
    float* __restrict__ dz,
    float* __restrict__ db,
    float* __restrict__ da,
    float* __restrict__ ddt,
    float* __restrict__ da_log,
    float* __restrict__ dnorm,
    float* __restrict__ grad_state_scratch,
    float* __restrict__ state_recompute_scratch,
    float* __restrict__ chunk_history_scratch,
    float* __restrict__ chunk_kv_scratch,
    const float* __restrict__ upstream,
    const float* __restrict__ z,
    const float* __restrict__ a_proj,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const float* __restrict__ norm_weight,
    const float* __restrict__ preact,
    const unsigned short* __restrict__ qkv_conv,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const float* __restrict__ chunk_state,
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

    constexpr int BT = 64;
    int tid = threadIdx.x;
    int q_dim = num_key_heads * key_dim;
    int v_offset = q_dim + q_dim;
    int key_head = value_head * num_key_heads / num_value_heads;
    int state_elems = key_dim * value_dim;
    int num_chunks = (seq_len + BT - 1) / BT;
    float* grad_state = grad_state_scratch + row * state_elems;
    float* state_cur = state_recompute_scratch + row * state_elems;
    float* history = chunk_history_scratch + row * BT * state_elems;
    float* kv_chunk = chunk_kv_scratch + row * BT * value_dim;

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

    for (int chunk_idx = num_chunks - 1; chunk_idx >= 0; --chunk_idx) {
        int base = chunk_idx * BT;
        int limit = seq_len - base;
        if (limit > BT) {
            limit = BT;
        }
        if (limit <= 0) {
            continue;
        }

        const float* chunk_state_base =
            chunk_state + (chunk_idx * num_value_heads + value_head) * state_elems;
        for (int idx = tid; idx < state_elems; idx += blockDim.x) {
            state_cur[idx] = chunk_state_base[idx];
        }
        __syncthreads();

        for (int local_t = 0; local_t < limit; ++local_t) {
            int seq_idx = base + local_t;
            for (int i = tid; i < key_dim; i += blockDim.x) {
                k_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
                    batch_idx, seq_idx, q_dim + key_head * key_dim + i, seq_len, qkv_dim)]);
            }
            for (int i = tid; i < value_dim; i += blockDim.x) {
                v_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
                    batch_idx, seq_idx, v_offset + value_head * value_dim + i, seq_len, qkv_dim)]);
            }
            __syncthreads();

            float local_k_sq = 0.0f;
            for (int i = tid; i < key_dim; i += blockDim.x) {
                local_k_sq += k_raw[i] * k_raw[i];
            }
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
                k_vec[i] = k_raw[i] / k_norm;
            }
            __syncthreads();

            float exp_g_value =
                expf(g[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)]);
            for (int idx = tid; idx < state_elems; idx += blockDim.x) {
                state_cur[idx] *= exp_g_value;
            }
            __syncthreads();

            float beta_value =
                beta[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
            for (int v = tid; v < value_dim; v += blockDim.x) {
                float accum = 0.0f;
                for (int k = 0; k < key_dim; ++k) {
                    accum += state_cur[k * value_dim + v] * k_vec[k];
                }
                kv_chunk[local_t * value_dim + v] = accum;
                delta[v] = (v_raw[v] - accum) * beta_value;
            }
            __syncthreads();

            for (int idx = tid; idx < state_elems; idx += blockDim.x) {
                int k = idx / value_dim;
                int v = idx - k * value_dim;
                state_cur[idx] += k_vec[k] * delta[v];
                history[local_t * state_elems + idx] = state_cur[idx];
            }
            __syncthreads();
        }

        for (int local_t = limit - 1; local_t >= 0; --local_t) {
            int seq_idx = base + local_t;
            for (int i = tid; i < key_dim; i += blockDim.x) {
                q_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
                    batch_idx, seq_idx, key_head * key_dim + i, seq_len, qkv_dim)]);
                k_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
                    batch_idx, seq_idx, q_dim + key_head * key_dim + i, seq_len, qkv_dim)]);
            }
            for (int i = tid; i < value_dim; i += blockDim.x) {
                v_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
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

            const float* state = history + local_t * state_elems;
            const float* prev_state =
                local_t == 0 ? chunk_state_base : history + (local_t - 1) * state_elems;

            for (int v = tid; v < value_dim; v += blockDim.x) {
                float accum = 0.0f;
                for (int k = 0; k < key_dim; ++k) {
                    accum += state[k * value_dim + v] * q_vec[k];
                }
                core_out[v] = accum;
            }
            __syncthreads();

            unsigned long long out_base = la_idx4(
                batch_idx, seq_idx, value_head, 0, seq_len, num_value_heads, value_dim);
            LaCoreRowF32 core{core_out, dcore};
            la_rms_gated_backward_row(
                core, reduce, dz + out_base, dnorm, z + out_base, upstream + out_base,
                norm_weight, tid, blockDim.x, value_dim, eps);
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

            float beta_value =
                beta[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
            float exp_g_value =
                expf(g[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)]);
            for (int v = tid; v < value_dim; v += blockDim.x) {
                float kv = kv_chunk[local_t * value_dim + v];
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
                float kv = kv_chunk[local_t * value_dim + v];
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
                local_dexp_g += prev_state[idx] * grad_state[idx];
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
                la_gate_param_backward(
                    da, ddt, da_log, db, a_proj, dt_bias,
                    la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads),
                    value_head, exp_a, dexp_g * exp_g_value, dbeta_scalar, beta_value);
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
                    q_scale * la_l2_norm_grad(dq_vec[i], q_raw[i], q_dot, q_norm, q_norm_cubed);
                float dk_raw = la_l2_norm_grad(dk_vec[i], k_raw[i], k_dot, k_norm, k_norm_cubed);
                atomicAdd(
                    &dqkv_conv[la_idx3(batch_idx, seq_idx, key_head * key_dim + i, seq_len, qkv_dim)],
                    dq_raw);
                atomicAdd(
                    &dqkv_conv[la_idx3(
                        batch_idx, seq_idx, q_dim + key_head * key_dim + i, seq_len, qkv_dim)],
                    dk_raw);
            }
            for (int v = tid; v < value_dim; v += blockDim.x) {
                float dv_raw = d_delta[v] * beta_value;
                atomicAdd(
                    &dqkv_conv[la_idx3(
                        batch_idx,
                        seq_idx,
                        v_offset + value_head * value_dim + v,
                        seq_len,
                        qkv_dim)],
                    dv_raw);
            }
            __syncthreads();

            for (int idx = tid; idx < state_elems; idx += blockDim.x) {
                grad_state[idx] = grad_state[idx] * exp_g_value;
            }
            __syncthreads();
        }
    }
}

// ── Staged chunk-parallel backward ───────────────────────────────────────────
// The reverse grad-state
// recurrence is affine in the incoming grad-state:
//   G_{t-1} = gamma_t (I - beta_t k_t k_t^T) (G_t + q_t dcore_t^T)
// so a chunk's action collapses to G_out = M_c G_in + B_c with
//   M_c = A_{t0} A_{t0+1} ... A_{t1},  B_c = sum_t (A_{t0}...A_t) q_t dcore_t^T,
// both computable chunk-locally during a forward recompute walk (stage 1,
// parallel over chunk x head). Only the num_chunks-step boundary carry is
// sequential (stage 2); stage 3 replays the exact per-token grad pass per chunk
// in parallel with the carried boundary grad-state.

// Stage 1: per (chunk, row) block, recompute the chunk from its saved incoming
// state and emit the transfer operator M_c [key_dim x key_dim] and offset B_c
// [key_dim x value_dim]. No grad outputs are written here.
extern "C" __global__ void linear_attention_chunk_transfer_f32(
    float* __restrict__ m_out,
    float* __restrict__ b_out,
    float* __restrict__ state_scratch,
    const float* __restrict__ upstream,
    const float* __restrict__ z,
    const float* __restrict__ norm_weight,
    const unsigned short* __restrict__ qkv_conv,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const float* __restrict__ chunk_state,
    int batch,
    int seq_len,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int value_dim,
    int qkv_dim,
    float eps
) {
    constexpr int BT = 64;
    int rows = batch * num_value_heads;
    int item = blockIdx.x;
    int chunk_idx = item / rows;
    int row = item - chunk_idx * rows;
    int batch_idx = row / num_value_heads;
    int value_head = row - (batch_idx * num_value_heads);
    if (batch_idx >= batch || key_dim > LINEAR_ATTENTION_MAX_DIM ||
        value_dim > LINEAR_ATTENTION_MAX_DIM) {
        return;
    }
    int base = chunk_idx * BT;
    int limit = seq_len - base;
    if (limit > BT) {
        limit = BT;
    }
    if (limit <= 0) {
        return;
    }

    int tid = threadIdx.x;
    int q_dim = num_key_heads * key_dim;
    int v_offset = q_dim + q_dim;
    int key_head = value_head * num_key_heads / num_value_heads;
    int state_elems = key_dim * value_dim;
    int m_elems = key_dim * key_dim;
    float* m_mat = m_out + item * m_elems;
    float* b_mat = b_out + item * state_elems;
    float* state_cur = state_scratch + item * state_elems;
    const float* chunk_state_base =
        chunk_state + (chunk_idx * num_value_heads + value_head) * state_elems;

    __shared__ float q_raw[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float k_raw[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float q_vec[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float k_vec[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float v_raw[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float delta[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float core_out[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float dcore[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float mvec[LINEAR_ATTENTION_MAX_DIM];
    __shared__ float reduce[LINEAR_ATTENTION_BLOCK];
    __shared__ float scalar0;

    for (int idx = tid; idx < m_elems; idx += blockDim.x) {
        m_mat[idx] = idx / key_dim == idx % key_dim ? 1.0f : 0.0f;
    }
    for (int idx = tid; idx < state_elems; idx += blockDim.x) {
        b_mat[idx] = 0.0f;
        state_cur[idx] = chunk_state_base[idx];
    }
    __syncthreads();

    float q_scale = rsqrtf((float)key_dim);

    for (int local_t = 0; local_t < limit; ++local_t) {
        int seq_idx = base + local_t;
        for (int i = tid; i < key_dim; i += blockDim.x) {
            q_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
                batch_idx, seq_idx, key_head * key_dim + i, seq_len, qkv_dim)]);
            k_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
                batch_idx, seq_idx, q_dim + key_head * key_dim + i, seq_len, qkv_dim)]);
        }
        for (int i = tid; i < value_dim; i += blockDim.x) {
            v_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
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

        float exp_g_value =
            expf(g[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)]);
        float beta_value =
            beta[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
        for (int idx = tid; idx < state_elems; idx += blockDim.x) {
            state_cur[idx] *= exp_g_value;
        }
        __syncthreads();

        for (int v = tid; v < value_dim; v += blockDim.x) {
            float accum = 0.0f;
            for (int k = 0; k < key_dim; ++k) {
                accum += state_cur[k * value_dim + v] * k_vec[k];
            }
            delta[v] = (v_raw[v] - accum) * beta_value;
        }
        __syncthreads();

        for (int idx = tid; idx < state_elems; idx += blockDim.x) {
            int k = idx / value_dim;
            int v = idx - k * value_dim;
            state_cur[idx] += k_vec[k] * delta[v];
        }
        __syncthreads();

        // dcore at token t — identical math to the reverse grad pass; it feeds
        // only the affine offset B_c here (dz/dnorm stay in the grad stage).
        for (int v = tid; v < value_dim; v += blockDim.x) {
            float accum = 0.0f;
            for (int k = 0; k < key_dim; ++k) {
                accum += state_cur[k * value_dim + v] * q_vec[k];
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
            unsigned long long out_idx = la_idx4(
                batch_idx, seq_idx, value_head, v, seq_len, num_value_heads, value_dim);
            float dcore_v = upstream[out_idx] * la_silu(z[out_idx]);
            dcore[v] = dcore_v;
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

        // M <- gamma (M - beta (M k) k^T), i.e. compose A_t on the right.
        for (int k = tid; k < key_dim; k += blockDim.x) {
            float accum = 0.0f;
            for (int j = 0; j < key_dim; ++j) {
                accum += m_mat[k * key_dim + j] * k_vec[j];
            }
            mvec[k] = accum;
        }
        __syncthreads();
        for (int idx = tid; idx < m_elems; idx += blockDim.x) {
            int k = idx / key_dim;
            int j = idx - k * key_dim;
            m_mat[idx] = exp_g_value * (m_mat[idx] - beta_value * mvec[k] * k_vec[j]);
        }
        __syncthreads();

        // B += (M q_t) dcore_t^T with the updated M (= A_{t0}..A_t).
        for (int k = tid; k < key_dim; k += blockDim.x) {
            float accum = 0.0f;
            for (int j = 0; j < key_dim; ++j) {
                accum += m_mat[k * key_dim + j] * q_vec[j];
            }
            mvec[k] = accum;
        }
        __syncthreads();
        for (int idx = tid; idx < state_elems; idx += blockDim.x) {
            int k = idx / value_dim;
            int v = idx - k * value_dim;
            b_mat[idx] += mvec[k] * dcore[v];
        }
        __syncthreads();
    }
}

// Stage 2: sequential reverse chunk-boundary carry, one block per row.
// g_in[c] is the grad-state entering chunk c from the right; the slot for the
// last chunk arrives pre-zeroed. g_in[c-1] = M_c g_in[c] + B_c.
extern "C" __global__ void linear_attention_chunk_carry_f32(
    float* __restrict__ g_in,
    const float* __restrict__ m_mats,
    const float* __restrict__ b_mats,
    int rows,
    int num_chunks,
    int key_dim,
    int value_dim
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x;
    int state_elems = key_dim * value_dim;
    int m_elems = key_dim * key_dim;
    for (int c = num_chunks - 1; c >= 1; --c) {
        const float* m_mat = m_mats + (c * rows + row) * m_elems;
        const float* b_mat = b_mats + (c * rows + row) * state_elems;
        const float* g_cur = g_in + (c * rows + row) * state_elems;
        float* g_prev = g_in + ((c - 1) * rows + row) * state_elems;
        for (int idx = tid; idx < state_elems; idx += blockDim.x) {
            int k = idx / value_dim;
            int v = idx - k * value_dim;
            float accum = b_mat[idx];
            for (int j = 0; j < key_dim; ++j) {
                accum += m_mat[k * key_dim + j] * g_cur[j * value_dim + v];
            }
            g_prev[idx] = accum;
        }
        __syncthreads();
    }
}

// Stage 3: exact per-token grad pass per chunk, parallel over `wave` chunk
// lanes x rows; each block strides chunks by `wave` and re-derives the chunk
// history from the saved incoming state (the per-block slab keeps scratch at
// grid x 64 x state instead of seq x state). Same math as the monolithic scan
// with grad_state seeded from the stage-2 carry instead of carried in-block.
extern "C" __global__ void linear_attention_chunk_grad_f32(
    float* __restrict__ dqkv_conv,
    float* __restrict__ dz,
    float* __restrict__ db,
    float* __restrict__ da,
    float* __restrict__ ddt,
    float* __restrict__ da_log,
    float* __restrict__ dnorm,
    float* __restrict__ grad_state_scratch,
    float* __restrict__ chunk_history_scratch,
    float* __restrict__ chunk_kv_scratch,
    const float* __restrict__ upstream,
    const float* __restrict__ z,
    const float* __restrict__ a_proj,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const float* __restrict__ norm_weight,
    const unsigned short* __restrict__ qkv_conv,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const float* __restrict__ chunk_state,
    const float* __restrict__ g_in,
    int batch,
    int seq_len,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int value_dim,
    int qkv_dim,
    int wave,
    float eps
) {
    constexpr int BT = 64;
    int rows = batch * num_value_heads;
    int lane = blockIdx.x / rows;
    int row = blockIdx.x - lane * rows;
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
    int num_chunks = (seq_len + BT - 1) / BT;
    float* grad_state = grad_state_scratch + blockIdx.x * state_elems;
    float* history = chunk_history_scratch + blockIdx.x * BT * state_elems;
    float* kv_chunk = chunk_kv_scratch + blockIdx.x * BT * value_dim;

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

    float exp_a = expf(a_log[value_head]);
    float q_scale = rsqrtf((float)key_dim);

    for (int chunk_idx = lane; chunk_idx < num_chunks; chunk_idx += wave) {
        int base = chunk_idx * BT;
        int limit = seq_len - base;
        if (limit > BT) {
            limit = BT;
        }
        if (limit <= 0) {
            continue;
        }

        const float* chunk_state_base =
            chunk_state + (chunk_idx * num_value_heads + value_head) * state_elems;
        const float* g_in_chunk = g_in + (chunk_idx * rows + row) * state_elems;
        for (int idx = tid; idx < state_elems; idx += blockDim.x) {
            grad_state[idx] = g_in_chunk[idx];
        }
        __syncthreads();

        for (int local_t = 0; local_t < limit; ++local_t) {
            int seq_idx = base + local_t;
            for (int i = tid; i < key_dim; i += blockDim.x) {
                k_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
                    batch_idx, seq_idx, q_dim + key_head * key_dim + i, seq_len, qkv_dim)]);
            }
            for (int i = tid; i < value_dim; i += blockDim.x) {
                v_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
                    batch_idx, seq_idx, v_offset + value_head * value_dim + i, seq_len, qkv_dim)]);
            }
            __syncthreads();

            float local_k_sq = 0.0f;
            for (int i = tid; i < key_dim; i += blockDim.x) {
                local_k_sq += k_raw[i] * k_raw[i];
            }
            float k_norm =
                sqrtf(la_block_sum(reduce, tid, blockDim.x, local_k_sq) + 1.0e-12f);
            for (int i = tid; i < key_dim; i += blockDim.x) {
                k_vec[i] = k_raw[i] / k_norm;
            }
            __syncthreads();

            const float* prev =
                local_t == 0 ? chunk_state_base : history + (local_t - 1) * state_elems;
            float* cur = history + local_t * state_elems;
            float exp_g_value =
                expf(g[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)]);
            for (int idx = tid; idx < state_elems; idx += blockDim.x) {
                cur[idx] = prev[idx] * exp_g_value;
            }
            __syncthreads();

            float beta_value =
                beta[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
            for (int v = tid; v < value_dim; v += blockDim.x) {
                float accum = 0.0f;
                for (int k = 0; k < key_dim; ++k) {
                    accum += cur[k * value_dim + v] * k_vec[k];
                }
                kv_chunk[local_t * value_dim + v] = accum;
                delta[v] = (v_raw[v] - accum) * beta_value;
            }
            __syncthreads();

            for (int idx = tid; idx < state_elems; idx += blockDim.x) {
                int k = idx / value_dim;
                int v = idx - k * value_dim;
                cur[idx] += k_vec[k] * delta[v];
            }
            __syncthreads();
        }

        for (int local_t = limit - 1; local_t >= 0; --local_t) {
            int seq_idx = base + local_t;
            for (int i = tid; i < key_dim; i += blockDim.x) {
                q_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
                    batch_idx, seq_idx, key_head * key_dim + i, seq_len, qkv_dim)]);
                k_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
                    batch_idx, seq_idx, q_dim + key_head * key_dim + i, seq_len, qkv_dim)]);
            }
            for (int i = tid; i < value_dim; i += blockDim.x) {
                v_raw[i] = la_bf16_to_float(qkv_conv[la_idx3(
                    batch_idx, seq_idx, v_offset + value_head * value_dim + i, seq_len, qkv_dim)]);
            }
            __syncthreads();

            float local_q_sq = 0.0f;
            float local_k_sq = 0.0f;
            for (int i = tid; i < key_dim; i += blockDim.x) {
                local_q_sq += q_raw[i] * q_raw[i];
                local_k_sq += k_raw[i] * k_raw[i];
            }
            float q_norm =
                sqrtf(la_block_sum(reduce, tid, blockDim.x, local_q_sq) + 1.0e-12f);

            float k_norm =
                sqrtf(la_block_sum(reduce, tid, blockDim.x, local_k_sq) + 1.0e-12f);

            for (int i = tid; i < key_dim; i += blockDim.x) {
                q_vec[i] = q_scale * q_raw[i] / q_norm;
                k_vec[i] = k_raw[i] / k_norm;
            }
            __syncthreads();

            const float* state = history + local_t * state_elems;
            const float* prev_state =
                local_t == 0 ? chunk_state_base : history + (local_t - 1) * state_elems;

            for (int v = tid; v < value_dim; v += blockDim.x) {
                float accum = 0.0f;
                for (int k = 0; k < key_dim; ++k) {
                    accum += state[k * value_dim + v] * q_vec[k];
                }
                core_out[v] = accum;
            }
            __syncthreads();

            unsigned long long out_base = la_idx4(
                batch_idx, seq_idx, value_head, 0, seq_len, num_value_heads, value_dim);
            LaCoreRowF32 core{core_out, dcore};
            la_rms_gated_backward_row(
                core, reduce, dz + out_base, dnorm, z + out_base, upstream + out_base,
                norm_weight, tid, blockDim.x, value_dim, eps);
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

            float beta_value =
                beta[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)];
            float exp_g_value =
                expf(g[la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads)]);
            for (int v = tid; v < value_dim; v += blockDim.x) {
                float kv = kv_chunk[local_t * value_dim + v];
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
                float kv = kv_chunk[local_t * value_dim + v];
                local_dbeta += d_delta[v] * (v_raw[v] - kv);
                dkv_mem[v] = -d_delta[v] * beta_value;
            }
            float dbeta_scalar = la_block_sum(reduce, tid, blockDim.x, local_dbeta);

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
                local_dexp_g += prev_state[idx] * grad_state[idx];
            }
            float dexp_g = la_block_sum(reduce, tid, blockDim.x, local_dexp_g);

            if (tid == 0) {
                la_gate_param_backward(
                    da, ddt, da_log, db, a_proj, dt_bias,
                    la_idx3(batch_idx, seq_idx, value_head, seq_len, num_value_heads),
                    value_head, exp_a, dexp_g * exp_g_value, dbeta_scalar, beta_value);
            }

            float local_q_dot = 0.0f;
            float local_k_dot = 0.0f;
            for (int i = tid; i < key_dim; i += blockDim.x) {
                local_q_dot += q_raw[i] * dq_vec[i];
                local_k_dot += k_raw[i] * dk_vec[i];
            }
            float q_dot = la_block_sum(reduce, tid, blockDim.x, local_q_dot);

            float k_dot = la_block_sum(reduce, tid, blockDim.x, local_k_dot);
            float q_norm_cubed = q_norm * q_norm * q_norm;
            float k_norm_cubed = k_norm * k_norm * k_norm;

            for (int i = tid; i < key_dim; i += blockDim.x) {
                float dq_raw =
                    q_scale * la_l2_norm_grad(dq_vec[i], q_raw[i], q_dot, q_norm, q_norm_cubed);
                float dk_raw = la_l2_norm_grad(dk_vec[i], k_raw[i], k_dot, k_norm, k_norm_cubed);
                atomicAdd(
                    &dqkv_conv[la_idx3(batch_idx, seq_idx, key_head * key_dim + i, seq_len, qkv_dim)],
                    dq_raw);
                atomicAdd(
                    &dqkv_conv[la_idx3(
                        batch_idx, seq_idx, q_dim + key_head * key_dim + i, seq_len, qkv_dim)],
                    dk_raw);
            }
            for (int v = tid; v < value_dim; v += blockDim.x) {
                float dv_raw = d_delta[v] * beta_value;
                atomicAdd(
                    &dqkv_conv[la_idx3(
                        batch_idx,
                        seq_idx,
                        v_offset + value_head * value_dim + v,
                        seq_len,
                        qkv_dim)],
                    dv_raw);
            }
            __syncthreads();

            for (int idx = tid; idx < state_elems; idx += blockDim.x) {
                grad_state[idx] = grad_state[idx] * exp_g_value;
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ void linear_attention_conv1d_silu_backward_f32(
    float* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    const float* __restrict__ grad_post_silu,
    const float* __restrict__ preact,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    unsigned long long total,
    int channels,
    int seq_len,
    int kernel_size,
    const float* __restrict__ conv_tail,
    int tail_len
) {
    unsigned long long idx =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int c = idx % channels;
    int t = idx / channels;
    float dpre = grad_post_silu[idx] * la_silu_grad(preact[idx]);
    for (int tap = 0; tap < kernel_size; ++tap) {
        int src_t = t + tap + 1 - kernel_size;
        if (src_t < 0) {
            // Carried history did contribute to the forward → its weight-grad is real. Carry is
            // frozen (requires_grad=false), so no grad_input flows back into it.
            if (conv_tail != nullptr) {
                atomicAdd(&grad_weight[c * kernel_size + tap],
                          dpre * conv_tail[(src_t + tail_len) * channels + c]);
            }
            continue;
        }
        unsigned long long input_idx = static_cast<unsigned long long>(src_t) * channels + c;
        float x = input[input_idx];
        float w = weight[c * kernel_size + tap];
        atomicAdd(&grad_input[input_idx], dpre * w);
        atomicAdd(&grad_weight[c * kernel_size + tap], dpre * x);
    }
}

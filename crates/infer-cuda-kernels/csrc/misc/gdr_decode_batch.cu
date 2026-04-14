#include "common.cuh"
#include <cmath>

// ============================================================================
// Batched Gated Delta Rule Decode — B requests in one kernel launch
//
// 2D grid: (num_value_heads, B). Each block handles one (head, request) pair.
// Same 512-thread j-slice parallelism as the single-request kernel — proven
// memory access pattern with perfect val_dim coalescing.
//
// Key differences from single-request kernel:
//   1. 2D grid: blockIdx.y selects request within batch
//   2. Per-request state via pointer array (no gather/scatter overhead)
//   3. QKV/b/a read from contiguous [B, dim] batch buffers
//   4. Output written to contiguous [B, num_value_heads * val_dim]
//
// Grid:  (num_value_heads, B)
// Block: 512 threads (128 val × 4 j_slices)
// ============================================================================

#define GDR_B_KEY_DIM 128
#define GDR_B_VAL_DIM 128
#define GDR_B_J_SLICES 4
#define GDR_B_BLOCK_DIM (GDR_B_VAL_DIM * GDR_B_J_SLICES)   // 512
#define GDR_B_J_PER_SLICE (GDR_B_KEY_DIM / GDR_B_J_SLICES)  // 32

__global__ void gdr_decode_batch_kernel(
    const __nv_bfloat16* __restrict__ qkv_batch,     // [B, q_dim + k_dim + v_dim]
    const __nv_bfloat16* __restrict__ b_proj_batch,  // [B, num_value_heads]
    const __nv_bfloat16* __restrict__ a_proj_batch,  // [B, num_value_heads]
    const __nv_bfloat16* __restrict__ dt_bias,       // [num_value_heads] (shared across batch)
    const float* __restrict__ A_log,                 // [num_value_heads] (shared across batch)
    float** __restrict__ state_ptrs,                 // [B] → [num_value_heads, key_dim, val_dim]
    __nv_bfloat16* __restrict__ output_batch,        // [B, num_value_heads * val_dim]
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int val_dim
) {
    int v_head = blockIdx.x;
    int b = blockIdx.y;
    int val_idx = threadIdx.x & 0x7F;
    int j_slice = threadIdx.x >> 7;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1F;

    int k_head = v_head * num_key_heads / num_value_heads;
    int q_dim_total = key_dim * num_key_heads;
    int k_dim_total = q_dim_total;
    int qkv_stride = q_dim_total + k_dim_total + val_dim * num_value_heads;

    __shared__ float smem_q[GDR_B_KEY_DIM];
    __shared__ float smem_k[GDR_B_KEY_DIM];
    __shared__ float smem_norm[2];
    __shared__ float warp_norms[GDR_B_BLOCK_DIM / WARP_SIZE];  // 16
    __shared__ float s_exp_g;
    __shared__ float s_beta;
    __shared__ float smem_kv_partial[GDR_B_J_SLICES][GDR_B_VAL_DIM];
    __shared__ float smem_out_partial[GDR_B_J_SLICES][GDR_B_VAL_DIM];

    // Read this request's QKV
    const __nv_bfloat16* my_qkv = qkv_batch + b * qkv_stride;
    float q_val = __bfloat162float(my_qkv[k_head * key_dim + val_idx]);
    float k_val = __bfloat162float(my_qkv[q_dim_total + k_head * key_dim + val_idx]);
    float v_val = __bfloat162float(my_qkv[q_dim_total + k_dim_total + v_head * val_dim + val_idx]);

    // ====================================================================
    // L2 normalize q and k — only j_slice=0 contributes
    // ====================================================================
    float q_sq = (j_slice == 0) ? q_val * q_val : 0.0f;
    q_sq = warp_reduce_sum(q_sq);
    if (lane_id == 0) warp_norms[warp_id] = q_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = warp_norms[0] + warp_norms[1] + warp_norms[2] + warp_norms[3];
        smem_norm[0] = rsqrtf(total + 1e-12f);
    }

    float k_sq = (j_slice == 0) ? k_val * k_val : 0.0f;
    k_sq = warp_reduce_sum(k_sq);
    if (lane_id == 0) warp_norms[warp_id] = k_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = warp_norms[0] + warp_norms[1] + warp_norms[2] + warp_norms[3];
        smem_norm[1] = rsqrtf(total + 1e-12f);
    }
    __syncthreads();

    q_val *= smem_norm[0] * rsqrtf((float)key_dim);
    k_val *= smem_norm[1];

    if (j_slice == 0) {
        smem_q[val_idx] = q_val;
        smem_k[val_idx] = k_val;
    }

    // ====================================================================
    // Compute g and beta for this (value_head, request)
    // ====================================================================
    if (threadIdx.x == 0) {
        float a_val = __bfloat162float(a_proj_batch[b * num_value_heads + v_head]);
        float b_val = __bfloat162float(b_proj_batch[b * num_value_heads + v_head]);
        float bias = __bfloat162float(dt_bias[v_head]);
        float a_log = A_log[v_head];

        float x = a_val + bias;
        float softplus_x = (x > 20.0f) ? x : logf(1.0f + expf(x));
        float g = -expf(a_log) * softplus_x;
        s_exp_g = expf(g);
        s_beta = 1.0f / (1.0f + expf(-b_val));
    }
    __syncthreads();

    float exp_g = s_exp_g;
    float beta = s_beta;

    // ====================================================================
    // State pointer — per-request
    // ====================================================================
    float* my_state = state_ptrs[b] + v_head * key_dim * val_dim;

    int j_start = j_slice * GDR_B_J_PER_SLICE;
    int j_end = j_start + GDR_B_J_PER_SLICE;

    // ====================================================================
    // Pass 1: Decay + partial kv_mem
    // ====================================================================
    float partial_kv = 0.0f;
    for (int j = j_start; j < j_end; j++) {
        float s = my_state[j * val_dim + val_idx];
        s *= exp_g;
        my_state[j * val_dim + val_idx] = s;
        partial_kv += s * smem_k[j];
    }

    smem_kv_partial[j_slice][val_idx] = partial_kv;
    __syncthreads();

    float kv_mem = smem_kv_partial[0][val_idx] + smem_kv_partial[1][val_idx]
                 + smem_kv_partial[2][val_idx] + smem_kv_partial[3][val_idx];

    float my_delta = (v_val - kv_mem) * beta;

    // ====================================================================
    // Pass 2: Rank-1 update + partial output
    // ====================================================================
    float partial_out = 0.0f;
    for (int j = j_start; j < j_end; j++) {
        float s = my_state[j * val_dim + val_idx];
        s += my_delta * smem_k[j];
        my_state[j * val_dim + val_idx] = s;
        partial_out += s * smem_q[j];
    }

    smem_out_partial[j_slice][val_idx] = partial_out;
    __syncthreads();

    if (j_slice == 0) {
        float out = smem_out_partial[0][val_idx] + smem_out_partial[1][val_idx]
                   + smem_out_partial[2][val_idx] + smem_out_partial[3][val_idx];
        int out_stride = num_value_heads * val_dim;
        output_batch[b * out_stride + v_head * val_dim + val_idx] = __float2bfloat16(out);
    }
}

extern "C" {

cudaError_t gdr_decode_batch_cuda(
    const __nv_bfloat16* qkv_batch,
    const __nv_bfloat16* b_proj_batch,
    const __nv_bfloat16* a_proj_batch,
    const __nv_bfloat16* dt_bias,
    const float* A_log,
    float** state_ptrs,
    __nv_bfloat16* output_batch,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int val_dim,
    int batch_size,
    cudaStream_t stream
) {
    dim3 grid(num_value_heads, batch_size);
    gdr_decode_batch_kernel<<<grid, GDR_B_BLOCK_DIM, 0, stream>>>(
        qkv_batch, b_proj_batch, a_proj_batch, dt_bias, A_log,
        state_ptrs, output_batch,
        num_key_heads, num_value_heads, key_dim, val_dim
    );
    return cudaGetLastError();
}

} // extern "C"

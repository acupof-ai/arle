#include "common.cuh"

#define PREFILL_PAGED_HD128 128
#define PREFILL_PAGED_HD256 256
#define PREFILL_PAGED_NUM_WARPS_HD128 (PREFILL_PAGED_HD128 / WARP_SIZE)
#define PREFILL_PAGED_NUM_WARPS_HD256 (PREFILL_PAGED_HD256 / WARP_SIZE)

__global__ void prefill_attention_paged_qk_norm_rope_hd128_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int q_dim,
    int kv_dim,
    int start_pos,
    float eps) {
  int head_global = blockIdx.x;
  int token = blockIdx.y;
  int d = threadIdx.x;

  bool is_q = head_global < num_q_heads;
  int head_local = is_q ? head_global : (head_global - num_q_heads);
  __nv_bfloat16* data = is_q ? q : k;
  int dim_stride = is_q ? q_dim : kv_dim;
  const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

  int offset = head_local * PREFILL_PAGED_HD128 + d + token * dim_stride;
  float val = __bfloat162float(data[offset]);

  float sq = warp_reduce_sum(val * val);
  int warp_id = d / WARP_SIZE;
  int lane_id = d % WARP_SIZE;

  __shared__ float warp_sums[PREFILL_PAGED_NUM_WARPS_HD128];
  if (lane_id == 0) {
    warp_sums[warp_id] = sq;
  }
  __syncthreads();

  __shared__ float inv_rms;
  if (warp_id == 0) {
    float warp_sum = (lane_id < PREFILL_PAGED_NUM_WARPS_HD128) ? warp_sums[lane_id] : 0.0f;
    float total = warp_reduce_sum(warp_sum);
    if (lane_id == 0) {
      inv_rms = rsqrtf(total / PREFILL_PAGED_HD128 + eps);
    }
  }
  __syncthreads();

  float normed = val * inv_rms * __bfloat162float(norm_w[d]);
  __shared__ float smem[PREFILL_PAGED_HD128];
  smem[d] = normed;
  __syncthreads();

  int pos = start_pos + token;
  int half = PREFILL_PAGED_HD128 / 2;
  __nv_bfloat16 result;
  if (d < half) {
    float lo = smem[d];
    float hi = smem[d + half];
    float c = __bfloat162float(cos_cache[pos * PREFILL_PAGED_HD128 + d]);
    float s = __bfloat162float(sin_cache[pos * PREFILL_PAGED_HD128 + d]);
    result = __float2bfloat16(lo * c - hi * s);
  } else {
    int pair = d - half;
    float lo = smem[pair];
    float hi = smem[d];
    float c = __bfloat162float(cos_cache[pos * PREFILL_PAGED_HD128 + pair]);
    float s = __bfloat162float(sin_cache[pos * PREFILL_PAGED_HD128 + pair]);
    result = __float2bfloat16(lo * s + hi * c);
  }

  data[offset] = result;
}

__global__ void prefill_attention_paged_kv_write_hd128_kernel(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int* __restrict__ page_table,
    int page_size,
    __nv_bfloat16* __restrict__ k_pool,
    __nv_bfloat16* __restrict__ v_pool,
    int num_kv_heads,
    int seq_len,
    int kv_dim,
    int start_pos) {
  int kv_head = blockIdx.x;
  int token = blockIdx.y;
  int d = threadIdx.x;

  int src_offset = kv_head * PREFILL_PAGED_HD128 + d + token * kv_dim;
  int logical_pos = start_pos + token;
  int physical_page = page_table[logical_pos / page_size];
  int token_in_page = logical_pos % page_size;
  int stride_page = num_kv_heads * page_size * PREFILL_PAGED_HD128;
  int pool_offset = physical_page * stride_page + kv_head * page_size * PREFILL_PAGED_HD128 +
                    token_in_page * PREFILL_PAGED_HD128 + d;

  k_pool[pool_offset] = k[src_offset];
  v_pool[pool_offset] = v[src_offset];
}

__device__ __forceinline__ float prefill_attention_paged_rms_norm_offset_hd256(
    float val,
    float weight,
    float eps,
    int tid) {
  float sq_sum = warp_reduce_sum(val * val);

  __shared__ float scratch[PREFILL_PAGED_NUM_WARPS_HD256];
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if (lane_id == 0) {
    scratch[warp_id] = sq_sum;
  }
  __syncthreads();

  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < PREFILL_PAGED_NUM_WARPS_HD256; ++i) {
      total += scratch[i];
    }
    scratch[0] = 1.0f / sqrtf(total / PREFILL_PAGED_HD256 + eps);
  }
  __syncthreads();

  return val * scratch[0] * (1.0f + weight);
}

__device__ __forceinline__ float prefill_attention_paged_apply_rope_partial_hd256(
    float* smem,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    int pos,
    int tid,
    int rotary_dim) {
  int half_rotary = rotary_dim / 2;
  if (tid < half_rotary) {
    float cos_val = __bfloat162float(cos_cache[pos * rotary_dim + tid]);
    float sin_val = __bfloat162float(sin_cache[pos * rotary_dim + tid]);
    return smem[tid] * cos_val - smem[tid + half_rotary] * sin_val;
  }
  if (tid < rotary_dim) {
    int pair = tid - half_rotary;
    float cos_val = __bfloat162float(cos_cache[pos * rotary_dim + pair]);
    float sin_val = __bfloat162float(sin_cache[pos * rotary_dim + pair]);
    return smem[pair] * sin_val + smem[tid] * cos_val;
  }
  return smem[tid];
}

__global__ void prefill_attention_paged_hd256_kernel(
    const __nv_bfloat16* __restrict__ q_full_batch,
    __nv_bfloat16* __restrict__ q_out_batch,
    const __nv_bfloat16* __restrict__ k_batch,
    const __nv_bfloat16* __restrict__ v_batch,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    const int* __restrict__ page_table,
    int page_size,
    __nv_bfloat16* __restrict__ k_pool,
    __nv_bfloat16* __restrict__ v_pool,
    int num_qo_heads,
    int num_kv_heads,
    int seq_len,
    int start_pos,
    int rotary_dim,
    float rms_eps) {
  int kv_head_idx = blockIdx.x;
  int token = blockIdx.y;
  int tid = threadIdx.x;
  int gqa_ratio = num_qo_heads / num_kv_heads;
  int pos = start_pos + token;

  __shared__ float smem_rope[PREFILL_PAGED_HD256];
  float q_norm_w = __bfloat162float(q_norm_weight[tid]);
  float k_norm_w = __bfloat162float(k_norm_weight[tid]);

  int q_full_dim = num_qo_heads * PREFILL_PAGED_HD256 * 2;
  int q_dim = num_qo_heads * PREFILL_PAGED_HD256;

  for (int g = 0; g < gqa_ratio; ++g) {
    int q_head = kv_head_idx * gqa_ratio + g;
    int q_src = token * q_full_dim + q_head * 2 * PREFILL_PAGED_HD256 + tid;

    float q_val = __bfloat162float(q_full_batch[q_src]);
    float q_normed =
        prefill_attention_paged_rms_norm_offset_hd256(q_val, q_norm_w, rms_eps, tid);

    smem_rope[tid] = q_normed;
    __syncthreads();

    float q_roped = prefill_attention_paged_apply_rope_partial_hd256(
        smem_rope, cos_cache, sin_cache, pos, tid, rotary_dim);
    __syncthreads();

    int q_dst = token * q_dim + q_head * PREFILL_PAGED_HD256 + tid;
    q_out_batch[q_dst] = __float2bfloat16(q_roped);
  }

  int kv_dim = num_kv_heads * PREFILL_PAGED_HD256;
  int kv_src = token * kv_dim + kv_head_idx * PREFILL_PAGED_HD256 + tid;
  float k_val = __bfloat162float(k_batch[kv_src]);
  float k_normed =
      prefill_attention_paged_rms_norm_offset_hd256(k_val, k_norm_w, rms_eps, tid);

  smem_rope[tid] = k_normed;
  __syncthreads();

  float k_roped = prefill_attention_paged_apply_rope_partial_hd256(
      smem_rope, cos_cache, sin_cache, pos, tid, rotary_dim);
  float v_val = __bfloat162float(v_batch[kv_src]);

  int physical_page = page_table[pos / page_size];
  int token_in_page = pos % page_size;
  int stride_page = num_kv_heads * page_size * PREFILL_PAGED_HD256;
  int pool_offset = physical_page * stride_page +
                    kv_head_idx * page_size * PREFILL_PAGED_HD256 +
                    token_in_page * PREFILL_PAGED_HD256 + tid;

  k_pool[pool_offset] = __float2bfloat16(k_roped);
  v_pool[pool_offset] = __float2bfloat16(v_val);
}

extern "C" {

cudaError_t prefill_attention_paged_prep_cuda(
    __nv_bfloat16* q_batch,
    __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    const int* page_table,
    int page_size,
    __nv_bfloat16* k_pool,
    __nv_bfloat16* v_pool,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int start_pos,
    float rms_eps,
    cudaStream_t stream) {
  (void)head_dim;
  int q_dim = num_q_heads * PREFILL_PAGED_HD128;
  int kv_dim = num_kv_heads * PREFILL_PAGED_HD128;

  dim3 norm_grid(num_q_heads + num_kv_heads, seq_len);
  prefill_attention_paged_qk_norm_rope_hd128_kernel<<<norm_grid, PREFILL_PAGED_HD128, 0, stream>>>(
      q_batch,
      k_batch,
      q_norm_weight,
      k_norm_weight,
      cos_cache,
      sin_cache,
      num_q_heads,
      num_kv_heads,
      seq_len,
      q_dim,
      kv_dim,
      start_pos,
      rms_eps);

  dim3 cache_grid(num_kv_heads, seq_len);
  prefill_attention_paged_kv_write_hd128_kernel<<<cache_grid, PREFILL_PAGED_HD128, 0, stream>>>(
      k_batch,
      v_batch,
      page_table,
      page_size,
      k_pool,
      v_pool,
      num_kv_heads,
      seq_len,
      kv_dim,
      start_pos);
  return cudaGetLastError();
}

cudaError_t prefill_attention_paged_prep_hd256_cuda(
    const __nv_bfloat16* q_full_batch,
    __nv_bfloat16* q_out_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    const int* page_table,
    int page_size,
    __nv_bfloat16* k_pool,
    __nv_bfloat16* v_pool,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int start_pos,
    int rotary_dim,
    float rms_eps,
    cudaStream_t stream) {
  dim3 grid(num_kv_heads, seq_len);
  prefill_attention_paged_hd256_kernel<<<grid, PREFILL_PAGED_HD256, 0, stream>>>(
      q_full_batch,
      q_out_batch,
      k_batch,
      v_batch,
      q_norm_weight,
      k_norm_weight,
      cos_cache,
      sin_cache,
      page_table,
      page_size,
      k_pool,
      v_pool,
      num_q_heads,
      num_kv_heads,
      seq_len,
      start_pos,
      rotary_dim,
      rms_eps);
  return cudaGetLastError();
}

}  // extern "C"

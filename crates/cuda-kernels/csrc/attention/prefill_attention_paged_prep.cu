#include "common.cuh"

#define PREFILL_PAGED_HD128 128
#define PREFILL_PAGED_NUM_WARPS_HD128 (PREFILL_PAGED_HD128 / WARP_SIZE)

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
    const int* __restrict__ start_pos_ptr,
    float eps) {
  int start_pos = *start_pos_ptr;
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
    const int* __restrict__ page_table_offset_ptr,
    int page_size,
    __nv_bfloat16* __restrict__ k_pool,
    __nv_bfloat16* __restrict__ v_pool,
    int num_kv_heads,
    int seq_len,
    int kv_dim,
    const int* __restrict__ start_pos_ptr) {
  int start_pos = *start_pos_ptr;
  int kv_head = blockIdx.x;
  int token = blockIdx.y;
  int d = threadIdx.x;

  int src_offset = kv_head * PREFILL_PAGED_HD128 + d + token * kv_dim;
  int logical_pos = start_pos + token;
  int page_table_offset = *page_table_offset_ptr;
  int physical_page = page_table[page_table_offset + logical_pos / page_size];
  int token_in_page = logical_pos % page_size;
  int stride_page = num_kv_heads * page_size * PREFILL_PAGED_HD128;
  int pool_offset = physical_page * stride_page + kv_head * page_size * PREFILL_PAGED_HD128 +
                    token_in_page * PREFILL_PAGED_HD128 + d;

  k_pool[pool_offset] = k[src_offset];
  v_pool[pool_offset] = v[src_offset];
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
    const int* page_table_offset_ptr,
    int page_size,
    __nv_bfloat16* k_pool,
    __nv_bfloat16* v_pool,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    const int* start_pos_ptr,
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
      start_pos_ptr,
      rms_eps);

  dim3 cache_grid(num_kv_heads, seq_len);
  prefill_attention_paged_kv_write_hd128_kernel<<<cache_grid, PREFILL_PAGED_HD128, 0, stream>>>(
      k_batch,
      v_batch,
      page_table,
      page_table_offset_ptr,
      page_size,
      k_pool,
      v_pool,
      num_kv_heads,
      seq_len,
      kv_dim,
      start_pos_ptr);
  return cudaGetLastError();
}

}  // extern "C"

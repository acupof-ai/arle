#include "common.cuh"

#define PREFILL_PAGED_HD256 256

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
    const int* __restrict__ start_pos_ptr,
    int rotary_dim,
    float rms_eps) {
  int start_pos = *start_pos_ptr;
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
        rms_norm_hd256(q_val, q_norm_w, rms_eps, tid);

    smem_rope[tid] = q_normed;
    __syncthreads();

    float q_roped = apply_rope_partial_hd256(
        smem_rope, cos_cache, sin_cache, pos, tid, rotary_dim);
    __syncthreads();

    int q_dst = token * q_dim + q_head * PREFILL_PAGED_HD256 + tid;
    q_out_batch[q_dst] = __float2bfloat16(q_roped);
  }

  int kv_dim = num_kv_heads * PREFILL_PAGED_HD256;
  int kv_src = token * kv_dim + kv_head_idx * PREFILL_PAGED_HD256 + tid;
  float k_val = __bfloat162float(k_batch[kv_src]);
  float k_normed =
      rms_norm_hd256(k_val, k_norm_w, rms_eps, tid);

  smem_rope[tid] = k_normed;
  __syncthreads();

  float k_roped = apply_rope_partial_hd256(
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
    const int* start_pos_ptr,
    int rotary_dim,
    float rms_eps,
    cudaStream_t stream) {
  if (q_full_batch == nullptr || q_out_batch == nullptr || k_batch == nullptr ||
      v_batch == nullptr || q_norm_weight == nullptr || k_norm_weight == nullptr ||
      cos_cache == nullptr || sin_cache == nullptr || page_table == nullptr ||
      k_pool == nullptr || v_pool == nullptr || start_pos_ptr == nullptr ||
      num_q_heads <= 0 || num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0 ||
      page_size <= 0 || seq_len < 0 || rotary_dim <= 0 ||
      rotary_dim > PREFILL_PAGED_HD256 || rotary_dim % 2 != 0) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0) {
    return cudaSuccess;
  }
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
      start_pos_ptr,
      rotary_dim,
      rms_eps);
  return cudaGetLastError();
}

}  // extern "C"

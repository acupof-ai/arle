// 2D context-parallel ring prefill kernels for Qwen3.6 HD256 full attention.
//
// The ring pass replaces the replicated-KV gather: each rank preps its own
// q-slice and KV slice into DENSE head-major buffers, rotates the KV slice
// around the cp ring, and scatters only the pages it owns (logical page
// `g` on shard `g % cp`) into the sharded paged pool as the blocks pass
// through. Norm+RoPE helpers live in common.cuh (shared with the paged
// prefill prep).
#include "common.cuh"

#define RING_PREFILL_HD 256
#define RING_PREFILL_NUM_WARPS (RING_PREFILL_HD / WARP_SIZE)

// Dense prep: q/k-norm + partial RoPE into DENSE head-major buffers
// `[heads, rows, hd]` (the ring core's tile-major layout), with NO pool write.
// `q_full` is the gated raw Q projection `[rows, q_heads*2*hd]`; `k_in`/`v_in`
// are `[rows, kv_heads*hd]`. One block per (kv_head, token); the block handles
// its gqa q heads plus its own k/v.
__global__ void ring_prefill_dense_prep_hd256_kernel(
    const __nv_bfloat16* __restrict__ q_full,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int num_qo_heads, int num_kv_heads, int rows, int start_pos,
    int rotary_dim, float rms_eps) {
  int kv_head_idx = blockIdx.x;
  int token = blockIdx.y;
  int tid = threadIdx.x;
  int gqa_ratio = num_qo_heads / num_kv_heads;
  int pos = start_pos + token;

  __shared__ float smem_rope[RING_PREFILL_HD];
  float q_norm_w = __bfloat162float(q_norm_weight[tid]);
  float k_norm_w = __bfloat162float(k_norm_weight[tid]);

  int q_full_dim = num_qo_heads * 2 * RING_PREFILL_HD;
  for (int g = 0; g < gqa_ratio; ++g) {
    int q_head = kv_head_idx * gqa_ratio + g;
    int q_src = token * q_full_dim + q_head * 2 * RING_PREFILL_HD + tid;
    float q_val = __bfloat162float(q_full[q_src]);
    float q_normed =
        rms_norm_hd256(q_val, q_norm_w, rms_eps, tid);
    smem_rope[tid] = q_normed;
    __syncthreads();
    float q_roped = apply_rope_partial_hd256(
        smem_rope, cos_cache, sin_cache, pos, tid, rotary_dim);
    __syncthreads();
    q_out[q_head * rows * RING_PREFILL_HD + token * RING_PREFILL_HD + tid] =
        __float2bfloat16(q_roped);
  }

  int kv_dim = num_kv_heads * RING_PREFILL_HD;
  int kv_src = token * kv_dim + kv_head_idx * RING_PREFILL_HD + tid;
  float k_val = __bfloat162float(k_in[kv_src]);
  float k_normed =
      rms_norm_hd256(k_val, k_norm_w, rms_eps, tid);
  smem_rope[tid] = k_normed;
  __syncthreads();
  float k_roped = apply_rope_partial_hd256(
      smem_rope, cos_cache, sin_cache, pos, tid, rotary_dim);
  __syncthreads();
  k_out[kv_head_idx * rows * RING_PREFILL_HD + token * RING_PREFILL_HD + tid] =
      __float2bfloat16(k_roped);
  v_out[kv_head_idx * rows * RING_PREFILL_HD + token * RING_PREFILL_HD + tid] =
      __bfloat162float(v_in[kv_src]);
}

// Block-cyclic scatter: write the current ring block's tokens whose global
// page `g = abs_pos / page_size` is owned by this shard (`g % cp_size ==
// cp_rank`) into the sharded HND pool. `k_dense`/`v_dense` are the block's
// prepped K/V `[kv_heads, blk_len, hd]` head-major (compact — the buffer may
// be pad-sized but only `blk_len` rows per head are live). `local_page_table`
// holds this shard's physical pages in local-index order (entry `j` backs
// global page `cp_rank + j*cp_size`).
__global__ void ring_prefill_scatter_sharded_hd256_kernel(
    const __nv_bfloat16* __restrict__ k_dense,
    const __nv_bfloat16* __restrict__ v_dense,
    const int* __restrict__ local_page_table,
    int local_page_count,
    int page_size,
    int kv_heads,
    int blk_start,
    int blk_len,
    int cp_rank,
    int cp_size,
    int stride_page,
    __nv_bfloat16* __restrict__ k_pool,
    __nv_bfloat16* __restrict__ v_pool) {
  int kv_head = blockIdx.x;
  int token = blockIdx.y;
  int tid = threadIdx.x;
  if (token >= blk_len) return;
  int abs_pos = blk_start + token;
  int g = abs_pos / page_size;
  if (g % cp_size != cp_rank) return;
  int j = g / cp_size;
  if (j >= local_page_count) return;
  int physical_page = local_page_table[j];
  int token_in_page = abs_pos % page_size;
  int pool_offset = physical_page * stride_page +
                    kv_head * page_size * RING_PREFILL_HD +
                    token_in_page * RING_PREFILL_HD + tid;
  int dense_offset =
      kv_head * blk_len * RING_PREFILL_HD + token * RING_PREFILL_HD + tid;
  k_pool[pool_offset] = k_dense[dense_offset];
  v_pool[pool_offset] = v_dense[dense_offset];
}

// Finalize the ring accumulator into the gate's row-major bf16 `attn_out`
// `[rows, q_heads*hd]`: `out = O / L` (the flash-2 normalized output),
// transposing from the accumulator's head-major `[q_heads, rows, hd]` layout.
// `acc_l == 0` (a row that saw no keys) writes 0. LSE is not needed in
// serving, so `acc_m` is not an input.
__global__ void ring_prefill_finalize_bf16_hd256_kernel(
    const float* __restrict__ acc_l,
    const float* __restrict__ acc_o,
    __nv_bfloat16* __restrict__ out,
    int q_heads,
    int rows) {
  int row = blockIdx.x;
  int head = blockIdx.y;
  int tid = threadIdx.x;
  int64_t acc_idx =
      (int64_t)head * rows * RING_PREFILL_HD + row * RING_PREFILL_HD + tid;
  float L = acc_l[(int64_t)head * rows + row];
  float v = (L > 0.0f) ? acc_o[acc_idx] / L : 0.0f;
  out[(int64_t)row * q_heads * RING_PREFILL_HD + head * RING_PREFILL_HD + tid] =
      __float2bfloat16(v);
}

extern "C" {

cudaError_t ring_prefill_dense_prep_hd256_cuda(
    const __nv_bfloat16* q_full,
    const __nv_bfloat16* k_in,
    const __nv_bfloat16* v_in,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    __nv_bfloat16* v_out,
    int num_qo_heads,
    int num_kv_heads,
    int rows,
    int start_pos,
    int rotary_dim,
    float rms_eps,
    cudaStream_t stream) {
  if (q_full == nullptr || k_in == nullptr || v_in == nullptr ||
      q_norm_weight == nullptr || k_norm_weight == nullptr ||
      cos_cache == nullptr || sin_cache == nullptr || q_out == nullptr ||
      k_out == nullptr || v_out == nullptr || num_qo_heads <= 0 ||
      num_kv_heads <= 0 || num_qo_heads % num_kv_heads != 0 || rows <= 0 ||
      rotary_dim <= 0 || rotary_dim > RING_PREFILL_HD || rotary_dim % 2 != 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(num_kv_heads, rows);
  ring_prefill_dense_prep_hd256_kernel<<<grid, RING_PREFILL_HD, 0, stream>>>(
      q_full, k_in, v_in, q_norm_weight, k_norm_weight, cos_cache, sin_cache,
      q_out, k_out, v_out, num_qo_heads, num_kv_heads, rows, start_pos,
      rotary_dim, rms_eps);
  return cudaGetLastError();
}

cudaError_t ring_prefill_scatter_sharded_hd256_cuda(
    const __nv_bfloat16* k_dense,
    const __nv_bfloat16* v_dense,
    const int* local_page_table,
    int local_page_count,
    int page_size,
    int kv_heads,
    int blk_start,
    int blk_len,
    int cp_rank,
    int cp_size,
    int stride_page,
    __nv_bfloat16* k_pool,
    __nv_bfloat16* v_pool,
    cudaStream_t stream) {
  if (k_dense == nullptr || v_dense == nullptr || local_page_table == nullptr ||
      k_pool == nullptr || v_pool == nullptr || page_size <= 0 ||
      kv_heads <= 0 || blk_len <= 0 || cp_size <= 1 || cp_rank < 0 ||
      cp_rank >= cp_size || stride_page <= 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(kv_heads, blk_len);
  ring_prefill_scatter_sharded_hd256_kernel<<<grid, RING_PREFILL_HD, 0, stream>>>(
      k_dense, v_dense, local_page_table, local_page_count, page_size, kv_heads,
      blk_start, blk_len, cp_rank, cp_size, stride_page, k_pool, v_pool);
  return cudaGetLastError();
}

cudaError_t ring_prefill_finalize_bf16_hd256_cuda(
    const float* acc_l,
    const float* acc_o,
    __nv_bfloat16* out,
    int q_heads,
    int rows,
    cudaStream_t stream) {
  if (acc_l == nullptr || acc_o == nullptr || out == nullptr || q_heads <= 0 ||
      rows <= 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(rows, q_heads);
  ring_prefill_finalize_bf16_hd256_kernel<<<grid, RING_PREFILL_HD, 0, stream>>>(
      acc_l, acc_o, out, q_heads, rows);
  return cudaGetLastError();
}

}  // extern "C"

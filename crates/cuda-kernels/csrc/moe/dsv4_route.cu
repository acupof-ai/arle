#include "common.cuh"
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define DSV4_ROUTE_BLOCK 256
#define DSV4_CONTIG_ROUTE_ALIGN 128
#define DSV4_ROUTE_MAX_EXPERTS 512
#define DSV4_ROUTE_MAX_TOPK 16

__device__ __forceinline__ float dsv4_route_bf16_to_f32(const uint16_t value) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&value));
}

__device__ __forceinline__ uint16_t dsv4_route_f32_to_bf16_bits(const float value) {
  __nv_bfloat16 out = __float2bfloat16(value);
  return *reinterpret_cast<uint16_t *>(&out);
}

__device__ __forceinline__ int32_t dsv4_route_f32_to_i32_bits(const float value) {
  return __float_as_int(value);
}

__device__ __forceinline__ float dsv4_route_i32_bits_to_f32(const int32_t value) {
  return __int_as_float(value);
}

__device__ __forceinline__ int dsv4_route_local_expert(
    const int32_t *__restrict__ meta,
    int route,
    int local_expert_start,
    int experts_per_rank) {
  int expert = meta[route * 3 + 1];
  int local = expert - local_expert_start;
  return (local >= 0 && local < experts_per_rank) ? local : -1;
}

__device__ __forceinline__ uint16_t dsv4_swiglu_clamped_one(
    uint16_t gate_bits,
    uint16_t up_bits,
    float limit) {
  float gate = dsv4_route_bf16_to_f32(gate_bits);
  float up = dsv4_route_bf16_to_f32(up_bits);
  gate = fminf(gate, limit);
  up = fminf(fmaxf(up, -limit), limit);
  float silu = gate / (1.0f + __expf(-gate));
  return dsv4_route_f32_to_bf16_bits(silu * up);
}

__global__ void dsv4_swiglu_clamped_routes_kernel(
    const uint16_t *__restrict__ gate,
    const uint16_t *__restrict__ up,
    uint16_t *__restrict__ out,
    const int32_t *__restrict__ route_meta,
    int num_routes,
    int hidden_dim,
    int local_expert_start,
    int experts_per_rank,
    float limit) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_routes * hidden_dim;
  if (idx >= total) return;
  int route = idx / hidden_dim;
  if (dsv4_route_local_expert(route_meta, route, local_expert_start, experts_per_rank) < 0) {
    return;
  }
  out[idx] = dsv4_swiglu_clamped_one(gate[idx], up[idx], limit);
}
__device__ __forceinline__ float dsv4_route_sigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + __expf(-value));
  }
  float expv = __expf(value);
  return expv / (1.0f + expv);
}

__device__ __forceinline__ float dsv4_route_softplus(float value) {
  return value > 20.0f ? value : __logf(1.0f + __expf(value));
}

__device__ __forceinline__ float dsv4_route_score(float logit, int scoring_kind) {
  if (scoring_kind == 0) {
    return logit;
  }
  if (scoring_kind == 1) {
    return dsv4_route_sigmoid(logit);
  }
  return sqrtf(dsv4_route_softplus(logit));
}

// Block-wide argmax with tie-break "lower expert wins". Returns (best_score,
// best_expert) broadcast to every thread of the block. Each thread brings one
// candidate via `score`/`expert`; pass -INFINITY/-1 for non-participants.
//
// Implementation: warp-level tournament via __shfl_xor_sync, then one slot per
// warp written to shared memory, then a single warp reduces across warps.
// blockDim.x must be a multiple of 32 and ≤ 1024. Tie-break uses
// `expert_a < expert_b` so lower expert index wins on equal scores — matches
// the prior serial selection-sort behavior.
__device__ __forceinline__ void dsv4_route_block_argmax(
    float score,
    int expert,
    float &best_score,
    int &best_expert) {
  __shared__ float warp_scores[DSV4_ROUTE_BLOCK / 32];
  __shared__ int warp_experts[DSV4_ROUTE_BLOCK / 32];

  // Warp tournament with tie-break.
  unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_score = __shfl_xor_sync(mask, score, offset);
    int other_expert = __shfl_xor_sync(mask, expert, offset);
    bool take_other = other_score > score ||
                      (other_score == score && other_expert >= 0 &&
                       (expert < 0 || other_expert < expert));
    if (take_other) {
      score = other_score;
      expert = other_expert;
    }
  }

  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  if (lane == 0) {
    warp_scores[warp_id] = score;
    warp_experts[warp_id] = expert;
  }
  __syncthreads();

  // Final reduction in warp 0.
  if (warp_id == 0) {
    int num_warps = blockDim.x >> 5;
    if (lane < num_warps) {
      score = warp_scores[lane];
      expert = warp_experts[lane];
    } else {
      score = -INFINITY;
      expert = -1;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other_score = __shfl_xor_sync(mask, score, offset);
      int other_expert = __shfl_xor_sync(mask, expert, offset);
      bool take_other = other_score > score ||
                        (other_score == score && other_expert >= 0 &&
                         (expert < 0 || other_expert < expert));
      if (take_other) {
        score = other_score;
        expert = other_expert;
      }
    }
    if (lane == 0) {
      warp_scores[0] = score;
      warp_experts[0] = expert;
    }
  }
  __syncthreads();

  best_score = warp_scores[0];
  best_expert = warp_experts[0];
  __syncthreads();
}

// Block-wide reduction: max(value) across all threads, broadcast to all.
__device__ __forceinline__ float dsv4_route_block_reduce_max(float value) {
  __shared__ float warp_max[DSV4_ROUTE_BLOCK / 32];
  unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, offset));
  }
  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  if (lane == 0) warp_max[warp_id] = value;
  __syncthreads();
  if (warp_id == 0) {
    int num_warps = blockDim.x >> 5;
    float v = lane < num_warps ? warp_max[lane] : -INFINITY;
    for (int offset = 16; offset > 0; offset >>= 1) {
      v = fmaxf(v, __shfl_xor_sync(mask, v, offset));
    }
    if (lane == 0) warp_max[0] = v;
  }
  __syncthreads();
  float result = warp_max[0];
  __syncthreads();
  return result;
}

// Block-wide reduction: sum(value) across all threads, broadcast to all.
__device__ __forceinline__ float dsv4_route_block_reduce_sum(float value) {
  __shared__ float warp_sum[DSV4_ROUTE_BLOCK / 32];
  unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(mask, value, offset);
  }
  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  if (lane == 0) warp_sum[warp_id] = value;
  __syncthreads();
  if (warp_id == 0) {
    int num_warps = blockDim.x >> 5;
    float v = lane < num_warps ? warp_sum[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_xor_sync(mask, v, offset);
    }
    if (lane == 0) warp_sum[0] = v;
  }
  __syncthreads();
  float result = warp_sum[0];
  __syncthreads();
  return result;
}

__global__ void dsv4_route_kernel(
    const uint16_t *__restrict__ logits,
    const uint16_t *__restrict__ bias,
    const int64_t *__restrict__ tid2eid,
    const uint32_t *__restrict__ token_ids,
    int32_t *__restrict__ indices,
    float *__restrict__ weights,
    int num_tokens,
    int n_experts,
    int topk,
    int routing_kind,
    int scoring_kind,
    float routed_scaling_factor) {
  int token = blockIdx.x;
  if (token >= num_tokens) return;

  // `scores[e]` holds the per-expert normalized probability (scoring_kind==0
  // softmax) or scoring_kind-specific score; this is preserved across the
  // top-k selection so the final renorm can read it. `combined[e]` is the
  // selection key (score + bias) and gets masked with -INFINITY each time an
  // expert is picked so the next argmax round skips it.
  __shared__ float scores[DSV4_ROUTE_MAX_EXPERTS];
  __shared__ float combined[DSV4_ROUTE_MAX_EXPERTS];
  __shared__ float rank_scores[DSV4_ROUTE_MAX_TOPK];
  __shared__ int rank_indices[DSV4_ROUTE_MAX_TOPK];

  int tid = threadIdx.x;

  // ---- Phase 1: per-expert scores in parallel. ------------------------------
  if (scoring_kind == 0) {
    // Block-parallel softmax over n_experts.
    float local_max = -INFINITY;
    for (int e = tid; e < n_experts; e += blockDim.x) {
      float l = dsv4_route_bf16_to_f32(logits[token * n_experts + e]);
      scores[e] = l;  // stash raw logit, overwritten below
      local_max = fmaxf(local_max, l);
    }
    __syncthreads();
    float max_logit = dsv4_route_block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int e = tid; e < n_experts; e += blockDim.x) {
      float v = expf(scores[e] - max_logit);
      scores[e] = v;  // exp(x - max)
      local_sum += v;
    }
    __syncthreads();
    float denom = dsv4_route_block_reduce_sum(local_sum);
    denom = fmaxf(denom, 1.0e-20f);
    float inv_denom = 1.0f / denom;
    for (int e = tid; e < n_experts; e += blockDim.x) {
      scores[e] *= inv_denom;
    }
  } else {
    for (int e = tid; e < n_experts; e += blockDim.x) {
      scores[e] = dsv4_route_score(
          dsv4_route_bf16_to_f32(logits[token * n_experts + e]), scoring_kind);
    }
  }
  __syncthreads();

  // ---- Phase 2: top-k selection. --------------------------------------------
  if (routing_kind == 0) {
    // Read the fixed routing table; topk ≤ 16, single thread suffices.
    if (tid == 0) {
      uint32_t token_id = token_ids[token];
      int64_t base = (int64_t)token_id * topk;
      for (int k = 0; k < topk; ++k) {
        int expert = (int)tid2eid[base + k];
        rank_indices[k] = expert;
        rank_scores[k] = (expert >= 0 && expert < n_experts) ? scores[expert] : 0.0f;
      }
    }
    __syncthreads();
  } else {
    // Build `combined[e] = scores[e] + bias[e]` in parallel. Subsequent
    // argmax loops only touch e < n_experts, so no upper-range init needed.
    for (int e = tid; e < n_experts; e += blockDim.x) {
      combined[e] = scores[e] + dsv4_route_bf16_to_f32(bias[e]);
    }
    __syncthreads();

    // Initialize ranks to "no pick".
    if (tid < topk) {
      rank_scores[tid] = -INFINITY;
      rank_indices[tid] = -1;
    }
    __syncthreads();

    // Repeat `topk` masked block-argmax rounds. Each round selects one expert
    // and masks it with -INFINITY for subsequent rounds. A round that finds
    // no candidate above -INFINITY leaves the rank slot at (-INFINITY, -1) —
    // matches the serial selection-sort behavior when n_experts < topk.
    for (int k = 0; k < topk; ++k) {
      float local_best_score = -INFINITY;
      int local_best_expert = -1;
      for (int e = tid; e < n_experts; e += blockDim.x) {
        float s = combined[e];
        // Strictly greater wins; tie-break (lower expert wins) only applies
        // among real candidates, never among -INFINITY-masked slots.
        bool take = s > local_best_score ||
                    (s == local_best_score && s > -INFINITY &&
                     local_best_expert >= 0 && e < local_best_expert) ||
                    (s == local_best_score && s > -INFINITY &&
                     local_best_expert < 0);
        if (take) {
          local_best_score = s;
          local_best_expert = e;
        }
      }
      float best_score;
      int best_expert;
      dsv4_route_block_argmax(local_best_score, local_best_expert, best_score,
                              best_expert);
      if (tid == 0) {
        if (best_expert >= 0 && best_score > -INFINITY) {
          rank_scores[k] = best_score;
          rank_indices[k] = best_expert;
          combined[best_expert] = -INFINITY;
        }
      }
      __syncthreads();
    }
  }

  // ---- Phase 3: renorm + write out. -----------------------------------------
  // topk ≤ 16 — keep the serial path here, branch-light, single thread.
  if (tid == 0) {
    float selected_sum = 0.0f;
    if (scoring_kind != 0) {
      for (int k = 0; k < topk; ++k) {
        int expert = rank_indices[k];
        if (expert >= 0 && expert < n_experts) {
          selected_sum += scores[expert];
        }
      }
    }
    float denom = scoring_kind == 0 ? 1.0f : selected_sum + 1.0e-9f;
    for (int k = 0; k < topk; ++k) {
      int expert = rank_indices[k];
      indices[token * topk + k] = expert;
      float score = (expert >= 0 && expert < n_experts) ? scores[expert] : 0.0f;
      weights[token * topk + k] = score / denom * routed_scaling_factor;
    }
  }
}

extern "C" CUresult dsv4_route_cuda(
    const uint16_t *logits,
    const uint16_t *bias,
    const int64_t *tid2eid,
    const uint32_t *token_ids,
    int32_t *indices,
    float *weights,
    int num_tokens,
    int n_experts,
    int topk,
    int routing_kind,
    int scoring_kind,
    float routed_scaling_factor,
    CUstream stream) {
  if (num_tokens < 0 || n_experts <= 0 || n_experts > DSV4_ROUTE_MAX_EXPERTS ||
      topk <= 0 || topk > DSV4_ROUTE_MAX_TOPK || routing_kind < 0 ||
      routing_kind > 1 || scoring_kind < 0 || scoring_kind > 2) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  dsv4_route_kernel<<<num_tokens, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      logits, bias, tid2eid, token_ids, indices, weights, num_tokens, n_experts,
      topk, routing_kind, scoring_kind, routed_scaling_factor);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_count_local_experts_kernel(
    const int32_t *__restrict__ indices,
    int32_t *__restrict__ counts,
    int num_tokens,
    int topk,
    int local_expert_start,
    int experts_per_rank) {
  int route = blockIdx.x * blockDim.x + threadIdx.x;
  int total_routes = num_tokens * topk;
  if (route >= total_routes) return;
  int expert = indices[route];
  int local = expert - local_expert_start;
  if (local >= 0 && local < experts_per_rank) {
    atomicAdd(&counts[local], 1);
  }
}

extern "C" CUresult dsv4_count_local_experts_cuda(
    const int32_t *indices,
    int32_t *counts,
    int num_tokens,
    int topk,
    int local_expert_start,
    int experts_per_rank,
    CUstream stream) {
  if (num_tokens < 0 || topk <= 0 || local_expert_start < 0 ||
      experts_per_rank <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total_routes = num_tokens * topk;
  if (total_routes == 0) return CUDA_SUCCESS;
  int grid = (total_routes + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_count_local_experts_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      indices, counts, num_tokens, topk, local_expert_start, experts_per_rank);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_exclusive_scan_i32_kernel(
    const int32_t *__restrict__ counts,
    int32_t *__restrict__ offsets,
    int32_t *__restrict__ total,
    int n) {
  __shared__ int32_t values[DSV4_ROUTE_BLOCK];
  int tid = threadIdx.x;
  int value = (tid < n) ? counts[tid] : 0;
  values[tid] = value;
  __syncthreads();

  for (int stride = 1; stride < DSV4_ROUTE_BLOCK; stride <<= 1) {
    int add = (tid >= stride) ? values[tid - stride] : 0;
    __syncthreads();
    values[tid] += add;
    __syncthreads();
  }

  if (tid < n) {
    offsets[tid] = values[tid] - value;
  }
  if (tid == 0 && total != nullptr) {
    total[0] = (n > 0) ? values[n - 1] : 0;
  }
}

extern "C" CUresult dsv4_exclusive_scan_i32_cuda(
    const int32_t *counts,
    int32_t *offsets,
    int32_t *total,
    int n,
    CUstream stream) {
  if (n < 0 || n > DSV4_ROUTE_BLOCK) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) return CUDA_SUCCESS;
  dsv4_exclusive_scan_i32_kernel<<<1, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      counts, offsets, total, n);
  return (CUresult)cudaGetLastError();
}

// Exclusive prefix-sum over per-expert counts with each group's span padded
// up to `alignment` rows: `offsets[g] = sum_{g' < g} align(counts[g'])`.
// This is the DeepGEMM m-grouped-contiguous layout builder: BLOCK_M = 128
// output tiles resolve their B group from `m_indices[tile_start]`, so every
// group's row segment must start 128-aligned (alignment = 128). Empty groups
// contribute zero rows. `total` (optional) receives the aligned row total.
__global__ void moe_exclusive_scan_aligned_i32_kernel(
    const int32_t *__restrict__ counts,
    int32_t *__restrict__ offsets,
    int32_t *__restrict__ total,
    int n,
    int alignment) {
  __shared__ int32_t values[DSV4_ROUTE_BLOCK];
  int tid = threadIdx.x;
  int value = 0;
  if (tid < n) {
    int count = counts[tid];
    value = ((count + alignment - 1) / alignment) * alignment;
  }
  values[tid] = value;
  __syncthreads();

  for (int stride = 1; stride < DSV4_ROUTE_BLOCK; stride <<= 1) {
    int add = (tid >= stride) ? values[tid - stride] : 0;
    __syncthreads();
    values[tid] += add;
    __syncthreads();
  }

  if (tid < n) {
    offsets[tid] = values[tid] - value;
  }
  if (tid == 0 && total != nullptr) {
    total[0] = (n > 0) ? values[n - 1] : 0;
  }
}

extern "C" CUresult moe_exclusive_scan_aligned_i32_cuda(
    const int32_t *counts,
    int32_t *offsets,
    int32_t *total,
    int n,
    int alignment,
    CUstream stream) {
  if (n < 0 || n > DSV4_ROUTE_BLOCK || alignment <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) return CUDA_SUCCESS;
  moe_exclusive_scan_aligned_i32_kernel<<<1, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      counts, offsets, total, n, alignment);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_pack_local_experts_with_slots_kernel(
    const uint16_t *__restrict__ hidden,
    const int32_t *__restrict__ indices,
    const float *__restrict__ weights,
    const int32_t *__restrict__ offsets,
    int32_t *__restrict__ cursors,
    uint16_t *__restrict__ packed_hidden,
    int32_t *__restrict__ packed_route_slot,
    float *__restrict__ packed_weight,
    int num_tokens,
    int hidden_dim,
    int topk,
    int local_expert_start,
    int experts_per_rank) {
  int route = blockIdx.x;
  int total_routes = num_tokens * topk;
  if (route >= total_routes) return;

  int token = route / topk;
  int expert = indices[route];
  int local = expert - local_expert_start;
  if (local < 0 || local >= experts_per_rank) return;

  __shared__ int slot;
  if (threadIdx.x == 0) {
    slot = offsets[local] + atomicAdd(&cursors[local], 1);
    packed_route_slot[slot] = route;
    packed_weight[slot] = weights[route];
  }
  __syncthreads();

  int src_base = token * hidden_dim;
  int dst_base = slot * hidden_dim;
  for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    packed_hidden[dst_base + col] = hidden[src_base + col];
  }
}

extern "C" CUresult dsv4_pack_local_experts_with_slots_cuda(
    const uint16_t *hidden,
    const int32_t *indices,
    const float *weights,
    const int32_t *offsets,
    int32_t *cursors,
    uint16_t *packed_hidden,
    int32_t *packed_route_slot,
    float *packed_weight,
    int num_tokens,
    int hidden_dim,
    int topk,
    int local_expert_start,
    int experts_per_rank,
    CUstream stream) {
  if (num_tokens < 0 || hidden_dim <= 0 || topk <= 0 ||
      local_expert_start < 0 || experts_per_rank <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total_routes = num_tokens * topk;
  if (total_routes == 0) return CUDA_SUCCESS;
  dsv4_pack_local_experts_with_slots_kernel<<<total_routes, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      hidden, indices, weights, offsets, cursors, packed_hidden, packed_route_slot,
      packed_weight, num_tokens, hidden_dim, topk, local_expert_start, experts_per_rank);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_fill_m_indices_from_counts_kernel(
    const int32_t *__restrict__ counts,
    const int32_t *__restrict__ offsets,
    int32_t *__restrict__ m_indices,
    int experts_per_rank,
    int row_capacity) {
  int local = blockIdx.x;
  if (local >= experts_per_rank) return;
  int count = counts[local];
  int offset = offsets[local];
  for (int row = threadIdx.x; row < count; row += blockDim.x) {
    int dst = offset + row;
    if (dst < row_capacity) {
      m_indices[dst] = local;
    }
  }
}

extern "C" CUresult dsv4_fill_m_indices_from_counts_cuda(
    const int32_t *counts,
    const int32_t *offsets,
    int32_t *m_indices,
    int experts_per_rank,
    int row_capacity,
    CUstream stream) {
  if (counts == nullptr || offsets == nullptr || m_indices == nullptr ||
      experts_per_rank <= 0 || row_capacity <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dsv4_fill_m_indices_from_counts_kernel<<<experts_per_rank, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      counts, offsets, m_indices, experts_per_rank, row_capacity);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_pack_local_experts_with_slots_and_indices_kernel(
    const uint16_t *__restrict__ hidden,
    const int32_t *__restrict__ indices,
    const float *__restrict__ weights,
    const int32_t *__restrict__ offsets,
    int32_t *__restrict__ cursors,
    uint16_t *__restrict__ packed_hidden,
    int32_t *__restrict__ packed_route_slot,
    float *__restrict__ packed_weight,
    int32_t *__restrict__ packed_m_indices,
    int num_tokens,
    int hidden_dim,
    int topk,
    int local_expert_start,
    int experts_per_rank) {
  int route = blockIdx.x;
  int total_routes = num_tokens * topk;
  if (route >= total_routes) return;

  int token = route / topk;
  int expert = indices[route];
  int local = expert - local_expert_start;
  if (local < 0 || local >= experts_per_rank) return;

  (void)offsets;
  (void)cursors;
  int slot = route * DSV4_CONTIG_ROUTE_ALIGN;
  if (threadIdx.x == 0) {
    packed_route_slot[slot] = route;
    packed_weight[slot] = weights[route];
    packed_m_indices[slot] = local;
  }

  int src_base = token * hidden_dim;
  int dst_base = slot * hidden_dim;
  for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    packed_hidden[dst_base + col] = hidden[src_base + col];
  }
}
__global__ void dsv4_scatter_all_route_slots_kernel(
    const uint16_t *__restrict__ expert_out,
    uint16_t *__restrict__ route_out,
    const int32_t *__restrict__ expert_route_slot,
    const float *__restrict__ expert_weight,
    int num_routes,
    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_routes * hidden_dim;
  if (idx >= total) return;
  int route = idx / hidden_dim;
  int col = idx - route * hidden_dim;
  int route_slot = expert_route_slot[route];
  if (route_slot < 0) return;
  float weight = expert_weight[route];
  float value = dsv4_route_bf16_to_f32(expert_out[idx]);
  route_out[route_slot * hidden_dim + col] =
      dsv4_route_f32_to_bf16_bits(weight * value);
}

extern "C" CUresult dsv4_scatter_all_route_slots_cuda(
    const uint16_t *expert_out,
    uint16_t *route_out,
    const int32_t *expert_route_slot,
    const float *expert_weight,
    int num_routes,
    int hidden_dim,
    CUstream stream) {
  if (num_routes < 0 || hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = num_routes * hidden_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_scatter_all_route_slots_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      expert_out, route_out, expert_route_slot, expert_weight, num_routes,
      hidden_dim);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_combine_route_slot_outputs_kernel(
    const uint16_t *__restrict__ route_slot_out,
    uint16_t *__restrict__ routed_out,
    int num_tokens,
    int topk,
    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * hidden_dim;
  if (idx >= total) return;
  int token = idx / hidden_dim;
  int col = idx - token * hidden_dim;
  float sum = 0.0f;
  int route_base = token * topk;
  for (int k = 0; k < topk; ++k) {
    sum += dsv4_route_bf16_to_f32(route_slot_out[(route_base + k) * hidden_dim + col]);
  }
  routed_out[idx] = dsv4_route_f32_to_bf16_bits(sum);
}

extern "C" CUresult dsv4_combine_route_slot_outputs_cuda(
    const uint16_t *route_slot_out,
    uint16_t *routed_out,
    int num_tokens,
    int topk,
    int hidden_dim,
    CUstream stream) {
  if (num_tokens < 0 || topk <= 0 || hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = num_tokens * hidden_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_combine_route_slot_outputs_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      route_slot_out, routed_out, num_tokens, topk, hidden_dim);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_cast_i32_to_i64_kernel(
    const int32_t *__restrict__ src,
    int64_t *__restrict__ dst,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  dst[idx] = static_cast<int64_t>(src[idx]);
}

extern "C" CUresult dsv4_cast_i32_to_i64_cuda(
    const int32_t *src,
    int64_t *dst,
    int n,
    CUstream stream) {
  if (n < 0) return CUDA_ERROR_INVALID_VALUE;
  if (n == 0) return CUDA_SUCCESS;
  int grid = (n + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_cast_i32_to_i64_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      src, dst, n);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_cast_i64_to_i32_kernel(
    const int64_t *__restrict__ src,
    int32_t *__restrict__ dst,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  dst[idx] = static_cast<int32_t>(src[idx]);
}

extern "C" CUresult dsv4_cast_i64_to_i32_cuda(
    const int64_t *src,
    int32_t *dst,
    int n,
    CUstream stream) {
  if (n < 0) return CUDA_ERROR_INVALID_VALUE;
  if (n == 0) return CUDA_SUCCESS;
  int grid = (n + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_cast_i64_to_i32_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      src, dst, n);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_interleave_gate_up_fp8_rows_kernel(
    const uint4 *__restrict__ gate,
    const uint4 *__restrict__ up,
    uint4 *__restrict__ output,
    int rows,
    int vectors_per_row) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * 2 * vectors_per_row;
  if (idx >= total) return;
  int dst_row = idx / vectors_per_row;
  int col = idx % vectors_per_row;
  int lane = dst_row & 15;
  int src_row = (dst_row >> 4) * 8 + (lane & 7);
  const uint4 *src = lane < 8 ? gate : up;
  output[idx] = src[static_cast<int64_t>(src_row) * vectors_per_row + col];
}

extern "C" CUresult dsv4_interleave_gate_up_fp8_rows_cuda(
    const uint8_t *gate,
    const uint8_t *up,
    uint8_t *output,
    int rows,
    int cols,
    CUstream stream) {
  if (gate == nullptr || up == nullptr || output == nullptr || rows <= 0 ||
      (rows & 7) != 0 || cols <= 0 || (cols & 15) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int vectors_per_row = cols / 16;
  int64_t total = static_cast<int64_t>(rows) * 2 * vectors_per_row;
  int grid = static_cast<int>((total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK);
  dsv4_interleave_gate_up_fp8_rows_kernel<<<grid, DSV4_ROUTE_BLOCK, 0,
                                            (cudaStream_t)stream>>>(
      reinterpret_cast<const uint4 *>(gate),
      reinterpret_cast<const uint4 *>(up),
      reinterpret_cast<uint4 *>(output), rows, vectors_per_row);
  return (CUresult)cudaGetLastError();
}

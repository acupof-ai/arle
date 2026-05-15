#include "common.cuh"
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define DSV4_ROUTE_BLOCK 256
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

extern "C" CUresult dsv4_zero_bf16_cuda(
    uint16_t *data,
    int elements,
    CUstream stream) {
  if (elements < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (elements == 0) return CUDA_SUCCESS;
  cudaError_t err =
      cudaMemsetAsync(data, 0, (size_t)elements * sizeof(uint16_t), (cudaStream_t)stream);
  return (CUresult)err;
}

__global__ void dsv4_dequantize_fp8_rows_to_bf16_kernel(
    const uint8_t *__restrict__ input,
    const float *__restrict__ scales,
    uint16_t *__restrict__ output,
    int rows,
    int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx >= total) return;
  int row = idx / cols;
  __nv_fp8_e4m3 fp8;
  *reinterpret_cast<uint8_t *>(&fp8) = input[idx];
  float value = static_cast<float>(fp8) * scales[row];
  output[idx] = dsv4_route_f32_to_bf16_bits(value);
}

extern "C" CUresult dsv4_dequantize_fp8_rows_to_bf16_cuda(
    const uint8_t *input,
    const float *scales,
    uint16_t *output,
    int rows,
    int cols,
    CUstream stream) {
  if (rows < 0 || cols <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = rows * cols;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_dequantize_fp8_rows_to_bf16_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      input, scales, output, rows, cols);
  return (CUresult)cudaGetLastError();
}

__device__ __forceinline__ uint16_t dsv4_swiglu_clamped_one(
    uint16_t gate_bits,
    uint16_t up_bits,
    float limit) {
  float gate = dsv4_route_bf16_to_f32(gate_bits);
  float up = dsv4_route_bf16_to_f32(up_bits);
  gate = fminf(gate, limit);
  up = fminf(fmaxf(up, -limit), limit);
  float silu = gate / (1.0f + expf(-gate));
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

extern "C" CUresult dsv4_swiglu_clamped_routes_cuda(
    const uint16_t *gate,
    const uint16_t *up,
    uint16_t *out,
    const int32_t *route_meta,
    int num_routes,
    int hidden_dim,
    int local_expert_start,
    int experts_per_rank,
    float limit,
    CUstream stream) {
  if (num_routes < 0 || hidden_dim <= 0 || local_expert_start < 0 ||
      experts_per_rank <= 0 || !(limit > 0.0f)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = num_routes * hidden_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_swiglu_clamped_routes_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      gate, up, out, route_meta, num_routes, hidden_dim, local_expert_start,
      experts_per_rank, limit);
  return (CUresult)cudaGetLastError();
}

__device__ __forceinline__ float dsv4_route_sigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + expf(-value));
  }
  float expv = expf(value);
  return expv / (1.0f + expv);
}

__device__ __forceinline__ float dsv4_route_softplus(float value) {
  return value > 20.0f ? value : log1pf(expf(value));
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
  __shared__ float scores[DSV4_ROUTE_MAX_EXPERTS];
  __shared__ float rank_scores[DSV4_ROUTE_MAX_TOPK];
  __shared__ int rank_indices[DSV4_ROUTE_MAX_TOPK];

  if (threadIdx.x == 0) {
    if (scoring_kind == 0) {
      float max_logit = -INFINITY;
      for (int expert = 0; expert < n_experts; ++expert) {
        max_logit = fmaxf(max_logit, dsv4_route_bf16_to_f32(logits[token * n_experts + expert]));
      }
      float denom = 0.0f;
      for (int expert = 0; expert < n_experts; ++expert) {
        float value = expf(dsv4_route_bf16_to_f32(logits[token * n_experts + expert]) - max_logit);
        scores[expert] = value;
        denom += value;
      }
      denom = fmaxf(denom, 1.0e-20f);
      for (int expert = 0; expert < n_experts; ++expert) {
        scores[expert] /= denom;
      }
    } else {
      for (int expert = 0; expert < n_experts; ++expert) {
        scores[expert] = dsv4_route_score(
            dsv4_route_bf16_to_f32(logits[token * n_experts + expert]), scoring_kind);
      }
    }

    for (int k = 0; k < topk; ++k) {
      rank_scores[k] = -INFINITY;
      rank_indices[k] = -1;
    }

    if (routing_kind == 0) {
      uint32_t token_id = token_ids[token];
      int64_t base = (int64_t)token_id * topk;
      for (int k = 0; k < topk; ++k) {
        int expert = (int)tid2eid[base + k];
        rank_indices[k] = expert;
        rank_scores[k] = (expert >= 0 && expert < n_experts) ? scores[expert] : 0.0f;
      }
    } else {
      for (int expert = 0; expert < n_experts; ++expert) {
        float top_score = scores[expert] + dsv4_route_bf16_to_f32(bias[expert]);
        for (int k = 0; k < topk; ++k) {
          bool better = top_score > rank_scores[k] ||
                        (top_score == rank_scores[k] && expert < rank_indices[k]);
          if (!better) continue;
          for (int shift = topk - 1; shift > k; --shift) {
            rank_scores[shift] = rank_scores[shift - 1];
            rank_indices[shift] = rank_indices[shift - 1];
          }
          rank_scores[k] = top_score;
          rank_indices[k] = expert;
          break;
        }
      }
    }

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

__global__ void dsv4_count_expert_ranks_kernel(
    const int32_t *__restrict__ indices,
    int32_t *__restrict__ counts,
    int num_tokens,
    int topk,
    int experts_per_rank,
    int ep_world_size) {
  int route = blockIdx.x * blockDim.x + threadIdx.x;
  int total_routes = num_tokens * topk;
  if (route >= total_routes) return;
  int expert = indices[route];
  if (expert < 0) return;
  int rank = expert / experts_per_rank;
  if (rank >= 0 && rank < ep_world_size) {
    atomicAdd(&counts[rank], 1);
  }
}

extern "C" CUresult dsv4_count_expert_ranks_cuda(
    const int32_t *indices,
    int32_t *counts,
    int num_tokens,
    int topk,
    int experts_per_rank,
    int ep_world_size,
    CUstream stream) {
  if (num_tokens < 0 || topk <= 0 || experts_per_rank <= 0 ||
      ep_world_size <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total_routes = num_tokens * topk;
  if (total_routes == 0) return CUDA_SUCCESS;
  int grid = (total_routes + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_count_expert_ranks_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      indices, counts, num_tokens, topk, experts_per_rank, ep_world_size);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_pack_local_experts_kernel(
    const uint16_t *__restrict__ hidden,
    const int32_t *__restrict__ indices,
    const float *__restrict__ weights,
    const int32_t *__restrict__ offsets,
    int32_t *__restrict__ cursors,
    uint16_t *__restrict__ packed_hidden,
    int32_t *__restrict__ packed_token,
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
    packed_token[slot] = token;
    packed_weight[slot] = weights[route];
  }
  __syncthreads();

  int src_base = token * hidden_dim;
  int dst_base = slot * hidden_dim;
  for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    packed_hidden[dst_base + col] = hidden[src_base + col];
  }
}

extern "C" CUresult dsv4_pack_local_experts_cuda(
    const uint16_t *hidden,
    const int32_t *indices,
    const float *weights,
    const int32_t *offsets,
    int32_t *cursors,
    uint16_t *packed_hidden,
    int32_t *packed_token,
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
  dsv4_pack_local_experts_kernel<<<total_routes, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      hidden, indices, weights, offsets, cursors, packed_hidden, packed_token,
      packed_weight, num_tokens, hidden_dim, topk, local_expert_start,
      experts_per_rank);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_scatter_packed_expert_kernel(
    const uint16_t *__restrict__ expert_out,
    uint16_t *__restrict__ routed_out,
    const int32_t *__restrict__ packed_token,
    const float *__restrict__ packed_weight,
    int start_slot,
    int count,
    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = count * hidden_dim;
  if (idx >= total) return;
  int row = idx / hidden_dim;
  int col = idx - row * hidden_dim;
  int slot = start_slot + row;
  int token = packed_token[slot];
  float weight = packed_weight[slot];
  int out_idx = token * hidden_dim + col;
  float prev = dsv4_route_bf16_to_f32(routed_out[out_idx]);
  float value = dsv4_route_bf16_to_f32(expert_out[idx]);
  routed_out[out_idx] = dsv4_route_f32_to_bf16_bits(prev + weight * value);
}

extern "C" CUresult dsv4_scatter_packed_expert_cuda(
    const uint16_t *expert_out,
    uint16_t *routed_out,
    const int32_t *packed_token,
    const float *packed_weight,
    int start_slot,
    int count,
    int hidden_dim,
    CUstream stream) {
  if (start_slot < 0 || count < 0 || hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = count * hidden_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_scatter_packed_expert_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      expert_out, routed_out, packed_token, packed_weight, start_slot, count,
      hidden_dim);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_pack_expert_ranks_kernel(
    const uint16_t *__restrict__ hidden,
    const int32_t *__restrict__ indices,
    const float *__restrict__ weights,
    const int32_t *__restrict__ offsets,
    int32_t *__restrict__ cursors,
    uint16_t *__restrict__ packed_hidden,
    int32_t *__restrict__ packed_token,
    int32_t *__restrict__ packed_route_slot,
    int32_t *__restrict__ packed_meta,
    int num_tokens,
    int hidden_dim,
    int topk,
    int experts_per_rank,
    int ep_world_size) {
  int route = blockIdx.x;
  int total_routes = num_tokens * topk;
  if (route >= total_routes) return;
  int token = route / topk;
  int expert = indices[route];
  if (expert < 0) return;
  int rank = expert / experts_per_rank;
  if (rank < 0 || rank >= ep_world_size) return;

  __shared__ int slot;
  if (threadIdx.x == 0) {
    slot = offsets[rank] + atomicAdd(&cursors[rank], 1);
    packed_token[slot] = token;
    packed_route_slot[slot] = route;
    int meta_base = slot * 3;
    packed_meta[meta_base] = token;
    packed_meta[meta_base + 1] = expert;
    packed_meta[meta_base + 2] = dsv4_route_f32_to_i32_bits(weights[route]);
  }
  __syncthreads();

  int src_base = token * hidden_dim;
  int dst_base = slot * hidden_dim;
  for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    packed_hidden[dst_base + col] = hidden[src_base + col];
  }
}

extern "C" CUresult dsv4_pack_expert_ranks_cuda(
    const uint16_t *hidden,
    const int32_t *indices,
    const float *weights,
    const int32_t *offsets,
    int32_t *cursors,
    uint16_t *packed_hidden,
    int32_t *packed_token,
    int32_t *packed_route_slot,
    int32_t *packed_meta,
    int num_tokens,
    int hidden_dim,
    int topk,
    int experts_per_rank,
    int ep_world_size,
    CUstream stream) {
  if (num_tokens < 0 || hidden_dim <= 0 || topk <= 0 ||
      experts_per_rank <= 0 || ep_world_size <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total_routes = num_tokens * topk;
  if (total_routes == 0) return CUDA_SUCCESS;
  dsv4_pack_expert_ranks_kernel<<<total_routes, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      hidden, indices, weights, offsets, cursors, packed_hidden, packed_token,
      packed_route_slot, packed_meta, num_tokens, hidden_dim, topk, experts_per_rank,
      ep_world_size);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_pack_dispatch_payload_kernel(
    const uint16_t *__restrict__ hidden,
    const int32_t *__restrict__ meta,
    uint16_t *__restrict__ payload,
    int num_routes,
    int hidden_dim,
    int stride_elems) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int meta_words = 3 * (int)sizeof(int32_t) / (int)sizeof(uint16_t);
  int64_t total = (int64_t)num_routes * stride_elems;
  if (idx >= total) return;

  int route = (int)(idx / stride_elems);
  int word = (int)(idx - (int64_t)route * stride_elems);
  if (word < hidden_dim) {
    payload[idx] = hidden[(int64_t)route * hidden_dim + word];
  } else if (word < hidden_dim + meta_words) {
    const uint8_t *meta_bytes_ptr = reinterpret_cast<const uint8_t *>(meta);
    int meta_word = word - hidden_dim;
    int64_t byte_offset = (int64_t)route * 3 * (int)sizeof(int32_t) +
                          meta_word * (int)sizeof(uint16_t);
    payload[idx] = (uint16_t)meta_bytes_ptr[byte_offset] |
                   ((uint16_t)meta_bytes_ptr[byte_offset + 1] << 8);
  } else {
    payload[idx] = 0;
  }
}

extern "C" CUresult dsv4_pack_dispatch_payload_cuda(
    const uint16_t *hidden,
    const int32_t *meta,
    uint16_t *payload,
    int num_routes,
    int hidden_dim,
    int stride_elems,
    CUstream stream) {
  if (num_routes < 0 || hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int meta_words = 3 * (int)sizeof(int32_t) / (int)sizeof(uint16_t);
  if (stride_elems < hidden_dim + meta_words) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int64_t total = (int64_t)num_routes * stride_elems;
  if (total == 0) return CUDA_SUCCESS;
  int64_t grid64 = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  if (grid64 > INT32_MAX) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dsv4_pack_dispatch_payload_kernel<<<(int)grid64, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      hidden, meta, payload, num_routes, hidden_dim, stride_elems);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_unpack_dispatch_payload_kernel(
    const uint16_t *__restrict__ payload,
    uint16_t *__restrict__ hidden,
    int32_t *__restrict__ meta,
    int num_routes,
    int hidden_dim,
    int stride_elems) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int meta_words = 3 * (int)sizeof(int32_t) / (int)sizeof(uint16_t);
  int64_t total = (int64_t)num_routes * stride_elems;
  if (idx >= total) return;

  int route = (int)(idx / stride_elems);
  int word = (int)(idx - (int64_t)route * stride_elems);
  if (word < hidden_dim) {
    hidden[(int64_t)route * hidden_dim + word] = payload[idx];
  } else if (word < hidden_dim + meta_words) {
    uint8_t *meta_bytes_ptr = reinterpret_cast<uint8_t *>(meta);
    int meta_word = word - hidden_dim;
    int64_t byte_offset = (int64_t)route * 3 * (int)sizeof(int32_t) +
                          meta_word * (int)sizeof(uint16_t);
    uint16_t value = payload[idx];
    meta_bytes_ptr[byte_offset] = (uint8_t)(value & 0xffu);
    meta_bytes_ptr[byte_offset + 1] = (uint8_t)(value >> 8);
  }
}

extern "C" CUresult dsv4_unpack_dispatch_payload_cuda(
    const uint16_t *payload,
    uint16_t *hidden,
    int32_t *meta,
    int num_routes,
    int hidden_dim,
    int stride_elems,
    CUstream stream) {
  if (num_routes < 0 || hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int meta_words = 3 * (int)sizeof(int32_t) / (int)sizeof(uint16_t);
  if (stride_elems < hidden_dim + meta_words) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int64_t total = (int64_t)num_routes * stride_elems;
  if (total == 0) return CUDA_SUCCESS;
  int64_t grid64 = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  if (grid64 > INT32_MAX) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dsv4_unpack_dispatch_payload_kernel<<<(int)grid64, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      payload, hidden, meta, num_routes, hidden_dim, stride_elems);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_init_padded_route_slots_kernel(
    int32_t *__restrict__ packed_token,
    int32_t *__restrict__ packed_route_slot,
    int32_t *__restrict__ packed_meta,
    int total_routes) {
  int route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route >= total_routes) return;
  packed_token[route] = -1;
  packed_route_slot[route] = -1;
  int meta_base = route * 3;
  packed_meta[meta_base] = -1;
  packed_meta[meta_base + 1] = -1;
  packed_meta[meta_base + 2] = 0;
}

extern "C" CUresult dsv4_init_padded_route_slots_cuda(
    int32_t *packed_token,
    int32_t *packed_route_slot,
    int32_t *packed_meta,
    int total_routes,
    CUstream stream) {
  if (total_routes < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (total_routes == 0) return CUDA_SUCCESS;
  int grid = (total_routes + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_init_padded_route_slots_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      packed_token, packed_route_slot, packed_meta, total_routes);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_count_packed_local_experts_kernel(
    const int32_t *__restrict__ packed_meta,
    int32_t *__restrict__ counts,
    int num_routes,
    int local_expert_start,
    int experts_per_rank) {
  int route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route >= num_routes) return;
  int expert = packed_meta[route * 3 + 1];
  int local = expert - local_expert_start;
  if (local >= 0 && local < experts_per_rank) {
    atomicAdd(&counts[local], 1);
  }
}

extern "C" CUresult dsv4_count_packed_local_experts_cuda(
    const int32_t *packed_meta,
    int32_t *counts,
    int num_routes,
    int local_expert_start,
    int experts_per_rank,
    CUstream stream) {
  if (num_routes < 0 || local_expert_start < 0 || experts_per_rank <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_routes == 0) return CUDA_SUCCESS;
  int grid = (num_routes + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_count_packed_local_experts_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      packed_meta, counts, num_routes, local_expert_start, experts_per_rank);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_prepare_packed_local_experts_small_kernel(
    const int32_t *__restrict__ packed_meta,
    int32_t *__restrict__ counts,
    int32_t *__restrict__ offsets,
    int32_t *__restrict__ cursors,
    int num_routes,
    int local_expert_start,
    int experts_per_rank) {
  int tid = threadIdx.x;
  for (int idx = tid; idx < experts_per_rank; idx += blockDim.x) {
    counts[idx] = 0;
    offsets[idx] = 0;
    cursors[idx] = 0;
  }
  __syncthreads();

  for (int route = tid; route < num_routes; route += blockDim.x) {
    int expert = packed_meta[route * 3 + 1];
    int local = expert - local_expert_start;
    if (local >= 0 && local < experts_per_rank) {
      atomicAdd(&counts[local], 1);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int running = 0;
    for (int idx = 0; idx < experts_per_rank; ++idx) {
      int count = counts[idx];
      offsets[idx] = running;
      running += count;
    }
  }
}

extern "C" CUresult dsv4_prepare_packed_local_experts_small_cuda(
    const int32_t *packed_meta,
    int32_t *counts,
    int32_t *offsets,
    int32_t *cursors,
    int num_routes,
    int local_expert_start,
    int experts_per_rank,
    CUstream stream) {
  if (num_routes < 0 || local_expert_start < 0 || experts_per_rank <= 0 ||
      num_routes > DSV4_ROUTE_BLOCK || experts_per_rank > DSV4_ROUTE_BLOCK) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dsv4_prepare_packed_local_experts_small_kernel<<<1, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      packed_meta, counts, offsets, cursors, num_routes, local_expert_start,
      experts_per_rank);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_pack_received_experts_kernel(
    const uint16_t *__restrict__ received_hidden,
    const int32_t *__restrict__ received_meta,
    const int32_t *__restrict__ offsets,
    int32_t *__restrict__ cursors,
    uint16_t *__restrict__ expert_hidden,
    float *__restrict__ expert_weight,
    int32_t *__restrict__ expert_route_slot,
    int num_routes,
    int hidden_dim,
    int local_expert_start,
    int experts_per_rank) {
  int route = blockIdx.x;
  if (route >= num_routes) return;
  int meta_base = route * 3;
  int local = received_meta[meta_base + 1] - local_expert_start;
  if (local < 0 || local >= experts_per_rank) return;

  __shared__ int slot;
  if (threadIdx.x == 0) {
    slot = offsets[local] + atomicAdd(&cursors[local], 1);
    expert_weight[slot] = dsv4_route_i32_bits_to_f32(received_meta[meta_base + 2]);
    expert_route_slot[slot] = route;
  }
  __syncthreads();

  int src_base = route * hidden_dim;
  int dst_base = slot * hidden_dim;
  for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    expert_hidden[dst_base + col] = received_hidden[src_base + col];
  }
}

extern "C" CUresult dsv4_pack_received_experts_cuda(
    const uint16_t *received_hidden,
    const int32_t *received_meta,
    const int32_t *offsets,
    int32_t *cursors,
    uint16_t *expert_hidden,
    float *expert_weight,
    int32_t *expert_route_slot,
    int num_routes,
    int hidden_dim,
    int local_expert_start,
    int experts_per_rank,
    CUstream stream) {
  if (num_routes < 0 || hidden_dim <= 0 || local_expert_start < 0 ||
      experts_per_rank <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_routes == 0) return CUDA_SUCCESS;
  dsv4_pack_received_experts_kernel<<<num_routes, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      received_hidden, received_meta, offsets, cursors, expert_hidden, expert_weight,
      expert_route_slot, num_routes, hidden_dim, local_expert_start, experts_per_rank);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_scatter_packed_route_slot_kernel(
    const uint16_t *__restrict__ expert_out,
    uint16_t *__restrict__ route_out,
    const int32_t *__restrict__ expert_route_slot,
    const float *__restrict__ expert_weight,
    int start_slot,
    int count,
    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = count * hidden_dim;
  if (idx >= total) return;
  int row = idx / hidden_dim;
  int col = idx - row * hidden_dim;
  int slot = start_slot + row;
  int route_slot = expert_route_slot[slot];
  float weight = expert_weight[slot];
  float value = dsv4_route_bf16_to_f32(expert_out[idx]);
  route_out[route_slot * hidden_dim + col] = dsv4_route_f32_to_bf16_bits(weight * value);
}

extern "C" CUresult dsv4_scatter_packed_route_slot_cuda(
    const uint16_t *expert_out,
    uint16_t *route_out,
    const int32_t *expert_route_slot,
    const float *expert_weight,
    int start_slot,
    int count,
    int hidden_dim,
    CUstream stream) {
  if (start_slot < 0 || count < 0 || hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = count * hidden_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_scatter_packed_route_slot_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      expert_out, route_out, expert_route_slot, expert_weight, start_slot, count,
      hidden_dim);
  return (CUresult)cudaGetLastError();
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

__global__ void dsv4_scatter_route_outputs_by_slot_kernel(
    const uint16_t *__restrict__ packed_route_out,
    uint16_t *__restrict__ route_slot_out,
    const int32_t *__restrict__ packed_route_slot,
    int num_routes,
    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_routes * hidden_dim;
  if (idx >= total) return;
  int route = idx / hidden_dim;
  int col = idx - route * hidden_dim;
  int route_slot = packed_route_slot[route];
  if (route_slot < 0) return;
  route_slot_out[route_slot * hidden_dim + col] = packed_route_out[idx];
}

extern "C" CUresult dsv4_scatter_route_outputs_by_slot_cuda(
    const uint16_t *packed_route_out,
    uint16_t *route_slot_out,
    const int32_t *packed_route_slot,
    int num_routes,
    int hidden_dim,
    CUstream stream) {
  if (num_routes < 0 || hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = num_routes * hidden_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_scatter_route_outputs_by_slot_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      packed_route_out, route_slot_out, packed_route_slot, num_routes, hidden_dim);
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

__global__ void dsv4_combine_route_outputs_kernel(
    const uint16_t *__restrict__ route_out,
    const int32_t *__restrict__ packed_token,
    uint16_t *__restrict__ routed_out,
    int num_tokens,
    int num_routes,
    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * hidden_dim;
  if (idx >= total) return;
  int token = idx / hidden_dim;
  int col = idx - token * hidden_dim;
  float sum = 0.0f;
  for (int route = 0; route < num_routes; ++route) {
    if (packed_token[route] == token) {
      sum += dsv4_route_bf16_to_f32(route_out[route * hidden_dim + col]);
    }
  }
  routed_out[idx] = dsv4_route_f32_to_bf16_bits(sum);
}

extern "C" CUresult dsv4_combine_route_outputs_cuda(
    const uint16_t *route_out,
    const int32_t *packed_token,
    uint16_t *routed_out,
    int num_tokens,
    int num_routes,
    int hidden_dim,
    CUstream stream) {
  if (num_tokens < 0 || num_routes < 0 || hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = num_tokens * hidden_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_combine_route_outputs_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      route_out, packed_token, routed_out, num_tokens, num_routes, hidden_dim);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_sum_padded_route_outputs_by_peer_kernel(
    const uint16_t *__restrict__ route_out,
    const int32_t *__restrict__ recv_meta,
    uint16_t *__restrict__ peer_out,
    int ep_world_size,
    int topk,
    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = ep_world_size * hidden_dim;
  if (idx >= total) return;
  int peer = idx / hidden_dim;
  int col = idx - peer * hidden_dim;
  int route_base = peer * topk;
  float sum = 0.0f;
  for (int k = 0; k < topk; ++k) {
    int route = route_base + k;
    if (recv_meta[route * 3 + 1] >= 0) {
      sum += dsv4_route_bf16_to_f32(route_out[route * hidden_dim + col]);
    }
  }
  peer_out[idx] = dsv4_route_f32_to_bf16_bits(sum);
}

extern "C" CUresult dsv4_sum_padded_route_outputs_by_peer_cuda(
    const uint16_t *route_out,
    const int32_t *recv_meta,
    uint16_t *peer_out,
    int ep_world_size,
    int topk,
    int hidden_dim,
    CUstream stream) {
  if (ep_world_size < 0 || topk <= 0 || hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = ep_world_size * hidden_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_sum_padded_route_outputs_by_peer_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      route_out, recv_meta, peer_out, ep_world_size, topk, hidden_dim);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_sum_bf16_rows_kernel(
    const uint16_t *__restrict__ rows,
    uint16_t *__restrict__ out,
    int num_rows,
    int hidden_dim) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= hidden_dim) return;
  float sum = 0.0f;
  for (int row = 0; row < num_rows; ++row) {
    sum += dsv4_route_bf16_to_f32(rows[row * hidden_dim + col]);
  }
  out[col] = dsv4_route_f32_to_bf16_bits(sum);
}

extern "C" CUresult dsv4_sum_bf16_rows_cuda(
    const uint16_t *rows,
    uint16_t *out,
    int num_rows,
    int hidden_dim,
    CUstream stream) {
  if (num_rows < 0 || hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_rows == 0) return CUDA_SUCCESS;
  int grid = (hidden_dim + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_sum_bf16_rows_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      rows, out, num_rows, hidden_dim);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_add_local_expert_kernel(
    const uint16_t *__restrict__ expert_out,
    uint16_t *__restrict__ routed_out,
    const int32_t *__restrict__ indices,
    const float *__restrict__ weights,
    int num_tokens,
    int hidden_dim,
    int topk,
    int global_expert_idx) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * hidden_dim;
  if (idx >= total) return;
  int token = idx / hidden_dim;
  float route_weight = 0.0f;
  for (int k = 0; k < topk; ++k) {
    int route_expert = indices[token * topk + k];
    if (route_expert == global_expert_idx) {
      route_weight += weights[token * topk + k];
    }
  }
  if (route_weight == 0.0f) return;
  float prev = dsv4_route_bf16_to_f32(routed_out[idx]);
  float value = dsv4_route_bf16_to_f32(expert_out[idx]);
  routed_out[idx] = dsv4_route_f32_to_bf16_bits(prev + route_weight * value);
}

extern "C" CUresult dsv4_add_local_expert_cuda(
    const uint16_t *expert_out,
    uint16_t *routed_out,
    const int32_t *indices,
    const float *weights,
    int num_tokens,
    int hidden_dim,
    int topk,
    int global_expert_idx,
    CUstream stream) {
  if (num_tokens < 0 || hidden_dim <= 0 || topk <= 0 ||
      global_expert_idx < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = num_tokens * hidden_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ROUTE_BLOCK - 1) / DSV4_ROUTE_BLOCK;
  dsv4_add_local_expert_kernel<<<grid, DSV4_ROUTE_BLOCK, 0, (cudaStream_t)stream>>>(
      expert_out, routed_out, indices, weights, num_tokens, hidden_dim, topk,
      global_expert_idx);
  return (CUresult)cudaGetLastError();
}

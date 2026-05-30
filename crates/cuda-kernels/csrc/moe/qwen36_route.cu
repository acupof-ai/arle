// Qwen3.5-MoE / Qwen3.6 single-GPU route-weight post-processing.
//
// The router itself reuses dsv4_route_cuda (scoring_kind=0 plain softmax,
// routing_kind=1 block-argmax top-k, bias=null, routed_scaling_factor=1.0),
// which writes per-route softmax probabilities into `weights[token*topk+k]`
// without renormalization (its scoring_kind==0 branch fixes denom=1.0).
//
// Qwen3.6's SparseMoeBlock optionally renormalizes the selected top-k scores
// so they sum to 1 per token (`norm_topk_prob`). This kernel applies that
// renorm in-place over the dsv4_route weight buffer — one block per token,
// topk lanes. When norm_topk_prob is false the runtime simply skips this
// launch (raw softmax probs are the gate weights).
//
// Mirrors the Metal reference (mlx_qwen35_moe_block.cpp): scores = softmax;
// take top-k; if norm_topk_prob: scores /= sum(scores, -1).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define QWEN36_ROUTE_MAX_TOPK 64

__global__ void qwen36_renorm_topk_weights_kernel(
    float* __restrict__ weights,
    int num_tokens,
    int topk) {
  int token = blockIdx.x;
  if (token >= num_tokens) return;

  __shared__ float partial[QWEN36_ROUTE_MAX_TOPK];
  int lane = threadIdx.x;
  float v = (lane < topk) ? weights[token * topk + lane] : 0.0f;
  partial[lane] = v;
  __syncthreads();

  // Single-thread reduction — topk is tiny (<= 64).
  if (lane == 0) {
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) sum += partial[k];
    float inv = sum > 1.0e-20f ? 1.0f / sum : 0.0f;
    for (int k = 0; k < topk; ++k) {
      weights[token * topk + k] = partial[k] * inv;
    }
  }
}

extern "C" cudaError_t qwen36_renorm_topk_weights_cuda(
    float* weights,
    int num_tokens,
    int topk,
    cudaStream_t stream) {
  if (num_tokens < 0 || topk <= 0 || topk > QWEN36_ROUTE_MAX_TOPK) {
    return cudaErrorInvalidValue;
  }
  if (num_tokens == 0) return cudaSuccess;
  int threads = topk < 32 ? 32 : ((topk + 31) / 32) * 32;
  qwen36_renorm_topk_weights_kernel<<<num_tokens, threads, 0, stream>>>(
      weights, num_tokens, topk);
  return cudaGetLastError();
}

// Shared-expert scalar sigmoid gate + accumulate into the routed output.
//
//   out[t, :] = routed[t, :] + sigmoid(gate_logit[t]) * shared_y[t, :]
//
// `gate_logit` is the [num_tokens, 1] output of `x @ shared_gate_router`
// (one logit per token); `shared_y` is the [num_tokens, hidden] dense
// shared-expert SwiGLU output; `routed` is the [num_tokens, hidden] routed
// MoE sum (modified in place). Mirrors the Metal reference:
// `shared_y = sigmoid(x @ shared_expert_gate) * shared_y; return y + shared_y`.

__device__ __forceinline__ float qwen36_sigmoid(float v) {
  if (v >= 0.0f) return 1.0f / (1.0f + expf(-v));
  float e = expf(v);
  return e / (1.0f + e);
}

__global__ void qwen36_add_shared_expert_gated_kernel(
    __nv_bfloat16* __restrict__ routed,
    const __nv_bfloat16* __restrict__ shared_y,
    const __nv_bfloat16* __restrict__ gate_logit,
    int num_tokens,
    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * hidden_dim;
  if (idx >= total) return;
  int token = idx / hidden_dim;
  float gate = qwen36_sigmoid(__bfloat162float(gate_logit[token]));
  float acc = __bfloat162float(routed[idx]);
  float sv = __bfloat162float(shared_y[idx]);
  routed[idx] = __float2bfloat16(acc + gate * sv);
}

extern "C" cudaError_t qwen36_add_shared_expert_gated_cuda(
    __nv_bfloat16* routed,
    const __nv_bfloat16* shared_y,
    const __nv_bfloat16* gate_logit,
    int num_tokens,
    int hidden_dim,
    cudaStream_t stream) {
  if (num_tokens < 0 || hidden_dim <= 0) return cudaErrorInvalidValue;
  int total = num_tokens * hidden_dim;
  if (total == 0) return cudaSuccess;
  int threads = 256;
  int grid = (total + threads - 1) / threads;
  qwen36_add_shared_expert_gated_kernel<<<grid, threads, 0, stream>>>(
      routed, shared_y, gate_logit, num_tokens, hidden_dim);
  return cudaGetLastError();
}

// Context-parallel ring-attention device kernels (Track A). One-block-pure per
// KV block so the ring feeds blocks in one at a time and the flash-2 (m,l,o)
// merge stays on-device across launches — never materializing the full-seq KV
// (the O(full_seq) slice_bwd buffer that OOMs option B at local seq > 65535).
//
// Layout: per-(batch·head) tiles, row-major [rows, dim]. q is [B*Hq, q_rows, hd];
// k/v blocks are [B*Hkv, blk_len, hd] (GQA repeat resolved here via q_head/gqa).
// One warp per (tile, q_row); lanes stride over head_dim, so ANY head_dim works
// (the parity model uses hd=2, the 27B uses hd=128). Math mirrors the verified
// host reference in crates/autograd/src/ops/ring_attention.rs (block_stats /
// merge_block / ring_backward_tile) — the U2 correctness spec.
#include "../common.cuh"
#include <cstdint>

#define RING_MAX_DPT 8  // ceil(256/32): max head_dim we support is 256

// Fused per-block forward + flash-2 merge. Reads the running (M,L,O) from the
// *_in accumulators, computes this block's (m,l,o) online (exact softmax), merges,
// writes to the *_out accumulators. Functional (in != out) so the ops-layer ring
// loop threads fresh handles per block — no Arc-mutation, no cross-launch aliasing
// (each (tile,row) is independent). Absolute causal mask: q row r (abs q_abs+r)
// attends block col c (abs k_abs+c) iff k_abs+c <= q_abs+r.
__global__ void ring_block_attention_fwd_merge_kernel(
    const __nv_bfloat16 *__restrict__ q,     // [Tq, q_rows, hd]
    const __nv_bfloat16 *__restrict__ k_blk, // [Tkv, blk_len, hd]
    const __nv_bfloat16 *__restrict__ v_blk, // [Tkv, blk_len, hd]
    const float *__restrict__ acc_m_in,      // [Tq, q_rows]
    const float *__restrict__ acc_l_in,      // [Tq, q_rows]
    const float *__restrict__ acc_o_in,      // [Tq, q_rows, hd]
    float *__restrict__ acc_m_out, float *__restrict__ acc_l_out,
    float *__restrict__ acc_o_out,
    int num_q_heads, int num_kv_heads, int head_dim,
    int q_rows, int blk_len, int q_abs, int k_abs, float sm_scale) {
  // grid.x = q_row (up to 2^31-1), grid.y = tile (= b*H, small): q_rows can exceed
  // the 65535 gridDim.y cap at CP local seq > 65535, so the LARGE dim is grid.x.
  int row = blockIdx.x;
  int tile = blockIdx.y;
  int lane = threadIdx.x; // 0..31 (one warp)
  if (row >= q_rows) return;

  int b = tile / num_q_heads;
  int qh = tile % num_q_heads;
  int gqa = num_q_heads / num_kv_heads;
  int kv_tile = b * num_kv_heads + (qh / gqa);
  int q_pos = q_abs + row;
  int64_t rid = (int64_t)tile * q_rows + row;

  const __nv_bfloat16 *q_ptr = q + rid * head_dim;
  const __nv_bfloat16 *k_base = k_blk + (int64_t)kv_tile * blk_len * head_dim;
  const __nv_bfloat16 *v_base = v_blk + (int64_t)kv_tile * blk_len * head_dim;
  const float *o_in = acc_o_in + rid * head_dim;
  float *o_out = acc_o_out + rid * head_dim;

  const int DPT = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
  float m_blk = -INFINITY, l_blk = 0.0f;
  float o_blk[RING_MAX_DPT];
#pragma unroll
  for (int i = 0; i < RING_MAX_DPT; ++i) o_blk[i] = 0.0f;

  for (int c = 0; c < blk_len; ++c) {
    if (k_abs + c > q_pos) break; // causal: cols ordered, rest are future
    float partial = 0.0f;
    for (int i = 0; i < DPT; ++i) {
      int d = lane + i * WARP_SIZE;
      if (d < head_dim)
        partial += __bfloat162float(q_ptr[d]) *
                   __bfloat162float(k_base[(int64_t)c * head_dim + d]);
    }
    float s = warp_reduce_sum(partial);       // full dot in lane 0
    s = __shfl_sync(0xffffffff, s, 0) * sm_scale;
    float m_new = fmaxf(m_blk, s);
    float rescale = (m_blk == -INFINITY) ? 0.0f : expf(m_blk - m_new);
    float p = expf(s - m_new);
    l_blk = l_blk * rescale + p;
    for (int i = 0; i < DPT; ++i) {
      int d = lane + i * WARP_SIZE;
      if (d < head_dim)
        o_blk[i] = o_blk[i] * rescale +
                   p * __bfloat162float(v_base[(int64_t)c * head_dim + d]);
    }
    m_blk = m_new;
  }

  float M_old = acc_m_in[rid];
  float L_old = acc_l_in[rid];
  if (m_blk == -INFINITY) {
    // Whole block future for this row → carry the running acc through unchanged.
    for (int i = 0; i < DPT; ++i) {
      int d = lane + i * WARP_SIZE;
      if (d < head_dim) o_out[d] = o_in[d];
    }
    if (lane == 0) { acc_m_out[rid] = M_old; acc_l_out[rid] = L_old; }
    return;
  }

  float M_new = fmaxf(M_old, m_blk);
  float a = (M_old == -INFINITY) ? 0.0f : expf(M_old - M_new);
  float bcoef = expf(m_blk - M_new);
  for (int i = 0; i < DPT; ++i) {
    int d = lane + i * WARP_SIZE;
    if (d < head_dim) o_out[d] = a * o_in[d] + bcoef * o_blk[i];
  }
  if (lane == 0) {
    acc_l_out[rid] = a * L_old + bcoef * l_blk;
    acc_m_out[rid] = M_new;
  }
}

// Normalize the running accumulator: out = O / L, lse = M + ln(L). One thread per
// (tile, row, dim) via a flat grid; lse written by dim 0. L==0 (row saw nothing)
// → out 0, lse -inf (matches ring_forward_tile's guard). out is f32 — the ring
// output stays on the f32 autograd tape (feeds o_proj as f32), no bf16 round-trip.
__global__ void ring_block_attention_finalize_kernel(
    const float *__restrict__ acc_m, const float *__restrict__ acc_l,
    const float *__restrict__ acc_o, float *__restrict__ out,
    float *__restrict__ lse, int total_rows, int head_dim) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)total_rows * head_dim;
  if (idx >= total) return;
  int64_t row = idx / head_dim;
  int d = idx % head_dim;
  float L = acc_l[row];
  out[idx] = (L > 0.0f) ? acc_o[idx] / L : 0.0f;
  if (d == 0) lse[row] = (L > 0.0f) ? acc_m[row] + logf(L) : -INFINITY;
}

// Per-block backward: reconstruct P = exp(S·scale − lse) from the SAVED final lse
// (flash-2 adjoint). grad_q accumulates across blocks (in/out; row-unique, calls
// sequential → safe non-atomic add). grad_k/grad_v accumulate across q_rows AND
// GQA q_heads that map to one kv_head → atomicAdd into fp32 accumulators. Mirrors
// ring_backward_tile.
__global__ void ring_block_attention_bwd_kernel(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k_blk,
    const __nv_bfloat16 *__restrict__ v_blk, const float *__restrict__ out,
    const float *__restrict__ lse, const __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ grad_q, float *__restrict__ grad_k_blk,
    float *__restrict__ grad_v_blk, int num_q_heads, int num_kv_heads,
    int head_dim, int q_rows, int blk_len, int q_abs, int k_abs, float sm_scale) {
  // grid.x = q_row, grid.y = tile (mirror the forward — q_rows may exceed 65535).
  int row = blockIdx.x;
  int tile = blockIdx.y;
  int lane = threadIdx.x;
  if (row >= q_rows) return;

  int b = tile / num_q_heads;
  int qh = tile % num_q_heads;
  int gqa = num_q_heads / num_kv_heads;
  int kv_tile = b * num_kv_heads + (qh / gqa);
  int q_pos = q_abs + row;
  int64_t rid = (int64_t)tile * q_rows + row;

  float row_lse = lse[rid];
  if (row_lse == -INFINITY) return; // row saw nothing → no grad from this block

  const __nv_bfloat16 *q_ptr = q + rid * head_dim;
  const float *out_ptr = out + rid * head_dim;
  const __nv_bfloat16 *do_ptr = d_out + rid * head_dim;
  const __nv_bfloat16 *k_base = k_blk + (int64_t)kv_tile * blk_len * head_dim;
  const __nv_bfloat16 *v_base = v_blk + (int64_t)kv_tile * blk_len * head_dim;
  float *gk_base = grad_k_blk + (int64_t)kv_tile * blk_len * head_dim;
  float *gv_base = grad_v_blk + (int64_t)kv_tile * blk_len * head_dim;
  float *gq_ptr = grad_q + rid * head_dim;

  const int DPT = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
  // delta_r = sum_d d_out[d]*out[d]  (flash backward row correction)
  float dpart = 0.0f;
  for (int i = 0; i < DPT; ++i) {
    int d = lane + i * WARP_SIZE;
    if (d < head_dim) dpart += __bfloat162float(do_ptr[d]) * out_ptr[d];
  }
  float delta = warp_reduce_sum(dpart);
  delta = __shfl_sync(0xffffffff, delta, 0);

  float gq[RING_MAX_DPT];
#pragma unroll
  for (int i = 0; i < RING_MAX_DPT; ++i) gq[i] = 0.0f;

  for (int c = 0; c < blk_len; ++c) {
    if (k_abs + c > q_pos) break;
    float sdot = 0.0f, dp = 0.0f;
    for (int i = 0; i < DPT; ++i) {
      int d = lane + i * WARP_SIZE;
      if (d < head_dim) {
        sdot += __bfloat162float(q_ptr[d]) * __bfloat162float(k_base[(int64_t)c * head_dim + d]);
        dp += __bfloat162float(do_ptr[d]) * __bfloat162float(v_base[(int64_t)c * head_dim + d]);
      }
    }
    sdot = warp_reduce_sum(sdot);
    sdot = __shfl_sync(0xffffffff, sdot, 0);
    dp = warp_reduce_sum(dp);
    dp = __shfl_sync(0xffffffff, dp, 0);
    float p = expf(sdot * sm_scale - row_lse);
    float d_score = p * (dp - delta);
    for (int i = 0; i < DPT; ++i) {
      int d = lane + i * WARP_SIZE;
      if (d < head_dim) {
        float qd = __bfloat162float(q_ptr[d]);
        gq[i] += sm_scale * d_score * __bfloat162float(k_base[(int64_t)c * head_dim + d]);
        atomicAdd(&gk_base[(int64_t)c * head_dim + d], sm_scale * d_score * qd);
        atomicAdd(&gv_base[(int64_t)c * head_dim + d], p * __bfloat162float(do_ptr[d]));
      }
    }
  }
  for (int i = 0; i < DPT; ++i) {
    int d = lane + i * WARP_SIZE;
    if (d < head_dim) gq_ptr[d] += gq[i]; // accumulate across blocks (sequential)
  }
}

static inline int ring_block_ok(int num_q_heads, int num_kv_heads, int head_dim,
                                int q_rows, int blk_len) {
  return num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0 && head_dim <= 256 &&
         q_rows >= 0 && blk_len >= 0 && num_q_heads % num_kv_heads == 0;
}

extern "C" cudaError_t ring_block_attention_fwd_merge_cuda(
    const uint16_t *q, const uint16_t *k_blk, const uint16_t *v_blk,
    const float *acc_m_in, const float *acc_l_in, const float *acc_o_in,
    float *acc_m_out, float *acc_l_out, float *acc_o_out, int num_q_tiles,
    int num_q_heads, int num_kv_heads, int head_dim, int q_rows, int blk_len,
    int q_abs, int k_abs, float sm_scale, cudaStream_t stream) {
  if (!ring_block_ok(num_q_heads, num_kv_heads, head_dim, q_rows, blk_len) || num_q_tiles <= 0)
    return cudaErrorInvalidValue;
  if (q_rows == 0 || blk_len == 0) return cudaSuccess;
  dim3 grid(q_rows, num_q_tiles); // x=row (may exceed 65535), y=tile
  ring_block_attention_fwd_merge_kernel<<<grid, WARP_SIZE, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(q),
      reinterpret_cast<const __nv_bfloat16 *>(k_blk),
      reinterpret_cast<const __nv_bfloat16 *>(v_blk), acc_m_in, acc_l_in,
      acc_o_in, acc_m_out, acc_l_out, acc_o_out, num_q_heads, num_kv_heads,
      head_dim, q_rows, blk_len, q_abs, k_abs, sm_scale);
  return cudaGetLastError();
}

extern "C" cudaError_t ring_block_attention_finalize_cuda(
    const float *acc_m, const float *acc_l, const float *acc_o, float *out,
    float *lse, int total_rows, int head_dim, cudaStream_t stream) {
  if (total_rows < 0 || head_dim <= 0 || head_dim > 256) return cudaErrorInvalidValue;
  if (total_rows == 0) return cudaSuccess;
  int64_t total = (int64_t)total_rows * head_dim;
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  ring_block_attention_finalize_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
      acc_m, acc_l, acc_o, out, lse, total_rows, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t ring_block_attention_bwd_cuda(
    const uint16_t *q, const uint16_t *k_blk, const uint16_t *v_blk,
    const float *out, const float *lse, const uint16_t *d_out, float *grad_q,
    float *grad_k_blk, float *grad_v_blk, int num_q_tiles, int num_q_heads,
    int num_kv_heads, int head_dim, int q_rows, int blk_len, int q_abs, int k_abs,
    float sm_scale, cudaStream_t stream) {
  if (!ring_block_ok(num_q_heads, num_kv_heads, head_dim, q_rows, blk_len) || num_q_tiles <= 0)
    return cudaErrorInvalidValue;
  if (q_rows == 0 || blk_len == 0) return cudaSuccess;
  dim3 grid(q_rows, num_q_tiles); // x=row (may exceed 65535), y=tile
  ring_block_attention_bwd_kernel<<<grid, WARP_SIZE, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(q),
      reinterpret_cast<const __nv_bfloat16 *>(k_blk),
      reinterpret_cast<const __nv_bfloat16 *>(v_blk), out, lse,
      reinterpret_cast<const __nv_bfloat16 *>(d_out), grad_q, grad_k_blk,
      grad_v_blk, num_q_heads, num_kv_heads, head_dim, q_rows, blk_len, q_abs,
      k_abs, sm_scale);
  return cudaGetLastError();
}

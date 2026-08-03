// FA3 pair-route helpers for the CP ring attention (Track A). FA3 emits a
// NORMALIZED (o, lse) per visible (q_run, k_run) pair; treated as a flash-2
// block-stats triple (m = lse, l = 1, o_unnorm = o) it merges into the ring's
// running f32 (M, L, O) accumulators with the exact rescale the scalar
// ring_block_attention merge uses — the finalize kernel is untouched. Plain
// CUDA (no FA3 headers), so this TU compiles for every arch and links in stub
// builds; the runtime marker gate keeps it unreachable there.
//
// Layout: accumulators / grads are full head-major [tiles, seq, d] / [tiles,
// seq]; pair buffers are compact head-major [heads, run_len, d] /
// [heads, run_len]. `tile_base` selects the batch's first tile; the run maps
// local rows [run_start, run_start+run_len) of every head.
#include "../common.cuh"
#include <cstdint>

// Merge one pair's (lse, o) into the running (M, L, O), in place on the
// caller's fresh accumulator copies. One block per (head, run row); threads
// stride head_dim.
__global__ void ring_fa3_merge_pair_kernel(
    float *__restrict__ acc_m, float *__restrict__ acc_l,
    float *__restrict__ acc_o,
    const float *__restrict__ lse_pair,       // [heads, run_len]
    const __nv_bfloat16 *__restrict__ o_pair, // [heads, run_len, d]
    int tile_base, int seq_len, int run_start, int run_len, int head_dim) {
  int r = blockIdx.x; // run row
  int h = blockIdx.y; // head
  float lse = lse_pair[(int64_t)h * run_len + r];
  if (isinf(lse)) return; // FA3 empty row — no contribution
  int64_t g = ((int64_t)(tile_base + h) * seq_len) + run_start + r;
  float m_old = acc_m[g];
  float m_new = fmaxf(m_old, lse);
  float a = isinf(m_old) ? 0.0f : expf(m_old - m_new);
  float bcoef = expf(lse - m_new);
  const __nv_bfloat16 *op = o_pair + ((int64_t)h * run_len + r) * head_dim;
  float *og = acc_o + g * head_dim;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    og[i] = a * og[i] + bcoef * __bfloat162float(op[i]);
  if (threadIdx.x == 0) {
    acc_l[g] = a * acc_l[g] + bcoef;
    acc_m[g] = m_new;
  }
}

// Gather the [tiles, seq] lse rows of one run into the compact [heads,
// run_len] layout FA3's bwd expects (it takes no lse stride).
__global__ void ring_fa3_gather_lse_kernel(float *__restrict__ dst,
                                           const float *__restrict__ lse,
                                           int tile_base, int seq_len,
                                           int run_start, int run_len,
                                           int num_heads) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (int64_t)num_heads * run_len) return;
  int h = (int)(i / run_len);
  int r = (int)(i % run_len);
  dst[i] = lse[((int64_t)(tile_base + h) * seq_len) + run_start + r];
}

// Accumulate a pair's compact bf16 grad ([heads, run_len, d]) into the full
// head-major f32 grad buffer at the run's rows.
__global__ void ring_fa3_accum_grad_bf16_kernel(
    float *__restrict__ dst, const __nv_bfloat16 *__restrict__ src,
    int tile_base, int seq_len, int run_start, int run_len, int head_dim) {
  int r = blockIdx.x;
  int h = blockIdx.y;
  const __nv_bfloat16 *sp = src + ((int64_t)h * run_len + r) * head_dim;
  float *dp = dst + (((int64_t)(tile_base + h) * seq_len) + run_start + r) *
                        head_dim;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    dp[i] += __bfloat162float(sp[i]);
}

extern "C" cudaError_t ring_fa3_merge_pair_cuda(
    float *acc_m, float *acc_l, float *acc_o, const float *lse_pair,
    const __nv_bfloat16 *o_pair, int num_heads, int tile_base, int seq_len,
    int run_start, int run_len, int head_dim, cudaStream_t stream) {
  if (run_len <= 0 || num_heads <= 0 || head_dim <= 0)
    return cudaErrorInvalidValue;
  dim3 grid((unsigned)run_len, (unsigned)num_heads);
  int threads = head_dim < 256 ? head_dim : 256;
  ring_fa3_merge_pair_kernel<<<grid, threads, 0, stream>>>(
      acc_m, acc_l, acc_o, lse_pair, o_pair, tile_base, seq_len, run_start,
      run_len, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t ring_fa3_gather_lse_cuda(float *dst, const float *lse,
                                                int num_heads, int tile_base,
                                                int seq_len, int run_start,
                                                int run_len,
                                                cudaStream_t stream) {
  if (run_len <= 0 || num_heads <= 0) return cudaErrorInvalidValue;
  int64_t total = (int64_t)num_heads * run_len;
  int threads = 256;
  unsigned blocks = (unsigned)((total + threads - 1) / threads);
  ring_fa3_gather_lse_kernel<<<blocks, threads, 0, stream>>>(
      dst, lse, tile_base, seq_len, run_start, run_len, num_heads);
  return cudaGetLastError();
}

extern "C" cudaError_t ring_fa3_accum_grad_bf16_cuda(
    float *dst, const __nv_bfloat16 *src, int num_heads, int tile_base,
    int seq_len, int run_start, int run_len, int head_dim,
    cudaStream_t stream) {
  if (run_len <= 0 || num_heads <= 0 || head_dim <= 0)
    return cudaErrorInvalidValue;
  dim3 grid((unsigned)run_len, (unsigned)num_heads);
  int threads = head_dim < 256 ? head_dim : 256;
  ring_fa3_accum_grad_bf16_kernel<<<grid, threads, 0, stream>>>(
      dst, src, tile_base, seq_len, run_start, run_len, head_dim);
  return cudaGetLastError();
}

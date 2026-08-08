// Self-contained micro-benchmark for the DSpark draft attention
// (nonpaged_prefill_attention_ring_varlen path).
//
// The serve-side profile of this kernel is confounded: ncu serializes launches,
// the batch collapses, and the grid shrinks to a regime that is not the one that
// costs 29 ms. This harness pins the shape instead.
//
// Build + run:
//   nvcc -O3 -arch=sm_90 nonpaged_attn_bench.cu -o nonpaged_attn_bench && ./nonpaged_attn_bench
//   ncu --set full -k nonpaged_prefill_attention_kernel -c 1 ./nonpaged_attn_bench 96
//
// arg1 (optional) = a single row count to run (default: sweep 12..96).
//
// Kernel body extracted VERBATIM from
//   crates/cuda-kernels/csrc/attention/nonpaged_prefill_attention.cu
// Shape from /host/dspark-fr/config.json: 32 q heads, 8 kv heads, head_dim 128,
// ring cap = sliding_window 2048 + block 16.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#define WARP_SIZE 32
#define NONPAGED_PREFILL_TILE 64
#define NONPAGED_PREFILL_MAX_WARPS 8

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA %s @ %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// `kv_len <= ring_modulus` bounds the walk to one wrap, so the runtime-modulus
// IDIV collapses to a conditional subtract.
__device__ __forceinline__ int ring_row(int base, int pos, int modulus, bool fast) {
  if (modulus <= 0) return pos;
  if (!fast) return (base + pos) % modulus;
  int row = base + pos;
  return row >= modulus ? row - modulus : row;
}

// ============================================================================
// VERBATIM from nonpaged_prefill_attention.cu, except the two `ring_row` calls.
// ============================================================================
template <bool FastRing>
__global__ void nonpaged_prefill_attention_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_cache,
    const __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int start_pos,
    const int *__restrict__ start_pos_dev,
    int max_seq_len,
    int ring_base,
    int ring_modulus,
    const int *__restrict__ ring_base_dev,
    const int *__restrict__ kv_len_dev,
    float sm_scale,
    // Slot batching: non-null promotes blockIdx.z to the slot axis, each slot
    // keeping its own ring cache exactly as `Qwen35DsparkSlotState` does.
    const __nv_bfloat16 *const *__restrict__ k_slots = nullptr,
    const __nv_bfloat16 *const *__restrict__ v_slots = nullptr) {
  int q_head = blockIdx.x;
  int token = blockIdx.y;
  int dim = threadIdx.x;
  int lane = dim % WARP_SIZE;
  int warp = dim / WARP_SIZE;
  int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  if (q_head >= num_q_heads || token >= seq_len || dim >= head_dim) {
    return;
  }

  if (start_pos_dev != nullptr) {
    start_pos = *start_pos_dev;
  }

  int gqa_ratio = num_q_heads / num_kv_heads;
  int kv_head = q_head / gqa_ratio;
  int q_dim = num_q_heads * head_dim;
  if (k_slots != nullptr) {
    int slot = blockIdx.z;
    k_cache = k_slots[slot];
    v_cache = v_slots[slot];
    q += (int64_t)slot * seq_len * q_dim;
    out += (int64_t)slot * seq_len * q_dim;
    ring_base_dev += slot * seq_len;
    kv_len_dev += slot * seq_len;
  }
  int kv_len = start_pos + token + 1;
  if (kv_len_dev != nullptr) {
    kv_len = kv_len_dev[token];
    ring_base = ring_base_dev[token];
  }
  // Fast path folds the modulus in once; the reference keeps the raw absolute
  // base so the A/B proves the two agree on unnormalized input.
  if (FastRing && ring_modulus > 0) {
    ring_base %= ring_modulus;
  }

  __shared__ float scores[NONPAGED_PREFILL_TILE];
  __shared__ float warp_partials[NONPAGED_PREFILL_MAX_WARPS *
                                 NONPAGED_PREFILL_TILE];
  __shared__ float warp_scratch[NONPAGED_PREFILL_MAX_WARPS];
  __shared__ float running_max_s;
  __shared__ float running_sum_s;
  __shared__ float rescale_s;

  if (dim == 0) {
    running_max_s = -INFINITY;
    running_sum_s = 0.0f;
  }
  __syncthreads();

  float q_val = __bfloat162float(q[token * q_dim + q_head * head_dim + dim]);
  float o_acc = 0.0f;
  const __nv_bfloat16 *k_base = k_cache + (int64_t)kv_head * max_seq_len * head_dim + dim;
  const __nv_bfloat16 *v_base = v_cache + (int64_t)kv_head * max_seq_len * head_dim + dim;

  for (int tile_start = 0; tile_start < kv_len; tile_start += NONPAGED_PREFILL_TILE) {
    int tile_len = min(NONPAGED_PREFILL_TILE, kv_len - tile_start);

    for (int pos = 0; pos < tile_len; ++pos) {
      int abs_pos = tile_start + pos;
      int row = ring_row(ring_base, abs_pos, ring_modulus, FastRing);
      float partial = q_val * __bfloat162float(k_base[row * head_dim]);
      partial = warp_reduce_sum(partial);
      if (lane == 0) {
        warp_partials[warp * NONPAGED_PREFILL_TILE + pos] = partial;
      }
    }
    __syncthreads();

    if (dim < tile_len) {
      float score = 0.0f;
      for (int w = 0; w < num_warps; ++w) {
        score += warp_partials[w * NONPAGED_PREFILL_TILE + dim];
      }
      scores[dim] = score * sm_scale;
    }
    __syncthreads();

    float local_max = -INFINITY;
    if (dim < tile_len) {
      local_max = scores[dim];
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) {
      warp_scratch[warp] = local_max;
    }
    __syncthreads();

    if (dim == 0) {
      float tile_max = warp_scratch[0];
      for (int w = 1; w < num_warps; ++w) {
        tile_max = fmaxf(tile_max, warp_scratch[w]);
      }
      float old_max = running_max_s;
      float new_max = fmaxf(old_max, tile_max);
      rescale_s = expf(old_max - new_max);
      running_sum_s *= rescale_s;
      running_max_s = new_max;
    }
    __syncthreads();

    o_acc *= rescale_s;
    float row_sum = 0.0f;
    float current_max = running_max_s;
    for (int pos = 0; pos < tile_len; ++pos) {
      float weight = expf(scores[pos] - current_max);
      int abs_pos = tile_start + pos;
      int row = ring_row(ring_base, abs_pos, ring_modulus, FastRing);
      row_sum += weight;
      o_acc += weight * __bfloat162float(v_base[row * head_dim]);
    }
    if (dim == 0) {
      running_sum_s += row_sum;
    }
    __syncthreads();
  }

  float denom = running_sum_s;
  float value = denom > 0.0f ? o_acc / denom : 0.0f;
  out[token * q_dim + q_head * head_dim + dim] = __float2bfloat16(value);
}
// ============================================================================

static const int NQ = 32, NKV = 8, HD = 128, WINDOW = 2048, BLOCK = 16;
static const int CAP = WINDOW + BLOCK;

// Serve shape measured 2026-08-08 (`/host/decodeanchor`, c=16 decode-shaped):
// grid (32, 6, 1) per launch, 5 draft layers x 16 slots serialized on one
// stream. Arm A reproduces that; arm B folds the slot axis into gridZ.
static int run_slot_batch_ab(int slots, int block, int kv_len) {
  const int CAP_L = kv_len + block;
  const size_t q_per_slot = (size_t)block * NQ * HD;
  const size_t kv_per_slot = (size_t)NKV * CAP_L * HD;

  std::vector<uint16_t> h_q(q_per_slot * slots), h_kv(kv_per_slot);
  for (size_t i = 0; i < h_q.size(); ++i) h_q[i] = 0x3C00 ^ (uint16_t)(i * 37);
  for (size_t i = 0; i < kv_per_slot; ++i) h_kv[i] = 0x3C00 ^ (uint16_t)(i * 61);

  __nv_bfloat16 *d_q, *d_out;
  CHECK(cudaMalloc(&d_q, h_q.size() * 2));
  CHECK(cudaMalloc(&d_out, h_q.size() * 2));
  CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * 2, cudaMemcpyHostToDevice));

  // One allocation per slot, as the slot states hold them.
  std::vector<__nv_bfloat16 *> k_ptrs(slots), v_ptrs(slots);
  for (int s = 0; s < slots; ++s) {
    CHECK(cudaMalloc(&k_ptrs[s], kv_per_slot * 2));
    CHECK(cudaMalloc(&v_ptrs[s], kv_per_slot * 2));
    CHECK(cudaMemcpy(k_ptrs[s], h_kv.data(), kv_per_slot * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(v_ptrs[s], h_kv.data(), kv_per_slot * 2, cudaMemcpyHostToDevice));
  }
  __nv_bfloat16 **d_kslots, **d_vslots;
  CHECK(cudaMalloc(&d_kslots, slots * sizeof(void *)));
  CHECK(cudaMalloc(&d_vslots, slots * sizeof(void *)));
  CHECK(cudaMemcpy(d_kslots, k_ptrs.data(), slots * sizeof(void *), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_vslots, v_ptrs.data(), slots * sizeof(void *), cudaMemcpyHostToDevice));

  const int rows = slots * block;
  std::vector<int> win(2 * rows);
  for (int t = 0; t < rows; ++t) {
    win[t] = (t % block) * 977;
    win[rows + t] = kv_len;
  }
  int *d_win;
  CHECK(cudaMalloc(&d_win, 2 * rows * sizeof(int)));
  CHECK(cudaMemcpy(d_win, win.data(), 2 * rows * sizeof(int), cudaMemcpyHostToDevice));

  float sm_scale = 1.0f / sqrtf((float)HD);
  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));

  printf("\nslot-batch A/B: slots=%d block=%d kv_len=%d cap=%d\n", slots, block, kv_len, CAP_L);
  printf("  arm A: %d serial launches of grid(%d,%d) = %d blocks each\n",
         slots, NQ, block, NQ * block);
  printf("  arm B: 1 launch of grid(%d,%d,%d) = %d blocks\n\n", NQ, block, slots,
         NQ * block * slots);

  std::vector<uint16_t> ref(h_q.size()), got(h_q.size());
  float ms[2] = {0, 0};
  for (int arm = 0; arm < 2; ++arm) {
    auto launch = [&] {
      if (arm == 0) {
        for (int s = 0; s < slots; ++s) {
          size_t qo = (size_t)s * q_per_slot;
          nonpaged_prefill_attention_kernel<false><<<dim3(NQ, block), HD>>>(
              d_q + qo, k_ptrs[s], v_ptrs[s], d_out + qo, NQ, NKV, HD, block, 0,
              nullptr, CAP_L, 0, CAP_L, d_win + s * block,
              d_win + rows + s * block, sm_scale);
        }
      } else {
        nonpaged_prefill_attention_kernel<false><<<dim3(NQ, block, slots), HD>>>(
            d_q, nullptr, nullptr, d_out, NQ, NKV, HD, block, 0, nullptr, CAP_L,
            0, CAP_L, d_win, d_win + rows, sm_scale, d_kslots, d_vslots);
      }
    };
    for (int i = 0; i < 3; ++i) launch();
    CHECK(cudaDeviceSynchronize());
    const int iters = 20;
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) launch();
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    CHECK(cudaEventElapsedTime(&ms[arm], t0, t1));
    ms[arm] /= iters;
    CHECK(cudaMemcpy((arm == 0 ? ref : got).data(), d_out, h_q.size() * 2,
                     cudaMemcpyDeviceToHost));
  }
  bool match = true;
  for (size_t i = 0; i < ref.size() && match; ++i) match = ref[i] == got[i];
  printf("%14s %10s %10s %8s\n", "", "per-slot", "batched", "delta");
  printf("%14s %10.3f %10.3f %7.1f%%\n", "ms / 5 layers", ms[0] * 5, ms[1] * 5,
         100.0 * (ms[1] - ms[0]) / ms[0]);
  printf("%14s %10.3f %10.3f\n", "ms / launch-set", ms[0], ms[1]);
  printf("output bit-identical: %s\n", match ? "yes" : "NO");

  CHECK(cudaGetLastError());
  return match ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "batch") {
    int slots = argc > 2 ? atoi(argv[2]) : 16;
    int block = argc > 3 ? atoi(argv[3]) : 6;
    int kv_len = argc > 4 ? atoi(argv[4]) : 2048;
    return run_slot_batch_ab(slots, block, kv_len);
  }
  std::vector<int> row_counts = {12, 24, 48, 72, 96};
  if (argc > 1) row_counts = {atoi(argv[1])};

  int max_rows = 0;
  for (int r : row_counts) max_rows = r > max_rows ? r : max_rows;

  size_t q_elems = (size_t)max_rows * NQ * HD;
  size_t kv_elems = (size_t)NKV * CAP * HD;
  std::vector<uint16_t> h_q(q_elems), h_kv(kv_elems);
  for (size_t i = 0; i < q_elems; ++i) h_q[i] = 0x3C00 ^ (uint16_t)(i * 37);
  for (size_t i = 0; i < kv_elems; ++i) h_kv[i] = 0x3C00 ^ (uint16_t)(i * 61);

  __nv_bfloat16 *d_q, *d_k, *d_v, *d_out;
  int *d_win;
  CHECK(cudaMalloc(&d_q, q_elems * 2));
  CHECK(cudaMalloc(&d_out, q_elems * 2));
  CHECK(cudaMalloc(&d_k, kv_elems * 2));
  CHECK(cudaMalloc(&d_v, kv_elems * 2));
  CHECK(cudaMalloc(&d_win, 2 * max_rows * sizeof(int)));
  CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * 2, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_k, h_kv.data(), kv_elems * 2, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_v, h_kv.data(), kv_elems * 2, cudaMemcpyHostToDevice));

  float sm_scale = 1.0f / sqrtf((float)HD);
  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));

  printf("shape: nq=%d nkv=%d (gqa %d) hd=%d cap=%d, steady-state windows (kv_len=%d)\n",
         NQ, NKV, NQ / NKV, HD, CAP, WINDOW);
  printf("%6s %10s %10s %8s %8s\n", "rows", "idiv ms", "fast ms", "delta", "match");

  std::vector<uint16_t> ref(q_elems), got(q_elems);
  for (int rows : row_counts) {
    // Steady state: every draft row sees a full window, bases walk the ring.
    std::vector<int> win(2 * rows);
    for (int t = 0; t < rows; ++t) {
      win[t] = t * 977;  // absolute positions, deliberately past the modulus
      win[rows + t] = WINDOW;
    }
    CHECK(cudaMemcpy(d_win, win.data(), 2 * rows * sizeof(int), cudaMemcpyHostToDevice));

    dim3 grid(NQ, rows);
    size_t out_bytes = (size_t)rows * NQ * HD * 2;
    float ms[2] = {0, 0};
    for (int variant = 0; variant < 2; ++variant) {
      auto launch = [&] {
        if (variant == 0) {
          nonpaged_prefill_attention_kernel<false><<<grid, HD>>>(
              d_q, d_k, d_v, d_out, NQ, NKV, HD, rows, 0, nullptr, CAP, 0, CAP,
              d_win, d_win + rows, sm_scale);
        } else {
          nonpaged_prefill_attention_kernel<true><<<grid, HD>>>(
              d_q, d_k, d_v, d_out, NQ, NKV, HD, rows, 0, nullptr, CAP, 0, CAP,
              d_win, d_win + rows, sm_scale);
        }
      };
      for (int i = 0; i < 3; ++i) launch();
      CHECK(cudaDeviceSynchronize());
      const int iters = 20;
      CHECK(cudaEventRecord(t0));
      for (int i = 0; i < iters; ++i) launch();
      CHECK(cudaEventRecord(t1));
      CHECK(cudaEventSynchronize(t1));
      CHECK(cudaEventElapsedTime(&ms[variant], t0, t1));
      ms[variant] /= iters;
      CHECK(cudaMemcpy((variant == 0 ? ref : got).data(), d_out, out_bytes,
                       cudaMemcpyDeviceToHost));
    }
    // Same values, cheaper arithmetic: the outputs must be bit-identical.
    bool match = true;
    for (size_t i = 0; i < out_bytes / 2 && match; ++i) match = ref[i] == got[i];
    printf("%6d %10.3f %10.3f %7.1f%% %8s\n", rows, ms[0], ms[1],
           100.0 * (ms[1] - ms[0]) / ms[0], match ? "yes" : "NO");
  }

  CHECK(cudaGetLastError());
  return 0;
}

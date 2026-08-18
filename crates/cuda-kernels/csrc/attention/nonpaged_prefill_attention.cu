#include "common.cuh"
#include <cstdint>

#define NONPAGED_PREFILL_TILE 64
#define NONPAGED_PREFILL_MAX_WARPS 8

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
    int ring_base,        // sliding-window ring: absolute position of logical key 0
    int ring_modulus,     // >0: physical row = (ring_base + abs_pos) % ring_modulus
    const int *__restrict__ ring_base_dev,  // per-token ring_base (ragged windows)
    const int *__restrict__ kv_len_dev,     // per-token key count; null = causal
    float sm_scale,
    // Slot batching: non-null promotes blockIdx.z to a slot axis, each slot
    // keeping the separate ring cache its `Qwen35DsparkSlotState` owns. Null
    // leaves the grid 2-D and the single `k_cache`/`v_cache` in force.
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

  // Device-resident position override (CUDA-graph decode replay): when set,
  // the per-step start_pos lives in device memory so ONE captured graph
  // replays across positions instead of baking the host scalar. Same math
  // either way (kv_len below derives from start_pos identically).
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
  // Ragged windows: each query row carries its own [ring_base, ring_base+kv_len)
  // key range (DSpark's per-row sliding lower bound). Absent → causal.
  int kv_len = start_pos + token + 1;
  if (kv_len_dev != nullptr) {
    kv_len = kv_len_dev[token];
    ring_base = ring_base_dev[token];
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
  // Fold the loop-invariant int64 offset into base pointers once (this is the only
  // term that can overflow int32); the hot loop then indexes with int32 row*head_dim
  // (< max_seq_len*head_dim < 2^31), keeping no live 64-bit reg pair across the loop.
  const __nv_bfloat16 *k_base = k_cache + (int64_t)kv_head * max_seq_len * head_dim + dim;
  const __nv_bfloat16 *v_base = v_cache + (int64_t)kv_head * max_seq_len * head_dim + dim;

  for (int tile_start = 0; tile_start < kv_len; tile_start += NONPAGED_PREFILL_TILE) {
    int tile_len = min(NONPAGED_PREFILL_TILE, kv_len - tile_start);

    for (int pos = 0; pos < tile_len; ++pos) {
      int abs_pos = tile_start + pos;
      // Softmax is permutation-invariant, so a wrapped ring walk accumulates
      // identically to a contiguous one — only the physical row differs.
      int row = ring_modulus > 0 ? ((ring_base + abs_pos) % ring_modulus) : abs_pos;
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
      int row = ring_modulus > 0 ? ((ring_base + abs_pos) % ring_modulus) : abs_pos;
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

extern "C" cudaError_t nonpaged_prefill_attention_cuda(
    const uint16_t *q,
    const uint16_t *k_cache,
    const uint16_t *v_cache,
    uint16_t *out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_len,
    int max_seq_len,
    float sm_scale,
    cudaStream_t stream) {
  if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len < 0 || kv_len < seq_len ||
      max_seq_len < kv_len || (head_dim != 128 && head_dim != 256) ||
      num_q_heads % num_kv_heads != 0) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0) {
    return cudaSuccess;
  }
  int start_pos = kv_len - seq_len;
  dim3 grid(num_q_heads, seq_len);
  nonpaged_prefill_attention_kernel<<<grid, head_dim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(q),
      reinterpret_cast<const __nv_bfloat16 *>(k_cache),
      reinterpret_cast<const __nv_bfloat16 *>(v_cache),
      reinterpret_cast<__nv_bfloat16 *>(out),
      num_q_heads,
      num_kv_heads,
      head_dim,
      seq_len,
      start_pos,
      /*start_pos_dev=*/nullptr,
      max_seq_len,
      /*ring_base=*/0,
      /*ring_modulus=*/0,
      /*ring_base_dev=*/nullptr,
      /*kv_len_dev=*/nullptr,
      sm_scale);
  return cudaGetLastError();
}

// Ragged-window ring variant: one launch for `seq_len` query rows whose key
// ranges differ. `ring_base_dev[t]` / `kv_len_dev[t]` (device i32, `seq_len`
// each) give row t's window `[base, base+len)`; the walk is non-causal within
// each row. Replaces DSpark's per-row `seq_len=1` launches — same math, one
// grid. Ranges live on device, so the host cannot bound-check them: the caller
// must guarantee `kv_len_dev[t] <= ring_modulus`.
extern "C" cudaError_t nonpaged_prefill_attention_ring_varlen_cuda(
    const uint16_t *q,
    const uint16_t *k_cache,
    const uint16_t *v_cache,
    uint16_t *out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    const int *ring_base_dev,
    const int *kv_len_dev,
    int ring_modulus,
    float sm_scale,
    cudaStream_t stream) {
  if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len < 0 || ring_modulus <= 0 ||
      ring_base_dev == nullptr || kv_len_dev == nullptr ||
      (head_dim != 128 && head_dim != 256) || num_q_heads % num_kv_heads != 0) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0) {
    return cudaSuccess;
  }
  dim3 grid(num_q_heads, seq_len);
  nonpaged_prefill_attention_kernel<<<grid, head_dim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(q),
      reinterpret_cast<const __nv_bfloat16 *>(k_cache),
      reinterpret_cast<const __nv_bfloat16 *>(v_cache),
      reinterpret_cast<__nv_bfloat16 *>(out),
      num_q_heads,
      num_kv_heads,
      head_dim,
      seq_len,
      /*start_pos=*/0,
      /*start_pos_dev=*/nullptr,
      /*max_seq_len=*/ring_modulus,
      /*ring_base=*/0,
      ring_modulus,
      ring_base_dev,
      kv_len_dev,
      sm_scale);
  return cudaGetLastError();
}

// Slot-batched ring-varlen: one launch over every draft slot, gridZ = slots.
// Each slot keeps its own ring cache, so the bases arrive as a device array of
// `slots` pointers. The window table is `[all bases][all lengths]`, each
// `slots * seq_len` long and slot-major. Same math as the per-slot entry — the
// per-slot loop it replaces ran 192-block grids that left the GPU 85% idle.
extern "C" cudaError_t nonpaged_prefill_attention_ring_varlen_batched_cuda(
    const uint16_t *q,
    const void *const *k_slots,
    const void *const *v_slots,
    uint16_t *out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int slots,
    const int *ring_base_dev,
    const int *kv_len_dev,
    int ring_modulus,
    float sm_scale,
    cudaStream_t stream) {
  if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len < 0 || slots < 0 ||
      ring_modulus <= 0 || k_slots == nullptr || v_slots == nullptr ||
      ring_base_dev == nullptr || kv_len_dev == nullptr ||
      (head_dim != 128 && head_dim != 256) || num_q_heads % num_kv_heads != 0) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0 || slots == 0) {
    return cudaSuccess;
  }
  dim3 grid(num_q_heads, seq_len, slots);
  nonpaged_prefill_attention_kernel<<<grid, head_dim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(q),
      /*k_cache=*/nullptr,
      /*v_cache=*/nullptr,
      reinterpret_cast<__nv_bfloat16 *>(out),
      num_q_heads,
      num_kv_heads,
      head_dim,
      seq_len,
      /*start_pos=*/0,
      /*start_pos_dev=*/nullptr,
      /*max_seq_len=*/ring_modulus,
      /*ring_base=*/0,
      ring_modulus,
      ring_base_dev,
      kv_len_dev,
      sm_scale,
      reinterpret_cast<const __nv_bfloat16 *const *>(k_slots),
      reinterpret_cast<const __nv_bfloat16 *const *>(v_slots));
  return cudaGetLastError();
}

// Device-position variant: start_pos is read from `start_pos_dev` (one i32 in
// device memory) instead of a host launch arg, so the launch is CUDA-graph
// capture-safe — replaying the captured graph re-reads the staged position
// every step (the Qwen3.5/3.6 whole-step decode graph stages it pre-replay).
// Grid/block depend only on (num_q_heads, seq_len, head_dim), all
// shape-constant at decode; the kv walk is bounded inside the kernel by the
// device-read start_pos. kv-length host validation is impossible here (the
// value lives on device); callers must enforce start_pos + seq_len <=
// max_seq_len host-side before staging.
extern "C" cudaError_t nonpaged_prefill_attention_devpos_cuda(
    const uint16_t *q,
    const uint16_t *k_cache,
    const uint16_t *v_cache,
    uint16_t *out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    const int *start_pos_dev,
    int max_seq_len,
    float sm_scale,
    cudaStream_t stream) {
  if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len < 0 ||
      max_seq_len < seq_len || (head_dim != 128 && head_dim != 256) ||
      num_q_heads % num_kv_heads != 0 || start_pos_dev == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0) {
    return cudaSuccess;
  }
  dim3 grid(num_q_heads, seq_len);
  nonpaged_prefill_attention_kernel<<<grid, head_dim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(q),
      reinterpret_cast<const __nv_bfloat16 *>(k_cache),
      reinterpret_cast<const __nv_bfloat16 *>(v_cache),
      reinterpret_cast<__nv_bfloat16 *>(out),
      num_q_heads,
      num_kv_heads,
      head_dim,
      seq_len,
      /*start_pos=*/0,
      start_pos_dev,
      max_seq_len,
      /*ring_base=*/0,
      /*ring_modulus=*/0,
      /*ring_base_dev=*/nullptr,
      /*kv_len_dev=*/nullptr,
      sm_scale);
  return cudaGetLastError();
}

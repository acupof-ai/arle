#include "common.cuh"

// ============================================================================
// Argmax: find index of maximum value (fast single-request version)
// ============================================================================
// 1 block of 1024 threads. Each thread processes ceil(n/1024) elements using
// vectorized bf16x2 loads where possible. Final reduction via warp shuffles
// to avoid shared memory bank conflicts in the tail.
// ============================================================================

#define ARGMAX_BLOCK 1024
#define ARGMAX_NUM_WARPS (ARGMAX_BLOCK / WARP_SIZE)

// Warp-level argmax reduction: returns (max_val, max_idx) in lane 0.
__device__ __forceinline__ void warp_reduce_argmax(float &val, int &idx) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

__global__ void argmax_kernel_fast(const __nv_bfloat16 *__restrict__ x,
                                   int *__restrict__ out, int n) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    float local_max = -INFINITY;
    int local_idx = 0;

    // Vectorized bf16x2 loads: process 2 elements per load
    int n_pairs = n / 2;
    const __nv_bfloat162 *x2 = (const __nv_bfloat162 *)x;
    for (int i = tid; i < n_pairs; i += ARGMAX_BLOCK) {
        __nv_bfloat162 pair = x2[i];
        float v0 = __bfloat162float(pair.x);
        float v1 = __bfloat162float(pair.y);
        int i0 = i * 2;
        int i1 = i0 + 1;
        if (v0 > local_max || (v0 == local_max && i0 < local_idx)) {
            local_max = v0;
            local_idx = i0;
        }
        if (v1 > local_max || (v1 == local_max && i1 < local_idx)) {
            local_max = v1;
            local_idx = i1;
        }
    }
    // Handle odd trailing element
    if (n & 1) {
        int last = n - 1;
        if ((last % ARGMAX_BLOCK) == tid || (tid == 0 && ARGMAX_BLOCK > n)) {
            float v = __bfloat162float(x[last]);
            if (v > local_max || (v == local_max && last < local_idx)) {
                local_max = v;
                local_idx = last;
            }
        }
    }

    // Warp-level reduction
    warp_reduce_argmax(local_max, local_idx);

    // Write warp results to shared memory
    __shared__ float warp_vals[ARGMAX_NUM_WARPS];
    __shared__ int warp_idxs[ARGMAX_NUM_WARPS];
    if (lane_id == 0) {
        warp_vals[warp_id] = local_max;
        warp_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        float val = (lane_id < ARGMAX_NUM_WARPS) ? warp_vals[lane_id] : -INFINITY;
        int idx = (lane_id < ARGMAX_NUM_WARPS) ? warp_idxs[lane_id] : 0;
        warp_reduce_argmax(val, idx);
        if (lane_id == 0) {
            out[0] = idx;
        }
    }
}

// ============================================================================
// Batched argmax: one block per request, processes B logit vectors in one launch
// ============================================================================
// Input:  logits [B, vocab_size] contiguous bf16
// Output: token_ids [B] int32
// Launch: B blocks of 1024 threads
// ============================================================================

__global__ void argmax_batch_kernel(const __nv_bfloat16 *__restrict__ logits,
                                    int *__restrict__ token_ids,
                                    int vocab_size) {
    int bid = blockIdx.x;   // request index
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    const __nv_bfloat16 *row = logits + (long long)bid * vocab_size;

    float local_max = -INFINITY;
    int local_idx = 0;

    // Vectorized bf16x2 loads
    int n_pairs = vocab_size / 2;
    const __nv_bfloat162 *row2 = (const __nv_bfloat162 *)row;
    for (int i = tid; i < n_pairs; i += ARGMAX_BLOCK) {
        __nv_bfloat162 pair = row2[i];
        float v0 = __bfloat162float(pair.x);
        float v1 = __bfloat162float(pair.y);
        int i0 = i * 2;
        int i1 = i0 + 1;
        if (v0 > local_max || (v0 == local_max && i0 < local_idx)) {
            local_max = v0;
            local_idx = i0;
        }
        if (v1 > local_max || (v1 == local_max && i1 < local_idx)) {
            local_max = v1;
            local_idx = i1;
        }
    }
    // Handle odd trailing element
    if ((vocab_size & 1) && tid == 0) {
        int last = vocab_size - 1;
        float v = __bfloat162float(row[last]);
        if (v > local_max || (v == local_max && last < local_idx)) {
            local_max = v;
            local_idx = last;
        }
    }

    // Warp-level reduction
    warp_reduce_argmax(local_max, local_idx);

    // Write warp results to shared memory
    __shared__ float warp_vals[ARGMAX_NUM_WARPS];
    __shared__ int warp_idxs[ARGMAX_NUM_WARPS];
    if (lane_id == 0) {
        warp_vals[warp_id] = local_max;
        warp_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        float val = (lane_id < ARGMAX_NUM_WARPS) ? warp_vals[lane_id] : -INFINITY;
        int idx = (lane_id < ARGMAX_NUM_WARPS) ? warp_idxs[lane_id] : 0;
        warp_reduce_argmax(val, idx);
        if (lane_id == 0) {
            token_ids[bid] = idx;
        }
    }
}

// ============================================================================
// GPU Sampling: temperature → softmax → top-k → top-p → multinomial
// Single block, 256 threads. Requires FP32 scratch buffer of vocab_size.
// Random number and sampling parameters passed from CPU.
// ============================================================================
#define SAMPLE_BLOCK 256
#define SAMPLE_NUM_WARPS (SAMPLE_BLOCK / WARP_SIZE)

__global__ void gpu_sample_kernel(
    const __nv_bfloat16 *__restrict__ logits,
    float *__restrict__ probs,     // scratch buffer [vocab_size]
    int *__restrict__ output,      // single int: sampled token id
    int vocab_size,
    float inv_temperature,
    int top_k,                     // ≤ 0 means disabled
    float top_p,                   // ≥ 1.0 means disabled
    float random_val               // uniform [0, 1)
) {
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  // ---- Pass 1: Temperature scale + find max ----
  float local_max = -INFINITY;
  for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
    float v = __bfloat162float(logits[i]) * inv_temperature;
    probs[i] = v;
    local_max = fmaxf(local_max, v);
  }

  local_max = warp_reduce_max(local_max);
  __shared__ float warp_vals[SAMPLE_NUM_WARPS];
  if (lane_id == 0) warp_vals[warp_id] = local_max;
  __syncthreads();
  if (warp_id == 0) {
    float v = (lane_id < SAMPLE_NUM_WARPS) ? warp_vals[lane_id] : -INFINITY;
    v = warp_reduce_max(v);
    if (lane_id == 0) warp_vals[0] = v;
  }
  __syncthreads();
  float global_max = warp_vals[0];

  // ---- Pass 2: Softmax (exp + sum + normalize) ----
  float local_sum = 0.0f;
  for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
    float v = expf(probs[i] - global_max);
    probs[i] = v;
    local_sum += v;
  }

  local_sum = warp_reduce_sum(local_sum);
  if (lane_id == 0) warp_vals[warp_id] = local_sum;
  __syncthreads();
  if (warp_id == 0) {
    float v = (lane_id < SAMPLE_NUM_WARPS) ? warp_vals[lane_id] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane_id == 0) warp_vals[0] = v;
  }
  __syncthreads();
  float inv_sum = 1.0f / warp_vals[0];

  for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
    probs[i] *= inv_sum;
  }
  __syncthreads();

  // ---- Pass 3: Top-k filtering (binary search for k-th largest prob) ----
  if (top_k > 0 && top_k < vocab_size) {
    // Find max prob for binary search upper bound
    float local_pmax = 0.0f;
    for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
      local_pmax = fmaxf(local_pmax, probs[i]);
    }
    local_pmax = warp_reduce_max(local_pmax);
    if (lane_id == 0) warp_vals[warp_id] = local_pmax;
    __syncthreads();
    if (warp_id == 0) {
      float v = (lane_id < SAMPLE_NUM_WARPS) ? warp_vals[lane_id] : 0.0f;
      v = warp_reduce_max(v);
      if (lane_id == 0) warp_vals[0] = v;
    }
    __syncthreads();
    float pmax = warp_vals[0];

    // Binary search: find threshold such that count(prob >= threshold) <= top_k
    float lo = 0.0f, hi = pmax;
    __shared__ int shared_count;

    for (int iter = 0; iter < 32; iter++) {
      float mid = (lo + hi) * 0.5f;
      // Count elements >= mid
      int local_count = 0;
      for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
        if (probs[i] >= mid) local_count++;
      }
      // Reduce count
      // Warp-reduce local_count (compatible with sm_75+)
      for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
      if (lane_id == 0) warp_vals[warp_id] = (float)local_count;
      __syncthreads();
      int total_count;
      if (warp_id == 0) {
        float v = (lane_id < SAMPLE_NUM_WARPS) ? warp_vals[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) {
          shared_count = (int)v;
        }
      }
      __syncthreads();
      total_count = shared_count;

      if (total_count > top_k) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    // Zero out elements below threshold and renormalize
    float threshold = lo;
    float local_kept_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
      if (probs[i] < threshold) {
        probs[i] = 0.0f;
      } else {
        local_kept_sum += probs[i];
      }
    }
    local_kept_sum = warp_reduce_sum(local_kept_sum);
    if (lane_id == 0) warp_vals[warp_id] = local_kept_sum;
    __syncthreads();
    if (warp_id == 0) {
      float v = (lane_id < SAMPLE_NUM_WARPS) ? warp_vals[lane_id] : 0.0f;
      v = warp_reduce_sum(v);
      if (lane_id == 0) warp_vals[0] = v;
    }
    __syncthreads();
    float kept_inv = 1.0f / warp_vals[0];
    for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
      probs[i] *= kept_inv;
    }
    __syncthreads();
  }

  // ---- Pass 4: Top-p filtering (binary search for cumulative sum threshold) ----
  if (top_p < 1.0f) {
    // Find threshold such that sum(prob >= threshold) >= top_p
    float local_pmax = 0.0f;
    for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
      local_pmax = fmaxf(local_pmax, probs[i]);
    }
    local_pmax = warp_reduce_max(local_pmax);
    if (lane_id == 0) warp_vals[warp_id] = local_pmax;
    __syncthreads();
    if (warp_id == 0) {
      float v = (lane_id < SAMPLE_NUM_WARPS) ? warp_vals[lane_id] : 0.0f;
      v = warp_reduce_max(v);
      if (lane_id == 0) warp_vals[0] = v;
    }
    __syncthreads();
    float pmax = warp_vals[0];

    float lo = 0.0f, hi = pmax;
    __shared__ float shared_sum_above;

    for (int iter = 0; iter < 32; iter++) {
      float mid = (lo + hi) * 0.5f;
      float local_above = 0.0f;
      for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
        if (probs[i] >= mid) local_above += probs[i];
      }
      local_above = warp_reduce_sum(local_above);
      if (lane_id == 0) warp_vals[warp_id] = local_above;
      __syncthreads();
      if (warp_id == 0) {
        float v = (lane_id < SAMPLE_NUM_WARPS) ? warp_vals[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) shared_sum_above = v;
      }
      __syncthreads();
      float total_above = shared_sum_above;

      if (total_above > top_p) {
        lo = mid; // threshold too low, raise it
      } else {
        hi = mid; // threshold too high, lower it
      }
    }

    // Zero out elements below threshold and renormalize
    float threshold = lo;
    float local_kept_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
      if (probs[i] < threshold) {
        probs[i] = 0.0f;
      } else {
        local_kept_sum += probs[i];
      }
    }
    local_kept_sum = warp_reduce_sum(local_kept_sum);
    if (lane_id == 0) warp_vals[warp_id] = local_kept_sum;
    __syncthreads();
    if (warp_id == 0) {
      float v = (lane_id < SAMPLE_NUM_WARPS) ? warp_vals[lane_id] : 0.0f;
      v = warp_reduce_sum(v);
      if (lane_id == 0) warp_vals[0] = v;
    }
    __syncthreads();
    float kept_inv = 1.0f / warp_vals[0];
    for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
      probs[i] *= kept_inv;
    }
    __syncthreads();
  }

  // ---- Pass 5: Multinomial sampling via parallel CDF ----
  // Each thread computes partial sum for its strided range
  __shared__ float partial_sums[SAMPLE_BLOCK];
  float my_sum = 0.0f;
  for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
    my_sum += probs[i];
  }
  partial_sums[tid] = my_sum;
  __syncthreads();

  // Prefix sum on partial_sums (thread 0 does sequential scan — only 256 elements)
  __shared__ float prefix[SAMPLE_BLOCK];
  if (tid == 0) {
    float running = 0.0f;
    for (int t = 0; t < SAMPLE_BLOCK; t++) {
      prefix[t] = running;
      running += partial_sums[t];
    }
  }
  __syncthreads();

  // Each thread checks if the sample falls in its range
  float range_start = prefix[tid];
  float range_end = range_start + partial_sums[tid];

  if (random_val >= range_start && random_val < range_end) {
    // Linear scan within this thread's strided elements
    float cumsum = range_start;
    for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
      cumsum += probs[i];
      if (random_val < cumsum) {
        output[0] = i;
        return;
      }
    }
    // Fallback: last element in this thread's range
    int last_idx = tid;
    for (int i = tid; i < vocab_size; i += SAMPLE_BLOCK) {
      if (probs[i] > 0.0f) last_idx = i;
    }
    output[0] = last_idx;
  }
}

extern "C" {
cudaError_t argmax_cuda(const __nv_bfloat16 *x, int *out, int n, cudaStream_t stream) {
  argmax_kernel_fast<<<1, ARGMAX_BLOCK, 0, stream>>>(x, out, n);
    return cudaGetLastError();
}

cudaError_t argmax_batch_cuda(const __nv_bfloat16 *logits, int *token_ids,
                       int batch_size, int vocab_size, cudaStream_t stream) {
  argmax_batch_kernel<<<batch_size, ARGMAX_BLOCK, 0, stream>>>(logits, token_ids, vocab_size);
    return cudaGetLastError();
}

cudaError_t gpu_sample_cuda(const __nv_bfloat16 *logits, float *probs_scratch, int *output,
                     int vocab_size, float inv_temperature, int top_k, float top_p,
                     float random_val, cudaStream_t stream) {
  gpu_sample_kernel<<<1, SAMPLE_BLOCK, 0, stream>>>(
      logits, probs_scratch, output, vocab_size, inv_temperature, top_k, top_p, random_val);
    return cudaGetLastError();
}
}

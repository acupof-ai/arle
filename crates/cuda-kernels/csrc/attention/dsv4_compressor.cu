#include "dsv4_attention_common.cuh"

template <typename T>
__device__ __forceinline__ float dsv4_compressor_load(T value);

template <>
__device__ __forceinline__ float dsv4_compressor_load(uint16_t value) {
  return dsv4_attn_bf16_to_f32(value);
}

template <>
__device__ __forceinline__ float dsv4_compressor_load(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ T dsv4_compressor_store(float value);

template <>
__device__ __forceinline__ uint16_t dsv4_compressor_store(float value) {
  return dsv4_attn_f32_to_bf16_bits(value);
}

template <>
__device__ __forceinline__ float dsv4_compressor_store(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ float dsv4_compressor_raw_value(
    const T *__restrict__ raw,
    const T *__restrict__ pending,
    int abs_pos,
    int start_pos,
    int block_start,
    int width,
    int col) {
  if (abs_pos < start_pos) {
    int pending_pos = abs_pos - block_start;
    return dsv4_compressor_load(pending[pending_pos * width + col]);
  }
  return dsv4_compressor_load(raw[(abs_pos - start_pos) * width + col]);
}

template <typename T>
__device__ __forceinline__ float dsv4_compressor_score_value(
    const T *__restrict__ raw,
    const T *__restrict__ pending,
    const T *__restrict__ ape,
    int abs_pos,
    int start_pos,
    int block_start,
    int ratio,
    int width,
    int col) {
  if (abs_pos < start_pos) {
    int pending_pos = abs_pos - block_start;
    return dsv4_compressor_load(pending[pending_pos * width + col]);
  }
  return dsv4_compressor_load(raw[(abs_pos - start_pos) * width + col]) +
         dsv4_compressor_load(ape[(abs_pos % ratio) * width + col]);
}

// Per-row compressor state update body, shared by the single-block global
// `dsv4_compressor_update_kernel` and the batched (grid-over-rows)
// `dsv4_compressor_update_batched_kernel`. Each invocation runs on ONE CUDA
// block and writes ONE row's per-slot `pending_kv`/`pending_score`/`compressed`.
// The batched kernel calls this with this row's pointers + this row's resolved
// start_pos so the math is byte-identical to the per-row launch. `start_pos`/
// `pending_len`/`compressed_base`/`has_prev_overlap` are ALREADY resolved by the
// caller (the batched kernel resolves them from `start_pos_ptr[row]` before
// calling).
//
// `prev_overlap_kv`/`prev_overlap_score` can be either a per-row/per-slot
// single register (one `ratio*head_dim` page, serially overwritten) or a flat,
// cross-request-shared pool (one `ratio*head_dim` page per compress-block
// position, `capacity_blocks = max_seq_len/ratio` pages total). `overlap_page_
// stride` selects which: `0` ⇒ every offset below collapses to `0` (the
// single-register form); `ratio*head_dim` ⇒ offset by `(compressed_base ±
// block) * overlap_page_stride`, keyed by absolute block position (not by
// which row/slot produced it), so any two rows sharing the same prefix
// position see the identical, correct carry-state.
template <typename T>
__device__ void dsv4_compressor_update_body(
    const T *__restrict__ kv_raw,
    const T *__restrict__ score_raw,
    const T *__restrict__ ape,
    const uint16_t *__restrict__ norm,
    T *__restrict__ pending_kv,
    T *__restrict__ pending_score,
    T *__restrict__ prev_overlap_kv,
    T *__restrict__ prev_overlap_score,
    uint16_t *__restrict__ compressed,
    int num_tokens,
    int start_pos,
    int pending_len,
    int compressed_base,
    int head_dim,
    int ratio,
    int width,
    int overlap,
    int has_prev_overlap,
    // Elements-per-page stride for `prev_overlap_kv/score` pool addressing.
    // `0` = single-register buffer; `ratio*head_dim` = shared pool, addressed
    // by absolute block position.
    int overlap_page_stride,
    float eps,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  __shared__ float row[DSV4_ATTN_MAX_HEAD_DIM];
  int total = pending_len + num_tokens;
  int completed = total / ratio;
  int block_start0 = start_pos - pending_len;

  if (completed == 0) {
    int raw_elems = num_tokens * width;
    for (int idx = threadIdx.x; idx < raw_elems; idx += blockDim.x) {
      int pos = idx / width;
      int col = idx - pos * width;
      int abs_pos = start_pos + pos;
      int dst = (pending_len + pos) * width + col;
      float kv = dsv4_compressor_load(kv_raw[idx]);
      float score = dsv4_compressor_load(score_raw[idx]) +
                    dsv4_compressor_load(ape[(abs_pos % ratio) * width + col]);
      pending_kv[dst] = dsv4_compressor_store<T>(kv);
      pending_score[dst] = dsv4_compressor_store<T>(score);
    }
    return;
  }

  for (int block = 0; block < completed; ++block) {
    int block_start = block_start0 + block * ratio;
    int block_end = block_start + ratio - 1;
    // `read_block = compressed_base + block - 1` is only ever MULTIPLIED into
    // an offset inside the `(has_prev_overlap || block > 0)` guard, which is
    // false exactly when `compressed_base == 0 && block == 0` — the ternary's
    // untaken branch is never evaluated, so the `-1` case never subscripts
    // memory.
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
      float max_logit = -INFINITY;
      int count = overlap ? 2 * ratio : ratio;
      for (int pos = 0; pos < count; ++pos) {
        float logit;
        if (overlap && pos < ratio) {
          int read_block = compressed_base + block - 1;
          logit = (has_prev_overlap || block > 0)
                      ? dsv4_compressor_load(prev_overlap_score[read_block * overlap_page_stride + pos * head_dim + col])
                      : -INFINITY;
        } else {
          int local_pos = overlap ? (pos - ratio) : pos;
          int abs_pos = block_start + local_pos;
          int score_col = overlap ? (head_dim + col) : col;
          logit = dsv4_compressor_score_value(
              score_raw, pending_score, ape, abs_pos, start_pos, block_start,
              ratio, width, score_col);
        }
        max_logit = fmaxf(max_logit, logit);
      }
      float denom = 0.0f;
      float acc = 0.0f;
      if (isfinite(max_logit)) {
        for (int pos = 0; pos < count; ++pos) {
          float logit;
          float raw_value;
          if (overlap && pos < ratio) {
            if (has_prev_overlap || block > 0) {
              int read_block = compressed_base + block - 1;
              logit = dsv4_compressor_load(prev_overlap_score[read_block * overlap_page_stride + pos * head_dim + col]);
              raw_value = dsv4_compressor_load(prev_overlap_kv[read_block * overlap_page_stride + pos * head_dim + col]);
            } else {
              logit = -INFINITY;
              raw_value = 0.0f;
            }
          } else {
            int local_pos = overlap ? (pos - ratio) : pos;
            int abs_pos = block_start + local_pos;
            int score_col = overlap ? (head_dim + col) : col;
            int kv_col = overlap ? (head_dim + col) : col;
            logit = dsv4_compressor_score_value(
                score_raw, pending_score, ape, abs_pos, start_pos, block_start,
                ratio, width, score_col);
            raw_value = dsv4_compressor_raw_value(
                kv_raw, pending_kv, abs_pos, start_pos, block_start, width, kv_col);
          }
          float weight = expf(logit - max_logit);
          denom += weight;
          acc += weight * raw_value;
        }
        if (denom > 0.0f) {
          acc /= denom;
        }
      }
      row[col] = acc;
    }
    __syncthreads();

    float sumsq = 0.0f;
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
      sumsq += row[col] * row[col];
    }
    sumsq = dsv4_attn_block_sum(sumsq);
    __shared__ float norm_scale;
    if (threadIdx.x == 0) {
      norm_scale = rsqrtf(sumsq / fmaxf((float)head_dim, 1.0f) + eps);
    }
    __syncthreads();

    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
      float value = row[col] * norm_scale * dsv4_attn_bf16_to_f32(norm[col]);
      if (rope_dim > 0 && col >= head_dim - rope_dim) {
        int local = col - (head_dim - rope_dim);
        int pair = local / 2;
        int pair_col = head_dim - rope_dim + pair * 2;
        float a = row[pair_col] * norm_scale * dsv4_attn_bf16_to_f32(norm[pair_col]);
        float b = row[pair_col + 1] * norm_scale * dsv4_attn_bf16_to_f32(norm[pair_col + 1]);
        float out_a;
        float out_b;
        dsv4_apply_rope_pair(
            a, b, pair, block_end, rope_dim, rope_base, original_seq_len,
            factor, beta_fast, beta_slow, 1.0f, &out_a, &out_b);
        value = (local & 1) == 0 ? out_a : out_b;
      }
      compressed[(compressed_base + block) * head_dim + col] =
          dsv4_attn_f32_to_bf16_bits(value);
    }
    __syncthreads();

    if (overlap) {
      // WRITE this block's own completed page (stride 0 ⇒ offset 0, old
      // single-register form). `write_block >= 0` always, no underflow risk.
      int write_block = compressed_base + block;
      for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
        for (int pos = 0; pos < ratio; ++pos) {
          int abs_pos = block_start + pos;
          float kv = dsv4_compressor_raw_value(
              kv_raw, pending_kv, abs_pos, start_pos, block_start, width, col);
          float score = dsv4_compressor_score_value(
              score_raw, pending_score, ape, abs_pos, start_pos, block_start,
              ratio, width, col);
          prev_overlap_kv[write_block * overlap_page_stride + pos * head_dim + col] = dsv4_compressor_store<T>(kv);
          prev_overlap_score[write_block * overlap_page_stride + pos * head_dim + col] = dsv4_compressor_store<T>(score);
        }
      }
    }
    __syncthreads();
  }

  int new_pending = total - completed * ratio;
  int tail_start = start_pos + num_tokens - new_pending;
  for (int idx = threadIdx.x; idx < new_pending * width; idx += blockDim.x) {
    int pos = idx / width;
    int col = idx - pos * width;
    int abs_pos = tail_start + pos;
    float kv = dsv4_compressor_raw_value(
        kv_raw, pending_kv, abs_pos, start_pos, block_start0, width, col);
    float score = dsv4_compressor_score_value(
        score_raw, pending_score, ape, abs_pos, start_pos, block_start0,
        ratio, width, col);
    pending_kv[idx] = dsv4_compressor_store<T>(kv);
    pending_score[idx] = dsv4_compressor_store<T>(score);
  }
}

// Single-row launch (`<<<1, BLOCK>>>`): resolves the decode start_pos/pending/
// base from `start_pos_ptr[0]` (or the scalar args), then runs the shared body.
// Behavior byte-identical to the pre-refactor monolithic kernel.
__global__ void dsv4_compressor_update_kernel(
    const uint16_t *__restrict__ kv_raw,
    const uint16_t *__restrict__ score_raw,
    const uint16_t *__restrict__ ape,
    const uint16_t *__restrict__ norm,
    uint16_t *__restrict__ pending_kv,
    uint16_t *__restrict__ pending_score,
    uint16_t *__restrict__ prev_overlap_kv,
    uint16_t *__restrict__ prev_overlap_score,
    uint16_t *__restrict__ compressed,
    int num_tokens,
    int start_pos,
    const int *__restrict__ start_pos_ptr,
    int pending_len,
    int compressed_base,
    int head_dim,
    int ratio,
    int width,
    int overlap,
    int has_prev_overlap,
    int overlap_page_stride,
    float eps,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  if (start_pos_ptr != nullptr) {
    start_pos = start_pos_ptr[0];
    pending_len = start_pos % ratio;
    compressed_base = start_pos / ratio;
    has_prev_overlap = compressed_base > 0;
  }
  dsv4_compressor_update_body(
      kv_raw, score_raw, ape, norm, pending_kv, pending_score, prev_overlap_kv,
      prev_overlap_score, compressed, num_tokens, start_pos, pending_len,
      compressed_base, head_dim, ratio, width, overlap, has_prev_overlap,
      overlap_page_stride, eps, rope_dim, rope_base, original_seq_len, factor,
      beta_fast, beta_slow);
}

// Batched decode compressor update (`<<<n, BLOCK>>>`, blockIdx.x = row): each
// block updates ONE row's per-slot ring state, reading this row's state buffer
// pointers from the host-gathered pointer ARRAYS and this row's absolute decode
// position from `start_pos_arr[row]`. `kv_raw`/`score_raw` are the batched m=N
// prepass outputs `[width, n]` (token-major: row r occupies the contiguous
// `[r*width, (r+1)*width)` span, == the single-row `[width,1]` slice). `ape`/
// `norm` are the SHARED compressor weights (same for all rows). num_tokens=1 per
// row (decode); start_pos/pending/base/has_prev_overlap are resolved per row
// EXACTLY as the single-row start_pos_ptr launcher does, so the body math is
// byte-identical to n single-row launches.
__global__ void dsv4_compressor_update_batched_kernel(
    const uint16_t *__restrict__ kv_raw,
    const uint16_t *__restrict__ score_raw,
    const uint16_t *__restrict__ ape,
    const uint16_t *__restrict__ norm,
    uint16_t *const *__restrict__ pending_kv_arr,
    uint16_t *const *__restrict__ pending_score_arr,
    uint16_t *const *__restrict__ prev_overlap_kv_arr,
    uint16_t *const *__restrict__ prev_overlap_score_arr,
    uint16_t *const *__restrict__ compressed_arr,
    int n,
    int num_tokens,
    const int *__restrict__ start_pos_arr,
    int head_dim,
    int ratio,
    int width,
    int overlap,
    // `0` (single-register) or `ratio*head_dim` (shared pool) — uniform across
    // all rows in one launch, since one launch is always one (layer,
    // compress_ratio) class.
    int overlap_page_stride,
    float eps,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  int rowi = blockIdx.x;
  if (rowi >= n) return;
  int start_pos = start_pos_arr[rowi];
  int pending_len = start_pos % ratio;
  int compressed_base = start_pos / ratio;
  int has_prev_overlap = compressed_base > 0;
  dsv4_compressor_update_body(
      kv_raw + rowi * num_tokens * width,
      score_raw + rowi * num_tokens * width,
      ape,
      norm,
      pending_kv_arr[rowi],
      pending_score_arr[rowi],
      prev_overlap_kv_arr[rowi],
      prev_overlap_score_arr[rowi],
      compressed_arr[rowi],
      num_tokens, start_pos, pending_len, compressed_base, head_dim, ratio,
      width, overlap, has_prev_overlap, overlap_page_stride, eps, rope_dim,
      rope_base, original_seq_len, factor, beta_fast, beta_slow);
}

// Parallel prefill compressor: one CUDA block per compressed-output block (grid =
// `completed`), so the whole SM array fills instead of a single block looping
// serially. Block b>0 reads its overlap "previous-block" tokens DIRECTLY from
// kv_raw/score_raw (first-half projection, col) — the serial `prev_overlap` carry
// was only a re-read optimization, the source tokens are addressable. Block 0's
// cross-chunk overlap still reads the (frozen, prior-chunk) prev_overlap input.
// prev_overlap/pending WRITES are deferred to dsv4_compressor_finalize_kernel so
// there is no in-flight read/write race on those buffers.
// T = uint16_t (bf16 default path) or float (FP32 prefill path, #146/#150 —
// exact carry, so the parallel re-read is bit-identical to the serial sweep).
template <typename T>
__global__ void dsv4_compressor_block_kernel(
    const T *__restrict__ kv_raw,
    const T *__restrict__ score_raw,
    const T *__restrict__ ape,
    const uint16_t *__restrict__ norm,
    const T *__restrict__ pending_kv,
    const T *__restrict__ pending_score,
    const T *__restrict__ prev_overlap_kv,
    const T *__restrict__ prev_overlap_score,
    uint16_t *__restrict__ compressed,
    int start_pos,
    int pending_len,
    int compressed_base,
    int head_dim,
    int ratio,
    int width,
    int overlap,
    int has_prev_overlap,
    // See `dsv4_compressor_update_body`: `0` collapses every offset below to
    // the single-register form.
    int overlap_page_stride,
    float eps,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  int block = blockIdx.x;
  int block_start0 = start_pos - pending_len;
  int block_start = block_start0 + block * ratio;
  int block_end = block_start + ratio - 1;
  int prev_block_start = block_start - ratio;
  __shared__ float row[DSV4_ATTN_MAX_HEAD_DIM];
  int count = overlap ? 2 * ratio : ratio;
  // Block 0's cross-chunk overlap reads the LAST block completed by a PRIOR
  // call, i.e. absolute block index `compressed_base - 1`. Only formed inside
  // the `has_prev_overlap` guard (false at compressed_base==0), so never
  // underflows a live subscript — matches `dsv4_compressor_update_body`.
  int prev_overlap_block = compressed_base - 1;

  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    float max_logit = -INFINITY;
    for (int pos = 0; pos < count; ++pos) {
      float logit;
      if (overlap && pos < ratio) {
        if (block == 0) {
          logit = has_prev_overlap
                      ? dsv4_compressor_load(prev_overlap_score[prev_overlap_block * overlap_page_stride + pos * head_dim + col])
                      : -INFINITY;
        } else {
          int abs_pos = prev_block_start + pos;
          logit = dsv4_compressor_score_value(
              score_raw, pending_score, ape, abs_pos, start_pos, prev_block_start,
              ratio, width, col);
        }
      } else {
        int local_pos = overlap ? (pos - ratio) : pos;
        int abs_pos = block_start + local_pos;
        int score_col = overlap ? (head_dim + col) : col;
        logit = dsv4_compressor_score_value(
            score_raw, pending_score, ape, abs_pos, start_pos, block_start,
            ratio, width, score_col);
      }
      max_logit = fmaxf(max_logit, logit);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    if (isfinite(max_logit)) {
      for (int pos = 0; pos < count; ++pos) {
        float logit;
        float raw_value;
        if (overlap && pos < ratio) {
          if (block == 0) {
            if (has_prev_overlap) {
              logit = dsv4_compressor_load(prev_overlap_score[prev_overlap_block * overlap_page_stride + pos * head_dim + col]);
              raw_value = dsv4_compressor_load(prev_overlap_kv[prev_overlap_block * overlap_page_stride + pos * head_dim + col]);
            } else {
              logit = -INFINITY;
              raw_value = 0.0f;
            }
          } else {
            int abs_pos = prev_block_start + pos;
            logit = dsv4_compressor_score_value(
                score_raw, pending_score, ape, abs_pos, start_pos, prev_block_start,
                ratio, width, col);
            raw_value = dsv4_compressor_raw_value(
                kv_raw, pending_kv, abs_pos, start_pos, prev_block_start, width, col);
          }
        } else {
          int local_pos = overlap ? (pos - ratio) : pos;
          int abs_pos = block_start + local_pos;
          int score_col = overlap ? (head_dim + col) : col;
          int kv_col = overlap ? (head_dim + col) : col;
          logit = dsv4_compressor_score_value(
              score_raw, pending_score, ape, abs_pos, start_pos, block_start,
              ratio, width, score_col);
          raw_value = dsv4_compressor_raw_value(
              kv_raw, pending_kv, abs_pos, start_pos, block_start, width, kv_col);
        }
        float weight = expf(logit - max_logit);
        denom += weight;
        acc += weight * raw_value;
      }
      if (denom > 0.0f) {
        acc /= denom;
      }
    }
    row[col] = acc;
  }
  __syncthreads();

  float sumsq = 0.0f;
  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    sumsq += row[col] * row[col];
  }
  sumsq = dsv4_attn_block_sum(sumsq);
  __shared__ float norm_scale;
  if (threadIdx.x == 0) {
    norm_scale = rsqrtf(sumsq / fmaxf((float)head_dim, 1.0f) + eps);
  }
  __syncthreads();

  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    float value = row[col] * norm_scale * dsv4_attn_bf16_to_f32(norm[col]);
    if (rope_dim > 0 && col >= head_dim - rope_dim) {
      int local = col - (head_dim - rope_dim);
      int pair = local / 2;
      int pair_col = head_dim - rope_dim + pair * 2;
      float a = row[pair_col] * norm_scale * dsv4_attn_bf16_to_f32(norm[pair_col]);
      float b = row[pair_col + 1] * norm_scale * dsv4_attn_bf16_to_f32(norm[pair_col + 1]);
      float out_a;
      float out_b;
      dsv4_apply_rope_pair(
          a, b, pair, block_end, rope_dim, rope_base, original_seq_len,
          factor, beta_fast, beta_slow, 1.0f, &out_a, &out_b);
      value = (local & 1) == 0 ? out_a : out_b;
    }
    compressed[(compressed_base + block) * head_dim + col] =
        dsv4_attn_f32_to_bf16_bits(value);
  }
}

// Finalize the prefill compressor: write the cross-chunk `prev_overlap` (the LAST
// produced block's first-half tokens) and the trailing `pending` partial block.
// Single block (both writes are O(ratio·head_dim), tiny). The __syncthreads
// separates the prev_overlap reads of the OLD pending from the pending writes.
// The FP32 path (T = float) also mirrors the prev_overlap AND pending writes to
// bf16 for the decode-lane readers; bf16 callers pass nullptr mirrors.
template <typename T>
__global__ void dsv4_compressor_finalize_kernel(
    const T *__restrict__ kv_raw,
    const T *__restrict__ score_raw,
    const T *__restrict__ ape,
    T *__restrict__ pending_kv,
    T *__restrict__ pending_score,
    T *__restrict__ prev_overlap_kv,
    T *__restrict__ prev_overlap_score,
    uint16_t *__restrict__ prev_overlap_kv_mirror,
    uint16_t *__restrict__ prev_overlap_score_mirror,
    uint16_t *__restrict__ pending_kv_mirror,
    uint16_t *__restrict__ pending_score_mirror,
    int num_tokens,
    int start_pos,
    int pending_len,
    int completed,
    int head_dim,
    int ratio,
    int width,
    int overlap,
    // See `dsv4_compressor_update_body`: `0` collapses the write below to the
    // single-register form.
    int overlap_page_stride) {
  int block_start0 = start_pos - pending_len;
  int total = pending_len + num_tokens;

  // completed == 0 leaves prev_overlap (FP32 and bf16 mirror alike) untouched:
  // the two carries are coherent at every probe entry (probe-exit mirror or the
  // host-side stale reseed), and a pending-only call changes neither — so the
  // deleted serial probe's unconditional re-mirror was redundant, not load-bearing.
  if (overlap && completed > 0) {
    int last_block_start = block_start0 + (completed - 1) * ratio;
    // Absolute block index of the LAST block this call completed — always
    // `>= 0` (guarded by `completed > 0`). `block_start0 / ratio` reconstructs
    // this call's `compressed_base` (== `start_pos / ratio`, since
    // `block_start0 == start_pos - start_pos % ratio` by construction) without
    // needing it as a separate parameter here.
    int last_block = block_start0 / ratio + (completed - 1);
    for (int idx = threadIdx.x; idx < ratio * head_dim; idx += blockDim.x) {
      int pos = idx / head_dim;
      int col = idx - pos * head_dim;
      int abs_pos = last_block_start + pos;
      float kv = dsv4_compressor_raw_value(
          kv_raw, pending_kv, abs_pos, start_pos, last_block_start, width, col);
      float score = dsv4_compressor_score_value(
          score_raw, pending_score, ape, abs_pos, start_pos, last_block_start,
          ratio, width, col);
      prev_overlap_kv[last_block * overlap_page_stride + pos * head_dim + col] = dsv4_compressor_store<T>(kv);
      prev_overlap_score[last_block * overlap_page_stride + pos * head_dim + col] = dsv4_compressor_store<T>(score);
      if (prev_overlap_kv_mirror != nullptr) {
        prev_overlap_kv_mirror[pos * head_dim + col] = dsv4_attn_f32_to_bf16_bits(kv);
        prev_overlap_score_mirror[pos * head_dim + col] = dsv4_attn_f32_to_bf16_bits(score);
      }
    }
  }
  __syncthreads();

  if (completed == 0) {
    for (int idx = threadIdx.x; idx < num_tokens * width; idx += blockDim.x) {
      int pos = idx / width;
      int col = idx - pos * width;
      int abs_pos = start_pos + pos;
      int dst = (pending_len + pos) * width + col;
      float kv = dsv4_compressor_load(kv_raw[idx]);
      float score = dsv4_compressor_load(score_raw[idx]) +
                    dsv4_compressor_load(ape[(abs_pos % ratio) * width + col]);
      pending_kv[dst] = dsv4_compressor_store<T>(kv);
      pending_score[dst] = dsv4_compressor_store<T>(score);
      if (pending_kv_mirror != nullptr) {
        pending_kv_mirror[dst] = dsv4_attn_f32_to_bf16_bits(kv);
        pending_score_mirror[dst] = dsv4_attn_f32_to_bf16_bits(score);
      }
    }
  } else {
    int new_pending = total - completed * ratio;
    int tail_start = start_pos + num_tokens - new_pending;
    for (int idx = threadIdx.x; idx < new_pending * width; idx += blockDim.x) {
      int pos = idx / width;
      int col = idx - pos * width;
      int abs_pos = tail_start + pos;
      float kv = dsv4_compressor_raw_value(
          kv_raw, pending_kv, abs_pos, start_pos, block_start0, width, col);
      float score = dsv4_compressor_score_value(
          score_raw, pending_score, ape, abs_pos, start_pos, block_start0,
          ratio, width, col);
      pending_kv[idx] = dsv4_compressor_store<T>(kv);
      pending_score[idx] = dsv4_compressor_store<T>(score);
      if (pending_kv_mirror != nullptr) {
        pending_kv_mirror[idx] = dsv4_attn_f32_to_bf16_bits(kv);
        pending_score_mirror[idx] = dsv4_attn_f32_to_bf16_bits(score);
      }
    }
  }
}

// FP32-carry reseed: bf16 → f32 upcast of the four per-state carry buffers
// (pending kv/score = `pending_elems` each, prev_overlap kv/score =
// `prev_elems` each). Launched when the bf16 decode lane / prefix restore /
// reset advanced the bf16 carry since the last FP32 probe — the bf16 carry is
// then the authority and the probe must not read the stale FP32 copy.
__global__ void dsv4_compressor_fp32_carry_reseed_kernel(
    const uint16_t *__restrict__ pending_kv_bf16,
    const uint16_t *__restrict__ pending_score_bf16,
    const uint16_t *__restrict__ prev_kv_bf16,
    const uint16_t *__restrict__ prev_score_bf16,
    float *__restrict__ pending_kv,
    float *__restrict__ pending_score,
    float *__restrict__ prev_kv,
    float *__restrict__ prev_score,
    int pending_elems,
    int prev_elems) {
  int total = pending_elems + prev_elems;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += gridDim.x * blockDim.x) {
    if (i < pending_elems) {
      pending_kv[i] = dsv4_attn_bf16_to_f32(pending_kv_bf16[i]);
      pending_score[i] = dsv4_attn_bf16_to_f32(pending_score_bf16[i]);
    } else {
      int j = i - pending_elems;
      prev_kv[j] = dsv4_attn_bf16_to_f32(prev_kv_bf16[j]);
      prev_score[j] = dsv4_attn_bf16_to_f32(prev_score_bf16[j]);
    }
  }
}

extern "C" CUresult dsv4_compressor_fp32_carry_reseed_cuda(
    const uint16_t *pending_kv_bf16,
    const uint16_t *pending_score_bf16,
    const uint16_t *prev_kv_bf16,
    const uint16_t *prev_score_bf16,
    float *pending_kv,
    float *pending_score,
    float *prev_kv,
    float *prev_score,
    int pending_elems,
    int prev_elems,
    CUstream stream) {
  if (pending_elems < 0 || prev_elems < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = pending_elems + prev_elems;
  if (total == 0) return CUDA_SUCCESS;
  int blocks = (total + DSV4_ATTN_BLOCK - 1) / DSV4_ATTN_BLOCK;
  dsv4_compressor_fp32_carry_reseed_kernel<<<blocks, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      pending_kv_bf16, pending_score_bf16, prev_kv_bf16, prev_score_bf16,
      pending_kv, pending_score, prev_kv, prev_score, pending_elems, prev_elems);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_compressor_update_cuda(
    const uint16_t *kv_raw,
    const uint16_t *score_raw,
    const uint16_t *ape,
    const uint16_t *norm,
    uint16_t *pending_kv,
    uint16_t *pending_score,
    uint16_t *prev_overlap_kv,
    uint16_t *prev_overlap_score,
    uint16_t *compressed,
    int num_tokens,
    int start_pos,
    int pending_len,
    int compressed_base,
    int head_dim,
    int ratio,
    int width,
    int overlap,
    int has_prev_overlap,
    // `0` = per-slot single-register prev_overlap buffer; `ratio*head_dim` =
    // shared, page-addressable pool. See `dsv4_compressor_update_body`.
    int overlap_page_stride,
    float eps,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    CUstream stream) {
  if (num_tokens < 0 || start_pos < 0 || pending_len < 0 || compressed_base < 0 ||
      head_dim <= 0 || head_dim > DSV4_ATTN_MAX_HEAD_DIM || ratio <= 0 ||
      ratio > 256 || width < head_dim || rope_dim < 0 || rope_dim > head_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // Parallel prefill: grid = number of compressed blocks (fills the SMs), then a
  // tiny single-block finalize for prev_overlap + pending. (Decode keeps the
  // single-block dsv4_compressor_update_kernel via the start_pos_ptr launcher
  // below — there completed<=1, so a grid is pointless.)
  int completed = (pending_len + num_tokens) / ratio;
  if (completed > 0) {
    dsv4_compressor_block_kernel<<<completed, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
        kv_raw, score_raw, ape, norm, pending_kv, pending_score, prev_overlap_kv,
        prev_overlap_score, compressed, start_pos, pending_len, compressed_base,
        head_dim, ratio, width, overlap, has_prev_overlap, overlap_page_stride,
        eps, rope_dim, rope_base, original_seq_len, factor, beta_fast, beta_slow);
  }
  dsv4_compressor_finalize_kernel<<<1, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      kv_raw, score_raw, ape, pending_kv, pending_score, prev_overlap_kv,
      prev_overlap_score, nullptr, nullptr, nullptr, nullptr, num_tokens,
      start_pos, pending_len, completed, head_dim, ratio, width, overlap,
      overlap_page_stride);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_compressor_fp32_prefill_probe_cuda(
    const float *kv_raw,
    const float *score_raw,
    const float *ape,
    const uint16_t *norm,
    float *pending_kv,
    float *pending_score,
    float *prev_overlap_kv,
    float *prev_overlap_score,
    uint16_t *prev_overlap_kv_bf16,
    uint16_t *prev_overlap_score_bf16,
    uint16_t *pending_kv_bf16,
    uint16_t *pending_score_bf16,
    uint16_t *compressed,
    int num_tokens,
    int start_pos,
    int pending_len,
    int compressed_base,
    int head_dim,
    int ratio,
    int width,
    int overlap,
    int has_prev_overlap,
    int overlap_page_stride,
    float eps,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    CUstream stream) {
  if (num_tokens <= 0 || start_pos < 0 || pending_len < 0 ||
      compressed_base < 0 || head_dim <= 0 ||
      head_dim > DSV4_ATTN_MAX_HEAD_DIM || ratio <= 0 || ratio > 256 ||
      width < head_dim || rope_dim < 0 || rope_dim > head_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // Same grid decomposition as the bf16 dsv4_compressor_update_cuda above:
  // one block per compressed-output block, then a tiny single-block finalize
  // (pending + prev_overlap + the bf16 mirrors the decode lane reads).
  int completed = (pending_len + num_tokens) / ratio;
  if (completed > 0) {
    dsv4_compressor_block_kernel<<<completed, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
        kv_raw, score_raw, ape, norm, pending_kv, pending_score, prev_overlap_kv,
        prev_overlap_score, compressed, start_pos, pending_len, compressed_base,
        head_dim, ratio, width, overlap, has_prev_overlap, overlap_page_stride,
        eps, rope_dim, rope_base, original_seq_len, factor, beta_fast, beta_slow);
  }
  dsv4_compressor_finalize_kernel<<<1, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      kv_raw, score_raw, ape, pending_kv, pending_score, prev_overlap_kv,
      prev_overlap_score, prev_overlap_kv_bf16, prev_overlap_score_bf16,
      pending_kv_bf16, pending_score_bf16, num_tokens, start_pos, pending_len,
      completed, head_dim, ratio, width, overlap, overlap_page_stride);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_compressor_update_start_pos_ptr_cuda(
    const uint16_t *kv_raw,
    const uint16_t *score_raw,
    const uint16_t *ape,
    const uint16_t *norm,
    uint16_t *pending_kv,
    uint16_t *pending_score,
    uint16_t *prev_overlap_kv,
    uint16_t *prev_overlap_score,
    uint16_t *compressed,
    int num_tokens,
    const int *start_pos,
    int head_dim,
    int ratio,
    int width,
    int overlap,
    int overlap_page_stride,
    float eps,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    CUstream stream) {
  if (num_tokens < 0 || start_pos == nullptr || head_dim <= 0 ||
      head_dim > DSV4_ATTN_MAX_HEAD_DIM || ratio <= 0 || ratio > 256 ||
      width < head_dim || rope_dim < 0 || rope_dim > head_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dsv4_compressor_update_kernel<<<1, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      kv_raw, score_raw, ape, norm, pending_kv, pending_score, prev_overlap_kv,
      prev_overlap_score, compressed, num_tokens, 0, start_pos, 0, 0, head_dim,
      ratio, width, overlap, 0, overlap_page_stride, eps, rope_dim, rope_base,
      original_seq_len, factor, beta_fast, beta_slow);
  return (CUresult)cudaGetLastError();
}

// Batched decode compressor update: ONE `<<<n, BLOCK>>>` launch replacing n
// single-row `dsv4_compressor_update_start_pos_ptr_cuda` calls. Each row r reads
// its state buffer pointers from the `*_arr` device pointer arrays (host-gathered
// from the per-slot Dsv4CompressorState) and its decode position from
// `start_pos_arr[r]`. `kv_raw`/`score_raw` are the batched m=N prepass outputs
// `[width, n]`. Math byte-identical to n single-row launches (each block runs the
// shared `dsv4_compressor_update_body` with the same resolved args).
extern "C" CUresult dsv4_compressor_update_batched_start_pos_ptr_cuda(
    const uint16_t *kv_raw,
    const uint16_t *score_raw,
    const uint16_t *ape,
    const uint16_t *norm,
    uint16_t *const *pending_kv_arr,
    uint16_t *const *pending_score_arr,
    uint16_t *const *prev_overlap_kv_arr,
    uint16_t *const *prev_overlap_score_arr,
    uint16_t *const *compressed_arr,
    int n,
    int num_tokens,
    const int *start_pos_arr,
    int head_dim,
    int ratio,
    int width,
    int overlap,
    // Uniform across all n rows — one launch is always one (layer,
    // compress_ratio) class, so every row's prev_overlap buffer is the same
    // kind (pool or single-register).
    int overlap_page_stride,
    float eps,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    CUstream stream) {
  if (n < 0 || num_tokens < 0 || start_pos_arr == nullptr || head_dim <= 0 ||
      head_dim > DSV4_ATTN_MAX_HEAD_DIM || ratio <= 0 || ratio > 256 ||
      width < head_dim || rope_dim < 0 || rope_dim > head_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) return CUDA_SUCCESS;
  if (pending_kv_arr == nullptr || pending_score_arr == nullptr ||
      prev_overlap_kv_arr == nullptr || prev_overlap_score_arr == nullptr ||
      compressed_arr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dsv4_compressor_update_batched_kernel<<<n, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      kv_raw, score_raw, ape, norm, pending_kv_arr, pending_score_arr,
      prev_overlap_kv_arr, prev_overlap_score_arr, compressed_arr, n, num_tokens,
      start_pos_arr, head_dim, ratio, width, overlap, overlap_page_stride, eps,
      rope_dim, rope_base, original_seq_len, factor, beta_fast, beta_slow);
  return (CUresult)cudaGetLastError();
}

#include "common.cuh"

// Causal Depthwise Conv1d for Gated Delta Net linear attention
//
// Prefill: Parallel causal conv1d over the entire sequence.

#define CONV1D_BLOCK 256

// Prefill kernel: parallel causal conv1d over sequence
// x_seq: [num_channels, seq_len] bf16 (column-major: token i at offset i * num_channels)
// Actually stored as [seq_len * num_channels] row-major (token i starts at i * num_channels)
// out_seq: [seq_len * num_channels] bf16
// conv_state: [num_channels, K-1] bf16 (updated with last K-1 values)
// Non-null `x_ptrs` selects the varlen form: `blockIdx.y` is one slot, reading
// its own buffer and ring, writing `row_len[s]` rows at `out_seq + s*seq_len`.
__global__ void conv1d_prefill_kernel(
    const __nv_bfloat16* __restrict__ x_seq,
    const __nv_bfloat16* const* __restrict__ x_ptrs,
    const __nv_bfloat16* __restrict__ conv_weight,
    __nv_bfloat16* __restrict__ conv_state,
    __nv_bfloat16* const* __restrict__ state_ptrs,
    const int* __restrict__ row_len,
    __nv_bfloat16* __restrict__ out_seq,
    int num_channels,
    int seq_len,
    int kernel_size
) {
    const int s = blockIdx.y;
    int len = seq_len;
    if (x_ptrs) {
        x_seq = x_ptrs[s];
        conv_state = state_ptrs[s];
        len = row_len[s];
        out_seq += static_cast<size_t>(s) * seq_len * num_channels;
    }

    // Each thread handles one (channel, position) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_channels * len;
    if (idx >= total) return;

    int c = idx % num_channels;
    int t = idx / num_channels;

    int state_width = kernel_size - 1;

    // Compute causal conv1d at position t for channel c
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int src_t = t - (kernel_size - 1) + k;  // position in sequence
        float val;
        if (src_t < 0) {
            // Read from conv_state (pre-existing state)
            int state_idx = state_width + src_t;  // maps [-state_width, -1] → [0, state_width-1]
            if (state_idx >= 0) {
                val = __bfloat162float(conv_state[c * state_width + state_idx]);
            } else {
                val = 0.0f;  // beyond state buffer
            }
        } else {
            val = __bfloat162float(x_seq[src_t * num_channels + c]);
        }
        sum += val * __bfloat162float(conv_weight[c * kernel_size + k]);
    }

    // Match HF/PyTorch: conv1d writes bf16, then SiLU consumes bf16 input.
    float sum_bf16 = __bfloat162float(__float2bfloat16(sum));
    float silu_out = sum_bf16 / (1.0f + __expf(-sum_bf16));
    out_seq[t * num_channels + c] = __float2bfloat16(silu_out);

    // NOTE: the conv ring update lives in conv1d_state_update_kernel, launched
    // AFTER this kernel on the same stream. Updating it here raced: threads at
    // t < kernel_size-1 READ conv_state (the previous chunk's tail) while the
    // t == seq_len-1 thread WROTE the new ring, with no grid-level ordering —
    // under chunked prefill early tokens could convolve against the CURRENT
    // chunk's tail instead of the previous chunk's.
}

// seq_len == 1 gives one thread per (slot, channel), so it reads the whole
// ring before rewriting it and the race above cannot occur.
__global__ void conv1d_decode_kernel(
    const __nv_bfloat16* __restrict__ x_seq,
    const __nv_bfloat16* const* __restrict__ x_ptrs,
    const __nv_bfloat16* __restrict__ conv_weight,
    __nv_bfloat16* __restrict__ conv_state,
    __nv_bfloat16* const* __restrict__ state_ptrs,
    __nv_bfloat16* __restrict__ out_seq,
    int num_channels,
    int kernel_size
) {
    const int s = blockIdx.y;
    if (x_ptrs) {
        x_seq = x_ptrs[s];
        conv_state = state_ptrs[s];
        out_seq += static_cast<size_t>(s) * num_channels;
    }

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_channels) return;

    const int state_width = kernel_size - 1;
    float ring[4];
    for (int i = 0; i < state_width; i++) {
        ring[i] = __bfloat162float(conv_state[c * state_width + i]);
    }
    const float x = __bfloat162float(x_seq[c]);

    float sum = 0.0f;
    for (int k = 0; k < state_width; k++) {
        sum += ring[k] * __bfloat162float(conv_weight[c * kernel_size + k]);
    }
    sum += x * __bfloat162float(conv_weight[c * kernel_size + state_width]);

    const float sum_bf16 = __bfloat162float(__float2bfloat16(sum));
    out_seq[c] = __float2bfloat16(sum_bf16 / (1.0f + __expf(-sum_bf16)));

    for (int i = 0; i + 1 < state_width; i++) {
        conv_state[c * state_width + i] = __float2bfloat16(ring[i + 1]);
    }
    if (state_width > 0) {
        conv_state[c * state_width + state_width - 1] = __float2bfloat16(x);
    }
}

// State-update kernel: rebuild the conv ring from the chunk tail. One thread
// per channel; reproduces the exact ring semantics of the writer branch that
// previously lived in conv1d_prefill_kernel (including seq_len < kernel-1,
// where the new ring = shifted old state ++ all of x_seq — old state is staged
// through registers before the in-place overwrite).
__global__ void conv1d_state_update_kernel(
    const __nv_bfloat16* __restrict__ x_seq,
    const __nv_bfloat16* const* __restrict__ x_ptrs,
    __nv_bfloat16* __restrict__ conv_state,
    __nv_bfloat16* const* __restrict__ state_ptrs,
    const int* __restrict__ row_len,
    int num_channels,
    int seq_len,
    int kernel_size
) {
    const int s = blockIdx.y;
    if (x_ptrs) {
        x_seq = x_ptrs[s];
        conv_state = state_ptrs[s];
        seq_len = row_len[s];
    }

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_channels) return;

    int state_width = kernel_size - 1;

    float old_state[4];
    for (int i = 0; i < state_width; i++) {
        old_state[i] = __bfloat162float(conv_state[c * state_width + i]);
    }
    for (int i = 0; i < state_width; i++) {
        int src_t = seq_len - state_width + i;
        if (src_t >= 0) {
            conv_state[c * state_width + i] = x_seq[src_t * num_channels + c];
        } else {
            int state_idx = state_width + src_t;  // maps [-state_width, -1] → [0, state_width-1]
            conv_state[c * state_width + i] =
                state_idx >= 0 ? __float2bfloat16(old_state[state_idx]) : __float2bfloat16(0.0f);
        }
    }
}

extern "C" {

cudaError_t conv1d_prefill_cuda(
    const __nv_bfloat16* x_seq,
    const __nv_bfloat16* conv_weight,
    __nv_bfloat16* conv_state,
    __nv_bfloat16* out_seq,
    int num_channels,
    int seq_len,
    int kernel_size,
    cudaStream_t stream
) {
    if (seq_len == 1) {
        int blocks = (num_channels + CONV1D_BLOCK - 1) / CONV1D_BLOCK;
        conv1d_decode_kernel<<<blocks, CONV1D_BLOCK, 0, stream>>>(
            x_seq, nullptr, conv_weight, conv_state, nullptr,
            out_seq, num_channels, kernel_size
        );
        return cudaGetLastError();
    }
    int total = num_channels * seq_len;
    int blocks = (total + CONV1D_BLOCK - 1) / CONV1D_BLOCK;
    conv1d_prefill_kernel<<<blocks, CONV1D_BLOCK, 0, stream>>>(
        x_seq, nullptr, conv_weight, conv_state, nullptr, nullptr,
        out_seq, num_channels, seq_len, kernel_size
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    // Ring update only after every (channel, position) thread has consumed the
    // previous state (same-stream ordering). Launched here, inside the
    // launcher, so every consumer keeps the original FFI contract
    // "state is advanced when the call's stream work completes".
    int state_blocks = (num_channels + CONV1D_BLOCK - 1) / CONV1D_BLOCK;
    conv1d_state_update_kernel<<<state_blocks, CONV1D_BLOCK, 0, stream>>>(
        x_seq, nullptr, conv_state, nullptr, nullptr, num_channels, seq_len, kernel_size
    );
    return cudaGetLastError();
}

// Varlen twin: slot `s` reads `x_ptrs[s]` for `row_len[s]` rows.
cudaError_t conv1d_prefill_varlen_cuda(
    const __nv_bfloat16* const* x_ptrs,
    const __nv_bfloat16* conv_weight,
    __nv_bfloat16* const* state_ptrs,
    const int* row_len,
    __nv_bfloat16* out_seq,
    int num_channels,
    int max_len,
    int kernel_size,
    int batch,
    cudaStream_t stream
) {
    if (batch <= 0 || max_len <= 0) {
        return cudaErrorInvalidValue;
    }
    // max_len == 1 forces every row_len to 1: the decode form covers all slots.
    if (max_len == 1) {
        dim3 dgrid((num_channels + CONV1D_BLOCK - 1) / CONV1D_BLOCK, batch);
        conv1d_decode_kernel<<<dgrid, CONV1D_BLOCK, 0, stream>>>(
            nullptr, x_ptrs, conv_weight, nullptr, state_ptrs,
            out_seq, num_channels, kernel_size
        );
        return cudaGetLastError();
    }
    int total = num_channels * max_len;
    dim3 grid((total + CONV1D_BLOCK - 1) / CONV1D_BLOCK, batch);
    conv1d_prefill_kernel<<<grid, CONV1D_BLOCK, 0, stream>>>(
        nullptr, x_ptrs, conv_weight, nullptr, state_ptrs, row_len,
        out_seq, num_channels, max_len, kernel_size
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    dim3 sgrid((num_channels + CONV1D_BLOCK - 1) / CONV1D_BLOCK, batch);
    conv1d_state_update_kernel<<<sgrid, CONV1D_BLOCK, 0, stream>>>(
        nullptr, x_ptrs, nullptr, state_ptrs, row_len, num_channels, max_len, kernel_size
    );
    return cudaGetLastError();
}

} // extern "C"

// `count` equal-sized device-to-device copies in one launch. The DSpark spec
// snapshot/restore is 48 layers x B slots of them per tick, and each
// `cuMemcpyDtoDAsync` costs ~11 µs of host driver time against ~2 µs of
// bandwidth.
// `words_each` non-null gives each buffer its own 16B-word count; null means
// they all copy `words`.
__global__ void batched_copy_uniform_kernel(
    void* const* __restrict__ dst_ptrs,
    const void* const* __restrict__ src_ptrs,
    const int* __restrict__ words_each,
    size_t words
) {
    const uint4* src = static_cast<const uint4*>(src_ptrs[blockIdx.y]);
    uint4* dst = static_cast<uint4*>(dst_ptrs[blockIdx.y]);
    if (words_each) words = words_each[blockIdx.y];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" {

// `bytes` (or every `words_each` entry x16) must be a multiple of 16 and every
// buffer 16B-aligned (cudaMalloc is). `max_words` sizes the grid.
cudaError_t batched_copy_uniform_cuda(
    void* const* dst_ptrs,
    const void* const* src_ptrs,
    const int* words_each,
    size_t bytes,
    size_t max_words,
    int count,
    cudaStream_t stream
) {
    if (count <= 0 || (bytes & 15)) {
        return cudaErrorInvalidValue;
    }
    size_t words = bytes >> 4;
    size_t span = words_each ? max_words : words;
    if (span == 0) {
        return cudaErrorInvalidValue;
    }
    size_t need = (span + CONV1D_BLOCK - 1) / CONV1D_BLOCK;
    dim3 grid(static_cast<unsigned>(need < 1024 ? need : 1024), count);
    batched_copy_uniform_kernel<<<grid, CONV1D_BLOCK, 0, stream>>>(
        dst_ptrs, src_ptrs, words_each, words
    );
    return cudaGetLastError();
}

} // extern "C"

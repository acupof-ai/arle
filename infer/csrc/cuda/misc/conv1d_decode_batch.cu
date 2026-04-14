#include "common.cuh"

// ============================================================================
// Batched Conv1d Decode — process B requests in one kernel launch
//
// Specialized for decode (seq_len=1) with kernel_size=4 (Qwen3.5 default).
// Each thread handles one (channel, request) pair.
//
// Optimizations vs naive implementation:
//   1. Conv weights cached in registers (K=4, only 4 floats per thread)
//   2. State values loaded into registers before computation — single pass
//   3. State update writes directly (no shift loop): overwrite s0←s1, s1←s2, s2←x
//      using pre-loaded register values to avoid RAW hazards
//   4. Coalesced reads from x_batch [B, num_channels] via blockIdx.y
//
// Grid:  (ceil(num_channels / BLOCK), B)
// Block: 256 threads
// ============================================================================

#define CONV1D_BATCH_BLOCK 256

__global__ void conv1d_decode_batch_kernel(
    const __nv_bfloat16* __restrict__ x_batch,        // [B, num_channels]
    const __nv_bfloat16* __restrict__ conv_weight,     // [num_channels, kernel_size]
    __nv_bfloat16** __restrict__ conv_state_ptrs,      // [B] → [num_channels, K-1]
    __nv_bfloat16* __restrict__ out_batch,             // [B, num_channels]
    int num_channels,
    int kernel_size
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (c >= num_channels) return;

    // Note: register unrolling below assumes kernel_size <= 4.
    // For kernel_size > 4, a generic loop would be needed.
    int sw = kernel_size - 1;  // state_width (3 for K=4)
    __nv_bfloat16* my_state = conv_state_ptrs[b] + c * sw;
    const __nv_bfloat16* my_weight = conv_weight + c * kernel_size;

    // Load conv weights into registers (K=4 → 4 floats, negligible register pressure)
    float w[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        w[k] = (k < kernel_size) ? __bfloat162float(my_weight[k]) : 0.0f;
    }

    // Load state into registers (avoid repeated global reads during shift)
    float s0 = __bfloat162float(my_state[0]);
    float s1 = (sw > 1) ? __bfloat162float(my_state[1]) : 0.0f;
    float s2 = (sw > 2) ? __bfloat162float(my_state[2]) : 0.0f;
    float x_val = __bfloat162float(x_batch[b * num_channels + c]);

    // Causal conv1d dot product: state[0..sw-1] * w[0..sw-1] + x * w[sw]
    float sum = s0 * w[0] + s1 * w[1] + s2 * w[2] + x_val * w[3];

    // SiLU activation (bf16 truncation for numerical parity with prefill kernel)
    float sum_bf16 = __bfloat162float(__float2bfloat16(sum));
    float silu_out = sum_bf16 / (1.0f + expf(-sum_bf16));
    out_batch[b * num_channels + c] = __float2bfloat16(silu_out);

    // Update state: shift left by 1, insert x at tail
    // Uses register values — no global read dependency, no shift loop
    my_state[0] = __float2bfloat16(s1);
    if (sw > 2) my_state[1] = __float2bfloat16(s2);
    my_state[sw - 1] = __float2bfloat16(x_val);
}

extern "C" {

cudaError_t conv1d_decode_batch_cuda(
    const __nv_bfloat16* x_batch,
    const __nv_bfloat16* conv_weight,
    __nv_bfloat16** conv_state_ptrs,
    __nv_bfloat16* out_batch,
    int num_channels,
    int kernel_size,
    int batch_size,
    cudaStream_t stream
) {
    if (kernel_size <= 0 || kernel_size > 4) {
        return cudaErrorInvalidValue;
    }

    dim3 grid((num_channels + CONV1D_BATCH_BLOCK - 1) / CONV1D_BATCH_BLOCK, batch_size);
    conv1d_decode_batch_kernel<<<grid, CONV1D_BATCH_BLOCK, 0, stream>>>(
        x_batch, conv_weight, conv_state_ptrs, out_batch,
        num_channels, kernel_size
    );
    return cudaGetLastError();
}

} // extern "C"

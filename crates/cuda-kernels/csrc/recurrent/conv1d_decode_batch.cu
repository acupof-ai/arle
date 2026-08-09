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

template <int KERNEL_SIZE>
__global__ void conv1d_decode_batch_kernel(
    const __nv_bfloat16* __restrict__ x_batch,        // [B, num_channels]
    const __nv_bfloat16* __restrict__ conv_weight,     // [num_channels, kernel_size]
    __nv_bfloat16** __restrict__ conv_state_ptrs,      // [B] → [num_channels, K-1]
    __nv_bfloat16* __restrict__ out_batch,             // [B, num_channels]
    int num_channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (c >= num_channels) return;

    constexpr int sw = KERNEL_SIZE - 1;
    __nv_bfloat16* my_state = conv_state_ptrs[b] + c * sw;
    const __nv_bfloat16* my_weight = conv_weight + c * KERNEL_SIZE;

    // Load weights and state once into registers. Decode always uses small
    // kernels (2..4), so template specialization removes the hot-path branches.
    float w0 = __bfloat162float(my_weight[0]);
    float w1 = __bfloat162float(my_weight[1]);
    float w2 = 0.0f;
    float w3 = 0.0f;
    if constexpr (KERNEL_SIZE > 2) {
        w2 = __bfloat162float(my_weight[2]);
    }
    if constexpr (KERNEL_SIZE > 3) {
        w3 = __bfloat162float(my_weight[3]);
    }

    float s0 = __bfloat162float(my_state[0]);
    float s1 = 0.0f;
    float s2 = 0.0f;
    if constexpr (KERNEL_SIZE > 2) {
        s1 = __bfloat162float(my_state[1]);
    }
    if constexpr (KERNEL_SIZE > 3) {
        s2 = __bfloat162float(my_state[2]);
    }
    float x_val = __bfloat162float(x_batch[b * num_channels + c]);

    float sum;
    if constexpr (KERNEL_SIZE == 2) {
        sum = s0 * w0 + x_val * w1;
    } else if constexpr (KERNEL_SIZE == 3) {
        sum = s0 * w0 + s1 * w1 + x_val * w2;
    } else {
        sum = s0 * w0 + s1 * w1 + s2 * w2 + x_val * w3;
    }

    // SiLU activation (bf16 truncation for numerical parity with prefill kernel)
    float sum_bf16 = __bfloat162float(__float2bfloat16(sum));
    float silu_out = sum_bf16 / (1.0f + expf(-sum_bf16));
    out_batch[b * num_channels + c] = __float2bfloat16(silu_out);

    // Update state: shift left by 1, insert x at tail.
    if constexpr (KERNEL_SIZE == 2) {
        my_state[0] = __float2bfloat16(x_val);
    } else if constexpr (KERNEL_SIZE == 3) {
        my_state[0] = __float2bfloat16(s1);
        my_state[1] = __float2bfloat16(x_val);
    } else {
        my_state[0] = __float2bfloat16(s1);
        my_state[1] = __float2bfloat16(s2);
        my_state[2] = __float2bfloat16(x_val);
    }
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
    switch (kernel_size) {
        case 2:
            conv1d_decode_batch_kernel<2><<<grid, CONV1D_BATCH_BLOCK, 0, stream>>>(
                x_batch, conv_weight, conv_state_ptrs, out_batch, num_channels
            );
            break;
        case 3:
            conv1d_decode_batch_kernel<3><<<grid, CONV1D_BATCH_BLOCK, 0, stream>>>(
                x_batch, conv_weight, conv_state_ptrs, out_batch, num_channels
            );
            break;
        case 4:
            conv1d_decode_batch_kernel<4><<<grid, CONV1D_BATCH_BLOCK, 0, stream>>>(
                x_batch, conv_weight, conv_state_ptrs, out_batch, num_channels
            );
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

} // extern "C"

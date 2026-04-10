// Qwen3.5 (Qwen3-Next) gated attention output.
//
// q_proj output layout is per-head concat:
//   [q_h0 | gate_h0 | q_h1 | gate_h1 | ... | q_h{H-1} | gate_h{H-1}]
// where each head owns 2 * head_dim contiguous elements. HF/vLLM/llama.cpp
// all agree on this layout and on the final formula:
//   attn_out = attention(q, k, v) * sigmoid(gate)
//
// split_q_gate_batch: de-interleaves q_full [B, 2*H*D] into q [B, H*D]
// and gate [B, H*D]. H = num_heads, D = head_dim.
//
// sigmoid_gate_mul_batch: attn_out[i] *= sigmoid(gate[i]) in-place.

#include "common.cuh"

__global__ void split_q_gate_batch_kernel(
    const __nv_bfloat16* __restrict__ q_full,  // [B, 2*H*D]
    __nv_bfloat16* __restrict__ q,              // [B, H*D]
    __nv_bfloat16* __restrict__ gate,           // [B, H*D]
    int head_dim,
    int num_heads
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0..H*D
    int row = blockIdx.y;                             // 0..B
    int qh_dim = num_heads * head_dim;                 // H*D
    if (col >= qh_dim) return;

    int head_idx = col / head_dim;
    int lane     = col - head_idx * head_dim;
    // Per-head stride in the source is 2*head_dim.
    int src_base = row * (2 * qh_dim) + head_idx * (2 * head_dim);
    q   [row * qh_dim + col] = q_full[src_base + lane];
    gate[row * qh_dim + col] = q_full[src_base + head_dim + lane];
}

__global__ void sigmoid_gate_mul_batch_kernel(
    __nv_bfloat16* __restrict__ attn_out,       // [B, H*D] in-place
    const __nv_bfloat16* __restrict__ gate,     // [B, H*D]
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = __bfloat162float(attn_out[idx]);
    float g = __bfloat162float(gate[idx]);
    float s = 1.0f / (1.0f + expf(-g));
    attn_out[idx] = __float2bfloat16(a * s);
}

extern "C" {

cudaError_t split_q_gate_batch_cuda(
    const __nv_bfloat16* q_full,
    __nv_bfloat16* q,
    __nv_bfloat16* gate,
    int batch_size,
    int num_heads,
    int head_dim,
    cudaStream_t stream
) {
    int qh_dim = num_heads * head_dim;
    int threads = 256;
    dim3 grid((qh_dim + threads - 1) / threads, batch_size);
    split_q_gate_batch_kernel<<<grid, threads, 0, stream>>>(
        q_full, q, gate, head_dim, num_heads
    );
    return cudaGetLastError();
}

cudaError_t sigmoid_gate_mul_batch_cuda(
    __nv_bfloat16* attn_out,
    const __nv_bfloat16* gate,
    int batch_size,
    int num_heads,
    int head_dim,
    cudaStream_t stream
) {
    int n = batch_size * num_heads * head_dim;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoid_gate_mul_batch_kernel<<<blocks, threads, 0, stream>>>(
        attn_out, gate, n
    );
    return cudaGetLastError();
}

} // extern "C"

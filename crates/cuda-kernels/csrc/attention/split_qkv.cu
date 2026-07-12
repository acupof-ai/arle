// Split merged QKV buffer into separate Q, K, V buffers.
// Input:  qkv [B, q_dim + 2*kv_dim] (merged GEMM output)
// Output: q [B, q_dim], k [B, kv_dim], v [B, kv_dim]
//
// Grid: (q_dim + 2*kv_dim, B), Block: 1

#include "common.cuh"

__global__ void split_qkv_kernel(
    const __nv_bfloat16* __restrict__ qkv,  // [B, qkv_dim]
    __nv_bfloat16* __restrict__ q,           // [B, q_dim]
    __nv_bfloat16* __restrict__ k,           // [B, kv_dim]
    __nv_bfloat16* __restrict__ v,           // [B, kv_dim]
    int q_dim, int kv_dim, int qkv_dim
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    if (col >= qkv_dim) return;

    __nv_bfloat16 val = qkv[row * qkv_dim + col];

    if (col < q_dim) {
        q[row * q_dim + col] = val;
    } else if (col < q_dim + kv_dim) {
        k[row * kv_dim + (col - q_dim)] = val;
    } else {
        v[row * kv_dim + (col - q_dim - kv_dim)] = val;
    }
}

// Fused silu_mul from merged gate+up buffer.
// Input:  gate_up [B, 2*inter_dim] where first half = gate, second half = up
// Output: out [B, inter_dim] = silu(gate) * up
__global__ void silu_mul_fused_kernel(
    const __nv_bfloat16* __restrict__ gate_up,  // [B, 2*inter_dim]
    __nv_bfloat16* __restrict__ out,             // [B, inter_dim]
    int inter_dim
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    if (col >= inter_dim) return;

    int gu_stride = 2 * inter_dim;
    float gate_val = __bfloat162float(gate_up[row * gu_stride + col]);
    float up_val = __bfloat162float(gate_up[row * gu_stride + inter_dim + col]);

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float silu = gate_val / (1.0f + expf(-gate_val));
    out[row * inter_dim + col] = __float2bfloat16(silu * up_val);
}

extern "C" {

cudaError_t split_qkv_cuda(
    const __nv_bfloat16* qkv,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    int batch_size, int q_dim, int kv_dim,
    cudaStream_t stream
) {
    int qkv_dim = q_dim + 2 * kv_dim;
    int threads = 256;
    dim3 grid((qkv_dim + threads - 1) / threads, batch_size);
    split_qkv_kernel<<<grid, threads, 0, stream>>>(
        qkv, q, k, v, q_dim, kv_dim, qkv_dim
    );
    return cudaGetLastError();
}

cudaError_t silu_mul_fused_cuda(
    const __nv_bfloat16* gate_up,
    __nv_bfloat16* out,
    int batch_size, int inter_dim,
    cudaStream_t stream
) {
    int threads = 256;
    dim3 grid((inter_dim + threads - 1) / threads, batch_size);
    silu_mul_fused_kernel<<<grid, threads, 0, stream>>>(
        gate_up, out, inter_dim
    );
    return cudaGetLastError();
}

} // extern "C"

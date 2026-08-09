// Split merged QKV buffer into separate Q, K, V buffers.
// Input:  qkv [B, q_dim + 2*kv_dim] (merged GEMM output)
// Output: q [B, q_dim], k [B, kv_dim], v [B, kv_dim]
//
// Vectorized: 8 bf16 (uint4) per thread. q_dim/kv_dim are multiples of 8,
// so each uint4 load stays within one section — no cross-boundary mixing.

#include "common.cuh"

__global__ void split_qkv_kernel(
    const __nv_bfloat16* __restrict__ qkv,  // [B, qkv_dim]
    __nv_bfloat16* __restrict__ q,           // [B, q_dim]
    __nv_bfloat16* __restrict__ k,           // [B, kv_dim]
    __nv_bfloat16* __restrict__ v,           // [B, kv_dim]
    int q_dim, int kv_dim, int qkv_dim
) {
    int col8 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int row = blockIdx.y;
    if (col8 >= qkv_dim) return;

    const uint4* src = reinterpret_cast<const uint4*>(qkv + row * qkv_dim + col8);
    uint4 val = *src;

    if (col8 < q_dim) {
        uint4* dst = reinterpret_cast<uint4*>(q + row * q_dim + col8);
        *dst = val;
    } else if (col8 < q_dim + kv_dim) {
        uint4* dst = reinterpret_cast<uint4*>(k + row * kv_dim + (col8 - q_dim));
        *dst = val;
    } else {
        uint4* dst = reinterpret_cast<uint4*>(v + row * kv_dim + (col8 - q_dim - kv_dim));
        *dst = val;
    }
}

// Split a row-fused [B, first_dim + second_dim] buffer into two parts.
// Vectorized: 8 bf16 per thread. Falls back to scalar for the tail when
// first_dim is not a multiple of 8.
__global__ void split2_kernel(
    const __nv_bfloat16* __restrict__ fused,  // [B, first_dim + second_dim]
    __nv_bfloat16* __restrict__ first,        // [B, first_dim]
    __nv_bfloat16* __restrict__ second,       // [B, second_dim]
    int first_dim, int second_dim
) {
    int col8 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int row = blockIdx.y;
    int fused_dim = first_dim + second_dim;
    if (col8 >= fused_dim) return;

    const __nv_bfloat16* src = fused + row * fused_dim + col8;

    if (col8 + 8 <= first_dim) {
        // Entirely in the first section — vectorized copy.
        *reinterpret_cast<uint4*>(first + row * first_dim + col8) =
            *reinterpret_cast<const uint4*>(src);
    } else if (col8 >= first_dim && col8 + 8 <= fused_dim) {
        // Entirely in the second section — vectorized copy.
        *reinterpret_cast<uint4*>(second + row * second_dim + (col8 - first_dim)) =
            *reinterpret_cast<const uint4*>(src);
    } else {
        // Straddles the first/second boundary, or the row tail — scalar per element.
        for (int i = 0; i < 8 && col8 + i < fused_dim; i++) {
            __nv_bfloat16 v = src[i];
            if (col8 + i < first_dim) {
                first[row * first_dim + col8 + i] = v;
            } else {
                second[row * second_dim + (col8 + i - first_dim)] = v;
            }
        }
    }
}

// Fused silu_mul from merged gate+up buffer.
// Input:  gate_up [B, 2*inter_dim] where first half = gate, second half = up
// Output: out [B, inter_dim] = silu(gate) * up
//
// Vectorized: 8 bf16 per thread. Uses __expf (fast math) for the sigmoid.
__global__ void silu_mul_fused_kernel(
    const __nv_bfloat16* __restrict__ gate_up,  // [B, 2*inter_dim]
    __nv_bfloat16* __restrict__ out,             // [B, inter_dim]
    int inter_dim
) {
    int col8 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int row = blockIdx.y;
    if (col8 >= inter_dim) return;

    int gu_stride = 2 * inter_dim;
    const uint4* gate_src = reinterpret_cast<const uint4*>(gate_up + row * gu_stride + col8);
    const uint4* up_src = reinterpret_cast<const uint4*>(gate_up + row * gu_stride + inter_dim + col8);

    uint4 g = *gate_src;
    uint4 u = *up_src;

    __nv_bfloat162 g01 = *reinterpret_cast<__nv_bfloat162*>(&g.x);
    __nv_bfloat162 g23 = *reinterpret_cast<__nv_bfloat162*>(&g.y);
    __nv_bfloat162 g45 = *reinterpret_cast<__nv_bfloat162*>(&g.z);
    __nv_bfloat162 g67 = *reinterpret_cast<__nv_bfloat162*>(&g.w);

    __nv_bfloat162 u01 = *reinterpret_cast<__nv_bfloat162*>(&u.x);
    __nv_bfloat162 u23 = *reinterpret_cast<__nv_bfloat162*>(&u.y);
    __nv_bfloat162 u45 = *reinterpret_cast<__nv_bfloat162*>(&u.z);
    __nv_bfloat162 u67 = *reinterpret_cast<__nv_bfloat162*>(&u.w);

    auto silu_mul = [](float gate, float up) -> __nv_bfloat16 {
        float s = gate / (1.0f + expf(-gate));
        return __float2bfloat16(s * up);
    };

    uint4 result;
    __nv_bfloat162 r01, r23, r45, r67;
    r01.x = silu_mul(__bfloat162float(g01.x), __bfloat162float(u01.x));
    r01.y = silu_mul(__bfloat162float(g01.y), __bfloat162float(u01.y));
    r23.x = silu_mul(__bfloat162float(g23.x), __bfloat162float(u23.x));
    r23.y = silu_mul(__bfloat162float(g23.y), __bfloat162float(u23.y));
    r45.x = silu_mul(__bfloat162float(g45.x), __bfloat162float(u45.x));
    r45.y = silu_mul(__bfloat162float(g45.y), __bfloat162float(u45.y));
    r67.x = silu_mul(__bfloat162float(g67.x), __bfloat162float(u67.x));
    r67.y = silu_mul(__bfloat162float(g67.y), __bfloat162float(u67.y));
    result.x = *reinterpret_cast<unsigned int*>(&r01);
    result.y = *reinterpret_cast<unsigned int*>(&r23);
    result.z = *reinterpret_cast<unsigned int*>(&r45);
    result.w = *reinterpret_cast<unsigned int*>(&r67);

    *reinterpret_cast<uint4*>(out + row * inter_dim + col8) = result;
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
    dim3 grid((qkv_dim / 8 + threads - 1) / threads, batch_size);
    split_qkv_kernel<<<grid, threads, 0, stream>>>(
        qkv, q, k, v, q_dim, kv_dim, qkv_dim
    );
    return cudaGetLastError();
}

cudaError_t split2_cuda(
    const __nv_bfloat16* fused,
    __nv_bfloat16* first, __nv_bfloat16* second,
    int batch_size, int first_dim, int second_dim,
    cudaStream_t stream
) {
    int threads = 256;
    int fused_dim = first_dim + second_dim;
    dim3 grid((fused_dim / 8 + threads - 1) / threads, batch_size);
    split2_kernel<<<grid, threads, 0, stream>>>(fused, first, second, first_dim, second_dim);
    return cudaGetLastError();
}

cudaError_t silu_mul_fused_cuda(
    const __nv_bfloat16* gate_up,
    __nv_bfloat16* out,
    int batch_size, int inter_dim,
    cudaStream_t stream
) {
    int threads = 256;
    dim3 grid((inter_dim / 8 + threads - 1) / threads, batch_size);
    silu_mul_fused_kernel<<<grid, threads, 0, stream>>>(
        gate_up, out, inter_dim
    );
    return cudaGetLastError();
}

} // extern "C"

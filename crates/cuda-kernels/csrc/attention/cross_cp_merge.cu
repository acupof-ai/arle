// Cross-cp flash-decoding merge (T3.2b Part E): combine the cp sequence-shard
// partials into the full attention output. Each shard's FA3 produced a
// NORMALIZED partial (out, lse) over its local KV; the merge is the
// flash-decoding weighted average
//   out = sum_c w_c * out_c / sum_c w_c,  w_c = exp(lse_c - max_c lse_c)
// accumulated in f32. SGLang #21637's "separate local combine kernel" shape.
//
// Grid: (rows)  Threads: head_dim (256)
// lse_gather: [cp, rows] f32 (this rank's partial at row cp_rank*rows)
// out_gather: [cp, rows, head_dim] bf16 (same rank-major layout)
// out:        [rows, head_dim] bf16

#include "common.cuh"

#define HD256 256

__global__ void cross_cp_merge_kernel(
    const float* __restrict__ lse_gather,
    const __nv_bfloat16* __restrict__ out_gather,
    __nv_bfloat16* __restrict__ out,
    int cp_size,
    int rows,
    int head_dim
) {
    int row = blockIdx.x;
    int d = threadIdx.x;
    if (row >= rows || d >= head_dim) return;

    // m = max over cp shards; every thread computes the same scalar.
    float m = -INFINITY;
    for (int r = 0; r < cp_size; r++) {
        m = fmaxf(m, lse_gather[r * rows + row]);
    }
    float sum_w = 0.0f;
    float acc = 0.0f;
    for (int r = 0; r < cp_size; r++) {
        float w = __expf(lse_gather[r * rows + row] - m);
        sum_w += w;
        acc += w * __bfloat162float(out_gather[(r * rows + row) * head_dim + d]);
    }
    out[row * head_dim + d] = __float2bfloat16(acc / sum_w);
}

extern "C" {

cudaError_t cross_cp_merge_bf16_hd256_cuda(
    const float* lse_gather,
    const __nv_bfloat16* out_gather,
    __nv_bfloat16* out,
    int cp_size,
    int rows,
    int head_dim,
    cudaStream_t stream
) {
    if (lse_gather == nullptr || out_gather == nullptr || out == nullptr ||
        cp_size <= 1 || rows <= 0 || head_dim != HD256) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(rows);
    dim3 threads(head_dim);
    cross_cp_merge_kernel<<<grid, threads, 0, stream>>>(
        lse_gather, out_gather, out, cp_size, rows, head_dim
    );
    return cudaGetLastError();
}

} // extern "C"

// Cross-cp flash-decoding merge (T3.2b Part E): combine the cp sequence-shard
// partials into the full attention output. Each shard's FA3 produced a
// NORMALIZED partial (out, lse) over its local KV; the merge is the
// flash-decoding weighted average
//   out = sum_c w_c * out_c / sum_c w_c,  w_c = exp(lse_c - max_c lse_c)
// accumulated in f32. SGLang #21637's "separate local combine kernel" shape.
//
// Grid: (rows)  Threads: head_dim (256)
// packed: [cp, section_bf16] bf16, rank-major. Per-rank section holds `rows`
// f32 lse (as bf16 pairs) then `rows * head_dim` bf16 out. lse_stride_f32 =
// section_bf16 / 2, out_off_bf16 = rows * 2, out_stride_bf16 = section_bf16.
// out:    [rows, head_dim] bf16

#include "common.cuh"

#define HD256 256

__global__ void cross_cp_merge_kernel(
    const __nv_bfloat16* __restrict__ packed,
    int lse_stride_f32,
    int out_off_bf16,
    int out_stride_bf16,
    __nv_bfloat16* __restrict__ out,
    int cp_size,
    int rows,
    int head_dim
) {
    int row = blockIdx.x;
    int d = threadIdx.x;
    if (row >= rows || d >= head_dim) return;

    const float* lse_base = reinterpret_cast<const float*>(packed);
    // m = max over cp shards; every thread computes the same scalar.
    float m = -INFINITY;
    for (int r = 0; r < cp_size; r++) {
        m = fmaxf(m, lse_base[r * lse_stride_f32 + row]);
    }
    float sum_w = 0.0f;
    float acc = 0.0f;
    for (int r = 0; r < cp_size; r++) {
        float w = __expf(lse_base[r * lse_stride_f32 + row] - m);
        sum_w += w;
        acc += w * __bfloat162float(
            packed[r * out_stride_bf16 + out_off_bf16 + row * head_dim + d]);
    }
    out[row * head_dim + d] = __float2bfloat16(acc / sum_w);
}

extern "C" {

cudaError_t cross_cp_merge_bf16_hd256_cuda(
    const __nv_bfloat16* packed,
    int lse_stride_f32,
    int out_off_bf16,
    int out_stride_bf16,
    __nv_bfloat16* out,
    int cp_size,
    int rows,
    int head_dim,
    cudaStream_t stream
) {
    if (packed == nullptr || out == nullptr ||
        cp_size <= 1 || rows <= 0 || head_dim != HD256) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(rows);
    dim3 threads(head_dim);
    cross_cp_merge_kernel<<<grid, threads, 0, stream>>>(
        packed, lse_stride_f32, out_off_bf16, out_stride_bf16,
        out, cp_size, rows, head_dim
    );
    return cudaGetLastError();
}

} // extern "C"

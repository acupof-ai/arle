// DSv4 TP attention helpers: AllGather-side Q repack and rank-slice for the
// FlashMLA SM90 sparse-prefill TP path. FlashMLA's SM90 sparse-prefill
// kernel hard-asserts h_q % B_H == 0 (B_H = 64 tile size, see
// vendor/flashmla/csrc/sm90/prefill/sparse/{config.h:26, phase1.cuh:579}).
// DSv4 has 64 total Q heads, ARLE TP=8 → 8 local heads — fails the assert.
// Work-around: AllGather local Q to produce h_q=64 on every rank, run
// FlashMLA once, slice this rank's 8 heads back to local_attn.
//
// The AllGather concatenates per-rank [s_q, h_local, d] slabs rank-major
// in recv buffer: layout [tp_world, s_q, h_local, d]. FlashMLA wants
// [s_q, tp_world*h_local, d] with rank w's heads at [w*h_local, (w+1)*h_local).
// `dsv4_tp_q_repack_cuda` performs that transpose. After FlashMLA runs and
// fills [s_q, global_heads, d_v], `dsv4_tp_out_slice_cuda` copies this
// rank's 8-head slab into the existing per-rank local_attn buffer.
//
// Refs:
//   docs/experience/errors/2026-05-27-dsv4-flashmla-v1-h_q-tp-shard-mismatch.md
//   crates/cuda-kernels/csrc/misc/arle_flashmla_shim.cu

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

// gathered: [tp_world, s_q, h_local, d] row-major
// packed:   [s_q, tp_world * h_local, d] row-major
// Grid: (s_q * tp_world, h_local)
// Block: 256 threads cooperatively copy d elements (d ∈ {512, 576}).
__global__ void dsv4_tp_q_repack_kernel(
    const __nv_bfloat16* __restrict__ gathered,
    __nv_bfloat16* __restrict__ packed,
    int tp_world,
    int s_q,
    int h_local,
    int d
) {
    const int s = blockIdx.x / tp_world;
    const int w = blockIdx.x % tp_world;
    const int h = blockIdx.y;
    if (s >= s_q || h >= h_local) return;

    const __nv_bfloat16* src = gathered
        + (((int64_t)w * s_q + s) * h_local + h) * d;
    __nv_bfloat16* dst = packed
        + (((int64_t)s) * (tp_world * h_local) + w * h_local + h) * d;

    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        dst[k] = src[k];
    }
}

// full_out: [s_q, h_global, d] row-major; h_global = global_width / d
// local:    [s_q, h_local, d] row-major; h_local = local_width / d
// head_offset = tp_rank * local_width (in bf16 elements, not bytes)
// Grid: (s_q, h_local). Block: 256.
__global__ void dsv4_tp_out_slice_kernel(
    const __nv_bfloat16* __restrict__ full_out,
    __nv_bfloat16* __restrict__ local,
    int s_q,
    int global_width,
    int local_width,
    int head_offset,
    int d
) {
    const int s = blockIdx.x;
    const int h = blockIdx.y;
    if (s >= s_q) return;
    if (h * d >= local_width) return;

    const __nv_bfloat16* src = full_out
        + (int64_t)s * global_width + head_offset + h * d;
    __nv_bfloat16* dst = local
        + (int64_t)s * local_width + h * d;

    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        dst[k] = src[k];
    }
}

}  // namespace

extern "C" {

cudaError_t dsv4_tp_q_repack_cuda(
    const void* gathered,   // bf16 [tp_world, s_q, h_local, d]
    void* packed,           // bf16 [s_q, tp_world*h_local, d]
    int tp_world,
    int s_q,
    int h_local,
    int d,
    cudaStream_t stream
) {
    if (tp_world <= 0 || s_q <= 0 || h_local <= 0 || d <= 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid((unsigned)(s_q * tp_world), (unsigned)h_local, 1);
    dim3 block(256, 1, 1);
    dsv4_tp_q_repack_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gathered),
        reinterpret_cast<__nv_bfloat16*>(packed),
        tp_world, s_q, h_local, d
    );
    return cudaGetLastError();
}

cudaError_t dsv4_tp_out_slice_cuda(
    const void* full_out,   // bf16 [s_q, global_width / d, d]
    void* local,            // bf16 [s_q, local_width / d, d]
    int s_q,
    int global_width,
    int local_width,
    int head_offset,
    cudaStream_t stream
) {
    if (s_q <= 0 || global_width <= 0 || local_width <= 0) {
        return cudaErrorInvalidValue;
    }
    if (head_offset < 0 || head_offset + local_width > global_width) {
        return cudaErrorInvalidValue;
    }
    // FlashMLA SM90 sparse prefill supports d_v = 512 only (see
    // vendor/flashmla/csrc/api/common.h DISPATCH_HEAD_DIM).
    constexpr int d = 512;
    if (local_width % d != 0 || global_width % d != 0) {
        return cudaErrorInvalidValue;
    }
    const int h_local = local_width / d;
    dim3 grid((unsigned)s_q, (unsigned)h_local, 1);
    dim3 block(256, 1, 1);
    dsv4_tp_out_slice_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(full_out),
        reinterpret_cast<__nv_bfloat16*>(local),
        s_q, global_width, local_width, head_offset, d
    );
    return cudaGetLastError();
}

}  // extern "C"

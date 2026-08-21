// Native persistent paged attention for 1-byte quantized KV (Path B).
//
// Reads the FP8 e4m3 / INT8 KV pools and the per-(token, kv_head) f32 scales
// directly from the paged pool — no dequant temp buffer. Path A
// (arle_fa3_shim.cu) instead materializes a per-call bf16 pool of every page
// the table names, which is 5x the KV traffic of this path (1 byte/elem in +
// 2 bytes/elem out + 2 bytes/elem FA3 read vs 1 byte/elem in here).
//
// FA3-inspired persistent split-KV: one CTA per (batch row, group of
// `heads_per_cta` q-heads sharing a kv-head, split), grid
// [num_q_heads / heads_per_cta * num_splits, batch]; each CTA walks its KV token
// range through the rectangular page table (the same metadata the FA3 lane
// consumes) and writes a partial (o, m, l); a merge kernel combines the
// splits. Decode-shaped: exactly one query token per batch row (the caller
// gates on qlen == 1).
//
// The dequant and online-softmax math is identical to
// decode_attention_varlen_quantized.cu (the production fallback), so the
// correctness contract is inherited. A bf16 tensor-core MMA stage is a
// follow-up: the f32 dot path is arch-portable (this TU builds for every
// target) and matches the fallback numerically.
//
// Durable pool layout (NHD, token-major):
//   data:   [(phys * page_size + off) * kv_dim + kv_head * head_dim + d]
//   scales: [(phys * page_size + off) * num_kv_heads + kv_head]

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

#define PAF3_NUM_WARPS 4
#define PAF3_WARP_SIZE 32
#define PAF3_BLOCK_SIZE (PAF3_NUM_WARPS * PAF3_WARP_SIZE)

namespace {
constexpr int kMaxSplits = 16;
}

// A lane's `EPT` KV bytes are contiguous, so they are one aligned vector load
// rather than EPT scalar byte loads. `ncu` measured the scalar form at 85%
// excessive sectors: the L1 absorbed them (87% hit, DRAM at 5.9% of peak) but
// the L1TEX pipeline still paid for every one, which is what pinned memory
// throughput at 66.7%.
template <int EPT>
struct Paf3LaneBytes {
    static_assert(EPT == 4 || EPT == 8, "head_dim must be 128 or 256");
    // Words, not a byte array behind a union: nvcc cannot keep a union in
    // registers here and round-trips it through memory, which SASS showed as
    // 48 LDG.E.U8 becoming a store plus 96 LDS. Shifting the byte out of a
    // register word keeps the whole row resident.
    uint32_t w[EPT / 4];

    __device__ __forceinline__ void load(const uint8_t* src) {
        // `src` is `row_off + lane_id * EPT` and `row_off` is a multiple of
        // HEAD_DIM, so the address is EPT-aligned.
        if constexpr (EPT == 8) {
            const uint2 v = *reinterpret_cast<const uint2*>(src);
            w[0] = v.x;
            w[1] = v.y;
        } else {
            w[0] = *reinterpret_cast<const uint32_t*>(src);
        }
    }

    // Unscaled: the KV scale is constant over `d`, so it multiplies once
    // outside the per-element loop rather than EPT times inside it.
    __device__ __forceinline__ float raw(int i, bool int8_kv) const {
        const uint32_t byte = (w[i >> 2] >> ((i & 3) * 8)) & 0xFFu;
        if (int8_kv) {
            return static_cast<float>(static_cast<int8_t>(static_cast<uint8_t>(byte)));
        }
        __nv_fp8_e4m3 v;
        v.__x = static_cast<unsigned char>(byte);
        return static_cast<float>(v);
    }
};

__device__ __forceinline__ float paf3_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Partial (o, m, l) for one (batch row, q-head, split). The four warps stride
// over the split's token range, then merge in shared memory — the same
// reduction as decode_attention_varlen_quantized.cu.
// One CTA covers `H` consecutive q-heads that share a kv-head: the KV row is
// loaded and dequantised once and reused for all H dot products (the kernel
// was issue-bound on per-q-head dequant, not on bytes — ncu 2026-08-21). Q
// lives in shared memory so the H x EPT accumulator fits the register budget.
template <int HEAD_DIM, bool INT8_KV, int H>
__global__ void __launch_bounds__(PAF3_BLOCK_SIZE, 4)
paged_attention_quantized_fa3_partial_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const void* __restrict__ K_pool,
    const void* __restrict__ V_pool,
    const float* __restrict__ K_scales,
    const float* __restrict__ V_scales,
    const int* __restrict__ page_table,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seqused_k,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    int page_table_stride,
    int total_q,
    float sm_scale,
    int num_splits)
{
    constexpr int EPT = HEAD_DIM / PAF3_WARP_SIZE;

    const int group = blockIdx.x / num_splits;
    const int split = blockIdx.x % num_splits;
    const int b = blockIdx.y;
    const int q_head0 = group * H;
    if (q_head0 >= num_q_heads) return;

    const int warp_id = threadIdx.x / PAF3_WARP_SIZE;
    const int lane_id = threadIdx.x % PAF3_WARP_SIZE;

    const int kv_len = seqused_k[b];
    const int q_token = cu_seqlens_q[b];
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head = q_head0 / gqa_ratio;
    const int kv_dim = num_kv_heads * HEAD_DIM;

    const int qh_stride = total_q * num_q_heads;
    const int out_idx0 = split * qh_stride + q_token * num_q_heads + q_head0;

    auto write_empty_partial = [&]() {
        if (threadIdx.x < H) {
            partial_m[out_idx0 + threadIdx.x] = -FLT_MAX;
            partial_l[out_idx0 + threadIdx.x] = 0.0f;
        }
        for (int e = threadIdx.x; e < H * HEAD_DIM; e += PAF3_BLOCK_SIZE) {
            partial_out[(size_t)out_idx0 * HEAD_DIM + e] = 0.0f;
        }
    };

    if (kv_len <= 0) {
        write_empty_partial();
        return;
    }

    const int chunk = (kv_len + num_splits - 1) / num_splits;
    const int tok_start = split * chunk;
    const int tok_end = min(tok_start + chunk, kv_len);
    if (tok_start >= tok_end) {
        write_empty_partial();
        return;
    }

    __shared__ __align__(16) float smem_q[H * HEAD_DIM];
    __shared__ float smem_m[PAF3_NUM_WARPS * H];
    __shared__ float smem_l[PAF3_NUM_WARPS * H];
    __shared__ __align__(16) float smem_o[PAF3_NUM_WARPS * H * HEAD_DIM];

    {
        const int q_base = q_token * num_q_heads * HEAD_DIM + q_head0 * HEAD_DIM;
        for (int e = threadIdx.x; e < H * HEAD_DIM; e += PAF3_BLOCK_SIZE) {
            smem_q[e] = __bfloat162float(Q[q_base + e]) * sm_scale;
        }
    }
    __syncthreads();

    float o_reg[H][EPT];
    float m_local[H];
    float l_local[H];
    #pragma unroll
    for (int h = 0; h < H; h++) {
        #pragma unroll
        for (int i = 0; i < EPT; i++) o_reg[h][i] = 0.0f;
        m_local[h] = -FLT_MAX;
        l_local[h] = 0.0f;
    }

    for (int t = tok_start + warp_id; t < tok_end; t += PAF3_NUM_WARPS) {
        const int phys = page_table[b * page_table_stride + t / page_size];
        const int off = t % page_size;
        const size_t row_off = (size_t)phys * page_size * kv_dim
                             + (size_t)off * kv_dim
                             + (size_t)kv_head * HEAD_DIM;
        const int scale_idx = (phys * page_size + off) * num_kv_heads + kv_head;

        Paf3LaneBytes<EPT> k_bytes;
        k_bytes.load(reinterpret_cast<const uint8_t*>(K_pool) + row_off + lane_id * EPT);
        float kf[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++) kf[i] = k_bytes.raw(i, INT8_KV);
        const float k_scale = K_scales[scale_idx];

        float qk[H];
        #pragma unroll
        for (int h = 0; h < H; h++) {
            float acc = 0.0f;
            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                acc += smem_q[h * HEAD_DIM + lane_id * EPT + i] * kf[i];
            }
            qk[h] = acc;
        }
        #pragma unroll
        for (int h = 0; h < H; h++) {
            qk[h] = paf3_warp_reduce_sum(qk[h]) * k_scale;
        }

        Paf3LaneBytes<EPT> v_bytes;
        v_bytes.load(reinterpret_cast<const uint8_t*>(V_pool) + row_off + lane_id * EPT);
        float vf[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++) vf[i] = v_bytes.raw(i, INT8_KV);
        const float v_scale = V_scales[scale_idx];

        #pragma unroll
        for (int h = 0; h < H; h++) {
            const float m_new = fmaxf(m_local[h], qk[h]);
            const float exp_diff = __expf(m_local[h] - m_new);
            const float exp_qk = __expf(qk[h] - m_new);
            l_local[h] = l_local[h] * exp_diff + exp_qk;
            m_local[h] = m_new;
            const float exp_v = exp_qk * v_scale;
            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                o_reg[h][i] = o_reg[h][i] * exp_diff + exp_v * vf[i];
            }
        }
    }

    #pragma unroll
    for (int h = 0; h < H; h++) {
        if (lane_id == 0) {
            smem_m[warp_id * H + h] = m_local[h];
            smem_l[warp_id * H + h] = l_local[h];
        }
        // `[lane * EPT + i]` keeps a lane's floats contiguous so nvcc merges
        // the stores into STS.128; the bank conflict costs less than losing it.
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            smem_o[(warp_id * H + h) * HEAD_DIM + lane_id * EPT + i] = o_reg[h][i];
        }
    }
    __syncthreads();

    // Warp w merges head w, w+4, ... across the four token-strided warps.
    for (int h = warp_id; h < H; h += PAF3_NUM_WARPS) {
        float final_m = smem_m[h];
        float final_l = smem_l[h];
        float final_o[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            final_o[i] = smem_o[h * HEAD_DIM + lane_id * EPT + i];
        }
        #pragma unroll
        for (int w = 1; w < PAF3_NUM_WARPS; w++) {
            const float m_w = smem_m[w * H + h];
            const float l_w = smem_l[w * H + h];
            if (l_w == 0.0f) continue;

            const float m_new = fmaxf(final_m, m_w);
            const float scale_prev = __expf(final_m - m_new);
            const float scale_w = __expf(m_w - m_new);
            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                const float o_w = smem_o[(w * H + h) * HEAD_DIM + lane_id * EPT + i];
                final_o[i] = final_o[i] * scale_prev + o_w * scale_w;
            }
            final_l = final_l * scale_prev + l_w * scale_w;
            final_m = m_new;
        }

        const int out_idx = out_idx0 + h;
        if (lane_id == 0) {
            partial_m[out_idx] = final_m;
            partial_l[out_idx] = final_l;
        }
        const float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            partial_out[(size_t)out_idx * HEAD_DIM + lane_id * EPT + i] = final_o[i] * inv_l;
        }
    }
}

// Combine the per-split partials into the final bf16 output. One thread per
// output element; same online merge as the varlen kernel's merge stage.
template <int HEAD_DIM>
__global__ void paged_attention_quantized_fa3_merge_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ O,
    int total_q,
    int num_q_heads,
    int num_splits)
{
    const int q_token = blockIdx.x;
    const int q_head = blockIdx.y;
    const int d = threadIdx.x;
    if (q_token >= total_q || q_head >= num_q_heads || d >= HEAD_DIM) return;

    const int qh_stride = total_q * num_q_heads;
    const int q_idx = q_token * num_q_heads + q_head;

    float final_m = -FLT_MAX;
    float final_l = 0.0f;
    float final_o = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        const int idx = s * qh_stride + q_idx;
        const float m_s = partial_m[idx];
        const float l_s = partial_l[idx];
        const float o_s = partial_out[(size_t)idx * HEAD_DIM + d];
        if (l_s == 0.0f) continue;

        const float m_new = fmaxf(final_m, m_s);
        const float s_prev = final_l * __expf(final_m - m_new);
        const float s_cur = l_s * __expf(m_s - m_new);
        const float l_new = s_prev + s_cur;

        final_o = (l_new > 0.0f) ? (final_o * s_prev + o_s * s_cur) / l_new : 0.0f;
        final_m = m_new;
        final_l = l_new;
    }

    const int o_base = q_token * num_q_heads * HEAD_DIM + q_head * HEAD_DIM;
    O[o_base + d] = __float2bfloat16(final_o);
}

extern "C" {

// Workspace layout: partial_out [splits, total_q, q_heads, head_dim] f32,
// then partial_m and partial_l [splits, total_q, q_heads] f32 each.
size_t paged_attention_quantized_fa3_workspace_bytes(
    int total_q,
    int num_q_heads,
    int head_dim,
    int num_splits)
{
    if (total_q <= 0 || num_q_heads <= 0 || head_dim <= 0 || num_splits <= 0) {
        return 0;
    }
    const size_t qh = (size_t)total_q * (size_t)num_q_heads;
    return (size_t)num_splits * qh * ((size_t)head_dim + 2) * sizeof(float);
}

cudaError_t paged_attention_quantized_fa3_cuda(
    const __nv_bfloat16* q_packed,
    const void* k_pool,
    const void* v_pool,
    const float* k_scales,
    const float* v_scales,
    const int* page_table,       // [batch, page_table_stride] rectangular
    const int* cu_seqlens_q,     // [batch + 1]
    const int* seqused_k,        // [batch] per-row KV extent in tokens
    __nv_bfloat16* output,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int page_table_stride,
    int batch,
    int total_q,
    float sm_scale,
    bool is_fp8,
    int num_splits,
    int heads_per_cta,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_bytes)
{
    if (batch <= 0 || total_q <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 ||
        page_size <= 0 || page_table_stride <= 0 || num_splits <= 0 ||
        num_splits > kMaxSplits || num_q_heads % num_kv_heads != 0 ||
        (head_dim != 128 && head_dim != 256)) {
        return cudaErrorInvalidValue;
    }
    // `heads_per_cta` must tile the GQA group so every head in a CTA shares
    // one kv-head; the instantiated set is what the caller may pick from.
    const int gqa_ratio = num_q_heads / num_kv_heads;
    if (heads_per_cta <= 0 || gqa_ratio % heads_per_cta != 0) {
        return cudaErrorInvalidValue;
    }
    if (q_packed == nullptr || k_pool == nullptr || v_pool == nullptr ||
        k_scales == nullptr || v_scales == nullptr || page_table == nullptr ||
        cu_seqlens_q == nullptr || seqused_k == nullptr || output == nullptr ||
        workspace == nullptr) {
        return cudaErrorInvalidValue;
    }
    const size_t needed = paged_attention_quantized_fa3_workspace_bytes(
        total_q, num_q_heads, head_dim, num_splits);
    if (workspace_bytes < needed) return cudaErrorInvalidValue;

    float* ws = reinterpret_cast<float*>(workspace);
    const size_t qh = (size_t)total_q * (size_t)num_q_heads;
    float* partial_out = ws;
    float* partial_m = partial_out + (size_t)num_splits * qh * head_dim;
    float* partial_l = partial_m + (size_t)num_splits * qh;

    const dim3 grid((num_q_heads / heads_per_cta) * num_splits, batch);
    const dim3 block(PAF3_BLOCK_SIZE);

    cudaError_t err = cudaSuccess;
#define LAUNCH_PARTIAL_H(HD, INT8, HPC)                                      \
    paged_attention_quantized_fa3_partial_kernel<HD, INT8, HPC>              \
        <<<grid, block, 0, stream>>>(                                        \
            q_packed, k_pool, v_pool, k_scales, v_scales, page_table,        \
            cu_seqlens_q, seqused_k, partial_out, partial_m, partial_l,      \
            num_q_heads, num_kv_heads, page_size, page_table_stride,         \
            total_q, sm_scale, num_splits)
#define LAUNCH_PARTIAL(HD, INT8)                                             \
    switch (heads_per_cta) {                                                 \
        case 1: LAUNCH_PARTIAL_H(HD, INT8, 1); break;                        \
        case 2: LAUNCH_PARTIAL_H(HD, INT8, 2); break;                        \
        case 3: LAUNCH_PARTIAL_H(HD, INT8, 3); break;                        \
        case 4: LAUNCH_PARTIAL_H(HD, INT8, 4); break;                        \
        case 6: LAUNCH_PARTIAL_H(HD, INT8, 6); break;                        \
        case 8: LAUNCH_PARTIAL_H(HD, INT8, 8); break;                        \
        default: return cudaErrorInvalidValue;                               \
    }

    if (head_dim == 128) {
        if (is_fp8) { LAUNCH_PARTIAL(128, false) } else { LAUNCH_PARTIAL(128, true) }
    } else {
        if (is_fp8) { LAUNCH_PARTIAL(256, false) } else { LAUNCH_PARTIAL(256, true) }
    }
#undef LAUNCH_PARTIAL
#undef LAUNCH_PARTIAL_H
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    const dim3 merge_grid(total_q, num_q_heads);
    const dim3 merge_block(head_dim);
#define LAUNCH_MERGE(HD)                                                     \
    paged_attention_quantized_fa3_merge_kernel<HD>                           \
        <<<merge_grid, merge_block, 0, stream>>>(                            \
            partial_out, partial_m, partial_l, output, total_q,              \
            num_q_heads, num_splits)

    if (head_dim == 128) LAUNCH_MERGE(128);
    else LAUNCH_MERGE(256);
#undef LAUNCH_MERGE
    return cudaGetLastError();
}

}  // extern "C"

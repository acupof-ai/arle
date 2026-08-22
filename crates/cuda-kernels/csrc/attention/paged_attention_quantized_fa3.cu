// Native paged attention for 1-byte quantized KV.
//
// Reads the FP8 e4m3 / INT8 KV pools and the per-(token, kv_head) f32 scales
// directly from the paged pool — no dequant temp buffer. The FA3 quant shim
// (arle_fa3_shim.cu) instead materializes a per-call bf16 pool of every page
// the table names, which is 5x the KV traffic of this path (1 byte/elem in +
// 2 bytes/elem out + 2 bytes/elem FA3 read vs 1 byte/elem in here).
//
// Persistent split-KV on tensor cores: one CTA per (batch row, kv-head,
// split, q-tile), grid [num_kv_heads * num_splits, batch, q_tiles]. A q-tile
// is 16 rows of (query token, q-head) pairs of the kv-head's GQA group —
// head-fastest, so a decode row (qlen 1, G <= 16) is one tile and a spec
// verify row (qlen <= PAF3_MAX_QLEN) is ceil(qlen * G / 16) tiles. Each CTA
// walks its KV token range through the rectangular page table (the same
// metadata the FA3 lane consumes) with the row's own causal bound and writes
// a partial (o, m, l); a merge kernel combines the splits.
// sm_80+ (bf16 mma.sync); the sm_70 lane serves BF16 KV only.
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

// ─── Tensor-core partial kernel ─────────────────────────────────────────────
//
// One CTA per (batch row, kv-head, split). The kv-head's `G = gqa_ratio`
// q-heads are the M rows of a 16-row MMA tile (rows >= G are zero padding);
// each warp walks 16-token tiles of the split's range, dequantises the K and V
// bytes once into a bf16 shared tile, and runs S = Q·Kᵀ and O = P·V on
// `mma.sync.m16n8k16`. The per-token K/V scales are applied to the S columns
// and to P, so the bf16 tiles hold the exact FP8 / INT8 values (both are
// subsets of bf16). The four warps merge their (o, m, l) at the end.

__device__ __forceinline__ void paf3_ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void paf3_ldmatrix_x4_trans(uint32_t (&r)[4], const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void paf3_mma_bf16(float (&c)[4], const uint32_t (&a)[4],
                                              uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t paf3_pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&b);
}

// Two quantized bytes -> bf16x2, exact (no scale).
template <bool INT8_KV>
__device__ __forceinline__ uint32_t paf3_bytes2_to_bf16x2(uint32_t pair) {
    if constexpr (INT8_KV) {
        const float lo = static_cast<float>(static_cast<int8_t>(pair & 0xFFu));
        const float hi = static_cast<float>(static_cast<int8_t>((pair >> 8) & 0xFFu));
        return paf3_pack_bf16x2(lo, hi);
    } else {
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<__nv_fp8x2_storage_t>(pair & 0xFFFFu), __NV_E4M3);
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
        return paf3_pack_bf16x2(f.x, f.y);
    }
}

#define PAF3_TILE_TOK 16
#define PAF3_MAX_QLEN 8

template <int HEAD_DIM, bool INT8_KV>
__global__ void __launch_bounds__(PAF3_BLOCK_SIZE, 2)
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __trap();
#else
    constexpr int LD = HEAD_DIM + 8;  // bf16 row stride: 16 B skew keeps ldmatrix conflict-free
    constexpr int NT = HEAD_DIM / 8;  // d n-tiles for P·V
    constexpr int KS = HEAD_DIM / 16; // k-steps for Q·Kᵀ
    constexpr int BYTES_PER_LANE = HEAD_DIM / 2;  // two lanes per token row
    constexpr int VEC_PER_LANE = BYTES_PER_LANE / 16;

    const int kv_head = blockIdx.x / num_splits;
    const int split = blockIdx.x % num_splits;
    const int b = blockIdx.y;
    const int qtile = blockIdx.z;
    const int G = num_q_heads / num_kv_heads;
    const int q_head0 = kv_head * G;

    const int warp_id = threadIdx.x / PAF3_WARP_SIZE;
    const int lane_id = threadIdx.x % PAF3_WARP_SIZE;

    const int kv_len = seqused_k[b];
    const int q_token0 = cu_seqlens_q[b];
    const int qlen = cu_seqlens_q[b + 1] - q_token0;
    const int kv_dim = num_kv_heads * HEAD_DIM;
    const int rows = qlen * G;
    const int row0 = qtile * 16;
    if (row0 >= rows) return;

    // Tile row r -> (query token t, head h); partial index of that row.
    const int qh_stride = total_q * num_q_heads;
    auto row_out_idx = [&](int r) {
        const int t = r / G, h = r % G;
        return split * qh_stride + (q_token0 + t) * num_q_heads + q_head0 + h;
    };

    auto write_empty_partial = [&]() {
        for (int r = row0; r < min(row0 + 16, rows); r++) {
            const int out_idx = row_out_idx(r);
            if (threadIdx.x == 0) {
                partial_m[out_idx] = -FLT_MAX;
                partial_l[out_idx] = 0.0f;
            }
            for (int e = threadIdx.x; e < HEAD_DIM; e += PAF3_BLOCK_SIZE) {
                partial_out[(size_t)out_idx * HEAD_DIM + e] = 0.0f;
            }
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

    __shared__ __align__(16) __nv_bfloat16 smem_q[16 * LD];
    __shared__ __align__(16) __nv_bfloat16 smem_kv[PAF3_NUM_WARPS][PAF3_TILE_TOK * LD];
    __shared__ float smem_ks[PAF3_NUM_WARPS][PAF3_TILE_TOK];
    __shared__ float smem_vs[PAF3_NUM_WARPS][PAF3_TILE_TOK];

    {
        const __nv_bfloat16 zero = __float2bfloat16(0.0f);
        for (int e = threadIdx.x; e < 16 * HEAD_DIM; e += PAF3_BLOCK_SIZE) {
            const int r = e / HEAD_DIM, d = e % HEAD_DIM;
            const int gr = row0 + r;
            smem_q[r * LD + d] = (gr < rows)
                ? Q[(size_t)(q_token0 + gr / G) * num_q_heads * HEAD_DIM
                    + (size_t)(q_head0 + gr % G) * HEAD_DIM + d]
                : zero;
        }
    }
    __syncthreads();

    // Causal bound of this lane's two rows: query token t sees kv positions
    // below kv_len - qlen + 1 + t (the new tokens are already in the pool).
    const int r0 = lane_id >> 2;
    const int c0 = (lane_id & 3) * 2;
    int row_lim[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int gr = row0 + r0 + 8 * i;
        row_lim[i] = gr < rows ? kv_len - qlen + 1 + gr / G : 0;
    }

    // Accumulator fragment layout (m16n8): lane holds rows r0 = lane/4 and
    // r0 + 8, columns (lane%4)*2 + {0,1}.

    float o_acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        o_acc[i][0] = o_acc[i][1] = o_acc[i][2] = o_acc[i][3] = 0.0f;
    }
    float m_row[2] = {-FLT_MAX, -FLT_MAX};
    float l_row[2] = {0.0f, 0.0f};

    __nv_bfloat16* kv_tile = smem_kv[warp_id];
    float* ks = smem_ks[warp_id];
    float* vs = smem_vs[warp_id];

    // Lane's token row and half-row for the tile load.
    const int ld_tok = lane_id >> 1;
    const int ld_half = lane_id & 1;

    // ldmatrix row addresses (bf16 elements, before the column offset).
    // A (Q, 16x16): m0 rows 0-7 k 0-7, m1 rows 8-15 k 0-7, m2 rows 0-7 k 8-15, m3 rows 8-15 k 8-15.
    const int a_row = (lane_id & 7) + 8 * ((lane_id >> 3) & 1);
    const int a_col = 8 * (lane_id >> 4);
    // B for K (n = tok, k = d): m0 tok 0-7 k 0-7, m1 tok 0-7 k 8-15, m2 tok 8-15 k 0-7, m3 tok 8-15 k 8-15.
    const int bk_row = (lane_id & 7) + 8 * (lane_id >> 4);
    const int bk_col = 8 * ((lane_id >> 3) & 1);
    // B for V (k = tok, n = d), trans: m0 tok 0-7 d-tile j, m1 tok 8-15 d-tile j, m2/m3 d-tile j+1.
    const int bv_row = (lane_id & 7) + 8 * ((lane_id >> 3) & 1);
    const int bv_col = 8 * (lane_id >> 4);

    for (int tb = tok_start + warp_id * PAF3_TILE_TOK; tb < tok_end; tb += PAF3_NUM_WARPS * PAF3_TILE_TOK) {
        const int t = tb + ld_tok;
        const bool t_ok = t < tok_end;
        size_t row_off = 0;
        int scale_idx = 0;
        if (t_ok) {
            const int phys = page_table[b * page_table_stride + t / page_size];
            const int off = t % page_size;
            row_off = (size_t)phys * page_size * kv_dim + (size_t)off * kv_dim
                    + (size_t)kv_head * HEAD_DIM;
            scale_idx = (phys * page_size + off) * num_kv_heads + kv_head;
        }

        // ── K tile ──
        if (ld_half == 0) {
            ks[ld_tok] = t_ok ? K_scales[scale_idx] * sm_scale : 0.0f;
            vs[ld_tok] = t_ok ? V_scales[scale_idx] : 0.0f;
        }
        {
            const uint8_t* src = reinterpret_cast<const uint8_t*>(K_pool) + row_off + ld_half * BYTES_PER_LANE;
            uint4* dst = reinterpret_cast<uint4*>(kv_tile + ld_tok * LD + ld_half * BYTES_PER_LANE);
            #pragma unroll
            for (int v = 0; v < VEC_PER_LANE; v++) {
                uint4 raw = make_uint4(0u, 0u, 0u, 0u);
                if (t_ok) raw = *reinterpret_cast<const uint4*>(src + v * 16);
                const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
                uint4 lo, hi;
                lo.x = paf3_bytes2_to_bf16x2<INT8_KV>(w[0]);
                lo.y = paf3_bytes2_to_bf16x2<INT8_KV>(w[0] >> 16);
                lo.z = paf3_bytes2_to_bf16x2<INT8_KV>(w[1]);
                lo.w = paf3_bytes2_to_bf16x2<INT8_KV>(w[1] >> 16);
                hi.x = paf3_bytes2_to_bf16x2<INT8_KV>(w[2]);
                hi.y = paf3_bytes2_to_bf16x2<INT8_KV>(w[2] >> 16);
                hi.z = paf3_bytes2_to_bf16x2<INT8_KV>(w[3]);
                hi.w = paf3_bytes2_to_bf16x2<INT8_KV>(w[3] >> 16);
                dst[v * 2] = lo;
                dst[v * 2 + 1] = hi;
            }
        }
        __syncwarp();

        // ── S = Q·Kᵀ (16 x 16 tokens) ──
        float s[2][4];
        #pragma unroll
        for (int j = 0; j < 2; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < KS; kk++) {
            uint32_t a[4], bk[4];
            paf3_ldmatrix_x4(a, smem_q + a_row * LD + kk * 16 + a_col);
            paf3_ldmatrix_x4(bk, kv_tile + bk_row * LD + kk * 16 + bk_col);
            paf3_mma_bf16(s[0], a, bk[0], bk[1]);
            paf3_mma_bf16(s[1], a, bk[2], bk[3]);
        }
        __syncwarp();

        // ── scale, mask, online softmax ──
        float p[2][4];
        float rmax[2] = {-FLT_MAX, -FLT_MAX};
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                const int tok = 8 * j + c0 + c;
                const int pos = tb + tok;
                const float sc = ks[tok];
                s[j][c] = (pos < tok_end && pos < row_lim[0]) ? s[j][c] * sc : -FLT_MAX;
                s[j][2 + c] = (pos < tok_end && pos < row_lim[1]) ? s[j][2 + c] * sc : -FLT_MAX;
                rmax[0] = fmaxf(rmax[0], s[j][c]);
                rmax[1] = fmaxf(rmax[1], s[j][2 + c]);
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            rmax[i] = fmaxf(rmax[i], __shfl_xor_sync(0xffffffff, rmax[i], 1));
            rmax[i] = fmaxf(rmax[i], __shfl_xor_sync(0xffffffff, rmax[i], 2));
        }
        float alpha[2], rsum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const float m_new = fmaxf(m_row[i], rmax[i]);
            alpha[i] = __expf(m_row[i] - m_new);
            m_row[i] = m_new;
        }
        // A row whose causal bound sits below this split's first tile has no
        // finite score yet: its probabilities must stay 0, not exp(0).
        const bool live0 = m_row[0] > -FLT_MAX, live1 = m_row[1] > -FLT_MAX;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                p[j][c] = live0 ? __expf(s[j][c] - m_row[0]) : 0.0f;
                p[j][2 + c] = live1 ? __expf(s[j][2 + c] - m_row[1]) : 0.0f;
                rsum[0] += p[j][c];
                rsum[1] += p[j][2 + c];
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            rsum[i] += __shfl_xor_sync(0xffffffff, rsum[i], 1);
            rsum[i] += __shfl_xor_sync(0xffffffff, rsum[i], 2);
            l_row[i] = l_row[i] * alpha[i] + rsum[i];
        }
        #pragma unroll
        for (int i = 0; i < NT; i++) {
            o_acc[i][0] *= alpha[0];
            o_acc[i][1] *= alpha[0];
            o_acc[i][2] *= alpha[1];
            o_acc[i][3] *= alpha[1];
        }

        // P as the A operand, with the per-token V scale folded in.
        uint32_t pa[4];
        {
            const float v00 = vs[c0], v01 = vs[c0 + 1], v10 = vs[8 + c0], v11 = vs[8 + c0 + 1];
            pa[0] = paf3_pack_bf16x2(p[0][0] * v00, p[0][1] * v01);
            pa[1] = paf3_pack_bf16x2(p[0][2] * v00, p[0][3] * v01);
            pa[2] = paf3_pack_bf16x2(p[1][0] * v10, p[1][1] * v11);
            pa[3] = paf3_pack_bf16x2(p[1][2] * v10, p[1][3] * v11);
        }

        // ── V tile (same buffer) ──
        {
            const uint8_t* src = reinterpret_cast<const uint8_t*>(V_pool) + row_off + ld_half * BYTES_PER_LANE;
            uint4* dst = reinterpret_cast<uint4*>(kv_tile + ld_tok * LD + ld_half * BYTES_PER_LANE);
            #pragma unroll
            for (int v = 0; v < VEC_PER_LANE; v++) {
                uint4 raw = make_uint4(0u, 0u, 0u, 0u);
                if (t_ok) raw = *reinterpret_cast<const uint4*>(src + v * 16);
                const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
                uint4 lo, hi;
                lo.x = paf3_bytes2_to_bf16x2<INT8_KV>(w[0]);
                lo.y = paf3_bytes2_to_bf16x2<INT8_KV>(w[0] >> 16);
                lo.z = paf3_bytes2_to_bf16x2<INT8_KV>(w[1]);
                lo.w = paf3_bytes2_to_bf16x2<INT8_KV>(w[1] >> 16);
                hi.x = paf3_bytes2_to_bf16x2<INT8_KV>(w[2]);
                hi.y = paf3_bytes2_to_bf16x2<INT8_KV>(w[2] >> 16);
                hi.z = paf3_bytes2_to_bf16x2<INT8_KV>(w[3]);
                hi.w = paf3_bytes2_to_bf16x2<INT8_KV>(w[3] >> 16);
                dst[v * 2] = lo;
                dst[v * 2 + 1] = hi;
            }
        }
        __syncwarp();

        // ── O += P·V ──
        #pragma unroll
        for (int j = 0; j < NT; j += 2) {
            uint32_t bv[4];
            paf3_ldmatrix_x4_trans(bv, kv_tile + bv_row * LD + j * 8 + bv_col);
            paf3_mma_bf16(o_acc[j], pa, bv[0], bv[1]);
            paf3_mma_bf16(o_acc[j + 1], pa, bv[2], bv[3]);
        }
        __syncwarp();
    }

    // ── merge the four warps into warp 0 ──
    float* stage_o = reinterpret_cast<float*>(&smem_kv[0][0]);  // [16][HEAD_DIM]
    __shared__ float stage_m[16];
    __shared__ float stage_l[16];
    for (int w = 1; w < PAF3_NUM_WARPS; w++) {
        __syncthreads();
        if (warp_id == w) {
            if ((lane_id & 3) == 0) {
                stage_m[r0] = m_row[0]; stage_l[r0] = l_row[0];
                stage_m[r0 + 8] = m_row[1]; stage_l[r0 + 8] = l_row[1];
            }
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                const int d = i * 8 + c0;
                *reinterpret_cast<float2*>(stage_o + r0 * HEAD_DIM + d) = make_float2(o_acc[i][0], o_acc[i][1]);
                *reinterpret_cast<float2*>(stage_o + (r0 + 8) * HEAD_DIM + d) = make_float2(o_acc[i][2], o_acc[i][3]);
            }
        }
        __syncthreads();
        if (warp_id == 0) {
            float sc_prev[2], sc_w[2];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int r = r0 + 8 * i;
                const float m_w = stage_m[r], l_w = stage_l[r];
                if (l_w == 0.0f) { sc_prev[i] = 1.0f; sc_w[i] = 0.0f; continue; }
                const float m_new = fmaxf(m_row[i], m_w);
                sc_prev[i] = __expf(m_row[i] - m_new);
                sc_w[i] = __expf(m_w - m_new);
                l_row[i] = l_row[i] * sc_prev[i] + l_w * sc_w[i];
                m_row[i] = m_new;
            }
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                const int d = i * 8 + c0;
                const float2 a = *reinterpret_cast<const float2*>(stage_o + r0 * HEAD_DIM + d);
                const float2 c = *reinterpret_cast<const float2*>(stage_o + (r0 + 8) * HEAD_DIM + d);
                o_acc[i][0] = o_acc[i][0] * sc_prev[0] + a.x * sc_w[0];
                o_acc[i][1] = o_acc[i][1] * sc_prev[0] + a.y * sc_w[0];
                o_acc[i][2] = o_acc[i][2] * sc_prev[1] + c.x * sc_w[1];
                o_acc[i][3] = o_acc[i][3] * sc_prev[1] + c.y * sc_w[1];
            }
        }
    }

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int gr = row0 + r0 + 8 * i;
            if (gr >= rows) continue;
            const int out_idx = row_out_idx(gr);
            if ((lane_id & 3) == 0) {
                partial_m[out_idx] = m_row[i];
                partial_l[out_idx] = l_row[i];
            }
            const float inv_l = (l_row[i] > 0.0f) ? (1.0f / l_row[i]) : 0.0f;
            float* dst = partial_out + (size_t)out_idx * HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < NT; j++) {
                const int d = j * 8 + c0;
                *reinterpret_cast<float2*>(dst + d) =
                    make_float2(o_acc[j][2 * i] * inv_l, o_acc[j][2 * i + 1] * inv_l);
            }
        }
    }
#endif
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
    int max_qlen,
    float sm_scale,
    bool is_fp8,
    int num_splits,
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
    // A decode row's GQA group is one 16-row MMA tile; a verify row of up to
    // PAF3_MAX_QLEN tokens spans ceil(max_qlen * G / 16) tiles.
    const int gqa = num_q_heads / num_kv_heads;
    if (gqa > 16 || max_qlen <= 0 || max_qlen > PAF3_MAX_QLEN) return cudaErrorInvalidValue;
    const int q_tiles = (max_qlen * gqa + 15) / 16;
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

    const dim3 grid(num_kv_heads * num_splits, batch, q_tiles);
    const dim3 block(PAF3_BLOCK_SIZE);

    cudaError_t err = cudaSuccess;
#define LAUNCH_PARTIAL(HD, INT8)                                             \
    paged_attention_quantized_fa3_partial_kernel<HD, INT8>                   \
        <<<grid, block, 0, stream>>>(                                        \
            q_packed, k_pool, v_pool, k_scales, v_scales, page_table,        \
            cu_seqlens_q, seqused_k, partial_out, partial_m, partial_l,      \
            num_q_heads, num_kv_heads, page_size, page_table_stride,         \
            total_q, sm_scale, num_splits)
    if (head_dim == 128) {
        if (is_fp8) LAUNCH_PARTIAL(128, false); else LAUNCH_PARTIAL(128, true);
    } else {
        if (is_fp8) LAUNCH_PARTIAL(256, false); else LAUNCH_PARTIAL(256, true);
    }
#undef LAUNCH_PARTIAL
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

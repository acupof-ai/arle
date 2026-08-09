// DSv4-Flash (MODEL1) FP8 KV pack kernel.
//
// Packs ARLE's bf16 DSv4 KV (NoPE 448 + RoPE 64) into the **MODEL1 FP8
// block-paged layout** consumed by upstream FlashMLA's
// `sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`.
//
// Per `vendor/flashmla/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`:
//   - L491: `fp8* k_ptr = (fp8*)params.kv;` — reinterprets KV as fp8.
//   - L554: `gK_base = k_ptr + block_index*k_block_stride
//                     + rel_idx_in_block*(HEAD_DIM_NOPE + HEAD_DIM_ROPE*sizeof(bf16));`
//   - L555: `gK_scales_base = (fp8_e8m0*)(k_ptr + block_index*k_block_stride
//                     + page_block_size*(HEAD_DIM_NOPE+HEAD_DIM_ROPE*sizeof(bf16))
//                     + rel_idx_in_block*NUM_SCALES*sizeof(fp8_e8m0));`
//   - L694-695: `BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE + 8 = 584`
//               `KU_ASSERT(stride_kv_row == BYTES_PER_TOKEN)`
//   - L560: scales are decoded via `__nv_cvt_e8m0x2_to_bf162raw` (e8m0 → bf16).
//   - L588 (dequant.h:cvt_fp8x8_bf16x8): `bf16 = __float22bfloat162_rn(fp8 -> f32) * scale_bf16`.
//
// MODEL1 constants (HEAD_DIM_NOPE=448, HEAD_DIM_ROPE=64, QUANT_TILE_SIZE=64,
// NUM_SCALES=8, page_block_size=64):
//
//   per-block layout (37376 B = 64 * 584):
//     offset 0      : [T0 NoPE 448 B][T0 RoPE 128 B]  (576 B per token AoS)
//     offset 576    : [T1 NoPE 448 B][T1 RoPE 128 B]
//     ...
//     offset 36288  : [T63 NoPE 448 B][T63 RoPE 128 B]
//     offset 36864  : [T0 scales 8 B]...[T63 scales 8 B]  (8 e8m0 bytes per token)
//
// E8M0 scale encoding (from `__nv_cvt_e8m0x2_to_bf162raw` semantics, exponent-only):
//   byte b  ∈ [1, 254]  ⇒  bf16 scale = 2^(b - 127)
//   byte b  = 0         ⇒  scale = 0 (zero-tile)
//   byte b  = 255       ⇒  NaN
//
// For each 64-element NoPE tile we compute the smallest e ∈ Z such that
//   2^e * 448 ≥ amax,
// i.e. `e = ceil(log2(amax / 448))`, clamped to [-126, 127], then store
// `byte = e + 127`. Each fp8_e4m3 value is then `__nv_fp8_e4m3(val / 2^e)`.
//
// This guarantees |val / scale| ≤ 448 (the E4M3 representable max) for all
// tile elements, matching what the kernel's reverse path expects.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// V32 (GLM-5.2) contract — DIFFERENT format from MODEL1.
//
// The vendored V32 sparse-FP8 decode reads a per-token INLINE layout (NOT the
// MODEL1 trailing-e8m0 scale region). Ground truth:
//   - `vendor/flashmla/csrc/sm90/decode/sparse_fp8/config.h`: V32 →
//     HEAD_DIM_NOPE=512, HEAD_DIM_ROPE=64, QUANT_TILE_SIZE=128, NUM_SCALES=4.
//   - `splitkv_mla.cuh` V32 branch (~543-562, 600-605):
//       gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*k_row_stride
//                 (per-token stride = k_row_stride = 656, FULLY INLINE)
//       scales  = *(float4*)load_128b((float*)(gK_base + HEAD_DIM_NOPE))
//                 (4× F32, INLINE, at offset 512)
//       RoPE    @ gK_base + HEAD_DIM_NOPE + NUM_SCALES*sizeof(float) = offset 528
//       dequant uses scale = scales[dim_idx/2] over dim_idx∈[0,8) 64-blocks, so
//       each F32 scale covers a 128-elem NoPE block.
//
// Definitive V32 per-token layout (656 bytes, INLINE, stride 656):
//   [0, 512)   NoPE fp8 (512 bytes)
//   [512, 528) 4× F32 scales (16 bytes) — block b covers NoPE elems
//              [b*128, (b+1)*128); scale[b] = amax(block_b) / 448.0 stored as
//              F32 (NO power-of-two rounding); 448 = E4M3 max
//   [528, 656) RoPE bf16 (64 dims × 2 = 128 bytes)
//   block stride = page_block_size * 656.
//
// This is implemented by the SEPARATE `dsv4_v32_fp8_kv_pack_kernel` below — it
// does NOT share the MODEL1 template (which writes a trailing e8m0 scale region
// at 64-elem QUANT_TILE granularity, the wrong format for V32).

namespace {

// Shape-INDEPENDENT constants shared by the MODEL1 pack body. The
// NoPE-dim-dependent constants (NUM_TILES, NUM_SCALES, TOKEN_DATA_BYTES,
// TOKEN_BYTES) are derived as `constexpr` LOCALS inside the
// `dsv4_fp8_kv_pack_kernel<HEAD_DIM_NOPE>` template. The E4M3-max=448 used in
// the SCALE MATH is the representable max, NOT the dim — it stays 448 verbatim.
// V32 (NoPE=512, inline F32 scales) is a SEPARATE kernel, not a template
// instantiation here.
static constexpr int HEAD_DIM_ROPE   = 64;
static constexpr int QUANT_TILE_SIZE = 64;
static constexpr int ROPE_BYTES      = HEAD_DIM_ROPE * sizeof(__nv_bfloat16);

// V32 inline-layout constants (config.h: HEAD_DIM_NOPE=512, QUANT_TILE_SIZE=128,
// NUM_SCALES=4). Fully inline 656 B/token: [512 NoPE fp8][4 F32 @512][128 rope bf16].
static constexpr int V32_HEAD_DIM_NOPE  = 512;
static constexpr int V32_QUANT_TILE     = 128;             // 128-elem NoPE block / F32 scale
static constexpr int V32_NUM_SCALES     = 4;
static constexpr int V32_SCALE_BYTES    = V32_NUM_SCALES * (int)sizeof(float);
static constexpr int V32_TOKEN_BYTES    =
    V32_HEAD_DIM_NOPE + V32_SCALE_BYTES + ROPE_BYTES;

static constexpr int THREADS_PER_BLOCK = 128;
static constexpr int NOPE_THREADS      = 64;   // 0..63 handle NoPE quant
// 64..127 handle RoPE copy

// Warp-shuffle reduce: max over 32 lanes.
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    }
    return v;
}

// One block per token. 128 threads per block (4 warps).
//
// Thread role split:
//   threads [0..63]  (warps 0+1): NoPE quantization, 7 tiles in series.
//   threads [64..127] (warps 2+3): RoPE bf16 copy (64 dims).
//
// All threads participate in `__syncthreads()` barriers each tile iteration
// so cross-warp shared-memory broadcast of the tile scale is safe.
//
// Per-tile flow (NoPE):
//   1. NoPE threads load bf16 value at (token, tile*64 + lane).
//   2. amax = warp_reduce_max(|val|) within each warp; combine the two
//      warps via two shared slots `s_warp_max[0..1]`.
//   3. Lane 0 of NoPE group computes e8m0 byte + bf16 scale (broadcast via
//      `s_scale_bits`), writes the byte to the per-token scales region.
//   4. All NoPE threads quantize: `__nv_fp8_e4m3(val / scale)` and write
//      the fp8 byte into the per-token NoPE region.
//
// RoPE threads do nothing during NoPE tiles (just sit at the barriers), then
// after the last tile copy 64 bf16 RoPE elements (1 thread per dim).
//
// Block-paged write addressing:
//   block_id  = token_block_id[t]
//   row       = token_in_block_row[t]   ∈ [0, page_block_size)
//   block_base = packed_kv + block_id * (page_block_size * TOKEN_BYTES)
//   NoPE+RoPE @ block_base + row * (HEAD_DIM_NOPE + ROPE_BYTES)
//   scales    @ block_base + page_block_size * (HEAD_DIM_NOPE + ROPE_BYTES)
//             + row * NUM_SCALES
//
// `stride_nope_elems` / `stride_rope_elems` are the per-token strides in
// **bf16 elements** for the NoPE and RoPE input buffers. The contiguous
// shipping caller passes (HEAD_DIM_NOPE=448, HEAD_DIM_ROPE=64); the
// interleaved-in-`k_prepared` caller passes (head_dim=512, head_dim=512)
// with separate base pointers `k_prepared + 0` (NoPE) and
// `k_prepared + 448` (RoPE).
//
// MODEL1 only (NoPE=448). V32 is handled by `dsv4_v32_fp8_kv_pack_kernel`.
//
// Stage-B page-table lookup (gated): when `page_table != nullptr`, `token_block_id[t]`
// carries a slot-LOGICAL page and the physical block is `page_table[token_block_id[t]]`.
// `page_table == nullptr` (Stage-A default) keeps `block_id = token_block_id[t]` verbatim
// — identity table (`page_table[i] = first + i`) reproduces the band path byte-for-byte.
template <int HEAD_DIM_NOPE>
__global__ void dsv4_fp8_kv_pack_kernel(
    const __nv_bfloat16* __restrict__ nope,   // [n_tokens, stride_nope_elems]
    const __nv_bfloat16* __restrict__ rope,   // [n_tokens, stride_rope_elems]
    uint8_t* __restrict__ packed_kv,
    const int* __restrict__ token_block_id,
    const int* __restrict__ token_in_block_row,
    int n_tokens,
    int page_block_size,
    int stride_nope_elems,
    int stride_rope_elems,
    const int* __restrict__ page_table,
    int num_logical_pages)
{
    // MODEL1-only NoPE-dim-dependent constants (NoPE=448 → NUM_TILES=7,
    // NUM_SCALES=8, TOKEN_BYTES=584). The trailing e8m0 scale region at 64-elem
    // QUANT_TILE granularity is the MODEL1 format; V32 uses a different inline
    // F32-scale layout handled by the SEPARATE `dsv4_v32_fp8_kv_pack_kernel`.
    static_assert(HEAD_DIM_NOPE == 448, "MODEL1 pack template is NoPE=448 only");
    constexpr int NUM_TILES       = HEAD_DIM_NOPE / QUANT_TILE_SIZE;
    constexpr int NUM_SCALES      = 8;                                      // 7 used + 1 pad
    constexpr int TOKEN_DATA_BYTES = HEAD_DIM_NOPE + ROPE_BYTES;
    constexpr int TOKEN_BYTES      = TOKEN_DATA_BYTES + NUM_SCALES;

    const int t = blockIdx.x;
    if (t >= n_tokens) return;

    const int tid = threadIdx.x;
    const int row = token_in_block_row[t];
    // Stage-B: route the logical page through the device table; Stage-A: identity.
    int block_id = token_block_id[t];
    if (page_table != nullptr) {
        if (block_id < 0 || block_id >= num_logical_pages) return;
        block_id = page_table[block_id];
    }

    // Block-paged base (in bytes).
    const int64_t block_stride = (int64_t)page_block_size * TOKEN_BYTES;
    uint8_t* block_base = packed_kv + (int64_t)block_id * block_stride;

    // Per-token NoPE+RoPE base (576 bytes/token AoS).
    uint8_t* token_data_base = block_base + (int64_t)row * TOKEN_DATA_BYTES;
    // Per-token scales (8 e8m0 bytes/token, appended after the AoS region).
    uint8_t* token_scales_base =
        block_base + (int64_t)page_block_size * TOKEN_DATA_BYTES
        + (int64_t)row * NUM_SCALES;

    // Shared scratch for the NoPE 64-lane reduction + broadcast.
    __shared__ float s_warp_max[2];   // 2 warps × 32 lanes = 64 NoPE threads
    // Broadcast the bf16 tile scale as a float (carries the exact bf16 value
    // after a round-trip __bfloat162float; the source is a power of 2 so the
    // conversion is exact). Threads read the float and convert to bf16 to
    // match the kernel's dequant numerics (which apply scale_bf16 × fp8→f32).
    __shared__ float s_scale_f;

    const bool is_nope = (tid < NOPE_THREADS);
    const int  nope_lane = is_nope ? tid : -1;     // 0..63 for NoPE threads
    const int  rope_lane = (tid >= NOPE_THREADS) ? (tid - NOPE_THREADS) : -1;

    // Iterate 7 tiles for NoPE. All 128 threads enter the loop body so
    // __syncthreads() barriers below are well-formed.
    #pragma unroll
    for (int tile = 0; tile < NUM_TILES; ++tile) {
        float v = 0.0f;
        if (is_nope) {
            const int dim_idx = tile * QUANT_TILE_SIZE + nope_lane;
            v = __bfloat162float(nope[(int64_t)t * stride_nope_elems + dim_idx]);
        }

        // Two-warp reduction (only NoPE warps participate in writes).
        if (is_nope) {
            float a = fabsf(v);
            float warp_max = warp_reduce_max(a);
            if ((nope_lane & 31) == 0) {
                s_warp_max[nope_lane >> 5] = warp_max;
            }
        }
        __syncthreads();

        if (is_nope && nope_lane == 0) {
            float amax = fmaxf(s_warp_max[0], s_warp_max[1]);

            // ceil(log2(amax / 448)) → e
            // bf16 scale = 2^e (exponent-only)
            uint8_t byte;
            __nv_bfloat16 scale_bf16;
            if (amax <= 0.0f || !isfinite(amax)) {
                // Zero tile → byte=0 → __nv_cvt_e8m0x2_to_bf162raw decodes to
                // bf16 zero per E8M0 spec. Locally use scale=1 so val/scale=0
                // doesn't introduce NaNs.
                byte = 0;
                scale_bf16 = __float2bfloat16(1.0f);
            } else {
                // Compute e = ceil(log2(amax / 448)). Use frexpf to avoid
                // log2 rounding traps near powers of 2.
                //
                // frexpf(amax, &e_amax): amax = m * 2^e_amax with m ∈ [0.5, 1.0).
                // ⇒ amax ∈ [0.5, 1.0) * 2^e_amax
                // ⇒ amax / 448 ∈ [0.5/448, 1.0/448) * 2^e_amax
                // 448 = 1.75 * 2^8, log2(448) ≈ 8.807.
                // ⇒ ⌈log2(amax/448)⌉ ∈ {e_amax - 9, e_amax - 8} depending on m.
                //
                // Safer than reasoning about ranges: try e = e_amax - 9 and
                // bump by 1 if the resulting scale is too small.
                int e_amax;
                (void)frexpf(amax, &e_amax);
                int e = e_amax - 9;
                float trial = ldexpf(448.0f, e);
                if (trial < amax) {
                    e += 1;
                }
                // Clamp e into the E8M0 storable range:
                //   byte = e + 127 must lie in [1, 254].
                if (e < -126) {
                    e = -126;
                } else if (e > 127) {
                    e = 127;
                }
                byte = (uint8_t)(e + 127);
                // bf16 scale = 2^e. Use ldexpf for an exact power of 2.
                scale_bf16 = __float2bfloat16(ldexpf(1.0f, e));
            }
            // Broadcast scale as float (exact, since scale is a power of 2).
            s_scale_f = __bfloat162float(scale_bf16);

            // Per-tile scale byte (slots [0, NUM_TILES) hold the tile scales).
            token_scales_base[tile] = byte;
            // Padding bytes (slots [NUM_TILES, NUM_SCALES)) — zeroed once at the
            // last tile by lane 0. MODEL1: slot 7 only. V32: slots 8..15.
            if (tile == NUM_TILES - 1) {
                for (int i = NUM_TILES; i < NUM_SCALES; ++i) {
                    token_scales_base[i] = 0;
                }
            }
        }
        __syncthreads();

        if (is_nope) {
            float scale_f = s_scale_f;
            // Quantize: out_fp8 = __nv_fp8_e4m3(v / scale_bf16).
            //
            // The kernel's reverse path is:
            //   f32 = (float)fp8_byte                    [E4M3 → f32]
            //   bf16 = __float2bfloat16_rn(f32) * scale  [scaled bf16]
            // So we want v ≈ (float)fp8_byte * scale_f, hence we encode
            // (v / scale_f) → fp8 and let the hardware rounding handle the
            // E4M3 representable set.
            float quantized = (scale_f != 0.0f) ? (v / scale_f) : 0.0f;
            __nv_fp8_e4m3 fp8_v = __nv_fp8_e4m3(quantized);
            const int dim_idx = tile * QUANT_TILE_SIZE + nope_lane;
            // NoPE bytes are contiguous in token_data_base[0..HEAD_DIM_NOPE).
            token_data_base[dim_idx] = (uint8_t)fp8_v.__x;
        }
    }

    // RoPE copy (after the last NoPE __syncthreads). 64 RoPE threads, one
    // per dim. NoPE threads sit out of this branch.
    if (!is_nope && rope_lane < HEAD_DIM_ROPE) {
        __nv_bfloat16 v = rope[(int64_t)t * stride_rope_elems + rope_lane];
        __nv_bfloat16* rope_base = reinterpret_cast<__nv_bfloat16*>(
            token_data_base + HEAD_DIM_NOPE);
        rope_base[rope_lane] = v;
    }
}

// V32 (GLM-5.2) inline-F32-scale pack kernel.
//
// Per-token INLINE 656-byte layout (stride 656), matching the vendored V32
// decode (`splitkv_mla.cuh` V32 branch):
//   [0, 512)   NoPE fp8 e4m3 (512 bytes)
//   [512, 528) 4× F32 scales — scale[b] = amax(NoPE[b*128:(b+1)*128]) / 448.0
//              stored as plain F32 (NO power-of-two rounding)
//   [528, 656) RoPE bf16 (64 dims)
//
// Block-paged write: block_base = packed_kv + block_id*page_block_size*656;
// token at block_base + row*656.
//
// Thread roles (128 threads / 4 warps): threads [0,64) quantize NoPE (4 blocks
// of 128, processed lane-strided over 128 lanes... — here 64 NoPE threads each
// handle 2 elements per 128-block, 4 blocks in series); threads [64,128) copy
// the 64 RoPE dims. The dequant kernel reads F32 scales directly, so we store
// the F32 amax/448 ratio with no exponent rounding (this is the whole point of
// the fix — MODEL1's e8m0 rounding cost up to ±40% per block).
__global__ void dsv4_v32_fp8_kv_pack_kernel(
    const __nv_bfloat16* __restrict__ nope,   // [n_tokens, stride_nope_elems]
    const __nv_bfloat16* __restrict__ rope,   // [n_tokens, stride_rope_elems]
    uint8_t* __restrict__ packed_kv,
    const int* __restrict__ token_block_id,
    const int* __restrict__ token_in_block_row,
    int n_tokens,
    int page_block_size,
    int stride_nope_elems,
    int stride_rope_elems)
{
    const int t = blockIdx.x;
    if (t >= n_tokens) return;

    const int tid = threadIdx.x;
    const int block_id = token_block_id[t];
    const int row = token_in_block_row[t];

    // Block-paged base (inline 656 B/token, stride 656).
    const int64_t block_stride = (int64_t)page_block_size * V32_TOKEN_BYTES;
    uint8_t* token_base =
        packed_kv + (int64_t)block_id * block_stride + (int64_t)row * V32_TOKEN_BYTES;

    uint8_t* nope_fp8_base   = token_base;                         // [0, 512)
    float*   scales_base     = (float*)(token_base + V32_HEAD_DIM_NOPE); // [512, 528)
    __nv_bfloat16* rope_base =
        (__nv_bfloat16*)(token_base + V32_HEAD_DIM_NOPE + V32_SCALE_BYTES); // [528, 656)

    // Shared scratch: per-128-block reduction across the 64 NoPE threads + the
    // broadcast scale. 64 NoPE threads = 2 warps; combine via 2 shared slots.
    __shared__ float s_warp_max[2];
    __shared__ float s_scale_f;

    const int NOPE_THREADS = 64;
    const bool is_nope = (tid < NOPE_THREADS);
    const int  nope_lane = is_nope ? tid : -1;     // 0..63 for NoPE threads
    const int  rope_lane = (tid >= NOPE_THREADS) ? (tid - NOPE_THREADS) : -1;

    // 4 NoPE blocks of 128 elems each → one F32 scale per block. Each NoPE thread
    // handles 2 elems per block (lane, lane+64).
    #pragma unroll
    for (int blk = 0; blk < V32_NUM_SCALES; ++blk) {
        const int base_dim = blk * V32_QUANT_TILE;
        float v0 = 0.0f, v1 = 0.0f;
        if (is_nope) {
            v0 = __bfloat162float(nope[(int64_t)t * stride_nope_elems + base_dim + nope_lane]);
            v1 = __bfloat162float(nope[(int64_t)t * stride_nope_elems + base_dim + nope_lane + 64]);
        }

        if (is_nope) {
            float a = fmaxf(fabsf(v0), fabsf(v1));
            float warp_max = warp_reduce_max(a);
            if ((nope_lane & 31) == 0) {
                s_warp_max[nope_lane >> 5] = warp_max;
            }
        }
        __syncthreads();

        if (is_nope && nope_lane == 0) {
            float amax = fmaxf(s_warp_max[0], s_warp_max[1]);
            // scale = amax / 448 (E4M3 max), stored as plain F32 (NO pow-2
            // rounding). Guards: zero/non-finite block → scale 0 (dequant reads
            // it back as 0; locally use 1.0 below so val/scale doesn't NaN).
            float scale = (amax > 0.0f && isfinite(amax)) ? (amax / 448.0f) : 0.0f;
            scales_base[blk] = scale;
            s_scale_f = scale;
        }
        __syncthreads();

        if (is_nope) {
            float scale_f = s_scale_f;
            float inv = (scale_f != 0.0f) ? (1.0f / scale_f) : 0.0f;
            __nv_fp8_e4m3 q0 = __nv_fp8_e4m3(v0 * inv);
            __nv_fp8_e4m3 q1 = __nv_fp8_e4m3(v1 * inv);
            nope_fp8_base[base_dim + nope_lane]      = (uint8_t)q0.__x;
            nope_fp8_base[base_dim + nope_lane + 64] = (uint8_t)q1.__x;
        }
    }

    // RoPE copy: 64 bf16 dims, one per RoPE thread.
    if (!is_nope && rope_lane < HEAD_DIM_ROPE) {
        rope_base[rope_lane] = rope[(int64_t)t * stride_rope_elems + rope_lane];
    }
}

__global__ void dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_kernel(
    int* __restrict__ token_block_id,
    int* __restrict__ token_in_block_row,
    const int* __restrict__ start_pos,
    int sliding_window,
    int page_block_size)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int pos = *start_pos;
    const int ring_idx = (sliding_window > 0 && pos >= 0) ? (pos % sliding_window) : 0;
    token_block_id[0] = ring_idx / page_block_size;
    token_in_block_row[0] = ring_idx % page_block_size;
}

__global__ void dsv4_fp8_kv_fill_sw_slots_from_start_pos_kernel(
    int* __restrict__ token_block_id,
    int* __restrict__ token_in_block_row,
    const int* __restrict__ start_pos,
    const int* __restrict__ slot_layer_block_offsets,
    int n_tokens,
    int sliding_window,
    int page_block_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_tokens) return;
    const int pos = start_pos[row];
    const int ring_idx = (sliding_window > 0 && pos >= 0) ? (pos % sliding_window) : 0;
    token_block_id[row] = slot_layer_block_offsets[row] + ring_idx / page_block_size;
    token_in_block_row[row] = ring_idx % page_block_size;
}

__global__ void dsv4_fp8_kv_pack_completed_compressor_row_start_pos_kernel(
    const __nv_bfloat16* __restrict__ compressed,
    uint8_t* __restrict__ packed_kv,
    const int* __restrict__ start_pos,
    int ratio,
    int sw_blocks,
    int page_block_size,
    int stride_elems,
    const int* __restrict__ page_table,
    int num_logical_pages)
{
    // MODEL1-only (DSv4 CSA) — GLM has no compressor so this kernel is never
    // launched on the GLM path. Self-contained MODEL1 constants (NoPE=448).
    constexpr int HEAD_DIM_NOPE    = 448;
    constexpr int NUM_TILES        = HEAD_DIM_NOPE / QUANT_TILE_SIZE;
    constexpr int NUM_SCALES       = 8;                                // 7 used + 1 pad
    constexpr int TOKEN_DATA_BYTES = HEAD_DIM_NOPE + ROPE_BYTES;
    constexpr int TOKEN_BYTES      = TOKEN_DATA_BYTES + NUM_SCALES;

    if (blockIdx.x != 0) return;
    const int pos = *start_pos;
    if (pos < 0 || ratio <= 0 || ((pos + 1) % ratio) != 0) return;
    const int compressed_row = pos / ratio;
    int block_id = sw_blocks + compressed_row / page_block_size;
    // Stage-B: route the slot-LOGICAL block to its physical pool block via the
    // page table (identity table == band path, byte-for-byte). null table = band.
    if (page_table != nullptr) {
        if (block_id >= num_logical_pages) return;
        block_id = page_table[block_id];
    }
    const int row = compressed_row % page_block_size;

    const int tid = threadIdx.x;
    const __nv_bfloat16* nope = compressed + (int64_t)compressed_row * stride_elems;
    const __nv_bfloat16* rope = nope + HEAD_DIM_NOPE;

    const int64_t block_stride = (int64_t)page_block_size * TOKEN_BYTES;
    uint8_t* block_base = packed_kv + (int64_t)block_id * block_stride;
    uint8_t* token_data_base = block_base + (int64_t)row * TOKEN_DATA_BYTES;
    uint8_t* token_scales_base =
        block_base + (int64_t)page_block_size * TOKEN_DATA_BYTES
        + (int64_t)row * NUM_SCALES;

    __shared__ float s_warp_max[2];
    __shared__ float s_scale_f;

    const bool is_nope = (tid < NOPE_THREADS);
    const int nope_lane = is_nope ? tid : -1;
    const int rope_lane = (tid >= NOPE_THREADS) ? (tid - NOPE_THREADS) : -1;

    #pragma unroll
    for (int tile = 0; tile < NUM_TILES; ++tile) {
        float v = 0.0f;
        if (is_nope) {
            const int dim_idx = tile * QUANT_TILE_SIZE + nope_lane;
            v = __bfloat162float(nope[dim_idx]);
        }

        if (is_nope) {
            float a = fabsf(v);
            float warp_max = warp_reduce_max(a);
            if ((nope_lane & 31) == 0) {
                s_warp_max[nope_lane >> 5] = warp_max;
            }
        }
        __syncthreads();

        if (is_nope && nope_lane == 0) {
            float amax = fmaxf(s_warp_max[0], s_warp_max[1]);
            uint8_t byte;
            __nv_bfloat16 scale_bf16;
            if (amax <= 0.0f || !isfinite(amax)) {
                byte = 0;
                scale_bf16 = __float2bfloat16(1.0f);
            } else {
                int e_amax;
                (void)frexpf(amax, &e_amax);
                int e = e_amax - 9;
                float trial = ldexpf(448.0f, e);
                if (trial < amax) {
                    e += 1;
                }
                if (e < -126) {
                    e = -126;
                } else if (e > 127) {
                    e = 127;
                }
                byte = (uint8_t)(e + 127);
                scale_bf16 = __float2bfloat16(ldexpf(1.0f, e));
            }
            s_scale_f = __bfloat162float(scale_bf16);
            token_scales_base[tile] = byte;
            if (tile == NUM_TILES - 1) {
                token_scales_base[NUM_SCALES - 1] = 0;
            }
        }
        __syncthreads();

        if (is_nope) {
            float scale_f = s_scale_f;
            float quantized = (scale_f != 0.0f) ? (v / scale_f) : 0.0f;
            __nv_fp8_e4m3 fp8_v = __nv_fp8_e4m3(quantized);
            const int dim_idx = tile * QUANT_TILE_SIZE + nope_lane;
            token_data_base[dim_idx] = (uint8_t)fp8_v.__x;
        }
    }

    if (!is_nope && rope_lane < HEAD_DIM_ROPE) {
        __nv_bfloat16* rope_base = reinterpret_cast<__nv_bfloat16*>(
            token_data_base + HEAD_DIM_NOPE);
        rope_base[rope_lane] = rope[rope_lane];
    }
}

// Batched (b=N) MODEL1 packs for the batched decode lane.
//
// KERNEL A — batched SW one-token pack. ONE launch (grid = (1, n, 1)) packs each
// slot's CURRENT decode token into its own SW ring slot. `blockIdx.y = row`
// selects the slot; the per-slot inputs are the device-pointer arrays
// `nope_arr[row]`/`rope_arr[row]` (each = that row's `k_prepared` NoPE/RoPE base)
// and the per-slot device page table `page_table_arr[row]`. The ring slot is
// computed ON DEVICE from `start_pos[row]` (folds `dsv4_fp8_kv_fill_one_sw_slot_*`
// inline) so CUDA-graph replay across positions stays correct. The inner
// NoPE-quant + RoPE copy body is BYTE-IDENTICAL to `dsv4_fp8_kv_pack_kernel<448>`
// at token 0 (n=1, row=0 reproduces `flashmla_pack_one_sw_token`). `packed_kv` is
// the single shared pool base (uniform across rows). MODEL1 only (NoPE=448);
// V32/SparseIndexed stay per-row.
__global__ void dsv4_fp8_kv_pack_strided_batched_kernel(
    const __nv_bfloat16* const* __restrict__ nope_arr,
    const __nv_bfloat16* const* __restrict__ rope_arr,
    uint8_t* __restrict__ packed_kv,
    const int* __restrict__ start_pos,
    int n,
    int page_block_size,
    int sliding_window,
    int stride_nope_elems,
    int stride_rope_elems,
    const int* const* __restrict__ page_table_arr,
    int num_logical_pages)
{
    constexpr int HEAD_DIM_NOPE    = 448;
    constexpr int NUM_TILES        = HEAD_DIM_NOPE / QUANT_TILE_SIZE;
    constexpr int NUM_SCALES       = 8;                                // 7 used + 1 pad
    constexpr int TOKEN_DATA_BYTES = HEAD_DIM_NOPE + ROPE_BYTES;
    constexpr int TOKEN_BYTES      = TOKEN_DATA_BYTES + NUM_SCALES;

    const int row = blockIdx.y;
    if (row >= n) return;

    // Ring-slot math folded inline from dsv4_fp8_kv_fill_one_sw_slot_*: the SW
    // pack always writes the CURRENT token (t=0) at ring = pos % sliding_window.
    const int pos = start_pos[row];
    const int ring_idx = (sliding_window > 0 && pos >= 0) ? (pos % sliding_window) : 0;
    int block_id = ring_idx / page_block_size;
    const int rrow = ring_idx % page_block_size;
    // Stage-B: route the slot-LOGICAL page through this row's device table.
    const int* page_table = page_table_arr[row];
    if (page_table != nullptr) {
        if (block_id < 0 || block_id >= num_logical_pages) return;
        block_id = page_table[block_id];
    }

    // Single token per row (t=0): the per-token strides only matter for n_tokens>1
    // inputs; here each row points at its own [1, head_dim] k_prepared base.
    (void)stride_nope_elems;
    (void)stride_rope_elems;
    const __nv_bfloat16* nope = nope_arr[row];
    const __nv_bfloat16* rope = rope_arr[row];

    const int64_t block_stride = (int64_t)page_block_size * TOKEN_BYTES;
    uint8_t* block_base = packed_kv + (int64_t)block_id * block_stride;
    uint8_t* token_data_base = block_base + (int64_t)rrow * TOKEN_DATA_BYTES;
    uint8_t* token_scales_base =
        block_base + (int64_t)page_block_size * TOKEN_DATA_BYTES
        + (int64_t)rrow * NUM_SCALES;

    __shared__ float s_warp_max[2];
    __shared__ float s_scale_f;

    const int tid = threadIdx.x;
    const bool is_nope = (tid < NOPE_THREADS);
    const int nope_lane = is_nope ? tid : -1;
    const int rope_lane = (tid >= NOPE_THREADS) ? (tid - NOPE_THREADS) : -1;

    // Byte-identical body of dsv4_fp8_kv_pack_kernel<448> at t=0.
    #pragma unroll
    for (int tile = 0; tile < NUM_TILES; ++tile) {
        float v = 0.0f;
        if (is_nope) {
            const int dim_idx = tile * QUANT_TILE_SIZE + nope_lane;
            v = __bfloat162float(nope[dim_idx]);
        }
        if (is_nope) {
            float a = fabsf(v);
            float warp_max = warp_reduce_max(a);
            if ((nope_lane & 31) == 0) {
                s_warp_max[nope_lane >> 5] = warp_max;
            }
        }
        __syncthreads();
        if (is_nope && nope_lane == 0) {
            float amax = fmaxf(s_warp_max[0], s_warp_max[1]);
            uint8_t byte;
            __nv_bfloat16 scale_bf16;
            if (amax <= 0.0f || !isfinite(amax)) {
                byte = 0;
                scale_bf16 = __float2bfloat16(1.0f);
            } else {
                int e_amax;
                (void)frexpf(amax, &e_amax);
                int e = e_amax - 9;
                float trial = ldexpf(448.0f, e);
                if (trial < amax) {
                    e += 1;
                }
                if (e < -126) {
                    e = -126;
                } else if (e > 127) {
                    e = 127;
                }
                byte = (uint8_t)(e + 127);
                scale_bf16 = __float2bfloat16(ldexpf(1.0f, e));
            }
            s_scale_f = __bfloat162float(scale_bf16);
            token_scales_base[tile] = byte;
            if (tile == NUM_TILES - 1) {
                for (int i = NUM_TILES; i < NUM_SCALES; ++i) {
                    token_scales_base[i] = 0;
                }
            }
        }
        __syncthreads();
        if (is_nope) {
            float scale_f = s_scale_f;
            float quantized = (scale_f != 0.0f) ? (v / scale_f) : 0.0f;
            __nv_fp8_e4m3 fp8_v = __nv_fp8_e4m3(quantized);
            const int dim_idx = tile * QUANT_TILE_SIZE + nope_lane;
            token_data_base[dim_idx] = (uint8_t)fp8_v.__x;
        }
    }
    if (!is_nope && rope_lane < HEAD_DIM_ROPE) {
        __nv_bfloat16* rope_base = reinterpret_cast<__nv_bfloat16*>(
            token_data_base + HEAD_DIM_NOPE);
        rope_base[rope_lane] = rope[rope_lane];
    }
}

// KERNEL B — batched compressed-delta pack. ONE launch (grid = (1, n, 1)). Each
// slot packs AT MOST one just-completed compressor row this step; the internal
// `(pos+1)%ratio != 0` early-out makes non-completing rows (and rows whose slot
// has no compressor, signalled by `compressed_arr[row] == nullptr`) no-op, BYTE-
// IDENTICAL to the per-row `flashmla_pack_compressed_delta` None/no-op. The body
// is the verbatim `dsv4_fp8_kv_pack_completed_compressor_row_start_pos_kernel`
// with per-row addressing (`compressed_arr[row]`, `start_pos[row]`,
// `page_table_arr[row]`). MODEL1 (CSA/HCA) only.
__global__ void dsv4_fp8_kv_pack_completed_compressor_row_batched_kernel(
    const __nv_bfloat16* const* __restrict__ compressed_arr,
    uint8_t* __restrict__ packed_kv,
    const int* __restrict__ start_pos,
    int n,
    int ratio,
    int sw_blocks,
    int page_block_size,
    int stride_elems,
    const int* const* __restrict__ page_table_arr,
    int num_logical_pages)
{
    constexpr int HEAD_DIM_NOPE    = 448;
    constexpr int NUM_TILES        = HEAD_DIM_NOPE / QUANT_TILE_SIZE;
    constexpr int NUM_SCALES       = 8;                                // 7 used + 1 pad
    constexpr int TOKEN_DATA_BYTES = HEAD_DIM_NOPE + ROPE_BYTES;
    constexpr int TOKEN_BYTES      = TOKEN_DATA_BYTES + NUM_SCALES;

    const int row = blockIdx.y;
    if (row >= n) return;
    // Rows whose slot has no compressor (SparseIndexed / want_compressed=false)
    // pass a null base → no-op (== the per-row `Some(compressed)` else None gate).
    const __nv_bfloat16* compressed = compressed_arr[row];
    if (compressed == nullptr) return;

    const int pos = start_pos[row];
    if (pos < 0 || ratio <= 0 || ((pos + 1) % ratio) != 0) return;
    const int compressed_row = pos / ratio;
    int block_id = sw_blocks + compressed_row / page_block_size;
    const int* page_table = page_table_arr[row];
    if (page_table != nullptr) {
        if (block_id >= num_logical_pages) return;
        block_id = page_table[block_id];
    }
    const int rrow = compressed_row % page_block_size;

    const int tid = threadIdx.x;
    const __nv_bfloat16* nope = compressed + (int64_t)compressed_row * stride_elems;
    const __nv_bfloat16* rope = nope + HEAD_DIM_NOPE;

    const int64_t block_stride = (int64_t)page_block_size * TOKEN_BYTES;
    uint8_t* block_base = packed_kv + (int64_t)block_id * block_stride;
    uint8_t* token_data_base = block_base + (int64_t)rrow * TOKEN_DATA_BYTES;
    uint8_t* token_scales_base =
        block_base + (int64_t)page_block_size * TOKEN_DATA_BYTES
        + (int64_t)rrow * NUM_SCALES;

    __shared__ float s_warp_max[2];
    __shared__ float s_scale_f;

    const bool is_nope = (tid < NOPE_THREADS);
    const int nope_lane = is_nope ? tid : -1;
    const int rope_lane = (tid >= NOPE_THREADS) ? (tid - NOPE_THREADS) : -1;

    #pragma unroll
    for (int tile = 0; tile < NUM_TILES; ++tile) {
        float v = 0.0f;
        if (is_nope) {
            const int dim_idx = tile * QUANT_TILE_SIZE + nope_lane;
            v = __bfloat162float(nope[dim_idx]);
        }
        if (is_nope) {
            float a = fabsf(v);
            float warp_max = warp_reduce_max(a);
            if ((nope_lane & 31) == 0) {
                s_warp_max[nope_lane >> 5] = warp_max;
            }
        }
        __syncthreads();
        if (is_nope && nope_lane == 0) {
            float amax = fmaxf(s_warp_max[0], s_warp_max[1]);
            uint8_t byte;
            __nv_bfloat16 scale_bf16;
            if (amax <= 0.0f || !isfinite(amax)) {
                byte = 0;
                scale_bf16 = __float2bfloat16(1.0f);
            } else {
                int e_amax;
                (void)frexpf(amax, &e_amax);
                int e = e_amax - 9;
                float trial = ldexpf(448.0f, e);
                if (trial < amax) {
                    e += 1;
                }
                if (e < -126) {
                    e = -126;
                } else if (e > 127) {
                    e = 127;
                }
                byte = (uint8_t)(e + 127);
                scale_bf16 = __float2bfloat16(ldexpf(1.0f, e));
            }
            s_scale_f = __bfloat162float(scale_bf16);
            token_scales_base[tile] = byte;
            if (tile == NUM_TILES - 1) {
                token_scales_base[NUM_SCALES - 1] = 0;
            }
        }
        __syncthreads();
        if (is_nope) {
            float scale_f = s_scale_f;
            float quantized = (scale_f != 0.0f) ? (v / scale_f) : 0.0f;
            __nv_fp8_e4m3 fp8_v = __nv_fp8_e4m3(quantized);
            const int dim_idx = tile * QUANT_TILE_SIZE + nope_lane;
            token_data_base[dim_idx] = (uint8_t)fp8_v.__x;
        }
    }
    if (!is_nope && rope_lane < HEAD_DIM_ROPE) {
        __nv_bfloat16* rope_base = reinterpret_cast<__nv_bfloat16*>(
            token_data_base + HEAD_DIM_NOPE);
        rope_base[rope_lane] = rope[rope_lane];
    }
}

} // namespace

// Contiguous-input variant: nope and rope are tightly-packed
// [n_tokens, HEAD_DIM_NOPE] / [n_tokens, HEAD_DIM_ROPE] bf16 buffers.
// Kept as a thin wrapper around the strided implementation so callers
// stay single-sourced on one kernel body.
extern "C" cudaError_t arle_dsv4_fp8_kv_pack_cuda(
    const __nv_bfloat16* nope,
    const __nv_bfloat16* rope,
    uint8_t* packed_kv,
    const int* token_block_id,
    const int* token_in_block_row,
    int n_tokens,
    int page_block_size,
    cudaStream_t stream)
{
    if (n_tokens == 0) return cudaSuccess;
    if (page_block_size <= 0) return cudaErrorInvalidValue;
    if (nope == nullptr || rope == nullptr || packed_kv == nullptr
        || token_block_id == nullptr || token_in_block_row == nullptr) {
        return cudaErrorInvalidValue;
    }

    dim3 grid((unsigned)n_tokens, 1, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    // MODEL1 contiguous pack: NoPE=448. Strides equal the dims (tight packing).
    // Stage-A only (no page table) — band addressing via host-physical block ids.
    dsv4_fp8_kv_pack_kernel<448><<<grid, block, 0, stream>>>(
        nope, rope, packed_kv,
        token_block_id, token_in_block_row,
        n_tokens, page_block_size,
        448, HEAD_DIM_ROPE,
        nullptr, 0);
    return cudaGetLastError();
}

// Strided-input variant for the runtime decode hooks: NoPE and RoPE may
// share a single `k_prepared`-style [n_tokens, head_dim=512] buffer where
// the caller passes (nope_ptr = k_prepared, rope_ptr = k_prepared+NOPE,
// stride_nope_elems = stride_rope_elems = 512) — see Phase D-4 SOLID
// survey Finding 1 in docs/experience/wins/2026-05-28-dsv4-flashmla-decode-d4-plumbing.md.
//
// Both strides must be ≥ HEAD_DIM_NOPE (448) / HEAD_DIM_ROPE (64) respectively.
// `page_table`/`num_logical_pages` are the OPTIONAL Stage-B lookup (pass
// nullptr/0 for the Stage-A band path). When set, `token_block_id[t]` is a
// slot-LOGICAL page indexed through `page_table`; identity table = byte-identical.
extern "C" cudaError_t arle_dsv4_fp8_kv_pack_strided_cuda(
    const __nv_bfloat16* nope,
    const __nv_bfloat16* rope,
    uint8_t* packed_kv,
    const int* token_block_id,
    const int* token_in_block_row,
    int n_tokens,
    int page_block_size,
    int stride_nope_elems,
    int stride_rope_elems,
    const int* page_table,
    int num_logical_pages,
    cudaStream_t stream)
{
    if (n_tokens == 0) return cudaSuccess;
    if (page_block_size <= 0) return cudaErrorInvalidValue;
    if (stride_nope_elems < 448 || stride_rope_elems < HEAD_DIM_ROPE) {
        return cudaErrorInvalidValue;
    }
    if (nope == nullptr || rope == nullptr || packed_kv == nullptr
        || token_block_id == nullptr || token_in_block_row == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (page_table != nullptr && num_logical_pages <= 0) {
        return cudaErrorInvalidValue;
    }

    dim3 grid((unsigned)n_tokens, 1, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    // MODEL1 strided pack: NoPE=448 (explicit instantiation).
    dsv4_fp8_kv_pack_kernel<448><<<grid, block, 0, stream>>>(
        nope, rope, packed_kv,
        token_block_id, token_in_block_row,
        n_tokens, page_block_size,
        stride_nope_elems, stride_rope_elems,
        page_table, num_logical_pages);
    return cudaGetLastError();
}

// V32 (GLM-5.2) strided pack: NoPE=512 / 656 B/tok INLINE layout. Launches the
// SEPARATE `dsv4_v32_fp8_kv_pack_kernel` (NOT the MODEL1 template): per token
// [512 NoPE fp8][4 F32 scales @512][128 rope bf16], stride 656, with F32
// (non-pow2) block scales = amax(128-block)/448. The k_prepared-style caller
// passes (nope=k_prepared, rope=k_prepared+512, stride_nope=stride_rope=head_dim=576).
extern "C" cudaError_t arle_dsv4_v32_fp8_kv_pack_strided_cuda(
    const __nv_bfloat16* nope,
    const __nv_bfloat16* rope,
    uint8_t* packed_kv,
    const int* token_block_id,
    const int* token_in_block_row,
    int n_tokens,
    int page_block_size,
    int stride_nope_elems,
    int stride_rope_elems,
    cudaStream_t stream)
{
    if (n_tokens == 0) return cudaSuccess;
    if (page_block_size <= 0) return cudaErrorInvalidValue;
    if (stride_nope_elems < V32_HEAD_DIM_NOPE || stride_rope_elems < HEAD_DIM_ROPE) {
        return cudaErrorInvalidValue;
    }
    if (nope == nullptr || rope == nullptr || packed_kv == nullptr
        || token_block_id == nullptr || token_in_block_row == nullptr) {
        return cudaErrorInvalidValue;
    }

    dim3 grid((unsigned)n_tokens, 1, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dsv4_v32_fp8_kv_pack_kernel<<<grid, block, 0, stream>>>(
        nope, rope, packed_kv,
        token_block_id, token_in_block_row,
        n_tokens, page_block_size,
        stride_nope_elems, stride_rope_elems);
    return cudaGetLastError();
}

extern "C" cudaError_t arle_dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_cuda(
    int* token_block_id,
    int* token_in_block_row,
    const int* start_pos,
    int sliding_window,
    int page_block_size,
    cudaStream_t stream)
{
    if (token_block_id == nullptr || token_in_block_row == nullptr || start_pos == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (sliding_window <= 0 || page_block_size <= 0) {
        return cudaErrorInvalidValue;
    }
    dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_kernel<<<1, 1, 0, stream>>>(
        token_block_id, token_in_block_row, start_pos, sliding_window, page_block_size);
    return cudaGetLastError();
}

extern "C" cudaError_t arle_dsv4_fp8_kv_fill_sw_slots_from_start_pos_cuda(
    int* token_block_id,
    int* token_in_block_row,
    const int* start_pos,
    const int* slot_layer_block_offsets,
    int n_tokens,
    int sliding_window,
    int page_block_size,
    cudaStream_t stream)
{
    if (n_tokens == 0) return cudaSuccess;
    if (token_block_id == nullptr || token_in_block_row == nullptr ||
        start_pos == nullptr || slot_layer_block_offsets == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (n_tokens < 0 || sliding_window <= 0 || page_block_size <= 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kBlock = 128;
    int grid = (n_tokens + kBlock - 1) / kBlock;
    dsv4_fp8_kv_fill_sw_slots_from_start_pos_kernel<<<grid, kBlock, 0, stream>>>(
        token_block_id, token_in_block_row, start_pos, slot_layer_block_offsets,
        n_tokens, sliding_window, page_block_size);
    return cudaGetLastError();
}

extern "C" cudaError_t arle_dsv4_fp8_kv_pack_completed_compressor_row_start_pos_cuda(
    const __nv_bfloat16* compressed,
    uint8_t* packed_kv,
    const int* start_pos,
    int ratio,
    int sw_blocks,
    int page_block_size,
    int stride_elems,
    const int* page_table,
    int num_logical_pages,
    cudaStream_t stream)
{
    if (compressed == nullptr || packed_kv == nullptr || start_pos == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (ratio <= 0 || sw_blocks < 0 || page_block_size <= 0 || stride_elems < 448 + HEAD_DIM_ROPE) {
        return cudaErrorInvalidValue;
    }
    if (page_table != nullptr && num_logical_pages <= 0) {
        return cudaErrorInvalidValue;
    }
    dsv4_fp8_kv_pack_completed_compressor_row_start_pos_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
        compressed, packed_kv, start_pos, ratio, sw_blocks, page_block_size, stride_elems, page_table, num_logical_pages);
    return cudaGetLastError();
}

// Batched (b=N) MODEL1 SW one-token pack. One launch, grid=(1,n,1), block=128.
// `nope_arr`/`rope_arr` are N device pointers (each = a row's k_prepared NoPE /
// RoPE base); `page_table_arr` is N device page-table pointers (one per slot,
// each `num_logical_pages` long); `packed_kv` is the single shared pool base;
// `start_pos` is the contiguous `[N]` decode positions. n=1 == the per-row
// `arle_dsv4_fp8_kv_pack_strided_cuda(n_tokens=1)` + fill (byte-identical).
extern "C" cudaError_t arle_dsv4_fp8_kv_pack_strided_batched_cuda(
    const __nv_bfloat16* const* nope_arr,
    const __nv_bfloat16* const* rope_arr,
    uint8_t* packed_kv,
    const int* start_pos,
    int n,
    int page_block_size,
    int sliding_window,
    int stride_nope_elems,
    int stride_rope_elems,
    const int* const* page_table_arr,
    int num_logical_pages,
    cudaStream_t stream)
{
    if (n == 0) return cudaSuccess;
    if (n < 0 || page_block_size <= 0 || sliding_window <= 0) return cudaErrorInvalidValue;
    if (stride_nope_elems < 448 || stride_rope_elems < HEAD_DIM_ROPE) {
        return cudaErrorInvalidValue;
    }
    if (nope_arr == nullptr || rope_arr == nullptr || packed_kv == nullptr
        || start_pos == nullptr || page_table_arr == nullptr) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(1, (unsigned)n, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dsv4_fp8_kv_pack_strided_batched_kernel<<<grid, block, 0, stream>>>(
        nope_arr, rope_arr, packed_kv, start_pos, n,
        page_block_size, sliding_window, stride_nope_elems, stride_rope_elems,
        page_table_arr, num_logical_pages);
    return cudaGetLastError();
}

// Batched (b=N) MODEL1 compressed-delta pack. One launch, grid=(1,n,1),
// block=128. `compressed_arr[row]` is a row's compressor `compressed` base
// (nullptr for rows with no compressor → no-op); the kernel early-outs per row
// on `(pos+1)%ratio != 0`, so non-completing rows write nothing. n=1 ==
// `arle_dsv4_fp8_kv_pack_completed_compressor_row_start_pos_cuda` (byte-identical).
extern "C" cudaError_t arle_dsv4_fp8_kv_pack_completed_compressor_row_batched_cuda(
    const __nv_bfloat16* const* compressed_arr,
    uint8_t* packed_kv,
    const int* start_pos,
    int n,
    int ratio,
    int sw_blocks,
    int page_block_size,
    int stride_elems,
    const int* const* page_table_arr,
    int num_logical_pages,
    cudaStream_t stream)
{
    if (n == 0) return cudaSuccess;
    if (n < 0 || ratio <= 0 || sw_blocks < 0 || page_block_size <= 0
        || stride_elems < 448 + HEAD_DIM_ROPE) {
        return cudaErrorInvalidValue;
    }
    if (compressed_arr == nullptr || packed_kv == nullptr || start_pos == nullptr
        || page_table_arr == nullptr) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(1, (unsigned)n, 1);
    dsv4_fp8_kv_pack_completed_compressor_row_batched_kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
        compressed_arr, packed_kv, start_pos, n, ratio, sw_blocks,
        page_block_size, stride_elems, page_table_arr, num_logical_pages);
    return cudaGetLastError();
}

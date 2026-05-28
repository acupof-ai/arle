// DSv4-Flash (MODEL1) FP8 KV pack kernel.
//
// Packs ARLE's bf16 DSv4 KV (NoPE 448 + RoPE 64) into the **MODEL1 FP8
// block-paged layout** consumed by upstream FlashMLA's
// `sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`.
//
// === Contract (evidence-anchored from upstream source) ============================
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
// =================================================================================

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {

// MODEL1 constants — DO NOT EDIT without also updating the upstream contract
// anchors above.
static constexpr int HEAD_DIM_NOPE   = 448;
static constexpr int HEAD_DIM_ROPE   = 64;
static constexpr int QUANT_TILE_SIZE = 64;
static constexpr int NUM_TILES       = HEAD_DIM_NOPE / QUANT_TILE_SIZE;   // 7
static constexpr int NUM_SCALES      = 8;                                 // 7 used + 1 pad
static constexpr int ROPE_BYTES      = HEAD_DIM_ROPE * sizeof(__nv_bfloat16); // 128
static constexpr int TOKEN_DATA_BYTES = HEAD_DIM_NOPE + ROPE_BYTES;       // 576
static constexpr int TOKEN_BYTES      = TOKEN_DATA_BYTES + NUM_SCALES;    // 584 (per-token stride)

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
__global__ void dsv4_fp8_kv_pack_kernel(
    const __nv_bfloat16* __restrict__ nope,   // [n_tokens, HEAD_DIM_NOPE]
    const __nv_bfloat16* __restrict__ rope,   // [n_tokens, HEAD_DIM_ROPE]
    uint8_t* __restrict__ packed_kv,
    const int* __restrict__ token_block_id,
    const int* __restrict__ token_in_block_row,
    int n_tokens,
    int page_block_size)
{
    const int t = blockIdx.x;
    if (t >= n_tokens) return;

    const int tid = threadIdx.x;
    const int block_id = token_block_id[t];
    const int row = token_in_block_row[t];

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
            v = __bfloat162float(nope[(int64_t)t * HEAD_DIM_NOPE + dim_idx]);
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

            // Per-tile scale byte (one of 7 used slots).
            token_scales_base[tile] = byte;
            // Padding byte (slot 7) — written once at the last tile.
            if (tile == NUM_TILES - 1) {
                token_scales_base[NUM_SCALES - 1] = 0;
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
            // NoPE bytes are contiguous in token_data_base[0..448).
            token_data_base[dim_idx] = (uint8_t)fp8_v.__x;
        }
    }

    // RoPE copy (after the last NoPE __syncthreads). 64 RoPE threads, one
    // per dim. NoPE threads sit out of this branch.
    if (!is_nope && rope_lane < HEAD_DIM_ROPE) {
        __nv_bfloat16 v = rope[(int64_t)t * HEAD_DIM_ROPE + rope_lane];
        __nv_bfloat16* rope_base = reinterpret_cast<__nv_bfloat16*>(
            token_data_base + HEAD_DIM_NOPE);
        rope_base[rope_lane] = v;
    }
}

} // namespace

// ===== Public C entry =====
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
    dsv4_fp8_kv_pack_kernel<<<grid, block, 0, stream>>>(
        nope, rope, packed_kv,
        token_block_id, token_in_block_row,
        n_tokens, page_block_size);
    return cudaGetLastError();
}

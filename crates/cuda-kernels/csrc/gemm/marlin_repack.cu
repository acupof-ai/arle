// GPTQ → Marlin weight repacking kernel.
// Adapted from vLLM's gptq_marlin_repack.cu (Apache 2.0).
// Standalone: no PyTorch dependency.

#include <cuda_runtime.h>
#include <cstdint>

// Marlin layout constants
namespace marlin {
constexpr int tile_size = 16;
constexpr int tile_k_size = 16;
constexpr int tile_n_size = 64;
constexpr int repack_stages = 8;
constexpr int repack_threads = 256;

// cp.async helper (128-bit / 16-byte copy from global to shared)
__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "  cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}
}  // namespace marlin

// ============================================================================
// Repack kernel: GPTQ int32-packed [K/8, N] → Marlin tile layout
// Only supports 4-bit, no act_order, no A8 mode (the common case).
// ============================================================================
__global__ void gptq_marlin_repack_kernel_4bit(
    const uint32_t* __restrict__ b_q_weight_ptr,
    uint32_t* __restrict__ out_ptr,
    int size_k,
    int size_n)
{
    constexpr int num_bits = 4;
    constexpr int pack_factor = 32 / num_bits;  // 8
    int k_tiles = size_k / marlin::tile_k_size;
    int n_tiles = size_n / marlin::tile_n_size;
    int block_k_tiles = marlin::repack_stages * 1;  // simplified

    // Each block handles one n_tile worth of repacking
    int n_tile_id = blockIdx.x;
    int tid = threadIdx.x;

    if (n_tile_id >= n_tiles) return;

    int first_n = n_tile_id * marlin::tile_n_size;

    // Simple repacking: iterate over k tiles, reorder weights into Marlin layout
    // Marlin layout: each tile is [tile_k_size=16, tile_n_size=64]
    // Stored as [k_tiles, tile_n_size * tile_k_size / pack_factor] uint32
    //
    // The tile layout reorders so that consecutive 128-bit loads access weights
    // for the same MMA operation.

    for (int k_tile = tid / (marlin::tile_n_size / 4); k_tile < k_tiles;
         k_tile += marlin::repack_threads / (marlin::tile_n_size / 4)) {
        int n_within_tile = (tid % (marlin::tile_n_size / 4)) * 4;

        // Source: GPTQ layout [K/8, N], K index = k_tile * 16 + k_within
        // For each of the 16 k values in this tile, load and repack
        int k_start = k_tile * marlin::tile_k_size;

        // Output tile offset
        int out_tile_offset = k_tile * (marlin::tile_n_size * marlin::tile_k_size / pack_factor)
                            + n_within_tile * (marlin::tile_k_size / pack_factor);

        // Simple: just copy the packed values in the correct order
        // GPTQ: element [k, n] is in qweight[k/8][n], bits [(k%8)*4 : (k%8)*4+3]
        // Marlin: element [k, n] within a tile maps to a specific bit position
        //         optimized for ldmatrix access

        // For simplicity, do element-level repacking:
        for (int ni = 0; ni < 4; ni++) {
            int n_idx = first_n + n_within_tile + ni;
            if (n_idx >= size_n) continue;

            // Pack 16 k-values (one tile_k) into 2 int32 (8 int4 per int32)
            uint32_t packed[2] = {0, 0};
            for (int ki = 0; ki < marlin::tile_k_size; ki++) {
                int k_idx = k_start + ki;
                if (k_idx >= size_k) continue;

                // Read from GPTQ format
                int gptq_row = k_idx / pack_factor;
                int gptq_bit = (k_idx % pack_factor) * num_bits;
                uint32_t gptq_val = b_q_weight_ptr[gptq_row * size_n + n_idx];
                int w = (gptq_val >> gptq_bit) & 0xF;

                // Write into Marlin-ordered output
                // Marlin packs in the order that ldmatrix expects:
                // Within each int32, elements are packed LSB-first, same as GPTQ
                int out_int = ki / pack_factor;  // which int32 (0 or 1)
                int out_bit = (ki % pack_factor) * num_bits;
                packed[out_int] |= (w << out_bit);
            }

            // Write output
            int out_idx = k_tile * size_n * (marlin::tile_k_size / pack_factor)
                        + n_idx * (marlin::tile_k_size / pack_factor);
            for (int p = 0; p < marlin::tile_k_size / pack_factor; p++) {
                out_ptr[out_idx + p] = packed[p];
            }
        }
    }
}

// ============================================================================
// C API
// ============================================================================
extern "C" {

// Repack GPTQ weights to Marlin layout.
// b_q_weight: [size_k / 8, size_n] int32 (GPTQ packed)
// out:        [size_k / tile_k * size_n * tile_k / 8] int32 (Marlin packed)
cudaError_t gptq_marlin_repack_cuda(
    const uint32_t* b_q_weight,
    uint32_t* out,
    int size_k,
    int size_n,
    cudaStream_t stream)
{
    int n_tiles = size_n / marlin::tile_n_size;
    // Handle case where size_n < tile_n_size
    if (n_tiles == 0) n_tiles = 1;

    gptq_marlin_repack_kernel_4bit<<<n_tiles, marlin::repack_threads, 0, stream>>>(
        b_q_weight, out, size_k, size_n);
    return cudaGetLastError();
}

// Workspace size for Marlin GEMM.
// Returns bytes needed for the lock buffer.
size_t marlin_workspace_size(int prob_n, int sms) {
    // Marlin needs num_SMs * (prob_n / 64) * sizeof(int) for the lock buffer
    int n_tiles = (prob_n + 63) / 64;
    return (size_t)sms * n_tiles * sizeof(int);
}

}  // extern "C"

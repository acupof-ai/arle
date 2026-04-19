// TurboQuant Fast: Hadamard-based rotation for O(D log D) quantize/dequantize.
//
// Replaces the naive O(D²) matmul rotation with a randomized Hadamard transform:
//   1. Element-wise sign flip: y[i] = x[i] * sign[i], sign ∈ {-1, +1}
//   2. Fast Walsh-Hadamard Transform (butterfly): log2(D) rounds
//   3. Scale by 1/√D
//
// For D=128: 7 butterfly rounds × 128 = 896 FMAs (vs 16,384 for full matmul).
// Provably achieves the same near-isotropic distribution for quantization.
//
// Reference: Ailon & Chazelle, "The Fast Johnson-Lindenstrauss Transform" (2009)

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// ─── In-place Fast Walsh-Hadamard Transform in shared memory ───
//
// D must be a power of 2. Each thread handles one element.
// After FWHT, output is scaled by 1/√D for normalization.
__device__ __forceinline__ void fwht_inplace(float* smem, int D, int tid) {
    for (int stride = 1; stride < D; stride <<= 1) {
        __syncthreads();
        int pair = tid ^ stride;
        if (pair > tid && pair < D) {
            float a = smem[tid];
            float b = smem[pair];
            smem[tid] = a + b;
            smem[pair] = a - b;
        }
    }
    __syncthreads();
    // Normalize
    smem[tid] *= rsqrtf((float)D);
    __syncthreads();
}

// ─── Inverse FWHT (same as forward — FWHT is its own inverse up to scaling) ───
__device__ __forceinline__ void ifwht_inplace(float* smem, int D, int tid) {
    // FWHT is self-inverse: H * H = D * I, so iH = H / D.
    // We already have 1/√D from forward, so applying again gives 1/D * D = identity.
    // Just apply FWHT again (with same 1/√D normalization).
    fwht_inplace(smem, D, tid);
}

// ─── Warp reduction helpers ───

__device__ __forceinline__ float warp_reduce_sum_fast(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ─── Cooperative bitpack without atomics ───
//
// Each thread computes its byte contribution and uses warp shuffle to
// collect bits from neighboring threads into complete bytes.
// For effective_bits=4 (covers bits=3,4): 2 indices per byte.
// For effective_bits=2: 4 indices per byte.
__device__ __forceinline__ uint8_t cooperative_pack(
    int idx, int d, int effective_bits, int indices_per_byte)
{
    uint8_t my_contrib = (uint8_t)(idx & ((1 << effective_bits) - 1));

    // Which byte does this thread contribute to?
    int byte_idx = d / indices_per_byte;
    int sub_idx = d % indices_per_byte;

    // Shift my contribution to the correct position within the byte.
    my_contrib <<= (sub_idx * effective_bits);

    // Collect contributions from all threads that share the same byte_idx.
    // Use warp shuffle to OR contributions together.
    uint8_t result = my_contrib;
    for (int i = 1; i < indices_per_byte; i++) {
        int src = byte_idx * indices_per_byte + i;
        // Only threads within the same warp can shuffle.
        // For cross-warp, we fall back to shared memory.
        if (src / 32 == d / 32) {
            uint32_t other = __shfl_sync(0xffffffff, (uint32_t)my_contrib, src % 32);
            // But we need the OTHER thread's contribution at position i, not ours.
            // This is tricky — let's use shared memory for simplicity.
        }
    }
    // Fallback: return just our contribution, let shared memory handle aggregation.
    return my_contrib;
}

// ============================================================================
// Kernel: TurboQuant Fast Quantize (Hadamard-based)
//
// Grid:  (num_kv_heads, batch_size)
// Block: (head_dim)  — D threads, one per coordinate
// ============================================================================
__global__ void turboquant_fast_quantize_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,
    uint8_t* __restrict__ packed_out,
    __half* __restrict__ norms_out,
    const int8_t* __restrict__ signs,       // [D] random signs {-1, +1} per layer
    const float* __restrict__ boundaries,   // [num_levels + 1]
    int head_dim,
    int kv_dim,
    int packed_per_head,
    int num_levels,
    int bits)
{
    int kv_head = blockIdx.x;
    int batch_idx = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    // ─── Load ───
    float x = __bfloat162float(kv_bf16[batch_idx * kv_dim + kv_head * head_dim + d]);

    // ─── Norm computation ───
    extern __shared__ float smem[];
    float sq = x * x;
    sq = warp_reduce_sum_fast(sq);
    int warp_id = d / 32, lane_id = d % 32;
    int num_warps = head_dim / 32;
    if (lane_id == 0) smem[head_dim + warp_id] = sq;
    __syncthreads();

    if (d == 0) {
        float total = 0.0f;
        for (int w = 0; w < num_warps; w++) total += smem[head_dim + w];
        smem[head_dim] = total;
    }
    __syncthreads();

    float norm = sqrtf(smem[head_dim]);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    if (d == 0) norms_out[batch_idx * gridDim.x + kv_head] = __float2half(norm);

    // ─── Normalize + sign flip ───
    float y = x * inv_norm * (float)signs[d];

    // ─── Fast Walsh-Hadamard Transform (in-place via shared memory) ───
    smem[d] = y;
    fwht_inplace(smem, head_dim, d);
    y = smem[d];

    // ─── Searchsorted ───
    int idx = 0;
    // Branchless for 3-bit (8 levels): just 7 comparisons
    for (int k = 1; k < num_levels; k++) {
        idx += (y >= boundaries[k]) ? 1 : 0;
    }

    // ─── Bitpack via shared memory (no atomics) ───
    int effective_bits = (bits == 3) ? 4 : bits;
    int indices_per_byte = 8 / effective_bits;
    int byte_idx = d / indices_per_byte;
    int sub_idx = d % indices_per_byte;

    // Each thread writes its shifted contribution; threads sharing a byte
    // are in the same warp (for D≤256, indices_per_byte≤4), so we use shuffle.
    uint32_t my_val = (uint32_t)(idx & ((1 << effective_bits) - 1)) << (sub_idx * effective_bits);

    // Collect contributions within each byte group using warp shuffle OR.
    uint32_t packed = my_val;
    int base = byte_idx * indices_per_byte;
    for (int i = 0; i < indices_per_byte; i++) {
        int src_lane = (base + i) % 32;
        int src_warp = (base + i) / 32;
        if (src_warp == warp_id) {
            uint32_t other = __shfl_sync(0xffffffff, my_val, src_lane);
            packed |= other;
        }
    }

    // Only the first thread in each byte group writes the packed byte.
    uint8_t* s_packed = (uint8_t*)&smem[0];
    if (sub_idx == 0) {
        s_packed[byte_idx] = (uint8_t)(packed & 0xFF);
    }
    __syncthreads();

    // Coalesced store to global memory.
    int dst = batch_idx * (gridDim.x * packed_per_head) + kv_head * packed_per_head;
    if (d < packed_per_head) {
        packed_out[dst + d] = s_packed[d];
    }
}

// ============================================================================
// Kernel: TurboQuant Fast Dequantize (Hadamard-based)
//
// Grid:  (num_kv_heads, token_count)
// Block: (head_dim)
// ============================================================================
__global__ void turboquant_fast_dequantize_kernel(
    const uint8_t* __restrict__ packed_in,
    const __half* __restrict__ norms_in,
    __nv_bfloat16* __restrict__ kv_bf16,
    const int8_t* __restrict__ signs,
    const float* __restrict__ centroids,
    int head_dim,
    int kv_dim,
    int packed_per_head,
    int num_kv_heads,
    int num_levels,
    int bits)
{
    int kv_head = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    // ─── Unpack ───
    int effective_bits = (bits == 3) ? 4 : bits;
    int indices_per_byte = 8 / effective_bits;
    int byte_idx = d / indices_per_byte;
    int sub_idx = d % indices_per_byte;

    int src = token * (gridDim.x * packed_per_head) + kv_head * packed_per_head;
    uint8_t packed_byte = packed_in[src + byte_idx];
    int idx = (packed_byte >> (sub_idx * effective_bits)) & ((1 << effective_bits) - 1);
    if (idx >= num_levels) idx = num_levels - 1;

    // ─── Gather centroid ───
    float y = centroids[idx];

    // ─── Inverse FWHT + sign flip ───
    extern __shared__ float smem[];
    smem[d] = y;
    ifwht_inplace(smem, head_dim, d);
    float x_hat = smem[d] * (float)signs[d];

    // ─── Scale by norm ───
    float norm = __half2float(norms_in[token * gridDim.x + kv_head]);
    x_hat *= norm;

    kv_bf16[token * kv_dim + kv_head * head_dim + d] = __float2bfloat16(x_hat);
}

// ============================================================================
// Kernel: Fast Dequant Pool → Working Buffer (in-place NHD for FlashInfer)
// ============================================================================
__global__ void turboquant_fast_dequantize_inplace_kernel(
    const uint8_t* __restrict__ pool_data,
    const __half* __restrict__ pool_norms,
    __nv_bfloat16* __restrict__ work_bf16,
    const int* __restrict__ pool_indices,
    const int8_t* __restrict__ signs,
    const float* __restrict__ centroids,
    int head_dim, int kv_dim, int packed_per_head,
    int num_kv_heads, int num_levels, int bits)
{
    int kv_head = blockIdx.x;
    int idx_pos = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int pool_idx = pool_indices[idx_pos];
    int effective_bits = (bits == 3) ? 4 : bits;
    int indices_per_byte = 8 / effective_bits;

    int src_off = pool_idx * (num_kv_heads * packed_per_head) + kv_head * packed_per_head;
    uint8_t packed_byte = pool_data[src_off + d / indices_per_byte];
    int qidx = (packed_byte >> ((d % indices_per_byte) * effective_bits)) & ((1 << effective_bits) - 1);
    if (qidx >= num_levels) qidx = num_levels - 1;

    extern __shared__ float smem[];
    smem[d] = centroids[qidx];
    ifwht_inplace(smem, head_dim, d);
    float x_hat = smem[d] * (float)signs[d];

    x_hat *= __half2float(pool_norms[pool_idx * num_kv_heads + kv_head]);
    work_bf16[pool_idx * kv_dim + kv_head * head_dim + d] = __float2bfloat16(x_hat);
}

// ============================================================================
// Kernel: Fast Quantize Single Token (paged pool, decode path)
// ============================================================================
__global__ void turboquant_fast_quantize_single_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,
    uint8_t* __restrict__ pool_data,
    __half* __restrict__ pool_norms,
    const int* __restrict__ pool_indices,
    const int8_t* __restrict__ signs,
    const float* __restrict__ boundaries,
    int head_dim, int kv_dim, int packed_per_head,
    int num_kv_heads, int num_levels, int bits)
{
    int kv_head = blockIdx.x;
    int batch_idx = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int pool_idx = pool_indices[batch_idx];
    float x = __bfloat162float(kv_bf16[batch_idx * kv_dim + kv_head * head_dim + d]);

    // Norm
    extern __shared__ float smem[];
    float sq = x * x;
    sq = warp_reduce_sum_fast(sq);
    int warp_id = d / 32, lane_id = d % 32, num_warps = head_dim / 32;
    if (lane_id == 0) smem[head_dim + warp_id] = sq;
    __syncthreads();
    if (d == 0) {
        float t = 0.0f;
        for (int w = 0; w < num_warps; w++) t += smem[head_dim + w];
        smem[head_dim] = t;
    }
    __syncthreads();
    float norm = sqrtf(smem[head_dim]);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    if (d == 0) pool_norms[pool_idx * num_kv_heads + kv_head] = __float2half(norm);

    // Sign flip + FWHT
    float y = x * inv_norm * (float)signs[d];
    smem[d] = y;
    fwht_inplace(smem, head_dim, d);
    y = smem[d];

    // Searchsorted
    int idx = 0;
    for (int k = 1; k < num_levels; k++) idx += (y >= boundaries[k]) ? 1 : 0;

    // Bitpack without shared-memory atomics: one thread writes one packed byte.
    int effective_bits = (bits == 3) ? 4 : bits;
    int indices_per_byte = 8 / effective_bits;
    uint8_t* s_quant = reinterpret_cast<uint8_t*>(&smem[0]);
    s_quant[d] = (uint8_t)(idx & ((1 << effective_bits) - 1));
    __syncthreads();

    int dst = pool_idx * (num_kv_heads * packed_per_head) + kv_head * packed_per_head;
    if (d < packed_per_head) {
        uint8_t packed = 0;
        int src_base = d * indices_per_byte;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (k < indices_per_byte) {
                packed |= (uint8_t)(s_quant[src_base + k] << (k * effective_bits));
            }
        }
        pool_data[dst + d] = packed;
    }
}

// ============================================================================
// Host-side: generate random signs for Hadamard rotation
// ============================================================================
extern "C" void turboquant_generate_signs(
    int8_t* signs,    // output: [D] signs ∈ {-1, +1}
    int head_dim,
    uint64_t seed)
{
    // Simple deterministic sign generation from seed.
    uint64_t state = seed;
    for (int i = 0; i < head_dim; i++) {
        // SplitMix64 step
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= (z >> 31);
        signs[i] = (z & 1) ? 1 : -1;
    }
}

// ============================================================================
// C API — Fast (Hadamard) launcher functions
// ============================================================================

extern "C" CUresult turboquant_fast_quantize_kv_cuda(
    const void* kv_bf16, void* packed_out, void* norms_out,
    const void* signs, const void* boundaries,
    int num_kv_heads, int head_dim, int kv_dim,
    int packed_per_head, int num_levels, int bits,
    int batch_size, CUstream stream)
{
    dim3 grid(num_kv_heads, batch_size);
    dim3 block(head_dim);
    int smem = (head_dim + (head_dim / 32) + 1) * sizeof(float);

    turboquant_fast_quantize_kernel<<<grid, block, smem, stream>>>(
        (const __nv_bfloat16*)kv_bf16, (uint8_t*)packed_out, (__half*)norms_out,
        (const int8_t*)signs, (const float*)boundaries,
        head_dim, kv_dim, packed_per_head, num_levels, bits);
    return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

extern "C" CUresult turboquant_fast_dequantize_kv_cuda(
    const void* packed_in, const void* norms_in, void* kv_bf16,
    const void* signs, const void* centroids,
    int num_kv_heads, int head_dim, int kv_dim,
    int packed_per_head, int num_levels, int bits,
    int token_count, CUstream stream)
{
    dim3 grid(num_kv_heads, token_count);
    dim3 block(head_dim);
    turboquant_fast_dequantize_kernel<<<grid, block, head_dim * (int)sizeof(float), stream>>>(
        (const uint8_t*)packed_in, (const __half*)norms_in, (__nv_bfloat16*)kv_bf16,
        (const int8_t*)signs, (const float*)centroids,
        head_dim, kv_dim, packed_per_head, num_kv_heads, num_levels, bits);
    return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

extern "C" CUresult turboquant_fast_dequantize_inplace_cuda(
    const void* pool_data, const void* pool_norms,
    void* work_bf16, const void* pool_indices,
    const void* signs, const void* centroids,
    int num_kv_heads, int head_dim, int kv_dim,
    int packed_per_head, int num_levels, int bits,
    int num_indices, CUstream stream)
{
    if (num_indices == 0) return CUDA_SUCCESS;
    dim3 grid(num_kv_heads, num_indices);
    dim3 block(head_dim);
    turboquant_fast_dequantize_inplace_kernel<<<grid, block, head_dim * (int)sizeof(float), stream>>>(
        (const uint8_t*)pool_data, (const __half*)pool_norms,
        (__nv_bfloat16*)work_bf16, (const int*)pool_indices,
        (const int8_t*)signs, (const float*)centroids,
        head_dim, kv_dim, packed_per_head, num_kv_heads, num_levels, bits);
    return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

extern "C" CUresult turboquant_fast_quantize_single_cuda(
    const void* kv_bf16, void* pool_data, void* pool_norms,
    const void* pool_indices, const void* signs, const void* boundaries,
    int num_kv_heads, int head_dim, int kv_dim,
    int packed_per_head, int num_levels, int bits,
    int batch_size, CUstream stream)
{
    dim3 grid(num_kv_heads, batch_size);
    dim3 block(head_dim);
    int smem = (head_dim + (head_dim / 32) + 1) * sizeof(float);

    turboquant_fast_quantize_single_kernel<<<grid, block, smem, stream>>>(
        (const __nv_bfloat16*)kv_bf16, (uint8_t*)pool_data, (__half*)pool_norms,
        (const int*)pool_indices, (const int8_t*)signs, (const float*)boundaries,
        head_dim, kv_dim, packed_per_head, num_kv_heads, num_levels, bits);
    return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

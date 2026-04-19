// TurboQuant: rotation-based near-optimal KV cache quantization.
//
// Algorithm (TurboQuantMSE):
//   quantize(x):
//     1. norm = ||x||_2
//     2. x_unit = x / norm
//     3. y = x_unit @ Pi^T          (random orthogonal rotation)
//     4. indices = searchsorted(boundaries, y)
//     5. packed = bitpack(indices)
//   dequantize(packed, norm):
//     1. y_hat = centroids[indices]
//     2. x_hat = y_hat @ Pi          (inverse rotation: Pi^{-1} = Pi^T for orthogonal)
//     3. return x_hat * norm
//
// Memory layout (NHD paged pool, per token per head):
//   packed_indices: [ceil(D * bits / 8)] bytes
//   norm:          f16 (2 bytes)
//
// Reference: arXiv:2504.19874 (Google Research, ICLR 2026)

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

// ─── Constants ───

// Maximum supported head_dim (128 for Qwen3/Llama).
#define TQ_MAX_HEAD_DIM 256

// Maximum bits supported.
#define TQ_MAX_BITS 4
#define TQ_MAX_CENTROIDS (1 << TQ_MAX_BITS)  // 16

// ─── Warp reduction helpers ───

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// ============================================================================
// Lloyd-Max codebook generation (host-side, called once at init)
// ============================================================================

// Beta((D-1)/2, (D-1)/2) PDF on [-1, 1], unnormalized.
// For D=128: f(x) ∝ (1 - x²)^62
static double beta_pdf(double x, int D) {
    double alpha = (D - 1.0) / 2.0;
    double base = 1.0 - x * x;
    if (base <= 0.0) return 0.0;
    return pow(base, alpha - 1.0);
}

// Numerical integration of x^p * f(x) over [a, b] using Simpson's rule.
static double integrate_moment(double a, double b, int p, int D, int nsteps) {
    double h = (b - a) / nsteps;
    double sum = 0.0;
    for (int i = 0; i <= nsteps; i++) {
        double x = a + i * h;
        double w = (i == 0 || i == nsteps) ? 1.0 : (i % 2 == 1) ? 4.0 : 2.0;
        double fx = beta_pdf(x, D);
        double xp = (p == 0) ? 1.0 : (p == 1) ? x : x * x;
        sum += w * fx * xp;
    }
    return sum * h / 3.0;
}

// Run Lloyd-Max iteration to compute optimal centroids and boundaries.
// Output: centroids[num_levels], boundaries[num_levels + 1] (boundaries[0] = -1, boundaries[num_levels] = 1).
extern "C" void turboquant_lloyd_max(
    float* centroids,
    float* boundaries,
    int num_levels,   // 2^bits
    int head_dim,
    int max_iters)
{
    int D = head_dim;
    int K = num_levels;
    int nsteps = 10000;  // Simpson integration steps

    // Initialize centroids at quantile midpoints.
    for (int i = 0; i < K; i++) {
        centroids[i] = -1.0f + (2.0f * (i + 0.5f)) / K;
    }

    // Boundaries always include endpoints.
    boundaries[0] = -1.0f;
    boundaries[K] = 1.0f;

    for (int iter = 0; iter < max_iters; iter++) {
        // Update boundaries: midpoint between adjacent centroids.
        for (int i = 1; i < K; i++) {
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0f;
        }

        // Update centroids: E[X | boundaries[i] <= X < boundaries[i+1]].
        for (int i = 0; i < K; i++) {
            double a = boundaries[i];
            double b = boundaries[i + 1];
            double m0 = integrate_moment(a, b, 0, D, nsteps);
            double m1 = integrate_moment(a, b, 1, D, nsteps);
            if (m0 > 1e-30) {
                centroids[i] = (float)(m1 / m0);
            }
        }
    }
}

// ============================================================================
// Generate random orthogonal rotation matrix via QR decomposition.
//
// Uses a simple seeded Gaussian → Householder QR on the host.
// Output: Pi[D * D] stored in row-major order.
// ============================================================================

// Simple xoshiro256** PRNG for deterministic cross-platform rotation generation.
static uint64_t tq_rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

struct TqRng {
    uint64_t s[4];

    void seed(uint64_t seed) {
        // SplitMix64 to initialize state
        for (int i = 0; i < 4; i++) {
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }

    uint64_t next() {
        uint64_t result = tq_rotl(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = tq_rotl(s[3], 45);
        return result;
    }

    // Box-Muller transform for Gaussian samples.
    float gaussian() {
        double u1 = (double)(next() >> 11) / (double)(1ULL << 53);
        double u2 = (double)(next() >> 11) / (double)(1ULL << 53);
        if (u1 < 1e-15) u1 = 1e-15;
        return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
    }
};

// Householder QR decomposition in-place. A is D×D column-major.
// After completion, Q is stored in A.
static void householder_qr_inplace(float* A, int D) {
    // R is not needed — we just want Q.
    // Use the implicit Q representation and accumulate Q = I - 2vv^T products.
    float* v = new float[D];
    float* tau = new float[D];

    for (int j = 0; j < D; j++) {
        // Compute Householder vector for column j, rows j..D-1.
        float norm_sq = 0.0f;
        for (int i = j; i < D; i++) {
            float aij = A[i + j * D];
            norm_sq += aij * aij;
        }
        float norm = sqrtf(norm_sq);
        float a_jj = A[j + j * D];
        float sign = (a_jj >= 0) ? 1.0f : -1.0f;
        float u1 = a_jj + sign * norm;

        // Store v in-place below diagonal.
        for (int i = j + 1; i < D; i++) {
            v[i] = A[i + j * D] / u1;
            A[i + j * D] = v[i];
        }
        v[j] = 1.0f;
        tau[j] = 2.0f * u1 * u1 / (norm_sq + fabsf(a_jj) * norm + norm * norm);
        if (norm_sq < 1e-30f) tau[j] = 0.0f;
        else tau[j] = sign * u1 / norm;

        // Compute tau properly: tau = 2 / (v^T v)
        float vdot = 0.0f;
        for (int i = j; i < D; i++) vdot += v[i] * v[i];
        tau[j] = (vdot > 1e-30f) ? 2.0f / vdot : 0.0f;

        // Apply Householder to remaining columns: A[:, j+1:D] -= tau * v * (v^T A[:, j+1:D])
        for (int k = j + 1; k < D; k++) {
            float dot = 0.0f;
            for (int i = j; i < D; i++) dot += v[i] * A[i + k * D];
            for (int i = j; i < D; i++) A[i + k * D] -= tau[j] * v[i] * dot;
        }

        // Set R diagonal and above.
        A[j + j * D] = -sign * norm;
    }

    // Accumulate Q from Householder vectors (backward accumulation).
    // Q = H_0 * H_1 * ... * H_{D-1} where H_k = I - tau_k * v_k * v_k^T
    float* Q = new float[D * D];
    // Start with identity.
    for (int i = 0; i < D * D; i++) Q[i] = 0.0f;
    for (int i = 0; i < D; i++) Q[i + i * D] = 1.0f;

    for (int j = D - 1; j >= 0; j--) {
        // Reconstruct v from below diagonal.
        v[j] = 1.0f;
        for (int i = j + 1; i < D; i++) v[i] = A[i + j * D];

        // Q = (I - tau * v * v^T) * Q  →  Q -= tau * v * (v^T Q)
        for (int k = j; k < D; k++) {
            float dot = 0.0f;
            for (int i = j; i < D; i++) dot += v[i] * Q[i + k * D];
            for (int i = j; i < D; i++) Q[i + k * D] -= tau[j] * v[i] * dot;
        }
    }

    // Copy Q back to A (row-major output).
    // A was column-major internally; Q is column-major. We want row-major output.
    // Pi[row][col] = Q[row + col * D] → output as Pi[row * D + col].
    float* Pi = new float[D * D];
    for (int r = 0; r < D; r++)
        for (int c = 0; c < D; c++)
            Pi[r * D + c] = Q[r + c * D];

    memcpy(A, Pi, D * D * sizeof(float));
    delete[] Pi;
    delete[] Q;
    delete[] v;
    delete[] tau;
}

extern "C" void turboquant_generate_rotation(
    float* Pi,       // output: D×D row-major, host memory
    int head_dim,
    uint64_t seed)
{
    int D = head_dim;
    TqRng rng;
    rng.seed(seed);

    // Fill D×D column-major Gaussian matrix.
    float* A = new float[D * D];
    for (int c = 0; c < D; c++)
        for (int r = 0; r < D; r++)
            A[r + c * D] = rng.gaussian();

    householder_qr_inplace(A, D);
    memcpy(Pi, A, D * D * sizeof(float));
    delete[] A;
}

// ============================================================================
// Kernel 1: TurboQuant Quantize KV
//
// BF16 KV → packed indices + f16 norm.
//
// Pool layout (NHD, per token):
//   packed: [packed_bytes_per_head] bytes per head
//   norms:  separate buffer, [f16] per head per token
//
// Grid:  (num_kv_heads, batch_size)
// Block: (head_dim)   — one thread per coordinate
// ============================================================================

// Shared memory: rotation matrix tile (D floats) + reduction scratch.
__global__ void turboquant_quantize_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,  // source: [batch, kv_dim] or working buffer
    uint8_t* __restrict__ packed_out,             // dest: [batch, num_kv_heads * packed_per_head]
    __half* __restrict__ norms_out,               // dest: [batch, num_kv_heads]
    const float* __restrict__ Pi,                 // rotation matrix [D, D] row-major
    const float* __restrict__ boundaries,         // [num_levels + 1] (including endpoints)
    int head_dim,
    int kv_dim,                                   // num_kv_heads * head_dim
    int packed_per_head,                          // ceil(D * bits / 8)
    int num_levels,                               // 2^bits
    int bits)
{
    int kv_head = blockIdx.x;
    int batch_idx = blockIdx.y;
    int d = threadIdx.x;

    if (d >= head_dim) return;

    // ─── Step 1: Load bf16 value ───
    int src_offset = batch_idx * kv_dim + kv_head * head_dim + d;
    float x_val = __bfloat162float(kv_bf16[src_offset]);

    // ─── Step 2: Compute ||x||_2 via shared memory reduction ───
    extern __shared__ float smem[];  // [head_dim] for reduction + [head_dim] for x_unit
    float* s_x = smem;
    float* s_norm = smem + head_dim;

    s_x[d] = x_val;
    __syncthreads();

    // Warp reduction for sum of squares.
    float sq = x_val * x_val;
    sq = warp_reduce_sum(sq);

    // Cross-warp reduction via shared memory.
    int warp_id = d / 32;
    int lane_id = d % 32;
    int num_warps = (head_dim + 31) / 32;

    if (lane_id == 0) s_norm[warp_id] = sq;
    __syncthreads();

    float norm_sq;
    if (d == 0) {
        norm_sq = 0.0f;
        for (int w = 0; w < num_warps; w++) norm_sq += s_norm[w];
        s_norm[0] = norm_sq;
    }
    __syncthreads();
    norm_sq = s_norm[0];

    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    // Store norm (thread 0 writes).
    if (d == 0) {
        int norm_offset = batch_idx * gridDim.x + kv_head;
        norms_out[norm_offset] = __float2half(norm);
    }

    // ─── Step 3: x_unit = x / norm ───
    float x_unit = x_val * inv_norm;

    // ─── Step 4: Rotation y[d] = sum_j(Pi[d][j] * x_unit[j]) ───
    // Load x_unit into shared memory for broadcast access.
    s_x[d] = x_unit;
    __syncthreads();

    float y = 0.0f;
    // Pi is [D, D] row-major: Pi[d][j] = Pi[d * D + j]
    const float* Pi_row = Pi + d * head_dim;
    for (int j = 0; j < head_dim; j++) {
        y += Pi_row[j] * s_x[j];
    }

    // ─── Step 5: Searchsorted (binary search in boundaries) ───
    // boundaries[0] = -1.0, boundaries[num_levels] = 1.0
    // Interior boundaries: boundaries[1..num_levels-1]
    // Find largest k such that boundaries[k] <= y.
    int idx = 0;
    for (int k = 1; k < num_levels; k++) {
        if (y >= boundaries[k]) idx = k;
    }

    // ─── Step 6: Cooperative bitpack ───
    // Pack `bits`-bit indices into bytes. LSB-first packing.
    // Each thread has one index (0..num_levels-1). Threads cooperate to write packed bytes.
    // Strategy: each group of (8/bits) threads packs into one byte.
    // For bits=3, we round up: 2 indices per byte (stored as 4-bit pairs, wastes 1 bit each).
    // For bits=2, 4 indices per byte. For bits=4, 2 indices per byte.

    int effective_bits = bits;
    if (bits == 3) effective_bits = 4;  // round up to nibble for simplicity

    int indices_per_byte = 8 / effective_bits;
    int byte_idx = d / indices_per_byte;
    int sub_idx = d % indices_per_byte;

    // Write into shared memory first, then coalesced store.
    // Reuse s_x as uint8 scratch.
    uint8_t* s_packed = (uint8_t*)smem;
    // Zero the packed buffer region first.
    if (d < packed_per_head) s_packed[d] = 0;
    __syncthreads();

    uint8_t shifted = (uint8_t)(idx & ((1 << effective_bits) - 1)) << (sub_idx * effective_bits);
    atomicOr((unsigned int*)(s_packed + (byte_idx & ~3)), (unsigned int)shifted << (8 * (byte_idx & 3)));
    __syncthreads();

    // ─── Step 7: Store packed bytes ───
    int dst_offset = batch_idx * (gridDim.x * packed_per_head) + kv_head * packed_per_head;
    if (d < packed_per_head) {
        packed_out[dst_offset + d] = s_packed[d];
    }
}

// ============================================================================
// Kernel 2: TurboQuant Dequantize KV
//
// Packed indices + f16 norm → BF16 KV.
//
// Grid:  (num_kv_heads, token_count)
// Block: (head_dim)
// ============================================================================

__global__ void turboquant_dequantize_kernel(
    const uint8_t* __restrict__ packed_in,        // [tokens, num_kv_heads * packed_per_head]
    const __half* __restrict__ norms_in,           // [tokens, num_kv_heads]
    __nv_bfloat16* __restrict__ kv_bf16,           // output: [tokens, kv_dim]
    const float* __restrict__ Pi,                  // rotation matrix [D, D] row-major
    const float* __restrict__ centroids,           // [num_levels]
    int head_dim,
    int kv_dim,
    int packed_per_head,
    int num_levels,
    int bits)
{
    int kv_head = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;

    if (d >= head_dim) return;

    // ─── Step 1: Unpack index for coordinate d ───
    int effective_bits = bits;
    if (bits == 3) effective_bits = 4;

    int indices_per_byte = 8 / effective_bits;
    int byte_idx = d / indices_per_byte;
    int sub_idx = d % indices_per_byte;

    int src_offset = token * (gridDim.x * packed_per_head) + kv_head * packed_per_head;
    uint8_t packed_byte = packed_in[src_offset + byte_idx];
    int idx = (packed_byte >> (sub_idx * effective_bits)) & ((1 << effective_bits) - 1);
    if (idx >= num_levels) idx = num_levels - 1;  // safety clamp

    // ─── Step 2: Gather centroid value ───
    float y_hat = centroids[idx];

    // ─── Step 3: Load y_hat into shared memory for rotation ───
    extern __shared__ float smem[];
    smem[d] = y_hat;
    __syncthreads();

    // ─── Step 4: Inverse rotation: x_hat[d] = sum_j(Pi[j][d] * y_hat[j]) ───
    // Pi^{-1} = Pi^T for orthogonal matrix.
    // Pi^T[d][j] = Pi[j][d], so x_hat[d] = sum_j Pi[j * D + d] * y_hat[j]
    float x_hat = 0.0f;
    for (int j = 0; j < head_dim; j++) {
        x_hat += Pi[j * head_dim + d] * smem[j];
    }

    // ─── Step 5: Scale by norm ───
    int norm_offset = token * gridDim.x + kv_head;
    float norm = __half2float(norms_in[norm_offset]);
    x_hat *= norm;

    // ─── Step 6: Store bf16 ───
    int dst_offset = token * kv_dim + kv_head * head_dim + d;
    kv_bf16[dst_offset] = __float2bfloat16(x_hat);
}

// ============================================================================
// Kernel 3: TurboQuant Quantize Single Token (paged pool, decode path)
//
// Quantize 1 new token per request from bf16 working buffer → TQ paged pool.
//
// Grid:  (num_kv_heads, batch_size)
// Block: (head_dim)
// ============================================================================

__global__ void turboquant_quantize_single_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,   // working buffer: [batch, kv_dim]
    uint8_t* __restrict__ pool_data,               // pool: [max_tokens, num_kv_heads * packed_per_head]
    __half* __restrict__ pool_norms,               // pool: [max_tokens, num_kv_heads]
    const int* __restrict__ pool_indices,           // [batch_size] — physical pool index per request
    const float* __restrict__ Pi,                  // rotation matrix [D, D] row-major
    const float* __restrict__ boundaries,          // [num_levels + 1]
    int head_dim,
    int kv_dim,
    int packed_per_head,
    int num_kv_heads,
    int num_levels,
    int bits)
{
    int kv_head = blockIdx.x;
    int batch_idx = blockIdx.y;
    int d = threadIdx.x;

    if (d >= head_dim) return;

    int pool_idx = pool_indices[batch_idx];

    // Load source bf16 value.
    int src_offset = batch_idx * kv_dim + kv_head * head_dim + d;
    float x_val = __bfloat162float(kv_bf16[src_offset]);

    // Shared memory for reduction and broadcast.
    extern __shared__ float smem[];
    float* s_x = smem;
    float* s_norm = smem + head_dim;

    s_x[d] = x_val;

    // Compute ||x||_2.
    float sq = x_val * x_val;
    sq = warp_reduce_sum(sq);
    int warp_id = d / 32;
    int lane_id = d % 32;
    int num_warps = (head_dim + 31) / 32;
    if (lane_id == 0) s_norm[warp_id] = sq;
    __syncthreads();

    float norm_sq;
    if (d == 0) {
        norm_sq = 0.0f;
        for (int w = 0; w < num_warps; w++) norm_sq += s_norm[w];
        s_norm[0] = norm_sq;
    }
    __syncthreads();
    norm_sq = s_norm[0];

    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    // Store norm.
    if (d == 0) {
        pool_norms[pool_idx * num_kv_heads + kv_head] = __float2half(norm);
    }

    // Normalize and rotate.
    float x_unit = x_val * inv_norm;
    s_x[d] = x_unit;
    __syncthreads();

    float y = 0.0f;
    const float* Pi_row = Pi + d * head_dim;
    for (int j = 0; j < head_dim; j++) {
        y += Pi_row[j] * s_x[j];
    }

    // Searchsorted.
    int idx = 0;
    for (int k = 1; k < num_levels; k++) {
        if (y >= boundaries[k]) idx = k;
    }

    // Cooperative bitpack.
    int effective_bits = bits;
    if (bits == 3) effective_bits = 4;

    int indices_per_byte = 8 / effective_bits;
    int byte_idx = d / indices_per_byte;
    int sub_idx = d % indices_per_byte;

    uint8_t* s_packed = (uint8_t*)smem;
    if (d < packed_per_head) s_packed[d] = 0;
    __syncthreads();

    uint8_t shifted = (uint8_t)(idx & ((1 << effective_bits) - 1)) << (sub_idx * effective_bits);
    atomicOr((unsigned int*)(s_packed + (byte_idx & ~3)), (unsigned int)shifted << (8 * (byte_idx & 3)));
    __syncthreads();

    // Store to pool.
    int dst_offset = pool_idx * (num_kv_heads * packed_per_head) + kv_head * packed_per_head;
    if (d < packed_per_head) {
        pool_data[dst_offset + d] = s_packed[d];
    }
}

// ============================================================================
// Kernel 4: TurboQuant Dequantize from Paged Pool
//
// Scatter-read from paged pool → contiguous bf16 working buffer.
//
// Grid:  (num_kv_heads, total_tokens)
// Block: (head_dim)
// ============================================================================

__global__ void turboquant_dequantize_paged_kernel(
    const uint8_t* __restrict__ pool_data,        // [max_tokens, num_kv_heads * packed_per_head]
    const __half* __restrict__ pool_norms,         // [max_tokens, num_kv_heads]
    __nv_bfloat16* __restrict__ kv_bf16,           // output: [total_tokens, kv_dim]
    const int* __restrict__ token_indices,          // [total_tokens] — physical pool indices
    const float* __restrict__ Pi,                  // rotation [D, D]
    const float* __restrict__ centroids,           // [num_levels]
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

    int pool_idx = token_indices[token];

    // Unpack index.
    int effective_bits = bits;
    if (bits == 3) effective_bits = 4;

    int indices_per_byte = 8 / effective_bits;
    int byte_idx = d / indices_per_byte;
    int sub_idx = d % indices_per_byte;

    int src_offset = pool_idx * (num_kv_heads * packed_per_head) + kv_head * packed_per_head;
    uint8_t packed_byte = pool_data[src_offset + byte_idx];
    int idx = (packed_byte >> (sub_idx * effective_bits)) & ((1 << effective_bits) - 1);
    if (idx >= num_levels) idx = num_levels - 1;

    float y_hat = centroids[idx];

    // Inverse rotation via shared memory.
    extern __shared__ float smem[];
    smem[d] = y_hat;
    __syncthreads();

    float x_hat = 0.0f;
    for (int j = 0; j < head_dim; j++) {
        x_hat += Pi[j * head_dim + d] * smem[j];
    }

    // Scale by norm.
    float norm = __half2float(pool_norms[pool_idx * num_kv_heads + kv_head]);
    x_hat *= norm;

    // Store.
    int dst_offset = token * kv_dim + kv_head * head_dim + d;
    kv_bf16[dst_offset] = __float2bfloat16(x_hat);
}

// ============================================================================
// C API — launcher functions
// ============================================================================

extern "C" CUresult turboquant_quantize_kv_cuda(
    const void* kv_bf16,
    void* packed_out,
    void* norms_out,
    const void* Pi,
    const void* boundaries,
    int num_kv_heads,
    int head_dim,
    int kv_dim,
    int packed_per_head,
    int num_levels,
    int bits,
    int batch_size,
    CUstream stream)
{
    dim3 grid(num_kv_heads, batch_size);
    dim3 block(head_dim);
    // Shared memory: head_dim floats (x values) + ceil(head_dim/32) floats (reduction)
    int smem = (head_dim + ((head_dim + 31) / 32)) * sizeof(float);
    // Ensure enough for packed scratch too.
    int packed_smem = packed_per_head + 4;  // +4 for alignment
    if (packed_smem > smem) smem = packed_smem;

    turboquant_quantize_kernel<<<grid, block, smem, stream>>>(
        (const __nv_bfloat16*)kv_bf16,
        (uint8_t*)packed_out,
        (__half*)norms_out,
        (const float*)Pi,
        (const float*)boundaries,
        head_dim, kv_dim, packed_per_head, num_levels, bits);

    return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

extern "C" CUresult turboquant_dequantize_kv_cuda(
    const void* packed_in,
    const void* norms_in,
    void* kv_bf16,
    const void* Pi,
    const void* centroids,
    int num_kv_heads,
    int head_dim,
    int kv_dim,
    int packed_per_head,
    int num_levels,
    int bits,
    int token_count,
    CUstream stream)
{
    dim3 grid(num_kv_heads, token_count);
    dim3 block(head_dim);
    int smem = head_dim * sizeof(float);

    turboquant_dequantize_kernel<<<grid, block, smem, stream>>>(
        (const uint8_t*)packed_in,
        (const __half*)norms_in,
        (__nv_bfloat16*)kv_bf16,
        (const float*)Pi,
        (const float*)centroids,
        head_dim, kv_dim, packed_per_head, num_levels, bits);

    return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

extern "C" CUresult turboquant_quantize_single_cuda(
    const void* kv_bf16,
    void* pool_data,
    void* pool_norms,
    const void* pool_indices,
    const void* Pi,
    const void* boundaries,
    int num_kv_heads,
    int head_dim,
    int kv_dim,
    int packed_per_head,
    int num_levels,
    int bits,
    int batch_size,
    CUstream stream)
{
    dim3 grid(num_kv_heads, batch_size);
    dim3 block(head_dim);
    int smem = (head_dim + ((head_dim + 31) / 32)) * sizeof(float);
    int packed_smem = packed_per_head + 4;
    if (packed_smem > smem) smem = packed_smem;

    turboquant_quantize_single_kernel<<<grid, block, smem, stream>>>(
        (const __nv_bfloat16*)kv_bf16,
        (uint8_t*)pool_data,
        (__half*)pool_norms,
        (const int*)pool_indices,
        (const float*)Pi,
        (const float*)boundaries,
        head_dim, kv_dim, packed_per_head, num_kv_heads, num_levels, bits);

    return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

extern "C" CUresult turboquant_dequantize_paged_cuda(
    const void* pool_data,
    const void* pool_norms,
    void* kv_bf16,
    const void* token_indices,
    const void* Pi,
    const void* centroids,
    int num_kv_heads,
    int head_dim,
    int kv_dim,
    int packed_per_head,
    int num_levels,
    int bits,
    int total_tokens,
    CUstream stream)
{
    dim3 grid(num_kv_heads, total_tokens);
    dim3 block(head_dim);
    int smem = head_dim * sizeof(float);

    turboquant_dequantize_paged_kernel<<<grid, block, smem, stream>>>(
        (const uint8_t*)pool_data,
        (const __half*)pool_norms,
        (__nv_bfloat16*)kv_bf16,
        (const int*)token_indices,
        (const float*)Pi,
        (const float*)centroids,
        head_dim, kv_dim, packed_per_head, num_kv_heads, num_levels, bits);

    return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Kernel 5: Dequantize Pool → Working Buffer (in-place NHD layout)
// ============================================================================

__global__ void turboquant_dequantize_inplace_kernel(
    const uint8_t* __restrict__ pool_data,
    const __half* __restrict__ pool_norms,
    __nv_bfloat16* __restrict__ work_bf16,
    const int* __restrict__ pool_indices,
    const float* __restrict__ Pi,
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
    __syncthreads();

    float x_hat = 0.0f;
    for (int j = 0; j < head_dim; j++)
        x_hat += Pi[j * head_dim + d] * smem[j];

    x_hat *= __half2float(pool_norms[pool_idx * num_kv_heads + kv_head]);
    work_bf16[pool_idx * kv_dim + kv_head * head_dim + d] = __float2bfloat16(x_hat);
}

extern "C" CUresult turboquant_dequantize_inplace_cuda(
    const void* pool_data, const void* pool_norms,
    void* work_bf16, const void* pool_indices,
    const void* Pi, const void* centroids,
    int num_kv_heads, int head_dim, int kv_dim,
    int packed_per_head, int num_levels, int bits,
    int num_indices, CUstream stream)
{
    if (num_indices == 0) return CUDA_SUCCESS;
    dim3 grid(num_kv_heads, num_indices);
    dim3 block(head_dim);
    turboquant_dequantize_inplace_kernel<<<grid, block, head_dim * (int)sizeof(float), stream>>>(
        (const uint8_t*)pool_data, (const __half*)pool_norms,
        (__nv_bfloat16*)work_bf16, (const int*)pool_indices,
        (const float*)Pi, (const float*)centroids,
        head_dim, kv_dim, packed_per_head, num_kv_heads, num_levels, bits);
    return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

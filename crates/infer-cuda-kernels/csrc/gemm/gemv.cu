#include "common.cuh"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <unordered_map>

// ============================================================================
// Hand-written GEMV: y = A @ x (row-major matrix)
// Each block processes GEMV_ROWS_PER_BLOCK rows.
// BF16×4 vectorized loads (8 bytes/thread/stride) for memory throughput.
// Warp shuffle reduction + shared memory for inter-warp reduce.
// BF16 inputs, FP32 accumulators, BF16 output.
// Graph-capture safe (no cuBLAS workspace allocation).
// ============================================================================
#define GEMV_BLOCK 256
#define GEMV_ROWS_PER_BLOCK 4
#define GEMV_NUM_WARPS (GEMV_BLOCK / WARP_SIZE)

__device__ __forceinline__ float bf16x4_dot(uint2 a_val, uint2 x_val) {
  __nv_bfloat162 a_lo = *reinterpret_cast<__nv_bfloat162 *>(&a_val.x);
  __nv_bfloat162 a_hi = *reinterpret_cast<__nv_bfloat162 *>(&a_val.y);
  __nv_bfloat162 x_lo = *reinterpret_cast<__nv_bfloat162 *>(&x_val.x);
  __nv_bfloat162 x_hi = *reinterpret_cast<__nv_bfloat162 *>(&x_val.y);
  float sum = 0.0f;
  sum += __bfloat162float(a_lo.x) * __bfloat162float(x_lo.x);
  sum += __bfloat162float(a_lo.y) * __bfloat162float(x_lo.y);
  sum += __bfloat162float(a_hi.x) * __bfloat162float(x_hi.x);
  sum += __bfloat162float(a_hi.y) * __bfloat162float(x_hi.y);
  return sum;
}

__device__ __forceinline__ float bf16x8_dot(uint4 a_val, uint4 x_val) {
  float sum = 0.0f;
  sum += bf16x4_dot(make_uint2(a_val.x, a_val.y), make_uint2(x_val.x, x_val.y));
  sum += bf16x4_dot(make_uint2(a_val.z, a_val.w), make_uint2(x_val.z, x_val.w));
  return sum;
}

__global__ void gemv_handwritten_kernel(
    const __nv_bfloat16 *__restrict__ A, // (M, K) row-major
    const __nv_bfloat16 *__restrict__ x, // (K,)
    __nv_bfloat16 *__restrict__ y,       // (M,)
    int M, int K) {

  extern __shared__ __align__(16) char smem[];

  int row_base = blockIdx.x * GEMV_ROWS_PER_BLOCK;
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  __nv_bfloat16 *x_shared = reinterpret_cast<__nv_bfloat16 *>(smem);

  // Vectorized BF16×8 / BF16×4 paths with scalar fallback for remainder.
  int K8 = K / 8;  // number of bf16x8 groups
  int K4 = K / 4;  // number of bf16x4 groups
  int K_tail = K - K4 * 4;  // remainder for scalar fallback
  bool use_bf16x8 = (K % 8) == 0;

  float sums[GEMV_ROWS_PER_BLOCK];
  #pragma unroll
  for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) sums[r] = 0.0f;

  if (use_bf16x8) {
    const uint4 *x_vec8 = reinterpret_cast<const uint4 *>(x);
    uint4 *x_shared_vec8 = reinterpret_cast<uint4 *>(x_shared);
    for (int k8 = tid; k8 < K8; k8 += GEMV_BLOCK) {
      x_shared_vec8[k8] = x_vec8[k8];
    }
  } else {
    const uint2 *x_vec4 = reinterpret_cast<const uint2 *>(x);
    uint2 *x_shared_vec4 = reinterpret_cast<uint2 *>(x_shared);
    for (int k4 = tid; k4 < K4; k4 += GEMV_BLOCK) {
      x_shared_vec4[k4] = x_vec4[k4];
    }
    if (K_tail > 0) {
      int k_start = K4 * 4;
      for (int k = k_start + tid; k < K; k += GEMV_BLOCK) {
        x_shared[k] = x[k];
      }
    }
  }
  __syncthreads();

  #pragma unroll
  for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) {
    int row = row_base + r;
    if (row < M) {
      float sum = 0.0f;

      if (use_bf16x8) {
        const uint4 *A_row_vec8 = reinterpret_cast<const uint4 *>(A + row * K);
        const uint4 *x_shared_vec8 = reinterpret_cast<const uint4 *>(x_shared);
        for (int k8 = tid; k8 < K8; k8 += GEMV_BLOCK) {
          sum += bf16x8_dot(A_row_vec8[k8], x_shared_vec8[k8]);
        }
      } else {
        const uint2 *A_row_vec4 = reinterpret_cast<const uint2 *>(A + row * K);
        const uint2 *x_shared_vec4 = reinterpret_cast<const uint2 *>(x_shared);
        for (int k4 = tid; k4 < K4; k4 += GEMV_BLOCK) {
          sum += bf16x4_dot(A_row_vec4[k4], x_shared_vec4[k4]);
        }
      }

      if (K_tail > 0) {
        const __nv_bfloat16 *A_row = A + row * K;
        int k_start = K4 * 4;
        for (int k = k_start + tid; k < K; k += GEMV_BLOCK) {
          sum += __bfloat162float(A_row[k]) * __bfloat162float(x_shared[k]);
        }
      }

      sums[r] = sum;
    }
  }

  // Warp-level reduction via shuffle
  #pragma unroll
  for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) {
    sums[r] = warp_reduce_sum(sums[r]);
  }

  // Inter-warp reduction via shared memory.
  // Layout: [WARPS][ROWS+1] — transposed + padded to avoid bank conflicts.
  // Old layout [ROWS][WARPS]: 8 warps write to same row → 8-way conflict.
  // New layout [WARPS][ROWS+1]: each warp writes to its own row → zero conflict.
  __shared__ float warp_sums[GEMV_NUM_WARPS][GEMV_ROWS_PER_BLOCK + 1];

  if (lane_id == 0) {
    #pragma unroll
    for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) {
      warp_sums[warp_id][r] = sums[r];
    }
  }
  __syncthreads();

  // First warp reduces across all warps
  if (warp_id == 0) {
    #pragma unroll
    for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) {
      float val = (lane_id < GEMV_NUM_WARPS) ? warp_sums[lane_id][r] : 0.0f;
      val = warp_reduce_sum(val);
      if (lane_id == 0) {
        int row = row_base + r;
        if (row < M) {
          y[row] = __float2bfloat16(val);
        }
      }
    }
  }
}

// Attention score: score = q @ k / sqrt(head_dim)
__global__ void attention_scores_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ scores,
    int seq_len, int head_dim, float scale) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < seq_len) {
    float dot = 0.0f;
    const __nv_bfloat16 *k = k_cache + pos * head_dim;
    for (int i = 0; i < head_dim; i++) {
      dot += __bfloat162float(q[i]) * __bfloat162float(k[i]);
    }
    scores[pos] = __float2bfloat16(dot * scale);
  }
}

// Attention weighted sum: out = sum(weights[i] * v[i])
__global__ void attention_weighted_sum_kernel(
    const __nv_bfloat16 *__restrict__ weights,
    const __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ out,
    int seq_len, int head_dim) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d < head_dim) {
    float sum = 0.0f;
    for (int pos = 0; pos < seq_len; pos++) {
      sum += __bfloat162float(weights[pos]) * __bfloat162float(v_cache[pos * head_dim + d]);
    }
    out[d] = __float2bfloat16(sum);
  }
}

// cuBLAS handle management (external linkage — shared with prefill_attention.cu)
// g_cublas_handle: workspace-free, safe for CUDA Graph capture (decode path).
// g_cublas_prefill_handle: has 32MB workspace, allows cuBLAS to pick faster algorithms
// for the 252 GEMMs per prefill. Never used under CUDA Graphs.
cublasHandle_t g_cublas_handle = nullptr;
cublasHandle_t g_cublas_prefill_handle = nullptr;
cublasLtHandle_t g_cublaslt_handle = nullptr;

static void *g_cublas_workspace = nullptr;
static const size_t CUBLAS_WORKSPACE_SIZE = 32 * 1024 * 1024; // 32MB
static void *g_cublaslt_workspace = nullptr;
static constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;

struct GemmKey {
  int M;
  int N;
  int K;

  bool operator==(const GemmKey &other) const {
    return M == other.M && N == other.N && K == other.K;
  }
};

struct GemmKeyHash {
  size_t operator()(const GemmKey &key) const {
    size_t h = static_cast<size_t>(key.M);
    h = h * 1315423911u + static_cast<size_t>(key.N);
    h = h * 1315423911u + static_cast<size_t>(key.K);
    return h;
  }
};

static std::unordered_map<GemmKey, cublasLtMatmulAlgo_t, GemmKeyHash> g_algo_cache;
static bool g_gemm_graphsafe_mode = false;

static cudaError_t gemm_cublas_fallback(const __nv_bfloat16 *W, const __nv_bfloat16 *X,
                                        __nv_bfloat16 *Y, int M, int N, int K,
                                        cudaStream_t stream, cublasHandle_t handle) {
  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;
  if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }
  if (cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                   M, N, K,
                   &h_alpha,
                   W, CUDA_R_16BF, K,
                   X, CUDA_R_16BF, K,
                   &h_beta,
                   Y, CUDA_R_16BF, M,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }
  return cudaGetLastError();
}

static cudaError_t gemm_cublaslt_impl(const __nv_bfloat16 *W, const __nv_bfloat16 *X,
                                      __nv_bfloat16 *Y, int M, int N, int K,
                                      cudaStream_t stream) {
  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t w_desc = nullptr;
  cublasLtMatrixLayout_t x_desc = nullptr;
  cublasLtMatrixLayout_t y_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;
  const GemmKey key{M, N, K};

  if (cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
      CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  if (cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &transa, sizeof(transa)) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &transb, sizeof(transb)) != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(operation_desc);
    return cudaErrorUnknown;
  }

  if (cublasLtMatrixLayoutCreate(&w_desc, CUDA_R_16BF, K, M, K) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&x_desc, CUDA_R_16BF, K, N, K) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&y_desc, CUDA_R_16BF, M, N, M) != CUBLAS_STATUS_SUCCESS) {
    if (y_desc != nullptr) cublasLtMatrixLayoutDestroy(y_desc);
    if (x_desc != nullptr) cublasLtMatrixLayoutDestroy(x_desc);
    if (w_desc != nullptr) cublasLtMatrixLayoutDestroy(w_desc);
    cublasLtMatmulDescDestroy(operation_desc);
    return cudaErrorUnknown;
  }

  auto algo_it = g_algo_cache.find(key);
  if (algo_it == g_algo_cache.end()) {
    if (g_gemm_graphsafe_mode) {
      cublasLtMatrixLayoutDestroy(y_desc);
      cublasLtMatrixLayoutDestroy(x_desc);
      cublasLtMatrixLayoutDestroy(w_desc);
      cublasLtMatmulDescDestroy(operation_desc);
      return gemm_cublas_fallback(W, X, Y, M, N, K, stream, g_cublas_handle);
    }

    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
      cublasLtMatrixLayoutDestroy(y_desc);
      cublasLtMatrixLayoutDestroy(x_desc);
      cublasLtMatrixLayoutDestroy(w_desc);
      cublasLtMatmulDescDestroy(operation_desc);
      return cudaErrorUnknown;
    }

    if (cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspaceBytes, sizeof(kWorkspaceBytes)) != CUBLAS_STATUS_SUCCESS) {
      cublasLtMatmulPreferenceDestroy(preference);
      cublasLtMatrixLayoutDestroy(y_desc);
      cublasLtMatrixLayoutDestroy(x_desc);
      cublasLtMatrixLayoutDestroy(w_desc);
      cublasLtMatmulDescDestroy(operation_desc);
      return cudaErrorUnknown;
    }

    cublasLtMatmulHeuristicResult_t heuristic_results[8];
    int returned_algo_count = 0;
    cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        g_cublaslt_handle, operation_desc, w_desc, x_desc, y_desc, y_desc,
        preference, 8, heuristic_results, &returned_algo_count);
    cublasLtMatmulPreferenceDestroy(preference);
    preference = nullptr;

    if (heuristic_status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0) {
      algo_it = g_algo_cache.emplace(key, heuristic_results[0].algo).first;
    } else {
      cublasLtMatrixLayoutDestroy(y_desc);
      cublasLtMatrixLayoutDestroy(x_desc);
      cublasLtMatrixLayoutDestroy(w_desc);
      cublasLtMatmulDescDestroy(operation_desc);
      return gemm_cublas_fallback(W, X, Y, M, N, K, stream, g_cublas_prefill_handle);
    }
  }

  if (cublasLtMatmul(g_cublaslt_handle, operation_desc,
                     &h_alpha,
                     W, w_desc,
                     X, x_desc,
                     &h_beta,
                     Y, y_desc,
                     Y, y_desc,
                     &algo_it->second,
                     g_gemm_graphsafe_mode ? nullptr : g_cublaslt_workspace,
                     g_gemm_graphsafe_mode ? 0 : kWorkspaceBytes,
                     stream) != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatrixLayoutDestroy(y_desc);
    cublasLtMatrixLayoutDestroy(x_desc);
    cublasLtMatrixLayoutDestroy(w_desc);
    cublasLtMatmulDescDestroy(operation_desc);
    return cudaErrorUnknown;
  }

  cublasLtMatrixLayoutDestroy(y_desc);
  cublasLtMatrixLayoutDestroy(x_desc);
  cublasLtMatrixLayoutDestroy(w_desc);
  cublasLtMatmulDescDestroy(operation_desc);
  return cudaGetLastError();
}

extern "C" {

void cublas_init() {
  if (g_cublas_handle == nullptr) {
    cublasCreate(&g_cublas_handle);
    cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
  }
  if (g_cublas_prefill_handle == nullptr) {
    cublasCreate(&g_cublas_prefill_handle);
    cublasSetMathMode(g_cublas_prefill_handle, CUBLAS_TENSOR_OP_MATH);
  }
  if (g_cublaslt_handle == nullptr) {
    cublasLtCreate(&g_cublaslt_handle);
  }
  if (g_cublas_workspace == nullptr) {
    cudaMalloc(&g_cublas_workspace, CUBLAS_WORKSPACE_SIZE);
  }
  if (g_cublas_prefill_handle != nullptr && g_cublas_workspace != nullptr) {
    cublasSetWorkspace(g_cublas_prefill_handle, g_cublas_workspace, CUBLAS_WORKSPACE_SIZE);
  }
  if (g_cublaslt_workspace == nullptr) {
    cudaMalloc(&g_cublaslt_workspace, kWorkspaceBytes);
  }
}

void cublas_destroy() {
  if (g_cublas_handle != nullptr) {
    cublasDestroy(g_cublas_handle);
    g_cublas_handle = nullptr;
  }
  if (g_cublas_prefill_handle != nullptr) {
    cublasDestroy(g_cublas_prefill_handle);
    g_cublas_prefill_handle = nullptr;
  }
  if (g_cublaslt_handle != nullptr) {
    cublasLtDestroy(g_cublaslt_handle);
    g_cublaslt_handle = nullptr;
  }
  if (g_cublas_workspace != nullptr) {
    cudaFree(g_cublas_workspace);
    g_cublas_workspace = nullptr;
  }
  if (g_cublaslt_workspace != nullptr) {
    cudaFree(g_cublaslt_workspace);
    g_cublaslt_workspace = nullptr;
  }
  g_algo_cache.clear();
}


void gemv_batched_qkv_cuda(const __nv_bfloat16 *Wq, const __nv_bfloat16 *Wk, const __nv_bfloat16 *Wv,
                           const __nv_bfloat16 *x, __nv_bfloat16 *q_out, __nv_bfloat16 *k_out,
                           __nv_bfloat16 *v_out, int Mq, int Mk, int K,
                           cudaStream_t stream) {
  int blocks_q = (Mq + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
  int blocks_k = (Mk + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
  size_t smem_bytes = static_cast<size_t>(K) * sizeof(__nv_bfloat16);

  gemv_handwritten_kernel<<<blocks_q, GEMV_BLOCK, smem_bytes, stream>>>(Wq, x, q_out, Mq, K);
  gemv_handwritten_kernel<<<blocks_k, GEMV_BLOCK, smem_bytes, stream>>>(Wk, x, k_out, Mk, K);
  gemv_handwritten_kernel<<<blocks_k, GEMV_BLOCK, smem_bytes, stream>>>(Wv, x, v_out, Mk, K);
}

cudaError_t gemv_cuda(const __nv_bfloat16 *A, const __nv_bfloat16 *x, __nv_bfloat16 *y, int M, int K,
               cudaStream_t stream) {
  int num_blocks = (M + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
  size_t smem_bytes = static_cast<size_t>(K) * sizeof(__nv_bfloat16);
  gemv_handwritten_kernel<<<num_blocks, GEMV_BLOCK, smem_bytes, stream>>>(A, x, y, M, K);
    return cudaGetLastError();
}

// General GEMM: Y = W @ X where W is [M, K] row-major, X is [K, N] col-major, Y is [M, N] col-major
// N=1 is equivalent to GEMV. N>1 enables batched prefill.
// Uses prefill handle (with workspace) — only called from prefill path, never under CUDA Graphs.
cudaError_t gemm_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X, __nv_bfloat16 *Y,
               int M, int N, int K, cudaStream_t stream) {
  g_gemm_graphsafe_mode = false;
  return gemm_cublaslt_impl(W, X, Y, M, N, K, stream);
}

// Graph-safe GEMM: same math as gemm_cuda but uses the workspace-free handle.
// Safe for CUDA Graph capture and decode path.
cudaError_t gemm_graphsafe_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X, __nv_bfloat16 *Y,
                          int M, int N, int K, cudaStream_t stream) {
  g_gemm_graphsafe_mode = true;
  cudaError_t status = gemm_cublaslt_impl(W, X, Y, M, N, K, stream);
  g_gemm_graphsafe_mode = false;
  return status;
}

void attention_scores_cuda(const __nv_bfloat16 *q, const __nv_bfloat16 *k_cache, __nv_bfloat16 *scores,
                           int seq_len, int head_dim, float scale,
                           cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (seq_len + block_size - 1) / block_size;
  attention_scores_kernel<<<num_blocks, block_size, 0, stream>>>(
      q, k_cache, scores, seq_len, head_dim, scale);
}

void attention_weighted_sum_cuda(const __nv_bfloat16 *weights, const __nv_bfloat16 *v_cache,
                                 __nv_bfloat16 *out, int seq_len, int head_dim,
                                 cudaStream_t stream) {
  int block_size = 128;
  int num_blocks = (head_dim + block_size - 1) / block_size;
  attention_weighted_sum_kernel<<<num_blocks, block_size, 0, stream>>>(
      weights, v_cache, out, seq_len, head_dim);
}

} // extern "C"

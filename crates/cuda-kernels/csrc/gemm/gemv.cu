#include "common.cuh"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// ============================================================================
// Hand-written GEMV: y = A @ x (row-major matrix)
// Each block processes GEMV_ROWS_PER_BLOCK rows.
// BF16×4 vectorized loads (8 bytes/thread/stride) for memory throughput.
// Warp shuffle reduction + shared memory for inter-warp reduce.
// BF16 inputs, FP32 accumulators, BF16 output.
// Graph-capture safe (no cuBLAS workspace allocation).
//
// ROWS_PER_BLOCK is the loads-in-flight knob: at K=2048 each thread issues
// exactly ONE 16B load per row (K/8/256 = 1), so 4 rows/block ran
// latency-bound at 32% of HBM peak on the H20 lm_head (nsys 2026-06-11:
// 491 us for 0.62 GB). 16 rows/block keeps every caller (lm_head-class
// M >= 100k => grid >= 6k blocks, far above SM capacity — unlike the MoE
// down-kernel grid-shrink trap) while 4x-ing per-thread loads in flight.
// ============================================================================
#define GEMV_BLOCK 256
#define GEMV_ROWS_PER_BLOCK 16
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

// Per-device cuBLAS state. Each CUDA device gets its own handles, workspaces,
// and algo cache because cuBLAS handles bind to a specific device at create
// time and cannot be reused across devices. F1+ multi-GPU rank threads each
// pin to one ordinal (via cudarc's CudaContext); the lookup keys off
// `cudaGetDevice()` so callers don't need to thread the ordinal explicitly.
//
// Single-GPU (F0 default) path: only ordinal 0 is ever populated; same
// behavior as the prior process-global state.
//
// `handle`: workspace-free, safe for CUDA Graph capture (decode path).
// `prefill_handle`: 32MB workspace, lets cuBLAS pick faster algorithms for the
//   252 GEMMs per prefill. Never used under CUDA Graphs.
// `lt_handle`: cublasLt for cublasLtMatmul.
// `algo_cache`: per-shape best algorithm chosen by heuristic / autotune.
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

struct CublasDeviceState {
  cublasHandle_t handle = nullptr;
  cublasHandle_t prefill_handle = nullptr;
  cublasLtHandle_t lt_handle = nullptr;
  void *cublas_workspace = nullptr;
  void *cublaslt_workspace = nullptr;
  std::unordered_map<GemmKey, cublasLtMatmulAlgo_t, GemmKeyHash> algo_cache;
  // Volta (sm_70) BF16↔FP16 cast scratch. Pre-allocated once, grown on
  // demand, freed at cublas_destroy. Avoids per-GEMM cudaMalloc/cudaFree
  // + the trailing cudaStreamSynchronize that guarded the cudaFree — both
  // serialized the entire decode pipeline on V100.
  void *cast_scratch = nullptr;
  size_t cast_scratch_bytes = 0;
};

static const size_t CUBLAS_WORKSPACE_SIZE = 32 * 1024 * 1024; // 32MB
static constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;

static std::mutex g_state_mutex;
static std::unordered_map<int, std::unique_ptr<CublasDeviceState>> g_per_device_state;

// Generation counter bumped on every `cublas_destroy()`. TLS caches stamp the
// generation they observed when populating; on mismatch they re-fault under
// the mutex. This prevents a worker thread from dereferencing a freed
// `CublasDeviceState` after another thread tore down + re-initialized state
// (codex R7 [P2]: per-thread TLS clear in destroy is insufficient for
// surviving worker threads).
static std::atomic<uint64_t> g_state_generation{0};

// Hot-path lookup avoids the mutex by caching the per-device pointer in TLS.
// Invalidated when `cudaGetDevice()` reports a different ordinal OR when the
// global generation moves (some other thread destroyed state).
thread_local CublasDeviceState *t_cached_state = nullptr;
thread_local int t_cached_ordinal = -1;
thread_local uint64_t t_cached_generation = 0;

static CublasDeviceState *current_device_state() {
  int ordinal = 0;
  cudaGetDevice(&ordinal);
  uint64_t gen = g_state_generation.load(std::memory_order_acquire);
  if (ordinal == t_cached_ordinal && t_cached_state != nullptr &&
      gen == t_cached_generation) {
    return t_cached_state;
  }
  std::lock_guard<std::mutex> lock(g_state_mutex);
  auto it = g_per_device_state.find(ordinal);
  if (it == g_per_device_state.end()) {
    return nullptr;
  }
  t_cached_ordinal = ordinal;
  t_cached_state = it->second.get();
  t_cached_generation = gen;
  return t_cached_state;
}

static int device_compute_major();

// Volta (sm_70, e.g. V100) has no BF16 tensor cores and no BF16 cuBLAS compute
// at all — only FP16/FP32. A BF16 cublasGemmEx faults at runtime with
// cudaErrorNotSupported, and the fault is asynchronous (the API returns
// SUCCESS, surfacing only at the next stream sync), so a post-hoc retry cannot
// catch it without a per-call cudaStreamSynchronize. Instead, value-convert the
// BF16 operands to FP16, run an FP16 GEMM (V100 has FP16 tensor cores), and
// value-convert the FP16 result back to BF16. Uses a pre-allocated per-device
// scratch (grown on demand, freed at cublas_destroy) — no per-call alloc or
// sync. This is the Volta-only path, never the sm_80+ hot path.
//
// BF16 and FP16 differ in exponent/mantissa width (BF16: 8 exp + 7 mantissa,
// same exponent range as FP32; FP16: 5 exp + 10 mantissa, narrower range), so a
// raw byte copy reinterprets the bits and corrupts every value (e.g. BF16 1.0
// == 0x3F80 reads as FP16 1.875). Promote through FP32 with round-to-nearest.
__global__ void convert_bf16_to_fp16_kernel(const __nv_bfloat16 *__restrict__ in,
                                            __half *__restrict__ out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = __float2half_rn(__bfloat162float(in[i]));
  }
}

__global__ void convert_fp16_to_bf16_kernel(const __half *__restrict__ in,
                                            __nv_bfloat16 *__restrict__ out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = __float2bfloat16_rn(__half2float(in[i]));
  }
}

// Grow the per-device BF16↔FP16 cast scratch to at least `needed` bytes.
// Common case (already large enough) is lock-free; the grow path takes
// `g_state_mutex` with double-checked locking so concurrent GEMM calls on
// the same device don't race. Scratch is freed at cublas_destroy.
static cudaError_t ensure_cast_scratch(CublasDeviceState *state, size_t needed) {
  if (state->cast_scratch_bytes >= needed) {
    return cudaSuccess;
  }
  std::lock_guard<std::mutex> lock(g_state_mutex);
  if (state->cast_scratch_bytes >= needed) {
    return cudaSuccess;
  }
  if (state->cast_scratch != nullptr) {
    cudaFree(state->cast_scratch);
    state->cast_scratch = nullptr;
    state->cast_scratch_bytes = 0;
  }
  // Round up to the next 1 MB boundary to avoid repeated tiny grows.
  size_t alloc_bytes = ((needed + (1 << 20) - 1) / (1 << 20)) * (1 << 20);
  if (cudaMalloc(&state->cast_scratch, alloc_bytes) != cudaSuccess) {
    return cudaErrorMemoryAllocation;
  }
  state->cast_scratch_bytes = alloc_bytes;
  return cudaSuccess;
}

static cudaError_t gemm_fp16_cast_cuda(CublasDeviceState *state,
                                       const __nv_bfloat16 *W, const __nv_bfloat16 *X,
                                       __nv_bfloat16 *Y, int M, int N, int K,
                                       cudaStream_t stream, cublasHandle_t handle) {
  if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }
  size_t w_elems = static_cast<size_t>(K) * M;
  size_t x_elems = static_cast<size_t>(K) * N;
  size_t y_elems = static_cast<size_t>(M) * N;
  size_t needed = (w_elems + x_elems + y_elems) * sizeof(__half);
  if (ensure_cast_scratch(state, needed) != cudaSuccess) {
    return cudaErrorMemoryAllocation;
  }
  // Slice the pre-allocated scratch into W/X/Y regions (no per-call alloc).
  __half *W_f16 = static_cast<__half *>(state->cast_scratch);
  __half *X_f16 = W_f16 + w_elems;
  __half *Y_f16 = X_f16 + x_elems;
  constexpr int CONV_BLOCK = 256;
  // Value-convert BF16 operands to FP16 (not a raw byte copy — see the
  // kernel comments above). Launch errors are sticky, so cudaGetLastError
  // after each.
  convert_bf16_to_fp16_kernel<<<(w_elems + CONV_BLOCK - 1) / CONV_BLOCK, CONV_BLOCK, 0,
                                stream>>>(W, W_f16, w_elems);
  cudaError_t cerr = cudaGetLastError();
  if (cerr != cudaSuccess) return cerr;
  convert_bf16_to_fp16_kernel<<<(x_elems + CONV_BLOCK - 1) / CONV_BLOCK, CONV_BLOCK, 0,
                                stream>>>(X, X_f16, x_elems);
  cerr = cudaGetLastError();
  if (cerr != cudaSuccess) return cerr;
  const float h_alpha = 1.0f, h_beta = 0.0f;
  // FP16 GEMM with FP32 accumulate + tensor cores: native on V100.
  cublasStatus_t stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                                     &h_alpha, W_f16, CUDA_R_16F, K, X_f16,
                                     CUDA_R_16F, K, &h_beta, Y_f16, CUDA_R_16F,
                                     M, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }
  convert_fp16_to_bf16_kernel<<<(y_elems + CONV_BLOCK - 1) / CONV_BLOCK, CONV_BLOCK, 0,
                                stream>>>(Y_f16, Y, y_elems);
  return cudaGetLastError();
}

static cudaError_t gemm_cublas_fallback(CublasDeviceState *state,
                                        const __nv_bfloat16 *W, const __nv_bfloat16 *X,
                                        __nv_bfloat16 *Y, int M, int N, int K,
                                        cudaStream_t stream, cublasHandle_t handle) {
  // Volta (sm_70, e.g. V100) has no BF16 compute at all — only FP16/FP32. A
  // BF16 cublasGemmEx faults asynchronously at runtime, so take the FP16-cast
  // path directly: BF16 operands → FP16 GEMM → BF16 result. Compute major is
  // cached per device (single cudaGetDeviceProperties at first use).
  if (device_compute_major() <= 7) {
    return gemm_fp16_cast_cuda(state, W, X, Y, M, N, K, stream, handle);
  }
  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;
  if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }
  // CUBLAS_GEMM_DEFAULT_TENSOR_OP lets cuBLAS pick a tensor-core kernel. On
  // hardware without BF16 tensor cores (Volta/sm_70 V100, only FP16/FP32),
  // cuBLAS still launches a BF16 tensor-core kernel that faults at runtime
  // with cudaErrorNotSupported (sticky). CUBLAS_GEMM_DEFAULT falls back to
  // CUDA-core math, which is correct on every device — slower but the only
  // correct path for BF16 on Volta. Retry once, only on NOT_SUPPORTED.
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
  cudaError_t err = cudaGetLastError();
  if (err == cudaErrorNotSupported) {
    // CUBLAS_GEMM_DEFAULT_TENSOR_OP let cuBLAS pick a BF16 tensor-core
    // kernel that faults at runtime on hardware without BF16 tensor cores
    // (Volta/sm_70 V100, only FP16/FP32). Retry once with CUBLAS_GEMM_DEFAULT
    // (CUDA-core math), which is correct on every device.
    if (cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     M, N, K,
                     &h_alpha,
                     W, CUDA_R_16BF, K,
                     X, CUDA_R_16BF, K,
                     &h_beta,
                     Y, CUDA_R_16BF, M,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
      return cudaErrorUnknown;
    }
    err = cudaGetLastError();
  }
  return err;
}

// M_pf-gemm Phase 0: at first miss for a given (M,N,K), benchmark
// all heuristic-returned algos (5 iters each) and cache the fastest.
// cuBLAS heuristic top-1 is optimized for "average cost across many
// shapes"; for a specific shape the best algo is often at index
// 1-3. Off by default; opt in with INFER_GEMM_AUTOTUNE=1.
// See the H_LP3 finding in
// docs/experience/wins/2026-05-07-h_lp3-diagnosed-cutlass-small-tile-gemm-bottleneck.md.
static bool gemm_autotune_enabled() {
  static const bool enabled = []() {
    const char *env = std::getenv("INFER_GEMM_AUTOTUNE");
    if (env == nullptr) return false;
    return std::strcmp(env, "1") == 0 ||
           std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0 ||
           std::strcmp(env, "on") == 0 || std::strcmp(env, "ON") == 0;
  }();
  return enabled;
}

static bool deterministic_gemm_enabled() {
  // INFER_DETERMINISTIC=1 forces every BF16 GEMM through the cublasGemmEx
  // fallback so B=1 (graphsafe path) and B>=2 (cublasLt path) hit the same
  // cuBLAS API. Without this, B=1 falls back to cublasGemmEx (cache miss
  // on graph capture) while B>=2 uses cublasLtMatmul, and the two
  // numerical paths diverge per-row even in greedy decoding (the deferred
  // bug tracked in 2026-04-13-batched-decode-high-concurrency.md).
  // Cached on first read; no perf impact on the fast path.
  static const bool enabled = []() {
    const char *env = std::getenv("INFER_DETERMINISTIC");
    if (env == nullptr) return false;
    // Match the explicit truthy spellings; reject "off"/"OFF"/empty.
    return std::strcmp(env, "1") == 0 ||
           std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0 ||
           std::strcmp(env, "on") == 0 || std::strcmp(env, "ON") == 0;
  }();
  return enabled;
}

// Compute capability major of the currently-active CUDA device, cached per
// device ordinal. A single cudaGetDeviceProperties runs at first use per
// device; subsequent calls read the cache. Returns 999 on a lookup failure so
// the caller conservatively takes the fast (cublasLt) path.
static int device_compute_major() {
  static constexpr int kUnknownMajor = 999;
  static std::atomic<int> cached_ordinal{-1};
  static std::atomic<int> cached_major{kUnknownMajor};
  int ordinal = 0;
  if (cudaGetDevice(&ordinal) != cudaSuccess) {
    return kUnknownMajor;
  }
  if (cached_ordinal.load(std::memory_order_acquire) == ordinal) {
    return cached_major.load(std::memory_order_acquire);
  }
  cudaDeviceProp prop{};
  int major = kUnknownMajor;
  if (cudaGetDeviceProperties(&prop, ordinal) == cudaSuccess) {
    major = prop.major;
  }
  cached_ordinal.store(ordinal, std::memory_order_release);
  cached_major.store(major, std::memory_order_release);
  return major;
}

static cudaError_t gemm_cublaslt_impl(const __nv_bfloat16 *W, const __nv_bfloat16 *X,
                                      __nv_bfloat16 *Y, int M, int N, int K,
                                      cudaStream_t stream, bool graphsafe) {
  CublasDeviceState *state = current_device_state();
  if (state == nullptr) {
    return cudaErrorNotReady;
  }

  // Volta (sm_70, e.g. V100) has no BF16 tensor cores — only FP16/FP32.
  // cublasLt's heuristic still selects a BF16 tensor-core algo that faults at
  // runtime with cudaErrorNotSupported, and the fault is asynchronous (the API
  // returns SUCCESS, surfacing only at the next stream sync). Route BF16 GEMMs
  // directly to the CUDA-core cublasGemmEx fallback on these devices: it skips
  // the heuristic entirely (never caching a bad algo) and needs no per-call
  // sync, so the sm_80+ hot path is untouched. Compute major is cached per
  // device (a single cudaGetDeviceProperties at first use) — the uncached
  // per-step lookup caused a -77% decode regression (2026-06-12).
  if (device_compute_major() <= 7) {
    return gemm_cublas_fallback(state, W, X, Y, M, N, K, stream,
                                graphsafe ? state->handle : state->prefill_handle);
  }

  if (deterministic_gemm_enabled()) {
    // Bypass cublasLt entirely. Graph-safe callers keep using the
    // workspace-free handle for CUDA Graph capture; eager callers use the
    // workspace-backed prefill handle. In deterministic decode, Rust splits
    // BF16 batched GEMM into per-row graph-safe N=1 calls, so every row hits
    // this same cublasGemmEx path.
    return gemm_cublas_fallback(state, W, X, Y, M, N, K, stream,
                                graphsafe ? state->handle : state->prefill_handle);
  }

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

  auto algo_it = state->algo_cache.find(key);
  if (algo_it == state->algo_cache.end()) {
    if (graphsafe) {
      cublasLtMatrixLayoutDestroy(y_desc);
      cublasLtMatrixLayoutDestroy(x_desc);
      cublasLtMatrixLayoutDestroy(w_desc);
      cublasLtMatmulDescDestroy(operation_desc);
      return gemm_cublas_fallback(state, W, X, Y, M, N, K, stream, state->handle);
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
        state->lt_handle, operation_desc, w_desc, x_desc, y_desc, y_desc,
        preference, 8, heuristic_results, &returned_algo_count);
    cublasLtMatmulPreferenceDestroy(preference);
    preference = nullptr;

    if (heuristic_status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0) {
      cublasLtMatmulAlgo_t selected_algo = heuristic_results[0].algo;
      // M_pf-gemm Phase 0: when INFER_GEMM_AUTOTUNE=1, benchmark all
      // returned algos at this (M,N,K) and keep the fastest. One-time
      // cost amortized across all subsequent calls of this shape.
      // Suppressed during CUDA Graph capture: cudaEventRecord on the
      // capture stream is not legal there, and graphsafe=false can be
      // active inside graph capture (e.g. batched-decode N>=2 path
      // routed through Bf16CublasGemm — observed crash 2026-05-07).
      cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(stream, &capture_status);
      bool inside_graph_capture =
          capture_status != cudaStreamCaptureStatusNone;
      if (gemm_autotune_enabled() && returned_algo_count > 1 &&
          !inside_graph_capture) {
        cudaEvent_t e_start = nullptr;
        cudaEvent_t e_stop = nullptr;
        if (cudaEventCreate(&e_start) == cudaSuccess &&
            cudaEventCreate(&e_stop) == cudaSuccess) {
          float best_ms = 0.0f;
          bool have_best = false;
          for (int i = 0; i < returned_algo_count; ++i) {
            cublasLtMatmulAlgo_t &candidate = heuristic_results[i].algo;
            // Warmup once
            if (cublasLtMatmul(state->lt_handle, operation_desc,
                               &h_alpha, W, w_desc, X, x_desc, &h_beta,
                               Y, y_desc, Y, y_desc, &candidate,
                               state->cublaslt_workspace, kWorkspaceBytes,
                               stream) != CUBLAS_STATUS_SUCCESS) {
              continue;
            }
            // Measure 5 iters
            cudaEventRecord(e_start, stream);
            bool failed = false;
            for (int it = 0; it < 5; ++it) {
              if (cublasLtMatmul(state->lt_handle, operation_desc,
                                 &h_alpha, W, w_desc, X, x_desc, &h_beta,
                                 Y, y_desc, Y, y_desc, &candidate,
                                 state->cublaslt_workspace, kWorkspaceBytes,
                                 stream) != CUBLAS_STATUS_SUCCESS) {
                failed = true;
                break;
              }
            }
            if (failed) continue;
            cudaEventRecord(e_stop, stream);
            if (cudaEventSynchronize(e_stop) != cudaSuccess) continue;
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, e_start, e_stop) != cudaSuccess) {
              continue;
            }
            if (!have_best || ms < best_ms) {
              best_ms = ms;
              selected_algo = candidate;
              have_best = true;
            }
          }
          cudaEventDestroy(e_start);
          cudaEventDestroy(e_stop);
        }
      }
      algo_it = state->algo_cache.emplace(key, selected_algo).first;
    } else {
      cublasLtMatrixLayoutDestroy(y_desc);
      cublasLtMatrixLayoutDestroy(x_desc);
      cublasLtMatrixLayoutDestroy(w_desc);
      cublasLtMatmulDescDestroy(operation_desc);
      return gemm_cublas_fallback(state, W, X, Y, M, N, K, stream, state->prefill_handle);
    }
  }

  cudaError_t err = cudaSuccess;
  if (cublasLtMatmul(state->lt_handle, operation_desc,
                     &h_alpha,
                     W, w_desc,
                     X, x_desc,
                     &h_beta,
                     Y, y_desc,
                     Y, y_desc,
                     &algo_it->second,
                     state->cublaslt_workspace,
                     kWorkspaceBytes,
                     stream) != CUBLAS_STATUS_SUCCESS) {
    err = cudaErrorUnknown;
  } else {
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    cublasLtMatrixLayoutDestroy(y_desc);
    cublasLtMatrixLayoutDestroy(x_desc);
    cublasLtMatrixLayoutDestroy(w_desc);
    cublasLtMatmulDescDestroy(operation_desc);
    // A failed launch (or an opaque cudaErrorUnknown from cublasLt) falls back
    // to cublasGemmEx, which retries with CUBLAS_GEMM_DEFAULT (CUDA-core math)
    // on hardware without BF16 tensor cores (Volta/sm_70 V100).
    return gemm_cublas_fallback(state, W, X, Y, M, N, K, stream, state->prefill_handle);
  }

  cublasLtMatrixLayoutDestroy(y_desc);
  cublasLtMatrixLayoutDestroy(x_desc);
  cublasLtMatrixLayoutDestroy(w_desc);
  cublasLtMatmulDescDestroy(operation_desc);
  return cudaSuccess;
}

extern "C" {

// Initialize cuBLAS state for the currently-active CUDA device. Idempotent;
// safe to call multiple times. Caller MUST set the CUDA context to the
// intended device before calling (cudarc's `CudaContext::new(ordinal)` does
// this on context creation).
void cublas_init() {
  int ordinal = 0;
  cudaGetDevice(&ordinal);

  std::lock_guard<std::mutex> lock(g_state_mutex);
  auto &slot = g_per_device_state[ordinal];
  if (slot != nullptr) {
    return;
  }
  auto state = std::make_unique<CublasDeviceState>();
  cublasCreate(&state->handle);
  cublasSetMathMode(state->handle, CUBLAS_TENSOR_OP_MATH);
  cublasCreate(&state->prefill_handle);
  cublasSetMathMode(state->prefill_handle, CUBLAS_TENSOR_OP_MATH);
  cublasLtCreate(&state->lt_handle);
  cudaMalloc(&state->cublas_workspace, CUBLAS_WORKSPACE_SIZE);
  if (state->prefill_handle != nullptr && state->cublas_workspace != nullptr) {
    cublasSetWorkspace(state->prefill_handle, state->cublas_workspace, CUBLAS_WORKSPACE_SIZE);
  }
  cudaMalloc(&state->cublaslt_workspace, kWorkspaceBytes);
  slot = std::move(state);
}

// Tear down cuBLAS state for ALL devices the process has initialized.
//
// CALLER CONTRACT: every thread that may issue GEMM (gemm_cuda /
// gemm_graphsafe_cuda / autotune_*) MUST be quiesced (joined, or proven
// idle) before this function is called. The hot path is intentionally
// lock-free past the TLS cache; we narrow but cannot fully eliminate the
// race window without an RWLock-on-every-call cost we don't want to pay
// in steady state. In practice this runs at process exit when worker
// threads have stopped, or in a hot-swap path where the caller already
// synchronizes a full quiesce-then-reinit cycle.
//
// Implementation details:
// - Generation is bumped BEFORE freeing handles so a concurrent reader
//   that loads `g_state_generation` after the bump observes mismatch and
//   re-faults under the mutex (where it blocks on us, then sees an empty
//   map). A reader that loaded the OLD generation before the bump is the
//   case the caller-contract above must rule out — we narrow the window
//   but don't eliminate it.
// - Each handle/workspace is destroyed under the device that owns it
//   (cuBLAS handles are bound to the device active at `cublasCreate`
//   time, but `cublasDestroy` works from any current device).
void cublas_destroy() {
  std::lock_guard<std::mutex> lock(g_state_mutex);
  // Bump generation FIRST under the mutex so any reader that loads gen
  // after this point sees the mismatch and re-faults. The actual destroy
  // happens after the bump so readers that have already passed the gen
  // check see live state until destroyed (race window narrows to the
  // gap between "reader loaded old gen" and "reader dereferences ptr").
  g_state_generation.fetch_add(1, std::memory_order_release);
  for (auto &kv : g_per_device_state) {
    auto &state = kv.second;
    if (state->handle != nullptr) {
      cublasDestroy(state->handle);
    }
    if (state->prefill_handle != nullptr) {
      cublasDestroy(state->prefill_handle);
    }
    if (state->lt_handle != nullptr) {
      cublasLtDestroy(state->lt_handle);
    }
    if (state->cublas_workspace != nullptr) {
      cudaFree(state->cublas_workspace);
    }
    if (state->cublaslt_workspace != nullptr) {
      cudaFree(state->cublaslt_workspace);
    }
    if (state->cast_scratch != nullptr) {
      cudaFree(state->cast_scratch);
    }
  }
  g_per_device_state.clear();
  // Clear our own TLS for fast-path consistency on this thread.
  t_cached_state = nullptr;
  t_cached_ordinal = -1;
  t_cached_generation = 0;
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
  return gemm_cublaslt_impl(W, X, Y, M, N, K, stream, /*graphsafe=*/false);
}

// GEMM with FP16 weights (already dequantized) and BF16 activations/output.
// On sm_70 this skips the BF16→FP16 weight cast that `gemm_cuda` does, since
// the W4A16 dequant already produced FP16 weights.
cudaError_t gemm_fp16_weight_cuda(const __half *W_f16, const __nv_bfloat16 *X,
                                  __nv_bfloat16 *Y, int M, int N, int K,
                                  cudaStream_t stream) {
  CublasDeviceState *state = current_device_state();
  if (state == nullptr) return cudaErrorNotReady;
  cublasHandle_t handle = state->prefill_handle;
  if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

  size_t x_elems = static_cast<size_t>(K) * N;
  size_t y_elems = static_cast<size_t>(M) * N;
  size_t needed = (x_elems + y_elems) * sizeof(__half);
  if (ensure_cast_scratch(state, needed) != cudaSuccess) return cudaErrorMemoryAllocation;

  __half *X_f16 = static_cast<__half *>(state->cast_scratch);
  __half *Y_f16 = X_f16 + x_elems;
  constexpr int CONV_BLOCK = 256;

  convert_bf16_to_fp16_kernel<<<(x_elems + CONV_BLOCK - 1) / CONV_BLOCK, CONV_BLOCK, 0,
                                stream>>>(X, X_f16, x_elems);
  cudaError_t cerr = cudaGetLastError();
  if (cerr != cudaSuccess) return cerr;

  const float h_alpha = 1.0f, h_beta = 0.0f;
  cublasStatus_t stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                                     &h_alpha, W_f16, CUDA_R_16F, K, X_f16,
                                     CUDA_R_16F, K, &h_beta, Y_f16, CUDA_R_16F,
                                     M, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

  convert_fp16_to_bf16_kernel<<<(y_elems + CONV_BLOCK - 1) / CONV_BLOCK, CONV_BLOCK, 0,
                                stream>>>(Y_f16, Y, y_elems);
  return cudaGetLastError();
}

cudaError_t gemm_bf16_f32_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X, float *Y,
                               int M, int N, int K, cudaStream_t stream) {
  CublasDeviceState *state = current_device_state();
  if (state == nullptr || state->lt_handle == nullptr) return cudaErrorNotReady;

  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t wl = nullptr, xl = nullptr, yl = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  const float alpha = 1.0f, beta = 0.0f;
  cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  cublasStatus_t status = cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transa, sizeof(transa));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
                                            &transb, sizeof(transb));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&wl, CUDA_R_16BF, K, M, K);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&xl, CUDA_R_16BF, K, N, K);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&yl, CUDA_R_32F, M, N, M);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulPreferenceCreate(&pref);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kWorkspaceBytes, sizeof(kWorkspaceBytes));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulAlgoGetHeuristic(state->lt_handle, op, wl, xl, yl, yl,
                                            pref, 1, &heuristic, &returned);
  if (status == CUBLAS_STATUS_SUCCESS && returned > 0)
    status = cublasLtMatmul(state->lt_handle, op, &alpha, W, wl, X, xl, &beta,
                            Y, yl, Y, yl, &heuristic.algo,
                            state->cublaslt_workspace, kWorkspaceBytes, stream);
  if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
  if (yl != nullptr) cublasLtMatrixLayoutDestroy(yl);
  if (xl != nullptr) cublasLtMatrixLayoutDestroy(xl);
  if (wl != nullptr) cublasLtMatrixLayoutDestroy(wl);
  if (op != nullptr) cublasLtMatmulDescDestroy(op);
  return status == CUBLAS_STATUS_SUCCESS && returned > 0 ? cudaGetLastError()
                                                         : cudaErrorUnknown;
}

// Graph-safe GEMM: same math as gemm_cuda but uses the workspace-free handle.
// Safe for CUDA Graph capture and decode path.
cudaError_t gemm_graphsafe_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X, __nv_bfloat16 *Y,
                          int M, int N, int K, cudaStream_t stream) {
  return gemm_cublaslt_impl(W, X, Y, M, N, K, stream, /*graphsafe=*/true);
}

// Benchmark all cublasLt heuristic algorithms for (M,N,K) and cache the fastest.
// Called during warmup before CUDA Graph capture.
cudaError_t autotune_gemm_cuda(int M, int N, int K, cudaStream_t stream) {
  CublasDeviceState *state = current_device_state();
  if (state == nullptr || state->lt_handle == nullptr ||
      state->cublaslt_workspace == nullptr)
    return cudaErrorNotReady;

  GemmKey key{M, N, K};

  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t wl = nullptr, xl = nullptr, yl = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  const float alpha = 1.0f, beta = 0.0f;

  if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
    return cudaErrorUnknown;

  cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

  cublasLtMatrixLayoutCreate(&wl, CUDA_R_16BF, K, M, K);
  cublasLtMatrixLayoutCreate(&xl, CUDA_R_16BF, K, N, K);
  cublasLtMatrixLayoutCreate(&yl, CUDA_R_16BF, M, N, M);

  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &kWorkspaceBytes, sizeof(kWorkspaceBytes));

  cublasLtMatmulHeuristicResult_t results[8];
  int count = 0;
  cublasLtMatmulAlgoGetHeuristic(state->lt_handle, op, wl, xl, yl, yl,
                                 pref, 8, results, &count);
  cublasLtMatmulPreferenceDestroy(pref);

  if (count <= 0) {
    cublasLtMatrixLayoutDestroy(yl);
    cublasLtMatrixLayoutDestroy(xl);
    cublasLtMatrixLayoutDestroy(wl);
    cublasLtMatmulDescDestroy(op);
    return cudaErrorUnknown;
  }

  if (count == 1) {
    state->algo_cache[key] = results[0].algo;
    cublasLtMatrixLayoutDestroy(yl);
    cublasLtMatrixLayoutDestroy(xl);
    cublasLtMatrixLayoutDestroy(wl);
    cublasLtMatmulDescDestroy(op);
    return cudaSuccess;
  }

  // Allocate temp buffers for benchmarking
  __nv_bfloat16 *d_W = nullptr, *d_X = nullptr, *d_Y = nullptr;
  size_t w_bytes = (size_t)M * K * 2;
  size_t x_bytes = (size_t)K * N * 2;
  size_t y_bytes = (size_t)M * N * 2;

  if (cudaMalloc(&d_W, w_bytes) != cudaSuccess ||
      cudaMalloc(&d_X, x_bytes) != cudaSuccess ||
      cudaMalloc(&d_Y, y_bytes) != cudaSuccess) {
    if (d_W) cudaFree(d_W);
    if (d_X) cudaFree(d_X);
    if (d_Y) cudaFree(d_Y);
    state->algo_cache[key] = results[0].algo;
    cublasLtMatrixLayoutDestroy(yl);
    cublasLtMatrixLayoutDestroy(xl);
    cublasLtMatrixLayoutDestroy(wl);
    cublasLtMatmulDescDestroy(op);
    return cudaSuccess;
  }
  cudaMemsetAsync(d_W, 0, w_bytes, stream);
  cudaMemsetAsync(d_X, 0, x_bytes, stream);

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  float best_ms = 1e30f;
  int best_idx = 0;

  for (int i = 0; i < count; i++) {
    bool failed = false;
    // Warmup
    for (int w = 0; w < 3; w++) {
      if (cublasLtMatmul(state->lt_handle, op, &alpha,
                         d_W, wl, d_X, xl, &beta, d_Y, yl, d_Y, yl,
                         &results[i].algo,
                         state->cublaslt_workspace, kWorkspaceBytes,
                         stream) != CUBLAS_STATUS_SUCCESS) {
        failed = true;
        break;
      }
    }
    if (failed) continue;

    // Timed iterations
    cudaEventRecord(ev_start, stream);
    for (int t = 0; t < 10; t++) {
      cublasLtMatmul(state->lt_handle, op, &alpha,
                     d_W, wl, d_X, xl, &beta, d_Y, yl, d_Y, yl,
                     &results[i].algo,
                     state->cublaslt_workspace, kWorkspaceBytes,
                     stream);
    }
    cudaEventRecord(ev_stop, stream);
    cudaEventSynchronize(ev_stop);

    float ms;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    if (ms < best_ms) {
      best_ms = ms;
      best_idx = i;
    }
  }

  state->algo_cache[key] = results[best_idx].algo;

  cudaFree(d_W);
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
  cublasLtMatrixLayoutDestroy(yl);
  cublasLtMatrixLayoutDestroy(xl);
  cublasLtMatrixLayoutDestroy(wl);
  cublasLtMatmulDescDestroy(op);
  return cudaSuccess;
}

// Autotune all GEMM shapes currently in the heuristic cache (for the
// currently-active CUDA device).
// Replaces heuristic-selected algorithms with benchmarked optimal ones.
void autotune_all_cached_gemms_cuda(cudaStream_t stream) {
  CublasDeviceState *state = current_device_state();
  if (state == nullptr) {
    return;
  }
  std::vector<GemmKey> keys;
  keys.reserve(state->algo_cache.size());
  for (auto &kv : state->algo_cache) {
    keys.push_back(kv.first);
  }
  for (auto &k : keys) {
    state->algo_cache.erase(k);
    autotune_gemm_cuda(k.M, k.N, k.K, stream);
  }
}

} // extern "C"

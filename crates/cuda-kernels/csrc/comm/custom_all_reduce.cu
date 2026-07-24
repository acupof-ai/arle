// ARLE C ABI around the vendored sgl-kernel/vLLM CustomAllreduce (one-shot /
// two-shot cross-device reduction over IPC-shared buffers), plus an ARLE-derived
// one-shot all-gather kernel reusing the same RankData/Signal/barrier framework.
//
// Scope: the isolated comm bench. Bootstrap is file-rendezvous based (atomic rename), one
// context per rank process; production integration (TpComm) would reuse the
// kernels but bootstrap over the multiproc relay instead.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <string>

#include "comm/custom_all_reduce.cuh"

namespace {

constexpr int kMaxWorld = 8;

struct ArleCarCtx {
  int rank = 0;
  int world = 0;
  size_t max_bytes = 0;
  // Owned device allocations.
  void* signal_region = nullptr;  // sizeof(Signal) + tmp staging (2stage)
  void* input = nullptr;          // registered AR/AG input (IPC-shared)
  void* output = nullptr;         // local result buffer
  void* rank_data = nullptr;      // RankData slots for the CustomAllreduce
  sglang::CustomAllreduce* car = nullptr;
};

// One IPC record per rank in the rendezvous dir: signal-region + input handles.
struct IpcRecord {
  cudaIpcMemHandle_t signal;
  cudaIpcMemHandle_t input;
};

bool write_atomic(const std::string& path, const void* data, size_t len) {
  std::string tmp = path + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) return false;
  bool ok = fwrite(data, 1, len, f) == len;
  ok = (fclose(f) == 0) && ok;
  if (!ok) return false;
  return rename(tmp.c_str(), path.c_str()) == 0;
}

bool read_blocking(const std::string& path, void* data, size_t len, int timeout_s) {
  for (int waited_ms = 0; waited_ms < timeout_s * 1000;) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f) {
      bool ok = fread(data, 1, len, f) == len;
      fclose(f);
      if (ok) return true;
    }
    // 2ms poll: rendezvous files appear once per process lifetime.
    struct timespec ts = {0, 2 * 1000 * 1000};
    nanosleep(&ts, nullptr);
    waited_ms += 2;
  }
  return false;
}

// ── One-shot all-gather: same Signal/RankData framework as the 1stage
// reduce — start barrier (peers' inputs visible), strided P2P reads laying
// out [world × per_rank] contiguously, end barrier (nobody overwrites their
// input while a peer still reads it). ARLE-authored, derived from
// cross_device_reduce_1stage.
template <typename T, int ngpus>
__global__ void __launch_bounds__(sglang::kMaxThreadsPerBlock, 1) cross_device_allgather_1stage(
    sglang::RankData* _dp,
    sglang::RankSignals sg,
    sglang::Signal* self_sg,
    T* __restrict__ result,
    int rank,
    int per_rank /* packed elems contributed by each rank */) {
  using P = typename sglang::packed_t<T>::P;
  auto dp = *_dp;
  sglang::multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  const int total = per_rank * ngpus;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
    const int src = idx / per_rank;
    const int off = idx % per_rank;
    ((P*)result)[idx] = ((const P*)dp.ptrs[src])[off];
  }
  sglang::multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);
}

// Deterministic bf16 fill pattern, distinct per (rank, idx).
__global__ void fill_pattern_bf16(__nv_bfloat16* buf, int elems, int seed) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += gridDim.x * blockDim.x) {
    // Small magnitudes: keep bf16 sums exactly representable order-independent
    // enough for tolerance checks while staying non-trivial.
    float v = ((i % 31) - 15) * 0.0625f + (seed % 7) * 0.03125f;
    buf[i] = __float2bfloat16(v);
  }
}

// 1-thread dependency hook: makes iteration k+1's input depend on iteration
// k's output, so a timing loop measures EXPOSED latency, not pipelined issue.
__global__ void chain_touch(__nv_bfloat16* output, __nv_bfloat16* input) {
  input[0] = output[0];
}

}  // namespace

extern "C" {

// Bootstrap one rank's context: allocate the IPC-shared regions, exchange
// handles via `dir` (atomic-rename rendezvous files), open peers, build the
// CustomAllreduce and register the shared input buffer set. Returns null on
// failure (diagnostics on stderr).
void* arle_car_bootstrap(int rank, int world, const char* dir, size_t max_bytes) {
  if (world < 2 || world > kMaxWorld || rank < 0 || rank >= world) {
    fprintf(stderr, "[arle_car] bad rank/world %d/%d\n", rank, world);
    return nullptr;
  }
  auto* ctx = new ArleCarCtx();
  ctx->rank = rank;
  ctx->world = world;
  ctx->max_bytes = max_bytes;

  const size_t signal_bytes = sizeof(sglang::Signal) + max_bytes;  // + 2stage tmp
  CHECK_CUDA_SUCCESS(cudaMalloc(&ctx->signal_region, signal_bytes));
  CHECK_CUDA_SUCCESS(cudaMemset(ctx->signal_region, 0, signal_bytes));
  CHECK_CUDA_SUCCESS(cudaMalloc(&ctx->input, max_bytes));
  CHECK_CUDA_SUCCESS(cudaMemset(ctx->input, 0, max_bytes));
  CHECK_CUDA_SUCCESS(cudaMalloc(&ctx->output, max_bytes));
  // 64 RankData slots — the bench registers exactly one buffer set.
  CHECK_CUDA_SUCCESS(cudaMalloc(&ctx->rank_data, 64 * sizeof(sglang::RankData)));

  IpcRecord mine;
  CHECK_CUDA_SUCCESS(cudaIpcGetMemHandle(&mine.signal, ctx->signal_region));
  CHECK_CUDA_SUCCESS(cudaIpcGetMemHandle(&mine.input, ctx->input));
  std::string base(dir);
  if (!write_atomic(base + "/car_rank" + std::to_string(rank) + ".ipc", &mine, sizeof(mine))) {
    fprintf(stderr, "[arle_car] rank %d rendezvous write failed: %s\n", rank, strerror(errno));
    return nullptr;
  }

  sglang::Signal* signals[kMaxWorld] = {nullptr};
  void* input_ptrs[kMaxWorld] = {nullptr};
  for (int r = 0; r < world; r++) {
    if (r == rank) {
      signals[r] = reinterpret_cast<sglang::Signal*>(ctx->signal_region);
      input_ptrs[r] = ctx->input;
      continue;
    }
    IpcRecord rec;
    if (!read_blocking(base + "/car_rank" + std::to_string(r) + ".ipc", &rec, sizeof(rec), 120)) {
      fprintf(stderr, "[arle_car] rank %d timed out waiting for rank %d rendezvous\n", rank, r);
      return nullptr;
    }
    void* sig = nullptr;
    void* inp = nullptr;
    CHECK_CUDA_SUCCESS(cudaIpcOpenMemHandle(&sig, rec.signal, cudaIpcMemLazyEnablePeerAccess));
    CHECK_CUDA_SUCCESS(cudaIpcOpenMemHandle(&inp, rec.input, cudaIpcMemLazyEnablePeerAccess));
    signals[r] = reinterpret_cast<sglang::Signal*>(sig);
    input_ptrs[r] = inp;
  }

  ctx->car = new sglang::CustomAllreduce(
      signals, ctx->rank_data, 64 * sizeof(sglang::RankData), rank, world, /*full_nvlink=*/true);
  ctx->car->register_buffer(input_ptrs);
  return ctx;
}

// Raw device pointers for the Rust side (NCCL arms reuse the same buffers).
uint64_t arle_car_input_ptr(void* h) {
  return reinterpret_cast<uint64_t>(static_cast<ArleCarCtx*>(h)->input);
}
uint64_t arle_car_output_ptr(void* h) {
  return reinterpret_cast<uint64_t>(static_cast<ArleCarCtx*>(h)->output);
}

// force_algo: 0 = upstream auto threshold, 1 = one-shot, 2 = two-shot.
// (Upstream selects via the SGLANG_CUSTOM_ALLREDUCE_ALGO env; setenv here keeps
// the vendored dispatch untouched.)
CUresult arle_car_allreduce_bf16(
    void* h, CUstream stream, int elems, int threads, int block_limit, int force_algo) {
  auto* ctx = static_cast<ArleCarCtx*>(h);
  if (force_algo == 1) {
    setenv("SGLANG_CUSTOM_ALLREDUCE_ALGO", "1stage", 1);
  } else if (force_algo == 2) {
    setenv("SGLANG_CUSTOM_ALLREDUCE_ALGO", "2stage", 1);
  } else {
    unsetenv("SGLANG_CUSTOM_ALLREDUCE_ALGO");
  }
  try {
    ctx->car->allreduce<nv_bfloat16>(
        reinterpret_cast<cudaStream_t>(stream),
        reinterpret_cast<nv_bfloat16*>(ctx->input),
        reinterpret_cast<nv_bfloat16*>(ctx->output),
        elems,
        threads,
        block_limit);
  } catch (const std::exception& e) {
    fprintf(stderr, "[arle_car] allreduce failed: %s\n", e.what());
    return CUDA_ERROR_UNKNOWN;
  }
  return (CUresult)cudaGetLastError();
}

// One-shot all-gather over the registered input set: result is
// [world × per_rank_elems] in rank order. bf16; per-rank chunk must be
// 16-byte packed (per_rank_elems % 8 == 0).
CUresult arle_car_allgather_bf16(
    void* h, CUstream stream, int per_rank_elems, int threads, int block_limit) {
  auto* ctx = static_cast<ArleCarCtx*>(h);
  using P = sglang::packed_t<nv_bfloat16>::P;
  constexpr int pack = P::size;
  if (per_rank_elems % pack != 0) return CUDA_ERROR_INVALID_VALUE;
  if ((size_t)per_rank_elems * ctx->world * sizeof(nv_bfloat16) > ctx->max_bytes) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int per_rank = per_rank_elems / pack;
  auto it = ctx->car->buffers_.find(ctx->input);
  if (it == ctx->car->buffers_.end()) return CUDA_ERROR_INVALID_VALUE;
  sglang::RankData* ptrs = it->second;
  const int total = per_rank * ctx->world;
  int blocks = std::min(block_limit, (total + threads - 1) / threads);
  blocks = std::max(blocks, 1);

#define ARLE_AG_CASE(ngpus)                                                                 \
  case ngpus:                                                                               \
    cross_device_allgather_1stage<nv_bfloat16, ngpus><<<blocks, threads, 0, (cudaStream_t)stream>>>( \
        ptrs, ctx->car->sg_, ctx->car->self_sg_, reinterpret_cast<nv_bfloat16*>(ctx->output), \
        ctx->rank, per_rank);                                                               \
    break;
  switch (ctx->world) {
    ARLE_AG_CASE(2)
    ARLE_AG_CASE(4)
    ARLE_AG_CASE(6)
    ARLE_AG_CASE(8)
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
#undef ARLE_AG_CASE
  return (CUresult)cudaGetLastError();
}

// Fill `elems` bf16 at `ptr` with the deterministic per-seed pattern.
CUresult arle_car_fill_bf16(CUstream stream, uint64_t ptr, int elems, int seed) {
  int blocks = std::min(64, (elems + 255) / 256);
  fill_pattern_bf16<<<std::max(blocks, 1), 256, 0, (cudaStream_t)stream>>>(
      reinterpret_cast<__nv_bfloat16*>(ptr), elems, seed);
  return (CUresult)cudaGetLastError();
}

// Dependency hook between timed iterations (exposed-latency mode).
CUresult arle_car_chain_touch(CUstream stream, uint64_t output, uint64_t input) {
  chain_touch<<<1, 1, 0, (cudaStream_t)stream>>>(
      reinterpret_cast<__nv_bfloat16*>(output), reinterpret_cast<__nv_bfloat16*>(input));
  return (CUresult)cudaGetLastError();
}

// ───────────────────────────── production ABI ─────────────────────────────
// The serve path exchanges IPC handles over the ALREADY-UP NCCL comm
// (collective.rs `all_gather_bytes`), not rendezvous files: Rust calls
// `alloc_shared` (own regions) → NCCL-allgathers the 2×64B handles →
// `open_peer` per peer → `create`. `create` takes OWNERSHIP of every pointer
// (own allocations freed, peer mappings IPC-closed, on `destroy_prod`).

struct ArleCarProd {
  sglang::CustomAllreduce* car = nullptr;
  int rank = 0;
  int world = 0;
  void* rank_data = nullptr;
  void* signal_ptrs[kMaxWorld] = {nullptr};
  void* input_ptrs[kMaxWorld] = {nullptr};
};

extern "C" CUresult arle_car_alloc_shared(size_t bytes, uint64_t* out_ptr, uint8_t* out_handle) {
  if (!out_ptr || !out_handle) return CUDA_ERROR_INVALID_VALUE;
  *out_ptr = 0;
  void* p = nullptr;
  cudaError_t e = cudaMalloc(&p, bytes);
  if (e != cudaSuccess) return (CUresult)e;
  e = cudaMemset(p, 0, bytes);
  if (e != cudaSuccess) {
    cudaFree(p);
    return (CUresult)e;
  }
  static_assert(sizeof(cudaIpcMemHandle_t) == 64, "IPC handle is 64 bytes");
  e = cudaIpcGetMemHandle(reinterpret_cast<cudaIpcMemHandle_t*>(out_handle), p);
  if (e != cudaSuccess) {
    cudaFree(p);
    return (CUresult)e;
  }
  *out_ptr = reinterpret_cast<uint64_t>(p);
  return CUDA_SUCCESS;
}

extern "C" CUresult arle_car_free_shared(uint64_t ptr) {
  if (ptr == 0) return CUDA_SUCCESS;
  return (CUresult)cudaFree(reinterpret_cast<void*>(ptr));
}

extern "C" CUresult arle_car_open_peer(const uint8_t* handle, uint64_t* out_ptr) {
  if (!handle || !out_ptr) return CUDA_ERROR_INVALID_VALUE;
  *out_ptr = 0;
  void* p = nullptr;
  cudaError_t e = cudaIpcOpenMemHandle(
      &p, *reinterpret_cast<const cudaIpcMemHandle_t*>(handle), cudaIpcMemLazyEnablePeerAccess);
  if (e != cudaSuccess) return (CUresult)e;
  *out_ptr = reinterpret_cast<uint64_t>(p);
  return CUDA_SUCCESS;
}

extern "C" CUresult arle_car_close_peer(uint64_t ptr) {
  if (ptr == 0) return CUDA_SUCCESS;
  return (CUresult)cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr));
}

extern "C" void* arle_car_create(
    int rank, int world, const uint64_t* signal_ptrs, const uint64_t* input_ptrs) {
  if (world < 2 || world > kMaxWorld || rank < 0 || rank >= world) return nullptr;
  auto* p = new ArleCarProd();
  p->rank = rank;
  p->world = world;
  sglang::Signal* signals[kMaxWorld] = {nullptr};
  void* inputs[kMaxWorld] = {nullptr};
  for (int r = 0; r < world; r++) {
    p->signal_ptrs[r] = reinterpret_cast<void*>(signal_ptrs[r]);
    p->input_ptrs[r] = reinterpret_cast<void*>(input_ptrs[r]);
    signals[r] = reinterpret_cast<sglang::Signal*>(p->signal_ptrs[r]);
    inputs[r] = p->input_ptrs[r];
  }
  cudaError_t e = cudaMalloc(&p->rank_data, 8 * sizeof(sglang::RankData));
  if (e != cudaSuccess) {
    delete p;
    return nullptr;
  }
  try {
    p->car = new sglang::CustomAllreduce(
        signals, p->rank_data, 8 * sizeof(sglang::RankData), rank, world, /*full_nvlink=*/true);
    p->car->register_buffer(inputs);
  } catch (const std::exception& ex) {
    fprintf(stderr, "[arle_car] create failed: %s\n", ex.what());
    delete p->car;
    cudaFree(p->rank_data);
    delete p;
    return nullptr;
  }
  return p;
}

// Sum-allreduce: registered own-input scratch → `out_ptr` (any local buffer).
// force_algo: 0 upstream auto threshold (one-shot <256KB at world 8), 1/2 force.
extern "C" CUresult arle_car_allreduce_bf16_into(
    void* h, CUstream stream, uint64_t out_ptr, int elems, int force_algo) {
  auto* p = static_cast<ArleCarProd*>(h);
  if (force_algo == 1) {
    setenv("SGLANG_CUSTOM_ALLREDUCE_ALGO", "1stage", 1);
  } else if (force_algo == 2) {
    setenv("SGLANG_CUSTOM_ALLREDUCE_ALGO", "2stage", 1);
  } else {
    unsetenv("SGLANG_CUSTOM_ALLREDUCE_ALGO");
  }
  try {
    p->car->allreduce<nv_bfloat16>(
        reinterpret_cast<cudaStream_t>(stream),
        reinterpret_cast<nv_bfloat16*>(p->input_ptrs[p->rank]),
        reinterpret_cast<nv_bfloat16*>(out_ptr),
        elems);
  } catch (const std::exception& ex) {
    fprintf(stderr, "[arle_car] allreduce_into failed: %s\n", ex.what());
    return CUDA_ERROR_UNKNOWN;
  }
  return (CUresult)cudaGetLastError();
}

// One-shot all-gather: every rank's registered scratch chunk → rank-major
// `out_ptr` ([world × per_rank_elems], ncclAllGather layout).
extern "C" CUresult arle_car_allgather_bf16_into(
    void* h, CUstream stream, uint64_t out_ptr, int per_rank_elems) {
  auto* p = static_cast<ArleCarProd*>(h);
  using P = sglang::packed_t<nv_bfloat16>::P;
  constexpr int pack = P::size;
  if (per_rank_elems % pack != 0) return CUDA_ERROR_INVALID_VALUE;
  const int per_rank = per_rank_elems / pack;
  auto it = p->car->buffers_.find(p->input_ptrs[p->rank]);
  if (it == p->car->buffers_.end()) return CUDA_ERROR_INVALID_VALUE;
  sglang::RankData* ptrs = it->second;
  const int total = per_rank * p->world;
  const int threads = 512;
  int blocks = std::min(36, (total + threads - 1) / threads);
  blocks = std::max(blocks, 1);

#define ARLE_AG_PROD_CASE(ngpus)                                                          \
  case ngpus:                                                                             \
    cross_device_allgather_1stage<nv_bfloat16, ngpus>                                     \
        <<<blocks, threads, 0, (cudaStream_t)stream>>>(                                   \
            ptrs, p->car->sg_, p->car->self_sg_, reinterpret_cast<nv_bfloat16*>(out_ptr), \
            p->rank, per_rank);                                                           \
    break;
  switch (p->world) {
    ARLE_AG_PROD_CASE(2)
    ARLE_AG_PROD_CASE(4)
    ARLE_AG_PROD_CASE(6)
    ARLE_AG_PROD_CASE(8)
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
#undef ARLE_AG_PROD_CASE
  return (CUresult)cudaGetLastError();
}

extern "C" void arle_car_destroy_prod(void* h) {
  auto* p = static_cast<ArleCarProd*>(h);
  if (!p) return;
  delete p->car;
  for (int r = 0; r < p->world; r++) {
    if (r == p->rank) {
      cudaFree(p->signal_ptrs[r]);
      cudaFree(p->input_ptrs[r]);
    } else {
      if (p->signal_ptrs[r]) cudaIpcCloseMemHandle(p->signal_ptrs[r]);
      if (p->input_ptrs[r]) cudaIpcCloseMemHandle(p->input_ptrs[r]);
    }
  }
  cudaFree(p->rank_data);
  delete p;
}

void arle_car_destroy(void* h) {
  auto* ctx = static_cast<ArleCarCtx*>(h);
  if (!ctx) return;
  delete ctx->car;  // closes opened IPC handles
  cudaFree(ctx->rank_data);
  cudaFree(ctx->output);
  cudaFree(ctx->input);
  cudaFree(ctx->signal_region);
  delete ctx;
}

}  // extern "C"

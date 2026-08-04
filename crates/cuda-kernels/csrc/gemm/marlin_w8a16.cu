// W8A16 Marlin GEMM + repack — extern "C" shim over the vendored SGLang kernels.
//
// Replaces SGLang's two TVM-FFI host wrappers (gptq_marlin_gemm / gptq_marlin_repack,
// both truncated out of the vendored .cuh) with raw-pointer + cudaStream_t entry
// points matching ARLE's crates/cuda-kernels/src/ffi/gemm.rs convention.
//
// Scope: symmetric per-group INT8 weights (scale = amax/127, no zero-point) →
// Marlin q_type = kU8B128 (uint8 with +128 bias), has_zp=false, has_act_order=false,
// BF16 activations/scales. group_size=128 (group_blocks=8) natively supported.
//
// The vendored device kernels are Ampere-MMA (`mma.sync.m16n8k16`, `cp.async`), gated
// `#if __CUDA_ARCH__ < 800` to empty stubs — one binary runs sm_80..sm_120. We add a
// runtime SM guard so a pre-sm_80 launch returns cudaErrorNotSupported instead of a
// silent no-op (the empty-stub trap).
//
// License: Apache 2.0 (SGLang / Neural Magic / Marlin — Elias Frantar). See the
// per-header attribution in marlin/*.cuh.

#include <cuda.h>  // Driver API: CUresult / CUDA_ERROR_* (the shim's return type)
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifdef ARLE_DISABLE_MARLIN_SM70

extern "C" {

cudaError_t marlin_gptq_repack_w8a16_cuda(
    const uint32_t* b_q_weight, uint32_t* out, int size_k, int size_n, cudaStream_t stream) {
  (void)b_q_weight; (void)out; (void)size_k; (void)size_n; (void)stream;
  return cudaErrorNotSupported;
}

CUresult marlin_w8a16_gemm_cuda(
    const __nv_bfloat16* A, const uint32_t* B_packed, const __nv_bfloat16* scales,
    __nv_bfloat16* C, float* c_tmp, int* workspace,
    int m, int n, int k, int group_size, cudaStream_t stream) {
  (void)A; (void)B_packed; (void)scales; (void)C; (void)c_tmp; (void)workspace;
  (void)m; (void)n; (void)k; (void)group_size; (void)stream;
  return CUDA_ERROR_NOT_SUPPORTED;
}

int marlin_w8a16_c_tmp_floats(int m, int sms) { (void)m; (void)sms; return 0; }
int marlin_w8a16_workspace_ints(int sms) { (void)sms; return 0; }

}  // extern "C"

#else

#include "marlin/gptq_marlin.cuh"
#include "marlin/gptq_marlin_repack.cuh"

namespace {

inline bool sm_supports_marlin(int dev, int* sms_out) {
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
  int minor = 0;
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  if (major * 10 + minor < 80) return false;  // Ampere MMA floor
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return false;
  *sms_out = sms;
  return true;
}

}  // namespace

extern "C" {

// GPTQ-packed uint8b128 [size_k/4, size_n] int32 → Marlin tile layout.
// weight: uint8 = int8 + 128 packed 4-per-int32 (see repack_for_marlin_w8a16 in
// tensor.rs). out: [size_k/16, size_n*16/4] int32.
cudaError_t marlin_gptq_repack_w8a16_cuda(
    const uint32_t* b_q_weight, uint32_t* out, int size_k, int size_n, cudaStream_t stream) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return cudaErrorInvalidDevice;
  int sms = 0;
  if (!sm_supports_marlin(dev, &sms)) return cudaErrorNotSupported;

  int max_shared_mem = 0;
  cudaError_t e = cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (e != cudaSuccess) return e;

  constexpr int num_bits = 8;
  // The repack kernel grid-strides over k-tiles: gridDim.x == SM count, and each
  // block sweeps all n-tiles for its k-tile range (see block_k_tiles at
  // gptq_marlin_repack.cuh:53). Matches upstream (blocks = multiProcessorCount).
  int blocks = sms;

  auto kernel = device::marlin::gptq_marlin_repack_kernel<device::marlin::repack_threads, num_bits, false>;
  e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
  if (e != cudaSuccess) return e;

  const uint32_t* perm_ptr = nullptr;  // has_perm=false
  kernel<<<blocks, device::marlin::repack_threads, max_shared_mem, stream>>>(
      b_q_weight, perm_ptr, out, size_k, size_n);
  return cudaGetLastError();
}

// Y[m,n] = X[m,k] @ dequant(W)  — BF16 activations, kU8B128 weights, per-group
// BF16 scales [k/group_size, n] (Marlin-permuted). C is written, not accumulated.
// c_tmp (fp32-reduce scratch) + workspace (zeroed int lock buffer) are
// CALLER-OWNED and reused across calls — the Qwen decode loop is CUDA-graph
// captured, so the shim must NOT cudaMalloc/cudaFree per call (that breaks
// capture). Both sizes depend only on sms (device-constant); the Rust side
// allocates once. c_tmp needs >= marlin_w8a16_c_tmp_floats(m, sms) floats;
// workspace >= sms ints, zero-initialized once (Marlin leaves locks at 0).
CUresult marlin_w8a16_gemm_cuda(
    const __nv_bfloat16* A,       // [m, k] row-major (lda = k)
    const uint32_t* B_packed,     // Marlin-repacked kU8B128 weight
    const __nv_bfloat16* scales,  // [k/group_size, n] permuted
    __nv_bfloat16* C,             // [m, n] output
    float* c_tmp,                 // caller-owned fp32-reduce scratch
    int* workspace,               // caller-owned zeroed int lock buffer
    int m, int n, int k, int group_size, cudaStream_t stream) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return CUDA_ERROR_INVALID_DEVICE;
  int sms = 0;
  if (!sm_supports_marlin(dev, &sms)) return CUDA_ERROR_NOT_SUPPORTED;
  if (m == 0) return CUDA_SUCCESS;

  int num_groups = k / group_size;

  try {
    device::marlin::marlin_mm<__nv_bfloat16>(
        A, B_packed, C,
        c_tmp,
        const_cast<__nv_bfloat16*>(scales),  // param #5 is void* (read-only here)
        nullptr,   // s2 (global_scale) — unused for kU8B128
        nullptr,   // zp — has_zp=false
        nullptr,   // g_idx
        nullptr,   // perm
        nullptr,   // a_tmp — has_act_order=false
        m, n, k,
        k,         // lda
        workspace,
        host::kU8B128,
        false,     // has_act_order
        true,      // is_k_full
        false,     // has_zp
        num_groups, group_size, dev, stream,
        -1, -1,    // thread_k/n auto
        sms,
        false,     // use_atomic_add
        true,      // use_fp32_reduce
        false);    // is_zp_float
  } catch (...) {
    // marlin_mm panics (host::Panic/RuntimeCheck throw) on an invalid config.
    // A C++ exception must not cross the extern-C boundary (UB) — map to a code.
    return CUDA_ERROR_INVALID_VALUE;
  }

  return (cudaGetLastError() == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

// c_tmp float count for the fp32-reduce path (SGLang gptq_marlin.py:76-81):
//   sms * min(ceil(m/16)*16, 64) * max_thread_n(256). The Rust scratch allocates
// the m-independent MAX (m >= 64 → max_m_block = 64) once.
int marlin_w8a16_c_tmp_floats(int m, int sms) {
  int max_m_block = ((m + 15) / 16) * 16;
  if (max_m_block > 64) max_m_block = 64;
  return sms * max_m_block * device::marlin::max_thread_n;
}

// Lock-buffer int count. Upstream sizes locks at sms * max_blocks_per_sm
// (marlin_utils.py:416, default max_blocks_per_sm=1 → sms); 4× is headroom.
// Zero-initialize once on the Rust side.
int marlin_w8a16_workspace_ints(int sms) { return sms * 4; }

}  // extern "C"

#endif  // ARLE_DISABLE_MARLIN_SM70

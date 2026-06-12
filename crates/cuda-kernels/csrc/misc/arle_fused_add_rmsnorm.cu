// ARLE → flashinfer (vendored) fused residual-add + RMSNorm shim.
//
// Adopts `flashinfer::norm::FusedAddRMSNormKernel` (csrc/vendor/flashinfer/
// norm.cuh:387) — the SGLang `sgl_fused_add_rmsnorm` kernel — to replace
// ARLE's separate `add_batch` + `rms_norm_batched_offset` pair on the Qwen3.6
// post-attention residual stream. Opt-in via `ARLE_QWEN35_FUSED_ADDNORM`;
// the default (flag-off) path stays on the hand kernels (byte-identical).
//
// Per row `bx` (one token, `d = hidden_size` contiguous bf16 elements):
//   x          = input[bx] + residual[bx]
//   residual[bx] = x                                   (running pre-norm sum)
//   rms        = rsqrt(mean(x^2) + eps)
//   input[bx]  = x * rms * (weight_bias + weight)
// Both buffers are mutated IN PLACE: `residual` becomes the new running
// residual, `input` becomes the normalized sublayer output.
//
// **weight_bias = 1.0f (NOT 0).** ARLE's `rms_norm_offset` / Qwen3.x apply the
// Gemma-style `(1 + weight)` gain, so the fused kernel must match — flashinfer's
// host launcher `FusedAddRMSNorm<T>` hardcodes `weight_bias = 0`, so we cannot
// call it directly. Instead we replicate its launch config here and pass
// `weight_bias = 1.0f`, which makes this fused path numerically equivalent to
// the `fused_add_rms_norm_offset` hand kernel it replaces.
//
// enable_pdl is hard-false: we are not inside a PDL graph, so the
// programmatic-stream-serialization attribute is inert. The `griddepcontrol`
// asm in the kernel is `__CUDA_ARCH__ >= 900`-guarded (Hopper/H20-safe).
//
// Refs:
//   csrc/vendor/flashinfer/norm.cuh (FusedAddRMSNormKernel @ 387, launcher @ 480)
//   sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu (torch wrapper)

#include <cstdint>
#include <numeric>

#include "vendor/flashinfer/norm.cuh"
#include "vendor/flashinfer/utils.cuh"

extern "C" {

// Fused residual-add + (1+weight) RMSNorm over a `[batch, d]` row-major bf16
// batch (each row = one token of `d` contiguous features; row stride = `d` for
// ARLE's `[seq_len, hidden_size]` residual buffers). `input` and `residual` are
// both in/out (see file header); `weight` is read-only. All pointers are CUDA
// device pointers; `stride_input` / `stride_residual` are in element count.
//
// `weight` is declared const (ARLE's norm weight is immutable, pool-shared) but
// flashinfer's kernel template takes a non-const `T*`; we cast away const in the
// launch — the kernel only ever reads `weight`.
cudaError_t arle_fused_add_rmsnorm_offset_bf16_cuda(__nv_bfloat16 *input,
                                                    __nv_bfloat16 *residual,
                                                    const __nv_bfloat16 *weight,
                                                    uint32_t batch, uint32_t d,
                                                    uint32_t stride_input,
                                                    uint32_t stride_residual, float eps,
                                                    cudaStream_t stream) {
  using T = __nv_bfloat16;
  // Mirror flashinfer::norm::FusedAddRMSNorm<T> (norm.cuh:480) launch config,
  // with weight_bias = 1.0f for the (1+weight) gain.
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);
  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  dim3 nblks(batch);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = (ceil_div(num_warps, 4) * 4 + d) * sizeof(float);
  float weight_bias = 1.0f;
  T *weight_mut = const_cast<T *>(weight);

  cudaLaunchConfig_t config;
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 0;  // enable_pdl = false
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaError_t status = cudaSuccess;
  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = flashinfer::norm::FusedAddRMSNormKernel<VEC_SIZE, T>;
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (status != cudaSuccess) return status;
    status = cudaLaunchKernelEx(&config, kernel, input, residual, weight_mut, d, stride_input,
                                stride_residual, weight_bias, eps);
    if (status != cudaSuccess) return status;
  });
  return cudaGetLastError();
}

}  // extern "C"

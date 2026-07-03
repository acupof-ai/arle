// CUDA->HIP source-compat shim, force-included (-include) when compiling
// ARLE's CUDA kernel sources with hipcc for the AIPC lane (#76/#77,
// docs/plans/2026-06-10-hip-backend-mvp.md §2.2/§4).
//
// Mapping table adapted from ggml-org/llama.cpp
// ggml/src/ggml-cuda/vendors/hip.h @ d2462f8f (MIT, vendored at
// vendor/llama.cpp/ggml-cuda/vendors/hip.h), scoped down to the exact
// CUDA-API inventory of the sources this crate compiles:
//   crates/cuda-kernels/csrc/misc/{dsv4_attention,dsv4_mhc,elementwise_basic,
//                                  norm,sampling}.cu
//   crates/cuda-kernels/csrc/attention/decode_prep_paged.cu
//   crates/cuda-kernels/csrc/gemm/{moe_grouped_gemm,quantized_gemv}.cu
//   crates/cuda-kernels/csrc/common.cuh
//   crates/hip-kernels/csrc/iq2_mmvq.cu
//
// Basic-op additions (norm/sampling/quantized_gemv, 2026-06-10) introduce NO
// new CUDA APIs beyond this inventory: bf16 conversions, __ldg, full-mask
// __shfl_{down,xor}_sync, cudaError_t/cudaStream_t/cudaGetLastError/
// cudaSuccess/cudaErrorInvalidValue, and the fp8 e4m3 cast — all mapped or
// HIP-native below.
//
// Inventory items needing NO mapping (provided by HIP as-is, verified
// against the ROCm HIP headers / llama.cpp HIP build at the pinned commit):
// kernel-launch chevrons, __global__/__device__/__host__/__shared__/
// __forceinline__/__restrict__/__launch_bounds__, threadIdx/blockIdx/
// blockDim/gridDim, __syncthreads, dim3, uint2/make_uint2, __uint_as_float,
// __popc, libm device math (expf/powf/logf/floorf/ceilf/cosf/sinf/rsqrtf/
// sqrtf/fminf/fmaxf/roundf/fabsf/isfinite/INFINITY), __bfloat162float,
// __float2bfloat16, half/half2 + make_half2/__half2float/__low2float/
// __half22float2 (hip_fp16.h). cudaMemcpy*/cudaMemset* appear in none of
// the sources above — deliberately not mapped.
//
// On-box verify (pending-remote, dry hipcc parse per plan §2.2): bf16
// __hadd/__hadd2 overloads and __hip_fp8_e4m3::operator float() in the
// pinned ROCm 7.x headers.
#pragma once

#ifdef __HIP_PLATFORM_AMD__

// ROCm 6.2+ ships native __shfl_*_sync builtins; disable them so the legacy
// wave-level shuffles below win — the known-good llama.cpp HIP configuration.
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#if HIP_VERSION >= 60200000
#include <hip/hip_fp8.h>
typedef __hip_fp8_e4m3 __nv_fp8_e4m3; // OCP E4M3 — same bit layout as NV E4M3
#else
#error "ARLE HIP kernels require ROCm >= 6.2 (hip_fp8.h)"
#endif

// Driver-API surface: HIP has no runtime/driver split, hipError_t covers both,
// so the (CUresult)cudaError_t casts in our launchers become identity casts.
typedef hipError_t CUresult;
typedef hipStream_t CUstream;
#define CUDA_SUCCESS hipSuccess
#define CUDA_ERROR_INVALID_VALUE hipErrorInvalidValue

// Runtime-API surface.
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaGetLastError hipGetLastError

typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
#define __hadd_rn __hadd   // HIP bf16 __hadd rounds to nearest even == CUDA _rn
#define __hadd2_rn __hadd2

// Mask argument dropped: all uses in our sources are full-mask (0xffffffff)
// butterfly/down reductions, and the gfx11/gfx12 targets this crate builds
// for are wave32, so warpSize==32 semantics hold without a mask.
// Variadic so both the 3-arg (lane) and 4-arg (lane, width) forms map.
#define __shfl_xor_sync(mask, var, ...) __shfl_xor(var, __VA_ARGS__)
#define __shfl_down_sync(mask, var, ...) __shfl_down(var, __VA_ARGS__)

// AMD has no separate read-only-data-cache load path; a plain load is
// semantically identical (defined after the HIP headers so their own __ldg
// overloads are not mangled).
#define __ldg(ptr) (*(ptr))

#endif // __HIP_PLATFORM_AMD__

// Torch/TVM-FFI-free compat shim for the vendored SGLang Marlin kernels.
//
// The upstream kernels include <sgl_kernel/utils.cuh> (+ utils.h, ffi.h,
// source_location.h), which drag in TVM-FFI, DLPack, and HIP. ARLE builds with
// cudarc + extern "C", none of that. This header reproduces ONLY the symbols the
// device kernels and the device::marlin::marlin_mm / repack host bodies actually
// reference — everything else in utils.cuh is TVM-FFI plumbing we replace with the
// extern "C" shim in marlin_w8a16.cu.
//
// Provided here:
//   - type aliases fp16_t / bf16_t / fp16x2_t / bf16x2_t / fp32_t
//   - SGL_DEVICE
//   - host::div_ceil, host::DebugInfo, host::panic, host::PanicError,
//     host::RuntimeCheck, host::Panic, host::RuntimeDeviceCheck, host::LaunchKernel
// host::ScalarType and the kU*/kFE* constants come from the verbatim
// scalar_type.hpp (torch-free upstream).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

// Upstream utils.cuh aliases; the kernels use these bare names.
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp16x2_t = __half2;
using bf16x2_t = __nv_bfloat162;

// The vendored kernels call div_ceil unqualified from inside namespace
// device::marlin (args are int/unsigned → no ADL). Define it at global scope so
// unqualified lookup finds it.
template <typename T, typename U>
__host__ __device__ inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

namespace host {

// Minimal source-location wrapper. The upstream DebugInfo derives from a
// std::source_location; we only need file/line for panic messages, and
// __builtin_* keeps it constexpr-friendly under nvcc without <source_location>.
struct DebugInfo {
  const char* file;
  int line;
  DebugInfo(const char* f = __builtin_FILE(), int l = __builtin_LINE()) : file(f), line(l) {}
  const char* file_name() const { return file; }
  int line_number() const { return line; }
};

struct PanicError : public std::runtime_error {
  explicit PanicError(std::string msg) : std::runtime_error(std::move(msg)) {}
};

template <typename... Args>
[[noreturn]] inline void panic(DebugInfo location, Args&&... args) {
  std::ostringstream os;
  os << "Marlin failed at " << location.file_name() << ":" << location.line_number();
  if constexpr (sizeof...(args) > 0) {
    os << ": ";
    (os << ... << std::forward<Args>(args));
  }
  throw PanicError(std::move(os).str());
}

// RuntimeCheck(cond, msg...): throw PanicError when cond is false. The trailing
// DebugInfo default + deduction guide mirror upstream so call sites compile
// verbatim (the variadic-with-trailing-default trick needs the guides below).
template <typename... Args>
struct RuntimeCheck {
  template <typename Cond>
  explicit RuntimeCheck(Cond&& condition, Args&&... args, DebugInfo location = {}) {
    if (condition) return;
    ::host::panic(location, std::forward<Args>(args)...);
  }
};
template <typename Cond, typename... Args>
explicit RuntimeCheck(Cond&&, Args&&...) -> RuntimeCheck<Args...>;

template <typename... Args>
struct Panic {
  explicit Panic(Args&&... args, DebugInfo location = {}) { ::host::panic(location, std::forward<Args>(args)...); }
};
template <typename... Args>
explicit Panic(Args&&...) -> Panic<Args...>;

inline void RuntimeDeviceCheck(::cudaError_t error, DebugInfo location = {}) {
  if (error != ::cudaSuccess) {
    ::host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

// Plain-CUDA replacement for upstream's PDL/cluster LaunchKernel. Marlin's
// marlin_mm/repack only ever call the 4-arg form (grid, block, stream, smem)
// then operator()(kernel, args...) — no PDL, no cluster. cudaLaunchKernelEx keeps
// the >48KB dynamic-smem opt-in (set via cudaFuncSetAttribute at the call site).
struct LaunchKernel {
  cudaLaunchConfig_t cfg{};
  DebugInfo loc;
  explicit LaunchKernel(dim3 grid, dim3 block, cudaStream_t stream, std::size_t smem = 0, DebugInfo location = {})
      : loc(location) {
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.numAttrs = 0;
    cfg.attrs = nullptr;
  }
  LaunchKernel(const LaunchKernel&) = delete;
  LaunchKernel& operator=(const LaunchKernel&) = delete;

  template <typename T, typename... Args>
  void operator()(T&& kernel, Args&&... args) const {
    RuntimeDeviceCheck(::cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...), loc);
  }
};

}

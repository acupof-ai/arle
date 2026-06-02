#include <cuda.h>

#include <cstddef>

#ifndef ARLE_ENABLE_DEEPGEMM_NATIVE
extern "C" CUresult dsv4_deepgemm_native_preflight_cuda(char* out, size_t out_len) {
  static constexpr const char* kMessage =
      "status=failed native_bridge=not_compiled "
      "reason=build_with_ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1";
  if (out != nullptr && out_len > 0) {
    size_t n = 0;
    while (n + 1 < out_len && kMessage[n] != '\0') {
      out[n] = kMessage[n];
      ++n;
    }
    out[n] = '\0';
  }
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda(
    const unsigned char*,
    const float*,
    const unsigned char*,
    const float*,
    unsigned short*,
    const int*,
    int,
    int,
    int,
    int,
    int,
    CUstream) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
#endif

#include <cuda.h>

#ifndef ARLE_ENABLE_DEEPGEMM_NATIVE
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

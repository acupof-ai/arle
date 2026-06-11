// HIP stand-in for crates/cuda-kernels/csrc/gemm/quantized_gemv_mma.cu,
// whose PTX `mma.sync.aligned.m16n8k16` asm cannot compile for RDNA.
// quantized_gemv.cu treats cudaErrorInvalidValue from this launcher as
// "shape not tiled" and falls through to its scalar kernels, so returning
// it unconditionally keeps every FP8 gemv shape on the portable path.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" cudaError_t dsv4_fp8_gemv_batch_mma_launch(
    const uint8_t* weight,
    const uint8_t* scales,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int B,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream) {
    (void)weight;
    (void)scales;
    (void)input;
    (void)output;
    (void)B;
    (void)N;
    (void)K;
    (void)scale_rows;
    (void)scale_cols;
    (void)stream;
    return cudaErrorInvalidValue;
}

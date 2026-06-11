// Link stubs for builds without the FA3 vendor tree / opt-in
// (ARLE_CUDA_ENABLE_FA3 unset). Mirrors arle_flashmla_decode_stubs.cu: the
// Rust FFI links unconditionally; the runtime gate keeps these unreachable,
// and the marker lets callers detect a stub binary instead of silently
// degrading.

#include <cuda_runtime.h>

extern "C" {

typedef struct {
    const void* q;
    const void* k;
    const void* v;
    void* o;
    float* softmax_lse;
    int* tile_count_semaphore;
    int seqlen_q;
    int seqlen_k;
    int num_heads;
    int num_heads_k;
    int head_dim;
    long long q_row_stride;
    long long k_row_stride;
    long long v_row_stride;
    long long o_row_stride;
    long long q_head_stride;
    long long k_head_stride;
    long long v_head_stride;
    long long o_head_stride;
    float softmax_scale;
    int is_causal;
} ArleFa3FwdHd256Args;

cudaError_t arle_fa3_fwd_hd256_bf16_cuda(const ArleFa3FwdHd256Args* a,
                                         cudaStream_t stream) {
    (void)a;
    (void)stream;
    return cudaErrorNotSupported;
}

int arle_fa3_real_kernel_marker_cuda(void) { return 0; }

}  // extern "C"

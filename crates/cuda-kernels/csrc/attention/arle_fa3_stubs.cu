// Link stubs for builds without the FA3 vendor tree / opt-in
// (ARLE_CUDA_ENABLE_FA3 unset). Mirrors arle_flashmla_decode_stubs.cu: the
// Rust FFI links unconditionally; the runtime gate keeps these unreachable,
// and the marker lets callers detect a stub binary instead of silently
// degrading.

#include <cuda_runtime.h>

extern "C" {

// Opaque: the stub never dereferences the args, so mirroring the real
// struct here only invites drift.
typedef struct ArleFa3FwdHd256Args ArleFa3FwdHd256Args;
typedef struct ArleFa3BwdHd256Args ArleFa3BwdHd256Args;

cudaError_t arle_fa3_fwd_hd256_bf16_cuda(const ArleFa3FwdHd256Args* a,
                                         cudaStream_t stream) {
    (void)a;
    (void)stream;
    return cudaErrorNotSupported;
}

cudaError_t arle_fa3_bwd_hd256_bf16_cuda(const ArleFa3BwdHd256Args* a,
                                         cudaStream_t stream) {
    (void)a;
    (void)stream;
    return cudaErrorNotSupported;
}

int arle_fa3_real_kernel_marker_cuda(void) { return 0; }

}  // extern "C"

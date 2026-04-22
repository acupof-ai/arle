#include "common.cuh"
#include <cstdint>

extern "C" {

cudaError_t conv1d_prefill_cuda(
    const __nv_bfloat16* x_seq,
    const __nv_bfloat16* conv_weight,
    __nv_bfloat16* conv_state,
    __nv_bfloat16* out_seq,
    int num_channels,
    int seq_len,
    int kernel_size,
    cudaStream_t stream
);

cudaError_t conv1d_prefill_packed_batch_cuda(
    const __nv_bfloat16* x_batch,
    const __nv_bfloat16* conv_weight,
    const uint64_t* conv_state_ptrs,
    const int32_t* seq_indptr,
    __nv_bfloat16* out_batch,
    int num_channels,
    int kernel_size,
    int batch_size,
    cudaStream_t stream
) {
    if (
        x_batch == nullptr || conv_weight == nullptr || conv_state_ptrs == nullptr
        || seq_indptr == nullptr || out_batch == nullptr || num_channels <= 0 || kernel_size <= 0
        || batch_size <= 0
    ) {
        return cudaErrorInvalidValue;
    }

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int token_start = seq_indptr[batch_idx];
        const int token_end = seq_indptr[batch_idx + 1];
        const int seq_len = token_end - token_start;
        if (token_start < 0 || seq_len <= 0 || conv_state_ptrs[batch_idx] == 0) {
            return cudaErrorInvalidValue;
        }

        cudaError_t err = conv1d_prefill_cuda(
            x_batch + static_cast<size_t>(token_start) * num_channels,
            conv_weight,
            reinterpret_cast<__nv_bfloat16*>(conv_state_ptrs[batch_idx]),
            out_batch + static_cast<size_t>(token_start) * num_channels,
            num_channels,
            seq_len,
            kernel_size,
            stream
        );
        if (err != cudaSuccess) {
            return err;
        }
    }

    return cudaSuccess;
}

} // extern "C"

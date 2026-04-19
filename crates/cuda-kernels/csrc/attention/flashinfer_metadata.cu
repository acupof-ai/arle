#include <cuda_runtime.h>
#include <cstdint>

__global__ void flashinfer_append_last_token_indices_kernel(
    int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ last_token_indices,
    int batch_size)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Serialize from the back so the in-place segment shifts behave like memmove.
    for (int i = batch_size - 1; i >= 0; --i) {
        int new_start = kv_indptr[i];
        int new_end = kv_indptr[i + 1];
        int old_start = new_start - i;
        int old_end = new_end - (i + 1);
        int old_len = old_end - old_start;

        for (int j = old_len - 1; j >= 0; --j) {
            kv_indices[new_start + j] = kv_indices[old_start + j];
        }
        kv_indices[new_end - 1] = last_token_indices[i];
    }
}

extern "C" {

cudaError_t flashinfer_append_last_token_indices_cuda(
    int32_t* kv_indices,
    const int32_t* kv_indptr,
    const int32_t* last_token_indices,
    int batch_size,
    cudaStream_t stream)
{
    if (batch_size <= 0) return cudaSuccess;

    flashinfer_append_last_token_indices_kernel<<<1, 1, 0, stream>>>(
        kv_indices, kv_indptr, last_token_indices, batch_size
    );
    return cudaGetLastError();
}

} // extern "C"

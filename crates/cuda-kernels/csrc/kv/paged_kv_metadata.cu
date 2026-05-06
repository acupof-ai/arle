#include <cuda_runtime.h>
#include <cstdint>

__global__ void paged_kv_append_last_token_indices_kernel(
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

__global__ void paged_kv_append_new_page_indices_kernel(
    int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ prev_kv_indptr,
    const int32_t* __restrict__ next_kv_indptr,
    const int32_t* __restrict__ append_indptr,
    const int32_t* __restrict__ appended_page_indices,
    int batch_size)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Serialize from the back so suffixes shifted by earlier slots behave like memmove.
    for (int i = batch_size - 1; i >= 0; --i) {
        int old_start = prev_kv_indptr[i];
        int old_end = prev_kv_indptr[i + 1];
        int new_start = next_kv_indptr[i];
        int new_end = next_kv_indptr[i + 1];
        int append_start = append_indptr[i];
        int append_end = append_indptr[i + 1];
        int append_count = append_end - append_start;

        for (int j = old_end - old_start - 1; j >= 0; --j) {
            kv_indices[new_start + j] = kv_indices[old_start + j];
        }
        for (int j = 0; j < append_count; ++j) {
            kv_indices[new_end - append_count + j] = appended_page_indices[append_start + j];
        }
    }
}

extern "C" {

cudaError_t paged_kv_append_last_token_indices_cuda(
    int32_t* kv_indices,
    const int32_t* kv_indptr,
    const int32_t* last_token_indices,
    int batch_size,
    cudaStream_t stream)
{
    if (batch_size <= 0) return cudaSuccess;

    paged_kv_append_last_token_indices_kernel<<<1, 1, 0, stream>>>(
        kv_indices, kv_indptr, last_token_indices, batch_size
    );
    return cudaGetLastError();
}

cudaError_t paged_kv_append_new_page_indices_cuda(
    int32_t* kv_indices,
    const int32_t* prev_kv_indptr,
    const int32_t* next_kv_indptr,
    const int32_t* append_indptr,
    const int32_t* appended_page_indices,
    int batch_size,
    cudaStream_t stream)
{
    if (batch_size <= 0) return cudaSuccess;

    paged_kv_append_new_page_indices_kernel<<<1, 1, 0, stream>>>(
        kv_indices,
        prev_kv_indptr,
        next_kv_indptr,
        append_indptr,
        appended_page_indices,
        batch_size
    );
    return cudaGetLastError();
}

} // extern "C"

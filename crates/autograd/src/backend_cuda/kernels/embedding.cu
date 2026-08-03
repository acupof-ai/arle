// Gather embedding rows by integer token ids.
// Inputs:
//   weight: [vocab, dim] row-major
//   ids:    [n_ids]        int32
// Output:
//   out:    [n_ids, dim] row-major
//
// Grid: (n_ids, 1, 1). Block: (256, 1, 1). Each block copies one row.
// Invalid ids (>= vocab) write zeros to keep the kernel well-defined; the
// host caller is responsible for bounds-checking correctness of ids.

extern "C" __global__ void embedding_f32(
    T* __restrict__ out,
    const float* __restrict__ weight,
    const int* __restrict__ ids,
    int n_ids,
    int vocab,
    int dim
) {
    int row = blockIdx.x;
    if (row >= n_ids) return;
    int tid = threadIdx.x;
    int block = blockDim.x;
    int id = ids[row];
    T* row_out = out + row * dim;
    if (id < 0 || id >= vocab) {
        for (int i = tid; i < dim; i += block) {
            row_out[i] = static_cast<T>(0.0f);
        }
        return;
    }
    const float* row_w = weight + id * dim;
    for (int i = tid; i < dim; i += block) {
        row_out[i] = static_cast<T>(row_w[i]);
    }
}

extern "C" __global__ void embedding_bf16_to_f32(
    T* __restrict__ out,
    const unsigned short* __restrict__ weight,
    const int* __restrict__ ids,
    int n_ids,
    int vocab,
    int dim
) {
    int row = blockIdx.x;
    if (row >= n_ids) return;
    int tid = threadIdx.x;
    int block = blockDim.x;
    int id = ids[row];
    T* row_out = out + row * dim;
    if (id < 0 || id >= vocab) {
        for (int i = tid; i < dim; i += block) {
            row_out[i] = static_cast<T>(0.0f);
        }
        return;
    }
    const unsigned short* row_w = weight + id * dim;
    for (int i = tid; i < dim; i += block) {
        unsigned int bits = static_cast<unsigned int>(row_w[i]) << 16;
        row_out[i] = static_cast<T>(__uint_as_float(bits));
    }
}

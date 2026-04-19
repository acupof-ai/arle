// Scatter-add rows into a [vocab, feature_dim] output.
//
// For each prefix row in [0, prefix_rows):
//   if indices[row] is in [0, vocab) :
//     atomicAdd into out[indices[row] * feature_dim .. +feature_dim] from
//     upstream[row * feature_dim .. +feature_dim]
//   else: skip (matches the CPU reference; the op-layer bounds-checks
//   against the high-level usize indices when it wants a hard error).
//
// Grid: (prefix_rows, 1, 1). Block: (min(256, feature_dim), 1, 1).
// Callers MUST zero-initialize `out` before launch — this kernel only adds.
// atomicAdd is required because distinct prefix rows may select the same
// `indices[row]` (embedding_backward: multiple token positions hitting the
// same vocab id).
extern "C" __global__ void scatter_add_rows_f32(
    float* __restrict__ out,
    const float* __restrict__ upstream,
    const int* __restrict__ indices,
    int prefix_rows,
    int feature_dim,
    int vocab
) {
    int row = blockIdx.x;
    if (row >= prefix_rows) return;
    int idx = indices[row];
    if (idx < 0 || idx >= vocab) return;
    const float* src = upstream + row * feature_dim;
    float* dst = out + idx * feature_dim;
    int tid = threadIdx.x;
    int block = blockDim.x;
    for (int i = tid; i < feature_dim; i += block) {
        atomicAdd(&dst[i], src[i]);
    }
}

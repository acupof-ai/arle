// atomicAdd is MANDATORY — duplicate token ids within one batch are normal
// (e.g. `the` appears N times in a 1024-token sequence) and must accumulate
// correctly. Candle's `index_add`/`scatter_add` skip atomics; we cannot
// vendor those.
//
// Caller MUST zero-initialize `grad_table` before launch (this kernel only
// adds — matches the `scatter_add_rows_f32` contract).
//
// Hidden dim ≤ ~512 in practice (small-25m: 160; production Qwen3.5: 4096),
// so the inner loop is short and the atomicAdd traffic is the dominant cost.

extern "C" __global__ void embedding_backward_f32(
    float* __restrict__ grad_table,
    const T* __restrict__ upstream,
    const int* __restrict__ ids,
    int n_ids,
    int hidden_dim,
    int vocab_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_ids) return;
    int id = ids[row];
    if (id < 0 || id >= vocab_size) return;
    const T* src = upstream + row * hidden_dim;
    float* dst = grad_table + id * hidden_dim;
    for (int c = 0; c < hidden_dim; ++c) {
        atomicAdd(&dst[c], static_cast<float>(src[c]));
    }
}

// Wave 1 (post-M5.3b nsys attribution): backward for gather_last_dim.
// Zero-fills a `[prefix_rows, vocab]` output (caller-owned, allocated via
// `alloc_zeros`) then writes `upstream[row]` into `(row, ids[row])`. Each
// prefix position writes exactly one slot — indices across rows are
// independent so no atomics are needed (the kernel only writes one
// element per launched thread, into a per-row strip that the next row's
// thread will not touch).
//
// Grid: `ceil(prefix_rows / block)`; one thread per prefix row. Negative
// or out-of-range indices are silently skipped (matches
// `cpu_gather_last_dim_backward` and the host scatter_add fallback).

extern "C" __global__ void gather_last_dim_backward_f32(
    float* __restrict__ grad_input,
    const float* __restrict__ upstream,
    const int* __restrict__ ids,
    int prefix_rows,
    int vocab
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= prefix_rows) {
        return;
    }
    int id = ids[row];
    if (id < 0 || id >= vocab) {
        return;
    }
    grad_input[row * vocab + id] = upstream[row];
}

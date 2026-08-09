// Right-aligned broadcast-add.
//
// `b_strides` is precomputed with right-alignment: broadcast axes (b-dim == 1
// or missing) get stride 0; matching axes get the contiguous row-major stride.
// The per-element b offset is sum(coord[d] * b_strides[d]).
extern "C" __global__ void add_broadcast_f32(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ b_strides,
    int out_rank,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int b_off = 0;
    int linear = idx;
    for (int d = out_rank - 1; d >= 0; --d) {
        int dim = out_shape[d];
        int coord = linear % dim;
        linear /= dim;
        b_off += coord * b_strides[d];
    }
    out[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[b_off]));
}

// Right-aligned broadcast-copy. Same stride convention as add_broadcast_f32.
//
// Pure expand for GQA repeat_kv: no zeroed `a` carrier, output written in full.
extern "C" __global__ void broadcast_copy_f32(
    const T* __restrict__ src,
    T* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ src_strides,
    int out_rank,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int src_off = 0;
    int linear = idx;
    for (int d = out_rank - 1; d >= 0; --d) {
        int dim = out_shape[d];
        int coord = linear % dim;
        linear /= dim;
        src_off += coord * src_strides[d];
    }
    out[idx] = src[src_off];
}

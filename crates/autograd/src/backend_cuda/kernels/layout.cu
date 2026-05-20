extern "C" __global__ void transpose_axes_swap_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const int* __restrict__ old_shape,
    const int* __restrict__ new_shape,
    int rank,
    int axis1,
    int axis2,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int linear = idx;
    int old_offset = 0;
    for (int d = rank - 1; d >= 0; --d) {
        int coord = linear % new_shape[d];
        linear /= new_shape[d];

        int old_axis = d;
        if (d == axis1) {
            old_axis = axis2;
        } else if (d == axis2) {
            old_axis = axis1;
        }

        int stride = 1;
        for (int s = rank - 1; s > old_axis; --s) {
            stride *= old_shape[s];
        }
        old_offset += coord * stride;
    }
    out[idx] = x[old_offset];
}

extern "C" __global__ void slice_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const int* __restrict__ old_shape,
    const int* __restrict__ starts,
    const int* __restrict__ new_shape,
    int rank,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int linear = idx;
    int old_offset = 0;
    for (int d = rank - 1; d >= 0; --d) {
        int coord = linear % new_shape[d];
        linear /= new_shape[d];

        int stride = 1;
        for (int s = rank - 1; s > d; --s) {
            stride *= old_shape[s];
        }
        old_offset += (starts[d] + coord) * stride;
    }
    out[idx] = x[old_offset];
}

extern "C" __global__ void slice_backward_f32(
    float* __restrict__ grad,
    const float* __restrict__ upstream,
    const int* __restrict__ input_shape,
    const int* __restrict__ starts,
    const int* __restrict__ upstream_shape,
    int rank,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int linear = idx;
    int input_offset = 0;
    for (int d = rank - 1; d >= 0; --d) {
        int coord = linear % upstream_shape[d];
        linear /= upstream_shape[d];

        int stride = 1;
        for (int s = rank - 1; s > d; --s) {
            stride *= input_shape[s];
        }
        input_offset += (starts[d] + coord) * stride;
    }
    grad[input_offset] = upstream[idx];
}

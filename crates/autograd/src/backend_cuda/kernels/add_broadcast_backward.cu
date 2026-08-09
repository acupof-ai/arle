// add_broadcast backward: grad_b[j] = sum of upstream over axes broadcast from b.
//
// Layout contract:
//   out_shape  = a_shape (row-major, length out_rank).
//   b_strides  : right-aligned to out_rank; 0 on contracted (broadcast) axes,
//                else b's contiguous row-major stride.
//   out_strides: contiguous row-major strides in upstream.
//
// One block per b output element. Threads stride over the cartesian product
// of contracted-axis ranges, partial-sum upstream, then shared-memory reduce.
// Grid: (b_size, 1, 1). Block: (BLOCK=256, 1, 1). Shared: BLOCK * sizeof(float).
// out_rank capped at 8 — Qwen3.5 broadcasts are rank <= 3.

#ifndef ARLE_AB_BWD_MAX_RANK
#define ARLE_AB_BWD_MAX_RANK 8
#endif

extern "C" __global__ void add_broadcast_backward_f32(
    T* __restrict__ grad_b,
    const T* __restrict__ upstream,
    const int* __restrict__ out_shape,
    const int* __restrict__ b_strides,    // 0 on contracted axes
    const int* __restrict__ out_strides,
    int out_rank,
    int b_idx_total,
    int contract_total
) {
    extern __shared__ float smem[];
    int b_idx = blockIdx.x;
    if (b_idx >= b_idx_total) return;
    int tid = threadIdx.x;
    int block = blockDim.x;

    int fixed_coord[ARLE_AB_BWD_MAX_RANK];
    int contract_dim[ARLE_AB_BWD_MAX_RANK];
    int contract_axis[ARLE_AB_BWD_MAX_RANK];
    int num_contract = 0;

    // Non-contracted axes (b_strides[d] > 0) carry b's contiguous row-major
    // strides, so coord = (b_idx / b_strides[d]) % out_shape[d] decodes them.
    int remaining = b_idx;
    // Contracted axes stay 0 — their coords come from the inner sweep.
    for (int d = 0; d < out_rank; ++d) {
        fixed_coord[d] = 0;
    }
    for (int d = 0; d < out_rank; ++d) {
        int s = b_strides[d];
        if (s != 0) {
            int dim = out_shape[d];
            fixed_coord[d] = (remaining / s) % dim;
        }
    }
    for (int d = 0; d < out_rank; ++d) {
        if (b_strides[d] == 0) {
            contract_axis[num_contract] = d;
            contract_dim[num_contract] = out_shape[d];
            num_contract++;
        }
    }

    float local_sum = 0.0f;
    for (int k = tid; k < contract_total; k += block) {
        int coord[ARLE_AB_BWD_MAX_RANK];
        int rem = k;
        for (int j = num_contract - 1; j >= 0; --j) {
            int dim = contract_dim[j];
            coord[j] = rem % dim;
            rem /= dim;
        }
        int lin = 0;
        for (int d = 0; d < out_rank; ++d) {
            lin += fixed_coord[d] * out_strides[d];
        }
        for (int j = 0; j < num_contract; ++j) {
            int d = contract_axis[j];
            lin += coord[j] * out_strides[d];
        }
        local_sum += static_cast<float>(upstream[lin]);
    }

    smem[tid] = local_sum;
    __syncthreads();
    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) smem[tid] += smem[tid + step];
        __syncthreads();
    }
    if (tid == 0) {
        grad_b[b_idx] = static_cast<T>(smem[0]);
    }
}

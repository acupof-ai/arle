// Wave 2 Commit A: device-resident backward for `add_broadcast`.
//
// Forward `add_broadcast`:
//   out[i] = a[i] + b[broadcast_offset(i, a_shape, b_shape)]   // shape a_shape
//
// Backward grad_b: sum over contracted (broadcast) axes:
//   grad_b[j] = sum_{i: broadcast_offset(i, a_shape, b_shape) == j} upstream[i]
//
// (grad_a is just upstream — handled in the ops layer.)
//
// Layout contract (mirrors add_broadcast.cu forward):
//   `out_shape`  is `a_shape`            (len = out_rank, row-major).
//   `b_strides`  is right-aligned of length `out_rank`. Entries where the
//   broadcast axis collapses (b-dim == 1 or axis missing from b_shape) are
//   stride 0; matching axes carry the contiguous row-major stride in b.
//   `out_strides` is the contiguous row-major stride in upstream, length
//   out_rank.
//
// Strategy: one BLOCK per output element `b_idx ∈ [0, b_size)`. The block:
//   1. Decodes `b_idx` into per-axis coordinates ONLY for non-contracted
//      axes (where b_strides[d] != 0). Contracted axes (b_strides[d] == 0)
//      get coordinate 0 in the b-decode but will be swept by threads.
//   2. Threads in the block stride through the cartesian product of all
//      contracted-axis ranges (size `contract_total`). Each thread computes
//      a per-thread partial sum, accumulating upstream[linear_in_a]
//      reconstructed from the fixed coords + contracted coords.
//   3. Shared-memory tree reduction → block thread 0 writes grad_b[b_idx].
//
// Grid: (b_size, 1, 1). Block: (BLOCK=256, 1, 1). Shared: BLOCK * sizeof(float).
// out_rank capped at 8 (more than enough for our tensor shapes; matches the
// recursion depth we never push past 4 in practice — Qwen3.5 broadcasts are
// rank ≤ 3).

#ifndef ARLE_AB_BWD_MAX_RANK
#define ARLE_AB_BWD_MAX_RANK 8
#endif

extern "C" __global__ void add_broadcast_backward_f32(
    float* __restrict__ grad_b,
    const float* __restrict__ upstream,
    const int* __restrict__ out_shape,    // a_shape, length out_rank
    const int* __restrict__ b_strides,    // length out_rank, 0 on contracted axes
    const int* __restrict__ out_strides,  // contiguous row-major strides in upstream
    int out_rank,
    int b_idx_total,
    int contract_total
) {
    extern __shared__ float smem[];
    int b_idx = blockIdx.x;
    if (b_idx >= b_idx_total) return;
    int tid = threadIdx.x;
    int block = blockDim.x;

    // 1. Decode `b_idx` into per-axis coords for non-contracted axes,
    //    and harvest the list of contracted-axis dims.
    int fixed_coord[ARLE_AB_BWD_MAX_RANK];
    int contract_dim[ARLE_AB_BWD_MAX_RANK];
    int contract_axis[ARLE_AB_BWD_MAX_RANK];
    int num_contract = 0;

    // Walk through `b_idx` in the canonical b-layout: the non-contracted
    // axes carry b's contiguous strides (their values match what
    // host-side `broadcast_strides(b_shape)` produces, just embedded in the
    // out_rank-length b_strides). We reverse-engineer coord by repeatedly
    // dividing by the matching axis dim, walking axes from low-stride to
    // high-stride. To do this without a sort we iterate `out_rank` times
    // finding the next-smallest non-zero stride.
    //
    // out_rank ≤ ARLE_AB_BWD_MAX_RANK (8), so the nested loops are tiny.
    int remaining = b_idx;
    // Initialize fixed_coord to 0; contracted axes will remain 0 (their
    // value is supplied by the inner sweep instead).
    for (int d = 0; d < out_rank; ++d) {
        fixed_coord[d] = 0;
    }
    // For each non-contracted axis (b_strides[d] > 0), decode coord =
    // (b_idx / b_strides[d]) % out_shape[d]. This works because b's
    // non-contracted axes are laid out contiguously in b's row-major
    // memory (matching the broadcast_strides() helper).
    for (int d = 0; d < out_rank; ++d) {
        int s = b_strides[d];
        if (s != 0) {
            int dim = out_shape[d];
            fixed_coord[d] = (remaining / s) % dim;
        }
    }
    // Harvest contracted axes (b_strides[d] == 0) into a compact list.
    for (int d = 0; d < out_rank; ++d) {
        if (b_strides[d] == 0) {
            contract_axis[num_contract] = d;
            contract_dim[num_contract] = out_shape[d];
            num_contract++;
        }
    }

    // 2. Stride through the cartesian product of contracted-axis coords.
    float local_sum = 0.0f;
    for (int k = tid; k < contract_total; k += block) {
        // Decode k → per-contracted-axis coord.
        int coord[ARLE_AB_BWD_MAX_RANK];
        int rem = k;
        for (int j = num_contract - 1; j >= 0; --j) {
            int dim = contract_dim[j];
            coord[j] = rem % dim;
            rem /= dim;
        }
        // Compute linear index in upstream from fixed_coord + contracted coords.
        int lin = 0;
        // Sum fixed contributions.
        for (int d = 0; d < out_rank; ++d) {
            lin += fixed_coord[d] * out_strides[d];
        }
        // Add contracted contributions.
        for (int j = 0; j < num_contract; ++j) {
            int d = contract_axis[j];
            lin += coord[j] * out_strides[d];
        }
        local_sum += upstream[lin];
    }

    // 3. Shared-memory tree reduction.
    smem[tid] = local_sum;
    __syncthreads();
    for (int step = block / 2; step > 0; step >>= 1) {
        if (tid < step) smem[tid] += smem[tid + step];
        __syncthreads();
    }
    if (tid == 0) {
        grad_b[b_idx] = smem[0];
    }
}

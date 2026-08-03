// Wave 2.1: NeoX-style RoPE backward. Identity:
//   forward:  y0 = x0*cos - x1*sin,    y1 = x1*cos + x0*sin
//   backward: gx0 = gy0*cos + gy1*sin, gx1 = gy1*cos - gy0*sin
//           = rope_forward(gy, cos, -sin)
//
// Same launch shape as `rope_f32` (grid: (batch*heads*seq, 1, 1); block:
// min(half_dim, 256)); the only sign difference is on the `sin` term —
// `cpu_rope_backward` does `neg_forward(sin)` host-side and re-invokes
// `rope_forward`. We mirror that on-device but skip the negate kernel:
// it's just sign flipping per-element which we can do inline in the
// backward kernel.

extern "C" __global__ void rope_backward_f32(
    T* __restrict__ grad_x,
    const T* __restrict__ upstream,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int batch,
    int heads,
    int seq,
    int head_dim,
    int rot_half  // rotary_dim/2; grads of the non-rotary tail pass through.
) {
    const int row = blockIdx.x;
    const int total_rows = batch * heads * seq;
    if (row >= total_rows) {
        return;
    }
    const int token = row % seq;
    const int row_base = row * head_dim;
    const int cache_base = token * rot_half;

    for (int i = threadIdx.x; i < rot_half; i += blockDim.x) {
        const float gy0 = static_cast<float>(upstream[row_base + i]);
        const float gy1 = static_cast<float>(upstream[row_base + i + rot_half]);
        const float c = cos_table[cache_base + i];
        const float s = sin_table[cache_base + i];
        // Inline `sin -> -sin` versus the forward kernel.
        grad_x[row_base + i] = static_cast<T>(gy0 * c + gy1 * s);
        grad_x[row_base + i + rot_half] = static_cast<T>(gy1 * c - gy0 * s);
    }
    for (int i = 2 * rot_half + threadIdx.x; i < head_dim; i += blockDim.x) {
        grad_x[row_base + i] = upstream[row_base + i];
    }
}

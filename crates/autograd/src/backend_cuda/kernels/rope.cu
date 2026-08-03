// NeoX-style rotary position embedding — matches autograd `cpu_rope_forward`
// and `ops::rope::rope`. Element i pairs with i+half_dim:
//   out[i]            = x[i]           * cos[i] - x[i+half_dim] * sin[i]
//   out[i+half_dim]   = x[i+half_dim]  * cos[i] + x[i]          * sin[i]
//
// Grid: (batch*heads*seq, 1, 1); block: (half_dim capped to 256, 1, 1).
// Each block handles one (batch, head, token) triple; `cos`/`sin` have
// shape [seq, half_dim] row-major and are indexed by token.
extern "C" __global__ void rope_f32(
    T* __restrict__ out,
    const T* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int batch,
    int heads,
    int seq,
    int head_dim,
    int rot_half  // rotary_dim/2 (== head_dim/2 for full rotary). Pairs live
                  // inside the rotary segment; dims beyond it pass through
                  // (partial_rotary_factor models, e.g. Qwen3.6's 0.25).
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
        const float x0 = static_cast<float>(x[row_base + i]);
        const float x1 = static_cast<float>(x[row_base + i + rot_half]);
        const float c = cos_table[cache_base + i];
        const float s = sin_table[cache_base + i];
        out[row_base + i] = static_cast<T>(x0 * c - x1 * s);
        out[row_base + i + rot_half] = static_cast<T>(x1 * c + x0 * s);
    }
    for (int i = 2 * rot_half + threadIdx.x; i < head_dim; i += blockDim.x) {
        out[row_base + i] = x[row_base + i];
    }
}

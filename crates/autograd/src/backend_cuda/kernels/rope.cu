// NeoX-style rotary position embedding — matches autograd `cpu_rope_forward`
// and `ops::rope::rope`. Element i pairs with i+half_dim:
//   out[i]            = x[i]           * cos[i] - x[i+half_dim] * sin[i]
//   out[i+half_dim]   = x[i+half_dim]  * cos[i] + x[i]          * sin[i]
//
// Grid: (batch*heads*seq, 1, 1); block: (half_dim capped to 256, 1, 1).
// Each block handles one (batch, head, token) triple; `cos`/`sin` have
// shape [seq, half_dim] row-major and are indexed by token.
extern "C" __global__ void rope_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int batch,
    int heads,
    int seq,
    int head_dim
) {
    const int row = blockIdx.x;
    const int total_rows = batch * heads * seq;
    if (row >= total_rows) {
        return;
    }
    const int half_dim = head_dim >> 1;
    const int token = row % seq;
    const int row_base = row * head_dim;
    const int cache_base = token * half_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        const float x0 = x[row_base + i];
        const float x1 = x[row_base + i + half_dim];
        const float c = cos_table[cache_base + i];
        const float s = sin_table[cache_base + i];
        out[row_base + i] = x0 * c - x1 * s;
        out[row_base + i + half_dim] = x1 * c + x0 * s;
    }
}

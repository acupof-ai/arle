// Gather along the last axis: out[i] = src[i * vocab + ids[i]].
// Grid: ceil(n / block); one thread per output element. Ids are i32 and
// assumed pre-validated by the caller (autograd op wrapper performs the
// bounds check so the error carries the original usize).
extern "C" __global__ void gather_last_dim_f32(
    float* __restrict__ out,
    const float* __restrict__ src,
    const int* __restrict__ ids,
    int n,
    int vocab
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int id = ids[idx];
    if (id < 0 || id >= vocab) {
        out[idx] = 0.0f;
        return;
    }
    out[idx] = src[idx * vocab + id];
}

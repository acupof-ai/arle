// Device-resident gradient accumulation. Reads `dest[i]` and `src[i]`,
// writes the sum into a fresh output buffer (not in-place — the caller
// allocates `out` so the previous `dest` handle remains valid for any
// other consumers still holding it on the autograd tape).
//
// Foundation for the post-G3 device-resident gradient tape — see
// docs/research/2026-05-17-cuda-training-architectural-correction.md.
extern "C" __global__ void add_into_f32(
    float* out,
    const float* dest,
    const float* src,
    int n
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        out[i] = dest[i] + src[i];
    }
}

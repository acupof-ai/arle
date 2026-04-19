// SiLU (Swish) = x * sigmoid(x). Qwen3 SwiGLU uses silu on the gate projection.
// One element per thread; simple 1D launch with block=256.

extern "C" __global__ void silu_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    int n
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        float v = x[i];
        // sigmoid(v) = 1 / (1 + exp(-v)) — use __expf for device throughput.
        float s = 1.0f / (1.0f + __expf(-v));
        out[i] = v * s;
    }
}

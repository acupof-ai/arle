// Wave 2.1: elementwise activation backward kernels. Each is a single 1D
// kernel `dx[i] = dy[i] * f'(x[i])` (or f'(y) where the derivative is
// expressed in terms of the saved output). Block=256 via the shared
// `launch_1d` helper.
//
// SiLU backward consumes the saved INPUT (x): silu'(x) = sigmoid(x) * (1 +
// x * (1 - sigmoid(x))) — matches `cpu_silu_backward` exactly.
// GELU backward consumes the saved INPUT (x): the erf-form derivative
//   gelu'(x) = 0.5 * (1 + erf(x / sqrt(2))) + x * (1/sqrt(2π)) * exp(-x²/2)
// — matches `cpu_gelu_backward` (and the autograd-side `gelu_host_eager`
// forward is the erf form, not the tanh approximation).
// Sigmoid backward consumes the saved OUTPUT (y): sigmoid'(y) = y * (1-y).
// Exp backward consumes the saved OUTPUT (y): exp'(x) = exp(x) = y.

extern "C" __global__ void silu_backward_f32(
    float* __restrict__ grad_input,
    const float* __restrict__ upstream,
    const float* __restrict__ x,
    int n
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float s = 1.0f / (1.0f + __expf(-v));
        float deriv = s + (v * s * (1.0f - s));
        grad_input[i] = upstream[i] * deriv;
    }
}

// Erf-form GELU derivative — matches the autograd `gelu_host_eager`
// forward and `cpu_gelu_backward`. INV_SQRT_2 = 0.707_106_77,
// INV_SQRT_2PI = 0.398_942_3.
extern "C" __global__ void gelu_backward_f32(
    float* __restrict__ grad_input,
    const float* __restrict__ upstream,
    const float* __restrict__ x,
    int n
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float erf_term = erff(v * 0.70710677f);
        float exp_term = __expf(-0.5f * v * v);
        float deriv = 0.5f * (1.0f + erf_term) + (v * 0.3989423f * exp_term);
        grad_input[i] = upstream[i] * deriv;
    }
}

extern "C" __global__ void sigmoid_backward_f32(
    float* __restrict__ grad_input,
    const float* __restrict__ upstream,
    const float* __restrict__ y,
    int n
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        float yv = y[i];
        grad_input[i] = upstream[i] * yv * (1.0f - yv);
    }
}

extern "C" __global__ void exp_backward_f32(
    float* __restrict__ grad_input,
    const float* __restrict__ upstream,
    const float* __restrict__ y,
    int n
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        grad_input[i] = upstream[i] * y[i];
    }
}

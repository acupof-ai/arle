extern "C" __global__ void add_f32(float* out, const float* a, const float* b, int n) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

extern "C" __global__ void mul_f32(float* out, const float* a, const float* b, int n) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

extern "C" __global__ void mul_scalar_f32(float* out, const float* a, float s, int n) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * s;
    }
}

extern "C" __global__ void gelu_f32(float* out, const float* a, int n) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        float x = a[i];
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + (0.044715f * x3));
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

extern "C" __global__ void exp_f32(float* out, const float* a, int n) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        out[i] = expf(a[i]);
    }
}

extern "C" __global__ void neg_f32(float* out, const float* a, int n) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        out[i] = -a[i];
    }
}

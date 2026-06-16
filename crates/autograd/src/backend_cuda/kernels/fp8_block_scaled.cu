extern "C" __device__ __forceinline__ float arle_fp8_e4m3_to_f32(unsigned char bits) {
    const float sign = (bits & 0x80) ? -1.0f : 1.0f;
    const int exp = (bits >> 3) & 0x0f;
    const int mant = bits & 0x07;
    if (exp == 0) {
        if (mant == 0) return sign * 0.0f;
        return sign * ldexpf(static_cast<float>(mant) * 0.125f, -6);
    }
    if (exp == 0x0f && mant == 0x07) {
        return nanf("");
    }
    return sign * ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
}

extern "C" __device__ __forceinline__ float arle_fp8_block_weight(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scales,
    int row,
    int col,
    int cols,
    int block_m,
    int block_k,
    int scale_cols)
{
    const float scale = scales[(row / block_m) * scale_cols + (col / block_k)];
    return arle_fp8_e4m3_to_f32(weight[row * cols + col]) * scale;
}

extern "C" __global__ void fp8_block_scaled_matmul_bt_f32(
    const float* __restrict__ a,              // [M, K]
    const unsigned char* __restrict__ weight, // [N, K] FP8 E4M3
    const float* __restrict__ scales,         // [ceil(N/BM), ceil(K/BK)]
    float* __restrict__ out,                  // [M, N]
    int M,
    int N,
    int K,
    int block_m,
    int block_k,
    int scale_cols)
{
    const int m = blockIdx.x;
    const int n = blockIdx.y;
    if (m >= M || n >= N) return;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float w = arle_fp8_block_weight(weight, scales, n, k, K, block_m, block_k, scale_cols);
        sum += a[m * K + k] * w;
    }
    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[m * N + n] = scratch[0];
    }
}

extern "C" __global__ void fp8_block_scaled_matmul_f32(
    const float* __restrict__ a,              // [M, N]
    const unsigned char* __restrict__ weight, // [N, K] FP8 E4M3
    const float* __restrict__ scales,         // [ceil(N/BM), ceil(K/BK)]
    float* __restrict__ out,                  // [M, K]
    int M,
    int N,
    int K,
    int block_m,
    int block_k,
    int scale_cols)
{
    const int m = blockIdx.x;
    const int k = blockIdx.y;
    if (m >= M || k >= K) return;

    float sum = 0.0f;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        const float w = arle_fp8_block_weight(weight, scales, n, k, K, block_m, block_k, scale_cols);
        sum += a[m * N + n] * w;
    }
    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[m * K + k] = scratch[0];
    }
}

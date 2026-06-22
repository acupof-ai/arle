// Self-contained CUDA micro-benchmark for the Qwen W4A16 (NVFP4 / FP4-E2M1
// group-scaled) decode GEMV — the load-bearing validation for the FP4 direction:
// does a 4-bit-weight GEMV actually run ~2x the FP8 GEMV at B=1 decode (halving
// the 27GB weight read → roofline ~296 t/s → 180 reachable), or is it
// dequant-bound (nibble-unpack + E2M1 LUT + group scale) and fails to deliver?
//
//   nvcc -O3 -arch=sm_90 gemv_fp4_bench.cu -o gemv_fp4_bench && ./gemv_fp4_bench 4000 [B]
//
// FP4 reads N*K/2 weight bytes (half of FP8's N*K). %roofline is computed on the
// FP4 byte count, so a same-HBM-ceiling kernel reports the SAME %roofline as FP8
// but at HALF the absolute ms — that 2x ms drop is the decode win.
//
// Kernel + helpers extracted VERBATIM from
//   crates/cuda-kernels/csrc/gemm/quantized_gemv.cu
//     (fp4_e2m1_group_gemv_batch_kernel + fp4_e2m1_group_scale + decoders).

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define WARP_SIZE 32
#define GEMV_THREADS 256
#define GEMV_ROWS 4

__device__ __constant__ float FP4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float decode_fp8_e4m3(uint8_t bits) {
    if ((bits & 0x7f) == 0) return 0.0f;
    if ((bits & 0x7f) == 0x7f) return (bits & 0x80) ? -448.0f : 448.0f;
    __nv_fp8_e4m3 v; v.__x = bits; return static_cast<float>(v);
}

__device__ __forceinline__ float decode_fp4_e2m1(uint8_t bits) {
    return FP4_E2M1_LUT[bits & 0x0f];
}

__device__ __forceinline__ float fp4_group_scale(
    const uint8_t* __restrict__ scales, const float* __restrict__ global_scales,
    int row, int col, int scale_cols, int group_size)
{
    const int g_raw = col / group_size;
    const int g = g_raw < scale_cols ? g_raw : (scale_cols - 1);
    return decode_fp8_e4m3(scales[row * scale_cols + g]) * global_scales[0];
}

// VERBATIM: fp4_e2m1_group_gemv_batch_kernel (quantized_gemv.cu)
__global__ void fp4_e2m1_group_gemv_batch_kernel(
    const uint8_t* __restrict__ weight, const uint8_t* __restrict__ scales,
    const float* __restrict__ global_scales, const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output, int B, int N, int K, int group_size, int scale_cols)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_idx = blockIdx.y;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N || batch_idx >= B) return;

    const int bytes_per_row = K / 2;
    const __nv_bfloat16* x = input + batch_idx * K;
    float sum = 0.0f;
    for (int pair = tid_in_row; pair < bytes_per_row; pair += threads_per_row) {
        const int k0 = pair << 1;
        const int k1 = k0 + 1;
        const uint8_t packed = weight[row * bytes_per_row + pair];
        const uint8_t lo = packed & 0x0f;
        const uint8_t hi = (packed >> 4) & 0x0f;
        const float w0 = decode_fp4_e2m1(lo) * fp4_group_scale(scales, global_scales, row, k0, scale_cols, group_size);
        const float w1 = decode_fp4_e2m1(hi) * fp4_group_scale(scales, global_scales, row, k1, scale_cols, group_size);
        sum += w0 * __bfloat162float(x[k0]);
        sum += w1 * __bfloat162float(x[k1]);
    }
    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++) total += smem[row_in_block * warps_per_row + w];
        output[batch_idx * N + row] = __float2bfloat16(total);
    }
}

// Oracle: one thread per (batch,row), same decode math, fp32 accumulate.
__global__ void fp4_reference_kernel(
    const uint8_t* __restrict__ weight, const uint8_t* __restrict__ scales,
    const float* __restrict__ global_scales, const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output, int N, int K, int group_size, int scale_cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const int bytes_per_row = K / 2;
    float sum = 0.0f;
    for (int pair = 0; pair < bytes_per_row; ++pair) {
        const int k0 = pair << 1, k1 = k0 + 1;
        const uint8_t packed = weight[(int64_t)row * bytes_per_row + pair];
        const float w0 = decode_fp4_e2m1(packed & 0x0f) * fp4_group_scale(scales, global_scales, row, k0, scale_cols, group_size);
        const float w1 = decode_fp4_e2m1((packed >> 4) & 0x0f) * fp4_group_scale(scales, global_scales, row, k1, scale_cols, group_size);
        sum += w0 * __bfloat162float(input[k0]) + w1 * __bfloat162float(input[k1]);
    }
    output[row] = __float2bfloat16(sum);
}

#define CUDA_CHECK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
    cudaGetErrorString(_e)); std::exit(1);} } while(0)

struct Shape { const char* name; int N; int K; };

int main(int argc, char** argv) {
    const double roofline_GBs = (argc > 1) ? atof(argv[1]) : 4000.0;
    const int B = (argc > 2) ? std::max(1, atoi(argv[2])) : 1;
    const int group_size = 16;  // NVFP4 group

    const size_t flush_bytes = 512ull << 20;
    unsigned char* d_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));

    const Shape shapes[] = {
        {"qkv", 6144, 5120}, {"o_proj", 5120, 6144},
        {"gate", 17408, 5120}, {"down", 5120, 17408},
    };
    const int num_shapes = 4;
    const int warmup_iters = 50, timed_iters = 200;
    std::printf("FP4 W4A16  Roofline=%.1f GB/s  B=%d group=%d  (bytes=N*K/2)\n\n",
                roofline_GBs, B, group_size);

    double sum_pct = 0.0, total_ms = 0.0; int passes = 0;
    for (int s = 0; s < num_shapes; ++s) {
        const int N = shapes[s].N, K = shapes[s].K;
        const int scale_cols = (K + group_size - 1) / group_size;
        const size_t wbytes = (size_t)N * K / 2;            // FP4 packed
        const size_t selems = (size_t)N * scale_cols;

        std::vector<uint8_t> h_w(wbytes), h_s(selems);
        std::vector<__nv_bfloat16> h_x((size_t)K * B);
        std::vector<float> h_g(1);
        uint64_t rng = 0x9E3779B97F4A7C15ull ^ (uint64_t)(s + 1);
        for (size_t i = 0; i < wbytes; ++i) { rng = rng*6364136223846793005ull+1442695040888963407ull; h_w[i] = (uint8_t)(rng & 0xFF); }
        for (size_t i = 0; i < selems; ++i) { rng = rng*6364136223846793005ull+1442695040888963407ull;
            // fp8 e4m3 scale ~1.0: code around 0x38..0x40, avoid 0/NaN
            uint8_t mag = 0x30 + (uint8_t)(rng % 0x10); h_s[i] = mag; }
        h_g[0] = 1.0f;
        for (size_t k = 0; k < (size_t)K*B; ++k) { rng = rng*6364136223846793005ull+1442695040888963407ull;
            double f = (double)((rng>>11)&0xFFFFF)/(double)0xFFFFF; h_x[k] = __float2bfloat16((float)(f*2.0-1.0)); }

        uint8_t *d_w, *d_s; float* d_g; __nv_bfloat16 *d_x, *d_o, *d_r;
        CUDA_CHECK(cudaMalloc(&d_w, wbytes)); CUDA_CHECK(cudaMalloc(&d_s, selems));
        CUDA_CHECK(cudaMalloc(&d_g, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x, (size_t)K*B*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_o, (size_t)N*B*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_r, (size_t)N*B*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), wbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s, h_s.data(), selems, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)K*B*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        dim3 grid((N + GEMV_ROWS - 1)/GEMV_ROWS, B), block(GEMV_THREADS);
        for (int b = 0; b < B; ++b)
            fp4_reference_kernel<<<(N+127)/128,128>>>(d_w, d_s, d_g, d_x+(size_t)b*K, d_r+(size_t)b*N, N, K, group_size, scale_cols);
        fp4_e2m1_group_gemv_batch_kernel<<<grid,block>>>(d_w,d_s,d_g,d_x,d_o,B,N,K,group_size,scale_cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<__nv_bfloat16> h_o((size_t)N*B), h_rf((size_t)N*B);
        CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, (size_t)N*B*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rf.data(), d_r, (size_t)N*B*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        double dot=0,na=0,nb=0,me=0;
        for (int i=0;i<N*B;++i){double a=__bfloat162float(h_o[i]),b=__bfloat162float(h_rf[i]);dot+=a*b;na+=a*a;nb+=b*b;me=std::max(me,std::fabs(a-b));}
        double cos = (na>0&&nb>0)?dot/(std::sqrt(na)*std::sqrt(nb)):0.0;
        bool pass = cos > 0.9999; if (pass) ++passes;

        for (int i=0;i<warmup_iters;++i) fp4_e2m1_group_gemv_batch_kernel<<<grid,block>>>(d_w,d_s,d_g,d_x,d_o,B,N,K,group_size,scale_cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t st,sp; CUDA_CHECK(cudaEventCreate(&st)); CUDA_CHECK(cudaEventCreate(&sp));
        std::vector<float> ts(timed_iters);
        for (int i=0;i<timed_iters;++i){
            CUDA_CHECK(cudaMemsetAsync(d_flush,(int)(i&0xFF),flush_bytes,0));
            CUDA_CHECK(cudaEventRecord(st,0));
            fp4_e2m1_group_gemv_batch_kernel<<<grid,block>>>(d_w,d_s,d_g,d_x,d_o,B,N,K,group_size,scale_cols);
            CUDA_CHECK(cudaEventRecord(sp,0)); CUDA_CHECK(cudaEventSynchronize(sp));
            float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,st,sp)); ts[i]=ms;
        }
        std::sort(ts.begin(),ts.end());
        double med = ts[timed_iters/2];
        double gbs = (double)wbytes/(med*1e-3)/1e9, pct = 100.0*gbs/roofline_GBs;
        total_ms += med; sum_pct += pct;
        std::printf("SHAPE %-7s N=%-6d K=%-6d  %s  cosine=%.5f max_err=%.3g  median_ms=%.4f  GB/s=%.0f  %%roofline=%.1f\n",
                    shapes[s].name,N,K,pass?"PASS":"FAIL",cos,me,med,gbs,pct);
        CUDA_CHECK(cudaEventDestroy(st)); CUDA_CHECK(cudaEventDestroy(sp));
        cudaFree(d_w);cudaFree(d_s);cudaFree(d_g);cudaFree(d_x);cudaFree(d_o);cudaFree(d_r);
    }
    std::printf("\nTOTAL  sum_median_ms=%.4f  avg_%%roofline=%.1f  correctness=%d/%d PASS\n",
                total_ms, sum_pct/num_shapes, passes, num_shapes);
    return passes==num_shapes ? 0 : 2;
}

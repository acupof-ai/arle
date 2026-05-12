// Standalone W4A8/FP4 KV dequant smoke.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_89 scripts/kv_w4a8_smoke.cu -o /tmp/kv_w4a8_smoke
//
// This is intentionally not wired into Cargo/build.rs. It is a Phase-0
// license-or-kill probe for KV read + scale + dequant cost before any runtime
// W4A8/FP4 KV implementation.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kBatch = 4;
// 64K context keeps every path above RTX 4070 Ti SUPER's 48 MiB L2:
// BF16 KV ~= 320 MiB, FP8 KV+scales ~= 168 MiB, FP4 KV+scales ~= 88 MiB.
constexpr int kSeqLen = 65536;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 80;
constexpr int kElems = kBatch * kSeqLen * kKvHeads * kHeadDim;
constexpr int kScaleElems = kBatch * kSeqLen * kKvHeads;
constexpr int kThreads = 256;
constexpr int kBlocks = 256;
constexpr int kIters = 50;

__constant__ float kFp4E2M1Lut[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

uint16_t float_to_bf16_bits(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounded = bits + 0x7fffu + lsb;
    return static_cast<uint16_t>(rounded >> 16);
}

template <typename T>
T* device_copy(const std::vector<T>& host) {
    T* ptr = nullptr;
    check(cudaMalloc(&ptr, host.size() * sizeof(T)), "cudaMalloc");
    check(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D");
    return ptr;
}

__device__ __forceinline__ float dim_weight(int elem_idx) {
    int d = elem_idx % kHeadDim;
    return 0.0009765625f * static_cast<float>((d % 17) + 1);
}

__device__ __forceinline__ void reduce_store(float acc, float* block_sums) {
    __shared__ float smem[kThreads];
    int tid = threadIdx.x;
    smem[tid] = acc;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_sums[blockIdx.x] = smem[0];
    }
}

__global__ void scan_bf16_kernel(const __nv_bfloat16* __restrict__ kv,
                                 float* __restrict__ block_sums) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (int i = tid; i < kElems; i += stride) {
        float v = __bfloat162float(kv[i]);
        acc = fmaf(v, dim_weight(i), acc);
    }
    reduce_store(acc, block_sums);
}

__global__ void scan_fp8_kernel(const __nv_fp8_e4m3* __restrict__ kv,
                                const float* __restrict__ scales,
                                float* __restrict__ block_sums) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (int i = tid; i < kElems; i += stride) {
        int scale_idx = i / kHeadDim;
        float v = static_cast<float>(kv[i]) * scales[scale_idx];
        acc = fmaf(v, dim_weight(i), acc);
    }
    reduce_store(acc, block_sums);
}

__global__ void scan_fp4_kernel(const uint8_t* __restrict__ kv_packed,
                                const float* __restrict__ scales,
                                float* __restrict__ block_sums) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (int i = tid; i < kElems; i += stride) {
        uint8_t packed = kv_packed[i >> 1];
        uint8_t nibble = (i & 1) ? (packed >> 4) : (packed & 0x0f);
        int scale_idx = i / kHeadDim;
        float v = kFp4E2M1Lut[nibble] * scales[scale_idx];
        acc = fmaf(v, dim_weight(i), acc);
    }
    reduce_store(acc, block_sums);
}

__device__ __forceinline__ float fp4_e2m1_to_float_bits(uint8_t nibble) {
    uint16_t sign = (nibble & 0x08) ? 0x8000u : 0u;
    uint8_t mag = nibble & 0x07;
    uint16_t bits = sign;
    if (mag != 0) {
        uint16_t exponent = static_cast<uint16_t>((mag >> 1) + 126u);
        uint16_t mantissa = ((mag & 1u) && mag != 1u) ? 0x0040u : 0u;
        bits |= static_cast<uint16_t>((exponent << 7) | mantissa);
    }
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__global__ void scan_fp4_bit_kernel(const uint8_t* __restrict__ kv_packed,
                                    const float* __restrict__ scales,
                                    float* __restrict__ block_sums) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (int i = tid; i < kElems; i += stride) {
        uint8_t packed = kv_packed[i >> 1];
        uint8_t nibble = (i & 1) ? (packed >> 4) : (packed & 0x0f);
        int scale_idx = i / kHeadDim;
        float v = fp4_e2m1_to_float_bits(nibble) * scales[scale_idx];
        acc = fmaf(v, dim_weight(i), acc);
    }
    reduce_store(acc, block_sums);
}

template <typename Launch>
float run_kernel(const char* label, Launch launch, float* block_sums) {
    cudaEvent_t start;
    cudaEvent_t stop;
    check(cudaEventCreate(&start), "cudaEventCreate start");
    check(cudaEventCreate(&stop), "cudaEventCreate stop");
    launch(block_sums);
    check(cudaGetLastError(), "warmup launch");
    check(cudaDeviceSynchronize(), "warmup sync");
    check(cudaEventRecord(start), "event start");
    for (int i = 0; i < kIters; ++i) {
        launch(block_sums);
    }
    check(cudaEventRecord(stop), "event stop");
    check(cudaEventSynchronize(stop), "event sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");

    std::vector<float> host_sums(kBlocks);
    check(cudaMemcpy(host_sums.data(), block_sums, host_sums.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy sums");
    double checksum = 0.0;
    for (float v : host_sums) {
        checksum += static_cast<double>(v);
    }
    float us = (ms * 1000.0f) / static_cast<float>(kIters);
    std::printf("%-8s time_us=%.3f checksum=%.6e\n", label, us, checksum);
    check(cudaEventDestroy(start), "destroy start");
    check(cudaEventDestroy(stop), "destroy stop");
    return us;
}

}  // namespace

int main() {
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::printf("gpu=%s sm=%d%d elems=%d scales=%d iters=%d\n",
                prop.name,
                prop.major,
                prop.minor,
                kElems,
                kScaleElems,
                kIters);

    std::vector<uint16_t> bf16(kElems);
    std::vector<uint8_t> fp8(kElems);
    std::vector<uint8_t> fp4((kElems + 1) / 2);
    std::vector<float> scales(kScaleElems);

    const uint8_t fp8_pattern[8] = {0x00, 0x38, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x34};
    const float bf16_pattern[8] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 0.75f};
    for (int i = 0; i < kElems; ++i) {
        bf16[i] = float_to_bf16_bits(bf16_pattern[i & 7]);
        fp8[i] = fp8_pattern[i & 7];
    }
    for (int i = 0; i < kElems; i += 2) {
        uint8_t lo = static_cast<uint8_t>((i / 2) & 0x0f);
        uint8_t hi = static_cast<uint8_t>(((i / 2) + 5) & 0x0f);
        fp4[i >> 1] = static_cast<uint8_t>(lo | (hi << 4));
    }
    for (int i = 0; i < kScaleElems; ++i) {
        scales[i] = 0.125f + static_cast<float>(i % 11) * 0.03125f;
    }

    uint16_t* bf16_dev = device_copy(bf16);
    uint8_t* fp8_dev_u8 = device_copy(fp8);
    uint8_t* fp4_dev = device_copy(fp4);
    float* scales_dev = device_copy(scales);
    float* block_sums = nullptr;
    check(cudaMalloc(&block_sums, kBlocks * sizeof(float)), "alloc sums");

    auto bf16_launch = [=](float* sums) {
        scan_bf16_kernel<<<kBlocks, kThreads>>>(
            reinterpret_cast<const __nv_bfloat16*>(bf16_dev), sums);
    };
    auto fp8_launch = [=](float* sums) {
        scan_fp8_kernel<<<kBlocks, kThreads>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(fp8_dev_u8), scales_dev, sums);
    };
    auto fp4_launch = [=](float* sums) {
        scan_fp4_kernel<<<kBlocks, kThreads>>>(fp4_dev, scales_dev, sums);
    };
    auto fp4_bit_launch = [=](float* sums) {
        scan_fp4_bit_kernel<<<kBlocks, kThreads>>>(fp4_dev, scales_dev, sums);
    };

    float bf16_us = run_kernel("bf16", bf16_launch, block_sums);
    float fp8_us = run_kernel("fp8", fp8_launch, block_sums);
    float fp4_lut_us = run_kernel("fp4_lut", fp4_launch, block_sums);
    float fp4_bit_us = run_kernel("fp4_bit", fp4_bit_launch, block_sums);

    double bf16_bytes = static_cast<double>(kElems) * 2.0;
    double fp8_bytes = static_cast<double>(kElems) * (1.0 + 4.0 / kHeadDim);
    double fp4_bytes = static_cast<double>(kElems) * (0.5 + 4.0 / kHeadDim);
    std::printf("effective_read_gb_s bf16=%.2f fp8=%.2f fp4_lut=%.2f fp4_bit=%.2f\n",
                bf16_bytes / (bf16_us * 1e-6) / 1e9,
                fp8_bytes / (fp8_us * 1e-6) / 1e9,
                fp4_bytes / (fp4_lut_us * 1e-6) / 1e9,
                fp4_bytes / (fp4_bit_us * 1e-6) / 1e9);
    std::printf("speedup_time_vs_bf16 fp8=%.3fx fp4_lut=%.3fx fp4_bit=%.3fx\n",
                bf16_us / fp8_us,
                bf16_us / fp4_lut_us,
                bf16_us / fp4_bit_us);
    std::printf("speedup_time_vs_fp8 fp4_lut=%.3fx fp4_bit=%.3fx\n",
                fp8_us / fp4_lut_us,
                fp8_us / fp4_bit_us);

    check(cudaFree(bf16_dev), "free bf16");
    check(cudaFree(fp8_dev_u8), "free fp8");
    check(cudaFree(fp4_dev), "free fp4");
    check(cudaFree(scales_dev), "free scales");
    check(cudaFree(block_sums), "free sums");
    return 0;
}

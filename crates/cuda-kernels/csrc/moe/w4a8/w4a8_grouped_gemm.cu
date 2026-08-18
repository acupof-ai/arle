// Adapted from SGLang (Apache-2.0, PR #7772) and TensorRT-LLM (BSD-3-Clause, PR #5027).
// SGLang's CUTLASS W4A8 grouped MoE GEMM, PyTorch deps replaced with a C ABI.
//
// Weight layout: signed INT4 two's complement, low nibble = even K, packed 2/int8.
// Scale layout: BF16 [E, K//512, N*4], interleaved per 512-K chunk (4 groups of 128).
// Activation: per-tensor scalar FP8 scale (same as SGLang).

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cutlass/cutlass.h"
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp"

using namespace cute;

namespace {

using MmaType = cutlass::float_e4m3_t;
using QuantType = cutlass::int4b_t;
using ElementAccumulator = float;
using ElementScale = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = ElementC;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;

using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

template <typename TileShape, typename ClusterShape, typename KernelSchedule, typename EpilogueSchedule>
struct cutlass_3x_w4a8_group_gemm {
  static constexpr int GroupSize = 128;
  static constexpr int PackedScalesNum = get<2>(TileShape{}) / GroupSize;
  using ElementScalePacked = cutlass::Array<ElementScale, PackedScalesNum>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC_Transpose*, AlignmentC,
      ElementD, LayoutD_Transpose*, AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
      ArchTag, OperatorClass,
      cute::tuple<QuantType, ElementScalePacked>, LayoutB_Transpose*, AlignmentB,
      MmaType, LayoutA_Transpose*, AlignmentA,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernelScaleOnly =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopScaleOnly, CollectiveEpilogue>;
  using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

  using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
  using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
  using StrideC = typename GemmKernelScaleOnly::InternalStrideC;
  using StrideD = typename GemmKernelScaleOnly::InternalStrideD;
  using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;
};

enum class Sched { PP, CO };

template <int M, int N, int K, int A, int B, int C, Sched S>
struct SM90W4A8Config {
  using KernelSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
  using EpilogueSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;
  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;
  using Cutlass3xW4A8Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <int M, int N, int K, int A, int B, int C>
using SM90_PP = SM90W4A8Config<M, N, K, A, B, C, Sched::PP>;
template <int M, int N, int K, int A, int B, int C>
using SM90_CO = SM90W4A8Config<M, N, K, A, B, C, Sched::CO>;

// Workspace layout: pointer arrays (4×E×8) + stride arrays (4×E×24) + CUTLASS workspace.
static constexpr size_t metadata_bytes(int num_experts) {
  return static_cast<size_t>(num_experts) * (4 * 8 + 4 * 24);
}

}  // namespace

// Device-side pointer + stride computation, copied from SGLang's
// int4_fp8_get_group_gemm_starts. Grid: 1 block, num_experts threads.
// Each thread writes its expert's 4 pointers and 12 stride values directly
// into the workspace, eliminating the host fill loop + cudaMemcpyAsync.
// Must be outside the anonymous namespace above — CUTLASS headers define their
// own anonymous namespace in the same TU, causing ambiguous reference errors (nvcc).
__global__ void w4a8_get_group_gemm_starts(
    const int32_t* __restrict__ expert_offsets,
    const cutlass::float_e4m3_t** a_offsets,
    const cutlass::int4b_t** b_offsets,
    cutlass::bfloat16_t** out_offsets,
    const cutlass::bfloat16_t** b_scales_offsets,
    int64_t* stride_base,
    const cutlass::float_e4m3_t* a_base,
    const cutlass::int4b_t* b_base,
    cutlass::bfloat16_t* out_base,
    const cutlass::bfloat16_t* b_scales_base,
    int64_t n, int64_t k, int num_experts) {
  int e = threadIdx.x;
  if (e >= num_experts) return;
  int32_t off = expert_offsets[e];
  a_offsets[e] = a_base + static_cast<size_t>(off) * k;
  b_offsets[e] = b_base + static_cast<size_t>(e) * k * n / 2;
  out_offsets[e] = out_base + static_cast<size_t>(off) * n;
  b_scales_offsets[e] = b_scales_base + static_cast<size_t>(e) * n * k / 128;
  // Strides: SGLang fills all three components with the leading dimension.
  int64_t* a_s = stride_base + static_cast<size_t>(e) * 3;
  int64_t* b_s = a_s + static_cast<size_t>(num_experts) * 3;
  int64_t* d_s = b_s + static_cast<size_t>(num_experts) * 3;
  int64_t* s_s = d_s + static_cast<size_t>(num_experts) * 3;
  for (int j = 0; j < 3; ++j) {
    a_s[j] = k;
    b_s[j] = k;
    d_s[j] = n;
    s_s[j] = n;
  }
}

namespace {

template <typename Gemm>
int run_grouped_gemm(
    void* d_output,
    const void* a_activations,
    const void* b_weights,
    const float* a_scale,
    const void* b_scales,
    const int32_t* expert_offsets,
    const int32_t* problem_sizes,
    int num_experts, int n, int k,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {
  using Args = typename Gemm::GemmScaleOnly::Arguments;

  const size_t meta_bytes = metadata_bytes(num_experts);
  if (workspace_bytes < meta_bytes + 1024) return -1;

  char* meta_d = static_cast<char*>(workspace);
  void* cutlass_ws = meta_d + meta_bytes;
  size_t cutlass_ws_bytes = workspace_bytes - meta_bytes;

  const int num_exp = num_experts;

  // Device-side pointer + stride computation (matches SGLang's
  // int4_fp8_get_group_gemm_starts).
  auto* d_b_ptrs = reinterpret_cast<const QuantType**>(meta_d);
  auto* d_a_ptrs = reinterpret_cast<const MmaType**>(meta_d + num_exp * 8);
  auto* d_out_ptrs = reinterpret_cast<ElementD**>(meta_d + 2 * num_exp * 8);
  auto* d_b_scales_ptrs = reinterpret_cast<const typename Gemm::ElementScalePacked**>(meta_d + 3 * num_exp * 8);
  auto* d_strides = reinterpret_cast<int64_t*>(meta_d + 4 * num_exp * 8);

  w4a8_get_group_gemm_starts<<<1, num_exp, 0, stream>>>(
      expert_offsets, d_a_ptrs, d_b_ptrs, d_out_ptrs,
      reinterpret_cast<const cutlass::bfloat16_t**>(d_b_scales_ptrs),
      d_strides,
      static_cast<const MmaType*>(a_activations),
      static_cast<const QuantType*>(b_weights),
      static_cast<ElementD*>(d_output),
      static_cast<const cutlass::bfloat16_t*>(b_scales),
      n, k, num_exp);

  auto* d_a_strides = reinterpret_cast<typename Gemm::StrideA*>(d_strides);
  auto* d_b_strides = reinterpret_cast<typename Gemm::StrideB*>(d_strides + num_exp * 3);
  auto* d_d_strides = reinterpret_cast<typename Gemm::StrideD*>(d_strides + 2 * num_exp * 3);
  auto* d_s_strides = reinterpret_cast<typename Gemm::StrideS*>(d_strides + 3 * num_exp * 3);

  cutlass::KernelHardwareInfo hw_info;
  int dev = 0;
  cudaGetDevice(&dev);
  hw_info.device_id = dev;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

  decltype(std::declval<Args>().epilogue.thread) fusion_args{};
  fusion_args.alpha = 0;
  fusion_args.beta = 0;
  fusion_args.alpha_ptr = const_cast<float*>(a_scale);
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  Args arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_exp, const_cast<ProblemShape::UnderlyingProblemShape*>(
           reinterpret_cast<const ProblemShape::UnderlyingProblemShape*>(problem_sizes)), nullptr},
      {d_b_ptrs, d_b_strides, d_a_ptrs, d_a_strides, d_b_scales_ptrs, d_s_strides, 128},
      {fusion_args, nullptr, nullptr, d_out_ptrs, d_d_strides},
      hw_info};

  typename Gemm::GemmScaleOnly gemm;
  size_t ws_needed = Gemm::GemmScaleOnly::get_workspace_size(arguments);
  if (ws_needed > cutlass_ws_bytes) return -2;

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return -3;

  status = gemm.initialize(arguments, cutlass_ws, stream);
  if (status != cutlass::Status::kSuccess) return -4;

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) return -5;

  return 0;
}

}  // namespace

// --- Per-tensor FP8 activation quantization for the W4AFP8 MoE path ---

// Multi-block amax: grid-stride reduction, block-level atomicMax to shared,
// block 0 writes the final value. Correct for prefill-scale tensors (25M+ elems).
__global__ void w4a8_per_tensor_amax_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ amax_out,
    int numel) {
  __shared__ float s_max;
  if (threadIdx.x == 0) s_max = 0.0f;
  __syncthreads();

  float local_max = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
       i += gridDim.x * blockDim.x) {
    float v = __bfloat162float(input[i]);
    local_max = fmaxf(local_max, fabsf(v));
  }
  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1)
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
  __shared__ float s_warp[32];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  if (lane == 0) s_warp[warp] = local_max;
  __syncthreads();
  if (warp == 0) {
    local_max = (lane < (blockDim.x + 31) / 32) ? s_warp[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    // atomicMax on int representation: correct for non-negative floats (amax).
    if (lane == 0)
      atomicMax(reinterpret_cast<int*>(&s_max), __float_as_int(local_max));
  }
  __syncthreads();
  // Write scale = amax / 448 (FP8 E4M3 max): the CUTLASS epilogue dequantizes
  // activations as `quantized * scale`, so scale must be the dequant factor.
  if (blockIdx.x == 0 && threadIdx.x == 0)
    *amax_out = (s_max > 0.0f) ? s_max / 448.0f : 1.0f;
}

// Quantize BF16 → FP8 E4M3 using a precomputed per-tensor scale.
__global__ void w4a8_per_tensor_quantize_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_storage_t* __restrict__ output,
    const float* __restrict__ scale,
    int numel) {
  float s = *scale;  // scale = amax / 448 (dequant factor); quant uses its reciprocal
  float inv = (s > 0.0f) ? 1.0f / s : 1.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
       i += gridDim.x * blockDim.x) {
    float v = __bfloat162float(input[i]) * inv;
    v = fmaxf(-448.0f, fminf(448.0f, v));
    output[i] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  }
}

// Build problem_sizes [E, 3] from per-expert token counts.
// SGLang convention: (N, M, K) — the CUTLASS mixed-input kernel reads the
// shape in this order; (M, N, K) swaps the GEMM dims and crashes.
__global__ void w4a8_compute_problem_sizes_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ problem_sizes,
    int num_experts, int n, int k) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < num_experts) {
    problem_sizes[e * 3 + 0] = n;
    problem_sizes[e * 3 + 1] = counts[e];
    problem_sizes[e * 3 + 2] = k;
  }
}

extern "C" void w4a8_per_tensor_fp8_quant(
    const void* input,   // BF16 [numel]
    void* output,       // FP8 E4M3 [numel]
    float* scale,       // [1] per-tensor scale (output of amax, input to quantize)
    int numel,
    void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  // Multi-block amax: 1024 threads/block, enough blocks to saturate the GPU.
  int amax_blocks = (numel + 1023) / 1024;
  if (amax_blocks > 128) amax_blocks = 128;
  w4a8_per_tensor_amax_kernel<<<amax_blocks, 1024, 0, s>>>(
      static_cast<const __nv_bfloat16*>(input), scale, numel);
  int blocks = (numel + 255) / 256;
  if (blocks > 256) blocks = 256;
  w4a8_per_tensor_quantize_kernel<<<blocks, 256, 0, s>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_fp8_storage_t*>(output), scale, numel);
}

extern "C" void w4a8_compute_problem_sizes(
    const int32_t* counts,
    int32_t* problem_sizes,
    int num_experts, int n, int k,
    void* stream) {
  w4a8_compute_problem_sizes_kernel<<<(num_experts + 31) / 32, 32, 0,
                                      static_cast<cudaStream_t>(stream)>>>(
      counts, problem_sizes, num_experts, n, k);
}

// Fused clamped SwiGLU on the CUTLASS gate+up output: [rows, 2*i_dim] → [rows, i_dim].
// Matches dsv4_swiglu_one (elementwise_basic.cu) exactly.
__global__ void w4a8_swiglu_fused_kernel(
    const __nv_bfloat16* __restrict__ gateup,
    __nv_bfloat16* __restrict__ out,
    int rows, int i_dim, float limit) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * i_dim) {
    int row = idx / i_dim;
    int col = idx % i_dim;
    int base = row * 2 * i_dim + col;
    float g = fminf(__bfloat162float(gateup[base]), limit);
    float u = fminf(fmaxf(__bfloat162float(gateup[base + i_dim]), -limit), limit);
    float silu = g / (1.0f + expf(-g));
    out[idx] = __float2bfloat16(silu * u);
  }
}

extern "C" void w4a8_swiglu_fused(
    const void* gateup,  // BF16 [rows, 2*i_dim]
    void* out,           // BF16 [rows, i_dim]
    int rows, int i_dim, float limit,
    void* stream) {
  int n = rows * i_dim;
  int blocks = (n + 255) / 256;
  if (blocks > 256) blocks = 256;
  w4a8_swiglu_fused_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(gateup),
      static_cast<__nv_bfloat16*>(out),
      rows, i_dim, limit);
}

// C ABI: one grouped GEMM call for all experts.
// d_output:       BF16 [total_m, N]
// a_activations:  FP8 E4M3 [total_m, K] (tokens sorted by expert)
// b_weights:      INT4 packed [E, N, K/2]
// a_scale:        pointer to single float (per-tensor activation scale)
// b_scales:       BF16 [E, K//512, N*4]
// expert_offsets: DEVICE int32 [E] cumulative token counts (read device-side
//                 by the w4a8_get_group_gemm_starts kernel)
// problem_sizes:  int32 [E, 3] (N, M, K per expert — SGLang order)
// total_m:        total token-expert pairs (= num_tokens * topk)
// topk:           experts per token (for tile shape heuristic)
// workspace:      pre-allocated buffer (64 MB recommended)
extern "C" int w4a8_moe_grouped_gemm_sm90(
    void* d_output,
    const void* a_activations,
    const void* b_weights,
    const float* a_scale,
    const void* b_scales,
    const int32_t* expert_offsets,
    const int32_t* problem_sizes,
    int num_experts,
    int n,
    int k,
    int total_m,
    int topk,
    void* workspace,
    size_t workspace_bytes,
    void* stream) {
  // The scale pointer formula (e * n * k / 128) requires 512-aligned K —
  // the BF16 scale layout is [E, K//512, N*4]. DSv4-Flash always satisfies
  // this (hidden 7168, intermediate 2048).
  if (k % 512 != 0) return -7;
  if (num_experts > 1024) return -8;  // one-block pointer kernel limit

  const int m = total_m / topk;  // original token count
  int rc;

  if (m <= 32) {
    rc = run_grouped_gemm<SM90_CO<128, 16, 512, 1, 1, 1>::Cutlass3xW4A8Gemm>(
        d_output, a_activations, b_weights, a_scale, b_scales,
        expert_offsets, problem_sizes, num_experts, n, k,
        workspace, workspace_bytes, static_cast<cudaStream_t>(stream));
  } else if (m <= 1024) {
    rc = run_grouped_gemm<SM90_CO<128, 32, 512, 1, 1, 1>::Cutlass3xW4A8Gemm>(
        d_output, a_activations, b_weights, a_scale, b_scales,
        expert_offsets, problem_sizes, num_experts, n, k,
        workspace, workspace_bytes, static_cast<cudaStream_t>(stream));
  } else {
    rc = run_grouped_gemm<SM90_CO<128, 64, 512, 1, 1, 1>::Cutlass3xW4A8Gemm>(
        d_output, a_activations, b_weights, a_scale, b_scales,
        expert_offsets, problem_sizes, num_experts, n, k,
        workspace, workspace_bytes, static_cast<cudaStream_t>(stream));
  }
  if (rc != 0) return rc;
  // Catch async kernel errors (illegal access, etc.) before they corrupt
  // the CUDA context and surface as unrelated cuModuleLoad failures.
  cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) return -10 - (int)err;
  return 0;
}

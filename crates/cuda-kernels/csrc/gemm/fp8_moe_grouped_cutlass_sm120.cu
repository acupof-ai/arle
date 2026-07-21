/***************************************************************************************************
 * sm_120 (Blackwell RTX PRO 6000) grouped blockwise-scaled FP8 MoE GEMM.
 *
 * The sm_120 replacement for the Hopper-only DeepGEMM
 * `m_grouped_fp8_gemm_nt_contiguous`. Per group (= expert) computes the NT GEMM
 *   D[Mg,N] (bf16) = A[Mg,K] (e4m3) @ B[N,K] (e4m3)^T
 * with DeepSeek blockwise scaling: per-token A scale (1x128) + 128x128 B block
 * scale, both f32.
 *
 * Collective/type stack copied verbatim from CUTLASS 4.3.5 example
 *   87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu
 * (the only in-tree sm_120a grouped blockwise-scaled FP8 instantiation),
 * de-risked standalone 2026-07-22 (bit-exact vs FP32, ragged + heavy M).
 *
 * Scale layout contract (source-verified against the DeepGEMM path this replaces):
 *   SFA — activation scales stay in DeepGEMM's packing:
 *         element (global_row, k_block) at `k_block*scale_stride_m + global_row`.
 *         Consumed directly by building LayoutSFA with the K-block stride =
 *         `scale_stride_m` (not Mg) and ptr_SFA[g] = sfa + group_offset[g].
 *   SFB — weight scales must arrive already N-contiguous per group
 *         (`n_block + k_block*n_blocks`), i.e. TRANSPOSED from the checkpoint's
 *         `weight_scale_inv` [n_blocks,k_blocks] row-major. The loader repacks
 *         once at load on sm_120 (weights are static). LayoutSFB is the standard
 *         `tile_atom_to_shape_SFB`.
 *
 * Group geometry (per-group M + row offset) is derived from the device
 * `group_offsets` (128-aligned per-group row start) + `group_counts` (real Mg),
 * D2H'd once per call (prefill only, not graph-captured).
 **************************************************************************************************/

// Build-gated: build.rs defines ARLE_SM120_GROUPED_FP8 (and forces sm_120a
// gencode) ONLY when an sm_120 target is present. Other builds compile the
// cheap `#else` stub — the heavy CUTLASS collective is never instantiated, so
// Hopper/Ampere builds stay clean and fast (mirrors the FA3/FlashMLA SM gate).
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef ARLE_SM120_GROUPED_FP8

#include <vector>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

using namespace cute;

namespace {

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

// ---- Type stack: verbatim from example 87c (the de-risked collective) ----
using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute = float;
using ElementSF = ElementAccumulator;

using MmaTileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

// DeepSeek scheme: per-token A scale (1x128), 128x128 B block scale.
constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;
using ScaleConfig =
    cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;

using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC*, AlignmentC,
    ElementD, LayoutD*, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA,
    ElementB, cute::tuple<LayoutB*, LayoutSFB*>, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

// Mirror cutlass::make_cute_packed_stride — the vendored cutlass tree is
// include-only (no tools/util/packed_stride.hpp). Rank-3 (d0, d1, L); L=1 grouped
// ⇒ batch stride 0. Two overloads keyed on the unit-stride (contiguous) mode:
//   RowMajor [d0,d1]     -> Stride<int64_t, _1, int64_t>, leading = d1
//   ColumnMajor [d0,d1]  -> Stride<_1, int64_t, int64_t>, second  = d0
template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<IntT, cute::Int<1>, IntT> arle_packed_stride(
    cute::Stride<IntT, cute::Int<1>, IntT> s, int d0, int d1, int l) {
  cute::get<0>(s) = static_cast<IntT>(d1);
  cute::get<2>(s) = (l > 1) ? static_cast<IntT>(static_cast<int64_t>(d0) * d1) : IntT(0);
  return s;
}
template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<cute::Int<1>, IntT, IntT> arle_packed_stride(
    cute::Stride<cute::Int<1>, IntT, IntT> s, int d0, int d1, int l) {
  cute::get<1>(s) = static_cast<IntT>(d0);
  cute::get<2>(s) = (l > 1) ? static_cast<IntT>(static_cast<int64_t>(d0) * d1) : IntT(0);
  return s;
}

// Persistent device scratch — allocated once, grown on demand. Reused across the
// per-layer / per-step GEMM calls (no per-call cudaMalloc in the hot loop).
struct GroupedScratch {
  int capacity = 0;
  size_t workspace_bytes = 0;
  ProblemShape::UnderlyingProblemShape* problem_sizes = nullptr;
  const ElementA** ptr_A = nullptr;
  const ElementB** ptr_B = nullptr;
  const ElementSF** ptr_SFA = nullptr;
  const ElementSF** ptr_SFB = nullptr;
  const ElementC** ptr_C = nullptr;
  ElementD** ptr_D = nullptr;
  StrideA* stride_A = nullptr;
  StrideB* stride_B = nullptr;
  StrideC* stride_C = nullptr;
  StrideD* stride_D = nullptr;
  LayoutSFA* layout_SFA = nullptr;
  LayoutSFB* layout_SFB = nullptr;
  uint8_t* workspace = nullptr;
  int sm_count = 0;

  cudaError_t ensure(int groups, size_t ws) {
    cudaError_t rc = cudaSuccess;
    if (groups > capacity) {
      free_arrays();
#define ARLE_ALLOC(field, T, n)                                       \
  if ((rc = cudaMalloc(reinterpret_cast<void**>(&field), (n) * sizeof(T))) != cudaSuccess) return rc
      ARLE_ALLOC(problem_sizes, ProblemShape::UnderlyingProblemShape, groups);
      ARLE_ALLOC(ptr_A, const ElementA*, groups);
      ARLE_ALLOC(ptr_B, const ElementB*, groups);
      ARLE_ALLOC(ptr_SFA, const ElementSF*, groups);
      ARLE_ALLOC(ptr_SFB, const ElementSF*, groups);
      ARLE_ALLOC(ptr_C, const ElementC*, groups);
      ARLE_ALLOC(ptr_D, ElementD*, groups);
      ARLE_ALLOC(stride_A, StrideA, groups);
      ARLE_ALLOC(stride_B, StrideB, groups);
      ARLE_ALLOC(stride_C, StrideC, groups);
      ARLE_ALLOC(stride_D, StrideD, groups);
      ARLE_ALLOC(layout_SFA, LayoutSFA, groups);
      ARLE_ALLOC(layout_SFB, LayoutSFB, groups);
#undef ARLE_ALLOC
      capacity = groups;
    }
    if (ws > workspace_bytes) {
      if (workspace) cudaFree(workspace);
      workspace = nullptr;
      if ((rc = cudaMalloc(reinterpret_cast<void**>(&workspace), ws)) != cudaSuccess) return rc;
      workspace_bytes = ws;
    }
    if (sm_count == 0) {
      sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    }
    return cudaSuccess;
  }

  void free_arrays() {
    void* fields[] = {problem_sizes, ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_C,
                      ptr_D, stride_A, stride_B, stride_C, stride_D,
                      layout_SFA, layout_SFB};
    for (void* p : fields)
      if (p) cudaFree(p);
    problem_sizes = nullptr; ptr_A = nullptr; ptr_B = nullptr; ptr_SFA = nullptr;
    ptr_SFB = nullptr; ptr_C = nullptr; ptr_D = nullptr; stride_A = nullptr;
    stride_B = nullptr; stride_C = nullptr; stride_D = nullptr;
    layout_SFA = nullptr; layout_SFB = nullptr; capacity = 0;
  }
};

// One scratch per device (single-GPU serve → index 0; guarded to a small fixed set).
GroupedScratch& scratch_for_current_device() {
  static GroupedScratch scratches[16];
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= 16) dev = 0;
  return scratches[dev];
}

}  // namespace

extern "C" CUresult arle_fp8_moe_grouped_gemm_nt_sm120(
    const uint8_t* a,          // e4m3 activations, contiguous [total_M, K]
    const float* sfa,          // per-token f32 act scales (DeepGEMM packing)
    const uint8_t* b,          // e4m3 grouped weights [G, N, K] row-major
    const float* sfb,          // f32 128-block weight scales, N-contiguous per group
    void* d,                   // bf16 out [total_M, N]
    const int* group_offsets,  // device [G]: 128-aligned per-group row start
    const int* group_counts,   // device [G]: real Mg per group
    int num_groups, int n, int k,
    int scale_stride_m,        // SFA K-block leading dim (DeepGEMM's scale_stride_m)
    cudaStream_t stream) {
  if (a == nullptr || sfa == nullptr || b == nullptr || sfb == nullptr || d == nullptr ||
      group_offsets == nullptr || group_counts == nullptr || num_groups <= 0 || n <= 0 ||
      k <= 0 || (n % 128) != 0 || (k % 128) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const int G = num_groups;
  const int n_blocks = n / 128;
  const int k_blocks = k / 128;
  auto* d_bf16 = reinterpret_cast<ElementD*>(d);

  // D2H the small group geometry (prefill path — a single sync is acceptable and
  // mirrors the DeepGEMM path's own host scans).
  std::vector<int> off(G), cnt(G);
  cudaError_t rc;
  if ((rc = cudaMemcpyAsync(off.data(), group_offsets, G * sizeof(int),
                            cudaMemcpyDeviceToHost, stream)) != cudaSuccess)
    return CUDA_ERROR_UNKNOWN;
  if ((rc = cudaMemcpyAsync(cnt.data(), group_counts, G * sizeof(int),
                            cudaMemcpyDeviceToHost, stream)) != cudaSuccess)
    return CUDA_ERROR_UNKNOWN;
  if ((rc = cudaStreamSynchronize(stream)) != cudaSuccess) return CUDA_ERROR_UNKNOWN;

  std::vector<ProblemShape::UnderlyingProblemShape> ps_host(G);
  std::vector<const ElementA*> pA(G);
  std::vector<const ElementB*> pB(G);
  std::vector<const ElementSF*> pSFA(G);
  std::vector<const ElementSF*> pSFB(G);
  std::vector<const ElementC*> pC(G);
  std::vector<ElementD*> pD(G);
  std::vector<StrideA> sA(G);
  std::vector<StrideB> sB(G);
  std::vector<StrideC> sC(G);
  std::vector<StrideD> sD(G);
  std::vector<LayoutSFA> lSFA(G);
  std::vector<LayoutSFB> lSFB(G);

  const auto* a_e4m3 = reinterpret_cast<const ElementA*>(a);
  const auto* b_e4m3 = reinterpret_cast<const ElementB*>(b);

  int max_m = 0;
  for (int g = 0; g < G; ++g) {
    int Mg = cnt[g];
    if (Mg < 0) Mg = 0;
    int row = off[g];
    if (Mg > max_m) max_m = Mg;
    ps_host[g] = {Mg, n, k};

    pA[g] = a_e4m3 + static_cast<int64_t>(row) * k;
    pB[g] = b_e4m3 + static_cast<int64_t>(g) * n * k;
    pC[g] = d_bf16 + static_cast<int64_t>(row) * n;  // beta=0; C unused, point at D.
    pD[g] = d_bf16 + static_cast<int64_t>(row) * n;
    // SFA stays in DeepGEMM packing: ptr offset by the group's global row start.
    pSFA[g] = sfa + row;
    pSFB[g] = sfb + static_cast<int64_t>(g) * n_blocks * k_blocks;

    sA[g] = arle_packed_stride(StrideA{}, Mg, k, 1);
    sB[g] = arle_packed_stride(StrideB{}, n, k, 1);
    sC[g] = arle_packed_stride(StrideC{}, Mg, n, 1);
    sD[g] = arle_packed_stride(StrideD{}, Mg, n, 1);

    // LayoutSFA — DeepGEMM packing: K-block stride = scale_stride_m (NOT Mg),
    // per-token (1) stride = 1. Matches `k_block*scale_stride_m + m`.
    lSFA[g] = make_layout(
        make_shape(make_shape(_1{}, Mg), make_shape(_128{}, k_blocks), 1),
        make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, scale_stride_m),
                    scale_stride_m * k_blocks));
    // LayoutSFB — standard (weights pre-transposed to N-contiguous at load).
    lSFB[g] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(Mg, n, k, 1));
  }

  if (max_m == 0) return CUDA_SUCCESS;  // nothing routed to any local expert.

  GroupedScratch& s = scratch_for_current_device();

  typename Gemm::Arguments arguments;
  {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    cudaGetDevice(&hw_info.device_id);
    hw_info.sm_count =
        (s.sm_count != 0) ? s.sm_count
                          : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
                                hw_info.device_id);

    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
    scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;

    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 0};
    fusion_args.dBeta = {_0{}, _0{}, 0};

    // Fill AFTER s.ensure so the device pointers exist; placeholders here get
    // overwritten below once the arrays are uploaded.
    arguments = {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {G, nullptr, ps_host.data()},
        {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
        {fusion_args, nullptr, nullptr, nullptr, nullptr},
        hw_info,
        scheduler};
  }

  size_t ws = Gemm::get_workspace_size(arguments);
  if ((rc = s.ensure(G, ws)) != cudaSuccess) return CUDA_ERROR_OUT_OF_MEMORY;

  // Upload the per-group arrays to the persistent scratch.
#define ARLE_UP(dst, src, T)                                                     \
  if (cudaMemcpyAsync(s.dst, src.data(), G * sizeof(T), cudaMemcpyHostToDevice,  \
                      stream) != cudaSuccess)                                    \
  return CUDA_ERROR_UNKNOWN
  ARLE_UP(problem_sizes, ps_host, ProblemShape::UnderlyingProblemShape);
  ARLE_UP(ptr_A, pA, const ElementA*);
  ARLE_UP(ptr_B, pB, const ElementB*);
  ARLE_UP(ptr_SFA, pSFA, const ElementSF*);
  ARLE_UP(ptr_SFB, pSFB, const ElementSF*);
  ARLE_UP(ptr_C, pC, const ElementC*);
  ARLE_UP(ptr_D, pD, ElementD*);
  ARLE_UP(stride_A, sA, StrideA);
  ARLE_UP(stride_B, sB, StrideB);
  ARLE_UP(stride_C, sC, StrideC);
  ARLE_UP(stride_D, sD, StrideD);
  ARLE_UP(layout_SFA, lSFA, LayoutSFA);
  ARLE_UP(layout_SFB, lSFB, LayoutSFB);
#undef ARLE_UP

  arguments.problem_shape = {G, s.problem_sizes, ps_host.data()};
  arguments.mainloop = {s.ptr_A, s.stride_A, s.ptr_B, s.stride_B,
                        s.ptr_SFA, s.layout_SFA, s.ptr_SFB, s.layout_SFB};
  arguments.epilogue.ptr_C = s.ptr_C;
  arguments.epilogue.dC = s.stride_C;
  arguments.epilogue.ptr_D = s.ptr_D;
  arguments.epilogue.dD = s.stride_D;

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess)
    return CUDA_ERROR_NOT_SUPPORTED;
  if (gemm.initialize(arguments, s.workspace, stream) != cutlass::Status::kSuccess)
    return CUDA_ERROR_UNKNOWN;
  if (gemm.run(stream) != cutlass::Status::kSuccess) return CUDA_ERROR_LAUNCH_FAILED;
  return CUDA_SUCCESS;
}

#else  // !ARLE_SM120_GROUPED_FP8

extern "C" CUresult arle_fp8_moe_grouped_gemm_nt_sm120(
    const uint8_t*, const float*, const uint8_t*, const float*, void*,
    const int*, const int*, int, int, int, int, cudaStream_t) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

#endif  // ARLE_SM120_GROUPED_FP8

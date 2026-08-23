/*
 * Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Raw-C ABI wrappers for the official DSv4 DSA indexer building blocks from
 * SGLang / sgl-kernel:
 *   - dsv4_fused_q_indexer_rope_hadamard_quant
 *   - fused_store_index_k_cache
 *   - deepseek_v4_topk_transform_512  (exported as dsv4_deepseek_v4_topk_transform_cuda)
 *
 * The kernel bodies below are direct ports of those OSS kernels with Torch/TVM
 * wrappers replaced by ARLE's raw CUDA ABI.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

using bf16_t = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;
using fp8x2_e4m3_t = __nv_fp8x2_e4m3;

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kFullMask = 0xffffffffu;
constexpr float kFP8Max = 448.0f;

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVec {
  T data[N];
  __device__ __forceinline__ T& operator[](int i) { return data[i]; }
  __device__ __forceinline__ T operator[](int i) const { return data[i]; }
  __device__ __forceinline__ void load(const void* ptr, int64_t offset = 0) {
    *this = reinterpret_cast<const AlignedVec*>(ptr)[offset];
  }
  __device__ __forceinline__ void store(void* ptr, int64_t offset = 0) const {
    reinterpret_cast<AlignedVec*>(ptr)[offset] = *this;
  }
};

__device__ __forceinline__ float bf16_to_float(bf16_t v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ bf16_t float_to_bf16(float v) {
  return __float2bfloat16_rn(v);
}

__device__ __forceinline__ float warp_reduce_max_xor(float val) {
#pragma unroll
  for (uint32_t mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(kFullMask, val, mask, kWarpSize));
  }
  return val;
}

__device__ __forceinline__ float fp8_e4m3_clip(float val) {
  return fmaxf(fminf(val, kFP8Max), -kFP8Max);
}

__device__ __forceinline__ fp8x2_e4m3_t pack_fp8(float x, float y) {
  return __nv_fp8x2_e4m3(make_float2(fp8_e4m3_clip(x), fp8_e4m3_clip(y)));
}

// SGLang dsv4_fused_q_indexer_rope_hadamard_quant

constexpr uint32_t kFusedQBlockSize = 128;
constexpr uint32_t kFusedQNumWarps = kFusedQBlockSize / kWarpSize;

struct FusedQIndexerRopeHadamardQuantParams {
  const void* __restrict__ q_input;
  void* __restrict__ q_fp8;
  const void* __restrict__ weight;
  float* __restrict__ weights_out;
  float weight_scale;
  const float* __restrict__ freqs_cis;
  const int32_t* __restrict__ positions;
  uint32_t batch_size;
  uint32_t num_heads;
};

__global__ __launch_bounds__(kFusedQBlockSize, 16) void fused_q_indexer_rope_hadamard_quant_kernel(
    const __grid_constant__ FusedQIndexerRopeHadamardQuantParams params) {
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim == kWarpSize * kVecSize);

  using Storage = AlignedVec<bf16_t, kVecSize>;
  using Float4 = AlignedVec<float, kVecSize>;
  using OutStorage = AlignedVec<fp8x2_e4m3_t, 2>;

  const auto warp_id = threadIdx.x / kWarpSize;
  const auto lane_id = threadIdx.x % kWarpSize;
  const auto work_id = blockIdx.x * kFusedQNumWarps + warp_id;
  const bool is_rope_lane = lane_id >= kWarpSize - kRopeSize;

  const uint32_t total_works = params.batch_size * params.num_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_heads;
  const auto input_ptr = static_cast<const bf16_t*>(params.q_input) + work_id * kHeadDim;
  const auto position = params.positions[batch_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  Float4 data, freq;
  const float weight_val = bf16_to_float(static_cast<const bf16_t*>(params.weight)[work_id]);

  Storage input_vec;
  input_vec.load(input_ptr, lane_id);
  if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpSize - kRopeSize));
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    data[i] = bf16_to_float(input_vec[i]);
  }

  if (is_rope_lane) {
    float x_r = data[0], x_i = data[1], y_r = data[2], y_i = data[3];
    float fxr = freq[0], fxi = freq[1], fyr = freq[2], fyi = freq[3];
    data[0] = x_r * fxr - x_i * fxi;
    data[1] = x_r * fxi + x_i * fxr;
    data[2] = y_r * fyr - y_i * fyi;
    data[3] = y_r * fyi + y_i * fyr;
  }

  {
    float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a1;
    data[1] = a0 - a1;
    data[2] = a2 + a3;
    data[3] = a2 - a3;
  }
  {
    float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a2;
    data[1] = a1 + a3;
    data[2] = a0 - a2;
    data[3] = a1 - a3;
  }
#pragma unroll
  for (uint32_t mask = 1; mask < kWarpSize; mask <<= 1) {
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      float other = __shfl_xor_sync(kFullMask, data[i], mask, kWarpSize);
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }
  const float kHadamardScale = rsqrtf(static_cast<float>(kHeadDim));
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    data[i] *= kHadamardScale;
  }

  float local_max = fabsf(data[0]);
#pragma unroll
  for (int i = 1; i < kVecSize; ++i) {
    local_max = fmaxf(local_max, fabsf(data[i]));
  }
  float abs_max = warp_reduce_max_xor(local_max);
  float scale = fmaxf(1e-4f, abs_max) / kFP8Max;
  float inv_scale = 1.0f / scale;

  OutStorage result;
  result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
  result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);

  auto out_row = static_cast<uint8_t*>(params.q_fp8) + work_id * kHeadDim;
  result.store(out_row, lane_id);
  params.weights_out[work_id] = weight_val * params.weight_scale * scale;
}

// SGLang rotate_activation for 128-dim BF16 rows.

__global__ __launch_bounds__(kFusedQBlockSize, 16) void hadamard128_bf16_kernel(
    const bf16_t* __restrict__ input,
    bf16_t* __restrict__ output,
    int rows) {
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kVecSize = 4;
  static_assert(kHeadDim == kWarpSize * kVecSize);

  using Storage = AlignedVec<bf16_t, kVecSize>;
  using Float4 = AlignedVec<float, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpSize;
  const auto lane_id = threadIdx.x % kWarpSize;
  const auto row = blockIdx.x * kFusedQNumWarps + warp_id;
  if (row >= rows) return;

  const auto in_ptr = input + static_cast<int64_t>(row) * kHeadDim;
  auto out_ptr = output + static_cast<int64_t>(row) * kHeadDim;

  Storage input_vec;
  input_vec.load(in_ptr, lane_id);
  Float4 data;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    data[i] = bf16_to_float(input_vec[i]);
  }

  {
    float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a1;
    data[1] = a0 - a1;
    data[2] = a2 + a3;
    data[3] = a2 - a3;
  }
  {
    float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a2;
    data[1] = a1 + a3;
    data[2] = a0 - a2;
    data[3] = a1 - a3;
  }
#pragma unroll
  for (uint32_t mask = 1; mask < kWarpSize; mask <<= 1) {
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      float other = __shfl_xor_sync(kFullMask, data[i], mask, kWarpSize);
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }

  Storage out_vec;
  const float scale = rsqrtf(static_cast<float>(kHeadDim));
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    out_vec[i] = float_to_bf16(data[i] * scale);
  }
  out_vec.store(out_ptr, lane_id);
}

// Batched (grid.y=slot) Hadamard rotate over n slots' newly-packed index-key
// rows. blockIdx.y selects the slot; blockIdx.x walks that slot's rows (4 warps
// per block, one 128-dim row per warp). Per-slot base pointers / offsets / row
// counts come from the host-gathered arrays; `src` is WINDOW-relative
// (`src_ring_row + r`) into the staging ring, `dst` is ABSOLUTE (`dst_row + r`)
// into rotated_keys. With n=1, slot=0, this is byte-identical to the per-row
// `hadamard128_bf16_kernel` (same warp body, same offsets). The per-warp body is
// copied verbatim.
__global__ __launch_bounds__(kFusedQBlockSize, 16) void hadamard128_bf16_batched_kernel(
    const bf16_t* const* __restrict__ keys_src_arr,
    const int32_t* __restrict__ src_ring_row_arr,
    bf16_t* const* __restrict__ rotated_dst_arr,
    const int32_t* __restrict__ dst_row_arr,
    const int32_t* __restrict__ newly_packed_arr,
    int n) {
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kVecSize = 4;
  static_assert(kHeadDim == kWarpSize * kVecSize);

  using Storage = AlignedVec<bf16_t, kVecSize>;
  using Float4 = AlignedVec<float, kVecSize>;

  const int slot = blockIdx.y;
  if (slot >= n) return;
  const auto warp_id = threadIdx.x / kWarpSize;
  const auto lane_id = threadIdx.x % kWarpSize;
  const int r = static_cast<int>(blockIdx.x * kFusedQNumWarps + warp_id);
  if (r >= newly_packed_arr[slot]) return;

  const auto in_ptr =
      keys_src_arr[slot] + static_cast<int64_t>(src_ring_row_arr[slot] + r) * kHeadDim;
  auto out_ptr =
      rotated_dst_arr[slot] + static_cast<int64_t>(dst_row_arr[slot] + r) * kHeadDim;

  Storage input_vec;
  input_vec.load(in_ptr, lane_id);
  Float4 data;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    data[i] = bf16_to_float(input_vec[i]);
  }

  {
    float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a1;
    data[1] = a0 - a1;
    data[2] = a2 + a3;
    data[3] = a2 - a3;
  }
  {
    float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a2;
    data[1] = a1 + a3;
    data[2] = a0 - a2;
    data[3] = a1 - a3;
  }
#pragma unroll
  for (uint32_t mask = 1; mask < kWarpSize; mask <<= 1) {
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      float other = __shfl_xor_sync(kFullMask, data[i], mask, kWarpSize);
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }

  Storage out_vec;
  const float scale = rsqrtf(static_cast<float>(kHeadDim));
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    out_vec[i] = float_to_bf16(data[i] * scale);
  }
  out_vec.store(out_ptr, lane_id);
}

// Decode-graph variant: ONE launch per step that packs AT MOST one compressed
// index-key row into the CSA key cache, gated on device by `start_pos`:
// the row `start_pos / ratio` completes when `(start_pos + 1) % ratio == 0`.
// Rotates keys[row] (absolute, full-retention ring base 0) into `rotated[0]`
// then quantizes into the cache band at `cache_locs[row]`, mirroring the two
// per-row kernels byte-for-byte. Captures into a CUDA graph (no host branch).
__global__ __launch_bounds__(kFusedQBlockSize, 16) void dsv4_dsa_pack_index_row_start_pos_kernel(
    const bf16_t* __restrict__ keys,
    bf16_t* __restrict__ rotated,
    uint8_t* __restrict__ cache,
    const int64_t* __restrict__ cache_locs,
    const int32_t* __restrict__ start_pos_ptr,
    int ratio) {
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kPageBits = 6;
  constexpr int64_t kPageBytes = 132ll << kPageBits;
  using Storage = AlignedVec<bf16_t, kVecSize>;
  using Float4 = AlignedVec<float, kVecSize>;
  using KeyT2 = bf16x2_t;
  using InStorage = AlignedVec<KeyT2, 2>;
  using OutStorage = AlignedVec<fp8x2_e4m3_t, 2>;

  const int start_pos = start_pos_ptr[0];
  if ((start_pos + 1) % ratio != 0) return;
  const int64_t row = start_pos / ratio;
  const auto lane_id = threadIdx.x % kWarpSize;
  if (threadIdx.x >= kWarpSize) return;

  // Hadamard rotate keys[row] -> rotated[0] (per-row kernel body).
  Storage input_vec;
  input_vec.load(keys + row * kHeadDim, lane_id);
  Float4 data;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) data[i] = bf16_to_float(input_vec[i]);
  {
    float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a1; data[1] = a0 - a1; data[2] = a2 + a3; data[3] = a2 - a3;
  }
  {
    float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a2; data[1] = a1 + a3; data[2] = a0 - a2; data[3] = a1 - a3;
  }
#pragma unroll
  for (uint32_t mask = 1; mask < kWarpSize; mask <<= 1) {
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      float other = __shfl_xor_sync(kFullMask, data[i], mask, kWarpSize);
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }
  Storage out_vec;
  const float scale_h = rsqrtf(static_cast<float>(kHeadDim));
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) out_vec[i] = float_to_bf16(data[i] * scale_h);
  out_vec.store(rotated, lane_id);
  __syncwarp();

  // FP8 store rotated[0] -> cache[cache_locs[row]] (per-row kernel body).
  const auto index = cache_locs[row];
  const auto elems = reinterpret_cast<const InStorage*>(rotated)[lane_id];
  float2 x = __bfloat1622float2(elems[0]);
  float2 y = __bfloat1622float2(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(y.x), fabsf(y.y)));
  const auto abs_max = warp_reduce_max_xor(local_max);
  const auto scale = fmaxf(1e-4f, abs_max) / kFP8Max;
  const auto inv_scale = 1.0f / scale;
  const int32_t page = static_cast<int32_t>(index >> kPageBits);
  const int32_t offset = static_cast<int32_t>(index & ((1 << kPageBits) - 1));
  auto page_ptr = cache + page * kPageBytes;
  auto value_ptr = page_ptr + offset * 128;
  auto scale_ptr = page_ptr + (128ll << kPageBits) + offset * 4;
  OutStorage result;
  result[0] = pack_fp8(x.x * inv_scale, x.y * inv_scale);
  result[1] = pack_fp8(y.x * inv_scale, y.y * inv_scale);
  reinterpret_cast<OutStorage*>(value_ptr)[lane_id] = result;
  reinterpret_cast<float*>(scale_ptr)[0] = scale;
}

extern "C" CUresult dsv4_dsa_pack_index_row_start_pos_cuda(
    const uint16_t* keys,
    uint16_t* rotated,
    uint8_t* cache,
    const int64_t* cache_locs,
    const int32_t* start_pos,
    int ratio,
    int page_size,
    CUstream stream) {
  if (keys == nullptr || rotated == nullptr || cache == nullptr || cache_locs == nullptr ||
      start_pos == nullptr || stream == nullptr || ratio <= 0 || page_size != 64) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dsv4_dsa_pack_index_row_start_pos_kernel<<<1, kFusedQBlockSize, 0,
                                             reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16_t*>(keys), reinterpret_cast<bf16_t*>(rotated), cache,
      cache_locs, start_pos, ratio);
  return static_cast<CUresult>(cudaGetLastError());
}

// SGLang fused_store_index_k_cache, raw C ABI.

template <typename IndicesT, uint32_t kPageBits>
__global__ void fused_store_indexer_cache_kernel(
    const bf16_t* __restrict__ input,
    uint8_t* __restrict__ cache,
    const IndicesT* __restrict__ indices,
    uint32_t num_tokens) {
  constexpr int64_t kPageBytes = 132ll << kPageBits;

  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / kWarpSize;
  const auto lane_id = threadIdx.x % kWarpSize;
  if (global_wid >= num_tokens) return;

  using KeyT2 = bf16x2_t;
  using InStorage = AlignedVec<KeyT2, 2>;
  using OutStorage = AlignedVec<fp8x2_e4m3_t, 2>;

  const auto index = static_cast<int64_t>(indices[global_wid]);
  const auto elems = reinterpret_cast<const InStorage*>(input)[global_tid];
  float2 x = __bfloat1622float2(elems[0]);
  float2 y = __bfloat1622float2(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(y.x), fabsf(y.y)));
  const auto abs_max = warp_reduce_max_xor(local_max);
  const auto scale = fmaxf(1e-4f, abs_max) / kFP8Max;
  const auto inv_scale = 1.0f / scale;

  const int32_t page = static_cast<int32_t>(index >> kPageBits);
  const int32_t offset = static_cast<int32_t>(index & ((1 << kPageBits) - 1));
  auto page_ptr = cache + page * kPageBytes;
  auto value_ptr = page_ptr + offset * 128;
  auto scale_ptr = page_ptr + (128ll << kPageBits) + offset * 4;

  OutStorage result;
  result[0] = pack_fp8(x.x * inv_scale, x.y * inv_scale);
  result[1] = pack_fp8(y.x * inv_scale, y.y * inv_scale);
  reinterpret_cast<OutStorage*>(value_ptr)[lane_id] = result;
  reinterpret_cast<float*>(scale_ptr)[0] = scale;
}

// Batched (grid.y=slot) FP8 fused-store over n slots' newly-packed rows. Each
// warp (32 lanes, one 128-dim key row) stores one token of slot `blockIdx.y`
// into that slot's cache band. The x-grid is PER-SLOT: `slot_local_wid` indexes
// this slot's token (the per-row kernel's `global_wid`), bounded by
// `newly_packed_arr[slot]`. The input read uses `key_arr[slot]` at the
// slot-local thread id, and the cache loc / band base are slot-local
// (`out_cache_loc_arr[slot][tok]` into `cache_arr[slot]`), so the page/offset
// math is unchanged. With n=1, slot=0, this is byte-identical to the per-row
// `fused_store_indexer_cache_kernel`. The per-warp body is copied verbatim.
template <typename IndicesT, uint32_t kPageBits>
__global__ void fused_store_indexer_cache_batched_kernel(
    const bf16_t* const* __restrict__ key_arr,
    uint8_t* const* __restrict__ cache_arr,
    const IndicesT* const* __restrict__ out_cache_loc_arr,
    const int32_t* __restrict__ newly_packed_arr,
    int n) {
  constexpr int64_t kPageBytes = 132ll << kPageBits;

  const int slot = blockIdx.y;
  if (slot >= n) return;
  const auto slot_local_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto slot_local_wid = slot_local_tid / kWarpSize;
  const auto lane_id = threadIdx.x % kWarpSize;
  if (slot_local_wid >= static_cast<uint32_t>(newly_packed_arr[slot])) return;

  using KeyT2 = bf16x2_t;
  using InStorage = AlignedVec<KeyT2, 2>;
  using OutStorage = AlignedVec<fp8x2_e4m3_t, 2>;

  const bf16_t* input = key_arr[slot];
  uint8_t* cache = cache_arr[slot];
  const auto index = static_cast<int64_t>(out_cache_loc_arr[slot][slot_local_wid]);
  const auto elems = reinterpret_cast<const InStorage*>(input)[slot_local_tid];
  float2 x = __bfloat1622float2(elems[0]);
  float2 y = __bfloat1622float2(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(y.x), fabsf(y.y)));
  const auto abs_max = warp_reduce_max_xor(local_max);
  const auto scale = fmaxf(1e-4f, abs_max) / kFP8Max;
  const auto inv_scale = 1.0f / scale;

  const int32_t page = static_cast<int32_t>(index >> kPageBits);
  const int32_t offset = static_cast<int32_t>(index & ((1 << kPageBits) - 1));
  auto page_ptr = cache + page * kPageBytes;
  auto value_ptr = page_ptr + offset * 128;
  auto scale_ptr = page_ptr + (128ll << kPageBits) + offset * 4;

  OutStorage result;
  result[0] = pack_fp8(x.x * inv_scale, x.y * inv_scale);
  result[1] = pack_fp8(y.x * inv_scale, y.y * inv_scale);
  reinterpret_cast<OutStorage*>(value_ptr)[lane_id] = result;
  reinterpret_cast<float*>(scale_ptr)[0] = scale;
}

// SGLang deepseek_v4_topk_transform_512 port, raw C ABI. Our symbol drops the
// `_512` suffix: `topk` is a runtime arg (`params.topk`, ≤ kMaxTopK=1024), NOT
// fixed at 512 — upstream only baked the canonical CSA k into the name.

constexpr uint32_t kMaxTopK = 1024;
constexpr uint32_t kTopKBlockSize = 512;
constexpr size_t kTopKSmem = 48 * 1024;
static_assert(kTopKSmem % (2 * sizeof(int32_t)) == 0, "topk smem must be aligned");

struct TopKParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t page_bits;
  uint32_t topk;
  int64_t output_stride;
};

__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ int32_t page_to_slot(
    const int32_t* __restrict__ page_table,
    uint32_t i,
    uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | static_cast<int32_t>(i & mask);
}

__device__ void naive_paged_transform(
    int32_t length,
    uint32_t topk,
    uint32_t page_bits,
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ page_indices_out,
    int32_t* __restrict__ raw_indices_out) {
  for (uint32_t i = threadIdx.x; i < topk; i += kTopKBlockSize) {
    if (i < static_cast<uint32_t>(length)) {
      page_indices_out[i] = page_to_slot(page_table, i, page_bits);
      if (raw_indices_out != nullptr) raw_indices_out[i] = static_cast<int32_t>(i);
    } else {
      page_indices_out[i] = -1;
      if (raw_indices_out != nullptr) raw_indices_out[i] = -1;
    }
  }
}

__device__ void radix_topk(
    const float* __restrict__ input,
    int32_t* __restrict__ output,
    uint32_t length,
    uint32_t topk) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t BLOCK_SIZE = kTopKBlockSize;
  constexpr uint32_t SMEM_INPUT_SIZE = kTopKSmem / (2 * sizeof(int32_t));

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ int32_t s_last_remain;

  extern __shared__ uint32_t s_input_idx[][SMEM_INPUT_SIZE];

  const uint32_t tx = threadIdx.x;
  uint32_t remain_topk = topk;
  auto& s_histogram = _s_histogram_buf[0];

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int32_t i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = _s_histogram_buf[k][tx];
        if (tx + j < RADIX) value += _s_histogram_buf[k][tx + j];
        _s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();
  for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx]);
    atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();
  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  {
    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin + 1];
    if (remain_topk == 0) {
      for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
        const uint32_t bin = convert_to_uint8(input[idx]);
        if (bin > threshold_bin) {
          const auto pos = atomicAdd(&s_counter, 1);
          output[pos] = static_cast<int32_t>(idx);
        }
      }
      __syncthreads();
      return;
    }
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
      const float raw_input = input[idx];
      const uint32_t bin = convert_to_uint8(raw_input);
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        output[pos] = static_cast<int32_t>(idx);
      } else if (bin == threshold_bin) {
        const auto pos = atomicAdd(&s_num_input[0], 1);
        if (pos < SMEM_INPUT_SIZE) {
          s_input_idx[0][pos] = idx;
          const auto bin32 = convert_to_uint32(raw_input);
          const auto sub_bin = (bin32 >> 24) & 0xFF;
          atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const auto r_idx = round % 2;
    const auto raw_num_input = s_num_input[r_idx];
    const auto num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = static_cast<int32_t>(remain_topk - s_histogram[tx + 1]);
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin + 1];

    if (remain_topk == 0) {
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = atomicAdd(&s_counter, 1);
          output[pos] = static_cast<int32_t>(idx);
        }
      }
      __syncthreads();
      break;
    }
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();
    for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
      const auto idx = s_input_idx[r_idx][i];
      const auto raw_input = input[idx];
      const auto offset = 24 - round * 8;
      const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        output[pos] = static_cast<int32_t>(idx);
      } else if (bin == threshold_bin) {
        if (round == 3) {
          const auto pos = atomicAdd(&s_last_remain, -1);
          if (pos > 0) output[topk - pos] = static_cast<int32_t>(idx);
        } else {
          const auto pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
          if (pos < SMEM_INPUT_SIZE) {
            s_input_idx[r_idx ^ 1][pos] = idx;
            const auto bin32 = convert_to_uint32(raw_input);
            const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
            atomicAdd(&s_histogram[sub_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kTopKBlockSize) void deepseek_v4_topk_transform_kernel(
    const TopKParams params) {
  const auto bid = blockIdx.x;
  const auto seq_len = params.seq_lens[bid];
  const auto topk = params.topk;
  const auto score_ptr = params.scores + bid * params.score_stride;
  const auto page_ptr = params.page_table + bid * params.page_table_stride;
  const auto indices_ptr = params.page_indices + bid * params.output_stride;
  const auto raw_indices_ptr =
      params.raw_indices != nullptr ? params.raw_indices + bid * params.output_stride : nullptr;

  if (seq_len <= static_cast<int32_t>(topk)) {
    naive_paged_transform(seq_len, topk, params.page_bits, page_ptr, indices_ptr, raw_indices_ptr);
    return;
  }

  __shared__ int32_t s_topk_indices[kMaxTopK];
  radix_topk(score_ptr, s_topk_indices, static_cast<uint32_t>(seq_len), topk);

  __syncthreads();
  for (uint32_t i = threadIdx.x; i < topk; i += kTopKBlockSize) {
    const auto raw = s_topk_indices[i];
    indices_ptr[i] = page_to_slot(page_ptr, static_cast<uint32_t>(raw), params.page_bits);
    if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = raw;
  }
}

}  // namespace

extern "C" CUresult dsv4_dsa_fused_q_indexer_rope_hadamard_quant_cuda(
    const uint16_t* q_input,
    uint8_t* q_fp8,
    const uint16_t* weight,
    float* weights_out,
    float weight_scale,
    const float* freqs_cis,
    const int32_t* positions,
    int batch_size,
    int num_heads,
    CUstream stream) {
  if (q_input == nullptr || q_fp8 == nullptr || weight == nullptr || weights_out == nullptr ||
      freqs_cis == nullptr || positions == nullptr || stream == nullptr || batch_size <= 0 ||
      num_heads <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const auto params = FusedQIndexerRopeHadamardQuantParams{
      .q_input = q_input,
      .q_fp8 = q_fp8,
      .weight = weight,
      .weights_out = weights_out,
      .weight_scale = weight_scale,
      .freqs_cis = freqs_cis,
      .positions = positions,
      .batch_size = static_cast<uint32_t>(batch_size),
      .num_heads = static_cast<uint32_t>(num_heads),
  };
  const uint32_t total_works = static_cast<uint32_t>(batch_size * num_heads);
  const uint32_t num_blocks = (total_works + kFusedQNumWarps - 1) / kFusedQNumWarps;
  fused_q_indexer_rope_hadamard_quant_kernel<<<num_blocks, kFusedQBlockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      params);
  return static_cast<CUresult>(cudaGetLastError());
}

extern "C" CUresult dsv4_dsa_hadamard128_bf16_cuda(
    const uint16_t* input,
    uint16_t* output,
    int rows,
    CUstream stream) {
  if (input == nullptr || output == nullptr || stream == nullptr || rows < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows == 0) return CUDA_SUCCESS;
  const uint32_t num_blocks = (static_cast<uint32_t>(rows) + kFusedQNumWarps - 1) / kFusedQNumWarps;
  hadamard128_bf16_kernel<<<num_blocks, kFusedQBlockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16_t*>(input), reinterpret_cast<bf16_t*>(output), rows);
  return static_cast<CUresult>(cudaGetLastError());
}

extern "C" CUresult dsv4_dsa_fused_store_index_k_cache_cuda(
    const uint16_t* key,
    uint8_t* index_k_with_scale,
    const int64_t* out_cache_loc,
    int num_tokens,
    int page_size,
    CUstream stream) {
  if (key == nullptr || index_k_with_scale == nullptr || out_cache_loc == nullptr ||
      stream == nullptr || num_tokens < 0 || page_size != 64) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  const int threads = 128;
  const int blocks = (num_tokens * 32 + threads - 1) / threads;
  fused_store_indexer_cache_kernel<int64_t, 6>
      <<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
          reinterpret_cast<const bf16_t*>(key), index_k_with_scale, out_cache_loc,
          static_cast<uint32_t>(num_tokens));
  return static_cast<CUresult>(cudaGetLastError());
}

// Batched (grid.y=slot) Hadamard rotate: ONE `<<<dim3(blocks_per_slot, n), 128>>>`
// launch replacing n single-row `dsv4_dsa_hadamard128_bf16_cuda` calls. Per-slot
// base ptrs / row offsets / row counts come from the host-gathered device arrays;
// `max_rows` (= max over slots of newly_packed) sizes the x-grid bound. Math is
// byte-identical to n single-row launches.
extern "C" CUresult dsv4_dsa_hadamard128_batched_cuda(
    const uint16_t* const* keys_src_arr,
    const int32_t* src_ring_row_arr,
    uint16_t* const* rotated_dst_arr,
    const int32_t* dst_row_arr,
    const int32_t* newly_packed_arr,
    int n,
    int max_rows,
    CUstream stream) {
  if (n < 0 || max_rows < 0 || stream == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0 || max_rows == 0) return CUDA_SUCCESS;
  if (keys_src_arr == nullptr || src_ring_row_arr == nullptr || rotated_dst_arr == nullptr ||
      dst_row_arr == nullptr || newly_packed_arr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const uint32_t blocks_per_slot =
      (static_cast<uint32_t>(max_rows) + kFusedQNumWarps - 1) / kFusedQNumWarps;
  dim3 grid(blocks_per_slot, static_cast<uint32_t>(n), 1);
  hadamard128_bf16_batched_kernel<<<grid, kFusedQBlockSize, 0,
                                    reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16_t* const*>(keys_src_arr), src_ring_row_arr,
      reinterpret_cast<bf16_t* const*>(rotated_dst_arr), dst_row_arr, newly_packed_arr, n);
  return static_cast<CUresult>(cudaGetLastError());
}

// Batched (grid.y=slot) FP8 fused-store: ONE `<<<dim3(blocks_per_slot, n), 128>>>`
// launch replacing n single-row `dsv4_dsa_fused_store_index_k_cache_cuda` calls.
// Per-slot rotated-key base / cache band base / slot-local cache-loc array come
// from the host-gathered device arrays; `max_tokens` (= max over slots of
// newly_packed) sizes the x-grid bound. Math is byte-identical to n single-row
// launches.
extern "C" CUresult dsv4_dsa_fused_store_index_k_cache_batched_cuda(
    const uint16_t* const* key_arr,
    uint8_t* const* cache_arr,
    const int64_t* const* out_cache_loc_arr,
    const int32_t* newly_packed_arr,
    int n,
    int max_tokens,
    int page_size,
    CUstream stream) {
  if (n < 0 || max_tokens < 0 || page_size != 64 || stream == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0 || max_tokens == 0) return CUDA_SUCCESS;
  if (key_arr == nullptr || cache_arr == nullptr || out_cache_loc_arr == nullptr ||
      newly_packed_arr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int threads = 128;
  const uint32_t blocks_per_slot =
      (static_cast<uint32_t>(max_tokens) * 32 + threads - 1) / threads;
  dim3 grid(blocks_per_slot, static_cast<uint32_t>(n), 1);
  fused_store_indexer_cache_batched_kernel<int64_t, 6>
      <<<grid, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
          reinterpret_cast<const bf16_t* const*>(key_arr), cache_arr, out_cache_loc_arr,
          newly_packed_arr, n);
  return static_cast<CUresult>(cudaGetLastError());
}

extern "C" CUresult dsv4_deepseek_v4_topk_transform_cuda(
    const float* scores,
    const int32_t* seq_lens,
    const int32_t* page_table,
    int32_t* page_indices,
    int32_t* raw_indices,
    int64_t score_stride,
    int64_t page_table_stride,
    int64_t output_stride,
    int batch_size,
    int topk,
    int page_size,
    CUstream stream) {
  if (scores == nullptr || seq_lens == nullptr || page_table == nullptr ||
      page_indices == nullptr || stream == nullptr || batch_size <= 0 || topk <= 0 ||
      topk > static_cast<int>(kMaxTopK) || page_size <= 0 ||
      (page_size & (page_size - 1)) != 0 || score_stride <= 0 ||
      page_table_stride <= 0 || output_stride < topk) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const auto page_bits = static_cast<uint32_t>(__builtin_ctz(static_cast<unsigned>(page_size)));
  const TopKParams params{
      .scores = scores,
      .seq_lens = seq_lens,
      .page_table = page_table,
      .page_indices = page_indices,
      .raw_indices = raw_indices,
      .score_stride = score_stride,
      .page_table_stride = page_table_stride,
      .page_bits = page_bits,
      .topk = static_cast<uint32_t>(topk),
      .output_stride = output_stride,
  };
  cudaError_t attr = cudaFuncSetAttribute(
      deepseek_v4_topk_transform_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kTopKSmem);
  if (attr != cudaSuccess) return static_cast<CUresult>(attr);
  deepseek_v4_topk_transform_kernel<<<batch_size, kTopKBlockSize, kTopKSmem, reinterpret_cast<cudaStream_t>(stream)>>>(
      params);
  return static_cast<CUresult>(cudaGetLastError());
}

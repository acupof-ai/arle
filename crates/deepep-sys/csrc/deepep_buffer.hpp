// SPDX-License-Identifier: Apache-2.0
// Phase B-2 — torch-free C interface to DeepEP intranode dispatch/combine.
//
// This is the Rust-callable surface of the same Buffer-lifecycle the
// phase 1.0a-iv spike + sidecar proved on 8 × H20. The implementation
// (deepep_buffer.cpp) links against DeepEP's torch-free
// csrc/kernels/{api.cuh, intranode.cu, layout.cu, runtime.cu} via the
// build.rs nvcc pipeline.
//
// Wire shape (no torch::Tensor, no pybind, no Python):
//   1. arle_deepep_buffer_create(rank, world_size) -> handle. Allocates
//      512 MiB NVL buffer + 32 MiB workspace + host-mapped MoE counters.
//   2. arle_deepep_buffer_local_ipc_handle(h, out64) -> int. Returns
//      cudaIpcMemHandle_t reserved bytes.
//   3. arle_deepep_buffer_sync(h, peer_ipc_handles, peer_device_ids,
//      world_size) -> int. Opens N-1 peer IPC handles, populates the
//      device-side peer pointer arrays, runs intranode::barrier.
//   4. arle_deepep_buffer_dispatch(h, params, raw_pointers...) -> int.
//      Runs notify_dispatch + intranode::dispatch on host-provided
//      x/topk_idx/topk_weights device buffers; writes recv_x and
//      handle-side metadata (rank_prefix_matrix, recv_channel_prefix,
//      send_head) to caller-allocated device buffers.
//   5. arle_deepep_buffer_combine(h, params, raw_pointers...) -> int.
//      Runs cached_notify_combine + intranode::combine.
//   6. arle_deepep_buffer_destroy(h). Frees buffers; closes IPC.
//
// Error codes (negative on failure):
//   0  = ok
//   -1 = bad args
//   -2 = cuda error (last cudaGetErrorString in arle_deepep_last_error)
//   -3 = kernel timeout / trap
//   -4 = sync precondition unmet (e.g. dispatch before sync)
//   -5 = world_size out of [2, 8]

#ifndef ARLE_DEEPEP_BUFFER_HPP_
#define ARLE_DEEPEP_BUFFER_HPP_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle. Internally a `BufferState*`.
typedef struct ArleDeepEpBuffer ArleDeepEpBuffer;

// Returned by every fallible call. Negative = error per code list above.
typedef int ArleDeepEpStatus;

// IPC handle size is the same 64-byte opaque as cudaIpcMemHandle_t.reserved.
#define ARLE_DEEPEP_IPC_HANDLE_BYTES 64

// `device_id` is the REAL CUDA device ordinal this rank runs on (from
// INFER_CUDA_DEVICES, e.g. 4 for rank 0 on a 4,5,6,7 set). It is NOT the
// TP rank — on a non-0-based GPU set rank != ordinal, and pinning by rank
// would select the wrong physical GPU (cudaErrorInvalidResourceHandle at
// barrier launch). All buffers/streams are created on this device.
ArleDeepEpStatus arle_deepep_buffer_create(
    uint32_t rank, uint32_t world_size, uint32_t device_id,
    ArleDeepEpBuffer **out_handle);

// Writes the local CUDA IPC handle into `out_ipc_handle[0..64]` and the
// local device_id into `out_device_id`.
ArleDeepEpStatus arle_deepep_buffer_local_ipc_handle(
    ArleDeepEpBuffer *handle,
    uint8_t out_ipc_handle[ARLE_DEEPEP_IPC_HANDLE_BYTES],
    uint32_t *out_device_id);

// Open peer IPC handles + run intranode::barrier. `peer_ipc_handles` is a
// contiguous array of `world_size * ARLE_DEEPEP_IPC_HANDLE_BYTES` bytes;
// `peer_device_ids` has length `world_size`. The slot at `rank` is
// ignored (filled with this rank's own handle by the caller for layout
// symmetry, but the implementation uses its own local_buf pointer).
ArleDeepEpStatus arle_deepep_buffer_sync(
    ArleDeepEpBuffer *handle,
    const uint8_t *peer_ipc_handles,
    const uint32_t *peer_device_ids,
    uint32_t world_size);

// Parameters for a single dispatch call. Mirrors `intranode::dispatch`
// signature shape; all device pointers are `uintptr_t` so this header
// stays CUDA-free for Rust binding generation.
typedef struct ArleDeepEpDispatchParams {
    uint32_t num_tokens;
    uint32_t hidden;
    uint32_t num_topk;
    uint32_t num_experts;
    uint32_t num_sms;
    uint32_t nvl_chunked_send;
    uint32_t nvl_chunked_recv;
    // Input device pointers (caller-owned).
    uintptr_t d_x;            // __nv_bfloat16[num_tokens, hidden]
    uintptr_t d_topk_idx;     // int64_t[num_tokens, num_topk]
    uintptr_t d_topk_weights; // float[num_tokens, num_topk]
    // Output device pointers (caller-allocated; size = worst-case slots).
    uintptr_t d_recv_x;             // __nv_bfloat16[max_recv_tokens, hidden]
    uintptr_t d_recv_src_idx;       // int[max_recv_tokens]
    uintptr_t d_recv_topk_idx;      // int64_t[max_recv_tokens, num_topk]
    uintptr_t d_recv_topk_weights;  // float[max_recv_tokens, num_topk]
    uintptr_t d_rank_prefix_matrix; // int[world_size, world_size]
    uintptr_t d_recv_channel_prefix;// int[world_size, num_channels]
    uintptr_t d_send_head;          // int[num_tokens, world_size]
    // Caller-allocated scratch (we don't allocate inside dispatch to keep
    // the C ABI simple).
    uintptr_t d_num_tokens_per_rank;   // int[world_size]
    uintptr_t d_num_tokens_per_expert; // int[num_experts]
    uintptr_t d_is_token_in_rank;      // bool[num_tokens, world_size]
    uintptr_t d_channel_prefix_matrix; // int[world_size, num_channels]
    // Caller's COMPUTE stream (cudaStream_t). When non-zero, the dispatch uses
    // event-based stream_wait instead of host cudaStreamSynchronize. 0 = host sync.
    uintptr_t compute_stream;
    // Out — actual recv token count, filled by host-poll after kernel.
    int32_t *out_num_recv_tokens;
} ArleDeepEpDispatchParams;

ArleDeepEpStatus arle_deepep_buffer_dispatch(
    ArleDeepEpBuffer *handle, const ArleDeepEpDispatchParams *params);

typedef struct ArleDeepEpCombineParams {
    uint32_t num_input_tokens;  // = dispatch's out_num_recv_tokens
    uint32_t num_output_tokens; // = original input token count
    uint32_t hidden;
    uint32_t num_topk;
    uint32_t num_sms;
    uint32_t nvl_chunked_send;
    uint32_t nvl_chunked_recv;
    // Input device pointers (caller-owned).
    uintptr_t d_x;               // __nv_bfloat16[num_input_tokens, hidden]
    uintptr_t d_topk_weights;    // float[num_input_tokens, num_topk]
    uintptr_t d_recv_src_idx;    // int[num_input_tokens]
    uintptr_t d_rank_prefix_matrix;   // int[world_size, world_size]
    uintptr_t d_recv_channel_prefix;  // int[world_size, num_channels]
    uintptr_t d_send_head;       // int[num_output_tokens, world_size]
    // Outputs (caller-allocated).
    uintptr_t d_combined_x;          // __nv_bfloat16[num_output_tokens, hidden]
    uintptr_t d_combined_topk_w;     // float[num_output_tokens, num_topk]
    // Caller's COMPUTE stream (cudaStream_t). When non-zero, the combine uses
    // event-based stream_wait instead of host cudaStreamSynchronize. 0 = host sync.
    uintptr_t compute_stream;
} ArleDeepEpCombineParams;

ArleDeepEpStatus arle_deepep_buffer_combine(
    ArleDeepEpBuffer *handle, const ArleDeepEpCombineParams *params);

void arle_deepep_buffer_destroy(ArleDeepEpBuffer *handle);

// Returns the static error string from the most recent CUDA failure on
// the calling thread. Read-only; do not free.
const char *arle_deepep_last_error(void);

// ===================================================================
// NVSHMEM low-latency (internode_ll) path — mirrors DeepEP
// csrc/legacy/buffer.hpp {low_latency_dispatch, low_latency_combine}
// + the NVSHMEM UID bootstrap in csrc/kernels/backend/nvshmem.cu.
//
// Lifecycle (distinct from the intranode path above):
//   1. arle_deepep_ll_get_uniqueid(out128) — rank 0 ONLY. Fills a 128-byte
//      nvshmemx_uniqueid_t. The CALLER broadcasts these bytes to all ranks
//      over its existing peer channel (NCCL / sidecar pipe), exactly like
//      the intranode IPC-handle exchange in arle_deepep_buffer_sync.
//   2. arle_deepep_buffer_ll_create(rank, world_size, num_max_dispatch_
//      tokens_per_rank, hidden, num_experts, root_uniqueid128, out_handle)
//      — every rank. Runs nvshmem::init(uid, rank, world) +
//      nvshmem_align(rdma_bytes) + memset, computes the LowLatencyLayout.
//      Collective: all ranks must call with the SAME uid + sizing.
//   3. arle_deepep_buffer_low_latency_dispatch(h, params) — packs the
//      per-rank tokens into the [num_local_experts, world*max_tok, hidden]
//      FP8/BF16 recv buffer; returns expected_m via out_expected_m.
//   4. arle_deepep_buffer_low_latency_combine(h, params) — reduces the
//      per-expert outputs back to [num_combined_tokens, hidden].
//   5. arle_deepep_buffer_destroy(h) frees the LL buffer + finalizes
//      NVSHMEM (shared with the intranode destroy).
// ===================================================================

#define ARLE_DEEPEP_LL_UNIQUEID_BYTES 128

// Rank-0 only: fill the 128-byte NVSHMEM unique id for broadcast.
ArleDeepEpStatus arle_deepep_ll_get_uniqueid(
    uint8_t out_uniqueid[ARLE_DEEPEP_LL_UNIQUEID_BYTES]);

// Create an LL buffer. `root_uniqueid` is the 128-byte blob produced by
// arle_deepep_ll_get_uniqueid on rank 0 and broadcast to all ranks. The
// rdma buffer is sized from (num_max_dispatch_tokens_per_rank, hidden,
// world_size, num_experts) per DeepEP's LowLatencyLayout.
// `device_id` is the REAL CUDA device ordinal this rank runs on (see
// arle_deepep_buffer_create); NOT the TP rank.
ArleDeepEpStatus arle_deepep_buffer_ll_create(
    uint32_t rank,
    uint32_t world_size,
    uint32_t device_id,
    uint32_t num_max_dispatch_tokens_per_rank,
    uint32_t hidden,
    uint32_t num_experts,
    const uint8_t root_uniqueid[ARLE_DEEPEP_LL_UNIQUEID_BYTES],
    ArleDeepEpBuffer **out_handle);

// Parameters for a single low_latency_dispatch. Device pointers are
// `uintptr_t` to keep this header CUDA-free.
typedef struct ArleDeepEpLowLatencyDispatchParams {
    uint32_t num_tokens;   // x.size(0), <= num_max_dispatch_tokens_per_rank
    uint32_t hidden;       // x.size(1); must match ll_create hidden
    uint32_t num_topk;     // topk_idx.size(1)
    uint32_t num_experts;  // global; must match ll_create num_experts
    uint32_t num_max_dispatch_tokens_per_rank; // must match ll_create
    uint8_t  use_fp8;      // 1 → packed FP8 e4m3 + scales; 0 → BF16
    uint8_t  round_scale;  // FP8 rounding mode (DeepEP `round_scale`)
    uint8_t  use_ue8m0;    // FP8 ue8m0 scale format (implies round_scale)
    // Inputs (caller-owned).
    uintptr_t d_x;         // __nv_bfloat16[num_tokens, hidden]
    uintptr_t d_topk_idx;  // int64_t[num_tokens, num_topk]
    // Outputs (caller-allocated; sized per the packed LL recv layout):
    //   d_recv_x:        [num_local_experts, world*max_tok, hidden]
    //                    dtype = FP8 e4m3 (use_fp8) else BF16
    //   d_recv_x_scales: FP8 only — [num_local_experts, world*max_tok,
    //                    hidden/128] float32 (column-major; pass the raw
    //                    contiguous buffer, the kernel writes transposed)
    //   d_recv_src_info: int[num_local_experts, world*max_tok]
    //   d_recv_layout_range: int64[num_local_experts, world]
    //   d_recv_count:    int[num_local_experts]  (masked_m / packed count)
    uintptr_t d_recv_x;
    uintptr_t d_recv_x_scales;     // nullable when use_fp8 == 0
    uintptr_t d_recv_src_info;
    uintptr_t d_recv_layout_range;
    uintptr_t d_recv_count;
    // Caller's COMPUTE stream (cudaStream_t). 0 ⇒ host sync.
    uintptr_t compute_stream;
    // Out — expected_m = ceil(num_tokens * world * num_topk / num_experts).
    int32_t *out_expected_m;
} ArleDeepEpLowLatencyDispatchParams;

ArleDeepEpStatus arle_deepep_buffer_low_latency_dispatch(
    ArleDeepEpBuffer *handle,
    const ArleDeepEpLowLatencyDispatchParams *params);

typedef struct ArleDeepEpLowLatencyCombineParams {
    uint32_t num_combined_tokens; // topk_weights.size(0)
    uint32_t hidden;              // must match ll_create hidden
    uint32_t num_topk;            // topk_weights.size(1)
    uint32_t num_experts;         // global; must match ll_create
    uint32_t num_max_dispatch_tokens_per_rank; // must match ll_create
    uint8_t  use_logfmt;          // DeepEP combine `use_logfmt`
    uint8_t  zero_copy;           // DeepEP combine `zero_copy`
    // Inputs (caller-owned).
    //   d_x:           __nv_bfloat16[num_local_experts, world*max_tok, hidden]
    //   d_topk_idx:    int64_t[num_combined_tokens, num_topk]
    //   d_topk_weights:float[num_combined_tokens, num_topk]
    //   d_src_info:    int[num_local_experts, world*max_tok]  (dispatch out)
    //   d_layout_range:int64[num_local_experts, world]        (dispatch out)
    uintptr_t d_x;
    uintptr_t d_topk_idx;
    uintptr_t d_topk_weights;
    uintptr_t d_src_info;
    uintptr_t d_layout_range;
    // Output (caller-allocated): __nv_bfloat16[num_combined_tokens, hidden]
    uintptr_t d_combined_x;
    // Caller's COMPUTE stream (cudaStream_t). 0 ⇒ host sync.
    uintptr_t compute_stream;
} ArleDeepEpLowLatencyCombineParams;

ArleDeepEpStatus arle_deepep_buffer_low_latency_combine(
    ArleDeepEpBuffer *handle,
    const ArleDeepEpLowLatencyCombineParams *params);

// Returns 1 if `handle` is an LL buffer (low_latency_mode), else 0.
int arle_deepep_buffer_is_low_latency(const ArleDeepEpBuffer *handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ARLE_DEEPEP_BUFFER_HPP_

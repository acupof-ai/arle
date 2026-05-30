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

ArleDeepEpStatus arle_deepep_buffer_create(
    uint32_t rank, uint32_t world_size, ArleDeepEpBuffer **out_handle);

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ARLE_DEEPEP_BUFFER_HPP_

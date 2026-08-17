// SPDX-License-Identifier: Apache-2.0
// Phase B-2 — torch-free C wrapper around DeepEP intranode kernels.
//
// Mirrors the Buffer-lifecycle proven in phase 1.0a-iv + sidecar_main.cpp.
// Differences from the sidecar:
//   - Library form (no fork/exec, no command loop, no pipe IO).
//   - C ABI for Rust binding (extern "C" + uintptr_t pointer args).
//   - No SHA-256 / no synthetic input — caller provides device buffers.
//   - Same NVL layout, same notify_dispatch/dispatch/cached_notify_
//     combine/combine call shape, same parameter naming traps fixed
//     (see feedback_deepep_kernel_api_inverted_naming +
//      feedback_deepep_combine_uses_recv_channel_prefix memories).

#include "deepep_buffer.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// DeepEP intranode/layout kernels (upstream csrc/kernels/legacy tree,
// pinned d4f41e4). The constant aliases hide upstream's NUM_MAX_*
// constant names behind the clean names our code uses.
#include "kernels/legacy/api.cuh"
#include "kernels/legacy/compiled.cuh"
// LowLatencyLayout + get_low_latency_rdma_size_hint — torch-free pointer
// arithmetic only (pulls deep_ep/common/exception.cuh + the same
// kernels/legacy/api.cuh already included). Reused verbatim so the LL
// recv-buffer sizing + handle layout can NEVER drift from the kernels.
#if defined(ARLE_DEEPEP_HAVE_NVSHMEM)
#include "legacy/config.hpp"
#endif
#ifndef NUM_MAX_NVL_PEERS
#define NUM_MAX_NVL_PEERS LEGACY_NUM_MAX_NVL_PEERS
#endif
#ifndef NUM_MAX_LOCAL_EXPERTS
#define NUM_MAX_LOCAL_EXPERTS LEGACY_NUM_MAX_LOCAL_EXPERTS
#endif
#ifndef NUM_WORKSPACE_BYTES
#define NUM_WORKSPACE_BYTES LEGACY_NUM_WORKSPACE_BYTES
#endif
namespace deep_ep_intranode_ns = deep_ep::legacy::intranode;
namespace deep_ep_layout_ns = deep_ep::legacy::layout;

namespace {

thread_local char g_last_error[256] = "";

void record_cuda_error(const char *ctx, cudaError_t err) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "%s: %s", ctx, cudaGetErrorString(err));
}

void record_error(const char *msg) {
    std::strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
}

constexpr int kMaxRanks = 8;
constexpr int64_t kNvlBytes = 512LL << 20;
constexpr int64_t kBarrierSignalBytes = NUM_MAX_NVL_PEERS * sizeof(int);
constexpr int64_t kBufferPtrBytes = NUM_MAX_NVL_PEERS * sizeof(void *);
constexpr int64_t kBarrierSignalPtrBytes = NUM_MAX_NVL_PEERS * sizeof(int *);
constexpr int64_t kTotalBytes = kNvlBytes + kBarrierSignalBytes +
                                kBufferPtrBytes + kBarrierSignalPtrBytes;

}  // namespace

struct ArleDeepEpBuffer {
    int rank = -1;
    int world_size = 0;
    int device_id = -1;

    void *local_buf = nullptr;
    void *workspace = nullptr;

    int *moe_recv_counter_host = nullptr;
    int *moe_recv_counter_dev = nullptr;
    int *moe_recv_expert_host = nullptr;
    int *moe_recv_expert_dev = nullptr;

    void *peer_buf_ptrs[NUM_MAX_NVL_PEERS] = {};
    int *peer_barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {};

    void **buffer_ptrs_gpu = nullptr;
    int **barrier_signal_ptrs_gpu = nullptr;

    cudaStream_t stream{};
    bool synced = false;

    // --- NVSHMEM low-latency (internode_ll) mode ---------------------
    // Populated by arle_deepep_buffer_ll_create; nullptr/0 for intranode.
    bool low_latency_mode = false;
    void *rdma_buffer_ptr = nullptr;       // nvshmem_align symmetric heap
    int64_t num_rdma_bytes = 0;
    int low_latency_buffer_idx = 0;        // double-buffer toggle (^=1 each call)
    int ll_num_max_dispatch_tokens_per_rank = 0;
    int ll_hidden = 0;
    int ll_num_experts = 0;
    int num_device_sms = 0;
};

#define CK(call, ctx)                                                          \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            record_cuda_error(ctx, err);                                       \
            return -2;                                                         \
        }                                                                      \
    } while (0)

extern "C" ArleDeepEpStatus arle_deepep_buffer_create(
    uint32_t rank, uint32_t world_size, uint32_t device_id,
    ArleDeepEpBuffer **out_handle) {
    if (!out_handle) {
        record_error("out_handle is null");
        return -1;
    }
    *out_handle = nullptr;
    if (world_size < 2 || world_size > kMaxRanks || rank >= world_size) {
        record_error("world_size must be in [2,8] and rank < world_size");
        return -5;
    }

    int device_count = 0;
    CK(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < (int)world_size) {
        record_error("not enough CUDA devices");
        return -1;
    }
    if (static_cast<int>(device_id) >= device_count) {
        record_error("device_id out of range");
        return -1;
    }

    auto *self = new ArleDeepEpBuffer();
    self->rank = static_cast<int>(rank);
    self->world_size = static_cast<int>(world_size);

    cudaError_t err;
    // Pin the REAL device ordinal this rank runs on (INFER_CUDA_DEVICES),
    // NOT the TP rank. On a non-0-based GPU set (e.g. 4,5,6,7) rank != ordinal;
    // setting the device by rank selected the wrong physical GPU, so the
    // buffers/streams created here mismatched the executor's context and the
    // intranode barrier launch failed with cudaErrorInvalidResourceHandle.
    err = cudaSetDevice(static_cast<int>(device_id));
    if (err != cudaSuccess) {
        record_cuda_error("cudaSetDevice", err);
        delete self;
        return -2;
    }
    err = cudaGetDevice(&self->device_id);
    if (err != cudaSuccess) {
        record_cuda_error("cudaGetDevice", err);
        delete self;
        return -2;
    }

    if ((err = cudaMalloc(&self->local_buf, kTotalBytes)) != cudaSuccess) {
        record_cuda_error("cudaMalloc local_buf", err);
        delete self;
        return -2;
    }
    // Zero the NVL IPC buffer. cudaMalloc leaves it uninitialized; the DeepEP
    // dispatch/combine kernels read header/metadata regions of this buffer that
    // are not fully written on the first pass, so an uninitialized read causes a
    // combine OOB / launch failure. compute-sanitizer zero-fills allocations,
    // which is exactly why the combine "worked" under memcheck — replicate it.
    if ((err = cudaMemset(self->local_buf, 0, kTotalBytes)) != cudaSuccess) {
        record_cuda_error("cudaMemset local_buf", err);
        cudaFree(self->local_buf);
        delete self;
        return -2;
    }
    if ((err = cudaMalloc(&self->workspace, NUM_WORKSPACE_BYTES)) != cudaSuccess) {
        record_cuda_error("cudaMalloc workspace", err);
        cudaFree(self->local_buf);
        delete self;
        return -2;
    }
    if ((err = cudaMemset(self->workspace, 0, NUM_WORKSPACE_BYTES)) != cudaSuccess) {
        record_cuda_error("cudaMemset workspace", err);
        cudaFree(self->workspace);
        cudaFree(self->local_buf);
        delete self;
        return -2;
    }
    if ((err = cudaHostAlloc(&self->moe_recv_counter_host, sizeof(int),
                             cudaHostAllocMapped)) != cudaSuccess) {
        record_cuda_error("cudaHostAlloc moe_recv_counter", err);
        cudaFree(self->workspace);
        cudaFree(self->local_buf);
        delete self;
        return -2;
    }
    if ((err = cudaHostGetDevicePointer(&self->moe_recv_counter_dev,
                                        self->moe_recv_counter_host, 0)) !=
        cudaSuccess) {
        record_cuda_error("cudaHostGetDevicePointer moe_recv_counter", err);
        cudaFreeHost(self->moe_recv_counter_host);
        cudaFree(self->workspace);
        cudaFree(self->local_buf);
        delete self;
        return -2;
    }
    if ((err = cudaHostAlloc(&self->moe_recv_expert_host,
                             sizeof(int) * NUM_MAX_LOCAL_EXPERTS,
                             cudaHostAllocMapped)) != cudaSuccess) {
        record_cuda_error("cudaHostAlloc moe_recv_expert", err);
        cudaFreeHost(self->moe_recv_counter_host);
        cudaFree(self->workspace);
        cudaFree(self->local_buf);
        delete self;
        return -2;
    }
    if ((err = cudaHostGetDevicePointer(&self->moe_recv_expert_dev,
                                        self->moe_recv_expert_host, 0)) !=
        cudaSuccess) {
        record_cuda_error("cudaHostGetDevicePointer moe_recv_expert", err);
        cudaFreeHost(self->moe_recv_expert_host);
        cudaFreeHost(self->moe_recv_counter_host);
        cudaFree(self->workspace);
        cudaFree(self->local_buf);
        delete self;
        return -2;
    }
    if ((err = cudaStreamCreate(&self->stream)) != cudaSuccess) {
        record_cuda_error("cudaStreamCreate", err);
        cudaFreeHost(self->moe_recv_expert_host);
        cudaFreeHost(self->moe_recv_counter_host);
        cudaFree(self->workspace);
        cudaFree(self->local_buf);
        delete self;
        return -2;
    }
    cudaMemsetAsync(self->local_buf, 0, kTotalBytes, self->stream);
    cudaMemsetAsync(self->workspace, 0, NUM_WORKSPACE_BYTES, self->stream);
    if ((err = cudaStreamSynchronize(self->stream)) != cudaSuccess) {
        record_cuda_error("init memset sync", err);
        arle_deepep_buffer_destroy(self);
        return -2;
    }

    *out_handle = self;
    return 0;
}

extern "C" ArleDeepEpStatus arle_deepep_buffer_local_ipc_handle(
    ArleDeepEpBuffer *self,
    uint8_t out_ipc_handle[ARLE_DEEPEP_IPC_HANDLE_BYTES],
    uint32_t *out_device_id) {
    if (!self || !out_ipc_handle || !out_device_id) {
        record_error("null arg");
        return -1;
    }
    cudaIpcMemHandle_t h{};
    cudaError_t err = cudaIpcGetMemHandle(&h, self->local_buf);
    if (err != cudaSuccess) {
        record_cuda_error("cudaIpcGetMemHandle", err);
        return -2;
    }
    static_assert(sizeof(h.reserved) == ARLE_DEEPEP_IPC_HANDLE_BYTES,
                  "IPC handle reserved field must be 64 bytes");
    std::memcpy(out_ipc_handle, h.reserved, sizeof(h.reserved));
    *out_device_id = static_cast<uint32_t>(self->device_id);
    return 0;
}

extern "C" ArleDeepEpStatus arle_deepep_buffer_sync(
    ArleDeepEpBuffer *self,
    const uint8_t *peer_ipc_handles,
    const uint32_t *peer_device_ids,
    uint32_t world_size) {
    if (!self || !peer_ipc_handles || !peer_device_ids) {
        record_error("null arg");
        return -1;
    }
    if (static_cast<int>(world_size) != self->world_size) {
        record_error("world_size mismatch with handle ctor");
        return -1;
    }
    // Belt-and-suspenders: every barrier/kernel launch on self->stream must
    // run under the context of the device the buffers were created on. Pin it
    // here (idempotent) so a caller thread on a different current device can't
    // launch the intranode barrier into the wrong context.
    CK(cudaSetDevice(self->device_id), "cudaSetDevice (sync)");

    for (int i = 0; i < self->world_size; ++i) {
        if (i == self->rank) {
            self->peer_buf_ptrs[i] = self->local_buf;
        } else {
            cudaIpcMemHandle_t h{};
            std::memcpy(h.reserved,
                        peer_ipc_handles + i * ARLE_DEEPEP_IPC_HANDLE_BYTES,
                        sizeof(h.reserved));
            void *p = nullptr;
            cudaError_t err =
                cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess);
            if (err != cudaSuccess) {
                record_cuda_error("cudaIpcOpenMemHandle", err);
                return -2;
            }
            self->peer_buf_ptrs[i] = p;
        }
        self->peer_barrier_signal_ptrs[i] = reinterpret_cast<int *>(
            static_cast<uint8_t *>(self->peer_buf_ptrs[i]) + kNvlBytes);
        (void)peer_device_ids;  // currently unused — reserved for IB cases
    }

    self->buffer_ptrs_gpu = reinterpret_cast<void **>(
        static_cast<uint8_t *>(self->local_buf) + kNvlBytes +
        kBarrierSignalBytes);
    self->barrier_signal_ptrs_gpu = reinterpret_cast<int **>(
        static_cast<uint8_t *>(self->local_buf) + kNvlBytes +
        kBarrierSignalBytes + kBufferPtrBytes);

    CK(cudaMemcpyAsync(self->buffer_ptrs_gpu, self->peer_buf_ptrs,
                       sizeof(self->peer_buf_ptrs), cudaMemcpyHostToDevice,
                       self->stream),
       "memcpy buffer_ptrs_gpu");
    CK(cudaMemcpyAsync(self->barrier_signal_ptrs_gpu,
                       self->peer_barrier_signal_ptrs,
                       sizeof(self->peer_barrier_signal_ptrs),
                       cudaMemcpyHostToDevice, self->stream),
       "memcpy barrier_signal_ptrs_gpu");
    CK(cudaStreamSynchronize(self->stream), "sync after peer ptr upload");

    deep_ep_intranode_ns::barrier(self->barrier_signal_ptrs_gpu, self->rank,
                                self->world_size, self->stream);
    CK(cudaStreamSynchronize(self->stream), "sync after intranode::barrier");

    self->synced = true;
    return 0;
}

extern "C" ArleDeepEpStatus arle_deepep_buffer_dispatch(
    ArleDeepEpBuffer *self, const ArleDeepEpDispatchParams *p) {
    // ABI guard: compute_stream must sit immediately before out_num_recv_tokens
    // so the Rust repr(C) struct and this C struct stay field-order-identical.
    static_assert(offsetof(ArleDeepEpDispatchParams, out_num_recv_tokens) ==
                          offsetof(ArleDeepEpDispatchParams, compute_stream) +
                              sizeof(uintptr_t),
                  "ArleDeepEpDispatchParams ABI drift: compute_stream/out_num_recv_tokens");
    if (!self || !p) {
        record_error("null arg");
        return -1;
    }
    if (!self->synced) {
        record_error("dispatch before sync");
        return -4;
    }
    // Pin the buffer's real device so every launch on self->stream runs under
    // the matching context (idempotent; see arle_deepep_buffer_sync).
    CK(cudaSetDevice(self->device_id), "cudaSetDevice (dispatch)");
    if (p->num_sms % 2 != 0 || p->num_sms == 0) {
        record_error("num_sms must be positive and even");
        return -1;
    }
    const int num_channels = p->num_sms / 2;
    const int hidden_int4 =
        p->hidden * sizeof(__nv_bfloat16) / sizeof(int4);

    auto *d_x = reinterpret_cast<__nv_bfloat16 *>(p->d_x);
    auto *d_topk_idx = reinterpret_cast<int64_t *>(p->d_topk_idx);
    auto *d_topk_w = reinterpret_cast<float *>(p->d_topk_weights);
    auto *d_recv_x = reinterpret_cast<__nv_bfloat16 *>(p->d_recv_x);
    auto *d_recv_src_idx = reinterpret_cast<int *>(p->d_recv_src_idx);
    auto *d_recv_topk_idx =
        reinterpret_cast<int64_t *>(p->d_recv_topk_idx);
    auto *d_recv_topk_w = reinterpret_cast<float *>(p->d_recv_topk_weights);
    auto *d_rank_prefix = reinterpret_cast<int *>(p->d_rank_prefix_matrix);
    auto *d_recv_channel_prefix =
        reinterpret_cast<int *>(p->d_recv_channel_prefix);
    auto *d_send_head = reinterpret_cast<int *>(p->d_send_head);
    auto *d_num_tokens_per_rank =
        reinterpret_cast<int *>(p->d_num_tokens_per_rank);
    auto *d_num_tokens_per_expert =
        reinterpret_cast<int *>(p->d_num_tokens_per_expert);
    auto *d_is_token_in_rank = reinterpret_cast<bool *>(p->d_is_token_in_rank);
    auto *d_channel_prefix =
        reinterpret_cast<int *>(p->d_channel_prefix_matrix);

    // Event-in: the dispatch reads d_x/d_topk_idx/d_topk_weights produced on
    // the caller's COMPUTE stream. Make the comm stream (self->stream) wait
    // ON-DEVICE rather than host-syncing the caller. (Falls back to a host
    // sync at the end when compute_stream == 0.)
    cudaStream_t compute_stream =
        reinterpret_cast<cudaStream_t>(p->compute_stream);
    if (compute_stream) {
        cudaEvent_t ev_in;
        cudaEventCreateWithFlags(&ev_in, cudaEventDisableTiming);
        cudaEventRecord(ev_in, compute_stream);
        cudaStreamWaitEvent(self->stream, ev_in, 0);
        cudaEventDestroy(ev_in);
    }

    // 1. layout
    deep_ep_layout_ns::get_dispatch_layout(
        d_topk_idx, d_num_tokens_per_rank, /*num_tokens_per_rdma_rank=*/nullptr,
        d_num_tokens_per_expert, d_is_token_in_rank,
        static_cast<int>(p->num_tokens), static_cast<int>(p->num_topk),
        self->world_size, static_cast<int>(p->num_experts), self->stream);

    // 2. notify_dispatch + host-poll
    *self->moe_recv_counter_host = -1;
    int experts_per_rank =
        static_cast<int>(p->num_experts) / self->world_size;
    for (int i = 0; i < experts_per_rank; ++i)
        self->moe_recv_expert_host[i] = -1;
    int num_memset_int = num_channels * self->world_size * 4;
    deep_ep_intranode_ns::notify_dispatch(
        d_num_tokens_per_rank, self->moe_recv_counter_dev, self->world_size,
        d_num_tokens_per_expert, self->moe_recv_expert_dev,
        static_cast<int>(p->num_experts),
        static_cast<int>(p->num_tokens), d_is_token_in_rank, d_channel_prefix,
        d_rank_prefix, num_memset_int, /*expert_alignment=*/1,
        self->buffer_ptrs_gpu, self->barrier_signal_ptrs_gpu, self->rank,
        self->stream, num_channels);

    int num_recv_tokens = -1;
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        num_recv_tokens = *self->moe_recv_counter_host;
        bool ready = (num_recv_tokens >= 0);
        for (int i = 0; i < experts_per_rank && ready; ++i)
            ready = ready && (self->moe_recv_expert_host[i] >= 0);
        if (ready) break;
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0)
                .count() > 15) {
            record_error("notify_dispatch host-poll timeout");
            return -3;
        }
    }

    // 3. dispatch
    deep_ep_intranode_ns::dispatch(
        d_recv_x, /*recv_x_scales=*/nullptr, d_recv_src_idx, d_recv_topk_idx,
        d_recv_topk_w, d_recv_channel_prefix, d_send_head, d_x,
        /*x_scales=*/nullptr, d_topk_idx, d_topk_w, d_is_token_in_rank,
        d_channel_prefix, static_cast<int>(p->num_tokens),
        /*num_worst_tokens=*/0, hidden_int4, static_cast<int>(p->num_topk),
        static_cast<int>(p->num_experts), /*num_scales=*/0,
        /*scale_token_stride=*/0, /*scale_hidden_stride=*/0,
        self->buffer_ptrs_gpu, self->rank, self->world_size, self->stream,
        static_cast<int>(p->num_sms),
        static_cast<int>(p->nvl_chunked_send),
        static_cast<int>(p->nvl_chunked_recv));
    if (compute_stream) {
        // Compute stream waits for the dispatch output ON-DEVICE; no host block.
        cudaEvent_t ev_out;
        cudaEventCreateWithFlags(&ev_out, cudaEventDisableTiming);
        cudaEventRecord(ev_out, self->stream);
        cudaStreamWaitEvent(compute_stream, ev_out, 0);
        cudaEventDestroy(ev_out);
    } else {
        CK(cudaStreamSynchronize(self->stream), "sync after dispatch");
    }

    if (p->out_num_recv_tokens) *p->out_num_recv_tokens = num_recv_tokens;
    return 0;
}

extern "C" ArleDeepEpStatus arle_deepep_buffer_combine(
    ArleDeepEpBuffer *self, const ArleDeepEpCombineParams *p) {
    if (!self || !p) {
        record_error("null arg");
        return -1;
    }
    if (!self->synced) {
        record_error("combine before sync");
        return -4;
    }
    // Pin the buffer's real device so every launch on self->stream runs under
    // the matching context (idempotent; see arle_deepep_buffer_sync).
    CK(cudaSetDevice(self->device_id), "cudaSetDevice (combine)");
    if (p->num_sms % 2 != 0 || p->num_sms == 0) {
        record_error("num_sms must be positive and even");
        return -1;
    }
    const int num_channels = p->num_sms / 2;

    auto *d_x = reinterpret_cast<__nv_bfloat16 *>(p->d_x);
    auto *d_topk_w = reinterpret_cast<float *>(p->d_topk_weights);
    auto *d_recv_src_idx = reinterpret_cast<int *>(p->d_recv_src_idx);
    auto *d_rank_prefix = reinterpret_cast<int *>(p->d_rank_prefix_matrix);
    auto *d_recv_channel_prefix =
        reinterpret_cast<int *>(p->d_recv_channel_prefix);
    auto *d_send_head = reinterpret_cast<int *>(p->d_send_head);
    auto *d_combined_x = reinterpret_cast<__nv_bfloat16 *>(p->d_combined_x);
    auto *d_combined_topk_w =
        reinterpret_cast<float *>(p->d_combined_topk_w);

    // Event-based cross-stream ordering (mirrors DeepEP's official
    // stream_wait(comm_stream, compute_stream)). The combine reads d_x (the
    // expert output) produced on the caller's COMPUTE stream, and writes
    // d_combined_x consumed there. Make the comm stream (self->stream) wait for
    // the compute stream ON-DEVICE rather than host-syncing the caller — so the
    // CPU isn't blocked for the ~ms expert-GEMM wait every layer. (Falls back to
    // a host sync at the end when compute_stream == 0.)
    cudaStream_t compute_stream =
        reinterpret_cast<cudaStream_t>(p->compute_stream);
    if (compute_stream) {
        cudaEvent_t ev_in;
        cudaEventCreateWithFlags(&ev_in, cudaEventDisableTiming);
        cudaEventRecord(ev_in, compute_stream);
        cudaStreamWaitEvent(self->stream, ev_in, 0);
        cudaEventDestroy(ev_in);
    }

    deep_ep_intranode_ns::cached_notify_combine(
        self->buffer_ptrs_gpu, d_send_head, num_channels,
        /*num_recv_tokens=*/static_cast<int>(p->num_output_tokens),
        /*num_memset_int=*/num_channels * self->world_size * 2,
        self->barrier_signal_ptrs_gpu, self->rank, self->world_size,
        self->stream);

    // CRITICAL: combine takes recv_channel_prefix (dispatch OUTPUT,
    // exclusive prefix), not channel_prefix_matrix (notify_dispatch
    // OUTPUT, inclusive prefix). See feedback_deepep_combine_uses_
    // recv_channel_prefix.md.
    deep_ep_intranode_ns::combine(
        CUDA_R_16BF, d_combined_x, d_combined_topk_w, d_x, /*topk_weights=*/d_topk_w,
        /*bias_0=*/nullptr, /*bias_1=*/nullptr, d_recv_src_idx, d_rank_prefix,
        /*channel_prefix_matrix=*/d_recv_channel_prefix, d_send_head,
        /*num_tokens=*/static_cast<int>(p->num_input_tokens),
        /*num_recv_tokens=*/static_cast<int>(p->num_output_tokens),
        static_cast<int>(p->hidden), static_cast<int>(p->num_topk),
        self->buffer_ptrs_gpu, self->rank, self->world_size, self->stream,
        static_cast<int>(p->num_sms),
        static_cast<int>(p->nvl_chunked_send),
        static_cast<int>(p->nvl_chunked_recv));
    if (compute_stream) {
        // Compute stream waits for the combine output ON-DEVICE; no host block.
        cudaEvent_t ev_out;
        cudaEventCreateWithFlags(&ev_out, cudaEventDisableTiming);
        cudaEventRecord(ev_out, self->stream);
        cudaStreamWaitEvent(compute_stream, ev_out, 0);
        cudaEventDestroy(ev_out);
    } else {
        CK(cudaStreamSynchronize(self->stream), "sync after combine");
    }
    return 0;
}

// Forward declaration: the LL teardown helper (defined in the
// ARLE_DEEPEP_HAVE_NVSHMEM block below) frees the nvshmem symmetric
// rdma buffer and finalizes NVSHMEM. Intranode-only builds get a no-op.
static void arle_deepep_ll_teardown(ArleDeepEpBuffer *self);

extern "C" void arle_deepep_buffer_destroy(ArleDeepEpBuffer *self) {
    if (!self) return;
    if (self->low_latency_mode) {
        // LL buffers never open IPC handles; tear down NVSHMEM instead.
        arle_deepep_ll_teardown(self);
        if (self->stream) cudaStreamDestroy(self->stream);
        if (self->workspace) cudaFree(self->workspace);
        delete self;
        return;
    }
    if (self->synced) {
        for (int i = 0; i < self->world_size; ++i) {
            if (i != self->rank && self->peer_buf_ptrs[i]) {
                cudaIpcCloseMemHandle(self->peer_buf_ptrs[i]);
            }
        }
    }
    if (self->stream) cudaStreamDestroy(self->stream);
    if (self->moe_recv_counter_host) cudaFreeHost(self->moe_recv_counter_host);
    if (self->moe_recv_expert_host) cudaFreeHost(self->moe_recv_expert_host);
    if (self->workspace) cudaFree(self->workspace);
    if (self->local_buf) cudaFree(self->local_buf);
    delete self;
}

extern "C" int arle_deepep_buffer_is_low_latency(const ArleDeepEpBuffer *self) {
    return (self && self->low_latency_mode) ? 1 : 0;
}

extern "C" const char *arle_deepep_last_error(void) { return g_last_error; }

// --- NVSHMEM low-latency force-link probe (T4-LL de-risk 2026-06-09) ---
//
// build.rs defines ARLE_DEEPEP_HAVE_NVSHMEM=1 when the internode_ll.cu +
// backend/nvshmem.cu objects are compiled into libarle_deepep.a. We must
// FORCE the final link to pull the NVSHMEM device + host libraries even
// though no ARLE caller invokes the LL path yet — otherwise the linker's
// default --as-needed drops libnvshmem_host and the archive's NVSHMEM
// members are never selected (a trivial "link OK" that proves nothing).
//
// Mechanism: a file-scope volatile function pointer initialised to the
// address of deep_ep::nvshmem::get_unique_id (a real host entry point in
// backend/nvshmem.cu). A static initialiser carrying a symbol relocation
// CANNOT be folded away by the optimiser, so:
//   probe .o  --references-->  deep_ep::nvshmem::get_unique_id
//   backend_nvshmem.o  --references-->  nvshmem_my_pe / nvshmemx_get_uniqueid
//                                       (undefined host syms)
//   => libnvshmem_host.so.3 must resolve them at final link.
// The pointer is read but NEVER called (no nvshmem_init runtime bootstrap
// in this unit). Returns 1 when NVSHMEM is compiled+linked in, else 0.
#if defined(ARLE_DEEPEP_HAVE_NVSHMEM)
#include <optional>

namespace deep_ep {
namespace nvshmem {
// Forward declarations of the real host entry points (defined in
// csrc/kernels/backend/nvshmem.cu). Declared here rather than including
// backend/api.cuh to avoid dragging <nccl.h>/<nccl_device.h> into this TU.
// Signatures MUST match backend/api.cuh exactly (verified 2026-06-09).
std::vector<unsigned char> get_unique_id();
int init(const std::vector<unsigned char> &root_unique_id_val,
         const int &rank,
         const int &num_ranks,
         const int &team_split_stride);
void *alloc(const size_t &size, const size_t &alignment);
void free(void *ptr);
void barrier(const bool &with_cpu_sync,
             const std::optional<cudaStream_t> &stream_opt);
void finalize();
}  // namespace nvshmem
}  // namespace deep_ep

namespace {
// Static initialiser → symbol relocation that survives -O2. `volatile`
// forces the load in the probe to actually happen.
using ArleNvshmemAnchorFn = std::vector<unsigned char> (*)();
volatile ArleNvshmemAnchorFn g_arle_nvshmem_anchor =
    &deep_ep::nvshmem::get_unique_id;
}  // namespace

extern "C" int arle_deepep_nvshmem_built(void) {
    return g_arle_nvshmem_anchor != nullptr ? 1 : 0;
}

// ===================================================================
// NVSHMEM low-latency (internode_ll) implementation.
//
// Mirrors DeepEP csrc/legacy/buffer.hpp::{low_latency_dispatch,
// low_latency_combine}, with the torch::Tensor allocations replaced by
// caller-provided raw device pointers (the moe-forward unit allocates +
// owns them). Buffer sizing + the {dispatch_,combine_}rdma_* handle
// layout come from the SAME deep_ep::legacy::LowLatencyLayout the kernels
// use, so they cannot drift.
//
// NVSHMEM bootstrap: nvshmem::init(uid, rank, world, team_split_stride=
// LEGACY_NUM_MAX_NVL_PEERS) — exactly the low_latency_mode branch of
// buffer.hpp::sync (line 262-265). The uid is the 128-byte blob the
// caller broadcast from rank 0.
// ===================================================================
namespace {

using deep_ep::legacy::LowLatencyBuffer;
using deep_ep::legacy::LowLatencyLayout;

namespace ll_kernels = deep_ep::legacy::internode_ll;

}  // namespace

extern "C" ArleDeepEpStatus arle_deepep_ll_get_uniqueid(
    uint8_t out_uniqueid[ARLE_DEEPEP_LL_UNIQUEID_BYTES]) {
    if (!out_uniqueid) {
        record_error("out_uniqueid is null");
        return -1;
    }
    std::vector<unsigned char> uid = deep_ep::nvshmem::get_unique_id();
    if (uid.size() != ARLE_DEEPEP_LL_UNIQUEID_BYTES) {
        record_error("nvshmem unique id size != 128 bytes");
        return -1;
    }
    std::memcpy(out_uniqueid, uid.data(), ARLE_DEEPEP_LL_UNIQUEID_BYTES);
    return 0;
}

extern "C" ArleDeepEpStatus arle_deepep_buffer_ll_create(
    uint32_t rank,
    uint32_t world_size,
    uint32_t device_id,
    uint32_t num_max_dispatch_tokens_per_rank,
    uint32_t hidden,
    uint32_t num_experts,
    const uint8_t root_uniqueid[ARLE_DEEPEP_LL_UNIQUEID_BYTES],
    ArleDeepEpBuffer **out_handle) {
    if (!out_handle || !root_uniqueid) {
        record_error("null arg");
        return -1;
    }
    *out_handle = nullptr;
    if (world_size < 2 || world_size > kMaxRanks || rank >= world_size) {
        record_error("world_size must be in [2,8] and rank < world_size");
        return -5;
    }
    if (num_experts % world_size != 0) {
        record_error("num_experts must be divisible by world_size");
        return -1;
    }
    // LowLatencyLayout requires hidden % 128 == 0 (num_scales = hidden/128)
    // and the FP8 path additionally requires hidden % 512 == 0; enforce the
    // base constraint here, the FP8 one at dispatch time.
    if (hidden == 0 || hidden % 128 != 0) {
        record_error("hidden must be a positive multiple of 128");
        return -1;
    }

    auto *self = new ArleDeepEpBuffer();
    self->rank = static_cast<int>(rank);
    self->world_size = static_cast<int>(world_size);
    self->low_latency_mode = true;
    self->ll_num_max_dispatch_tokens_per_rank =
        static_cast<int>(num_max_dispatch_tokens_per_rank);
    self->ll_hidden = static_cast<int>(hidden);
    self->ll_num_experts = static_cast<int>(num_experts);

    cudaError_t err;
    // LL uses one device per rank (the rank's own GPU). Pin the REAL device
    // ordinal this rank runs on (INFER_CUDA_DEVICES), NOT the TP rank — on a
    // non-0-based GPU set rank != ordinal (see arle_deepep_buffer_create).
    if ((err = cudaSetDevice(static_cast<int>(device_id))) != cudaSuccess) {
        record_cuda_error("cudaSetDevice", err);
        delete self;
        return -2;
    }
    if ((err = cudaGetDevice(&self->device_id)) != cudaSuccess) {
        record_cuda_error("cudaGetDevice", err);
        delete self;
        return -2;
    }
    cudaDeviceProp prop{};
    if ((err = cudaGetDeviceProperties(&prop, self->device_id)) != cudaSuccess) {
        record_cuda_error("cudaGetDeviceProperties", err);
        delete self;
        return -2;
    }
    self->num_device_sms = prop.multiProcessorCount;

    // 32 MiB workspace (same as intranode + DeepEP Buffer ctor).
    if ((err = cudaMalloc(&self->workspace, NUM_WORKSPACE_BYTES)) != cudaSuccess) {
        record_cuda_error("cudaMalloc workspace", err);
        delete self;
        return -2;
    }
    if ((err = cudaMemset(self->workspace, 0, NUM_WORKSPACE_BYTES)) != cudaSuccess) {
        record_cuda_error("cudaMemset workspace", err);
        cudaFree(self->workspace);
        delete self;
        return -2;
    }
    if ((err = cudaStreamCreate(&self->stream)) != cudaSuccess) {
        record_cuda_error("cudaStreamCreate", err);
        cudaFree(self->workspace);
        delete self;
        return -2;
    }

    // Size the symmetric RDMA buffer from the SAME layout helper the kernels
    // use. get_low_latency_rdma_size_hint already ceils to the 128-byte
    // buffer alignment LEGACY_NUM_BUFFER_ALIGNMENT_BYTES.
    self->num_rdma_bytes = static_cast<int64_t>(
        deep_ep::legacy::get_low_latency_rdma_size_hint(
            static_cast<int>(num_max_dispatch_tokens_per_rank),
            static_cast<int>(hidden), static_cast<int>(world_size),
            static_cast<int>(num_experts)));

    // NVSHMEM bootstrap. team_split_stride = LEGACY_NUM_MAX_NVL_PEERS for
    // low_latency_mode (buffer.hpp::sync line 265). init returns my_pe; it
    // MUST equal rank (single-node UID bootstrap maps pe == rank).
    std::vector<unsigned char> uid(ARLE_DEEPEP_LL_UNIQUEID_BYTES);
    std::memcpy(uid.data(), root_uniqueid, ARLE_DEEPEP_LL_UNIQUEID_BYTES);
    int my_pe = deep_ep::nvshmem::init(uid, self->rank, self->world_size,
                                       LEGACY_NUM_MAX_NVL_PEERS);
    if (my_pe != self->rank) {
        std::snprintf(g_last_error, sizeof(g_last_error),
                      "nvshmem::init returned pe=%d != rank=%d", my_pe,
                      self->rank);
        cudaStreamDestroy(self->stream);
        cudaFree(self->workspace);
        delete self;
        return -2;
    }

    // Allocate the symmetric heap buffer + zero it (DeepEP sync does
    // cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes) — mainly for LL: the
    // first dispatch reads the count/signaling region before any write).
    self->rdma_buffer_ptr = deep_ep::nvshmem::alloc(
        static_cast<size_t>(self->num_rdma_bytes),
        LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);
    if (self->rdma_buffer_ptr == nullptr) {
        record_error("nvshmem::alloc returned null (out of symmetric heap?)");
        deep_ep::nvshmem::finalize();
        cudaStreamDestroy(self->stream);
        cudaFree(self->workspace);
        delete self;
        return -2;
    }
    if ((err = cudaMemset(self->rdma_buffer_ptr, 0,
                          static_cast<size_t>(self->num_rdma_bytes))) !=
        cudaSuccess) {
        record_cuda_error("cudaMemset rdma_buffer", err);
        arle_deepep_ll_teardown(self);
        cudaStreamDestroy(self->stream);
        cudaFree(self->workspace);
        delete self;
        return -2;
    }

    // Cross-rank barrier so every rank's symmetric heap is allocated +
    // zeroed before any dispatch. Mirrors DeepEP sync's nvshmem::barrier.
    deep_ep::nvshmem::barrier(true, std::nullopt);
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        record_cuda_error("ll_create barrier sync", err);
        arle_deepep_ll_teardown(self);
        cudaStreamDestroy(self->stream);
        cudaFree(self->workspace);
        delete self;
        return -2;
    }

    self->synced = true;  // LL needs no IPC sync; ready after barrier.
    *out_handle = self;
    return 0;
}

extern "C" ArleDeepEpStatus arle_deepep_buffer_low_latency_dispatch(
    ArleDeepEpBuffer *self,
    const ArleDeepEpLowLatencyDispatchParams *p) {
    if (!self || !p) {
        record_error("null arg");
        return -1;
    }
    if (!self->low_latency_mode || self->rdma_buffer_ptr == nullptr) {
        record_error("low_latency_dispatch on a non-LL buffer");
        return -4;
    }
    // Pin the buffer's real device so every launch on self->stream runs under
    // the matching context (idempotent; see arle_deepep_buffer_create).
    CK(cudaSetDevice(self->device_id), "cudaSetDevice (ll dispatch)");
    if (static_cast<int>(p->hidden) != self->ll_hidden ||
        static_cast<int>(p->num_experts) != self->ll_num_experts ||
        static_cast<int>(p->num_max_dispatch_tokens_per_rank) !=
            self->ll_num_max_dispatch_tokens_per_rank) {
        record_error("dispatch sizing mismatch with ll_create");
        return -1;
    }
    if (static_cast<int>(p->num_tokens) >
        self->ll_num_max_dispatch_tokens_per_rank) {
        record_error("num_tokens exceeds num_max_dispatch_tokens_per_rank");
        return -1;
    }
    const bool use_fp8 = p->use_fp8 != 0;
    if (use_fp8 && (self->ll_hidden % 512 != 0)) {
        record_error("FP8 dispatch requires hidden % 512 == 0");
        return -1;
    }

    const int num_tokens = static_cast<int>(p->num_tokens);
    const int hidden = self->ll_hidden;
    const int num_topk = static_cast<int>(p->num_topk);
    const int num_experts = self->ll_num_experts;
    const int num_max_tok = self->ll_num_max_dispatch_tokens_per_rank;

    // Recompute the layout + select the active double-buffer, then toggle —
    // identical to buffer.hpp (buffer = buffers[idx]; next = buffers[idx^=1]).
    LowLatencyLayout layout(self->rdma_buffer_ptr, num_max_tok, hidden,
                            self->world_size, num_experts);
    if (layout.total_bytes > static_cast<size_t>(self->num_rdma_bytes)) {
        record_error("LL layout total_bytes exceeds allocated rdma buffer");
        return -1;
    }
    LowLatencyBuffer buffer = layout.buffers[self->low_latency_buffer_idx];
    LowLatencyBuffer next_buffer =
        layout.buffers[self->low_latency_buffer_idx ^= 1];
    auto next_clean_meta = next_buffer.clean_meta();

    // Stream selection: run the LL kernel on the comm stream (self->stream),
    // ordered after the caller's compute stream via an on-device event, so
    // the host isn't blocked. Fall back to host sync when compute_stream==0.
    cudaStream_t compute_stream =
        reinterpret_cast<cudaStream_t>(p->compute_stream);
    cudaStream_t launch_stream = self->stream;
    if (compute_stream) {
        cudaEvent_t ev_in;
        cudaEventCreateWithFlags(&ev_in, cudaEventDisableTiming);
        cudaEventRecord(ev_in, compute_stream);
        cudaStreamWaitEvent(launch_stream, ev_in, 0);
        cudaEventDestroy(ev_in);
    }

    // internode_ll::dispatch — arg order mirrors buffer.hpp::low_latency_
    // dispatch launcher EXACTLY (DeepEP param-order trap discipline). Both
    // SEND and RECV phases in one launch (no recv-hook split here).
    ll_kernels::dispatch(
        reinterpret_cast<void *>(p->d_recv_x),
        reinterpret_cast<void *>(p->d_recv_x_scales),  // null when !use_fp8
        reinterpret_cast<int *>(p->d_recv_src_info),
        reinterpret_cast<int64_t *>(p->d_recv_layout_range),
        reinterpret_cast<int *>(p->d_recv_count),
        /*mask_buffer=*/nullptr,                       // shrink disabled
        /*cumulative_local_expert_recv_stats=*/nullptr,
        /*dispatch_wait_recv_cost_stats=*/nullptr,
        buffer.dispatch_rdma_recv_data_buffer,
        buffer.dispatch_rdma_recv_count_buffer,
        buffer.dispatch_rdma_send_buffer,
        reinterpret_cast<const void *>(p->d_x),
        reinterpret_cast<const int64_t *>(p->d_topk_idx),
        next_clean_meta.first,
        next_clean_meta.second,
        num_tokens,
        hidden,
        num_max_tok,
        num_topk,
        num_experts,
        self->rank,
        self->world_size,
        use_fp8,
        p->round_scale != 0,
        p->use_ue8m0 != 0,
        self->workspace,
        self->num_device_sms,
        launch_stream,
        LEGACY_LOW_LATENCY_SEND_PHASE | LEGACY_LOW_LATENCY_RECV_PHASE);

    if (compute_stream) {
        cudaEvent_t ev_out;
        cudaEventCreateWithFlags(&ev_out, cudaEventDisableTiming);
        cudaEventRecord(ev_out, launch_stream);
        cudaStreamWaitEvent(compute_stream, ev_out, 0);
        cudaEventDestroy(ev_out);
    } else {
        cudaError_t err = cudaStreamSynchronize(launch_stream);
        if (err != cudaSuccess) {
            record_cuda_error("sync after ll dispatch", err);
            return -2;
        }
    }

    // expected_m = ceil(num_tokens * world * num_topk / num_experts) — the
    // average per-local-expert token count the GroupedGEMM should size for.
    // (DeepEP test_low_latency.py computes this; the masked_m per-expert
    // counts are in d_recv_count.)
    if (p->out_expected_m) {
        const int64_t numer = static_cast<int64_t>(num_tokens) *
                              self->world_size * num_topk;
        *p->out_expected_m =
            static_cast<int32_t>((numer + num_experts - 1) / num_experts);
    }
    return 0;
}

extern "C" ArleDeepEpStatus arle_deepep_buffer_low_latency_combine(
    ArleDeepEpBuffer *self,
    const ArleDeepEpLowLatencyCombineParams *p) {
    if (!self || !p) {
        record_error("null arg");
        return -1;
    }
    if (!self->low_latency_mode || self->rdma_buffer_ptr == nullptr) {
        record_error("low_latency_combine on a non-LL buffer");
        return -4;
    }
    // Pin the buffer's real device so every launch on self->stream runs under
    // the matching context (idempotent; see arle_deepep_buffer_create).
    CK(cudaSetDevice(self->device_id), "cudaSetDevice (ll combine)");
    if (static_cast<int>(p->hidden) != self->ll_hidden ||
        static_cast<int>(p->num_experts) != self->ll_num_experts ||
        static_cast<int>(p->num_max_dispatch_tokens_per_rank) !=
            self->ll_num_max_dispatch_tokens_per_rank) {
        record_error("combine sizing mismatch with ll_create");
        return -1;
    }
    if (static_cast<int>(p->num_combined_tokens) >
        self->ll_num_max_dispatch_tokens_per_rank) {
        record_error(
            "num_combined_tokens exceeds num_max_dispatch_tokens_per_rank");
        return -1;
    }

    const int hidden = self->ll_hidden;
    const int num_topk = static_cast<int>(p->num_topk);
    const int num_experts = self->ll_num_experts;
    const int num_max_tok = self->ll_num_max_dispatch_tokens_per_rank;
    const int num_combined_tokens = static_cast<int>(p->num_combined_tokens);

    LowLatencyLayout layout(self->rdma_buffer_ptr, num_max_tok, hidden,
                            self->world_size, num_experts);
    if (layout.total_bytes > static_cast<size_t>(self->num_rdma_bytes)) {
        record_error("LL layout total_bytes exceeds allocated rdma buffer");
        return -1;
    }
    LowLatencyBuffer buffer = layout.buffers[self->low_latency_buffer_idx];
    LowLatencyBuffer next_buffer =
        layout.buffers[self->low_latency_buffer_idx ^= 1];
    auto next_clean_meta = next_buffer.clean_meta();

    cudaStream_t compute_stream =
        reinterpret_cast<cudaStream_t>(p->compute_stream);
    cudaStream_t launch_stream = self->stream;
    if (compute_stream) {
        cudaEvent_t ev_in;
        cudaEventCreateWithFlags(&ev_in, cudaEventDisableTiming);
        cudaEventRecord(ev_in, compute_stream);
        cudaStreamWaitEvent(launch_stream, ev_in, 0);
        cudaEventDestroy(ev_in);
    }

    // internode_ll::combine — arg order mirrors buffer.hpp::low_latency_
    // combine launcher EXACTLY.
    ll_kernels::combine(
        reinterpret_cast<void *>(p->d_combined_x),
        buffer.combine_rdma_recv_data_buffer,
        buffer.combine_rdma_recv_flag_buffer,
        buffer.combine_rdma_send_buffer,
        reinterpret_cast<const void *>(p->d_x),
        reinterpret_cast<const int64_t *>(p->d_topk_idx),
        reinterpret_cast<const float *>(p->d_topk_weights),
        reinterpret_cast<const int *>(p->d_src_info),
        reinterpret_cast<const int64_t *>(p->d_layout_range),
        /*mask_buffer=*/nullptr,                        // shrink disabled
        /*combine_wait_recv_cost_stats=*/nullptr,
        next_clean_meta.first,
        next_clean_meta.second,
        num_combined_tokens,
        hidden,
        num_max_tok,
        num_topk,
        num_experts,
        self->rank,
        self->world_size,
        p->use_logfmt != 0,
        self->workspace,
        self->num_device_sms,
        launch_stream,
        LEGACY_LOW_LATENCY_SEND_PHASE | LEGACY_LOW_LATENCY_RECV_PHASE,
        p->zero_copy != 0);

    if (compute_stream) {
        cudaEvent_t ev_out;
        cudaEventCreateWithFlags(&ev_out, cudaEventDisableTiming);
        cudaEventRecord(ev_out, launch_stream);
        cudaStreamWaitEvent(compute_stream, ev_out, 0);
        cudaEventDestroy(ev_out);
    } else {
        cudaError_t err = cudaStreamSynchronize(launch_stream);
        if (err != cudaSuccess) {
            record_cuda_error("sync after ll combine", err);
            return -2;
        }
    }
    return 0;
}

// LL teardown: free the symmetric heap buffer + finalize NVSHMEM. Called
// from arle_deepep_buffer_destroy. Idempotent on the rdma buffer.
static void arle_deepep_ll_teardown(ArleDeepEpBuffer *self) {
    if (!self || !self->low_latency_mode) return;
    cudaDeviceSynchronize();
    if (self->rdma_buffer_ptr) {
        deep_ep::nvshmem::free(self->rdma_buffer_ptr);
        self->rdma_buffer_ptr = nullptr;
    }
    deep_ep::nvshmem::finalize();
}

#else  // !ARLE_DEEPEP_HAVE_NVSHMEM

extern "C" int arle_deepep_nvshmem_built(void) { return 0; }

// Stub the LL surface so the header's extern "C" decls always resolve even
// in an intranode-only (no-NVSHMEM) archive. They report "not built".
extern "C" ArleDeepEpStatus arle_deepep_ll_get_uniqueid(
    uint8_t out_uniqueid[ARLE_DEEPEP_LL_UNIQUEID_BYTES]) {
    (void)out_uniqueid;
    record_error("deepep built without NVSHMEM — LL path unavailable");
    return -1;
}
extern "C" ArleDeepEpStatus arle_deepep_buffer_ll_create(
    uint32_t rank, uint32_t world_size, uint32_t device_id,
    uint32_t num_max_dispatch_tokens_per_rank, uint32_t hidden,
    uint32_t num_experts,
    const uint8_t root_uniqueid[ARLE_DEEPEP_LL_UNIQUEID_BYTES],
    ArleDeepEpBuffer **out_handle) {
    (void)rank; (void)world_size; (void)device_id;
    (void)num_max_dispatch_tokens_per_rank;
    (void)hidden; (void)num_experts; (void)root_uniqueid;
    if (out_handle) *out_handle = nullptr;
    record_error("deepep built without NVSHMEM — LL path unavailable");
    return -1;
}
extern "C" ArleDeepEpStatus arle_deepep_buffer_low_latency_dispatch(
    ArleDeepEpBuffer *handle,
    const ArleDeepEpLowLatencyDispatchParams *params) {
    (void)handle; (void)params;
    record_error("deepep built without NVSHMEM — LL path unavailable");
    return -1;
}
extern "C" ArleDeepEpStatus arle_deepep_buffer_low_latency_combine(
    ArleDeepEpBuffer *handle,
    const ArleDeepEpLowLatencyCombineParams *params) {
    (void)handle; (void)params;
    record_error("deepep built without NVSHMEM — LL path unavailable");
    return -1;
}

// No-op teardown referenced by arle_deepep_buffer_destroy.
static void arle_deepep_ll_teardown(ArleDeepEpBuffer *self) { (void)self; }

#endif

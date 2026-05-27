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

// Header layout differs between the old flat DeepEP tree and the new
// "csrc/kernels/legacy/" refactor. build.rs defines ARLE_DEEPEP_LEGACY_
// LAYOUT=1 when api.cuh sits under legacy/ — pick the right include +
// shim the renamed constants (NUM_MAX_NVL_PEERS → LEGACY_NUM_MAX_NVL_PEERS).
#ifdef ARLE_DEEPEP_LEGACY_LAYOUT
#include "kernels/legacy/api.cuh"
#include "kernels/legacy/compiled.cuh"
#ifndef NUM_MAX_NVL_PEERS
#define NUM_MAX_NVL_PEERS LEGACY_NUM_MAX_NVL_PEERS
#endif
#ifndef NUM_MAX_LOCAL_EXPERTS
#define NUM_MAX_LOCAL_EXPERTS LEGACY_NUM_MAX_LOCAL_EXPERTS
#endif
#ifndef NUM_WORKSPACE_BYTES
#define NUM_WORKSPACE_BYTES LEGACY_NUM_WORKSPACE_BYTES
#endif
// Legacy refactor put intranode/layout under deep_ep::legacy::*.
namespace deep_ep_intranode_ns = deep_ep::legacy::intranode;
namespace deep_ep_layout_ns = deep_ep::legacy::layout;
#else
#include "kernels/api.cuh"
#include "kernels/configs.cuh"
namespace deep_ep_intranode_ns = deep_ep::intranode;
namespace deep_ep_layout_ns = deep_ep::layout;
#endif

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
    uint32_t rank, uint32_t world_size, ArleDeepEpBuffer **out_handle) {
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

    auto *self = new ArleDeepEpBuffer();
    self->rank = static_cast<int>(rank);
    self->world_size = static_cast<int>(world_size);

    cudaError_t err;
    err = cudaSetDevice(self->rank);
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
    if ((err = cudaMalloc(&self->workspace, NUM_WORKSPACE_BYTES)) != cudaSuccess) {
        record_cuda_error("cudaMalloc workspace", err);
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
    if (!self || !p) {
        record_error("null arg");
        return -1;
    }
    if (!self->synced) {
        record_error("dispatch before sync");
        return -4;
    }
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
    CK(cudaStreamSynchronize(self->stream), "sync after dispatch");

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
    CK(cudaStreamSynchronize(self->stream), "sync after combine");
    return 0;
}

extern "C" void arle_deepep_buffer_destroy(ArleDeepEpBuffer *self) {
    if (!self) return;
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

extern "C" const char *arle_deepep_last_error(void) { return g_last_error; }

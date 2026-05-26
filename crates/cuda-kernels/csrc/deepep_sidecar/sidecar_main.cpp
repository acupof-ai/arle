// SPDX-License-Identifier: Apache-2.0
// ARLE native DeepEP sidecar — production binary (phase 1.1).
//
// Long-running C++ process forked one-per-rank by ARLE's Rust host. Boot
// sequence + IPC handshake match the phase 1.0a-iii spike validated on
// 8 × H20 (see `docs/experience/wins/2026-05-26-dsv4-deepep-cpp-full-
// dispatch-combine.md`). Production differences:
//
//   - Command loop over fixed fds (kChildP2cFd=10, kChildC2pFd=11) instead
//     of single-shot dispatch+combine.
//   - Structured wire protocol (protocol.hpp).
//   - Clean shutdown frees all CUDA resources before exit.
//   - SIGTERM is honored (parent uses it on supervised teardown).
//
// Build: nvcc -DDISABLE_NVSHMEM --expt-relaxed-constexpr
//           --expt-extended-lambda -gencode arch=compute_90,code=sm_90
//           -I$DEEPEP/csrc intranode.cu layout.cu runtime.cu
//           sidecar_main.cpp -lcudart
// See `crates/cuda-kernels/build.rs` for how the host build wires this.

#include "protocol.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels/api.cuh"
#include "kernels/configs.cuh"

namespace {

using arle::deepep_sidecar::BootRequest;
using arle::deepep_sidecar::BootResponse;
using arle::deepep_sidecar::CommandId;
using arle::deepep_sidecar::MessageHeader;
using arle::deepep_sidecar::RoundTripRequest;
using arle::deepep_sidecar::RoundTripResponse;
using arle::deepep_sidecar::Status;
using arle::deepep_sidecar::kChildC2pFd;
using arle::deepep_sidecar::kChildP2cFd;
using arle::deepep_sidecar::kMaxNvlPeers;
using arle::deepep_sidecar::kProtocolVersion;

// ─── SHA-256 (no openssl) ────────────────────────────────────────────────

constexpr std::uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static inline std::uint32_t rotr32(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

struct Sha256Ctx {
    std::uint32_t h[8];
    std::uint8_t buf[64];
    std::uint64_t len_bits;
    int buf_len;
};

void sha256_init(Sha256Ctx& s) {
    static const std::uint32_t kInit[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    for (int i = 0; i < 8; ++i) s.h[i] = kInit[i];
    s.len_bits = 0;
    s.buf_len = 0;
}

void sha256_block(Sha256Ctx& s, const std::uint8_t* p) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (std::uint32_t(p[i * 4]) << 24) | (std::uint32_t(p[i * 4 + 1]) << 16) |
               (std::uint32_t(p[i * 4 + 2]) << 8) | std::uint32_t(p[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        std::uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        std::uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = s.h[0], b = s.h[1], c = s.h[2], d = s.h[3];
    std::uint32_t e = s.h[4], f = s.h[5], g = s.h[6], h = s.h[7];
    for (int i = 0; i < 64; ++i) {
        std::uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        std::uint32_t ch = (e & f) ^ ((~e) & g);
        std::uint32_t t1 = h + s1 + ch + kSha256K[i] + w[i];
        std::uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        std::uint32_t t2 = s0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s.h[0] += a; s.h[1] += b; s.h[2] += c; s.h[3] += d;
    s.h[4] += e; s.h[5] += f; s.h[6] += g; s.h[7] += h;
}

void sha256_update(Sha256Ctx& s, const void* data, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    s.len_bits += static_cast<std::uint64_t>(n) * 8;
    while (n) {
        std::size_t take = 64 - s.buf_len;
        if (take > n) take = n;
        std::memcpy(s.buf + s.buf_len, p, take);
        s.buf_len += static_cast<int>(take);
        p += take;
        n -= take;
        if (s.buf_len == 64) {
            sha256_block(s, s.buf);
            s.buf_len = 0;
        }
    }
}

void sha256_final(Sha256Ctx& s, std::uint8_t out[32]) {
    s.buf[s.buf_len++] = 0x80;
    if (s.buf_len > 56) {
        while (s.buf_len < 64) s.buf[s.buf_len++] = 0;
        sha256_block(s, s.buf);
        s.buf_len = 0;
    }
    while (s.buf_len < 56) s.buf[s.buf_len++] = 0;
    for (int i = 7; i >= 0; --i) s.buf[s.buf_len++] = static_cast<std::uint8_t>(s.len_bits >> (i * 8));
    sha256_block(s, s.buf);
    for (int i = 0; i < 8; ++i) {
        out[i * 4]     = static_cast<std::uint8_t>(s.h[i] >> 24);
        out[i * 4 + 1] = static_cast<std::uint8_t>(s.h[i] >> 16);
        out[i * 4 + 2] = static_cast<std::uint8_t>(s.h[i] >> 8);
        out[i * 4 + 3] = static_cast<std::uint8_t>(s.h[i]);
    }
}

// ─── pipe IO helpers ─────────────────────────────────────────────────────

ssize_t read_all(int fd, void* buf, std::size_t n) {
    auto* p = static_cast<std::uint8_t*>(buf);
    std::size_t got = 0;
    while (got < n) {
        ssize_t r = ::read(fd, p + got, n - got);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) return static_cast<ssize_t>(got);
        got += static_cast<std::size_t>(r);
    }
    return static_cast<ssize_t>(got);
}

ssize_t write_all(int fd, const void* buf, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(buf);
    std::size_t put = 0;
    while (put < n) {
        ssize_t w = ::write(fd, p + put, n - put);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        put += static_cast<std::size_t>(w);
    }
    return static_cast<ssize_t>(put);
}

bool send_response(int fd, Status status, const void* payload, std::size_t bytes) {
    MessageHeader hdr{};
    hdr.cmd_or_status = static_cast<std::uint32_t>(status);
    hdr.payload_bytes = static_cast<std::uint32_t>(bytes);
    if (write_all(fd, &hdr, sizeof(hdr)) != static_cast<ssize_t>(sizeof(hdr))) return false;
    if (bytes && write_all(fd, payload, bytes) != static_cast<ssize_t>(bytes)) return false;
    return true;
}

#define CK_CUDA(call, ctx)                                                                                    \
    do {                                                                                                      \
        cudaError_t err = (call);                                                                             \
        if (err != cudaSuccess) {                                                                             \
            std::fprintf(stderr, "[arle-deepep-sidecar] CUDA error at %s:%d %s: %s\n",                        \
                         __FILE__, __LINE__, ctx, cudaGetErrorString(err));                                   \
            std::fflush(stderr);                                                                              \
            return Status::kCudaError;                                                                        \
        }                                                                                                     \
    } while (0)

// ─── Sidecar state ───────────────────────────────────────────────────────

// Mirror the phase 1.0a-iii Buffer ctor layout exactly.
constexpr std::int64_t kNvlBytes = 512LL << 20;  // 512 MiB
constexpr std::int64_t kBarrierSignalBytes = kMaxNvlPeers * sizeof(int);
constexpr std::int64_t kBufferPtrBytes = kMaxNvlPeers * sizeof(void*);
constexpr std::int64_t kBarrierSignalPtrBytes = kMaxNvlPeers * sizeof(int*);
constexpr std::int64_t kTotalBytes =
    kNvlBytes + kBarrierSignalBytes + kBufferPtrBytes + kBarrierSignalPtrBytes;

struct SidecarState {
    int rank = -1;
    int world_size = 0;
    int device_id = -1;

    void* local_buf = nullptr;
    void* workspace = nullptr;

    int* moe_recv_counter_host = nullptr;
    int* moe_recv_counter_dev = nullptr;
    int* moe_recv_expert_host = nullptr;
    int* moe_recv_expert_dev = nullptr;

    void* peer_buf_ptrs[kMaxNvlPeers] = {};
    int* peer_barrier_signal_ptrs[kMaxNvlPeers] = {};

    void** buffer_ptrs_gpu = nullptr;
    int** barrier_signal_ptrs_gpu = nullptr;

    cudaStream_t stream{};
    bool booted = false;
    bool synced = false;
};

Status sidecar_boot(SidecarState& s, const BootRequest& req, BootResponse& resp) {
    if (req.protocol_version != kProtocolVersion) {
        std::fprintf(stderr, "[arle-deepep-sidecar] protocol mismatch host=%u sidecar=%u\n",
                     req.protocol_version, kProtocolVersion);
        return Status::kProtocolMismatch;
    }
    if (req.world_size <= 0 || req.world_size > kMaxNvlPeers || req.rank >= req.world_size) {
        return Status::kBadArgs;
    }

    s.rank = static_cast<int>(req.rank);
    s.world_size = static_cast<int>(req.world_size);

    int device_count = 0;
    CK_CUDA(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < s.world_size) return Status::kBadArgs;

    CK_CUDA(cudaSetDevice(s.rank), "cudaSetDevice");
    CK_CUDA(cudaGetDevice(&s.device_id), "cudaGetDevice");

    CK_CUDA(cudaMalloc(&s.local_buf, kTotalBytes), "cudaMalloc local_buf");
    CK_CUDA(cudaMalloc(&s.workspace, NUM_WORKSPACE_BYTES), "cudaMalloc workspace");
    CK_CUDA(cudaHostAlloc(&s.moe_recv_counter_host, sizeof(int), cudaHostAllocMapped),
            "cudaHostAlloc moe_recv_counter");
    CK_CUDA(cudaHostGetDevicePointer(&s.moe_recv_counter_dev, s.moe_recv_counter_host, 0),
            "cudaHostGetDevicePointer moe_recv_counter");
    CK_CUDA(cudaHostAlloc(&s.moe_recv_expert_host, sizeof(int) * NUM_MAX_LOCAL_EXPERTS,
                          cudaHostAllocMapped),
            "cudaHostAlloc moe_recv_expert");
    CK_CUDA(cudaHostGetDevicePointer(&s.moe_recv_expert_dev, s.moe_recv_expert_host, 0),
            "cudaHostGetDevicePointer moe_recv_expert");

    CK_CUDA(cudaStreamCreate(&s.stream), "cudaStreamCreate");
    CK_CUDA(cudaMemsetAsync(s.local_buf, 0, kTotalBytes, s.stream), "memset local_buf");
    CK_CUDA(cudaMemsetAsync(s.workspace, 0, NUM_WORKSPACE_BYTES, s.stream), "memset workspace");
    CK_CUDA(cudaStreamSynchronize(s.stream), "sync after init memset");

    cudaIpcMemHandle_t handle{};
    CK_CUDA(cudaIpcGetMemHandle(&handle, s.local_buf), "cudaIpcGetMemHandle");

    resp = {};
    resp.device_id = static_cast<std::uint32_t>(s.device_id);
    static_assert(sizeof(resp.ipc_handle) == sizeof(handle.reserved),
                  "IPC handle wire size must match cudaIpcMemHandle_t reserved field");
    std::memcpy(resp.ipc_handle, handle.reserved, sizeof(handle.reserved));

    s.booted = true;
    return Status::kOk;
}

// Sync payload mirrors the BootResponse array: kMaxNvlPeers tuples.
struct alignas(8) PeerEntry {
    std::uint32_t device_id;
    std::uint32_t reserved;
    std::uint8_t ipc_handle[64];
};
static_assert(sizeof(PeerEntry) == sizeof(BootResponse),
              "PeerEntry must match BootResponse on the wire");

Status sidecar_sync(SidecarState& s, const PeerEntry* peers) {
    if (!s.booted) return Status::kBadArgs;

    for (int i = 0; i < s.world_size; ++i) {
        if (i == s.rank) {
            s.peer_buf_ptrs[i] = s.local_buf;
        } else {
            cudaIpcMemHandle_t handle{};
            std::memcpy(handle.reserved, peers[i].ipc_handle, sizeof(handle.reserved));
            void* p = nullptr;
            CK_CUDA(cudaIpcOpenMemHandle(&p, handle, cudaIpcMemLazyEnablePeerAccess),
                    "cudaIpcOpenMemHandle peer");
            s.peer_buf_ptrs[i] = p;
        }
        s.peer_barrier_signal_ptrs[i] = reinterpret_cast<int*>(
            static_cast<std::uint8_t*>(s.peer_buf_ptrs[i]) + kNvlBytes);
    }

    s.buffer_ptrs_gpu = reinterpret_cast<void**>(
        static_cast<std::uint8_t*>(s.local_buf) + kNvlBytes + kBarrierSignalBytes);
    s.barrier_signal_ptrs_gpu = reinterpret_cast<int**>(
        static_cast<std::uint8_t*>(s.local_buf) + kNvlBytes + kBarrierSignalBytes + kBufferPtrBytes);

    CK_CUDA(cudaMemcpyAsync(s.buffer_ptrs_gpu, s.peer_buf_ptrs, sizeof(s.peer_buf_ptrs),
                            cudaMemcpyHostToDevice, s.stream),
            "memcpyAsync buffer_ptrs_gpu");
    CK_CUDA(cudaMemcpyAsync(s.barrier_signal_ptrs_gpu, s.peer_barrier_signal_ptrs,
                            sizeof(s.peer_barrier_signal_ptrs), cudaMemcpyHostToDevice, s.stream),
            "memcpyAsync barrier_signal_ptrs_gpu");
    CK_CUDA(cudaStreamSynchronize(s.stream), "sync after peer pointer upload");

    deep_ep::intranode::barrier(s.barrier_signal_ptrs_gpu, s.rank, s.world_size, s.stream);
    CK_CUDA(cudaStreamSynchronize(s.stream), "sync after intranode::barrier");

    s.synced = true;
    return Status::kOk;
}

Status sidecar_round_trip(SidecarState& s, const RoundTripRequest& req, RoundTripResponse& resp) {
    if (!s.synced) return Status::kBadArgs;
    if (req.num_tokens == 0 || req.hidden == 0 || req.num_topk == 0 ||
        req.num_experts == 0 || req.num_sms == 0 || req.num_sms % 2 != 0) {
        return Status::kBadArgs;
    }
    if (req.num_topk > static_cast<std::uint32_t>(s.world_size)) return Status::kBadArgs;
    if (req.num_experts % static_cast<std::uint32_t>(s.world_size) != 0) return Status::kBadArgs;
    if ((req.hidden * sizeof(__nv_bfloat16)) % sizeof(int4) != 0) return Status::kBadArgs;

    const int num_tokens = static_cast<int>(req.num_tokens);
    const int hidden = static_cast<int>(req.hidden);
    const int num_topk = static_cast<int>(req.num_topk);
    const int num_experts = static_cast<int>(req.num_experts);
    const int num_sms = static_cast<int>(req.num_sms);
    const int num_channels = num_sms / 2;
    const int nvl_chunked_send = static_cast<int>(req.nvl_chunked_send);
    const int nvl_chunked_recv = static_cast<int>(req.nvl_chunked_recv);
    const int experts_per_rank = num_experts / s.world_size;
    const int hidden_int4 = hidden * sizeof(__nv_bfloat16) / sizeof(int4);

    // ─── synthesize input (rank-tagged, hidden-position-tagged, symmetric routing)
    std::vector<__nv_bfloat16> host_x(num_tokens * hidden);
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < hidden; ++j) {
            float v = static_cast<float>(s.rank) + static_cast<float>(j) * 1e-4f;
            host_x[i * hidden + j] = __float2bfloat16(v);
        }
    }
    std::vector<std::int64_t> host_topk(num_tokens * num_topk);
    std::vector<float> host_topk_w(num_tokens * num_topk);
    for (int i = 0; i < num_tokens; ++i) {
        for (int k = 0; k < num_topk; ++k) {
            int target_rank = (s.rank + k) % s.world_size;
            host_topk[i * num_topk + k] = static_cast<std::int64_t>(target_rank * experts_per_rank + k);
            host_topk_w[i * num_topk + k] = 1.0f / static_cast<float>(num_topk);
        }
    }

    // ─── GPU buffers
    __nv_bfloat16* d_x = nullptr;
    std::int64_t* d_topk = nullptr;
    float* d_topk_w = nullptr;
    CK_CUDA(cudaMalloc(&d_x, sizeof(__nv_bfloat16) * host_x.size()), "alloc d_x");
    CK_CUDA(cudaMalloc(&d_topk, sizeof(std::int64_t) * host_topk.size()), "alloc d_topk");
    CK_CUDA(cudaMalloc(&d_topk_w, sizeof(float) * host_topk_w.size()), "alloc d_topk_w");
    CK_CUDA(cudaMemcpyAsync(d_x, host_x.data(), sizeof(__nv_bfloat16) * host_x.size(),
                            cudaMemcpyHostToDevice, s.stream), "h2d d_x");
    CK_CUDA(cudaMemcpyAsync(d_topk, host_topk.data(), sizeof(std::int64_t) * host_topk.size(),
                            cudaMemcpyHostToDevice, s.stream), "h2d d_topk");
    CK_CUDA(cudaMemcpyAsync(d_topk_w, host_topk_w.data(), sizeof(float) * host_topk_w.size(),
                            cudaMemcpyHostToDevice, s.stream), "h2d d_topk_w");

    int* d_num_tokens_per_rank = nullptr;
    int* d_num_tokens_per_expert = nullptr;
    bool* d_is_token_in_rank = nullptr;
    CK_CUDA(cudaMalloc(&d_num_tokens_per_rank, sizeof(int) * s.world_size), "alloc num_tokens_per_rank");
    CK_CUDA(cudaMalloc(&d_num_tokens_per_expert, sizeof(int) * num_experts), "alloc num_tokens_per_expert");
    CK_CUDA(cudaMalloc(&d_is_token_in_rank, sizeof(bool) * num_tokens * s.world_size),
            "alloc is_token_in_rank");

    deep_ep::layout::get_dispatch_layout(
        d_topk, d_num_tokens_per_rank, /*num_tokens_per_rdma_rank=*/nullptr,
        d_num_tokens_per_expert, d_is_token_in_rank, num_tokens, num_topk,
        s.world_size, num_experts, s.stream);

    int* d_rank_prefix_matrix = nullptr;
    int* d_channel_prefix_matrix = nullptr;
    CK_CUDA(cudaMalloc(&d_rank_prefix_matrix, sizeof(int) * s.world_size * s.world_size),
            "alloc rank_prefix_matrix");
    CK_CUDA(cudaMalloc(&d_channel_prefix_matrix, sizeof(int) * s.world_size * num_channels),
            "alloc channel_prefix_matrix");
    *s.moe_recv_counter_host = -1;
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i) s.moe_recv_expert_host[i] = -1;
    int num_memset_int = num_channels * s.world_size * 4;

    deep_ep::intranode::notify_dispatch(
        d_num_tokens_per_rank, s.moe_recv_counter_dev, s.world_size,
        d_num_tokens_per_expert, s.moe_recv_expert_dev, num_experts,
        num_tokens, d_is_token_in_rank, d_channel_prefix_matrix,
        d_rank_prefix_matrix, num_memset_int, /*expert_alignment=*/1,
        s.buffer_ptrs_gpu, s.barrier_signal_ptrs_gpu, s.rank, s.stream, num_channels);

    // Host-poll for notify completion. Timeout 15 s mirrors phase 1.0a-iii.
    int num_recv_tokens = -1;
    auto t_poll0 = std::chrono::steady_clock::now();
    while (true) {
        num_recv_tokens = *s.moe_recv_counter_host;
        bool ready = (num_recv_tokens >= 0);
        for (int i = 0; i < experts_per_rank && ready; ++i)
            ready = ready && (s.moe_recv_expert_host[i] >= 0);
        if (ready) break;
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t_poll0).count() > 15) {
            std::fprintf(stderr, "[arle-deepep-sidecar] rank %d notify_dispatch timeout\n", s.rank);
            return Status::kKernelTimeout;
        }
    }

    __nv_bfloat16* d_recv_x = nullptr;
    int* d_recv_src_idx = nullptr;
    std::int64_t* d_recv_topk_idx = nullptr;
    float* d_recv_topk_w = nullptr;
    int* d_recv_channel_prefix = nullptr;
    int* d_send_head = nullptr;
    CK_CUDA(cudaMalloc(&d_recv_x, sizeof(__nv_bfloat16) * num_recv_tokens * hidden), "alloc recv_x");
    CK_CUDA(cudaMalloc(&d_recv_src_idx, sizeof(int) * num_recv_tokens), "alloc recv_src_idx");
    CK_CUDA(cudaMalloc(&d_recv_topk_idx, sizeof(std::int64_t) * num_recv_tokens * num_topk),
            "alloc recv_topk_idx");
    CK_CUDA(cudaMalloc(&d_recv_topk_w, sizeof(float) * num_recv_tokens * num_topk),
            "alloc recv_topk_w");
    CK_CUDA(cudaMalloc(&d_recv_channel_prefix, sizeof(int) * s.world_size * num_channels),
            "alloc recv_channel_prefix");
    CK_CUDA(cudaMalloc(&d_send_head, sizeof(int) * num_tokens * s.world_size), "alloc send_head");

    deep_ep::intranode::dispatch(
        d_recv_x, /*recv_x_scales=*/nullptr, d_recv_src_idx, d_recv_topk_idx,
        d_recv_topk_w, d_recv_channel_prefix, d_send_head,
        d_x, /*x_scales=*/nullptr, d_topk, d_topk_w,
        d_is_token_in_rank, d_channel_prefix_matrix,
        num_tokens, /*num_worst_tokens=*/0, hidden_int4, num_topk, num_experts,
        /*num_scales=*/0, /*scale_token_stride=*/0, /*scale_hidden_stride=*/0,
        s.buffer_ptrs_gpu, s.rank, s.world_size, s.stream, num_sms,
        nvl_chunked_send, nvl_chunked_recv);
    CK_CUDA(cudaStreamSynchronize(s.stream), "sync after dispatch");

    // Identity expert step — combine on recv_x directly.
    __nv_bfloat16* d_combined_x = nullptr;
    float* d_combined_topk_weights = nullptr;
    CK_CUDA(cudaMalloc(&d_combined_x, sizeof(__nv_bfloat16) * num_tokens * hidden),
            "alloc combined_x");
    CK_CUDA(cudaMalloc(&d_combined_topk_weights, sizeof(float) * num_tokens * num_topk),
            "alloc combined_topk_w");

    deep_ep::intranode::cached_notify_combine(
        s.buffer_ptrs_gpu, d_send_head, num_channels, num_tokens,
        num_channels * s.world_size * 2,
        s.barrier_signal_ptrs_gpu, s.rank, s.world_size, s.stream);

    // CRITICAL — pass recv_channel_prefix (exclusive prefix) NOT
    // channel_prefix_matrix (inclusive prefix). Phase 1.0a-iv root cause.
    deep_ep::intranode::combine(
        CUDA_R_16BF, d_combined_x, d_combined_topk_weights,
        d_recv_x, /*topk_weights=*/d_recv_topk_w,
        /*bias_0=*/nullptr, /*bias_1=*/nullptr,
        d_recv_src_idx, d_rank_prefix_matrix, /*channel_prefix_matrix=*/d_recv_channel_prefix,
        d_send_head,
        /*num_tokens=*/num_recv_tokens, /*num_recv_tokens=*/num_tokens,
        hidden, num_topk,
        s.buffer_ptrs_gpu, s.rank, s.world_size, s.stream, num_sms,
        nvl_chunked_send, nvl_chunked_recv);
    CK_CUDA(cudaStreamSynchronize(s.stream), "sync after combine");

    // SHA-256 of combined output + first 8 preview values.
    std::vector<__nv_bfloat16> host_combined(num_tokens * hidden);
    CK_CUDA(cudaMemcpy(host_combined.data(), d_combined_x,
                       sizeof(__nv_bfloat16) * host_combined.size(),
                       cudaMemcpyDeviceToHost), "d2h combined_x");

    Sha256Ctx sha{};
    sha256_init(sha);
    sha256_update(sha, host_combined.data(), sizeof(__nv_bfloat16) * host_combined.size());
    resp = {};
    resp.num_recv_tokens = static_cast<std::uint32_t>(num_recv_tokens);
    sha256_final(sha, resp.sha256);
    for (int i = 0; i < 8; ++i) {
        resp.preview[i] = (i < num_tokens * hidden) ? __bfloat162float(host_combined[i]) : 0.0f;
    }

    // Cleanup the per-call scratch (NVL buffer / pinned counters / pointer
    // arrays persist across calls).
    cudaFree(d_x); cudaFree(d_topk); cudaFree(d_topk_w);
    cudaFree(d_num_tokens_per_rank); cudaFree(d_num_tokens_per_expert); cudaFree(d_is_token_in_rank);
    cudaFree(d_rank_prefix_matrix); cudaFree(d_channel_prefix_matrix);
    cudaFree(d_recv_x); cudaFree(d_recv_src_idx); cudaFree(d_recv_topk_idx);
    cudaFree(d_recv_topk_w); cudaFree(d_recv_channel_prefix); cudaFree(d_send_head);
    cudaFree(d_combined_x); cudaFree(d_combined_topk_weights);

    return Status::kOk;
}

void sidecar_teardown(SidecarState& s) {
    if (s.synced) {
        for (int i = 0; i < s.world_size; ++i) {
            if (i != s.rank && s.peer_buf_ptrs[i]) {
                cudaIpcCloseMemHandle(s.peer_buf_ptrs[i]);
                s.peer_buf_ptrs[i] = nullptr;
            }
        }
    }
    if (s.stream) cudaStreamDestroy(s.stream);
    if (s.moe_recv_counter_host) cudaFreeHost(s.moe_recv_counter_host);
    if (s.moe_recv_expert_host) cudaFreeHost(s.moe_recv_expert_host);
    if (s.workspace) cudaFree(s.workspace);
    if (s.local_buf) cudaFree(s.local_buf);
    s = {};
}

volatile std::sig_atomic_t g_sigterm = 0;
void on_sigterm(int) { g_sigterm = 1; }

}  // anonymous namespace

int main(int /*argc*/, char** /*argv*/) {
    // Ignore SIGPIPE so a parent disconnect surfaces as read/write EOF rather
    // than killing the sidecar mid-flight.
    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGTERM, on_sigterm);

    SidecarState state{};

    while (!g_sigterm) {
        MessageHeader hdr{};
        ssize_t n = read_all(kChildP2cFd, &hdr, sizeof(hdr));
        if (n == 0) {
            // Parent closed the pipe — treat as clean shutdown.
            sidecar_teardown(state);
            return 0;
        }
        if (n < 0 || n != static_cast<ssize_t>(sizeof(hdr))) {
            std::fprintf(stderr, "[arle-deepep-sidecar] short header read (%zd)\n", n);
            sidecar_teardown(state);
            return 1;
        }

        std::vector<std::uint8_t> payload(hdr.payload_bytes);
        if (hdr.payload_bytes &&
            read_all(kChildP2cFd, payload.data(), hdr.payload_bytes) !=
                static_cast<ssize_t>(hdr.payload_bytes)) {
            std::fprintf(stderr, "[arle-deepep-sidecar] short payload read\n");
            sidecar_teardown(state);
            return 1;
        }

        const auto cmd = static_cast<CommandId>(hdr.cmd_or_status);
        Status status = Status::kInternal;

        switch (cmd) {
        case CommandId::kBoot: {
            if (hdr.payload_bytes != sizeof(BootRequest)) {
                status = Status::kBadArgs;
                send_response(kChildC2pFd, status, nullptr, 0);
                break;
            }
            BootRequest req{};
            std::memcpy(&req, payload.data(), sizeof(req));
            BootResponse resp{};
            status = sidecar_boot(state, req, resp);
            send_response(kChildC2pFd, status,
                          status == Status::kOk ? static_cast<const void*>(&resp) : nullptr,
                          status == Status::kOk ? sizeof(resp) : 0);
            break;
        }
        case CommandId::kSync: {
            const std::size_t expected = sizeof(PeerEntry) * kMaxNvlPeers;
            if (hdr.payload_bytes != expected) {
                status = Status::kBadArgs;
                send_response(kChildC2pFd, status, nullptr, 0);
                break;
            }
            status = sidecar_sync(state, reinterpret_cast<const PeerEntry*>(payload.data()));
            send_response(kChildC2pFd, status, nullptr, 0);
            break;
        }
        case CommandId::kRoundTrip: {
            if (hdr.payload_bytes != sizeof(RoundTripRequest)) {
                status = Status::kBadArgs;
                send_response(kChildC2pFd, status, nullptr, 0);
                break;
            }
            RoundTripRequest req{};
            std::memcpy(&req, payload.data(), sizeof(req));
            RoundTripResponse resp{};
            status = sidecar_round_trip(state, req, resp);
            send_response(kChildC2pFd, status,
                          status == Status::kOk ? static_cast<const void*>(&resp) : nullptr,
                          status == Status::kOk ? sizeof(resp) : 0);
            break;
        }
        case CommandId::kShutdown: {
            send_response(kChildC2pFd, Status::kOk, nullptr, 0);
            sidecar_teardown(state);
            return 0;
        }
        default: {
            std::fprintf(stderr, "[arle-deepep-sidecar] unknown command 0x%x\n",
                         hdr.cmd_or_status);
            send_response(kChildC2pFd, Status::kBadArgs, nullptr, 0);
            break;
        }
        }
    }

    sidecar_teardown(state);
    return 0;
}

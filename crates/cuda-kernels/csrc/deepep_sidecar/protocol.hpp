// SPDX-License-Identifier: Apache-2.0
// ARLE native DeepEP sidecar — wire protocol shared with the Rust host.
//
// All multi-byte integers are little-endian. Commands and responses use the
// same 8-byte header layout. Payloads are command-specific.
//
// Versioning: a single u32 PROTOCOL_VERSION lives in the boot exchange and
// must match host/sidecar exactly. Bump on any breaking change.

#ifndef ARLE_DEEPEP_SIDECAR_PROTOCOL_HPP_
#define ARLE_DEEPEP_SIDECAR_PROTOCOL_HPP_

#include <cstdint>

namespace arle::deepep_sidecar {

constexpr std::uint32_t kProtocolVersion = 1;

// File descriptor numbers the host pins for the child (per phase 1.0a-iii
// validated layout — see `phase1a_iii_spike.cpp` parent fork sequence).
constexpr int kChildP2cFd = 10;  // host → sidecar (commands)
constexpr int kChildC2pFd = 11;  // sidecar → host (responses)

// Maximum number of intra-node ranks supported (matches DeepEP NUM_MAX_NVL_PEERS).
constexpr int kMaxNvlPeers = 8;

// Sidecar status codes (returned in every response header).
enum class Status : std::uint32_t {
    kOk = 0,
    kProtocolMismatch = 1,
    kCudaError = 2,
    kKernelTimeout = 3,
    kBadArgs = 4,
    kInternal = 5,
};

// Command identifiers. The numeric values are part of the wire protocol —
// never reorder, only append.
enum class CommandId : std::uint32_t {
    kBoot = 0x01,         // Host posts BootRequest → sidecar replies BootResponse + IPC handle.
    kSync = 0x02,         // Host posts SyncRequest (peer handles) → sidecar opens IPC + barrier.
    kRoundTrip = 0x10,    // Smoke-test full dispatch+identity+combine on inline shape. Phase 1.1 smoke.
    kDispatch = 0x20,     // Real dispatch (host owns x/topk_*; sidecar fills recv_x via IPC). Phase 1.2+.
    kCombine = 0x21,      // Real combine (host owns recv_x post-expert; sidecar fills combined_x). Phase 1.2+.
    kShutdown = 0x7f,     // Clean shutdown — drain stream, free buffers, exit 0.
};

// Common 8-byte header for every message in both directions.
struct alignas(8) MessageHeader {
    std::uint32_t cmd_or_status;  // CommandId (request) or Status (response).
    std::uint32_t payload_bytes;  // Bytes of payload that follow this header.
};

// kBoot request — host posts after fork. Sidecar replies with its
// (device_id, IPC handle) tuple wrapped in BootResponse.
struct alignas(8) BootRequest {
    std::uint32_t protocol_version;  // Must equal kProtocolVersion.
    std::uint32_t rank;
    std::uint32_t world_size;
    std::uint32_t reserved;
};

struct alignas(8) BootResponse {
    std::uint32_t device_id;
    std::uint32_t reserved;
    std::uint8_t ipc_handle[64];  // cudaIpcMemHandle_t — 64-byte opaque blob.
};

// kSync request — host gathers all BootResponses, then broadcasts the
// full array of N peer (device_id, IPC handle) tuples back to each child.
// world_size is implied from the previous BootRequest.
// Payload layout: kMaxNvlPeers × { u32 device_id; u32 reserved; u8 ipc_handle[64]; }
//
// Sidecar opens each peer's IPC handle, populates pointer arrays, and runs
// intranode::barrier. Replies with an empty kOk header.

// kRoundTrip request — runs the phase 1.0a-iii smoke pattern end-to-end:
// (synthesize deterministic input | dispatch | identity expert step |
//  combine | SHA-256 of combined output). Sidecar generates the input
// internally — no host-side tensor exchange. Used to confirm the sidecar
// can drive the full DeepEP cycle without any real MoE wiring yet.
struct alignas(8) RoundTripRequest {
    std::uint32_t num_tokens;
    std::uint32_t hidden;
    std::uint32_t num_topk;
    std::uint32_t num_experts;
    std::uint32_t num_sms;            // dispatch + combine both use this.
    std::uint32_t nvl_chunked_send;
    std::uint32_t nvl_chunked_recv;
    std::uint32_t reserved;
};

struct alignas(8) RoundTripResponse {
    std::uint32_t num_recv_tokens;
    std::uint32_t reserved;
    std::uint8_t sha256[32];          // SHA-256 of the combined_x tensor bytes.
    float preview[8];                 // First 8 bf16-as-float values for eyeball check.
};

// kShutdown request has no payload. Sidecar acks then exits.

}  // namespace arle::deepep_sidecar

#endif  // ARLE_DEEPEP_SIDECAR_PROTOCOL_HPP_

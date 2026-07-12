// DSpark draft dense MLA-latent attention (T4.1) — POD-SIDE STUB.
//
// The Rust draft forward (crates/infer-cuda/src/dsv4/dspark.rs) is gated
// dead_code and nothing launches this until the DSpark stacking/verify tranche,
// so the body is unimplemented. Signature + math contract are frozen here for
// the pod implementer.
//
// Contract (mirrors ffi/attention.rs dsv4_dspark_draft_attention_cuda):
//   q         [block_size, local_heads, head_dim]  token-major
//   latent_kv [kv_len,     head_dim]               kv-major, head-SHARED (K==V)
//   out       [block_size, local_heads, nope_dim]  token-major
//   head_dim == nope_dim + rope_dim.
//
//   for each block row r, head h:
//     score[j] = sm_scale * dot(q[r,h,0..head_dim], latent_kv[j,0..head_dim])
//                // full head_dim (NoPE + RoPE)
//     w[0..kv_len] = online_softmax(score[0..kv_len])   // non-causal, all keys
//     out[r,h,0..nope_dim] = sum_j w[j] * latent_kv[j,0..nope_dim]
//                // weighted sum of the NoPE part only (V-side is the NoPE latent)

#include <cuda_runtime.h>
#include <cstdint>

// TODO(pod): implement small dense MLA-latent attention (see contract above).
extern "C" cudaError_t dsv4_dspark_draft_attention_cuda(
    const uint16_t *q,
    const uint16_t *latent_kv,
    uint16_t *out,
    int kv_len,
    int block_size,
    int local_heads,
    int head_dim,
    int nope_dim,
    int rope_dim,
    float sm_scale,
    cudaStream_t stream) {
  (void)q;
  (void)latent_kv;
  (void)out;
  (void)kv_len;
  (void)block_size;
  (void)local_heads;
  (void)head_dim;
  (void)nope_dim;
  (void)rope_dim;
  (void)sm_scale;
  (void)stream;
  return cudaErrorNotYetImplemented;
}

// metal_fused_ops.cpp — Fused transformer block for Apple Silicon Metal backend.
//
// Compiled only on macOS when the `metal` feature is active (see build.rs).
// Uses the MLX C++ API directly, reducing ~40 FFI round-trips per layer to 1.
//
// Memory model:
//   mlx_array = { void* ctx } where ctx = heap-allocated mlx::core::array*
//   Ownership: Rust side manages lifetimes; we borrow (never free) input arrays.
//   Output arrays: we heap-allocate new mlx::core::array objects and pass ctx back.
//
// Weight convention: all Dense weights are PRE-TRANSPOSED at load time (P1).
//   Dense weight shape is [in, out].  So projection is: out = x @ w_t   (no extra transpose).

// MLX C++ API (mlx::core::array, fast ops, transforms)
#include "mlx/mlx.h"
#include "mlx/fast.h"
#include "mlx/transforms.h"
#include "mlx/ops.h"

// mlx-c C ABI: mlx_array = { void* ctx } — our FFI boundary type.
// This header lives under build/include (installed mlx-c headers).
#include "mlx/c/array.h"

#include <cstdint>
#include <optional>
#include <vector>

// ── Helpers ───────────────────────────────────────────────────────────────────

// Borrow: extract a reference to the mlx::core::array stored in an mlx_array handle.
// The ctx pointer is a raw `new mlx::core::array*` (see mlx-c/private/array.h).
static inline mlx::core::array& arr_ref(mlx_array a) {
    return *static_cast<mlx::core::array*>(a.ctx);
}

// Own: move a C++ array into a new heap-allocated mlx_array (Rust takes ownership).
static inline mlx_array arr_own(mlx::core::array&& a) {
    return mlx_array{new mlx::core::array(std::move(a))};
}

// ── Dense linear: x @ w_t (w_t is already [in, out]) ─────────────────────────
static inline mlx::core::array dense_linear(
    const mlx::core::array& x,
    const mlx::core::array& w_t)   // pre-transposed [in, out]
{
    return mlx::core::matmul(x, w_t);
}

extern "C" {

// ── metal_fused_block_cached ──────────────────────────────────────────────────
//
// Full transformer block (norm → attention with KV cache → residual →
//                          norm → SwiGLU MLP → residual) in one C++ call.
//
// Inputs (all Dense weights are pre-transposed [in, out]):
//   x              [seq, hidden]
//   input_norm_w   [hidden]
//   post_attn_norm_w [hidden]
//   q_proj_t / k_proj_t / v_proj_t / o_proj_t  [in, out] (pre-transposed)
//   q_norm_w / k_norm_w   [head_dim]
//   gate_proj_t / up_proj_t / down_proj_t  [in, out]
//   k_cache / v_cache  [1, n_kv_heads, max_seq, head_dim] (mutable)
//   cache_len        filled prefix length (read position)
//   seq              current token count (== 1 for decode, > 1 for prefill)
//
// Outputs:
//   result_out  — new [seq, hidden] array (Rust takes ownership)
//   k_cache / v_cache updated in-place via slice_update
void metal_fused_block_cached(
    // ── input hidden state ─────────────────────────────────────────────────
    mlx_array x_h,
    // ── layer-norm weights ──────────────────────────────────────────────────
    mlx_array input_norm_w_h,
    mlx_array post_attn_norm_w_h,
    // ── attention projection weights (pre-transposed) ──────────────────────
    mlx_array q_proj_t_h,
    mlx_array k_proj_t_h,
    mlx_array v_proj_t_h,
    mlx_array o_proj_t_h,
    // ── per-head QK norms ───────────────────────────────────────────────────
    mlx_array q_norm_w_h,
    mlx_array k_norm_w_h,
    // ── MLP weights (pre-transposed) ────────────────────────────────────────
    mlx_array gate_proj_t_h,
    mlx_array up_proj_t_h,
    mlx_array down_proj_t_h,
    // ── attention hyper-params ──────────────────────────────────────────────
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float attn_scale,
    float rope_base,
    int rope_dims,
    float norm_eps,
    // ── KV cache (in/out, pre-allocated [1, n_kv_heads, max_seq, head_dim]) ─
    mlx_array* k_cache_h,   // mutable: updated via slice_update
    mlx_array* v_cache_h,
    int cache_len,          // start of new tokens in cache
    int seq,                // number of new tokens
    // ── output ─────────────────────────────────────────────────────────────
    mlx_array* result_out)
{
    using namespace mlx::core;
    namespace fast = mlx::core::fast;

    const array& x               = arr_ref(x_h);
    const array& input_norm_w    = arr_ref(input_norm_w_h);
    const array& post_attn_norm_w = arr_ref(post_attn_norm_w_h);
    const array& q_proj_t        = arr_ref(q_proj_t_h);
    const array& k_proj_t        = arr_ref(k_proj_t_h);
    const array& v_proj_t        = arr_ref(v_proj_t_h);
    const array& o_proj_t        = arr_ref(o_proj_t_h);
    const array& q_norm_w        = arr_ref(q_norm_w_h);
    const array& k_norm_w        = arr_ref(k_norm_w_h);
    const array& gate_proj_t     = arr_ref(gate_proj_t_h);
    const array& up_proj_t       = arr_ref(up_proj_t_h);
    const array& down_proj_t     = arr_ref(down_proj_t_h);
    array&       k_cache         = arr_ref(*k_cache_h);
    array&       v_cache         = arr_ref(*v_cache_h);

    // ── 1. Input residual + RMSNorm ─────────────────────────────────────────
    const array residual = x;
    array normed = fast::rms_norm(x, std::optional<array>(input_norm_w), norm_eps);

    // ── 2. QKV projections (x @ w_t, weights are pre-transposed) ───────────
    // normed: [seq, hidden]  →  q_raw: [seq, n_heads * head_dim]
    array q_raw = dense_linear(normed, q_proj_t);
    array k_raw = dense_linear(normed, k_proj_t);
    array v_raw = dense_linear(normed, v_proj_t);

    // ── 3. Per-head QK norms ────────────────────────────────────────────────
    // Reshape to [seq, n_heads, head_dim], norm, reshape back.
    q_raw = reshape(q_raw, {seq, n_heads, head_dim});
    k_raw = reshape(k_raw, {seq, n_kv_heads, head_dim});
    v_raw = reshape(v_raw, {seq, n_kv_heads, head_dim});

    q_raw = fast::rms_norm(q_raw, std::optional<array>(q_norm_w), norm_eps);
    k_raw = fast::rms_norm(k_raw, std::optional<array>(k_norm_w), norm_eps);

    // ── 4+5+6. Reshape → transpose → RoPE (→ [1, n_heads, seq, head_dim]) ───
    // fast::rope treats the second-to-last axis as the sequence dimension (T).
    // Transpose first so T = seq, not n_heads.
    array q = reshape(q_raw, {1, seq, n_heads, head_dim});
    array k = reshape(k_raw, {1, seq, n_kv_heads, head_dim});
    array v = reshape(v_raw, {1, seq, n_kv_heads, head_dim});

    q = transpose(q, {0, 2, 1, 3}); // [1, n_heads,   seq, head_dim]
    k = transpose(k, {0, 2, 1, 3}); // [1, n_kv_heads,seq, head_dim]
    v = transpose(v, {0, 2, 1, 3}); // [1, n_kv_heads,seq, head_dim]

    q = fast::rope(q, rope_dims, /*traditional=*/false,
                   std::optional<float>(rope_base), 1.0f, cache_len);
    k = fast::rope(k, rope_dims, /*traditional=*/false,
                   std::optional<float>(rope_base), 1.0f, cache_len);

    // ── 7. KV cache slice_update: write new tokens at [.., .., cache_len..end, ..] ─
    int end_pos = cache_len + seq;
    k_cache = slice_update(k_cache, k, {0, 0, cache_len, 0}, {1, n_kv_heads, end_pos, head_dim});
    v_cache = slice_update(v_cache, v, {0, 0, cache_len, 0}, {1, n_kv_heads, end_pos, head_dim});

    // ── 8. Read filled prefix [.., .., 0..end_pos, ..] ─────────────────────
    array k_full = slice(k_cache, {0, 0, 0, 0}, {1, n_kv_heads, end_pos, head_dim});
    array v_full = slice(v_cache, {0, 0, 0, 0}, {1, n_kv_heads, end_pos, head_dim});

    // ── 9. Grouped-query attention ──────────────────────────────────────────
    // q: [1, n_heads, seq, head_dim]  k/v: [1, n_kv_heads, end_pos, head_dim]
    bool use_causal = (cache_len == 0 && seq > 1);
    std::string mask_mode = use_causal ? "causal" : "";
    array attn_out = fast::scaled_dot_product_attention(
        q, k_full, v_full, attn_scale, mask_mode);

    // ── 10. Reshape back to [seq, hidden] and output projection ─────────────
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {seq, n_heads * head_dim});
    attn_out = dense_linear(attn_out, o_proj_t);

    // ── 11. Attention residual ───────────────────────────────────────────────
    array h = add(residual, attn_out);

    // ── 12. Post-attention norm + SwiGLU MLP ────────────────────────────────
    const array residual2 = h;
    array xn = fast::rms_norm(h, std::optional<array>(post_attn_norm_w), norm_eps);

    array gate  = dense_linear(xn, gate_proj_t);
    array up    = dense_linear(xn, up_proj_t);
    // SiLU(gate) = gate * sigmoid(gate)
    array gate_act = multiply(gate, sigmoid(gate));
    array down_in  = multiply(gate_act, up);
    array mlp_out  = dense_linear(down_in, down_proj_t);

    // ── 13. MLP residual ─────────────────────────────────────────────────────
    array output = add(residual2, mlp_out);

    *result_out = arr_own(std::move(output));
}

// ── Quantized linear: x @ dequant(w, scales, biases).T ──────────────────────
static inline mlx::core::array quant_linear(
    const mlx::core::array& x,
    const mlx::core::array& w,        // packed uint32 [out, packed_in]
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int group_size,
    int bits)
{
    return mlx::core::quantized_matmul(
        x, w, scales, biases, /*transpose=*/true, group_size, bits);
}

// ── metal_quantized_fused_block_cached ───────────────────────────────────────
//
// Full transformer block for quantized models.  Same structure as the dense
// variant above, but uses `quantized_matmul` for all projections and accepts
// merged QKV / gate-up weights with explicit split dimensions.
//
// Weight convention (quantized):
//   Each projection is a triple (w, scales, biases).
//   w: [out, packed_in] uint32  (packed quantized weights)
//   scales/biases: [out, n_groups]
//   quantized_matmul(x, w, scales, biases, transpose=true, group_size, bits)
//   produces x @ dequant(w).T → shape [seq, out].
//
// Merged projections:
//   qkv_proj: out = q_dim + k_dim + v_dim
//   gate_up_proj: out = gate_dim + up_dim
void metal_quantized_fused_block_cached(
    // ── input hidden state ─────────────────────────────────────────────────
    mlx_array x_h,
    // ── layer-norm weights ──────────────────────────────────────────────────
    mlx_array input_norm_w_h,
    mlx_array post_attn_norm_w_h,
    // ── quantized QKV projection (merged) ──────────────────────────────────
    mlx_array qkv_w_h,
    mlx_array qkv_scales_h,
    mlx_array qkv_biases_h,
    // ── quantized output projection ────────────────────────────────────────
    mlx_array o_w_h,
    mlx_array o_scales_h,
    mlx_array o_biases_h,
    // ── per-head QK norms ───────────────────────────────────────────────────
    mlx_array q_norm_w_h,
    mlx_array k_norm_w_h,
    // ── quantized MLP gate+up projection (merged) ──────────────────────────
    mlx_array gate_up_w_h,
    mlx_array gate_up_scales_h,
    mlx_array gate_up_biases_h,
    // ── quantized MLP down projection ──────────────────────────────────────
    mlx_array down_w_h,
    mlx_array down_scales_h,
    mlx_array down_biases_h,
    // ── quantization parameters (shared across all projections) ─────────────
    int group_size,
    int bits,
    // ── split dimensions ────────────────────────────────────────────────────
    int q_dim,
    int k_dim,
    int v_dim,
    int gate_dim,
    int up_dim,
    // ── attention hyper-params ──────────────────────────────────────────────
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float attn_scale,
    float rope_base,
    int rope_dims,
    float norm_eps,
    // ── KV cache (in/out, pre-allocated [1, n_kv_heads, max_seq, head_dim]) ─
    mlx_array* k_cache_h,   // mutable: updated via slice_update
    mlx_array* v_cache_h,
    int cache_len,          // start of new tokens in cache
    int seq,                // number of new tokens
    // ── output ─────────────────────────────────────────────────────────────
    mlx_array* result_out)
{
    using namespace mlx::core;
    namespace fast = mlx::core::fast;

    const array& x               = arr_ref(x_h);
    const array& input_norm_w    = arr_ref(input_norm_w_h);
    const array& post_attn_norm_w = arr_ref(post_attn_norm_w_h);
    const array& qkv_w           = arr_ref(qkv_w_h);
    const array& qkv_scales      = arr_ref(qkv_scales_h);
    const array& qkv_biases      = arr_ref(qkv_biases_h);
    const array& o_w             = arr_ref(o_w_h);
    const array& o_scales        = arr_ref(o_scales_h);
    const array& o_biases        = arr_ref(o_biases_h);
    const array& q_norm_w        = arr_ref(q_norm_w_h);
    const array& k_norm_w        = arr_ref(k_norm_w_h);
    const array& gate_up_w       = arr_ref(gate_up_w_h);
    const array& gate_up_scales  = arr_ref(gate_up_scales_h);
    const array& gate_up_biases  = arr_ref(gate_up_biases_h);
    const array& down_w          = arr_ref(down_w_h);
    const array& down_scales     = arr_ref(down_scales_h);
    const array& down_biases     = arr_ref(down_biases_h);
    array&       k_cache         = arr_ref(*k_cache_h);
    array&       v_cache         = arr_ref(*v_cache_h);

    // ── 1. Input residual + RMSNorm ─────────────────────────────────────────
    const array residual = x;
    array normed = fast::rms_norm(x, std::optional<array>(input_norm_w), norm_eps);

    // ── 2. Merged QKV projection (quantized) ───────────────────────────────
    array qkv = quant_linear(normed, qkv_w, qkv_scales, qkv_biases,
                             group_size, bits);
    // Split into q, k, v along the last axis.
    // split_sections expects cumulative indices: [q_dim, q_dim+k_dim].
    std::vector<int> qkv_split_at = {q_dim, q_dim + k_dim};
    auto qkv_parts = split(qkv, qkv_split_at, -1);
    array q_raw = qkv_parts[0];  // [seq, q_dim]
    array k_raw = qkv_parts[1];  // [seq, k_dim]
    array v_raw = qkv_parts[2];  // [seq, v_dim]

    // ── 3. Per-head QK norms ────────────────────────────────────────────────
    q_raw = reshape(q_raw, {seq, n_heads, head_dim});
    k_raw = reshape(k_raw, {seq, n_kv_heads, head_dim});
    v_raw = reshape(v_raw, {seq, n_kv_heads, head_dim});

    q_raw = fast::rms_norm(q_raw, std::optional<array>(q_norm_w), norm_eps);
    k_raw = fast::rms_norm(k_raw, std::optional<array>(k_norm_w), norm_eps);

    // ── 4+5+6. Reshape → transpose → RoPE (→ [1, n_heads, seq, head_dim]) ───
    array q = reshape(q_raw, {1, seq, n_heads, head_dim});
    array k = reshape(k_raw, {1, seq, n_kv_heads, head_dim});
    array v = reshape(v_raw, {1, seq, n_kv_heads, head_dim});

    q = transpose(q, {0, 2, 1, 3}); // [1, n_heads,   seq, head_dim]
    k = transpose(k, {0, 2, 1, 3}); // [1, n_kv_heads,seq, head_dim]
    v = transpose(v, {0, 2, 1, 3}); // [1, n_kv_heads,seq, head_dim]

    q = fast::rope(q, rope_dims, /*traditional=*/false,
                   std::optional<float>(rope_base), 1.0f, cache_len);
    k = fast::rope(k, rope_dims, /*traditional=*/false,
                   std::optional<float>(rope_base), 1.0f, cache_len);

    // ── 7. KV cache slice_update ────────────────────────────────────────────
    int end_pos = cache_len + seq;
    k_cache = slice_update(k_cache, k, {0, 0, cache_len, 0}, {1, n_kv_heads, end_pos, head_dim});
    v_cache = slice_update(v_cache, v, {0, 0, cache_len, 0}, {1, n_kv_heads, end_pos, head_dim});

    // ── 8. Read filled prefix [.., .., 0..end_pos, ..] ─────────────────────
    array k_full = slice(k_cache, {0, 0, 0, 0}, {1, n_kv_heads, end_pos, head_dim});
    array v_full = slice(v_cache, {0, 0, 0, 0}, {1, n_kv_heads, end_pos, head_dim});

    // ── 9. Grouped-query attention ──────────────────────────────────────────
    bool use_causal = (cache_len == 0 && seq > 1);
    std::string mask_mode = use_causal ? "causal" : "";
    array attn_out = fast::scaled_dot_product_attention(
        q, k_full, v_full, attn_scale, mask_mode);

    // ── 10. Reshape back to [seq, hidden] and output projection (quantized) ─
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {seq, n_heads * head_dim});
    attn_out = quant_linear(attn_out, o_w, o_scales, o_biases, group_size, bits);

    // ── 11. Attention residual ───────────────────────────────────────────────
    array h = add(residual, attn_out);

    // ── 12. Post-attention norm + SwiGLU MLP (quantized) ────────────────────
    const array residual2 = h;
    array xn = fast::rms_norm(h, std::optional<array>(post_attn_norm_w), norm_eps);

    // Merged gate+up projection, then split.
    array gate_up = quant_linear(xn, gate_up_w, gate_up_scales, gate_up_biases,
                                 group_size, bits);
    std::vector<int> gu_split_at = {gate_dim};
    auto gu_parts = split(gate_up, gu_split_at, -1);
    array gate = gu_parts[0];  // [seq, gate_dim]
    array up   = gu_parts[1];  // [seq, up_dim]

    // SiLU(gate) = gate * sigmoid(gate)
    array gate_act = multiply(gate, sigmoid(gate));
    array down_in  = multiply(gate_act, up);
    array mlp_out  = quant_linear(down_in, down_w, down_scales, down_biases,
                                  group_size, bits);

    // ── 13. MLP residual ─────────────────────────────────────────────────────
    array output = add(residual2, mlp_out);

    *result_out = arr_own(std::move(output));
}

} // extern "C"

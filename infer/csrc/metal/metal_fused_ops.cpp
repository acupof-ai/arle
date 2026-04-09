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

// ── ABI compat shims ─────────────────────────────────────────────────────────
// mlx-sys ships headers where Shape = SmallVector<int> but the compiled library
// was built when Shape = std::vector<int>.  We call the vector-ABI symbols
// directly through these shims so the linker finds them regardless of the
// header's Shape typedef.
using Vec = std::vector<int>;

// Linker-level declarations matching the actual exported symbols (std::vector).
namespace mlx { namespace core {
    extern array reshape(const array&, Vec, StreamOrDevice);
    extern array broadcast_to(const array&, const Vec&, StreamOrDevice);
    // Old-ABI quantized_matmul (no optionals, no string).
    extern array quantized_matmul(array, array, array, array, bool, int, int, StreamOrDevice);
    extern array sum(const array&, const Vec&, bool, StreamOrDevice);
    extern array slice(const array&, Vec, Vec, Vec, StreamOrDevice);
    extern array slice(const array&, Vec, Vec, StreamOrDevice);
    extern array slice_update(const array&, const array&, Vec, Vec, StreamOrDevice);
    extern array slice_update(const array&, const array&, Vec, Vec, Vec, StreamOrDevice);
    extern std::vector<array> split(const array&, const Vec&, int, StreamOrDevice);
}}

static inline mlx::core::array vec_reshape(
    const mlx::core::array& a, Vec shape) {
    return mlx::core::reshape(a, std::move(shape), mlx::core::StreamOrDevice{});
}

static inline mlx::core::array vec_broadcast_to(
    const mlx::core::array& a, Vec shape) {
    return mlx::core::broadcast_to(a, std::move(shape), mlx::core::StreamOrDevice{});
}

static inline mlx::core::array vec_slice(
    const mlx::core::array& src, Vec start, Vec stop) {
    return mlx::core::slice(src, std::move(start), std::move(stop), mlx::core::StreamOrDevice{});
}
static inline mlx::core::array vec_slice_update(
    const mlx::core::array& src, const mlx::core::array& update,
    Vec start, Vec stop) {
    return mlx::core::slice_update(src, update, std::move(start), std::move(stop), mlx::core::StreamOrDevice{});
}
static inline std::vector<mlx::core::array> vec_split(
    const mlx::core::array& a, Vec indices, int axis) {
    return mlx::core::split(a, std::move(indices), axis, mlx::core::StreamOrDevice{});
}

static inline mlx::core::array vec_sum(
    const mlx::core::array& a, Vec axes, bool keepdims = false) {
    return mlx::core::sum(a, axes, keepdims, mlx::core::StreamOrDevice{});
}

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

// ── SDPA: go through C API to avoid header/library ABI mismatch ──────────────
#include "mlx/c/fast.h"

/// SDPA via C API — returns ownership via mlx_array handle, caller wraps with arr_ref.
static inline mlx_array c_sdpa(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, float scale, const std::string& mask_mode) {
    mlx_array q_h = arr_own(mlx::core::array(q));
    mlx_array k_h = arr_own(mlx::core::array(k));
    mlx_array v_h = arr_own(mlx::core::array(v));
    mlx_array no_mask = mlx_array_new();
    mlx_array no_sinks = mlx_array_new();
    mlx_stream s = mlx_stream_new();
    {
        mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
        mlx_get_default_stream(&s, dev);
        mlx_device_free(dev);
    }
    mlx_array res;
    mlx_fast_scaled_dot_product_attention(
        &res, q_h, k_h, v_h, scale, mask_mode.c_str(), no_mask, no_sinks, s);
    mlx_array_free(q_h);
    mlx_array_free(k_h);
    mlx_array_free(v_h);
    mlx_array_free(no_mask);
    mlx_array_free(no_sinks);
    mlx_stream_free(s);
    return res; // caller takes ownership
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
    q_raw = vec_reshape(q_raw, {seq, n_heads, head_dim});
    k_raw = vec_reshape(k_raw, {seq, n_kv_heads, head_dim});
    v_raw = vec_reshape(v_raw, {seq, n_kv_heads, head_dim});

    q_raw = fast::rms_norm(q_raw, std::optional<array>(q_norm_w), norm_eps);
    k_raw = fast::rms_norm(k_raw, std::optional<array>(k_norm_w), norm_eps);

    // ── 4+5+6. Reshape → transpose → RoPE (→ [1, n_heads, seq, head_dim]) ───
    // fast::rope treats the second-to-last axis as the sequence dimension (T).
    // Transpose first so T = seq, not n_heads.
    array q = vec_reshape(q_raw, {1, seq, n_heads, head_dim});
    array k = vec_reshape(k_raw, {1, seq, n_kv_heads, head_dim});
    array v = vec_reshape(v_raw, {1, seq, n_kv_heads, head_dim});

    q = transpose(q, {0, 2, 1, 3}); // [1, n_heads,   seq, head_dim]
    k = transpose(k, {0, 2, 1, 3}); // [1, n_kv_heads,seq, head_dim]
    v = transpose(v, {0, 2, 1, 3}); // [1, n_kv_heads,seq, head_dim]

    q = fast::rope(q, rope_dims, /*traditional=*/false,
                   std::optional<float>(rope_base), 1.0f, cache_len);
    k = fast::rope(k, rope_dims, /*traditional=*/false,
                   std::optional<float>(rope_base), 1.0f, cache_len);

    // ── 7. KV cache slice_update: write new tokens at [.., .., cache_len..end, ..] ─
    int end_pos = cache_len + seq;
    k_cache = vec_slice_update(k_cache, k, Vec{0, 0, cache_len, 0}, Vec{1, n_kv_heads, end_pos, head_dim});
    v_cache = vec_slice_update(v_cache, v, Vec{0, 0, cache_len, 0}, Vec{1, n_kv_heads, end_pos, head_dim});

    // ── 8. Read filled prefix [.., .., 0..end_pos, ..] ─────────────────────
    array k_full = vec_slice(k_cache, Vec{0, 0, 0, 0}, Vec{1, n_kv_heads, end_pos, head_dim});
    array v_full = vec_slice(v_cache, Vec{0, 0, 0, 0}, Vec{1, n_kv_heads, end_pos, head_dim});

    // ── 9. Grouped-query attention ──────────────────────────────────────────
    // q: [1, n_heads, seq, head_dim]  k/v: [1, n_kv_heads, end_pos, head_dim]
    bool use_causal = (cache_len == 0 && seq > 1);
    std::string mask_mode = use_causal ? "causal" : "";
    array attn_out = arr_ref(c_sdpa(
        q, k_full, v_full, attn_scale, mask_mode));

    // ── 10. Reshape back to [seq, hidden] and output projection ─────────────
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = vec_reshape(attn_out, {seq, n_heads * head_dim});
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
        x, w, scales, biases, /*transpose=*/true, group_size, bits, {});
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
    auto qkv_parts = vec_split(qkv, qkv_split_at, -1);
    array q_raw = qkv_parts[0];  // [seq, q_dim]
    array k_raw = qkv_parts[1];  // [seq, k_dim]
    array v_raw = qkv_parts[2];  // [seq, v_dim]

    // ── 3. Per-head QK norms ────────────────────────────────────────────────
    q_raw = vec_reshape(q_raw, {seq, n_heads, head_dim});
    k_raw = vec_reshape(k_raw, {seq, n_kv_heads, head_dim});
    v_raw = vec_reshape(v_raw, {seq, n_kv_heads, head_dim});

    q_raw = fast::rms_norm(q_raw, std::optional<array>(q_norm_w), norm_eps);
    k_raw = fast::rms_norm(k_raw, std::optional<array>(k_norm_w), norm_eps);

    // ── 4+5+6. Reshape → transpose → RoPE (→ [1, n_heads, seq, head_dim]) ───
    array q = vec_reshape(q_raw, {1, seq, n_heads, head_dim});
    array k = vec_reshape(k_raw, {1, seq, n_kv_heads, head_dim});
    array v = vec_reshape(v_raw, {1, seq, n_kv_heads, head_dim});

    q = transpose(q, {0, 2, 1, 3}); // [1, n_heads,   seq, head_dim]
    k = transpose(k, {0, 2, 1, 3}); // [1, n_kv_heads,seq, head_dim]
    v = transpose(v, {0, 2, 1, 3}); // [1, n_kv_heads,seq, head_dim]

    q = fast::rope(q, rope_dims, /*traditional=*/false,
                   std::optional<float>(rope_base), 1.0f, cache_len);
    k = fast::rope(k, rope_dims, /*traditional=*/false,
                   std::optional<float>(rope_base), 1.0f, cache_len);

    // ── 7. KV cache slice_update ────────────────────────────────────────────
    int end_pos = cache_len + seq;
    k_cache = vec_slice_update(k_cache, k, Vec{0, 0, cache_len, 0}, Vec{1, n_kv_heads, end_pos, head_dim});
    v_cache = vec_slice_update(v_cache, v, Vec{0, 0, cache_len, 0}, Vec{1, n_kv_heads, end_pos, head_dim});

    // ── 8. Read filled prefix [.., .., 0..end_pos, ..] ─────────────────────
    array k_full = vec_slice(k_cache, Vec{0, 0, 0, 0}, Vec{1, n_kv_heads, end_pos, head_dim});
    array v_full = vec_slice(v_cache, Vec{0, 0, 0, 0}, Vec{1, n_kv_heads, end_pos, head_dim});

    // ── 9. Grouped-query attention ──────────────────────────────────────────
    bool use_causal = (cache_len == 0 && seq > 1);
    std::string mask_mode = use_causal ? "causal" : "";
    array attn_out = arr_ref(c_sdpa(
        q, k_full, v_full, attn_scale, mask_mode));

    // ── 10. Reshape back to [seq, hidden] and output projection (quantized) ─
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = vec_reshape(attn_out, {seq, n_heads * head_dim});
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
    auto gu_parts = vec_split(gate_up, gu_split_at, -1);
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

// ── Qwen3.5 helpers ─────────────────────────────────────────────────────────

/// Projection that handles both dense and quantized weights.
/// If is_quantized: uses quantized_matmul with w/scales/biases.
/// If !is_quantized: w is pre-transposed [in, out], uses dense matmul.
static inline mlx::core::array flex_linear(
    const mlx::core::array& x,
    const mlx::core::array& w,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int group_size, int bits, bool is_quantized)
{
    if (is_quantized) {
        return mlx::core::quantized_matmul(
            x, w, scales, biases, /*transpose=*/true, group_size, bits, {});
    } else {
        return mlx::core::matmul(x, w);
    }
}

/// RMS normalize with optional +1 offset on weight.
static inline mlx::core::array rms_norm_weighted(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    float eps,
    bool offset)
{
    using namespace mlx::core;
    namespace fast = mlx::core::fast;
    if (offset) {
        array scale = add(astype(weight, float32), array(1.0f));
        array normed = fast::rms_norm(astype(x, float32), std::nullopt, eps);
        return astype(multiply(normed, scale), bfloat16);
    } else {
        return fast::rms_norm(x, std::optional<array>(weight), eps);
    }
}

/// softplus(x) = log(1 + exp(x)), numerically stable for large x.
static inline mlx::core::array softplus(const mlx::core::array& x) {
    using namespace mlx::core;
    array threshold = array(20.0f);
    return where(greater(x, threshold), x, log1p(exp(x)));
}

// ── metal_qwen35_full_attn_block ────────────────────────────────────────────
//
// Qwen3.5 full-attention layer with q/gate split, per-head QK norm, partial
// RoPE, KV cache, SDPA, sigmoid gating, and output projection.
// Supports both dense and quantized weights.

void metal_qwen35_full_attn_block(
    // input
    mlx_array x_h,
    mlx_array input_norm_w_h,
    // attention projections (dense or quantized)
    mlx_array q_proj_w_h, mlx_array q_proj_scales_h, mlx_array q_proj_biases_h,
    mlx_array k_proj_w_h, mlx_array k_proj_scales_h, mlx_array k_proj_biases_h,
    mlx_array v_proj_w_h, mlx_array v_proj_scales_h, mlx_array v_proj_biases_h,
    mlx_array o_proj_w_h, mlx_array o_proj_scales_h, mlx_array o_proj_biases_h,
    // per-head norms
    mlx_array q_norm_w_h, mlx_array k_norm_w_h,
    // config
    int n_heads, int n_kv_heads, int head_dim, float attn_scale,
    float rope_base, int rotary_dim, float norm_eps,
    int group_size, int bits, bool is_quantized, bool norm_offset,
    // KV cache
    mlx_array* k_cache_h, mlx_array* v_cache_h, int cache_len,
    // output
    mlx_array* result_out)
{
    using namespace mlx::core;
    namespace fast = mlx::core::fast;

    const array& x         = arr_ref(x_h);
    const array& norm_w    = arr_ref(input_norm_w_h);
    const array& q_proj_w  = arr_ref(q_proj_w_h);
    const array& q_scales  = arr_ref(q_proj_scales_h);
    const array& q_biases  = arr_ref(q_proj_biases_h);
    const array& k_proj_w  = arr_ref(k_proj_w_h);
    const array& k_scales  = arr_ref(k_proj_scales_h);
    const array& k_biases  = arr_ref(k_proj_biases_h);
    const array& v_proj_w  = arr_ref(v_proj_w_h);
    const array& v_scales  = arr_ref(v_proj_scales_h);
    const array& v_biases  = arr_ref(v_proj_biases_h);
    const array& o_proj_w  = arr_ref(o_proj_w_h);
    const array& o_scales  = arr_ref(o_proj_scales_h);
    const array& o_biases  = arr_ref(o_proj_biases_h);
    const array& q_norm_w  = arr_ref(q_norm_w_h);
    const array& k_norm_w  = arr_ref(k_norm_w_h);
    array& k_cache         = arr_ref(*k_cache_h);
    array& v_cache         = arr_ref(*v_cache_h);

    int q_dim = n_heads * head_dim;

    // 1. Input norm
    array normed = rms_norm_weighted(x, norm_w, norm_eps, norm_offset);

    // 2. Q projection → [1, n_heads * head_dim * 2] (q + gate interleaved)
    array q_full = astype(
        flex_linear(normed, q_proj_w, q_scales, q_biases, group_size, bits, is_quantized),
        bfloat16);
    q_full = vec_reshape(q_full, {1, 1, n_heads, head_dim * 2});

    // Split q and gate per head
    array q_heads = vec_slice(q_full, Vec{0,0,0,0}, Vec{1,1,n_heads,head_dim});
    array gate_heads = vec_slice(q_full, Vec{0,0,0,head_dim}, Vec{1,1,n_heads,head_dim*2});

    // 3. K, V projections
    array k_raw = astype(
        flex_linear(normed, k_proj_w, k_scales, k_biases, group_size, bits, is_quantized),
        bfloat16);
    array v_raw = astype(
        flex_linear(normed, v_proj_w, v_scales, v_biases, group_size, bits, is_quantized),
        bfloat16);

    // 4. Per-head QK norm + RoPE
    array q = rms_norm_weighted(q_heads, q_norm_w, norm_eps, norm_offset);
    q = transpose(q, {0, 2, 1, 3}); // [1, n_heads, 1, head_dim]
    q = fast::rope(q, rotary_dim, false, std::optional<float>(rope_base), 1.0f, cache_len);

    array k = vec_reshape(k_raw, {1, 1, n_kv_heads, head_dim});
    k = rms_norm_weighted(k, k_norm_w, norm_eps, norm_offset);
    k = transpose(k, {0, 2, 1, 3}); // [1, n_kv_heads, 1, head_dim]
    k = fast::rope(k, rotary_dim, false, std::optional<float>(rope_base), 1.0f, cache_len);

    array v = vec_reshape(v_raw, {1, 1, n_kv_heads, head_dim});
    v = transpose(v, {0, 2, 1, 3}); // [1, n_kv_heads, 1, head_dim]

    // 5. KV cache update
    int end_pos = cache_len + 1;
    std::vector<int> kv_start = {0, 0, cache_len, 0};
    std::vector<int> kv_end = {1, n_kv_heads, end_pos, head_dim};
    k_cache = vec_slice_update(k_cache, k, kv_start, kv_end);
    v_cache = vec_slice_update(v_cache, v, kv_start, kv_end);

    std::vector<int> kv_origin = {0, 0, 0, 0};
    array k_full_cache = vec_slice(k_cache, kv_origin, kv_end);
    array v_full_cache = vec_slice(v_cache, kv_origin, kv_end);

    // 6. SDPA
    array attn_out = arr_ref(c_sdpa(
        q, k_full_cache, v_full_cache, attn_scale, ""));

    // 7. Reshape + sigmoid gating + output projection
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = vec_reshape(attn_out, {1, q_dim});
    array gate = vec_reshape(gate_heads, {1, q_dim});
    gate = sigmoid(astype(gate, float32));
    array gated = astype(multiply(astype(attn_out, float32), gate), bfloat16);

    array output = astype(
        flex_linear(gated, o_proj_w, o_scales, o_biases, group_size, bits, is_quantized),
        bfloat16);

    *result_out = arr_own(std::move(output));
}

// ── metal_qwen35_gdr_block ──────────────────────────────────────────────────
//
// Gated Delta Rule (GDR) linear attention decode step.
// Full GDR pipeline: project → conv1d → normalize → gate → state update → output.

void metal_qwen35_gdr_block(
    // input
    mlx_array x_h,                          // [1, hidden_size]
    // projections (dense or quantized)
    mlx_array qkv_w_h, mlx_array qkv_scales_h, mlx_array qkv_biases_h,
    mlx_array z_w_h, mlx_array z_scales_h, mlx_array z_biases_h,
    mlx_array beta_w_h, mlx_array beta_scales_h, mlx_array beta_biases_h,
    mlx_array alpha_w_h, mlx_array alpha_scales_h, mlx_array alpha_biases_h,
    mlx_array out_w_h, mlx_array out_scales_h, mlx_array out_biases_h,
    // fixed params
    mlx_array conv1d_weight_h,              // [qkv_dim, kernel_size] f32
    mlx_array dt_bias_h,                    // [num_value_heads] f32
    mlx_array a_log_h,                      // [num_value_heads] f32
    mlx_array norm_weight_h,                // [value_dim] f32
    // config
    int num_key_heads, int key_dim, int num_value_heads, int value_dim,
    int conv_kernel, int hidden_size, float rms_norm_eps,
    int group_size, int bits, bool is_quantized,
    // mutable state
    mlx_array* recurrent_state_h,           // [num_value_heads, key_dim, value_dim]
    mlx_array* conv_state_h,                // [qkv_dim, conv_kernel-1]
    // output
    mlx_array* result_out)
{
    using namespace mlx::core;
    namespace fast = mlx::core::fast;

    const array& x            = arr_ref(x_h);
    const array& qkv_w        = arr_ref(qkv_w_h);
    const array& qkv_scales   = arr_ref(qkv_scales_h);
    const array& qkv_biases   = arr_ref(qkv_biases_h);
    const array& z_w          = arr_ref(z_w_h);
    const array& z_scales     = arr_ref(z_scales_h);
    const array& z_biases     = arr_ref(z_biases_h);
    const array& beta_w       = arr_ref(beta_w_h);
    const array& beta_scales  = arr_ref(beta_scales_h);
    const array& beta_biases  = arr_ref(beta_biases_h);
    const array& alpha_w      = arr_ref(alpha_w_h);
    const array& alpha_scales = arr_ref(alpha_scales_h);
    const array& alpha_biases = arr_ref(alpha_biases_h);
    const array& out_w        = arr_ref(out_w_h);
    const array& out_scales   = arr_ref(out_scales_h);
    const array& out_biases   = arr_ref(out_biases_h);
    const array& conv_weight  = arr_ref(conv1d_weight_h);
    const array& dt_bias      = arr_ref(dt_bias_h);
    const array& a_log        = arr_ref(a_log_h);
    const array& norm_w       = arr_ref(norm_weight_h);
    array& state              = arr_ref(*recurrent_state_h);
    array& conv_state         = arr_ref(*conv_state_h);

    int q_dim  = num_key_heads * key_dim;
    int k_dim  = q_dim;
    int v_dim  = num_value_heads * value_dim;
    int qkv_dim = q_dim + k_dim + v_dim;

    // ── 1. Projections ──────────────────────────────────────────────────────
    array x_flat = vec_reshape(x, {1, hidden_size});
    array qkv_raw = astype(
        flex_linear(x_flat, qkv_w, qkv_scales, qkv_biases, group_size, bits, is_quantized),
        bfloat16);
    array z = astype(
        flex_linear(x_flat, z_w, z_scales, z_biases, group_size, bits, is_quantized),
        bfloat16);
    array beta_raw = astype(
        flex_linear(x_flat, beta_w, beta_scales, beta_biases, group_size, bits, is_quantized),
        bfloat16);
    array alpha_raw = astype(
        flex_linear(x_flat, alpha_w, alpha_scales, alpha_biases, group_size, bits, is_quantized),
        bfloat16);

    // ── 2. Conv1d step ──────────────────────────────────────────────────────
    array qkv_1d = vec_reshape(qkv_raw, {qkv_dim});
    array qkv_f32 = astype(qkv_1d, float32);
    array x_col = vec_reshape(qkv_f32, {qkv_dim, 1});
    array full_window = concatenate({conv_state, x_col}, 1);
    array conv_out = vec_sum(multiply(full_window, conv_weight), {1});
    conv_out = astype(conv_out, bfloat16);
    // SiLU activation
    array activated = astype(multiply(conv_out, sigmoid(conv_out)), bfloat16);
    activated = astype(activated, float32);

    // Update conv state: shift left
    int state_width = conv_kernel - 1;
    if (state_width > 0) {
        conv_state = vec_slice(full_window, Vec{0, 1}, Vec{qkv_dim, conv_kernel});
    }

    // ── 3. Split QKV + RMS normalize ────────────────────────────────────────
    array q_raw = vec_slice(activated, Vec{0}, Vec{q_dim});
    array k_raw = vec_slice(activated, Vec{q_dim}, Vec{q_dim + k_dim});
    array v_raw = vec_slice(activated, Vec{q_dim + k_dim}, Vec{qkv_dim});

    // Per-key-head RMS normalize
    array q_per_head = vec_reshape(q_raw, {num_key_heads, key_dim});
    array k_per_head = vec_reshape(k_raw, {num_key_heads, key_dim});

    // Manual RMS norm (no weight): norm(x) = x / sqrt(mean(x^2) + eps)
    auto rms_no_weight = [rms_norm_eps](const array& x) -> array {
        int d = x.shape().back();
        array sq = multiply(x, x);
        array mean_sq = multiply(vec_sum(sq, {-1}, true), array(1.0f / d));
        return multiply(x, reciprocal(sqrt(add(mean_sq, array(rms_norm_eps)))));
    };

    array q_norm = rms_no_weight(q_per_head);
    array k_norm = rms_no_weight(k_per_head);

    // Scale: q *= inv_scale^2, k *= inv_scale (matching mlx_lm)
    float inv_scale = 1.0f / std::sqrt((float)key_dim);
    array q_scaled = multiply(q_norm, array(inv_scale * inv_scale));
    array k_scaled = multiply(k_norm, array(inv_scale));

    // ── 4. Compute gate + beta ──────────────────────────────────────────────
    array alpha_1d = astype(vec_reshape(alpha_raw, {num_value_heads}), float32);
    array beta_1d = astype(vec_reshape(beta_raw, {num_value_heads}), float32);

    // g = exp(-exp(A_log) * softplus(alpha + dt_bias))
    array alpha_plus_bias = add(alpha_1d, dt_bias);
    array sp = softplus(alpha_plus_bias);
    array neg_a_exp = negative(exp(a_log));
    array exp_g = exp(multiply(neg_a_exp, sp)); // per-head decay [num_value_heads]

    array beta = sigmoid(beta_1d);

    // ── 5–6. State update (GDR delta rule) ──────────────────────────────────
    int heads_per_key = num_value_heads / num_key_heads;

    // Expand k to [num_value_heads, key_dim] by repeating key heads
    array k_expanded = k_scaled;
    if (heads_per_key > 1) {
        k_expanded = vec_reshape(
            vec_broadcast_to(
                expand_dims(k_scaled, 1),
                {num_key_heads, heads_per_key, key_dim}),
            {num_value_heads, key_dim});
    }
    array q_expanded = q_scaled;
    if (heads_per_key > 1) {
        q_expanded = vec_reshape(
            vec_broadcast_to(
                expand_dims(q_scaled, 1),
                {num_key_heads, heads_per_key, key_dim}),
            {num_value_heads, key_dim});
    }

    // v: [v_dim] → [num_value_heads, value_dim]
    array v_heads = vec_reshape(v_raw, {num_value_heads, value_dim});

    // Pass 1: Decay state
    array exp_g_3d = vec_reshape(exp_g, {num_value_heads, 1, 1});
    array s_decayed = multiply(state, exp_g_3d);

    // kv_mem = sum_j S_decayed[h,j,v] * k[h,j]
    array k_3d = vec_reshape(k_expanded, {num_value_heads, key_dim, 1});
    array kv_mem = vec_sum(multiply(s_decayed, k_3d), {1}); // [H, V]

    // delta = (v - kv_mem) * beta
    array v_minus_kv = subtract(v_heads, kv_mem);
    array beta_2d = vec_reshape(beta, {num_value_heads, 1});
    array delta = multiply(v_minus_kv, beta_2d); // [H, V]

    // Pass 2: Rank-1 update
    array delta_3d = vec_reshape(delta, {num_value_heads, 1, value_dim});
    array update = multiply(delta_3d, k_3d); // [H, K, V]
    state = add(s_decayed, update);

    // Output: o[h,v] = sum_j S_updated[h,j,v] * q[h,j]
    array q_3d = vec_reshape(q_expanded, {num_value_heads, key_dim, 1});
    array output_heads = vec_sum(multiply(state, q_3d), {1}); // [H, V]

    // ── 7. Per-head RMSNorm + output gate ───────────────────────────────────
    array out_bf16 = astype(output_heads, bfloat16);
    array normed = fast::rms_norm(out_bf16, std::optional<array>(norm_w), rms_norm_eps);

    array normed_flat = astype(vec_reshape(normed, {1, num_value_heads * value_dim}), bfloat16);

    // Output gate: o = normed * silu(z)
    array z_f32 = astype(z, float32);
    array z_silu = multiply(z_f32, sigmoid(z_f32));
    array gated = astype(
        multiply(astype(normed_flat, float32), z_silu),
        bfloat16);

    // ── 8. Output projection ────────────────────────────────────────────────
    array output = astype(
        flex_linear(gated, out_w, out_scales, out_biases, group_size, bits, is_quantized),
        bfloat16);

    *result_out = arr_own(std::move(output));
}

} // extern "C"

// metal_fused_capi.cpp — Fused transformer block using pure MLX C API.
//
// Avoids the C++ SmallVector/vector ABI mismatch in mlx-sys 0.2 by using
// only mlx_* C functions.  Compiled as C++ but only calls extern "C" funcs.
//
// Ownership: input arrays are borrowed; result_out transfers ownership to Rust.

// Include only the base types, avoid headers with overloaded C++ functions.
#include "mlx/c/array.h"
#include "mlx/c/device.h"
#include "mlx/c/stream.h"
#include "mlx/c/vector.h"

#include <cstring>

// Types from mlx/c headers that we need.
typedef struct { float value; bool has_value; } mlx_optional_float;

// Manually declare the specific C API functions we need (avoiding overload ambiguity).
extern "C" {
    int mlx_matmul(mlx_array* res, mlx_array a, mlx_array b, mlx_stream s);
    int mlx_add(mlx_array* res, mlx_array a, mlx_array b, mlx_stream s);
    int mlx_multiply(mlx_array* res, mlx_array a, mlx_array b, mlx_stream s);
    int mlx_sigmoid(mlx_array* res, mlx_array a, mlx_stream s);
    int mlx_reshape(mlx_array* res, mlx_array a, const int* shape, size_t n, mlx_stream s);
    int mlx_transpose_axes(mlx_array* res, mlx_array a, const int* axes, size_t n, mlx_stream s);
    int mlx_slice(mlx_array* res, mlx_array a, const int* start, size_t sn,
                  const int* stop, size_t en, const int* strides, size_t tn, mlx_stream s);
    int mlx_slice_update(mlx_array* res, mlx_array src, mlx_array upd,
                         const int* start, size_t sn, const int* stop, size_t en,
                         const int* strides, size_t tn, mlx_stream s);
    int mlx_quantized_matmul(mlx_array* res, mlx_array x, mlx_array w,
                             mlx_array scales, mlx_array biases,
                             bool transpose, int group_size, int bits, mlx_stream s);
    int mlx_fast_rms_norm(mlx_array* res, mlx_array x, mlx_array w, float eps, mlx_stream s);
    int mlx_fast_rope(mlx_array* res, mlx_array x, int dims, bool traditional,
                      mlx_optional_float base, float scale, int offset,
                      mlx_array freqs, mlx_stream s);
    int mlx_fast_scaled_dot_product_attention(mlx_array* res,
        mlx_array q, mlx_array k, mlx_array v, float scale,
        const char* mask_mode, mlx_vector_array mask_arrs, mlx_stream s);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

static mlx_stream gpu_stream(void) {
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_stream s = mlx_stream_new();
    mlx_get_default_stream(&s, dev);
    mlx_device_free(dev);
    return s;
}

// All ops use this cached stream.
static mlx_stream _stream;
static int _stream_init = 0;

static mlx_stream S(void) {
    if (!_stream_init) {
        _stream = gpu_stream();
        _stream_init = 1;
    }
    return _stream;
}

// Wrappers that return a new mlx_array (caller must free).
static mlx_array c_matmul(mlx_array a, mlx_array b) {
    mlx_array r = mlx_array_new();
    mlx_matmul(&r, a, b, S());
    return r;
}

static mlx_array c_add(mlx_array a, mlx_array b) {
    mlx_array r = mlx_array_new();
    mlx_add(&r, a, b, S());
    return r;
}

static mlx_array c_multiply(mlx_array a, mlx_array b) {
    mlx_array r = mlx_array_new();
    mlx_multiply(&r, a, b, S());
    return r;
}

static mlx_array c_sigmoid(mlx_array a) {
    mlx_array r = mlx_array_new();
    mlx_sigmoid(&r, a, S());
    return r;
}

static mlx_array c_reshape(mlx_array a, const int* shape, int ndim) {
    mlx_array r = mlx_array_new();
    mlx_reshape(&r, a, shape, (size_t)ndim, S());
    return r;
}

static mlx_array c_transpose(mlx_array a, const int* axes, int naxes) {
    mlx_array r = mlx_array_new();
    mlx_transpose_axes(&r, a, axes, (size_t)naxes, S());
    return r;
}

static mlx_array c_rms_norm(mlx_array x, mlx_array w, float eps) {
    mlx_array r = mlx_array_new();
    mlx_fast_rms_norm(&r, x, w, eps, S());
    return r;
}

static mlx_array c_rope(mlx_array x, int dims, float base, int offset) {
    mlx_array r = mlx_array_new();
    mlx_optional_float opt_base = { base, 1 };
    mlx_array no_freqs = mlx_array_new();
    mlx_fast_rope(&r, x, dims, 0/*traditional*/, opt_base, 1.0f, offset, no_freqs, S());
    mlx_array_free(no_freqs);
    return r;
}

static mlx_array c_sdpa_fn(mlx_array q, mlx_array k, mlx_array v, float scale, const char* mask_mode) {
    mlx_array r = mlx_array_new();
    mlx_vector_array masks = mlx_vector_array_new();
    mlx_fast_scaled_dot_product_attention(&r, q, k, v, scale, mask_mode, masks, S());
    mlx_vector_array_free(masks);
    return r;
}

static mlx_array c_slice(mlx_array a, const int* start, const int* stop, int ndim) {
    mlx_array r = mlx_array_new();
    int strides[4] = {1, 1, 1, 1};
    mlx_slice(&r, a, start, (size_t)ndim, stop, (size_t)ndim, strides, (size_t)ndim, S());
    return r;
}

static mlx_array c_slice_update(mlx_array src, mlx_array update, const int* start, const int* stop, int ndim) {
    mlx_array r = mlx_array_new();
    int strides[4] = {1, 1, 1, 1};
    mlx_slice_update(&r, src, update, start, (size_t)ndim, stop, (size_t)ndim, strides, (size_t)ndim, S());
    return r;
}

static mlx_array c_qmatmul(mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
                            int group_size, int bits) {
    mlx_array r = mlx_array_new();
    mlx_quantized_matmul(&r, x, w, scales, biases, 1/*transpose*/, group_size, bits, S());
    return r;
}

// ── metal_capi_fused_block ──────────────────────────────────────────────────
//
// Full transformer block via pure C API.  Same semantics as
// metal_fused_block_cached in metal_fused_ops.cpp.

extern "C" void metal_capi_fused_block(
    mlx_array x,
    mlx_array input_norm_w, mlx_array post_attn_norm_w,
    mlx_array q_proj_t, mlx_array k_proj_t, mlx_array v_proj_t, mlx_array o_proj_t,
    mlx_array q_norm_w, mlx_array k_norm_w,
    mlx_array gate_proj_t, mlx_array up_proj_t, mlx_array down_proj_t,
    int n_heads, int n_kv_heads, int head_dim, float attn_scale,
    float rope_base, int rope_dims, float norm_eps,
    mlx_array* k_cache, mlx_array* v_cache, int cache_len, int seq,
    mlx_array* result_out)
{
    // 1. Residual + RMSNorm
    mlx_array normed = c_rms_norm(x, input_norm_w, norm_eps);

    // 2. QKV projections
    mlx_array q_raw = c_matmul(normed, q_proj_t);
    mlx_array k_raw = c_matmul(normed, k_proj_t);
    mlx_array v_raw = c_matmul(normed, v_proj_t);
    mlx_array_free(normed);

    // 3. Per-head reshape + QK norms
    int qshape3[] = {seq, n_heads, head_dim};
    int kshape3[] = {seq, n_kv_heads, head_dim};
    mlx_array q3 = c_reshape(q_raw, qshape3, 3); mlx_array_free(q_raw);
    mlx_array k3 = c_reshape(k_raw, kshape3, 3); mlx_array_free(k_raw);
    mlx_array v3 = c_reshape(v_raw, kshape3, 3); mlx_array_free(v_raw);

    mlx_array qn = c_rms_norm(q3, q_norm_w, norm_eps); mlx_array_free(q3);
    mlx_array kn = c_rms_norm(k3, k_norm_w, norm_eps); mlx_array_free(k3);

    // 4. Reshape to [1, seq, heads, dim] then transpose to [1, heads, seq, dim]
    int qshape4[] = {1, seq, n_heads, head_dim};
    int kshape4[] = {1, seq, n_kv_heads, head_dim};
    int axes[] = {0, 2, 1, 3};

    mlx_array q4 = c_reshape(qn, qshape4, 4); mlx_array_free(qn);
    mlx_array k4 = c_reshape(kn, kshape4, 4); mlx_array_free(kn);
    mlx_array v4 = c_reshape(v3, kshape4, 4); mlx_array_free(v3);

    mlx_array qt = c_transpose(q4, axes, 4); mlx_array_free(q4);
    mlx_array kt = c_transpose(k4, axes, 4); mlx_array_free(k4);
    mlx_array vt = c_transpose(v4, axes, 4); mlx_array_free(v4);

    // 5. RoPE
    mlx_array q = c_rope(qt, rope_dims, rope_base, cache_len); mlx_array_free(qt);
    mlx_array k = c_rope(kt, rope_dims, rope_base, cache_len); mlx_array_free(kt);
    mlx_array v = vt; // no rope on v

    // 6. KV cache update
    int end_pos = cache_len + seq;
    int kv_start[] = {0, 0, cache_len, 0};
    int kv_end[]   = {1, n_kv_heads, end_pos, head_dim};
    int kv_zero[]  = {0, 0, 0, 0};

    mlx_array k_updated = c_slice_update(*k_cache, k, kv_start, kv_end, 4);
    mlx_array_free(*k_cache); *k_cache = k_updated;
    mlx_array v_updated = c_slice_update(*v_cache, v, kv_start, kv_end, 4);
    mlx_array_free(*v_cache); *v_cache = v_updated;
    mlx_array_free(k);
    mlx_array_free(v);

    mlx_array k_full = c_slice(*k_cache, kv_zero, kv_end, 4);
    mlx_array v_full = c_slice(*v_cache, kv_zero, kv_end, 4);

    // 7. Attention
    const char* mask_mode = (cache_len == 0 && seq > 1) ? "causal" : "";
    mlx_array attn_out = c_sdpa_fn(q, k_full, v_full, attn_scale, mask_mode);
    mlx_array_free(q); mlx_array_free(k_full); mlx_array_free(v_full);

    // 8. Reshape + output projection + residual
    mlx_array at2 = c_transpose(attn_out, axes, 4); mlx_array_free(attn_out);
    int oshape[] = {seq, n_heads * head_dim};
    mlx_array at3 = c_reshape(at2, oshape, 2); mlx_array_free(at2);
    mlx_array at4 = c_matmul(at3, o_proj_t); mlx_array_free(at3);
    mlx_array h = c_add(x, at4); mlx_array_free(at4);

    // 9. Post-attention norm + SwiGLU MLP
    mlx_array xn = c_rms_norm(h, post_attn_norm_w, norm_eps);
    mlx_array gate = c_matmul(xn, gate_proj_t);
    mlx_array up = c_matmul(xn, up_proj_t);
    mlx_array_free(xn);
    mlx_array gate_act = c_multiply(gate, c_sigmoid(gate)); mlx_array_free(gate);
    mlx_array down_in = c_multiply(gate_act, up); mlx_array_free(gate_act); mlx_array_free(up);
    mlx_array mlp_out = c_matmul(down_in, down_proj_t); mlx_array_free(down_in);

    // 10. MLP residual
    *result_out = c_add(h, mlp_out);
    mlx_array_free(h); mlx_array_free(mlp_out);
}

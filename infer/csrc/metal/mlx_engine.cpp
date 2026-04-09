// mlx_engine.cpp — MLX C++ inference engine for Qwen3/3.5.
//
// Uses MLX C++ API directly (same path as pybind11), eliminating the
// per-op C wrapper overhead that makes the Rust→C API path 1.8x slower
// than mlx_lm's Python path.
//
// Design: weights are loaded via mx::load (memory-mapped safetensors).
// Forward pass uses mlx::core ops directly. Caches are C++ vectors.
// Exposed to Rust via a simple C API (mlx_engine.h).

#include "mlx_engine.h"

#include "mlx/mlx.h"
#include "mlx/fast.h"
#include "mlx/ops.h"

// MLX C API for safetensors loading (memory-mapped).
#include "mlx/c/io.h"
#include "mlx/c/map.h"
#include "mlx/c/string.h"
#include "mlx/c/stream.h"
#include "mlx/c/device.h"
#include "mlx/c/array.h"

#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>


using namespace mlx::core;
namespace fast = mlx::core::fast;

// ── Weight container ────────────────────────────────────────────────────────

struct QuantWeight {
    array w, scales, biases;
    int group_size, bits;
    bool quantized;

    array operator()(const array& x) const {
        if (quantized) {
            return quantized_matmul(x, w, scales, biases, true, group_size, bits);
        }
        return matmul(x, w); // w is pre-transposed [in, out]
    }
};

// ── Full-attention layer ────────────────────────────────────────────────────

struct FullAttentionLayer {
    QuantWeight q_proj, k_proj, v_proj, o_proj;
    array q_norm, k_norm;
    // Per-layer KV cache: [1, n_kv_heads, max_seq, head_dim]
};

// ── Linear attention (GDR) layer ────────────────────────────────────────────

struct GDRLayer {
    QuantWeight in_proj_qkv, in_proj_z, in_proj_beta, in_proj_alpha, out_proj;
    array conv1d_weight; // [qkv_dim, kernel_size]
    array dt_bias;       // [num_value_heads]
    array a_log;         // [num_value_heads]
    array norm_weight;   // [value_dim]
};

// ── Transformer layer ───────────────────────────────────────────────────────

struct TransformerLayer {
    array input_layernorm, post_attention_layernorm;
    QuantWeight gate_proj, up_proj, down_proj;
    bool is_full_attention;
    // Union of attention types
    FullAttentionLayer full_attn;
    GDRLayer gdr;
};

// ── Model ───────────────────────────────────────────────────────────────────

struct mlx_engine_model {
    array embed_tokens;
    std::vector<TransformerLayer> layers;
    array final_norm;
    QuantWeight lm_head;

    // Config
    int hidden_size, n_heads, n_kv_heads, head_dim;
    float rope_base, norm_eps, attn_scale;
    int rotary_dim;
    bool norm_offset;
    int group_size, bits;
    bool is_quantized;

    // GDR config
    int gdr_num_key_heads, gdr_key_dim;
    int gdr_num_value_heads, gdr_value_dim;
    int gdr_conv_kernel;
};

// ── Decode state ────────────────────────────────────────────────────────────

struct mlx_engine_state {
    // Full-attention KV caches
    std::vector<array> k_caches, v_caches;
    // GDR recurrent states
    std::vector<array> recurrent_states, conv_states;
    int cache_len;
};

// ── Helpers ─────────────────────────────────────────────────────────────────

static inline array rms_norm_w(const array& x, const array& w, float eps, bool offset) {
    if (offset) {
        auto scale = add(astype(w, float32), array(1.0f));
        return astype(multiply(fast::rms_norm(astype(x, float32), std::nullopt, eps), scale), bfloat16);
    }
    return fast::rms_norm(x, std::optional<array>(w), eps);
}

static inline array stable_softplus(const array& x) {
    return where(greater(x, array(20.0f)), x, log1p(exp(x)));
}

// ── Full-attention step ─────────────────────────────────────────────────────

static array full_attn_step(
    const array& x, const TransformerLayer& layer,
    const mlx_engine_model& model,
    array& k_cache, array& v_cache, int cache_len)
{
    auto& attn = layer.full_attn;
    int n_heads = model.n_heads, n_kv = model.n_kv_heads, hd = model.head_dim;
    int q_dim = n_heads * hd;

    auto normed = rms_norm_w(x, layer.input_layernorm, model.norm_eps, model.norm_offset);

    // Q → [1, n_heads * hd * 2] → split into q + gate
    auto q_full = astype(attn.q_proj(normed), bfloat16);
    q_full = reshape(q_full, {1, 1, n_heads, hd * 2});
    auto q_gate = split(q_full, Shape{hd}, -1);
    auto q_heads = q_gate[0], gate_heads = q_gate[1];

    auto k_raw = astype(attn.k_proj(normed), bfloat16);
    auto v_raw = astype(attn.v_proj(normed), bfloat16);

    // QK norm + RoPE
    auto q = rms_norm_w(q_heads, attn.q_norm, model.norm_eps, model.norm_offset);
    q = fast::rope(transpose(q, {0,2,1,3}), model.rotary_dim, false,
                   std::optional<float>(model.rope_base), 1.0f, cache_len);

    auto k = reshape(k_raw, {1, 1, n_kv, hd});
    k = fast::rope(transpose(rms_norm_w(k, attn.k_norm, model.norm_eps, model.norm_offset), {0,2,1,3}),
                   model.rotary_dim, false, std::optional<float>(model.rope_base), 1.0f, cache_len);

    auto v = transpose(reshape(v_raw, {1, 1, n_kv, hd}), {0,2,1,3});

    // KV cache
    int end_pos = cache_len + 1;
    k_cache = slice_update(k_cache, k, {0,0,cache_len,0}, {1,n_kv,end_pos,hd});
    v_cache = slice_update(v_cache, v, {0,0,cache_len,0}, {1,n_kv,end_pos,hd});
    auto k_full = slice(k_cache, {0,0,0,0}, {1,n_kv,end_pos,hd});
    auto v_full = slice(v_cache, {0,0,0,0}, {1,n_kv,end_pos,hd});

    // SDPA
    auto attn_out = fast::scaled_dot_product_attention(q, k_full, v_full, model.attn_scale, "");
    attn_out = reshape(transpose(attn_out, {0,2,1,3}), {1, q_dim});

    // Sigmoid gating
    auto gate = sigmoid(astype(reshape(gate_heads, {1, q_dim}), float32));
    auto gated = astype(multiply(astype(attn_out, float32), gate), bfloat16);

    return astype(attn.o_proj(gated), bfloat16);
}

// ── GDR linear attention step ───────────────────────────────────────────────

static array gdr_step(
    const array& x, const TransformerLayer& layer,
    const mlx_engine_model& model,
    array& recurrent_state, array& conv_state)
{
    auto& gdr = layer.gdr;
    int nkh = model.gdr_num_key_heads, kd = model.gdr_key_dim;
    int nvh = model.gdr_num_value_heads, vd = model.gdr_value_dim;
    int q_dim = nkh * kd, k_dim = q_dim, v_dim = nvh * vd;
    int qkv_dim = q_dim + k_dim + v_dim;
    int hpk = nvh / nkh; // heads per key

    auto normed = rms_norm_w(x, layer.input_layernorm, model.norm_eps, model.norm_offset);
    auto x_flat = reshape(normed, {1, model.hidden_size});

    // Projections
    auto qkv_raw = astype(gdr.in_proj_qkv(x_flat), bfloat16);
    auto z = astype(gdr.in_proj_z(x_flat), bfloat16);
    auto beta_raw = astype(gdr.in_proj_beta(x_flat), bfloat16);
    auto alpha_raw = astype(gdr.in_proj_alpha(x_flat), bfloat16);

    // Conv1d step
    auto qkv_1d = astype(reshape(qkv_raw, {qkv_dim}), float32);
    auto x_col = reshape(qkv_1d, {qkv_dim, 1});
    auto full_window = concatenate({conv_state, x_col}, 1);
    auto conv_out = astype(sum(multiply(full_window, gdr.conv1d_weight), std::vector<int>{1}), bfloat16);
    auto activated = astype(multiply(conv_out, sigmoid(conv_out)), float32); // SiLU → f32
    conv_state = slice(full_window, {0, 1}, {qkv_dim, model.gdr_conv_kernel});

    // Split QKV + normalize
    auto q_raw = slice(activated, {0}, {q_dim});
    auto k_raw = slice(activated, {q_dim}, {q_dim + k_dim});
    auto v_raw = slice(activated, {q_dim + k_dim}, {qkv_dim});

    auto q_norm = fast::rms_norm(reshape(q_raw, {nkh, kd}), std::nullopt, 1e-6f);
    auto k_norm = fast::rms_norm(reshape(k_raw, {nkh, kd}), std::nullopt, 1e-6f);
    float inv = 1.0f / std::sqrt((float)kd);
    auto q_scaled = multiply(q_norm, array(inv * inv));
    auto k_scaled = multiply(k_norm, array(inv));

    // Gate
    auto alpha_1d = astype(reshape(alpha_raw, {nvh}), float32);
    auto beta_1d = astype(reshape(beta_raw, {nvh}), float32);
    auto sp = stable_softplus(add(alpha_1d, gdr.dt_bias));
    auto exp_g = exp(multiply(negative(exp(gdr.a_log)), sp));
    auto beta = sigmoid(beta_1d);

    // Expand q/k for GQA
    auto k_exp = k_scaled, q_exp = q_scaled;
    if (hpk > 1) {
        k_exp = reshape(broadcast_to(expand_dims(k_scaled, 1), {nkh, hpk, kd}), {nvh, kd});
        q_exp = reshape(broadcast_to(expand_dims(q_scaled, 1), {nkh, hpk, kd}), {nvh, kd});
    }
    auto v_heads = reshape(v_raw, {nvh, vd});

    // State update (delta rule)
    auto g3 = reshape(exp_g, {nvh, 1, 1});
    auto s_dec = multiply(recurrent_state, g3);
    auto k3 = reshape(k_exp, {nvh, kd, 1});
    auto kv_mem = sum(multiply(s_dec, k3), std::vector<int>{1});
    auto beta2 = reshape(beta, {nvh, 1});
    auto delta = multiply(subtract(v_heads, kv_mem), beta2);
    recurrent_state = add(s_dec, multiply(reshape(delta, {nvh, 1, vd}), k3));
    auto q3 = reshape(q_exp, {nvh, kd, 1});
    auto out_heads = sum(multiply(recurrent_state, q3), std::vector<int>{1});

    // Norm + gate + output proj
    auto out_bf = astype(out_heads, bfloat16);
    auto normed_out = fast::rms_norm(out_bf, std::optional<array>(gdr.norm_weight), model.norm_eps);
    auto flat = astype(reshape(normed_out, {1, nvh * vd}), bfloat16);
    auto z_silu = multiply(astype(z, float32), sigmoid(astype(z, float32)));
    auto gated = astype(multiply(astype(flat, float32), z_silu), bfloat16);

    return astype(gdr.out_proj(gated), bfloat16);
}

// ── Forward step ────────────────────────────────────────────────────────────

static array forward_step(
    int token_id,
    mlx_engine_model& model,
    mlx_engine_state& state)
{
    auto tok = array({token_id}, {1}, int32);
    auto x = take(model.embed_tokens, tok, 0);

    int full_idx = 0, linear_idx = 0;
    for (auto& layer : model.layers) {
        auto residual = x;
        array attn_out = array(0.0f);

        if (layer.is_full_attention) {
            attn_out = full_attn_step(x, layer, model,
                state.k_caches[full_idx], state.v_caches[full_idx], state.cache_len);
            full_idx++;
        } else {
            attn_out = gdr_step(x, layer, model,
                state.recurrent_states[linear_idx], state.conv_states[linear_idx]);
            linear_idx++;
        }

        x = add(residual, attn_out);

        // MLP
        auto residual2 = x;
        auto xn = rms_norm_w(x, layer.post_attention_layernorm, model.norm_eps, model.norm_offset);
        auto gate = astype(layer.gate_proj(xn), bfloat16);
        auto up = astype(layer.up_proj(xn), bfloat16);
        auto silu_gate = multiply(gate, sigmoid(gate));
        auto mlp = astype(layer.down_proj(astype(multiply(silu_gate, up), bfloat16)), bfloat16);
        x = add(residual2, mlp);
    }

    auto final_x = rms_norm_w(x, model.final_norm, model.norm_eps, model.norm_offset);
    return model.lm_head(final_x); // logits
}

// ── C API implementation ────────────────────────────────────────────────────

extern "C" {

mlx_engine_model_t mlx_engine_load(const char* model_dir, char* err_buf, int err_buf_len) {
    // TODO: implement model loading from safetensors + config.json
    // For now, return NULL (not implemented)
    if (err_buf && err_buf_len > 0) {
        snprintf(err_buf, err_buf_len, "mlx_engine_load not yet implemented");
    }
    return nullptr;
}

void mlx_engine_free_model(mlx_engine_model_t model) {
    delete model;
}

mlx_engine_state_t mlx_engine_new_state(mlx_engine_model_t model) {
    if (!model) return nullptr;
    auto* s = new mlx_engine_state();
    // TODO: allocate KV caches and recurrent state
    s->cache_len = 0;
    return s;
}

void mlx_engine_free_state(mlx_engine_state_t state) {
    delete state;
}

int32_t mlx_engine_decode_step(
    mlx_engine_model_t model, mlx_engine_state_t state,
    int32_t token_id, float temperature)
{
    if (!model || !state) return -1;

    auto logits = forward_step(token_id, *model, *state);
    state->cache_len++;

    // Greedy sampling
    if (temperature <= 1e-6f) {
        auto token = argmax(logits, -1);
        eval({token});
        return token.item<int32_t>();
    }

    // Temperature sampling
    auto scaled = multiply(logits, array(1.0f / temperature));
    auto token = random::categorical(scaled);
    eval({token});
    return token.item<int32_t>();
}

void mlx_engine_prefill(
    mlx_engine_model_t model, mlx_engine_state_t state,
    const int32_t* token_ids, int num_tokens)
{
    if (!model || !state) return;
    for (int i = 0; i < num_tokens; i++) {
        forward_step(token_ids[i], *model, *state);
        state->cache_len++;
    }
    // Force eval of all caches
    std::vector<array> to_eval;
    for (auto& k : state->k_caches) to_eval.push_back(k);
    for (auto& v : state->v_caches) to_eval.push_back(v);
    for (auto& s : state->recurrent_states) to_eval.push_back(s);
    eval(to_eval);
}

} // extern "C"

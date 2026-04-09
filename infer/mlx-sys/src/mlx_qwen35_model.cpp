//! Qwen3.5 C++ forward model used to collapse per-op Rust/FFI overhead.
//!
//! The implementation currently runs the C++ forward path directly. It does not
//! call `mx::compile()` because the position-dependent KV cache updates still
//! force retracing on each decode step.
//!
//! API:
//!   model = qwen35_compiled_new()
//!   qwen35_compiled_set_config(model, ...)
//!   qwen35_compiled_push_layer_full_attn(model, ...) // ×8
//!   qwen35_compiled_push_layer_gdr(model, ...)       // ×24
//!   qwen35_compiled_finalize(model)                  // validates/prepares model
//!   qwen35_compiled_step(model, token, caches_in, caches_out)
//!   qwen35_compiled_free(model)

#include "mlx_common.h"
#include <cstdlib>

using namespace mlx::core;

namespace {

bool use_gdr_metal_kernel() {
    const char* env = std::getenv("AGENT_INFER_GDR_METAL_KERNEL");
    return !(env && std::string(env) == "0");
}

auto& gated_delta_kernel() {
    static auto kernel = fast::metal_kernel(
        "gated_delta_step",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"},
        R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[hv_idx];
                kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
            }
            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            g_ += Hv;
            beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[i]);
        }
        )",
        "",
        true,
        false);
    return kernel;
}

// Compiled compute_g: g = exp(-exp(A_log.f32) * softplus(a + dt_bias))
// Matches mlx_lm's @mx.compile(shapeless=True) — fuses ~10 elementwise ops
// into a single compiled kernel, saving ~240 kernel launches per step.
std::vector<array> compute_g_impl(const std::vector<array>& inputs) {
    auto A_log = astype(inputs[0], float32);
    auto ab = inputs[1] + inputs[2];
    auto sp = where(greater(ab, array(20.0f)), ab, log1p(exp(ab)));
    return {exp(negative(exp(A_log)) * sp)};
}

auto& compiled_compute_g() {
    static auto fn = mlx::core::compile(compute_g_impl, true /*shapeless*/);
    return fn;
}

// Compiled SiLU: x * sigmoid(x) — matches mlx_lm's @mx.compile(shapeless=True)
// Fuses 2 ops (sigmoid + multiply) into 1 compiled kernel.
std::vector<array> silu_impl(const std::vector<array>& inputs) {
    return {inputs[0] * sigmoid(inputs[0])};
}

auto& compiled_silu() {
    static auto fn = mlx::core::compile(silu_impl, true /*shapeless*/);
    return fn;
}

// Compiled SwiGLU: silu(gate) * up — fuses 3 ops into 1 compiled kernel.
std::vector<array> swiglu_impl(const std::vector<array>& inputs) {
    auto gate = inputs[0];
    auto up = inputs[1];
    return {(gate * sigmoid(gate)) * up};
}

auto& compiled_swiglu() {
    static auto fn = mlx::core::compile(swiglu_impl, true /*shapeless*/);
    return fn;
}

} // namespace

// ── Quantized linear helper ────────────────────────────────────────────────

struct QWeight {
    array w = array(0);
    array scales = array(0);
    array biases = array(0);
    int group_size = 64;
    int bits = 4;
    bool is_dense = false;  // if true, w is already transposed, use matmul directly

    array apply(const array& x) const {
        if (is_dense) {
            return matmul(x, w);  // w is already transposed at load time
        }
        return quantized_matmul(x, w, scales, biases, true, group_size, bits);
    }
};

// ── Layer weight structs ───────────────────────────────────────────────────

struct FullAttnLayerWeights {
    array input_ln_w = array(0), post_attn_ln_w = array(0);
    QWeight q_proj, k_proj, v_proj, o_proj;
    array q_norm_w = array(0), k_norm_w = array(0);
    QWeight gate_up, down;
    int gate_dim = 0;
};

struct GdrLayerWeights {
    array input_ln_w = array(0), post_attn_ln_w = array(0);
    // Separate projections (matching mlx_lm — 4 matmul, no slice overhead)
    QWeight qkv_proj, z_proj, b_proj, a_proj, out_proj;
    // Legacy fused projections (used if separate not provided)
    QWeight qkvz_proj, ba_proj;
    int qkv_split = 0, z_split = 0, ba_num_heads = 0;
    bool use_separate_proj = false;
    array conv1d_w = array(0);
    array a_log = array(0), dt_bias = array(0);
    array norm_w = array(0);
    QWeight gate_proj, up_proj, down;
    bool use_separate_mlp = false;
    QWeight gate_up;
    int gate_dim = 0;
    int num_key_heads = 0, key_dim = 0, num_value_heads = 0, value_dim = 0;
    array q_scale_arr = array(0.0f);
    array k_scale_arr = array(0.0f);
    int conv_kernel = 4;
    float rms_eps = 1e-6f;
};

struct LayerWeights {
    bool is_gdr = false;
    FullAttnLayerWeights full;
    GdrLayerWeights gdr;
};

// ── Model struct ───────────────────────────────────────────────────────────

struct Qwen35CompiledModel {
    // Weights
    array embed_tokens = array(0);
    array final_norm_w = array(0);
    QWeight lm_head;
    std::vector<LayerWeights> layers;

    // Config
    float rope_theta = 1e6f;
    float rms_eps = 1e-6f;
    int n_heads = 16, n_kv_heads = 4, head_dim = 256;
    int rotary_dim = 256;
    int hidden_size = 2560;
    int n_full_attn = 0, n_gdr = 0;

    // Compiled function
    std::function<std::vector<array>(const std::vector<array>&)> compiled_fn;
    bool is_compiled = false;

    // Runtime state (set before each forward call)
    int current_cache_pos = 0;
    // Keep previous step's arrays alive to prevent premature GPU buffer release.
    // This mimics Python's lazy GC behavior where intermediates survive until
    // the next GC cycle, allowing MLX to reuse GPU buffers efficiently.
    std::vector<array> prev_outputs;
    // Collect ALL intermediate arrays during forward() to keep them alive.
    // Cleared at start of each step, populated during forward().
    mutable std::vector<array> intermediates;

    // ── Full attention decode step ─────────────────────────────────────

    array full_attn_step(
        const array& x, const FullAttnLayerWeights& lw,
        const array& k_cache, const array& v_cache, int cache_pos,
        array& new_k_cache, array& new_v_cache
    ) const {
        int nh = n_heads, nkv = n_kv_heads, hd = head_dim;
        float attn_scale = 1.0f / std::sqrt((float)hd);

        auto q_full = reshape(lw.q_proj.apply(x), {1, 1, nh, hd * 2});
        auto q_gate = split(q_full, Shape{hd}, -1);  // split at hd along last dim
        auto& q_heads = q_gate[0];
        auto& gate_heads = q_gate[1];

        auto k_raw = lw.k_proj.apply(x);
        auto v_raw = lw.v_proj.apply(x);

        // Q norm + RoPE
        auto q = fast::rms_norm(q_heads, lw.q_norm_w, rms_eps);
        q = transpose(q, {0, 2, 1, 3});
        q = fast::rope(q, rotary_dim, false, rope_theta, 1.0f, cache_pos);

        // K norm + RoPE
        auto k = reshape(k_raw, {1, 1, nkv, hd});
        k = fast::rms_norm(k, lw.k_norm_w, rms_eps);
        k = transpose(k, {0, 2, 1, 3});
        k = fast::rope(k, rotary_dim, false, rope_theta, 1.0f, cache_pos);

        auto v = reshape(v_raw, {1, 1, nkv, hd});
        v = transpose(v, {0, 2, 1, 3});

        // KV cache update
        int end = cache_pos + 1;
        new_k_cache = slice_update(k_cache, k, {0,0,cache_pos,0}, {1,nkv,end,hd});
        new_v_cache = slice_update(v_cache, v, {0,0,cache_pos,0}, {1,nkv,end,hd});

        auto k_full = slice(new_k_cache, {0,0,0,0}, {1,nkv,end,hd});
        auto v_full = slice(new_v_cache, {0,0,0,0}, {1,nkv,end,hd});

        // SDPA + gate
        auto attn_out = fast::scaled_dot_product_attention(q, k_full, v_full, attn_scale, "");
        attn_out = reshape(transpose(attn_out, {0,2,1,3}), {1, nh*hd});
        auto gate = reshape(gate_heads, {1, 1, nh*hd});
        gate = sigmoid(astype(gate, float32));
        auto gated = astype(astype(attn_out, float32) * gate, bfloat16);
        auto result = lw.o_proj.apply(gated);

        // Keep intermediates alive for GPU buffer reuse
        intermediates.push_back(q);
        intermediates.push_back(k);
        intermediates.push_back(attn_out);
        intermediates.push_back(gated);
        return result;
    }

    // ── GDR decode step ────────────────────────────────────────────────

    array gdr_step(
        const array& x, const GdrLayerWeights& lw,
        const array& gdr_state_in, const array& conv_state_in,
        array& gdr_state_out, array& conv_state_out
    ) const {
        int hk = lw.num_key_heads, dk = lw.key_dim;
        int hv = lw.num_value_heads, dv = lw.value_dim;
        int q_dim = hk * dk, k_dim = q_dim, v_dim = hv * dv;
        int qkv_dim = q_dim + k_dim + v_dim;

        auto x_3d = reshape(x, {1, 1, hidden_size});

        // Projections — separate matmuls matching mlx_lm (4 matmul, no split overhead)
        array qkv(0), z_raw(0), b_raw(0), a_raw(0);
        if (lw.use_separate_proj) {
            // 4 separate matmuls — no split/slice overhead (matches mlx_lm)
            qkv = lw.qkv_proj.apply(x_3d);
            z_raw = reshape(lw.z_proj.apply(x_3d), {1, 1, hv, dv}); // reshape like mlx_lm
            b_raw = lw.b_proj.apply(x_3d);
            a_raw = lw.a_proj.apply(x_3d);
        } else {
            auto qkvz = lw.qkvz_proj.apply(x_3d);
            auto qkv_z = split(qkvz, Shape{lw.qkv_split}, -1);
            qkv = qkv_z[0];
            z_raw = qkv_z[1];
            auto ba = lw.ba_proj.apply(x_3d);
            auto ba_parts = split(ba, Shape{lw.ba_num_heads}, -1);
            b_raw = ba_parts[0];
            a_raw = ba_parts[1];
        }

        // Conv1d
        auto conv_input = concatenate({conv_state_in, qkv}, 1);
        int n_keep = lw.conv_kernel - 1;
        conv_state_out = contiguous(slice(conv_input, {0,1,0}, {1,n_keep+1,qkv_dim}));
        auto conv_out = conv1d(conv_input, lw.conv1d_w, 1, 0, 1, qkv_dim);
        conv_out = compiled_silu()({conv_out})[0]; // SiLU (compiled)

        // Split conv output: 1 split op instead of 3 separate slices
        auto qkv_parts = split(conv_out, Shape{q_dim, q_dim + k_dim}, -1);
        auto q_raw = reshape(qkv_parts[0], {1, 1, hk, dk});
        auto k_raw = reshape(qkv_parts[1], {1, 1, hk, dk});
        auto v_raw = reshape(qkv_parts[2], {1, 1, hv, dv});

        auto q = fast::rms_norm(q_raw, std::nullopt, 1e-6f) * lw.q_scale_arr;
        auto k = fast::rms_norm(k_raw, std::nullopt, 1e-6f) * lw.k_scale_arr;

        // Gate computation: compiled (matching mlx_lm's @mx.compile(shapeless=True))
        auto beta = sigmoid(b_raw);
        auto g = compiled_compute_g()({lw.a_log, a_raw, lw.dt_bias})[0];

        array y(0);
        if (use_gdr_metal_kernel()) {
            // q, k are already bf16 (from rms_norm * scalar). v_raw is bf16 from conv.
            // Skip unnecessary casts — 3 fewer ops per GDR layer × 24 = 72 fewer ops.
            auto& q_bf16 = q;
            auto& k_bf16 = k;
            auto& v_bf16 = v_raw;
            auto g_3d = reshape(g, {1, 1, hv});
            auto beta_3d = reshape(beta, {1, 1, hv});
            auto t_arr = array(1);

            std::vector<array> inputs = {
                q_bf16, k_bf16, v_bf16, g_3d, beta_3d, gdr_state_in, t_arr
            };
            std::vector<Shape> out_shapes = {{1, 1, hv, dv}, gdr_state_in.shape()};
            std::vector<Dtype> out_dtypes = {bfloat16, float32};
            std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
                {"Dk", fast::TemplateArg(dk)},
                {"Dv", fast::TemplateArg(dv)},
                {"Hk", fast::TemplateArg(hk)},
                {"Hv", fast::TemplateArg(hv)},
                {"InT", fast::TemplateArg(bfloat16)},
                {"StT", fast::TemplateArg(float32)},
            };

            auto result = gated_delta_kernel()(
                inputs,
                out_shapes,
                out_dtypes,
                std::make_tuple(32, dv, hv),
                std::make_tuple(32, 4, 1),
                tmpl,
                std::nullopt,
                false,
                {});

            y = std::move(result[0]);
            gdr_state_out = std::move(result[1]);
        } else {
            int heads_per_key = hv / hk;
            auto g_4d = reshape(g, {1, hv, 1, 1});
            auto s_decayed = gdr_state_in * g_4d;

            // GQA: repeat k/q from [1,1,Hk,Dk] to [1,Hv,Dk] by inserting + broadcasting
            array k_exp = (heads_per_key > 1)
                ? reshape(broadcast_to(expand_dims(k, 3), {1,1,hk,heads_per_key,dk}), {1,hv,dk})
                : reshape(k, {1,hv,dk});
            array q_exp = (heads_per_key > 1)
                ? reshape(broadcast_to(expand_dims(q, 3), {1,1,hk,heads_per_key,dk}), {1,hv,dk})
                : reshape(q, {1,hv,dk});

            auto v_3d = reshape(v_raw, {1, hv, dv});
            auto k_4d = reshape(k_exp, {1, hv, 1, dk});
            auto kv_mem = sum(s_decayed * k_4d, -1, false);
            auto beta_3d = reshape(beta, {1, hv, 1});
            auto delta = (v_3d - kv_mem) * beta_3d;
            gdr_state_out = s_decayed + reshape(delta, {1,hv,dv,1}) * k_4d;
            auto q_4d = reshape(q_exp, {1, hv, 1, dk});
            y = reshape(sum(gdr_state_out * q_4d, -1, false), {1,1,hv,dv});
        }

        // Output norm + gate
        auto y_heads = reshape(y, {hv, dv});
        auto normed = fast::rms_norm(y_heads, lw.norm_w, lw.rms_eps);
        auto z_gated = reshape(z_raw, {hv, dv});
        auto out = normed * compiled_silu()({z_gated})[0];
        auto result = lw.out_proj.apply(reshape(out, {1, 1, hv*dv}));

        // Keep ALL available intermediates alive for GPU buffer reuse.
        auto& im = intermediates;
        im.push_back(x_3d);
        im.push_back(qkv); im.push_back(z_raw);
        im.push_back(b_raw); im.push_back(a_raw);
        im.push_back(conv_input); im.push_back(conv_out);
        for (auto& a : qkv_parts) im.push_back(a);
        im.push_back(q_raw); im.push_back(k_raw); im.push_back(v_raw);
        im.push_back(q); im.push_back(k);
        im.push_back(beta); im.push_back(g);
        im.push_back(y);
        im.push_back(y_heads); im.push_back(normed);
        im.push_back(z_gated); im.push_back(out);
        return result;
    }

    // ── MLP block ──────────────────────────────────────────────────────

    // Separate MLP: 2 matmul (matching mlx_lm, no split overhead)
    array mlp_separate(const array& x, const QWeight& gate, const QWeight& up, const QWeight& down) const {
        auto g = gate.apply(x);
        auto u = up.apply(x);
        auto h = compiled_swiglu()({g, u})[0];
        return down.apply(h);
    }

    // Fused MLP: 1 matmul + split
    array mlp(const array& x, const QWeight& gate_up, const QWeight& down, int gate_dim) const {
        auto gu = gate_up.apply(x);
        auto gu_parts = split(gu, Shape{gate_dim}, -1);
        auto& g = gu_parts[0];
        auto& u = gu_parts[1];
        auto h = compiled_swiglu()({g, u})[0]; // SiLU(gate) * up (compiled)
        return down.apply(h);
    }

    // ── Full forward pass ──────────────────────────────────────────────
    // inputs layout:
    //   [0]        : token_id (int32 scalar)
    //   [1]        : cache_pos (int32 scalar)
    //   [2..2+2*F) : k_cache_i, v_cache_i for F full-attn layers
    //   [2+2*F .. 2+2*F+2*G) : gdr_state_i, conv_state_i for G GDR layers
    // outputs layout:
    //   [0]        : logits
    //   [1..1+2*F) : new k/v caches
    //   [1+2*F .. 1+2*F+2*G) : new gdr/conv states

    std::vector<array> forward(const std::vector<array>& inputs) const {
        // Clear intermediates from previous step (releases old GPU buffers)
        intermediates.clear();
        intermediates.reserve(2048);  // pre-alloc to avoid realloc

        auto token_id = inputs[0];
        int cache_pos = current_cache_pos;

        int F = n_full_attn, G = n_gdr;
        // Ensure x is [1, 1, hidden] (3D) matching mlx_lm's tensor layout.
        // token_id may be [1] (1D) — reshape to [1, 1] so take returns [1, 1, H].
        auto tid = (token_id.ndim() == 1) ? reshape(token_id, {1, 1}) : token_id;
        auto x = take(embed_tokens, flatten(tid), 0);
        x = reshape(x, {1, 1, hidden_size});

        std::vector<array> new_kv_caches(2 * F, array(0));
        std::vector<array> new_gdr_states(G, array(0));
        std::vector<array> new_conv_states(G, array(0));
        int full_idx = 0, gdr_idx = 0;

        for (int i = 0; i < (int)layers.size(); ++i) {
            auto& layer = layers[i];
            auto residual = x;

            // Input layernorm
            auto ln_w = layer.is_gdr ? layer.gdr.input_ln_w : layer.full.input_ln_w;
            auto xn = fast::rms_norm(x, ln_w, rms_eps);

            // Attention
            array attn_out(0);
            if (layer.is_gdr) {
                int si = 1 + 2*F + 2*gdr_idx;
                attn_out = gdr_step(xn, layer.gdr,
                    inputs[si], inputs[si+1],
                    new_gdr_states[gdr_idx], new_conv_states[gdr_idx]);
                gdr_idx++;
            } else {
                int si = 1 + 2*full_idx;
                attn_out = full_attn_step(xn, layer.full,
                    inputs[si], inputs[si+1],
                    cache_pos,
                    new_kv_caches[2*full_idx], new_kv_caches[2*full_idx+1]);
                full_idx++;
            }

            x = residual + attn_out;

            // MLP
            auto residual2 = x;
            auto post_ln_w = layer.is_gdr ? layer.gdr.post_attn_ln_w : layer.full.post_attn_ln_w;
            auto xn2 = fast::rms_norm(x, post_ln_w, rms_eps);
            if (layer.is_gdr && layer.gdr.use_separate_mlp) {
                x = residual2 + mlp_separate(xn2, layer.gdr.gate_proj, layer.gdr.up_proj, layer.gdr.down);
            } else {
                auto& gu = layer.is_gdr ? layer.gdr.gate_up : layer.full.gate_up;
                auto& dw = layer.is_gdr ? layer.gdr.down : layer.full.down;
                int gd = layer.is_gdr ? layer.gdr.gate_dim : layer.full.gate_dim;
                x = residual2 + mlp(xn2, gu, dw, gd);
            }

            // Keep key intermediates alive for GPU buffer reuse.
            // Without this, C++ RAII destroys them immediately, causing MLX
            // to release GPU buffers that could be reused next step.
            intermediates.push_back(xn);
            intermediates.push_back(attn_out);
            intermediates.push_back(xn2);
        }

        // Final norm + lm_head
        auto final_x = fast::rms_norm(x, final_norm_w, rms_eps);
        auto logits = lm_head.apply(final_x);

        // Build output
        std::vector<array> outputs;
        outputs.reserve(1 + 2*F + 2*G);
        outputs.push_back(std::move(logits));
        for (auto& kv : new_kv_caches) outputs.push_back(std::move(kv));
        for (int j = 0; j < G; ++j) {
            outputs.push_back(std::move(new_gdr_states[j]));
            outputs.push_back(std::move(new_conv_states[j]));
        }
        return outputs;
    }

    void prepare_forward() {
        // NOTE: mx::compile() cannot handle position-dependent KV cache slicing
        // (cache_pos changes each step, forcing re-trace). For now, skip JIT
        // compilation and run the C++ forward directly. This still eliminates
        // most Rust/FFI overhead.
        //
        // Future: compile individual GDR+MLP sublayers (no position deps) while
        // keeping full-attention layers uncompiled.
        is_compiled = false;
    }
};

// ── FFI ────────────────────────────────────────────────────────────────────

extern "C" {

void* qwen35_compiled_new() {
    MLX_TRY_RETURN(new Qwen35CompiledModel());
}

void qwen35_compiled_free(void* model) {
    MLX_TRY_VOID(delete static_cast<Qwen35CompiledModel*>(model));
}

void qwen35_compiled_set_config(
    void* model,
    float rope_theta, float rms_eps,
    int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
    int32_t rotary_dim, int32_t hidden_size
) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->rope_theta = rope_theta;
        m->rms_eps = rms_eps;
        m->n_heads = n_heads;
        m->n_kv_heads = n_kv_heads;
        m->head_dim = head_dim;
        m->rotary_dim = rotary_dim;
        m->hidden_size = hidden_size;
    });
}

void qwen35_compiled_set_embed(
    void* model,
    mlx_array* embed_tokens,
    mlx_array* final_norm_w,
    mlx_array* lm_head_w, mlx_array* lm_head_s, mlx_array* lm_head_b,
    int32_t lm_gs, int32_t lm_bits
) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->embed_tokens = *to_arr(embed_tokens);
        m->final_norm_w = *to_arr(final_norm_w);
        if (lm_head_s == nullptr || lm_bits == 0) {
            // Dense lm_head (tied to embed_tokens transpose)
            m->lm_head = {*to_arr(lm_head_w), array(0), array(0), 0, 0, true};
        } else {
            m->lm_head = {*to_arr(lm_head_w), *to_arr(lm_head_s), *to_arr(lm_head_b), lm_gs, lm_bits, false};
        }
    });
}

void qwen35_compiled_push_full_attn(
    void* model,
    mlx_array* input_ln, mlx_array* post_ln,
    mlx_array* q_w, mlx_array* q_s, mlx_array* q_b, int32_t q_gs, int32_t q_bits,
    mlx_array* k_w, mlx_array* k_s, mlx_array* k_b,
    mlx_array* v_w, mlx_array* v_s, mlx_array* v_b,
    mlx_array* o_w, mlx_array* o_s, mlx_array* o_b,
    mlx_array* q_norm, mlx_array* k_norm,
    mlx_array* gu_w, mlx_array* gu_s, mlx_array* gu_b, int32_t gu_gs, int32_t gu_bits, int32_t gate_dim,
    mlx_array* dw_w, mlx_array* dw_s, mlx_array* dw_b
) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        LayerWeights lw;
        lw.is_gdr = false;
        lw.full.input_ln_w = *to_arr(input_ln);
        lw.full.post_attn_ln_w = *to_arr(post_ln);
        lw.full.q_proj = {*to_arr(q_w), *to_arr(q_s), *to_arr(q_b), q_gs, q_bits};
        lw.full.k_proj = {*to_arr(k_w), *to_arr(k_s), *to_arr(k_b), q_gs, q_bits};
        lw.full.v_proj = {*to_arr(v_w), *to_arr(v_s), *to_arr(v_b), q_gs, q_bits};
        lw.full.o_proj = {*to_arr(o_w), *to_arr(o_s), *to_arr(o_b), q_gs, q_bits};
        lw.full.q_norm_w = *to_arr(q_norm);
        lw.full.k_norm_w = *to_arr(k_norm);
        lw.full.gate_up = {*to_arr(gu_w), *to_arr(gu_s), *to_arr(gu_b), gu_gs, gu_bits};
        lw.full.down = {*to_arr(dw_w), *to_arr(dw_s), *to_arr(dw_b), gu_gs, gu_bits};
        lw.full.gate_dim = gate_dim;
        m->layers.push_back(std::move(lw));
        m->n_full_attn++;
    });
}

void qwen35_compiled_push_gdr(
    void* model,
    mlx_array* input_ln, mlx_array* post_ln,
    mlx_array* qkvz_w, mlx_array* qkvz_s, mlx_array* qkvz_b, int32_t qkvz_gs, int32_t qkvz_bits,
    int32_t qkv_split, int32_t z_split,
    mlx_array* ba_w, mlx_array* ba_s, mlx_array* ba_b, int32_t ba_gs, int32_t ba_bits,
    int32_t ba_num_heads,
    mlx_array* conv1d_w, int32_t conv_kernel,
    mlx_array* a_log, mlx_array* dt_bias,
    mlx_array* norm_w, float gdr_rms_eps,
    mlx_array* out_w, mlx_array* out_s, mlx_array* out_b, int32_t out_gs, int32_t out_bits,
    int32_t num_key_heads, int32_t key_dim, int32_t num_value_heads, int32_t value_dim,
    mlx_array* gu_w, mlx_array* gu_s, mlx_array* gu_b, int32_t gu_gs, int32_t gu_bits, int32_t gate_dim,
    mlx_array* dw_w, mlx_array* dw_s, mlx_array* dw_b
) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        LayerWeights lw;
        lw.is_gdr = true;
        lw.gdr.input_ln_w = *to_arr(input_ln);
        lw.gdr.post_attn_ln_w = *to_arr(post_ln);
        lw.gdr.qkvz_proj = {*to_arr(qkvz_w), *to_arr(qkvz_s), *to_arr(qkvz_b), qkvz_gs, qkvz_bits};
        lw.gdr.qkv_split = qkv_split;
        lw.gdr.z_split = z_split;
        lw.gdr.ba_proj = {*to_arr(ba_w), *to_arr(ba_s), *to_arr(ba_b), ba_gs, ba_bits};
        lw.gdr.ba_num_heads = ba_num_heads;
        lw.gdr.conv1d_w = *to_arr(conv1d_w);
        lw.gdr.conv_kernel = conv_kernel;
        lw.gdr.a_log = *to_arr(a_log);
        lw.gdr.dt_bias = *to_arr(dt_bias);
        lw.gdr.norm_w = *to_arr(norm_w);
        lw.gdr.rms_eps = gdr_rms_eps;
        lw.gdr.out_proj = {*to_arr(out_w), *to_arr(out_s), *to_arr(out_b), out_gs, out_bits};
        lw.gdr.num_key_heads = num_key_heads;
        lw.gdr.key_dim = key_dim;
        lw.gdr.num_value_heads = num_value_heads;
        lw.gdr.value_dim = value_dim;
        float inv = 1.0f / std::sqrt((float)key_dim);
        lw.gdr.q_scale_arr = astype(array(inv * inv), bfloat16);
        lw.gdr.k_scale_arr = astype(array(inv), bfloat16);
        lw.gdr.gate_up = {*to_arr(gu_w), *to_arr(gu_s), *to_arr(gu_b), gu_gs, gu_bits};
        lw.gdr.down = {*to_arr(dw_w), *to_arr(dw_s), *to_arr(dw_b), gu_gs, gu_bits};
        lw.gdr.gate_dim = gate_dim;
        m->layers.push_back(std::move(lw));
        m->n_gdr++;
    });
}

// Set individual (unfused) projections for the last-pushed GDR layer.
// Call AFTER qwen35_compiled_push_gdr. Enables the unfused matmul path
// which has fewer graph nodes (matches mlx_lm's op pattern).
void qwen35_compiled_set_separate_proj(
    void* model,
    mlx_array* qkv_w, mlx_array* qkv_s, mlx_array* qkv_b, int32_t qkv_gs, int32_t qkv_bits,
    mlx_array* z_w, mlx_array* z_s, mlx_array* z_b,
    mlx_array* b_w, mlx_array* b_s, mlx_array* b_b,
    mlx_array* a_w, mlx_array* a_s, mlx_array* a_b,
    mlx_array* gate_w, mlx_array* gate_s, mlx_array* gate_b, int32_t mlp_gs, int32_t mlp_bits,
    mlx_array* up_w, mlx_array* up_s, mlx_array* up_b
) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        auto& lw = m->layers.back().gdr;
        lw.qkv_proj = {*to_arr(qkv_w), *to_arr(qkv_s), *to_arr(qkv_b), qkv_gs, qkv_bits};
        lw.z_proj = {*to_arr(z_w), *to_arr(z_s), *to_arr(z_b), qkv_gs, qkv_bits};
        lw.b_proj = {*to_arr(b_w), *to_arr(b_s), *to_arr(b_b), qkv_gs, qkv_bits};
        lw.a_proj = {*to_arr(a_w), *to_arr(a_s), *to_arr(a_b), qkv_gs, qkv_bits};
        lw.use_separate_proj = true;
        lw.gate_proj = {*to_arr(gate_w), *to_arr(gate_s), *to_arr(gate_b), mlp_gs, mlp_bits};
        lw.up_proj = {*to_arr(up_w), *to_arr(up_s), *to_arr(up_b), mlp_gs, mlp_bits};
        lw.use_separate_mlp = true;
    });
}

int32_t qwen35_compiled_finalize(void* model) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    try {
        mlx_clear_error();
        m->prepare_forward();
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

// Run one decode step. Returns logits + updated caches.
// inputs/outputs arrays are allocated by caller.
int32_t qwen35_compiled_step(
    void* model,
    mlx_array* token_id,     // int32 scalar
    int32_t cache_pos,
    // KV caches: [n_full_attn * 2] — k0, v0, k1, v1, ...
    mlx_array** kv_caches, int32_t n_kv,
    // GDR states: [n_gdr * 2] — gdr0, conv0, gdr1, conv1, ...
    mlx_array** gdr_states, int32_t n_gdr,
    // Output
    mlx_array** out_logits,
    mlx_array** out_kv_caches,  // same count as kv_caches
    mlx_array** out_gdr_states  // same count as gdr_states
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    try {
        mlx_clear_error();

        m->current_cache_pos = cache_pos;

        // Build inputs vector
        std::vector<array> inputs;
        inputs.reserve(1 + n_kv + n_gdr);
        inputs.push_back(*to_arr(token_id));
        for (int i = 0; i < n_kv; ++i) inputs.push_back(*to_arr(kv_caches[i]));
        for (int i = 0; i < n_gdr; ++i) inputs.push_back(*to_arr(gdr_states[i]));

        // Run forward — returns lazy arrays.
        // IMPORTANT: keep the previous step's intermediate arrays alive by
        // storing outputs in a member variable. This prevents premature
        // GPU buffer deallocation when C++ locals are destructed.
        m->prev_outputs = m->forward(inputs);
        auto& outputs = m->prev_outputs;

        // Distribute outputs to caller (still lazy — no GPU work yet)
        *out_logits = from_arr(std::move(outputs[0]));
        for (int i = 0; i < n_kv; ++i)
            out_kv_caches[i] = from_arr(std::move(outputs[1 + i]));
        for (int i = 0; i < n_gdr; ++i)
            out_gdr_states[i] = from_arr(std::move(outputs[1 + n_kv + i]));

        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

} // extern "C"

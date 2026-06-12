//! DiffusionGemma / Gemma4 MLX forward model.
//!
//! This is a first Metal implementation optimized for correctness and backend
//! integration, not latency: each denoise pass recomputes the full
//! `context_tokens + canvas` sequence with an explicit block attention mask.

#include "mlx_common.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace {

using mlx::core::array;

array array_from_i32(const int32_t* data, int32_t len) {
    if (len < 0) {
        throw std::invalid_argument("negative token length");
    }
    if (len > 0 && data == nullptr) {
        throw std::invalid_argument("non-empty token input has null data pointer");
    }
    auto buf = allocator::malloc(static_cast<size_t>(len) * sizeof(int32_t));
    if (len > 0) {
        std::memcpy(buf.raw_ptr(), data, static_cast<size_t>(len) * sizeof(int32_t));
    }
    return array(std::move(buf), Shape{len}, int32);
}

array array_from_f32(const std::vector<float>& data, const Shape& shape) {
    auto buf = allocator::malloc(data.size() * sizeof(float));
    if (!data.empty()) {
        std::memcpy(buf.raw_ptr(), data.data(), data.size() * sizeof(float));
    }
    return array(std::move(buf), shape, float32);
}

array gelu_tanh(const array& x) {
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    auto x3 = x * x * x;
    return array(0.5f) * x * (array(1.0f) + tanh(array(kAlpha) * (x + array(kBeta) * x3)));
}

array gelu_exact(const array& x) {
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    return array(0.5f) * x * (array(1.0f) + erf(x * array(kInvSqrt2)));
}

struct QWeight {
    array w = array(0);
    array scales = array(0);
    array biases = array(0);
    int group_size = 64;
    int bits = 4;
    bool is_dense = false;

    array apply(const array& x) const {
        if (is_dense) {
            return matmul(x, w);
        }
        return quantized_matmul(x, w, scales, biases, true, group_size, bits);
    }
};

struct LayerWeights {
    bool is_full_attention = false;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int rotary_dim = 0;
    float rope_theta = 10000.0f;
    int sliding_window = 0;

    array input_ln = array(0);
    QWeight q_proj;
    QWeight k_proj;
    QWeight v_proj;
    bool k_eq_v = false;
    QWeight o_proj;
    array q_norm = array(0);
    array k_norm = array(0);
    array post_attn_ln = array(0);

    array pre_ff_ln = array(0);
    QWeight gate_proj;
    QWeight up_proj;
    QWeight down_proj;
    array post_ff_ln = array(0);

    bool has_moe = false;
    array pre_ff2_ln = array(0);
    array post_ff1_ln = array(0);
    array post_ff2_ln = array(0);
    QWeight router;
    array router_scale = array(0);
    array per_expert_scale = array(0);
    QWeight expert_gate_up;
    QWeight expert_down;
    int num_experts = 0;
    int top_k = 0;

    array layer_scalar = array(1.0f);
};

struct SelfConditioningWeights {
    array pre_norm = array(0);
    QWeight gate_proj;
    QWeight up_proj;
    QWeight down_proj;
    bool ready = false;
};

struct DiffusionGemmaModel {
    int hidden_size = 0;
    int vocab_size = 0;
    float rms_eps = 1e-6f;
    float final_logit_softcap = 0.0f;

    std::vector<QWeight> weights;
    QWeight embed;
    array embed_tokens = array(0);
    array lm_head_t = array(0);
    array final_norm = array(0);
    std::vector<LayerWeights> layers;
    SelfConditioningWeights self_conditioning;

    std::vector<int32_t> context_tokens;
    array self_conditioning_embeds = array(0);
    int self_conditioning_len = 0;
    bool finalized = false;

    QWeight& weight_by_id(int32_t id) {
        if (id < 0 || id >= static_cast<int32_t>(weights.size())) {
            throw std::invalid_argument("DiffusionGemma weight id out of range");
        }
        return weights[static_cast<size_t>(id)];
    }

    array array_by_id(int32_t id) {
        return weight_by_id(id).w;
    }

    array qmm(int32_t id, const array& x) {
        return weight_by_id(id).apply(x);
    }

    array rms(const array& x, const array& weight) const {
        return fast::rms_norm(x, weight, rms_eps);
    }

    array rms_no_weight(const array& x) const {
        return fast::rms_norm(x, std::nullopt, rms_eps);
    }

    array build_mask(int total_len, int prompt_len, const LayerWeights& layer) const {
        std::vector<float> mask(static_cast<size_t>(total_len) * total_len, -1.0e9f);
        for (int i = 0; i < total_len; ++i) {
            for (int j = 0; j < total_len; ++j) {
                bool allowed = false;
                if (i < prompt_len) {
                    allowed = j <= i;
                } else {
                    allowed = j < total_len;
                }
                if (allowed && !layer.is_full_attention && layer.sliding_window > 0) {
                    const int left = std::max(0, i - layer.sliding_window + 1);
                    int right = i;
                    if (i >= prompt_len) {
                        right = std::min(total_len - 1, i + layer.sliding_window - 1);
                    }
                    allowed = j >= left && j <= right;
                }
                if (allowed) {
                    mask[static_cast<size_t>(i) * total_len + j] = 0.0f;
                }
            }
        }
        return array_from_f32(mask, Shape{1, 1, total_len, total_len});
    }

    array attention(const array& x, const LayerWeights& layer, int prompt_len) const {
        const int s = x.shape(0);
        const int nh = layer.num_heads;
        const int nkv = layer.num_kv_heads;
        const int hd = layer.head_dim;
        if (s <= 0 || nh <= 0 || nkv <= 0 || hd <= 0) {
            throw std::runtime_error("invalid DiffusionGemma attention shape");
        }

        auto x3 = reshape(x, {1, s, hidden_size});
        auto q_raw = layer.q_proj.apply(x3);
        auto k_raw = layer.k_proj.apply(x3);
        auto v_raw = layer.k_eq_v ? k_raw : layer.v_proj.apply(x3);

        auto q = reshape(q_raw, {1, s, nh, hd});
        q = rms(q, layer.q_norm);
        q = transpose(q, {0, 2, 1, 3});

        auto k = reshape(k_raw, {1, s, nkv, hd});
        k = rms(k, layer.k_norm);
        k = transpose(k, {0, 2, 1, 3});

        q = fast::rope(q, layer.rotary_dim, false, layer.rope_theta, 1.0f, 0);
        k = fast::rope(k, layer.rotary_dim, false, layer.rope_theta, 1.0f, 0);

        auto v = reshape(v_raw, {1, s, nkv, hd});
        v = rms_no_weight(v);
        v = transpose(v, {0, 2, 1, 3});

        auto mask = build_mask(s, prompt_len, layer);
        // Gemma4 folds query scaling into q_norm weights; attention scale is literally 1.
        float scale = 1.0f;
        auto attn = fast::scaled_dot_product_attention(q, k, v, scale, "", mask);
        auto flat = reshape(transpose(attn, {0, 2, 1, 3}), {1, s, nh * hd});
        auto out = layer.o_proj.apply(flat);
        return reshape(out, {s, hidden_size});
    }

    array dense_mlp(const array& x, const LayerWeights& layer) const {
        auto gate = layer.gate_proj.apply(x);
        auto up = layer.up_proj.apply(x);
        auto h = gelu_tanh(gate) * up;
        return layer.down_proj.apply(h);
    }

    array expert_switch(const array& x, const array& inds, const LayerWeights& layer) const {
        auto x5 = expand_dims(x, std::vector<int>{-2, -3});
        auto gate_up = gather_qmm(
            x5,
            layer.expert_gate_up.w,
            layer.expert_gate_up.scales,
            layer.expert_gate_up.biases,
            std::nullopt,
            inds,
            true,
            layer.expert_gate_up.group_size,
            layer.expert_gate_up.bits,
            "affine",
            false);
        auto parts = split(gate_up, Shape{gate_up.shape(gate_up.ndim() - 1) / 2}, -1);
        // Dense MLP uses gelu_tanh, but Gemma4 MoE experts use plain GELU.
        auto h = gelu_exact(parts[0]) * parts[1];
        auto y = gather_qmm(
            h,
            layer.expert_down.w,
            layer.expert_down.scales,
            layer.expert_down.biases,
            std::nullopt,
            inds,
            true,
            layer.expert_down.group_size,
            layer.expert_down.bits,
            "affine",
            false);
        return squeeze(y, -2);
    }

    array moe(const array& expert_input, const array& router_input, const LayerWeights& layer) const {
        if (!layer.has_moe) {
            return array(0);
        }
        auto router_x = rms_no_weight(router_input);
        router_x = router_x * array(1.0f / std::sqrt(static_cast<float>(hidden_size)));
        router_x = router_x * layer.router_scale;
        auto logits = layer.router.apply(router_x);

        auto gates = softmax(logits, -1, true);
        const int kth = layer.num_experts - layer.top_k;
        auto part = argpartition(gates, kth, -1);
        Shape start(part.ndim(), 0);
        Shape stop = part.shape();
        Shape strides(part.ndim(), 1);
        start[part.ndim() - 1] = kth;
        auto inds = slice(part, start, stop, strides);

        auto scores = take_along_axis(gates, inds, -1);
        auto denom = sum(scores, -1, true);
        scores = scores / denom;
        auto expert_scales = take(layer.per_expert_scale, inds, 0);
        scores = scores * expert_scales;
        if (scores.dtype() != expert_input.dtype()) {
            scores = astype(scores, expert_input.dtype());
        }

        auto y = expert_switch(expert_input, inds, layer);
        return sum(y * expand_dims(scores, -1), -2, false);
    }

    array apply_self_conditioning(const array& x, int prompt_len, int valid_len) const {
        if (!self_conditioning.ready || self_conditioning_len != valid_len || valid_len <= 0) {
            return x;
        }
        auto canvas = slice(x, Shape{prompt_len, 0}, Shape{prompt_len + valid_len, hidden_size});
        auto sc_x = rms(self_conditioning_embeds, self_conditioning.pre_norm);
        auto gate = self_conditioning.gate_proj.apply(sc_x);
        auto up = self_conditioning.up_proj.apply(sc_x);
        auto signal = self_conditioning.down_proj.apply(gelu_tanh(gate) * up);
        auto conditioned = rms_no_weight(canvas + signal);
        return slice_update(
            x,
            conditioned,
            Shape{prompt_len, 0},
            Shape{prompt_len + valid_len, hidden_size});
    }

    array forward_logits(const int32_t* canvas, int canvas_len, int valid_len) {
        if (!finalized) {
            throw std::runtime_error("DiffusionGemma model was not finalized");
        }
        if (canvas_len <= 0 || valid_len <= 0 || valid_len > canvas_len) {
            throw std::invalid_argument("invalid DiffusionGemma canvas length");
        }
        const int prompt_len = static_cast<int>(context_tokens.size());
        std::vector<int32_t> tokens = context_tokens;
        tokens.insert(tokens.end(), canvas, canvas + valid_len);
        auto token_ids = array_from_i32(tokens.data(), static_cast<int32_t>(tokens.size()));
        auto x = take(embed_tokens, token_ids, 0) * array(std::sqrt(static_cast<float>(hidden_size)));
        x = apply_self_conditioning(x, prompt_len, valid_len);

        for (const auto& layer : layers) {
            auto residual = x;
            auto h = rms(x, layer.input_ln);
            h = attention(h, layer, prompt_len);
            h = rms(h, layer.post_attn_ln);
            x = h + residual;

            residual = x;
            h = rms(x, layer.pre_ff_ln);
            h = dense_mlp(h, layer);
            if (layer.has_moe) {
                auto h1 = rms(h, layer.post_ff1_ln);
                auto h2 = rms(residual, layer.pre_ff2_ln);
                auto m = moe(h2, residual, layer);
                h = h1 + rms(m, layer.post_ff2_ln);
            }
            h = rms(h, layer.post_ff_ln);
            x = (h + residual) * layer.layer_scalar;
        }

        x = rms(x, final_norm);
        auto canvas_hidden = slice(
            x,
            Shape{prompt_len, 0},
            Shape{prompt_len + valid_len, hidden_size});
        auto logits = matmul(canvas_hidden, lm_head_t);
        if (final_logit_softcap > 0.0f) {
            logits = tanh(logits / array(final_logit_softcap)) * array(final_logit_softcap);
        }
        return logits;
    }

    void update_self_conditioning(const array& probs) {
        auto soft = matmul(probs, embed_tokens) * array(std::sqrt(static_cast<float>(hidden_size)));
        self_conditioning_embeds = soft;
        self_conditioning_len = soft.shape(0);
    }
};

DiffusionGemmaModel* as_model(void* raw) {
    if (raw == nullptr) {
        throw std::invalid_argument("DiffusionGemma model pointer is null");
    }
    return reinterpret_cast<DiffusionGemmaModel*>(raw);
}

QWeight make_dense(mlx_array* w) {
    if (w == nullptr) {
        throw std::invalid_argument("dense weight is null");
    }
    QWeight weight;
    weight.w = *to_arr(w);
    weight.is_dense = true;
    return weight;
}

QWeight make_affine(
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,
    int32_t group_size,
    int32_t bits) {
    if (w == nullptr || scales == nullptr || biases == nullptr) {
        throw std::invalid_argument("affine weight received null tensor");
    }
    QWeight weight;
    weight.w = *to_arr(w);
    weight.scales = *to_arr(scales);
    weight.biases = *to_arr(biases);
    weight.group_size = group_size;
    weight.bits = bits;
    weight.is_dense = false;
    return weight;
}

int32_t catch_to_rc(const std::function<void()>& fn) {
    try {
        mlx_clear_error();
        fn();
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

} // namespace

extern "C" {

void* diffusion_gemma_new() {
    MLX_TRY_RETURN_VALUE(nullptr, new DiffusionGemmaModel());
}

void diffusion_gemma_free(void* model) {
    MLX_TRY_VOID(delete reinterpret_cast<DiffusionGemmaModel*>(model));
}

int32_t diffusion_gemma_add_dense_weight(void* model, mlx_array* w) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        m->weights.push_back(make_dense(w));
        return static_cast<int32_t>(m->weights.size() - 1);
    }());
}

int32_t diffusion_gemma_add_affine_weight(
    void* model,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,
    int32_t group_size,
    int32_t bits) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        m->weights.push_back(make_affine(w, scales, biases, group_size, bits));
        return static_cast<int32_t>(m->weights.size() - 1);
    }());
}

void diffusion_gemma_set_config(
    void* model,
    int32_t hidden_size,
    int32_t vocab_size,
    float rms_eps,
    float final_logit_softcap) {
    MLX_TRY_VOID([&]() {
        auto* m = as_model(model);
        m->hidden_size = hidden_size;
        m->vocab_size = vocab_size;
        m->rms_eps = rms_eps;
        m->final_logit_softcap = final_logit_softcap;
    }());
}

void diffusion_gemma_set_embed(void* model, int32_t embed_id, int32_t final_norm_id) {
    MLX_TRY_VOID([&]() {
        auto* m = as_model(model);
        m->embed = m->weight_by_id(embed_id);
        if (!m->embed.is_dense) {
            throw std::invalid_argument("DiffusionGemma embed must be registered dense/dequantized");
        }
        m->embed_tokens = m->embed.w;
        m->lm_head_t = transpose(m->embed_tokens);
        m->final_norm = m->array_by_id(final_norm_id);
    }());
}

int32_t diffusion_gemma_push_layer(
    void* model,
    bool is_full_attention,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t rotary_dim,
    float rope_theta,
    int32_t sliding_window,
    int32_t input_ln_id,
    int32_t q_id,
    int32_t k_id,
    int32_t v_id,
    int32_t o_id,
    int32_t q_norm_id,
    int32_t k_norm_id,
    int32_t post_attn_ln_id,
    int32_t pre_ff_ln_id,
    int32_t gate_id,
    int32_t up_id,
    int32_t down_id,
    int32_t post_ff_ln_id,
    int32_t pre_ff2_ln_id,
    int32_t post_ff1_ln_id,
    int32_t post_ff2_ln_id,
    int32_t router_id,
    int32_t router_scale_id,
    int32_t per_expert_scale_id,
    int32_t expert_gate_up_id,
    int32_t expert_down_id,
    int32_t layer_scalar_id,
    int32_t num_experts,
    int32_t top_k) {
    return catch_to_rc([&]() {
        auto* m = as_model(model);
        LayerWeights layer;
        layer.is_full_attention = is_full_attention;
        layer.num_heads = num_heads;
        layer.num_kv_heads = num_kv_heads;
        layer.head_dim = head_dim;
        layer.rotary_dim = rotary_dim;
        layer.rope_theta = rope_theta;
        layer.sliding_window = sliding_window;
        layer.input_ln = m->array_by_id(input_ln_id);
        layer.q_proj = m->weight_by_id(q_id);
        layer.k_proj = m->weight_by_id(k_id);
        layer.k_eq_v = v_id < 0;
        if (!layer.k_eq_v) {
            layer.v_proj = m->weight_by_id(v_id);
        }
        layer.o_proj = m->weight_by_id(o_id);
        layer.q_norm = m->array_by_id(q_norm_id);
        layer.k_norm = m->array_by_id(k_norm_id);
        layer.post_attn_ln = m->array_by_id(post_attn_ln_id);
        layer.pre_ff_ln = m->array_by_id(pre_ff_ln_id);
        layer.gate_proj = m->weight_by_id(gate_id);
        layer.up_proj = m->weight_by_id(up_id);
        layer.down_proj = m->weight_by_id(down_id);
        layer.post_ff_ln = m->array_by_id(post_ff_ln_id);
        layer.layer_scalar = m->array_by_id(layer_scalar_id);

        layer.has_moe = router_id >= 0;
        if (layer.has_moe) {
            layer.pre_ff2_ln = m->array_by_id(pre_ff2_ln_id);
            layer.post_ff1_ln = m->array_by_id(post_ff1_ln_id);
            layer.post_ff2_ln = m->array_by_id(post_ff2_ln_id);
            layer.router = m->weight_by_id(router_id);
            layer.router_scale = m->array_by_id(router_scale_id);
            layer.per_expert_scale = m->array_by_id(per_expert_scale_id);
            layer.expert_gate_up = m->weight_by_id(expert_gate_up_id);
            layer.expert_down = m->weight_by_id(expert_down_id);
            layer.num_experts = num_experts;
            layer.top_k = top_k;
        }
        m->layers.push_back(layer);
    });
}

int32_t diffusion_gemma_set_self_conditioning(
    void* model,
    int32_t pre_norm_id,
    int32_t gate_id,
    int32_t up_id,
    int32_t down_id) {
    return catch_to_rc([&]() {
        auto* m = as_model(model);
        m->self_conditioning.pre_norm = m->array_by_id(pre_norm_id);
        m->self_conditioning.gate_proj = m->weight_by_id(gate_id);
        m->self_conditioning.up_proj = m->weight_by_id(up_id);
        m->self_conditioning.down_proj = m->weight_by_id(down_id);
        m->self_conditioning.ready = true;
    });
}

int32_t diffusion_gemma_finalize(void* model) {
    return catch_to_rc([&]() {
        auto* m = as_model(model);
        if (m->hidden_size <= 0 || m->vocab_size <= 0) {
            throw std::runtime_error("DiffusionGemma config was not set");
        }
        if (m->layers.empty()) {
            throw std::runtime_error("DiffusionGemma has no registered layers");
        }
        if (m->embed_tokens.ndim() != 2 || m->final_norm.ndim() == 0) {
            throw std::runtime_error("DiffusionGemma embed/final norm not registered");
        }
        if (!m->self_conditioning.ready) {
            throw std::runtime_error("DiffusionGemma self-conditioning weights not registered");
        }
        m->finalized = true;
    });
}

int32_t diffusion_gemma_begin_request(void* model, uint64_t seed) {
    return catch_to_rc([&]() {
        auto* m = as_model(model);
        random::seed(seed);
        m->context_tokens.clear();
        m->self_conditioning_len = 0;
        m->self_conditioning_embeds = array(0);
    });
}

int32_t diffusion_gemma_prefill(void* model, const int32_t* tokens, int32_t len) {
    return catch_to_rc([&]() {
        auto* m = as_model(model);
        if (len < 0 || (len > 0 && tokens == nullptr)) {
            throw std::invalid_argument("DiffusionGemma prefill received invalid token buffer");
        }
        if (len == 0) {
            m->context_tokens.clear();
        } else {
            m->context_tokens.assign(tokens, tokens + len);
        }
        m->self_conditioning_len = 0;
        m->self_conditioning_embeds = array(0);
    });
}

int32_t diffusion_gemma_commit(void* model, const int32_t* tokens, int32_t len) {
    return catch_to_rc([&]() {
        auto* m = as_model(model);
        if (len < 0 || (len > 0 && tokens == nullptr)) {
            throw std::invalid_argument("DiffusionGemma commit received invalid token buffer");
        }
        if (len > 0) {
            m->context_tokens.insert(m->context_tokens.end(), tokens, tokens + len);
        }
        m->self_conditioning_len = 0;
        m->self_conditioning_embeds = array(0);
    });
}

int32_t diffusion_gemma_predict_canvas(
    void* model,
    const int32_t* canvas,
    int32_t canvas_len,
    int32_t valid_len,
    int32_t /*step*/,
    float temperature,
    uint32_t* out_sampled,
    uint32_t* out_argmax,
    float* out_entropy) {
    return catch_to_rc([&]() {
        if (out_sampled == nullptr || out_argmax == nullptr || out_entropy == nullptr) {
            throw std::invalid_argument("DiffusionGemma prediction outputs must be non-null");
        }
        auto* m = as_model(model);
        auto logits = m->forward_logits(canvas, canvas_len, valid_len);
        auto log_z = logsumexp(logits, -1, true);
        auto log_probs = logits - log_z;
        auto probs = exp(log_probs);
        auto entropy = -sum(probs * log_probs, -1, false);
        auto argmax_tokens = astype(argmax(logits, -1, false), int32);
        array sampled_tokens(0);
        if (temperature <= 0.0f) {
            sampled_tokens = argmax_tokens;
        } else {
            sampled_tokens = astype(random::categorical(logits / array(temperature), -1), int32);
        }

        m->update_self_conditioning(probs);

        auto sampled_c = contiguous(sampled_tokens);
        auto argmax_c = contiguous(argmax_tokens);
        auto entropy_c = contiguous(astype(entropy, float32));
        eval({sampled_c, argmax_c, entropy_c, m->self_conditioning_embeds});

        auto sampled_ptr = sampled_c.data<int32_t>();
        auto argmax_ptr = argmax_c.data<int32_t>();
        auto entropy_ptr = entropy_c.data<float>();
        for (int i = 0; i < valid_len; ++i) {
            out_sampled[i] = static_cast<uint32_t>(sampled_ptr[i]);
            out_argmax[i] = static_cast<uint32_t>(argmax_ptr[i]);
            out_entropy[i] = entropy_ptr[i];
        }
        for (int i = valid_len; i < canvas_len; ++i) {
            out_sampled[i] = 0;
            out_argmax[i] = 0;
            out_entropy[i] = 0.0f;
        }
    });
}

} // extern "C"

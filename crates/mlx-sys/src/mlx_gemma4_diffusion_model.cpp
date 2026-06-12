//! DiffusionGemma / Gemma4 MLX forward model.
//!
//! The prompt / committed blocks are encoded into per-layer KV caches, and each
//! denoise pass runs only the bidirectional canvas decoder against that cache.

#include "mlx_common.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
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

std::vector<array> geglu_impl(const std::vector<array>& inputs) {
    return {gelu_tanh(inputs[0]) * inputs[1]};
}

auto& compiled_geglu() {
    static auto fn = mlx::core::compile(geglu_impl, true /*shapeless*/);
    return fn;
}

array geglu(const array& gate, const array& up) {
    return compiled_geglu()({gate, up})[0];
}

std::vector<array> softcap_impl(const std::vector<array>& inputs) {
    auto logits = astype(inputs[0], float32);
    auto cap = inputs[1];
    return {tanh(logits / cap) * cap};
}

auto& compiled_softcap() {
    static auto fn = mlx::core::compile(softcap_impl, true /*shapeless*/);
    return fn;
}

array softcap_logits(const array& logits, float cap) {
    if (cap <= 0.0f) {
        return logits;
    }
    return compiled_softcap()({logits, array(cap)})[0];
}

std::vector<array> entropy_probs_impl(const std::vector<array>& inputs) {
    auto logits = astype(inputs[0], float32);
    auto log_z = logsumexp(logits, -1, true);
    auto log_probs = logits - log_z;
    auto probs = exp(log_probs);
    auto entropy = -sum(probs * log_probs, -1, false);
    return {probs, entropy};
}

auto& compiled_entropy_probs() {
    static auto fn = mlx::core::compile(entropy_probs_impl, false /*shapeless*/);
    return fn;
}

std::vector<array> entropy_probs(const array& logits) {
    return compiled_entropy_probs()({logits});
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

struct LayerCache {
    array keys = array(0);
    array values = array(0);
    int len = 0;
};

struct SortedSwitchInputs {
    array x = array(0);
    array indices = array(0);
    array inv_order = array(0);
};

struct DiffusionGenerateResult {
    int tokens = 0;
    int blocks = 0;
    int steps = 0;
    int forced = 0;
    int adaptive = 0;
    int finish = 0;
};

SortedSwitchInputs gather_sort_switch_inputs(const array& x, const array& indices) {
    const auto& shape = indices.shape();
    const int last_dim = shape.back();
    auto flat_indices = flatten(indices);
    auto order = astype(argsort(flat_indices), int32);
    auto inv_order = astype(argsort(order), int32);
    auto rows = floor_divide(order, array(last_dim, int32));
    auto flat_x = flatten(x, 0, -3);
    return {
        take(flat_x, rows, 0),
        take(flat_indices, order, 0),
        inv_order,
    };
}

array scatter_unsort_switch_outputs(
    const array& x,
    const array& inv_order,
    const Shape& indices_shape) {
    auto unsorted = take(x, inv_order, 0);
    return unflatten(unsorted, 0, indices_shape);
}

struct DiffusionGemmaModel {
    int hidden_size = 0;
    int vocab_size = 0;
    float rms_eps = 1e-6f;
    float final_logit_softcap = 0.0f;

    std::vector<QWeight> weights;
    QWeight embed;
    array embed_tokens = array(0);
    QWeight lm_head;
    array final_norm = array(0);
    std::vector<LayerWeights> layers;
    SelfConditioningWeights self_conditioning;

    std::vector<int32_t> context_tokens;
    std::vector<LayerCache> layer_caches;
    int context_len = 0;
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

    void reset_request_state() {
        context_tokens.clear();
        context_len = 0;
        layer_caches.assign(layers.size(), LayerCache{});
        self_conditioning_len = 0;
        self_conditioning_embeds = array(0);
    }

    array build_encoder_mask(
        int query_len,
        int retained_past_len,
        int absolute_offset,
        const LayerWeights& layer) const {
        const int key_len = retained_past_len + query_len;
        const int first_key_abs = absolute_offset - retained_past_len;
        std::vector<float> mask(static_cast<size_t>(query_len) * key_len, -1.0e9f);
        for (int i = 0; i < query_len; ++i) {
            const int q_pos = absolute_offset + i;
            for (int j = 0; j < key_len; ++j) {
                const int key_pos = first_key_abs + j;
                bool allowed = key_pos <= q_pos;
                if (allowed && !layer.is_full_attention && layer.sliding_window > 0) {
                    allowed = q_pos < key_pos + layer.sliding_window;
                }
                if (allowed) {
                    mask[static_cast<size_t>(i) * key_len + j] = 0.0f;
                }
            }
        }
        return array_from_f32(mask, Shape{1, 1, query_len, key_len});
    }

    struct AttentionProjection {
        array q = array(0);
        array k = array(0);
        array v = array(0);
    };

    AttentionProjection project_attention(const array& x, const LayerWeights& layer, int offset) const {
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

        q = fast::rope(q, layer.rotary_dim, false, layer.rope_theta, 1.0f, offset);
        k = fast::rope(k, layer.rotary_dim, false, layer.rope_theta, 1.0f, offset);

        auto v = reshape(v_raw, {1, s, nkv, hd});
        v = rms_no_weight(v);
        v = transpose(v, {0, 2, 1, 3});
        return AttentionProjection{q, k, v};
    }

    array attention_encode(
        const array& x,
        const LayerWeights& layer,
        LayerCache& cache,
        int offset) const {
        const int s = x.shape(0);
        const int nh = layer.num_heads;
        const int nkv = layer.num_kv_heads;
        const int hd = layer.head_dim;
        auto proj = project_attention(x, layer, offset);
        const int retained_past_len = cache.len;

        array k_full = proj.k;
        array v_full = proj.v;
        if (cache.len > 0) {
            k_full = concatenate(std::vector<array>{cache.keys, proj.k}, 2);
            v_full = concatenate(std::vector<array>{cache.values, proj.v}, 2);
        }
        auto mask = build_encoder_mask(s, retained_past_len, offset, layer);
        // Gemma4 folds query scaling into q_norm weights; attention scale is literally 1.
        float scale = 1.0f;
        auto attn = fast::scaled_dot_product_attention(proj.q, k_full, v_full, scale, "", mask);

        int full_len = retained_past_len + s;
        if (!layer.is_full_attention && layer.sliding_window > 0 && full_len > layer.sliding_window) {
            const int start = full_len - layer.sliding_window;
            k_full = slice(k_full, Shape{0, 0, start, 0}, Shape{1, nkv, full_len, hd});
            v_full = slice(v_full, Shape{0, 0, start, 0}, Shape{1, nkv, full_len, hd});
            full_len = layer.sliding_window;
        }
        cache.keys = k_full;
        cache.values = v_full;
        cache.len = full_len;

        auto flat = reshape(transpose(attn, {0, 2, 1, 3}), {1, s, nh * hd});
        auto out = layer.o_proj.apply(flat);
        return reshape(out, {s, hidden_size});
    }

    array attention_decode(
        const array& x,
        const LayerWeights& layer,
        const LayerCache& cache,
        int offset) const {
        const int s = x.shape(0);
        const int nh = layer.num_heads;
        const int nkv = layer.num_kv_heads;
        const int hd = layer.head_dim;
        auto proj = project_attention(x, layer, offset);

        array k_prefix = cache.keys;
        array v_prefix = cache.values;
        int prefix_len = cache.len;
        if (!layer.is_full_attention && layer.sliding_window > 0) {
            const int keep = std::max(layer.sliding_window - 1, 0);
            if (keep == 0) {
                prefix_len = 0;
            } else if (prefix_len > keep) {
                const int start = prefix_len - keep;
                k_prefix = slice(k_prefix, Shape{0, 0, start, 0}, Shape{1, nkv, prefix_len, hd});
                v_prefix = slice(v_prefix, Shape{0, 0, start, 0}, Shape{1, nkv, prefix_len, hd});
                prefix_len = keep;
            }
        }

        array k_full = proj.k;
        array v_full = proj.v;
        if (prefix_len > 0) {
            k_full = concatenate(std::vector<array>{k_prefix, proj.k}, 2);
            v_full = concatenate(std::vector<array>{v_prefix, proj.v}, 2);
        }
        // Decoder canvas positions are bidirectional. Prefix pruning above
        // makes the sliding-attention mask all-true for the retained keys.
        float scale = 1.0f;
        auto attn = fast::scaled_dot_product_attention(proj.q, k_full, v_full, scale, "");
        auto flat = reshape(transpose(attn, {0, 2, 1, 3}), {1, s, nh * hd});
        auto out = layer.o_proj.apply(flat);
        return reshape(out, {s, hidden_size});
    }

    array dense_mlp(const array& x, const LayerWeights& layer) const {
        auto gate = layer.gate_proj.apply(x);
        auto up = layer.up_proj.apply(x);
        auto h = geglu(gate, up);
        return layer.down_proj.apply(h);
    }

    array expert_switch(const array& x, const array& inds, const LayerWeights& layer) const {
        auto x5 = expand_dims(x, std::vector<int>{-2, -3});
        // Match mlx-vlm diffusion_gemma: sort large route sets by expert id so
        // gather_qmm can use the coalesced `sorted_indices` path.
        const bool do_sort = inds.size() >= 64;
        auto idx = inds;
        array inv_order(0);
        if (do_sort) {
            auto sorted = gather_sort_switch_inputs(x5, inds);
            x5 = sorted.x;
            idx = sorted.indices;
            inv_order = sorted.inv_order;
        }
        auto gate_up = gather_qmm(
            x5,
            layer.expert_gate_up.w,
            layer.expert_gate_up.scales,
            layer.expert_gate_up.biases,
            std::nullopt,
            idx,
            true,
            layer.expert_gate_up.group_size,
            layer.expert_gate_up.bits,
            "affine",
            do_sort);
        auto parts = split(gate_up, Shape{gate_up.shape(gate_up.ndim() - 1) / 2}, -1);
        auto h = geglu(parts[0], parts[1]);
        auto y = gather_qmm(
            h,
            layer.expert_down.w,
            layer.expert_down.scales,
            layer.expert_down.biases,
            std::nullopt,
            idx,
            true,
            layer.expert_down.group_size,
            layer.expert_down.bits,
            "affine",
            do_sort);
        if (do_sort) {
            y = scatter_unsort_switch_outputs(y, inv_order, inds.shape());
        }
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

        const int kth = layer.num_experts - layer.top_k;
        auto part = argpartition(logits, kth, -1);
        Shape start(part.ndim(), 0);
        Shape stop = part.shape();
        Shape strides(part.ndim(), 1);
        start[part.ndim() - 1] = kth;
        auto inds = slice(part, start, stop, strides);

        auto scores = take_along_axis(logits, inds, -1);
        scores = softmax(scores, -1, true);
        auto expert_scales = take(layer.per_expert_scale, inds, 0);
        scores = scores * expert_scales;
        if (scores.dtype() != expert_input.dtype()) {
            scores = astype(scores, expert_input.dtype());
        }

        auto y = expert_switch(expert_input, inds, layer);
        return sum(y * expand_dims(scores, -1), -2, false);
    }

    array apply_self_conditioning(const array& x, int valid_len) const {
        if (!self_conditioning.ready || self_conditioning_len != valid_len || valid_len <= 0) {
            return x;
        }
        auto sc_x = rms(self_conditioning_embeds, self_conditioning.pre_norm);
        auto gate = self_conditioning.gate_proj.apply(sc_x);
        auto up = self_conditioning.up_proj.apply(sc_x);
        auto signal = self_conditioning.down_proj.apply(geglu(gate, up));
        return rms_no_weight(x + signal);
    }

    void encode_tokens_into(
        const int32_t* tokens,
        int len,
        std::vector<LayerCache>& caches,
        int offset) const {
        if (len <= 0) {
            return;
        }
        if (!finalized) {
            throw std::runtime_error("DiffusionGemma model was not finalized");
        }
        if (caches.size() != layers.size()) {
            throw std::runtime_error("DiffusionGemma cache size mismatch");
        }
        auto token_ids = array_from_i32(tokens, static_cast<int32_t>(len));
        auto x = take(embed_tokens, token_ids, 0) * array(std::sqrt(static_cast<float>(hidden_size)));

        for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
            const auto& layer = layers[layer_idx];
            auto residual = x;
            auto h = rms(x, layer.input_ln);
            h = attention_encode(h, layer, caches[layer_idx], offset);
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
    }

    array forward_logits_from_ids(const array& token_ids) {
        if (!finalized) {
            throw std::runtime_error("DiffusionGemma model was not finalized");
        }
        const int valid_len = token_ids.shape(0);
        if (valid_len <= 0) {
            throw std::invalid_argument("invalid DiffusionGemma canvas length");
        }
        if (layer_caches.size() != layers.size()) {
            throw std::runtime_error("DiffusionGemma cache was not initialized");
        }
        auto x = take(embed_tokens, token_ids, 0) * array(std::sqrt(static_cast<float>(hidden_size)));
        x = apply_self_conditioning(x, valid_len);

        for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
            const auto& layer = layers[layer_idx];
            const auto& cache = layer_caches[layer_idx];
            auto residual = x;
            auto h = rms(x, layer.input_ln);
            h = attention_decode(h, layer, cache, context_len);
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
        auto logits = lm_head.apply(x);
        return softcap_logits(logits, final_logit_softcap);
    }

    array forward_logits(const int32_t* canvas, int canvas_len, int valid_len) {
        if (canvas_len <= 0 || valid_len <= 0 || valid_len > canvas_len) {
            throw std::invalid_argument("invalid DiffusionGemma canvas length");
        }
        auto token_ids = array_from_i32(canvas, static_cast<int32_t>(valid_len));
        return forward_logits_from_ids(token_ids);
    }

    void update_self_conditioning(const array& probs) {
        auto soft = matmul(probs, embed_tokens) * array(std::sqrt(static_cast<float>(hidden_size)));
        self_conditioning_embeds = soft;
        self_conditioning_len = soft.shape(0);
    }

    array random_canvas(int len) const {
        return random::randint(0, vocab_size, Shape{len}, int32);
    }

    array entropy_acceptance_mask(const array& entropy, float entropy_bound) const {
        auto sorted_indices = argsort(entropy, -1);
        auto sorted_entropy = take_along_axis(entropy, sorted_indices, -1);
        auto cumulative_entropy = cumsum(sorted_entropy, -1);
        auto cumulative_max = cummax(sorted_entropy, -1);
        auto sorted_selection = less_equal(
            cumulative_entropy - cumulative_max,
            array(entropy_bound));
        return put_along_axis(zeros_like(sorted_selection), sorted_indices, sorted_selection, -1);
    }

    void commit_tokens_checked(const int32_t* tokens, int len) {
        if (len <= 0) {
            return;
        }
        auto next_caches = layer_caches;
        encode_tokens_into(tokens, len, next_caches, context_len);
        layer_caches = std::move(next_caches);
        context_tokens.insert(context_tokens.end(), tokens, tokens + len);
        context_len += len;
        self_conditioning_len = 0;
        self_conditioning_embeds = array(0);
    }

    DiffusionGenerateResult generate_into(
        const int32_t* prompt,
        int prompt_len,
        int max_new_tokens,
        int canvas_len,
        int max_steps,
        float entropy_bound,
        float confidence_threshold,
        float t_min,
        float t_max,
        int stability_threshold,
        uint64_t seed,
        const uint32_t* stop_ids,
        int stop_ids_len,
        uint32_t* out_tokens) {
        using Clock = std::chrono::steady_clock;
        const bool profile = std::getenv("ARLE_DIFFUSION_CPP_PROFILE") != nullptr;
        const auto t0 = Clock::now();
        double prefill_ms = 0.0;
        double denoise_ms = 0.0;
        double final_ms = 0.0;
        if (prompt_len < 0 || (prompt_len > 0 && prompt == nullptr)) {
            throw std::invalid_argument("DiffusionGemma generate received invalid prompt");
        }
        if (max_new_tokens < 0 || canvas_len <= 0 || max_steps <= 0) {
            throw std::invalid_argument("DiffusionGemma generate received invalid config");
        }
        if (max_new_tokens > 0 && out_tokens == nullptr) {
            throw std::invalid_argument("DiffusionGemma generate output buffer is null");
        }
        if (stop_ids_len < 0 || (stop_ids_len > 0 && stop_ids == nullptr)) {
            throw std::invalid_argument("DiffusionGemma generate received invalid stop ids");
        }

        random::seed(seed);
        reset_request_state();
        const auto prefill_start = Clock::now();
        if (prompt_len > 0) {
            auto next_caches = std::vector<LayerCache>(layers.size());
            encode_tokens_into(prompt, prompt_len, next_caches, 0);
            layer_caches = std::move(next_caches);
            context_tokens.assign(prompt, prompt + prompt_len);
            context_len = prompt_len;
            std::vector<array> cache_arrays;
            cache_arrays.reserve(layer_caches.size() * 2);
            for (const auto& cache : layer_caches) {
                if (cache.len > 0) {
                    cache_arrays.push_back(cache.keys);
                    cache_arrays.push_back(cache.values);
                }
            }
            if (!cache_arrays.empty()) {
                eval(cache_arrays);
            }
        }
        prefill_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - prefill_start).count();

        DiffusionGenerateResult result;
        result.finish = 0;
        std::vector<int32_t> output;
        output.reserve(static_cast<size_t>(max_new_tokens));
        const int stable_need = std::max(stability_threshold, 1);

        while (static_cast<int>(output.size()) < max_new_tokens) {
            const int remaining = max_new_tokens - static_cast<int>(output.size());
            const int valid_len = std::min(remaining, canvas_len);
            auto current_canvas = random_canvas(valid_len);
            array previous_argmax(0);
            bool have_previous = false;

            for (int step = 0; step < max_steps; ++step) {
                const auto step_start = Clock::now();
                const int remaining_steps = std::max(max_steps - step, 1);
                const float schedule_temperature =
                    t_min + (t_max - t_min) *
                    (static_cast<float>(remaining_steps) / static_cast<float>(max_steps));
                auto logits = forward_logits_from_ids(current_canvas);
                if (schedule_temperature > 0.0f) {
                    logits = logits / array(schedule_temperature);
                }
                auto argmax_tokens = astype(argmax(logits, -1, false), int32);
                const bool forced = step + 1 >= max_steps;
                result.steps += 1;

                if (forced) {
                    previous_argmax = argmax_tokens;
                    result.forced += 1;
                    denoise_ms += std::chrono::duration<double, std::milli>(
                        Clock::now() - step_start).count();
                    break;
                }

                auto probs_entropy = entropy_probs(logits);
                auto probs = probs_entropy[0];
                auto entropy = probs_entropy[1];
                auto acceptance = entropy_acceptance_mask(entropy, entropy_bound);
                auto renoise = random_canvas(valid_len);
                current_canvas = where(acceptance, argmax_tokens, renoise);

                auto mean_entropy = mean(entropy);
                bool stable = false;
                if (have_previous && stable_need <= 1) {
                    stable = all(equal(argmax_tokens, previous_argmax)).item<bool>();
                }
                bool confident = mean_entropy.item<float>() < confidence_threshold;
                update_self_conditioning(probs);
                if (stable && confident) {
                    previous_argmax = argmax_tokens;
                    result.adaptive += 1;
                    denoise_ms += std::chrono::duration<double, std::milli>(
                        Clock::now() - step_start).count();
                    break;
                }
                previous_argmax = argmax_tokens;
                have_previous = true;
                denoise_ms += std::chrono::duration<double, std::milli>(
                    Clock::now() - step_start).count();
            }

            const auto final_start = Clock::now();
            auto final_canvas = contiguous(previous_argmax);
            eval({final_canvas});
            const auto* final_ptr = final_canvas.data<int32_t>();
            int commit_len = valid_len;
            bool stopped = false;
            for (int i = 0; i < valid_len; ++i) {
                const uint32_t token = static_cast<uint32_t>(final_ptr[i]);
                for (int j = 0; j < stop_ids_len; ++j) {
                    if (token == stop_ids[j]) {
                        commit_len = i + 1;
                        stopped = true;
                        break;
                    }
                }
                if (stopped) {
                    break;
                }
            }
            std::vector<int32_t> commit_tokens(final_ptr, final_ptr + commit_len);
            const bool needs_next_block =
                !stopped &&
                static_cast<int>(output.size()) + commit_len < max_new_tokens;
            if (needs_next_block) {
                commit_tokens_checked(commit_tokens.data(), commit_len);
            }
            output.insert(output.end(), commit_tokens.begin(), commit_tokens.end());
            result.blocks += 1;
            if (stopped) {
                result.finish = 1;
                break;
            }
            if (result.blocks > max_new_tokens) {
                break;
            }
            final_ms += std::chrono::duration<double, std::milli>(
                Clock::now() - final_start).count();
        }

        if (static_cast<int>(output.size()) > max_new_tokens) {
            output.resize(static_cast<size_t>(max_new_tokens));
        }
        for (size_t i = 0; i < output.size(); ++i) {
            out_tokens[i] = static_cast<uint32_t>(output[i]);
        }
        result.tokens = static_cast<int>(output.size());
        if (profile) {
            const double total_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - t0).count();
            std::cerr
                << "diffusion cpp profile: prompt_tokens=" << prompt_len
                << " generated_tokens=" << result.tokens
                << " blocks=" << result.blocks
                << " steps=" << result.steps
                << " prefill_ms=" << prefill_ms
                << " denoise_ms=" << denoise_ms
                << " final_commit_ms=" << final_ms
                << " total_ms=" << total_ms
                << std::endl;
        }
        return result;
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

void diffusion_gemma_set_embed(
    void* model,
    int32_t embed_id,
    int32_t lm_head_id,
    int32_t final_norm_id) {
    MLX_TRY_VOID([&]() {
        auto* m = as_model(model);
        m->embed = m->weight_by_id(embed_id);
        if (!m->embed.is_dense) {
            throw std::invalid_argument("DiffusionGemma embed must be registered dense/dequantized");
        }
        m->embed_tokens = m->embed.w;
        m->lm_head = m->weight_by_id(lm_head_id);
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
        m->layer_caches.assign(m->layers.size(), LayerCache{});
        m->finalized = true;
    });
}

int32_t diffusion_gemma_begin_request(void* model, uint64_t seed) {
    return catch_to_rc([&]() {
        auto* m = as_model(model);
        random::seed(seed);
        m->reset_request_state();
    });
}

int32_t diffusion_gemma_prefill(void* model, const int32_t* tokens, int32_t len) {
    return catch_to_rc([&]() {
        auto* m = as_model(model);
        if (len < 0 || (len > 0 && tokens == nullptr)) {
            throw std::invalid_argument("DiffusionGemma prefill received invalid token buffer");
        }
        m->reset_request_state();
        if (len == 0) {
            m->context_tokens.clear();
        } else {
            auto next_caches = std::vector<LayerCache>(m->layers.size());
            m->encode_tokens_into(tokens, len, next_caches, 0);
            m->layer_caches = std::move(next_caches);
            m->context_tokens.assign(tokens, tokens + len);
            m->context_len = len;
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
            auto next_caches = m->layer_caches;
            m->encode_tokens_into(tokens, len, next_caches, m->context_len);
            m->layer_caches = std::move(next_caches);
            m->context_tokens.insert(m->context_tokens.end(), tokens, tokens + len);
            m->context_len += len;
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
        if (temperature > 0.0f) {
            logits = logits / array(temperature);
        }
        auto probs_entropy = entropy_probs(logits);
        auto probs = probs_entropy[0];
        auto entropy = probs_entropy[1];
        auto argmax_tokens = astype(argmax(logits, -1, false), int32);
        auto sampled_tokens = argmax_tokens;

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

int32_t diffusion_gemma_generate(
    void* model,
    const int32_t* prompt,
    int32_t prompt_len,
    int32_t max_new_tokens,
    int32_t canvas_len,
    int32_t max_steps,
    float entropy_bound,
    float confidence_threshold,
    float t_min,
    float t_max,
    int32_t stability_threshold,
    uint64_t seed,
    const uint32_t* stop_ids,
    int32_t stop_ids_len,
    uint32_t* out_tokens,
    int32_t* out_len,
    int32_t* out_finish,
    int32_t* out_blocks,
    int32_t* out_steps,
    int32_t* out_forced,
    int32_t* out_adaptive) {
    return catch_to_rc([&]() {
        if (out_len == nullptr || out_finish == nullptr || out_blocks == nullptr ||
            out_steps == nullptr || out_forced == nullptr || out_adaptive == nullptr) {
            throw std::invalid_argument("DiffusionGemma generate stats outputs must be non-null");
        }
        auto* m = as_model(model);
        auto result = m->generate_into(
            prompt,
            prompt_len,
            max_new_tokens,
            canvas_len,
            max_steps,
            entropy_bound,
            confidence_threshold,
            t_min,
            t_max,
            stability_threshold,
            seed,
            stop_ids,
            stop_ids_len,
            out_tokens);
        *out_len = result.tokens;
        *out_finish = result.finish;
        *out_blocks = result.blocks;
        *out_steps = result.steps;
        *out_forced = result.forced;
        *out_adaptive = result.adaptive;
    });
}

} // extern "C"

#include "mlx_common.h"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace mlx::core;

namespace {

std::vector<array> swiglu_impl(const std::vector<array>& inputs) {
    auto gate = inputs[0];
    auto up = inputs[1];
    return {(gate * sigmoid(gate)) * up};
}

auto& compiled_swiglu() {
    static auto fn = mlx::core::compile(swiglu_impl, true /* shapeless */);
    return fn;
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

QWeight make_weight(
    mlx_array* w,
    mlx_array* s,
    mlx_array* b,
    int32_t group_size,
    int32_t bits
) {
    if (w == nullptr) {
        throw std::runtime_error("DFlash draft weight pointer must not be null");
    }
    if (s == nullptr || b == nullptr || bits == 0) {
        return {*to_arr(w), array(0), array(0), 0, 0, true};
    }
    return {*to_arr(w), *to_arr(s), *to_arr(b), group_size, bits, false};
}

struct DraftLayerWeights {
    QWeight q_proj;
    QWeight k_proj;
    QWeight v_proj;
    QWeight o_proj;
    QWeight gate_proj;
    QWeight up_proj;
    QWeight down_proj;
    array input_layernorm = array(0);
    array post_attention_layernorm = array(0);
    array q_norm = array(0);
    array k_norm = array(0);
};

struct DFlashDraftModel {
    std::vector<DraftLayerWeights> layers;

    QWeight fc;
    array hidden_norm = array(0);
    array norm = array(0);

    int hidden_size = 2560;
    int num_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 128;
    int num_layers = 5;
    float rope_theta = 1e6f;
    float rms_eps = 1e-6f;

    std::function<std::vector<array>(const std::vector<array>&)> compiled_forward;
    std::function<std::vector<array>(const std::vector<array>&)> compiled_forward_batched;
    // Separate compile slot for the mixed-length batched path: the mask is
    // threaded as the LAST input so the compiled graph never captures it by
    // reference across compile boundaries (mirrors the verify_block_batched
    // pattern in mlx_qwen35_model.cpp, but adapted because draft IS compiled
    // while verify is not).
    std::function<std::vector<array>(const std::vector<array>&)> compiled_forward_batched_masked;
    bool is_compiled = false;

    // Set by the FFI entrypoint when the caller passes a non-null attn_mask.
    // Read inside forward_batched() to pick which compiled slot to dispatch.
    // The actual mask array is appended to the inputs vector — never captured
    // via this member by the compiled lambda.
    bool current_has_attn_mask = false;

    // Keep inputs/outputs alive across the C boundary so MLX can safely retain
    // graph references until Rust forces materialization downstream.
    std::vector<array> prev_inputs;
    std::vector<array> prev_outputs;

    void set_config(
        int32_t hidden_size_,
        int32_t num_heads_,
        int32_t num_kv_heads_,
        int32_t head_dim_,
        int32_t num_layers_,
        float rope_theta_,
        float rms_eps_
    ) {
        hidden_size = hidden_size_;
        num_heads = num_heads_;
        num_kv_heads = num_kv_heads_;
        head_dim = head_dim_;
        num_layers = num_layers_;
        rope_theta = rope_theta_;
        rms_eps = rms_eps_;
    }

    void push_layer(DraftLayerWeights layer) {
        layers.push_back(std::move(layer));
    }

    void set_fc_norms(QWeight fc_, array hidden_norm_, array norm_) {
        fc = std::move(fc_);
        hidden_norm = std::move(hidden_norm_);
        norm = std::move(norm_);
    }

    std::vector<array> forward_impl(const std::vector<array>& inputs) const {
        const size_t expected_inputs = 4 + layers.size() * 2;
        if (inputs.size() != expected_inputs) {
            throw std::runtime_error("DFlash draft forward input count mismatch");
        }

        auto hidden_states = inputs[0];
        auto target_hidden = inputs[1];
        auto q_offset = inputs[2];
        auto k_offset = inputs[3];

        if (hidden_states.ndim() != 2 || target_hidden.ndim() != 2) {
            throw std::runtime_error("DFlash draft forward expects rank-2 noise/target inputs");
        }

        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        auto target_hidden_proj = fc.apply(target_hidden);
        target_hidden_proj = fast::rms_norm(target_hidden_proj, hidden_norm, rms_eps);

        std::vector<array> new_caches;
        new_caches.reserve(layers.size() * 2);
        size_t input_idx = 4;

        for (const auto& layer : layers) {
            auto residual = hidden_states;
            auto normed_hidden = fast::rms_norm(hidden_states, layer.input_layernorm, rms_eps);

            auto q_raw = layer.q_proj.apply(normed_hidden);
            auto kv_states = concatenate({target_hidden_proj, normed_hidden}, 0);
            auto k_raw = layer.k_proj.apply(kv_states);
            auto v_raw = layer.v_proj.apply(kv_states);

            auto q = reshape(q_raw, {1, -1, num_heads, head_dim});
            q = fast::rms_norm(q, layer.q_norm, rms_eps);
            q = transpose(q, {0, 2, 1, 3});
            q = fast::rope(q, head_dim, false, rope_theta, 1.0f, q_offset);

            auto k = reshape(k_raw, {1, -1, num_kv_heads, head_dim});
            k = fast::rms_norm(k, layer.k_norm, rms_eps);
            k = transpose(k, {0, 2, 1, 3});
            k = fast::rope(k, head_dim, false, rope_theta, 1.0f, k_offset);

            auto v = reshape(v_raw, {1, -1, num_kv_heads, head_dim});
            v = transpose(v, {0, 2, 1, 3});

            auto new_k_cache = concatenate({inputs[input_idx++], k}, 2);
            auto new_v_cache = concatenate({inputs[input_idx++], v}, 2);

            auto attn = fast::scaled_dot_product_attention(
                q,
                new_k_cache,
                new_v_cache,
                attn_scale,
                "");
            attn = transpose(attn, {0, 2, 1, 3});
            attn = reshape(attn, {-1, num_heads * head_dim});
            attn = layer.o_proj.apply(attn);
            hidden_states = residual + attn;

            auto residual2 = hidden_states;
            auto post_norm = fast::rms_norm(hidden_states, layer.post_attention_layernorm, rms_eps);
            auto gate = layer.gate_proj.apply(post_norm);
            auto up = layer.up_proj.apply(post_norm);
            auto mlp = layer.down_proj.apply(compiled_swiglu()({gate, up})[0]);
            hidden_states = residual2 + mlp;

            new_caches.push_back(std::move(new_k_cache));
            new_caches.push_back(std::move(new_v_cache));
        }

        std::vector<array> outputs;
        outputs.reserve(1 + new_caches.size());
        outputs.push_back(fast::rms_norm(hidden_states, norm, rms_eps));
        for (auto& cache : new_caches) {
            outputs.push_back(std::move(cache));
        }
        return outputs;
    }

    std::vector<array> forward_batched_impl(
        const std::vector<array>& inputs,
        bool with_mask
    ) const {
        const size_t expected_inputs = 4 + layers.size() * 2 + (with_mask ? 1 : 0);
        if (inputs.size() != expected_inputs) {
            throw std::runtime_error("DFlash draft batched forward input count mismatch");
        }

        auto hidden_states = inputs[0];
        auto target_hidden = inputs[1];
        auto q_offsets = inputs[2];
        auto k_offsets = inputs[3];
        // Mask, if present, lives at the very end of `inputs` (after all KV
        // caches) so the input layout for the unmasked path stays unchanged.
        const array* attn_mask_ptr = with_mask ? &inputs.back() : nullptr;

        if (hidden_states.ndim() != 3 || target_hidden.ndim() != 3) {
            throw std::runtime_error(
                "DFlash draft batched forward expects rank-3 noise/target inputs");
        }
        if (q_offsets.ndim() != 1 || k_offsets.ndim() != 1) {
            throw std::runtime_error(
                "DFlash draft batched forward expects rank-1 q/k offset arrays");
        }

        const int B = hidden_states.shape(0);
        const int seq_len = hidden_states.shape(1);
        const int context_len = target_hidden.shape(1);
        if (target_hidden.shape(0) != B) {
            throw std::runtime_error("DFlash draft batched forward batch size mismatch");
        }
        if (q_offsets.shape(0) != B || k_offsets.shape(0) != B) {
            throw std::runtime_error("DFlash draft batched forward offset length mismatch");
        }

        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        auto target_hidden_proj = fc.apply(target_hidden);
        target_hidden_proj = fast::rms_norm(target_hidden_proj, hidden_norm, rms_eps);

        std::vector<array> new_caches;
        new_caches.reserve(layers.size() * 2);
        size_t input_idx = 4;

        for (const auto& layer : layers) {
            auto residual = hidden_states;
            auto normed_hidden = fast::rms_norm(hidden_states, layer.input_layernorm, rms_eps);

            auto q_raw = layer.q_proj.apply(normed_hidden);
            auto kv_states = concatenate({target_hidden_proj, normed_hidden}, 1);
            auto k_raw = layer.k_proj.apply(kv_states);
            auto v_raw = layer.v_proj.apply(kv_states);

            auto q = reshape(q_raw, {B, seq_len, num_heads, head_dim});
            q = fast::rms_norm(q, layer.q_norm, rms_eps);
            q = transpose(q, {0, 2, 1, 3});
            q = fast::rope(q, head_dim, false, rope_theta, 1.0f, q_offsets);

            const int total_len = context_len + seq_len;
            auto k = reshape(k_raw, {B, total_len, num_kv_heads, head_dim});
            k = fast::rms_norm(k, layer.k_norm, rms_eps);
            k = transpose(k, {0, 2, 1, 3});
            k = fast::rope(k, head_dim, false, rope_theta, 1.0f, k_offsets);

            auto v = reshape(v_raw, {B, total_len, num_kv_heads, head_dim});
            v = transpose(v, {0, 2, 1, 3});

            auto new_k_cache = concatenate({inputs[input_idx++], k}, 2);
            auto new_v_cache = concatenate({inputs[input_idx++], v}, 2);

            array attn(0);
            if (attn_mask_ptr != nullptr) {
                // Mixed-length packed batch: caller supplies an additive mask
                // that zeroes out left-padded KV columns per row. Causal-only
                // would attend to those padded slots and corrupt logits.
                attn = fast::scaled_dot_product_attention(
                    q,
                    new_k_cache,
                    new_v_cache,
                    attn_scale,
                    "",
                    *attn_mask_ptr);
            } else {
                attn = fast::scaled_dot_product_attention(
                    q,
                    new_k_cache,
                    new_v_cache,
                    attn_scale,
                    "");
            }
            attn = transpose(attn, {0, 2, 1, 3});
            attn = reshape(attn, {B, seq_len, num_heads * head_dim});
            attn = layer.o_proj.apply(attn);
            hidden_states = residual + attn;

            auto residual2 = hidden_states;
            auto post_norm = fast::rms_norm(hidden_states, layer.post_attention_layernorm, rms_eps);
            auto gate = layer.gate_proj.apply(post_norm);
            auto up = layer.up_proj.apply(post_norm);
            auto mlp = layer.down_proj.apply(compiled_swiglu()({gate, up})[0]);
            hidden_states = residual2 + mlp;

            new_caches.push_back(std::move(new_k_cache));
            new_caches.push_back(std::move(new_v_cache));
        }

        std::vector<array> outputs;
        outputs.reserve(1 + new_caches.size());
        outputs.push_back(fast::rms_norm(hidden_states, norm, rms_eps));
        for (auto& cache : new_caches) {
            outputs.push_back(std::move(cache));
        }
        return outputs;
    }

    void finalize() {
        if (static_cast<int>(layers.size()) != num_layers) {
            throw std::runtime_error("DFlash draft layer count mismatch at finalize");
        }
        compiled_forward = mlx::core::compile(
            [this](const std::vector<array>& inputs) { return this->forward_impl(inputs); },
            true /* shapeless */);
        is_compiled = true;
    }

    std::vector<array> forward(const std::vector<array>& inputs) {
        prev_inputs = inputs;
        if (is_compiled) {
            prev_outputs = compiled_forward(prev_inputs);
        } else {
            prev_outputs = forward_impl(prev_inputs);
        }
        return prev_outputs;
    }

    std::vector<array> forward_batched(const std::vector<array>& inputs) {
        prev_inputs = inputs;
        const bool with_mask = current_has_attn_mask;
        if (is_compiled) {
            if (with_mask) {
                if (!compiled_forward_batched_masked) {
                    compiled_forward_batched_masked = mlx::core::compile(
                        [this](const std::vector<array>& call_inputs) {
                            return this->forward_batched_impl(call_inputs, true);
                        },
                        true /* shapeless */);
                }
                prev_outputs = compiled_forward_batched_masked(prev_inputs);
            } else {
                if (!compiled_forward_batched) {
                    compiled_forward_batched = mlx::core::compile(
                        [this](const std::vector<array>& call_inputs) {
                            return this->forward_batched_impl(call_inputs, false);
                        },
                        true /* shapeless */);
                }
                prev_outputs = compiled_forward_batched(prev_inputs);
            }
        } else {
            prev_outputs = forward_batched_impl(prev_inputs, with_mask);
        }
        return prev_outputs;
    }
};

}  // namespace

extern "C" {

void* dflash_draft_new() {
    MLX_TRY_RETURN(new DFlashDraftModel());
}

void dflash_draft_free(void* model) {
    MLX_TRY_VOID(delete static_cast<DFlashDraftModel*>(model));
}

void dflash_draft_set_config(
    void* model,
    int32_t hidden_size,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t num_layers,
    float rope_theta,
    float rms_eps
) {
    MLX_TRY_VOID({
        auto* m = static_cast<DFlashDraftModel*>(model);
        m->set_config(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            rope_theta,
            rms_eps);
    });
}

void dflash_draft_push_layer(
    void* model,
    mlx_array* q_w, mlx_array* q_s, mlx_array* q_b, int32_t q_gs, int32_t q_bits,
    mlx_array* k_w, mlx_array* k_s, mlx_array* k_b, int32_t k_gs, int32_t k_bits,
    mlx_array* v_w, mlx_array* v_s, mlx_array* v_b, int32_t v_gs, int32_t v_bits,
    mlx_array* o_w, mlx_array* o_s, mlx_array* o_b, int32_t o_gs, int32_t o_bits,
    mlx_array* gate_w, mlx_array* gate_s, mlx_array* gate_b, int32_t gate_gs, int32_t gate_bits,
    mlx_array* up_w, mlx_array* up_s, mlx_array* up_b, int32_t up_gs, int32_t up_bits,
    mlx_array* down_w, mlx_array* down_s, mlx_array* down_b, int32_t down_gs, int32_t down_bits,
    mlx_array* input_norm,
    mlx_array* post_attn_norm,
    mlx_array* q_norm,
    mlx_array* k_norm
) {
    MLX_TRY_VOID({
        auto* m = static_cast<DFlashDraftModel*>(model);
        DraftLayerWeights layer;
        layer.q_proj = make_weight(q_w, q_s, q_b, q_gs, q_bits);
        layer.k_proj = make_weight(k_w, k_s, k_b, k_gs, k_bits);
        layer.v_proj = make_weight(v_w, v_s, v_b, v_gs, v_bits);
        layer.o_proj = make_weight(o_w, o_s, o_b, o_gs, o_bits);
        layer.gate_proj = make_weight(gate_w, gate_s, gate_b, gate_gs, gate_bits);
        layer.up_proj = make_weight(up_w, up_s, up_b, up_gs, up_bits);
        layer.down_proj = make_weight(down_w, down_s, down_b, down_gs, down_bits);
        layer.input_layernorm = *to_arr(input_norm);
        layer.post_attention_layernorm = *to_arr(post_attn_norm);
        layer.q_norm = *to_arr(q_norm);
        layer.k_norm = *to_arr(k_norm);
        m->push_layer(std::move(layer));
    });
}

void dflash_draft_set_fc_norms(
    void* model,
    mlx_array* fc_w,
    mlx_array* fc_s,
    mlx_array* fc_b,
    int32_t fc_gs,
    int32_t fc_bits,
    mlx_array* hidden_norm,
    mlx_array* norm
) {
    MLX_TRY_VOID({
        auto* m = static_cast<DFlashDraftModel*>(model);
        m->set_fc_norms(
            make_weight(fc_w, fc_s, fc_b, fc_gs, fc_bits),
            *to_arr(hidden_norm),
            *to_arr(norm));
    });
}

int32_t dflash_draft_finalize(void* model) {
    auto* m = static_cast<DFlashDraftModel*>(model);
    try {
        mlx_clear_error();
        m->finalize();
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

int32_t dflash_draft_forward(
    void* model,
    mlx_array* noise_embedding,
    mlx_array* target_hidden,
    mlx_array** kv_caches,
    int32_t n_kv,
    int32_t rope_offset,
    mlx_array** out_hidden,
    mlx_array** out_kv_caches
) {
    auto* m = static_cast<DFlashDraftModel*>(model);
    try {
        mlx_clear_error();

        if (n_kv != static_cast<int32_t>(m->layers.size() * 2)) {
            throw std::runtime_error("DFlash draft forward cache count mismatch");
        }

        const int32_t context_len = to_arr(target_hidden)->shape(0);
        const int32_t q_offset_value = rope_offset + context_len;
        int32_t q_offset_data[1] = {q_offset_value};
        int32_t k_offset_data[1] = {rope_offset};

        std::vector<array> inputs;
        inputs.reserve(4 + static_cast<size_t>(n_kv));
        inputs.push_back(*to_arr(noise_embedding));
        inputs.push_back(*to_arr(target_hidden));
        inputs.emplace_back(q_offset_data, Shape{1}, int32);
        inputs.emplace_back(k_offset_data, Shape{1}, int32);
        for (int32_t idx = 0; idx < n_kv; ++idx) {
            inputs.push_back(*to_arr(kv_caches[idx]));
        }

        auto outputs = m->forward(inputs);
        *out_hidden = from_arr(std::move(outputs[0]));
        for (int32_t idx = 0; idx < n_kv; ++idx) {
            out_kv_caches[idx] = from_arr(std::move(outputs[1 + idx]));
        }
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

int32_t dflash_draft_forward_batched(
    void* model,
    mlx_array* noise_embedding,
    mlx_array* target_hidden,
    int32_t batch_size,
    mlx_array* q_offsets,
    mlx_array* k_offsets,
    mlx_array** kv_caches,
    int32_t n_kv,
    mlx_array* attn_mask,           // additive [B, 1, seq, key_len], nullable
    mlx_array** out_hidden,
    mlx_array** out_kv_caches
) {
    auto* m = static_cast<DFlashDraftModel*>(model);
    try {
        mlx_clear_error();

        if (batch_size <= 0) {
            throw std::runtime_error("DFlash draft batched forward requires batch_size > 0");
        }
        if (noise_embedding == nullptr || target_hidden == nullptr) {
            throw std::runtime_error("DFlash draft batched forward inputs must not be null");
        }
        if (q_offsets == nullptr || k_offsets == nullptr) {
            throw std::runtime_error("DFlash draft batched forward offsets must not be null");
        }
        if (n_kv != static_cast<int32_t>(m->layers.size() * 2)) {
            throw std::runtime_error("DFlash draft batched forward cache count mismatch");
        }

        const auto* noise_arr = to_arr(noise_embedding);
        const auto* target_arr = to_arr(target_hidden);
        const auto* q_offsets_arr = to_arr(q_offsets);
        const auto* k_offsets_arr = to_arr(k_offsets);
        if (noise_arr->ndim() != 3 || target_arr->ndim() != 3) {
            throw std::runtime_error(
                "DFlash draft batched forward expects rank-3 noise/target inputs");
        }
        if (noise_arr->shape(0) != batch_size || target_arr->shape(0) != batch_size) {
            throw std::runtime_error("DFlash draft batched forward batch size mismatch");
        }
        if (q_offsets_arr->ndim() != 1 || k_offsets_arr->ndim() != 1 ||
            q_offsets_arr->shape(0) != batch_size || k_offsets_arr->shape(0) != batch_size) {
            throw std::runtime_error("DFlash draft batched forward offset shape mismatch");
        }

        // Mirror qwen35_compiled_verify_block_batched: stash the has-mask flag
        // on the model struct so forward_batched() can pick the right compiled
        // slot. The mask array itself is appended to `inputs` (NOT captured by
        // the compiled lambda) to avoid stale references across compile cache
        // hits.
        m->current_has_attn_mask = attn_mask != nullptr;

        std::vector<array> inputs;
        inputs.reserve(4 + static_cast<size_t>(n_kv) + (attn_mask != nullptr ? 1 : 0));
        inputs.push_back(*noise_arr);
        inputs.push_back(*target_arr);
        inputs.push_back(*q_offsets_arr);
        inputs.push_back(*k_offsets_arr);
        for (int32_t idx = 0; idx < n_kv; ++idx) {
            inputs.push_back(*to_arr(kv_caches[idx]));
        }
        if (attn_mask != nullptr) {
            inputs.push_back(*to_arr(attn_mask));
        }

        auto outputs = m->forward_batched(inputs);
        *out_hidden = from_arr(std::move(outputs[0]));
        for (int32_t idx = 0; idx < n_kv; ++idx) {
            out_kv_caches[idx] = from_arr(std::move(outputs[1 + idx]));
        }
        m->current_has_attn_mask = false;
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        m->current_has_attn_mask = false;
        return -1;
    }
}

}  // extern "C"

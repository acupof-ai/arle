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

// Precise sigmoid-mul: sigmoid(gate.f32) * x.f32 -> x.dtype. Mirrors the base
// Qwen3.5 model's attention output gate (see mlx_qwen35_model.cpp). Used only by
// the Qwen35Mtp draft kind, whose single layer is a gated full-attention layer.
std::vector<array> precise_sigmoid_mul_impl(const std::vector<array>& inputs) {
    auto gate = sigmoid(astype(inputs[0], float32));
    auto x = astype(inputs[1], float32);
    return {astype(gate * x, inputs[1].dtype())};
}

auto& compiled_precise_sigmoid_mul() {
    static auto fn = mlx::core::compile(precise_sigmoid_mul_impl, true /* shapeless */);
    return fn;
}

// One EAGLE draft head, two configs. DFlashEagle = z-lab DFlash (fc consumes the
// base hidden only, single hidden_norm, non-gated layer). Qwen35Mtp = Qwen3.5/3.6
// NextN/MTP head (fc consumes concat[pre_fc_norm_embedding(embed),
// pre_fc_norm_hidden(hidden)], the single layer is a gated full-attention layer
// with partial rotary). The two branch ONCE for the fc-input/norm + layer-gate
// step and share everything after (KV growth, final norm, the batched verify).
enum class DraftKind { DFlashEagle = 0, Qwen35Mtp = 1 };

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

    // Qwen35Mtp-only: the two pre-fc norms applied to the token embedding and the
    // base hidden before they are concatenated and fed to `fc`. Unused (and left
    // as scalar placeholders) for the DFlashEagle kind.
    array pre_fc_norm_embedding = array(0);
    array pre_fc_norm_hidden = array(0);

    DraftKind draft_kind = DraftKind::DFlashEagle;

    int hidden_size = 2560;
    int num_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 128;
    int num_layers = 5;
    int rotary_dim = 0;          // Qwen35Mtp partial rotary; 0 => full head_dim.
    bool attn_output_gate = false;  // Qwen35Mtp: q_proj carries a gated half.
    float rope_theta = 1e6f;
    float rms_eps = 1e-6f;

    std::function<std::vector<array>(const std::vector<array>&)> compiled_forward;
    bool is_compiled = false;

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
        int32_t rotary_dim_,
        int32_t attn_output_gate_,
        int32_t draft_kind_,
        float rope_theta_,
        float rms_eps_
    ) {
        hidden_size = hidden_size_;
        num_heads = num_heads_;
        num_kv_heads = num_kv_heads_;
        head_dim = head_dim_;
        num_layers = num_layers_;
        rotary_dim = rotary_dim_ > 0 ? rotary_dim_ : head_dim_;
        attn_output_gate = attn_output_gate_ != 0;
        draft_kind = draft_kind_ == 1 ? DraftKind::Qwen35Mtp : DraftKind::DFlashEagle;
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

    // Qwen35Mtp: replace the single hidden_norm with the two pre-fc norms and the
    // final norm. `fc` and `norm` are still set via set_fc_norms (hidden_norm is
    // ignored for this kind).
    void set_qwen35_mtp_norms(array pre_fc_norm_embedding_, array pre_fc_norm_hidden_) {
        pre_fc_norm_embedding = std::move(pre_fc_norm_embedding_);
        pre_fc_norm_hidden = std::move(pre_fc_norm_hidden_);
    }

    // Compute the fc output (= the single layer's input residual stream) for the
    // active draft kind. For DFlashEagle this is fc(hidden) -> hidden_norm and the
    // result feeds the KV-context concat. For Qwen35Mtp this is
    // fc(concat[pre_fc_norm_embedding(embed), pre_fc_norm_hidden(hidden)]) along the
    // feature axis, and the result IS the layer input (one row per position). `embed`
    // and `hidden` are expected to be aligned row-for-row for Qwen35Mtp.
    array compute_fc_input(const array& embed, const array& hidden) const {
        if (draft_kind == DraftKind::Qwen35Mtp) {
            auto ne = fast::rms_norm(embed, pre_fc_norm_embedding, rms_eps);
            auto nh = fast::rms_norm(hidden, pre_fc_norm_hidden, rms_eps);
            return fc.apply(concatenate({ne, nh}, -1));
        }
        auto proj = fc.apply(hidden);
        return fast::rms_norm(proj, hidden_norm, rms_eps);
    }

    std::vector<array> forward_impl(const std::vector<array>& inputs) const {
        const size_t expected_inputs = 4 + layers.size() * 2;
        if (inputs.size() != expected_inputs) {
            throw std::runtime_error("DFlash draft forward input count mismatch");
        }

        auto noise_embedding = inputs[0];
        auto target_hidden = inputs[1];
        auto q_offset = inputs[2];
        auto k_offset = inputs[3];

        if (noise_embedding.ndim() != 2 || target_hidden.ndim() != 2) {
            throw std::runtime_error("DFlash draft forward expects rank-2 noise/target inputs");
        }

        if (draft_kind == DraftKind::Qwen35Mtp) {
            return forward_impl_mtp(inputs, noise_embedding, target_hidden, q_offset);
        }

        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        auto hidden_states = noise_embedding;

        auto target_hidden_proj = compute_fc_input(noise_embedding, target_hidden);

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

    // Qwen3.5/3.6 NextN/MTP forward for one set of `seq` aligned positions.
    //   x      = fc(concat[pre_fc_norm_embedding(embed), pre_fc_norm_hidden(hidden)])
    //   x      = gated full-attention layer (own causal KV grows by `seq`)
    //   return rms_norm(x, norm)  -> fed to the BASE lm_head for draft logits
    // `q_offset` carries each position's rope index. The head appends `seq` rows to
    // its own KV (no dual-stream context concat — that is the structural difference
    // from DFlashEagle).
    std::vector<array> forward_impl_mtp(
        const std::vector<array>& inputs,
        const array& noise_embedding,
        const array& target_hidden,
        const array& q_offset
    ) const {
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        // fc input is one row per position; embed/hidden are aligned [seq, H].
        auto hidden_states = compute_fc_input(noise_embedding, target_hidden);

        const auto& layer = layers[0];
        auto residual = hidden_states;
        auto normed_hidden = fast::rms_norm(hidden_states, layer.input_layernorm, rms_eps);

        // seq is concrete here (forward_impl_mtp runs EAGERLY, never compiled — see
        // finalize()), so reshapes use the explicit length rather than -1. That is
        // what lets `split` infer its output shapes (the shapeless-compile trace
        // could not).
        const int seq = noise_embedding.shape(0);
        auto q_raw = layer.q_proj.apply(normed_hidden);
        auto k_raw = layer.k_proj.apply(normed_hidden);
        auto v_raw = layer.v_proj.apply(normed_hidden);

        array q(0), gate_val(0);
        if (attn_output_gate) {
            auto q_full = reshape(q_raw, {1, seq, num_heads, head_dim * 2});
            auto q_gate = split(q_full, Shape{head_dim}, -1);
            q = fast::rms_norm(q_gate[0], layer.q_norm, rms_eps);
            gate_val = q_gate[1];
        } else {
            q = fast::rms_norm(
                reshape(q_raw, {1, seq, num_heads, head_dim}), layer.q_norm, rms_eps);
        }
        q = transpose(q, {0, 2, 1, 3});
        q = fast::rope(q, rotary_dim, false, rope_theta, 1.0f, q_offset);

        auto k = reshape(k_raw, {1, seq, num_kv_heads, head_dim});
        k = fast::rms_norm(k, layer.k_norm, rms_eps);
        k = transpose(k, {0, 2, 1, 3});
        k = fast::rope(k, rotary_dim, false, rope_theta, 1.0f, q_offset);

        auto v = reshape(v_raw, {1, seq, num_kv_heads, head_dim});
        v = transpose(v, {0, 2, 1, 3});

        auto new_k_cache = concatenate({inputs[4], k}, 2);
        auto new_v_cache = concatenate({inputs[5], v}, 2);

        // seq>1 (initial cold draft) needs causal masking; seq==1 (warm step) does not.
        std::string mask_mode = (seq > 1) ? "causal" : "";
        auto attn = fast::scaled_dot_product_attention(
            q, new_k_cache, new_v_cache, attn_scale, mask_mode);
        attn = transpose(attn, {0, 2, 1, 3});
        attn = reshape(attn, {1, seq, num_heads * head_dim});

        array attn_out(0);
        if (attn_output_gate) {
            auto gate = reshape(gate_val, {1, seq, num_heads * head_dim});
            attn_out = compiled_precise_sigmoid_mul()({gate, attn})[0];
        } else {
            attn_out = attn;
        }
        attn_out = reshape(attn_out, {seq, num_heads * head_dim});
        attn_out = layer.o_proj.apply(attn_out);
        hidden_states = residual + attn_out;

        auto residual2 = hidden_states;
        auto post_norm = fast::rms_norm(hidden_states, layer.post_attention_layernorm, rms_eps);
        auto gate_proj = layer.gate_proj.apply(post_norm);
        auto up = layer.up_proj.apply(post_norm);
        auto mlp = layer.down_proj.apply(compiled_swiglu()({gate_proj, up})[0]);
        hidden_states = residual2 + mlp;

        std::vector<array> outputs;
        outputs.reserve(1 + layers.size() * 2);
        outputs.push_back(fast::rms_norm(hidden_states, norm, rms_eps));
        outputs.push_back(std::move(new_k_cache));
        outputs.push_back(std::move(new_v_cache));
        return outputs;
    }

    void finalize() {
        if (static_cast<int>(layers.size()) != num_layers) {
            throw std::runtime_error("DFlash draft layer count mismatch at finalize");
        }
        // The Qwen35Mtp forward uses split + gated reshapes whose output shapes the
        // shapeless-compile trace cannot infer; it runs eagerly instead (the head is
        // one layer, so per-call graph build is cheap). The z-lab DFlash forward
        // compiles as before.
        if (draft_kind == DraftKind::Qwen35Mtp) {
            is_compiled = false;
            return;
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
    int32_t rotary_dim,
    int32_t attn_output_gate,
    int32_t draft_kind,
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
            rotary_dim,
            attn_output_gate,
            draft_kind,
            rope_theta,
            rms_eps);
    });
}

void dflash_draft_set_qwen35_mtp_norms(
    void* model,
    mlx_array* pre_fc_norm_embedding,
    mlx_array* pre_fc_norm_hidden
) {
    MLX_TRY_VOID({
        auto* m = static_cast<DFlashDraftModel*>(model);
        m->set_qwen35_mtp_norms(
            *to_arr(pre_fc_norm_embedding),
            *to_arr(pre_fc_norm_hidden));
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

        std::vector<array> inputs;
        inputs.reserve(4 + static_cast<size_t>(n_kv));
        inputs.push_back(*to_arr(noise_embedding));
        inputs.push_back(*to_arr(target_hidden));

        if (m->draft_kind == DraftKind::Qwen35Mtp) {
            // No context prepend: the head's KV holds only its own draft tokens, so
            // each of the `seq` positions gets a consecutive absolute rope index
            // starting at rope_offset. q and k share these offsets (forward_impl_mtp
            // reads inputs[2] for both).
            const int32_t seq = to_arr(noise_embedding)->shape(0);
            std::vector<int32_t> offsets(static_cast<size_t>(seq));
            for (int32_t i = 0; i < seq; ++i) {
                offsets[static_cast<size_t>(i)] = rope_offset + i;
            }
            inputs.emplace_back(offsets.data(), Shape{seq}, int32);
            inputs.emplace_back(offsets.data(), Shape{seq}, int32);
        } else {
            const int32_t context_len = to_arr(target_hidden)->shape(0);
            const int32_t q_offset_value = rope_offset + context_len;
            int32_t q_offset_data[1] = {q_offset_value};
            int32_t k_offset_data[1] = {rope_offset};
            inputs.emplace_back(q_offset_data, Shape{1}, int32);
            inputs.emplace_back(k_offset_data, Shape{1}, int32);
        }
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

}  // extern "C"

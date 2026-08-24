//! LFM2.5 C++ forward model — collapses per-op Rust/FFI overhead, same
//! pattern as `mlx_qwen35_model.cpp`.
//!
//! Architecture (see docs/reference: HuggingFace transformers
//! `modeling_lfm2_moe.py`):
//!   - 24 layers, each either a gated short-conv block (18) or full
//!     attention (6), with pre-norm (operator_norm / ffn_norm).
//!   - Conv block: y = out_proj(C * conv1d(B * x)); in_proj splits into
//!     B|C|x, gates are multiplicative with NO activation, conv has NO
//!     activation. Conv state keeps the last (kernel-1) post-gate frames.
//!   - Attention: standard MHA with per-head-dim Q/K RMSNorm, RoPE, GQA.
//!   - FFN: SwiGLU — dense (intermediate 7168) for the first 2 layers,
//!     32-expert top-4 MoE (sigmoid routing + expert_bias, no shared
//!     expert) for the rest.
//!   - Final RMSNorm (`embedding_norm`, despite the name) + tied lm_head.
//!
//! API:
//!   model = lfm2_compiled_new()
//!   lfm2_compiled_set_config(model, ...)
//!   lfm2_compiled_set_embed(model, embed, embedding_norm, lm_head_id)
//!   lfm2_compiled_set_embed_as_linear(model, embed_quant_id)  // tied lm_head
//!   lfm2_compiled_push_conv_layer(model, ...)  // ×18
//!   lfm2_compiled_push_attn_layer(model, ...)  // ×6
//!   lfm2_compiled_set_last_moe(model, ...)     // ×22 (after the layer push)
//!   lfm2_compiled_finalize(model)
//!   lfm2_session_begin/end, lfm2_compiled_prefill/step[_paged]_session
//!   lfm2_compiled_free(model)

#include "mlx_common.h"
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <functional>
#include <map>
#include <tuple>
#include <stdexcept>

namespace {

bool parse_env_bool(const char* name, bool fallback) {
    const char* env = std::getenv(name);
    return env ? std::string(env) != "0" : fallback;
}

bool use_prefill_last_logits_only() {
    static const bool enabled =
        parse_env_bool("AGENT_INFER_LFM2_CPP_PREFILL_LAST_LOGITS_ONLY", true);
    return enabled;
}

bool keep_prefill_intermediates() {
    static const bool enabled =
        parse_env_bool("AGENT_INFER_LFM2_CPP_KEEP_PREFILL_INTERMEDIATES", false);
    return enabled;
}

std::optional<array> bias_if_affine(const array& biases, int mode) {
    return mode == 0 ? std::optional(biases) : std::nullopt;
}

bool contains_layer_id(const std::vector<int>& ids, int id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

// Compiled SwiGLU: silu(gate) * up.
std::vector<array> swiglu_impl(const std::vector<array>& inputs) {
    auto gate = inputs[0];
    auto up = inputs[1];
    return {(gate * mlx::core::sigmoid(gate)) * up};
}

auto& compiled_swiglu() {
    static auto fn = mlx::core::compile(swiglu_impl, /*shapeless=*/true);
    return fn;
}

// Compiled fused dense MLP: gate_up matmul -> split -> swiglu -> down matmul.
// Encoded once per (gate_dim, gs, bits, mode); decode (S=1) only. Mirrors
// compiled_mlp_fn in mlx_qwen35_model.cpp.
using CompiledFn = std::function<std::vector<array>(const std::vector<array>&)>;
CompiledFn& compiled_mlp_fn(int gate_dim, int gs, int bits, int mode) {
    static std::map<std::tuple<int, int, int, int>, CompiledFn> cache;
    auto key = std::make_tuple(gate_dim, gs, bits, mode);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    auto impl = [gate_dim, gs, bits, mode](const std::vector<array>& in) -> std::vector<array> {
        auto gu = quantized_matmul(in[0], in[1], in[2], bias_if_affine(in[3], mode), true,
                                   gs, bits, quant_mode_str(mode));
        auto parts = split(gu, Shape{gate_dim}, -1);
        auto h = (parts[0] * sigmoid(parts[0])) * parts[1];
        return {quantized_matmul(h, in[4], in[5], bias_if_affine(in[6], mode), true, gs,
                                 bits, quant_mode_str(mode))};
    };
    return cache.emplace(key, mlx::core::compile(impl, /*shapeless=*/false)).first->second;
}

} // namespace

struct Lfm2Weight {
    array w = array(0);
    array scales = array(0);
    array biases = array(0);
    int group_size = 64;
    int bits = 4;
    bool is_dense = false;  // w pre-transposed [in, out]
    int mode = 0;           // 0=affine, 1=mxfp4

    array apply(const array& x) const {
        if (is_dense) {
            return matmul(x, w);
        }
        return quantized_matmul(
            x, w, scales, bias_if_affine(biases, mode), true, group_size, bits,
            quant_mode_str(mode));
    }

    // Sub-weight for output dimensions [start, end).  Slice is on the weight
    // (a compile-time constant), so it folds away under mlx::core::compile —
    // this is what lets the compiled verify forward avoid split/slice on
    // intermediate activations (which shapeless compile cannot infer).
    Lfm2Weight sub_weight(int start, int end) const {
        Lfm2Weight sub;
        sub.is_dense = is_dense;
        sub.group_size = group_size;
        sub.bits = bits;
        sub.mode = mode;
        if (is_dense) {
            sub.w = slice(w, {0, start}, {w.shape(0), end});
        } else {
            sub.w = slice(w, {start, 0}, {end, w.shape(1)});
            sub.scales = slice(scales, {start, 0}, {end, scales.shape(1)});
            if (biases.ndim() > 0) {
                sub.biases = slice(biases, {start, 0}, {end, biases.shape(1)});
            }
        }
        return sub;
    }
};

struct Lfm2ConvLayer {
    array op_norm_w = array(0), ffn_norm_w = array(0);
    Lfm2Weight in_proj;   // H -> 3H
    // Pre-split projections (constant-folded at setup time, avoiding
    // slice/split on dynamic-shape intermediates in the compiled verify path).
    Lfm2Weight b_proj, c_proj, x_proj;
    array conv_w = array(0);  // [H, kernel, 1]
    Lfm2Weight out_proj;  // H -> H
    Lfm2Weight gate_up;   // dense FFN (merged gate+up), empty for MoE layers
    Lfm2Weight gate_proj, up_proj;  // pre-split for compiled verify
    Lfm2Weight down;
    int gate_dim = 0;
};

struct Lfm2AttnLayer {
    array op_norm_w = array(0), ffn_norm_w = array(0);
    Lfm2Weight q_proj, k_proj, v_proj, o_proj;
    array q_norm_w = array(0), k_norm_w = array(0);  // [head_dim]
    Lfm2Weight gate_up;
    Lfm2Weight gate_proj, up_proj;  // pre-split for compiled verify
    Lfm2Weight down;
    int gate_dim = 0;
};

struct Lfm2MoeFFN {
    array router_w = array(0);    // dense [H, E] (pre-transposed)
    array expert_bias = array(0); // [E]
    Lfm2Weight switch_gate, switch_up, switch_down;  // stacked [E, I, H/pack]
    int num_experts = 0;
    int top_k = 0;
    bool norm_topk_prob = true;
    int expert_bits = 4;
    int expert_group_size = 64;
    // Dense BF16 expert weights (optional — when non-empty, bypasses quantized path).
    array dense_gate_w = array(0);  // [E, I, H]
    array dense_up_w = array(0);    // [E, I, H]
    array dense_down_w = array(0);  // [E, H, I]
    // Lazily-compiled MoE forward (shapeless=false, fixed verify-block shape).
    // Cuts ~23 eager kernel launches per MoE layer × 22 layers.
    // mutable: lazily initialized in const forward_impl.
    mutable std::function<std::vector<array>(const std::vector<array>&)> compiled_moe;
    // The seq_len the MoE was compiled for (shapeless=false specializes on S).
    mutable int compiled_moe_seq_len = 0;
};

struct Lfm2Layer {
    bool is_conv = false;
    Lfm2ConvLayer conv;
    Lfm2AttnLayer attn;
    bool has_moe = false;
    Lfm2MoeFFN moe;
};

// Defined in mlx_lfm2_moe_block.cpp.
array lfm2_moe_block_forward_cpp(
    const array& x,
    const array& router_w,
    const array& expert_bias,
    const array& expert_gate_w, const array& expert_gate_s, const array& expert_gate_b,
    const array& expert_up_w,   const array& expert_up_s,   const array& expert_up_b,
    const array& expert_down_w, const array& expert_down_s, const array& expert_down_b,
    int32_t expert_group_size, int32_t expert_bits,
    int32_t num_experts, int32_t top_k, bool norm_topk_prob);
array lfm2_moe_block_forward_dense_cpp(
    const array& x,
    const array& router_w,
    const array& expert_bias,
    const array& expert_gate_w,
    const array& expert_up_w,
    const array& expert_down_w,
    int32_t num_experts, int32_t top_k, bool norm_topk_prob);

struct Lfm2CompiledModel {
    // Weights
    array embed_tokens = array(0);       // dequantized bf16 for take()
    array embedding_norm_w = array(0);   // final RMSNorm (despite the name)
    Lfm2Weight lm_head;
    Lfm2Weight embed_as_linear;          // quantized embed for tied lm_head
    bool use_embed_as_linear = false;
    std::vector<Lfm2Layer> layers;
    std::vector<Lfm2Weight> weight_pool;

    // Config
    float rope_theta = 5e6f;
    float rms_eps = 1e-5f;
    int n_heads = 32, n_kv_heads = 8, head_dim = 64;
    int hidden_size = 2048;
    int conv_kernel = 3;
    int n_full_attn = 0, n_conv = 0;

    // Session state
    std::vector<array> session_kv_caches;   // [k0, v0, ...] per full-attn layer
    std::vector<array> session_conv_states; // one per conv layer
    bool session_active = false;

    // Per-step runtime state
    int current_cache_pos = 0;
    int current_seq_len = 1;
    int current_batch_size = 1;

    // Lazily-compiled verify forward (shapeless=true, dynamic KV cache length).
    mutable std::function<std::vector<array>(const std::vector<array>&)> compiled_verify;
    bool current_last_logits_only = false;
    bool current_has_paged_prefix = false;
    std::vector<array> current_paged_k;
    std::vector<array> current_paged_v;
    std::vector<array> prev_outputs;

    // Compiled verify forward (shapeless; shapes are fixed for spec decode).
    std::function<std::vector<array>(const std::vector<array>&)> compiled_verify_forward;

    // DSpark speculative-decode capture. When non-empty, forward_impl captures
    // the residual stream after each listed layer and (for conv layers) the
    // full conv_input window, appending both to the outputs tail. Read back via
    // lfm2_get_captured_hidden / lfm2_get_captured_conv_input after the forward.
    std::vector<int> capture_layer_ids;
    bool capture_conv_inputs = false;

    static const bool mlp_compile() {
        static const bool on = std::getenv("INFER_METAL_NO_MLP_COMPILE") == nullptr;
        return on;
    }

    array dense_mlp(const array& x, const Lfm2Weight& gate_up,
                    const Lfm2Weight& gate, const Lfm2Weight& up,
                    const Lfm2Weight& down, int gate_dim) const {
        if (mlp_compile() && x.ndim() == 3 && x.shape(1) == 1 && !gate_up.is_dense
            && !down.is_dense && gate_up.group_size == down.group_size
            && gate_up.bits == down.bits && gate_up.mode == down.mode) {
            return compiled_mlp_fn(
                gate_dim, gate_up.group_size, gate_up.bits, gate_up.mode)(
                {x, gate_up.w, gate_up.scales, gate_up.biases,
                 down.w, down.scales, down.biases})[0];
        }
        // Pre-split gate/up projections (setup-time sub_weight) — avoids
        // slice/split on the activation which shapeless compile can't infer.
        auto h = compiled_swiglu()({gate.apply(x), up.apply(x)})[0];
        return down.apply(h);
    }

    array conv_step(
        const array& x, const Lfm2ConvLayer& lw,
        const array& conv_state_in,
        array& conv_state_out,
        std::vector<array>* captured_conv_inputs) const {
        int B = current_batch_size;
        int S = current_seq_len;
        int H = hidden_size;

        // Pre-split projections (computed at setup time, avoiding slice on
        // dynamic-shape intermediates in the compiled verify path).
        auto gate_b = lw.b_proj.apply(x);
        auto gate_c = lw.c_proj.apply(x);
        auto conv_in = lw.x_proj.apply(x);
        // in_proj output order is B|C|x: input gate, output gate, conv input.
        auto h = gate_b * conv_in;

        int n_keep = conv_kernel - 1;
        // Pad conv state if it has fewer than n_keep frames (e.g. the DSpark
        // draft model doesn't carry conv states). Left-pad with zeros so the
        // conv1d has enough context.
        array state = conv_state_in;
        if (state.ndim() < 3 || state.shape(1) < n_keep) {
            int have = (state.ndim() >= 3) ? state.shape(1) : 0;
            int pad = n_keep - have;
            array zeros = mlx::core::zeros({B, pad, H}, state.dtype());
            state = concatenate({zeros, state}, 1);
        }
        auto conv_input = concatenate({state, h}, 1);  // [B, S+n_keep, H]
        // DSpark: save the full conv window so the spec loop can slice the
        // 2-frame conv state at any accepted position (avoids a re-run on
        // partial acceptance).
        if (captured_conv_inputs) captured_conv_inputs->push_back(conv_input);
        // Slice the last n_keep frames using the runtime shape (not S, which
        // the compiled graph bakes in from the first trace).
        int total_frames = conv_input.shape(1);
        conv_state_out = contiguous(slice(
            conv_input, {0, total_frames - n_keep, 0}, {B, total_frames, H}));
        auto conv_out = conv1d(conv_input, lw.conv_w, 1, 0, 1, H);  // [B, S, H]

        auto y = gate_c * conv_out;
        return lw.out_proj.apply(y);
    }

    array full_attn_step(
        const array& x, const Lfm2AttnLayer& lw,
        const array& k_cache, const array& v_cache,
        const array& cache_pos,
        int full_layer_idx,
        array& new_k_cache, array& new_v_cache) const {
        int B = current_batch_size;
        int nh = n_heads, nkv = n_kv_heads, hd = head_dim;
        int S = current_seq_len;
        float attn_scale = 1.0f / std::sqrt((float)hd);

        auto q = reshape(lw.q_proj.apply(x), {B, -1, nh, hd});
        q = fast::rms_norm(q, lw.q_norm_w, rms_eps);
        q = transpose(q, {0, 2, 1, 3});

        auto k = reshape(lw.k_proj.apply(x), {B, -1, nkv, hd});
        k = fast::rms_norm(k, lw.k_norm_w, rms_eps);
        k = transpose(k, {0, 2, 1, 3});

        q = fast::rope(q, hd, false, rope_theta, 1.0f, cache_pos);
        k = fast::rope(k, hd, false, rope_theta, 1.0f, cache_pos);

        auto v = transpose(reshape(lw.v_proj.apply(x), {B, -1, nkv, hd}), {0, 2, 1, 3});

        // Grow KV cache via concatenate (cache is pre-trimmed to cache_pos by
        // the caller). This avoids slice_update whose Shape indices would bake
        // cache_pos as a stale constant under shapeless compile.
        new_k_cache = concatenate({k_cache, k}, 2);
        new_v_cache = concatenate({v_cache, v}, 2);

        array k_full(0), v_full(0);
        if (current_has_paged_prefix) {
            if (S != 1 || B != 1) {
                throw std::runtime_error("paged KV read supports only single-token decode");
            }
            if (full_layer_idx < 0 || full_layer_idx >= (int)current_paged_k.size()
                || full_layer_idx >= (int)current_paged_v.size()) {
                throw std::runtime_error("paged KV read missing layer input");
            }
            k_full = concatenate(std::vector<array>{current_paged_k[full_layer_idx], k}, 2);
            v_full = concatenate(std::vector<array>{current_paged_v[full_layer_idx], v}, 2);
        } else {
            // Cache is already exactly [1, nkv, cache_pos+S, hd] after concat.
            k_full = new_k_cache;
            v_full = new_v_cache;
        }

        // Causal mask is correct for S=1 too (single token attends to itself),
        // so hardcode it to avoid baking a C++ conditional into the compiled graph.
        auto attn = fast::scaled_dot_product_attention(q, k_full, v_full, attn_scale, "causal");
        attn = reshape(transpose(attn, {0, 2, 1, 3}), {B, -1, nh * hd});
        return lw.o_proj.apply(attn);
    }

    // inputs layout:
    //   [0]            : token ids
    //   [1 .. 1+2F)    : k_cache_i, v_cache_i for F full-attn layers
    //   [1+2F .. 1+2F+C) : conv_state_i for C conv layers
    //   [1+2F+C]       : cache_pos (int32 scalar array)
    // outputs: [logits, new_kv..., new_conv..., captured_conv_inputs..., captured_hidden...]
    // The captured tails are only present when capture_layer_ids is non-empty.
    std::vector<array> forward_impl(const std::vector<array>& inputs) const {
        auto token_id = inputs[0];
        int B = current_batch_size;
        int S = current_seq_len;
        int F = n_full_attn, C = n_conv;
        auto cache_pos = inputs[1 + 2 * F + C];

        auto x = take(embed_tokens, flatten(token_id), 0);
        x = reshape(x, {B, -1, hidden_size});

        std::vector<array> new_kv(2 * F, array(0));
        std::vector<array> new_conv(C, array(0));
        int full_idx = 0, conv_idx = 0;

        const bool capture = !capture_layer_ids.empty();
        std::vector<array> captured_hidden;
        std::vector<array> captured_conv_inputs;
        if (capture) {
            captured_hidden.reserve(capture_layer_ids.size());
            if (capture_conv_inputs) captured_conv_inputs.reserve(C);
        }

        for (int i = 0; i < (int)layers.size(); ++i) {
            auto& layer = layers[i];
            auto residual = x;
            auto op_norm_w = layer.is_conv ? layer.conv.op_norm_w : layer.attn.op_norm_w;
            auto xn = fast::rms_norm(x, op_norm_w, rms_eps);

            array attn_out(0);
            if (layer.is_conv) {
                int si = 1 + 2 * F + conv_idx;
                attn_out = conv_step(xn, layer.conv, inputs[si], new_conv[conv_idx],
                                     capture_conv_inputs ? &captured_conv_inputs : nullptr);
                conv_idx++;
            } else {
                int si = 1 + 2 * full_idx;
                attn_out = full_attn_step(
                    xn, layer.attn, inputs[si], inputs[si + 1], cache_pos, full_idx,
                    new_kv[2 * full_idx], new_kv[2 * full_idx + 1]);
                full_idx++;
            }
            x = residual + attn_out;

            auto residual2 = x;
            auto ffn_norm_w = layer.is_conv ? layer.conv.ffn_norm_w : layer.attn.ffn_norm_w;
            auto xn2 = fast::rms_norm(x, ffn_norm_w, rms_eps);
            if (layer.has_moe) {
                auto& moe = layer.moe;
                if (moe.dense_gate_w.ndim() > 0) {
                    x = residual2 + lfm2_moe_block_forward_dense_cpp(
                        xn2,
                        moe.router_w, moe.expert_bias,
                        moe.dense_gate_w, moe.dense_up_w, moe.dense_down_w,
                        moe.num_experts, moe.top_k, moe.norm_topk_prob);
                } else {
                    static bool eager_moe = getenv("LFM2_EAGER_MOE") != nullptr;
                    if (!moe.compiled_moe && !eager_moe) {
                        auto router_w = moe.router_w;
                        auto expert_bias = moe.expert_bias;
                        auto gw = moe.switch_gate.w, gs = moe.switch_gate.scales, gb = moe.switch_gate.biases;
                        auto uw = moe.switch_up.w, us = moe.switch_up.scales, ub = moe.switch_up.biases;
                        auto dw = moe.switch_down.w, ds = moe.switch_down.scales, db = moe.switch_down.biases;
                        int egs = moe.expert_group_size, eb = moe.expert_bits;
                        int ne = moe.num_experts, tk = moe.top_k;
                        bool ntp = moe.norm_topk_prob;
                        std::function<std::vector<array>(const std::vector<array>&)> fn =
                            [router_w, expert_bias, gw, gs, gb, uw, us, ub, dw, ds, db,
                             egs, eb, ne, tk, ntp](const std::vector<array>& inputs) -> std::vector<array> {
                                return {lfm2_moe_block_forward_cpp(
                                    inputs[0], router_w, expert_bias,
                                    gw, gs, gb, uw, us, ub, dw, ds, db,
                                    egs, eb, ne, tk, ntp)};
                            };
                        moe.compiled_moe = mlx::core::compile(fn, false /* shapeless */);
                        moe.compiled_moe_seq_len = S;
                    }
                    // shapeless=false specializes on S; eager fallback for other S.
                    if (moe.compiled_moe_seq_len == S) {
                        x = residual2 + moe.compiled_moe({xn2})[0];
                    } else {
                        x = residual2 + lfm2_moe_block_forward_cpp(
                            xn2, moe.router_w, moe.expert_bias,
                            moe.switch_gate.w, moe.switch_gate.scales, moe.switch_gate.biases,
                            moe.switch_up.w, moe.switch_up.scales, moe.switch_up.biases,
                            moe.switch_down.w, moe.switch_down.scales, moe.switch_down.biases,
                            moe.expert_group_size, moe.expert_bits,
                            moe.num_experts, moe.top_k, moe.norm_topk_prob);
                    }
                }
            } else if (layer.is_conv) {
                x = residual2 + dense_mlp(xn2, layer.conv.gate_up,
                                          layer.conv.gate_proj, layer.conv.up_proj,
                                          layer.conv.down, layer.conv.gate_dim);
            } else {
                x = residual2 + dense_mlp(xn2, layer.attn.gate_up,
                                          layer.attn.gate_proj, layer.attn.up_proj,
                                          layer.attn.down, layer.attn.gate_dim);
            }
            // DSpark: capture the post-layer residual stream at target layers.
            if (capture && contains_layer_id(capture_layer_ids, i)) {
                captured_hidden.push_back(x);
            }
        }

        auto final_x = fast::rms_norm(x, embedding_norm_w, rms_eps);
        if (current_last_logits_only && S > 1) {
            final_x = slice(final_x, {0, S - 1, 0}, {B, S, hidden_size});
        }
        auto logits = use_embed_as_linear ? embed_as_linear.apply(final_x) : lm_head.apply(final_x);

        std::vector<array> outputs;
        outputs.reserve(1 + 2 * F + C + captured_conv_inputs.size() + captured_hidden.size());
        outputs.push_back(std::move(logits));
        for (auto& kv : new_kv) outputs.push_back(std::move(kv));
        for (auto& c : new_conv) outputs.push_back(std::move(c));
        for (auto& ci : captured_conv_inputs) outputs.push_back(std::move(ci));
        for (auto& h : captured_hidden) outputs.push_back(std::move(h));
        return outputs;
    }

    std::vector<array> forward(const std::vector<array>& inputs) {
        auto outputs = forward_impl(inputs);
        // Keep previous outputs alive so lazy graphs don't release GPU buffers
        // mid-pipeline (mirrors the qwen35 model's prev_outputs).
        prev_outputs = outputs;
        return prev_outputs;  // copies of handles; buffers stay owned here
    }

    // Compiled verify path — shapeless=true handles the growing KV cache.
    // MoE blocks are compiled separately (shapeless=false) to avoid the
    // argpartition/slice shape inference issue. The conv state slice is the
    // remaining risk; if compile fails, fall back to eager.
    std::vector<array> forward_verify(const std::vector<array>& inputs) {
        static bool eager = getenv("LFM2_EAGER_VERIFY") != nullptr;
        std::vector<array> outputs;
        if (eager) {
            outputs = forward_impl(inputs);
        } else {
            if (!compiled_verify) {
                compiled_verify = mlx::core::compile(
                    [this](const std::vector<array>& ins) { return this->forward_impl(ins); },
                    true /* shapeless */);
            }
            outputs = compiled_verify(inputs);
        }
        // Store outputs so drain_captured_hidden/conv_inputs can read the
        // capture tail (forward_verify is the only forward in the DSpark path;
        // without this, prev_outputs stays stale from prefill).
        prev_outputs = outputs;
        return outputs;
    }
};

Lfm2Weight& lfm2_weight_by_id(Lfm2CompiledModel* model, int32_t id) {
    if (id < 0 || id >= (int32_t)model->weight_pool.size()) {
        throw std::runtime_error("invalid LFM2 compiled weight id");
    }
    return model->weight_pool[(size_t)id];
}

extern "C" {

void* lfm2_compiled_new() {
    MLX_TRY_RETURN(new Lfm2CompiledModel());
}

void lfm2_compiled_free(void* model) {
    MLX_TRY_VOID(delete static_cast<Lfm2CompiledModel*>(model));
}

int32_t lfm2_compiled_add_dense_weight(void* model, mlx_array* w) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = static_cast<Lfm2CompiledModel*>(model);
        m->weight_pool.push_back({*to_arr(w), array(0), array(0), 0, 0, true});
        return (int32_t)(m->weight_pool.size() - 1);
    }());
}

int32_t lfm2_compiled_add_quant_weight(
    void* model, mlx_array* w, mlx_array* scales, mlx_array* biases,
    int32_t group_size, int32_t bits, int32_t mode) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = static_cast<Lfm2CompiledModel*>(model);
        m->weight_pool.push_back({
            *to_arr(w), *to_arr(scales),
            biases ? *to_arr(biases) : array(0),
            group_size, bits, false, mode});
        return (int32_t)(m->weight_pool.size() - 1);
    }());
}

void lfm2_compiled_set_config(
    void* model,
    float rope_theta, float rms_eps,
    int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
    int32_t hidden_size, int32_t conv_kernel) {
    MLX_TRY_VOID({
        auto* m = static_cast<Lfm2CompiledModel*>(model);
        m->rope_theta = rope_theta;
        m->rms_eps = rms_eps;
        m->n_heads = n_heads;
        m->n_kv_heads = n_kv_heads;
        m->head_dim = head_dim;
        m->hidden_size = hidden_size;
        m->conv_kernel = conv_kernel;
    });
}

void lfm2_compiled_set_embed(
    void* model, mlx_array* embed_tokens, mlx_array* embedding_norm_w,
    int32_t lm_head_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Lfm2CompiledModel*>(model);
        m->embed_tokens = embed_tokens ? *to_arr(embed_tokens) : array(0);
        m->embedding_norm_w = *to_arr(embedding_norm_w);
        m->lm_head = lfm2_weight_by_id(m, lm_head_id);
        m->use_embed_as_linear = false;
    });
}

void lfm2_compiled_set_embed_as_linear(void* model, int32_t embed_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Lfm2CompiledModel*>(model);
        m->embed_as_linear = lfm2_weight_by_id(m, embed_id);
        m->use_embed_as_linear = true;
    });
}

void lfm2_compiled_push_conv_layer(
    void* model,
    mlx_array* op_norm, mlx_array* ffn_norm,
    int32_t in_proj_id, mlx_array* conv_w, int32_t out_proj_id,
    int32_t gate_up_id, int32_t gate_dim, int32_t down_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Lfm2CompiledModel*>(model);
        Lfm2Layer layer;
        layer.is_conv = true;
        layer.conv.op_norm_w = *to_arr(op_norm);
        layer.conv.ffn_norm_w = *to_arr(ffn_norm);
        layer.conv.in_proj = lfm2_weight_by_id(m, in_proj_id);
        int H = m->hidden_size;
        layer.conv.b_proj = layer.conv.in_proj.sub_weight(0, H);
        layer.conv.c_proj = layer.conv.in_proj.sub_weight(H, 2 * H);
        layer.conv.x_proj = layer.conv.in_proj.sub_weight(2 * H, 3 * H);
        layer.conv.conv_w = *to_arr(conv_w);
        layer.conv.out_proj = lfm2_weight_by_id(m, out_proj_id);
        if (gate_up_id >= 0) {
            layer.conv.gate_up = lfm2_weight_by_id(m, gate_up_id);
            layer.conv.gate_proj = layer.conv.gate_up.sub_weight(0, gate_dim);
            layer.conv.up_proj = layer.conv.gate_up.sub_weight(gate_dim, 2 * gate_dim);
            layer.conv.gate_dim = gate_dim;
        }
        if (down_id >= 0) {
            layer.conv.down = lfm2_weight_by_id(m, down_id);
        }
        m->layers.push_back(std::move(layer));
        m->n_conv++;
    });
}

void lfm2_compiled_push_attn_layer(
    void* model,
    mlx_array* op_norm, mlx_array* ffn_norm,
    int32_t q_id, int32_t k_id, int32_t v_id, int32_t o_id,
    mlx_array* q_norm, mlx_array* k_norm,
    int32_t gate_up_id, int32_t gate_dim, int32_t down_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Lfm2CompiledModel*>(model);
        Lfm2Layer layer;
        layer.is_conv = false;
        layer.attn.op_norm_w = *to_arr(op_norm);
        layer.attn.ffn_norm_w = *to_arr(ffn_norm);
        layer.attn.q_proj = lfm2_weight_by_id(m, q_id);
        layer.attn.k_proj = lfm2_weight_by_id(m, k_id);
        layer.attn.v_proj = lfm2_weight_by_id(m, v_id);
        layer.attn.o_proj = lfm2_weight_by_id(m, o_id);
        layer.attn.q_norm_w = *to_arr(q_norm);
        layer.attn.k_norm_w = *to_arr(k_norm);
        if (gate_up_id >= 0) {
            layer.attn.gate_up = lfm2_weight_by_id(m, gate_up_id);
            layer.attn.gate_proj = layer.attn.gate_up.sub_weight(0, gate_dim);
            layer.attn.up_proj = layer.attn.gate_up.sub_weight(gate_dim, 2 * gate_dim);
            layer.attn.gate_dim = gate_dim;
        }
        if (down_id >= 0) {
            layer.attn.down = lfm2_weight_by_id(m, down_id);
        }
        m->layers.push_back(std::move(layer));
        m->n_full_attn++;
    });
}

void lfm2_compiled_set_last_moe(
    void* model,
    mlx_array* router_w, mlx_array* expert_bias,
    int32_t switch_gate_id, int32_t switch_up_id, int32_t switch_down_id,
    int32_t expert_group_size, int32_t expert_bits,
    int32_t num_experts, int32_t top_k, bool norm_topk_prob) {
    MLX_TRY_VOID({
        auto* m = static_cast<Lfm2CompiledModel*>(model);
        if (m->layers.empty()) {
            throw std::runtime_error("lfm2_compiled_set_last_moe requires an existing layer");
        }
        auto& layer = m->layers.back();
        layer.has_moe = true;
        layer.moe.router_w = *to_arr(router_w);
        layer.moe.expert_bias = *to_arr(expert_bias);
        layer.moe.switch_gate = lfm2_weight_by_id(m, switch_gate_id);
        layer.moe.switch_up = lfm2_weight_by_id(m, switch_up_id);
        layer.moe.switch_down = lfm2_weight_by_id(m, switch_down_id);
        layer.moe.expert_group_size = expert_group_size;
        layer.moe.expert_bits = expert_bits;
        layer.moe.num_experts = num_experts;
        layer.moe.top_k = top_k;
        layer.moe.norm_topk_prob = norm_topk_prob;
    });
}

void lfm2_compiled_set_last_moe_dense(
    void* model,
    mlx_array* router_w, mlx_array* expert_bias,
    mlx_array* gate_w, mlx_array* up_w, mlx_array* down_w,
    int32_t num_experts, int32_t top_k, bool norm_topk_prob) {
    MLX_TRY_VOID({
        auto* m = static_cast<Lfm2CompiledModel*>(model);
        if (m->layers.empty()) {
            throw std::runtime_error("lfm2_compiled_set_last_moe_dense requires an existing layer");
        }
        auto& layer = m->layers.back();
        layer.has_moe = true;
        layer.moe.router_w = *to_arr(router_w);
        layer.moe.expert_bias = *to_arr(expert_bias);
        layer.moe.dense_gate_w = *to_arr(gate_w);
        layer.moe.dense_up_w = *to_arr(up_w);
        layer.moe.dense_down_w = *to_arr(down_w);
        layer.moe.num_experts = num_experts;
        layer.moe.top_k = top_k;
        layer.moe.norm_topk_prob = norm_topk_prob;
    });
}

int32_t lfm2_compiled_finalize(void* model) {
    auto* m = static_cast<Lfm2CompiledModel*>(model);
    try {
        mlx_clear_error();
        if ((int)m->layers.size() != m->n_conv + m->n_full_attn) {
            throw std::runtime_error("LFM2 layer count mismatch");
        }
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

int32_t lfm2_session_begin(
    void* model, mlx_array** kv_caches, int32_t n_kv,
    mlx_array** conv_states, int32_t n_conv) {
    auto* m = static_cast<Lfm2CompiledModel*>(model);
    mlx_clear_error();
    try {
        if (m->session_active) {
            throw std::runtime_error("lfm2_session_begin requires an inactive session");
        }
        if (n_kv != 2 * m->n_full_attn) {
            throw std::runtime_error(
                "lfm2_session_begin KV cache count must be 2*full_attn_layers");
        }
        if (n_conv != m->n_conv) {
            throw std::runtime_error(
                "lfm2_session_begin conv state count must match conv layers");
        }
        m->session_kv_caches.clear();
        m->session_conv_states.clear();
        for (int i = 0; i < n_kv; ++i) {
            m->session_kv_caches.push_back(*to_arr(kv_caches[i]));
        }
        for (int i = 0; i < n_conv; ++i) {
            m->session_conv_states.push_back(*to_arr(conv_states[i]));
        }
        m->session_active = true;
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

int32_t lfm2_session_end(
    void* model, mlx_array** out_kv, int32_t n_kv,
    mlx_array** out_conv, int32_t n_conv) {
    auto* m = static_cast<Lfm2CompiledModel*>(model);
    mlx_clear_error();
    try {
        if (!m->session_active) {
            throw std::runtime_error("lfm2_session_end requires an active session");
        }
        if ((int)m->session_kv_caches.size() != n_kv
            || (int)m->session_conv_states.size() != n_conv) {
            throw std::runtime_error("lfm2_session_end cache counts do not match the session");
        }
        for (int i = 0; i < n_kv; ++i) {
            out_kv[i] = from_arr(std::move(m->session_kv_caches[i]));
        }
        for (int i = 0; i < n_conv; ++i) {
            out_conv[i] = from_arr(std::move(m->session_conv_states[i]));
        }
        m->session_kv_caches.clear();
        m->session_conv_states.clear();
        m->session_active = false;
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

// Build forward inputs: token_ids + KV caches (trimmed to cache_pos) + conv
// states + cache_pos (as int32 array). Trimming is an eager view (no copy);
// the compiled forward then grows the cache via concatenate.
static std::vector<array> build_forward_inputs(
    const Lfm2CompiledModel* m, const array& token_ids, int32_t cache_pos) {
    std::vector<array> inputs;
    inputs.reserve(1 + m->session_kv_caches.size() + m->session_conv_states.size() + 1);
    inputs.push_back(token_ids);
    for (const auto& kv : m->session_kv_caches) {
        int nkv = kv.shape(1);
        int hd = kv.shape(3);
        inputs.push_back(slice(kv, {0, 0, 0, 0}, {1, nkv, cache_pos, hd}));
    }
    for (const auto& c : m->session_conv_states) inputs.push_back(c);
    int32_t cp_data[1] = {cache_pos};
    inputs.emplace_back(cp_data, Shape{1}, int32);
    return inputs;
}

static void extract_forward_outputs(
    Lfm2CompiledModel* m, std::vector<array>& outputs, mlx_array** out_logits) {
    std::vector<array> next_kv, next_conv;
    for (size_t i = 0; i < m->session_kv_caches.size(); ++i) {
        next_kv.push_back(std::move(outputs[1 + i]));
    }
    for (size_t i = 0; i < m->session_conv_states.size(); ++i) {
        next_conv.push_back(std::move(outputs[1 + m->session_kv_caches.size() + i]));
    }
    *out_logits = from_arr(std::move(outputs[0]));
    m->session_kv_caches = std::move(next_kv);
    m->session_conv_states = std::move(next_conv);
}

int32_t lfm2_compiled_step_session(
    void* model, mlx_array* token_id, int32_t cache_pos, mlx_array** out_logits) {
    auto* m = static_cast<Lfm2CompiledModel*>(model);
    try {
        mlx_clear_error();
        if (!m->session_active) {
            throw std::runtime_error("lfm2_compiled_step_session requires an active session");
        }
        m->current_cache_pos = cache_pos;
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->current_has_paged_prefix = false;
        m->current_paged_k.clear();
        m->current_paged_v.clear();

        auto inputs = build_forward_inputs(m, *to_arr(token_id), cache_pos);
        // Use the compiled verify path for single-token decode too — the
        // shapeless-compiled forward fuses element-wise ops across layers,
        // cutting kernel launch overhead ~2ms/token vs the eager forward().
        auto outputs = m->forward_verify(inputs);
        extract_forward_outputs(m, outputs, out_logits);
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

// Eager single-token decode for the DSpark adaptive-skip fallback. The compiled
// forward_verify traces with S=5 (verify block) and bakes slice/reshape indices
// that fail on S=1; the eager path reads S at runtime.
int32_t lfm2_eager_step_session(
    void* model, mlx_array* token_id, int32_t cache_pos, mlx_array** out_logits) {
    auto* m = static_cast<Lfm2CompiledModel*>(model);
    try {
        mlx_clear_error();
        if (!m->session_active) {
            throw std::runtime_error("lfm2_eager_step_session requires an active session");
        }
        m->current_cache_pos = cache_pos;
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->current_has_paged_prefix = false;
        m->current_paged_k.clear();
        m->current_paged_v.clear();

        auto inputs = build_forward_inputs(m, *to_arr(token_id), cache_pos);
        auto outputs = m->forward(inputs);
        extract_forward_outputs(m, outputs, out_logits);
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

int32_t lfm2_compiled_step_session_paged(
    void* model, mlx_array* token_id, int32_t cache_pos,
    mlx_array** k_full_per_layer, mlx_array** v_full_per_layer, int32_t n_layers,
    mlx_array** out_logits) {
    auto* m = static_cast<Lfm2CompiledModel*>(model);
    try {
        mlx_clear_error();
        if (!m->session_active) {
            throw std::runtime_error("lfm2_compiled_step_session_paged requires an active session");
        }
        if (n_layers != m->n_full_attn) {
            throw std::runtime_error("lfm2_compiled_step_session_paged layer count mismatch");
        }
        m->current_cache_pos = cache_pos;
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->current_has_paged_prefix = true;
        m->current_paged_k.clear();
        m->current_paged_v.clear();
        m->current_paged_k.reserve(n_layers);
        m->current_paged_v.reserve(n_layers);
        for (int32_t i = 0; i < n_layers; ++i) {
            if (!k_full_per_layer[i] || !v_full_per_layer[i]) {
                throw std::runtime_error("lfm2_compiled_step_session_paged received null layer input");
            }
            m->current_paged_k.push_back(*to_arr(k_full_per_layer[i]));
            m->current_paged_v.push_back(*to_arr(v_full_per_layer[i]));
        }

        auto inputs = build_forward_inputs(m, *to_arr(token_id), cache_pos);
        auto outputs = m->forward(inputs);

        m->current_has_paged_prefix = false;
        m->current_paged_k.clear();
        m->current_paged_v.clear();

        extract_forward_outputs(m, outputs, out_logits);
        return 0;
    } catch (const std::exception& e) {
        m->current_has_paged_prefix = false;
        m->current_paged_k.clear();
        m->current_paged_v.clear();
        mlx_set_error(e.what());
        return -1;
    }
}

int32_t lfm2_compiled_prefill_session(
    void* model, mlx_array* token_ids, int32_t prompt_len, int32_t cache_pos,
    mlx_array** out_logits) {
    auto* m = static_cast<Lfm2CompiledModel*>(model);
    try {
        mlx_clear_error();
        if (!m->session_active) {
            throw std::runtime_error("lfm2_compiled_prefill_session requires an active session");
        }
        m->current_cache_pos = cache_pos;
        m->current_batch_size = 1;
        m->current_seq_len = prompt_len;
        m->current_last_logits_only = use_prefill_last_logits_only();
        m->current_has_paged_prefix = false;
        m->current_paged_k.clear();
        m->current_paged_v.clear();

        auto inputs = build_forward_inputs(m, *to_arr(token_ids), cache_pos);
        auto outputs = m->forward(inputs);
        extract_forward_outputs(m, outputs, out_logits);

        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        return 0;
    } catch (const std::exception& e) {
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        mlx_set_error(e.what());
        return -1;
    }
}

int32_t lfm2_compiled_verify_block_session(
    void* model, mlx_array* token_ids, int32_t block_len, int32_t cache_pos,
    const int32_t* capture_layer_ids, int32_t capture_count,
    mlx_array** out_logits) {
    auto* m = reinterpret_cast<Lfm2CompiledModel*>(model);
    try {
        mlx_clear_error();
        if (!m->session_active) {
            throw std::runtime_error("lfm2 verify_block requires an active session");
        }
        m->capture_layer_ids.clear();
        m->capture_conv_inputs = false;
        if (capture_layer_ids && capture_count > 0) {
            m->capture_layer_ids.assign(capture_layer_ids, capture_layer_ids + capture_count);
            m->capture_conv_inputs = true;
        }
        m->current_cache_pos = cache_pos;
        m->current_batch_size = 1;
        m->current_seq_len = block_len;
        m->current_last_logits_only = false;  // full logits for verification
        m->current_has_paged_prefix = false;
        m->current_paged_k.clear();
        m->current_paged_v.clear();

        auto inputs = build_forward_inputs(m, *to_arr(token_ids), cache_pos);
        auto outputs = m->forward_verify(inputs);
        extract_forward_outputs(m, outputs, out_logits);

        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        return 0;
    } catch (const std::exception& e) {
        m->capture_layer_ids.clear();
        m->capture_conv_inputs = false;
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        mlx_set_error(e.what());
        return -1;
    }
}

void lfm2_set_capture_layers(void* model, const int32_t* layer_ids, int32_t count) {
    auto* m = reinterpret_cast<Lfm2CompiledModel*>(model);
    m->capture_layer_ids.clear();
    m->capture_conv_inputs = false;
    if (layer_ids && count > 0) {
        m->capture_layer_ids.assign(layer_ids, layer_ids + count);
        m->capture_conv_inputs = true;
    }
}

int32_t lfm2_get_captured_hidden_count(void* model) {
    auto* m = reinterpret_cast<Lfm2CompiledModel*>(model);
    int hidden_count = static_cast<int>(m->capture_layer_ids.size());
    if (hidden_count <= 0) return 0;
    if ((int)m->prev_outputs.size() < hidden_count) return 0;
    return static_cast<int32_t>(hidden_count);
}

int32_t lfm2_get_captured_hidden(void* model, int32_t idx, mlx_array** out) {
    try {
        auto* m = reinterpret_cast<Lfm2CompiledModel*>(model);
        int hidden_count = static_cast<int>(m->capture_layer_ids.size());
        if (hidden_count <= 0)
            throw std::out_of_range("no captured hidden states are active");
        if ((int)m->prev_outputs.size() < hidden_count)
            throw std::out_of_range("captured hidden output tail is shorter than capture count");
        int hi = static_cast<int>(m->prev_outputs.size()) - hidden_count + idx;
        if (hi < 0 || hi >= (int)m->prev_outputs.size())
            throw std::out_of_range("captured hidden index out of range");
        *out = reinterpret_cast<mlx_array*>(new array(m->prev_outputs[hi]));
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

int32_t lfm2_get_captured_conv_count(void* model) {
    auto* m = reinterpret_cast<Lfm2CompiledModel*>(model);
    if (!m->capture_conv_inputs) return 0;
    int conv_count = m->n_conv;
    int hidden_count = static_cast<int>(m->capture_layer_ids.size());
    if ((int)m->prev_outputs.size() < hidden_count + conv_count) return 0;
    return static_cast<int32_t>(conv_count);
}

int32_t lfm2_get_captured_conv_input(void* model, int32_t idx, mlx_array** out) {
    try {
        auto* m = reinterpret_cast<Lfm2CompiledModel*>(model);
        if (!m->capture_conv_inputs)
            throw std::out_of_range("conv input capture is not active");
        int conv_count = m->n_conv;
        int hidden_count = static_cast<int>(m->capture_layer_ids.size());
        if ((int)m->prev_outputs.size() < hidden_count + conv_count)
            throw std::out_of_range("captured conv output tail is too short");
        int hi = static_cast<int>(m->prev_outputs.size()) - hidden_count - conv_count + idx;
        if (hi < 0 || hi >= (int)m->prev_outputs.size())
            throw std::out_of_range("captured conv input index out of range");
        *out = reinterpret_cast<mlx_array*>(new array(m->prev_outputs[hi]));
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

int32_t lfm2_session_set_conv_states(void* model, mlx_array** conv_states, int32_t n) {
    auto* m = reinterpret_cast<Lfm2CompiledModel*>(model);
    try {
        mlx_clear_error();
        if (!m->session_active) {
            throw std::runtime_error("lfm2_session_set_conv_states requires an active session");
        }
        if (n != m->n_conv) {
            throw std::runtime_error("lfm2_session_set_conv_states conv count mismatch");
        }
        m->session_conv_states.clear();
        for (int32_t i = 0; i < n; ++i) {
            if (!conv_states[i]) throw std::runtime_error("null conv state input");
            m->session_conv_states.push_back(*to_arr(conv_states[i]));
        }
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

} // extern "C"

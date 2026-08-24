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
//!   qwen35_compiled_step_session(model, ...)
//!   qwen35_compiled_free(model)

#include "mlx_common.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <tuple>
#include <stdexcept>

namespace {

int parse_env_int(const char* name, int fallback) {
    const char* env = std::getenv(name);
    if (!env || *env == '\0') {
        return fallback;
    }
    int value = fallback;
    auto first = env;
    auto last = env + std::char_traits<char>::length(env);
    auto [ptr, ec] = std::from_chars(first, last, value);
    if (ec != std::errc() || ptr != last || value <= 0) {
        return fallback;
    }
    return value;
}

bool parse_env_bool(const char* name, bool fallback) {
    const char* env = std::getenv(name);
    return env ? std::string(env) != "0" : fallback;
}

bool use_gdr_metal_kernel() {
    static const bool enabled = parse_env_bool("AGENT_INFER_GDR_METAL_KERNEL", true);
    return enabled;
}

bool keep_prefill_intermediates() {
    static const bool enabled =
        parse_env_bool("AGENT_INFER_QWEN35_CPP_KEEP_PREFILL_INTERMEDIATES", false);
    return enabled;
}

bool use_qwen35_cpp_prefill_last_logits_only() {
    static const bool enabled =
        parse_env_bool("AGENT_INFER_QWEN35_CPP_PREFILL_LAST_LOGITS_ONLY", true);
    return enabled;
}

bool use_qwen35_cpp_separate_mlp() {
    static const bool enabled = parse_env_bool("AGENT_INFER_QWEN35_CPP_SEPARATE_MLP", false);
    return enabled;
}

bool use_qwen35_cpp_prefill_gbeta_helper() {
    static const bool enabled = parse_env_bool("AGENT_INFER_QWEN35_CPP_PREFILL_GBETA_HELPER", true);
    return enabled;
}

bool use_qwen35_cpp_qk_norm_helper() {
    static const bool enabled = parse_env_bool("AGENT_INFER_QWEN35_CPP_QK_NORM_HELPER", false);
    return enabled;
}

array suppress_last_axis_token(const array& logits, int32_t suppress_token_id) {
    if (suppress_token_id < 0 || logits.ndim() == 0) {
        return logits;
    }
    int axis = logits.ndim() - 1;
    int vocab = logits.shape(axis);
    if (suppress_token_id >= vocab) {
        return logits;
    }

    auto update_shape = logits.shape();
    update_shape[axis] = 1;
    auto floor = astype(zeros(update_shape, float32) + array(-1.0e9f), logits.dtype());
    auto start = logits.shape();
    auto stop = logits.shape();
    std::fill(start.begin(), start.end(), 0);
    start[axis] = suppress_token_id;
    stop[axis] = suppress_token_id + 1;
    return slice_update(logits, floor, start, stop);
}

int qwen35_cpp_gdr_threadgroup_y(int seq_len) {
    static const int tg_y = parse_env_int("AGENT_INFER_QWEN35_CPP_GDR_TG_Y", 4);
    static const int prefill_tg_y =
        parse_env_int("AGENT_INFER_QWEN35_CPP_PREFILL_GDR_TG_Y", tg_y);
    static const int decode_tg_y =
        parse_env_int("AGENT_INFER_QWEN35_CPP_DECODE_GDR_TG_Y", tg_y);
    return seq_len > 1 ? prefill_tg_y : decode_tg_y;
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

// Tape-recording variant of gated_delta_kernel — same computation but
// additionally outputs the innovation tape (delta at each timestep).
auto& gated_delta_tape_kernel() {
    static auto kernel = fast::metal_kernel(
        "gated_delta_step_tape",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out", "innovation_tape"},
        R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        auto tape_ = innovation_tape + b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

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

            // Record innovation tape
            if (thread_index_in_simdgroup == 0) {
                tape_[dv_idx] = static_cast<InT>(delta);
            }

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
            tape_ += Hv * Dv;
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

// Compiled compute_g: g = exp(neg_exp_a * softplus(a + dt_bias))
// `neg_exp_a = -exp(A_log.f32)` is precomputed once at load time.
// Matches mlx_lm's runtime math while saving one per-step exp per layer.
std::vector<array> compute_g_impl(const std::vector<array>& inputs) {
    auto neg_exp_a = inputs[0];
    auto ab = inputs[1] + inputs[2];
    auto sp = where(greater(ab, array(20.0f)), ab, log1p(exp(ab)));
    return {exp(neg_exp_a * sp)};
}

auto& compiled_compute_g() {
    static auto fn = mlx::core::compile(compute_g_impl, true /*shapeless*/);
    return fn;
}

// Compiled compute_g + beta: both are per-token elementwise transforms used by
// every GDR layer during prefill/decode. Keeping them in one helper reduces one
// extra elementwise kernel launch and one temporary array per layer.
std::vector<array> compute_g_beta_impl(const std::vector<array>& inputs) {
    auto neg_exp_a = inputs[0];
    auto a_raw = inputs[1];
    auto dt_bias = inputs[2];
    auto b_raw = inputs[3];
    auto ab = a_raw + dt_bias;
    auto sp = where(greater(ab, array(20.0f)), ab, log1p(exp(ab)));
    return {exp(neg_exp_a * sp), sigmoid(b_raw)};
}

auto& compiled_compute_g_beta() {
    static auto fn = mlx::core::compile(compute_g_beta_impl, true /*shapeless*/);
    return fn;
}

// Compiled Q/K norm + scale for GDR. This keeps the two per-layer RMSNorm calls
// together so MLX can optimize them as one helper-level graph instead of two
// separate launches from host code.
std::vector<array> qk_norm_scale_impl(const std::vector<array>& inputs) {
    auto q = fast::rms_norm(inputs[0], std::nullopt, 1e-6f) * inputs[2];
    auto k = fast::rms_norm(inputs[1], std::nullopt, 1e-6f) * inputs[3];
    return {q, k};
}

auto& compiled_qk_norm_scale() {
    static auto fn = mlx::core::compile(qk_norm_scale_impl, true /*shapeless*/);
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

// Affine weights carry per-group biases; mxfp4 has none. Mode is the QuantMode
// code from Rust (see quant_mode_str in mlx_common.h). Inlined at the call site
// so the matmul argument stays a prvalue (one array-handle copy).
static std::optional<array> bias_if_affine(const array& biases, int mode) {
    return mode == 0 ? std::optional(biases) : std::nullopt;
}

// Compiled fused MLP: gate_up matmul -> split -> swiglu -> down matmul. Encoded ONCE per
// (gate_dim, gs, bits, mode) — all MLP layers share the config, so one cached graph serves
// every layer, cutting the per-step re-encode of the matmul-heavy MLP (~51% of the decode
// step).
using CompiledFn = std::function<std::vector<array>(const std::vector<array>&)>;
CompiledFn& compiled_mlp_fn(int gate_dim, int gs, int bits, int mode) {
    static std::map<std::tuple<int, int, int, int>, CompiledFn> cache;
    auto key = std::make_tuple(gate_dim, gs, bits, mode);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    auto impl = [gate_dim, gs, bits, mode](const std::vector<array>& in) -> std::vector<array> {
        // in = [x, gate_up_w, gate_up_scales, gate_up_biases, down_w, down_scales, down_biases]
        // biases slots are array(0) placeholders for mxfp4, never read.
        auto gu = quantized_matmul(in[0], in[1], in[2], bias_if_affine(in[3], mode), true, gs, bits,
                                   quant_mode_str(mode));
        auto parts = split(gu, Shape{gate_dim}, -1);
        auto h = (parts[0] * sigmoid(parts[0])) * parts[1];  // swiglu, inlined for fusion
        return {quantized_matmul(h, in[4], in[5], bias_if_affine(in[6], mode), true, gs, bits,
                                 quant_mode_str(mode))};
    };
    // SHAPED (not shapeless): the split needs concrete shapes. Gated to decode (S=1) at the
    // call site, so the shape is fixed [1,1,hidden] → compiled once, reused every step.
    return cache.emplace(key, mlx::core::compile(impl, false)).first->second;
}

// Compiled fused separate-MLP: gate matmul + up matmul -> swiglu -> down matmul.
// Used for mixed-bit MLP (OptiQ gate=4-bit/up=8-bit) where gate/up are kept as
// two separate quantized weights (no merged gate_up). Encoded ONCE per
// (gate_dim, gate_bits, up_bits, down_bits, gs, mode) — no split needed since gate and
// up are already separate. Mirrors compiled_mlp_fn (shaped, decode S=1 only).
CompiledFn& compiled_mlp_separate_fn(
    int gate_dim, int gate_bits, int up_bits, int down_bits, int gs, int mode) {
    static std::map<std::tuple<int, int, int, int, int, int>, CompiledFn> cache;
    auto key = std::make_tuple(gate_dim, gate_bits, up_bits, down_bits, gs, mode);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    auto impl = [gs, gate_bits, up_bits, down_bits, mode](
                    const std::vector<array>& in) -> std::vector<array> {
        // in = [x, gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b]
        auto g = quantized_matmul(in[0], in[1], in[2], bias_if_affine(in[3], mode), true, gs, gate_bits,
                                  quant_mode_str(mode));
        auto u = quantized_matmul(in[0], in[4], in[5], bias_if_affine(in[6], mode), true, gs, up_bits,
                                  quant_mode_str(mode));
        auto h = (g * sigmoid(g)) * u;  // swiglu, inlined for fusion
        return {quantized_matmul(h, in[7], in[8], bias_if_affine(in[9], mode), true, gs, down_bits,
                                 quant_mode_str(mode))};
    };
    return cache.emplace(key, mlx::core::compile(impl, false)).first->second;
}

// Compiled precise SiLU-mul: silu(gate.f32) * x.f32 -> x.dtype
std::vector<array> precise_silu_mul_impl(const std::vector<array>& inputs) {
    auto gate = astype(inputs[0], float32);
    auto x = astype(inputs[1], float32);
    return {astype(gate * sigmoid(gate) * x, inputs[1].dtype())};
}

auto& compiled_precise_silu_mul() {
    static auto fn = mlx::core::compile(precise_silu_mul_impl, true /*shapeless*/);
    return fn;
}

// Compiled precise sigmoid-mul: sigmoid(gate.f32) * x.f32 -> x.dtype
std::vector<array> precise_sigmoid_mul_impl(const std::vector<array>& inputs) {
    auto gate = sigmoid(astype(inputs[0], float32));
    auto x = astype(inputs[1], float32);
    return {astype(gate * x, inputs[1].dtype())};
}

auto& compiled_precise_sigmoid_mul() {
    static auto fn = mlx::core::compile(precise_sigmoid_mul_impl, true /*shapeless*/);
    return fn;
}

} // namespace


struct QWeight {
    array w = array(0);
    array scales = array(0);
    array biases = array(0);
    int group_size = 64;
    int bits = 4;
    bool is_dense = false;  // if true, w is already transposed, use matmul directly
    int mode = 0;  // 0=affine (scale+bias), 1=mxfp4 (E2M1 + E8M0 per-32 scale, no bias)

    array apply(const array& x, bool prefer_verify_m16 = false) const {
        if (is_dense) {
            return matmul(x, w);  // w is already transposed at load time
        }
        // The custom MMA2 kernel is affine-only; mxfp4 uses the stock kernel.
        if (prefer_verify_m16 && mode == 0) {
            return verify_quantized_matmul_cpp(x, w, scales, biases, group_size, bits);
        }
        return quantized_matmul(
            x, w, scales, bias_if_affine(biases, mode), true, group_size, bits, quant_mode_str(mode));
    }
};


struct FullAttnLayerWeights {
    array input_ln_w = array(0), post_attn_ln_w = array(0);
    QWeight q_proj, k_proj, v_proj, o_proj;
    array q_norm_w = array(0), k_norm_w = array(0);
    QWeight gate_up, down;
    // Separate gate/up projections for mixed-bit MLP (e.g. OptiQ gate=4-bit,
    // up=8-bit) that cannot row-merge into a single quantized gate_up. When set,
    // forward routes through mlp_separate (two quantized matmuls) instead of the
    // merged-then-dequantized dense gate_up path.
    QWeight gate_proj, up_proj;
    bool has_separate_mlp = false;
    bool has_qk_gate = true;  // true for Qwen3.5 (q_dim = nh*hd*2), false for Qwen3
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
    bool has_separate_mlp = false;
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
    bool has_moe = false;
    FullAttnLayerWeights full;
    GdrLayerWeights gdr;
    struct MoeLayerWeights {
        QWeight router;
        QWeight switch_gate;
        QWeight switch_up;
        QWeight switch_down;
        QWeight shared_gate;
        QWeight shared_up;
        QWeight shared_down;
        QWeight shared_expert_gate;
        int num_experts = 0;
        int top_k = 0;
        bool norm_topk_prob = true;
        int router_bits = 8;
        int router_group_size = 64;
        int expert_bits = 4;
        int expert_group_size = 64;
    } moe;
};


struct Qwen35CompiledModel {
    struct GdrTapeEntry {
        array innovation_tape = array(0);
        array k = array(0);
        array g = array(0);
        array qkv = array(0);
    };

    struct ForwardContext {
        int cache_pos = 0;
        int seq_len = 1;
        int batch_size = 1;
        bool last_logits_only = false;
        bool is_verify = false;
        bool kv_cache_int8 = false;
        bool keep_intermediates = false;
        bool record_tapes = false;
        const std::vector<int>* capture_layer_ids = nullptr;
    };

    struct ForwardArtifacts {
        std::vector<array> intermediates;
        std::vector<GdrTapeEntry> gdr_tapes;
    };

    // Weights
    array embed_tokens = array(0);  // dequantized bf16 for take() lookup
    array final_norm_w = array(0);
    QWeight lm_head;
    // Quantized embed weights for as_linear lm_head (when tied)
    QWeight embed_as_linear;
    bool use_embed_as_linear = false;
    std::vector<LayerWeights> layers;
    std::vector<QWeight> weight_pool;

    // Config
    float rope_theta = 1e6f;
    float rms_eps = 1e-6f;
    int n_heads = 16, n_kv_heads = 4, head_dim = 256;
    int rotary_dim = 256;
    int hidden_size = 2560;
    int n_full_attn = 0, n_gdr = 0;
    // Whether full-attn Q projection includes the gated half (q_dim = nh*hd*2).
    // Qwen3.5 always gates Q; Qwen3 never does. Set explicitly by the Rust
    // builder before finalize so dense full-attn-only Qwen3.5 fixtures are
    // routed correctly (n_gdr alone cannot tell the families apart).
    bool model_has_qk_gate = false;

    // Compiled function
    std::function<std::vector<array>(const std::vector<array>&)> compiled_fn;
    bool is_compiled = false;

    // Runtime state (set before each forward call)
    int current_cache_pos = 0;
    int current_seq_len = 1;  // 1 for decode, >1 for batch prefill
    int current_batch_size = 1;
    bool current_last_logits_only = false;
    bool current_is_verify = false;
    mutable array current_gdr_t_arr = array(1);
    mutable bool current_kv_cache_int8 = false;
    // Keep previous step's arrays alive to prevent premature GPU buffer release.
    // This mimics Python's lazy GC behavior where intermediates survive until
    // the next GC cycle, allowing MLX to reuse GPU buffers efficiently.
    std::vector<array> prev_outputs;
    // Session state for FFI-cost-amortized single-request decode.
    // BF16 session: [k0, v0, k1, v1, ...].
    // INT8 session: [k_q0, k_s0, k_b0, v_q0, v_s0, v_b0, ...].
    std::vector<array> session_kv_caches;
    std::vector<array> session_gdr_states;  // [gdr0, conv0, gdr1, conv1, ...]
    bool session_active = false;
    // Collect ALL intermediate arrays during forward() to keep them alive.
    // Cleared at start of each step, populated during forward().
    mutable std::vector<array> intermediates;


    // When tape_mode is on, gdr_step() records innovation tapes for each GDR layer.
    bool tape_mode = false;
    // When non-empty, forward() captures hidden states after the listed layers
    // and appends them to the output vector (after logits + caches + gdr states).
    std::vector<int> capture_layer_ids;
    // Per-GDR-layer tape recordings: (innovation_tape, k, g, qkv).
    // Populated during forward() when tape_mode=true, cleared at start of each step.
    mutable std::vector<GdrTapeEntry> gdr_tapes;

    bool keep_step_intermediates(int seq_len) const {
        return seq_len == 1 || keep_prefill_intermediates();
    }

    bool use_separate_mlp_for_current_step(const GdrLayerWeights& lw) const {
        // Env-gated when a merged gate_up exists (gate_dim > 0); unconditional when
        // only separate gate/up were registered (mixed-bit MLP has no merged path).
        return lw.has_separate_mlp
            && (use_qwen35_cpp_separate_mlp() || lw.gate_dim == 0);
    }

    static bool contains_layer_id(const std::vector<int>* layer_ids, int layer_id) {
        if (!layer_ids) {
            return false;
        }
        return std::find(layer_ids->begin(), layer_ids->end(), layer_id) != layer_ids->end();
    }

    static bool should_prefer_verify_m16(const ForwardContext& ctx) {
        return ctx.is_verify && ctx.batch_size == 1 && ctx.seq_len == 16;
    }

    void clear_optional_batch_inputs() {
        current_is_verify = false;
    }

    bool can_use_verify_sdpa_2pass(
        const ForwardContext& ctx,
        const array& q,
        const array& k_full,
        const array& v_full,
        int nh,
        int nkv,
        int hd
    ) const {
        if (!ctx.is_verify || ctx.seq_len != 16) {
            return false;
        }
        // Valid for both mask-free packed verify and the native single-row
        // verify-summary path. We intentionally do not require cache_pos_arr:
        // B=1 summary keeps the scalar cache contract and should still take
        // the exact 2-pass kernel when the shapes line up.
        if (ctx.batch_size <= 0) {
            return false;
        }
        if ((hd != 128 && hd != 256) || nkv <= 0 || (nh % nkv) != 0) {
            return false;
        }
        if (q.ndim() != 4 || k_full.ndim() != 4 || v_full.ndim() != 4) {
            return false;
        }
        if (q.dtype() != bfloat16 || k_full.dtype() != bfloat16 || v_full.dtype() != bfloat16) {
            return false;
        }
        return q.shape(0) == k_full.shape(0)
            && q.shape(0) == v_full.shape(0)
            && q.shape(1) == nh
            && k_full.shape(1) == nkv
            && v_full.shape(1) == nkv
            && q.shape(2) == 16
            && q.shape(3) == hd
            && k_full.shape(3) == hd
            && v_full.shape(3) == hd;
    }


    array full_attn_step(
        const array& x, const FullAttnLayerWeights& lw,
        const array& k_cache, const array& v_cache, int cache_pos,
        const ForwardContext& ctx,
        ForwardArtifacts* artifacts,
        array& new_k_cache, array& new_v_cache
    ) const {
        int B = ctx.batch_size;
        int nh = n_heads, nkv = n_kv_heads, hd = head_dim;
        int S = ctx.seq_len;
        float attn_scale = 1.0f / std::sqrt((float)hd);
        bool keep_intermediates = ctx.keep_intermediates && artifacts;
        bool prefer_verify_m16 = should_prefer_verify_m16(ctx);

        auto q_proj_out = lw.q_proj.apply(x, prefer_verify_m16);
        auto k_raw = lw.k_proj.apply(x, prefer_verify_m16);
        auto v_raw = lw.v_proj.apply(x, prefer_verify_m16);

        array q(0), gate_val(0);
        if (lw.has_qk_gate) {
            // Qwen3.5: Q has gate — split at head_dim
            auto q_full = reshape(q_proj_out, {B, S, nh, hd * 2});
            auto q_gate = split(q_full, Shape{hd}, -1);
            q = fast::rms_norm(q_gate[0], lw.q_norm_w, rms_eps);
            gate_val = q_gate[1];
        } else {
            // Qwen3: standard Q, no gate
            q = fast::rms_norm(reshape(q_proj_out, {B, S, nh, hd}), lw.q_norm_w, rms_eps);
        }
        q = transpose(q, {0, 2, 1, 3});

        auto k = reshape(k_raw, {B, S, nkv, hd});
        k = fast::rms_norm(k, lw.k_norm_w, rms_eps);
        k = transpose(k, {0, 2, 1, 3});

        q = fast::rope(q, rotary_dim, false, rope_theta, 1.0f, cache_pos);
        k = fast::rope(k, rotary_dim, false, rope_theta, 1.0f, cache_pos);

        auto v = reshape(v_raw, {B, S, nkv, hd});
        v = transpose(v, {0, 2, 1, 3});

        array k_full(0), v_full(0);
        int end = cache_pos + S;
        new_k_cache = slice_update(k_cache, k, {0,0,cache_pos,0}, {B,nkv,end,hd});
        new_v_cache = slice_update(v_cache, v, {0,0,cache_pos,0}, {B,nkv,end,hd});
        k_full = slice(new_k_cache, {0,0,0,0}, {B,nkv,end,hd});
        v_full = slice(new_v_cache, {0,0,0,0}, {B,nkv,end,hd});

        array attn_out(0);
        if (can_use_verify_sdpa_2pass(ctx, q, k_full, v_full, nh, nkv, hd)) {
            attn_out = batched_sdpa_2pass_cpp(q, k_full, v_full, attn_scale, nh / nkv);
        } else {
            std::string mask_mode = (S > 1) ? "causal" : "";
            attn_out = fast::scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                attn_scale,
                mask_mode);
        }
        attn_out = reshape(transpose(attn_out, {0,2,1,3}), {B, S, nh*hd});

        array result(0);
        if (lw.has_qk_gate) {
            auto gate = reshape(gate_val, {B, S, nh*hd});
            result = lw
                .o_proj
                .apply(compiled_precise_sigmoid_mul()({gate, attn_out})[0], prefer_verify_m16);
        } else {
            result = lw.o_proj.apply(attn_out, prefer_verify_m16);
        }

        // Keep intermediates alive for GPU buffer reuse
        if (keep_intermediates) {
            auto& intermediates = artifacts->intermediates;
            intermediates.push_back(q);
            intermediates.push_back(k);
            intermediates.push_back(attn_out);
            intermediates.push_back(result);
        }
        return result;
    }

    int kv_int8_group_size() const {
        if (head_dim % 128 == 0) return 128;
        if (head_dim % 64 == 0) return 64;
        if (head_dim % 32 == 0) return 32;
        throw std::runtime_error("Metal int8 KV requires head_dim divisible by 32/64/128");
    }

    array full_attn_step_int8(
        const array& x, const FullAttnLayerWeights& lw,
        const array& k_q_cache, const array& k_s_cache, const array& k_b_cache,
        const array& v_q_cache, const array& v_s_cache, const array& v_b_cache,
        int cache_pos,
        const ForwardContext& ctx,
        ForwardArtifacts* artifacts,
        array& new_k_q_cache, array& new_k_s_cache, array& new_k_b_cache,
        array& new_v_q_cache, array& new_v_s_cache, array& new_v_b_cache
    ) const {
        int B = ctx.batch_size;
        int nh = n_heads, nkv = n_kv_heads, hd = head_dim;
        int S = ctx.seq_len;
        int group_size = kv_int8_group_size();
        float attn_scale = 1.0f / std::sqrt((float)hd);
        bool keep_intermediates = ctx.keep_intermediates && artifacts;
        bool prefer_verify_m16 = should_prefer_verify_m16(ctx);

        auto q_proj_out = lw.q_proj.apply(x, prefer_verify_m16);
        auto k_raw = lw.k_proj.apply(x, prefer_verify_m16);
        auto v_raw = lw.v_proj.apply(x, prefer_verify_m16);

        array q(0), gate_val(0);
        if (lw.has_qk_gate) {
            auto q_full = reshape(q_proj_out, {B, S, nh, hd * 2});
            auto q_gate = split(q_full, Shape{hd}, -1);
            q = fast::rms_norm(q_gate[0], lw.q_norm_w, rms_eps);
            gate_val = q_gate[1];
        } else {
            q = fast::rms_norm(reshape(q_proj_out, {B, S, nh, hd}), lw.q_norm_w, rms_eps);
        }
        q = transpose(q, {0, 2, 1, 3});

        auto k = reshape(k_raw, {B, S, nkv, hd});
        k = fast::rms_norm(k, lw.k_norm_w, rms_eps);
        k = transpose(k, {0, 2, 1, 3});

        q = fast::rope(q, rotary_dim, false, rope_theta, 1.0f, cache_pos);
        k = fast::rope(k, rotary_dim, false, rope_theta, 1.0f, cache_pos);

        auto v = reshape(v_raw, {B, S, nkv, hd});
        v = transpose(v, {0, 2, 1, 3});

        auto kq = quantize(k, group_size, 8);
        auto vq = quantize(v, group_size, 8);
        if (kq.size() != 3 || vq.size() != 3) {
            throw std::runtime_error("Metal int8 KV quantize expected data/scale/bias triples");
        }

        int end = cache_pos + S;
        int packed_hd = hd / 4;
        int scale_hd = hd / group_size;
        new_k_q_cache = slice_update(k_q_cache, kq[0], {0,0,cache_pos,0}, {B,nkv,end,packed_hd});
        new_k_s_cache = slice_update(k_s_cache, kq[1], {0,0,cache_pos,0}, {B,nkv,end,scale_hd});
        new_k_b_cache = slice_update(k_b_cache, kq[2], {0,0,cache_pos,0}, {B,nkv,end,scale_hd});
        new_v_q_cache = slice_update(v_q_cache, vq[0], {0,0,cache_pos,0}, {B,nkv,end,packed_hd});
        new_v_s_cache = slice_update(v_s_cache, vq[1], {0,0,cache_pos,0}, {B,nkv,end,scale_hd});
        new_v_b_cache = slice_update(v_b_cache, vq[2], {0,0,cache_pos,0}, {B,nkv,end,scale_hd});

        array k_q_full = slice(new_k_q_cache, {0,0,0,0}, {B,nkv,end,packed_hd});
        array k_s_full = slice(new_k_s_cache, {0,0,0,0}, {B,nkv,end,scale_hd});
        array k_b_full = slice(new_k_b_cache, {0,0,0,0}, {B,nkv,end,scale_hd});
        array v_q_full = slice(new_v_q_cache, {0,0,0,0}, {B,nkv,end,packed_hd});
        array v_s_full = slice(new_v_s_cache, {0,0,0,0}, {B,nkv,end,scale_hd});
        array v_b_full = slice(new_v_b_cache, {0,0,0,0}, {B,nkv,end,scale_hd});
        auto k_full = dequantize(k_q_full, k_s_full, k_b_full, group_size, 8);
        auto v_full = dequantize(v_q_full, v_s_full, v_b_full, group_size, 8);

        array attn_out(0);
        if (can_use_verify_sdpa_2pass(ctx, q, k_full, v_full, nh, nkv, hd)) {
            attn_out = batched_sdpa_2pass_cpp(q, k_full, v_full, attn_scale, nh / nkv);
        } else {
            std::string mask_mode = (S > 1) ? "causal" : "";
            attn_out = fast::scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                attn_scale,
                mask_mode);
        }
        attn_out = reshape(transpose(attn_out, {0,2,1,3}), {B, S, nh*hd});

        array result(0);
        if (lw.has_qk_gate) {
            auto gate = reshape(gate_val, {B, S, nh*hd});
            result = lw
                .o_proj
                .apply(compiled_precise_sigmoid_mul()({gate, attn_out})[0], prefer_verify_m16);
        } else {
            result = lw.o_proj.apply(attn_out, prefer_verify_m16);
        }

        if (keep_intermediates) {
            auto& intermediates = artifacts->intermediates;
            intermediates.push_back(q);
            intermediates.push_back(k);
            intermediates.push_back(kq[0]);
            intermediates.push_back(vq[0]);
            intermediates.push_back(attn_out);
            intermediates.push_back(result);
        }
        return result;
    }


    array gdr_step(
        const array& x, const GdrLayerWeights& lw,
        const array& gdr_state_in, const array& conv_state_in,
        const ForwardContext& ctx,
        ForwardArtifacts* artifacts,
        array& gdr_state_out, array& conv_state_out,
        const array& gdr_t_arr
    ) const {
        int B = ctx.batch_size;
        int hk = lw.num_key_heads, dk = lw.key_dim;
        int hv = lw.num_value_heads, dv = lw.value_dim;
        int q_dim = hk * dk, k_dim = q_dim, v_dim = hv * dv;
        int qkv_dim = q_dim + k_dim + v_dim;
        int S = ctx.seq_len;
        bool keep_intermediates = ctx.keep_intermediates && artifacts;
        bool prefer_verify_m16 = should_prefer_verify_m16(ctx);

        auto x_3d = reshape(x, {B, S, hidden_size});

        // Projections
        array qkv(0), z_raw(0), b_raw(0), a_raw(0);
        if (lw.use_separate_proj) {
            qkv = lw.qkv_proj.apply(x_3d, prefer_verify_m16);
            z_raw = reshape(lw.z_proj.apply(x_3d, prefer_verify_m16), {B, S, hv, dv});
            b_raw = lw.b_proj.apply(x_3d, prefer_verify_m16);
            a_raw = lw.a_proj.apply(x_3d, prefer_verify_m16);
        } else {
            auto qkvz = lw.qkvz_proj.apply(x_3d, prefer_verify_m16);
            auto qkv_z = split(qkvz, Shape{lw.qkv_split}, -1);
            qkv = qkv_z[0];
            z_raw = qkv_z[1];
            auto ba = lw.ba_proj.apply(x_3d, prefer_verify_m16);
            auto ba_parts = split(ba, Shape{lw.ba_num_heads}, -1);
            b_raw = ba_parts[0];
            a_raw = ba_parts[1];
            if (keep_intermediates) {
                auto& intermediates = artifacts->intermediates;
                intermediates.push_back(qkvz);
                intermediates.push_back(ba);
                for (auto& a : qkv_z) {
                    intermediates.push_back(a);
                }
                for (auto& a : ba_parts) {
                    intermediates.push_back(a);
                }
            }
        }

        // Conv1d (naturally handles S > 1)
        auto conv_input = concatenate({conv_state_in, qkv}, 1);
        int n_keep = lw.conv_kernel - 1;
        int conv_total = n_keep + S;
        conv_state_out = contiguous(slice(conv_input, {0, conv_total - n_keep, 0}, {B, conv_total, qkv_dim}));
        auto conv_out = conv1d(conv_input, lw.conv1d_w, 1, 0, 1, qkv_dim);
        conv_out = compiled_silu()({conv_out})[0];

        // Split conv output
        auto qkv_parts = split(conv_out, Shape{q_dim, q_dim + k_dim}, -1);
        if (keep_intermediates) {
            auto& intermediates = artifacts->intermediates;
            for (auto& a : qkv_parts) {
                intermediates.push_back(a);
            }
        }
        auto q_raw = reshape(qkv_parts[0], {B, S, hk, dk});
        auto k_raw = reshape(qkv_parts[1], {B, S, hk, dk});
        auto v_raw = reshape(qkv_parts[2], {B, S, hv, dv});

        array q(0), k(0);
        if (use_qwen35_cpp_qk_norm_helper()) {
            auto qk = compiled_qk_norm_scale()({q_raw, k_raw, lw.q_scale_arr, lw.k_scale_arr});
            q = qk[0];
            k = qk[1];
        } else {
            q = fast::rms_norm(q_raw, std::nullopt, 1e-6f) * lw.q_scale_arr;
            k = fast::rms_norm(k_raw, std::nullopt, 1e-6f) * lw.k_scale_arr;
        }

        array g(0), beta(0);
        if (use_qwen35_cpp_prefill_gbeta_helper()) {
            auto gb = compiled_compute_g_beta()({lw.a_log, a_raw, lw.dt_bias, b_raw});
            g = gb[0];
            beta = gb[1];
        } else {
            beta = sigmoid(b_raw);
            g = compiled_compute_g()({lw.a_log, a_raw, lw.dt_bias})[0];
        }

        array y(0);
        if (use_gdr_metal_kernel()) {
            // The raw Metal kernel does direct pointer arithmetic and assumes
            // compact row-major inputs. GGUF Q4 projections and split/reshape
            // results may be lazy views, so materialize the exact kernel
            // contract before dispatch.
            auto q_kernel = contiguous(astype(reshape(q, {B, S, hk, dk}), bfloat16));
            auto k_kernel = contiguous(astype(reshape(k, {B, S, hk, dk}), bfloat16));
            auto v_kernel = contiguous(astype(reshape(v_raw, {B, S, hv, dv}), bfloat16));
            auto g_kernel = contiguous(reshape(g, {B, S, hv}));
            auto beta_kernel = contiguous(reshape(beta, {B, S, hv}));
            int threadgroup_y = qwen35_cpp_gdr_threadgroup_y(S);
            std::vector<array> inputs = {
                q_kernel, k_kernel, v_kernel, g_kernel, beta_kernel, gdr_state_in, gdr_t_arr
            };
            std::vector<Shape> out_shapes = {{B, S, hv, dv}, gdr_state_in.shape()};
            std::vector<Dtype> out_dtypes = {bfloat16, float32};
            std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
                {"Dk", fast::TemplateArg(dk)},
                {"Dv", fast::TemplateArg(dv)},
                {"Hk", fast::TemplateArg(hk)},
                {"Hv", fast::TemplateArg(hv)},
                {"InT", fast::TemplateArg(bfloat16)},
                {"StT", fast::TemplateArg(float32)},
            };

            if (ctx.record_tapes) {
                // Tape-recording variant: same computation + records innovation_tape
                std::vector<Shape> tape_out_shapes = {{B, S, hv, dv}, gdr_state_in.shape(), {B, S, hv, dv}};
                std::vector<Dtype> tape_out_dtypes = {bfloat16, float32, bfloat16};
                auto result = gated_delta_tape_kernel()(
                    inputs,
                    tape_out_shapes,
                    tape_out_dtypes,
                    std::make_tuple(32, dv, B * hv),
                    std::make_tuple(32, threadgroup_y, 1),
                    tmpl,
                    std::nullopt,
                    false,
                    {});
                y = std::move(result[0]);
                gdr_state_out = std::move(result[1]);
                // Record tape for rollback. tape_replay accepts bf16 or f32 g;
                // k and tape must be bf16 (kernel dtype gate).
                artifacts->gdr_tapes.push_back({
                    std::move(result[2]),            // innovation_tape (bf16 from kernel)
                    k_kernel,                        // k
                    g_kernel,                        // g (f32 from compiled_compute_g_beta)
                    contiguous(qkv),                 // qkv for conv rebuild
                });
            } else {
                auto result = gated_delta_kernel()(
                    inputs,
                    out_shapes,
                    out_dtypes,
                    std::make_tuple(32, dv, B * hv),
                    std::make_tuple(32, threadgroup_y, 1),
                    tmpl,
                    std::nullopt,
                    false,
                    {});
                y = std::move(result[0]);
                gdr_state_out = std::move(result[1]);
            }
            if (keep_intermediates) {
                auto& intermediates = artifacts->intermediates;
                intermediates.push_back(g_kernel);
                intermediates.push_back(beta_kernel);
            }
        } else {
            if (ctx.record_tapes) {
                throw std::runtime_error(
                    "Qwen3.5 GDR tape mode requires the custom Metal recurrent kernel");
            }
            int heads_per_key = hv / hk;
            array state = gdr_state_in;
            std::vector<array> y_steps;
            y_steps.reserve(S);

            for (int t = 0; t < S; ++t) {
                auto q_t = slice(q, {0, t, 0, 0}, {B, t + 1, hk, dk});
                auto k_t = slice(k, {0, t, 0, 0}, {B, t + 1, hk, dk});
                auto v_t = slice(v_raw, {0, t, 0, 0}, {B, t + 1, hv, dv});
                auto g_t = slice(g, {0, t, 0}, {B, t + 1, hv});
                auto beta_t = slice(beta, {0, t, 0}, {B, t + 1, hv});

                auto g_4d = reshape(g_t, {B, hv, 1, 1});
                auto s_decayed = state * g_4d;

                auto k_exp = (heads_per_key > 1)
                    ? reshape(
                        broadcast_to(expand_dims(k_t, 3), {B, 1, hk, heads_per_key, dk}),
                        {B, hv, dk})
                    : reshape(k_t, {B, hv, dk});
                auto q_exp = (heads_per_key > 1)
                    ? reshape(
                        broadcast_to(expand_dims(q_t, 3), {B, 1, hk, heads_per_key, dk}),
                        {B, hv, dk})
                    : reshape(q_t, {B, hv, dk});

                auto v_3d = reshape(v_t, {B, hv, dv});
                auto k_4d = reshape(k_exp, {B, hv, 1, dk});
                auto kv_mem = sum(s_decayed * k_4d, -1, false);
                auto beta_3d = reshape(beta_t, {B, hv, 1});
                auto delta = (v_3d - kv_mem) * beta_3d;
                state = s_decayed + reshape(delta, {B, hv, dv, 1}) * k_4d;

                auto q_4d = reshape(q_exp, {B, hv, 1, dk});
                y_steps.push_back(reshape(sum(state * q_4d, -1, false), {B, 1, hv, dv}));
            }

            gdr_state_out = state;
            y = y_steps.size() == 1 ? y_steps[0] : concatenate(y_steps, 1);
        }

        // Output norm + gate (S-aware)
        auto y_heads = reshape(y, {B * S * hv, dv});
        auto normed = fast::rms_norm(y_heads, lw.norm_w, lw.rms_eps);
        auto z_gated = reshape(z_raw, {B * S * hv, dv});
        auto out = compiled_precise_silu_mul()({z_gated, normed})[0];
        auto result = lw.out_proj.apply(reshape(out, {B, S, hv*dv}), prefer_verify_m16);

        // Keep ALL available intermediates alive for GPU buffer reuse.
        if (keep_intermediates) {
            auto& im = artifacts->intermediates;
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
        }
        return result;
    }


    // Separate MLP: 2 matmul (matching mlx_lm, no split overhead)
    array mlp_separate(
        const array& x,
        const QWeight& gate,
        const QWeight& up,
        const QWeight& down,
        bool prefer_verify_m16
    ) const {
        // Compiled fast path: standard quantized weights at decode (S=1) — the two
        // matmuls + swiglu + down matmul encode once per (gate_dim, bits..., gs).
        // Falls back to the per-op path for dense/verify weights.
        // Gated under INFER_METAL_NO_MLP_COMPILE (same flag as compiled_mlp_fn) so
        // it can be A/B'd against the uncompiled path.
        static const bool mlp_compile = std::getenv("INFER_METAL_NO_MLP_COMPILE") == nullptr;
        if (mlp_compile && !prefer_verify_m16
            && x.ndim() == 3 && x.shape(1) == 1  // decode only (fixed shape for the shaped compile)
            && !gate.is_dense && !up.is_dense && !down.is_dense
            && gate.group_size == up.group_size && gate.group_size == down.group_size
            && gate.mode == up.mode && gate.mode == down.mode) {
            int gate_dim = gate.w.shape(0);  // output rows of the gate projection
            return compiled_mlp_separate_fn(
                gate_dim, gate.bits, up.bits, down.bits, gate.group_size, gate.mode)(
                {x, gate.w, gate.scales, gate.biases,
                 up.w, up.scales, up.biases,
                 down.w, down.scales, down.biases})[0];
        }
        auto g = gate.apply(x, prefer_verify_m16);
        auto u = up.apply(x, prefer_verify_m16);
        auto h = compiled_swiglu()({g, u})[0];
        return down.apply(h, prefer_verify_m16);
    }

    // Fused MLP: 1 matmul + split
    array mlp(
        const array& x,
        const QWeight& gate_up,
        const QWeight& down,
        int gate_dim,
        bool prefer_verify_m16
    ) const {
        // Compiled fast path: standard quantized weights (the common decode case) — the two
        // matmuls + split + swiglu encode once. Falls back to the per-op path for
        // dense/verify weights, which the compiled graph does not cover.
        static const bool mlp_compile = std::getenv("INFER_METAL_NO_MLP_COMPILE") == nullptr;
        if (mlp_compile && !prefer_verify_m16
            && x.ndim() == 3 && x.shape(1) == 1  // decode only (fixed shape for the shaped compile)
            && !gate_up.is_dense && !down.is_dense
            && gate_up.group_size == down.group_size && gate_up.bits == down.bits
            && gate_up.mode == down.mode) {
            return compiled_mlp_fn(
                gate_dim, gate_up.group_size, gate_up.bits, gate_up.mode)(
                {x, gate_up.w, gate_up.scales, gate_up.biases,
                 down.w, down.scales, down.biases})[0];
        }
        auto gu = gate_up.apply(x, prefer_verify_m16);
        auto gu_parts = split(gu, Shape{gate_dim}, -1);
        auto& g = gu_parts[0];
        auto& u = gu_parts[1];
        auto h = compiled_swiglu()({g, u})[0]; // SiLU(gate) * up (compiled)
        return down.apply(h, prefer_verify_m16);
    }

    array moe_mlp(
        const array& x,
        const LayerWeights::MoeLayerWeights& moe,
        bool prefer_verify_m16
    ) const {
        if (prefer_verify_m16 && x.ndim() == 3 && x.shape(0) == 1 && x.shape(1) == 16) {
            auto x_2d = reshape(x, {16, x.shape(2)});
            auto y_2d = qwen35_moe_block_forward_cpp(
                x_2d,
                moe.router.w, moe.router.scales, moe.router.biases,
                moe.router_bits, moe.router_group_size,
                moe.switch_gate.w, moe.switch_gate.scales, moe.switch_gate.biases,
                moe.switch_up.w, moe.switch_up.scales, moe.switch_up.biases,
                moe.switch_down.w, moe.switch_down.scales, moe.switch_down.biases,
                moe.expert_bits, moe.expert_group_size,
                moe.shared_gate.w, moe.shared_gate.scales, moe.shared_gate.biases,
                moe.shared_up.w, moe.shared_up.scales, moe.shared_up.biases,
                moe.shared_down.w, moe.shared_down.scales, moe.shared_down.biases,
                moe.shared_expert_gate.w,
                moe.shared_expert_gate.scales,
                moe.shared_expert_gate.biases,
                moe.num_experts, moe.top_k, moe.norm_topk_prob);
            return reshape(y_2d, {1, 16, y_2d.shape(1)});
        }

        return qwen35_moe_block_forward_cpp(
            x,
            moe.router.w, moe.router.scales, moe.router.biases,
            moe.router_bits, moe.router_group_size,
            moe.switch_gate.w, moe.switch_gate.scales, moe.switch_gate.biases,
            moe.switch_up.w, moe.switch_up.scales, moe.switch_up.biases,
            moe.switch_down.w, moe.switch_down.scales, moe.switch_down.biases,
            moe.expert_bits, moe.expert_group_size,
            moe.shared_gate.w, moe.shared_gate.scales, moe.shared_gate.biases,
            moe.shared_up.w, moe.shared_up.scales, moe.shared_up.biases,
            moe.shared_down.w, moe.shared_down.scales, moe.shared_down.biases,
            moe.shared_expert_gate.w,
            moe.shared_expert_gate.scales,
            moe.shared_expert_gate.biases,
            moe.num_experts, moe.top_k, moe.norm_topk_prob);
    }

    // inputs layout:
    //   [0]        : token ids / token batch
    //   BF16:
    //     [1..1+2*F) : k_cache_i, v_cache_i for F full-attn layers
    //     [1+2*F .. 1+2*F+2*G) : gdr_state_i, conv_state_i for G GDR layers
    //   INT8:
    //     [1..1+6*F) : k_q,k_s,k_b,v_q,v_s,v_b per full-attn layer
    //     [1+6*F .. 1+6*F+2*G) : gdr_state_i, conv_state_i for G GDR layers
    // outputs layout:
    //   [0]        : logits
    //   followed by the same KV/GDR layout as the inputs.

    std::vector<array> forward_impl(
        const std::vector<array>& inputs,
        const ForwardContext& ctx,
        ForwardArtifacts* artifacts
    ) const {
        auto token_id = inputs[0];
        int cache_pos = ctx.cache_pos;
        int B = ctx.batch_size;
        int S = ctx.seq_len;  // 1 for decode, >1 for batch prefill

        int F = n_full_attn, G = n_gdr;
        int kv_per_full = ctx.kv_cache_int8 ? 6 : 2;
        int full_kv_count = kv_per_full * F;
        auto x = take(embed_tokens, flatten(token_id), 0);
        x = reshape(x, {B, S, hidden_size});
        // op-profile (INFER_METAL_OP_PROFILE): eval-based per-section breakdown of the decode
        // forward. Serializes the step (sync per section) — RELATIVE breakdown only. Default-off.
        static const bool op_prof = std::getenv("INFER_METAL_OP_PROFILE") != nullptr;
        static thread_local double op_ms[5] = {0, 0, 0, 0, 0};  // embed, gdr, full, mlp, head
        static thread_local long op_steps = 0;
        if (op_prof) {
            auto s = std::chrono::high_resolution_clock::now();
            eval(x);
            op_ms[0] += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - s).count();
        }
        bool keep_intermediates = ctx.keep_intermediates && artifacts;
        bool prefer_verify_m16 = should_prefer_verify_m16(ctx);
        auto gdr_t_arr = array(S);

        std::vector<array> new_kv_caches(full_kv_count, array(0));
        std::vector<array> new_gdr_states(G, array(0));
        std::vector<array> new_conv_states(G, array(0));
        std::vector<array> captured_hidden;
        if (ctx.capture_layer_ids && !ctx.capture_layer_ids->empty()) {
            captured_hidden.reserve(ctx.capture_layer_ids->size());
        }
        int full_idx = 0, gdr_idx = 0;

        for (int i = 0; i < (int)layers.size(); ++i) {
            auto& layer = layers[i];
            auto residual = x;

            // Input layernorm
            auto ln_w = layer.is_gdr ? layer.gdr.input_ln_w : layer.full.input_ln_w;
            auto xn = fast::rms_norm(x, ln_w, rms_eps);

            // Attention
            auto op_attn_s = op_prof ? std::chrono::high_resolution_clock::now()
                                     : std::chrono::high_resolution_clock::time_point{};
            array attn_out(0);
            if (layer.is_gdr) {
                int si = 1 + full_kv_count + 2*gdr_idx;
                attn_out = gdr_step(xn, layer.gdr,
                    inputs[si], inputs[si+1],
                    ctx,
                    artifacts,
                    new_gdr_states[gdr_idx], new_conv_states[gdr_idx], gdr_t_arr);
                gdr_idx++;
            } else {
                int si = 1 + kv_per_full*full_idx;
                int oi = kv_per_full*full_idx;
                if (ctx.kv_cache_int8) {
                    attn_out = full_attn_step_int8(xn, layer.full,
                        inputs[si], inputs[si+1], inputs[si+2],
                        inputs[si+3], inputs[si+4], inputs[si+5],
                        cache_pos,
                        ctx,
                        artifacts,
                        new_kv_caches[oi], new_kv_caches[oi+1], new_kv_caches[oi+2],
                        new_kv_caches[oi+3], new_kv_caches[oi+4], new_kv_caches[oi+5]);
                } else {
                    attn_out = full_attn_step(xn, layer.full,
                        inputs[si], inputs[si+1],
                        cache_pos,
                        ctx,
                        artifacts,
                        new_kv_caches[oi], new_kv_caches[oi+1]);
                }
                full_idx++;
            }

            if (op_prof) {
                eval(attn_out);
                op_ms[layer.is_gdr ? 1 : 2] += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - op_attn_s).count();
            }
            x = residual + attn_out;

            // MLP
            auto op_mlp_s = op_prof ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};
            auto residual2 = x;
            auto post_ln_w = layer.is_gdr ? layer.gdr.post_attn_ln_w : layer.full.post_attn_ln_w;
            auto xn2 = fast::rms_norm(x, post_ln_w, rms_eps);
            if (layer.has_moe) {
                x = residual2 + moe_mlp(xn2, layer.moe, prefer_verify_m16);
            } else if (layer.is_gdr && use_separate_mlp_for_current_step(layer.gdr)) {
                x = residual2
                    + mlp_separate(
                        xn2,
                        layer.gdr.gate_proj,
                        layer.gdr.up_proj,
                        layer.gdr.down,
                        prefer_verify_m16);
            } else if (!layer.is_gdr && layer.full.has_separate_mlp) {
                // Mixed-bit MLP (e.g. OptiQ gate=4-bit/up=8-bit): gate and up
                // cannot row-merge into one quantized gate_up, so they run as two
                // separate quantized matmuls. This is unconditional (no env gate):
                // a separate-mlp full-attn layer has no merged gate_up fallback.
                x = residual2
                    + mlp_separate(
                        xn2,
                        layer.full.gate_proj,
                        layer.full.up_proj,
                        layer.full.down,
                        prefer_verify_m16);
            } else {
                auto& gu = layer.is_gdr ? layer.gdr.gate_up : layer.full.gate_up;
                auto& dw = layer.is_gdr ? layer.gdr.down : layer.full.down;
                int gd = layer.is_gdr ? layer.gdr.gate_dim : layer.full.gate_dim;
                x = residual2 + mlp(xn2, gu, dw, gd, prefer_verify_m16);
            }
            if (op_prof) {
                eval(x);
                op_ms[3] += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - op_mlp_s).count();
            }

            // Keep key intermediates alive for GPU buffer reuse.
            if (keep_intermediates) {
                auto& intermediates = artifacts->intermediates;
                intermediates.push_back(xn);
                intermediates.push_back(attn_out);
                intermediates.push_back(xn2);
            }

            // DFlash: capture hidden states at specified layers.
            if (contains_layer_id(ctx.capture_layer_ids, i)) {
                captured_hidden.push_back(x);
            }
        }

        // Final norm + lm_head
        auto final_x = fast::rms_norm(x, final_norm_w, rms_eps);
        if (ctx.last_logits_only && ctx.seq_len > 1) {
            final_x = slice(
                final_x,
                {0, ctx.seq_len - 1, 0},
                {B, ctx.seq_len, hidden_size}
            );
        }
        // Use quantized matmul for tied lm_head (same as mlx_lm's as_linear).
        // Dense bf16 matmul reads 1.2GB vs quantized reads 0.3GB — 7.5ms difference.
        auto logits = use_embed_as_linear
            ? embed_as_linear.apply(final_x, prefer_verify_m16)
            : lm_head.apply(final_x, prefer_verify_m16);
        if (op_prof) {
            auto s = std::chrono::high_resolution_clock::now();
            eval(logits);
            op_ms[4] += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - s).count();
            if (++op_steps % 16 == 0) {
                double n = (double)op_steps;
                fprintf(stderr,
                    "[op-profile] %ld steps avg/step: embed=%.2f gdr=%.2f full=%.2f "
                    "mlp=%.2f head=%.2f total=%.2f ms\n",
                    op_steps, op_ms[0] / n, op_ms[1] / n, op_ms[2] / n, op_ms[3] / n,
                    op_ms[4] / n,
                    (op_ms[0] + op_ms[1] + op_ms[2] + op_ms[3] + op_ms[4]) / n);
            }
        }

        // Build output: [logits, kv_caches..., gdr_states..., captured_hidden...]
        std::vector<array> outputs;
        outputs.reserve(1 + full_kv_count + 2*G + captured_hidden.size());
        outputs.push_back(std::move(logits));
        for (auto& kv : new_kv_caches) outputs.push_back(std::move(kv));
        for (int j = 0; j < G; ++j) {
            outputs.push_back(std::move(new_gdr_states[j]));
            outputs.push_back(std::move(new_conv_states[j]));
        }
        for (auto& h : captured_hidden) outputs.push_back(std::move(h));
        return outputs;
    }

    std::vector<array> forward(const std::vector<array>& inputs) const {
        ForwardContext ctx;
        ctx.cache_pos = current_cache_pos;
        ctx.seq_len = current_seq_len;
        ctx.batch_size = current_batch_size;
        ctx.last_logits_only = current_last_logits_only;
        ctx.is_verify = current_is_verify;
        ctx.kv_cache_int8 = current_kv_cache_int8;
        ctx.keep_intermediates = keep_step_intermediates(current_seq_len);
        ctx.record_tapes = tape_mode;
        ctx.capture_layer_ids = &capture_layer_ids;

        ForwardArtifacts artifacts;
        if (ctx.keep_intermediates) {
            artifacts.intermediates.reserve(current_seq_len == 1 ? 2048 : 128);
        }
        if (ctx.record_tapes) {
            artifacts.gdr_tapes.reserve(n_gdr);
        }

        auto outputs = forward_impl(inputs, ctx, &artifacts);
        intermediates = std::move(artifacts.intermediates);
        gdr_tapes = std::move(artifacts.gdr_tapes);
        return outputs;
    }

    void prepare_forward() {
        // The Rust builder sets `model_has_qk_gate` explicitly per family
        // (Qwen3 → false, Qwen3.5 → true). A `n_gdr > 0` heuristic would
        // misclassify dense-only Qwen3.5 checkpoints (no GDR layers but
        // Q is still gated) as Qwen3.
        for (auto& lw : layers) {
            if (!lw.is_gdr) {
                lw.full.has_qk_gate = model_has_qk_gate;
            }
        }

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


QWeight& qwen35_weight_by_id(Qwen35CompiledModel* model, int32_t id) {
    if (id < 0 || id >= static_cast<int32_t>(model->weight_pool.size())) {
        throw std::runtime_error("invalid Qwen3.5 compiled weight id");
    }
    return model->weight_pool[static_cast<size_t>(id)];
}

extern "C" {

void* qwen35_compiled_new() {
    MLX_TRY_RETURN(new Qwen35CompiledModel());
}

void qwen35_compiled_free(void* model) {
    MLX_TRY_VOID(delete static_cast<Qwen35CompiledModel*>(model));
}

int32_t qwen35_compiled_add_dense_weight(void* model, mlx_array* w) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->weight_pool.push_back({*to_arr(w), array(0), array(0), 0, 0, true});
        return static_cast<int32_t>(m->weight_pool.size() - 1);
    }());
}

int32_t qwen35_compiled_add_quant_weight(
    void* model,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,
    int32_t group_size,
    int32_t bits,
    int32_t mode) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->weight_pool.push_back({
            *to_arr(w), *to_arr(scales),
            biases ? *to_arr(biases) : array(0),
            group_size, bits, false, mode});
        return static_cast<int32_t>(m->weight_pool.size() - 1);
    }());
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

void qwen35_compiled_set_qk_gate(void* model, int32_t enabled) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->model_has_qk_gate = enabled != 0;
    });
}

void qwen35_compiled_set_embed_v2(
    void* model,
    mlx_array* embed_tokens,
    mlx_array* final_norm_w,
    int32_t lm_head_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->embed_tokens = embed_tokens == nullptr ? array(0) : *to_arr(embed_tokens);
        m->final_norm_w = *to_arr(final_norm_w);
        m->lm_head = qwen35_weight_by_id(m, lm_head_id);
        m->use_embed_as_linear = false;
    });
}

void qwen35_compiled_set_embed_as_linear_v2(void* model, int32_t embed_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->embed_as_linear = qwen35_weight_by_id(m, embed_id);
        m->use_embed_as_linear = true;
    });
}

void qwen35_compiled_push_full_attn_v2(
    void* model,
    mlx_array* input_ln,
    mlx_array* post_ln,
    int32_t q_id,
    int32_t k_id,
    int32_t v_id,
    int32_t o_id,
    mlx_array* q_norm,
    mlx_array* k_norm,
    int32_t gate_up_id,
    int32_t gate_dim,
    int32_t down_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        LayerWeights lw;
        lw.is_gdr = false;
        lw.full.input_ln_w = *to_arr(input_ln);
        lw.full.post_attn_ln_w = *to_arr(post_ln);
        lw.full.q_proj = qwen35_weight_by_id(m, q_id);
        lw.full.k_proj = qwen35_weight_by_id(m, k_id);
        lw.full.v_proj = qwen35_weight_by_id(m, v_id);
        lw.full.o_proj = qwen35_weight_by_id(m, o_id);
        lw.full.q_norm_w = *to_arr(q_norm);
        lw.full.k_norm_w = *to_arr(k_norm);
        // down is always set when present; gate_up only when a merged projection
        // exists (gate_up_id < 0 for mixed-bit MLP, which uses set_full_separate_mlp).
        if (down_id >= 0) {
            lw.full.down = qwen35_weight_by_id(m, down_id);
        }
        if (gate_up_id >= 0) {
            lw.full.gate_up = qwen35_weight_by_id(m, gate_up_id);
            lw.full.gate_dim = gate_dim;
        }
        m->layers.push_back(std::move(lw));
        m->n_full_attn++;
    });
}

void qwen35_compiled_push_gdr_v2(
    void* model,
    mlx_array* input_ln,
    mlx_array* post_ln,
    int32_t qkvz_id,
    int32_t qkv_split,
    int32_t z_split,
    int32_t ba_id,
    int32_t ba_num_heads,
    mlx_array* conv1d_w,
    int32_t conv_kernel,
    mlx_array* a_log,
    mlx_array* dt_bias,
    mlx_array* norm_w,
    float gdr_rms_eps,
    int32_t out_id,
    int32_t num_key_heads,
    int32_t key_dim,
    int32_t num_value_heads,
    int32_t value_dim,
    int32_t gate_up_id,
    int32_t gate_dim,
    int32_t down_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        LayerWeights lw;
        lw.is_gdr = true;
        lw.gdr.input_ln_w = *to_arr(input_ln);
        lw.gdr.post_attn_ln_w = *to_arr(post_ln);
        if (qkvz_id >= 0) {
            lw.gdr.qkvz_proj = qwen35_weight_by_id(m, qkvz_id);
        }
        lw.gdr.qkv_split = qkv_split;
        lw.gdr.z_split = z_split;
        if (ba_id >= 0) {
            lw.gdr.ba_proj = qwen35_weight_by_id(m, ba_id);
        }
        lw.gdr.ba_num_heads = ba_num_heads;
        lw.gdr.conv1d_w = *to_arr(conv1d_w);
        lw.gdr.conv_kernel = conv_kernel;
        lw.gdr.a_log = negative(exp(astype(*to_arr(a_log), float32)));
        lw.gdr.dt_bias = *to_arr(dt_bias);
        lw.gdr.norm_w = *to_arr(norm_w);
        lw.gdr.rms_eps = gdr_rms_eps;
        lw.gdr.out_proj = qwen35_weight_by_id(m, out_id);
        lw.gdr.num_key_heads = num_key_heads;
        lw.gdr.key_dim = key_dim;
        lw.gdr.num_value_heads = num_value_heads;
        lw.gdr.value_dim = value_dim;
        float inv = 1.0f / std::sqrt((float)key_dim);
        lw.gdr.q_scale_arr = astype(array(inv * inv), bfloat16);
        lw.gdr.k_scale_arr = astype(array(inv), bfloat16);
        // down is always set when present; gate_up only when a merged projection
        // exists (gate_up_id < 0 for mixed-bit MLP, which uses set_separate_mlp).
        if (down_id >= 0) {
            lw.gdr.down = qwen35_weight_by_id(m, down_id);
        }
        if (gate_up_id >= 0) {
            lw.gdr.gate_up = qwen35_weight_by_id(m, gate_up_id);
            lw.gdr.gate_dim = gate_dim;
        }
        m->layers.push_back(std::move(lw));
        m->n_gdr++;
    });
}

void qwen35_compiled_set_last_moe_mlp(
    void* model,
    mlx_array* router_w, mlx_array* router_s, mlx_array* router_b, int32_t router_gs, int32_t router_bits,
    mlx_array* expert_gate_w, mlx_array* expert_gate_s, mlx_array* expert_gate_b,
    mlx_array* expert_up_w, mlx_array* expert_up_s, mlx_array* expert_up_b,
    mlx_array* expert_down_w, mlx_array* expert_down_s, mlx_array* expert_down_b,
    int32_t expert_gs, int32_t expert_bits,
    mlx_array* shared_gate_w, mlx_array* shared_gate_s, mlx_array* shared_gate_b,
    mlx_array* shared_up_w, mlx_array* shared_up_s, mlx_array* shared_up_b,
    mlx_array* shared_down_w, mlx_array* shared_down_s, mlx_array* shared_down_b,
    mlx_array* shared_gate_router_w, mlx_array* shared_gate_router_s, mlx_array* shared_gate_router_b,
    int32_t num_experts, int32_t top_k, bool norm_topk_prob
) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        if (m->layers.empty()) {
            throw std::runtime_error("qwen35_compiled_set_last_moe_mlp requires an existing layer");
        }
        auto& lw = m->layers.back();
        lw.has_moe = true;
        lw.moe.router = {*to_arr(router_w), *to_arr(router_s), *to_arr(router_b), router_gs, router_bits};
        lw.moe.switch_gate = {*to_arr(expert_gate_w), *to_arr(expert_gate_s), *to_arr(expert_gate_b), expert_gs, expert_bits};
        lw.moe.switch_up = {*to_arr(expert_up_w), *to_arr(expert_up_s), *to_arr(expert_up_b), expert_gs, expert_bits};
        lw.moe.switch_down = {*to_arr(expert_down_w), *to_arr(expert_down_s), *to_arr(expert_down_b), expert_gs, expert_bits};
        lw.moe.shared_gate = {*to_arr(shared_gate_w), *to_arr(shared_gate_s), *to_arr(shared_gate_b), expert_gs, expert_bits};
        lw.moe.shared_up = {*to_arr(shared_up_w), *to_arr(shared_up_s), *to_arr(shared_up_b), expert_gs, expert_bits};
        lw.moe.shared_down = {*to_arr(shared_down_w), *to_arr(shared_down_s), *to_arr(shared_down_b), expert_gs, expert_bits};
        lw.moe.shared_expert_gate = {
            *to_arr(shared_gate_router_w),
            *to_arr(shared_gate_router_s),
            *to_arr(shared_gate_router_b),
            router_gs,
            router_bits,
        };
        lw.moe.num_experts = num_experts;
        lw.moe.top_k = top_k;
        lw.moe.norm_topk_prob = norm_topk_prob;
        lw.moe.router_bits = router_bits;
        lw.moe.router_group_size = router_gs;
        lw.moe.expert_bits = expert_bits;
        lw.moe.expert_group_size = expert_gs;
    });
}

void qwen35_compiled_set_separate_proj_v2(
    void* model,
    int32_t qkv_id,
    int32_t z_id,
    int32_t b_id,
    int32_t a_id,
    int32_t gate_id,
    int32_t up_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        auto& lw = m->layers.back().gdr;
        lw.qkv_proj = qwen35_weight_by_id(m, qkv_id);
        lw.z_proj = qwen35_weight_by_id(m, z_id);
        lw.b_proj = qwen35_weight_by_id(m, b_id);
        lw.a_proj = qwen35_weight_by_id(m, a_id);
        if (gate_id >= 0 && up_id >= 0) {
            lw.gate_proj = qwen35_weight_by_id(m, gate_id);
            lw.up_proj = qwen35_weight_by_id(m, up_id);
            lw.has_separate_mlp = true;
        }
        lw.use_separate_proj = true;
    });
}

void qwen35_compiled_set_separate_mlp_v2(
    void* model,
    int32_t gate_id,
    int32_t up_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        auto& lw = m->layers.back().gdr;
        lw.gate_proj = qwen35_weight_by_id(m, gate_id);
        lw.up_proj = qwen35_weight_by_id(m, up_id);
        lw.has_separate_mlp = true;
    });
}

// Set separate gate/up MLP projections for the last-pushed FULL-ATTENTION layer.
// Used for mixed-bit MLP (OptiQ gate=4-bit/up=8-bit) that cannot row-merge into
// a single quantized gate_up. Call AFTER qwen35_compiled_push_full_attn_v2 with
// gate_up_id=-1/down_id passed for the down projection only.
void qwen35_compiled_set_full_separate_mlp_v2(
    void* model,
    int32_t gate_id,
    int32_t up_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        auto& lw = m->layers.back().full;
        lw.gate_proj = qwen35_weight_by_id(m, gate_id);
        lw.up_proj = qwen35_weight_by_id(m, up_id);
        lw.has_separate_mlp = true;
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

int32_t qwen35_session_begin(
    void* model,
    mlx_array** kv_caches, int32_t n_kv,
    mlx_array** gdr_states, int32_t n_gdr
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    mlx_clear_error();

    if (m->session_active) {
        mlx_set_error("qwen35_session_begin requires an inactive session");
        return -1;
    }
    if (n_kv < 0 || n_gdr < 0) {
        mlx_set_error("qwen35_session_begin received negative cache counts");
        return -1;
    }

    try {
        bool kv_int8 = false;
        if (m->n_full_attn == 0) {
            if (n_kv != 0) {
                throw std::runtime_error("qwen35_session_begin expected zero KV caches");
            }
        } else if (n_kv == 2 * m->n_full_attn) {
            kv_int8 = false;
        } else if (n_kv == 6 * m->n_full_attn) {
            kv_int8 = true;
        } else {
            throw std::runtime_error(
                "qwen35_session_begin KV cache count must be 2*full_layers (bf16) "
                "or 6*full_layers (int8)");
        }
        std::vector<array> session_kv_caches;
        std::vector<array> session_gdr_states;
        session_kv_caches.reserve(n_kv);
        session_gdr_states.reserve(n_gdr);
        for (int i = 0; i < n_kv; ++i) {
            session_kv_caches.push_back(*to_arr(kv_caches[i]));
        }
        for (int i = 0; i < n_gdr; ++i) {
            session_gdr_states.push_back(*to_arr(gdr_states[i]));
        }

        m->session_kv_caches = std::move(session_kv_caches);
        m->session_gdr_states = std::move(session_gdr_states);
        m->current_kv_cache_int8 = kv_int8;
        m->clear_optional_batch_inputs();
        m->session_active = true;
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

int32_t qwen35_session_end(
    void* model,
    mlx_array** out_kv_caches, int32_t n_kv,
    mlx_array** out_gdr_states, int32_t n_gdr
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    mlx_clear_error();

    if (!m->session_active) {
        mlx_set_error("qwen35_session_end requires an active session");
        return -1;
    }
    if (n_kv < 0 || n_gdr < 0) {
        mlx_set_error("qwen35_session_end received negative cache counts");
        return -1;
    }
    if (static_cast<int32_t>(m->session_kv_caches.size()) != n_kv ||
        static_cast<int32_t>(m->session_gdr_states.size()) != n_gdr) {
        mlx_set_error("qwen35_session_end cache counts do not match the active session");
        return -1;
    }

    try {
        for (int i = 0; i < n_kv; ++i) {
            out_kv_caches[i] = from_arr(std::move(m->session_kv_caches[i]));
        }
        for (int i = 0; i < n_gdr; ++i) {
            out_gdr_states[i] = from_arr(std::move(m->session_gdr_states[i]));
        }

        m->session_kv_caches.clear();
        m->session_gdr_states.clear();
        m->current_kv_cache_int8 = false;
        m->session_active = false;
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

// Env-gated MTLCaptureManager hook — default no-op, enabled by
// INFER_CAPTURE_STEP=N (see crates/mlx-sys/src/mlx_metal_capture.mm).
extern "C" int32_t maybe_capture_qwen35_step_begin(void);
extern "C" void maybe_capture_qwen35_step_end(int32_t started);

int32_t qwen35_compiled_step_session(
    void* model,
    mlx_array* token_id,
    int32_t cache_pos,
    mlx_array** out_logits
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    const int32_t capture_started = maybe_capture_qwen35_step_begin();
    try {
        mlx_clear_error();

        if (!m->session_active) {
            throw std::runtime_error("qwen35_compiled_step_session requires an active session");
        }

        const int32_t n_kv = static_cast<int32_t>(m->session_kv_caches.size());
        const int32_t n_gdr = static_cast<int32_t>(m->session_gdr_states.size());

        m->current_cache_pos = cache_pos;
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->clear_optional_batch_inputs();
        m->current_kv_cache_int8 = (n_kv == 6 * m->n_full_attn && m->n_full_attn > 0);

        std::vector<array> inputs;
        inputs.reserve(1 + n_kv + n_gdr);
        inputs.push_back(*to_arr(token_id));
        for (const auto& kv : m->session_kv_caches) {
            inputs.push_back(kv);
        }
        for (const auto& gdr : m->session_gdr_states) {
            inputs.push_back(gdr);
        }

        m->prev_outputs = m->forward(inputs);
        auto& outputs = m->prev_outputs;

        // Force GPU work to flush inside the capture window so the .gputrace
        // actually contains this step's dispatches. Do it BEFORE mutating
        // session state: if eval() throws (OOM / Metal runtime error), the
        // catch below fires without `outputs` having been moved-from, without
        // `*out_logits` being set, and without the session caches being
        // advanced — caller observes -1 with clean rollback.
        // No-op branch when capture is disabled.
        if (capture_started) {
            eval(outputs);
        }

        std::vector<array> next_kv_caches;
        std::vector<array> next_gdr_states;
        next_kv_caches.reserve(n_kv);
        next_gdr_states.reserve(n_gdr);
        for (int i = 0; i < n_kv; ++i) {
            next_kv_caches.push_back(std::move(outputs[1 + i]));
        }
        for (int i = 0; i < n_gdr; ++i) {
            next_gdr_states.push_back(std::move(outputs[1 + n_kv + i]));
        }

        auto* logits = from_arr(std::move(outputs[0]));
        m->session_kv_caches = std::move(next_kv_caches);
        m->session_gdr_states = std::move(next_gdr_states);
        *out_logits = logits;
        maybe_capture_qwen35_step_end(capture_started);
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        maybe_capture_qwen35_step_end(capture_started);
        return -1;
    }
}

int32_t qwen35_compiled_prefill_session(
    void* model,
    mlx_array* token_ids,
    int32_t prompt_len,
    int32_t cache_pos,
    mlx_array** out_logits
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    try {
        mlx_clear_error();

        if (!m->session_active) {
            throw std::runtime_error("qwen35_compiled_prefill_session requires an active session");
        }

        const int32_t n_kv = static_cast<int32_t>(m->session_kv_caches.size());
        const int32_t n_gdr = static_cast<int32_t>(m->session_gdr_states.size());

        m->current_cache_pos = cache_pos;
        m->current_batch_size = 1;
        m->current_seq_len = prompt_len;
        m->current_last_logits_only = use_qwen35_cpp_prefill_last_logits_only();
        m->clear_optional_batch_inputs();
        m->current_kv_cache_int8 = (n_kv == 6 * m->n_full_attn && m->n_full_attn > 0);

        std::vector<array> inputs;
        inputs.reserve(1 + n_kv + n_gdr);
        inputs.push_back(*to_arr(token_ids));
        for (const auto& kv : m->session_kv_caches) {
            inputs.push_back(kv);
        }
        for (const auto& gdr : m->session_gdr_states) {
            inputs.push_back(gdr);
        }

        m->prev_outputs = m->forward(inputs);
        auto& outputs = m->prev_outputs;

        std::vector<array> next_kv_caches;
        std::vector<array> next_gdr_states;
        next_kv_caches.reserve(n_kv);
        next_gdr_states.reserve(n_gdr);
        for (int i = 0; i < n_kv; ++i) {
            next_kv_caches.push_back(std::move(outputs[1 + i]));
        }
        for (int i = 0; i < n_gdr; ++i) {
            next_gdr_states.push_back(std::move(outputs[1 + n_kv + i]));
        }

        auto* logits = from_arr(std::move(outputs[0]));
        m->session_kv_caches = std::move(next_kv_caches);
        m->session_gdr_states = std::move(next_gdr_states);
        *out_logits = logits;

        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->clear_optional_batch_inputs();
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->clear_optional_batch_inputs();
        return -1;
    }
}

int32_t qwen35_compiled_verify_block_summary(
    void* model,
    mlx_array* token_ids,    // int32 [block_size]
    int32_t block_size,
    int32_t cache_pos,
    mlx_array** kv_caches, int32_t n_kv,
    mlx_array** gdr_states, int32_t n_gdr,
    float temperature,
    bool greedy,
    int32_t suppress_token_id,
    int32_t accept_topk,
    int32_t* out_matched_prefix_len,
    int32_t* out_next_token,
    mlx_array** out_kv_caches,
    mlx_array** out_gdr_states
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    try {
        mlx_clear_error();

        if (out_matched_prefix_len == nullptr || out_next_token == nullptr) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_summary requires non-null summary outputs");
        }
        if (block_size <= 0) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_summary requires block_size > 0");
        }

        auto tokens = *to_arr(token_ids);
        if (tokens.dtype() != int32) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_summary requires int32 token_ids");
        }
        if (tokens.ndim() != 1 || tokens.shape(0) != block_size) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_summary requires token_ids shape [block_size]");
        }

        m->current_cache_pos = cache_pos;
        m->current_batch_size = 1;
        m->current_seq_len = block_size;
        m->current_last_logits_only = false;
        m->clear_optional_batch_inputs();
        m->current_is_verify = true;
        m->current_kv_cache_int8 = (n_kv == 6 * m->n_full_attn && m->n_full_attn > 0);

        std::vector<array> inputs;
        inputs.reserve(1 + n_kv + n_gdr);
        inputs.push_back(*to_arr(token_ids));
        for (int i = 0; i < n_kv; ++i) inputs.push_back(*to_arr(kv_caches[i]));
        for (int i = 0; i < n_gdr; ++i) inputs.push_back(*to_arr(gdr_states[i]));

        m->prev_outputs = m->forward(inputs);
        auto& outputs = m->prev_outputs;

        auto logits = outputs[0];
        logits = suppress_last_axis_token(logits, suppress_token_id);
        auto sampled = greedy
            ? argmax(logits, -1, false)
            : random::categorical(logits * array(1.0f / temperature), -1);
        sampled = reshape(sampled, {block_size});
        eval(sampled);
        eval(tokens);
        const int32_t* sampled_data = sampled.data<int32_t>();
        const int32_t* token_data = tokens.data<int32_t>();

        int32_t matched_prefix_len = 0;
        const int32_t drafted_len = block_size - 1;
        if (accept_topk > 1) {
            // Top-k acceptance (option A): accept a draft token iff it lies in the
            // target's top-k for that position, then commit the DRAFT token. Lossy
            // vs exact greedy spec-decode (deviates from target argmax) — a
            // speed-vs-quality knob. Tie-robust: count tokens strictly greater than
            // the draft's logit; in top-k iff that count < k.
            const int32_t vocab = static_cast<int32_t>(logits.shape(-1));
            const int32_t kk = std::min(accept_topk, vocab);
            // logits may be [1, block_size, vocab]; normalise to [block_size, vocab].
            auto logits2d = reshape(logits, {block_size, vocab});
            auto draft_ids = reshape(slice(tokens, {1}, {block_size}), {drafted_len, 1});
            auto logits_pref = slice(logits2d, {0, 0}, {drafted_len, vocab});
            auto v_draft = take_along_axis(logits_pref, draft_ids, -1);
            auto num_greater = sum(astype(greater(logits_pref, v_draft), int32), -1, false);
            auto accept = astype(less(num_greater, array(kk)), int32);
            eval(accept);
            const int32_t* acc = accept.data<int32_t>();
            while (matched_prefix_len < drafted_len && acc[matched_prefix_len] != 0) {
                matched_prefix_len += 1;
            }
        } else {
            while (matched_prefix_len < drafted_len &&
                   sampled_data[matched_prefix_len] == token_data[matched_prefix_len + 1]) {
                matched_prefix_len += 1;
            }
        }
        *out_matched_prefix_len = matched_prefix_len;
        *out_next_token = sampled_data[matched_prefix_len];

        for (int i = 0; i < n_kv; ++i) {
            out_kv_caches[i] = from_arr(std::move(outputs[1 + i]));
        }
        for (int i = 0; i < n_gdr; ++i) {
            out_gdr_states[i] = from_arr(std::move(outputs[1 + n_kv + i]));
        }

        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->current_kv_cache_int8 = false;
        m->clear_optional_batch_inputs();
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->current_kv_cache_int8 = false;
        m->clear_optional_batch_inputs();
        return -1;
    }
}

void qwen35_set_tape_mode(void* model, bool enabled) {
    auto* m = reinterpret_cast<Qwen35CompiledModel*>(model);
    m->tape_mode = enabled;
    if (!enabled) m->gdr_tapes.clear();
}

int32_t qwen35_read_and_clear_gdr_tapes(
    void* model,
    mlx_array** out_tapes,
    mlx_array** out_k,
    mlx_array** out_g,
    mlx_array** out_qkv,
    int32_t capacity
) {
    try {
        auto* m = reinterpret_cast<Qwen35CompiledModel*>(model);
        auto tape_count = static_cast<int32_t>(m->gdr_tapes.size());
        if (capacity < tape_count) {
            throw std::runtime_error("gdr tape output buffer too small");
        }
        for (int32_t idx = 0; idx < tape_count; ++idx) {
            auto& tape = m->gdr_tapes[idx];
            out_tapes[idx] = reinterpret_cast<mlx_array*>(new array(tape.innovation_tape));
            out_k[idx] = reinterpret_cast<mlx_array*>(new array(tape.k));
            out_g[idx] = reinterpret_cast<mlx_array*>(new array(tape.g));
            out_qkv[idx] = reinterpret_cast<mlx_array*>(new array(tape.qkv));
        }
        m->gdr_tapes.clear();
        return tape_count;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}


void qwen35_set_capture_layers(void* model, const int32_t* layer_ids, int32_t count) {
    auto* m = reinterpret_cast<Qwen35CompiledModel*>(model);
    m->capture_layer_ids.clear();
    if (layer_ids && count > 0) {
        m->capture_layer_ids.assign(layer_ids, layer_ids + count);
    }
}

int32_t qwen35_get_captured_hidden_count(void* model) {
    auto* m = reinterpret_cast<Qwen35CompiledModel*>(model);
    int capture_count = static_cast<int>(m->capture_layer_ids.size());
    if (capture_count <= 0) return 0;
    auto& outputs = m->prev_outputs;
    if ((int)outputs.size() < capture_count) return 0;
    return static_cast<int32_t>(capture_count);
}

/// Get a captured hidden state by index. Returns new array handle (caller must free).
int32_t qwen35_get_captured_hidden(void* model, int32_t idx, mlx_array** out) {
    try {
        auto* m = reinterpret_cast<Qwen35CompiledModel*>(model);
        int capture_count = static_cast<int>(m->capture_layer_ids.size());
        auto& outputs = m->prev_outputs;
        if (capture_count <= 0)
            throw std::out_of_range("no captured hidden states are active");
        if ((int)outputs.size() < capture_count)
            throw std::out_of_range("captured hidden output tail is shorter than capture count");
        int hi = static_cast<int>(outputs.size()) - capture_count + idx;
        if (hi < 0 || hi >= (int)outputs.size())
            throw std::out_of_range("captured hidden index out of range");
        *out = reinterpret_cast<mlx_array*>(new array(outputs[hi]));
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

} // extern "C"

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
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <stdexcept>

extern "C" mlx_array* qwen35_moe_block_forward(
    mlx_array* hidden,
    mlx_array* router_w,
    mlx_array* router_scales,
    mlx_array* router_biases,
    int32_t router_bits,
    int32_t router_group_size,
    mlx_array* expert_gate_w,
    mlx_array* expert_gate_scales,
    mlx_array* expert_gate_biases,
    mlx_array* expert_up_w,
    mlx_array* expert_up_scales,
    mlx_array* expert_up_biases,
    mlx_array* expert_down_w,
    mlx_array* expert_down_scales,
    mlx_array* expert_down_biases,
    int32_t expert_bits,
    int32_t expert_group_size,
    mlx_array* shared_gate_w,
    mlx_array* shared_gate_scales,
    mlx_array* shared_gate_biases,
    mlx_array* shared_up_w,
    mlx_array* shared_up_scales,
    mlx_array* shared_up_biases,
    mlx_array* shared_down_w,
    mlx_array* shared_down_scales,
    mlx_array* shared_down_biases,
    mlx_array* shared_gate_router_w,
    mlx_array* shared_gate_router_scales,
    mlx_array* shared_gate_router_biases,
    int32_t num_experts,
    int32_t top_k,
    bool norm_topk_prob);

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

bool use_gdr_metal_kernel() {
    const char* env = std::getenv("AGENT_INFER_GDR_METAL_KERNEL");
    return !(env && std::string(env) == "0");
}

bool keep_prefill_intermediates() {
    const char* env = std::getenv("AGENT_INFER_QWEN35_CPP_KEEP_PREFILL_INTERMEDIATES");
    return env && std::string(env) != "0";
}

bool use_qwen35_cpp_clear_cache() {
    const char* env = std::getenv("AGENT_INFER_QWEN35_CPP_CLEAR_CACHE");
    return !(env && std::string(env) == "0");
}

bool use_qwen35_cpp_prefill_last_logits_only() {
    const char* env = std::getenv("AGENT_INFER_QWEN35_CPP_PREFILL_LAST_LOGITS_ONLY");
    return !(env && std::string(env) == "0");
}

bool use_qwen35_cpp_separate_mlp() {
    const char* env = std::getenv("AGENT_INFER_QWEN35_CPP_SEPARATE_MLP");
    if (env) {
        return std::string(env) != "0";
    }
    // Revalidation with longer runs shows separate gate/up matmuls are
    // equal-or-better across mixed, decode-heavy, and prefill-heavy
    // workloads, so keep the default simple.
    return true;
}

bool use_qwen35_cpp_prefill_gbeta_helper() {
    const char* env = std::getenv("AGENT_INFER_QWEN35_CPP_PREFILL_GBETA_HELPER");
    return !(env && std::string(env) == "0");
}

bool use_qwen35_cpp_qk_norm_helper() {
    const char* env = std::getenv("AGENT_INFER_QWEN35_CPP_QK_NORM_HELPER");
    return !(env && std::string(env) == "0");
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
    int fallback = parse_env_int("AGENT_INFER_QWEN35_CPP_GDR_TG_Y", 4);
    if (seq_len > 1) {
        return parse_env_int("AGENT_INFER_QWEN35_CPP_PREFILL_GDR_TG_Y", fallback);
    }
    return parse_env_int("AGENT_INFER_QWEN35_CPP_DECODE_GDR_TG_Y", fallback);
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

// ── Quantized linear helper ────────────────────────────────────────────────

struct QWeight {
    array w = array(0);
    array scales = array(0);
    array biases = array(0);
    int group_size = 64;
    int bits = 4;
    bool is_dense = false;  // if true, w is already transposed, use matmul directly
    int gguf_format = 0;
    int rows = 0;
    int cols = 0;

    array apply(const array& x, bool prefer_verify_m16 = false) const {
        if (gguf_format != 0) {
            return gguf_quantized_matmul_cpp(x, w, gguf_format, rows, cols);
        }
        if (is_dense) {
            return matmul(x, w);  // w is already transposed at load time
        }
        if (prefer_verify_m16) {
            return verify_quantized_matmul_cpp(x, w, scales, biases, group_size, bits);
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

// ── Model struct ───────────────────────────────────────────────────────────

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
        bool has_attn_mask = false;
        array attn_mask = array(0);
        bool has_cache_pos_arr = false;
        const int32_t* cache_pos_arr = nullptr;
        bool has_rope_offsets = false;
        array rope_offsets = array(0);
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
    QWeight embed_packed;
    bool use_packed_embed = false;
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
    mutable array current_attn_mask = array(0);
    mutable bool current_has_attn_mask = false;
    // Batched verify can carry a per-row physical KV write position.
    //
    // RoPE offsets already encode each row's logical token positions, but
    // `cache_pos` is also used to select the KV slice-update window. Route 2
    // therefore threads a host int32[B] cache_pos slice through the forward
    // context and uses it only when the batched verify entrypoint requests it.
    mutable const int32_t* current_cache_pos_arr = nullptr;
    mutable bool current_has_cache_pos_arr = false;
    // Per-row RoPE offsets (int32 array of length batch_size).
    //
    // Workaround for MLX 0.31.1: `fast::rope(..., int offset)` on a
    // `[B, H, S=1, D]` input with `B > 1` silently zeroes batch rows > 0.
    // The array-offset overload (`fast::rope(..., const array& offset)`) works
    // correctly for B=1 AND B>1, so we always route batched-decode RoPE
    // through it. The same array slot also carries per-row offsets for
    // variable-length batches (each row rotated at its own logical position).
    mutable array current_rope_offsets = array(0);
    mutable bool current_has_rope_offsets = false;
    // Keep previous step's arrays alive to prevent premature GPU buffer release.
    // This mimics Python's lazy GC behavior where intermediates survive until
    // the next GC cycle, allowing MLX to reuse GPU buffers efficiently.
    std::vector<array> prev_outputs;
    // Session state for FFI-cost-amortized single-request decode.
    std::vector<array> session_kv_caches;   // [k0, v0, k1, v1, ...]
    std::vector<array> session_gdr_states;  // [gdr0, conv0, gdr1, conv1, ...]
    bool session_active = false;
    // Collect ALL intermediate arrays during forward() to keep them alive.
    // Cleared at start of each step, populated during forward().
    mutable std::vector<array> intermediates;

    // ── DFlash support ────────────────────────────────────────────────
    // When tape_mode is on, gdr_step() records innovation tapes for each GDR layer.
    bool tape_mode = false;
    bool gdr_metal_kernel_enabled = true;
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
        return lw.has_separate_mlp && use_qwen35_cpp_separate_mlp();
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
        current_attn_mask = array(0);
        current_has_attn_mask = false;
        current_cache_pos_arr = nullptr;
        current_has_cache_pos_arr = false;
        current_rope_offsets = array(0);
        current_has_rope_offsets = false;
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
        if (!ctx.is_verify || ctx.has_attn_mask || ctx.seq_len != 16) {
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

    // ── Full attention decode step ─────────────────────────────────────

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

        if (ctx.has_rope_offsets) {
            // Array-offset path. Handles B>1 correctly AND carries per-row
            // logical positions for variable-length decode (each row's
            // offset = batch_cache_len - left_padding[row]).
            q = fast::rope(q, rotary_dim, false, rope_theta, 1.0f, ctx.rope_offsets);
            k = fast::rope(k, rotary_dim, false, rope_theta, 1.0f, ctx.rope_offsets);
        } else {
            // Scalar path. Only safe for B == 1 (prefill or single-request
            // decode); batched decode callers MUST set rope_offsets.
            q = fast::rope(q, rotary_dim, false, rope_theta, 1.0f, cache_pos);
            k = fast::rope(k, rotary_dim, false, rope_theta, 1.0f, cache_pos);
        }

        auto v = reshape(v_raw, {B, S, nkv, hd});
        v = transpose(v, {0, 2, 1, 3});

        array k_full(0), v_full(0);
        if (ctx.has_cache_pos_arr) {
            // Batched verify can mix rows with different physical KV cursors.
            // RoPE still comes from ctx.rope_offsets; cache_pos_arr is only for
            // where each row writes into the packed KV cache.
            const int32_t* cache_pos_data = ctx.cache_pos_arr;

            std::vector<array> new_k_rows;
            std::vector<array> new_v_rows;
            std::vector<array> k_full_rows;
            std::vector<array> v_full_rows;
            new_k_rows.reserve(B);
            new_v_rows.reserve(B);
            k_full_rows.reserve(B);
            v_full_rows.reserve(B);

            int key_len = 0;
            for (int b = 0; b < B; ++b) {
                int row_cache_pos = cache_pos_data[b];
                int row_end = row_cache_pos + S;
                key_len = std::max(key_len, row_end);

                auto k_cache_row = slice(k_cache, {b, 0, 0, 0}, {b + 1, nkv, k_cache.shape(2), hd});
                auto v_cache_row = slice(v_cache, {b, 0, 0, 0}, {b + 1, nkv, v_cache.shape(2), hd});
                auto k_row = slice(k, {b, 0, 0, 0}, {b + 1, nkv, S, hd});
                auto v_row = slice(v, {b, 0, 0, 0}, {b + 1, nkv, S, hd});

                auto new_k_row = slice_update(
                    k_cache_row,
                    k_row,
                    {0, 0, row_cache_pos, 0},
                    {1, nkv, row_end, hd});
                auto new_v_row = slice_update(
                    v_cache_row,
                    v_row,
                    {0, 0, row_cache_pos, 0},
                    {1, nkv, row_end, hd});
                new_k_rows.push_back(new_k_row);
                new_v_rows.push_back(new_v_row);
            }

            if (ctx.has_attn_mask) {
                key_len = ctx.attn_mask.shape(3);
            } else {
                for (int b = 0; b < B; ++b) {
                    int row_end = cache_pos_data[b] + S;
                    if (row_end != key_len) {
                        throw std::runtime_error(
                            "qwen35 batched verify requires attn_mask when cache_pos_arr differs across rows");
                    }
                }
            }

            for (int b = 0; b < B; ++b) {
                k_full_rows.push_back(slice(new_k_rows[b], {0, 0, 0, 0}, {1, nkv, key_len, hd}));
                v_full_rows.push_back(slice(new_v_rows[b], {0, 0, 0, 0}, {1, nkv, key_len, hd}));
            }

            new_k_cache = concatenate(new_k_rows, 0);
            new_v_cache = concatenate(new_v_rows, 0);
            k_full = concatenate(k_full_rows, 0);
            v_full = concatenate(v_full_rows, 0);
        } else {
            int end = cache_pos + S;
            new_k_cache = slice_update(k_cache, k, {0,0,cache_pos,0}, {B,nkv,end,hd});
            new_v_cache = slice_update(v_cache, v, {0,0,cache_pos,0}, {B,nkv,end,hd});
            k_full = slice(new_k_cache, {0,0,0,0}, {B,nkv,end,hd});
            v_full = slice(new_v_cache, {0,0,0,0}, {B,nkv,end,hd});
        }

        array attn_out(0);
        if (can_use_verify_sdpa_2pass(ctx, q, k_full, v_full, nh, nkv, hd)) {
            attn_out = batched_sdpa_2pass_cpp(q, k_full, v_full, attn_scale, nh / nkv);
        } else if (ctx.has_attn_mask) {
            attn_out = fast::scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                attn_scale,
                "",
                ctx.attn_mask);
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

    // ── GDR decode step ────────────────────────────────────────────────

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
        if (S > 1 && use_qwen35_cpp_prefill_gbeta_helper()) {
            auto gb = compiled_compute_g_beta()({lw.a_log, a_raw, lw.dt_bias, b_raw});
            g = gb[0];
            beta = gb[1];
        } else {
            beta = sigmoid(b_raw);
            g = compiled_compute_g()({lw.a_log, a_raw, lw.dt_bias})[0];
        }

        array y(0);
        if (gdr_metal_kernel_enabled && use_gdr_metal_kernel()) {
            auto& v_bf16 = v_raw;
            // g and beta are [1, S, Hv] — no reshape needed
            auto& g_3d = g;
            auto& beta_3d = beta;
            int threadgroup_y = qwen35_cpp_gdr_threadgroup_y(S);
            std::vector<array> inputs = {
                q, k, v_bf16, g_3d, beta_3d, gdr_state_in, gdr_t_arr
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
                // Record tape for rollback. tape_replay requires bf16 for
                // g/k/tape, but compute_g_impl produces f32 because neg_exp_a
                // is f32. Cast here so the tape kernel's dtype gate holds.
                artifacts->gdr_tapes.push_back({
                    std::move(result[2]),            // innovation_tape (bf16 from kernel)
                    astype(contiguous(k), bfloat16), // k
                    astype(contiguous(g_3d), bfloat16), // g (was f32)
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
                intermediates.push_back(g_3d);
                intermediates.push_back(beta_3d);
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

    // ── MLP block ──────────────────────────────────────────────────────

    // Separate MLP: 2 matmul (matching mlx_lm, no split overhead)
    array mlp_separate(
        const array& x,
        const QWeight& gate,
        const QWeight& up,
        const QWeight& down,
        bool prefer_verify_m16
    ) const {
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

    // ── Full forward pass ──────────────────────────────────────────────
    // inputs layout:
    //   [0]        : token ids / token batch
    //   [1..1+2*F) : k_cache_i, v_cache_i for F full-attn layers
    //   [1+2*F .. 1+2*F+2*G) : gdr_state_i, conv_state_i for G GDR layers
    // outputs layout:
    //   [0]        : logits
    //   [1..1+2*F) : new k/v caches
    //   [1+2*F .. 1+2*F+2*G) : new gdr/conv states

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
        auto x = use_packed_embed
            ? gguf_embedding_cpp(token_id, embed_packed.w, embed_packed.gguf_format, embed_packed.rows, embed_packed.cols)
            : take(embed_tokens, flatten(token_id), 0);
        x = reshape(x, {B, S, hidden_size});
        bool keep_intermediates = ctx.keep_intermediates && artifacts;
        bool prefer_verify_m16 = should_prefer_verify_m16(ctx);
        auto gdr_t_arr = array(S);

        std::vector<array> new_kv_caches(2 * F, array(0));
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
            array attn_out(0);
            if (layer.is_gdr) {
                int si = 1 + 2*F + 2*gdr_idx;
                attn_out = gdr_step(xn, layer.gdr,
                    inputs[si], inputs[si+1],
                    ctx,
                    artifacts,
                    new_gdr_states[gdr_idx], new_conv_states[gdr_idx], gdr_t_arr);
                gdr_idx++;
            } else {
                int si = 1 + 2*full_idx;
                attn_out = full_attn_step(xn, layer.full,
                    inputs[si], inputs[si+1],
                    cache_pos,
                    ctx,
                    artifacts,
                    new_kv_caches[2*full_idx], new_kv_caches[2*full_idx+1]);
                full_idx++;
            }

            x = residual + attn_out;

            // MLP
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
            } else {
                auto& gu = layer.is_gdr ? layer.gdr.gate_up : layer.full.gate_up;
                auto& dw = layer.is_gdr ? layer.gdr.down : layer.full.down;
                int gd = layer.is_gdr ? layer.gdr.gate_dim : layer.full.gate_dim;
                x = residual2 + mlp(xn2, gu, dw, gd, prefer_verify_m16);
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

        // Build output: [logits, kv_caches..., gdr_states..., captured_hidden...]
        std::vector<array> outputs;
        outputs.reserve(1 + 2*F + 2*G + captured_hidden.size());
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
        ctx.has_attn_mask = current_has_attn_mask;
        ctx.attn_mask = current_attn_mask;
        ctx.has_cache_pos_arr = current_has_cache_pos_arr;
        ctx.cache_pos_arr = current_cache_pos_arr;
        ctx.has_rope_offsets = current_has_rope_offsets;
        ctx.rope_offsets = current_rope_offsets;
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

// ── FFI ────────────────────────────────────────────────────────────────────

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

void qwen35_compiled_set_gdr_metal_kernel_enabled(void* model, int32_t enabled) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->gdr_metal_kernel_enabled = enabled != 0;
    });
}

int32_t qwen35_compiled_add_dense_weight(void* model, mlx_array* w) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->weight_pool.push_back({*to_arr(w), array(0), array(0), 0, 0, true});
        return static_cast<int32_t>(m->weight_pool.size() - 1);
    }());
}

int32_t qwen35_compiled_add_affine_weight(
    void* model,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,
    int32_t group_size,
    int32_t bits) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->weight_pool.push_back({
            *to_arr(w), *to_arr(scales), *to_arr(biases), group_size, bits, false});
        return static_cast<int32_t>(m->weight_pool.size() - 1);
    }());
}

int32_t qwen35_compiled_add_gguf_weight(
    void* model,
    mlx_array* w,
    int32_t format,
    int32_t rows,
    int32_t cols) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        QWeight weight;
        weight.w = *to_arr(w);
        weight.gguf_format = format;
        weight.rows = rows;
        weight.cols = cols;
        m->weight_pool.push_back(std::move(weight));
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
        m->use_embed_as_linear = false;
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
        m->use_packed_embed = false;
        m->final_norm_w = *to_arr(final_norm_w);
        m->lm_head = qwen35_weight_by_id(m, lm_head_id);
        m->use_embed_as_linear = false;
    });
}

void qwen35_compiled_set_packed_embed_v2(
    void* model,
    int32_t embed_id,
    mlx_array* final_norm_w,
    int32_t lm_head_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->embed_packed = qwen35_weight_by_id(m, embed_id);
        m->use_packed_embed = true;
        m->final_norm_w = *to_arr(final_norm_w);
        m->lm_head = qwen35_weight_by_id(m, lm_head_id);
        m->use_embed_as_linear = false;
    });
}

// Set quantized embed weights for as_linear lm_head (when tie_word_embeddings=true).
// This uses quantized_matmul instead of dense matmul, reading 0.3GB vs 1.2GB per step.
void qwen35_compiled_set_embed_as_linear(
    void* model,
    mlx_array* w, mlx_array* s, mlx_array* b,
    int32_t gs, int32_t bits
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    m->embed_as_linear = {*to_arr(w), *to_arr(s), *to_arr(b), gs, bits, false};
    m->use_embed_as_linear = true;
}

void qwen35_compiled_set_embed_as_linear_v2(void* model, int32_t embed_id) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        m->embed_as_linear = qwen35_weight_by_id(m, embed_id);
        m->use_embed_as_linear = true;
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
        if (gu_w != nullptr && gu_s != nullptr && gu_b != nullptr &&
            dw_w != nullptr && dw_s != nullptr && dw_b != nullptr) {
            lw.full.gate_up = {*to_arr(gu_w), *to_arr(gu_s), *to_arr(gu_b), gu_gs, gu_bits};
            lw.full.down = {*to_arr(dw_w), *to_arr(dw_s), *to_arr(dw_b), gu_gs, gu_bits};
            lw.full.gate_dim = gate_dim;
        }
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
        lw.gdr.a_log = negative(exp(astype(*to_arr(a_log), float32)));
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
        if (gu_w != nullptr && gu_s != nullptr && gu_b != nullptr &&
            dw_w != nullptr && dw_s != nullptr && dw_b != nullptr) {
            lw.gdr.gate_up = {*to_arr(gu_w), *to_arr(gu_s), *to_arr(gu_b), gu_gs, gu_bits};
            lw.gdr.down = {*to_arr(dw_w), *to_arr(dw_s), *to_arr(dw_b), gu_gs, gu_bits};
            lw.gdr.gate_dim = gate_dim;
        }
        m->layers.push_back(std::move(lw));
        m->n_gdr++;
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
        lw.full.gate_up = qwen35_weight_by_id(m, gate_up_id);
        lw.full.down = qwen35_weight_by_id(m, down_id);
        lw.full.gate_dim = gate_dim;
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
        lw.gdr.gate_up = qwen35_weight_by_id(m, gate_up_id);
        lw.gdr.down = qwen35_weight_by_id(m, down_id);
        lw.gdr.gate_dim = gate_dim;
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

// Set individual (unfused) projections for the last-pushed GDR layer.
// Call AFTER qwen35_compiled_push_gdr. Enables the unfused matmul path
// for the attention-side projections.
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
        lw.gate_proj = {*to_arr(gate_w), *to_arr(gate_s), *to_arr(gate_b), mlp_gs, mlp_bits};
        lw.up_proj = {*to_arr(up_w), *to_arr(up_s), *to_arr(up_b), mlp_gs, mlp_bits};
        lw.has_separate_mlp = true;
        lw.use_separate_proj = true;
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
        lw.gate_proj = qwen35_weight_by_id(m, gate_id);
        lw.up_proj = qwen35_weight_by_id(m, up_id);
        lw.has_separate_mlp = true;
        lw.use_separate_proj = true;
    });
}

void qwen35_compiled_set_separate_mlp(
    void* model,
    mlx_array* gate_w, mlx_array* gate_s, mlx_array* gate_b, int32_t mlp_gs, int32_t mlp_bits,
    mlx_array* up_w, mlx_array* up_s, mlx_array* up_b
) {
    MLX_TRY_VOID({
        auto* m = static_cast<Qwen35CompiledModel*>(model);
        auto& lw = m->layers.back().gdr;
        lw.gate_proj = {*to_arr(gate_w), *to_arr(gate_s), *to_arr(gate_b), mlp_gs, mlp_bits};
        lw.up_proj = {*to_arr(up_w), *to_arr(up_s), *to_arr(up_b), mlp_gs, mlp_bits};
        lw.has_separate_mlp = true;
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
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->clear_optional_batch_inputs();

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

int32_t qwen35_compiled_step_batch(
    void* model,
    mlx_array* token_ids,    // int32 vector [batch]
    int32_t batch_size,
    int32_t cache_pos,
    mlx_array** kv_caches,
    int32_t n_kv_per_request,
    mlx_array** gdr_states,
    int32_t n_gdr_per_request,
    mlx_array* attn_mask,
    mlx_array* rope_offsets,  // nullable int32[batch_size] per-row RoPE offsets
    mlx_array** out_logits,
    mlx_array** out_kv_caches,
    mlx_array** out_gdr_states
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    try {
        mlx_clear_error();

        if (batch_size <= 0) {
            throw std::runtime_error("qwen35_compiled_step_batch requires batch_size > 0");
        }

        m->current_cache_pos = cache_pos;
        m->current_batch_size = batch_size;
        m->current_seq_len = 1;
        m->current_has_attn_mask = attn_mask != nullptr;
        if (m->current_has_attn_mask) {
            m->current_attn_mask = *to_arr(attn_mask);
        } else {
            m->current_attn_mask = array(0);
        }
        m->current_has_cache_pos_arr = false;
        m->current_cache_pos_arr = nullptr;
        m->current_has_rope_offsets = rope_offsets != nullptr;
        if (m->current_has_rope_offsets) {
            m->current_rope_offsets = *to_arr(rope_offsets);
        } else {
            m->current_rope_offsets = array(0);
        }

        std::vector<array> inputs;
        inputs.reserve(1 + n_kv_per_request + n_gdr_per_request);
        inputs.push_back(*to_arr(token_ids));

        for (int kv_idx = 0; kv_idx < n_kv_per_request; ++kv_idx) {
            std::vector<array> per_request;
            per_request.reserve(batch_size);
            for (int b = 0; b < batch_size; ++b) {
                per_request.push_back(*to_arr(kv_caches[b * n_kv_per_request + kv_idx]));
            }
            inputs.push_back(concatenate(per_request, 0));
        }

        for (int gdr_idx = 0; gdr_idx < n_gdr_per_request; ++gdr_idx) {
            std::vector<array> per_request;
            per_request.reserve(batch_size);
            for (int b = 0; b < batch_size; ++b) {
                per_request.push_back(*to_arr(gdr_states[b * n_gdr_per_request + gdr_idx]));
            }
            inputs.push_back(concatenate(per_request, 0));
        }

        m->prev_outputs = m->forward(inputs);
        auto& outputs = m->prev_outputs;

        *out_logits = from_arr(std::move(outputs[0]));

        auto slice_batch_row = [](const array& arr, int row) {
            Shape start = arr.shape();
            Shape end = arr.shape();
            std::fill(start.begin(), start.end(), 0);
            start[0] = row;
            end[0] = row + 1;
            return slice(arr, start, end);
        };

        for (int kv_idx = 0; kv_idx < n_kv_per_request; ++kv_idx) {
            const auto& batched = outputs[1 + kv_idx];
            for (int b = 0; b < batch_size; ++b) {
                out_kv_caches[b * n_kv_per_request + kv_idx] =
                    from_arr(slice_batch_row(batched, b));
            }
        }

        for (int gdr_idx = 0; gdr_idx < n_gdr_per_request; ++gdr_idx) {
            const auto& batched = outputs[1 + n_kv_per_request + gdr_idx];
            for (int b = 0; b < batch_size; ++b) {
                out_gdr_states[b * n_gdr_per_request + gdr_idx] =
                    from_arr(slice_batch_row(batched, b));
            }
        }

        m->clear_optional_batch_inputs();
        m->current_batch_size = 1;
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        m->clear_optional_batch_inputs();
        m->current_batch_size = 1;
        return -1;
    }
}

int32_t qwen35_compiled_step_batch_packed(
    void* model,
    mlx_array* token_ids,    // int32 vector [batch]
    int32_t batch_size,
    int32_t cache_pos,
    mlx_array** packed_kv_caches,
    int32_t n_kv,
    mlx_array** packed_gdr_states,
    int32_t n_gdr,
    mlx_array* attn_mask,
    mlx_array* rope_offsets,  // nullable int32[batch_size] per-row RoPE offsets
    mlx_array** out_logits,
    mlx_array** out_packed_kv_caches,
    mlx_array** out_packed_gdr_states
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    try {
        mlx_clear_error();

        if (batch_size <= 0) {
            throw std::runtime_error("qwen35_compiled_step_batch_packed requires batch_size > 0");
        }

        m->current_cache_pos = cache_pos;
        m->current_batch_size = batch_size;
        m->current_seq_len = 1;
        m->current_has_attn_mask = attn_mask != nullptr;
        if (m->current_has_attn_mask) {
            m->current_attn_mask = *to_arr(attn_mask);
        } else {
            m->current_attn_mask = array(0);
        }
        m->current_has_cache_pos_arr = false;
        m->current_cache_pos_arr = nullptr;
        m->current_has_rope_offsets = rope_offsets != nullptr;
        if (m->current_has_rope_offsets) {
            m->current_rope_offsets = *to_arr(rope_offsets);
        } else {
            m->current_rope_offsets = array(0);
        }

        std::vector<array> inputs;
        inputs.reserve(1 + n_kv + n_gdr);
        inputs.push_back(*to_arr(token_ids));

        for (int kv_idx = 0; kv_idx < n_kv; ++kv_idx) {
            inputs.push_back(*to_arr(packed_kv_caches[kv_idx]));
        }

        for (int gdr_idx = 0; gdr_idx < n_gdr; ++gdr_idx) {
            inputs.push_back(*to_arr(packed_gdr_states[gdr_idx]));
        }

        m->prev_outputs = m->forward(inputs);
        auto& outputs = m->prev_outputs;

        *out_logits = from_arr(std::move(outputs[0]));

        for (int kv_idx = 0; kv_idx < n_kv; ++kv_idx) {
            out_packed_kv_caches[kv_idx] = from_arr(std::move(outputs[1 + kv_idx]));
        }

        for (int gdr_idx = 0; gdr_idx < n_gdr; ++gdr_idx) {
            out_packed_gdr_states[gdr_idx] = from_arr(std::move(outputs[1 + n_kv + gdr_idx]));
        }

        m->clear_optional_batch_inputs();
        m->current_batch_size = 1;
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        m->clear_optional_batch_inputs();
        m->current_batch_size = 1;
        return -1;
    }
}

int32_t qwen35_compiled_prefill(
    void* model,
    mlx_array* token_ids,    // int32 vector [prompt_len]
    int32_t prompt_len,
    int32_t cache_pos,
    mlx_array** kv_caches, int32_t n_kv,
    mlx_array** gdr_states, int32_t n_gdr,
    mlx_array** out_logits,
    mlx_array** out_kv_caches,
    mlx_array** out_gdr_states
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    try {
        mlx_clear_error();

        m->current_cache_pos = cache_pos;
        m->current_batch_size = 1;
        m->current_seq_len = prompt_len;
        m->current_last_logits_only = use_qwen35_cpp_prefill_last_logits_only();
        m->clear_optional_batch_inputs();

        std::vector<array> inputs;
        inputs.reserve(1 + n_kv + n_gdr);
        inputs.push_back(*to_arr(token_ids));
        for (int i = 0; i < n_kv; ++i) inputs.push_back(*to_arr(kv_caches[i]));
        for (int i = 0; i < n_gdr; ++i) inputs.push_back(*to_arr(gdr_states[i]));

        m->prev_outputs = m->forward(inputs);
        auto& outputs = m->prev_outputs;

        *out_logits = from_arr(std::move(outputs[0]));
        for (int i = 0; i < n_kv; ++i)
            out_kv_caches[i] = from_arr(std::move(outputs[1 + i]));
        for (int i = 0; i < n_gdr; ++i)
            out_gdr_states[i] = from_arr(std::move(outputs[1 + n_kv + i]));

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

// ── DFlash speculative verify — parallel forward over a draft block ───────
// Same forward path as prefill (current_seq_len = block_size) but always
// returns all-position logits (current_last_logits_only = false). DFlash
// needs logits for every drafted token to compute greedy acceptance.
// Respects model-level tape_mode / capture_layer_ids so one call emits
// per-step GDR tapes [1, block_size, hv, dv] and captured hidden
// [1, block_size, hidden] for the whole block.

int32_t qwen35_compiled_verify_block(
    void* model,
    mlx_array* token_ids,    // int32 [block_size]
    int32_t block_size,
    int32_t cache_pos,
    mlx_array** kv_caches, int32_t n_kv,
    mlx_array** gdr_states, int32_t n_gdr,
    mlx_array** out_logits,  // [1, block_size, vocab]
    mlx_array** out_kv_caches,
    mlx_array** out_gdr_states
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    try {
        mlx_clear_error();

        m->current_cache_pos = cache_pos;
        m->current_batch_size = 1;
        m->current_seq_len = block_size;
        m->current_last_logits_only = false;
        m->clear_optional_batch_inputs();
        m->current_is_verify = true;

        std::vector<array> inputs;
        inputs.reserve(1 + n_kv + n_gdr);
        inputs.push_back(*to_arr(token_ids));
        for (int i = 0; i < n_kv; ++i) inputs.push_back(*to_arr(kv_caches[i]));
        for (int i = 0; i < n_gdr; ++i) inputs.push_back(*to_arr(gdr_states[i]));

        m->prev_outputs = m->forward(inputs);
        auto& outputs = m->prev_outputs;

        *out_logits = from_arr(std::move(outputs[0]));
        for (int i = 0; i < n_kv; ++i)
            out_kv_caches[i] = from_arr(std::move(outputs[1 + i]));
        for (int i = 0; i < n_gdr; ++i)
            out_gdr_states[i] = from_arr(std::move(outputs[1 + n_kv + i]));

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
        int32_t drafted_len = block_size - 1;
        while (matched_prefix_len < drafted_len &&
               sampled_data[matched_prefix_len] == token_data[matched_prefix_len + 1]) {
            matched_prefix_len += 1;
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

int32_t qwen35_compiled_verify_block_batched(
    void* model,
    mlx_array* token_ids,           // int32 [B, block_size]
    int32_t batch_size,
    int32_t block_size,
    const int32_t* cache_pos_arr,   // host int32 [B] per-row cache_pos
    mlx_array** packed_kv_caches, int32_t n_kv,
    mlx_array** packed_gdr_states, int32_t n_gdr,
    mlx_array* attn_mask,           // additive [B, 1, block_size, key_len], nullable
    mlx_array* rope_offsets,        // int32 [B] per-row RoPE base offset
    mlx_array** out_logits,         // [B, block_size, vocab]
    mlx_array** out_packed_kv_caches,
    mlx_array** out_packed_gdr_states
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    try {
        mlx_clear_error();

        if (batch_size <= 0) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_batched requires batch_size > 0");
        }
        if (block_size <= 0) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_batched requires block_size > 0");
        }
        if (cache_pos_arr == nullptr) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_batched requires cache_pos_arr");
        }
        if (rope_offsets == nullptr) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_batched requires rope_offsets");
        }

        // Route 2: physical KV slot indexing is per row for batched verify, so
        // keep the scalar cache_pos at 0 and read the actual write windows from
        // current_cache_pos_arr inside full_attn_step.
        m->current_cache_pos = 0;
        m->current_batch_size = batch_size;
        m->current_seq_len = block_size;
        m->current_last_logits_only = false;
        m->current_has_attn_mask = attn_mask != nullptr;
        if (m->current_has_attn_mask) {
            m->current_attn_mask = *to_arr(attn_mask);
        } else {
            m->current_attn_mask = array(0);
        }
        m->current_has_cache_pos_arr = true;
        m->current_cache_pos_arr = cache_pos_arr;
        m->current_has_rope_offsets = true;
        m->current_rope_offsets = *to_arr(rope_offsets);
        m->current_is_verify = true;

        std::vector<array> inputs;
        inputs.reserve(1 + n_kv + n_gdr);
        inputs.push_back(*to_arr(token_ids));
        for (int i = 0; i < n_kv; ++i) inputs.push_back(*to_arr(packed_kv_caches[i]));
        for (int i = 0; i < n_gdr; ++i) inputs.push_back(*to_arr(packed_gdr_states[i]));

        m->prev_outputs = m->forward(inputs);
        auto& outputs = m->prev_outputs;

        *out_logits = from_arr(std::move(outputs[0]));
        for (int i = 0; i < n_kv; ++i) {
            out_packed_kv_caches[i] = from_arr(std::move(outputs[1 + i]));
        }
        for (int i = 0; i < n_gdr; ++i) {
            out_packed_gdr_states[i] = from_arr(std::move(outputs[1 + n_kv + i]));
        }

        m->current_cache_pos = 0;
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->clear_optional_batch_inputs();
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        m->current_cache_pos = 0;
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->clear_optional_batch_inputs();
        return -1;
    }
}

int32_t qwen35_compiled_verify_block_batched_sampled(
    void* model,
    mlx_array* token_ids,           // int32 [B, block_size]
    int32_t batch_size,
    int32_t block_size,
    const int32_t* cache_pos_arr,   // host int32 [B] per-row cache_pos
    mlx_array** packed_kv_caches, int32_t n_kv,
    mlx_array** packed_gdr_states, int32_t n_gdr,
    mlx_array* attn_mask,           // additive [B, 1, block_size, key_len], nullable
    mlx_array* rope_offsets,        // int32 [B] per-row RoPE base offset
    float temperature,
    bool greedy,
    int32_t suppress_token_id,
    mlx_array** out_sampled,        // [B, block_size]
    mlx_array** out_packed_kv_caches,
    mlx_array** out_packed_gdr_states
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    try {
        mlx_clear_error();

        if (batch_size <= 0) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_batched_sampled requires batch_size > 0");
        }
        if (block_size <= 0) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_batched_sampled requires block_size > 0");
        }
        if (cache_pos_arr == nullptr) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_batched_sampled requires cache_pos_arr");
        }
        if (rope_offsets == nullptr) {
            throw std::runtime_error(
                "qwen35_compiled_verify_block_batched_sampled requires rope_offsets");
        }

        m->current_cache_pos = 0;
        m->current_batch_size = batch_size;
        m->current_seq_len = block_size;
        m->current_last_logits_only = false;
        m->current_has_attn_mask = attn_mask != nullptr;
        if (m->current_has_attn_mask) {
            m->current_attn_mask = *to_arr(attn_mask);
        } else {
            m->current_attn_mask = array(0);
        }
        m->current_has_cache_pos_arr = true;
        m->current_cache_pos_arr = cache_pos_arr;
        m->current_has_rope_offsets = true;
        m->current_rope_offsets = *to_arr(rope_offsets);
        m->current_is_verify = true;

        std::vector<array> inputs;
        inputs.reserve(1 + n_kv + n_gdr);
        inputs.push_back(*to_arr(token_ids));
        for (int i = 0; i < n_kv; ++i) inputs.push_back(*to_arr(packed_kv_caches[i]));
        for (int i = 0; i < n_gdr; ++i) inputs.push_back(*to_arr(packed_gdr_states[i]));

        m->prev_outputs = m->forward(inputs);
        auto& outputs = m->prev_outputs;

        auto logits = outputs[0];
        logits = suppress_last_axis_token(logits, suppress_token_id);
        auto sampled = greedy
            ? argmax(logits, -1, false)
            : random::categorical(logits * array(1.0f / temperature), -1);

        *out_sampled = from_arr(std::move(sampled));
        for (int i = 0; i < n_kv; ++i) {
            out_packed_kv_caches[i] = from_arr(std::move(outputs[1 + i]));
        }
        for (int i = 0; i < n_gdr; ++i) {
            out_packed_gdr_states[i] = from_arr(std::move(outputs[1 + n_kv + i]));
        }

        m->current_cache_pos = 0;
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->clear_optional_batch_inputs();
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        m->current_cache_pos = 0;
        m->current_batch_size = 1;
        m->current_seq_len = 1;
        m->current_last_logits_only = false;
        m->clear_optional_batch_inputs();
        return -1;
    }
}

// ── Full decode loop in C++ ────────────────────────────────────────────────
// Keeps ALL intermediate arrays alive within the loop body, matching
// Python's behavior where locals survive until the next loop iteration.
// This eliminates the GPU buffer release/realloc overhead that caused
// our per-step time to be 5ms slower than Python.

int32_t qwen35_compiled_generate(
    void* model,
    const int32_t* prompt_ids, int32_t prompt_len,
    int32_t max_new_tokens,
    float temperature,
    bool greedy,
    // Output
    int32_t* out_tokens,
    int32_t* out_count,
    double* out_prefill_ms,   // prefill time in ms (nullable)
    double* out_decode_ms,    // decode time in ms (nullable)
    // Callbacks
    int32_t (*on_token)(int32_t token_id, void* ctx),
    void* callback_ctx,
    // Stop tokens
    const int32_t* stop_tokens, int32_t n_stop_tokens
) {
    auto* m = static_cast<Qwen35CompiledModel*>(model);
    try {
        mlx_clear_error();
        auto t_start = std::chrono::high_resolution_clock::now();
        int F = m->n_full_attn;

        // Initialize caches
        constexpr int KV_CACHE_CHUNK = 256;
        const int total_tokens_needed = std::max(1, prompt_len + max_new_tokens);
        const int kv_cap =
            ((total_tokens_needed + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK) * KV_CACHE_CHUNK;
        auto cache_shape = [&](int heads, int dim) {
            return Shape{1, heads, kv_cap, dim};
        };
        std::vector<array> kv_caches;
        for (int i = 0; i < F; ++i) {
            kv_caches.push_back(zeros(cache_shape(m->n_kv_heads, m->head_dim), bfloat16));
            kv_caches.push_back(zeros(cache_shape(m->n_kv_heads, m->head_dim), bfloat16));
        }
        std::vector<array> gdr_states;
        for (int i = 0; i < (int)m->layers.size(); ++i) {
            if (m->layers[i].is_gdr) {
                auto& lw = m->layers[i].gdr;
                int hv = lw.num_value_heads, dv = lw.value_dim, dk = lw.key_dim;
                gdr_states.push_back(zeros({1, hv, dv, dk}, float32));  // gdr state
                gdr_states.push_back(zeros({1, (int)lw.conv_kernel-1, lw.num_key_heads*dk*2 + hv*dv}, bfloat16));  // conv state
            }
        }
        eval(kv_caches); eval(gdr_states);

        int cache_pos = 0;

        // Batch prefill — all prompt tokens in one forward pass.
        {
            m->current_cache_pos = 0;
            m->current_seq_len = prompt_len;
            m->current_last_logits_only = use_qwen35_cpp_prefill_last_logits_only();

            std::vector<int32_t> pv(prompt_ids, prompt_ids + prompt_len);
            auto tokens = array(pv.data(), {prompt_len}, int32);

            std::vector<array> inputs;
            inputs.push_back(tokens);
            for (auto& c : kv_caches) inputs.push_back(c);
            for (auto& s : gdr_states) inputs.push_back(s);

            auto outputs = m->forward(inputs);
            m->current_seq_len = 1;  // back to decode mode
            m->current_last_logits_only = false;

            for (int j = 0; j < (int)kv_caches.size(); ++j)
                kv_caches[j] = outputs[1 + j];
            for (int j = 0; j < (int)gdr_states.size(); ++j)
                gdr_states[j] = outputs[1 + (int)kv_caches.size() + j];

            cache_pos = prompt_len;

            {
                auto all_logits = outputs[0];
                auto logits = (all_logits.shape(1) == 1)
                    ? all_logits
                    : take(all_logits, array(prompt_len - 1), 1);
                auto y = greedy
                    ? argmax(logits, true)
                    : random::categorical(logits * array(1.0f / temperature), -1);
                eval(y);  // single eval for all prefill tokens
                auto t_prefill_end = std::chrono::high_resolution_clock::now();
                if (out_prefill_ms) {
                    *out_prefill_ms = std::chrono::duration<double, std::milli>(
                        t_prefill_end - t_start).count();
                }
                if (use_qwen35_cpp_clear_cache()) {
                    mlx::core::clear_cache();
                }
                async_eval(y);

                // ── Decode loop ──
                int generated = 0;
                // Keep previous step's locals alive in these vectors
                std::vector<array> prev_step_arrays;

                while (generated < max_new_tokens) {
                    // Save current step's key arrays to keep them alive
                    prev_step_arrays.clear();
                    prev_step_arrays.reserve(512);

                    // Build NEXT graph with lazy token y
                    m->current_cache_pos = cache_pos;
                    std::vector<array> step_inputs;
                    step_inputs.push_back(y);
                    for (auto& c : kv_caches) step_inputs.push_back(c);
                    for (auto& s : gdr_states) step_inputs.push_back(s);

                    auto step_outputs = m->forward(step_inputs);

                    // Update caches (lazy)
                    for (int j = 0; j < (int)kv_caches.size(); ++j)
                        kv_caches[j] = step_outputs[1 + j];
                    for (int j = 0; j < (int)gdr_states.size(); ++j)
                        gdr_states[j] = step_outputs[1 + (int)kv_caches.size() + j];

                    auto next_logits = step_outputs[0];
                    auto next_y = greedy
                        ? argmax(next_logits, true)
                        : random::categorical(next_logits * array(1.0f / temperature), -1);
                    async_eval(next_y);

                    // Wait for CURRENT y
                    eval(y);
                    int32_t token_id = y.item<int32_t>();

                    // Check stop
                    bool stop = false;
                    for (int s = 0; s < n_stop_tokens; ++s) {
                        if (token_id == stop_tokens[s]) { stop = true; break; }
                    }
                    out_tokens[generated] = token_id;
                    generated++;
                    cache_pos++;

                    if (on_token && on_token(token_id, callback_ctx) != 0)
                        break;
                    if (stop) break;

                    if (use_qwen35_cpp_clear_cache() && generated % 256 == 0) {
                        mlx::core::clear_cache();
                    }

                    // Keep step_outputs and step_inputs alive until next iteration
                    // (they're already in local scope — will survive until next loop clear)
                    prev_step_arrays = std::move(step_outputs);
                    for (auto& a : step_inputs) prev_step_arrays.push_back(std::move(a));
                    // Also keep intermediates from forward()
                    for (auto& a : m->intermediates) prev_step_arrays.push_back(std::move(a));

                    y = next_y;
                }

                // Handle last pending token
                if (generated < max_new_tokens) {
                    eval(y);
                    // Already handled above
                }

                *out_count = generated;
                if (out_decode_ms) {
                    auto t_decode_end = std::chrono::high_resolution_clock::now();
                    *out_decode_ms = std::chrono::duration<double, std::milli>(
                        t_decode_end - t_prefill_end).count();
                }
            }
        }

        if (prompt_len == 0) {
            *out_count = 0;
        }

        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
}

// ── DFlash tape mode API ─────────────────────────────────────────────

void qwen35_set_tape_mode(void* model, bool enabled) {
    auto* m = reinterpret_cast<Qwen35CompiledModel*>(model);
    m->tape_mode = enabled;
    if (!enabled) m->gdr_tapes.clear();
}

int32_t qwen35_get_tape_count(void* model) {
    auto* m = reinterpret_cast<Qwen35CompiledModel*>(model);
    return static_cast<int32_t>(m->gdr_tapes.size());
}

/// Get tape arrays for a GDR layer. Returns new array handles (caller must free).
int32_t qwen35_get_tape(void* model, int32_t idx,
                        mlx_array** out_tape, mlx_array** out_k,
                        mlx_array** out_g, mlx_array** out_qkv) {
    try {
        auto* m = reinterpret_cast<Qwen35CompiledModel*>(model);
        if (idx < 0 || idx >= static_cast<int32_t>(m->gdr_tapes.size()))
            throw std::out_of_range("tape index out of range");
        auto& t = m->gdr_tapes[idx];
        // Allocate new shared_ptr-wrapped arrays (same pattern as qwen35 step outputs)
        *out_tape = reinterpret_cast<mlx_array*>(new array(t.innovation_tape));
        *out_k = reinterpret_cast<mlx_array*>(new array(t.k));
        *out_g = reinterpret_cast<mlx_array*>(new array(t.g));
        *out_qkv = reinterpret_cast<mlx_array*>(new array(t.qkv));
        return 0;
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        return -1;
    }
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

// ── Hidden state capture API ─────────────────────────────────────────

void qwen35_set_capture_layers(void* model, const int32_t* layer_ids, int32_t count) {
    auto* m = reinterpret_cast<Qwen35CompiledModel*>(model);
    m->capture_layer_ids.clear();
    if (layer_ids && count > 0) {
        m->capture_layer_ids.assign(layer_ids, layer_ids + count);
    }
}

int32_t qwen35_get_captured_hidden_count(void* model) {
    auto* m = reinterpret_cast<Qwen35CompiledModel*>(model);
    int total_out = 1 + 2 * m->n_full_attn + 2 * m->n_gdr;
    auto& outputs = m->prev_outputs;
    if ((int)outputs.size() <= total_out) return 0;
    return static_cast<int32_t>(outputs.size() - total_out);
}

/// Get a captured hidden state by index. Returns new array handle (caller must free).
int32_t qwen35_get_captured_hidden(void* model, int32_t idx, mlx_array** out) {
    try {
        auto* m = reinterpret_cast<Qwen35CompiledModel*>(model);
        int total_out = 1 + 2 * m->n_full_attn + 2 * m->n_gdr;
        auto& outputs = m->prev_outputs;
        int hi = total_out + idx;
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

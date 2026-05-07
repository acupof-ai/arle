//! Qwen3.5/3.6 SparseMoeBlock — C++ forward (Metal).
//!
//! Ports `Qwen3NextSparseMoeBlock.__call__` from the mlx-lm reference
//! (`qwen3_next.py` lines 308–354) into a single C++ helper that composes the
//! MLX C++ API. The helper is intended to be called from the per-layer dispatch
//! inside `mlx_qwen35_model.cpp` (Phase 1C wires it up) and from Rust via the
//! `qwen35_moe_block_forward` FFI.
//!
//! Reference flow (all MLX ops, stays in the graph):
//!
//!   gates   = softmax(quantized_matmul(x, router), axis=-1, precise=true)
//!   inds    = argpartition(gates, kth=E-k, axis=-1)[..., -k:]
//!   scores  = take_along_axis(gates, inds, axis=-1)
//!   if norm_topk_prob: scores = scores / sum(scores, -1, keepdims)
//!   y       = SwitchGLU(x, inds)               // gather_qmm × 3 + SiLU
//!   y       = sum(y * scores[..., None], axis=-2)
//!   shared  = SwiGLU(x, shared_{gate,up,down}) // dense, 4-bit
//!   shared  = sigmoid(router_shared(x)) * shared
//!   return y + shared
//!
//! SwitchGLU (from mlx-lm `switch_layers.py`) packs tokens through a 5-D shape
//! before hitting `gather_qmm`: `x = expand_dims(x, {-2, -3})` makes the
//! token-dim precede the expert-dim so the gather computes one `[k, 1, hidden]`
//! slab per token. We skip the `do_sort` fast-path (an index permutation
//! optimisation that only kicks in for very large batches and requires extra
//! scatter helpers); correctness-wise it is an optimisation, not a semantic
//! change. The un-sorted path is what all seq_len-1 decode steps take in
//! mlx-lm anyway.

#include "mlx_common.h"
#include <stdexcept>

namespace {

using mlx::core::array;

// M_e.4 — Compiled SwiGLU: `silu(gate) * up = (gate * sigmoid(gate)) * up`.
// Replaces the per-call 3-op {sigmoid, multiply, multiply} chain with one
// shapeless-compiled kernel, dropping the encoder primitive count by 2 per
// SwiGLU invocation. With 40 layers × 2 SwiGLU sites/layer (switch experts +
// shared expert) that's ~160 fewer primitives encoded per Qwen3.6 step.
// Mirrors the analogous helper in `mlx_qwen35_model.cpp:compiled_swiglu` —
// kept file-local here so this TU doesn't depend on Qwen3.5 internals.
std::vector<array> swiglu_impl(const std::vector<array>& inputs) {
    auto gate = inputs[0];
    auto up = inputs[1];
    return {(gate * mlx::core::sigmoid(gate)) * up};
}

auto& compiled_swiglu() {
    static auto fn = mlx::core::compile(swiglu_impl, /*shapeless=*/true);
    return fn;
}

inline array swiglu(const array& gate, const array& up) {
    return compiled_swiglu()({gate, up})[0];
}

struct SortedSwitchInputs {
    array x;
    array indices;
    array inv_order;
};

// Quantized linear: y = x @ dequantize(w).T — matches QWeight::apply() in
// mlx_qwen35_model.cpp. Kept local so this TU doesn't depend on Qwen3.5 model
// internals.
array qmm(const array& x, const array& w, const array& scales,
          const array& biases, int group_size, int bits) {
    return verify_quantized_matmul_cpp(
        x,
        w,
        scales,
        biases,
        group_size,
        bits,
        /*transpose=*/true);
}

// Fused quantized-SwiGLU: down(silu(gate(x)) * up(x)) with three 4-bit proj.
// Mirrors `mlx_fused_quantized_gated_mlp` in mlx_bridge.cpp (kept local so
// this TU doesn't link against that symbol ordering).
array quantized_swiglu(const array& x,
                       const array& gate_w, const array& gate_s, const array& gate_b,
                       const array& up_w,   const array& up_s,   const array& up_b,
                       const array& down_w, const array& down_s, const array& down_b,
                       int group_size, int bits) {
    auto gate = qmm(x, gate_w, gate_s, gate_b, group_size, bits);
    auto up   = qmm(x, up_w,   up_s,   up_b,   group_size, bits);
    // M_e.4: silu(gate) * up via compiled-shapeless kernel — saves 2
    // encoded primitives vs the manual {sigmoid, multiply, multiply} chain.
    auto h = swiglu(gate, up);
    return qmm(h, down_w, down_s, down_b, group_size, bits);
}

// Match mlx-lm's `_gather_sort`/`_scatter_unsort` optimization in
// `switch_layers.py`: when many token→expert routes are active, sort routes by
// expert id so `gather_qmm` sees expert-locality and can use its
// `sorted_indices=true` fast path.
SortedSwitchInputs gather_sort_switch_inputs(const array& x, const array& indices) {
    const auto& shape = indices.shape();
    const int last_dim = shape.back();
    auto flat_indices = mlx::core::flatten(indices);
    auto order = mlx::core::astype(mlx::core::argsort(flat_indices), mlx::core::int32);
    auto inv_order = mlx::core::astype(mlx::core::argsort(order), mlx::core::int32);
    auto rows = mlx::core::floor_divide(order, array(last_dim, mlx::core::int32));
    auto flat_x = mlx::core::flatten(x, 0, -3);
    return {
        mlx::core::take(flat_x, rows, 0),
        mlx::core::take(flat_indices, order, 0),
        inv_order,
    };
}

array scatter_unsort_switch_outputs(
    const array& x,
    const array& inv_order,
    const mlx::core::Shape& indices_shape) {
    auto unsorted = mlx::core::take(x, inv_order, 0);
    return mlx::core::unflatten(unsorted, 0, indices_shape);
}

// SwitchGLU forward — batched quantized gather_qmm × 3 with SiLU gate.
// Shapes:
//   x:    [..., H]            (typically [B, S, H] or [N, H])
//   inds: [..., top_k]        (int32, ndim matches x.ndim up to axis -1)
// Output: [..., top_k, H]
//
// Internal layout (matches mlx-lm `SwitchGLU.__call__`):
//   x5 = expand_dims(x, {-2, -3})                 -> [..., 1, 1, H]
//   x_gate = gather_qmm(x5, gate_w, ..., rhs_indices=inds, transpose=true)
//          -> [..., top_k, 1, hidden]
//   same for x_up
//   h = silu(x_gate) * x_up                        -> [..., top_k, 1, hidden]
//   y = gather_qmm(h, down_w, ..., rhs_indices=inds, transpose=true)
//          -> [..., top_k, 1, H]
//   y = squeeze(y, -2)                             -> [..., top_k, H]
array switch_glu_forward(
    const array& x, const array& inds,
    const array& gate_w, const array& gate_s, const array& gate_b,
    const array& up_w,   const array& up_s,   const array& up_b,
    const array& down_w, const array& down_s, const array& down_b,
    int group_size, int bits) {
    // expand_dims accepts a vector of axes; add 1 at both -2 and -3.
    auto x5 = mlx::core::expand_dims(x, std::vector<int>{-2, -3});
    // gather_qmm sorted-indices fast path threshold. mlx-lm uses 64 (no comment
    // explaining the choice — see mlx_lm/models/switch_layers.py:178). At 32
    // we cover c=4 decode on Qwen3.6 35B-A3B-4bit (c=4 × top_k=8 = 32 indices),
    // routing the c=4 decode hot path through the coalesced expert-row reads.
    // Apple Silicon's narrower memory bandwidth makes the sort overhead worth
    // it for fewer indices than NVIDIA. See
    // docs/experience/wins/2026-05-07-bench-qwen36-baseline.md and the MoE
    // research subagent report at the same date — technique #2.
    const bool do_sort = inds.size() >= 32;
    auto idx = inds;
    array inv_order(0);
    if (do_sort) {
        auto sorted = gather_sort_switch_inputs(x5, inds);
        x5 = sorted.x;
        idx = sorted.indices;
        inv_order = sorted.inv_order;
    }

    auto x_gate = mlx::core::gather_qmm(
        x5, gate_w, gate_s, /*biases=*/gate_b,
        /*lhs_indices=*/std::nullopt,
        /*rhs_indices=*/idx,
        /*transpose=*/true,
        /*group_size=*/group_size,
        /*bits=*/bits,
        /*mode=*/"affine",
        /*sorted_indices=*/do_sort);
    auto x_up = mlx::core::gather_qmm(
        x5, up_w, up_s, /*biases=*/up_b,
        std::nullopt, idx, true, group_size, bits, "affine", do_sort);

    // M_e.4: silu(gate) * up via compiled-shapeless kernel — saves 2
    // encoded primitives per call (this is the per-layer SwitchGLU site,
    // hit 40 times per Qwen3.6 step).
    auto h = swiglu(x_gate, x_up);

    auto y = mlx::core::gather_qmm(
        h, down_w, down_s, /*biases=*/down_b,
        std::nullopt, idx, true, group_size, bits, "affine", do_sort);

    if (do_sort) {
        y = scatter_unsort_switch_outputs(y, inv_order, inds.shape());
    }

    // Drop the trailing unit dim: [..., top_k, 1, H] -> [..., top_k, H].
    return mlx::core::squeeze(y, -2);
}

} // namespace

// Qwen3.5/3.6 sparse-MoE forward. Returns the MLX array pointer on success
// (caller owns → must `mlx_array_free`), nullptr on failure (check
// `mlx_last_error()`).
//
// Array ownership: all inputs are borrowed; the helper never frees or clones
// them. On success it allocates exactly one new `mlx_array`.
//
// Dense-SwiGLU router vs switch-mlp: the router gate (`router_*`) and the
// shared-expert gate (`shared_gate_router_*`) are 8-bit quantized linears —
// their `group_size`/`bits` come in via `router_bits`/`router_group_size`.
// The switch-mlp experts and the dense shared expert are 4-bit and share
// `expert_bits`/`expert_group_size`.
//
// Layout expectations (following mlx-lm `qwen3_5_moe.py` sanitize output):
//   hidden                : [..., H]             any rank >= 2
//   router_w              : [E, H / pack]        packed quantized
//   router_scales/biases  : [E, H / group_size]
//   expert_{gate,up}_w    : [E, Hmoe, H / pack]
//   expert_{gate,up}_s/b  : [E, Hmoe, H / group_size]
//   expert_down_w         : [E, H, Hmoe / pack]
//   expert_down_s/b       : [E, H, Hmoe / group_size]
//   shared_{gate,up}_w    : [Hshared, H / pack]
//   shared_down_w         : [H, Hshared / pack]
//   shared_gate_router_w  : [1, H / pack]
//
// The helper does not validate shapes beyond what MLX itself will catch
// during op evaluation; pre-checks would duplicate mlx's shape checks and
// cost an extra allocation. Errors surface through `mlx_last_error()`.
array qwen35_moe_block_forward_cpp(
    const array& x,
    // Router (8-bit quantized linear: H -> E, no bias at runtime)
    const array& router_w,
    const array& router_scales,
    const array& router_biases,
    int32_t router_bits,
    int32_t router_group_size,
    // Switch MLP experts (4-bit quantized, stacked on expert axis)
    const array& expert_gate_w,
    const array& expert_gate_scales,
    const array& expert_gate_biases,
    const array& expert_up_w,
    const array& expert_up_scales,
    const array& expert_up_biases,
    const array& expert_down_w,
    const array& expert_down_scales,
    const array& expert_down_biases,
    int32_t expert_bits,
    int32_t expert_group_size,
    // Dense shared expert (4-bit quantized SwiGLU)
    const array& shared_gate_w,
    const array& shared_gate_scales,
    const array& shared_gate_biases,
    const array& shared_up_w,
    const array& shared_up_scales,
    const array& shared_up_biases,
    const array& shared_down_w,
    const array& shared_down_scales,
    const array& shared_down_biases,
    // Shared-expert scalar router (8-bit quantized linear: H -> 1)
    const array& shared_gate_router_w,
    const array& shared_gate_router_scales,
    const array& shared_gate_router_biases,
    int32_t num_experts,
    int32_t top_k,
    bool norm_topk_prob) {
    if (num_experts <= 0 || top_k <= 0 || top_k > num_experts) {
        throw std::invalid_argument(
            "qwen35_moe_block_forward: invalid num_experts/top_k");
    }

    // INFER_MOE_TOP_K=N (1..top_k) — runtime knob to reduce active-expert
    // count below the model's configured top_k. Per the parallel research
    // subagent (2026-05-07): vllm-mlx ships this as `--moe-top-k` and reports
    // +7-16% throughput on Qwen3-30B-A3B with ~3% MMLU drop at top_k=6 vs 8.
    // Cached env probe; clamps to valid range. Env unset = passthrough.
    {
        static int env_top_k = -2;
        if (env_top_k == -2) {
            const char* v = std::getenv("INFER_MOE_TOP_K");
            if (v && *v) {
                int parsed = 0;
                for (const char* p = v; *p; ++p) {
                    if (*p < '0' || *p > '9') { parsed = 0; break; }
                    parsed = parsed * 10 + (*p - '0');
                }
                env_top_k = (parsed > 0 && parsed <= top_k) ? parsed : -1;
                if (env_top_k > 0) {
                    std::fprintf(stderr,
                        "INFO MoE top_k overridden to %d via INFER_MOE_TOP_K (model default=%d)\n",
                        env_top_k, top_k);
                }
            } else {
                env_top_k = -1;
            }
        }
        if (env_top_k > 0) {
            top_k = env_top_k;
        }
    }

    const int rank = static_cast<int>(x.ndim());
    if (rank < 2) {
        throw std::invalid_argument(
            "qwen35_moe_block_forward: hidden must have rank >= 2 (got rank < 2)");
    }

    // ── Router ────────────────────────────────────────────────────────
    // gates: [..., E]
    auto gates = qmm(x, router_w, router_scales, router_biases, router_group_size, router_bits);
    gates = mlx::core::softmax(gates, /*axis=*/-1, /*precise=*/true);

    // ── Top-k via argpartition + slice the last k of the last axis ────
    // argpartition places the k largest elements at positions [E-k, E).
    const int kth = num_experts - top_k;
    auto part = mlx::core::argpartition(gates, kth, /*axis=*/-1);

    // Slice last axis from E-k to E. Build per-rank start/stop/strides.
    const int gates_rank = static_cast<int>(part.ndim());
    mlx::core::Shape start(gates_rank, 0);
    mlx::core::Shape stop = part.shape();
    mlx::core::Shape strides(gates_rank, 1);
    start[gates_rank - 1] = num_experts - top_k;
    // stop[-1] already num_experts
    auto inds = mlx::core::slice(part, start, stop, strides);

    // ── Gather scores and optional renormalization ────────────────────
    auto scores = mlx::core::take_along_axis(gates, inds, /*axis=*/-1);
    if (norm_topk_prob) {
        auto denom = mlx::core::sum(scores, /*axis=*/-1, /*keepdims=*/true);
        scores = mlx::core::divide(scores, denom);
    }

    // Match scores dtype to hidden for the eventual multiply/sum.
    // (softmax returns the input dtype; router_w may upcast.)
    if (scores.dtype() != x.dtype()) {
        scores = mlx::core::astype(scores, x.dtype());
    }

    // ── Switch-MLP experts ───────────────────────────────────────────
    // y_switch: [..., top_k, H]
    auto y_switch = switch_glu_forward(
        x, inds,
        expert_gate_w, expert_gate_scales, expert_gate_biases,
        expert_up_w, expert_up_scales, expert_up_biases,
        expert_down_w, expert_down_scales, expert_down_biases,
        expert_group_size, expert_bits);

    // Weighted sum over top_k: y = sum(y_switch * scores[..., None], -2)
    auto scores_bcast = mlx::core::expand_dims(scores, -1);
    auto y_weighted = mlx::core::multiply(y_switch, scores_bcast);
    auto y = mlx::core::sum(y_weighted, /*axis=*/-2, /*keepdims=*/false);

    // ── Dense shared expert ──────────────────────────────────────────
    auto shared_y = quantized_swiglu(
        x,
        shared_gate_w, shared_gate_scales, shared_gate_biases,
        shared_up_w, shared_up_scales, shared_up_biases,
        shared_down_w, shared_down_scales, shared_down_biases,
        expert_group_size, expert_bits);

    // Scalar shared-expert gate: sigmoid(linear_8bit(x)) * shared_y
    // linear_8bit output shape: [..., 1]. Broadcast against shared_y [..., H].
    auto gate_logit = qmm(x,
                          shared_gate_router_w,
                          shared_gate_router_scales,
                          shared_gate_router_biases,
                          router_group_size, router_bits);
    auto gate_val = mlx::core::sigmoid(gate_logit);
    if (gate_val.dtype() != shared_y.dtype()) {
        gate_val = mlx::core::astype(gate_val, shared_y.dtype());
    }
    shared_y = mlx::core::multiply(gate_val, shared_y);

    return mlx::core::add(y, shared_y);
}

extern "C" mlx_array* qwen35_moe_block_forward(
    mlx_array* hidden,
    // Router (8-bit quantized linear: H -> E, no bias at runtime)
    mlx_array* router_w,
    mlx_array* router_scales,
    mlx_array* router_biases,
    int32_t router_bits,
    int32_t router_group_size,
    // Switch MLP experts (4-bit quantized, stacked on expert axis)
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
    // Dense shared expert (4-bit quantized SwiGLU)
    mlx_array* shared_gate_w,
    mlx_array* shared_gate_scales,
    mlx_array* shared_gate_biases,
    mlx_array* shared_up_w,
    mlx_array* shared_up_scales,
    mlx_array* shared_up_biases,
    mlx_array* shared_down_w,
    mlx_array* shared_down_scales,
    mlx_array* shared_down_biases,
    // Shared-expert scalar router (8-bit quantized linear: H -> 1)
    mlx_array* shared_gate_router_w,
    mlx_array* shared_gate_router_scales,
    mlx_array* shared_gate_router_biases,
    int32_t num_experts,
    int32_t top_k,
    bool norm_topk_prob) {
    MLX_TRY_RETURN([&]() {
        if (hidden == nullptr) {
            throw std::invalid_argument("qwen35_moe_block_forward: hidden is null");
        }
        return from_arr(qwen35_moe_block_forward_cpp(
            *to_arr(hidden),
            *to_arr(router_w),
            *to_arr(router_scales),
            *to_arr(router_biases),
            router_bits,
            router_group_size,
            *to_arr(expert_gate_w),
            *to_arr(expert_gate_scales),
            *to_arr(expert_gate_biases),
            *to_arr(expert_up_w),
            *to_arr(expert_up_scales),
            *to_arr(expert_up_biases),
            *to_arr(expert_down_w),
            *to_arr(expert_down_scales),
            *to_arr(expert_down_biases),
            expert_bits,
            expert_group_size,
            *to_arr(shared_gate_w),
            *to_arr(shared_gate_scales),
            *to_arr(shared_gate_biases),
            *to_arr(shared_up_w),
            *to_arr(shared_up_scales),
            *to_arr(shared_up_biases),
            *to_arr(shared_down_w),
            *to_arr(shared_down_scales),
            *to_arr(shared_down_biases),
            *to_arr(shared_gate_router_w),
            *to_arr(shared_gate_router_scales),
            *to_arr(shared_gate_router_biases),
            num_experts,
            top_k,
            norm_topk_prob));
    }());
}

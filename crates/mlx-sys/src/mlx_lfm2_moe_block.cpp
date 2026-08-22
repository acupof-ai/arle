//! LFM2.5 SparseMoeBlock — C++ forward (Metal).
//!
//! Ports `Lfm2MoeSparseMoeBlock` from the HuggingFace transformers reference
//! (`modeling_lfm2_moe.py`). Differences from the Qwen3.5 MoE block
//! (`mlx_qwen35_moe_block.cpp`):
//!   - sigmoid routing (not softmax): scores = sigmoid(x @ W_gate.T)
//!   - a persistent per-expert bias (`expert_bias`, F32, nonzero in the
//!     checkpoint — aux-loss-free balancing baked in at train time) is added
//!     to the scores ONLY for top-k selection, never to the routing weights
//!   - no shared expert
//!
//! Reference flow:
//!   s       = sigmoid(matmul(x, router_w))          // [..., E]
//!   idx     = topk(s + expert_bias, k).indices       // [..., top_k]
//!   w       = take_along_axis(s, idx)                // un-biased sigmoid
//!   if norm_topk_prob: w = w / (sum(w, -1) + 1e-6)
//!   y       = SwitchGLU(x, idx)                      // gather_qmm x 3 + SiLU
//!   y       = sum(y * w[..., None], axis=-2)

#include "mlx_common.h"
#include <stdexcept>

namespace {

using mlx::core::array;

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

// Quantized linear: y = x @ dequantize(w).T — matches QWeight::apply() in
// mlx_qwen35_model.cpp.
array qmm(const array& x, const array& w, const array& scales,
          const array& biases, int group_size, int bits) {
    return verify_quantized_matmul_cpp(
        x, w, scales, biases, group_size, bits, /*transpose=*/true);
}

struct SortedSwitchInputs {
    array x;
    array indices;
    array inv_order;
};

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

// SwitchGLU forward — batched quantized gather_qmm x 3 with SiLU gate.
// Mirrors switch_glu_forward in mlx_qwen35_moe_block.cpp; see that file for
// the layout contract ([..., 1, 1, H] slabs, one per routed expert).
array switch_glu_forward(
    const array& x, const array& inds,
    const array& gate_w, const array& gate_s, const array& gate_b,
    const array& up_w,   const array& up_s,   const array& up_b,
    const array& down_w, const array& down_s, const array& down_b,
    int group_size, int bits) {
    auto x5 = mlx::core::expand_dims(x, std::vector<int>{-2, -3});
    // 32 experts x top_k=4 = 16 routes at c=4; sort threshold matches the
    // Qwen3.5 block (coalesced expert-row reads on Apple Silicon).
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
        x5, gate_w, gate_s, gate_b,
        /*lhs_indices=*/std::nullopt, /*rhs_indices=*/idx,
        /*transpose=*/true, group_size, bits, "affine", do_sort);
    auto x_up = mlx::core::gather_qmm(
        x5, up_w, up_s, up_b,
        std::nullopt, idx, true, group_size, bits, "affine", do_sort);

    auto h = swiglu(x_gate, x_up);

    auto y = mlx::core::gather_qmm(
        h, down_w, down_s, down_b,
        std::nullopt, idx, true, group_size, bits, "affine", do_sort);

    if (do_sort) {
        y = scatter_unsort_switch_outputs(y, inv_order, inds.shape());
    }
    return mlx::core::squeeze(y, -2);
}

} // namespace

// LFM2.5 sparse-MoE forward. Throws on invalid arguments.
//
//   x            : [..., H]
//   router_w     : [H, E] dense (pre-transposed at load)
//   expert_bias  : [E] persistent aux-loss-free bias (f32)
//   switch_*     : [E, I, H/pack] stacked 4-bit affine experts
array lfm2_moe_block_forward_cpp(
    const array& x,
    const array& router_w,
    const array& expert_bias,
    const array& expert_gate_w, const array& expert_gate_s, const array& expert_gate_b,
    const array& expert_up_w,   const array& expert_up_s,   const array& expert_up_b,
    const array& expert_down_w, const array& expert_down_s, const array& expert_down_b,
    int32_t expert_group_size, int32_t expert_bits,
    int32_t num_experts, int32_t top_k, bool norm_topk_prob) {
    if (num_experts <= 0 || top_k <= 0 || top_k > num_experts) {
        throw std::invalid_argument("lfm2_moe_block_forward_cpp: invalid num_experts/top_k");
    }
    const int rank = static_cast<int>(x.ndim());
    if (rank < 2) {
        throw std::invalid_argument("lfm2_moe_block_forward_cpp: hidden must have rank >= 2");
    }

    // Sigmoid routing in f32 (matches the transformers reference upcast).
    auto logits = mlx::core::matmul(astype(x, float32), astype(router_w, float32));
    auto s = mlx::core::sigmoid(logits);

    // top-k selection on biased scores; weights come from the un-biased sigmoid.
    // The vendored MLX topk returns values only, so use argpartition (same
    // pattern as the Qwen3.5 MoE block): the last-k slice of the partitioned
    // indices holds the top-k expert ids.
    auto sel = s + astype(expert_bias, float32);
    const int kth = num_experts - top_k;
    auto part = mlx::core::argpartition(sel, kth, /*axis=*/-1);
    const int sel_rank = static_cast<int>(part.ndim());
    mlx::core::Shape start(sel_rank, 0);
    mlx::core::Shape stop = part.shape();
    mlx::core::Shape strides(sel_rank, 1);
    start[sel_rank - 1] = num_experts - top_k;
    auto idx = astype(mlx::core::slice(part, start, stop, strides), mlx::core::int32);
    auto w = mlx::core::take_along_axis(s, idx, /*axis=*/-1);
    if (norm_topk_prob) {
        auto denom = mlx::core::sum(w, /*axis=*/-1, /*keepdims=*/true) + array(1e-6f);
        w = mlx::core::divide(w, denom);
    }

    auto y_switch = switch_glu_forward(
        x, idx,
        expert_gate_w, expert_gate_s, expert_gate_b,
        expert_up_w, expert_up_s, expert_up_b,
        expert_down_w, expert_down_s, expert_down_b,
        expert_group_size, expert_bits);

    auto y = mlx::core::sum(
        y_switch * mlx::core::expand_dims(w, -1), /*axis=*/-2, /*keepdims=*/false);
    return astype(y, x.dtype());
}

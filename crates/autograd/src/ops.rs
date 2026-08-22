#[path = "ops/activation.rs"]
pub mod activation;
#[path = "ops/attention.rs"]
pub mod attention;
#[path = "ops/broadcast.rs"]
pub mod broadcast;
#[path = "ops/checkpoint.rs"]
pub mod checkpoint;
#[path = "ops/chunk_accum.rs"]
pub mod chunk_accum;
#[path = "ops/collective.rs"]
pub mod collective;
#[path = "ops/elementwise.rs"]
pub mod elementwise;
#[path = "ops/embed.rs"]
pub mod embed;
#[path = "ops/fused_linear_distill.rs"]
pub mod fused_linear_distill;
#[path = "ops/gather.rs"]
pub mod gather;
#[path = "ops/layout.rs"]
pub mod layout;
#[path = "ops/linear_attention.rs"]
pub mod linear_attention;
#[path = "ops/matmul.rs"]
pub mod matmul;
#[path = "ops/moe.rs"]
pub mod moe;
#[path = "ops/norm.rs"]
pub mod norm;
#[path = "ops/reduce.rs"]
pub mod reduce;
#[path = "ops/ring_attention.rs"]
pub mod ring_attention;
#[path = "ops/rope.rs"]
pub mod rope;
#[path = "ops/softmax.rs"]
pub mod softmax;

use crate::{
    Result,
    tape::Tape,
    tensor::{TensorId, TensorStore},
};

pub(crate) use activation::{exp_backward, gelu_backward, sigmoid_backward, silu_backward};
pub(crate) use attention::{cat_heads_backward, cat_seq_backward, causal_sdpa_recompute_backward};
pub(crate) use broadcast::add_broadcast_backward;
pub(crate) use collective::{
    all_gather_seq_backward, all_reduce_sum_backward, all_to_all_backward,
    reduce_scatter_sum_backward,
};
pub(crate) use elementwise::{abs_backward, add_backward, mul_backward, mul_scalar_backward};
pub(crate) use embed::embedding_backward;
pub(crate) use fused_linear_distill::{fused_linear_distill_backward, generalized_jsd_backward};
pub(crate) use gather::gather_last_dim_backward;
pub(crate) use layout::{
    broadcast_expand_backward, cat_backward, permute_seq_blocks_backward, reshape_backward,
    slice_backward, slice_backward_into, transpose_backward,
};
pub(crate) use linear_attention::linear_attention_backward;
pub(crate) use matmul::{matmul_backward, matmul_bt_backward};
pub(crate) use moe::{
    moe_gather_rows_backward, moe_grouped_linear_backward, moe_grouped_weighted_scatter_backward,
    moe_topk_softmax_backward, moe_weighted_scatter_backward,
};
pub(crate) use norm::rmsnorm_backward;
pub(crate) use reduce::{mean_backward, sum_backward};
pub(crate) use rope::rope_backward;
pub(crate) use softmax::{log_softmax_backward, softmax_backward};

pub use checkpoint::{checkpoint, checkpoint_seq_chunked, checkpoint_sequential};

pub fn exp(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Inner `activation::exp` dispatches on `dirty`; a Dirty::Device
    // input stays lazy via `backend.exp` (MLX `mlx_exp`), while Dirty::Host
    // / Dirty::Both take the eager host path. No `ensure_host` here — that
    // would force a readback before the inner fn could see the device state.
    activation::exp(x, store, tape)
}

pub fn all_reduce_sum(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    collective::all_reduce_sum(x, store, tape)
}

pub fn all_gather_seq(
    x: TensorId,
    full_shape: Vec<usize>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    collective::all_gather_seq(x, full_shape, store, tape)
}

pub fn reduce_scatter_sum(
    x: TensorId,
    local_shape: Vec<usize>,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    collective::reduce_scatter_sum(x, local_shape, store, tape)
}

pub fn all_to_all(
    x: TensorId,
    scatter_axis: usize,
    gather_axis: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    collective::all_to_all(x, scatter_axis, gather_axis, store, tape)
}

pub fn gelu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Inner dispatcher picks lazy-device vs host-eager; stripping the
    // eager readback here lets the lazy branch keep x Dirty::Device.
    activation::gelu(x, store, tape)
}

pub fn silu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // `silu` is device-resident on Metal for Dirty::Device
    // inputs — `activation::silu` routes to `backend.silu` (composes
    // `mlx_multiply(x, mlx_sigmoid(x))` into the MLX lazy graph, no eval).
    // Dirty::Host / Dirty::Both inputs stay on the host fast path. CPU/CUDA
    // use the default trait fallback (readback → host → upload); lazy
    // semantics are Metal-only.
    activation::silu(x, store, tape)
}

pub fn sigmoid(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Inner `activation::sigmoid` dispatches on `x.dirty`; a
    // Dirty::Device/Both input stays lazy via `backend.sigmoid`
    // (`mlx_sigmoid` single node into the MLX graph), Dirty::Host takes
    // the eager host path. No `ensure_host` here — Qwen3.5 attention's
    // `gate = sigmoid(gate_proj)` × 28 layers would otherwise flush the
    // q_full slice to host before the gate multiply.
    activation::sigmoid(x, store, tape)
}

pub fn repeat_kv(
    x: TensorId,
    n_rep: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    attention::repeat_kv(x, n_rep, store, tape)
}

pub fn causal_sdpa(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Composite op — body is pure dispatch over reshape /
    // transpose / matmul / mul_scalar / add_broadcast / softmax, all of
    // which are lazy on Metal. No `ensure_host` here, so the entire
    // attention chain stays in the MLX graph end-to-end for each layer
    // (Qwen3.5 × 28 layers).
    attention::causal_sdpa(q, k, v, store, tape)
}

pub fn causal_sdpa_recompute(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    attention::causal_sdpa_recompute(q, k, v, store, tape)
}

pub fn causal_sdpa_with_q_start(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_start: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    attention::causal_sdpa_with_q_start(q, k, v, q_start, store, tape)
}

pub fn causal_sdpa_recompute_with_q_start(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_start: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    attention::causal_sdpa_recompute_with_q_start(q, k, v, q_start, store, tape)
}

pub fn cat_seq(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    attention::cat_seq(a, b, store, tape)
}

pub fn cat_heads(
    inputs: &[TensorId],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    attention::cat_heads(inputs, store, tape)
}

pub fn causal_sdpa_decode_gqa(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_start: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    attention::causal_sdpa_decode_gqa(q, k, v, q_start, store, tape)
}

pub fn add_broadcast(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Inner `broadcast::add_broadcast` dispatches on
    // `a.dirty`/`b.dirty`; both-Device/Both inputs stay lazy via
    // `backend.add_broadcast` (MLX `mlx_add` broadcasts natively via
    // right-alignment). Mixed Host/Device inputs fall back to the host
    // path. No `ensure_host` here — the hot paths are Qwen3.5 attention's
    // causal-mask add (`scaled + causal_mask` per layer × 28 layers) and
    // Linear bias add (`linear_out + bias` per projection × many projections).
    broadcast::add_broadcast(a, b, store, tape)
}

pub fn add(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    elementwise::add(a, b, store, tape)
}

/// Elementwise `|x|`. Backward is `grad * sign(x)`, with `sign(0) = 0`.
pub fn abs(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    elementwise::abs(x, store, tape)
}

pub fn mul(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Inner `elementwise::mul` dispatches OR-lazy; if either
    // operand is Dirty::Device/Both the pair stays on `mlx_multiply`
    // in the MLX graph, else the host-eager path kicks in. No `ensure_host` —
    // Qwen3.5 hot paths are `attn * gate` and `silu(gate) * up` per
    // attention/MLP layer × 28 layers.
    elementwise::mul(a, b, store, tape)
}

pub fn mul_scalar(
    a: TensorId,
    k: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Inner `elementwise::mul_scalar` dispatches on `a.dirty`;
    // Dirty::Device/Both inputs stay lazy via `backend.mul_scalar` (MLX
    // `mlx_multiply(x, scalar_arr)` — broadcast rank-0 scalar).
    // Dirty::Host takes the host-eager path. No `ensure_host` here —
    // Qwen3.5 attention scales q by `1/sqrt(d_head)` once per layer, which
    // would otherwise force a readback of every layer's q projection before
    // softmax.
    elementwise::mul_scalar(a, k, store, tape)
}

pub fn embedding(
    table: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Inner dispatcher picks lazy-device vs host-eager via `table`'s dirty
    // bit + device handle; each branch materializes the table on its own
    // side. No eager readback here — that would defeat the lazy branch.
    embed::embedding(table, indices, store, tape)
}

pub fn gather_last_dim(
    src: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Inner `gather::gather_last_dim` dispatches on `src.dirty`;
    // a Dirty::Device/Both input stays lazy via `backend.gather_last_dim`
    // (composes `mlx_reshape → mlx_take_axis → mlx_reshape` into the MLX
    // graph), while Dirty::Host takes the eager host path. No `ensure_host`
    // here — logits coming out of the final matmul would otherwise be
    // flushed to host before the gather.
    gather::gather_last_dim(src, indices, store, tape)
}

pub use moe::{MoeGroupedLinearExpert, MoeGroupedLinearInput, MoeGroupedRoute, MoeRoute, MoeTopK};

pub fn moe_topk_softmax(
    logits: TensorId,
    top_k: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<MoeTopK> {
    moe::moe_topk_softmax(logits, top_k, store, tape)
}

pub fn moe_topk_softmax_with_indices(
    logits: TensorId,
    top_k: usize,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<MoeTopK> {
    moe::moe_topk_softmax_with_indices(logits, top_k, indices, store, tape)
}

pub fn moe_gather_rows(
    src: TensorId,
    rows: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    moe::moe_gather_rows(src, rows, store, tape)
}

pub fn moe_weighted_scatter(
    values: TensorId,
    weights: TensorId,
    routes: &[MoeRoute],
    out_rows: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    moe::moe_weighted_scatter(values, weights, routes, out_rows, store, tape)
}

pub fn moe_grouped_linear(
    input: TensorId,
    experts: &[MoeGroupedLinearExpert],
    routes: &[MoeGroupedRoute],
    input_kind: MoeGroupedLinearInput,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    moe::moe_grouped_linear(input, experts, routes, input_kind, store, tape)
}

pub fn moe_grouped_weighted_scatter(
    values: TensorId,
    weights: TensorId,
    routes: &[MoeGroupedRoute],
    out_rows: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    moe::moe_grouped_weighted_scatter(values, weights, routes, out_rows, store, tape)
}

pub fn reshape(
    x: TensorId,
    shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Inner `layout::reshape` dispatches on `x.dirty`; a
    // Dirty::Device input stays lazy via `backend.reshape` (MLX
    // `mlx_reshape` — metadata-only, no compute), Dirty::Host takes the
    // host-eager path. No `ensure_host` here — it would flush the q/k/v
    // projection matmul's output to host before every attention-layer reshape.
    layout::reshape(x, shape, store, tape)
}

pub fn broadcast_expand(
    src: TensorId,
    target_shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    layout::broadcast_expand(src, target_shape, store, tape)
}

pub fn transpose(
    x: TensorId,
    axis1: usize,
    axis2: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Inner `layout::transpose` dispatches on `x.dirty`;
    // device-resident input stays lazy via `backend.transpose_axes_swap`
    // (MLX `mlx_transpose_axes` — a lazy view fused into downstream
    // GEMMs). Same rationale as `reshape` — the hot path is Qwen3.5
    // q/k/v projections.
    layout::transpose(x, axis1, axis2, store, tape)
}

pub fn slice(
    x: TensorId,
    starts: &[usize],
    ends: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Inner `layout::slice` dispatches on `x.dirty`; a
    // Dirty::Device input stays lazy via `backend.slice` (composes
    // `mlx_slice → mlx_contiguous` into the MLX graph), Dirty::Host takes
    // the host-eager path. No `ensure_host` here — it would flush the fused
    // q_full projection's matmul output to host before every Qwen3.5
    // attention-layer q/gate split.
    layout::slice(x, starts, ends, store, tape)
}

pub fn cat(
    inputs: &[TensorId],
    axis: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    layout::cat(inputs, axis, store, tape)
}

pub fn matmul(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    matmul::matmul(a, b, store, tape)
}

pub fn matmul_bt(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    matmul::matmul_bt(a, b, store, tape)
}

pub fn matmul_bt_with_site(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
    site: &'static str,
) -> Result<TensorId> {
    matmul::matmul_bt_with_site(a, b, store, tape, site)
}

pub use linear_attention::{
    LinearAttentionParams, linear_attention_boundary, linear_attention_core_carry,
    linear_attention_core_cp, linear_attention_core_with_carry,
    linear_attention_core_with_carry_taped, linear_attention_ctx_bytes,
    linear_attention_row_transient_bytes,
};

pub fn linear_attention_core(
    qkv: TensorId,
    z: TensorId,
    b_proj: TensorId,
    a_proj: TensorId,
    conv1d_weight: TensorId,
    dt_bias: TensorId,
    a_log: TensorId,
    norm_weight: TensorId,
    params: LinearAttentionParams,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    linear_attention::linear_attention_core(
        qkv,
        z,
        b_proj,
        a_proj,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        params,
        store,
        tape,
    )
}

pub fn rmsnorm(
    x: TensorId,
    weight: TensorId,
    eps: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Inner dispatcher chooses lazy-device vs host-eager based on x's
    // dirty bit + device handle; it handles `ensure_host(weight)` and
    // `ensure_device(x)` on the lazy branch, `store.tensor(x).clone()`
    // on the eager branch. No eager readback of `x` here — that would
    // pay an mlx_eval the lazy path is designed to skip.
    norm::rmsnorm(x, weight, eps, store, tape)
}

pub fn rope(
    x: TensorId,
    cos: TensorId,
    sin: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // Inner `rope::rope` dispatches on `x.dirty`; a Dirty::Device
    // `x` stays lazy via `backend.rope` (half-split rotation on device),
    // while Dirty::Host/Both take the eager host path. cos/sin are
    // `ensure_host`-ed inside the lazy branch (caches are typically host
    // already; the readback is a no-op in the common case). No `ensure_host(x)`
    // here — it would force a readback of every q/k before each layer's rope.
    rope::rope(x, cos, sin, store, tape)
}

pub fn mean(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Inner `reduce::mean` dispatches on `a.dirty`; a Dirty::Device
    // input stays lazy by composing `sum_all + mul_scalar(1/numel)` on the
    // MLX graph (reusing the lazy `sum_all` + `mul_scalar` — no new trait
    // method needed), while Dirty::Host / Dirty::Both take the eager host
    // path. No `ensure_host` here — the CE-loss path
    // `log_softmax → gather_last_dim → mean` would otherwise flush the full
    // log-probs tensor to host per step, reversing every upstream lazy win.
    reduce::mean(a, store, tape)
}

pub fn sum(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // `sum` is device-resident on Metal — `reduce::sum` calls
    // `store.ensure_device(a)` and `backend.sum_all`, composing into the
    // MLX lazy graph instead of forcing a host readback. CPU/CUDA still get
    // a fully-realized scalar handle; lazy semantics are Metal-only.
    reduce::sum(a, store, tape)
}

pub fn softmax(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // `softmax` is device-resident on Metal for Dirty::Device
    // inputs — `softmax::softmax` routes to `backend.softmax_last_axis`
    // (composes `mlx_softmax_axis` into the MLX lazy graph, no eval).
    // Dirty::Host / Dirty::Both inputs stay on the host fast path. CPU/CUDA
    // use the default trait fallback (readback → host → upload); lazy
    // semantics are Metal-only.
    softmax::softmax(x, store, tape)
}

pub fn log_softmax(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // See `softmax`. Metal path composes `mlx_logsumexp_axis + mlx_subtract`.
    softmax::log_softmax(x, store, tape)
}

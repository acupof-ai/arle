// Thin public dispatch layer. No `ensure_host` / `ensure_device` here — the
// inner ops own residency decisions; a readback at this layer defeats the
// lazy device path.

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
    activation::gelu(x, store, tape)
}

pub fn silu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    activation::silu(x, store, tape)
}

pub fn sigmoid(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
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
    elementwise::mul(a, b, store, tape)
}

pub fn mul_scalar(
    a: TensorId,
    k: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    elementwise::mul_scalar(a, k, store, tape)
}

pub fn embedding(
    table: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    embed::embedding(table, indices, store, tape)
}

pub fn gather_last_dim(
    src: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
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
    layout::transpose(x, axis1, axis2, store, tape)
}

pub fn slice(
    x: TensorId,
    starts: &[usize],
    ends: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
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
    norm::rmsnorm(x, weight, eps, store, tape)
}

pub fn rope(
    x: TensorId,
    cos: TensorId,
    sin: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    rope::rope(x, cos, sin, store, tape)
}

pub fn mean(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    reduce::mean(a, store, tape)
}

pub fn sum(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    reduce::sum(a, store, tape)
}

pub fn softmax(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    softmax::softmax(x, store, tape)
}

pub fn log_softmax(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    softmax::log_softmax(x, store, tape)
}

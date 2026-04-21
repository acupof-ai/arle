#[path = "ops/activation.rs"]
pub mod activation;
#[path = "ops/attention.rs"]
pub mod attention;
#[path = "ops/broadcast.rs"]
pub mod broadcast;
#[path = "ops/elementwise.rs"]
pub mod elementwise;
#[path = "ops/embed.rs"]
pub mod embed;
#[path = "ops/gather.rs"]
pub mod gather;
#[path = "ops/layout.rs"]
pub mod layout;
#[path = "ops/linear_attention.rs"]
pub mod linear_attention;
#[path = "ops/matmul.rs"]
pub mod matmul;
#[path = "ops/norm.rs"]
pub mod norm;
#[path = "ops/reduce.rs"]
pub mod reduce;
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
pub(crate) use broadcast::add_broadcast_backward;
pub(crate) use elementwise::{add_backward, mul_backward, mul_scalar_backward};
pub(crate) use embed::embedding_backward;
pub(crate) use gather::gather_last_dim_backward;
pub(crate) use layout::{reshape_backward, slice_backward, transpose_backward};
pub(crate) use linear_attention::linear_attention_backward;
pub(crate) use matmul::matmul_backward;
pub(crate) use norm::rmsnorm_backward;
pub(crate) use reduce::{mean_backward, sum_backward};
pub(crate) use rope::rope_backward;
pub(crate) use softmax::{log_softmax_backward, softmax_backward};

pub fn exp(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.4: inner `activation::exp` dispatches on `dirty`; a Dirty::Device
    // input stays lazy via `backend.exp` (MLX `mlx_exp`), while Dirty::Host
    // / Dirty::Both take the eager host path. Stripping `ensure_host` here
    // is the critical enabler — previously it forced a readback before the
    // inner fn could see the device state.
    activation::exp(x, store, tape)
}

pub fn gelu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // Inner dispatcher picks lazy-device vs host-eager; stripping the
    // eager readback here lets the lazy branch keep x Dirty::Device.
    activation::gelu(x, store, tape)
}

pub fn silu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.3: `silu` is now device-resident on Metal for Dirty::Device
    // inputs — `activation::silu` routes to `backend.silu` (composes
    // `mlx_multiply(x, mlx_sigmoid(x))` into the MLX lazy graph, no eval).
    // Dirty::Host / Dirty::Both inputs stay on the host fast path. CPU/CUDA
    // use the default trait fallback (readback → host → upload); lazy
    // semantics are Metal-only.
    activation::silu(x, store, tape)
}

pub fn sigmoid(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.18: inner `activation::sigmoid` dispatches on `x.dirty`; a
    // Dirty::Device/Both input stays lazy via `backend.sigmoid`
    // (`mlx_sigmoid` single node into the MLX graph), Dirty::Host takes
    // the eager host path. Stripping `ensure_host` here is the enabler —
    // Qwen3.5 attention's `gate = sigmoid(gate_proj)` × 28 layers
    // previously flushed the q_full slice to host before the gate
    // multiply.
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
    // M5.3b.15: composite op — body is pure dispatch over reshape /
    // transpose / matmul / mul_scalar / add_broadcast / softmax, all of
    // which are lazy on Metal post M5.3b.1–14. Stripping `ensure_host`
    // here lets the entire attention chain stay in the MLX graph end-
    // to-end for each layer (Qwen3.5 × 28 layers).
    attention::causal_sdpa(q, k, v, store, tape)
}

pub fn add_broadcast(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // M5.3b.14: inner `broadcast::add_broadcast` dispatches on
    // `a.dirty`/`b.dirty`; both-Device/Both inputs stay lazy via
    // `backend.add_broadcast` (MLX `mlx_add` broadcasts natively via
    // right-alignment). Mixed Host/Device inputs fall back to the host
    // path. Stripping `ensure_host` here is the enabler — the hot paths
    // are Qwen3.5 attention's causal-mask add (`scaled + causal_mask`
    // per layer × 28 layers) and Linear bias add (`linear_out + bias`
    // per projection × many projections).
    broadcast::add_broadcast(a, b, store, tape)
}

pub fn add(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    elementwise::add(a, b, store, tape)
}

pub fn mul(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.17: inner `elementwise::mul` dispatches OR-lazy; if either
    // operand is Dirty::Device/Both the pair stays on `mlx_multiply`
    // in the MLX graph, else the host-eager path kicks in. Stripping
    // `ensure_host` is the enabler — Qwen3.5 hot paths are `attn * gate`
    // and `silu(gate) * up` per attention/MLP layer × 28 layers.
    elementwise::mul(a, b, store, tape)
}

pub fn mul_scalar(
    a: TensorId,
    k: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // M5.3b.13: inner `elementwise::mul_scalar` dispatches on `a.dirty`;
    // Dirty::Device/Both inputs stay lazy via `backend.mul_scalar` (MLX
    // `mlx_multiply(x, scalar_arr)` — broadcast rank-0 scalar).
    // Dirty::Host takes the host-eager path. Stripping `ensure_host` here
    // is the enabler — Qwen3.5 attention scales q by `1/sqrt(d_head)`
    // once per layer, and previously this forced a readback of every
    // layer's q projection before softmax.
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
    // M5.3b.9: inner `gather::gather_last_dim` dispatches on `src.dirty`;
    // a Dirty::Device/Both input stays lazy via `backend.gather_last_dim`
    // (composes `mlx_reshape → mlx_take_axis → mlx_reshape` into the MLX
    // graph), while Dirty::Host takes the eager host path. Stripping
    // `ensure_host` here is the enabler — previously logits coming out of
    // the final matmul were flushed to host before the gather.
    gather::gather_last_dim(src, indices, store, tape)
}

pub fn reshape(
    x: TensorId,
    shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // M5.3b.12: inner `layout::reshape` dispatches on `x.dirty`; a
    // Dirty::Device input stays lazy via `backend.reshape` (MLX
    // `mlx_reshape` — metadata-only, no compute), Dirty::Host takes the
    // host-eager path. Stripping `ensure_host` here is the enabler — it
    // previously flushed the q/k/v projection matmul's output to host
    // before every attention-layer reshape.
    layout::reshape(x, shape, store, tape)
}

pub fn transpose(
    x: TensorId,
    axis1: usize,
    axis2: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    // M5.3b.12: inner `layout::transpose` dispatches on `x.dirty`;
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
    // M5.3b.16: inner `layout::slice` dispatches on `x.dirty`; a
    // Dirty::Device input stays lazy via `backend.slice` (composes
    // `mlx_slice → mlx_contiguous` into the MLX graph), Dirty::Host takes
    // the host-eager path. Stripping `ensure_host` here is the enabler —
    // it previously flushed the fused q_full projection's matmul output
    // to host before every Qwen3.5 attention-layer q/gate split.
    layout::slice(x, starts, ends, store, tape)
}

pub fn matmul(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    matmul::matmul(a, b, store, tape)
}

pub use linear_attention::LinearAttentionParams;

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
    // M5.3b.5: inner `rope::rope` dispatches on `x.dirty`; a Dirty::Device
    // `x` stays lazy via `backend.rope` (half-split rotation on device),
    // while Dirty::Host/Both take the eager host path. cos/sin are
    // `ensure_host`-ed inside the lazy branch (caches are typically host
    // already; the readback is a no-op in the common case). Stripping the
    // `ensure_host(x)` here is the enabler — it previously forced a
    // readback of every q/k before each layer's rope.
    rope::rope(x, cos, sin, store, tape)
}

pub fn mean(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.19: inner `reduce::mean` dispatches on `a.dirty`; a Dirty::Device
    // input stays lazy by composing `sum_all + mul_scalar(1/numel)` on the
    // MLX graph (reusing the M5.3b.1 lazy `sum_all` + M5.3b.13 lazy
    // `mul_scalar` — no new trait method needed), while Dirty::Host /
    // Dirty::Both take the eager host path. Stripping `ensure_host` here is
    // the enabler — the CE-loss path `log_softmax → gather_last_dim → mean`
    // previously flushed the full log-probs tensor to host per step,
    // reversing every upstream M5.3b lazy win.
    reduce::mean(a, store, tape)
}

pub fn sum(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.1: `sum` is now device-resident on Metal — `reduce::sum` calls
    // `store.ensure_device(a)` and `backend.sum_all`, composing into the
    // MLX lazy graph instead of forcing a host readback. CPU/CUDA still get
    // a fully-realized scalar handle; lazy semantics are Metal-only.
    reduce::sum(a, store, tape)
}

pub fn softmax(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    // M5.3b.2: `softmax` is now device-resident on Metal for Dirty::Device
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

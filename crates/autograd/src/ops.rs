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
pub(crate) use matmul::matmul_backward;
pub(crate) use norm::rmsnorm_backward;
pub(crate) use reduce::{mean_backward, sum_backward};
pub(crate) use rope::rope_backward;
pub(crate) use softmax::{log_softmax_backward, softmax_backward};

pub fn exp(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_host(x)?;
    activation::exp(x, store, tape)
}

pub fn gelu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_host(x)?;
    activation::gelu(x, store, tape)
}

pub fn silu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_host(x)?;
    activation::silu(x, store, tape)
}

pub fn sigmoid(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_host(x)?;
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
    store.ensure_host(q)?;
    store.ensure_host(k)?;
    store.ensure_host(v)?;
    attention::causal_sdpa(q, k, v, store, tape)
}

pub fn add_broadcast(
    a: TensorId,
    b: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(a)?;
    store.ensure_host(b)?;
    broadcast::add_broadcast(a, b, store, tape)
}

pub fn add(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    elementwise::add(a, b, store, tape)
}

pub fn mul(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_host(a)?;
    store.ensure_host(b)?;
    elementwise::mul(a, b, store, tape)
}

pub fn mul_scalar(
    a: TensorId,
    k: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(a)?;
    elementwise::mul_scalar(a, k, store, tape)
}

pub fn embedding(
    table: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(table)?;
    embed::embedding(table, indices, store, tape)
}

pub fn gather_last_dim(
    src: TensorId,
    indices: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(src)?;
    gather::gather_last_dim(src, indices, store, tape)
}

pub fn reshape(
    x: TensorId,
    shape: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(x)?;
    layout::reshape(x, shape, store, tape)
}

pub fn transpose(
    x: TensorId,
    axis1: usize,
    axis2: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(x)?;
    layout::transpose(x, axis1, axis2, store, tape)
}

pub fn slice(
    x: TensorId,
    starts: &[usize],
    ends: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(x)?;
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

pub fn rmsnorm(
    x: TensorId,
    weight: TensorId,
    eps: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(x)?;
    store.ensure_host(weight)?;
    norm::rmsnorm(x, weight, eps, store, tape)
}

pub fn rope(
    x: TensorId,
    cos: TensorId,
    sin: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    store.ensure_host(x)?;
    store.ensure_host(cos)?;
    store.ensure_host(sin)?;
    rope::rope(x, cos, sin, store, tape)
}

pub fn mean(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_host(a)?;
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
    store.ensure_host(x)?;
    softmax::softmax(x, store, tape)
}

pub fn log_softmax(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_host(x)?;
    softmax::log_softmax(x, store, tape)
}

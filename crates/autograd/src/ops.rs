#[path = "ops/activation.rs"]
pub mod activation;
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
#[path = "ops/softmax.rs"]
pub mod softmax;

use crate::{
    Result,
    tape::Tape,
    tensor::{TensorId, TensorStore},
};

pub(crate) use activation::{exp_backward, gelu_backward};
pub(crate) use broadcast::add_broadcast_backward;
pub(crate) use elementwise::{add_backward, mul_backward, mul_scalar_backward};
pub(crate) use embed::embedding_backward;
pub(crate) use gather::gather_last_dim_backward;
pub(crate) use layout::{reshape_backward, transpose_backward};
pub(crate) use matmul::matmul_backward;
pub(crate) use norm::rmsnorm_backward;
pub(crate) use reduce::{mean_backward, sum_backward};
pub(crate) use softmax::{log_softmax_backward, softmax_backward};

pub fn exp(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_host(x)?;
    activation::exp(x, store, tape)
}

pub fn gelu(x: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_host(x)?;
    activation::gelu(x, store, tape)
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
    store.ensure_host(a)?;
    store.ensure_host(b)?;
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

pub fn mean(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_host(a)?;
    reduce::mean(a, store, tape)
}

pub fn sum(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> Result<TensorId> {
    store.ensure_host(a)?;
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

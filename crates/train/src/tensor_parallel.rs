//! Tensor-parallel (TP) placement + collectives — the model-agnostic core.
//!
//! TP shards a layer's projection weights column/row-wise across N ranks and
//! all-reduces the partial outputs. This module owns the pure coordinate view
//! (`TpContext`, the `attn_tp` sub-axis of the one device mesh), the divisibility
//! rule every sharded dimension obeys (`divide`), and the collective wrapper
//! (`maybe_all_reduce`) — all of it independent of any model's config. A model
//! layer applies these to its own dimensions; the transport lives in `autograd`.
//!
//! Mirror of `context_parallel::CpContext` — same mesh, a different sharded axis
//! (weights, not sequence). `TpContext::single()` is the byte-identical
//! single-card path.

use autograd::{Result, Tape, TensorId, TensorStore, ops::all_reduce_sum};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpContext {
    pub rank: usize,
    pub world_size: usize,
}

impl TpContext {
    pub const fn single() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }

    pub const fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    pub fn is_enabled(self) -> bool {
        self.world_size > 1
    }

    pub fn divide(self, value: usize) -> Option<usize> {
        (self.world_size != 0 && value.is_multiple_of(self.world_size))
            .then_some(value / self.world_size)
    }
}

pub fn maybe_all_reduce(
    x: TensorId,
    tp: TpContext,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    if tp.is_enabled() {
        all_reduce_sum(x, store, tape)
    } else {
        Ok(x)
    }
}

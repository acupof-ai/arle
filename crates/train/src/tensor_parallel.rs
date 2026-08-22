//! Model-agnostic TP coordinate view; mirror of `context_parallel::CpContext`
//! on the weights axis. `TpContext::single()` is the byte-identical single-card path.

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

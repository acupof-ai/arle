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

    pub fn from_coord(cfg: infer_topo::MultiAxisConfig, coord: infer_topo::RankCoord) -> Self {
        Self {
            rank: coord.attn_tp_rank,
            world_size: cfg.attn_tp_size(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_is_disabled_and_divides_trivially() {
        let tp = TpContext::single();
        assert!(!tp.is_enabled());
        assert_eq!(tp.divide(128), Some(128));
    }

    #[test]
    fn divide_shards_evenly_and_rejects_indivisible() {
        let tp = TpContext::new(1, 4);
        assert!(tp.is_enabled());
        assert_eq!(tp.divide(128), Some(32));
        assert_eq!(tp.divide(30), None);
    }

    #[test]
    fn from_coord_pure_tp_matches_rank_size() {
        use infer_topo::{MultiAxisConfig, RankCoord};
        for size in [1usize, 2, 4, 8] {
            let cfg = MultiAxisConfig {
                tp_size: size,
                pp_size: 1,
                ep_size: 1,
                attn_dp_size: 1,
                attn_cp_size: 1,
                moe_dp_size: 1,
            };
            for rank in 0..size {
                let coord = RankCoord::from_world_rank(cfg, rank).unwrap();
                let tp = TpContext::from_coord(cfg, coord);
                assert_eq!(
                    tp,
                    TpContext {
                        rank,
                        world_size: size
                    }
                );
            }
        }
    }
}

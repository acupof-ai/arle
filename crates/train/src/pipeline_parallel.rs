//! Pipeline-parallel (PP) layer partitioning for OPD training.
//!
//! PP splits the LAYER stack across `pp_size` stages; a stage owns a contiguous
//! layer range, sends its output activation to the next stage, and receives the
//! grad on the way back. This module owns the pure layer-range math (which layers
//! a `pp_rank` owns); the cross-stage activation send/recv transport is the
//! pending-remote data-plane.
//!
//! HONEST FIT NOTE — PP is a poor match for single-pass OPD writeback, by design,
//! not by effort. A 1F1B schedule exists to overlap the forward and backward of
//! DIFFERENT microbatches so the pipeline bubble is amortized. OPD writeback runs
//! ONE trajectory per step, so there is no stream of microbatches to interleave:
//! true 1F1B has nothing to amortize and degenerates to fill-drain (bubble
//! = (stages-1)/stages idle). The coherent increment for this workload is
//! layer-partition pipeline MODEL-parallelism (this helper + activation send/recv),
//! NOT a 1F1B scheduler. 1F1B only pays off once writeback micro-batches the batch
//! axis — tracked, not built.

use crate::lora_shard::shard_range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PpContext {
    pub rank: usize,
    pub size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRange {
    pub start: usize,
    pub end: usize,
}

impl PpContext {
    pub const fn single() -> Self {
        Self { rank: 0, size: 1 }
    }

    pub fn from_mesh(pp_size: usize, tp_size: usize, world_rank: usize) -> Self {
        let cfg = infer_topo::MultiAxisConfig {
            tp_size: tp_size.max(1),
            pp_size: pp_size.max(1),
            ep_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        match infer_topo::RankCoord::from_world_rank(cfg, world_rank) {
            Ok(coord) => Self {
                rank: coord.pp_rank,
                size: cfg.pp_size,
            },
            Err(_) => Self::single(),
        }
    }

    pub fn is_enabled(self) -> bool {
        self.size > 1
    }

    pub fn is_first(self) -> bool {
        self.rank == 0
    }

    pub fn is_last(self) -> bool {
        self.rank + 1 == self.size
    }

    pub fn layers(self, num_layers: usize) -> LayerRange {
        if self.size <= 1 {
            return LayerRange {
                start: 0,
                end: num_layers,
            };
        }
        let range = shard_range(num_layers, self.rank, self.size);
        LayerRange {
            start: range.start,
            end: range.end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_owns_all_layers() {
        let pp = PpContext::single();
        assert!(!pp.is_enabled());
        assert!(pp.is_first() && pp.is_last());
        assert_eq!(pp.layers(64), LayerRange { start: 0, end: 64 });
    }

    #[test]
    fn stages_partition_layers_contiguously() {
        let (size, layers) = (4usize, 64usize);
        let mut next = 0;
        for rank in 0..size {
            let pp = PpContext::from_mesh(size, 1, rank);
            let r = pp.layers(layers);
            assert_eq!(r.start, next, "gap/overlap at stage {rank}");
            assert_eq!(pp.is_first(), rank == 0);
            assert_eq!(pp.is_last(), rank + 1 == size);
            next = r.end;
        }
        assert_eq!(next, layers, "last stage must reach num_layers");
    }

    #[test]
    fn remainder_layers_go_to_last_stage() {
        assert_eq!(
            PpContext::from_mesh(3, 1, 0).layers(10),
            LayerRange { start: 0, end: 3 }
        );
        assert_eq!(
            PpContext::from_mesh(3, 1, 1).layers(10),
            LayerRange { start: 3, end: 6 }
        );
        assert_eq!(
            PpContext::from_mesh(3, 1, 2).layers(10),
            LayerRange { start: 6, end: 10 }
        );
    }
}

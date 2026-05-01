//! Tensor-parallel weight-load context.

use anyhow::Result;

use crate::tensor_parallel::{ShardingSpec, TpConfig, column_shard, head_shard, row_shard};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TpShardAxis {
    Row,
    Column,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TpLoadContext {
    pub rank: usize,
    pub world_size: usize,
    pub sharding: ShardingSpec,
    pub axis: TpShardAxis,
}

impl TpLoadContext {
    pub fn single(total: usize, axis: TpShardAxis) -> Self {
        Self {
            rank: 0,
            world_size: 1,
            sharding: ShardingSpec {
                offset: 0,
                size: total,
                total,
            },
            axis,
        }
    }

    pub fn column(rank: usize, world_size: usize, total_out_features: usize) -> Result<Self> {
        let tp = TpConfig::new(world_size, rank)?;
        Ok(Self {
            rank,
            world_size,
            sharding: column_shard(total_out_features, &tp),
            axis: TpShardAxis::Column,
        })
    }

    pub fn row(rank: usize, world_size: usize, total_in_features: usize) -> Result<Self> {
        let tp = TpConfig::new(world_size, rank)?;
        Ok(Self {
            rank,
            world_size,
            sharding: row_shard(total_in_features, &tp),
            axis: TpShardAxis::Row,
        })
    }

    pub fn head(
        rank: usize,
        world_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) -> Result<(usize, usize)> {
        let tp = TpConfig::new(world_size, rank)?;
        head_shard(num_q_heads, num_kv_heads, &tp)
    }

    pub fn is_single(&self) -> bool {
        self.world_size == 1 && self.sharding.is_full()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_context_uses_existing_shard_math() {
        let ctx = TpLoadContext::column(1, 2, 10).unwrap();
        assert_eq!(ctx.rank, 1);
        assert_eq!(ctx.world_size, 2);
        assert_eq!(ctx.axis, TpShardAxis::Column);
        assert_eq!(ctx.sharding.offset, 5);
        assert_eq!(ctx.sharding.size, 5);
    }

    #[test]
    fn row_context_keeps_remainder_on_last_rank() {
        let ctx = TpLoadContext::row(2, 3, 8).unwrap();
        assert_eq!(ctx.axis, TpShardAxis::Row);
        assert_eq!(ctx.sharding.offset, 4);
        assert_eq!(ctx.sharding.size, 4);
    }

    #[test]
    fn head_context_validates_gqa_divisibility() {
        assert_eq!(TpLoadContext::head(0, 2, 32, 8).unwrap(), (16, 4));
        assert!(TpLoadContext::head(0, 3, 32, 8).is_err());
    }
}

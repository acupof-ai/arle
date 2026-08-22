//! `infer-topo` — pure, CPU-verifiable, device-independent topology + sharding
//! math (TP rank placement, weight/head sharding, the SGLang-style multi-axis
//! layout, and rank-group builders); depends only on `std`.
//!
//! Column-parallel splits the output dim (all-reduce to sum partials);
//! row-parallel splits the input dim (pre-sharded input, all-reduce at output).

#[path = "error.rs"]
mod error;
#[path = "sharding.rs"]
mod sharding;
#[path = "topology.rs"]
mod topology;

pub use error::{Result, TopoError};
pub use sharding::{
    ParallelLinearKind, ShardingSpec, TpConfig, column_shard, head_shard, kv_load_block_index,
    row_shard,
};
pub use topology::{
    MultiAxisConfig, RankCoord, build_attn_cp_groups, build_attn_tp_groups, build_moe_ep_groups,
    build_tp_groups,
};

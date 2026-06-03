//! `infer-topo` — Phase-1 (Tensor Parallel) FOUNDATION for the ARLE rewrite.
//!
//! Pure, **CPU-verifiable**, device-independent topology + sharding math.
//! This is a faithful port of the numerics in the legacy
//! `infer/src/tensor_parallel.rs`, with all GPU / NCCL / `Communicator`
//! coupling deliberately dropped. NCCL wiring is gated on H20 verification and
//! lands in a later phase — it does **not** belong in this foundation crate.
//!
//! The crate depends on nothing but `std` (plus an optional `serde` feature for
//! (de)serializing the config/topology types).
//!
//! # What lives here
//!
//! - [`TpConfig`] — a single tensor-parallel rank placement (`world_size`, `rank`).
//! - [`ShardingSpec`] — a rank's `offset`/`size`/`total` slice of one dimension.
//! - [`column_shard`] / [`row_shard`] — split a weight dimension across TP ranks.
//! - [`head_shard`] — split attention Q/KV heads across TP ranks (GQA-aware).
//! - [`TpLinearConfig`] / [`ParallelLinearKind`] — column- vs row-parallel linear.
//! - [`MultiAxisConfig`] — the SGLang-style multi-axis parallelism layout
//!   (tp/pp/ep + attention-DP/CP + MoE-DP), with derived `attn_tp` / `moe_tp`.
//! - [`RankCoord`] — decompose a global rank into its per-axis coordinate.
//! - `build_*_groups` — the rank-list of each communication/compute group.
//!
//! # Tensor Parallel overview
//!
//! ```text
//! ColumnParallelLinear:   output dim split → each rank holds W[:, offset..offset+size]
//!                         Requires all-reduce on output to sum partial results.
//!
//! RowParallelLinear:      input dim split  → each rank holds W[offset..offset+size, :]
//!                         Input is pre-sharded; all-reduce needed at output.
//! ```

// Flat module layout, no `mod.rs` (ARLE convention).
#[path = "error.rs"]
mod error;
#[path = "sharding.rs"]
mod sharding;
#[path = "topology.rs"]
mod topology;

pub use error::{Result, TopoError};
pub use sharding::{
    ParallelLinearKind, ShardingSpec, TpConfig, TpLinearConfig, column_shard, head_shard, row_shard,
};
pub use topology::{
    MultiAxisConfig, RankCoord, build_attn_cp_groups, build_attn_dp_groups,
    build_attn_owner_groups, build_attn_tp_groups, build_moe_dp_groups, build_moe_ep_groups,
    build_moe_tp_groups, build_pp_groups, build_tp_groups,
};

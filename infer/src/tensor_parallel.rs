//! Tensor Parallel configuration and sharding utilities.
//!
//! This module provides the **CPU-side** configuration and sharding math for
//! tensor parallelism.  GPU communication primitives (NCCL all-reduce /
//! all-gather) are declared as stubs and live behind `#[cfg(feature = "cuda")]`.
//!
//! # Tensor Parallel overview
//!
//! Tensor parallelism (TP) splits model weight matrices across multiple GPUs:
//!
//! ```text
//! ColumnParallelLinear:   output dim split → each GPU holds W[:, offset..offset+size]
//!                         Requires all-reduce on output to sum partial results.
//!
//! RowParallelLinear:      input dim split  → each GPU holds W[offset..offset+size, :]
//!                         Input is pre-sharded; all-reduce needed at output.
//! ```
//!
//! # CPU-verifiable
//!
//! - [`TpConfig`] validation
//! - [`ShardingSpec`] computation via [`column_shard`] / [`row_shard`]
//! - Head assignment via [`head_shard`]
//! - [`TpLinearConfig`] builder for both parallel linear types

use anyhow::{Result, bail};

// ============================================================================
// TpConfig
// ============================================================================

/// Tensor parallel configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TpConfig {
    /// Total number of TP ranks (GPUs in the tensor-parallel group).
    pub world_size: usize,
    /// This rank's index within the TP group (0 ≤ rank < world_size).
    pub rank: usize,
}

impl TpConfig {
    /// Single-GPU configuration (no parallelism).
    pub fn single() -> Self {
        Self {
            world_size: 1,
            rank: 0,
        }
    }

    /// Multi-GPU configuration.
    pub fn new(world_size: usize, rank: usize) -> Result<Self> {
        if world_size == 0 {
            bail!("world_size must be ≥ 1");
        }
        if rank >= world_size {
            bail!("rank ({rank}) must be < world_size ({world_size})");
        }
        Ok(Self { world_size, rank })
    }

    /// True when running on a single GPU (no all-reduce needed).
    pub fn is_single(&self) -> bool {
        self.world_size == 1
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.world_size == 0 {
            bail!("world_size must be ≥ 1");
        }
        if self.rank >= self.world_size {
            bail!("rank {} ≥ world_size {}", self.rank, self.world_size);
        }
        Ok(())
    }
}

impl Default for TpConfig {
    fn default() -> Self {
        Self::single()
    }
}

// ============================================================================
// ShardingSpec
// ============================================================================

/// Describes a rank's slice of a dimension of size `total`.
///
/// The rank owns `self.size` elements starting at `self.offset`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardingSpec {
    /// Starting index of this rank's shard.
    pub offset: usize,
    /// Number of elements owned by this rank.
    pub size: usize,
    /// Total size of the dimension (sum of all ranks' sizes).
    pub total: usize,
}

impl ShardingSpec {
    /// Exclusive end index: `offset + size`.
    pub fn end(&self) -> usize {
        self.offset + self.size
    }

    /// Return the range as a `std::ops::Range`.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.end()
    }

    /// True if this rank owns the entire dimension (single-GPU case).
    pub fn is_full(&self) -> bool {
        self.offset == 0 && self.size == self.total
    }
}

// ============================================================================
// Sharding functions
// ============================================================================

/// Compute the shard for a **column-parallel** dimension (output features split
/// across TP ranks).
///
/// The last rank absorbs any remainder so that `sum(all sizes) == total`.
///
/// # Panics
/// Panics if `total < world_size` (cannot give each rank at least 1 element).
pub fn column_shard(total: usize, tp: &TpConfig) -> ShardingSpec {
    assert!(
        total >= tp.world_size,
        "total ({total}) < world_size ({}): cannot shard",
        tp.world_size
    );
    let base = total / tp.world_size;
    let remainder = total % tp.world_size;
    // Distribute remainder to the last rank.
    let offset = tp.rank * base;
    let size = if tp.rank == tp.world_size - 1 {
        base + remainder
    } else {
        base
    };
    ShardingSpec {
        offset,
        size,
        total,
    }
}

/// Compute the shard for a **row-parallel** dimension (input features split
/// across TP ranks).
///
/// Identical formula to column_shard — differs only in semantic interpretation.
pub fn row_shard(total: usize, tp: &TpConfig) -> ShardingSpec {
    column_shard(total, tp)
}

/// Compute the assignment of attention heads for this TP rank.
///
/// Returns `(num_q_heads_local, num_kv_heads_local)`.
///
/// # Errors
/// Returns an error if `num_kv_heads` is not divisible by `world_size`
/// (GQA head assignment must be uniform across TP ranks).
pub fn head_shard(
    num_q_heads: usize,
    num_kv_heads: usize,
    tp: &TpConfig,
) -> Result<(usize, usize)> {
    if !num_q_heads.is_multiple_of(tp.world_size) {
        bail!(
            "num_q_heads ({num_q_heads}) not divisible by world_size ({})",
            tp.world_size
        );
    }
    if !num_kv_heads.is_multiple_of(tp.world_size) {
        bail!(
            "num_kv_heads ({num_kv_heads}) not divisible by world_size ({})",
            tp.world_size
        );
    }
    Ok((num_q_heads / tp.world_size, num_kv_heads / tp.world_size))
}

// ============================================================================
// TpLinearConfig
// ============================================================================

/// Type of parallel linear layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelLinearKind {
    /// Split output dimension across TP ranks; all-reduce result.
    Column,
    /// Split input dimension across TP ranks; all-reduce result.
    Row,
}

/// Configuration for a tensor-parallel linear layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TpLinearConfig {
    pub kind: ParallelLinearKind,
    pub shard: ShardingSpec,
    /// Whether an all-reduce is needed after this layer (always true for both kinds,
    /// unless this is an intermediate result that will be combined in the next layer).
    pub needs_all_reduce: bool,
}

impl TpLinearConfig {
    /// Build config for a column-parallel linear layer.
    pub fn column(out_features: usize, tp: &TpConfig) -> Self {
        Self {
            kind: ParallelLinearKind::Column,
            shard: column_shard(out_features, tp),
            needs_all_reduce: true,
        }
    }

    /// Build config for a row-parallel linear layer.
    pub fn row(in_features: usize, tp: &TpConfig) -> Self {
        Self {
            kind: ParallelLinearKind::Row,
            shard: row_shard(in_features, tp),
            needs_all_reduce: true,
        }
    }
}

// ============================================================================
// NcclComm (GPU required)
// ============================================================================

/// NCCL communicator handle. GPU required.
///
/// On CPU builds this struct exists but all methods panic with
/// `todo!("GPU required: ...")`.
pub struct NcclComm {
    tp: TpConfig,
}

impl NcclComm {
    /// Create a new NCCL communicator for the given TP config.
    ///
    /// **GPU required** — panics on CPU builds.
    #[allow(unused_variables)]
    pub fn new(tp: TpConfig) -> Result<Self> {
        // GPU required: NCCL communicator initialization
        // In production: ncclCommInitRank / ncclGetUniqueId exchange via shared store
        #[cfg(not(feature = "cuda"))]
        todo!("GPU required: NCCL communicator initialization");
        #[cfg(feature = "cuda")]
        Ok(Self { tp })
    }

    /// All-reduce (sum) across all TP ranks.
    ///
    /// **GPU required**.
    #[allow(unused_variables)]
    pub fn all_reduce_sum_f16(&self, data: *mut u16, numel: usize) -> Result<()> {
        // GPU required: ncclAllReduce(data, data, numel, ncclFloat16, ncclSum, comm, stream)
        todo!("GPU required: NCCL all-reduce f16")
    }

    /// All-gather across all TP ranks, concatenating along dim 0.
    ///
    /// **GPU required**.
    #[allow(unused_variables)]
    pub fn all_gather_f16(
        &self,
        send: *const u16,
        recv: *mut u16,
        numel_per_rank: usize,
    ) -> Result<()> {
        // GPU required: ncclAllGather(send, recv, numel_per_rank, ncclFloat16, comm, stream)
        todo!("GPU required: NCCL all-gather f16")
    }

    pub fn tp(&self) -> &TpConfig {
        &self.tp
    }
}

// ============================================================================
// MultiAxisConfig + RankCoord (multi-GPU rank-layout math, port of SGLang
// `parallel_state.py::initialize_model_parallel`)
// ============================================================================

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultiAxisConfig {
    pub tp_size: usize,
    pub pp_size: usize,
    pub ep_size: usize,
    pub attn_dp_size: usize,
    pub attn_cp_size: usize,
    pub moe_dp_size: usize,
}

impl MultiAxisConfig {
    pub fn single() -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            ep_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        }
    }

    // SGLang parallel_state.py:1781,1827-1829,1897-1899
    pub fn validate(&self) -> Result<()> {
        if self.tp_size == 0
            || self.pp_size == 0
            || self.ep_size == 0
            || self.attn_dp_size == 0
            || self.attn_cp_size == 0
            || self.moe_dp_size == 0
        {
            bail!(
                "all axis sizes must be >= 1 (tp={}, pp={}, ep={}, attn_dp={}, attn_cp={}, moe_dp={})",
                self.tp_size,
                self.pp_size,
                self.ep_size,
                self.attn_dp_size,
                self.attn_cp_size,
                self.moe_dp_size,
            );
        }
        let attn_div = self.attn_dp_size * self.attn_cp_size;
        if !self.tp_size.is_multiple_of(attn_div) {
            bail!(
                "assert tp_size % (attn_dp_size * attn_cp_size) == 0 failed: tp={}, attn_dp={}, attn_cp={}",
                self.tp_size,
                self.attn_dp_size,
                self.attn_cp_size,
            );
        }
        let moe_div = self.ep_size * self.moe_dp_size;
        if !self.tp_size.is_multiple_of(moe_div) {
            bail!(
                "assert tp_size % (ep_size * moe_dp_size) == 0 failed: tp={}, ep={}, moe_dp={}",
                self.tp_size,
                self.ep_size,
                self.moe_dp_size,
            );
        }
        Ok(())
    }

    // SGLang parallel_state.py:1781
    pub fn world_size(&self) -> usize {
        self.tp_size * self.pp_size
    }

    // SGLang parallel_state.py:1829
    pub fn attn_tp_size(&self) -> usize {
        self.tp_size / self.attn_dp_size / self.attn_cp_size
    }

    // SGLang parallel_state.py:1899
    pub fn moe_tp_size(&self) -> usize {
        self.tp_size / self.ep_size / self.moe_dp_size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankCoord {
    pub world_rank: usize,
    pub tp_rank: usize,
    pub pp_rank: usize,
    pub attn_tp_rank: usize,
    pub attn_dp_rank: usize,
    pub attn_cp_rank: usize,
    pub moe_tp_rank: usize,
    pub moe_ep_rank: usize,
    pub moe_dp_rank: usize,
}

impl RankCoord {
    // SGLang dp_attention.py:240-254 + parallel_state.py:1789,1981
    pub fn from_world_rank(cfg: MultiAxisConfig, world_rank: usize) -> Result<Self> {
        cfg.validate()?;
        let world = cfg.world_size();
        if world_rank >= world {
            bail!("world_rank ({world_rank}) must be < world_size ({world})");
        }
        let tp_rank = world_rank % cfg.tp_size;
        let pp_rank = world_rank / cfg.tp_size;
        let attn_tp = cfg.attn_tp_size();
        let attn_tp_rank = tp_rank % attn_tp;
        let attn_cp_rank = (tp_rank / attn_tp) % cfg.attn_cp_size;
        let attn_dp_rank = tp_rank / (attn_tp * cfg.attn_cp_size);
        let moe_tp = cfg.moe_tp_size();
        let moe_tp_rank = tp_rank % moe_tp;
        let moe_ep_rank = (tp_rank / moe_tp) % cfg.ep_size;
        let moe_dp_rank = tp_rank / (moe_tp * cfg.ep_size);
        Ok(Self {
            world_rank,
            tp_rank,
            pp_rank,
            attn_tp_rank,
            attn_dp_rank,
            attn_cp_rank,
            moe_tp_rank,
            moe_ep_rank,
            moe_dp_rank,
        })
    }
}

// SGLang parallel_state.py:1789-1800
pub fn build_tp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    let world = cfg.world_size();
    let num_tp_groups = world / cfg.tp_size;
    let mut group_ranks = Vec::with_capacity(num_tp_groups);
    for tp_group_idx in 0..num_tp_groups {
        let st = tp_group_idx * cfg.tp_size;
        let en = (tp_group_idx + 1) * cfg.tp_size;
        group_ranks.push((st..en).collect());
    }
    group_ranks
}

// SGLang parallel_state.py:1981-1989
pub fn build_pp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    let world = cfg.world_size();
    let num_pp_groups = world / cfg.pp_size;
    let mut group_ranks = Vec::with_capacity(num_pp_groups);
    for pp_group_idx in 0..num_pp_groups {
        let ranks: Vec<usize> = (pp_group_idx..world).step_by(num_pp_groups).collect();
        group_ranks.push(ranks);
    }
    group_ranks
}

// SGLang parallel_state.py:1838-1853
pub fn build_attn_cp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    if cfg.attn_cp_size == cfg.tp_size {
        return build_tp_groups(cfg);
    }
    let world = cfg.world_size();
    let num_tp_groups = world / cfg.tp_size;
    let attn_tp = cfg.attn_tp_size();
    let mut group_ranks = Vec::new();
    for tp_group_idx in 0..num_tp_groups {
        for dp_idx in 0..cfg.attn_dp_size {
            for attn_tp_idx in 0..attn_tp {
                let st =
                    tp_group_idx * cfg.tp_size + dp_idx * attn_tp * cfg.attn_cp_size + attn_tp_idx;
                let en = tp_group_idx * cfg.tp_size
                    + (dp_idx + 1) * attn_tp * cfg.attn_cp_size
                    + attn_tp_idx;
                let ranks: Vec<usize> = (st..en).step_by(attn_tp).collect();
                group_ranks.push(ranks);
            }
        }
    }
    group_ranks
}

// SGLang parallel_state.py:1871-1883
pub fn build_attn_tp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    let attn_tp = cfg.attn_tp_size();
    if attn_tp == cfg.tp_size {
        return build_tp_groups(cfg);
    }
    let world = cfg.world_size();
    let num_tp_groups = world / cfg.tp_size;
    let mut group_ranks = Vec::new();
    for tp_group_idx in 0..num_tp_groups {
        for cp_dp_combined_idx in 0..(cfg.attn_cp_size * cfg.attn_dp_size) {
            let st = tp_group_idx * cfg.tp_size + cp_dp_combined_idx * attn_tp;
            let en = tp_group_idx * cfg.tp_size + (cp_dp_combined_idx + 1) * attn_tp;
            group_ranks.push((st..en).collect());
        }
    }
    group_ranks
}

// SGLang parallel_state.py:1838-1853 (attn_dp uses same outer layout as attn_cp;
// each attn_dp group is a stride across (attn_cp_size * attn_tp_size))
pub fn build_attn_dp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    if cfg.attn_dp_size == 1 {
        return (0..cfg.world_size()).map(|r| vec![r]).collect();
    }
    let world = cfg.world_size();
    let num_tp_groups = world / cfg.tp_size;
    let attn_tp = cfg.attn_tp_size();
    let stride = attn_tp * cfg.attn_cp_size;
    let mut group_ranks = Vec::new();
    for tp_group_idx in 0..num_tp_groups {
        for cp_idx in 0..cfg.attn_cp_size {
            for attn_tp_idx in 0..attn_tp {
                let st = tp_group_idx * cfg.tp_size + cp_idx * attn_tp + attn_tp_idx;
                let en = tp_group_idx * cfg.tp_size + cfg.tp_size;
                let ranks: Vec<usize> = (st..en).step_by(stride).collect();
                group_ranks.push(ranks);
            }
        }
    }
    group_ranks
}

// SGLang parallel_state.py:1903-1919
pub fn build_moe_dp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    if cfg.attn_cp_size > cfg.moe_dp_size {
        return build_attn_cp_groups(cfg);
    }
    if cfg.moe_dp_size == cfg.tp_size {
        return build_tp_groups(cfg);
    }
    let world = cfg.world_size();
    let num_tp_groups = world / cfg.tp_size;
    let moe_tp = cfg.moe_tp_size();
    let stride = moe_tp * cfg.ep_size;
    let mut group_ranks = Vec::new();
    for tp_group_idx in 0..num_tp_groups {
        for tp_ep_combined_idx in 0..(moe_tp * cfg.ep_size) {
            let st = tp_group_idx * cfg.tp_size + tp_ep_combined_idx;
            let en = (tp_group_idx + 1) * cfg.tp_size + tp_ep_combined_idx;
            let ranks: Vec<usize> = (st..en).step_by(stride).collect();
            group_ranks.push(ranks);
        }
    }
    group_ranks
}

// SGLang parallel_state.py:1929-1943
pub fn build_moe_ep_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    if cfg.ep_size == cfg.tp_size {
        return build_tp_groups(cfg);
    }
    let world = cfg.world_size();
    let num_tp_groups = world / cfg.tp_size;
    let moe_tp = cfg.moe_tp_size();
    let mut group_ranks = Vec::new();
    for tp_group_idx in 0..num_tp_groups {
        for moe_dp_idx in 0..cfg.moe_dp_size {
            for moe_tp_idx in 0..moe_tp {
                let st =
                    tp_group_idx * cfg.tp_size + moe_dp_idx * cfg.ep_size * moe_tp + moe_tp_idx;
                let en = st + cfg.ep_size * moe_tp;
                let ranks: Vec<usize> = (st..en).step_by(moe_tp).collect();
                group_ranks.push(ranks);
            }
        }
    }
    group_ranks
}

// SGLang parallel_state.py:1955-1970
pub fn build_moe_tp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    let moe_tp = cfg.moe_tp_size();
    if moe_tp == cfg.tp_size {
        return build_tp_groups(cfg);
    }
    let world = cfg.world_size();
    let num_tp_groups = world / cfg.tp_size;
    let mut group_ranks = Vec::new();
    for tp_group_idx in 0..num_tp_groups {
        for ep_dp_combined_idx in 0..(cfg.ep_size * cfg.moe_dp_size) {
            let st = tp_group_idx * cfg.tp_size + ep_dp_combined_idx * moe_tp;
            let en = tp_group_idx * cfg.tp_size + (ep_dp_combined_idx + 1) * moe_tp;
            group_ranks.push((st..en).collect());
        }
    }
    group_ranks
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------- TpConfig

    #[test]
    fn tp_config_single() {
        let tp = TpConfig::single();
        assert!(tp.is_single());
        tp.validate().unwrap();
    }

    #[test]
    fn tp_config_valid_multi() {
        let tp = TpConfig::new(4, 2).unwrap();
        assert!(!tp.is_single());
        assert_eq!(tp.world_size, 4);
        assert_eq!(tp.rank, 2);
    }

    #[test]
    fn tp_config_invalid_rank() {
        assert!(TpConfig::new(4, 4).is_err());
        assert!(TpConfig::new(0, 0).is_err());
    }

    // ---------------------------------------------------------------- column_shard

    #[test]
    fn column_shard_even_division() {
        let tp = TpConfig::new(4, 0).unwrap();
        let s = column_shard(16, &tp);
        assert_eq!(s.offset, 0);
        assert_eq!(s.size, 4);
        assert_eq!(s.total, 16);
        assert_eq!(s.end(), 4);

        let tp3 = TpConfig::new(4, 3).unwrap();
        let s3 = column_shard(16, &tp3);
        assert_eq!(s3.offset, 12);
        assert_eq!(s3.size, 4);
    }

    #[test]
    fn column_shard_with_remainder() {
        // 10 / 4: base=2, remainder=2; last rank gets 2+2=4
        let tp0 = TpConfig::new(4, 0).unwrap();
        let tp3 = TpConfig::new(4, 3).unwrap();
        let s0 = column_shard(10, &tp0);
        let s3 = column_shard(10, &tp3);
        assert_eq!(s0.size, 2);
        assert_eq!(s3.size, 4); // absorbs remainder
        // All shards together cover the full dimension
        let total_covered: usize = (0..4)
            .map(|r| column_shard(10, &TpConfig::new(4, r).unwrap()).size)
            .sum();
        assert_eq!(total_covered, 10);
    }

    #[test]
    fn column_shard_single_gpu() {
        let tp = TpConfig::single();
        let s = column_shard(1024, &tp);
        assert!(s.is_full());
        assert_eq!(s.offset, 0);
        assert_eq!(s.size, 1024);
    }

    // ---------------------------------------------------------------- row_shard (same formula)

    #[test]
    fn row_shard_matches_column_shard() {
        let tp = TpConfig::new(8, 3).unwrap();
        assert_eq!(row_shard(128, &tp), column_shard(128, &tp));
    }

    // ---------------------------------------------------------------- head_shard

    #[test]
    fn head_shard_gqa() {
        // Llama-70B: 64 Q heads, 8 KV heads, TP=8
        let tp = TpConfig::new(8, 0).unwrap();
        let (q, kv) = head_shard(64, 8, &tp).unwrap();
        assert_eq!(q, 8);
        assert_eq!(kv, 1);
    }

    #[test]
    fn head_shard_mha() {
        // Standard MHA: 32 Q == 32 KV, TP=4
        let tp = TpConfig::new(4, 2).unwrap();
        let (q, kv) = head_shard(32, 32, &tp).unwrap();
        assert_eq!(q, 8);
        assert_eq!(kv, 8);
    }

    #[test]
    fn head_shard_indivisible_kv() {
        // 7 KV heads not divisible by 4
        let tp = TpConfig::new(4, 0).unwrap();
        assert!(head_shard(32, 7, &tp).is_err());
    }

    // ---------------------------------------------------------------- TpLinearConfig

    #[test]
    fn tp_linear_config_column() {
        let tp = TpConfig::new(4, 1).unwrap();
        let cfg = TpLinearConfig::column(512, &tp);
        assert_eq!(cfg.kind, ParallelLinearKind::Column);
        assert_eq!(cfg.shard.offset, 128);
        assert_eq!(cfg.shard.size, 128);
        assert!(cfg.needs_all_reduce);
    }

    #[test]
    fn tp_linear_config_row() {
        let tp = TpConfig::new(2, 0).unwrap();
        let cfg = TpLinearConfig::row(4096, &tp);
        assert_eq!(cfg.kind, ParallelLinearKind::Row);
        assert_eq!(cfg.shard.offset, 0);
        assert_eq!(cfg.shard.size, 2048);
    }

    // ---------------------------------------------------------------- ShardingSpec helpers

    #[test]
    fn sharding_spec_range() {
        let s = ShardingSpec {
            offset: 8,
            size: 4,
            total: 16,
        };
        assert_eq!(s.end(), 12);
        assert_eq!(s.range(), 8..12);
        assert!(!s.is_full());
    }

    #[test]
    fn sharding_spec_full() {
        let s = ShardingSpec {
            offset: 0,
            size: 1024,
            total: 1024,
        };
        assert!(s.is_full());
    }

    // ---------------------------------------------------------------- MultiAxisConfig + RankCoord

    #[test]
    fn single_config_world_size_1() {
        let cfg = MultiAxisConfig::single();
        assert_eq!(cfg.world_size(), 1);
        assert_eq!(cfg.attn_tp_size(), 1);
        assert_eq!(cfg.moe_tp_size(), 1);
        cfg.validate().unwrap();
        let coord = RankCoord::from_world_rank(cfg, 0).unwrap();
        assert_eq!(
            coord,
            RankCoord {
                world_rank: 0,
                tp_rank: 0,
                pp_rank: 0,
                attn_tp_rank: 0,
                attn_dp_rank: 0,
                attn_cp_rank: 0,
                moe_tp_rank: 0,
                moe_ep_rank: 0,
                moe_dp_rank: 0,
            }
        );
    }

    // SGLang parallel_state.py:1749-1756
    #[test]
    fn tp2_pp4_groups_sglang_docstring_1749_1756() {
        let cfg = MultiAxisConfig {
            tp_size: 2,
            pp_size: 4,
            ep_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.world_size(), 8);
        assert_eq!(
            build_tp_groups(cfg),
            vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
        );
        assert_eq!(
            build_pp_groups(cfg),
            vec![vec![0, 2, 4, 6], vec![1, 3, 5, 7]],
        );
    }

    // SGLang parallel_state.py:1758-1769
    #[test]
    fn attn_cp2_attn_tp4_moe_dp2_moe_ep4_groups_sglang_docstring_1758_1769() {
        // Per docstring: tp=8, pp=1, attn_cp=2, attn_tp=4 (=> attn_dp=1),
        // moe_ep=4, moe_dp=2 (=> moe_tp=1).
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 4,
            attn_dp_size: 1,
            attn_cp_size: 2,
            moe_dp_size: 2,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.attn_tp_size(), 4);
        assert_eq!(cfg.moe_tp_size(), 1);
        assert_eq!(
            build_attn_tp_groups(cfg),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
        );
        assert_eq!(
            build_attn_cp_groups(cfg),
            vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]],
        );
        assert_eq!(
            build_moe_ep_groups(cfg),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
        );
        assert_eq!(
            build_moe_dp_groups(cfg),
            vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]],
        );
    }

    #[test]
    fn validate_rejects_world_size_mismatch() {
        // tp=3 not divisible by ep*moe_dp=4
        let cfg = MultiAxisConfig {
            tp_size: 3,
            pp_size: 1,
            ep_size: 2,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 2,
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("ep_size * moe_dp_size"), "got: {err}");
    }

    #[test]
    fn rank_coord_decomposition_round_trip() {
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 2,
            ep_size: 4,
            attn_dp_size: 1,
            attn_cp_size: 2,
            moe_dp_size: 2,
        };
        cfg.validate().unwrap();
        for world_rank in 0..cfg.world_size() {
            let coord = RankCoord::from_world_rank(cfg, world_rank).unwrap();
            assert_eq!(coord.world_rank, world_rank);
            // Reassembly from sub-ranks must give back tp_rank.
            let attn_tp = cfg.attn_tp_size();
            let reassembled_tp = (coord.attn_dp_rank * cfg.attn_cp_size + coord.attn_cp_rank)
                * attn_tp
                + coord.attn_tp_rank;
            assert_eq!(reassembled_tp, coord.tp_rank);
            let moe_tp = cfg.moe_tp_size();
            let reassembled_tp_moe =
                (coord.moe_dp_rank * cfg.ep_size + coord.moe_ep_rank) * moe_tp + coord.moe_tp_rank;
            assert_eq!(reassembled_tp_moe, coord.tp_rank);
            // pp_rank * tp_size + tp_rank == world_rank
            assert_eq!(coord.pp_rank * cfg.tp_size + coord.tp_rank, world_rank);
        }
    }

    // SGLang dp_attention.py:240-254
    #[test]
    fn dp_attention_math_matches_sglang() {
        // tp=8, dp=2, attn_cp=2 => attn_tp=2.
        // tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 1,
            attn_dp_size: 2,
            attn_cp_size: 2,
            moe_dp_size: 1,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.attn_tp_size(), 2);
        for tp_rank in 0..cfg.tp_size {
            let coord = RankCoord::from_world_rank(cfg, tp_rank).unwrap();
            assert_eq!(coord.attn_tp_rank, tp_rank % 2);
            assert_eq!(coord.attn_cp_rank, (tp_rank / 2) % 2);
            assert_eq!(coord.attn_dp_rank, tp_rank / (2 * 2));
        }
    }

    #[test]
    fn attn_tp_size_when_dp_off() {
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.attn_tp_size(), cfg.tp_size);
    }

    #[test]
    fn moe_tp_size_division() {
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 2,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 2,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.moe_tp_size(), 2);
    }

    #[test]
    fn from_world_rank_rejects_out_of_range() {
        let cfg = MultiAxisConfig::single();
        assert!(RankCoord::from_world_rank(cfg, 1).is_err());
    }

    #[test]
    fn build_pp_groups_single_tp() {
        // pp_size == world_size: each PP group is a single rank.
        let cfg = MultiAxisConfig {
            tp_size: 1,
            pp_size: 4,
            ep_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        cfg.validate().unwrap();
        assert_eq!(build_pp_groups(cfg), vec![vec![0, 1, 2, 3]]);
        assert_eq!(
            build_tp_groups(cfg),
            vec![vec![0], vec![1], vec![2], vec![3]],
        );
    }
}

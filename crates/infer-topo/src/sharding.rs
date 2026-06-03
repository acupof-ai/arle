//! Tensor-parallel rank placement + per-rank sharding math.
//!
//! Ported verbatim from `infer/src/tensor_parallel.rs`
//! ([`TpConfig`] L32-96, [`ShardingSpec`] L102-130, [`column_shard`] L143-163,
//! [`row_shard`] L169-171, [`head_shard`] L195-213,
//! [`ParallelLinearKind`]/[`TpLinearConfig`] L220-256). The `anyhow` error
//! handling is swapped for the crate-local std-only [`crate::TopoError`]; the
//! arithmetic and `bail!` message strings are unchanged.

use crate::error::{Result, bail};

// ============================================================================
// TpConfig
// ============================================================================

/// Tensor parallel configuration: one rank's placement in the TP group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TpConfig {
    /// Total number of TP ranks (GPUs in the tensor-parallel group).
    pub world_size: usize,
    /// This rank's index within the TP group (`0 <= rank < world_size`).
    pub rank: usize,
}

impl TpConfig {
    /// Single-GPU configuration (no parallelism).
    #[must_use]
    pub fn single() -> Self {
        Self {
            world_size: 1,
            rank: 0,
        }
    }

    /// Multi-GPU configuration.
    ///
    /// # Errors
    /// Errors if `world_size == 0` or `rank >= world_size`.
    pub fn new(world_size: usize, rank: usize) -> Result<Self> {
        if world_size == 0 {
            bail!("world_size must be >= 1");
        }
        if rank >= world_size {
            bail!("rank ({rank}) must be < world_size ({world_size})");
        }
        Ok(Self { world_size, rank })
    }

    /// True when running on a single GPU (no all-reduce needed).
    #[must_use]
    pub fn is_single(&self) -> bool {
        self.world_size == 1
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// Errors if `world_size == 0` or `rank >= world_size`.
    pub fn validate(&self) -> Result<()> {
        if self.world_size == 0 {
            bail!("world_size must be >= 1");
        }
        if self.rank >= self.world_size {
            bail!("rank {} >= world_size {}", self.rank, self.world_size);
        }
        Ok(())
    }

    /// Parse tensor-parallel rank placement from environment.
    ///
    /// Primary names match the serving binary; `ARLE_*` aliases keep the
    /// lower-level runtime scripts usable while DSv4 bring-up is still moving.
    ///
    /// # Errors
    /// Errors on a non-`usize` env value or an invalid `(world_size, rank)` pair.
    pub fn from_env() -> Result<Self> {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Result<Self> {
        let world_size = parse_parallel_env_usize("INFER_TP_SIZE", "ARLE_TP_SIZE", 1, &mut lookup)?;
        let rank = parse_parallel_env_usize("INFER_TP_RANK", "ARLE_TP_RANK", 0, &mut lookup)?;
        Self::new(world_size, rank)
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    #[must_use]
    pub fn end(&self) -> usize {
        self.offset + self.size
    }

    /// Return the range as a [`std::ops::Range`].
    #[must_use]
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.end()
    }

    /// True if this rank owns the entire dimension (single-GPU case).
    #[must_use]
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
#[must_use]
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
/// Identical formula to [`column_shard`] — differs only in semantic interpretation.
///
/// # Panics
/// Panics if `total < world_size` (cannot give each rank at least 1 element).
#[must_use]
pub fn row_shard(total: usize, tp: &TpConfig) -> ShardingSpec {
    column_shard(total, tp)
}

fn parse_parallel_env_usize(
    primary: &str,
    alias: &str,
    default: usize,
    lookup: &mut impl FnMut(&str) -> Option<String>,
) -> Result<usize> {
    let value = lookup(primary).or_else(|| lookup(alias));
    let Some(value) = value else {
        return Ok(default);
    };
    value.parse::<usize>().map_err(|err| {
        crate::error::TopoError::new(format!(
            "invalid {primary}/{alias} value `{value}`: expected usize: {err}"
        ))
    })
}

/// Compute the assignment of attention heads for this TP rank.
///
/// Returns `(num_q_heads_local, num_kv_heads_local)`.
///
/// # Errors
/// Returns an error if `num_q_heads` or `num_kv_heads` is not divisible by
/// `world_size` (GQA head assignment must be uniform across TP ranks).
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParallelLinearKind {
    /// Split output dimension across TP ranks; all-reduce result.
    Column,
    /// Split input dimension across TP ranks; all-reduce result.
    Row,
}

/// Configuration for a tensor-parallel linear layer.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TpLinearConfig {
    pub kind: ParallelLinearKind,
    pub shard: ShardingSpec,
    /// Whether an all-reduce is needed after this layer (always true for both
    /// kinds, unless this is an intermediate result combined in the next layer).
    pub needs_all_reduce: bool,
}

impl TpLinearConfig {
    /// Build config for a column-parallel linear layer.
    ///
    /// # Panics
    /// Panics if `out_features < tp.world_size` (via [`column_shard`]).
    #[must_use]
    pub fn column(out_features: usize, tp: &TpConfig) -> Self {
        Self {
            kind: ParallelLinearKind::Column,
            shard: column_shard(out_features, tp),
            needs_all_reduce: true,
        }
    }

    /// Build config for a row-parallel linear layer.
    ///
    /// # Panics
    /// Panics if `in_features < tp.world_size` (via [`row_shard`]).
    #[must_use]
    pub fn row(in_features: usize, tp: &TpConfig) -> Self {
        Self {
            kind: ParallelLinearKind::Row,
            shard: row_shard(in_features, tp),
            needs_all_reduce: true,
        }
    }
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

    #[test]
    fn tp_config_from_lookup_reads_primary_names() {
        let tp = TpConfig::from_lookup(|key| match key {
            "INFER_TP_SIZE" => Some("8".to_string()),
            "INFER_TP_RANK" => Some("3".to_string()),
            _ => None,
        })
        .unwrap();
        assert_eq!(tp, TpConfig::new(8, 3).unwrap());
    }

    #[test]
    fn tp_config_from_lookup_accepts_arle_aliases() {
        let tp = TpConfig::from_lookup(|key| match key {
            "ARLE_TP_SIZE" => Some("4".to_string()),
            "ARLE_TP_RANK" => Some("1".to_string()),
            _ => None,
        })
        .unwrap();
        assert_eq!(tp, TpConfig::new(4, 1).unwrap());
    }

    #[test]
    fn tp_config_from_lookup_rejects_non_usize() {
        let err = TpConfig::from_lookup(|key| match key {
            "INFER_TP_SIZE" => Some("not-a-number".to_string()),
            _ => None,
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected usize"), "got: {err}");
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

    // TP=8 target shape: [4096, 1024] column-parallel -> each rank [512, 1024].
    // (The `in` dim 1024 is unchanged for column-parallel; we shard `out`=4096.)
    #[test]
    fn column_shard_4096_across_tp8_target_shape() {
        // Each rank owns 512 of the 4096 output rows; offsets are contiguous and
        // reassembling all 8 shards reproduces the original dimension exactly.
        let mut covered = 0usize;
        let mut expected_offset = 0usize;
        for rank in 0..8 {
            let tp = TpConfig::new(8, rank).unwrap();
            let s = column_shard(4096, &tp);
            assert_eq!(s.size, 512, "rank {rank} local out-dim");
            assert_eq!(s.offset, expected_offset, "rank {rank} offset");
            assert_eq!(s.total, 4096);
            expected_offset += s.size;
            covered += s.size;
        }
        // Round-trip: contiguous, non-overlapping, exact cover of [0, 4096).
        assert_eq!(covered, 4096);
        assert_eq!(expected_offset, 4096);
    }

    // ---------------------------------------------------------------- row_shard (same formula)

    #[test]
    fn row_shard_matches_column_shard() {
        let tp = TpConfig::new(8, 3).unwrap();
        assert_eq!(row_shard(128, &tp), column_shard(128, &tp));
    }

    // TP=8 target shape: [4096, 1024] row-parallel -> shard the `in` dim 1024 ->
    // each rank owns [4096, 128] (here we model the sharded `in` dim = 1024).
    #[test]
    fn row_shard_1024_across_tp8_target_shape() {
        let mut covered = 0usize;
        let mut expected_offset = 0usize;
        for rank in 0..8 {
            let tp = TpConfig::new(8, rank).unwrap();
            let s = row_shard(1024, &tp);
            assert_eq!(s.size, 128, "rank {rank} local in-dim");
            assert_eq!(s.offset, expected_offset, "rank {rank} offset");
            assert_eq!(s.total, 1024);
            expected_offset += s.size;
            covered += s.size;
        }
        assert_eq!(covered, 1024);
        assert_eq!(expected_offset, 1024);
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

    // 32 Q heads + 8 GQA KV heads across TP=8 -> 4 Q heads/rank, 1 KV head/rank.
    #[test]
    fn head_shard_32q_8kv_tp8() {
        for rank in 0..8 {
            let tp = TpConfig::new(8, rank).unwrap();
            let (q, kv) = head_shard(32, 8, &tp).unwrap();
            assert_eq!(q, 4, "rank {rank} local Q heads");
            assert_eq!(
                kv, 1,
                "rank {rank} local KV heads (>=1, exactly replicated-uniform)"
            );
        }
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

    #[test]
    fn head_shard_indivisible_q() {
        // 30 Q heads not divisible by 8 — legacy rule errors (no padding).
        let tp = TpConfig::new(8, 0).unwrap();
        let err = head_shard(30, 8, &tp).unwrap_err().to_string();
        assert!(err.contains("num_q_heads"), "got: {err}");
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

    // ---------------------------------------------------------------- boundary: total < world_size

    #[test]
    #[should_panic(expected = "cannot shard")]
    fn column_shard_panics_when_total_below_world_size() {
        // Legacy behavior: `assert!` panics, it does NOT pad. We mirror it.
        let tp = TpConfig::new(8, 0).unwrap();
        let _ = column_shard(4, &tp);
    }
}

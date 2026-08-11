//! Tensor-parallel rank placement + per-rank sharding math.
//!
//! Ported from the legacy `infer/src/tensor_parallel.rs`; arithmetic and `bail!`
//! messages unchanged, `anyhow` swapped for the std-only [`crate::TopoError`].

use crate::error::{Result, bail};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TpConfig {
    pub world_size: usize,
    pub rank: usize,
}

impl TpConfig {
    #[must_use]
    pub fn single() -> Self {
        Self {
            world_size: 1,
            rank: 0,
        }
    }

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

    #[must_use]
    pub fn is_single(&self) -> bool {
        self.world_size == 1
    }

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

/// The rank owns `self.size` elements starting at `self.offset`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardingSpec {
    pub offset: usize,
    pub size: usize,
    pub total: usize,
}

impl ShardingSpec {
    #[must_use]
    pub fn end(&self) -> usize {
        self.offset + self.size
    }

    #[must_use]
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.end()
    }

    #[must_use]
    pub fn is_full(&self) -> bool {
        self.offset == 0 && self.size == self.total
    }
}

/// Column-parallel shard (output features split across TP ranks).
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

/// Row-parallel shard (input features split across TP ranks).
///
/// Identical formula to [`column_shard`] — differs only in semantic interpretation.
///
/// # Panics
/// Panics if `total < world_size` (cannot give each rank at least 1 element).
#[must_use]
pub fn row_shard(total: usize, tp: &TpConfig) -> ShardingSpec {
    column_shard(total, tp)
}

pub(crate) fn parse_parallel_env_usize(
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
/// Two regimes:
/// * **Shard** (`num_kv_heads >= world_size`, divisible): each rank owns
///   `num_kv_heads / world_size` distinct KV heads — the original, byte-identical
///   path (e.g. 64Q/8KV @ TP8 → 8Q/1KV; 32Q/32KV @ TP4 → 8Q/8KV).
/// * **Replicate** (`num_kv_heads < world_size`, `world_size % num_kv_heads == 0`):
///   too few KV heads to give each rank one, so each KV head is *replicated*
///   across `world_size / num_kv_heads` ranks. Every rank holds exactly ONE KV
///   head (`local_kv_heads == 1`) and its `num_q_heads / world_size` Q heads
///   (e.g. 32Q/2KV @ TP4 → 8Q/1KV, ranks 0,1 replicate KV head 0, ranks 2,3 KV
///   head 1). Replicated ranks load IDENTICAL K/V weights ([`kv_load_block_index`])
///   and write identical cache rows, so each computes correct GQA locally.
///
/// `num_q_heads` must always be divisible by `world_size` (Q is partitioned,
/// never replicated, in either regime).
///
/// # Errors
/// Returns an error if `num_q_heads` is not divisible by `world_size`, or if
/// `num_kv_heads` neither divides nor evenly-replicates across `world_size`.
pub fn head_shard(
    num_q_heads: usize,
    num_kv_heads: usize,
    tp: &TpConfig,
) -> Result<(usize, usize)> {
    let world = tp.world_size;
    if !num_q_heads.is_multiple_of(world) {
        bail!("num_q_heads ({num_q_heads}) not divisible by world_size ({world})");
    }
    let local_q = num_q_heads / world;
    if num_kv_heads.is_multiple_of(world) {
        return Ok((local_q, num_kv_heads / world));
    }
    if num_kv_heads != 0 && world.is_multiple_of(num_kv_heads) {
        return Ok((local_q, 1));
    }
    bail!(
        "num_kv_heads ({num_kv_heads}) neither divides world_size ({world}) nor \
         replicates evenly (world_size % num_kv_heads != 0)"
    )
}

/// Which global KV-head block index this rank loads from the K/V projection
/// weight (head-block-major). The loader multiplies this by `local_kv_heads *
/// head_dim` to get the row offset of its KV weight slice.
///
/// * **Shard**: returns `tp.rank` (byte-identical to the original
///   `rank * local_rows` offset — distinct heads per rank).
/// * **Replicate**: returns `tp.rank / replicas` where `replicas = world /
///   num_kv_heads`, so each replica group of consecutive ranks loads the SAME KV
///   head (the source of the replicated-K/V-identity invariant).
///
/// # Errors
/// Mirrors [`head_shard`]'s divisibility/replication contract.
pub fn kv_load_block_index(num_kv_heads: usize, tp: &TpConfig) -> Result<usize> {
    let world = tp.world_size;
    if num_kv_heads.is_multiple_of(world) {
        return Ok(tp.rank);
    }
    if num_kv_heads != 0 && world.is_multiple_of(num_kv_heads) {
        let replicas = world / num_kv_heads;
        return Ok(tp.rank / replicas);
    }
    bail!(
        "num_kv_heads ({num_kv_heads}) neither divides world_size ({world}) nor \
         replicates evenly (world_size % num_kv_heads != 0)"
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelLinearKind {
    /// Split output dimension across TP ranks; all-reduce result.
    Column,
    /// Split input dimension across TP ranks; all-reduce result.
    Row,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TpLinearConfig {
    pub kind: ParallelLinearKind,
    pub shard: ShardingSpec,
    /// Always true for both kinds, unless this is an intermediate result
    /// combined in the next layer.
    pub needs_all_reduce: bool,
}

impl TpLinearConfig {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(s3.size, 4);
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
    #[test]
    fn column_shard_4096_across_tp8_target_shape() {
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
        assert_eq!(covered, 4096);
        assert_eq!(expected_offset, 4096);
    }

    // ---------------------------------------------------------------- row_shard (same formula)

    #[test]
    fn row_shard_matches_column_shard() {
        let tp = TpConfig::new(8, 3).unwrap();
        assert_eq!(row_shard(128, &tp), column_shard(128, &tp));
    }

    // TP=8 target shape: [4096, 1024] row-parallel -> shard the `in` dim 1024.
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
            assert_eq!(kv, 1, "rank {rank} local KV heads");
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

    // ---------------------------------------------------------------- head_shard: KV replication

    // Qwen3.5-122B-A10B: 32 Q heads, 2 KV heads, TP=4. 2 KV heads < 4 ranks ->
    // replicate each KV head 4/2 = 2 ways. Every rank: 8 Q heads, 1 KV head.
    #[test]
    fn head_shard_122b_q32_kv2_tp4_replicates() {
        for rank in 0..4 {
            let tp = TpConfig::new(4, rank).unwrap();
            let (q, kv) = head_shard(32, 2, &tp).unwrap();
            assert_eq!(q, 8, "rank {rank} local Q heads");
            assert_eq!(kv, 1, "rank {rank} local KV heads");
            let block = kv_load_block_index(2, &tp).unwrap();
            assert_eq!(block, rank / 2, "rank {rank} replicated KV-head block");
        }
        assert_eq!(
            kv_load_block_index(2, &TpConfig::new(4, 0).unwrap()).unwrap(),
            kv_load_block_index(2, &TpConfig::new(4, 1).unwrap()).unwrap(),
        );
        assert_eq!(
            kv_load_block_index(2, &TpConfig::new(4, 2).unwrap()).unwrap(),
            kv_load_block_index(2, &TpConfig::new(4, 3).unwrap()).unwrap(),
        );
        assert_ne!(
            kv_load_block_index(2, &TpConfig::new(4, 0).unwrap()).unwrap(),
            kv_load_block_index(2, &TpConfig::new(4, 2).unwrap()).unwrap(),
        );
    }

    // Divisible case stays byte-identical: 64Q/8KV @ TP8 unchanged.
    #[test]
    fn head_shard_64q_8kv_tp8_shard_regime_unchanged() {
        for rank in 0..8 {
            let tp = TpConfig::new(8, rank).unwrap();
            let (q, kv) = head_shard(64, 8, &tp).unwrap();
            assert_eq!(q, 8, "rank {rank} local Q heads");
            assert_eq!(kv, 1, "rank {rank} local KV heads");
            assert_eq!(
                kv_load_block_index(8, &tp).unwrap(),
                rank,
                "rank {rank} shard-regime block index == rank"
            );
        }
    }

    // 1 KV head (full MQA) across TP=4 -> replicate the single head onto all 4 ranks.
    #[test]
    fn head_shard_mqa_q32_kv1_tp4_replicates_all() {
        for rank in 0..4 {
            let tp = TpConfig::new(4, rank).unwrap();
            let (q, kv) = head_shard(32, 1, &tp).unwrap();
            assert_eq!(q, 8);
            assert_eq!(kv, 1);
            assert_eq!(
                kv_load_block_index(1, &tp).unwrap(),
                0,
                "rank {rank} -> head 0"
            );
        }
    }

    // Replication requires world_size % num_kv_heads == 0.
    #[test]
    fn head_shard_kv3_tp4_rejects_unclean_replication() {
        let tp = TpConfig::new(4, 0).unwrap();
        let err = head_shard(32, 3, &tp).unwrap_err().to_string();
        assert!(err.contains("num_kv_heads"), "got: {err}");
        assert!(kv_load_block_index(3, &tp).is_err());
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
        let tp = TpConfig::new(8, 0).unwrap();
        let _ = column_shard(4, &tp);
    }
}

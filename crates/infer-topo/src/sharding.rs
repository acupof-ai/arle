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
    alias: Option<&str>,
    default: usize,
    lookup: &mut impl FnMut(&str) -> Option<String>,
) -> Result<usize> {
    let value = lookup(primary).or_else(|| alias.and_then(lookup));
    let Some(value) = value else {
        return Ok(default);
    };
    value.parse::<usize>().map_err(|err| {
        crate::error::TopoError::new(format!(
            "invalid {primary} value `{value}`: expected usize: {err}"
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

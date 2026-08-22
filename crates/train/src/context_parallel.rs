//! Context-parallel (CP) sequence sharding for OPD 256K writeback.
//!
//! CP splits the sequence dimension across N ranks (per-card activation
//! O(seq/N)); weights are replicated, so weight gradients are all-reduced
//! DP-style after backward. `CpContext` is a view over the one device mesh
//! (`infer_topo::MultiAxisConfig` / `RankCoord`), not a second source of truth.
//! `CpContext::single()` is the byte-identical single-card path.

use infer_topo::{MultiAxisConfig, RankCoord};

fn train_mesh(
    attn_dp: usize,
    attn_cp: usize,
    world_rank: usize,
) -> Option<(MultiAxisConfig, RankCoord)> {
    // PP/EP/MoE-DP are 1 until those axes land in training; cp/dp are the only
    // CLI-declared axes (--cp-size/--dp-size via ARLE_TRAIN_* env to workers).
    let cfg = MultiAxisConfig {
        tp_size: attn_dp.max(1) * attn_cp.max(1),
        pp_size: 1,
        ep_size: 1,
        attn_dp_size: attn_dp.max(1),
        attn_cp_size: attn_cp.max(1),
        moe_dp_size: 1,
    };
    RankCoord::from_world_rank(cfg, world_rank)
        .ok()
        .map(|coord| (cfg, coord))
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub fn world_rank(cp: CpContext, dp: DpContext) -> usize {
    dp.rank * cp.size + cp.rank
}

fn mesh_env() -> (usize, usize, usize) {
    (
        env_usize("ARLE_TRAIN_DP_SIZE", 1),
        env_usize("ARLE_TRAIN_CP_SIZE", 1),
        env_usize("ARLE_TRAIN_WORLD_RANK", 0),
    )
}

/// The `attn_cp` (sequence) axis of the mesh; mirrors `TpContext`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpContext {
    pub rank: usize,
    pub size: usize,
}

/// A sequence shard as an ordered list of contiguous chunks of the global
/// sequence. CP zigzag load-balancing is two chunks — chunk `r` and chunk
/// `2N-1-r` of a `2N`-way split, so every rank carries the same causal-attention
/// work (Megatron `get_batch_on_this_cp_rank`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqShard {
    chunks: Vec<(usize, usize)>,
}

impl SeqShard {
    pub fn contiguous(start: usize, end: usize) -> Self {
        Self {
            chunks: vec![(start, end)],
        }
    }

    pub fn len(&self) -> usize {
        self.chunks.iter().map(|&(s, e)| e - s).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn local_rows(&self) -> Vec<usize> {
        self.chunks.iter().flat_map(|&(s, e)| s..e).collect()
    }

    pub fn local_of(&self, pos: usize) -> Option<usize> {
        let mut base = 0;
        for &(s, e) in &self.chunks {
            if pos >= s && pos < e {
                return Some(base + (pos - s));
            }
            base += e - s;
        }
        None
    }
}

impl CpContext {
    pub const fn single() -> Self {
        Self { rank: 0, size: 1 }
    }

    pub const fn new(rank: usize, size: usize) -> Self {
        Self { rank, size }
    }

    pub fn from_env() -> Self {
        let (dp, cp, world_rank) = mesh_env();
        Self::from_mesh(dp, cp, world_rank)
    }

    pub fn from_mesh(attn_dp: usize, attn_cp: usize, world_rank: usize) -> Self {
        match train_mesh(attn_dp, attn_cp, world_rank) {
            Some((cfg, coord)) => Self {
                rank: coord.attn_cp_rank,
                size: cfg.attn_cp_size,
            },
            None => Self::single(),
        }
    }

    pub fn is_enabled(self) -> bool {
        self.size > 1
    }

    /// Pad rows go at the tail, past every real row, so causal attention never reaches them.
    pub fn padded_seq_len(self, seq_len: usize) -> usize {
        if self.size <= 1 {
            return seq_len;
        }
        let period = 2 * self.size;
        seq_len.div_ceil(period) * period
    }

    /// Zigzag: split the sequence into `2*size` equal chunks, own chunk `rank`
    /// and chunk `2*size-1-rank`. Under a causal mask the tail attends ~N× the
    /// keys the head does, so pairing front+back equalizes per-rank work.
    pub fn shard(self, seq_len: usize) -> SeqShard {
        if self.size <= 1 {
            return SeqShard::contiguous(0, seq_len);
        }
        let two_n = 2 * self.size;
        // assert (not debug_assert): a silent tail-drop would corrupt gradients invisibly.
        assert_eq!(
            seq_len % two_n,
            0,
            "CP zigzag requires seq_len % (2*cp_size) == 0; pad the sequence up"
        );
        let chunk = seq_len / two_n;
        let front = self.rank;
        let back = two_n - self.rank - 1;
        SeqShard {
            chunks: vec![
                (front * chunk, (front + 1) * chunk),
                (back * chunk, (back + 1) * chunk),
            ],
        }
    }
}

/// DP shards the BATCH (disjoint trajectories per rank) and replicates weights.
/// The correctness crux vs CP: under CP every rank shares ONE trajectory, so its
/// masked-target count is already global. Under DP each rank owns a DIFFERENT
/// trajectory, so the global-mean `inv_n` needs the SUM of per-rank counts over
/// the DP group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DpContext {
    pub rank: usize,
    pub size: usize,
}

impl DpContext {
    pub const fn single() -> Self {
        Self { rank: 0, size: 1 }
    }

    pub fn from_mesh(attn_dp: usize, attn_cp: usize, world_rank: usize) -> Self {
        match train_mesh(attn_dp, attn_cp, world_rank) {
            Some((cfg, coord)) => Self {
                rank: coord.attn_dp_rank,
                size: cfg.attn_dp_size,
            },
            None => Self::single(),
        }
    }

    pub fn from_env() -> Self {
        let (dp, cp, world_rank) = mesh_env();
        Self::from_mesh(dp, cp, world_rank)
    }

    pub fn is_enabled(self) -> bool {
        self.size > 1
    }
}

pub fn global_inv_n(dp_group_sum: usize) -> Option<f32> {
    (dp_group_sum > 0).then(|| 1.0 / dp_group_sum as f32)
}

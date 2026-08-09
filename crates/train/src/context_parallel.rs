//! Context-parallel (CP) sequence sharding for OPD 256K writeback.
//!
//! CP splits the sequence dimension across N ranks so per-card activation memory
//! is O(seq/N). Weights are REPLICATED (not sharded), so weight gradients are
//! all-reduced (DP-style) after backward. This module owns the pure host-side
//! shard arithmetic; the collectives live in `autograd::ops` and the launcher in
//! the CLI. `CpContext::single()` is the byte-identical single-card path.
//!
//! `CpContext` is a view over the one device mesh (`infer_topo::MultiAxisConfig`
//! / `RankCoord`) — the same mesh serving reads — not a second source of truth.

use infer_topo::{MultiAxisConfig, RankCoord};

fn train_mesh(
    attn_dp: usize,
    attn_cp: usize,
    world_rank: usize,
) -> Option<(MultiAxisConfig, RankCoord)> {
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

/// A rank's position in the context-parallel group — the `attn_cp` axis of the
/// mesh. Mirrors `TpContext`, but the axis is SEQUENCE.
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

    pub fn batch_shard(self, batch_len: usize) -> SeqShard {
        if self.size <= 1 {
            return SeqShard::contiguous(0, batch_len);
        }
        let range = crate::lora_shard::shard_range(batch_len, self.rank, self.size);
        SeqShard::contiguous(range.start, range.end)
    }
}

pub fn global_inv_n(dp_group_sum: usize) -> Option<f32> {
    (dp_group_sum > 0).then(|| 1.0 / dp_group_sum as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_is_whole_sequence() {
        let cp = CpContext::single();
        assert!(!cp.is_enabled());
        assert_eq!(cp.shard(1000), SeqShard::contiguous(0, 1000));
    }

    #[test]
    fn zigzag_covers_sequence_disjointly() {
        let size = 4;
        let seq = 32;
        let mut covered = vec![false; seq];
        for rank in 0..size {
            let s = CpContext::new(rank, size).shard(seq);
            assert_eq!(s.len(), 8, "each rank owns seq/size rows");
            for row in s.local_rows() {
                assert!(!covered[row], "position {row} covered twice");
                covered[row] = true;
            }
        }
        assert!(
            covered.iter().all(|&c| c),
            "every position covered exactly once"
        );
    }

    #[test]
    fn zigzag_pairs_front_and_back() {
        let (size, seq) = (2usize, 8usize);
        assert_eq!(
            CpContext::new(0, size).shard(seq).local_rows(),
            vec![0, 1, 6, 7]
        );
        assert_eq!(
            CpContext::new(1, size).shard(seq).local_rows(),
            vec![2, 3, 4, 5]
        );
    }

    #[test]
    fn local_targets_partition_by_owner() {
        let size = 2;
        let seq = 8;
        let positions = [1usize, 4, 7];
        let targets = vec![100u32, 200, 300];
        let local = |rank| {
            let shard = CpContext::new(rank, size).shard(seq);
            positions
                .iter()
                .zip(&targets)
                .filter_map(|(&p, &t)| shard.local_of(p).map(|l| (l, t)))
                .unzip::<_, _, Vec<usize>, Vec<u32>>()
        };

        let (p0, t0) = local(0);
        assert_eq!(p0, vec![1, 3]);
        assert_eq!(t0, vec![100, 300]);

        let (p1, t1) = local(1);
        assert_eq!(p1, vec![2]);
        assert_eq!(t1, vec![200]);

        assert_eq!(t0.len() + t1.len(), targets.len());
    }

    #[test]
    fn from_mesh_pure_cp_matches_legacy_rank_size() {
        for size in [1usize, 2, 4, 8] {
            for rank in 0..size {
                assert_eq!(
                    CpContext::from_mesh(1, size, rank),
                    CpContext { rank, size },
                    "pure CP size={size} rank={rank}"
                );
            }
        }
    }

    #[test]
    fn from_mesh_dp_times_cp_derives_cp_axis() {
        let (dp, cp) = (2usize, 2usize);
        for (world, cp_rank) in [(0, 0), (1, 1), (2, 0), (3, 1)] {
            let ctx = CpContext::from_mesh(dp, cp, world);
            assert_eq!(ctx.size, cp, "CP view size is the CP axis");
            assert_eq!(ctx.rank, cp_rank, "world {world} → cp_rank {cp_rank}");
        }
    }

    #[test]
    fn from_mesh_out_of_range_rank_falls_back_to_single() {
        assert_eq!(CpContext::from_mesh(1, 2, 5), CpContext::single());
    }

    #[test]
    fn dp_from_mesh_derives_dp_axis() {
        let (dp, cp) = (2usize, 2usize);
        for (world, dp_rank) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
            let ctx = DpContext::from_mesh(dp, cp, world);
            assert_eq!(ctx.size, dp, "DP view size is the DP axis");
            assert_eq!(ctx.rank, dp_rank, "world {world} -> dp_rank {dp_rank}");
        }
    }

    #[test]
    fn dp_batch_shard_partitions_disjointly() {
        let (size, batch) = (3usize, 10usize);
        let mut covered = vec![false; batch];
        for rank in 0..size {
            let s = DpContext::from_mesh(size, 1, rank).batch_shard(batch);
            for row in s.local_rows() {
                assert!(!covered[row], "overlap");
                covered[row] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "every trajectory covered once");
    }

    #[test]
    fn global_inv_n_uses_dp_group_sum() {
        assert_eq!(global_inv_n(10), Some(0.1));
        assert_eq!(global_inv_n(8), Some(0.125));
        assert_eq!(global_inv_n(0), None);
    }

    #[test]
    fn shard_union_reconstructs_single_card_targets() {
        let seq = 32;
        let size = 4;
        let positions: Vec<usize> = (0..seq).filter(|p| p % 3 == 0 || p % 7 == 0).collect();
        let targets: Vec<u32> = positions.iter().map(|&p| (p * 11 + 1) as u32).collect();

        let mut reconstructed: Vec<(usize, u32)> = Vec::new();
        for rank in 0..size {
            let shard = CpContext::new(rank, size).shard(seq);
            let rows = shard.local_rows();
            let (local_p, local_t): (Vec<usize>, Vec<u32>) = positions
                .iter()
                .zip(&targets)
                .filter_map(|(&p, &t)| shard.local_of(p).map(|l| (l, t)))
                .unzip();
            for (lp, t) in local_p.iter().zip(&local_t) {
                reconstructed.push((rows[*lp], *t));
            }
        }
        reconstructed.sort_unstable();
        let want: Vec<(usize, u32)> = positions.iter().copied().zip(targets).collect();
        assert_eq!(
            reconstructed, want,
            "shard union must equal single-card set"
        );
    }

    #[test]
    fn padded_shard_covers_every_target_when_seq_not_divisible() {
        let size = 2;
        let raw = 14;
        let padded = CpContext::new(0, size).padded_seq_len(raw);
        assert_eq!(padded, 16);
        for target in 3..=raw - 2 {
            let owners: Vec<usize> = (0..size)
                .filter(|&r| {
                    CpContext::new(r, size)
                        .shard(padded)
                        .local_of(target)
                        .is_some()
                })
                .collect();
            assert_eq!(
                owners.len(),
                1,
                "target {target} owned by {owners:?}, want exactly one"
            );
        }
    }

    #[test]
    #[should_panic(expected = "seq_len % (2*cp_size) == 0")]
    fn shard_panics_on_indivisible_seq_never_silently_drops() {
        let _ = CpContext::new(0, 2).shard(14);
    }
}

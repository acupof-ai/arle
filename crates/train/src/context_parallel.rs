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

/// This rank's train parallelism mesh from explicit axis sizes + world rank.
/// Pure (no env) so it unit-tests without racing the process-global env. Train
/// drives only attention-DP × attention-CP: in-layer TP is world=1 on the
/// writeback path, so `tp_size = dp*cp` places every rank on the attn sub-mesh
/// (`attn_tp_size` derives to 1). The launcher's world rank composes as
/// `dp_rank*cp + cp_rank` (CP inner, DP outer); pure CP gives `world_rank = cp_rank`.
/// `None` = misconfigured sizes/rank (caller falls back to single card).
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

/// A rank's position in the context-parallel group — the `attn_cp` axis of the
/// mesh. Mirrors `Qwen35TensorParallelConfig`, but the axis is SEQUENCE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpContext {
    pub rank: usize,
    pub size: usize,
}

/// A contiguous sequence shard: local rows `[start, end)` of the global sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqShard {
    pub start: usize,
    pub end: usize,
}

impl SeqShard {
    pub fn len(self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(self) -> bool {
        self.start >= self.end
    }

    pub fn contains(self, pos: usize) -> bool {
        pos >= self.start && pos < self.end
    }

    /// Filter `(global_position, target)` loss pairs to those this shard owns,
    /// rebased to shard-local row indices. A target at global position `p` lives on
    /// the shard containing `p`; its local row is `p - start`. The global target
    /// count (for `inv_n = 1/global_targets`) is the sum of per-shard lengths.
    pub fn local_targets(self, positions: &[usize], targets: &[u32]) -> (Vec<usize>, Vec<u32>) {
        positions
            .iter()
            .zip(targets)
            .filter(|&(&p, _)| self.contains(p))
            .map(|(&p, &t)| (p - self.start, t))
            .unzip()
    }
}

impl CpContext {
    pub const fn single() -> Self {
        Self { rank: 0, size: 1 }
    }

    pub const fn new(rank: usize, size: usize) -> Self {
        Self { rank, size }
    }

    /// Read the CP group from the launcher's env (`ARLE_TRAIN_CP_RANK` /
    /// `ARLE_TRAIN_CP_SIZE`), derived through the one mesh so `rank`/`size` are the
    /// mesh's `attn_cp_rank`/`attn_cp_size`, not a private second derivation.
    /// Defaults to `single()` when unset or misconfigured — the byte-identical
    /// single-card path — so callers can read it unconditionally.
    pub fn from_env() -> Self {
        let cp = env_usize("ARLE_TRAIN_CP_SIZE", 1);
        let world_rank = env_usize("ARLE_TRAIN_CP_RANK", 0);
        Self::from_mesh(1, cp, world_rank)
    }

    /// The CP view of the mesh for explicit axis sizes + world rank (pure, no env).
    /// Falls back to `single()` on a misconfigured mesh.
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

    /// This rank's sequence shard of a `seq_len`-long trajectory. Delegates to the
    /// canonical even-split-remainder-to-last-rank formula (`lora_shard::shard_range`
    /// → `infer_topo::column_shard`), so CP sequence shards line up with how base
    /// weights and LoRA deltas are split. Equal-length shards when `seq_len % size
    /// == 0` (an NCCL all-gather precondition); the remainder path keeps a single
    /// node correct, callers needing the ring pad to a multiple.
    pub fn shard(self, seq_len: usize) -> SeqShard {
        if self.size <= 1 {
            return SeqShard {
                start: 0,
                end: seq_len,
            };
        }
        let range = crate::lora_shard::shard_range(seq_len, self.rank, self.size);
        SeqShard {
            start: range.start,
            end: range.end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_is_whole_sequence() {
        let cp = CpContext::single();
        assert!(!cp.is_enabled());
        assert_eq!(
            cp.shard(1000),
            SeqShard {
                start: 0,
                end: 1000
            }
        );
    }

    #[test]
    fn even_split_covers_sequence_disjointly() {
        let size = 4;
        let seq = 32;
        let mut covered = vec![false; seq];
        for rank in 0..size {
            let s = CpContext::new(rank, size).shard(seq);
            assert_eq!(s.len(), 8);
            for slot in &mut covered[s.start..s.end] {
                assert!(!*slot);
                *slot = true;
            }
        }
        assert!(
            covered.iter().all(|&c| c),
            "every position covered exactly once"
        );
    }

    #[test]
    fn remainder_goes_to_last_rank() {
        let size = 3;
        let seq = 10; // base=3, last rank takes 3..10 (4 rows)
        assert_eq!(
            CpContext::new(0, size).shard(seq),
            SeqShard { start: 0, end: 3 }
        );
        assert_eq!(
            CpContext::new(1, size).shard(seq),
            SeqShard { start: 3, end: 6 }
        );
        assert_eq!(
            CpContext::new(2, size).shard(seq),
            SeqShard { start: 6, end: 10 }
        );
    }

    #[test]
    fn local_targets_partition_by_owner() {
        let size = 2;
        let seq = 8; // shards [0,4) and [4,8)
        // global targets at positions 1,4,7
        let positions = vec![1usize, 4, 7];
        let targets = vec![100u32, 200, 300];

        let (p0, t0) = CpContext::new(0, size)
            .shard(seq)
            .local_targets(&positions, &targets);
        assert_eq!(p0, vec![1]); // pos 1 → local row 1
        assert_eq!(t0, vec![100]);

        let (p1, t1) = CpContext::new(1, size)
            .shard(seq)
            .local_targets(&positions, &targets);
        assert_eq!(p1, vec![0, 3]); // pos 4→row 0, pos 7→row 3
        assert_eq!(t1, vec![200, 300]);

        // Global count = sum of local counts (the inv_n = 1/global_targets invariant).
        assert_eq!(t0.len() + t1.len(), targets.len());
    }

    // Pure-CP through the mesh must reproduce the pre-convergence {rank, size}
    // byte-identically: attn_cp_rank == world_rank and attn_cp_size == CP.
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

    // DP×CP: world rank composes CP-inner, DP-outer (world = dp_rank*cp + cp_rank).
    // The mesh must split it back into the right (attn_dp_rank, attn_cp_rank), and
    // the CP view must expose only the CP axis so DP replicas shard sequence
    // independently. This is the coordinate foundation the DP data-plane builds on.
    #[test]
    fn from_mesh_dp_times_cp_derives_cp_axis() {
        let (dp, cp) = (2usize, 2usize);
        // (world_rank, expected cp_rank)
        for (world, cp_rank) in [(0, 0), (1, 1), (2, 0), (3, 1)] {
            let ctx = CpContext::from_mesh(dp, cp, world);
            assert_eq!(ctx.size, cp, "CP view size is the CP axis");
            assert_eq!(ctx.rank, cp_rank, "world {world} → cp_rank {cp_rank}");
        }
    }

    // A misconfigured world rank (>= world_size) degrades to the safe single card,
    // never a panic or an out-of-range shard.
    #[test]
    fn from_mesh_out_of_range_rank_falls_back_to_single() {
        assert_eq!(CpContext::from_mesh(1, 2, 5), CpContext::single());
    }

    // The full parity invariant: the union of every CP shard's rebased targets
    // reconstructs the single-card target set exactly (no lost, dup, or misrebased
    // pair), and the counts sum to the global — a wrong split silently corrupts
    // inv_n and every gradient.
    #[test]
    fn shard_union_reconstructs_single_card_targets() {
        let seq = 30;
        let size = 4;
        let positions: Vec<usize> = (0..seq).filter(|p| p % 3 == 0 || p % 7 == 0).collect();
        let targets: Vec<u32> = positions.iter().map(|&p| (p * 11 + 1) as u32).collect();

        let mut reconstructed: Vec<(usize, u32)> = Vec::new();
        for rank in 0..size {
            let shard = CpContext::new(rank, size).shard(seq);
            let (local_p, local_t) = shard.local_targets(&positions, &targets);
            // Rebase local rows back to absolute and collect.
            for (lp, t) in local_p.iter().zip(&local_t) {
                reconstructed.push((shard.start + lp, *t));
            }
        }
        reconstructed.sort_unstable();
        let want: Vec<(usize, u32)> = positions.iter().copied().zip(targets).collect();
        assert_eq!(
            reconstructed, want,
            "shard union must equal single-card set"
        );
    }
}

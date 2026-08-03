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

/// `world_rank = dp_rank*cp + cp_rank` (CP inner) — the one composition site.
pub fn world_rank(cp: CpContext, dp: DpContext) -> usize {
    dp.rank * cp.size + cp.rank
}

/// The launcher's mesh env contract: `(dp_size, cp_size, world_rank)`, with
/// `world_rank = dp_rank*cp + cp_rank` (CP inner). Unset ⇒ single card.
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
/// sequence. Contiguous shards (DP batch, single-card CP) are one chunk; CP zigzag
/// load-balancing is two — chunk `r` and chunk `2N-1-r` of a `2N`-way split, so
/// every rank carries the same causal-attention work (Megatron
/// `get_batch_on_this_cp_rank`). Local row order is chunk order: chunk 0's rows,
/// then chunk 1's. `local_rows()` is the gather index into the global sequence;
/// `local_of()` maps a global position to its local row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqShard {
    /// Owned global ranges in local order. Each `(start, end)` is `[start, end)`.
    chunks: Vec<(usize, usize)>,
}

impl SeqShard {
    /// A single contiguous range `[start, end)` — DP batch shards and the
    /// single-card degenerate case.
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

    /// Global rows this rank owns, in local order — the gather index for slicing
    /// the embedded sequence and positions down to this shard.
    pub fn local_rows(&self) -> Vec<usize> {
        self.chunks.iter().flat_map(|&(s, e)| s..e).collect()
    }

    /// Global position → local row index, or `None` if this shard doesn't own it.
    /// Chunks concatenate in order, so the offset is the summed length of prior
    /// chunks plus the in-chunk offset. This is the shard's only membership query —
    /// callers filter loss targets by mapping through it (`opd.rs`).
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

    /// CP view of the launcher's mesh env; `single()` when unset or
    /// misconfigured, so callers read it unconditionally.
    pub fn from_env() -> Self {
        let (dp, cp, world_rank) = mesh_env();
        Self::from_mesh(dp, cp, world_rank)
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

    /// Sequence length rounded up to a multiple of `2*size` — the zigzag split
    /// precondition (`shard` panics otherwise). Identity when CP is off. Pad rows go
    /// at the tail, past every real row, so causal attention never reaches them.
    pub fn padded_seq_len(self, seq_len: usize) -> usize {
        if self.size <= 1 {
            return seq_len;
        }
        let period = 2 * self.size;
        seq_len.div_ceil(period) * period
    }

    /// This rank's zigzag load-balanced sequence shard: split the sequence into
    /// `2*size` equal chunks, own chunk `rank` and chunk `2*size-1-rank` (one from
    /// the front, one from the back). Under a causal mask the tail attends ~N× the
    /// keys the head does, so pairing a front and back chunk equalizes per-rank work
    /// and stops the ring stalling on the slowest rank (Megatron
    /// `get_batch_on_this_cp_rank`). Requires `seq_len % (2*size) == 0` — callers pad
    /// up. Single card is the whole sequence as one chunk (byte-identical).
    pub fn shard(self, seq_len: usize) -> SeqShard {
        if self.size <= 1 {
            return SeqShard::contiguous(0, seq_len);
        }
        let two_n = 2 * self.size;
        // Always checked (not debug_assert): a silent tail-drop when seq_len isn't a
        // multiple of 2*size would corrupt every rank's gradient invisibly. Callers
        // (opd.rs) pad up before sharding; this fires only on a broken contract.
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

/// A rank's position on the data-parallel axis (`attn_dp` of the mesh). DP shards
/// the BATCH (disjoint trajectories per rank) and replicates weights, so weight
/// grads all-reduce like CP's — the same collective, a different sharded axis.
///
/// The correctness crux vs CP: under CP every rank shares ONE trajectory, so its
/// masked-target count is already global. Under DP each rank owns a DIFFERENT
/// trajectory, so the global-mean `inv_n` needs the SUM of per-rank counts over
/// the DP group — otherwise every rank scales by its own local count and the mean
/// is wrong. `global_target_count` is that reduction (host arithmetic; the wire
/// all-reduce feeds it the summed count). This module owns the math; the launcher
/// + cross-rank reduce are the pending-remote data-plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DpContext {
    pub rank: usize,
    pub size: usize,
}

impl DpContext {
    pub const fn single() -> Self {
        Self { rank: 0, size: 1 }
    }

    /// The DP view of the mesh for explicit axis sizes + world rank (pure, no env).
    /// Falls back to `single()` on a misconfigured mesh.
    pub fn from_mesh(attn_dp: usize, attn_cp: usize, world_rank: usize) -> Self {
        match train_mesh(attn_dp, attn_cp, world_rank) {
            Some((cfg, coord)) => Self {
                rank: coord.attn_dp_rank,
                size: cfg.attn_dp_size,
            },
            None => Self::single(),
        }
    }

    /// DP view of the launcher's mesh env; `single()` when unset.
    pub fn from_env() -> Self {
        let (dp, cp, world_rank) = mesh_env();
        Self::from_mesh(dp, cp, world_rank)
    }

    pub fn is_enabled(self) -> bool {
        self.size > 1
    }

    /// This rank's disjoint slice of a `batch_len`-long trajectory list. DP shards
    /// the BATCH contiguously (even-split, remainder to last rank) — no zigzag: batch
    /// items are independent, so there's no causal imbalance to balance.
    pub fn batch_shard(self, batch_len: usize) -> SeqShard {
        if self.size <= 1 {
            return SeqShard::contiguous(0, batch_len);
        }
        let range = crate::lora_shard::shard_range(batch_len, self.rank, self.size);
        SeqShard::contiguous(range.start, range.end)
    }
}

/// Global masked-target count for the mean-CE `inv_n`, given this rank's local
/// count and the SUM of local counts already reduced over the DP group
/// (`dp_group_sum`; equals `local_count` when DP is off). CP shares one trajectory
/// so it contributes no extra factor — `dp_group_sum` is the global count directly.
/// Returns `None` when the count is zero (nothing to train), so the caller keeps
/// the fused op's local default.
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

    // Zigzag: size=4 splits seq=32 into 2N=8 chunks of 4; rank r owns chunk r and
    // chunk 2N-1-r. Every position covered exactly once, and each rank owns 2 chunks
    // (front + back) totalling seq/size rows.
    #[test]
    fn zigzag_covers_sequence_disjointly() {
        let size = 4;
        let seq = 32; // 2N=8 chunks of 4
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

    // Zigzag pairing for size=2, seq=8 (2N=4 chunks of 2): rank 0 owns chunks 0,3
    // (rows 0,1,6,7), rank 1 owns chunks 1,2 (rows 2,3,4,5) — front+back balance.
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
        let seq = 8; // rank0 owns rows {0,1,6,7}, rank1 owns {2,3,4,5}
        // global targets at positions 1,4,7
        let positions = [1usize, 4, 7];
        let targets = vec![100u32, 200, 300];
        // Mirror opd.rs's filter: keep targets this shard owns, rebased via local_of.
        let local = |rank| {
            let shard = CpContext::new(rank, size).shard(seq);
            positions
                .iter()
                .zip(&targets)
                .filter_map(|(&p, &t)| shard.local_of(p).map(|l| (l, t)))
                .unzip::<_, _, Vec<usize>, Vec<u32>>()
        };

        let (p0, t0) = local(0);
        // rank0 local order [0,1,6,7]: pos1→local 1, pos7→local 3.
        assert_eq!(p0, vec![1, 3]);
        assert_eq!(t0, vec![100, 300]);

        let (p1, t1) = local(1);
        // rank1 local order [2,3,4,5]: pos4→local 2.
        assert_eq!(p1, vec![2]);
        assert_eq!(t1, vec![200]);

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

    // DP view of the mesh: world rank composes CP-inner, DP-outer, so the DP axis
    // is world_rank / cp. Mirror of the CP-axis test — one mesh, two views.
    #[test]
    fn dp_from_mesh_derives_dp_axis() {
        let (dp, cp) = (2usize, 2usize);
        // (world_rank, expected dp_rank)
        for (world, dp_rank) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
            let ctx = DpContext::from_mesh(dp, cp, world);
            assert_eq!(ctx.size, dp, "DP view size is the DP axis");
            assert_eq!(ctx.rank, dp_rank, "world {world} -> dp_rank {dp_rank}");
        }
    }

    // DP batch shards tile the trajectory list disjointly, remainder to last rank
    // (contiguous — DP does not zigzag).
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

    // The global-mean inv_n: DP sums per-rank counts (each rank a different
    // trajectory) into the global count; CP shares one trajectory so its count is
    // already global. A zero count yields None (nothing to train).
    #[test]
    fn global_inv_n_uses_dp_group_sum() {
        // DP=2 with local counts 4 and 6 -> global 10 -> inv_n 0.1.
        assert_eq!(global_inv_n(10), Some(0.1));
        // Single card / CP: dp_group_sum == the one trajectory's count.
        assert_eq!(global_inv_n(8), Some(0.125));
        assert_eq!(global_inv_n(0), None);
    }

    // The full parity invariant: the union of every CP shard's rebased targets
    // reconstructs the single-card target set exactly (no lost, dup, or misrebased
    // pair), and the counts sum to the global — a wrong split silently corrupts
    // inv_n and every gradient. Zigzag: seq divisible by 2*size.
    #[test]
    fn shard_union_reconstructs_single_card_targets() {
        let seq = 32; // 2*size=8 divides 32
        let size = 4;
        let positions: Vec<usize> = (0..seq).filter(|p| p % 3 == 0 || p % 7 == 0).collect();
        let targets: Vec<u32> = positions.iter().map(|&p| (p * 11 + 1) as u32).collect();

        let mut reconstructed: Vec<(usize, u32)> = Vec::new();
        for rank in 0..size {
            let shard = CpContext::new(rank, size).shard(seq);
            let rows = shard.local_rows();
            // Filter+rebase as opd.rs does: map each target through local_of.
            let (local_p, local_t): (Vec<usize>, Vec<u32>) = positions
                .iter()
                .zip(&targets)
                .filter_map(|(&p, &t)| shard.local_of(p).map(|l| (l, t)))
                .unzip();
            // Rebase local rows back to absolute (via the gather index) and collect.
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

    // Regression: seq=14, cp=2 is NOT a multiple of 2*size=4, so the raw zigzag
    // dropped the tail rows 12,13 — position 12 is a loss target, so CP loss came out
    // ~10% low (pod baseline FAIL). After padded_seq_len(14)->16 every real target
    // position is owned by exactly one shard.
    #[test]
    fn padded_shard_covers_every_target_when_seq_not_divisible() {
        let size = 2;
        let raw = 14;
        let padded = CpContext::new(0, size).padded_seq_len(raw);
        assert_eq!(padded, 16);
        // Targets predict positions prompt_len-1..=raw-2 (prompt 4): 3..=12.
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
        let _ = CpContext::new(0, 2).shard(14); // 14 % 4 != 0
    }
}

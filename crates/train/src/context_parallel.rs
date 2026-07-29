//! Context-parallel (CP) sequence sharding for OPD 256K writeback.
//!
//! CP splits the sequence dimension across N ranks so per-card activation memory
//! is O(seq/N). Weights are REPLICATED (not sharded), so weight gradients are
//! all-reduced (DP-style) after backward. This module owns the pure host-side
//! shard arithmetic; the collectives live in `autograd::ops` and the launcher in
//! the CLI. `CpContext::single()` is the byte-identical single-card path.

/// A rank's position in the context-parallel group. Mirrors
/// `Qwen35TensorParallelConfig`, but the axis is SEQUENCE, not heads/hidden.
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
    /// `ARLE_TRAIN_CP_SIZE`). Defaults to `single()` when unset — the byte-identical
    /// single-card path — so callers can read it unconditionally.
    pub fn from_env() -> Self {
        let rank = std::env::var("ARLE_TRAIN_CP_RANK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let size = std::env::var("ARLE_TRAIN_CP_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        Self { rank, size }
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
        let seq = 32; // 32 % 4 == 0 → equal shards of 8
        let mut covered = vec![false; seq];
        for rank in 0..size {
            let s = CpContext::new(rank, size).shard(seq);
            assert_eq!(s.len(), 8);
            for p in s.start..s.end {
                assert!(!covered[p], "position {p} double-covered");
                covered[p] = true;
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
}

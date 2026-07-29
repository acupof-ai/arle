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
}

impl CpContext {
    pub const fn single() -> Self {
        Self { rank: 0, size: 1 }
    }

    pub const fn new(rank: usize, size: usize) -> Self {
        Self { rank, size }
    }

    pub fn is_enabled(self) -> bool {
        self.size > 1
    }

    /// This rank's sequence shard of a `seq_len`-long trajectory. Even split when
    /// `seq_len % size == 0`; otherwise the remainder rows go to the LAST rank
    /// (equal-length shards are an NCCL all-gather precondition, so callers that
    /// need the ring should pad to a multiple — the remainder path keeps a
    /// single-node run correct meanwhile).
    pub fn shard(self, seq_len: usize) -> SeqShard {
        if self.size <= 1 {
            return SeqShard {
                start: 0,
                end: seq_len,
            };
        }
        let base = seq_len / self.size;
        let start = self.rank * base;
        let end = if self.rank + 1 == self.size {
            seq_len
        } else {
            start + base
        };
        SeqShard { start, end }
    }

    /// Filter `(global_position, target)` loss pairs to those the local shard owns,
    /// rebased to shard-local row indices. A target at global position `p` lives on
    /// the rank whose shard contains `p`; its local row is `p - shard.start`.
    pub fn local_targets(
        self,
        shard: SeqShard,
        positions: &[usize],
        targets: &[u32],
    ) -> (Vec<usize>, Vec<u32>) {
        positions
            .iter()
            .zip(targets)
            .filter(|&(&p, _)| shard.contains(p))
            .map(|(&p, &t)| (p - shard.start, t))
            .unzip()
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

        let (p0, t0) = CpContext::new(0, size).local_targets(
            CpContext::new(0, size).shard(seq),
            &positions,
            &targets,
        );
        assert_eq!(p0, vec![1]); // pos 1 → local row 1
        assert_eq!(t0, vec![100]);

        let (p1, t1) = CpContext::new(1, size).local_targets(
            CpContext::new(1, size).shard(seq),
            &positions,
            &targets,
        );
        assert_eq!(p1, vec![0, 3]); // pos 4→row 0, pos 7→row 3
        assert_eq!(t1, vec![200, 300]);

        // Global count = sum of local counts (the inv_n = 1/global_targets invariant).
        assert_eq!(t0.len() + t1.len(), targets.len());
    }
}

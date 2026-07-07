//! Streaming replay buffer for the in-process actor-learner RL loop (WS1).
//!
//! Bounded, depth-limited store of accepted trajectories. A producer pushes as
//! each rollout completes (evicting the oldest at `cap`); a learner drains from
//! the front, dropping entries staler than `max_staleness` LoRA epochs.

use std::collections::VecDeque;

/// One accepted trajectory — verl-style masked-CE record plus the LoRA epoch it
/// was rolled out under. `mask[i] == 1` marks an LLM-generated response token,
/// `0` a tool/env token (mirrors `agent_opd.rs:399`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trajectory {
    pub prompt: Vec<u32>,
    pub resp: Vec<u32>,
    pub mask: Vec<u8>,
    pub rollout_lora_epoch: u64,
}

/// FIFO ring of trajectories with capacity eviction and depth-bound staleness.
pub struct ReplayBuffer {
    deque: VecDeque<Trajectory>,
    cap: usize,
    max_staleness: u64,
}

impl ReplayBuffer {
    pub fn new(cap: usize, max_staleness: u64) -> Self {
        Self {
            deque: VecDeque::with_capacity(cap),
            cap,
            max_staleness,
        }
    }

    /// Push, evicting the oldest entry first when already at `cap`.
    pub fn push(&mut self, t: Trajectory) {
        if self.deque.len() >= self.cap {
            self.deque.pop_front();
        }
        self.deque.push_back(t);
    }

    /// Drain up to `n` from the front, DROPPING any entry staler than
    /// `max_staleness` (by `current_epoch - rollout_lora_epoch`); returns only
    /// the fresh-enough ones, order preserved.
    pub fn pop_batch(&mut self, n: usize, current_epoch: u64) -> Vec<Trajectory> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let Some(t) = self.deque.pop_front() else {
                break;
            };
            if current_epoch.saturating_sub(t.rollout_lora_epoch) <= self.max_staleness {
                out.push(t);
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.deque.len()
    }

    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traj(epoch: u64) -> Trajectory {
        Trajectory {
            prompt: vec![epoch as u32],
            resp: vec![epoch as u32],
            mask: vec![1],
            rollout_lora_epoch: epoch,
        }
    }

    #[test]
    fn push_beyond_cap_evicts_oldest() {
        let mut buf = ReplayBuffer::new(2, 100);
        buf.push(traj(0));
        buf.push(traj(1));
        buf.push(traj(2));
        assert_eq!(buf.len(), 2);
        // Oldest (epoch 0) gone; front is now epoch 1.
        let out = buf.pop_batch(10, 2);
        assert_eq!(
            out.iter().map(|t| t.rollout_lora_epoch).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn pop_batch_drops_stale_keeps_fresh_in_order() {
        let mut buf = ReplayBuffer::new(8, 1);
        for e in [0u64, 2, 3, 4] {
            buf.push(traj(e));
        }
        // current_epoch=4, max_staleness=1 → keep epoch >= 3, drop 0 and 2.
        let out = buf.pop_batch(10, 4);
        assert_eq!(
            out.iter().map(|t| t.rollout_lora_epoch).collect::<Vec<_>>(),
            vec![3, 4]
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn pop_batch_respects_n_and_fifo_order() {
        let mut buf = ReplayBuffer::new(8, 100);
        for e in 0..5u64 {
            buf.push(traj(e));
        }
        let out = buf.pop_batch(3, 4);
        assert_eq!(
            out.iter().map(|t| t.rollout_lora_epoch).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(buf.len(), 2);
    }
}

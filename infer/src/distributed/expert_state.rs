//! Expert-parallel placement metadata for sparse MoE models.
//!
//! F4 keeps this as CPU-verifiable routing state. Real all-to-all dispatch and
//! expert execution are later F5+ work.

use anyhow::{Result, bail};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertGroup {
    pub rank: usize,
    pub world_size: usize,
    pub num_experts: usize,
    pub experts_per_rank: usize,
    expert_to_rank: Vec<usize>,
}

impl ExpertGroup {
    pub fn new(rank: usize, world_size: usize, num_experts: usize) -> Result<Self> {
        if world_size == 0 {
            bail!("expert world_size must be >= 1");
        }
        if rank >= world_size {
            bail!("expert rank {rank} must be < world_size {world_size}");
        }
        if num_experts == 0 {
            bail!("num_experts must be >= 1");
        }
        if !num_experts.is_multiple_of(world_size) {
            bail!("num_experts {num_experts} must be divisible by world_size {world_size}");
        }

        let experts_per_rank = num_experts / world_size;
        let expert_to_rank = (0..num_experts)
            .map(|expert_idx| expert_idx / experts_per_rank)
            .collect();

        Ok(Self {
            rank,
            world_size,
            num_experts,
            experts_per_rank,
            expert_to_rank,
        })
    }

    pub fn rank_for_expert(&self, expert_idx: usize) -> Option<usize> {
        self.expert_to_rank.get(expert_idx).copied()
    }

    pub fn owns_expert(&self, expert_idx: usize) -> bool {
        self.rank_for_expert(expert_idx) == Some(self.rank)
    }

    pub fn local_expert_range(&self) -> std::ops::Range<usize> {
        let start = self.rank * self.experts_per_rank;
        start..start + self.experts_per_rank
    }

    pub fn expert_to_rank_map(&self) -> &[usize] {
        &self.expert_to_rank
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertRoute {
    pub token_idx: usize,
    pub expert_idx: usize,
    pub weight: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertRoutingWeights {
    pub num_experts: usize,
    pub routes: Vec<ExpertRoute>,
}

impl ExpertRoutingWeights {
    pub fn new(num_experts: usize, routes: impl Into<Vec<ExpertRoute>>) -> Self {
        Self {
            num_experts,
            routes: routes.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertOutput {
    pub rank: usize,
    pub expert_idx: usize,
    pub token_indices: Vec<usize>,
    pub hidden_states: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_group_maps_contiguous_expert_ranges() {
        let rank0 = ExpertGroup::new(0, 2, 8).unwrap();
        let rank1 = ExpertGroup::new(1, 2, 8).unwrap();

        assert_eq!(rank0.experts_per_rank, 4);
        assert_eq!(rank0.local_expert_range(), 0..4);
        assert_eq!(rank1.local_expert_range(), 4..8);
        assert_eq!(rank0.expert_to_rank_map(), &[0, 0, 0, 0, 1, 1, 1, 1]);
        assert!(rank0.owns_expert(3));
        assert!(!rank0.owns_expert(4));
        assert!(rank1.owns_expert(7));
    }

    #[test]
    fn expert_group_rejects_invalid_layouts() {
        assert!(ExpertGroup::new(0, 0, 8).is_err());
        assert!(ExpertGroup::new(2, 2, 8).is_err());
        assert!(ExpertGroup::new(0, 2, 0).is_err());
        assert!(ExpertGroup::new(0, 3, 8).is_err());
    }
}

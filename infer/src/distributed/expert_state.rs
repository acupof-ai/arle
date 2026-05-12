//! Expert-parallel placement metadata for sparse MoE models.
//!
//! This module owns the CPU-visible EP contract used by DeepSeek-style routed
//! MoE layers: contiguous global expert ownership, global-to-local expert
//! remapping, and the TileKernels-compatible TP/EP masking formula for fused
//! expert dispatch.

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

    pub fn local_expert_idx(&self, expert_idx: usize) -> Option<usize> {
        if !self.owns_expert(expert_idx) {
            return None;
        }
        Some(expert_idx - self.local_expert_range().start)
    }

    pub fn global_expert_idx(&self, local_expert_idx: usize) -> Option<usize> {
        if local_expert_idx >= self.experts_per_rank {
            return None;
        }
        Some(self.local_expert_range().start + local_expert_idx)
    }

    /// Return a tensor-shaped mask/remap for global expert ids.
    ///
    /// Non-local entries become `-1`; local entries are compacted to
    /// `[0, experts_per_rank)`. Negative input values are preserved as masked
    /// entries. Out-of-range non-negative expert ids are rejected because they
    /// indicate a router bug on the host side.
    pub fn mask_indices_to_local(&self, indices: &[i64]) -> Result<Vec<i64>> {
        indices
            .iter()
            .map(|&expert_idx| {
                if expert_idx < 0 {
                    return Ok(-1);
                }
                let expert_idx = usize::try_from(expert_idx)
                    .map_err(|_| anyhow::anyhow!("expert index does not fit usize"))?;
                if expert_idx >= self.num_experts {
                    bail!(
                        "expert index {expert_idx} out of range for num_experts {}",
                        self.num_experts
                    );
                }
                Ok(self
                    .local_expert_idx(expert_idx)
                    .map_or(-1, |local| local as i64))
            })
            .collect()
    }

    pub fn localize_routing(&self, routing: &ExpertRoutingWeights) -> Result<LocalExpertRouting> {
        if routing.num_experts != self.num_experts {
            bail!(
                "routing num_experts {} does not match EP group num_experts {}",
                routing.num_experts,
                self.num_experts
            );
        }
        let mut routes = Vec::new();
        for route in &routing.routes {
            if route.expert_idx >= self.num_experts {
                bail!(
                    "route expert {} out of range for num_experts {}",
                    route.expert_idx,
                    self.num_experts
                );
            }
            let Some(local_expert_idx) = self.local_expert_idx(route.expert_idx) else {
                continue;
            };
            routes.push(LocalExpertRoute {
                token_idx: route.token_idx,
                global_expert_idx: route.expert_idx,
                local_expert_idx,
                weight: route.weight,
            });
        }
        Ok(LocalExpertRouting {
            num_global_experts: self.num_experts,
            experts_per_rank: self.experts_per_rank,
            routes,
        })
    }

    pub fn from_env(num_experts: usize) -> Result<Self> {
        let world_size = parse_parallel_env_usize("INFER_EP_SIZE", "ARLE_EP_SIZE", 1)?;
        let rank = parse_parallel_env_usize("INFER_EP_RANK", "ARLE_EP_RANK", 0)?;
        Self::new(rank, world_size, num_experts)
    }

    pub fn expert_to_rank_map(&self) -> &[usize] {
        &self.expert_to_rank
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalExpertRoute {
    pub token_idx: usize,
    pub global_expert_idx: usize,
    pub local_expert_idx: usize,
    pub weight: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalExpertRouting {
    pub num_global_experts: usize,
    pub experts_per_rank: usize,
    pub routes: Vec<LocalExpertRoute>,
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

/// CPU reference for DeepSeek TileKernels' `mask_indices_by_tp` formula.
///
/// The upstream TileLang kernel keeps only experts assigned to `tp_rank`, then
/// compacts global expert ids by removing gaps introduced by other TP ranks.
/// This is the host-side truth function used by ARLE tests and by future AOT
/// wrappers.
pub fn tilekernels_mask_indices_by_tp_layout(
    indices: &[i64],
    num_experts: usize,
    num_ep_ranks: usize,
    tp_rank: usize,
    num_tp_ranks: usize,
) -> Result<Vec<i64>> {
    if num_experts == 0 {
        bail!("num_experts must be >= 1");
    }
    if num_ep_ranks == 0 {
        bail!("num_ep_ranks must be >= 1");
    }
    if num_tp_ranks == 0 {
        bail!("num_tp_ranks must be >= 1");
    }
    if tp_rank >= num_tp_ranks {
        bail!("tp_rank {tp_rank} must be < num_tp_ranks {num_tp_ranks}");
    }
    if !num_experts.is_multiple_of(num_ep_ranks) {
        bail!("num_experts {num_experts} must be divisible by num_ep_ranks {num_ep_ranks}");
    }

    let per_gpu = num_experts / num_ep_ranks;
    let per_dp = num_tp_ranks * per_gpu;
    indices
        .iter()
        .map(|&expert_idx| {
            if expert_idx < 0 {
                return Ok(-1);
            }
            let expert_idx = usize::try_from(expert_idx)
                .map_err(|_| anyhow::anyhow!("expert index does not fit usize"))?;
            if expert_idx >= num_experts {
                bail!("expert index {expert_idx} out of range for num_experts {num_experts}");
            }
            if (expert_idx / per_gpu) % num_tp_ranks != tp_rank {
                return Ok(-1);
            }
            let value = expert_idx - tp_rank * per_gpu;
            let dp_rank = value / per_dp;
            let local = value.saturating_sub(dp_rank * (per_dp - per_gpu));
            Ok(local as i64)
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertOutput {
    pub rank: usize,
    pub expert_idx: usize,
    pub token_indices: Vec<usize>,
    pub hidden_states: Vec<Vec<f32>>,
}

fn parse_parallel_env_usize(primary: &str, alias: &str, default: usize) -> Result<usize> {
    let value = std::env::var(primary)
        .ok()
        .or_else(|| std::env::var(alias).ok());
    let Some(value) = value else {
        return Ok(default);
    };
    value.parse::<usize>().map_err(|err| {
        anyhow::anyhow!("invalid {primary}/{alias} value `{value}`: expected usize: {err}")
    })
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
        assert_eq!(rank0.local_expert_idx(3), Some(3));
        assert_eq!(rank1.local_expert_idx(4), Some(0));
        assert_eq!(rank1.global_expert_idx(3), Some(7));
    }

    #[test]
    fn expert_group_rejects_invalid_layouts() {
        assert!(ExpertGroup::new(0, 0, 8).is_err());
        assert!(ExpertGroup::new(2, 2, 8).is_err());
        assert!(ExpertGroup::new(0, 2, 0).is_err());
        assert!(ExpertGroup::new(0, 3, 8).is_err());
    }

    #[test]
    fn expert_group_masks_global_indices_to_local() {
        let rank1 = ExpertGroup::new(1, 2, 8).unwrap();

        assert_eq!(
            rank1.mask_indices_to_local(&[0, 3, 4, 7, -1]).unwrap(),
            vec![-1, -1, 0, 3, -1]
        );
        assert!(rank1.mask_indices_to_local(&[8]).is_err());
    }

    #[test]
    fn expert_group_localizes_routes_for_owned_experts() {
        let rank1 = ExpertGroup::new(1, 2, 8).unwrap();
        let routing = ExpertRoutingWeights::new(
            8,
            vec![
                ExpertRoute {
                    token_idx: 0,
                    expert_idx: 1,
                    weight: 0.25,
                },
                ExpertRoute {
                    token_idx: 0,
                    expert_idx: 5,
                    weight: 0.75,
                },
            ],
        );

        let local = rank1.localize_routing(&routing).unwrap();

        assert_eq!(local.num_global_experts, 8);
        assert_eq!(local.experts_per_rank, 4);
        assert_eq!(
            local.routes,
            vec![LocalExpertRoute {
                token_idx: 0,
                global_expert_idx: 5,
                local_expert_idx: 1,
                weight: 0.75,
            }]
        );
    }

    #[test]
    fn tilekernels_tp_layout_mask_matches_deepseek_formula() {
        let indices = [0, 3, 4, 7, 8, 11, 12, 15, -1];

        assert_eq!(
            tilekernels_mask_indices_by_tp_layout(&indices, 16, 4, 0, 2).unwrap(),
            vec![0, 3, -1, -1, 4, 7, -1, -1, -1]
        );
        assert_eq!(
            tilekernels_mask_indices_by_tp_layout(&indices, 16, 4, 1, 2).unwrap(),
            vec![-1, -1, 0, 3, -1, -1, 4, 7, -1]
        );
    }
}

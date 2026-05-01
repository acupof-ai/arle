//! Group coordinator metadata and collective dispatch.
//!
//! F1 keeps this intentionally narrow: it owns group rank metadata and can wrap
//! an existing NCCL group when the `nccl` feature is enabled. Single-rank groups
//! are pure no-ops so default serving can use the same surface without pulling
//! CUDA collectives into the hot path.

#[cfg(feature = "nccl")]
use std::sync::Arc;

use anyhow::{Result, bail};

use super::expert_state::{ExpertGroup, ExpertOutput, ExpertRoutingWeights};
use super::parallel_state::{RankGroup, get_ep_group, get_pp_group};

#[cfg(feature = "nccl")]
use super::nccl::NcclGroup;

#[derive(Clone)]
pub struct GroupCoordinator {
    name: String,
    ranks: Vec<usize>,
    rank: usize,
    rank_in_group: usize,
    #[cfg(feature = "nccl")]
    nccl: Option<Arc<NcclGroup>>,
}

impl GroupCoordinator {
    pub fn from_rank_group(name: impl Into<String>, group: &RankGroup) -> Self {
        Self {
            name: name.into(),
            ranks: group.ranks.clone(),
            rank: group.rank,
            rank_in_group: group.rank_in_group,
            #[cfg(feature = "nccl")]
            nccl: None,
        }
    }

    pub fn pipeline_group() -> Result<Self> {
        let group = get_pp_group()?;
        Ok(Self::from_rank_group("pp", &group))
    }

    pub fn expert_group() -> Result<Self> {
        let group = get_ep_group()?;
        Ok(Self::from_rank_group("ep", &group))
    }

    #[cfg(feature = "nccl")]
    pub fn with_nccl_group(
        name: impl Into<String>,
        group: &RankGroup,
        nccl: Arc<NcclGroup>,
    ) -> Result<Self> {
        if group.world_size() != nccl.world_size {
            bail!(
                "group {} has {} ranks but NCCL world_size is {}",
                group.kind.name(),
                group.world_size(),
                nccl.world_size
            );
        }
        Ok(Self {
            name: name.into(),
            ranks: group.ranks.clone(),
            rank: group.rank,
            rank_in_group: group.rank_in_group,
            nccl: Some(nccl),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ranks(&self) -> &[usize] {
        &self.ranks
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn rank_in_group(&self) -> usize {
        self.rank_in_group
    }

    pub fn world_size(&self) -> usize {
        self.ranks.len()
    }

    pub fn all_reduce_f32(&self, input: &[f32]) -> Result<Vec<f32>> {
        if self.world_size() == 1 {
            return Ok(input.to_vec());
        }
        #[cfg(feature = "nccl")]
        {
            let Some(nccl) = &self.nccl else {
                bail!("group {} has no NCCL backend for all_reduce", self.name);
            };
            return nccl.all_reduce_f32(input);
        }
        #[cfg(not(feature = "nccl"))]
        bail!(
            "GPU required: NCCL all_reduce for multi-rank group {}",
            self.name
        )
    }

    pub fn all_gather_f32(&self, input: &[f32], per_rank_count: usize) -> Result<Vec<f32>> {
        if input.len() != per_rank_count {
            bail!(
                "all_gather input len {} must equal per-rank count {per_rank_count} for group {}",
                input.len(),
                self.name
            );
        }
        if self.world_size() == 1 {
            return Ok(input.to_vec());
        }
        #[cfg(feature = "nccl")]
        {
            let Some(nccl) = &self.nccl else {
                bail!("group {} has no NCCL backend for all_gather", self.name);
            };
            return nccl.all_gather_f32(input, per_rank_count);
        }
        #[cfg(not(feature = "nccl"))]
        bail!(
            "GPU required: NCCL all_gather for multi-rank group {}",
            self.name
        )
    }

    pub fn broadcast_f32(
        &self,
        input: &[f32],
        count: usize,
        root_rank_in_group: usize,
    ) -> Result<Vec<f32>> {
        if root_rank_in_group >= self.world_size() {
            bail!(
                "broadcast root {} out of range for group {} size {}",
                root_rank_in_group,
                self.name,
                self.world_size()
            );
        }
        if self.rank_in_group == root_rank_in_group && input.len() != count {
            bail!(
                "broadcast root input len {} must equal count {count} for group {}",
                input.len(),
                self.name
            );
        }
        if self.world_size() == 1 {
            return Ok(input.to_vec());
        }
        #[cfg(feature = "nccl")]
        {
            let Some(nccl) = &self.nccl else {
                bail!("group {} has no NCCL backend for broadcast", self.name);
            };
            return nccl.broadcast_f32(input, count, root_rank_in_group);
        }
        #[cfg(not(feature = "nccl"))]
        bail!(
            "GPU required: NCCL broadcast for multi-rank group {}",
            self.name
        )
    }

    pub fn dispatch_to_experts(
        &self,
        routing_weights: &ExpertRoutingWeights,
        hidden_states: &[Vec<f32>],
    ) -> Result<Vec<ExpertOutput>> {
        let expert_group = ExpertGroup::new(
            self.rank_in_group,
            self.world_size(),
            routing_weights.num_experts,
        )?;
        for route in &routing_weights.routes {
            if route.expert_idx >= routing_weights.num_experts {
                bail!(
                    "route expert {} out of range for num_experts {}",
                    route.expert_idx,
                    routing_weights.num_experts
                );
            }
            if route.token_idx >= hidden_states.len() {
                bail!(
                    "route token {} out of range for hidden_states len {}",
                    route.token_idx,
                    hidden_states.len()
                );
            }
        }
        let mut outputs = Vec::new();
        for expert_idx in expert_group.local_expert_range() {
            let mut token_indices = Vec::new();
            let mut routed_hidden = Vec::new();
            for route in &routing_weights.routes {
                if route.expert_idx != expert_idx {
                    continue;
                }
                let hidden = &hidden_states[route.token_idx];
                token_indices.push(route.token_idx);
                routed_hidden.push(hidden.clone());
            }
            if !token_indices.is_empty() {
                outputs.push(ExpertOutput {
                    rank: self.rank,
                    expert_idx,
                    token_indices,
                    hidden_states: routed_hidden,
                });
            }
        }
        Ok(outputs)
    }

    pub fn combine_from_experts(
        &self,
        outputs: &[ExpertOutput],
        routing_weights: &ExpertRoutingWeights,
    ) -> Result<Vec<Vec<f32>>> {
        if routing_weights.routes.is_empty() {
            return Ok(Vec::new());
        }
        let max_token_idx = routing_weights
            .routes
            .iter()
            .map(|route| route.token_idx)
            .max()
            .unwrap_or(0);
        let hidden_dim = outputs
            .iter()
            .flat_map(|output| output.hidden_states.iter())
            .next()
            .map_or(0, Vec::len);
        let mut combined = vec![vec![0.0_f32; hidden_dim]; max_token_idx + 1];

        for output in outputs {
            if output.token_indices.len() != output.hidden_states.len() {
                bail!(
                    "expert output token row count {} does not match hidden row count {}",
                    output.token_indices.len(),
                    output.hidden_states.len()
                );
            }
            for (token_idx, hidden) in output.token_indices.iter().zip(&output.hidden_states) {
                if hidden.len() != hidden_dim {
                    bail!(
                        "expert output hidden dim {} does not match expected {hidden_dim}",
                        hidden.len()
                    );
                }
                let Some(route) = routing_weights.routes.iter().find(|route| {
                    route.token_idx == *token_idx && route.expert_idx == output.expert_idx
                }) else {
                    bail!(
                        "expert output for token {} expert {} has no matching route",
                        token_idx,
                        output.expert_idx
                    );
                };
                for (dst, src) in combined[*token_idx].iter_mut().zip(hidden) {
                    *dst += src * route.weight;
                }
            }
        }
        for route in &routing_weights.routes {
            let produced = outputs.iter().any(|output| {
                output.expert_idx == route.expert_idx
                    && output.token_indices.contains(&route.token_idx)
            });
            if !produced {
                bail!(
                    "route token {} expert {} has no expert output",
                    route.token_idx,
                    route.expert_idx
                );
            }
        }

        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::expert_state::{ExpertRoute, ExpertRoutingWeights};
    use crate::distributed::parallel_state::{
        ParallelGroupKind, RankGroup, destroy_model_parallel, initialize_model_parallel,
    };
    use crate::tensor_parallel::MultiAxisConfig;

    #[test]
    fn single_rank_collectives_are_noops() {
        let rank_group = RankGroup {
            kind: ParallelGroupKind::Tensor,
            ranks: vec![0],
            rank: 0,
            rank_in_group: 0,
        };
        let group = GroupCoordinator::from_rank_group("tp", &rank_group);
        assert_eq!(group.all_reduce_f32(&[1.0, 2.0]).unwrap(), vec![1.0, 2.0]);
        assert_eq!(
            group.all_gather_f32(&[1.0, 2.0], 2).unwrap(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            group.broadcast_f32(&[1.0, 2.0], 2, 0).unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn broadcast_rejects_bad_root() {
        let rank_group = RankGroup {
            kind: ParallelGroupKind::Tensor,
            ranks: vec![0],
            rank: 0,
            rank_in_group: 0,
        };
        let group = GroupCoordinator::from_rank_group("tp", &rank_group);
        assert!(group.broadcast_f32(&[1.0], 1, 1).is_err());
    }

    #[test]
    fn pipeline_group_accessor_uses_parallel_state() {
        destroy_model_parallel();
        initialize_model_parallel(
            MultiAxisConfig {
                tp_size: 1,
                pp_size: 2,
                ep_size: 1,
                attn_dp_size: 1,
                attn_cp_size: 1,
                moe_dp_size: 1,
            },
            1,
        )
        .unwrap();

        let group = GroupCoordinator::pipeline_group().unwrap();

        assert_eq!(group.name(), "pp");
        assert_eq!(group.ranks(), &[0, 1]);
        assert_eq!(group.rank(), 1);
        assert_eq!(group.rank_in_group(), 1);
        destroy_model_parallel();
    }

    #[test]
    fn expert_group_accessor_uses_parallel_state() {
        destroy_model_parallel();
        initialize_model_parallel(
            MultiAxisConfig {
                tp_size: 2,
                pp_size: 1,
                ep_size: 2,
                attn_dp_size: 1,
                attn_cp_size: 1,
                moe_dp_size: 1,
            },
            1,
        )
        .unwrap();

        let group = GroupCoordinator::expert_group().unwrap();

        assert_eq!(group.name(), "ep");
        assert_eq!(group.ranks(), &[0, 1]);
        assert_eq!(group.rank(), 1);
        assert_eq!(group.rank_in_group(), 1);
        destroy_model_parallel();
    }

    #[test]
    fn mock_expert_dispatch_filters_by_rank_owned_experts() {
        let rank0_group = RankGroup {
            kind: ParallelGroupKind::Expert,
            ranks: vec![0, 1],
            rank: 0,
            rank_in_group: 0,
        };
        let rank1_group = RankGroup {
            kind: ParallelGroupKind::Expert,
            ranks: vec![0, 1],
            rank: 1,
            rank_in_group: 1,
        };
        let rank0 = GroupCoordinator::from_rank_group("ep", &rank0_group);
        let rank1 = GroupCoordinator::from_rank_group("ep", &rank1_group);
        let hidden_states = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let route_to_rank0 = ExpertRoutingWeights::new(
            8,
            vec![
                ExpertRoute {
                    token_idx: 0,
                    expert_idx: 1,
                    weight: 1.0,
                },
                ExpertRoute {
                    token_idx: 1,
                    expert_idx: 3,
                    weight: 0.5,
                },
            ],
        );
        assert_eq!(
            rank0
                .dispatch_to_experts(&route_to_rank0, &hidden_states)
                .unwrap()
                .len(),
            2
        );
        assert!(
            rank1
                .dispatch_to_experts(&route_to_rank0, &hidden_states)
                .unwrap()
                .is_empty()
        );

        let route_to_rank1 = ExpertRoutingWeights::new(
            8,
            vec![ExpertRoute {
                token_idx: 0,
                expert_idx: 5,
                weight: 1.0,
            }],
        );
        assert!(
            rank0
                .dispatch_to_experts(&route_to_rank1, &hidden_states)
                .unwrap()
                .is_empty()
        );
        let rank1_outputs = rank1
            .dispatch_to_experts(&route_to_rank1, &hidden_states)
            .unwrap();
        assert_eq!(rank1_outputs.len(), 1);
        assert_eq!(rank1_outputs[0].rank, 1);
        assert_eq!(rank1_outputs[0].expert_idx, 5);
        assert_eq!(rank1_outputs[0].hidden_states, vec![vec![1.0, 2.0]]);
    }

    #[test]
    fn expert_dispatch_reports_global_rank_for_non_contiguous_group() {
        let rank_group = RankGroup {
            kind: ParallelGroupKind::Expert,
            ranks: vec![4, 6],
            rank: 6,
            rank_in_group: 1,
        };
        let coordinator = GroupCoordinator::from_rank_group("ep", &rank_group);
        let routing = ExpertRoutingWeights::new(
            8,
            vec![ExpertRoute {
                token_idx: 0,
                expert_idx: 5,
                weight: 1.0,
            }],
        );

        let outputs = coordinator
            .dispatch_to_experts(&routing, &[vec![1.0, 2.0]])
            .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].rank, 6);
    }

    #[test]
    fn combine_rejects_output_without_matching_route() {
        let rank_group = RankGroup {
            kind: ParallelGroupKind::Expert,
            ranks: vec![0],
            rank: 0,
            rank_in_group: 0,
        };
        let coordinator = GroupCoordinator::from_rank_group("ep", &rank_group);
        let routing = ExpertRoutingWeights::new(
            4,
            vec![ExpertRoute {
                token_idx: 0,
                expert_idx: 1,
                weight: 0.5,
            }],
        );
        let stale_output = ExpertOutput {
            rank: 0,
            expert_idx: 2,
            token_indices: vec![0],
            hidden_states: vec![vec![1.0, 2.0]],
        };

        assert!(
            coordinator
                .combine_from_experts(&[stale_output], &routing)
                .is_err()
        );
    }

    #[test]
    fn expert_dispatch_rejects_invalid_expert_ids_before_filtering() {
        let rank_group = RankGroup {
            kind: ParallelGroupKind::Expert,
            ranks: vec![0, 1],
            rank: 0,
            rank_in_group: 0,
        };
        let coordinator = GroupCoordinator::from_rank_group("ep", &rank_group);
        let routing = ExpertRoutingWeights::new(
            8,
            vec![ExpertRoute {
                token_idx: 0,
                expert_idx: 8,
                weight: 1.0,
            }],
        );

        assert!(
            coordinator
                .dispatch_to_experts(&routing, &[vec![1.0, 2.0]])
                .is_err()
        );
    }

    #[test]
    fn combine_rejects_mismatched_expert_output_rows() {
        let rank_group = RankGroup {
            kind: ParallelGroupKind::Expert,
            ranks: vec![0],
            rank: 0,
            rank_in_group: 0,
        };
        let coordinator = GroupCoordinator::from_rank_group("ep", &rank_group);
        let routing = ExpertRoutingWeights::new(
            4,
            vec![ExpertRoute {
                token_idx: 0,
                expert_idx: 1,
                weight: 0.5,
            }],
        );
        let malformed_output = ExpertOutput {
            rank: 0,
            expert_idx: 1,
            token_indices: vec![0, 1],
            hidden_states: vec![vec![1.0, 2.0]],
        };

        assert!(
            coordinator
                .combine_from_experts(&[malformed_output], &routing)
                .is_err()
        );
    }

    #[test]
    fn combine_rejects_missing_routed_expert_output() {
        let rank_group = RankGroup {
            kind: ParallelGroupKind::Expert,
            ranks: vec![0],
            rank: 0,
            rank_in_group: 0,
        };
        let coordinator = GroupCoordinator::from_rank_group("ep", &rank_group);
        let routing = ExpertRoutingWeights::new(
            8,
            vec![
                ExpertRoute {
                    token_idx: 0,
                    expert_idx: 1,
                    weight: 0.5,
                },
                ExpertRoute {
                    token_idx: 0,
                    expert_idx: 5,
                    weight: 0.5,
                },
            ],
        );
        let partial_output = ExpertOutput {
            rank: 0,
            expert_idx: 1,
            token_indices: vec![0],
            hidden_states: vec![vec![1.0, 2.0]],
        };

        assert!(
            coordinator
                .combine_from_experts(&[partial_output], &routing)
                .is_err()
        );
    }
}

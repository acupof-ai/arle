//! Group coordinator metadata and collective dispatch.
//!
//! F1 keeps this intentionally narrow: it owns group rank metadata and can wrap
//! an existing NCCL group when the `nccl` feature is enabled. Single-rank groups
//! are pure no-ops so default serving can use the same surface without pulling
//! CUDA collectives into the hot path.

#[cfg(feature = "nccl")]
use std::sync::Arc;

use anyhow::{Result, bail};

use super::parallel_state::{RankGroup, get_pp_group};

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
}

#[cfg(test)]
mod tests {
    use super::*;
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
}

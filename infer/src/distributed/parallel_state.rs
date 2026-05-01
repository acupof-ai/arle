//! Pure-Rust parallel-state layout for single-node multi-GPU execution.
//!
//! This mirrors the SGLang/vLLM access pattern: initialize one process/thread
//! with a world rank, then retrieve named groups such as TP, PP, attention TP,
//! and MoE EP through small accessors. The groups here are metadata only; NCCL
//! binding is provided by [`super::group_coordinator`].

use std::cell::RefCell;

use anyhow::{Context, Result};

use crate::tensor_parallel::{
    MultiAxisConfig, RankCoord, build_attn_cp_groups, build_attn_dp_groups, build_attn_tp_groups,
    build_moe_dp_groups, build_moe_ep_groups, build_moe_tp_groups, build_pp_groups,
    build_tp_groups,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelGroupKind {
    World,
    Tensor,
    Pipeline,
    Expert,
    AttentionTensor,
    AttentionData,
    AttentionContext,
    MoeTensor,
    MoeExpert,
    MoeData,
}

impl ParallelGroupKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::World => "world",
            Self::Tensor => "tp",
            Self::Pipeline => "pp",
            Self::Expert => "ep",
            Self::AttentionTensor => "attn_tp",
            Self::AttentionData => "attn_dp",
            Self::AttentionContext => "attn_cp",
            Self::MoeTensor => "moe_tp",
            Self::MoeExpert => "moe_ep",
            Self::MoeData => "moe_dp",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankGroup {
    pub kind: ParallelGroupKind,
    pub ranks: Vec<usize>,
    pub rank: usize,
    pub rank_in_group: usize,
}

impl RankGroup {
    fn new(kind: ParallelGroupKind, ranks: Vec<usize>, rank: usize) -> Result<Self> {
        let rank_in_group = ranks
            .iter()
            .position(|&candidate| candidate == rank)
            .with_context(|| {
                format!(
                    "rank {rank} is not a member of {} group {ranks:?}",
                    kind.name()
                )
            })?;
        Ok(Self {
            kind,
            ranks,
            rank,
            rank_in_group,
        })
    }

    pub fn world_size(&self) -> usize {
        self.ranks.len()
    }

    pub fn first_rank(&self) -> usize {
        self.ranks[0]
    }

    pub fn last_rank(&self) -> usize {
        self.ranks[self.ranks.len() - 1]
    }

    pub fn is_first_rank(&self) -> bool {
        self.rank == self.first_rank()
    }

    pub fn is_last_rank(&self) -> bool {
        self.rank == self.last_rank()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParallelState {
    pub config: MultiAxisConfig,
    pub coord: RankCoord,
    world_group: RankGroup,
    tp_group: RankGroup,
    pp_group: RankGroup,
    ep_group: RankGroup,
    attn_tp_group: RankGroup,
    attn_dp_group: RankGroup,
    attn_cp_group: RankGroup,
    moe_tp_group: RankGroup,
    moe_ep_group: RankGroup,
    moe_dp_group: RankGroup,
}

impl ParallelState {
    pub fn new(config: MultiAxisConfig, world_rank: usize) -> Result<Self> {
        config.validate()?;
        let coord = RankCoord::from_world_rank(config, world_rank)?;
        let world_group = RankGroup::new(
            ParallelGroupKind::World,
            (0..config.world_size()).collect(),
            world_rank,
        )?;
        let tp_group = group_containing(
            ParallelGroupKind::Tensor,
            build_tp_groups(config),
            world_rank,
        )?;
        let pp_group = group_containing(
            ParallelGroupKind::Pipeline,
            build_pp_groups(config),
            world_rank,
        )?;
        let moe_ep_groups = build_moe_ep_groups(config);
        let ep_group =
            group_containing(ParallelGroupKind::Expert, moe_ep_groups.clone(), world_rank)?;
        let attn_tp_group = group_containing(
            ParallelGroupKind::AttentionTensor,
            build_attn_tp_groups(config),
            world_rank,
        )?;
        let attn_dp_group = group_containing(
            ParallelGroupKind::AttentionData,
            build_attn_dp_groups(config),
            world_rank,
        )?;
        let attn_cp_group = group_containing(
            ParallelGroupKind::AttentionContext,
            build_attn_cp_groups(config),
            world_rank,
        )?;
        let moe_tp_group = group_containing(
            ParallelGroupKind::MoeTensor,
            build_moe_tp_groups(config),
            world_rank,
        )?;
        let moe_ep_group =
            group_containing(ParallelGroupKind::MoeExpert, moe_ep_groups, world_rank)?;
        let moe_dp_group = group_containing(
            ParallelGroupKind::MoeData,
            build_moe_dp_groups(config),
            world_rank,
        )?;

        Ok(Self {
            config,
            coord,
            world_group,
            tp_group,
            pp_group,
            ep_group,
            attn_tp_group,
            attn_dp_group,
            attn_cp_group,
            moe_tp_group,
            moe_ep_group,
            moe_dp_group,
        })
    }

    pub fn get_world_group(&self) -> &RankGroup {
        &self.world_group
    }

    pub fn get_tp_group(&self) -> &RankGroup {
        &self.tp_group
    }

    pub fn get_pp_group(&self) -> &RankGroup {
        &self.pp_group
    }

    pub fn get_ep_group(&self) -> &RankGroup {
        &self.ep_group
    }

    pub fn get_attention_tp_group(&self) -> &RankGroup {
        &self.attn_tp_group
    }

    pub fn get_attention_dp_group(&self) -> &RankGroup {
        &self.attn_dp_group
    }

    pub fn get_attention_cp_group(&self) -> &RankGroup {
        &self.attn_cp_group
    }

    pub fn get_moe_tp_group(&self) -> &RankGroup {
        &self.moe_tp_group
    }

    pub fn get_moe_ep_group(&self) -> &RankGroup {
        &self.moe_ep_group
    }

    pub fn get_moe_dp_group(&self) -> &RankGroup {
        &self.moe_dp_group
    }
}

fn group_containing(
    kind: ParallelGroupKind,
    groups: Vec<Vec<usize>>,
    rank: usize,
) -> Result<RankGroup> {
    let ranks = groups
        .into_iter()
        .find(|group| group.contains(&rank))
        .with_context(|| format!("rank {rank} has no {} group", kind.name()))?;
    RankGroup::new(kind, ranks, rank)
}

thread_local! {
    static THREAD_PARALLEL_STATE: RefCell<Option<ParallelState>> = const { RefCell::new(None) };
}

pub fn initialize_model_parallel(config: MultiAxisConfig, world_rank: usize) -> Result<()> {
    let state = ParallelState::new(config, world_rank)?;
    THREAD_PARALLEL_STATE.with(|slot| {
        *slot.borrow_mut() = Some(state);
    });
    Ok(())
}

pub fn destroy_model_parallel() {
    THREAD_PARALLEL_STATE.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

pub fn with_parallel_state<T>(f: impl FnOnce(&ParallelState) -> T) -> Result<T> {
    THREAD_PARALLEL_STATE.with(|slot| {
        let borrowed = slot.borrow();
        let state = borrowed
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("parallel state is not initialized"))?;
        Ok(f(state))
    })
}

pub fn get_world_group() -> Result<RankGroup> {
    with_parallel_state(|state| state.get_world_group().clone())
}

pub fn get_tp_group() -> Result<RankGroup> {
    with_parallel_state(|state| state.get_tp_group().clone())
}

pub fn get_pp_group() -> Result<RankGroup> {
    with_parallel_state(|state| state.get_pp_group().clone())
}

pub fn get_ep_group() -> Result<RankGroup> {
    with_parallel_state(|state| state.get_ep_group().clone())
}

pub fn get_attention_tp_group() -> Result<RankGroup> {
    with_parallel_state(|state| state.get_attention_tp_group().clone())
}

pub fn get_attention_dp_group() -> Result<RankGroup> {
    with_parallel_state(|state| state.get_attention_dp_group().clone())
}

pub fn get_attention_cp_group() -> Result<RankGroup> {
    with_parallel_state(|state| state.get_attention_cp_group().clone())
}

pub fn get_moe_tp_group() -> Result<RankGroup> {
    with_parallel_state(|state| state.get_moe_tp_group().clone())
}

pub fn get_moe_ep_group() -> Result<RankGroup> {
    with_parallel_state(|state| state.get_moe_ep_group().clone())
}

pub fn get_moe_dp_group() -> Result<RankGroup> {
    with_parallel_state(|state| state.get_moe_dp_group().clone())
}

pub fn assert_initialized() -> Result<()> {
    with_parallel_state(|_| ())
        .with_context(|| "parallel state must be initialized before distributed accessors are used")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_rank_groups_all_point_to_rank_zero() {
        let state = ParallelState::new(MultiAxisConfig::single(), 0).unwrap();
        for group in [
            state.get_world_group(),
            state.get_tp_group(),
            state.get_pp_group(),
            state.get_ep_group(),
            state.get_attention_tp_group(),
            state.get_attention_dp_group(),
            state.get_attention_cp_group(),
            state.get_moe_tp_group(),
            state.get_moe_ep_group(),
            state.get_moe_dp_group(),
        ] {
            assert_eq!(group.ranks, vec![0]);
            assert_eq!(group.rank_in_group, 0);
        }
    }

    #[test]
    fn accessors_match_multi_axis_layout() {
        let cfg = MultiAxisConfig {
            tp_size: 4,
            pp_size: 2,
            ep_size: 2,
            attn_dp_size: 2,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        let state = ParallelState::new(cfg, 6).unwrap();
        assert_eq!(state.get_world_group().ranks, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(state.get_tp_group().ranks, vec![4, 5, 6, 7]);
        assert_eq!(state.get_pp_group().ranks, vec![2, 6]);
        assert_eq!(state.get_attention_tp_group().ranks, vec![6, 7]);
        assert_eq!(state.get_attention_dp_group().ranks, vec![4, 6]);
        assert_eq!(state.get_attention_cp_group().ranks, vec![6]);
        assert_eq!(state.get_moe_tp_group().ranks, vec![6, 7]);
        assert_eq!(state.get_moe_ep_group().ranks, vec![4, 6]);
        assert_eq!(state.get_moe_dp_group().ranks, vec![6]);
    }

    #[test]
    fn thread_local_accessors_require_initialization() {
        destroy_model_parallel();
        assert!(get_tp_group().is_err());
        initialize_model_parallel(MultiAxisConfig::single(), 0).unwrap();
        assert_eq!(get_tp_group().unwrap().ranks, vec![0]);
        destroy_model_parallel();
    }
}

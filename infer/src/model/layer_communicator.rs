//! Layer-level communication skeleton for tensor/context/data parallel forward.
//!
//! F0.8 intentionally keeps this module detached from Qwen forward call sites.
//! It defines the method surface that F1+ TP/DP/CP forward paths will call, with
//! exact single-rank pass-through behavior so the default runtime path remains
//! inert.

use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerCollective {
    PostAttentionAllReduce,
    PostMlpAllReduce,
    DpAttentionGather,
    DpAttentionScatter,
    CpAttentionSplit,
    CpAttentionGather,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerCommStatus {
    NoopSingleRank,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerCommunicator {
    tp_rank: usize,
    tp_world_size: usize,
    dp_rank: usize,
    dp_world_size: usize,
    cp_rank: usize,
    cp_world_size: usize,
}

impl LayerCommunicator {
    pub fn single() -> Self {
        Self {
            tp_rank: 0,
            tp_world_size: 1,
            dp_rank: 0,
            dp_world_size: 1,
            cp_rank: 0,
            cp_world_size: 1,
        }
    }

    pub fn new(
        tp_rank: usize,
        tp_world_size: usize,
        dp_rank: usize,
        dp_world_size: usize,
        cp_rank: usize,
        cp_world_size: usize,
    ) -> Result<Self> {
        validate_axis("tp", tp_rank, tp_world_size)?;
        validate_axis("dp", dp_rank, dp_world_size)?;
        validate_axis("cp", cp_rank, cp_world_size)?;
        Ok(Self {
            tp_rank,
            tp_world_size,
            dp_rank,
            dp_world_size,
            cp_rank,
            cp_world_size,
        })
    }

    pub fn tp_rank(&self) -> usize {
        self.tp_rank
    }

    pub fn tp_world_size(&self) -> usize {
        self.tp_world_size
    }

    pub fn dp_rank(&self) -> usize {
        self.dp_rank
    }

    pub fn dp_world_size(&self) -> usize {
        self.dp_world_size
    }

    pub fn cp_rank(&self) -> usize {
        self.cp_rank
    }

    pub fn cp_world_size(&self) -> usize {
        self.cp_world_size
    }

    pub fn is_single_rank(&self) -> bool {
        self.tp_world_size == 1 && self.dp_world_size == 1 && self.cp_world_size == 1
    }

    pub fn post_attn_all_reduce<T>(&self, hidden: &mut [T]) -> Result<LayerCommStatus> {
        Self::ensure_noop(
            LayerCollective::PostAttentionAllReduce,
            self.tp_world_size,
            hidden.len(),
        )
    }

    pub fn post_mlp_all_reduce<T>(&self, hidden: &mut [T]) -> Result<LayerCommStatus> {
        Self::ensure_noop(
            LayerCollective::PostMlpAllReduce,
            self.tp_world_size,
            hidden.len(),
        )
    }

    pub fn all_reduce_post_attention<T>(&self, hidden: &mut [T]) -> Result<LayerCommStatus> {
        self.post_attn_all_reduce(hidden)
    }

    pub fn all_reduce_post_mlp<T>(&self, hidden: &mut [T]) -> Result<LayerCommStatus> {
        self.post_mlp_all_reduce(hidden)
    }

    pub fn dp_attn_gather<T: Clone>(&self, local: &[T]) -> Result<Vec<T>> {
        Self::ensure_noop(
            LayerCollective::DpAttentionGather,
            self.dp_world_size,
            local.len(),
        )?;
        Ok(local.to_vec())
    }

    pub fn dp_attn_scatter<T: Clone>(&self, gathered: &[T]) -> Result<Vec<T>> {
        Self::ensure_noop(
            LayerCollective::DpAttentionScatter,
            self.dp_world_size,
            gathered.len(),
        )?;
        Ok(gathered.to_vec())
    }

    pub fn cp_split<T: Clone>(&self, sequence: &[T]) -> Result<Vec<T>> {
        Self::ensure_noop(
            LayerCollective::CpAttentionSplit,
            self.cp_world_size,
            sequence.len(),
        )?;
        Ok(sequence.to_vec())
    }

    pub fn cp_attention_split<T: Clone>(&self, sequence: &[T]) -> Result<Vec<T>> {
        self.cp_split(sequence)
    }

    pub fn cp_attention_gather<T: Clone>(&self, local: &[T]) -> Result<Vec<T>> {
        Self::ensure_noop(
            LayerCollective::CpAttentionGather,
            self.cp_world_size,
            local.len(),
        )?;
        Ok(local.to_vec())
    }

    pub fn fused_allreduce_residual_rmsnorm<T>(
        &self,
        hidden: &mut [T],
        residual: &mut [T],
    ) -> Result<LayerCommStatus> {
        if hidden.len() != residual.len() {
            bail!(
                "fused_allreduce_residual_rmsnorm requires matching hidden/residual lengths, got {} and {}",
                hidden.len(),
                residual.len()
            );
        }
        Self::ensure_noop(
            LayerCollective::PostMlpAllReduce,
            self.tp_world_size,
            hidden.len(),
        )
    }

    fn ensure_noop(
        collective: LayerCollective,
        world_size: usize,
        _len: usize,
    ) -> Result<LayerCommStatus> {
        if world_size == 1 {
            return Ok(LayerCommStatus::NoopSingleRank);
        }
        bail!(
            "{collective:?} requires world_size={world_size}; LayerCommunicator F0.8 only supports single-rank no-op"
        )
    }
}

impl Default for LayerCommunicator {
    fn default() -> Self {
        Self::single()
    }
}

fn validate_axis(name: &str, rank: usize, world_size: usize) -> Result<()> {
    if world_size == 0 {
        bail!("{name}_world_size must be >= 1");
    }
    if rank >= world_size {
        bail!("{name}_rank ({rank}) must be < {name}_world_size ({world_size})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_rank_post_attention_all_reduce_preserves_buffer() {
        let comm = LayerCommunicator::single();
        let mut hidden = vec![1.0f32, 2.0, 3.0];
        let before = hidden.clone();

        let status = comm.post_attn_all_reduce(&mut hidden).unwrap();

        assert_eq!(status, LayerCommStatus::NoopSingleRank);
        assert_eq!(hidden, before);
    }

    #[test]
    fn single_rank_post_mlp_all_reduce_preserves_buffer() {
        let comm = LayerCommunicator::single();
        let mut hidden = vec![7u32, 8, 9];
        let before = hidden.clone();

        let status = comm.post_mlp_all_reduce(&mut hidden).unwrap();

        assert_eq!(status, LayerCommStatus::NoopSingleRank);
        assert_eq!(hidden, before);
    }

    #[test]
    fn single_rank_dp_and_cp_paths_are_pass_through() {
        let comm = LayerCommunicator::single();
        let tokens = vec![10u32, 11, 12, 13];

        assert_eq!(comm.dp_attn_gather(&tokens).unwrap(), tokens);
        assert_eq!(comm.dp_attn_scatter(&tokens).unwrap(), tokens);
        assert_eq!(comm.cp_split(&tokens).unwrap(), tokens);
        assert_eq!(comm.cp_attention_gather(&tokens).unwrap(), tokens);
    }

    #[test]
    fn multi_rank_collectives_reject_until_wired() {
        let comm = LayerCommunicator::new(0, 2, 0, 1, 0, 1).unwrap();
        let mut hidden = vec![1u8, 2, 3];

        assert!(comm.post_attn_all_reduce(&mut hidden).is_err());
        assert_eq!(hidden, vec![1, 2, 3]);
    }

    #[test]
    fn constructor_validates_axis_ranks() {
        assert!(LayerCommunicator::new(1, 1, 0, 1, 0, 1).is_err());
        assert!(LayerCommunicator::new(0, 0, 0, 1, 0, 1).is_err());
    }
}

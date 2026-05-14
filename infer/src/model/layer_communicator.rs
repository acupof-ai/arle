//! Layer-level communication skeleton for tensor/context/data parallel forward.
//!
//! F0.8 intentionally keeps this module detached from Qwen forward call sites.
//! It defines the method surface that F1+ TP/DP/CP forward paths will call, with
//! exact single-rank pass-through behavior so the default runtime path remains
//! inert.

use anyhow::{Result, bail};

#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceVec, HiddenStates};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use cudarc::driver::CudaSlice;
#[cfg(all(feature = "cuda", feature = "nccl"))]
use half::bf16;
#[cfg(feature = "nccl")]
use std::sync::Arc;

#[cfg(feature = "nccl")]
use crate::distributed::nccl::NcclGroup;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerCollective {
    PostAttentionAllReduce,
    PostMlpAllReduce,
    PostMoeExpertAllReduce,
    DpAttentionGather,
    DpAttentionScatter,
    CpAttentionSplit,
    CpAttentionGather,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerCommStatus {
    NoopSingleRank,
    AllReduceSum,
    GroupedSendRecv,
}

#[derive(Clone, Debug)]
pub struct LayerCommunicator {
    tp_rank: usize,
    tp_world_size: usize,
    dp_rank: usize,
    dp_world_size: usize,
    cp_rank: usize,
    cp_world_size: usize,
    ep_rank: usize,
    ep_world_size: usize,
    #[cfg(feature = "nccl")]
    tp_nccl: Option<Arc<NcclGroup>>,
    #[cfg(feature = "nccl")]
    ep_nccl: Option<Arc<NcclGroup>>,
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
            ep_rank: 0,
            ep_world_size: 1,
            #[cfg(feature = "nccl")]
            tp_nccl: None,
            #[cfg(feature = "nccl")]
            ep_nccl: None,
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
        Self::new_with_ep(
            tp_rank,
            tp_world_size,
            dp_rank,
            dp_world_size,
            cp_rank,
            cp_world_size,
            0,
            1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_ep(
        tp_rank: usize,
        tp_world_size: usize,
        dp_rank: usize,
        dp_world_size: usize,
        cp_rank: usize,
        cp_world_size: usize,
        ep_rank: usize,
        ep_world_size: usize,
    ) -> Result<Self> {
        validate_axis("tp", tp_rank, tp_world_size)?;
        validate_axis("dp", dp_rank, dp_world_size)?;
        validate_axis("cp", cp_rank, cp_world_size)?;
        validate_axis("ep", ep_rank, ep_world_size)?;
        Ok(Self {
            tp_rank,
            tp_world_size,
            dp_rank,
            dp_world_size,
            cp_rank,
            cp_world_size,
            ep_rank,
            ep_world_size,
            #[cfg(feature = "nccl")]
            tp_nccl: None,
            #[cfg(feature = "nccl")]
            ep_nccl: None,
        })
    }

    #[cfg(feature = "nccl")]
    pub fn with_tp_nccl(mut self, nccl: Arc<NcclGroup>) -> Result<Self> {
        if self.tp_world_size != nccl.world_size {
            bail!(
                "LayerCommunicator TP world_size {} does not match NCCL world_size {}",
                self.tp_world_size,
                nccl.world_size
            );
        }
        if self.tp_rank != nccl.rank {
            bail!(
                "LayerCommunicator TP rank {} does not match NCCL rank {}",
                self.tp_rank,
                nccl.rank
            );
        }
        self.tp_nccl = Some(nccl);
        Ok(self)
    }

    #[cfg(feature = "nccl")]
    pub fn with_ep_nccl(mut self, nccl: Arc<NcclGroup>) -> Result<Self> {
        if self.ep_world_size != nccl.world_size {
            bail!(
                "LayerCommunicator EP world_size {} does not match NCCL world_size {}",
                self.ep_world_size,
                nccl.world_size
            );
        }
        if self.ep_rank != nccl.rank {
            bail!(
                "LayerCommunicator EP rank {} does not match NCCL rank {}",
                self.ep_rank,
                nccl.rank
            );
        }
        self.ep_nccl = Some(nccl);
        Ok(self)
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

    pub fn ep_rank(&self) -> usize {
        self.ep_rank
    }

    pub fn ep_world_size(&self) -> usize {
        self.ep_world_size
    }

    pub fn is_single_rank(&self) -> bool {
        self.tp_world_size == 1
            && self.dp_world_size == 1
            && self.cp_world_size == 1
            && self.ep_world_size == 1
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

    pub fn post_moe_expert_all_reduce<T>(&self, hidden: &mut [T]) -> Result<LayerCommStatus> {
        Self::ensure_noop(
            LayerCollective::PostMoeExpertAllReduce,
            self.ep_world_size,
            hidden.len(),
        )
    }

    #[cfg(feature = "cuda")]
    pub fn post_attn_all_reduce_hidden_states(
        &self,
        hidden: &mut HiddenStates,
    ) -> Result<LayerCommStatus> {
        self.all_reduce_bf16(
            LayerCollective::PostAttentionAllReduce,
            ParallelAxis::Tensor,
            &mut hidden.data,
            hidden.hidden_dim.saturating_mul(hidden.seq_len),
        )
    }

    #[cfg(feature = "cuda")]
    pub fn post_mlp_all_reduce_hidden_states(
        &self,
        hidden: &mut HiddenStates,
    ) -> Result<LayerCommStatus> {
        self.all_reduce_bf16(
            LayerCollective::PostMlpAllReduce,
            ParallelAxis::Tensor,
            &mut hidden.data,
            hidden.hidden_dim.saturating_mul(hidden.seq_len),
        )
    }

    #[cfg(feature = "cuda")]
    pub fn post_moe_expert_all_reduce_hidden_states(
        &self,
        hidden: &mut HiddenStates,
    ) -> Result<LayerCommStatus> {
        self.all_reduce_bf16(
            LayerCollective::PostMoeExpertAllReduce,
            ParallelAxis::Expert,
            &mut hidden.data,
            hidden.hidden_dim.saturating_mul(hidden.seq_len),
        )
    }

    #[cfg(feature = "cuda")]
    pub fn dp_attn_gather_hidden_states(
        &self,
        hidden: &mut HiddenStates,
    ) -> Result<LayerCommStatus> {
        Self::ensure_noop(
            LayerCollective::DpAttentionGather,
            self.dp_world_size,
            hidden.hidden_dim.saturating_mul(hidden.seq_len),
        )
    }

    #[cfg(feature = "cuda")]
    pub fn post_attn_all_reduce_device_vec(
        &self,
        hidden: &mut DeviceVec,
    ) -> Result<LayerCommStatus> {
        self.all_reduce_bf16(
            LayerCollective::PostAttentionAllReduce,
            ParallelAxis::Tensor,
            &mut hidden.data,
            hidden.len,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn post_mlp_all_reduce_device_vec(
        &self,
        hidden: &mut DeviceVec,
    ) -> Result<LayerCommStatus> {
        self.all_reduce_bf16(
            LayerCollective::PostMlpAllReduce,
            ParallelAxis::Tensor,
            &mut hidden.data,
            hidden.len,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn post_moe_expert_all_reduce_device_vec(
        &self,
        hidden: &mut DeviceVec,
    ) -> Result<LayerCommStatus> {
        self.all_reduce_bf16(
            LayerCollective::PostMoeExpertAllReduce,
            ParallelAxis::Expert,
            &mut hidden.data,
            hidden.len,
        )
    }

    /// EP-axis grouped BF16 send/recv for DeepEP-style MoE token exchange.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub fn moe_grouped_send_recv_bf16(
        &self,
        sendbuf: &CudaSlice<bf16>,
        send_offsets: &[usize],
        send_counts: &[usize],
        recvbuf: &mut CudaSlice<bf16>,
        recv_offsets: &[usize],
        recv_counts: &[usize],
    ) -> Result<LayerCommStatus> {
        if self.ep_world_size == 1 {
            return Ok(LayerCommStatus::NoopSingleRank);
        }
        let Some(nccl) = self.ep_nccl.as_ref() else {
            bail!(
                "MoE grouped BF16 send/recv requires EP NCCL backend for world_size={}",
                self.ep_world_size
            );
        };
        nccl.grouped_send_recv_bf16(
            sendbuf,
            send_offsets,
            send_counts,
            recvbuf,
            recv_offsets,
            recv_counts,
        )?;
        Ok(LayerCommStatus::GroupedSendRecv)
    }

    /// EP-axis grouped I32 send/recv for DeepEP-style MoE metadata exchange.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub fn moe_grouped_send_recv_i32(
        &self,
        sendbuf: &CudaSlice<i32>,
        send_offsets: &[usize],
        send_counts: &[usize],
        recvbuf: &mut CudaSlice<i32>,
        recv_offsets: &[usize],
        recv_counts: &[usize],
    ) -> Result<LayerCommStatus> {
        if self.ep_world_size == 1 {
            return Ok(LayerCommStatus::NoopSingleRank);
        }
        let Some(nccl) = self.ep_nccl.as_ref() else {
            bail!(
                "MoE grouped I32 send/recv requires EP NCCL backend for world_size={}",
                self.ep_world_size
            );
        };
        nccl.grouped_send_recv_i32(
            sendbuf,
            send_offsets,
            send_counts,
            recvbuf,
            recv_offsets,
            recv_counts,
        )?;
        Ok(LayerCommStatus::GroupedSendRecv)
    }

    /// EP-axis I32 all-gather for tiny DeepEP-style route-count exchange.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub fn moe_all_gather_i32(
        &self,
        sendbuf: &CudaSlice<i32>,
        send_count: usize,
        recvbuf: &mut CudaSlice<i32>,
    ) -> Result<LayerCommStatus> {
        if self.ep_world_size == 1 {
            return Ok(LayerCommStatus::NoopSingleRank);
        }
        let Some(nccl) = self.ep_nccl.as_ref() else {
            bail!(
                "MoE I32 all-gather requires EP NCCL backend for world_size={}",
                self.ep_world_size
            );
        };
        nccl.all_gather_i32_device(sendbuf, send_count, recvbuf)?;
        Ok(LayerCommStatus::GroupedSendRecv)
    }

    /// EP-axis grouped F32 send/recv for DeepEP-style route weights.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub fn moe_grouped_send_recv_f32(
        &self,
        sendbuf: &CudaSlice<f32>,
        send_offsets: &[usize],
        send_counts: &[usize],
        recvbuf: &mut CudaSlice<f32>,
        recv_offsets: &[usize],
        recv_counts: &[usize],
    ) -> Result<LayerCommStatus> {
        if self.ep_world_size == 1 {
            return Ok(LayerCommStatus::NoopSingleRank);
        }
        let Some(nccl) = self.ep_nccl.as_ref() else {
            bail!(
                "MoE grouped F32 send/recv requires EP NCCL backend for world_size={}",
                self.ep_world_size
            );
        };
        nccl.grouped_send_recv_f32(
            sendbuf,
            send_offsets,
            send_counts,
            recvbuf,
            recv_offsets,
            recv_counts,
        )?;
        Ok(LayerCommStatus::GroupedSendRecv)
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

    #[cfg(feature = "cuda")]
    fn all_reduce_bf16(
        &self,
        collective: LayerCollective,
        axis: ParallelAxis,
        buffer: &mut cudarc::driver::CudaSlice<half::bf16>,
        len: usize,
    ) -> Result<LayerCommStatus> {
        if len != buffer.len() {
            bail!(
                "{collective:?} buffer len {} does not match logical len {len}",
                buffer.len()
            );
        }
        let world_size = match axis {
            ParallelAxis::Tensor => self.tp_world_size,
            ParallelAxis::Expert => self.ep_world_size,
        };
        if world_size == 1 {
            return Ok(LayerCommStatus::NoopSingleRank);
        }
        #[cfg(feature = "nccl")]
        {
            let nccl = match axis {
                ParallelAxis::Tensor => self.tp_nccl.as_ref(),
                ParallelAxis::Expert => self.ep_nccl.as_ref(),
            };
            let Some(nccl) = nccl else {
                bail!(
                    "{collective:?} requires {} NCCL backend for world_size={world_size}",
                    axis.name()
                );
            };
            nccl.all_reduce_bf16_in_place(buffer)?;
            return Ok(LayerCommStatus::AllReduceSum);
        }
        #[cfg(not(feature = "nccl"))]
        bail!(
            "{collective:?} requires {} NCCL backend for world_size={world_size}; build with --features nccl",
            axis.name()
        )
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum ParallelAxis {
    Tensor,
    Expert,
}

#[cfg(feature = "cuda")]
impl ParallelAxis {
    fn name(self) -> &'static str {
        match self {
            Self::Tensor => "TP",
            Self::Expert => "EP",
        }
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

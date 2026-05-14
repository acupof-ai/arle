//! Per-request mutable state for the DeepSeek V4 model scaffold.
//!
//! Mirrors `Qwen3State`: wraps `GenerationStateBase` and forwards every
//! `GenerationState` method through it. V4-specific scratch (hybrid attention,
//! compressor/indexer metadata, and MoE route buffers) will live alongside
//! `decode_bufs` once Phase 2A exposes the kernel surfaces.

#[cfg(all(test, feature = "cuda"))]
use std::collections::VecDeque;

use anyhow::Result;

use crate::model::GenerationState;
#[cfg(feature = "cuda")]
use crate::model::generation_state::GenerationStateBase;
#[cfg(feature = "cuda")]
use crate::model::kv_cache::KVCacheDtype;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates, PagedKVPool};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
use half::bf16;

/// Per-request DeepSeek mutable state.
///
/// Currently a thin wrapper over `GenerationStateBase` so the trait surface
/// matches `Qwen3State`. V4 attention/MoE scratch lands here when the kernel
/// surface is real.
pub struct DeepseekState {
    #[cfg(feature = "cuda")]
    pub(crate) base: GenerationStateBase,
    #[cfg(feature = "cuda")]
    pub(crate) decode_logits: DeviceVec,
    #[cfg(feature = "cuda")]
    pub(crate) sample_probs: CudaSlice<f32>,
    #[cfg(feature = "cuda")]
    pub(crate) sample_out: CudaSlice<i32>,
    #[cfg(feature = "cuda")]
    pub(crate) reference_tokens: Vec<u32>,
    #[cfg(feature = "cuda")]
    pub(crate) incremental: DeepseekIncrementalState,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepseekIncrementalState {
    pub(crate) processed_tokens: usize,
    pub(crate) layers: Vec<DeepseekLayerRuntimeCache>,
}

#[cfg(feature = "cuda")]
impl DeepseekIncrementalState {
    pub(crate) fn clear(&mut self) {
        self.processed_tokens = 0;
        self.layers.clear();
    }

    pub(crate) fn ensure_layers(&mut self, layers: usize) {
        if self.layers.len() < layers {
            self.layers
                .resize_with(layers, DeepseekLayerRuntimeCache::default);
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepseekLayerRuntimeCache {
    pub(crate) attention: DeepseekAttentionRuntimeCache,
    pub(crate) moe: DeepseekMoeRuntimeCache,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepseekMoeRuntimeCache {
    pub(crate) expert: Option<DeepseekExpertRuntimeScratch>,
    pub(crate) grouped: Option<DeepseekGroupedExpertRuntimeScratch>,
    pub(crate) route_combine: Option<DeepseekRouteCombineRuntimeScratch>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekExpertRuntimeScratch {
    pub(crate) capacity_tokens: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) intermediate_dim: usize,
    pub(crate) output_dim: usize,
    pub(crate) input: HiddenStates,
    pub(crate) gate: HiddenStates,
    pub(crate) up: HiddenStates,
    pub(crate) act: HiddenStates,
    pub(crate) out: HiddenStates,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekGroupedExpertRuntimeScratch {
    pub(crate) capacity_routes: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) intermediate_dim: usize,
    pub(crate) gate: HiddenStates,
    pub(crate) up: HiddenStates,
    pub(crate) act: HiddenStates,
    pub(crate) out: HiddenStates,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekRouteCombineRuntimeScratch {
    pub(crate) capacity_routes: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) combine_recv: HiddenStates,
    pub(crate) route_slot_out: HiddenStates,
}

#[cfg(feature = "cuda")]
impl DeepseekMoeRuntimeCache {
    pub(crate) fn ensure_expert_scratch(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        intermediate_dim: usize,
        output_dim: usize,
        capacity_tokens: usize,
    ) -> Result<&mut DeepseekExpertRuntimeScratch> {
        let capacity_tokens = capacity_tokens.max(1);
        let needs_alloc = self
            .expert
            .as_ref()
            .map(|scratch| {
                scratch.capacity_tokens < capacity_tokens
                    || scratch.hidden_dim != hidden_dim
                    || scratch.intermediate_dim != intermediate_dim
                    || scratch.output_dim != output_dim
            })
            .unwrap_or(true);
        if needs_alloc {
            self.expert = Some(DeepseekExpertRuntimeScratch {
                capacity_tokens,
                hidden_dim,
                intermediate_dim,
                output_dim,
                input: HiddenStates::zeros(ctx, hidden_dim, capacity_tokens)?,
                gate: HiddenStates::zeros(ctx, intermediate_dim, capacity_tokens)?,
                up: HiddenStates::zeros(ctx, intermediate_dim, capacity_tokens)?,
                act: HiddenStates::zeros(ctx, intermediate_dim, capacity_tokens)?,
                out: HiddenStates::zeros(ctx, output_dim, capacity_tokens)?,
            });
        }
        Ok(self
            .expert
            .as_mut()
            .expect("DeepSeek V4 expert scratch allocated"))
    }

    pub(crate) fn ensure_grouped_expert_scratch(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        intermediate_dim: usize,
        capacity_routes: usize,
    ) -> Result<&mut DeepseekGroupedExpertRuntimeScratch> {
        let capacity_routes = capacity_routes.max(1);
        let needs_alloc = self
            .grouped
            .as_ref()
            .map(|scratch| {
                scratch.capacity_routes < capacity_routes
                    || scratch.hidden_dim != hidden_dim
                    || scratch.intermediate_dim != intermediate_dim
            })
            .unwrap_or(true);
        if needs_alloc {
            self.grouped = Some(DeepseekGroupedExpertRuntimeScratch {
                capacity_routes,
                hidden_dim,
                intermediate_dim,
                gate: HiddenStates::zeros(ctx, intermediate_dim, capacity_routes)?,
                up: HiddenStates::zeros(ctx, intermediate_dim, capacity_routes)?,
                act: HiddenStates::zeros(ctx, intermediate_dim, capacity_routes)?,
                out: HiddenStates::zeros(ctx, hidden_dim, capacity_routes)?,
            });
        }
        Ok(self
            .grouped
            .as_mut()
            .expect("DeepSeek V4 grouped expert scratch allocated"))
    }

    pub(crate) fn ensure_route_combine_scratch(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        capacity_routes: usize,
    ) -> Result<&mut DeepseekRouteCombineRuntimeScratch> {
        let capacity_routes = capacity_routes.max(1);
        let needs_alloc = self
            .route_combine
            .as_ref()
            .map(|scratch| {
                scratch.capacity_routes < capacity_routes || scratch.hidden_dim != hidden_dim
            })
            .unwrap_or(true);
        if needs_alloc {
            self.route_combine = Some(DeepseekRouteCombineRuntimeScratch {
                capacity_routes,
                hidden_dim,
                combine_recv: HiddenStates::zeros(ctx, hidden_dim, capacity_routes)?,
                route_slot_out: HiddenStates::zeros(ctx, hidden_dim, capacity_routes)?,
            });
        }
        Ok(self
            .route_combine
            .as_mut()
            .expect("DeepSeek V4 route combine scratch allocated"))
    }
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepseekAttentionRuntimeCache {
    #[cfg(test)]
    pub(crate) window: VecDeque<DeepseekKvRow>,
    pub(crate) window_gpu: Option<CudaSlice<bf16>>,
    pub(crate) window_gpu_len: usize,
    #[cfg(test)]
    pub(crate) compressed: Option<DeepseekCompressorRuntimeCache>,
    #[cfg(test)]
    pub(crate) indexer: Option<DeepseekCompressorRuntimeCache>,
    pub(crate) compressed_gpu: Option<DeepseekGpuCompressorRuntimeCache>,
    pub(crate) indexer_gpu: Option<DeepseekGpuCompressorRuntimeCache>,
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) struct DeepseekKvRow {
    pub(crate) pos: usize,
    pub(crate) values: Vec<f32>,
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) struct DeepseekCompressedRow {
    pub(crate) end_pos: usize,
    pub(crate) values: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepseekGpuCompressorRuntimeCache {
    pub(crate) pending_kv: Option<CudaSlice<bf16>>,
    pub(crate) pending_score: Option<CudaSlice<bf16>>,
    pub(crate) prev_overlap_kv: Option<CudaSlice<bf16>>,
    pub(crate) prev_overlap_score: Option<CudaSlice<bf16>>,
    pub(crate) compressed: Option<CudaSlice<bf16>>,
    pub(crate) pending_len: usize,
    pub(crate) compressed_rows: usize,
    pub(crate) compressed_capacity: usize,
    pub(crate) pending_width: usize,
    pub(crate) head_dim: usize,
}

#[cfg(all(test, feature = "cuda"))]
#[derive(Default)]
pub(crate) struct DeepseekCompressorRuntimeCache {
    pub(crate) pending_kv: Vec<f32>,
    pub(crate) pending_score: Vec<f32>,
    pub(crate) prev_overlap_kv: Vec<f32>,
    pub(crate) prev_overlap_score: Vec<f32>,
    pub(crate) compressed: Vec<DeepseekCompressedRow>,
}

// SAFETY: identical invariant to `Qwen3State` — every `DeepseekState` is owned
// by exactly one scheduler slot, accessed only from the single inference
// thread that runs `Scheduler::run()`. CUDA resources held inside
// `GenerationStateBase` are not shared across threads.
#[cfg(feature = "cuda")]
unsafe impl Send for DeepseekState {}

#[cfg(feature = "cuda")]
impl GenerationState for DeepseekState {
    fn logits(&self) -> &DeviceVec {
        self.base.logits_or(&self.decode_logits)
    }

    fn reset(&mut self) -> Result<()> {
        self.reference_tokens.clear();
        self.incremental.clear();
        self.base.reset()
    }

    fn reset_for_warmup_clear(&mut self) -> Result<()> {
        self.reference_tokens.clear();
        self.incremental.clear();
        self.base.reset()
    }

    fn truncate_to(&mut self, len: usize) -> Result<()> {
        self.reference_tokens.truncate(len);
        if self.incremental.processed_tokens > len {
            self.incremental.clear();
        }
        self.base.truncate_to(len)
    }

    fn supports_partial_prefix(&self) -> bool {
        false
    }

    fn set_max_seq_len(&mut self, max_seq: usize) {
        self.base.set_max_seq_len(max_seq);
    }

    fn set_kv_dtype(&mut self, dtype: KVCacheDtype) {
        self.base.set_kv_dtype(dtype);
    }

    fn migrate_kv_to_paged(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
    ) -> Result<()> {
        self.base.migrate_kv_to_paged(ctx, pool, slot)
    }

    fn migrate_kv_range_to_paged(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        start_pos: usize,
        token_count: usize,
    ) -> Result<()> {
        self.base
            .migrate_kv_range_to_paged(ctx, pool, slot, start_pos, token_count)
    }
}

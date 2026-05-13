//! Per-request mutable state for the DeepSeek V4 model scaffold.
//!
//! Mirrors `Qwen3State`: wraps `GenerationStateBase` and forwards every
//! `GenerationState` method through it. V4-specific scratch (hybrid attention,
//! compressor/indexer metadata, and MoE route buffers) will live alongside
//! `decode_bufs` once Phase 2A exposes the kernel surfaces.

use anyhow::Result;

use crate::model::GenerationState;
#[cfg(feature = "cuda")]
use crate::model::generation_state::GenerationStateBase;
#[cfg(feature = "cuda")]
use crate::model::kv_cache::KVCacheDtype;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};

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
    pub(crate) reference_tokens: Vec<u32>,
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
        self.base.reset()
    }

    fn reset_for_warmup_clear(&mut self) -> Result<()> {
        self.reference_tokens.clear();
        self.base.reset()
    }

    fn truncate_to(&mut self, len: usize) -> Result<()> {
        self.reference_tokens.truncate(len);
        self.base.truncate_to(len)
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

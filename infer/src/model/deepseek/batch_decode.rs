//! DeepSeek V4 batched-decode scratch buffers.
//!
//! Tracks scheduler batch capacity for DSv4 decode. The model currently uses a
//! sequential per-slot decode fallback for B>1; vectorized V4 decode kernels
//! will add real scratch buffers here.

use anyhow::{Result, ensure};

use crate::model::DecodeContextOps;
use crate::model::kv_cache::KVFormat;
use cuda_kernels::prelude::{DeviceContext, PagedKVPool};

/// Pre-allocated decode context. Stub: kernel-shaped fields land alongside the
/// V4 vectorized decode kernels.
///
/// Public so the `ModelForward::DecodeContext` associated type (a `pub`
/// surface on the trait) does not leak a private name.
pub struct DeepseekBatchDecodeBuffers {
    max_batch_size: usize,
}

impl DeepseekBatchDecodeBuffers {
    /// Allocate the decode context. The current serial fallback still needs to
    /// validate scheduler batch capacity.
    pub(super) fn new(
        _ctx: &DeviceContext,
        max_batch_size: usize,
        _max_total_pages: usize,
    ) -> Result<Self> {
        ensure!(
            max_batch_size > 0,
            "DeepSeek V4 decode context needs batch capacity"
        );
        Ok(Self { max_batch_size })
    }
}

impl DecodeContextOps for DeepseekBatchDecodeBuffers {
    fn upload_token_ids(&mut self, _ctx: &DeviceContext, tokens: &[u32]) -> Result<()> {
        ensure!(
            tokens.len() <= self.max_batch_size,
            "DeepSeek V4 Phase 2A.0 decode batch {} exceeds context capacity {}",
            tokens.len(),
            self.max_batch_size
        );
        Ok(())
    }

    fn update_metadata(
        &mut self,
        _ctx: &DeviceContext,
        _pool: &PagedKVPool,
        slot_indices: &[usize],
    ) -> Result<bool> {
        ensure!(
            slot_indices.len() <= self.max_batch_size,
            "DeepSeek V4 Phase 2A.0 slot batch {} exceeds context capacity {}",
            slot_indices.len(),
            self.max_batch_size
        );
        Ok(false)
    }

    fn plan_attention(
        &mut self,
        _ctx: &DeviceContext,
        batch_size: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _page_size: usize,
        _head_dim: usize,
        _kv_format: KVFormat,
    ) -> Result<()> {
        ensure!(
            batch_size <= self.max_batch_size,
            "DeepSeek V4 Phase 2A.0 attention batch {batch_size} exceeds context capacity {}",
            self.max_batch_size
        );
        Ok(())
    }

    fn set_batch_size(&mut self, _bs: usize) {}

    fn invalidate_graph_cache(&mut self, _batch_size: usize) {}
}

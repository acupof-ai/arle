//! DeepSeek batched-decode scratch buffers.
//!
//! Mirrors `qwen3::batch_decode::BatchDecodeBuffers` once the MLA decode kernel
//! exposes its required scratch shape. Until then the type is an empty marker
//! that satisfies the `ModelForward::DecodeContext` associated type.

use anyhow::Result;

use crate::model::DecodeContextOps;
use crate::model::kv_cache::KVFormat;
use cuda_kernels::prelude::{DeviceContext, PagedKVPool};

/// Pre-allocated buffers for batched decode. Stub: kernel-shaped fields land
/// alongside the MLA decode kernel.
pub(super) struct DeepseekBatchDecodeBuffers;

impl DeepseekBatchDecodeBuffers {
    /// Allocate the decode context. Returns an empty marker until the MLA
    /// decode kernel lands.
    pub(super) fn new(
        _ctx: &DeviceContext,
        _max_batch_size: usize,
        _max_total_pages: usize,
    ) -> Result<Self> {
        Ok(Self)
    }
}

impl DecodeContextOps for DeepseekBatchDecodeBuffers {
    fn upload_token_ids(&mut self, _ctx: &DeviceContext, _tokens: &[u32]) -> Result<()> {
        todo!("MLA kernel — see docs/plans/2026-05-01-mla-kernel-design.md")
    }

    fn update_metadata(
        &mut self,
        _ctx: &DeviceContext,
        _pool: &PagedKVPool,
        _slot_indices: &[usize],
    ) -> Result<bool> {
        todo!("MLA kernel — see docs/plans/2026-05-01-mla-kernel-design.md")
    }

    fn plan_attention(
        &mut self,
        _ctx: &DeviceContext,
        _batch_size: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _page_size: usize,
        _head_dim: usize,
        _kv_format: KVFormat,
    ) -> Result<()> {
        todo!("MLA kernel — see docs/plans/2026-05-01-mla-kernel-design.md")
    }

    fn set_batch_size(&mut self, _bs: usize) {}

    fn invalidate_graph_cache(&mut self, _batch_size: usize) {}
}

//! DeepSeek V4 batched-decode scratch buffers.
//!
//! Tracks scheduler batch capacity for DSv4 decode. The model currently uses a
//! sequential per-slot decode fallback for B>1; vectorized V4 decode kernels
//! will add real scratch buffers here.

use anyhow::{Result, ensure};

use crate::model::DecodeContextOps;
use crate::model::kv_cache::KVFormat;
use cuda_kernels::prelude::{DeviceContext, PagedKVPool};
use cuda_kernels::tensor::CudaAllocTraceExt;
use cudarc::driver::{CudaSlice, DevicePtrMut};

/// MODEL1 FlashMLA sparse-FP8 KV block byte size — `page_block_size (64) ×
/// bytes_per_token (584)` = 37376 B per block per layer. Kept here (instead of
/// re-importing the private `weights.rs` constant) so the shared-pool sizing is
/// readable at its allocation site; both must stay in lock-step with the
/// `DSV4_FLASHMLA_MODEL1_*` constants in `weights.rs` (compile-time `const`s, no
/// drift across a single build).
const DSV4_FLASHMLA_MODEL1_BLOCK_BYTES: usize = 64 * 584;

/// Pre-allocated decode context.
///
/// Optionally owns the **shared persistent FP8 KV pool** that backs FlashMLA
/// sparse decode (Phase D-4) — but ONLY when `ARLE_DSV4_SHARED_KV_POOL=1`. When
/// the env knob is OFF (the default), `fp8_kv_pool` stays `None` and the model
/// uses the per-(slot, layer) lazy allocation (`ensure_dsv4_flashmla_fp8_kv_pool`
/// in `weights.rs`), byte-identical to the path shipped on `main`.
///
/// When ON, the pool is allocated once at `create_decode_context` sized for
/// `num_slots × layers × slot_blocks × block_bytes` so every concurrent
/// sequence gets a fixed, budgeted sub-range — replacing the per-state lazy
/// allocation that OOMed at c≥8 (each of `num_slots × layers` pools was a
/// separate unbudgeted `alloc_zeros`, and the compressed sub-pool was sized to
/// the full `max_position_embeddings` capacity).
///
/// Slot `s`, layer `l` owns the contiguous byte range
/// `[(s*layers + l) * slot_layer_bytes, (s*layers + l + 1) * slot_layer_bytes)`
/// where `slot_layer_bytes = slot_blocks * block_bytes`. The per-row pack/decode
/// logic is byte-identical to the per-state pool — only the base device pointer
/// differs (the slot's sub-range start instead of an owned buffer's byte 0).
///
/// Public so the `ModelForward::DecodeContext` associated type (a `pub` surface
/// on the trait) does not leak a private name.
pub struct DeepseekBatchDecodeBuffers {
    max_batch_size: usize,
    /// Shared owning FP8 KV pool, or `None` when `ARLE_DSV4_SHARED_KV_POOL` is
    /// off (the default — per-state lazy pool in use) or when the FlashMLA
    /// decode env knob is off / no layers are loaded.
    fp8_kv_pool: Option<CudaSlice<u8>>,
    /// Number of `num_slots` the pool was sized for.
    fp8_kv_slots: usize,
    /// Number of `layers` the pool was sized for.
    fp8_kv_layers: usize,
    /// `slot_blocks = sw_blocks + comp_blocks` per (slot, layer) sub-range.
    fp8_kv_slot_blocks: usize,
    /// Served `max_seq_len` the pool was sized for; the per-step binding reads
    /// it back so the layout matches the allocation.
    fp8_kv_max_seq_len: usize,
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
        Ok(Self {
            max_batch_size,
            fp8_kv_pool: None,
            fp8_kv_slots: 0,
            fp8_kv_layers: 0,
            fp8_kv_slot_blocks: 0,
            fp8_kv_max_seq_len: 0,
        })
    }

    /// Record the served `max_seq_len` the shared pool was sized for.
    pub(super) fn set_fp8_kv_max_seq_len(&mut self, max_seq_len: usize) {
        self.fp8_kv_max_seq_len = max_seq_len;
    }

    /// Served `max_seq_len` the shared pool was sized for, or `None` when the
    /// shared pool is not allocated (env knob off → per-state path in use).
    /// Used as the single "is the shared pool active?" predicate at the decode
    /// bind site, so OFF is a pure no-op.
    pub(super) fn fp8_kv_max_seq_len(&self) -> Option<usize> {
        if self.fp8_kv_pool.is_some() {
            Some(self.fp8_kv_max_seq_len)
        } else {
            None
        }
    }

    /// Byte size of the shared FP8 KV pool for the given shape. Used both by
    /// `ensure_fp8_kv_pool` and by the model's `scheduler_runtime_workspace_bytes`
    /// budget estimate so the static KV-pool sizing reserves headroom for it.
    pub(super) fn fp8_kv_pool_bytes(num_slots: usize, layers: usize, slot_blocks: usize) -> usize {
        num_slots
            .saturating_mul(layers)
            .saturating_mul(slot_blocks)
            .saturating_mul(DSV4_FLASHMLA_MODEL1_BLOCK_BYTES)
    }

    /// Allocate (or grow) the shared FP8 KV pool to cover
    /// `num_slots × layers × slot_blocks` blocks. Allocated once for the session
    /// shape; a later call with the same-or-smaller shape is a no-op (the buffer
    /// is monotonic, mirroring the per-state `ensure_*` it replaces). Only
    /// called when `ARLE_DSV4_SHARED_KV_POOL=1`.
    pub(super) fn ensure_fp8_kv_pool(
        &mut self,
        ctx: &DeviceContext,
        num_slots: usize,
        layers: usize,
        slot_blocks: usize,
    ) -> Result<()> {
        ensure!(
            num_slots > 0 && layers > 0 && slot_blocks > 0,
            "DSv4 shared FP8 KV pool sizing requires positive shape (num_slots={num_slots}, layers={layers}, slot_blocks={slot_blocks})"
        );
        let want_bytes = Self::fp8_kv_pool_bytes(num_slots, layers, slot_blocks);
        ensure!(
            want_bytes > 0,
            "DSv4 shared FP8 KV pool byte size overflow (num_slots={num_slots}, layers={layers}, slot_blocks={slot_blocks})"
        );
        let need_grow = self
            .fp8_kv_pool
            .as_ref()
            .is_none_or(|buf| buf.len() < want_bytes)
            || self.fp8_kv_slots != num_slots
            || self.fp8_kv_layers != layers
            || self.fp8_kv_slot_blocks != slot_blocks;
        if need_grow {
            self.fp8_kv_pool = Some(ctx.stream.alloc_zeros_traced::<u8>(want_bytes).map_err(
                |err| anyhow::anyhow!("DSv4 shared FlashMLA FP8 KV pool alloc failed: {err}"),
            )?);
            self.fp8_kv_slots = num_slots;
            self.fp8_kv_layers = layers;
            self.fp8_kv_slot_blocks = slot_blocks;
        }
        Ok(())
    }

    /// Device base pointer + byte length of the (slot, layer) sub-range inside
    /// the shared pool. The returned pointer is the start of the slot's
    /// `slot_blocks`-block window; the pack/decode kernels index block ids
    /// relative to it exactly as they did against a per-state buffer's byte 0.
    ///
    /// `slot_blocks` is the caller's current per-(slot, layer) requirement; it
    /// must be `<=` the pool's `fp8_kv_slot_blocks` (guaranteed because the
    /// caller sizes the pool from the same `(sw_blocks, comp_blocks)` formula
    /// just before binding).
    pub(super) fn fp8_kv_slot_layer_view(
        &mut self,
        ctx: &DeviceContext,
        slot_idx: usize,
        layer_idx: usize,
        slot_blocks: usize,
    ) -> Result<(u64, usize)> {
        ensure!(
            slot_idx < self.fp8_kv_slots,
            "DSv4 FP8 KV pool slot {slot_idx} out of range for {} slots",
            self.fp8_kv_slots
        );
        ensure!(
            layer_idx < self.fp8_kv_layers,
            "DSv4 FP8 KV pool layer {layer_idx} out of range for {} layers",
            self.fp8_kv_layers
        );
        ensure!(
            slot_blocks <= self.fp8_kv_slot_blocks,
            "DSv4 FP8 KV pool slot_blocks {slot_blocks} exceeds pool capacity {}",
            self.fp8_kv_slot_blocks
        );
        let slot_layer_bytes = self.fp8_kv_slot_blocks * DSV4_FLASHMLA_MODEL1_BLOCK_BYTES;
        let byte_offset = (slot_idx * self.fp8_kv_layers + layer_idx) * slot_layer_bytes;
        let view_bytes = slot_blocks * DSV4_FLASHMLA_MODEL1_BLOCK_BYTES;
        let pool = self
            .fp8_kv_pool
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("DSv4 shared FP8 KV pool not allocated"))?;
        ensure!(
            byte_offset + view_bytes <= pool.len(),
            "DSv4 FP8 KV pool sub-range [{byte_offset}, {}) exceeds pool len {}",
            byte_offset + view_bytes,
            pool.len()
        );
        let (base_ptr, _g) = pool.device_ptr_mut(&ctx.stream);
        Ok((base_ptr + byte_offset as u64, view_bytes))
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

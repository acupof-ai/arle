//! Token-level KV Cache Pool — FlashInfer-compatible token-granularity KV storage.
//!
//! Instead of fixed-size pages of N tokens, every token gets its own slot index
//! in a pre-allocated pool (like SGLang's `TokenToKVPool`). This simplifies
//! bookkeeping: no page tables, no last_page_len calculations, no partial pages.
//!
//! For FlashInfer compatibility we use `page_size = 1`, so each "page" is one
//! token. The pool buffers use NHD layout:
//!   `[max_total_tokens, num_kv_heads * head_dim]` row-major bf16 per layer.
//!
//! Token at pool index `idx`, head `h`, dim `d`:
//!   offset = `idx * kv_dim + h * head_dim + d`

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaSlice, DevicePtr};
use log::info;

use super::tensor::DeviceContext;
use crate::model::kv_cache::{KVCacheDtype, KVFormat};

/// Token-level KV cache pool — shared across all request slots.
///
/// Storage is format-aware via `KVFormat`:
/// - `BF16`: `k_data`/`v_data` are `CudaSlice<u8>` holding bf16 (2 bytes/elem)
/// - `FP8E4M3`: `k_data`/`v_data` hold FP8 E4M3 (1 byte/elem), no scales
/// - `INT8`: `k_data`/`v_data` hold int8 (1 byte/elem), + `k_scales`/`v_scales`
///
/// For FP8/INT8, a shared bf16 working buffer (1 layer) is used as the write
/// target for `decode_prep_paged`, which outputs bf16. After the prep kernel,
/// new tokens are quantized from the working buffer into the pool.
pub struct TokenKVPool {
    /// K data per layer: raw bytes, layout `[max_total_tokens, kv_dim]` × bytes_per_elem
    k_data: Vec<CudaSlice<u8>>,
    /// V data per layer: same layout
    v_data: Vec<CudaSlice<u8>>,
    /// Per-head per-token f32 scales (INT8 only). `[max_total_tokens, num_kv_heads]`
    k_scales: Vec<CudaSlice<f32>>,
    v_scales: Vec<CudaSlice<f32>>,
    /// Shared bf16 working buffers (1 layer, for decode_prep write target).
    /// Only allocated when format != BF16.
    k_work: Option<CudaSlice<u8>>,
    v_work: Option<CudaSlice<u8>>,
    /// Workspace for split-KV fused-dequant attention (INT8 only).
    pub(crate) int8_attn_workspace: Option<CudaSlice<u8>>,
    pub(crate) int8_attn_workspace_bytes: usize,
    /// Per-head per-token f16 norms (TurboQuant only). `[max_total_tokens, num_kv_heads]`
    pub(crate) k_norms: Vec<CudaSlice<u16>>,
    pub(crate) v_norms: Vec<CudaSlice<u16>>,
    /// TurboQuant per-layer state: rotation matrices + codebook (K and V).
    /// Only populated when format is TurboQuant.
    pub(crate) tq_k_state: Option<crate::model::turboquant_state::TurboQuantLayerState>,
    pub(crate) tq_v_state: Option<crate::model::turboquant_state::TurboQuantLayerState>,

    /// Free token slot indices (stack-based allocator, LIFO).
    free_slots: Vec<u32>,

    /// Per-request token mappings: `token_indices[slot][i]` = physical pool index
    /// for logical position `i` of the request occupying that slot.
    token_indices: Vec<Vec<u32>>,
    /// Monotonic slot epoch bumped whenever a slot is released.
    /// Lets decode metadata distinguish "same slot index, different request".
    slot_epochs: Vec<u64>,

    // Config
    pub format: KVFormat,
    /// Legacy compat — maps to format.
    pub dtype: KVCacheDtype,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_total_tokens: usize,
    pub num_slots: usize,
    /// `num_kv_heads * head_dim` — stride for one token row in the pool buffer.
    pub kv_dim: usize,
}

/// FlashInfer-compatible metadata for a batch of requests.
///
/// With `page_size = 1`:
/// - `indptr[i+1] - indptr[i]` = number of tokens (= pages) for request `i`
/// - `indices` = concatenated physical pool indices for all requests
/// - `last_page_len` = all 1s (every "page" is exactly 1 token)
pub struct FlashInferMeta {
    /// Cumulative token counts: `[batch_size + 1]`
    pub indptr: Vec<i32>,
    /// Concatenated physical pool indices for the batch.
    pub indices: Vec<i32>,
    /// All 1s — each page holds exactly 1 token.
    pub last_page_len: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BudgetBreakdown {
    storage_bytes_per_token: usize,
    work_bytes_per_token: usize,
    total_bytes_per_token: usize,
    max_total_tokens: usize,
}

fn compute_budget_breakdown(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_slots: usize,
    budget_bytes: usize,
    format: KVFormat,
) -> BudgetBreakdown {
    let kv_dim = num_kv_heads * head_dim;
    let bpe = format.bytes_per_element();
    let scale_bytes_per_token = if format.has_scales() {
        num_kv_heads * 4 * 2 // f32 per-head, K+V
    } else {
        0
    };
    let norm_bytes_per_token = if format.has_norms() {
        num_kv_heads * 2 * 2 // f16 per-head, K+V
    } else {
        0
    };
    let data_bytes_per_token = kv_dim * bpe * 2; // K+V
    let storage_bytes_per_token =
        (data_bytes_per_token + scale_bytes_per_token + norm_bytes_per_token) * num_layers;
    let work_bytes_per_token = if format.needs_work_buffer() {
        kv_dim * 2 * 2 // K+V bf16 working buffers for one layer
    } else {
        0
    };
    let total_bytes_per_token = storage_bytes_per_token + work_bytes_per_token;
    let max_total_tokens = if total_bytes_per_token > 0 {
        (budget_bytes / total_bytes_per_token).max(num_slots)
    } else {
        0
    };

    BudgetBreakdown {
        storage_bytes_per_token,
        work_bytes_per_token,
        total_bytes_per_token,
        max_total_tokens,
    }
}

impl TokenKVPool {
    /// Create a new token-level KV pool.
    ///
    /// `budget_bytes` controls how much GPU memory to allocate for the pool.
    /// `max_total_tokens` is derived from the budget: all memory is allocated
    /// up-front at construction time.
    pub fn new(
        ctx: &DeviceContext,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_slots: usize,
        budget_bytes: usize,
        dtype: KVCacheDtype,
    ) -> Result<Self> {
        // Map legacy KVCacheDtype to KVFormat.
        let format = match dtype {
            KVCacheDtype::BF16 => KVFormat::BF16,
            KVCacheDtype::INT8 => KVFormat::INT8,
        };
        Self::with_format(
            ctx,
            num_layers,
            num_kv_heads,
            head_dim,
            num_slots,
            budget_bytes,
            format,
        )
    }

    /// Create a new token-level KV pool with explicit format.
    pub fn with_format(
        ctx: &DeviceContext,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_slots: usize,
        budget_bytes: usize,
        format: KVFormat,
    ) -> Result<Self> {
        let kv_dim = num_kv_heads * head_dim;
        let bpe = format.bytes_per_element();
        let budget = compute_budget_breakdown(
            num_layers,
            num_kv_heads,
            head_dim,
            num_slots,
            budget_bytes,
            format,
        );
        let max_total_tokens = budget.max_total_tokens;

        info!(
            "TokenKVPool: {} max tokens, {:.1} GB for {} layers \
             ({} kv_heads x {} head_dim, kv_dim={}, format={:?})",
            max_total_tokens,
            (max_total_tokens as u64 * budget.total_bytes_per_token as u64) as f64 / 1e9,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dim,
            format,
        );

        let pool_bytes_per_layer = max_total_tokens * kv_dim * bpe;
        let scale_elements = max_total_tokens * num_kv_heads;

        let mut k_data = Vec::new();
        let mut v_data = Vec::new();
        let mut k_scales = Vec::new();
        let mut v_scales = Vec::new();
        let mut k_norms = Vec::new();
        let mut v_norms = Vec::new();
        let mut k_work = None;
        let mut v_work = None;

        if pool_bytes_per_layer > 0 {
            // Data buffers (all formats)
            for _ in 0..num_layers {
                k_data.push(
                    ctx.stream
                        .alloc_zeros::<u8>(pool_bytes_per_layer)
                        .map_err(|e| anyhow!("TokenKVPool K data alloc failed: {e}"))?,
                );
                v_data.push(
                    ctx.stream
                        .alloc_zeros::<u8>(pool_bytes_per_layer)
                        .map_err(|e| anyhow!("TokenKVPool V data alloc failed: {e}"))?,
                );
            }

            // Scale buffers (INT8 only)
            if format.has_scales() {
                for _ in 0..num_layers {
                    k_scales.push(
                        ctx.stream
                            .alloc_zeros::<f32>(scale_elements)
                            .map_err(|e| anyhow!("TokenKVPool K scales alloc failed: {e}"))?,
                    );
                    v_scales.push(
                        ctx.stream
                            .alloc_zeros::<f32>(scale_elements)
                            .map_err(|e| anyhow!("TokenKVPool V scales alloc failed: {e}"))?,
                    );
                }
            }

            // Norm buffers (TurboQuant only): f16 per-head per-token
            if format.has_norms() {
                for _ in 0..num_layers {
                    k_norms.push(
                        ctx.stream
                            .alloc_zeros::<u16>(scale_elements)
                            .map_err(|e| anyhow!("TokenKVPool K norms alloc failed: {e}"))?,
                    );
                    v_norms.push(
                        ctx.stream
                            .alloc_zeros::<u16>(scale_elements)
                            .map_err(|e| anyhow!("TokenKVPool V norms alloc failed: {e}"))?,
                    );
                }
            }

            // Working buffer (FP8/INT8: 1-layer bf16 for decode_prep write target)
            if format.needs_work_buffer() {
                let work_bytes = max_total_tokens * kv_dim * 2; // bf16 = 2 bytes
                k_work = Some(
                    ctx.stream
                        .alloc_zeros::<u8>(work_bytes)
                        .map_err(|e| anyhow!("TokenKVPool K work alloc failed: {e}"))?,
                );
                v_work = Some(
                    ctx.stream
                        .alloc_zeros::<u8>(work_bytes)
                        .map_err(|e| anyhow!("TokenKVPool V work alloc failed: {e}"))?,
                );
            }

            info!(
                "TokenKVPool {format:?}: data={:.1}MB/layer scales={:.1}MB/layer working={:.1}MB",
                (pool_bytes_per_layer * 2) as f64 / 1e6,
                if format.has_scales() {
                    (scale_elements * 4 * 2) as f64 / 1e6
                } else {
                    0.0
                },
                (max_total_tokens * budget.work_bytes_per_token) as f64 / 1e6,
            );
        }

        let free_slots: Vec<u32> = (0..max_total_tokens as u32).rev().collect();
        let token_indices = vec![Vec::new(); num_slots];
        let slot_epochs = vec![0; num_slots];

        // Quantized split-KV attention workspace.
        // FP8 reuses the same two-phase reduction scratch layout as INT8.
        let num_splits = 32;
        let (int8_attn_workspace, int8_attn_workspace_bytes) =
            if matches!(format, KVFormat::INT8 | KVFormat::FP8E4M3) && pool_bytes_per_layer > 0 {
                let ws_bytes = crate::ops::kv_quant::decode_attention_int8_workspace_bytes(
                    num_slots,
                    num_kv_heads * (head_dim / 128).max(1) * 4, // approximate max q_heads
                    head_dim,
                    num_splits,
                );
                // Use a reasonable upper bound: max_batch * max_heads * head_dim * num_splits * 3 floats
                let ws_bytes_safe = num_splits * num_slots * num_kv_heads * 4 * (head_dim + 2) * 4;
                let ws_bytes = ws_bytes.max(ws_bytes_safe);
                let ws = ctx
                    .stream
                    .alloc_zeros::<u8>(ws_bytes)
                    .map_err(|e| anyhow!("Quantized attn workspace alloc failed: {e}"))?;
                (Some(ws), ws_bytes)
            } else {
                (None, 0)
            };

        // TurboQuant state: rotation matrices + codebook
        let (tq_k_state, tq_v_state) = if let KVFormat::TurboQuant { key_bits, val_bits } = format {
            use crate::model::turboquant_state::TurboQuantLayerState;
            let k_state = TurboQuantLayerState::new(ctx, num_layers, head_dim, key_bits, 42)?;
            let v_state = TurboQuantLayerState::new(ctx, num_layers, head_dim, val_bits, 137)?;
            (Some(k_state), Some(v_state))
        } else {
            (None, None)
        };

        // Legacy dtype mapping
        let dtype = match format {
            KVFormat::BF16 => KVCacheDtype::BF16,
            KVFormat::FP8E4M3 | KVFormat::INT8 | KVFormat::TurboQuant { .. } => KVCacheDtype::INT8,
        };

        Ok(Self {
            k_data,
            v_data,
            k_scales,
            v_scales,
            k_work,
            v_work,
            int8_attn_workspace,
            int8_attn_workspace_bytes,
            k_norms,
            v_norms,
            tq_k_state,
            tq_v_state,
            free_slots,
            token_indices,
            slot_epochs,
            format,
            dtype,
            num_layers,
            num_kv_heads,
            head_dim,
            max_total_tokens,
            num_slots,
            kv_dim,
        })
    }

    /// Allocate `count` token slots for the request in `slot`.
    ///
    /// Returns the newly allocated physical pool indices. These are appended to
    /// the slot's token_indices list.
    pub fn alloc_tokens(&mut self, slot: usize, count: usize) -> Result<Vec<u32>> {
        if count > self.free_slots.len() {
            return Err(anyhow!(
                "TokenKVPool: out of token slots (requested {}, available {})",
                count,
                self.free_slots.len()
            ));
        }

        let mut new_indices = Vec::with_capacity(count);
        for _ in 0..count {
            // SAFETY: we checked len >= count above.
            let idx = self
                .free_slots
                .pop()
                .expect("invariant: free_slots.len() >= count checked above");
            new_indices.push(idx);
        }
        self.token_indices[slot].extend_from_slice(&new_indices);
        Ok(new_indices)
    }

    /// Free all token slots for a request, returning them to the pool.
    pub fn free_slot(&mut self, slot: usize) {
        if !self.token_indices[slot].is_empty() {
            self.slot_epochs[slot] = self.slot_epochs[slot].saturating_add(1);
        }
        for &idx in &self.token_indices[slot] {
            self.free_slots.push(idx);
        }
        self.token_indices[slot].clear();
    }

    /// Get token indices for a request (physical pool indices, in logical order).
    pub fn token_indices(&self, slot: usize) -> &[u32] {
        &self.token_indices[slot]
    }

    /// Get the sequence length for a request (number of tokens allocated).
    pub fn seq_len(&self, slot: usize) -> usize {
        self.token_indices[slot].len()
    }

    /// Monotonic identifier for the current logical occupant of `slot`.
    pub fn slot_epoch(&self, slot: usize) -> u64 {
        self.slot_epochs[slot]
    }

    /// Number of free token slots remaining in the pool.
    pub fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Whether the pool has allocated buffers.
    pub fn is_active(&self) -> bool {
        !self.k_data.is_empty()
    }

    // ── Pointer accessors ──
    //
    // `k_ptr` / `v_ptr` = the "write target" for decode_prep_paged:
    //   BF16 → per-layer data buffer (also read by FlashInfer)
    //   FP8/INT8 → shared bf16 working buffer (quantized to pool after write)
    //
    // `k_data_ptr` / `v_data_ptr` = the quantized data buffer (read by attention):
    //   Used by FlashInfer FP8 and fused-dequant INT8 attention.

    /// Write-target pointer for decode_prep_paged (bf16 for all formats).
    pub fn k_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        if self.format.needs_work_buffer() {
            let (ptr, _guard) = self.k_work.as_ref().expect("k_work").device_ptr(stream);
            ptr as u64
        } else {
            let (ptr, _guard) = self.k_data[layer].device_ptr(stream);
            ptr as u64
        }
    }

    /// Write-target pointer for decode_prep_paged (bf16 for all formats).
    pub fn v_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        if self.format.needs_work_buffer() {
            let (ptr, _guard) = self.v_work.as_ref().expect("v_work").device_ptr(stream);
            ptr as u64
        } else {
            let (ptr, _guard) = self.v_data[layer].device_ptr(stream);
            ptr as u64
        }
    }

    /// Quantized K data pointer for a layer (read by attention kernels).
    pub fn k_data_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.k_data[layer].device_ptr(stream);
        ptr as u64
    }

    /// Quantized V data pointer for a layer (read by attention kernels).
    pub fn v_data_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.v_data[layer].device_ptr(stream);
        ptr as u64
    }

    /// K scales device pointer for a layer (INT8 only).
    pub fn k_scales_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.k_scales[layer].device_ptr(stream);
        ptr as u64
    }

    /// V scales device pointer for a layer (INT8 only).
    pub fn v_scales_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.v_scales[layer].device_ptr(stream);
        ptr as u64
    }

    /// K norms device pointer for a layer (TurboQuant only).
    pub fn k_norms_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.k_norms[layer].device_ptr(stream);
        ptr as u64
    }

    /// V norms device pointer for a layer (TurboQuant only).
    pub fn v_norms_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.v_norms[layer].device_ptr(stream);
        ptr as u64
    }

    /// K norms CudaSlice ref for a layer (TurboQuant only).
    pub fn k_norms_slice(&self, layer: usize) -> &CudaSlice<u16> {
        &self.k_norms[layer]
    }

    /// V norms CudaSlice ref for a layer (TurboQuant only).
    pub fn v_norms_slice(&self, layer: usize) -> &CudaSlice<u16> {
        &self.v_norms[layer]
    }

    /// K data CudaSlice ref for a layer.
    pub fn k_data_slice(&self, layer: usize) -> &CudaSlice<u8> {
        &self.k_data[layer]
    }

    /// V data CudaSlice ref for a layer.
    pub fn v_data_slice(&self, layer: usize) -> &CudaSlice<u8> {
        &self.v_data[layer]
    }

    /// K working buffer pointer (bf16, shared across layers).
    pub fn k_work_ptr(&self, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.k_work.as_ref().expect("k_work").device_ptr(stream);
        ptr as u64
    }

    /// V working buffer pointer (bf16, shared across layers).
    pub fn v_work_ptr(&self, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.v_work.as_ref().expect("v_work").device_ptr(stream);
        ptr as u64
    }

    /// Build FlashInfer-compatible metadata for a batch of slots.
    ///
    /// With `page_size = 1`:
    /// - `indptr[i+1] - indptr[i]` = token count for request `i`
    /// - `indices` = concatenated physical pool indices
    /// - `last_page_len` = all 1s
    pub fn build_flashinfer_metadata(&self, slots: &[usize]) -> FlashInferMeta {
        let mut indptr = Vec::with_capacity(slots.len() + 1);
        let mut indices = Vec::new();
        let mut last_page_len = Vec::with_capacity(slots.len());

        indptr.push(0i32);
        for &slot in slots {
            let toks = &self.token_indices[slot];
            for &idx in toks {
                indices.push(idx as i32);
            }
            let prev = *indptr
                .last()
                .expect("invariant: indptr always has at least one element (initialized with 0)");
            indptr.push(prev + toks.len() as i32);
            // page_size=1 ⇒ last_page_len is always 1 (if seq_len > 0).
            last_page_len.push(if toks.is_empty() { 0 } else { 1 });
        }

        FlashInferMeta {
            indptr,
            indices,
            last_page_len,
        }
    }

    // ── Convenience accessors that mirror the old PagedKVPool API so callers ──
    // ── can transition incrementally.                                         ──

    /// Build FlashInfer indptr array for a batch of slots.
    /// `indptr[i+1] - indptr[i]` = token count (= page count with page_size=1).
    pub fn build_indptr(&self, slots: &[usize]) -> Vec<i32> {
        let mut indptr = Vec::with_capacity(slots.len() + 1);
        indptr.push(0);
        for &slot in slots {
            let last = *indptr
                .last()
                .expect("invariant: indptr always has at least one element (initialized with 0)");
            indptr.push(last + self.token_indices[slot].len() as i32);
        }
        indptr
    }

    /// Build FlashInfer page-indices array (concatenated token pool indices).
    pub fn build_indices(&self, slots: &[usize]) -> Vec<i32> {
        let mut indices = Vec::new();
        for &slot in slots {
            for &idx in &self.token_indices[slot] {
                indices.push(idx as i32);
            }
        }
        indices
    }

    /// Build only the LAST token index per slot (for incremental GPU update).
    /// Returns B values — the most recently allocated pool index for each slot.
    pub fn build_last_indices(&self, slots: &[usize]) -> Vec<i32> {
        slots
            .iter()
            .map(|&slot| *self.token_indices[slot].last().expect("slot has no tokens") as i32)
            .collect()
    }

    /// Build FlashInfer last_page_len array — always all-1s for page_size=1.
    pub fn build_last_page_lens(&self, slots: &[usize]) -> Vec<i32> {
        slots
            .iter()
            .map(|&slot| {
                if self.token_indices[slot].is_empty() {
                    0
                } else {
                    1
                }
            })
            .collect()
    }

    /// Migrate KV data from contiguous per-slot cache into the token pool.
    ///
    /// Called after prefill completes. Copies `seq_len(slot)` tokens of K/V
    /// from each contiguous layer buffer into the corresponding token slots
    /// in the pool.
    ///
    /// The contiguous cache layout is `[max_seq_len_contiguous, kv_dim]` per layer.
    fn upload_token_indices(
        &self,
        ctx: &super::tensor::DeviceContext,
        token_indices: &[u32],
    ) -> Result<cudarc::driver::CudaSlice<i32>> {
        let token_indices_i32: Vec<i32> = token_indices.iter().map(|&p| p as i32).collect();
        ctx.stream
            .clone_htod(&token_indices_i32)
            .map_err(|e| anyhow!("H2D token_indices failed: {e}"))
    }

    fn migrate_from_contiguous_range_bf16(
        &self,
        ctx: &super::tensor::DeviceContext,
        contiguous_k_caches: &[super::tensor::DeviceVec],
        contiguous_v_caches: &[super::tensor::DeviceVec],
        max_seq_len_contiguous: usize,
        start_pos: usize,
        new_token_indices: &[u32],
        k_dst_ptr: impl Fn(usize) -> u64,
        v_dst_ptr: impl Fn(usize) -> u64,
    ) -> Result<()> {
        let token_count = new_token_indices.len();
        if token_count == 0 || self.k_data.is_empty() {
            return Ok(());
        }

        let token_indices_gpu = self.upload_token_indices(ctx, new_token_indices)?;
        let (ti_ptr, _gti) = token_indices_gpu.device_ptr(&ctx.stream);

        for layer in 0..self.num_layers.min(contiguous_k_caches.len()) {
            let (k_src_ptr, _gk) = contiguous_k_caches[layer].data.device_ptr(&ctx.stream);
            let (v_src_ptr, _gv) = contiguous_v_caches[layer].data.device_ptr(&ctx.stream);
            unsafe {
                super::ffi::kv_cache_to_paged_range_cuda(
                    k_src_ptr as *const super::ffi::Half,
                    v_src_ptr as *const super::ffi::Half,
                    k_dst_ptr(layer) as *mut super::ffi::Half,
                    v_dst_ptr(layer) as *mut super::ffi::Half,
                    ti_ptr as *const i32,
                    start_pos as i32,
                    max_seq_len_contiguous as i32,
                    token_count as i32,
                    self.num_kv_heads as i32,
                    self.head_dim as i32,
                    self.kv_dim as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        Ok(())
    }

    pub fn migrate_from_contiguous_range(
        &self,
        ctx: &super::tensor::DeviceContext,
        contiguous_k_caches: &[super::tensor::DeviceVec],
        contiguous_v_caches: &[super::tensor::DeviceVec],
        max_seq_len_contiguous: usize,
        start_pos: usize,
        new_token_indices: &[u32],
    ) -> Result<()> {
        self.migrate_from_contiguous_range_bf16(
            ctx,
            contiguous_k_caches,
            contiguous_v_caches,
            max_seq_len_contiguous,
            start_pos,
            new_token_indices,
            |layer| self.k_data_ptr(layer, &ctx.stream),
            |layer| self.v_data_ptr(layer, &ctx.stream),
        )
    }

    pub fn migrate_from_contiguous(
        &self,
        ctx: &super::tensor::DeviceContext,
        slot: usize,
        contiguous_k_caches: &[super::tensor::DeviceVec],
        contiguous_v_caches: &[super::tensor::DeviceVec],
        max_seq_len_contiguous: usize,
    ) -> Result<()> {
        let token_idxs = &self.token_indices[slot];
        self.migrate_from_contiguous_range(
            ctx,
            contiguous_k_caches,
            contiguous_v_caches,
            max_seq_len_contiguous,
            0,
            token_idxs,
        )
    }

    /// Migrate INT8 KV data from contiguous per-slot cache into the INT8 token pool.
    ///
    /// Copies quantized INT8 data + scales from HND contiguous layout to NHD paged
    /// layout with scale transposition.
    pub fn migrate_from_contiguous_int8_range(
        &self,
        ctx: &super::tensor::DeviceContext,
        contiguous_k_q: &[cudarc::driver::CudaSlice<i8>],
        contiguous_v_q: &[cudarc::driver::CudaSlice<i8>],
        contiguous_k_scales: &[cudarc::driver::CudaSlice<f32>],
        contiguous_v_scales: &[cudarc::driver::CudaSlice<f32>],
        max_seq_len_contiguous: usize,
        start_pos: usize,
        new_token_indices: &[u32],
    ) -> Result<()> {
        let token_count = new_token_indices.len();
        if token_count == 0 || self.k_data.is_empty() {
            return Ok(());
        }

        let token_indices_gpu = self.upload_token_indices(ctx, new_token_indices)?;
        let (ti_ptr, _gti) = token_indices_gpu.device_ptr(&ctx.stream);

        for layer in 0..self.num_layers.min(contiguous_k_q.len()) {
            let (k_src_ptr, _gk) = contiguous_k_q[layer].device_ptr(&ctx.stream);
            let (v_src_ptr, _gv) = contiguous_v_q[layer].device_ptr(&ctx.stream);
            let (ks_src_ptr, _gks) = contiguous_k_scales[layer].device_ptr(&ctx.stream);
            let (vs_src_ptr, _gvs) = contiguous_v_scales[layer].device_ptr(&ctx.stream);
            let (k_dst_ptr, _gkd) = self.k_data[layer].device_ptr(&ctx.stream);
            let (v_dst_ptr, _gvd) = self.v_data[layer].device_ptr(&ctx.stream);
            let (ks_dst_ptr, _gksd) = self.k_scales[layer].device_ptr(&ctx.stream);
            let (vs_dst_ptr, _gvsd) = self.v_scales[layer].device_ptr(&ctx.stream);

            unsafe {
                super::ffi::kv_cache_to_paged_int8_range_cuda(
                    k_src_ptr as *const i8,
                    v_src_ptr as *const i8,
                    ks_src_ptr as *const f32,
                    vs_src_ptr as *const f32,
                    k_dst_ptr as *mut i8,
                    v_dst_ptr as *mut i8,
                    ks_dst_ptr as *mut f32,
                    vs_dst_ptr as *mut f32,
                    ti_ptr as *const i32,
                    start_pos as i32,
                    max_seq_len_contiguous as i32,
                    token_count as i32,
                    self.num_kv_heads as i32,
                    self.head_dim as i32,
                    self.kv_dim as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        Ok(())
    }

    pub fn migrate_from_contiguous_int8(
        &self,
        ctx: &super::tensor::DeviceContext,
        slot: usize,
        contiguous_k_q: &[cudarc::driver::CudaSlice<i8>],
        contiguous_v_q: &[cudarc::driver::CudaSlice<i8>],
        contiguous_k_scales: &[cudarc::driver::CudaSlice<f32>],
        contiguous_v_scales: &[cudarc::driver::CudaSlice<f32>],
        max_seq_len_contiguous: usize,
    ) -> Result<()> {
        let token_idxs = &self.token_indices[slot];
        self.migrate_from_contiguous_int8_range(
            ctx,
            contiguous_k_q,
            contiguous_v_q,
            contiguous_k_scales,
            contiguous_v_scales,
            max_seq_len_contiguous,
            0,
            token_idxs,
        )
    }

    /// Migrate BF16 contiguous KV to FP8 paged pool (quantize + scatter).
    ///
    /// Reads bf16 from contiguous HND layout, quantizes to FP8 E4M3, and
    /// scatters to NHD paged layout in a single fused kernel per layer.
    pub fn migrate_from_contiguous_fp8_range(
        &self,
        ctx: &super::tensor::DeviceContext,
        contiguous_k_caches: &[super::tensor::DeviceVec],
        contiguous_v_caches: &[super::tensor::DeviceVec],
        max_seq_len_contiguous: usize,
        start_pos: usize,
        new_token_indices: &[u32],
    ) -> Result<()> {
        let token_count = new_token_indices.len();
        if token_count == 0 || self.k_data.is_empty() {
            return Ok(());
        }

        let token_indices_gpu = self.upload_token_indices(ctx, new_token_indices)?;
        for layer in 0..self.num_layers.min(contiguous_k_caches.len()) {
            crate::ops::kv_quant::quantize_scatter_kv_fp8_range(
                ctx,
                &contiguous_k_caches[layer],
                self.k_data_ptr(layer, &ctx.stream),
                &token_indices_gpu,
                start_pos,
                max_seq_len_contiguous,
                token_count,
                self.num_kv_heads,
                self.head_dim,
                self.kv_dim,
            )?;
            crate::ops::kv_quant::quantize_scatter_kv_fp8_range(
                ctx,
                &contiguous_v_caches[layer],
                self.v_data_ptr(layer, &ctx.stream),
                &token_indices_gpu,
                start_pos,
                max_seq_len_contiguous,
                token_count,
                self.num_kv_heads,
                self.head_dim,
                self.kv_dim,
            )?;
        }

        Ok(())
    }

    pub fn migrate_from_contiguous_fp8(
        &self,
        ctx: &super::tensor::DeviceContext,
        slot: usize,
        contiguous_k_caches: &[super::tensor::DeviceVec],
        contiguous_v_caches: &[super::tensor::DeviceVec],
        max_seq_len_contiguous: usize,
    ) -> Result<()> {
        let token_idxs = &self.token_indices[slot];
        self.migrate_from_contiguous_fp8_range(
            ctx,
            contiguous_k_caches,
            contiguous_v_caches,
            max_seq_len_contiguous,
            0,
            token_idxs,
        )
    }

    pub fn migrate_from_contiguous_turboquant_range(
        &self,
        ctx: &super::tensor::DeviceContext,
        contiguous_k_caches: &[super::tensor::DeviceVec],
        contiguous_v_caches: &[super::tensor::DeviceVec],
        max_seq_len_contiguous: usize,
        start_pos: usize,
        new_token_indices: &[u32],
    ) -> Result<()> {
        let token_count = new_token_indices.len();
        if token_count == 0 || self.k_data.is_empty() {
            return Ok(());
        }

        let token_indices_gpu = self.upload_token_indices(ctx, new_token_indices)?;
        let k_state = self
            .tq_k_state
            .as_ref()
            .ok_or_else(|| anyhow!("TurboQuant K state missing"))?;
        let v_state = self
            .tq_v_state
            .as_ref()
            .ok_or_else(|| anyhow!("TurboQuant V state missing"))?;

        self.migrate_from_contiguous_range_bf16(
            ctx,
            contiguous_k_caches,
            contiguous_v_caches,
            max_seq_len_contiguous,
            start_pos,
            new_token_indices,
            |_| self.k_work_ptr(&ctx.stream),
            |_| self.v_work_ptr(&ctx.stream),
        )?;

        for layer in 0..self.num_layers.min(contiguous_k_caches.len()) {
            crate::ops::kv_turboquant::turboquant_quantize_paged_single(
                ctx,
                self.k_work_ptr(&ctx.stream),
                self.k_data_slice(layer),
                self.k_norms_slice(layer),
                &token_indices_gpu,
                k_state,
                layer,
                self.num_kv_heads,
                self.head_dim,
                token_count,
            )?;
            crate::ops::kv_turboquant::turboquant_quantize_paged_single(
                ctx,
                self.v_work_ptr(&ctx.stream),
                self.v_data_slice(layer),
                self.v_norms_slice(layer),
                &token_indices_gpu,
                v_state,
                layer,
                self.num_kv_heads,
                self.head_dim,
                token_count,
            )?;
        }

        Ok(())
    }
}

// ── Type alias for backward compatibility ──────────────────────────────────

/// Backward-compatible alias. New code should use `TokenKVPool` directly.
pub type PagedKVPool = TokenKVPool;

/// Page size is 1 for token-level pool (used by callers that pass `page_size`
/// to FlashInfer / CUDA kernels).
pub const DEFAULT_PAGE_SIZE: usize = 1;

#[cfg(test)]
mod tests {
    use super::{BudgetBreakdown, compute_budget_breakdown};
    use crate::model::kv_cache::KVFormat;

    #[test]
    fn bf16_budget_has_no_work_buffer_component() {
        let budget = compute_budget_breakdown(2, 8, 16, 4, 16_384, KVFormat::BF16);
        assert_eq!(
            budget,
            BudgetBreakdown {
                storage_bytes_per_token: 1024,
                work_bytes_per_token: 0,
                total_bytes_per_token: 1024,
                max_total_tokens: 16,
            }
        );
    }

    #[test]
    fn int8_budget_counts_work_buffer_per_token() {
        let budget = compute_budget_breakdown(2, 8, 16, 4, 16_384, KVFormat::INT8);
        assert_eq!(budget.storage_bytes_per_token, 640);
        assert_eq!(budget.work_bytes_per_token, 512);
        assert_eq!(budget.total_bytes_per_token, 1152);
        assert_eq!(budget.max_total_tokens, 14);
    }

    #[test]
    fn budget_respects_slot_floor_when_budget_is_tiny() {
        let budget = compute_budget_breakdown(2, 8, 16, 32, 1, KVFormat::FP8E4M3);
        assert_eq!(budget.max_total_tokens, 32);
    }

    #[test]
    fn slot_epoch_advances_only_when_a_live_slot_is_released() {
        let mut epochs = vec![0_u64, 0_u64];
        let mut token_indices = vec![vec![10_u32, 11_u32], Vec::<u32>::new()];
        let mut free_slots = Vec::<u32>::new();

        if !token_indices[0].is_empty() {
            epochs[0] = epochs[0].saturating_add(1);
        }
        for &idx in &token_indices[0] {
            free_slots.push(idx);
        }
        token_indices[0].clear();

        if !token_indices[1].is_empty() {
            epochs[1] = epochs[1].saturating_add(1);
        }

        assert_eq!(epochs, vec![1, 0]);
        assert_eq!(free_slots, vec![10, 11]);
        assert!(token_indices[0].is_empty());
    }
}

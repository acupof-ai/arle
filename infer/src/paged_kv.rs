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

use crate::tensor::DeviceContext;

/// Token-level KV cache pool — shared across all request slots.
pub struct TokenKVPool {
    /// K buffers per layer: each `[max_total_tokens * kv_dim]` bf16
    pub(crate) k_buffers: Vec<CudaSlice<u16>>,
    /// V buffers per layer: each `[max_total_tokens * kv_dim]` bf16
    v_buffers: Vec<CudaSlice<u16>>,

    /// Free token slot indices (stack-based allocator, LIFO).
    free_slots: Vec<u32>,

    /// Per-request token mappings: `token_indices[slot][i]` = physical pool index
    /// for logical position `i` of the request occupying that slot.
    token_indices: Vec<Vec<u32>>,

    // Config
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
    ) -> Result<Self> {
        let kv_dim = num_kv_heads * head_dim;
        let bytes_per_token_per_layer = kv_dim * 2; // bf16 = 2 bytes
        let bytes_per_token = bytes_per_token_per_layer * num_layers * 2; // K + V
        let max_total_tokens = if bytes_per_token > 0 {
            budget_bytes / bytes_per_token
        } else {
            0
        };
        // At least 1 token slot per request slot (may be 0-capacity in stub mode).
        let max_total_tokens = max_total_tokens.max(num_slots);

        info!(
            "TokenKVPool: {} max tokens, {:.1} GB for {} layers \
             ({} kv_heads x {} head_dim, kv_dim={})",
            max_total_tokens,
            (max_total_tokens as u64 * bytes_per_token as u64) as f64 / 1e9,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dim,
        );

        // Allocate K and V buffers per layer (skip if budget is 0 — stub mode).
        let pool_elements = max_total_tokens * kv_dim;
        let mut k_buffers = Vec::with_capacity(num_layers);
        let mut v_buffers = Vec::with_capacity(num_layers);

        if pool_elements > 0 {
            for _ in 0..num_layers {
                let k: CudaSlice<u16> = ctx
                    .stream
                    .alloc_zeros(pool_elements)
                    .map_err(|e| anyhow!("TokenKVPool K alloc failed: {e}"))?;
                let v: CudaSlice<u16> = ctx
                    .stream
                    .alloc_zeros(pool_elements)
                    .map_err(|e| anyhow!("TokenKVPool V alloc failed: {e}"))?;
                k_buffers.push(k);
                v_buffers.push(v);
            }
        }

        // Initialize free list: all token slots free, highest indices first (LIFO).
        let free_slots: Vec<u32> = (0..max_total_tokens as u32).rev().collect();

        // Initialize per-slot state.
        let token_indices = vec![Vec::new(); num_slots];

        Ok(Self {
            k_buffers,
            v_buffers,
            free_slots,
            token_indices,
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

    /// Number of free token slots remaining in the pool.
    pub fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Get raw K buffer device pointer for a layer (for FFI / kernel launches).
    pub fn k_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.k_buffers[layer].device_ptr(stream);
        ptr as u64
    }

    /// Get raw V buffer device pointer for a layer (for FFI / kernel launches).
    pub fn v_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.v_buffers[layer].device_ptr(stream);
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
    pub fn migrate_from_contiguous(
        &self,
        ctx: &crate::tensor::DeviceContext,
        slot: usize,
        contiguous_k_caches: &[crate::tensor::DeviceVec],
        contiguous_v_caches: &[crate::tensor::DeviceVec],
        max_seq_len_contiguous: usize,
    ) -> Result<()> {
        use cudarc::driver::DevicePtr;

        let seq_len = self.seq_len(slot);
        if seq_len == 0 || self.k_buffers.is_empty() {
            return Ok(());
        }

        let token_idxs = &self.token_indices[slot];
        if token_idxs.is_empty() {
            return Ok(());
        }

        // Upload token indices to GPU as the "page table" for the CUDA kernel.
        // With page_size=1 the existing kv_cache_to_paged_cuda kernel works:
        // each "page" is one token, stride_page = kv_dim.
        let page_indices_i32: Vec<i32> = token_idxs.iter().map(|&p| p as i32).collect();
        let page_indices_gpu: cudarc::driver::CudaSlice<i32> = ctx
            .stream
            .memcpy_stod(&page_indices_i32)
            .map_err(|e| anyhow!("H2D page_indices failed: {e}"))?;

        for layer in 0..self.num_layers.min(contiguous_k_caches.len()) {
            let (k_src_ptr, _gk) = contiguous_k_caches[layer].data.device_ptr(&ctx.stream);
            let (v_src_ptr, _gv) = contiguous_v_caches[layer].data.device_ptr(&ctx.stream);
            let (k_dst_ptr, _gkd) = self.k_buffers[layer].device_ptr(&ctx.stream);
            let (v_dst_ptr, _gvd) = self.v_buffers[layer].device_ptr(&ctx.stream);
            let (pi_ptr, _gpi) = page_indices_gpu.device_ptr(&ctx.stream);

            unsafe {
                crate::ffi::kv_cache_to_paged_cuda(
                    k_src_ptr as *const crate::ffi::Half,
                    v_src_ptr as *const crate::ffi::Half,
                    k_dst_ptr as *mut crate::ffi::Half,
                    v_dst_ptr as *mut crate::ffi::Half,
                    pi_ptr as *const i32,
                    max_seq_len_contiguous as i32,
                    seq_len as i32,
                    self.num_kv_heads as i32,
                    1, // page_size = 1
                    self.head_dim as i32,
                    self.kv_dim as i32, // stride_page = kv_dim (one token row)
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
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

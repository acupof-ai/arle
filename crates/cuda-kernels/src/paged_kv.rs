//! Paged KV cache pool — FlashInfer-compatible KV storage with runtime
//! `page_size`.
//!
//! The pool keeps **token-level sequence accounting** (`seq_len(slot)` is always
//! in logical tokens) while allocating and retaining storage in **physical
//! pages**. This matches FlashInfer's HND paged layout:
//!   `[max_pages, num_kv_heads, page_size, head_dim]`.
//!
//! BF16 / INT8 / FP8 E4M3 now all use `page_size = 16`. TurboQuant remains
//! token-granular (`page_size = 1`) until its decode and migration kernels are
//! rewritten around paged layout.

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use log::info;

use super::ffi;
use super::tensor::DeviceContext;
use crate::kv_quant::{decode_attention_int8_workspace_bytes, quantize_scatter_kv_fp8_range};
use crate::kv_turboquant::turboquant_quantize_paged_single;
use crate::kv_types::{KVCacheDtype, KVFormat};
use crate::turboquant_state::TurboQuantLayerState;

/// Paged KV cache pool — shared across all request slots.
///
/// Storage is format-aware via `KVFormat`:
/// - `BF16`: `k_data`/`v_data` are `CudaSlice<u8>` holding bf16 (2 bytes/elem)
/// - `FP8E4M3`: `k_data`/`v_data` hold FP8 E4M3 (1 byte/elem), + `k_scales`/`v_scales`
/// - `INT8`: `k_data`/`v_data` hold int8 (1 byte/elem), + `k_scales`/`v_scales`
///
/// For FP8/INT8, a shared bf16 working buffer (1 layer) is used as the write
/// target for `decode_prep_paged`, which outputs bf16. After the prep kernel,
/// new tokens are quantized from the working buffer into the pool.
pub struct TokenKVPool {
    /// K data per layer. Backing bytes are sized for
    /// `[max_total_pages, num_kv_heads, page_size, head_dim]`, which is bytewise
    /// identical to `[max_total_tokens, kv_dim]` because
    /// `max_total_tokens = max_total_pages * page_size`.
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
    pub int8_attn_workspace: Option<CudaSlice<u8>>,
    pub int8_attn_workspace_bytes: usize,
    /// Per-head per-token f16 norms (TurboQuant only). `[max_total_tokens, num_kv_heads]`
    pub k_norms: Vec<CudaSlice<u16>>,
    pub v_norms: Vec<CudaSlice<u16>>,
    /// TurboQuant per-layer state: rotation matrices + codebook (K and V).
    /// Only populated when format is TurboQuant.
    pub tq_k_state: Option<TurboQuantLayerState>,
    pub tq_v_state: Option<TurboQuantLayerState>,

    /// Free physical pages (stack-based allocator, LIFO).
    free_pages: Vec<u32>,

    /// Per-request page tables: `page_indices[slot][i]` = physical page id for
    /// logical page `i` of the request occupying that slot.
    page_indices: Vec<Vec<u32>>,
    /// Per-request logical token lengths.
    seq_lens: Vec<usize>,
    /// Monotonic slot epoch bumped whenever a slot is released.
    /// Lets decode metadata distinguish "same slot index, different request".
    slot_epochs: Vec<u64>,

    /// Per-physical-page slot attachment count.
    ///
    /// `page_attach_count[p]` is how many live slots currently include
    /// page `p` in their page table. New allocations start at 1, direct
    /// prefix attachment bumps the count, and `free_slot` drops one attachment
    /// for every page in the released slot.
    page_attach_count: Vec<u32>,

    /// Per-physical-page non-slot retain count.
    ///
    /// This is the radix / detached-page pin count: pages with
    /// `page_ref_count[p] > 0` must not be reclaimed even when no live slot
    /// currently attaches them. `retain_pages` / `release_pages` manipulate
    /// this counter; `free_slot` only returns a page to the free list once
    /// both `page_attach_count[p] == 0` and `page_ref_count[p] == 0`.
    page_ref_count: Vec<u32>,

    // Config
    pub format: KVFormat,
    /// Legacy compat — maps to format.
    pub dtype: KVCacheDtype,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_total_tokens: usize,
    pub max_total_pages: usize,
    pub page_size: usize,
    pub num_slots: usize,
    /// `num_kv_heads * head_dim` — stride for one token row in the pool buffer.
    pub kv_dim: usize,
}

/// FlashInfer-compatible metadata for a batch of requests.
///
/// With `page_size = 16`:
/// - `indptr[i+1] - indptr[i]` = number of pages for request `i`
/// - `indices` = concatenated physical pool indices for all requests
/// - `last_page_len` = tokens used in the final page for each request
pub struct FlashInferMeta {
    /// Cumulative token counts: `[batch_size + 1]`
    pub indptr: Vec<i32>,
    /// Concatenated physical pool indices for the batch.
    pub indices: Vec<i32>,
    /// Tokens used in the final page for each request.
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
    let max_total_tokens = budget_bytes
        .checked_div(total_bytes_per_token)
        .map_or(0, |tokens| tokens.max(num_slots));

    BudgetBreakdown {
        storage_bytes_per_token,
        work_bytes_per_token,
        total_bytes_per_token,
        max_total_tokens,
    }
}

impl TokenKVPool {
    fn storage_bytes_per_token(&self) -> usize {
        let data_bytes = self.kv_dim * self.format.bytes_per_element() * 2;
        let scale_bytes = if self.format.has_scales() {
            self.num_kv_heads * std::mem::size_of::<f32>() * 2
        } else {
            0
        };
        let norm_bytes = if self.format.has_norms() {
            self.num_kv_heads * std::mem::size_of::<u16>() * 2
        } else {
            0
        };
        (data_bytes + scale_bytes + norm_bytes) * self.num_layers
    }

    pub fn storage_bytes_for_tokens(&self, token_count: usize) -> usize {
        self.storage_bytes_per_token() * token_count
    }

    fn storage_bytes_per_page(&self) -> usize {
        self.storage_bytes_for_tokens(self.page_size)
    }

    fn slot_hot_tail_len(&self, slot: usize) -> usize {
        self.seq_lens[slot] % self.page_size
    }

    fn slot_last_page_len(&self, slot: usize) -> usize {
        let seq_len = self.seq_lens[slot];
        if seq_len == 0 {
            0
        } else {
            let hot_tail_len = self.slot_hot_tail_len(slot);
            if hot_tail_len == 0 {
                self.page_size
            } else {
                hot_tail_len
            }
        }
    }

    fn slot_hot_tail_page(&self, slot: usize) -> Option<u32> {
        if self.slot_hot_tail_len(slot) == 0 {
            None
        } else {
            self.page_indices[slot].last().copied()
        }
    }

    fn page_is_shared_read_only(&self, page: u32) -> bool {
        let page_idx = page as usize;
        self.page_ref_count[page_idx] > 0 || self.page_attach_count[page_idx] > 1
    }

    fn slot_shared_hot_tail_page(&self, slot: usize) -> Option<u32> {
        let hot_tail_page = self.slot_hot_tail_page(slot)?;
        self.page_is_shared_read_only(hot_tail_page)
            .then_some(hot_tail_page)
    }

    /// Extra physical pages needed to detach a shared partial tail before append.
    pub fn append_cow_pages_needed(&self, slot: usize) -> usize {
        usize::from(self.slot_shared_hot_tail_page(slot).is_some())
    }

    /// Physical pages needed to append `count` logical tokens to `slot`.
    ///
    /// This includes both the optional COW page for a radix-shared hot tail and
    /// any fresh pages required after filling the current tail.
    pub fn append_pages_needed(&self, slot: usize, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        let page_size = self.page_size.max(1);
        let hot_tail_len = self.slot_hot_tail_len(slot);
        let available_in_last_page = if hot_tail_len == 0 {
            0
        } else {
            page_size - hot_tail_len
        };
        self.append_cow_pages_needed(slot)
            + count
                .saturating_sub(available_in_last_page)
                .div_ceil(page_size)
    }

    fn recycle_page_if_unreferenced(&mut self, page: u32) {
        let page_idx = page as usize;
        if self.page_attach_count[page_idx] == 0 && self.page_ref_count[page_idx] == 0 {
            self.free_pages.push(page);
        }
    }

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
        let page_size = format.default_page_size();
        let budget = compute_budget_breakdown(
            num_layers,
            num_kv_heads,
            head_dim,
            num_slots,
            budget_bytes,
            format,
        );
        let max_total_pages = budget.max_total_tokens.div_ceil(page_size).max(num_slots);
        let max_total_tokens = max_total_pages * page_size;

        info!(
            "TokenKVPool: {} max tokens ({} pages @ page_size={}), {:.1} GB for {} layers \
             ({} kv_heads x {} head_dim, kv_dim={}, format={:?})",
            max_total_tokens,
            max_total_pages,
            page_size,
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

            // Scale buffers (FP8/INT8)
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

        let free_pages: Vec<u32> = (0..max_total_pages as u32).rev().collect();
        let page_indices = vec![Vec::new(); num_slots];
        let seq_lens = vec![0; num_slots];
        let slot_epochs = vec![0; num_slots];
        let page_attach_count = vec![0_u32; max_total_pages];
        let page_ref_count = vec![0_u32; max_total_pages];

        // Quantized split-KV attention workspace.
        // FP8 reuses the same two-phase reduction scratch layout as INT8.
        let num_splits = 32;
        let (int8_attn_workspace, int8_attn_workspace_bytes) =
            if matches!(format, KVFormat::INT8 | KVFormat::FP8E4M3) && pool_bytes_per_layer > 0 {
                let ws_bytes = decode_attention_int8_workspace_bytes(
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
            free_pages,
            page_indices,
            seq_lens,
            slot_epochs,
            page_attach_count,
            page_ref_count,
            format,
            dtype,
            num_layers,
            num_kv_heads,
            head_dim,
            max_total_tokens,
            max_total_pages,
            page_size,
            num_slots,
            kv_dim,
        })
    }

    /// Allocate `count` logical tokens for the request in `slot`.
    ///
    /// Returns the newly allocated physical page ids. Existing callers mostly
    /// ignore the return value; the canonical slot state lives inside
    /// `page_indices[slot]` + `seq_lens[slot]`.
    pub fn alloc_tokens(&mut self, slot: usize, count: usize) -> Result<Vec<u32>> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let hot_tail_len = self.slot_hot_tail_len(slot);
        let available_in_last_page = if hot_tail_len == 0 {
            0
        } else {
            self.page_size - hot_tail_len
        };
        let remaining_after_fill = count.saturating_sub(available_in_last_page);
        let new_page_count = remaining_after_fill.div_ceil(self.page_size);

        if new_page_count > self.free_pages.len() {
            return Err(anyhow!(
                "TokenKVPool: out of pages (requested {} tokens / {} new pages, available {} pages)",
                count,
                new_page_count,
                self.free_pages.len()
            ));
        }

        let mut new_pages = Vec::with_capacity(new_page_count);
        for _ in 0..new_page_count {
            let idx = self
                .free_pages
                .pop()
                .expect("invariant: free_pages.len() >= new_page_count checked above");
            self.page_attach_count[idx as usize] = 1;
            new_pages.push(idx);
        }
        self.page_indices[slot].extend_from_slice(&new_pages);
        self.seq_lens[slot] += count;
        Ok(new_pages)
    }

    /// Allocate detached physical pages that are not yet owned by any slot.
    ///
    /// This is the minimal pool primitive needed by the session-restore path:
    /// restored blocks must reserve stable physical pages before a live slot
    /// claims them.
    pub fn alloc_detached_pages(&mut self, count: usize) -> Result<Vec<u32>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if count > self.free_pages.len() {
            return Err(anyhow!(
                "TokenKVPool: out of pages (requested {count}, available {} pages)",
                self.free_pages.len()
            ));
        }

        let mut new_pages = Vec::with_capacity(count);
        for _ in 0..count {
            let idx = self
                .free_pages
                .pop()
                .expect("invariant: free_pages.len() >= count checked above");
            self.page_ref_count[idx as usize] = 1;
            new_pages.push(idx);
        }
        Ok(new_pages)
    }

    /// Attach already-live pages to an empty slot.
    ///
    /// Used by the CUDA scheduler's GPU-prefix admission path: the radix owns
    /// the retained pages, and a fresh slot borrows them as its prefix KV
    /// table before suffix prefill / decode appends new tokens.
    ///
    /// Borrowed full pages are sealed shared prefix blocks. If `token_count`
    /// leaves the final page partial, that borrowed frontier is a read-only hot
    /// tail and the caller must pass through
    /// [`Self::cow_tail_page_for_append`] before mutating it.
    pub fn attach_pages(&mut self, slot: usize, pages: &[u32], token_count: usize) -> Result<()> {
        if !self.page_indices[slot].is_empty() || self.seq_lens[slot] != 0 {
            return Err(anyhow!(
                "TokenKVPool::attach_pages requires an empty slot (slot={slot})"
            ));
        }
        if token_count > pages.len().saturating_mul(self.page_size) {
            return Err(anyhow!(
                "TokenKVPool::attach_pages token_count={} exceeds page capacity={}",
                token_count,
                pages.len().saturating_mul(self.page_size)
            ));
        }

        for &page in pages {
            let idx = page as usize;
            if idx >= self.max_total_pages {
                return Err(anyhow!(
                    "TokenKVPool::attach_pages page index out of bounds: {page}"
                ));
            }
            if self.page_attach_count[idx] == 0 && self.page_ref_count[idx] == 0 {
                return Err(anyhow!(
                    "TokenKVPool::attach_pages page {page} is not live in any tier"
                ));
            }
            self.page_attach_count[idx] = self.page_attach_count[idx].saturating_add(1);
        }

        self.page_indices[slot].extend_from_slice(pages);
        self.seq_lens[slot] = token_count;
        Ok(())
    }

    pub fn copy_pages_to_host(&self, ctx: &DeviceContext, pages: &[u32]) -> Result<Vec<u8>> {
        #[cfg(feature = "cuda")]
        {
            let token_bytes = self.page_size * self.kv_dim * self.format.bytes_per_element();
            let scale_len = self.page_size * self.num_kv_heads;
            let mut out = Vec::with_capacity(pages.len() * self.storage_bytes_per_page());

            for &page in pages {
                let page_idx = page as usize;
                let data_start = page_idx * token_bytes;
                let data_end = data_start + token_bytes;
                let scale_start = page_idx * scale_len;
                let scale_end = scale_start + scale_len;

                for layer in 0..self.num_layers {
                    out.extend_from_slice(
                        &ctx.stream
                            .clone_dtoh(&self.k_data[layer].slice(data_start..data_end))
                            .map_err(|e| anyhow!("paged_kv copy K page dtoh failed: {e}"))?,
                    );
                    out.extend_from_slice(
                        &ctx.stream
                            .clone_dtoh(&self.v_data[layer].slice(data_start..data_end))
                            .map_err(|e| anyhow!("paged_kv copy V page dtoh failed: {e}"))?,
                    );

                    if self.format.has_scales() {
                        for value in ctx
                            .stream
                            .clone_dtoh(&self.k_scales[layer].slice(scale_start..scale_end))
                            .map_err(|e| anyhow!("paged_kv copy K scales dtoh failed: {e}"))?
                        {
                            out.extend_from_slice(&value.to_le_bytes());
                        }
                        for value in ctx
                            .stream
                            .clone_dtoh(&self.v_scales[layer].slice(scale_start..scale_end))
                            .map_err(|e| anyhow!("paged_kv copy V scales dtoh failed: {e}"))?
                        {
                            out.extend_from_slice(&value.to_le_bytes());
                        }
                    }

                    if self.format.has_norms() {
                        for value in ctx
                            .stream
                            .clone_dtoh(&self.k_norms[layer].slice(scale_start..scale_end))
                            .map_err(|e| anyhow!("paged_kv copy K norms dtoh failed: {e}"))?
                        {
                            out.extend_from_slice(&value.to_le_bytes());
                        }
                        for value in ctx
                            .stream
                            .clone_dtoh(&self.v_norms[layer].slice(scale_start..scale_end))
                            .map_err(|e| anyhow!("paged_kv copy V norms dtoh failed: {e}"))?
                        {
                            out.extend_from_slice(&value.to_le_bytes());
                        }
                    }
                }
            }

            ctx.sync()?;
            Ok(out)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ctx;
            let _ = pages;
            Err(anyhow!(
                "PagedKVPool::copy_pages_to_host is unavailable without feature=cuda"
            ))
        }
    }

    pub fn copy_pages_from_host(
        &mut self,
        ctx: &DeviceContext,
        pages: &[u32],
        payload: &[u8],
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let token_bytes = self.page_size * self.kv_dim * self.format.bytes_per_element();
            let scale_len = self.page_size * self.num_kv_heads;
            let expected_len = pages.len() * self.storage_bytes_per_page();
            if payload.len() != expected_len {
                return Err(anyhow!(
                    "paged_kv host payload length mismatch: got {} expected {}",
                    payload.len(),
                    expected_len
                ));
            }

            let mut cursor = 0usize;
            for &page in pages {
                let page_idx = page as usize;
                let data_start = page_idx * token_bytes;
                let data_end = data_start + token_bytes;
                let scale_start = page_idx * scale_len;
                let scale_end = scale_start + scale_len;

                for layer in 0..self.num_layers {
                    let mut k_view = self.k_data[layer].slice_mut(data_start..data_end);
                    ctx.stream
                        .memcpy_htod(&payload[cursor..cursor + token_bytes], &mut k_view)
                        .map_err(|e| anyhow!("paged_kv copy K page htod failed: {e}"))?;
                    cursor += token_bytes;

                    let mut v_view = self.v_data[layer].slice_mut(data_start..data_end);
                    ctx.stream
                        .memcpy_htod(&payload[cursor..cursor + token_bytes], &mut v_view)
                        .map_err(|e| anyhow!("paged_kv copy V page htod failed: {e}"))?;
                    cursor += token_bytes;

                    if self.format.has_scales() {
                        let mut k_scales = Vec::with_capacity(scale_len);
                        for chunk in payload
                            [cursor..cursor + scale_len * std::mem::size_of::<f32>()]
                            .chunks_exact(std::mem::size_of::<f32>())
                        {
                            let bytes: [u8; 4] =
                                chunk.try_into().expect("f32 chunk size must be exact");
                            k_scales.push(f32::from_le_bytes(bytes));
                        }
                        cursor += scale_len * std::mem::size_of::<f32>();
                        let mut k_scale_view =
                            self.k_scales[layer].slice_mut(scale_start..scale_end);
                        ctx.stream
                            .memcpy_htod(&k_scales, &mut k_scale_view)
                            .map_err(|e| anyhow!("paged_kv copy K scales htod failed: {e}"))?;

                        let mut v_scales = Vec::with_capacity(scale_len);
                        for chunk in payload
                            [cursor..cursor + scale_len * std::mem::size_of::<f32>()]
                            .chunks_exact(std::mem::size_of::<f32>())
                        {
                            let bytes: [u8; 4] =
                                chunk.try_into().expect("f32 chunk size must be exact");
                            v_scales.push(f32::from_le_bytes(bytes));
                        }
                        cursor += scale_len * std::mem::size_of::<f32>();
                        let mut v_scale_view =
                            self.v_scales[layer].slice_mut(scale_start..scale_end);
                        ctx.stream
                            .memcpy_htod(&v_scales, &mut v_scale_view)
                            .map_err(|e| anyhow!("paged_kv copy V scales htod failed: {e}"))?;
                    }

                    if self.format.has_norms() {
                        let mut k_norms = Vec::with_capacity(scale_len);
                        for chunk in payload
                            [cursor..cursor + scale_len * std::mem::size_of::<u16>()]
                            .chunks_exact(std::mem::size_of::<u16>())
                        {
                            let bytes: [u8; 2] =
                                chunk.try_into().expect("u16 chunk size must be exact");
                            k_norms.push(u16::from_le_bytes(bytes));
                        }
                        cursor += scale_len * std::mem::size_of::<u16>();
                        let mut k_norm_view = self.k_norms[layer].slice_mut(scale_start..scale_end);
                        ctx.stream
                            .memcpy_htod(&k_norms, &mut k_norm_view)
                            .map_err(|e| anyhow!("paged_kv copy K norms htod failed: {e}"))?;

                        let mut v_norms = Vec::with_capacity(scale_len);
                        for chunk in payload
                            [cursor..cursor + scale_len * std::mem::size_of::<u16>()]
                            .chunks_exact(std::mem::size_of::<u16>())
                        {
                            let bytes: [u8; 2] =
                                chunk.try_into().expect("u16 chunk size must be exact");
                            v_norms.push(u16::from_le_bytes(bytes));
                        }
                        cursor += scale_len * std::mem::size_of::<u16>();
                        let mut v_norm_view = self.v_norms[layer].slice_mut(scale_start..scale_end);
                        ctx.stream
                            .memcpy_htod(&v_norms, &mut v_norm_view)
                            .map_err(|e| anyhow!("paged_kv copy V norms htod failed: {e}"))?;
                    }
                }
            }

            debug_assert_eq!(cursor, payload.len());
            ctx.sync()?;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ctx;
            let _ = pages;
            let _ = payload;
            Err(anyhow!(
                "PagedKVPool::copy_pages_from_host is unavailable without feature=cuda"
            ))
        }
    }

    #[cfg(feature = "cuda")]
    fn copy_page_device_to_device(
        &mut self,
        ctx: &DeviceContext,
        src_page: u32,
        dst_page: u32,
    ) -> Result<()> {
        fn copy_same_buffer_region<T>(
            ctx: &DeviceContext,
            buffer: &mut CudaSlice<T>,
            src_elem_start: usize,
            dst_elem_start: usize,
            elem_count: usize,
            label: &str,
        ) -> Result<()> {
            use cudarc::driver::sys::{cuMemcpyDtoDAsync_v2, cudaError_enum::CUDA_SUCCESS};

            let (base_ptr, _guard) = buffer.device_ptr_mut(&ctx.stream);
            let src_ptr = unsafe { (base_ptr as *mut T).add(src_elem_start) } as u64;
            let dst_ptr = unsafe { (base_ptr as *mut T).add(dst_elem_start) } as u64;
            let bytes = elem_count * std::mem::size_of::<T>();
            let result =
                unsafe { cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, bytes, ctx.stream.cu_stream()) };
            if result != CUDA_SUCCESS {
                return Err(anyhow!("{label} failed: {result:?}"));
            }
            Ok(())
        }

        let token_bytes = self.page_size * self.kv_dim * self.format.bytes_per_element();
        let scale_len = self.page_size * self.num_kv_heads;
        let src_data_start = src_page as usize * token_bytes;
        let dst_data_start = dst_page as usize * token_bytes;
        let src_scale_start = src_page as usize * scale_len;
        let dst_scale_start = dst_page as usize * scale_len;

        for layer in 0..self.num_layers {
            copy_same_buffer_region(
                ctx,
                &mut self.k_data[layer],
                src_data_start,
                dst_data_start,
                token_bytes,
                "paged_kv copy K page dtod",
            )?;
            copy_same_buffer_region(
                ctx,
                &mut self.v_data[layer],
                src_data_start,
                dst_data_start,
                token_bytes,
                "paged_kv copy V page dtod",
            )?;

            if self.format.has_scales() {
                copy_same_buffer_region(
                    ctx,
                    &mut self.k_scales[layer],
                    src_scale_start,
                    dst_scale_start,
                    scale_len,
                    "paged_kv copy K scales dtod",
                )?;
                copy_same_buffer_region(
                    ctx,
                    &mut self.v_scales[layer],
                    src_scale_start,
                    dst_scale_start,
                    scale_len,
                    "paged_kv copy V scales dtod",
                )?;
            }

            if self.format.has_norms() {
                copy_same_buffer_region(
                    ctx,
                    &mut self.k_norms[layer],
                    src_scale_start,
                    dst_scale_start,
                    scale_len,
                    "paged_kv copy K norms dtod",
                )?;
                copy_same_buffer_region(
                    ctx,
                    &mut self.v_norms[layer],
                    src_scale_start,
                    dst_scale_start,
                    scale_len,
                    "paged_kv copy V norms dtod",
                )?;
            }
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn detach_shared_hot_tail_page_for_append(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        shared_tail_page: u32,
    ) -> Result<()> {
        debug_assert_eq!(
            self.slot_shared_hot_tail_page(slot),
            Some(shared_tail_page),
            "detach_shared_hot_tail_page_for_append requires the live shared hot tail",
        );

        let new_page = self
            .free_pages
            .pop()
            .ok_or_else(|| anyhow!("TokenKVPool::cow_tail_page_for_append out of free pages"))?;
        self.page_attach_count[new_page as usize] = 1;
        self.copy_page_device_to_device(ctx, shared_tail_page, new_page)?;

        let old_tail_page = {
            let tail_slot_page = self.page_indices[slot]
                .last_mut()
                .expect("slot with shared hot tail must have a tail page");
            let old_tail_page = *tail_slot_page;
            *tail_slot_page = new_page;
            old_tail_page
        };
        let old_tail_idx = old_tail_page as usize;

        debug_assert!(
            self.page_attach_count[old_tail_idx] > 0,
            "detach_shared_hot_tail_page_for_append: detached page had zero slot refs"
        );
        self.page_attach_count[old_tail_idx] -= 1;
        self.recycle_page_if_unreferenced(old_tail_page);
        Ok(())
    }

    /// Detach the shared hot tail before append.
    ///
    /// Sealed full pages stay shared and read-only. The only mutable
    /// shared-prefix write path is detaching a partially-filled shared hot tail
    /// page immediately before append. Once that tail fills, the next append
    /// allocates a fresh page instead of mutating the sealed prefix in place.
    ///
    /// Returns `true` only when a shared hot tail was detached.
    pub fn cow_tail_page_for_append(&mut self, ctx: &DeviceContext, slot: usize) -> Result<bool> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ctx;
            let _ = slot;
            return Ok(false);
        }

        #[cfg(feature = "cuda")]
        {
            let Some(shared_tail_page) = self.slot_shared_hot_tail_page(slot) else {
                return Ok(false);
            };

            self.detach_shared_hot_tail_page_for_append(ctx, slot, shared_tail_page)?;
            Ok(true)
        }
    }

    /// Free all token slots for a request.
    ///
    /// Each page in the slot transitions based on its external reference
    /// count:
    /// - `page_ref_count == 0` → pushed back onto `free_slots`, reusable
    ///   by the next `alloc_tokens` call immediately
    /// - `page_ref_count > 0`  → **limbo**: the physical HBM row stays
    ///   live, but it is no longer owned by any slot. It will rejoin the
    ///   free list the next time [`Self::release_pages`] drops its refcount to
    ///   zero. This is the M2 dual-residency path: the
    ///   `crate::prefix_cache::RadixCache` on the scheduler thread holds
    ///   the refcount, and a future admission whose prompt prefix
    ///   matches those pages can read the KV data directly without
    ///   re-prefilling.
    ///
    /// Slot epoch advances as before whenever the slot had any pages,
    /// so decode metadata invalidation logic stays correct even when
    /// pages are retained in limbo.
    pub fn free_slot(&mut self, slot: usize) {
        let slot_pages = std::mem::take(&mut self.page_indices[slot]);
        if !slot_pages.is_empty() {
            self.slot_epochs[slot] = self.slot_epochs[slot].saturating_add(1);
        }
        for idx in slot_pages {
            let usize_idx = idx as usize;
            debug_assert!(
                self.page_attach_count[usize_idx] > 0,
                "free_slot: page {idx} had zero slot refs"
            );
            self.page_attach_count[usize_idx] = self.page_attach_count[usize_idx].saturating_sub(1);
            self.recycle_page_if_unreferenced(idx);
        }
        self.seq_lens[slot] = 0;
    }

    /// Truncate a live slot to `new_len` logical tokens and recycle any full
    /// trailing pages that are no longer reachable.
    pub fn truncate_slot(&mut self, slot: usize, new_len: usize) -> Result<Vec<u32>> {
        let old_len = self.seq_lens[slot];
        if new_len > old_len {
            return Err(anyhow!(
                "TokenKVPool: cannot grow slot {slot} via truncate ({new_len} > {old_len})"
            ));
        }
        if new_len == old_len {
            return Ok(Vec::new());
        }

        let keep_pages = new_len.div_ceil(self.page_size);
        let slot_pages = &mut self.page_indices[slot];
        let removed = slot_pages.split_off(keep_pages.min(slot_pages.len()));
        if !removed.is_empty() {
            self.slot_epochs[slot] = self.slot_epochs[slot].saturating_add(1);
        }
        let mut recycled = Vec::new();
        for idx in removed {
            let usize_idx = idx as usize;
            debug_assert!(
                self.page_attach_count[usize_idx] > 0,
                "truncate_slot: page {idx} had zero slot refs"
            );
            self.page_attach_count[usize_idx] = self.page_attach_count[usize_idx].saturating_sub(1);
            let before = self.free_pages.len();
            self.recycle_page_if_unreferenced(idx);
            if self.free_pages.len() > before {
                recycled.push(idx);
            }
        }
        self.seq_lens[slot] = new_len;
        Ok(recycled)
    }

    /// Bump the external reference count on each of the given pages by one.
    ///
    /// Used by the scheduler's `publish_to_prefix_cache` path: when a
    /// finished request's prompt is folded into the radix, the
    /// scheduler calls `retain_pages` on exactly the pages that are
    /// being indexed so they survive the subsequent `free_slot` call.
    ///
    /// Pages must currently be valid pool indices (`< max_total_tokens`).
    /// Calling with a page that is already in `free_slots` is safe but
    /// will not move it out — a page becomes pinned only when it is
    /// retained *before* being freed. The scheduler's ordering
    /// (`retain_pages` → `free_slot`) enforces that invariant.
    pub fn retain_pages(&mut self, pages: &[u32]) {
        for &idx in pages {
            self.page_ref_count[idx as usize] = self.page_ref_count[idx as usize].saturating_add(1);
        }
    }

    /// Decrement the external reference count on each page by one and
    /// return the set of pages whose refcount just dropped to zero.
    ///
    /// For each returned page, the caller is expected to treat the page
    /// as "just became reclaimable". The pool itself pushes those pages
    /// onto `free_slots` internally before returning so the caller does
    /// not have to worry about double-frees; the returned `Vec<u32>` is
    /// informational — scheduler logs, metrics, or radix-cache bookkeeping.
    ///
    /// Pages that still have refcount > 0 after the decrement stay in
    /// their current state (in a live slot or in limbo).
    ///
    /// Panics in debug builds if any page in `pages` has refcount 0
    /// (that would be a double-release, which signals a scheduler /
    /// radix book-keeping bug). In release builds the saturating
    /// subtraction keeps the counter at 0 silently — same conservative
    /// stance as `retain_pages`'s `saturating_add`.
    pub fn release_pages(&mut self, pages: &[u32]) -> Vec<u32> {
        let mut newly_freed = Vec::new();
        for &idx in pages {
            let usize_idx = idx as usize;
            let cur = self.page_ref_count[usize_idx];
            debug_assert!(
                cur > 0,
                "release_pages: double-release on page {idx} (refcount already 0)",
            );
            let next = cur.saturating_sub(1);
            self.page_ref_count[usize_idx] = next;
            if next == 0 {
                self.recycle_page_if_unreferenced(idx);
                newly_freed.push(idx);
            }
        }
        newly_freed
    }

    /// Number of pages currently pinned by an external reference
    /// (i.e. `page_ref_count > 0`). M2 observability: the scheduler
    /// `/v1/stats` endpoint will want this alongside `free_count` so
    /// operators can see how much of the pool is owned by the radix
    /// cache vs available for fresh allocation.
    pub fn retained_count(&self) -> usize {
        self.page_ref_count.iter().filter(|&&rc| rc > 0).count()
    }

    /// Get the page table for a request (physical page ids in logical-page order).
    pub fn page_indices(&self, slot: usize) -> &[u32] {
        &self.page_indices[slot]
    }

    /// Get the sequence length for a request (number of tokens allocated).
    pub fn seq_len(&self, slot: usize) -> usize {
        self.seq_lens[slot]
    }

    /// Monotonic identifier for the current logical occupant of `slot`.
    pub fn slot_epoch(&self, slot: usize) -> u64 {
        self.slot_epochs[slot]
    }

    /// Number of logical tokens that can still be allocated without eviction.
    ///
    /// This includes:
    /// - every token position inside completely free pages
    /// - unused tail space in each live slot's last partially-filled page
    pub fn free_count(&self) -> usize {
        let partial_capacity = self
            .seq_lens
            .iter()
            .enumerate()
            .map(|(slot, _)| {
                let hot_tail_len = self.slot_hot_tail_len(slot);
                if hot_tail_len == 0 {
                    0
                } else {
                    self.page_size - hot_tail_len
                }
            })
            .sum::<usize>();
        self.free_pages.len() * self.page_size + partial_capacity
    }

    /// Number of currently free physical pages.
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }

    fn page_span_for_token_range(
        &self,
        slot: usize,
        start_pos: usize,
        token_count: usize,
    ) -> std::ops::Range<usize> {
        let seq_len = self.seq_len(slot);
        debug_assert!(
            start_pos + token_count <= seq_len,
            "token range [{start_pos}, {}) exceeds seq_len={seq_len}",
            start_pos + token_count
        );
        let start_page = start_pos / self.page_size;
        let end_page = (start_pos + token_count).div_ceil(self.page_size);
        start_page..end_page
    }

    pub fn page_indices_for_token_range(
        &self,
        slot: usize,
        start_pos: usize,
        token_count: usize,
    ) -> &[u32] {
        let span = self.page_span_for_token_range(slot, start_pos, token_count);
        &self.page_indices[slot][span]
    }

    pub fn token_rows_for_range(
        &self,
        slot: usize,
        start_pos: usize,
        token_count: usize,
    ) -> Vec<u32> {
        let seq_len = self.seq_len(slot);
        debug_assert!(
            start_pos + token_count <= seq_len,
            "token range [{start_pos}, {}) exceeds seq_len={seq_len}",
            start_pos + token_count
        );
        let mut rows = Vec::with_capacity(token_count);
        for pos in start_pos..start_pos + token_count {
            let page_idx = self.page_indices[slot][pos / self.page_size];
            let offset = (pos % self.page_size) as u32;
            rows.push(page_idx * self.page_size as u32 + offset);
        }
        rows
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
            ptr
        } else {
            let (ptr, _guard) = self.k_data[layer].device_ptr(stream);
            ptr
        }
    }

    /// Write-target pointer for decode_prep_paged (bf16 for all formats).
    pub fn v_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        if self.format.needs_work_buffer() {
            let (ptr, _guard) = self.v_work.as_ref().expect("v_work").device_ptr(stream);
            ptr
        } else {
            let (ptr, _guard) = self.v_data[layer].device_ptr(stream);
            ptr
        }
    }

    /// Quantized K data pointer for a layer (read by attention kernels).
    pub fn k_data_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.k_data[layer].device_ptr(stream);
        ptr
    }

    /// Quantized V data pointer for a layer (read by attention kernels).
    pub fn v_data_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.v_data[layer].device_ptr(stream);
        ptr
    }

    /// K scales device pointer for a layer (FP8/INT8).
    pub fn k_scales_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.k_scales[layer].device_ptr(stream);
        ptr
    }

    /// V scales device pointer for a layer (FP8/INT8).
    pub fn v_scales_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.v_scales[layer].device_ptr(stream);
        ptr
    }

    /// K norms device pointer for a layer (TurboQuant only).
    pub fn k_norms_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.k_norms[layer].device_ptr(stream);
        ptr
    }

    /// V norms device pointer for a layer (TurboQuant only).
    pub fn v_norms_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.v_norms[layer].device_ptr(stream);
        ptr
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
        ptr
    }

    /// V working buffer pointer (bf16, shared across layers).
    pub fn v_work_ptr(&self, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.v_work.as_ref().expect("v_work").device_ptr(stream);
        ptr
    }

    /// Build FlashInfer-compatible metadata for a batch of slots.
    pub fn build_flashinfer_metadata(&self, slots: &[usize]) -> FlashInferMeta {
        let mut indptr = Vec::with_capacity(slots.len() + 1);
        let mut indices = Vec::new();
        let mut last_page_len = Vec::with_capacity(slots.len());

        indptr.push(0i32);
        for &slot in slots {
            let pages = &self.page_indices[slot];
            for &idx in pages {
                indices.push(idx as i32);
            }
            let prev = *indptr
                .last()
                .expect("invariant: indptr always has at least one element (initialized with 0)");
            indptr.push(prev + pages.len() as i32);
            last_page_len.push(self.slot_last_page_len(slot) as i32);
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
    /// `indptr[i+1] - indptr[i]` = page count for request `i`.
    pub fn build_indptr(&self, slots: &[usize]) -> Vec<i32> {
        let mut indptr = Vec::with_capacity(slots.len() + 1);
        indptr.push(0);
        for &slot in slots {
            let last = *indptr
                .last()
                .expect("invariant: indptr always has at least one element (initialized with 0)");
            indptr.push(last + self.page_indices[slot].len() as i32);
        }
        indptr
    }

    /// Build FlashInfer page-indices array (concatenated physical page ids).
    pub fn build_indices(&self, slots: &[usize]) -> Vec<i32> {
        let mut indices = Vec::new();
        for &slot in slots {
            for &idx in &self.page_indices[slot] {
                indices.push(idx as i32);
            }
        }
        indices
    }

    /// Build the token-row index of the newest token in each slot.
    ///
    /// For `page_size=1` this is identical to the last physical page id. For
    /// paged quantized pools (`page_size=16`), the quantize-single fast path
    /// needs the exact token row, not just the page id.
    pub fn build_last_indices(&self, slots: &[usize]) -> Vec<i32> {
        slots
            .iter()
            .map(|&slot| {
                let seq_len = self.seq_lens[slot];
                debug_assert!(seq_len > 0, "slot has no live tokens");
                let last_pos = seq_len - 1;
                let page = self.page_indices[slot][last_pos / self.page_size];
                (page as usize * self.page_size + (last_pos % self.page_size)) as i32
            })
            .collect()
    }

    /// Build FlashInfer last_page_len array.
    pub fn build_last_page_lens(&self, slots: &[usize]) -> Vec<i32> {
        slots
            .iter()
            .map(|&slot| self.slot_last_page_len(slot) as i32)
            .collect()
    }

    /// Build packed decode metadata for quantized page-aware kernels:
    /// `[page_indptr..., last_page_len...]`.
    ///
    /// `page_indptr` is length `batch + 1` and indexes into the page-granular
    /// `kv_indices` array. `last_page_len` is length `batch` and records how
    /// many logical tokens in the final page are valid.
    pub fn build_quantized_decode_indptr(&self, slots: &[usize]) -> Vec<i32> {
        let mut packed = self.build_indptr(slots);
        packed.extend(self.build_last_page_lens(slots));
        packed
    }

    /// Migrate KV data from contiguous per-slot cache into the paged pool.
    ///
    /// Called after prefill completes. Copies `seq_len(slot)` tokens of K/V
    /// from each contiguous layer buffer into the corresponding token slots
    /// in the pool.
    ///
    /// The contiguous cache layout is `[max_seq_len_contiguous, kv_dim]` per layer.
    fn upload_page_indices(
        &self,
        ctx: &super::tensor::DeviceContext,
        page_indices: &[u32],
    ) -> Result<cudarc::driver::CudaSlice<i32>> {
        let page_indices_i32: Vec<i32> = page_indices.iter().map(|&p| p as i32).collect();
        ctx.stream
            .clone_htod(&page_indices_i32)
            .map_err(|e| anyhow!("H2D page_indices failed: {e}"))
    }

    fn migrate_from_contiguous_range_bf16(
        &self,
        ctx: &super::tensor::DeviceContext,
        contiguous_k_caches: &[super::tensor::DeviceVec],
        contiguous_v_caches: &[super::tensor::DeviceVec],
        max_seq_len_contiguous: usize,
        slot: usize,
        start_pos: usize,
        token_count: usize,
        k_dst_ptr: impl Fn(usize) -> u64,
        v_dst_ptr: impl Fn(usize) -> u64,
    ) -> Result<()> {
        if token_count == 0 || self.k_data.is_empty() {
            return Ok(());
        }

        let page_indices_gpu = self.upload_page_indices(
            ctx,
            self.page_indices_for_token_range(slot, 0, self.seq_len(slot)),
        )?;
        let (pi_ptr, _gpi) = page_indices_gpu.device_ptr(&ctx.stream);
        let stride_page = self.kv_dim * self.page_size;

        for layer in 0..self.num_layers.min(contiguous_k_caches.len()) {
            let (k_src_ptr, _gk) = contiguous_k_caches[layer].data.device_ptr(&ctx.stream);
            let (v_src_ptr, _gv) = contiguous_v_caches[layer].data.device_ptr(&ctx.stream);
            unsafe {
                ffi::kv_cache_to_paged_range_hnd_cuda(
                    k_src_ptr as *const ffi::Half,
                    v_src_ptr as *const ffi::Half,
                    k_dst_ptr(layer) as *mut ffi::Half,
                    v_dst_ptr(layer) as *mut ffi::Half,
                    pi_ptr as *const i32,
                    start_pos as i32,
                    max_seq_len_contiguous as i32,
                    token_count as i32,
                    self.num_kv_heads as i32,
                    self.page_size as i32,
                    self.head_dim as i32,
                    stride_page as i32,
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
        slot: usize,
        start_pos: usize,
        token_count: usize,
    ) -> Result<()> {
        self.migrate_from_contiguous_range_bf16(
            ctx,
            contiguous_k_caches,
            contiguous_v_caches,
            max_seq_len_contiguous,
            slot,
            start_pos,
            token_count,
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
        self.migrate_from_contiguous_range(
            ctx,
            contiguous_k_caches,
            contiguous_v_caches,
            max_seq_len_contiguous,
            slot,
            0,
            self.seq_len(slot),
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

        let token_indices_gpu = self.upload_page_indices(ctx, new_token_indices)?;
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
                ffi::kv_cache_to_paged_int8_range_cuda(
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
        let token_idxs = self.token_rows_for_range(slot, 0, self.seq_len(slot));
        self.migrate_from_contiguous_int8_range(
            ctx,
            contiguous_k_q,
            contiguous_v_q,
            contiguous_k_scales,
            contiguous_v_scales,
            max_seq_len_contiguous,
            0,
            &token_idxs,
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

        let token_indices_gpu = self.upload_page_indices(ctx, new_token_indices)?;
        for layer in 0..self.num_layers.min(contiguous_k_caches.len()) {
            quantize_scatter_kv_fp8_range(
                ctx,
                &contiguous_k_caches[layer],
                self.k_data_ptr(layer, &ctx.stream),
                self.k_scales_ptr(layer, &ctx.stream),
                &token_indices_gpu,
                start_pos,
                max_seq_len_contiguous,
                token_count,
                self.num_kv_heads,
                self.head_dim,
                self.kv_dim,
            )?;
            quantize_scatter_kv_fp8_range(
                ctx,
                &contiguous_v_caches[layer],
                self.v_data_ptr(layer, &ctx.stream),
                self.v_scales_ptr(layer, &ctx.stream),
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
        let token_idxs = self.token_rows_for_range(slot, 0, self.seq_len(slot));
        self.migrate_from_contiguous_fp8_range(
            ctx,
            contiguous_k_caches,
            contiguous_v_caches,
            max_seq_len_contiguous,
            0,
            &token_idxs,
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

        let token_indices_gpu = self.upload_page_indices(ctx, new_token_indices)?;
        let k_state = self
            .tq_k_state
            .as_ref()
            .ok_or_else(|| anyhow!("TurboQuant K state missing"))?;
        let v_state = self
            .tq_v_state
            .as_ref()
            .ok_or_else(|| anyhow!("TurboQuant V state missing"))?;

        let (ti_ptr, _gti) = token_indices_gpu.device_ptr(&ctx.stream);
        for layer in 0..self.num_layers.min(contiguous_k_caches.len()) {
            let (k_src_ptr, _gk) = contiguous_k_caches[layer].data.device_ptr(&ctx.stream);
            let (v_src_ptr, _gv) = contiguous_v_caches[layer].data.device_ptr(&ctx.stream);
            unsafe {
                ffi::kv_cache_to_paged_range_cuda(
                    k_src_ptr as *const ffi::Half,
                    v_src_ptr as *const ffi::Half,
                    self.k_work_ptr(&ctx.stream) as *mut ffi::Half,
                    self.v_work_ptr(&ctx.stream) as *mut ffi::Half,
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

        for layer in 0..self.num_layers.min(contiguous_k_caches.len()) {
            turboquant_quantize_paged_single(
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
            turboquant_quantize_paged_single(
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

/// Default BF16 paged-KV page size used by M0.3.
pub const DEFAULT_PAGE_SIZE: usize = 16;

#[cfg(test)]
mod tests {
    use super::{BudgetBreakdown, compute_budget_breakdown};
    use crate::KVFormat;

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

    // ------------------------------------------------------------------
    // M2a pool refcount mechanics — pure book-keeping tests that exercise
    // `retain_pages` / `release_pages` / `free_slot` without needing a
    // real CUDA context. Mirrors the exact call pattern the CUDA
    // scheduler will run in `publish_to_prefix_cache` → `free_slot` →
    // (later) `release_pages`-from-radix-evict.
    // ------------------------------------------------------------------

    /// Minimal mock of the pool's book-keeping fields so tests can
    /// exercise refcount / free-slot / retain / release logic without
    /// standing up a real CUDA context. Keeps the same invariants as
    /// `TokenKVPool` for the M2a paths.
    struct MockPool {
        page_size: usize,
        free_pages: Vec<u32>,
        page_indices: Vec<Vec<u32>>,
        seq_lens: Vec<usize>,
        slot_epochs: Vec<u64>,
        page_attach_count: Vec<u32>,
        page_ref_count: Vec<u32>,
    }

    impl MockPool {
        fn new(max_total_pages: usize, num_slots: usize, page_size: usize) -> Self {
            Self {
                page_size,
                free_pages: (0..max_total_pages as u32).rev().collect(),
                page_indices: vec![Vec::new(); num_slots],
                seq_lens: vec![0; num_slots],
                slot_epochs: vec![0; num_slots],
                page_attach_count: vec![0_u32; max_total_pages],
                page_ref_count: vec![0_u32; max_total_pages],
            }
        }

        fn slot_hot_tail_len(&self, slot: usize) -> usize {
            self.seq_lens[slot] % self.page_size
        }

        fn slot_last_page_len(&self, slot: usize) -> usize {
            let seq_len = self.seq_lens[slot];
            if seq_len == 0 {
                0
            } else {
                let hot_tail_len = self.slot_hot_tail_len(slot);
                if hot_tail_len == 0 {
                    self.page_size
                } else {
                    hot_tail_len
                }
            }
        }

        fn slot_hot_tail_page(&self, slot: usize) -> Option<u32> {
            if self.slot_hot_tail_len(slot) == 0 {
                None
            } else {
                self.page_indices[slot].last().copied()
            }
        }

        fn page_is_shared_read_only(&self, page: u32) -> bool {
            let page_idx = page as usize;
            self.page_ref_count[page_idx] > 0 || self.page_attach_count[page_idx] > 1
        }

        fn slot_shared_hot_tail_page(&self, slot: usize) -> Option<u32> {
            let hot_tail_page = self.slot_hot_tail_page(slot)?;
            self.page_is_shared_read_only(hot_tail_page)
                .then_some(hot_tail_page)
        }

        fn recycle_page_if_unreferenced(&mut self, page: u32) {
            let page_idx = page as usize;
            if self.page_ref_count[page_idx] == 0 && self.page_attach_count[page_idx] == 0 {
                self.free_pages.push(page);
            }
        }

        fn alloc_tokens(&mut self, slot: usize, count: usize) -> Vec<u32> {
            let hot_tail_len = self.slot_hot_tail_len(slot);
            let available_in_last_page = if hot_tail_len == 0 {
                0
            } else {
                self.page_size - hot_tail_len
            };
            let remaining_after_fill = count.saturating_sub(available_in_last_page);
            let new_page_count = remaining_after_fill.div_ceil(self.page_size);
            assert!(self.free_pages.len() >= new_page_count, "mock pool OOM");

            let mut new_pages = Vec::with_capacity(new_page_count);
            for _ in 0..new_page_count {
                let page = self.free_pages.pop().unwrap();
                self.page_attach_count[page as usize] = 1;
                new_pages.push(page);
            }
            self.page_indices[slot].extend_from_slice(&new_pages);
            self.seq_lens[slot] += count;
            new_pages
        }

        fn alloc_detached_pages(&mut self, count: usize) -> Vec<u32> {
            let mut new_pages = Vec::with_capacity(count);
            for _ in 0..count {
                let page = self.free_pages.pop().unwrap();
                self.page_ref_count[page as usize] = 1;
                new_pages.push(page);
            }
            new_pages
        }

        fn attach_pages(&mut self, slot: usize, pages: &[u32], token_count: usize) {
            assert!(self.page_indices[slot].is_empty());
            assert!(token_count <= pages.len() * self.page_size);
            for &page in pages {
                assert!(
                    self.page_attach_count[page as usize] > 0
                        || self.page_ref_count[page as usize] > 0
                );
                self.page_attach_count[page as usize] += 1;
            }
            self.page_indices[slot].extend_from_slice(pages);
            self.seq_lens[slot] = token_count;
        }

        fn retain(&mut self, pages: &[u32]) {
            for &p in pages {
                self.page_ref_count[p as usize] = self.page_ref_count[p as usize].saturating_add(1);
            }
        }

        fn release(&mut self, pages: &[u32]) -> Vec<u32> {
            let mut freed = Vec::new();
            for &p in pages {
                let pu = p as usize;
                let cur = self.page_ref_count[pu];
                self.page_ref_count[pu] = cur.saturating_sub(1);
                if self.page_ref_count[pu] == 0 {
                    self.recycle_page_if_unreferenced(p);
                    freed.push(p);
                }
            }
            freed
        }

        fn free_slot(&mut self, slot: usize) {
            let slot_pages = std::mem::take(&mut self.page_indices[slot]);
            if !slot_pages.is_empty() {
                self.slot_epochs[slot] = self.slot_epochs[slot].saturating_add(1);
            }
            for idx in slot_pages {
                self.page_attach_count[idx as usize] =
                    self.page_attach_count[idx as usize].saturating_sub(1);
                self.recycle_page_if_unreferenced(idx);
            }
            self.seq_lens[slot] = 0;
        }

        fn truncate_slot(&mut self, slot: usize, new_len: usize) -> Vec<u32> {
            assert!(new_len <= self.seq_lens[slot]);
            let keep_pages = new_len.div_ceil(self.page_size);
            let removed = self.page_indices[slot].split_off(keep_pages);
            if !removed.is_empty() {
                self.slot_epochs[slot] = self.slot_epochs[slot].saturating_add(1);
            }
            let mut recycled = Vec::new();
            for page in removed {
                self.page_attach_count[page as usize] =
                    self.page_attach_count[page as usize].saturating_sub(1);
                let before = self.free_pages.len();
                self.recycle_page_if_unreferenced(page);
                if self.free_pages.len() > before {
                    recycled.push(page);
                }
            }
            self.seq_lens[slot] = new_len;
            recycled
        }

        fn detach_shared_hot_tail_page_for_append(&mut self, slot: usize, shared_tail_page: u32) {
            debug_assert_eq!(
                self.slot_shared_hot_tail_page(slot),
                Some(shared_tail_page),
                "detach_shared_hot_tail_page_for_append requires the live shared hot tail",
            );

            let new_page = self.free_pages.pop().unwrap();
            self.page_attach_count[new_page as usize] = 1;

            let old_tail_page = {
                let tail_slot_page = self.page_indices[slot].last_mut().unwrap();
                let old_tail_page = *tail_slot_page;
                *tail_slot_page = new_page;
                old_tail_page
            };
            self.page_attach_count[old_tail_page as usize] -= 1;
            self.recycle_page_if_unreferenced(old_tail_page);
        }

        fn cow_tail_page_for_append(&mut self, slot: usize) -> bool {
            let Some(shared_tail_page) = self.slot_shared_hot_tail_page(slot) else {
                return false;
            };
            self.detach_shared_hot_tail_page_for_append(slot, shared_tail_page);
            true
        }

        fn retained_count(&self) -> usize {
            self.page_ref_count.iter().filter(|&&rc| rc > 0).count()
        }

        fn free_count(&self) -> usize {
            let partial_capacity = self
                .seq_lens
                .iter()
                .enumerate()
                .map(|(slot, _)| {
                    let hot_tail_len = self.slot_hot_tail_len(slot);
                    if hot_tail_len == 0 {
                        0
                    } else {
                        self.page_size - hot_tail_len
                    }
                })
                .sum::<usize>();
            self.free_pages.len() * self.page_size + partial_capacity
        }

        fn page_indices_for_token_range(
            &self,
            slot: usize,
            start_pos: usize,
            token_count: usize,
        ) -> &[u32] {
            let start_page = start_pos / self.page_size;
            let end_page = (start_pos + token_count).div_ceil(self.page_size);
            &self.page_indices[slot][start_page..end_page]
        }

        fn token_rows_for_range(
            &self,
            slot: usize,
            start_pos: usize,
            token_count: usize,
        ) -> Vec<u32> {
            let mut rows = Vec::with_capacity(token_count);
            for pos in start_pos..start_pos + token_count {
                let page = self.page_indices[slot][pos / self.page_size];
                rows.push(page * self.page_size as u32 + (pos % self.page_size) as u32);
            }
            rows
        }

        fn build_last_indices(&self, slots: &[usize]) -> Vec<i32> {
            slots
                .iter()
                .map(|&slot| {
                    let seq_len = self.seq_lens[slot];
                    let last_pos = seq_len - 1;
                    let page = self.page_indices[slot][last_pos / self.page_size];
                    (page as usize * self.page_size + (last_pos % self.page_size)) as i32
                })
                .collect()
        }

        fn build_indptr(&self, slots: &[usize]) -> Vec<i32> {
            let mut indptr = Vec::with_capacity(slots.len() + 1);
            indptr.push(0);
            for &slot in slots {
                let last = *indptr.last().unwrap();
                indptr.push(last + self.page_indices[slot].len() as i32);
            }
            indptr
        }

        fn build_last_page_lens(&self, slots: &[usize]) -> Vec<i32> {
            slots
                .iter()
                .map(|&slot| self.slot_last_page_len(slot) as i32)
                .collect()
        }

        fn build_quantized_decode_indptr(&self, slots: &[usize]) -> Vec<i32> {
            let mut packed = self.build_indptr(slots);
            packed.extend(self.build_last_page_lens(slots));
            packed
        }
    }

    #[test]
    fn format_default_page_sizes_match_quantized_page16_dispatch() {
        assert_eq!(KVFormat::BF16.default_page_size(), 16);
        assert_eq!(KVFormat::INT8.default_page_size(), 16);
        assert_eq!(KVFormat::FP8E4M3.default_page_size(), 16);
        assert_eq!(
            KVFormat::TurboQuant {
                key_bits: 3,
                val_bits: 4,
            }
            .default_page_size(),
            1
        );
    }

    #[test]
    fn alloc_tokens_reuses_tail_before_grabbing_a_new_page() {
        let mut pool = MockPool::new(4, 1, 16);
        let first = pool.alloc_tokens(0, 8);
        let second = pool.alloc_tokens(0, 4);

        assert_eq!(first, vec![0]);
        assert!(
            second.is_empty(),
            "tail room should satisfy the second alloc"
        );
        assert_eq!(pool.page_indices[0], vec![0]);
        assert_eq!(pool.seq_lens[0], 12);
        assert_eq!(pool.free_pages, vec![3, 2, 1]);
    }

    #[test]
    fn token_ranges_map_to_pages_and_rows_under_page_size_16() {
        let mut pool = MockPool::new(4, 1, 16);
        pool.alloc_tokens(0, 20);

        assert_eq!(pool.page_indices[0], vec![0, 1]);
        assert_eq!(pool.page_indices_for_token_range(0, 0, 16), &[0]);
        assert_eq!(pool.page_indices_for_token_range(0, 12, 8), &[0, 1]);
        assert_eq!(pool.token_rows_for_range(0, 14, 4), vec![14, 15, 16, 17]);
    }

    #[test]
    fn last_indices_track_last_token_row_under_page_size_16() {
        // Pool needs at least 5 pages to satisfy 20 + 33 = 53 tokens at
        // page_size=16 (slot 0 → 2 pages, slot 1 → 3 pages). Use 6 to mirror
        // the adjacent `quantized_decode_indptr_*` test.
        // After alloc: slot 0 = pages [0, 1], slot 1 = pages [2, 3, 4].
        // - slot 0 last_pos = 19 → page_indices[0][1] = 1 → row = 1*16+3 = 19
        // - slot 1 last_pos = 32 → page_indices[1][2] = 4 → row = 4*16+0 = 64
        let mut pool = MockPool::new(6, 2, 16);
        pool.alloc_tokens(0, 20);
        pool.alloc_tokens(1, 33);

        assert_eq!(pool.build_last_indices(&[0, 1]), vec![19, 64]);
    }

    #[test]
    fn quantized_decode_indptr_packs_page_offsets_and_last_page_lens() {
        let mut pool = MockPool::new(6, 2, 16);
        pool.alloc_tokens(0, 20);
        pool.alloc_tokens(1, 33);

        assert_eq!(pool.build_indptr(&[0, 1]), vec![0, 2, 5]);
        assert_eq!(pool.build_last_page_lens(&[0, 1]), vec![4, 1]);
        assert_eq!(
            pool.build_quantized_decode_indptr(&[0, 1]),
            vec![0, 2, 5, 4, 1]
        );
    }

    #[test]
    fn free_count_includes_partial_tail_capacity() {
        let mut pool = MockPool::new(3, 2, 16);
        pool.alloc_tokens(0, 8);
        pool.alloc_tokens(1, 16);

        assert_eq!(pool.free_pages, vec![2]);
        assert_eq!(pool.free_count(), 24, "1 free page + 8 token tail room");
    }

    #[test]
    fn slot_epoch_advances_only_when_a_live_slot_is_released() {
        let mut pool = MockPool::new(4, 2, 16);
        pool.alloc_tokens(0, 8);

        pool.free_slot(0);
        pool.free_slot(1);

        assert_eq!(pool.slot_epochs, vec![1, 0]);
        assert!(pool.page_indices[0].is_empty());
    }

    #[test]
    fn retain_then_free_slot_keeps_page_in_limbo() {
        // M2a core invariant: pages retained by the radix survive a
        // `free_slot` call. Pages without a retain get freed back as
        // before. Exactly the ordering the scheduler runs on cleanup:
        //   1. scheduler.publish_to_prefix_cache → retain_pages
        //   2. paged_kv_pool.free_slot
        let mut pool = MockPool::new(4, 2, 16);
        let _ = pool.alloc_tokens(0, 20); // slot 0 takes 2 pages
        assert_eq!(pool.page_indices[0].len(), 2);
        assert_eq!(pool.free_pages.len(), 2);

        let retained = vec![pool.page_indices[0][0]];
        pool.retain(&retained);
        assert_eq!(pool.retained_count(), 1);

        pool.free_slot(0);
        // Slot is empty but the retained page did NOT go back to
        // the free list.
        assert!(pool.page_indices[0].is_empty());
        assert_eq!(pool.free_pages.len(), 3, "1 pinned page remains in limbo");
        assert_eq!(pool.retained_count(), 1);
        assert_eq!(pool.slot_epochs[0], 1);
    }

    #[test]
    fn release_after_free_slot_reclaims_limbo_pages() {
        // Second half of the M2a cycle: once the radix evicts / drops
        // the retained pages, `release_pages` pushes them back to the
        // free list with no double-free and no lost pages.
        let mut pool = MockPool::new(4, 2, 16);
        let alloc = pool.alloc_tokens(0, 20);
        let retained: Vec<u32> = alloc[..1].to_vec();
        pool.retain(&retained);
        pool.free_slot(0);
        assert_eq!(pool.free_pages.len(), 3);

        // Radix eviction path drops the pages.
        let freed_now = pool.release(&retained);
        assert_eq!(freed_now.len(), 1);
        assert_eq!(pool.retained_count(), 0);
        assert_eq!(pool.free_pages.len(), 4, "every page back in the free pool");
    }

    #[test]
    fn truncate_slot_reclaims_only_full_trailing_pages() {
        let mut pool = MockPool::new(4, 1, 16);
        pool.alloc_tokens(0, 40);
        assert_eq!(pool.page_indices[0], vec![0, 1, 2]);

        let freed = pool.truncate_slot(0, 20);
        assert_eq!(freed, vec![2]);
        assert_eq!(pool.page_indices[0], vec![0, 1]);
        assert_eq!(pool.seq_lens[0], 20);

        let freed = pool.truncate_slot(0, 17);
        assert!(
            freed.is_empty(),
            "shrinking within the hot page must not recycle storage"
        );
        assert_eq!(pool.page_indices[0], vec![0, 1]);
        assert_eq!(pool.seq_lens[0], 17);
    }

    #[test]
    fn retain_release_without_free_slot_does_not_move_pages() {
        // Invariant: retain/release only flip the refcount; they do
        // NOT move pages out of a live slot's token_indices. This
        // matches the scheduler's "shadow observer" fallback lookup
        // pattern: bump → log → release, page stays in its owning
        // slot the whole time.
        let mut pool = MockPool::new(4, 2, 16);
        let alloc = pool.alloc_tokens(0, 20);
        let pg = alloc[0];
        pool.retain(&[pg]);
        assert_eq!(pool.page_ref_count[pg as usize], 1);
        assert_eq!(pool.page_attach_count[pg as usize], 1);
        assert_eq!(pool.page_indices[0].len(), 2);
        assert_eq!(pool.free_pages.len(), 2);

        let freed = pool.release(&[pg]);
        assert!(
            freed.is_empty(),
            "release must not free a page that is still attached to a live slot",
        );
        assert_eq!(pool.page_ref_count[pg as usize], 0);
        assert_eq!(pool.page_attach_count[pg as usize], 1);
    }

    #[test]
    fn double_retain_needs_double_release_to_free() {
        // Two radix insert passes on the same prefix should not free
        // the underlying pages after a single release cycle.
        let mut pool = MockPool::new(2, 1, 16);
        let alloc = pool.alloc_tokens(0, 20);
        pool.retain(&alloc);
        pool.retain(&alloc);
        assert_eq!(pool.page_ref_count[alloc[0] as usize], 2);
        pool.free_slot(0);
        assert_eq!(
            pool.free_pages.len(),
            0,
            "both pages pinned, neither in free list"
        );

        let freed = pool.release(&alloc);
        assert!(
            freed.is_empty(),
            "first release only drops refcount from 2 to 1; nothing freed yet",
        );
        assert_eq!(pool.retained_count(), 2);

        let freed = pool.release(&alloc);
        assert_eq!(
            freed.len(),
            2,
            "second release drops to 0 → both pages freed"
        );
        assert_eq!(pool.free_pages.len(), 2);
        assert_eq!(pool.retained_count(), 0);
    }

    #[test]
    fn direct_attach_of_full_shared_prefix_keeps_sealed_page_shared() {
        let mut pool = MockPool::new(4, 2, 16);
        let pages = pool.alloc_tokens(0, 16);
        pool.retain(&pages);
        pool.free_slot(0);

        pool.attach_pages(1, &pages, 16);
        assert_eq!(pool.page_indices[1], pages);
        assert!(!pool.cow_tail_page_for_append(1));
        assert_eq!(pool.page_attach_count[pages[0] as usize], 1);
        assert_eq!(pool.page_ref_count[pages[0] as usize], 1);

        let new_hot_tail = pool.alloc_tokens(1, 1);
        assert_eq!(new_hot_tail.len(), 1);
        assert_eq!(pool.page_indices[1][0], pages[0]);
        assert_ne!(pool.page_indices[1][1], pages[0]);
        assert_eq!(pool.seq_lens[1], 17);
    }

    #[test]
    fn private_hot_tail_append_does_not_detach() {
        let mut pool = MockPool::new(2, 1, 16);
        let pages = pool.alloc_tokens(0, 15);

        assert_eq!(pages.len(), 1);
        assert_eq!(pool.slot_hot_tail_page(0), Some(pages[0]));
        assert_eq!(pool.slot_shared_hot_tail_page(0), None);
        assert!(!pool.cow_tail_page_for_append(0));
        assert_eq!(pool.page_indices[0], pages);
        assert_eq!(pool.page_attach_count[pages[0] as usize], 1);
        assert_eq!(pool.page_ref_count[pages[0] as usize], 0);
    }

    #[test]
    fn shared_hot_tail_detaches_before_append_and_full_page_transition_stays_private() {
        let mut pool = MockPool::new(4, 2, 16);
        let detached = pool.alloc_detached_pages(1);
        pool.attach_pages(0, &detached, 15);

        assert_eq!(pool.slot_hot_tail_page(0), Some(detached[0]));
        assert_eq!(pool.slot_shared_hot_tail_page(0), Some(detached[0]));
        assert!(pool.cow_tail_page_for_append(0));
        let private_tail = pool.page_indices[0][0];
        assert_ne!(private_tail, detached[0]);
        assert_eq!(pool.page_attach_count[detached[0] as usize], 0);
        assert_eq!(pool.page_ref_count[detached[0] as usize], 1);
        assert_eq!(pool.page_attach_count[private_tail as usize], 1);
        assert_eq!(pool.page_ref_count[private_tail as usize], 0);
        assert_eq!(pool.slot_shared_hot_tail_page(0), None);

        let filled = pool.alloc_tokens(0, 1);
        assert!(
            filled.is_empty(),
            "detached private tail should fill in place"
        );
        assert_eq!(pool.seq_lens[0], 16);
        assert_eq!(pool.page_indices[0], vec![private_tail]);
        assert!(!pool.cow_tail_page_for_append(0));

        let next_hot_tail = pool.alloc_tokens(0, 1);
        assert_eq!(next_hot_tail.len(), 1);
        assert_eq!(pool.page_indices[0], vec![private_tail, next_hot_tail[0]]);
    }
}

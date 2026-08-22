//! Paged KV cache pool — TileLang-compatible KV storage with runtime
//! `page_size`.
//!
//! The pool keeps **token-level sequence accounting** (`seq_len(slot)` is always
//! in logical tokens) while allocating and retaining storage in **physical
//! pages**. This matches TileLang's HND paged layout:
//!   `[max_pages, num_kv_heads, page_size, head_dim]`.
//!
//! BF16 / INT8 / FP8 E4M3 now all use `page_size = 16`. TurboQuant remains
//! token-granular (`page_size = 1`) until its decode and migration kernels are
//! rewritten around paged layout. `PackedBytes` (MLA latent records) uses
//! `page_size = 64` — the FlashMLA block size — and stores one opaque
//! `bytes_per_token` record per token in the K plane only.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr};
use log::info;

use super::tensor::DeviceContext;
use crate::kv_quant::paged_attention_quantized_fa3_workspace_bytes;
use crate::kv_types::{KVCacheDtype, KVFormat};
use crate::turboquant_state::TurboQuantLayerState;

/// Logical-page marker for a page that has been **evict-dropped** out of HBM
/// under the write-through tiered KV model ([`TokenKVPool::evict_slot_page`]).
///
/// A recall slot keeps its `page_indices[slot]` vector at full logical length so
/// a logical page index still maps to the right entry; an evicted middle page
/// leaves this sentinel in its logical slot while the physical HBM page returns
/// to `free_pages`. The sentinel never names a real page (`max_total_pages` is
/// far below `u32::MAX`) and only appears on the opt-in recall path — the
/// default decode path keeps a contiguous, sentinel-free page table. This value
/// MUST match `infer_seam::host_paged_kv_pool::EVICTED_PAGE` so a host-mirrored
/// sentinel survives the round-trip through `mirror_slot`.
pub const EVICTED_PAGE: u32 = u32::MAX;

/// PHYSICAL FlashMLA band page id. Engine LOGICAL page ids share the same u32
/// representation; constructing `BandPage` is the explicit domain assertion
/// (f7891c3f0 fed logical ids into physical band tables undetected).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BandPage(pub u32);

/// Paged KV cache pool — shared across all request slots.
///
/// Storage is format-aware via `KVFormat`:
/// - `BF16`: `k_data`/`v_data` are `CudaSlice<u8>` holding bf16 (2 bytes/elem)
/// - `FP8E4M3`: `k_data`/`v_data` hold FP8 E4M3 (1 byte/elem), + `k_scales`/`v_scales`
/// - `INT8`: `k_data`/`v_data` hold int8 (1 byte/elem), + `k_scales`/`v_scales`
/// - `PackedBytes`: `k_data` holds one opaque `bytes_per_token` record per
///   token (MLA latent); `v_data` and all scale/norm buffers stay empty
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
    /// Per-head per-token f32 scales (INT8 + FP8 default). `[max_total_tokens, num_kv_heads]`
    k_scales: Vec<CudaSlice<f32>>,
    v_scales: Vec<CudaSlice<f32>>,
    /// Shared bf16 working buffers (1 layer, for decode_prep write target).
    /// Only allocated when format != BF16.
    k_work: Option<CudaSlice<u8>>,
    v_work: Option<CudaSlice<u8>>,
    /// Workspace for split-KV fused-dequant attention (INT8 only).
    pub quantized_attn_workspace: Option<CudaSlice<u8>>,
    pub quantized_attn_workspace_bytes: usize,
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
    let data_bytes_per_token = match format.packed_record_bytes_per_token() {
        // Single packed plane (K only) — the MLA latent record has no V plane.
        Some(record_bytes) => record_bytes,
        None => kv_dim * format.bytes_per_element() * 2, // K+V
    };
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

/// Host-side shape validation shared by pool construction (kept free-standing
/// so the no-GPU unit tests can exercise it without a `DeviceContext`).
fn validate_format_shape(format: KVFormat, num_kv_heads: usize) -> Result<()> {
    if let KVFormat::PackedBytes { bytes_per_token } = format {
        ensure!(
            num_kv_heads == 1,
            "KVFormat::PackedBytes requires num_kv_heads == 1 (the MLA latent record is \
             head-less), got {num_kv_heads}"
        );
        ensure!(
            bytes_per_token > 0,
            "KVFormat::PackedBytes requires bytes_per_token > 0"
        );
    }
    Ok(())
}

impl TokenKVPool {
    /// Return the budget bytes needed to allocate at least `token_count` logical
    /// tokens for this pool shape and format.
    pub fn budget_bytes_for_tokens(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        token_count: usize,
        format: KVFormat,
    ) -> usize {
        let budget =
            compute_budget_breakdown(num_layers, num_kv_heads, head_dim, 1, usize::MAX, format);
        token_count.saturating_mul(budget.total_bytes_per_token)
    }

    fn storage_bytes_per_token(&self) -> usize {
        let data_bytes = match self.format.packed_record_bytes_per_token() {
            // Single packed plane (K only) — no V plane for packed records.
            Some(record_bytes) => record_bytes,
            None => self.kv_dim * self.format.bytes_per_element() * 2,
        };
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

    /// Full per-page host-image size (all layers, K+V, scales/norms) — the
    /// payload unit `copy_pages_to_host`/`copy_pages_from_host` move.
    pub fn storage_bytes_per_page(&self) -> usize {
        self.storage_bytes_for_tokens(self.page_size)
    }

    /// Exact requested device bytes this pool owns: Σ over every `CudaSlice<T>`
    /// field of `slice.len() * size_of::<T>()`. This is the *requested* byte
    /// count, NOT the cudaMalloc-rounded physical reservation. Covers the
    /// per-layer K/V data planes, per-token scales (FP8/INT8), per-token norms
    /// (TurboQuant), the 1-layer bf16 work buffers, the split-KV attention
    /// workspace.
    ///
    /// `tq_k_state`/`tq_v_state` (TurboQuant rotation/codebook device buffers)
    /// are NOT summed here — they are always `None` for the DSv4 FlashMLA pool
    /// (`KVFormat::PackedBytes`, single-plane) and the DSv4 ledger this method
    /// feeds never exercises TurboQuant.
    pub fn device_bytes(&self) -> usize {
        let mut total = 0usize;
        for s in &self.k_data {
            total += s.len(); // u8
        }
        for s in &self.v_data {
            total += s.len(); // u8
        }
        for s in &self.k_scales {
            total += s.len() * std::mem::size_of::<f32>();
        }
        for s in &self.v_scales {
            total += s.len() * std::mem::size_of::<f32>();
        }
        for s in &self.k_norms {
            total += s.len() * std::mem::size_of::<u16>();
        }
        for s in &self.v_norms {
            total += s.len() * std::mem::size_of::<u16>();
        }
        if let Some(s) = &self.k_work {
            total += s.len(); // u8
        }
        if let Some(s) = &self.v_work {
            total += s.len(); // u8
        }
        if let Some(s) = &self.quantized_attn_workspace {
            total += s.len(); // u8
        }
        total
    }

    /// Whether this pool stores one packed record per token in the K plane
    /// only (`v_data` and all scale/norm buffers are empty).
    fn is_single_plane(&self) -> bool {
        self.format.packed_record_bytes_per_token().is_some()
    }

    /// Bytes of one page in a single data plane (K or V; packed-record
    /// formats store everything in the K plane).
    fn data_plane_bytes_per_page(&self) -> usize {
        match self.format.packed_record_bytes_per_token() {
            Some(record_bytes) => self.page_size * record_bytes,
            None => self.page_size * self.kv_dim * self.format.bytes_per_element(),
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

    fn recycle_page_if_unreferenced(&mut self, page: u32) -> bool {
        let page_idx = page as usize;
        if self.page_attach_count[page_idx] == 0 && self.page_ref_count[page_idx] == 0 {
            self.free_pages.push(page);
            true
        } else {
            false
        }
    }

    fn claim_mirrored_page(&mut self, page: u32) {
        let idx = page as usize;
        if self.page_attach_count[idx] == 0
            && self.page_ref_count[idx] == 0
            && let Some(pos) = self.free_pages.iter().position(|&p| p == page)
        {
            self.free_pages.swap_remove(pos);
        }
        self.page_attach_count[idx] = self.page_attach_count[idx].saturating_add(1);
    }

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

    pub fn with_format(
        ctx: &DeviceContext,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_slots: usize,
        budget_bytes: usize,
        format: KVFormat,
    ) -> Result<Self> {
        validate_format_shape(format, num_kv_heads)?;
        let kv_dim = num_kv_heads * head_dim;
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

        // PackedBytes stores one opaque record per token (K plane only).
        // Other formats: kv_dim * bytes_per_element bytes per token.
        let pool_bytes_per_layer = match format {
            KVFormat::PackedBytes { bytes_per_token } => max_total_tokens * bytes_per_token,
            _ => max_total_tokens * kv_dim * format.bytes_per_element(),
        };
        let single_plane = format.packed_record_bytes_per_token().is_some();
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
            // Data buffers. Packed-record formats are single-plane: the
            // whole record lives in `k_data`, `v_data` stays empty.
            for _ in 0..num_layers {
                k_data.push(
                    ctx.stream
                        .alloc_zeros::<u8>(pool_bytes_per_layer)
                        .map_err(|e| anyhow!("TokenKVPool K data alloc failed: {e}"))?,
                );
                if !single_plane {
                    v_data.push(
                        ctx.stream
                            .alloc_zeros::<u8>(pool_bytes_per_layer)
                            .map_err(|e| anyhow!("TokenKVPool V data alloc failed: {e}"))?,
                    );
                }
            }

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
                (pool_bytes_per_layer * if single_plane { 1 } else { 2 }) as f64 / 1e6,
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

        // Split-KV attention workspace for the quantized decode kernel: every
        // slot in one batch, 16 splits, GQA ratio up to 8.
        let (quantized_attn_workspace, quantized_attn_workspace_bytes) =
            if matches!(format, KVFormat::INT8 | KVFormat::FP8E4M3) && pool_bytes_per_layer > 0 {
                let ws_bytes = paged_attention_quantized_fa3_workspace_bytes(
                    num_slots,
                    num_kv_heads * 8,
                    head_dim,
                    16,
                );
                let ws = ctx
                    .stream
                    .alloc_zeros::<u8>(ws_bytes)
                    .map_err(|e| anyhow!("Quantized attn workspace alloc failed: {e}"))?;
                (Some(ws), ws_bytes)
            } else {
                (None, 0)
            };

        let (tq_k_state, tq_v_state) = if let KVFormat::TurboQuant { key_bits, val_bits } = format {
            let k_state = TurboQuantLayerState::new(ctx, num_layers, head_dim, key_bits, 42)?;
            let v_state = TurboQuantLayerState::new(ctx, num_layers, head_dim, val_bits, 137)?;
            (Some(k_state), Some(v_state))
        } else {
            (None, None)
        };

        // Legacy dtype mapping. PackedBytes carries no per-head quant
        // dispatch — BF16 is the inert legacy mapping (P2's FlashMLA
        // consumer reads the packed record directly, never this field).
        let dtype = match format {
            KVFormat::BF16 | KVFormat::PackedBytes { .. } => KVCacheDtype::BF16,
            KVFormat::FP8E4M3 | KVFormat::INT8 | KVFormat::TurboQuant { .. } => KVCacheDtype::INT8,
        };

        Ok(Self {
            k_data,
            v_data,
            k_scales,
            v_scales,
            k_work,
            v_work,
            quantized_attn_workspace,
            quantized_attn_workspace_bytes,
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

    /// Extend a fixed-layout band's page table by `count` pages from the free
    /// list WITHOUT moving the logical token cursor — the band-demand-paging
    /// counterpart of [`Self::alloc_tokens`] (#154 Phase 3b: DSv4 grows the
    /// comp region at page boundaries; `set_band_cursor` owns the cursor).
    /// Returns the newly attached page ids; the caller must zero them (a
    /// recycled page carries a prior occupant's bytes).
    pub fn band_extend(&mut self, slot: usize, count: usize) -> Result<Vec<BandPage>> {
        ensure!(
            slot < self.num_slots,
            "TokenKVPool::band_extend slot {slot} out of range {}",
            self.num_slots
        );
        if count == 0 {
            return Ok(Vec::new());
        }
        if count > self.free_pages.len() {
            return Err(anyhow!(
                "TokenKVPool: out of pages (band_extend {count} pages, available {})",
                self.free_pages.len()
            ));
        }
        let mut new_pages = Vec::with_capacity(count);
        for _ in 0..count {
            let idx = self
                .free_pages
                .pop()
                .expect("invariant: free_pages.len() >= count checked above");
            self.page_attach_count[idx as usize] = 1;
            self.page_indices[slot].push(idx);
            new_pages.push(BandPage(idx));
        }
        Ok(new_pages)
    }

    /// Zero `pages`' data planes with DEVICE memsets (async, stream-ordered).
    /// The band-demand-paging claim-zero path: an H2D of host zeros blocks on
    /// pageable memory per call (~290 blocking copies per DSv4 request —
    /// measured +3.7% on the E6 shape), a memset does not.
    pub fn zero_pages(&mut self, ctx: &DeviceContext, pages: &[BandPage]) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let token_bytes = self.data_plane_bytes_per_page();
            let single_plane = self.is_single_plane();
            for &BandPage(page) in pages {
                anyhow::ensure!(
                    (page as usize) < self.max_total_pages,
                    "paged_kv zero_pages: page id {page} outside pool ({} pages)",
                    self.max_total_pages
                );
                let start = page as usize * token_bytes;
                for layer in 0..self.num_layers {
                    let mut k_view = self.k_data[layer].slice_mut(start..start + token_bytes);
                    ctx.stream
                        .memset_zeros(&mut k_view)
                        .map_err(|e| anyhow!("paged_kv zero_pages K memset failed: {e}"))?;
                    if !single_plane {
                        let mut v_view = self.v_data[layer].slice_mut(start..start + token_bytes);
                        ctx.stream
                            .memset_zeros(&mut v_view)
                            .map_err(|e| anyhow!("paged_kv zero_pages V memset failed: {e}"))?;
                    }
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (ctx, pages);
            anyhow::bail!("GPU required: TokenKVPool::zero_pages")
        }
    }

    /// Roll the logical cursor back to `new_len` tokens WITHOUT recycling any
    /// band pages — the fixed-layout-band counterpart of [`Self::truncate_slot`].
    ///
    /// [`Self::truncate_slot`] recycles trailing pages (`keep = ceil(new_len /
    /// page_size)`), which is correct for a sequential cache but would dismantle
    /// a fixed-layout band (the SW ring / comp region must stay fully resident).
    /// Used on a DSv4 MTP reject: only the cursor shrinks so the next tick's
    /// `seq_len == append_pos` invariant holds; the band is untouched.
    pub fn set_band_cursor(&mut self, slot: usize, new_len: usize) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "TokenKVPool::set_band_cursor slot {slot} out of range {}",
            self.num_slots
        );
        // The cursor is the LOGICAL sequence position — unbounded by the physical
        // band: a fixed-layout band wraps (SW ring mod sw) and compresses (comp
        // region to max_seq/cr), so the logical position routinely exceeds the
        // band's page capacity. It is bounded by max_seq at ingress, NOT here.
        // (A band-capacity check here wrongly broke both decode-advance and MTP
        // truncation once a sequence outgrew the band — #review long-ctx.)
        self.seq_lens[slot] = new_len;
        Ok(())
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

    /// Mirror a slot's page table and logical length from a host-authoritative
    /// pool view.
    ///
    /// The rewrite's Qwen-dense executor does not run this pool's allocator at
    /// all: the engine's host `CudaKvPool` is the single page allocator, and the
    /// executor lowers each scheduled row's `KvBatchDescriptor` page list into
    /// this device pool via `mirror_slot` before the forward. Host page ids
    /// index this pool's storage rows 1:1 (both pools are sized to the same
    /// `total_pages`), which is what makes radix prefix attach work: pages a
    /// finished request published keep their KV rows until the host pool
    /// recycles the ids, so a fresh slot mirroring those ids reads the prefix
    /// KV directly.
    ///
    /// Mirrored slots bypass `free_pages`/`page_attach_count`; do not mix
    /// `mirror_slot` with [`Self::alloc_tokens`]/[`Self::free_slot`] on the
    /// same pool.
    pub fn mirror_slot(&mut self, slot: usize, pages: &[u32], seq_len: usize) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "TokenKVPool::mirror_slot slot {slot} out of range {}",
            self.num_slots
        );
        // Under 2D the host pool passes only this shard's local pages (1/cp of
        // the global count), so `<=` covers both the replicated (==) and
        // sharded (<) cases.
        ensure!(
            pages.len() <= seq_len.div_ceil(self.page_size),
            "TokenKVPool::mirror_slot pages {} exceed seq_len {} at page_size {}",
            pages.len(),
            seq_len,
            self.page_size
        );
        for &page in pages {
            // The evict sentinel marks a logical page whose physical HBM page was
            // returned to the pool under the write-through tiered KV model
            // (`evict_slot_page`); it is not a real page id, so it bypasses the
            // bounds check. It only appears on the opt-in recall path — the
            // default decode path mirrors a fully contiguous, sentinel-free table.
            ensure!(
                page == EVICTED_PAGE || (page as usize) < self.max_total_pages,
                "TokenKVPool::mirror_slot page {page} out of range {} \
                 (host pool total_pages exceeds device pool budget?)",
                self.max_total_pages
            );
        }
        let dst = &mut self.page_indices[slot];
        // Steady decode extends the same table by one page every 16 tokens;
        // avoid re-copying the whole prefix when the host view is a superset.
        if pages.len() >= dst.len() && pages[..dst.len()] == dst[..] {
            dst.extend_from_slice(&pages[dst.len()..]);
        } else {
            dst.clear();
            dst.extend_from_slice(pages);
        }
        self.seq_lens[slot] = seq_len;
        Ok(())
    }

    /// Mirror a fixed logical page band for a slot while setting an independent
    /// logical token cursor. Used by DSv4 FlashMLA: the slot page table is
    /// `[SW ring | compressed region]`, not `ceil(seq_len / page_size)`.
    /// Returns whether the slot's page list CHANGED (#154 Phase 0: the caller's
    /// device-table refresh is dirty-driven). An unchanged page list only
    /// updates the token cursor — no release/claim refcount churn.
    pub fn mirror_band(&mut self, slot: usize, pages: &[BandPage], seq_len: usize) -> Result<bool> {
        ensure!(
            slot < self.num_slots,
            "TokenKVPool::mirror_band slot {slot} out of range {}",
            self.num_slots
        );
        for &BandPage(page) in pages {
            ensure!(
                page == EVICTED_PAGE || (page as usize) < self.max_total_pages,
                "TokenKVPool::mirror_band page {page} out of range {}",
                self.max_total_pages
            );
        }
        if pages
            .iter()
            .map(|p| p.0)
            .eq(self.page_indices[slot].iter().copied())
        {
            self.seq_lens[slot] = seq_len;
            return Ok(false);
        }
        let old_pages = std::mem::take(&mut self.page_indices[slot]);
        for page in old_pages.iter().copied() {
            if page == EVICTED_PAGE {
                continue;
            }
            let idx = page as usize;
            self.page_attach_count[idx] = self.page_attach_count[idx].saturating_sub(1);
            self.recycle_page_if_unreferenced(page);
        }
        for &BandPage(page) in pages {
            if page == EVICTED_PAGE {
                continue;
            }
            self.claim_mirrored_page(page);
        }
        self.page_indices[slot].extend(pages.iter().map(|p| p.0));
        self.seq_lens[slot] = seq_len;
        Ok(true)
    }

    /// Attach already-live pages to an empty slot.
    ///
    /// Monolith-era allocator-mode prefix attach (the deleted scheduler drove
    /// this directly). The rewrite path instead mirrors the host pool's page
    /// table via [`Self::mirror_slot`]; this method and the retain/COW
    /// machinery below stay for allocator-mode users and tests.
    ///
    /// Borrowed full pages are sealed shared prefix blocks. If `token_count`
    /// leaves the final page partial, that borrowed frontier is a read-only
    /// hot tail; the caller must not mutate it in place.
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

    /// Reject any page id outside the pool before byte-offset math runs.
    /// Codex review (9d63682d): the tier (#82/#83) and whole-slot swap
    /// transports feed these copies caller-built page lists; a corrupt list
    /// must fail loudly, not slice out of bounds.
    fn validate_page_ids(&self, pages: &[u32], what: &str) -> Result<()> {
        for &page in pages {
            anyhow::ensure!(
                (page as usize) < self.max_total_pages,
                "paged_kv {what}: page id {page} outside pool ({} pages)",
                self.max_total_pages
            );
        }
        Ok(())
    }

    pub fn copy_pages_to_host(&self, ctx: &DeviceContext, pages: &[u32]) -> Result<Vec<u8>> {
        let out = self.copy_pages_to_host_impl(ctx, pages)?;
        ctx.sync()?;
        Ok(out)
    }

    /// Like [`copy_pages_to_host`] but omits the trailing `ctx.sync()`. Use when
    /// the caller batches many per-layer copies and issues a single sync after
    /// the loop (avoids N-1 unnecessary stream fences).
    pub fn copy_pages_to_host_no_sync(
        &self,
        ctx: &DeviceContext,
        pages: &[u32],
    ) -> Result<Vec<u8>> {
        self.copy_pages_to_host_impl(ctx, pages)
    }

    fn copy_pages_to_host_impl(&self, ctx: &DeviceContext, pages: &[u32]) -> Result<Vec<u8>> {
        #[cfg(feature = "cuda")]
        {
            // Tier/swap transports hand this caller-built page lists; validate
            // every id before any byte math so a corrupt table fails loudly
            // here instead of slicing out of bounds below.
            self.validate_page_ids(pages, "copy_pages_to_host")?;
            let token_bytes = self.data_plane_bytes_per_page();
            let single_plane = self.is_single_plane();
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
                    // Packed records have no V plane (scale/norm branches
                    // below skip themselves via has_scales/has_norms).
                    if !single_plane {
                        out.extend_from_slice(
                            &ctx.stream
                                .clone_dtoh(&self.v_data[layer].slice(data_start..data_end))
                                .map_err(|e| anyhow!("paged_kv copy V page dtoh failed: {e}"))?,
                        );
                    }

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
            // See copy_pages_to_host: validate ids before any byte math.
            self.validate_page_ids(pages, "copy_pages_from_host")?;
            let stream = &ctx.stream;
            let token_bytes = self.data_plane_bytes_per_page();
            let single_plane = self.is_single_plane();
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
                    stream
                        .memcpy_htod(&payload[cursor..cursor + token_bytes], &mut k_view)
                        .map_err(|e| anyhow!("paged_kv copy K page htod failed: {e}"))?;
                    cursor += token_bytes;

                    // Packed records have no V plane (scale/norm branches
                    // below skip themselves via has_scales/has_norms).
                    if !single_plane {
                        let mut v_view = self.v_data[layer].slice_mut(data_start..data_end);
                        stream
                            .memcpy_htod(&payload[cursor..cursor + token_bytes], &mut v_view)
                            .map_err(|e| anyhow!("paged_kv copy V page htod failed: {e}"))?;
                        cursor += token_bytes;
                    }

                    if self.format.has_scales() {
                        let k_scales: Vec<f32> = payload
                            [cursor..cursor + scale_len * size_of::<f32>()]
                            .chunks_exact(size_of::<f32>())
                            .map(|c| f32::from_le_bytes(c.try_into().expect("f32 chunk")))
                            .collect();
                        cursor += scale_len * size_of::<f32>();
                        let mut k_scale_view =
                            self.k_scales[layer].slice_mut(scale_start..scale_end);
                        stream
                            .memcpy_htod(&k_scales, &mut k_scale_view)
                            .map_err(|e| anyhow!("paged_kv copy K scales htod failed: {e}"))?;

                        let v_scales: Vec<f32> = payload
                            [cursor..cursor + scale_len * size_of::<f32>()]
                            .chunks_exact(size_of::<f32>())
                            .map(|c| f32::from_le_bytes(c.try_into().expect("f32 chunk")))
                            .collect();
                        cursor += scale_len * size_of::<f32>();
                        let mut v_scale_view =
                            self.v_scales[layer].slice_mut(scale_start..scale_end);
                        stream
                            .memcpy_htod(&v_scales, &mut v_scale_view)
                            .map_err(|e| anyhow!("paged_kv copy V scales htod failed: {e}"))?;
                    }

                    if self.format.has_norms() {
                        let k_norms: Vec<u16> = payload
                            [cursor..cursor + scale_len * std::mem::size_of::<u16>()]
                            .chunks_exact(std::mem::size_of::<u16>())
                            .map(|c| u16::from_le_bytes(c.try_into().expect("u16 chunk")))
                            .collect();
                        cursor += scale_len * std::mem::size_of::<u16>();
                        let mut k_norm_view = self.k_norms[layer].slice_mut(scale_start..scale_end);
                        stream
                            .memcpy_htod(&k_norms, &mut k_norm_view)
                            .map_err(|e| anyhow!("paged_kv copy K norms htod failed: {e}"))?;

                        let v_norms: Vec<u16> = payload
                            [cursor..cursor + scale_len * std::mem::size_of::<u16>()]
                            .chunks_exact(std::mem::size_of::<u16>())
                            .map(|c| u16::from_le_bytes(c.try_into().expect("u16 chunk")))
                            .collect();
                        cursor += scale_len * std::mem::size_of::<u16>();
                        let mut v_norm_view = self.v_norms[layer].slice_mut(scale_start..scale_end);
                        stream
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
            if idx == EVICTED_PAGE {
                continue; // mid-decode recall already freed this page; no slot ref to drop
            }
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
            if idx == EVICTED_PAGE {
                continue; // mid-decode recall already freed this page; no slot ref to drop
            }
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

    /// **Write-through evict-drop**: release one *middle* physical page of a live
    /// slot back to the pool while the slot keeps decoding, leaving its logical
    /// length (`seq_lens[slot]`) and logical page-table length unchanged.
    ///
    /// The page is named by its *logical* page index within the slot. Its
    /// physical HBM page is recycled to `free_pages` (the real free — this is the
    /// flat-VRAM win), and the logical slot is overwritten with [`EVICTED_PAGE`]
    /// so the surviving pages stay logically addressable. Returns the freed
    /// physical page id, or `None` if that logical page was already evicted,
    /// pinned by a radix/detached refcount, or out of range.
    ///
    /// The dropped page's KV is the tier's responsibility (it was mirrored by the
    /// write-through verb before this call), so nothing is written back. No
    /// in-tree caller today (the `--kv-recall` driver was deleted, 3f826c204);
    /// kept for the remote-L3 hole-tolerance path.
    pub fn evict_slot_page(&mut self, slot: usize, logical_page: usize) -> Option<u32> {
        let page = *self.page_indices.get(slot)?.get(logical_page)?;
        if page == EVICTED_PAGE {
            return None; // already evicted
        }
        let page_idx = page as usize;
        // A page pinned by the radix/detached store (ref > 0) or shared by more
        // than this slot must not be freed out from under the other owner.
        if self.page_ref_count[page_idx] > 0 || self.page_attach_count[page_idx] > 1 {
            return None;
        }
        self.page_attach_count[page_idx] = self.page_attach_count[page_idx].saturating_sub(1);
        let freed = self.recycle_page_if_unreferenced(page);
        debug_assert!(
            freed,
            "evict_slot_page: unpinned single-owner page must recycle"
        );
        // Keep logical length intact: replace with the sentinel, never shrink.
        self.page_indices[slot][logical_page] = EVICTED_PAGE;
        Some(page)
    }

    /// Return a previously parked evicted page to the free list (the
    /// keepalive's one-step-later free). The page was parked OUT of `free_pages`
    /// for the whole keepalive step, so no `alloc_tokens` / `reinstate_slot_page`
    /// could have re-popped it; at release it is single-owner detached (attach 0,
    /// ref 0) and recycles cleanly.
    pub fn release_evicted_page(&mut self, page: u32) {
        if page == EVICTED_PAGE {
            return;
        }
        debug_assert_eq!(
            self.page_attach_count[page as usize], 0,
            "release_evicted_page: parked page {page} should have zero slot refs"
        );
        self.recycle_page_if_unreferenced(page);
    }

    /// **Re-recall prefetch reinstate**: the inverse of [`Self::evict_slot_page`].
    /// Allocate one fresh physical page from `free_pages` and bind it to the given
    /// *logical* page index of `slot`, which MUST currently hold an [`EVICTED_PAGE`]
    /// sentinel (its KV was evict-dropped and mirrored to the tier). Returns the
    /// fresh physical page id so the caller can H2D-copy the tier payload into it.
    ///
    /// Restores the dual-residency invariant exactly: `page_attach_count` for the
    /// new page starts at 1 (single owner, this slot), logical length is unchanged
    /// (a sentinel is swapped for a real id, never grown). `None` if the logical
    /// index is out of range, was not a sentinel (already resident — nothing to
    /// do), or the pool is out of free pages (caller keeps it evicted; no KV loss
    /// because the tier copy persists). NEVER called on the default decode path.
    pub fn reinstate_slot_page(&mut self, slot: usize, logical_page: usize) -> Option<u32> {
        let cur = *self.page_indices.get(slot)?.get(logical_page)?;
        if cur != EVICTED_PAGE {
            return None; // already resident
        }
        let new_page = self.free_pages.pop()?;
        self.page_attach_count[new_page as usize] = 1;
        self.page_indices[slot][logical_page] = new_page;
        Some(new_page)
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

    /// Decrement the external reference count on each page by one and return
    /// the set of pages that actually moved back to the free-page stack.
    ///
    /// A page whose refcount drops to zero is still not reclaimable while a
    /// live slot attaches it. In that case the page remains in its owner slot
    /// and is not returned. The returned `Vec<u32>` is informational —
    /// scheduler logs, metrics, or radix-cache bookkeeping — and means
    /// "pushed to `free_pages` during this call".
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
            if next == 0 && self.recycle_page_if_unreferenced(idx) {
                newly_freed.push(idx);
            }
        }
        newly_freed
    }

    pub fn page_indices(&self, slot: usize) -> &[u32] {
        &self.page_indices[slot]
    }

    pub fn seq_len(&self, slot: usize) -> usize {
        self.seq_lens[slot]
    }

    /// Monotonic identifier for the current logical occupant of `slot`.
    pub fn slot_epoch(&self, slot: usize) -> u64 {
        self.slot_epochs[slot]
    }

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
        (start_pos..start_pos + token_count)
            .map(|pos| {
                let page_idx = self.page_indices[slot][pos / self.page_size];
                page_idx * self.page_size as u32 + (pos % self.page_size) as u32
            })
            .collect()
    }

    /// Whether the pool has allocated buffers.
    pub fn is_active(&self) -> bool {
        !self.k_data.is_empty()
    }

    // `k_ptr` / `v_ptr` = the "write target" for decode_prep_paged:
    //   BF16 -> per-layer data buffer (also read by TileLang)
    //   FP8/INT8 → shared bf16 working buffer (quantized to pool after write)
    //
    // `k_data_ptr` / `v_data_ptr` = the quantized data buffer (read by attention):
    //   Used by fused-dequant INT8/FP8 attention.

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

    /// Split-KV attention workspace (FP8/INT8). Allocated at pool
    /// construction for quantized formats.
    pub fn quantized_attn_workspace(&self) -> anyhow::Result<&CudaSlice<u8>> {
        self.quantized_attn_workspace
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("quantized KV pool missing quantized_attn_workspace"))
    }

    pub fn k_data_slice(&self, layer: usize) -> &CudaSlice<u8> {
        &self.k_data[layer]
    }

    /// Mutable K data CudaSlice ref for a layer. Packed-record pools
    /// (`KVFormat::PackedBytes`) are single-plane - the whole record lives
    /// here - and their adapters (DSv4 #85 P2) need `slice_mut` views for
    /// memset / D2D restore paths that the raw-pointer accessors can't serve.
    pub fn k_data_slice_mut(&mut self, layer: usize) -> &mut CudaSlice<u8> {
        &mut self.k_data[layer]
    }

    // Convenience accessors that mirror the old PagedKVPool API so callers can
    // transition incrementally.

    /// Build TileLang paged-KV indptr array for a batch of slots.
    /// `indptr[i+1] - indptr[i]` = page count for request `i`.
    pub fn build_indptr(&self, slots: &[usize]) -> Vec<i32> {
        let mut indptr = Vec::with_capacity(slots.len() + 1);
        self.fill_indptr(slots, &mut indptr);
        indptr
    }

    pub fn fill_indptr<'a>(&self, slots: &[usize], scratch: &'a mut Vec<i32>) -> &'a [i32] {
        scratch.clear();
        scratch.reserve(slots.len() + 1);
        scratch.push(0);
        for &slot in slots {
            let last = *scratch
                .last()
                .expect("invariant: indptr always has at least one element (initialized with 0)");
            scratch.push(last + self.page_indices[slot].len() as i32);
        }
        scratch.as_slice()
    }

    pub fn build_indices(&self, slots: &[usize]) -> Vec<i32> {
        slots
            .iter()
            .flat_map(|&slot| self.page_indices[slot].iter().map(|&idx| idx as i32))
            .collect()
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

    pub fn build_last_page_lens(&self, slots: &[usize]) -> Vec<i32> {
        slots
            .iter()
            .map(|&slot| self.slot_last_page_len(slot) as i32)
            .collect()
    }

    pub fn fill_last_page_lens<'a>(&self, slots: &[usize], scratch: &'a mut Vec<i32>) -> &'a [i32] {
        scratch.clear();
        scratch.reserve(slots.len());
        scratch.extend(
            slots
                .iter()
                .map(|&slot| self.slot_last_page_len(slot) as i32),
        );
        scratch.as_slice()
    }
}

// Type alias for backward compatibility

/// Backward-compatible alias. New code should use `TokenKVPool` directly.
pub type PagedKVPool = TokenKVPool;

/// Default BF16 paged-KV page size used by M0.3.
pub const DEFAULT_PAGE_SIZE: usize = 16;

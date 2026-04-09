//! KV Cache — contiguous buffers for fused attention, with CPU offload support
//! and optional INT8 quantization.
//!
//! When the sequence length exceeds `max_gpu_seq_len`, the oldest KV blocks are
//! offloaded to CPU (host) memory. Before attention kernels run, `ensure_on_gpu()`
//! restores the full sequence to GPU so the kernels see a contiguous `0..seq_len` range.
//! After attention, `offload_to_host()` moves the prefix back to CPU to free GPU memory.
//!
//! ## INT8 Quantization
//!
//! When `dtype = KVCacheDtype::INT8`, KV data is stored as INT8 with per-head per-token
//! f32 scales. A shared bf16 working buffer (one layer's worth) is used for attention:
//! - `prepare_layer()`: dequantize INT8 → bf16 working buffer
//! - Attention kernels read/write the bf16 working buffer as usual
//! - `commit_layer()`: quantize newly written bf16 tokens → INT8 storage
//!
//! Memory savings: ~46% (INT8 storage + scales + 1 layer bf16 working vs full bf16).

use anyhow::{Result, anyhow};
use cudarc::driver::CudaSlice;
use half::bf16;
use log::info;

use crate::ops::kv_quant;
use crate::tensor::{DeviceContext, DeviceVec};

/// Block size for offloading (in tokens). Offload happens in multiples of this.
const OFFLOAD_BLOCK_SIZE: usize = 64;

/// KV cache data type (contiguous cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVCacheDtype {
    /// Full precision bf16 (default).
    BF16,
    /// Per-head per-token symmetric INT8 quantization.
    INT8,
}

/// KV pool storage format (paged pool).
///
/// Determines how KV data is stored in the TokenKVPool and which attention
/// kernel is used during batched decode:
/// - `FP8E4M3` → FlashInfer native (zero dequant overhead)
/// - `INT8` → self-built fused-dequant decode attention
/// - `BF16` → FlashInfer native (baseline)
/// - `TurboQuant` → rotation-based 2-4 bit (dequant → FlashInfer, Phase 1)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVFormat {
    /// Full precision bf16, 2 bytes/element. FlashInfer native.
    BF16,
    /// FP8 E4M3, 1 byte/element, no separate scale. FlashInfer native.
    FP8E4M3,
    /// INT8 + per-head per-token f32 scale. Fused-dequant attention.
    INT8,
    /// TurboQuant: rotation + Lloyd-Max quantization.
    /// `key_bits` and `val_bits` control compression (2-4 bits each).
    /// Phase 1: dequantize → bf16 working buffer → FlashInfer.
    TurboQuant { key_bits: u8, val_bits: u8 },
}

impl Default for KVFormat {
    fn default() -> Self {
        Self::BF16
    }
}

impl KVFormat {
    /// Bytes per element for the data buffer (excluding scales/norms).
    ///
    /// For TurboQuant this returns the packed bytes per coordinate (approximate),
    /// but the actual allocation uses [`Self::pool_bytes_per_token`] for accuracy.
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::FP8E4M3 => 1,
            Self::INT8 => 1,
            // Approximate: ceil(bits / 8) — actual layout is per-head packed.
            Self::TurboQuant { key_bits, .. } => {
                let effective = if key_bits == 3 { 4 } else { key_bits as usize };
                (effective + 7) / 8
            }
        }
    }

    /// Whether this format uses separate per-head per-token scale buffers (f32).
    pub fn has_scales(self) -> bool {
        matches!(self, Self::INT8)
    }

    /// Whether this format uses separate per-head per-token norm buffers (f16).
    pub fn has_norms(self) -> bool {
        matches!(self, Self::TurboQuant { .. })
    }

    /// Whether a bf16 working buffer is needed (for decode_prep_paged write target).
    /// FP8, INT8, and TurboQuant need a working buffer because decode_prep_paged
    /// outputs bf16, which then gets quantized into the pool.
    pub fn needs_work_buffer(self) -> bool {
        !matches!(self, Self::BF16)
    }

    /// Whether this is a TurboQuant format.
    pub fn is_turboquant(self) -> bool {
        matches!(self, Self::TurboQuant { .. })
    }

    /// Total bytes per token per KV head in the pool (data + norms).
    /// Used for accurate budget calculation.
    pub fn pool_bytes_per_kv_head(self, head_dim: usize) -> usize {
        match self {
            Self::BF16 => head_dim * 2,
            Self::FP8E4M3 => head_dim,
            Self::INT8 => head_dim + 4, // 1 byte/elem + 4 bytes f32 scale
            Self::TurboQuant { key_bits, .. } => {
                let packed =
                    crate::model::turboquant_state::packed_bytes_per_head(head_dim, key_bits);
                packed + 2 // + 2 bytes f16 norm
            }
        }
    }
}

impl Default for KVCacheDtype {
    fn default() -> Self {
        Self::BF16
    }
}

/// KV Cache — contiguous buffers for fused attention.
pub(crate) struct KVCache {
    // ─── BF16 storage (used when dtype=BF16) ───
    // [layer] -> contiguous buffer (num_kv_heads * max_seq * head_dim)
    k_cache: Vec<DeviceVec>,
    v_cache: Vec<DeviceVec>,

    // ─── INT8 storage (used when dtype=INT8) ───
    k_cache_q: Vec<CudaSlice<i8>>,
    v_cache_q: Vec<CudaSlice<i8>>,
    k_scales: Vec<CudaSlice<f32>>,
    v_scales: Vec<CudaSlice<f32>>,
    // Shared bf16 working buffers (1 layer, reused across layers)
    k_work: Option<DeviceVec>,
    v_work: Option<DeviceVec>,

    // ─── Common fields ───
    dtype: KVCacheDtype,
    seq_len: usize,
    head_dim: usize,
    num_layers: usize,
    num_kv_heads: usize,
    max_seq_len: usize,

    // --- CPU offload fields ---
    /// Maximum tokens to keep on GPU. When seq_len exceeds this, oldest tokens
    /// are offloaded to CPU. Defaults to max_seq_len (no offload).
    max_gpu_seq_len: usize,
    /// Number of tokens currently offloaded to CPU host memory.
    offloaded_len: usize,
    /// CPU shadow buffers for offloaded K data. [layer] -> Vec<bf16>.
    /// Each stores `offloaded_len * num_kv_heads * head_dim` elements.
    k_host: Vec<Vec<bf16>>,
    /// CPU shadow buffers for offloaded V data. [layer] -> Vec<bf16>.
    v_host: Vec<Vec<bf16>>,
    /// Whether the GPU buffers currently contain the full sequence (including
    /// data restored from CPU). This is set by `ensure_on_gpu()` and cleared
    /// by `offload_to_host()`.
    gpu_has_full_seq: bool,
}

impl KVCache {
    pub(crate) fn new(num_layers: usize, num_kv_heads: usize) -> Self {
        Self {
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            k_cache_q: Vec::new(),
            v_cache_q: Vec::new(),
            k_scales: Vec::new(),
            v_scales: Vec::new(),
            k_work: None,
            v_work: None,
            dtype: KVCacheDtype::BF16,
            seq_len: 0,
            head_dim: 0,
            num_layers,
            num_kv_heads,
            max_seq_len: 32768,
            max_gpu_seq_len: 32768,
            offloaded_len: 0,
            k_host: Vec::new(),
            v_host: Vec::new(),
            gpu_has_full_seq: true,
        }
    }

    /// Set KV cache quantization dtype. Must be called before `init_if_needed()`.
    pub(crate) fn set_dtype(&mut self, dtype: KVCacheDtype) {
        self.dtype = dtype;
        if dtype == KVCacheDtype::INT8 {
            info!("KV cache: INT8 quantization enabled (per-head per-token symmetric)");
        }
    }

    pub(crate) fn dtype(&self) -> KVCacheDtype {
        self.dtype
    }

    /// Set the maximum number of tokens to keep on GPU.
    /// Tokens beyond this limit will be offloaded to CPU.
    /// Must be called before `init_if_needed()`. The value is rounded down
    /// to the nearest `OFFLOAD_BLOCK_SIZE` boundary.
    pub(crate) fn set_max_gpu_seq_len(&mut self, max_gpu: usize) {
        // Round down to block boundary so offloads are block-aligned.
        let aligned = (max_gpu / OFFLOAD_BLOCK_SIZE) * OFFLOAD_BLOCK_SIZE;
        // Ensure at least one block stays on GPU.
        self.max_gpu_seq_len = aligned.max(OFFLOAD_BLOCK_SIZE);
        info!(
            "KV cache: max_gpu_seq_len set to {} tokens ({} blocks of {})",
            self.max_gpu_seq_len,
            self.max_gpu_seq_len / OFFLOAD_BLOCK_SIZE,
            OFFLOAD_BLOCK_SIZE,
        );
    }

    /// Set the maximum sequence length (total, GPU + CPU).
    /// Must be called before `init_if_needed()`.
    pub(crate) fn set_max_seq_len(&mut self, max_seq: usize) {
        self.max_seq_len = max_seq;
        // If max_gpu_seq_len hasn't been explicitly set (still at old default),
        // update it to match.
        if self.max_gpu_seq_len == 32768 && max_seq != 32768 {
            self.max_gpu_seq_len = max_seq;
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.seq_len
    }

    /// Returns true if offloading is active (some tokens are on CPU).
    pub(crate) fn has_offloaded(&self) -> bool {
        self.offloaded_len > 0
    }

    /// Number of tokens currently offloaded to CPU.
    pub(crate) fn offloaded_len(&self) -> usize {
        self.offloaded_len
    }

    /// Number of tokens currently on GPU.
    pub(crate) fn gpu_seq_len(&self) -> usize {
        self.seq_len - self.offloaded_len
    }

    /// Elements per token per layer in the KV cache (num_kv_heads * head_dim).
    fn elems_per_token(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    // ========================================================================
    // Initialization
    // ========================================================================

    pub(crate) fn init_if_needed(&mut self, ctx: &DeviceContext, head_dim: usize) -> Result<()> {
        if self.head_dim != 0 {
            return Ok(());
        }

        let cache_size = self.num_kv_heads * self.max_seq_len * head_dim;

        match self.dtype {
            KVCacheDtype::BF16 => {
                let mut k_tmp = Vec::with_capacity(self.num_layers);
                let mut v_tmp = Vec::with_capacity(self.num_layers);
                for _ in 0..self.num_layers {
                    k_tmp.push(DeviceVec::zeros(ctx, cache_size)?);
                    v_tmp.push(DeviceVec::zeros(ctx, cache_size)?);
                }
                self.k_cache = k_tmp;
                self.v_cache = v_tmp;
            }
            KVCacheDtype::INT8 => {
                let scale_size = self.num_kv_heads * self.max_seq_len;

                let mut kq = Vec::with_capacity(self.num_layers);
                let mut vq = Vec::with_capacity(self.num_layers);
                let mut ks = Vec::with_capacity(self.num_layers);
                let mut vs = Vec::with_capacity(self.num_layers);
                for _ in 0..self.num_layers {
                    kq.push(
                        ctx.stream
                            .alloc_zeros::<i8>(cache_size)
                            .map_err(|e| anyhow!("INT8 K alloc failed: {e}"))?,
                    );
                    vq.push(
                        ctx.stream
                            .alloc_zeros::<i8>(cache_size)
                            .map_err(|e| anyhow!("INT8 V alloc failed: {e}"))?,
                    );
                    ks.push(
                        ctx.stream
                            .alloc_zeros::<f32>(scale_size)
                            .map_err(|e| anyhow!("K scales alloc failed: {e}"))?,
                    );
                    vs.push(
                        ctx.stream
                            .alloc_zeros::<f32>(scale_size)
                            .map_err(|e| anyhow!("V scales alloc failed: {e}"))?,
                    );
                }
                self.k_cache_q = kq;
                self.v_cache_q = vq;
                self.k_scales = ks;
                self.v_scales = vs;

                // Shared bf16 working buffers (1 layer's worth)
                self.k_work = Some(DeviceVec::zeros(ctx, cache_size)?);
                self.v_work = Some(DeviceVec::zeros(ctx, cache_size)?);

                let int8_bytes = self.num_layers * cache_size; // K+V each
                let scale_bytes = self.num_layers * scale_size * 4;
                let work_bytes = cache_size * 2 * 2; // K+V, bf16=2 bytes
                let bf16_bytes = self.num_layers * cache_size * 2 * 2;
                info!(
                    "KV cache INT8: storage={:.1}MB scales={:.1}MB working={:.1}MB (was {:.1}MB bf16, saving {:.0}%)",
                    (int8_bytes * 2) as f64 / 1e6,
                    (scale_bytes * 2) as f64 / 1e6,
                    work_bytes as f64 / 1e6,
                    bf16_bytes as f64 / 1e6,
                    (1.0 - (int8_bytes * 2 + scale_bytes * 2 + work_bytes) as f64
                        / bf16_bytes as f64)
                        * 100.0,
                );
            }
        }

        self.head_dim = head_dim;
        self.k_host = vec![Vec::new(); self.num_layers];
        self.v_host = vec![Vec::new(); self.num_layers];

        Ok(())
    }

    // ========================================================================
    // Layer access: prepare / commit pattern
    // ========================================================================

    /// Get mutable references to K/V cache for a layer (BF16 mode only).
    ///
    /// For INT8 mode, use `prepare_layer()` / `commit_layer()` instead.
    pub(crate) fn get_cache_mut(
        &mut self,
        ctx: &DeviceContext,
        layer: usize,
    ) -> Result<(&mut DeviceVec, &mut DeviceVec)> {
        // Initialize on first access (legacy path)
        if self.head_dim == 0 {
            // Can't determine head_dim here — caller should call init_if_needed first.
            let cache_size = self.num_kv_heads * self.max_seq_len * 128; // fallback
            let mut k_tmp = Vec::with_capacity(self.num_layers);
            let mut v_tmp = Vec::with_capacity(self.num_layers);
            for _ in 0..self.num_layers {
                k_tmp.push(DeviceVec::zeros(ctx, cache_size)?);
                v_tmp.push(DeviceVec::zeros(ctx, cache_size)?);
            }
            self.k_cache = k_tmp;
            self.v_cache = v_tmp;
            self.head_dim = 128;
        }
        Ok((&mut self.k_cache[layer], &mut self.v_cache[layer]))
    }

    /// Prepare a layer's KV cache for attention.
    ///
    /// - **BF16**: Returns references to the per-layer bf16 buffers (no-op).
    /// - **INT8**: Dequantizes `[0..seq_len)` from INT8 storage into shared bf16
    ///   working buffers, then returns references to those working buffers.
    ///
    /// After attention completes, call `commit_layer()` to quantize new tokens back.
    pub(crate) fn prepare_layer(
        &mut self,
        ctx: &DeviceContext,
        layer: usize,
    ) -> Result<(&mut DeviceVec, &mut DeviceVec)> {
        match self.dtype {
            KVCacheDtype::BF16 => Ok((&mut self.k_cache[layer], &mut self.v_cache[layer])),
            KVCacheDtype::INT8 => {
                // Dequantize existing tokens [0..seq_len) → bf16 working buffers
                let seq_len = self.seq_len;
                let num_kv_heads = self.num_kv_heads;
                let head_dim = self.head_dim;
                let max_seq_len = self.max_seq_len;

                if seq_len > 0 {
                    // Borrow INT8 storage immutably, working buffers mutably
                    // Need to use raw pointer trick to satisfy borrow checker
                    let k_int8 = &self.k_cache_q[layer];
                    let v_int8 = &self.v_cache_q[layer];
                    let k_sc = &self.k_scales[layer];
                    let v_sc = &self.v_scales[layer];
                    let k_work = self.k_work.as_mut().expect("INT8 k_work not initialized");
                    kv_quant::dequantize_kv(
                        ctx,
                        k_int8,
                        k_sc,
                        k_work,
                        num_kv_heads,
                        head_dim,
                        max_seq_len,
                        seq_len,
                    )?;
                    let v_work = self.v_work.as_mut().expect("INT8 v_work not initialized");
                    kv_quant::dequantize_kv(
                        ctx,
                        v_int8,
                        v_sc,
                        v_work,
                        num_kv_heads,
                        head_dim,
                        max_seq_len,
                        seq_len,
                    )?;
                }

                let k_work = self.k_work.as_mut().expect("INT8 k_work not initialized");
                let v_work = self.v_work.as_mut().expect("INT8 v_work not initialized");
                Ok((k_work, v_work))
            }
        }
    }

    /// Commit newly written tokens from bf16 working buffer → INT8 storage.
    ///
    /// - **BF16**: No-op.
    /// - **INT8**: Quantizes `[start_pos..start_pos+token_count)` from the bf16
    ///   working buffer into INT8 storage.
    ///
    /// # Arguments
    /// * `start_pos` — First token position written by the attention kernel.
    /// * `token_count` — Number of new tokens written.
    pub(crate) fn commit_layer(
        &mut self,
        ctx: &DeviceContext,
        layer: usize,
        start_pos: usize,
        token_count: usize,
    ) -> Result<()> {
        if self.dtype != KVCacheDtype::INT8 || token_count == 0 {
            return Ok(());
        }

        let num_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;
        let max_seq_len = self.max_seq_len;

        // Quantize K working → INT8
        {
            let k_work = self.k_work.as_ref().expect("INT8 k_work not initialized");
            let k_int8 = &mut self.k_cache_q[layer];
            let k_sc = &mut self.k_scales[layer];
            kv_quant::quantize_kv(
                ctx,
                k_work,
                k_int8,
                k_sc,
                num_kv_heads,
                head_dim,
                max_seq_len,
                start_pos,
                token_count,
            )?;
        }

        // Quantize V working → INT8
        {
            let v_work = self.v_work.as_ref().expect("INT8 v_work not initialized");
            let v_int8 = &mut self.v_cache_q[layer];
            let v_sc = &mut self.v_scales[layer];
            kv_quant::quantize_kv(
                ctx,
                v_work,
                v_int8,
                v_sc,
                num_kv_heads,
                head_dim,
                max_seq_len,
                start_pos,
                token_count,
            )?;
        }

        Ok(())
    }

    // ========================================================================
    // Sequence management
    // ========================================================================

    pub(crate) fn increment_seq_len(&mut self) {
        self.seq_len += 1;
    }

    /// Access contiguous K caches for migration to paged pool.
    pub(crate) fn k_caches(&self) -> &[DeviceVec] {
        &self.k_cache
    }

    /// Access contiguous V caches for migration to paged pool.
    pub(crate) fn v_caches(&self) -> &[DeviceVec] {
        &self.v_cache
    }

    /// Access contiguous INT8 K caches for migration to paged pool.
    pub(crate) fn k_caches_q(&self) -> &[CudaSlice<i8>] {
        &self.k_cache_q
    }

    /// Access contiguous INT8 V caches for migration to paged pool.
    pub(crate) fn v_caches_q(&self) -> &[CudaSlice<i8>] {
        &self.v_cache_q
    }

    /// Access contiguous K scales for migration to paged pool.
    pub(crate) fn k_scales(&self) -> &[CudaSlice<f32>] {
        &self.k_scales
    }

    /// Access contiguous V scales for migration to paged pool.
    pub(crate) fn v_scales(&self) -> &[CudaSlice<f32>] {
        &self.v_scales
    }

    /// Maximum sequence length (for contiguous offset calculation).
    pub(crate) fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub(crate) fn advance_seq_len(&mut self, count: usize) {
        self.seq_len += count;
    }

    /// Truncate KV cache to `new_len` tokens, discarding everything after.
    /// Used for partial prefix reuse: keep the common prefix, discard divergent suffix.
    /// If data was offloaded to CPU, keeps offloaded data up to `new_len`.
    pub(crate) fn truncate_to(&mut self, new_len: usize) {
        if new_len >= self.seq_len {
            return; // Nothing to truncate
        }
        if new_len == 0 {
            self.reset();
            return;
        }
        if new_len <= self.offloaded_len {
            // Truncation point is within the offloaded region.
            // Keep only the CPU data up to new_len.
            let ept = self.elems_per_token();
            let keep_elems = new_len * ept;
            for buf in &mut self.k_host {
                buf.truncate(keep_elems);
            }
            for buf in &mut self.v_host {
                buf.truncate(keep_elems);
            }
            self.offloaded_len = new_len;
            // GPU portion is now empty (all remaining data was past new_len).
            self.seq_len = new_len;
            self.gpu_has_full_seq = false;
        } else {
            // Truncation point is within the GPU region.
            // Just update seq_len; GPU data past new_len is stale but won't be read.
            self.seq_len = new_len;
        }
        info!(
            "KV cache truncated to {} tokens (offloaded: {}, gpu: {})",
            self.seq_len,
            self.offloaded_len,
            self.gpu_seq_len()
        );
    }

    /// Prefetch offloaded KV data from CPU back to GPU.
    /// Call this before prefill to ensure the full prefix is on GPU.
    /// This is a no-op if nothing is offloaded.
    pub(crate) fn prefetch_to_gpu(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.offloaded_len == 0 || self.gpu_has_full_seq {
            return Ok(());
        }
        info!(
            "KV cache prefetch: restoring {} tokens from CPU to GPU",
            self.offloaded_len
        );
        self.ensure_on_gpu(ctx)?;
        Ok(())
    }

    /// Reset sequence length to 0 for reuse across requests.
    /// Keeps allocated GPU buffers (stable GPU pointers for CUDA Graph replay).
    /// Clears CPU offload state.
    pub(crate) fn reset(&mut self) {
        self.seq_len = 0;
        self.offloaded_len = 0;
        self.gpu_has_full_seq = true;
        // Clear host buffers but keep the Vec allocations.
        for buf in &mut self.k_host {
            buf.clear();
        }
        for buf in &mut self.v_host {
            buf.clear();
        }
    }

    // ========================================================================
    // CPU offload (BF16 mode only — INT8 offload is future work)
    // ========================================================================

    /// Offload oldest KV blocks to CPU if GPU seq_len exceeds the budget.
    ///
    /// Call this after `advance_seq_len()` or `increment_seq_len()` when the
    /// GPU buffer might be over capacity. This copies the oldest blocks to CPU
    /// and shifts the remaining GPU data to the start of the buffer.
    ///
    /// The GPU buffer layout after offload:
    /// - Positions `0..gpu_seq_len` contain the most recent tokens.
    /// - The CUDA kernels will be told `seq_len = gpu_seq_len` until
    ///   `ensure_on_gpu()` is called.
    pub(crate) fn offload_if_needed(&mut self, ctx: &DeviceContext) -> Result<()> {
        // INT8 mode: offload not yet supported
        if self.dtype == KVCacheDtype::INT8 {
            return Ok(());
        }

        // If the full sequence was restored to GPU by ensure_on_gpu(), we need to
        // first move the prefix back to its CPU-only state before computing what
        // else needs offloading. Otherwise we'd re-offload already-offloaded data.
        if self.gpu_has_full_seq && self.offloaded_len > 0 {
            // GPU has [0..seq_len] with [0..offloaded_len] being the restored prefix.
            // Shift the new portion left so GPU has [0..gpu_tokens] = only the new data.
            let ept = self.elems_per_token();
            let offloaded_elems = self.offloaded_len * ept;
            let gpu_tokens = self.seq_len - self.offloaded_len;
            let gpu_elems = gpu_tokens * ept;

            if gpu_elems > 0 {
                for layer in 0..self.num_layers {
                    let mut temp = vec![bf16::ZERO; gpu_elems];

                    self.k_cache[layer].copy_region_to_host(
                        ctx,
                        offloaded_elems,
                        gpu_elems,
                        &mut temp,
                    )?;
                    self.k_cache[layer].copy_region_from_host(ctx, 0, &temp)?;

                    self.v_cache[layer].copy_region_to_host(
                        ctx,
                        offloaded_elems,
                        gpu_elems,
                        &mut temp,
                    )?;
                    self.v_cache[layer].copy_region_from_host(ctx, 0, &temp)?;
                }
                ctx.sync()?;
            }
            self.gpu_has_full_seq = false;
        }

        let gpu_tokens = self.seq_len - self.offloaded_len;
        if gpu_tokens <= self.max_gpu_seq_len {
            return Ok(());
        }

        // Calculate how many tokens to offload (in whole blocks).
        let excess = gpu_tokens - self.max_gpu_seq_len;
        let blocks_to_offload = (excess + OFFLOAD_BLOCK_SIZE - 1) / OFFLOAD_BLOCK_SIZE;
        let tokens_to_offload = blocks_to_offload * OFFLOAD_BLOCK_SIZE;
        // Don't offload more than what's on GPU.
        let tokens_to_offload =
            tokens_to_offload.min(gpu_tokens.saturating_sub(OFFLOAD_BLOCK_SIZE));

        if tokens_to_offload == 0 {
            return Ok(());
        }

        let ept = self.elems_per_token();
        let offload_elems = tokens_to_offload * ept;

        info!(
            "KV cache offload: moving {} tokens ({} blocks) to CPU (total offloaded: {})",
            tokens_to_offload,
            tokens_to_offload / OFFLOAD_BLOCK_SIZE,
            self.offloaded_len + tokens_to_offload,
        );

        // GPU buffer now has [0..gpu_tokens] = only the non-offloaded portion.
        // Copy oldest `tokens_to_offload` tokens to CPU, then shift remaining left.
        for layer in 0..self.num_layers {
            let mut host_buf = vec![bf16::ZERO; offload_elems];

            self.k_cache[layer].copy_region_to_host(ctx, 0, offload_elems, &mut host_buf)?;
            self.k_host[layer].extend_from_slice(&host_buf);

            self.v_cache[layer].copy_region_to_host(ctx, 0, offload_elems, &mut host_buf)?;
            self.v_host[layer].extend_from_slice(&host_buf);

            let remaining_tokens = gpu_tokens - tokens_to_offload;
            let remaining_elems = remaining_tokens * ept;
            if remaining_elems > 0 {
                let src_offset = offload_elems;
                let mut temp = vec![bf16::ZERO; remaining_elems];

                self.k_cache[layer].copy_region_to_host(
                    ctx,
                    src_offset,
                    remaining_elems,
                    &mut temp,
                )?;
                self.k_cache[layer].copy_region_from_host(ctx, 0, &temp)?;

                self.v_cache[layer].copy_region_to_host(
                    ctx,
                    src_offset,
                    remaining_elems,
                    &mut temp,
                )?;
                self.v_cache[layer].copy_region_from_host(ctx, 0, &temp)?;
            }
        }

        ctx.sync()?;

        self.offloaded_len += tokens_to_offload;
        self.gpu_has_full_seq = false;

        Ok(())
    }

    /// Ensure the full sequence (including offloaded tokens) is on GPU.
    ///
    /// Call this before attention kernels that need to scan the full KV range.
    /// This copies CPU-offloaded data back to the GPU buffer, shifting current
    /// GPU data to make room for the restored prefix.
    ///
    /// After this call, the GPU buffer contains `0..seq_len` contiguously,
    /// and `gpu_has_full_seq` is true.
    pub(crate) fn ensure_on_gpu(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.offloaded_len == 0 || self.gpu_has_full_seq {
            return Ok(());
        }

        let ept = self.elems_per_token();
        let offloaded_elems = self.offloaded_len * ept;
        let gpu_tokens = self.seq_len - self.offloaded_len;
        let gpu_elems = gpu_tokens * ept;

        info!(
            "KV cache restore: copying {} offloaded tokens back to GPU",
            self.offloaded_len,
        );

        for layer in 0..self.num_layers {
            // 1. Shift current GPU data right to make room for the prefix.
            // Move [0..gpu_elems] to [offloaded_elems..offloaded_elems+gpu_elems].
            // Bounce through host to avoid borrow conflicts on CudaSlice.
            if gpu_elems > 0 {
                let mut temp = vec![bf16::ZERO; gpu_elems];

                self.k_cache[layer].copy_region_to_host(ctx, 0, gpu_elems, &mut temp)?;
                self.k_cache[layer].copy_region_from_host(ctx, offloaded_elems, &temp)?;

                self.v_cache[layer].copy_region_to_host(ctx, 0, gpu_elems, &mut temp)?;
                self.v_cache[layer].copy_region_from_host(ctx, offloaded_elems, &temp)?;
            }

            // 2. Copy offloaded prefix from CPU to GPU [0..offloaded_elems].
            self.k_cache[layer].copy_region_from_host(ctx, 0, &self.k_host[layer])?;
            self.v_cache[layer].copy_region_from_host(ctx, 0, &self.v_host[layer])?;
        }

        ctx.sync()?;
        self.gpu_has_full_seq = true;

        Ok(())
    }

    /// Move restored prefix data back to CPU after attention is done.
    ///
    /// Call this after attention kernels have finished reading the full sequence.
    /// Shifts GPU data back to position 0 so new tokens can be appended at
    /// the correct offset within the GPU-resident portion.
    pub(crate) fn offload_to_host(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.offloaded_len == 0 || !self.gpu_has_full_seq {
            return Ok(());
        }

        let ept = self.elems_per_token();
        let offloaded_elems = self.offloaded_len * ept;
        let gpu_tokens = self.seq_len - self.offloaded_len;
        let gpu_elems = gpu_tokens * ept;

        for layer in 0..self.num_layers {
            // Shift GPU data left: move [offloaded_elems..offloaded_elems+gpu_elems]
            // to [0..gpu_elems]. Bounce through host to avoid borrow conflicts.
            if gpu_elems > 0 {
                let mut temp = vec![bf16::ZERO; gpu_elems];

                self.k_cache[layer].copy_region_to_host(
                    ctx,
                    offloaded_elems,
                    gpu_elems,
                    &mut temp,
                )?;
                self.k_cache[layer].copy_region_from_host(ctx, 0, &temp)?;

                self.v_cache[layer].copy_region_to_host(
                    ctx,
                    offloaded_elems,
                    gpu_elems,
                    &mut temp,
                )?;
                self.v_cache[layer].copy_region_from_host(ctx, 0, &temp)?;
            }
        }

        ctx.sync()?;
        self.gpu_has_full_seq = false;

        Ok(())
    }
}

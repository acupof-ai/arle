//! KV Cache — contiguous buffers for fused attention, with optional INT8
//! quantization.
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
use log::info;

use infer_cuda_kernels::kv_quant;
use infer_cuda_kernels::prelude::{DeviceContext, DeviceVec};
pub use infer_cuda_kernels::{KVCacheDtype, KVFormat};

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

    /// Set the maximum contiguous sequence length.
    /// Must be called before `init_if_needed()`.
    pub(crate) fn set_max_seq_len(&mut self, max_seq: usize) {
        self.max_seq_len = max_seq;
    }

    pub(crate) fn len(&self) -> usize {
        self.seq_len
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

    /// Mutable access to the K/V cache pair for one layer.
    pub(crate) fn layer_kv_caches_mut(&mut self, layer: usize) -> (&mut DeviceVec, &mut DeviceVec) {
        (&mut self.k_cache[layer], &mut self.v_cache[layer])
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
    pub(crate) fn truncate_to(&mut self, new_len: usize) {
        if new_len >= self.seq_len {
            return; // Nothing to truncate
        }
        if new_len == 0 {
            self.reset();
            return;
        }
        self.seq_len = new_len;
        info!("KV cache truncated to {} tokens", self.seq_len);
    }

    /// Reset sequence length to 0 for reuse across requests.
    /// Keeps allocated GPU buffers (stable GPU pointers for CUDA Graph replay).
    pub(crate) fn reset(&mut self) {
        self.seq_len = 0;
    }
}

//! Metal paged KV cache pool — token-level KV storage for Apple Silicon.
//!
//! Mirrors the design of [`TokenKVPool`](crate::paged_kv::TokenKVPool) (CUDA)
//! but uses MLX `Array` tensors on Metal unified memory instead of `CudaSlice`
//! buffers.
//!
//! Since MLX does not have native paged attention, the strategy is:
//!   1. Store KV in a flat pool: `[max_total_tokens, kv_dim]` per layer.
//!   2. `write_kv` — scatter-write new K/V rows into the pool at allocated slots.
//!   3. `gather_kv` — gather rows by index, reshape to `[1, n_kv_heads, seq, head_dim]`.
//!   4. Feed gathered contiguous tensors to `fast::scaled_dot_product_attention`.
//!
//! This enables multi-sequence support when wired to a scheduler, without
//! requiring any changes to the attention implementation.
//!
//! # Feature flag
//!
//! All GPU-touching types/methods are behind `#[cfg(feature = "metal")]`.
//! The module itself is always compiled (like `metal_backend`).

#[cfg(feature = "metal")]
use std::collections::HashMap;

#[cfg(feature = "metal")]
use anyhow::{Result, anyhow};

#[cfg(feature = "metal")]
use mlx_rs::Array;

/// Token-level KV cache pool for the Metal backend.
///
/// Each layer has a K pool and a V pool, each shaped `[max_total_tokens, kv_dim]`.
/// Individual token slots are allocated from a LIFO free list and tracked per
/// request via `slot_indices`.
// GPU required: Array is backed by Metal unified memory.
#[cfg(feature = "metal")]
pub struct MetalKVPool {
    /// K buffers per layer: each `[max_total_tokens, kv_dim]` bf16/f16.
    k_pool: Vec<Array>,
    /// V buffers per layer: each `[max_total_tokens, kv_dim]` bf16/f16.
    v_pool: Vec<Array>,

    /// Free token slot indices (stack-based allocator, LIFO).
    free_slots: Vec<u32>,

    /// Per-request token mappings: `slot_indices[request_id][i]` = physical pool
    /// index for logical position `i` of that request.
    slot_indices: HashMap<usize, Vec<u32>>,

    // ── Config ────────────────────────────────────────────────────────────────
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// `num_kv_heads * head_dim` — stride for one token row in the pool buffer.
    kv_dim: usize,
    max_total_tokens: usize,
}

#[cfg(feature = "metal")]
impl MetalKVPool {
    /// Create a new token-level KV pool on Metal unified memory.
    ///
    /// All pool buffers are allocated up-front as zeros with the given dtype
    /// (typically `Bfloat16` or `Float16`, matching the model's KV dtype).
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_total_tokens: usize,
        dtype: mlx_rs::Dtype,
    ) -> Result<Self> {
        use mlx_rs::{StreamOrDevice, ops::zeros_dtype_device};

        let kv_dim = num_kv_heads * head_dim;
        let pool_shape = [max_total_tokens as i32, kv_dim as i32];

        let mut k_pool = Vec::with_capacity(num_layers);
        let mut v_pool = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let k = zeros_dtype_device(&pool_shape, dtype, StreamOrDevice::default())
                .map_err(|e| anyhow!("MetalKVPool K alloc failed: {e}"))?;
            let v = zeros_dtype_device(&pool_shape, dtype, StreamOrDevice::default())
                .map_err(|e| anyhow!("MetalKVPool V alloc failed: {e}"))?;
            k_pool.push(k);
            v_pool.push(v);
        }

        // Initialize free list: all token slots free, highest indices first (LIFO).
        let free_slots: Vec<u32> = (0..max_total_tokens as u32).rev().collect();

        log::info!(
            "MetalKVPool: {} max tokens, {} layers ({} kv_heads x {} head_dim, kv_dim={})",
            max_total_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dim,
        );

        Ok(Self {
            k_pool,
            v_pool,
            free_slots,
            slot_indices: HashMap::new(),
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dim,
            max_total_tokens,
        })
    }

    /// Allocate `count` token slots for a request.
    ///
    /// Returns the newly allocated physical pool indices. These are appended
    /// to the request's token list in `slot_indices`.
    pub fn alloc_tokens(&mut self, request_id: usize, count: usize) -> Result<Vec<u32>> {
        if count > self.free_slots.len() {
            return Err(anyhow!(
                "MetalKVPool: out of token slots (requested {}, available {})",
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
        self.slot_indices
            .entry(request_id)
            .or_default()
            .extend_from_slice(&new_indices);
        Ok(new_indices)
    }

    /// Free all token slots for a request, returning them to the pool.
    pub fn free_request(&mut self, request_id: usize) {
        if let Some(indices) = self.slot_indices.remove(&request_id) {
            for idx in indices {
                self.free_slots.push(idx);
            }
        }
    }

    /// Scatter-write K/V tensors into the pool at the request's token positions.
    ///
    /// `k` and `v` should each be shaped `[num_new_tokens, kv_dim]` (2D), where
    /// `num_new_tokens` matches the number of most-recently-allocated slots for
    /// this request that haven't been written yet.
    ///
    /// Internally this writes each token row to its allocated pool index using
    /// MLX scatter via `try_index_mut`.
    pub fn write_kv(
        &mut self,
        layer: usize,
        request_id: usize,
        k: &Array,
        v: &Array,
    ) -> Result<()> {
        use mlx_rs::ops::indexing::{TryIndexMutOp, take_axis};

        let indices = self
            .slot_indices
            .get(&request_id)
            .ok_or_else(|| anyhow!("MetalKVPool: unknown request_id {request_id}"))?;

        let num_tokens = k.shape().first().copied().unwrap_or(0) as usize;
        if num_tokens == 0 {
            return Ok(());
        }

        // The new tokens correspond to the LAST `num_tokens` entries in the
        // request's slot_indices (the most recently allocated ones).
        if num_tokens > indices.len() {
            return Err(anyhow!(
                "MetalKVPool: write_kv got {} tokens but request {request_id} only has {} slots",
                num_tokens,
                indices.len()
            ));
        }
        let write_indices = &indices[indices.len() - num_tokens..];

        let k_pool = &mut self.k_pool[layer];

        // MLX `try_index_mut` supports contiguous slice ranges. We write one row
        // at a time using `pool[pi..pi+1] = row`. This is correct but not optimal;
        // a future batched scatter would be faster for large writes.
        for (i, &pool_idx) in write_indices.iter().enumerate() {
            // take_axis with a 1-element index returns [1, kv_dim] — matches the
            // slice range pi..pi+1 for try_index_mut.
            let row_k = take_axis(k, Array::from_slice(&[i as i32], &[1]), 0)
                .map_err(|e| anyhow!("take k row {i}: {e}"))?;
            let row_v = take_axis(v, Array::from_slice(&[i as i32], &[1]), 0)
                .map_err(|e| anyhow!("take v row {i}: {e}"))?;

            let pi = pool_idx as i32;
            k_pool
                .try_index_mut(pi..pi + 1, &row_k)
                .map_err(|e| anyhow!("scatter k[{pool_idx}]: {e}"))?;
            self.v_pool[layer]
                .try_index_mut(pi..pi + 1, &row_v)
                .map_err(|e| anyhow!("scatter v[{pool_idx}]: {e}"))?;
        }

        Ok(())
    }

    /// Gather K/V from the pool for a request, returning contiguous tensors
    /// shaped `[1, n_kv_heads, seq_len, head_dim]` ready for attention.
    pub fn gather_kv(&self, layer: usize, request_id: usize) -> Result<(Array, Array)> {
        use mlx_rs::ops::{indexing::take_axis, reshape, transpose_axes};

        let indices = self
            .slot_indices
            .get(&request_id)
            .ok_or_else(|| anyhow!("MetalKVPool: unknown request_id {request_id}"))?;

        let seq_len = indices.len() as i32;
        if seq_len == 0 {
            return Err(anyhow!(
                "MetalKVPool: gather_kv called with no tokens for request {request_id}"
            ));
        }

        // Build index array from the request's token positions.
        let idx_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
        let idx_arr = Array::from_slice(&idx_i32, &[seq_len]);

        // Gather rows from pool: result is [seq_len, kv_dim].
        let k_gathered = take_axis(&self.k_pool[layer], &idx_arr, 0)
            .map_err(|e| anyhow!("gather k layer {layer}: {e}"))?;
        let v_gathered = take_axis(&self.v_pool[layer], &idx_arr, 0)
            .map_err(|e| anyhow!("gather v layer {layer}: {e}"))?;

        // Reshape to [1, n_kv_heads, seq_len, head_dim] for attention.
        let n_kv = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;
        let k_out =
            reshape(&k_gathered, &[1, seq_len, n_kv, hd]).map_err(|e| anyhow!("reshape k: {e}"))?;
        let v_out =
            reshape(&v_gathered, &[1, seq_len, n_kv, hd]).map_err(|e| anyhow!("reshape v: {e}"))?;

        // Transpose from [1, seq_len, n_kv_heads, head_dim] to
        //                 [1, n_kv_heads, seq_len, head_dim].
        let k_out =
            transpose_axes(&k_out, &[0, 2, 1, 3]).map_err(|e| anyhow!("transpose k: {e}"))?;
        let v_out =
            transpose_axes(&v_out, &[0, 2, 1, 3]).map_err(|e| anyhow!("transpose v: {e}"))?;

        Ok((k_out, v_out))
    }

    /// Number of token slots currently in use across all requests.
    pub fn total_tokens_used(&self) -> usize {
        self.max_total_tokens - self.free_slots.len()
    }

    /// Number of free token slots remaining in the pool.
    pub fn available_tokens(&self) -> usize {
        self.free_slots.len()
    }

    /// Get token indices for a request (physical pool indices, in logical order).
    pub fn token_indices(&self, request_id: usize) -> Option<&[u32]> {
        self.slot_indices.get(&request_id).map(Vec::as_slice)
    }

    /// Number of layers in the pool.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of KV heads per layer.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// `num_kv_heads * head_dim` — row stride in the pool buffers.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Maximum total tokens the pool can hold.
    pub fn max_total_tokens(&self) -> usize {
        self.max_total_tokens
    }
}

// ── Stub for non-Metal builds ────────────────────────────────────────────────

/// Stub `MetalKVPool` when the `metal` feature is not enabled.
///
/// All methods panic at runtime — this exists only so that code referencing
/// the type can compile on non-Metal targets.
#[cfg(not(feature = "metal"))]
pub struct MetalKVPool {
    _private: (),
}

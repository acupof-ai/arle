//! Metal paged KV cache pool — token-level KV storage for Apple Silicon.
//!
//! Mirrors the design of [`TokenKVPool`](crate::backend::cuda::paged_kv::TokenKVPool) (CUDA)
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

use std::collections::HashMap;

use anyhow::{anyhow, Result};

#[cfg(feature = "metal")]
use super::mlx::MlxArray;

/// Pure Rust token-slot ledger used by the Metal KV pool.
///
/// This is the metadata layer only:
/// - slot allocation / freeing
/// - request -> slot mappings
/// - prefix sharing via reference counts
///
/// It has no GPU dependency and can be unit-tested on CPU-only builds.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
#[derive(Debug, Clone)]
struct SlotLedger {
    max_total_tokens: usize,
    free_slots: Vec<u32>,
    slot_refcounts: Vec<u32>,
    request_slots: HashMap<usize, Vec<u32>>,
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
impl SlotLedger {
    fn new(max_total_tokens: usize) -> Self {
        let free_slots = (0..max_total_tokens as u32).rev().collect();
        Self {
            max_total_tokens,
            free_slots,
            slot_refcounts: vec![0; max_total_tokens],
            request_slots: HashMap::new(),
        }
    }

    fn available_tokens(&self) -> usize {
        self.free_slots.len()
    }

    fn total_tokens_used(&self) -> usize {
        self.max_total_tokens - self.free_slots.len()
    }

    fn token_indices(&self, request_id: usize) -> Option<&[u32]> {
        self.request_slots.get(&request_id).map(Vec::as_slice)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn slot_refcount(&self, slot: u32) -> Result<u32> {
        let idx = self.slot_index(slot)?;
        Ok(self.slot_refcounts[idx])
    }

    fn alloc_tokens(&mut self, request_id: usize, count: usize) -> Result<Vec<u32>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if count > self.free_slots.len() {
            return Err(anyhow!(
                "MetalKVPool: out of token slots (requested {}, available {})",
                count,
                self.free_slots.len()
            ));
        }

        let mut new_indices = Vec::with_capacity(count);
        for _ in 0..count {
            let idx = self
                .free_slots
                .pop()
                .expect("invariant: free_slots.len() >= count checked above");
            self.bump_refcount(idx)?;
            new_indices.push(idx);
        }
        self.request_slots
            .entry(request_id)
            .or_default()
            .extend_from_slice(&new_indices);
        Ok(new_indices)
    }

    fn share_slots(&mut self, request_id: usize, slots: &[u32]) -> Result<Vec<u32>> {
        if slots.is_empty() {
            return Ok(Vec::new());
        }

        for &slot in slots {
            self.bump_refcount(slot)?;
        }

        let entry = self.request_slots.entry(request_id).or_default();
        entry.extend_from_slice(slots);
        Ok(slots.to_vec())
    }

    fn share_prefix_from(
        &mut self,
        request_id: usize,
        source_request_id: usize,
        prefix_tokens: usize,
    ) -> Result<Vec<u32>> {
        if prefix_tokens == 0 {
            return Ok(Vec::new());
        }

        let source_slots = self
            .request_slots
            .get(&source_request_id)
            .ok_or_else(|| anyhow!("MetalKVPool: unknown source request_id {source_request_id}"))?;
        let prefix = source_slots.get(0..prefix_tokens).ok_or_else(|| {
            anyhow!(
                "MetalKVPool: source request {source_request_id} only has {} slots",
                source_slots.len()
            )
        })?;
        let prefix = prefix.to_vec();
        self.share_slots(request_id, &prefix)
    }

    fn free_request(&mut self, request_id: usize) {
        if let Some(indices) = self.request_slots.remove(&request_id) {
            for idx in indices {
                self.release_slot(idx);
            }
        }
    }

    fn max_total_tokens(&self) -> usize {
        self.max_total_tokens
    }

    fn slot_index(&self, slot: u32) -> Result<usize> {
        let idx =
            usize::try_from(slot).map_err(|_| anyhow!("MetalKVPool: invalid token slot {slot}"))?;
        if idx >= self.max_total_tokens {
            return Err(anyhow!(
                "MetalKVPool: token slot {} out of range (max {})",
                slot,
                self.max_total_tokens
            ));
        }
        Ok(idx)
    }

    fn bump_refcount(&mut self, slot: u32) -> Result<()> {
        let idx = self.slot_index(slot)?;
        self.slot_refcounts[idx] = self
            .slot_refcounts
            .get(idx)
            .copied()
            .ok_or_else(|| anyhow!("MetalKVPool: slot index {slot} missing"))?
            .saturating_add(1);
        Ok(())
    }

    fn release_slot(&mut self, slot: u32) {
        let Ok(idx) = self.slot_index(slot) else {
            return;
        };
        let Some(count) = self.slot_refcounts.get_mut(idx) else {
            return;
        };
        if *count == 0 {
            debug_assert!(*count > 0, "MetalKVPool: release_slot saw zero refcount");
            return;
        }
        *count -= 1;
        if *count == 0 {
            self.free_slots.push(slot);
        }
    }
}

/// Token-level KV cache pool for the Metal backend.
///
/// Each layer has a K pool and a V pool, each shaped `[max_total_tokens, kv_dim]`.
/// Individual token slots are allocated from a LIFO free list and tracked per
/// request via `slot_indices`.
// GPU required: Array is backed by Metal unified memory.
#[cfg(feature = "metal")]
pub struct MetalKVPool {
    ledger: SlotLedger,
    /// K buffers per layer: each `[max_total_tokens, kv_dim]` bf16/f16.
    k_pool: Vec<MlxArray>,
    /// V buffers per layer: each `[max_total_tokens, kv_dim]` bf16/f16.
    v_pool: Vec<MlxArray>,

    // ── Config ────────────────────────────────────────────────────────────────
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// `num_kv_heads * head_dim` — stride for one token row in the pool buffer.
    kv_dim: usize,
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
        dtype: super::mlx::Dtype,
    ) -> Result<Self> {
        use super::mlx::zeros;

        let kv_dim = num_kv_heads * head_dim;
        let pool_shape = [max_total_tokens as i32, kv_dim as i32];
        let ledger = SlotLedger::new(max_total_tokens);

        let mut k_pool = Vec::with_capacity(num_layers);
        let mut v_pool = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            k_pool.push(zeros(&pool_shape, dtype));
            v_pool.push(zeros(&pool_shape, dtype));
        }

        log::info!(
            "MetalKVPool: {} max tokens, {} layers ({} kv_heads x {} head_dim, kv_dim={})",
            max_total_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dim,
        );

        Ok(Self {
            ledger,
            k_pool,
            v_pool,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dim,
        })
    }

    /// Allocate `count` token slots for a request.
    ///
    /// Returns the newly allocated physical pool indices. These are appended
    /// to the request's token list in `slot_indices`.
    pub fn alloc_tokens(&mut self, request_id: usize, count: usize) -> Result<Vec<u32>> {
        self.ledger.alloc_tokens(request_id, count)
    }

    /// Share an existing token-slot slice with `request_id`.
    ///
    /// Each slot in `slots` has its reference count incremented and is appended
    /// to the request's logical token sequence.
    pub fn share_slots(&mut self, request_id: usize, slots: &[u32]) -> Result<Vec<u32>> {
        self.ledger.share_slots(request_id, slots)
    }

    /// Share a prefix of another request's token slots with `request_id`.
    pub fn share_prefix_from(
        &mut self,
        request_id: usize,
        source_request_id: usize,
        prefix_tokens: usize,
    ) -> Result<Vec<u32>> {
        self.ledger
            .share_prefix_from(request_id, source_request_id, prefix_tokens)
    }

    /// Free all token slots for a request, returning them to the pool.
    pub fn free_request(&mut self, request_id: usize) {
        self.ledger.free_request(request_id);
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
        k: &MlxArray,
        v: &MlxArray,
    ) -> Result<()> {
        use super::mlx::{slice_update, take_axis};

        let indices = self
            .ledger
            .token_indices(request_id)
            .ok_or_else(|| anyhow!("MetalKVPool: unknown request_id {request_id}"))?;

        let num_tokens = k.shape().first().copied().unwrap_or(0) as usize;
        if num_tokens == 0 {
            return Ok(());
        }

        if num_tokens > indices.len() {
            return Err(anyhow!(
                "MetalKVPool: write_kv got {} tokens but request {request_id} only has {} slots",
                num_tokens,
                indices.len()
            ));
        }
        let write_indices = &indices[indices.len() - num_tokens..];
        let kv_dim = self.kv_dim as i32;

        for (i, &pool_idx) in write_indices.iter().enumerate() {
            let idx_arr = MlxArray::from_slice_i32(&[i as i32], &[1]);
            let row_k = take_axis(k, &idx_arr, 0);
            let row_v = take_axis(v, &idx_arr, 0);

            let pi = pool_idx as i32;
            let start = [pi, 0];
            let stop = [pi + 1, kv_dim];
            self.k_pool[layer] = slice_update(&mut self.k_pool[layer], &row_k, &start, &stop);
            self.v_pool[layer] = slice_update(&mut self.v_pool[layer], &row_v, &start, &stop);
        }

        Ok(())
    }

    /// Gather K/V from the pool for a request, returning contiguous tensors
    /// shaped `[1, n_kv_heads, seq_len, head_dim]` ready for attention.
    pub fn gather_kv(&self, layer: usize, request_id: usize) -> Result<(MlxArray, MlxArray)> {
        use super::mlx::{reshape, take_axis, transpose_axes};

        let indices = self
            .ledger
            .token_indices(request_id)
            .ok_or_else(|| anyhow!("MetalKVPool: unknown request_id {request_id}"))?;

        let seq_len = indices.len() as i32;
        if seq_len == 0 {
            return Err(anyhow!(
                "MetalKVPool: gather_kv called with no tokens for request {request_id}"
            ));
        }

        let idx_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
        let idx_arr = MlxArray::from_slice_i32(&idx_i32, &[seq_len]);

        let k_gathered = take_axis(&self.k_pool[layer], &idx_arr, 0);
        let v_gathered = take_axis(&self.v_pool[layer], &idx_arr, 0);

        let n_kv = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;
        let k_out = reshape(&k_gathered, &[1, seq_len, n_kv, hd]);
        let v_out = reshape(&v_gathered, &[1, seq_len, n_kv, hd]);

        let k_out = transpose_axes(&k_out, &[0, 2, 1, 3]);
        let v_out = transpose_axes(&v_out, &[0, 2, 1, 3]);

        Ok((k_out, v_out))
    }

    /// Number of token slots currently in use across all requests.
    pub fn total_tokens_used(&self) -> usize {
        self.ledger.total_tokens_used()
    }

    /// Number of free token slots remaining in the pool.
    pub fn available_tokens(&self) -> usize {
        self.ledger.available_tokens()
    }

    /// Get token indices for a request (physical pool indices, in logical order).
    pub fn token_indices(&self, request_id: usize) -> Option<&[u32]> {
        self.ledger.token_indices(request_id)
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
        self.ledger.max_total_tokens()
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

#[cfg(test)]
mod tests {
    use super::SlotLedger;

    #[test]
    fn alloc_and_free_round_trip() {
        let mut ledger = SlotLedger::new(4);
        let slots = ledger.alloc_tokens(1, 2).expect("alloc");
        assert_eq!(slots, vec![0, 1]);
        assert_eq!(ledger.available_tokens(), 2);
        assert_eq!(ledger.slot_refcount(0).unwrap(), 1);
        assert_eq!(ledger.slot_refcount(1).unwrap(), 1);

        ledger.free_request(1);
        assert_eq!(ledger.available_tokens(), 4);
        assert_eq!(ledger.slot_refcount(0).unwrap(), 0);
        assert_eq!(ledger.slot_refcount(1).unwrap(), 0);
        assert!(ledger.token_indices(1).is_none());
    }

    #[test]
    fn share_prefix_increments_refcounts() {
        let mut ledger = SlotLedger::new(8);
        let source = ledger.alloc_tokens(10, 4).expect("source alloc");
        assert_eq!(source, vec![0, 1, 2, 3]);

        let shared = ledger.share_prefix_from(11, 10, 2).expect("share prefix");
        assert_eq!(shared, vec![0, 1]);
        assert_eq!(ledger.token_indices(11).unwrap(), &[0, 1]);
        assert_eq!(ledger.slot_refcount(0).unwrap(), 2);
        assert_eq!(ledger.slot_refcount(1).unwrap(), 2);
        assert_eq!(ledger.slot_refcount(2).unwrap(), 1);
        assert_eq!(ledger.slot_refcount(3).unwrap(), 1);
        assert_eq!(ledger.available_tokens(), 4);

        ledger.free_request(10);
        assert_eq!(ledger.available_tokens(), 6);
        assert_eq!(ledger.slot_refcount(0).unwrap(), 1);
        assert_eq!(ledger.slot_refcount(1).unwrap(), 1);
        assert_eq!(ledger.slot_refcount(2).unwrap(), 0);
        assert_eq!(ledger.slot_refcount(3).unwrap(), 0);

        ledger.free_request(11);
        assert_eq!(ledger.available_tokens(), 8);
    }

    #[test]
    fn free_request_is_idempotent() {
        let mut ledger = SlotLedger::new(2);
        ledger.alloc_tokens(1, 1).expect("alloc");
        ledger.free_request(1);
        let available_after_first_free = ledger.available_tokens();
        ledger.free_request(1);
        assert_eq!(ledger.available_tokens(), available_after_first_free);
    }

    #[test]
    fn share_prefix_rejects_bad_requests() {
        let mut ledger = SlotLedger::new(2);
        assert!(ledger.share_prefix_from(2, 1, 1).is_err());

        ledger.alloc_tokens(1, 1).expect("alloc");
        assert!(ledger.share_prefix_from(2, 1, 2).is_err());
        assert!(ledger.token_indices(2).is_none());
        assert_eq!(ledger.slot_refcount(0).unwrap(), 1);
    }

    #[test]
    fn share_slots_allows_manual_aliasing() {
        let mut ledger = SlotLedger::new(4);
        ledger.alloc_tokens(1, 2).expect("alloc");
        let shared = ledger.share_slots(2, &[1, 0, 1]).expect("share");
        assert_eq!(shared, vec![1, 0, 1]);
        assert_eq!(ledger.token_indices(2).unwrap(), &[1, 0, 1]);
        assert_eq!(ledger.slot_refcount(0).unwrap(), 2);
        assert_eq!(ledger.slot_refcount(1).unwrap(), 3);

        ledger.free_request(1);
        assert_eq!(ledger.slot_refcount(0).unwrap(), 1);
        assert_eq!(ledger.slot_refcount(1).unwrap(), 2);

        ledger.free_request(2);
        assert_eq!(ledger.available_tokens(), 4);
    }
}

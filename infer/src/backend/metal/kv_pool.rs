//! Metal paged KV cache pool — token-level KV storage for Apple Silicon.
//!
//! Mirrors the design of `TokenKVPool` (CUDA, in `cuda-kernels::paged_kv`)
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

use anyhow::{Result, anyhow};

#[cfg(feature = "metal")]
use super::mlx::{MlxArray, reshape, transpose_axes};

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
    slot_last_access: Vec<u64>,
    request_slots: HashMap<usize, Vec<u32>>,
    next_access_tick: u64,
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
impl SlotLedger {
    fn new(max_total_tokens: usize) -> Self {
        let free_slots = (0..max_total_tokens as u32).rev().collect();
        Self {
            max_total_tokens,
            free_slots,
            slot_refcounts: vec![0; max_total_tokens],
            slot_last_access: vec![0; max_total_tokens],
            request_slots: HashMap::new(),
            next_access_tick: 1,
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
        let new_indices = self.alloc_detached_slots(count)?;
        self.request_slots
            .entry(request_id)
            .or_default()
            .extend_from_slice(&new_indices);
        Ok(new_indices)
    }

    fn alloc_detached_slots(&mut self, count: usize) -> Result<Vec<u32>> {
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

        let access_tick = self.bump_access_tick();
        let mut new_indices = Vec::with_capacity(count);
        for _ in 0..count {
            let idx = self
                .free_slots
                .pop()
                .expect("invariant: free_slots.len() >= count checked above");
            self.bump_refcount(idx)?;
            self.mark_access_at(idx, access_tick)?;
            new_indices.push(idx);
        }
        Ok(new_indices)
    }

    fn share_slots(&mut self, request_id: usize, slots: &[u32]) -> Result<Vec<u32>> {
        if slots.is_empty() {
            return Ok(Vec::new());
        }

        for &slot in slots {
            self.slot_index(slot)?;
        }
        let access_tick = self.bump_access_tick();
        for &slot in slots {
            self.bump_refcount(slot)?;
            self.mark_access_at(slot, access_tick)?;
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

    fn release_slots(&mut self, slots: &[u32]) -> Result<()> {
        for &slot in slots {
            self.slot_index(slot)?;
        }
        for &slot in slots {
            self.release_slot(slot);
        }
        Ok(())
    }

    fn max_total_tokens(&self) -> usize {
        self.max_total_tokens
    }

    fn register_access(&mut self, slots: &[u32]) -> Result<()> {
        if slots.is_empty() {
            return Ok(());
        }
        for &slot in slots {
            self.slot_index(slot)?;
        }
        let access_tick = self.bump_access_tick();
        for &slot in slots {
            self.mark_access_at(slot, access_tick)?;
        }
        Ok(())
    }

    fn reclaim_target_tokens(
        &self,
        high_watermark: f64,
        low_watermark: f64,
    ) -> Result<Option<usize>> {
        if !high_watermark.is_finite()
            || !low_watermark.is_finite()
            || !(0.0..=1.0).contains(&high_watermark)
            || !(0.0..=1.0).contains(&low_watermark)
            || low_watermark > high_watermark
        {
            return Err(anyhow!(
                "MetalKVPool: invalid watermarks low={} high={}",
                low_watermark,
                high_watermark
            ));
        }

        let used = self.total_tokens_used();
        let high_tokens = (self.max_total_tokens as f64) * high_watermark;
        if (used as f64) <= high_tokens {
            return Ok(None);
        }

        let low_tokens = ((self.max_total_tokens as f64) * low_watermark).floor() as usize;
        Ok(Some(used.saturating_sub(low_tokens)))
    }

    fn select_eviction_candidates(
        &self,
        candidate_blocks: &[Vec<u32>],
        target_tokens: usize,
    ) -> Result<Vec<Vec<u32>>> {
        if target_tokens == 0 || candidate_blocks.is_empty() {
            return Ok(Vec::new());
        }

        let active_slots = self.active_slot_bitmap();
        let mut candidates = Vec::new();
        for block in candidate_blocks {
            if block.is_empty() {
                continue;
            }

            let mut newest_access = 0;
            let mut eligible = true;
            for &slot in block {
                let idx = self.slot_index(slot)?;
                if active_slots[idx] || self.slot_refcounts[idx] == 0 {
                    eligible = false;
                    break;
                }
                newest_access = newest_access.max(self.slot_last_access[idx]);
            }
            if eligible {
                candidates.push((newest_access, block.clone()));
            }
        }

        candidates.sort_by_key(|(last_access, _)| *last_access);

        let mut selected = Vec::new();
        let mut selected_tokens = 0usize;
        for (_, block) in candidates {
            selected_tokens = selected_tokens.saturating_add(block.len());
            selected.push(block);
            if selected_tokens >= target_tokens {
                break;
            }
        }
        Ok(selected)
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

    fn active_slot_bitmap(&self) -> Vec<bool> {
        let mut active = vec![false; self.max_total_tokens];
        for slots in self.request_slots.values() {
            for &slot in slots {
                if let Ok(idx) = self.slot_index(slot) {
                    active[idx] = true;
                }
            }
        }
        active
    }

    fn bump_access_tick(&mut self) -> u64 {
        let tick = self.next_access_tick;
        self.next_access_tick = self.next_access_tick.saturating_add(1).max(1);
        tick
    }

    fn mark_access_at(&mut self, slot: u32, tick: u64) -> Result<()> {
        let idx = self.slot_index(slot)?;
        self.slot_last_access[idx] = tick;
        Ok(())
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
            if let Some(last_access) = self.slot_last_access.get_mut(idx) {
                *last_access = 0;
            }
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

    /// Allocate detached token slots not associated with any active request.
    pub fn alloc_slots(&mut self, count: usize) -> Result<Vec<u32>> {
        self.ledger.alloc_detached_slots(count)
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

    /// Release detached token slots back into the pool.
    pub fn release_slots(&mut self, slots: &[u32]) -> Result<()> {
        self.ledger.release_slots(slots)
    }

    /// Mark token slots as recently accessed for LRU eviction accounting.
    pub fn register_access(&mut self, slots: &[u32]) -> Result<()> {
        self.ledger.register_access(slots)
    }

    /// Return how many tokens should be reclaimed to move from high to low
    /// watermark, or `None` when the pool is below the high watermark.
    pub fn reclaim_target_tokens(
        &self,
        high_watermark: f64,
        low_watermark: f64,
    ) -> Result<Option<usize>> {
        self.ledger
            .reclaim_target_tokens(high_watermark, low_watermark)
    }

    /// Select eviction candidates from caller-supplied cache blocks in LRU
    /// order. The ledger protects active request slots; the caller owns block
    /// alignment and cache metadata.
    pub fn select_eviction_candidates(
        &self,
        candidate_blocks: &[Vec<u32>],
        target_tokens: usize,
    ) -> Result<Vec<Vec<u32>>> {
        self.ledger
            .select_eviction_candidates(candidate_blocks, target_tokens)
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
        let write_indices = indices[indices.len() - num_tokens..].to_vec();
        self.write_kv_slots(layer, &write_indices, k, v)
    }

    /// Scatter-write K/V tensors into explicitly provided token slots.
    pub fn write_kv_slots(
        &mut self,
        layer: usize,
        slot_indices: &[u32],
        k: &MlxArray,
        v: &MlxArray,
    ) -> Result<()> {
        use super::mlx::{slice_update, take_axis};

        let num_tokens = k.shape().first().copied().unwrap_or(0) as usize;
        if num_tokens == 0 {
            return Ok(());
        }
        if num_tokens != slot_indices.len() {
            return Err(anyhow!(
                "MetalKVPool: write_kv_slots got {} tokens for {} explicit slots",
                num_tokens,
                slot_indices.len()
            ));
        }
        let kv_dim = self.kv_dim as i32;

        for (i, &pool_idx) in slot_indices.iter().enumerate() {
            self.ledger.slot_index(pool_idx)?;
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
        let indices = self
            .ledger
            .token_indices(request_id)
            .ok_or_else(|| anyhow!("MetalKVPool: unknown request_id {request_id}"))?;
        let (k_gathered, v_gathered) = self.gather_kv_rows(layer, indices)?;

        let seq_len = indices.len() as i32;
        let n_kv = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;
        let k_out = reshape(&k_gathered, &[1, seq_len, n_kv, hd]);
        let v_out = reshape(&v_gathered, &[1, seq_len, n_kv, hd]);

        let k_out = transpose_axes(&k_out, &[0, 2, 1, 3]);
        let v_out = transpose_axes(&v_out, &[0, 2, 1, 3]);

        Ok((k_out, v_out))
    }

    /// Gather raw `[seq_len, kv_dim]` K/V rows from explicit token slots.
    pub fn gather_kv_rows(
        &self,
        layer: usize,
        slot_indices: &[u32],
    ) -> Result<(MlxArray, MlxArray)> {
        use super::mlx::take_axis;

        let seq_len = slot_indices.len() as i32;
        if seq_len == 0 {
            return Err(anyhow!("MetalKVPool: gather_kv_rows called with no slots"));
        }
        for &slot in slot_indices {
            self.ledger.slot_index(slot)?;
        }

        let idx_i32: Vec<i32> = slot_indices.iter().map(|&i| i as i32).collect();
        let idx_arr = MlxArray::from_slice_i32(&idx_i32, &[seq_len]);

        let k_gathered = take_axis(&self.k_pool[layer], &idx_arr, 0);
        let v_gathered = take_axis(&self.v_pool[layer], &idx_arr, 0);
        Ok((k_gathered, v_gathered))
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

    #[test]
    fn access_tracking_selects_oldest_detached_blocks() {
        let mut ledger = SlotLedger::new(8);
        let old = ledger.alloc_detached_slots(2).expect("old block");
        let active = ledger.alloc_tokens(7, 2).expect("active block");
        let new = ledger.alloc_detached_slots(2).expect("new block");

        ledger.register_access(&new).expect("touch new");
        ledger.register_access(&old).expect("touch old");
        ledger.register_access(&new).expect("retouch new");

        let candidates = vec![new.clone(), active, old.clone()];
        let selected = ledger
            .select_eviction_candidates(&candidates, 2)
            .expect("select candidates");

        assert_eq!(selected, vec![old]);
    }

    #[test]
    fn eviction_selection_skips_free_and_active_slots() {
        let mut ledger = SlotLedger::new(6);
        let detached = ledger.alloc_detached_slots(2).expect("detached");
        let active = ledger.alloc_tokens(1, 2).expect("active");
        let free = ledger.alloc_detached_slots(2).expect("free");
        ledger.release_slots(&free).expect("release free");

        let candidates = vec![active, free, detached.clone()];
        let selected = ledger
            .select_eviction_candidates(&candidates, 4)
            .expect("select candidates");

        assert_eq!(selected, vec![detached]);
    }

    #[test]
    fn watermarks_compute_reclaim_target() {
        let mut ledger = SlotLedger::new(10);
        ledger.alloc_detached_slots(7).expect("alloc");

        assert_eq!(ledger.reclaim_target_tokens(0.6, 0.4).unwrap(), Some(3));
        assert_eq!(ledger.reclaim_target_tokens(0.7, 0.4).unwrap(), None);
        assert_eq!(ledger.reclaim_target_tokens(0.95, 0.8).unwrap(), None);
        assert!(ledger.reclaim_target_tokens(0.3, 0.4).is_err());
    }

    #[test]
    fn watermarks_trigger_when_used_exceeds_fractional_high_mark() {
        let mut ledger = SlotLedger::new(10);
        ledger.alloc_detached_slots(10).expect("alloc");

        assert_eq!(ledger.reclaim_target_tokens(0.95, 0.8).unwrap(), Some(2));
    }

    #[test]
    fn detached_slots_round_trip_without_request_ownership() {
        let mut ledger = SlotLedger::new(4);
        let detached = ledger.alloc_detached_slots(2).expect("alloc detached");
        assert_eq!(detached, vec![0, 1]);
        assert_eq!(ledger.available_tokens(), 2);
        assert_eq!(ledger.slot_refcount(0).unwrap(), 1);
        assert_eq!(ledger.slot_refcount(1).unwrap(), 1);

        ledger.release_slots(&detached).expect("release detached");
        assert_eq!(ledger.available_tokens(), 4);
        assert_eq!(ledger.slot_refcount(0).unwrap(), 0);
        assert_eq!(ledger.slot_refcount(1).unwrap(), 0);
    }
}

//! Metal prefix cache bridge.
//!
//! This layer adapts [`crate::prefix_cache::RadixCache`] to the Metal token-slot
//! model used by `MetalKVPool`:
//! - inserts are token prefixes plus their physical slot indices
//! - lookup returns the longest block-aligned shared prefix and the slots to reuse
//! - release/evict operate on slot indices, not opaque block IDs

use std::collections::HashMap;

use anyhow::{Result, anyhow, ensure};

use crate::prefix_cache::{BlockId, RadixCache};

/// Result of a Metal prefix lookup.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MetalPrefixHit {
    /// Number of tokens covered by the shared prefix.
    pub matched_len: usize,
    /// Physical token slots that back the matched prefix.
    pub slot_indices: Vec<u32>,
}

/// Bridge cache between radix prefix hits and Metal token slots.
pub struct MetalPrefixCache {
    radix: RadixCache,
    block_slots: HashMap<BlockId, Vec<u32>>,
    slot_to_block: HashMap<u32, BlockId>,
    next_block_id: u32,
    block_size: usize,
}

impl MetalPrefixCache {
    /// Create an empty cache.
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        Self {
            radix: RadixCache::new(block_size),
            block_slots: HashMap::new(),
            slot_to_block: HashMap::new(),
            next_block_id: 0,
            block_size,
        }
    }

    /// Block size shared by the radix tree and the Metal token ledger.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of cached full blocks.
    pub fn cached_block_count(&self) -> usize {
        self.radix.cached_block_count()
    }

    /// Insert a token prefix together with its physical slot indices.
    ///
    /// Only full blocks are cached. Partial trailing tokens remain uncached.
    /// Returns the number of cached tokens, rounded down to a block boundary.
    pub fn insert(&mut self, tokens: &[u32], slot_indices: &[u32]) -> Result<usize> {
        ensure!(
            tokens.len() == slot_indices.len(),
            "token and slot lengths must match"
        );

        let full_len = self.aligned_len(tokens.len());
        if full_len == 0 {
            return Ok(0);
        }

        let (matched_len, matched_blocks) = self.radix.lookup(&tokens[..full_len]);
        self.radix.release(&matched_blocks);

        ensure!(
            matched_len <= full_len,
            "radix lookup returned an invalid match length"
        );

        let matched_blocks_len = matched_len / self.block_size;
        let total_blocks = full_len / self.block_size;

        let mut blocks = Vec::with_capacity(total_blocks);

        for (block_idx, block_id) in matched_blocks.iter().copied().enumerate() {
            let slots = self.block_slots.get(&block_id).ok_or_else(|| {
                anyhow!("MetalPrefixCache: missing slot mapping for cached block {block_id:?}")
            })?;
            let expected =
                &slot_indices[block_idx * self.block_size..(block_idx + 1) * self.block_size];
            ensure!(
                slots.as_slice() == expected,
                "prefix already cached with different slot indices"
            );
            blocks.push(block_id);
        }

        for block_idx in matched_blocks_len..total_blocks {
            let block_id = BlockId(self.next_block_id);
            self.next_block_id = self
                .next_block_id
                .checked_add(1)
                .ok_or_else(|| anyhow!("MetalPrefixCache: block id overflow"))?;
            let slots = slot_indices
                [block_idx * self.block_size..(block_idx + 1) * self.block_size]
                .to_vec();
            self.record_block(block_id, &slots)?;
            blocks.push(block_id);
        }

        if matched_blocks_len < total_blocks {
            self.radix.insert(&tokens[..full_len], &blocks);
        }

        Ok(full_len)
    }

    /// Lookup the longest cached prefix for `tokens`.
    pub fn lookup(&mut self, tokens: &[u32]) -> Result<MetalPrefixHit> {
        let (matched_len, blocks) = self.radix.lookup(tokens);
        let slot_indices = match self.blocks_to_slots(&blocks) {
            Ok(slots) => slots,
            Err(err) => {
                self.radix.release(&blocks);
                return Err(err);
            }
        };

        Ok(MetalPrefixHit {
            matched_len,
            slot_indices,
        })
    }

    /// Release a previously looked-up shared prefix.
    ///
    /// The provided slots must be block-aligned and correspond to a cached prefix.
    pub fn release(&mut self, slot_indices: &[u32]) -> Result<()> {
        if slot_indices.is_empty() {
            return Ok(());
        }
        ensure!(
            slot_indices.len().is_multiple_of(self.block_size),
            "release requires block-aligned slot indices"
        );

        let mut blocks = Vec::with_capacity(slot_indices.len() / self.block_size);
        for chunk in slot_indices.chunks(self.block_size) {
            let block = self.block_for_slots(chunk)?;
            blocks.push(block);
        }
        blocks.sort_unstable();
        blocks.dedup();
        self.radix.release(&blocks);
        Ok(())
    }

    /// Evict up to `n` cached blocks in LRU order.
    ///
    /// Returns the freed slot indices, flattened in block order.
    pub fn evict(&mut self, n: usize) -> Vec<u32> {
        let freed_blocks = self.radix.evict(n);
        let mut freed_slots = Vec::new();

        for block_id in freed_blocks {
            if let Some(slots) = self.block_slots.remove(&block_id) {
                for slot in &slots {
                    self.slot_to_block.remove(slot);
                }
                freed_slots.extend(slots);
            }
        }

        freed_slots
    }

    /// Total number of slot indices pinned by cached blocks.
    pub fn cached_slot_count(&self) -> usize {
        self.block_slots.len() * self.block_size
    }

    fn aligned_len(&self, len: usize) -> usize {
        (len / self.block_size) * self.block_size
    }

    fn blocks_to_slots(&self, blocks: &[BlockId]) -> Result<Vec<u32>> {
        let mut slots = Vec::with_capacity(blocks.len() * self.block_size);
        for &block in blocks {
            let block_slots = self.block_slots.get(&block).ok_or_else(|| {
                anyhow!("MetalPrefixCache: missing slot mapping for cached block {block:?}")
            })?;
            slots.extend_from_slice(block_slots);
        }
        Ok(slots)
    }

    fn block_for_slots(&self, slots: &[u32]) -> Result<BlockId> {
        ensure!(
            slots.len() == self.block_size,
            "slot chunk must be exactly one block"
        );
        let first = self
            .slot_to_block
            .get(&slots[0])
            .copied()
            .ok_or_else(|| anyhow!("MetalPrefixCache: unknown slot {}", slots[0]))?;
        for &slot in &slots[1..] {
            let block = self
                .slot_to_block
                .get(&slot)
                .copied()
                .ok_or_else(|| anyhow!("MetalPrefixCache: unknown slot {slot}"))?;
            ensure!(block == first, "slot chunk spans multiple blocks");
        }
        Ok(first)
    }

    fn record_block(&mut self, block_id: BlockId, slots: &[u32]) -> Result<()> {
        ensure!(
            slots.len() == self.block_size,
            "block mapping must contain exactly one block of slots"
        );

        if let Some(existing) = self.block_slots.get(&block_id) {
            ensure!(
                existing.as_slice() == slots,
                "block {block_id:?} already mapped to different slots"
            );
            return Ok(());
        }

        for &slot in slots {
            if let Some(existing_block) = self.slot_to_block.get(&slot).copied() {
                ensure!(
                    existing_block == block_id,
                    "slot {slot} already belongs to {existing_block:?}"
                );
            }
        }

        self.block_slots.insert(block_id, slots.to_vec());
        for &slot in slots {
            self.slot_to_block.insert(slot, block_id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    fn slots(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    #[test]
    fn exact_lookup_returns_matching_slots() {
        let mut cache = MetalPrefixCache::new(4);
        let toks = tokens(1, 8);
        let sls = slots(100, 8);

        assert_eq!(cache.insert(&toks, &sls).unwrap(), 8);
        let hit = cache.lookup(&toks).unwrap();

        assert_eq!(hit.matched_len, 8);
        assert_eq!(hit.slot_indices, sls);
    }

    #[test]
    fn partial_block_is_not_cached() {
        let mut cache = MetalPrefixCache::new(4);
        let toks = tokens(1, 6);
        let sls = slots(200, 6);

        assert_eq!(cache.insert(&toks, &sls).unwrap(), 4);

        let hit = cache.lookup(&toks).unwrap();
        assert_eq!(hit.matched_len, 4);
        assert_eq!(hit.slot_indices, slots(200, 4));

        let short_hit = cache.lookup(&toks[..3]).unwrap();
        assert_eq!(short_hit.matched_len, 0);
        assert!(short_hit.slot_indices.is_empty());
    }

    #[test]
    fn repeated_insert_is_idempotent() {
        let mut cache = MetalPrefixCache::new(4);
        let toks = tokens(10, 4);
        let sls = slots(300, 4);

        assert_eq!(cache.insert(&toks, &sls).unwrap(), 4);
        assert_eq!(cache.insert(&toks, &sls).unwrap(), 4);
        assert_eq!(cache.cached_block_count(), 1);
        assert_eq!(cache.cached_slot_count(), 4);

        let hit = cache.lookup(&toks).unwrap();
        assert_eq!(hit.slot_indices, sls);
    }

    #[test]
    fn release_allows_evicting_a_shared_block() {
        let mut cache = MetalPrefixCache::new(4);
        let toks = tokens(20, 4);
        let sls = slots(400, 4);

        cache.insert(&toks, &sls).unwrap();
        let hit = cache.lookup(&toks).unwrap();

        assert!(cache.evict(1).is_empty());
        cache.release(&hit.slot_indices).unwrap();
        assert_eq!(cache.evict(1), sls);
    }
}

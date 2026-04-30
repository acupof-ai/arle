#![allow(unreachable_pub)]
#![allow(warnings)]
/*!
 * Thread-safe RadixCache coordination for multi-threaded inference
 *
 * Solves memory management complexity by implementing concurrent access patterns,
 * atomic reference counting, and coordinated eviction between scheduler and GPU workers
 */

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use tokio::sync::{Mutex, RwLock};

use crate::kv_tier::{BlockLocation, IndexEntryState, StoreState};
use crate::prefix_cache::BlockId;
use crate::types::{BlockFingerprint, SessionId};

/// Thread-safe wrapper around RadixCache with coordinated access
pub struct ThreadSafeRadixCache {
    /// Core cache state protected by RwLock for concurrent reads
    cache: Arc<RwLock<RadixCacheCore>>,
    /// Eviction coordinator for managing memory pressure
    eviction_coordinator: Arc<EvictionCoordinator>,
    /// Global access counter for LRU tracking
    global_clock: Arc<AtomicU64>,
    /// Configuration parameters
    config: RadixCacheConfig,
}

/// Core cache implementation (non-thread-safe, wrapped by ThreadSafeRadixCache)
struct RadixCacheCore {
    /// Root node index
    root: usize,
    /// All nodes in the tree
    nodes: Vec<ThreadSafeNode>,
    /// Block size for chunking
    block_size: usize,
    /// Free node indices for reuse
    free_nodes: Vec<usize>,
}

/// Thread-safe node with atomic reference counting
struct ThreadSafeNode {
    /// Token sequence stored on this edge
    tokens: Vec<u32>,
    /// GPU block ID cached for this node
    block_id: Option<BlockId>,
    /// Atomic reference count for concurrent access
    ref_count: Arc<AtomicU32>,
    /// Last access time for LRU
    last_access: Arc<AtomicU64>,
    /// Hit count statistics
    hit_count: Arc<AtomicU32>,
    /// Tier location information
    tier_location: Arc<Mutex<Option<BlockLocation>>>,
    /// Session affinity
    session_id: Arc<Mutex<Option<SessionId>>>,
    /// Content fingerprint
    fingerprint: Arc<Mutex<Option<BlockFingerprint>>>,
    /// Byte length
    byte_len: Arc<AtomicU32>,
    /// Soft pin deadline
    soft_pin_until: Arc<AtomicU64>, // 0 means no pin
    /// Host spill pin deadline
    host_spill_pin_until: Arc<AtomicU64>, // 0 means no pin
    /// Entry state
    entry_state: Arc<Mutex<IndexEntryState>>,
    /// Store state
    store_state: Arc<Mutex<StoreState>>,
    /// Children nodes
    children: Arc<RwLock<HashMap<u32, usize>>>,
}

/// Configuration for thread-safe RadixCache
#[derive(Debug, Clone)]
pub struct RadixCacheConfig {
    /// Block size for token chunking
    pub block_size: usize,
    /// Maximum number of cached blocks
    pub max_blocks: usize,
    /// Enable automatic eviction when near capacity
    pub auto_eviction: bool,
    /// Eviction threshold (fraction of max_blocks)
    pub eviction_threshold: f64,
    /// Number of blocks to evict per pass
    pub eviction_batch_size: usize,
}

impl Default for RadixCacheConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            max_blocks: 10000,
            auto_eviction: true,
            eviction_threshold: 0.9,
            eviction_batch_size: 100,
        }
    }
}

/// Coordinates eviction across multiple threads
pub struct EvictionCoordinator {
    /// Current number of allocated blocks
    allocated_blocks: Arc<AtomicU32>,
    /// Maximum allowed blocks
    max_blocks: usize,
    /// Eviction in progress flag
    eviction_in_progress: Arc<AtomicU32>, // 0 = not in progress, 1 = in progress
    /// LRU eviction policy state
    lru_state: Arc<Mutex<LruState>>,
}

#[derive(Debug)]
struct LruState {
    /// Candidate blocks for eviction (block_id -> last_access_time)
    candidates: Vec<EvictionCandidate>,
    /// Last eviction timestamp
    last_eviction: u64,
}

#[derive(Debug, Clone)]
struct EvictionCandidate {
    block_id: BlockId,
    node_index: usize,
    last_access: u64,
    ref_count: u32,
    priority: EvictionPriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EvictionPriority {
    High = 0,      // Unreferenced, old access
    Medium = 1,    // Unreferenced, recent access
    Low = 2,       // Referenced but evictable
    Protected = 3, // Never evict
}

impl ThreadSafeRadixCache {
    /// Create a new thread-safe RadixCache
    pub fn new(config: RadixCacheConfig) -> Self {
        let cache_core = RadixCacheCore {
            root: 0,
            nodes: vec![ThreadSafeNode::new_root()],
            block_size: config.block_size,
            free_nodes: Vec::new(),
        };

        let eviction_coordinator = Arc::new(EvictionCoordinator {
            allocated_blocks: Arc::new(AtomicU32::new(0)),
            max_blocks: config.max_blocks,
            eviction_in_progress: Arc::new(AtomicU32::new(0)),
            lru_state: Arc::new(Mutex::new(LruState {
                candidates: Vec::new(),
                last_eviction: 0,
            })),
        });

        Self {
            cache: Arc::new(RwLock::new(cache_core)),
            eviction_coordinator,
            global_clock: Arc::new(AtomicU64::new(1)),
            config,
        }
    }

    /// Look up a token sequence and return matching prefix
    pub async fn lookup(&self, tokens: &[u32]) -> Result<CacheLookupResult> {
        let now = self.global_clock.fetch_add(1, Ordering::Relaxed);

        let cache = self.cache.read().await;
        let mut current_node = 0; // Start from root
        let mut matched_tokens = 0;
        let mut matched_blocks = Vec::new();

        let mut remaining_tokens = tokens;

        while !remaining_tokens.is_empty() && current_node < cache.nodes.len() {
            let node = &cache.nodes[current_node];

            // Try to match against this node's tokens
            let match_len = self.find_common_prefix(&node.tokens, remaining_tokens);

            if match_len == 0 {
                // No match, check children
                if let Some(&child_idx) = node.children.read().await.get(&remaining_tokens[0]) {
                    current_node = child_idx;
                    continue;
                } else {
                    // No matching child
                    break;
                }
            }

            // Partial or full match with this node
            matched_tokens += match_len;

            // Update access time atomically
            node.last_access.store(now, Ordering::Relaxed);

            // If we have a block and matched completely
            if match_len == node.tokens.len() {
                if let Some(block_id) = node.block_id {
                    // Increment hit count
                    node.hit_count.fetch_add(1, Ordering::Relaxed);

                    matched_blocks.push(CacheBlock {
                        block_id,
                        token_offset: matched_tokens - match_len,
                        token_count: match_len,
                    });
                }

                // Continue with remaining tokens
                remaining_tokens = &remaining_tokens[match_len..];

                // If no more tokens, we're done
                if remaining_tokens.is_empty() {
                    break;
                }

                // Look for a child that can handle the remaining tokens
                if let Some(&child_idx) = node.children.read().await.get(&remaining_tokens[0]) {
                    current_node = child_idx;
                } else {
                    // No child for remaining tokens
                    break;
                }
            } else {
                // Partial match within a node - need to split
                break;
            }
        }

        Ok(CacheLookupResult {
            matched_tokens,
            matched_blocks,
            cache_hit_rate: if tokens.is_empty() {
                1.0
            } else {
                matched_tokens as f64 / tokens.len() as f64
            },
        })
    }

    /// Insert a token sequence with associated blocks
    pub async fn insert(&self, tokens: &[u32], blocks: Vec<CacheBlock>) -> Result<()> {
        if tokens.is_empty() || blocks.is_empty() {
            return Ok(());
        }

        // Check if we need eviction before inserting
        if self.config.auto_eviction {
            self.check_and_evict().await?;
        }

        let now = self.global_clock.fetch_add(1, Ordering::Relaxed);
        let mut cache = self.cache.write().await;

        // Find insertion point
        let insertion_result = self.find_insertion_point(&mut cache, tokens, now).await?;

        // Insert blocks at the appropriate nodes
        for block in blocks {
            let node_idx = insertion_result.target_node;
            if node_idx < cache.nodes.len() {
                let node = &mut cache.nodes[node_idx];
                node.block_id = Some(block.block_id);
                node.last_access.store(now, Ordering::Relaxed);

                // Update allocation count
                self.eviction_coordinator
                    .allocated_blocks
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    /// Increment reference count for a block (pin it)
    pub async fn pin_blocks(&self, block_ids: &[BlockId]) -> Result<()> {
        let cache = self.cache.read().await;

        for &block_id in block_ids {
            if let Some(node_idx) = self.find_node_by_block_id(&cache, block_id).await {
                if node_idx < cache.nodes.len() {
                    cache.nodes[node_idx]
                        .ref_count
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        Ok(())
    }

    /// Decrement reference count for a block (unpin it)
    pub async fn unpin_blocks(&self, block_ids: &[BlockId]) -> Result<()> {
        let cache = self.cache.read().await;

        for &block_id in block_ids {
            if let Some(node_idx) = self.find_node_by_block_id(&cache, block_id).await {
                if node_idx < cache.nodes.len() {
                    let prev_count = cache.nodes[node_idx]
                        .ref_count
                        .fetch_sub(1, Ordering::Relaxed);
                    if prev_count == 0 {
                        log::warn!(
                            "Attempted to unpin block {:?} with ref_count already 0",
                            block_id
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Force eviction of specific number of blocks
    pub async fn evict(&self, num_blocks: usize) -> Result<Vec<BlockId>> {
        if num_blocks == 0 {
            return Ok(Vec::new());
        }

        // Use compare-and-swap to ensure only one eviction runs at a time
        if self
            .eviction_coordinator
            .eviction_in_progress
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // Eviction already in progress
            return Ok(Vec::new());
        }

        let result = self.perform_eviction(num_blocks).await;

        // Clear eviction in progress flag
        self.eviction_coordinator
            .eviction_in_progress
            .store(0, Ordering::Release);

        result
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        let allocated_blocks = self
            .eviction_coordinator
            .allocated_blocks
            .load(Ordering::Relaxed);

        let mut total_nodes = 0;
        let mut nodes_with_blocks = 0;
        let mut total_ref_count = 0;
        let mut total_hit_count = 0;

        for node in &cache.nodes {
            total_nodes += 1;
            if node.block_id.is_some() {
                nodes_with_blocks += 1;
            }
            total_ref_count += node.ref_count.load(Ordering::Relaxed);
            total_hit_count += node.hit_count.load(Ordering::Relaxed);
        }

        CacheStats {
            total_nodes,
            nodes_with_blocks,
            allocated_blocks,
            max_blocks: self.config.max_blocks as u32,
            utilization: allocated_blocks as f64 / self.config.max_blocks as f64,
            total_ref_count,
            total_hit_count,
            eviction_in_progress: self
                .eviction_coordinator
                .eviction_in_progress
                .load(Ordering::Relaxed)
                != 0,
        }
    }

    // Helper methods

    fn find_common_prefix(&self, a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    async fn find_node_by_block_id(
        &self,
        cache: &RadixCacheCore,
        block_id: BlockId,
    ) -> Option<usize> {
        for (idx, node) in cache.nodes.iter().enumerate() {
            if node.block_id == Some(block_id) {
                return Some(idx);
            }
        }
        None
    }

    async fn find_insertion_point(
        &self,
        cache: &mut RadixCacheCore,
        tokens: &[u32],
        now: u64,
    ) -> Result<InsertionResult> {
        // Simplified insertion logic - in practice this would be more complex
        let target_node = if cache.free_nodes.is_empty() {
            let new_idx = cache.nodes.len();
            cache
                .nodes
                .push(ThreadSafeNode::new(tokens.to_vec(), None, now));
            new_idx
        } else {
            let idx = cache.free_nodes.pop().unwrap();
            cache.nodes[idx] = ThreadSafeNode::new(tokens.to_vec(), None, now);
            idx
        };

        Ok(InsertionResult { target_node })
    }

    async fn check_and_evict(&self) -> Result<()> {
        let allocated = self
            .eviction_coordinator
            .allocated_blocks
            .load(Ordering::Relaxed);
        let threshold = (self.config.max_blocks as f64 * self.config.eviction_threshold) as u32;

        if allocated >= threshold {
            self.evict(self.config.eviction_batch_size).await?;
        }

        Ok(())
    }

    async fn perform_eviction(&self, num_blocks: usize) -> Result<Vec<BlockId>> {
        let mut cache = self.cache.write().await;
        let mut candidates = Vec::new();

        // Collect eviction candidates
        for (idx, node) in cache.nodes.iter().enumerate() {
            if let Some(block_id) = node.block_id {
                let ref_count = node.ref_count.load(Ordering::Relaxed);
                let last_access = node.last_access.load(Ordering::Relaxed);

                let priority = if ref_count > 0 {
                    EvictionPriority::Low
                } else if last_access < 1000 {
                    // Very old access
                    EvictionPriority::High
                } else {
                    EvictionPriority::Medium
                };

                candidates.push(EvictionCandidate {
                    block_id,
                    node_index: idx,
                    last_access,
                    ref_count,
                    priority,
                });
            }
        }

        // Sort by priority (high priority = evict first)
        candidates.sort_by_key(|c| (c.priority, c.last_access));

        // Evict up to num_blocks
        let mut evicted = Vec::new();
        for candidate in candidates.into_iter().take(num_blocks) {
            if candidate.ref_count == 0 {
                // Safe to evict
                if candidate.node_index < cache.nodes.len() {
                    cache.nodes[candidate.node_index].block_id = None;
                    evicted.push(candidate.block_id);

                    // Update allocation count
                    self.eviction_coordinator
                        .allocated_blocks
                        .fetch_sub(1, Ordering::Relaxed);
                }
            }
        }

        log::debug!("Evicted {} blocks", evicted.len());
        Ok(evicted)
    }
}

impl ThreadSafeNode {
    fn new_root() -> Self {
        Self {
            tokens: Vec::new(),
            block_id: None,
            ref_count: Arc::new(AtomicU32::new(0)),
            last_access: Arc::new(AtomicU64::new(0)),
            hit_count: Arc::new(AtomicU32::new(0)),
            tier_location: Arc::new(Mutex::new(None)),
            session_id: Arc::new(Mutex::new(None)),
            fingerprint: Arc::new(Mutex::new(None)),
            byte_len: Arc::new(AtomicU32::new(0)),
            soft_pin_until: Arc::new(AtomicU64::new(0)),
            host_spill_pin_until: Arc::new(AtomicU64::new(0)),
            entry_state: Arc::new(Mutex::new(IndexEntryState::Ready)),
            store_state: Arc::new(Mutex::new(StoreState::Idle)),
            children: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn new(tokens: Vec<u32>, block_id: Option<BlockId>, now: u64) -> Self {
        Self {
            tokens,
            block_id,
            ref_count: Arc::new(AtomicU32::new(0)),
            last_access: Arc::new(AtomicU64::new(now)),
            hit_count: Arc::new(AtomicU32::new(0)),
            tier_location: Arc::new(Mutex::new(None)),
            session_id: Arc::new(Mutex::new(None)),
            fingerprint: Arc::new(Mutex::new(None)),
            byte_len: Arc::new(AtomicU32::new(0)),
            soft_pin_until: Arc::new(AtomicU64::new(0)),
            host_spill_pin_until: Arc::new(AtomicU64::new(0)),
            entry_state: Arc::new(Mutex::new(IndexEntryState::Ready)),
            store_state: Arc::new(Mutex::new(StoreState::Idle)),
            children: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

// Result types

#[derive(Debug, Clone)]
pub struct CacheLookupResult {
    pub matched_tokens: usize,
    pub matched_blocks: Vec<CacheBlock>,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Clone)]
pub struct CacheBlock {
    pub block_id: BlockId,
    pub token_offset: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_nodes: usize,
    pub nodes_with_blocks: usize,
    pub allocated_blocks: u32,
    pub max_blocks: u32,
    pub utilization: f64,
    pub total_ref_count: u32,
    pub total_hit_count: u32,
    pub eviction_in_progress: bool,
}

struct InsertionResult {
    target_node: usize,
}

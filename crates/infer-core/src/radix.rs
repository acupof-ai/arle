//! Host-side radix prefix cache keyed by page-sized token blocks.
//!
//! The block size is aligned to `KvPool::page_size()`: one cached block maps to
//! one host page id. Partial tail blocks are deliberately not published, which
//! keeps prefix reuse page-aligned.

use std::collections::BTreeMap;

/// Host page id used as the prefix-cache block id.
pub type BlockId = u32;

/// Longest cached prefix result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Number of prompt tokens covered by cached full blocks.
    pub matched_len: usize,
    /// Host page ids backing the matched prefix in prompt order.
    pub block_ids: Vec<BlockId>,
}

impl PrefixMatch {
    /// Return an empty prefix match.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            matched_len: 0,
            block_ids: Vec::new(),
        }
    }

    /// Return whether this match contains at least one cached block.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.block_ids.is_empty()
    }
}

/// Fixed-block radix cache for prompt KV reuse.
#[derive(Debug, Clone)]
pub struct RadixCache {
    block_size: usize,
    nodes: Vec<Node>,
    page_to_node: BTreeMap<BlockId, usize>,
    clock: u64,
}

#[derive(Debug, Clone)]
struct Node {
    block: Vec<u32>,
    page_id: Option<BlockId>,
    ref_count: usize,
    last_access: u64,
    parent: Option<usize>,
    children: BTreeMap<Vec<u32>, usize>,
    evicted: bool,
}

impl Node {
    fn root() -> Self {
        Self {
            block: Vec::new(),
            page_id: None,
            ref_count: 0,
            last_access: 0,
            parent: None,
            children: BTreeMap::new(),
            evicted: false,
        }
    }

    fn child(block: Vec<u32>, page_id: BlockId, parent: usize, last_access: u64) -> Self {
        Self {
            block,
            page_id: Some(page_id),
            ref_count: 0,
            last_access,
            parent: Some(parent),
            children: BTreeMap::new(),
            evicted: false,
        }
    }

    fn is_evictable_leaf(&self) -> bool {
        !self.evicted && self.page_id.is_some() && self.ref_count == 0 && self.children.is_empty()
    }
}

impl RadixCache {
    /// Create a cache whose block size is the KV pool page size in tokens.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
            nodes: vec![Node::root()],
            page_to_node: BTreeMap::new(),
            clock: 0,
        }
    }

    /// Return the number of tokens in one cached block.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Return the number of cached full blocks.
    #[must_use]
    pub fn cached_page_count(&self) -> usize {
        self.page_to_node.len()
    }

    /// Find the longest cached full-block prefix and bump recency.
    pub fn longest_prefix_match(&mut self, tokens: &[u32]) -> PrefixMatch {
        let matched = self.match_inner(tokens);
        if !matched.block_ids.is_empty() {
            for page_id in &matched.block_ids {
                if let Some(&node_idx) = self.page_to_node.get(page_id) {
                    let last_access = self.tick();
                    self.nodes[node_idx].last_access = last_access;
                }
            }
        }
        matched
    }

    /// Find the longest cached full-block prefix without changing recency.
    #[must_use]
    pub fn peek_longest_prefix_match(&self, tokens: &[u32]) -> PrefixMatch {
        self.match_inner(tokens)
    }

    fn match_inner(&self, tokens: &[u32]) -> PrefixMatch {
        let mut node_idx = 0usize;
        let mut block_ids = Vec::new();
        let mut matched_len = 0usize;

        for block in tokens.chunks_exact(self.block_size) {
            let Some(&child_idx) = self.nodes[node_idx].children.get(block) else {
                break;
            };
            let child = &self.nodes[child_idx];
            let Some(page_id) = child.page_id else {
                break;
            };
            block_ids.push(page_id);
            matched_len += self.block_size;
            node_idx = child_idx;
        }

        PrefixMatch {
            matched_len,
            block_ids,
        }
    }

    /// Publish full token blocks with their host page ids.
    ///
    /// Returns page ids that became newly owned by the cache. Existing matching
    /// blocks are left in place and are not returned, so callers can retain only
    /// newly published pages.
    pub fn insert(&mut self, tokens: &[u32], page_ids: &[BlockId]) -> Vec<BlockId> {
        let full_blocks = tokens.len() / self.block_size;
        let publish_blocks = full_blocks.min(page_ids.len());
        let mut node_idx = 0usize;
        let mut newly_cached = Vec::new();

        for (block_idx, block) in tokens
            .chunks_exact(self.block_size)
            .take(publish_blocks)
            .enumerate()
        {
            let block = block.to_vec();
            let page_id = page_ids[block_idx];
            let child_idx = if let Some(&child_idx) = self.nodes[node_idx].children.get(&block) {
                child_idx
            } else {
                let child_idx = self.nodes.len();
                let last_access = self.tick();
                self.nodes
                    .push(Node::child(block.clone(), page_id, node_idx, last_access));
                self.nodes[node_idx]
                    .children
                    .insert(block.clone(), child_idx);
                if self.page_to_node.insert(page_id, child_idx).is_none() {
                    newly_cached.push(page_id);
                }
                child_idx
            };

            if self.nodes[child_idx].page_id.is_none() {
                self.nodes[child_idx].page_id = Some(page_id);
                if self.page_to_node.insert(page_id, child_idx).is_none() {
                    newly_cached.push(page_id);
                }
            }
            let last_access = self.tick();
            self.nodes[child_idx].last_access = last_access;
            node_idx = child_idx;
        }

        newly_cached
    }

    /// Mark cached blocks as attached to an active slot.
    pub fn retain_blocks(&mut self, pages: &[BlockId]) {
        for page_id in pages {
            if let Some(&node_idx) = self.page_to_node.get(page_id) {
                self.nodes[node_idx].ref_count = self.nodes[node_idx].ref_count.saturating_add(1);
                let last_access = self.tick();
                self.nodes[node_idx].last_access = last_access;
            }
        }
    }

    /// Release active-slot refs for cached blocks.
    pub fn release_blocks(&mut self, pages: &[BlockId]) {
        for page_id in pages {
            if let Some(&node_idx) = self.page_to_node.get(page_id) {
                self.nodes[node_idx].ref_count = self.nodes[node_idx].ref_count.saturating_sub(1);
                let last_access = self.tick();
                self.nodes[node_idx].last_access = last_access;
            }
        }
    }

    /// Evict up to `n_pages_needed` least-recently-used inactive leaf blocks.
    ///
    /// Blocks with a nonzero active ref are never returned.
    pub fn evict_lru(&mut self, n_pages_needed: usize) -> Vec<BlockId> {
        let mut evicted = Vec::new();
        while evicted.len() < n_pages_needed {
            let Some(node_idx) = self.least_recent_evictable_leaf() else {
                break;
            };
            let Some(page_id) = self.nodes[node_idx].page_id.take() else {
                self.nodes[node_idx].evicted = true;
                continue;
            };
            if let Some(parent_idx) = self.nodes[node_idx].parent {
                let block = self.nodes[node_idx].block.clone();
                self.nodes[parent_idx].children.remove(&block);
            }
            self.nodes[node_idx].evicted = true;
            self.page_to_node.remove(&page_id);
            evicted.push(page_id);
        }
        evicted
    }

    fn least_recent_evictable_leaf(&self) -> Option<usize> {
        self.nodes
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, node)| node.is_evictable_leaf())
            .min_by_key(|(_, node)| node.last_access)
            .map(|(idx, _)| idx)
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }
}

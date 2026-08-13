//! Host-side radix prefix cache keyed by page-sized token blocks.
//!
//! The block size is aligned to `KvPool::page_size()`: one cached block maps to
//! one host page id. Partial tail blocks are deliberately not published, which
//! keeps prefix reuse page-aligned.

use std::collections::BTreeMap;

use infer_seam::PrefixBlock;

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
    #[must_use]
    pub fn empty() -> Self {
        Self {
            matched_len: 0,
            block_ids: Vec::new(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.block_ids.is_empty()
    }
}

/// Longest cached prefix including demoted (host-tier) blocks.
///
/// Used only by the tier-enabled engine path; the resident-only
/// [`PrefixMatch`] surface is unchanged for backends without a tier store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TieredPrefixMatch {
    /// Matched blocks in prompt order (resident and demoted interleaved).
    pub blocks: Vec<PrefixBlock>,
}

/// Fixed-block radix cache for prompt KV reuse.
#[derive(Debug, Clone)]
pub struct RadixCache {
    block_size: usize,
    nodes: Vec<Node>,
    /// Reclaimed `nodes` slots (severed, awaiting reuse). `insert` pops here
    /// before appending, bounding `nodes` growth to the live-node high-water.
    free: Vec<usize>,
    page_to_node: BTreeMap<BlockId, usize>,
    tier_to_node: BTreeMap<u64, usize>,
    /// Tier keys invalidated by sever/revive since the last drain. The engine
    /// drains these after every cache mutation batch and forwards them to
    /// `BackendExecutor::drop_kv_tier_entries`, so no path can leak a key.
    dropped_tier_keys: Vec<u64>,
    clock: u64,
}

#[derive(Debug, Clone)]
struct Node {
    block: Vec<u32>,
    page_id: Option<BlockId>,
    /// Backend tier-store key while this block's contents live in the host
    /// tier instead of a device page. Mutually exclusive with `page_id`.
    tier_key: Option<u64>,
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
            tier_key: None,
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
            tier_key: None,
            ref_count: 0,
            last_access,
            parent: Some(parent),
            children: BTreeMap::new(),
            evicted: false,
        }
    }

    fn is_demoted(&self) -> bool {
        !self.evicted && self.page_id.is_none() && self.tier_key.is_some()
    }
}

impl RadixCache {
    /// Create a cache whose block size is the KV pool page size in tokens.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
            nodes: vec![Node::root()],
            free: Vec::new(),
            page_to_node: BTreeMap::new(),
            tier_to_node: BTreeMap::new(),
            dropped_tier_keys: Vec::new(),
            clock: 0,
        }
    }

    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    #[must_use]
    pub fn cached_page_count(&self) -> usize {
        self.page_to_node.len()
    }

    #[must_use]
    pub fn demoted_block_count(&self) -> usize {
        self.tier_to_node.len()
    }

    /// Drain the tier keys invalidated since the last call (severed or revived
    /// demoted nodes). The caller forwards them to the backend tier store.
    pub fn take_dropped_tier_keys(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.dropped_tier_keys)
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
        let block_ids: Vec<u32> = tokens
            .chunks_exact(self.block_size)
            .scan(0usize, |node_idx, block| {
                let &child_idx = self.nodes[*node_idx].children.get(block)?;
                let page_id = self.nodes[child_idx].page_id?;
                *node_idx = child_idx;
                Some(page_id)
            })
            .collect();
        PrefixMatch {
            matched_len: block_ids.len() * self.block_size,
            block_ids,
        }
    }

    /// Find the longest cached prefix including demoted blocks, bumping recency.
    ///
    /// Unlike [`RadixCache::longest_prefix_match`], the walk continues through
    /// demoted nodes (contents in the backend host tier) so the engine can
    /// promote them into fresh pages instead of re-prefilling. Only used by
    /// the tier-enabled engine path.
    pub fn tiered_longest_prefix_match(&mut self, tokens: &[u32]) -> TieredPrefixMatch {
        let mut node_idx = 0usize;
        let mut blocks = Vec::new();
        for block in tokens.chunks_exact(self.block_size) {
            let Some(&child_idx) = self.nodes[node_idx].children.get(block) else {
                break;
            };
            let child = &self.nodes[child_idx];
            let entry = if let Some(page_id) = child.page_id {
                PrefixBlock::ResidentPage(page_id)
            } else if let Some(tier_key) = child.tier_key {
                PrefixBlock::DemotedKey(tier_key)
            } else {
                break;
            };
            blocks.push(entry);
            let last_access = self.tick();
            self.nodes[child_idx].last_access = last_access;
            node_idx = child_idx;
        }
        TieredPrefixMatch { blocks }
    }

    /// Demote a cached resident block: drop its device page id and remember the
    /// backend tier key its contents now live under. The node stays linked and
    /// matchable. Returns `false` (no state change) if `page` is not an
    /// idle cached block.
    pub fn demote_block(&mut self, page: BlockId, tier_key: u64) -> bool {
        let Some(&node_idx) = self.page_to_node.get(&page) else {
            return false;
        };
        // Only idle leaves (no resident descendants) demote, preserving the
        // invariant that demoted subtrees are entirely non-resident.
        if !self.is_evictable_leaf(node_idx) {
            return false;
        }
        self.page_to_node.remove(&page);
        let node = &mut self.nodes[node_idx];
        node.page_id = None;
        node.tier_key = Some(tier_key);
        self.tier_to_node.insert(tier_key, node_idx);
        true
    }

    /// Restore a demoted block to device residency under a freshly promoted
    /// page. The tier key is consumed (the caller drops the store entry via
    /// the dropped-keys drain). Returns `false` if the key is unknown.
    pub fn promote_block(&mut self, tier_key: u64, page: BlockId) -> bool {
        let Some(&node_idx) = self.tier_to_node.get(&tier_key) else {
            return false;
        };
        self.tier_to_node.remove(&tier_key);
        self.dropped_tier_keys.push(tier_key);
        let node = &mut self.nodes[node_idx];
        node.tier_key = None;
        node.page_id = Some(page);
        self.page_to_node.insert(page, node_idx);
        let last_access = self.tick();
        self.nodes[node_idx].last_access = last_access;
        true
    }

    /// Return up to `limit` currently evictable resident pages in LRU order.
    ///
    /// This is a snapshot of the current frontier. Demoting a child can expose
    /// its parent as a new frontier node, so callers that need more pages should
    /// re-query after each accepted batch.
    #[must_use]
    pub fn lru_evictable_pages(&self, limit: usize) -> Vec<BlockId> {
        if limit == 0 {
            return Vec::new();
        }
        let mut pages =
            self.evictable_pages_where(limit, |cache, idx| cache.is_strict_evictable_leaf(idx));
        if pages.len() < limit {
            let remaining = limit - pages.len();
            pages.extend(self.evictable_pages_where(remaining, |cache, idx| {
                cache.is_evictable_leaf(idx) && !cache.is_strict_evictable_leaf(idx)
            }));
        }
        pages
    }

    /// Peek the tier key of the least-recently-used demoted block whose
    /// subtree holds no resident pages (safe to sever for tier-store room).
    #[must_use]
    pub fn lru_demoted_key(&self) -> Option<u64> {
        self.tier_to_node
            .iter()
            .map(|(&key, &idx)| (key, idx))
            .filter(|&(_, idx)| !self.subtree_has_resident(idx))
            .min_by_key(|&(_, idx)| self.nodes[idx].last_access)
            .map(|(key, _)| key)
    }

    /// Sever a demoted block (and its demoted-only subtree) from the cache,
    /// pushing all invalidated tier keys to the dropped-keys drain. Returns
    /// `false` if the key is unknown or the subtree still holds resident pages.
    pub fn drop_demoted(&mut self, tier_key: u64) -> bool {
        let Some(&node_idx) = self.tier_to_node.get(&tier_key) else {
            return false;
        };
        if self.subtree_has_resident(node_idx) {
            return false;
        }
        self.sever_subtree(node_idx);
        true
    }

    /// Sever an idle resident block selected by page id, exactly like one
    /// `evict_lru` step. Demoted descendants are dropped via the tier-key
    /// drain. Returns `false` if `page` is not an idle cached block.
    pub fn evict_page(&mut self, page: BlockId) -> bool {
        let Some(&node_idx) = self.page_to_node.get(&page) else {
            return false;
        };
        if !self.is_evictable_leaf(node_idx) {
            return false;
        }
        self.page_to_node.remove(&page);
        self.sever_subtree(node_idx);
        true
    }

    fn subtree_has_resident(&self, idx: usize) -> bool {
        if self.nodes[idx].page_id.is_some() {
            return true;
        }
        self.nodes[idx]
            .children
            .values()
            .any(|&child| self.subtree_has_resident(child))
    }

    /// Detach `idx` from its parent and mark its subtree evicted, draining
    /// every tier key found into `dropped_tier_keys`. Callers must have
    /// removed any `page_to_node` entries for resident nodes in the subtree
    /// (the engine paths only sever subtrees whose only contents are the
    /// severed node's own page plus demoted descendants).
    fn sever_subtree(&mut self, idx: usize) {
        if let Some(parent_idx) = self.nodes[idx].parent {
            let block = self.nodes[idx].block.clone();
            self.nodes[parent_idx].children.remove(&block);
        }
        let mut stack = vec![idx];
        while let Some(current) = stack.pop() {
            let node = &mut self.nodes[current];
            node.evicted = true;
            node.page_id = None;
            node.block = Vec::new();
            let tier_key = node.tier_key.take();
            let children = std::mem::take(&mut node.children);
            if let Some(key) = tier_key {
                self.tier_to_node.remove(&key);
                self.dropped_tier_keys.push(key);
            }
            stack.extend(children.into_values());
            // evicted=true + page_id/tier_key=None keeps every scan predicate
            // skipping this slot until `insert` fully re-initializes it.
            self.free.push(current);
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
                let last_access = self.tick();
                let node = Node::child(block.clone(), page_id, node_idx, last_access);
                let child_idx = if let Some(free_idx) = self.free.pop() {
                    self.nodes[free_idx] = node;
                    free_idx
                } else {
                    self.nodes.push(node);
                    self.nodes.len() - 1
                };
                self.nodes[node_idx]
                    .children
                    .insert(block.clone(), child_idx);
                if self.page_to_node.insert(page_id, child_idx).is_none() {
                    newly_cached.push(page_id);
                }
                child_idx
            };

            if self.nodes[child_idx].page_id.is_none() {
                // Reviving a page-less node: if it was demoted, the re-prefilled
                // page supersedes the tier copy — drop the stale tier entry.
                if let Some(key) = self.nodes[child_idx].tier_key.take() {
                    self.tier_to_node.remove(&key);
                    self.dropped_tier_keys.push(key);
                }
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

    pub fn retain_blocks(&mut self, pages: &[BlockId]) {
        for page_id in pages {
            if let Some(&node_idx) = self.page_to_node.get(page_id) {
                self.nodes[node_idx].ref_count = self.nodes[node_idx].ref_count.saturating_add(1);
                let last_access = self.tick();
                self.nodes[node_idx].last_access = last_access;
            }
        }
    }

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
    /// Blocks with a nonzero active ref are never returned. Demoted-only
    /// subtrees under an evicted block are severed with it (their tier keys
    /// land in the dropped-keys drain).
    pub fn evict_lru(&mut self, n_pages_needed: usize) -> Vec<BlockId> {
        let mut evicted = Vec::new();
        while evicted.len() < n_pages_needed {
            let Some(node_idx) = self.least_recent_evictable_leaf() else {
                break;
            };
            let Some(page_id) = self.nodes[node_idx].page_id else {
                break;
            };
            self.page_to_node.remove(&page_id);
            self.sever_subtree(node_idx);
            evicted.push(page_id);
        }
        evicted
    }

    /// A resident block is on the relaxed evictable frontier when it is idle
    /// and holds no resident descendants. Demoted children do not pin their
    /// parent (otherwise a demoted leaf would freeze its whole ancestor chain
    /// on the device); demoted nodes never have resident descendants —
    /// inserting through a demoted node revives it first (see `insert`) — so
    /// one level suffices.
    fn is_evictable_leaf(&self, idx: usize) -> bool {
        let node = &self.nodes[idx];
        !node.evicted
            && node.page_id.is_some()
            && node.ref_count == 0
            && node
                .children
                .values()
                .all(|&child| self.nodes[child].is_demoted())
    }

    fn is_strict_evictable_leaf(&self, idx: usize) -> bool {
        self.is_evictable_leaf(idx) && self.nodes[idx].children.is_empty()
    }

    fn least_recent_evictable_leaf(&self) -> Option<usize> {
        // Prefer true leaves. A resident parent with only demoted descendants is
        // safe to sever, but it drops more prefix shape than a strict leaf.
        // Use that relaxed frontier only after all strict leaves are gone.
        (1..self.nodes.len())
            .filter(|&idx| self.is_strict_evictable_leaf(idx))
            .min_by_key(|&idx| self.nodes[idx].last_access)
            .or_else(|| {
                (1..self.nodes.len())
                    .filter(|&idx| self.is_evictable_leaf(idx))
                    .min_by_key(|&idx| self.nodes[idx].last_access)
            })
    }

    fn evictable_pages_where(
        &self,
        limit: usize,
        predicate: impl Fn(&Self, usize) -> bool,
    ) -> Vec<BlockId> {
        let mut candidates: Vec<_> = (1..self.nodes.len())
            .filter(|&idx| predicate(self, idx))
            .filter_map(|idx| {
                self.nodes[idx]
                    .page_id
                    .map(|page| (self.nodes[idx].last_access, page))
            })
            .collect();
        candidates.sort_unstable_by_key(|&(last_access, _)| last_access);
        candidates
            .into_iter()
            .take(limit)
            .map(|(_, page)| page)
            .collect()
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }
}

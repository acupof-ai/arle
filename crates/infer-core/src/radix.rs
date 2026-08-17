//! Host-side radix prefix cache keyed by page-sized token blocks.
//!
//! The block size is aligned to `KvPool::page_size()`: one cached block maps to
//! one host page id. Partial tail blocks are deliberately not published, which
//! keeps prefix reuse page-aligned.
//!
//! Under CP sequence-sharding (2D), the tree is REPLICATED across cp ranks:
//! every rank holds the full token tree, but a block's `page_id` is `Some` only
//! on the owning shard — block `B` lives on shard `B % cp_size`, a pure
//! function, so no location table is stored. Non-owning ranks hold replica
//! nodes (`page_id == None`, never in `page_to_node`); a prefix match walks
//! through them and emits [`REPLICA_PAGE`], and the engine attaches only the
//! local subset. Block `B` is resident iff shard `B % cp_size`'s page is live;
//! the engine min-reduces the matched length across ranks, so a missing block
//! on any shard truncates the match for all.

use std::collections::BTreeMap;

use infer_seam::{PrefixBlock, ShardSpec};

/// Host page id used as the prefix-cache block id.
pub type BlockId = u32;

/// Matched-block marker for a block whose KV page lives on another CP shard.
///
/// Never a real pool page id (`total_pages` is far below `u32::MAX`) and
/// distinct from `infer_seam::EVICTED_PAGE`; every pool/backend touch filters
/// it out first (see [`PrefixMatch::local_block_ids`]).
pub const REPLICA_PAGE: BlockId = u32::MAX - 1;

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

    /// Page ids backed by THIS rank's CP shard, in global block order.
    ///
    /// Block positions owned by other shards carry [`REPLICA_PAGE`]; this
    /// drops them, leaving exactly the pages the host pool can retain or
    /// attach (block `B` lives on shard `B % size`). An unsharded cache
    /// (`size <= 1`) returns the full list.
    #[must_use]
    pub fn local_block_ids(&self, shard: ShardSpec) -> Vec<BlockId> {
        if shard.size <= 1 {
            return self.block_ids.clone();
        }
        self.block_ids
            .iter()
            .enumerate()
            .filter(|(block_idx, _)| shard.owns_page(*block_idx))
            .map(|(_, &page)| page)
            .filter(|&page| page != REPLICA_PAGE)
            .collect()
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
    /// CP shard coordinates for the replicated tree. `size <= 1` (the
    /// default) keeps the cache rank-local: every block is local, no
    /// [`REPLICA_PAGE`] markers are emitted, and eviction is unchanged.
    cp_shard: ShardSpec,
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
    /// Resident (page-owning) nodes in this node's subtree, EXCLUDING self.
    /// Replica and demoted descendants do not count. Maintained on every
    /// `page_id` transition so `is_evictable_leaf` is O(1): a resident node
    /// evicts only when no resident descendant survives it — a replica child
    /// never pins its parent (it holds no local page).
    resident_below: usize,
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
            resident_below: 0,
        }
    }

    fn child(block: Vec<u32>, page_id: Option<BlockId>, parent: usize, last_access: u64) -> Self {
        Self {
            block,
            page_id,
            tier_key: None,
            ref_count: 0,
            last_access,
            parent: Some(parent),
            children: BTreeMap::new(),
            evicted: false,
            resident_below: 0,
        }
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
            cp_shard: ShardSpec::default(),
        }
    }

    /// Set the CP sequence-shard coordinates for the replicated tree.
    ///
    /// Called by the engine once the pool's shard assignment is known
    /// (idempotent — the same coordinates on every call). `size <= 1` keeps
    /// the cache rank-local. Must be called before the first publish under
    /// 2D, so a block's owning shard is known while the tree grows.
    pub fn set_cp_shard(&mut self, shard: ShardSpec) {
        self.cp_shard = shard;
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
        let mut node_idx = 0usize;
        let mut block_ids = Vec::new();
        for (block_idx, block) in tokens.chunks_exact(self.block_size).enumerate() {
            let Some(&child_idx) = self.nodes[node_idx].children.get(block) else {
                break;
            };
            let child = &self.nodes[child_idx];
            let local = self.cp_shard.owns_page(block_idx);
            match (local, child.page_id) {
                // Local block with its page: the normal resident entry.
                (true, Some(page)) => block_ids.push(page),
                // Block owned by another shard: its page lives there. Walk on
                // — the tree is replicated, the content is known.
                (false, _) => block_ids.push(REPLICA_PAGE),
                // Local block without a page: demoted or evicted. The
                // resident match ends here (the tiered match may continue).
                (true, None) => break,
            }
            node_idx = child_idx;
        }
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
        self.adjust_resident_ancestors(node_idx, -1);
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
        self.adjust_resident_ancestors(node_idx, 1);
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
        let node = &self.nodes[idx];
        node.page_id.is_some() || node.resident_below > 0
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
        let mut removed_residents = 0usize;
        let mut stack = vec![idx];
        while let Some(current) = stack.pop() {
            let node = &mut self.nodes[current];
            if node.page_id.is_some() {
                removed_residents += 1;
            }
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
        if removed_residents > 0 {
            self.adjust_resident_ancestors(idx, -(removed_residents as i64));
        }
    }

    /// Publish full token blocks into the replicated tree.
    ///
    /// `tokens` is the rank-identical published span (the engine's collective
    /// exchange aligned it across cp ranks); `page_of` maps a GLOBAL block
    /// index within the span to this rank's local page, or `None` for blocks
    /// owned by another shard (replica nodes: content known, no local page).
    ///
    /// Returns the local pages that became newly owned by the cache. Existing
    /// matching blocks are left in place and are not returned, so callers can
    /// retain only newly published pages. Replica nodes never enter
    /// `page_to_node` and never pin eviction.
    pub fn insert_replicated(
        &mut self,
        tokens: &[u32],
        page_of: &dyn Fn(usize) -> Option<BlockId>,
    ) -> Vec<BlockId> {
        let full_blocks = tokens.len() / self.block_size;
        let mut node_idx = 0usize;
        let mut newly_cached = Vec::new();

        for (block_idx, block) in tokens
            .chunks_exact(self.block_size)
            .take(full_blocks)
            .enumerate()
        {
            let block = block.to_vec();
            let page = page_of(block_idx);
            let child_idx = if let Some(&child_idx) = self.nodes[node_idx].children.get(&block) {
                child_idx
            } else {
                let last_access = self.tick();
                let node = Node::child(block.clone(), page, node_idx, last_access);
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
                if let Some(page_id) = page
                    && self.page_to_node.insert(page_id, child_idx).is_none()
                {
                    newly_cached.push(page_id);
                    self.adjust_resident_ancestors(child_idx, 1);
                }
                child_idx
            };

            if let Some(page_id) = page
                && self.nodes[child_idx].page_id.is_none()
            {
                // Reviving a page-less node: if it was demoted, the re-prefilled
                // page supersedes the tier copy — drop the stale tier entry.
                // (A replica node stays page-less: `page` is None for it.)
                if let Some(key) = self.nodes[child_idx].tier_key.take() {
                    self.tier_to_node.remove(&key);
                    self.dropped_tier_keys.push(key);
                }
                self.nodes[child_idx].page_id = Some(page_id);
                if self.page_to_node.insert(page_id, child_idx).is_none() {
                    newly_cached.push(page_id);
                }
                self.adjust_resident_ancestors(child_idx, 1);
            }
            let last_access = self.tick();
            self.nodes[child_idx].last_access = last_access;
            node_idx = child_idx;
        }

        newly_cached
    }

    /// Adjust `resident_below` on every ancestor of `idx` by `delta`.
    fn adjust_resident_ancestors(&mut self, mut idx: usize, delta: i64) {
        while let Some(parent) = self.nodes[idx].parent {
            let node = &mut self.nodes[parent];
            node.resident_below = if delta >= 0 {
                node.resident_below.saturating_add(delta as usize)
            } else {
                node.resident_below
                    .saturating_sub(delta.unsigned_abs() as usize)
            };
            idx = parent;
        }
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
    /// and holds no resident descendants. Demoted and replica children do not
    /// pin their parent (they hold no local page — otherwise a demoted leaf
    /// would freeze its whole ancestor chain, and under CP replication every
    /// other block is a replica); `resident_below` makes the descendant check
    /// O(1) and exact across the replicated tree.
    fn is_evictable_leaf(&self, idx: usize) -> bool {
        let node = &self.nodes[idx];
        !node.evicted && node.page_id.is_some() && node.ref_count == 0 && node.resident_below == 0
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn unsharded_insert_matches_legacy() {
        let bs = 4;
        let toks = tokens(bs * 3);
        let mut cache = RadixCache::new(bs);
        let pages: Vec<BlockId> = vec![10, 11, 12];
        let newly = cache.insert_replicated(&toks, &|j| Some(pages[j]));
        assert_eq!(newly, pages);
        let m = cache.peek_longest_prefix_match(&toks);
        assert_eq!(m.matched_len, toks.len());
        assert_eq!(m.block_ids, pages);
        assert_eq!(m.local_block_ids(ShardSpec::default()), pages);
    }

    #[test]
    fn replicated_tree_matches_full_span_with_replica_markers() {
        let bs = 4;
        let toks = tokens(bs * 6);
        let pages0: Vec<BlockId> = vec![10, 12, 14]; // global blocks 0,2,4
        let pages1: Vec<BlockId> = vec![11, 13, 15]; // global blocks 1,3,5

        let mut rank0 = RadixCache::new(bs);
        rank0.set_cp_shard(ShardSpec::new(0, 2));
        let n0 = rank0.insert_replicated(&toks, &|j| {
            if j % 2 == 0 {
                Some(pages0[j / 2])
            } else {
                None
            }
        });
        let mut rank1 = RadixCache::new(bs);
        rank1.set_cp_shard(ShardSpec::new(1, 2));
        let n1 = rank1.insert_replicated(&toks, &|j| {
            if j % 2 == 1 {
                Some(pages1[j / 2])
            } else {
                None
            }
        });
        assert_eq!(n0, pages0);
        assert_eq!(n1, pages1);

        // Both ranks match the FULL span; non-owning blocks are REPLICA_PAGE.
        let m0 = rank0.peek_longest_prefix_match(&toks);
        let m1 = rank1.peek_longest_prefix_match(&toks);
        assert_eq!(m0.matched_len, toks.len());
        assert_eq!(m1.matched_len, toks.len());
        assert_eq!(
            m0.block_ids,
            vec![10, REPLICA_PAGE, 12, REPLICA_PAGE, 14, REPLICA_PAGE]
        );
        assert_eq!(
            m1.block_ids,
            vec![REPLICA_PAGE, 11, REPLICA_PAGE, 13, REPLICA_PAGE, 15]
        );
        assert_eq!(m0.local_block_ids(ShardSpec::new(0, 2)), pages0);
        assert_eq!(m1.local_block_ids(ShardSpec::new(1, 2)), pages1);
        // Replica nodes never enter page_to_node.
        assert_eq!(rank0.cached_page_count(), 3);
        assert_eq!(rank1.cached_page_count(), 3);
    }

    #[test]
    fn replicated_match_stops_at_local_gap_but_walks_replicas() {
        let bs = 4;
        let toks = tokens(bs * 4);
        let mut rank0 = RadixCache::new(bs);
        rank0.set_cp_shard(ShardSpec::new(0, 2));
        // Rank 0 owns blocks 0,2; block 1 is a replica.
        rank0.insert_replicated(&toks, &|j| {
            if j % 2 == 0 {
                Some((100 + j) as BlockId)
            } else {
                None
            }
        });
        // Evict the deepest local page (block 2): the match now ends at
        // block 2's position, having walked the block-1 replica.
        assert_eq!(rank0.evict_lru(1), vec![102]);
        let m = rank0.peek_longest_prefix_match(&toks);
        assert_eq!(m.matched_len, bs * 2);
        assert_eq!(m.block_ids, vec![100, REPLICA_PAGE]);
    }

    #[test]
    fn replica_children_do_not_pin_but_live_local_descendants_do() {
        let bs = 4;
        let toks = tokens(bs * 4);
        let mut cache = RadixCache::new(bs);
        cache.set_cp_shard(ShardSpec::new(0, 2));
        cache.insert_replicated(&toks, &|j| {
            if j % 2 == 0 {
                Some((100 + j) as BlockId)
            } else {
                None
            }
        });
        // Block 0 has a live local descendant (block 2): not evictable.
        let victims = cache.lru_evictable_pages(usize::MAX);
        assert_eq!(victims, vec![102]);
        // Once block 2 is gone, block 0's only descendants are replicas:
        // it becomes evictable and drains cleanly.
        assert_eq!(cache.evict_lru(1), vec![102]);
        assert_eq!(cache.evict_lru(1), vec![100]);
        assert_eq!(cache.cached_page_count(), 0);
    }

    #[test]
    fn demote_promote_keeps_evictability_consistent() {
        let bs = 4;
        let toks = tokens(bs * 2);
        let mut cache = RadixCache::new(bs);
        cache.insert_replicated(&toks, &|j| Some((10 + j) as BlockId));
        // Demote the leaf: its parent becomes evictable (demoted child does
        // not pin), and the demoted block itself leaves the resident set.
        assert!(cache.demote_block(11, 777));
        assert_eq!(cache.lru_evictable_pages(usize::MAX), vec![10]);
        assert_eq!(cache.demoted_block_count(), 1);
        // Promote back: the parent is pinned again, the leaf is evictable.
        assert!(cache.promote_block(777, 11));
        assert_eq!(cache.lru_evictable_pages(usize::MAX), vec![11]);
        assert_eq!(cache.demoted_block_count(), 0);
    }
}

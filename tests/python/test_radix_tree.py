"""
Radix tree prefix cache tests — no GPU required.

Pure Python port of infer/src/prefix_cache.rs.
Tests node insertion, longest-prefix matching, reference counting,
LRU eviction, and edge cases.

Run:
    pytest tests/python/test_radix_tree.py -v
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Python mirror of prefix_cache.rs
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class BlockId:
    value: int

    def __repr__(self):
        return f"BlockId({self.value})"


@dataclass
class _Node:
    tokens: list[int]
    block_id: Optional[BlockId]
    ref_count: int
    last_access: int
    children: dict[int, int]  # first_token → node_index

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RadixCache:
    """Pure Python port of infer::prefix_cache::RadixCache."""

    def __init__(self, block_size: int):
        assert block_size > 0, "block_size must be > 0"
        self._nodes: list[_Node] = [
            _Node(tokens=[], block_id=None, ref_count=0, last_access=0, children={})
        ]
        self._block_size = block_size
        self._clock = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _root(self) -> int:
        return 0

    def _dec_refs_by_block_id(self, blocks: list[BlockId]):
        block_set = set(blocks)
        for node in self._nodes:
            if node.block_id in block_set and node.ref_count > 0:
                node.ref_count -= 1

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, tokens: list[int]) -> tuple[int, list[BlockId]]:
        """Find longest cached prefix. Returns (matched_len, matched_blocks)."""
        now = self._tick()
        node_idx = self._root()
        pos = 0
        matched_blocks: list[BlockId] = []

        while True:
            self._nodes[node_idx].last_access = now

            if pos >= len(tokens):
                break

            next_token = tokens[pos]
            child_idx = self._nodes[node_idx].children.get(next_token)
            if child_idx is None:
                break

            child_tokens = self._nodes[child_idx].tokens
            remaining = tokens[pos:]
            match_len = sum(
                1 for a, b in zip(child_tokens, remaining) if a == b
            )

            if match_len < len(child_tokens):
                break  # partial edge match — stop

            # Full edge match
            pos += match_len
            self._nodes[child_idx].last_access = now

            if self._nodes[child_idx].block_id is not None:
                self._nodes[child_idx].ref_count += 1
                matched_blocks.append(self._nodes[child_idx].block_id)

            node_idx = child_idx

        # Round down to block boundary
        block_count = len(matched_blocks)
        rounded = block_count * self._block_size

        # Trim if we went past rounded boundary
        actual_block_count = rounded // self._block_size
        if len(matched_blocks) > actual_block_count:
            extra = matched_blocks[actual_block_count:]
            matched_blocks = matched_blocks[:actual_block_count]
            self._dec_refs_by_block_id(extra)

        return rounded, matched_blocks

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, tokens: list[int], blocks: list[BlockId]) -> int:
        """Insert token sequence with corresponding blocks."""
        if not blocks or not tokens:
            return 0

        total_tokens = len(blocks) * self._block_size
        tokens = tokens[:min(len(tokens), total_tokens)]
        now = self._tick()

        node_idx = self._root()
        pos = 0
        block_idx = 0

        while pos < len(tokens) and block_idx < len(blocks):
            next_token = tokens[pos]

            if next_token in self._nodes[node_idx].children:
                child_idx = self._nodes[node_idx].children[next_token]
                child_tokens = list(self._nodes[child_idx].tokens)
                remaining = tokens[pos:]
                match_len = sum(
                    1 for a, b in zip(child_tokens, remaining) if a == b
                )

                if match_len == len(child_tokens):
                    # Full edge match — descend
                    self._nodes[child_idx].last_access = now
                    pos += match_len

                    if len(child_tokens) == self._block_size and block_idx < len(blocks):
                        self._nodes[child_idx].block_id = blocks[block_idx]
                        block_idx += 1

                    node_idx = child_idx
                else:
                    # Partial match — split edge
                    split_point = match_len
                    old_suffix = child_tokens[split_point:]
                    old_block = self._nodes[child_idx].block_id
                    old_children = dict(self._nodes[child_idx].children)

                    # Shared prefix node
                    shared_tokens = child_tokens[:split_point]
                    shared_idx = len(self._nodes)
                    self._nodes.append(_Node(
                        tokens=shared_tokens,
                        block_id=None,
                        ref_count=0,
                        last_access=now,
                        children={},
                    ))

                    # Rewire original child
                    self._nodes[child_idx].tokens = old_suffix
                    self._nodes[child_idx].block_id = old_block
                    self._nodes[child_idx].children = old_children

                    first_old = old_suffix[0]
                    self._nodes[shared_idx].children[first_old] = child_idx

                    # Replace pointer in parent
                    self._nodes[node_idx].children[next_token] = shared_idx

                    pos += split_point
                    node_idx = shared_idx
                    break
            else:
                # No matching child — insert remaining as new subtree
                while pos < len(tokens) and block_idx < len(blocks):
                    end = min(pos + self._block_size, len(tokens))
                    edge_tokens = tokens[pos:end]
                    if len(edge_tokens) == self._block_size:
                        bid = blocks[block_idx]
                        block_idx += 1
                    else:
                        bid = None

                    new_idx = len(self._nodes)
                    self._nodes.append(_Node(
                        tokens=list(edge_tokens),
                        block_id=bid,
                        ref_count=0,
                        last_access=now,
                        children={},
                    ))

                    first_tok = edge_tokens[0]
                    self._nodes[node_idx].children[first_tok] = new_idx

                    pos = end
                    node_idx = new_idx

                    if len(edge_tokens) < self._block_size:
                        break
                break

        return block_idx * self._block_size

    # ------------------------------------------------------------------
    # Reference counting
    # ------------------------------------------------------------------

    def release(self, blocks: list[BlockId]):
        """Decrement ref counts for the given blocks."""
        self._dec_refs_by_block_id(blocks)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict(self, n: int) -> list[BlockId]:
        """Evict up to n LRU leaf nodes with ref_count == 0."""
        if n == 0:
            return []

        candidates = [
            (node.last_access, idx)
            for idx, node in enumerate(self._nodes)
            if idx != self._root()
            and node.is_leaf
            and node.ref_count == 0
            and node.block_id is not None
        ]
        candidates.sort()  # ascending by last_access (LRU first)

        to_evict = {idx for _, idx in candidates[:n]}
        freed = []

        for idx in to_evict:
            if self._nodes[idx].block_id is not None:
                freed.append(self._nodes[idx].block_id)
                self._nodes[idx].block_id = None

        # Remove evicted nodes from their parents
        for node in self._nodes:
            node.children = {
                k: v for k, v in node.children.items() if v not in to_evict
            }

        return freed

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return len(self._nodes)

    def cached_block_count(self) -> int:
        return sum(1 for n in self._nodes if n.block_id is not None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bids(*values: int) -> list[BlockId]:
    return [BlockId(v) for v in values]


# ---------------------------------------------------------------------------
# Tests: basic lookup/insert
# ---------------------------------------------------------------------------

class TestBasicInsertLookup:

    def test_empty_cache_no_match(self):
        cache = RadixCache(4)
        length, blocks = cache.lookup([1, 2, 3, 4])
        assert length == 0
        assert blocks == []

    def test_insert_and_lookup_exact_one_block(self):
        cache = RadixCache(4)
        tokens = [10, 20, 30, 40]
        cache.insert(tokens, bids(100))
        length, blocks = cache.lookup(tokens)
        assert length == 4
        assert blocks == bids(100)

    def test_lookup_longer_query_than_cached(self):
        cache = RadixCache(4)
        cache.insert([1, 2, 3, 4], bids(10))
        length, blocks = cache.lookup([1, 2, 3, 4, 5, 6, 7, 8])
        assert length == 4
        assert blocks == bids(10)

    def test_lookup_mismatch_returns_zero(self):
        cache = RadixCache(4)
        cache.insert([1, 2, 3, 4], bids(10))
        length, blocks = cache.lookup([5, 6, 7, 8])
        assert length == 0
        assert blocks == []

    def test_insert_multiple_blocks(self):
        cache = RadixCache(2)
        tokens = list(range(1, 7))  # 6 tokens = 3 blocks
        cache.insert(tokens, bids(10, 20, 30))
        length, blocks = cache.lookup(tokens)
        assert length == 6
        assert blocks == bids(10, 20, 30)

    def test_empty_insert_returns_zero(self):
        cache = RadixCache(4)
        result = cache.insert([], bids(10))
        assert result == 0

    def test_empty_blocks_returns_zero(self):
        cache = RadixCache(4)
        result = cache.insert([1, 2, 3], [])
        assert result == 0

    def test_insert_idempotent(self):
        cache = RadixCache(4)
        tokens = [1, 2, 3, 4]
        cache.insert(tokens, bids(10))
        cache.insert(tokens, bids(10))  # second insert
        length, blocks = cache.lookup(tokens)
        assert length == 4
        assert blocks == bids(10)


# ---------------------------------------------------------------------------
# Tests: prefix sharing
# ---------------------------------------------------------------------------

class TestPrefixSharing:

    def test_two_requests_share_prefix(self):
        cache = RadixCache(4)
        shared = [1, 2, 3, 4]
        cache.insert(shared, bids(10))

        req_a = shared + [5, 6, 7, 8]
        req_b = shared + [9, 10, 11, 12]
        cache.insert(req_a, bids(10, 20))
        cache.insert(req_b, bids(10, 30))

        len_a, blk_a = cache.lookup(req_a)
        len_b, blk_b = cache.lookup(req_b)
        cache.release(blk_a)
        cache.release(blk_b)

        assert len_a == 8
        assert blk_a == bids(10, 20)
        assert len_b == 8
        assert blk_b == bids(10, 30)

    def test_prefix_only_cached_first(self):
        cache = RadixCache(4)
        prefix = [1, 2, 3, 4]
        cache.insert(prefix, bids(10))

        # Query with extension — only prefix matched
        length, blocks = cache.lookup(prefix + [99, 100, 101, 102])
        cache.release(blocks)
        assert length == 4

    def test_long_shared_prefix_three_requests(self):
        cache = RadixCache(2)
        # All share first 4 tokens = 2 blocks
        shared = [1, 2, 3, 4]
        cache.insert(shared, bids(10, 20))
        cache.insert(shared + [5, 6], bids(10, 20, 30))
        cache.insert(shared + [7, 8], bids(10, 20, 40))

        for ext, bid in [([5, 6], 30), ([7, 8], 40)]:
            l, b = cache.lookup(shared + ext)
            cache.release(b)
            assert l == 6
            assert b[-1] == BlockId(bid)


# ---------------------------------------------------------------------------
# Tests: partial block handling
# ---------------------------------------------------------------------------

class TestPartialBlock:

    def test_partial_block_not_cached(self):
        cache = RadixCache(4)
        # 3 tokens < 1 block of size 4 → nothing inserted
        result = cache.insert([1, 2, 3], [])
        assert result == 0
        length, _ = cache.lookup([1, 2, 3])
        assert length == 0

    def test_trailing_partial_tokens_not_cached(self):
        # 5 tokens with block_size=4 → only first 4 inserted
        cache = RadixCache(4)
        tokens = [1, 2, 3, 4, 5]
        cache.insert(tokens, bids(10))
        length, blocks = cache.lookup(tokens)
        cache.release(blocks)
        assert length == 4  # only 1 full block

    def test_block_size_1(self):
        cache = RadixCache(1)
        tokens = [10, 20, 30]
        cache.insert(tokens, bids(1, 2, 3))
        length, blocks = cache.lookup(tokens)
        cache.release(blocks)
        assert length == 3
        assert blocks == bids(1, 2, 3)


# ---------------------------------------------------------------------------
# Tests: reference counting
# ---------------------------------------------------------------------------

class TestRefCounting:

    def test_lookup_increments_ref(self):
        cache = RadixCache(4)
        cache.insert([1, 2, 3, 4], bids(10))
        _, blocks = cache.lookup([1, 2, 3, 4])
        # ref_count should be 1; can't evict now
        freed = cache.evict(1)
        assert freed == []  # pinned
        cache.release(blocks)
        freed = cache.evict(1)
        assert freed == bids(10)

    def test_multiple_lookups_increment_multiple(self):
        cache = RadixCache(4)
        cache.insert([1, 2, 3, 4], bids(10))

        _, b1 = cache.lookup([1, 2, 3, 4])
        _, b2 = cache.lookup([1, 2, 3, 4])

        # Still pinned after first release
        cache.release(b1)
        freed = cache.evict(1)
        assert freed == []  # still ref_count > 0

        cache.release(b2)
        freed = cache.evict(1)
        assert freed == bids(10)

    def test_release_without_lookup_no_op(self):
        cache = RadixCache(4)
        # Releasing a block that was never looked up should not crash
        cache.release(bids(999))

    def test_evict_respects_ref_count(self):
        cache = RadixCache(4)
        cache.insert([1, 2, 3, 4], bids(10))
        _, _blocks = cache.lookup([1, 2, 3, 4])
        # Don't release → ref_count > 0
        freed = cache.evict(1)
        assert freed == []


# ---------------------------------------------------------------------------
# Tests: LRU eviction
# ---------------------------------------------------------------------------

class TestLruEviction:

    def test_evict_lru_leaf(self):
        cache = RadixCache(4)
        cache.insert([1, 2, 3, 4], bids(10))
        cache.insert([5, 6, 7, 8], bids(20))

        # Access second sequence (makes it more recently used)
        _, b = cache.lookup([5, 6, 7, 8])
        cache.release(b)

        # Both were just inserted; but second was accessed later.
        # Release first block's ref too (insert doesn't add refs, only lookup does).
        freed = cache.evict(1)
        # Should evict BlockId(10) (older access) not BlockId(20).
        assert freed == bids(10)

    def test_evict_n_blocks(self):
        cache = RadixCache(4)
        for i in range(4):
            cache.insert(
                [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3],
                [BlockId(i * 10)]
            )
        freed = cache.evict(2)
        assert len(freed) == 2

    def test_evict_more_than_available(self):
        cache = RadixCache(4)
        cache.insert([1, 2, 3, 4], bids(10))
        freed = cache.evict(100)  # only 1 evictable
        assert freed == bids(10)

    def test_evict_zero_returns_empty(self):
        cache = RadixCache(4)
        cache.insert([1, 2, 3, 4], bids(10))
        assert cache.evict(0) == []

    def test_evict_empty_cache(self):
        cache = RadixCache(4)
        assert cache.evict(5) == []

    def test_evicted_block_no_longer_returned_by_lookup(self):
        cache = RadixCache(4)
        tokens = [1, 2, 3, 4]
        cache.insert(tokens, bids(10))
        cache.evict(1)
        length, blocks = cache.lookup(tokens)
        assert length == 0
        assert blocks == []


# ---------------------------------------------------------------------------
# Tests: node_count / cached_block_count
# ---------------------------------------------------------------------------

class TestStats:

    def test_initial_node_count(self):
        cache = RadixCache(4)
        assert cache.node_count() == 1  # just root

    def test_cached_block_count_starts_zero(self):
        cache = RadixCache(4)
        assert cache.cached_block_count() == 0

    def test_cached_block_count_increments(self):
        cache = RadixCache(4)
        cache.insert([1, 2, 3, 4], bids(10))
        assert cache.cached_block_count() == 1
        cache.insert([5, 6, 7, 8], bids(20))
        assert cache.cached_block_count() == 2

    def test_cached_block_count_after_evict(self):
        cache = RadixCache(4)
        cache.insert([1, 2, 3, 4], bids(10))
        cache.evict(1)
        assert cache.cached_block_count() == 0


# ---------------------------------------------------------------------------
# Tests: edge cases / corner cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_block_size_equals_query_length(self):
        cache = RadixCache(8)
        tokens = list(range(8))
        cache.insert(tokens, bids(42))
        length, blocks = cache.lookup(tokens)
        cache.release(blocks)
        assert length == 8
        assert blocks == bids(42)

    def test_single_token_block(self):
        cache = RadixCache(1)
        cache.insert([7], bids(99))
        length, blocks = cache.lookup([7])
        cache.release(blocks)
        assert length == 1
        assert blocks == bids(99)

    def test_no_match_different_prefix(self):
        cache = RadixCache(4)
        cache.insert([1, 2, 3, 4], bids(10))
        length, blocks = cache.lookup([1, 2, 3, 99])
        assert length == 0  # partial match within a node = no match

    def test_repeated_same_token(self):
        cache = RadixCache(4)
        tokens = [5, 5, 5, 5]
        cache.insert(tokens, bids(77))
        length, blocks = cache.lookup(tokens)
        cache.release(blocks)
        assert length == 4
        assert blocks == bids(77)

    def test_large_block_size(self):
        cache = RadixCache(16)
        tokens = list(range(16))
        cache.insert(tokens, bids(1))
        length, blocks = cache.lookup(tokens)
        cache.release(blocks)
        assert length == 16

    def test_insert_then_extend(self):
        cache = RadixCache(4)
        base = [1, 2, 3, 4]
        ext = [5, 6, 7, 8]
        cache.insert(base, bids(10))
        cache.insert(base + ext, bids(10, 20))

        length, blocks = cache.lookup(base + ext)
        cache.release(blocks)
        assert length == 8
        assert blocks == bids(10, 20)

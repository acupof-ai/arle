"""
KV cache metadata and management tests — no GPU required.

Models the resident-only CPU-side metadata tracking of infer/src/model/kv_cache.rs.
Tests sequence length tracking, prefix truncation, reset behavior, and TokenKVPool
budget accounting without touching actual GPU memory.

Run:
    pytest tests/test_kv_cache.py -v
"""

import pytest


class KVCacheMeta:
    """CPU-side mirror of the resident-only contiguous KV cache metadata."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self._seq_len: int = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def increment_seq_len(self):
        self._seq_len += 1

    def advance_seq_len(self, n: int):
        self._seq_len += n

    def truncate_to(self, target_len: int):
        if target_len > self._seq_len:
            raise ValueError(
                f"Cannot truncate to {target_len} > current seq_len {self._seq_len}"
            )
        if target_len < 0:
            raise ValueError("target_len must be >= 0")
        self._seq_len = target_len

    def reset(self):
        self._seq_len = 0

    def bytes_per_token(self) -> int:
        """Memory footprint per token (K + V, all layers), in bf16 (2 bytes)."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2

    def gpu_memory_bytes(self) -> int:
        return self._seq_len * self.bytes_per_token()

    def total_memory_bytes(self) -> int:
        return self._seq_len * self.bytes_per_token()


def make_cache(
    num_layers=4,
    num_kv_heads=8,
    head_dim=64,
    max_seq_len=512,
):
    return KVCacheMeta(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
    )


class TestInitialState:
    def test_zero_seq_len(self):
        cache = make_cache()
        assert cache.seq_len == 0

    def test_gpu_memory_zero_initially(self):
        cache = make_cache()
        assert cache.gpu_memory_bytes() == 0


class TestSeqLenTracking:
    def test_increment_seq_len(self):
        cache = make_cache()
        cache.increment_seq_len()
        assert cache.seq_len == 1

    def test_increment_multiple_times(self):
        cache = make_cache()
        for _ in range(10):
            cache.increment_seq_len()
        assert cache.seq_len == 10

    def test_advance_seq_len(self):
        cache = make_cache()
        cache.advance_seq_len(128)
        assert cache.seq_len == 128

    def test_advance_and_increment(self):
        cache = make_cache()
        cache.advance_seq_len(50)
        cache.increment_seq_len()
        cache.increment_seq_len()
        assert cache.seq_len == 52


class TestReset:
    def test_reset_clears_seq_len(self):
        cache = make_cache()
        cache.advance_seq_len(200)
        cache.reset()
        assert cache.seq_len == 0

    def test_double_reset_safe(self):
        cache = make_cache()
        cache.reset()
        cache.reset()
        assert cache.seq_len == 0


class TestTruncateTo:
    def test_truncate_to_shorter(self):
        cache = make_cache()
        cache.advance_seq_len(100)
        cache.truncate_to(50)
        assert cache.seq_len == 50

    def test_truncate_to_same_length(self):
        cache = make_cache()
        cache.advance_seq_len(100)
        cache.truncate_to(100)
        assert cache.seq_len == 100

    def test_truncate_to_zero(self):
        cache = make_cache()
        cache.advance_seq_len(50)
        cache.truncate_to(0)
        assert cache.seq_len == 0

    def test_truncate_beyond_seq_len_raises(self):
        cache = make_cache()
        cache.advance_seq_len(50)
        with pytest.raises(ValueError):
            cache.truncate_to(51)

    def test_truncate_negative_raises(self):
        cache = make_cache()
        cache.advance_seq_len(50)
        with pytest.raises(ValueError):
            cache.truncate_to(-1)


class TestMemoryAccounting:
    def test_bytes_per_token(self):
        cache = make_cache(num_layers=4, num_kv_heads=8, head_dim=64)
        expected = 2 * 4 * 8 * 64 * 2
        assert cache.bytes_per_token() == expected

    def test_gpu_memory_proportional_to_seq_len(self):
        cache = make_cache()
        cache.advance_seq_len(100)
        assert cache.gpu_memory_bytes() == 100 * cache.bytes_per_token()

    def test_total_memory_matches_gpu_memory(self):
        cache = make_cache()
        cache.advance_seq_len(256)
        assert cache.total_memory_bytes() == cache.gpu_memory_bytes()


class TestMultiRequestSimulation:
    def test_two_requests_independent(self):
        c1 = make_cache()
        c2 = make_cache()

        c1.advance_seq_len(100)
        c2.advance_seq_len(200)

        assert c1.seq_len == 100
        assert c2.seq_len == 200

    def test_reuse_slot_after_reset(self):
        cache = make_cache()
        cache.advance_seq_len(300)
        cache.reset()
        cache.advance_seq_len(50)
        assert cache.seq_len == 50

    def test_prefix_reuse_workflow(self):
        cache = make_cache()
        cache.advance_seq_len(200)
        cache.truncate_to(100)
        assert cache.seq_len == 100
        cache.advance_seq_len(50)
        assert cache.seq_len == 150


class TokenKVPoolMeta:
    """CPU-side metadata mirror of TokenKVPool for budget/capacity tests."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_slots: int,
        budget_bytes: int,
        dtype: str = "bf16",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.num_slots = num_slots
        self.dtype = dtype

        if dtype == "bf16":
            bytes_per_token = self.kv_dim * 2 * num_layers * 2
            fixed_cost = 0
        elif dtype == "int8":
            bytes_per_token = num_layers * 2 * (self.kv_dim + num_kv_heads * 4)
            fixed_cost = self.kv_dim * 2 * 2
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        effective = budget_bytes - fixed_cost
        self.max_total_tokens = (
            max(effective // bytes_per_token, num_slots)
            if bytes_per_token > 0
            else num_slots
        )
        self.bytes_per_token = bytes_per_token
        self.fixed_cost = fixed_cost

        self.free_slots = list(range(self.max_total_tokens))
        self.token_indices: list[list[int]] = [[] for _ in range(num_slots)]

    def alloc_tokens(self, slot: int, count: int) -> list[int]:
        if count > len(self.free_slots):
            raise RuntimeError(f"Out of token slots: need {count}, have {len(self.free_slots)}")
        new_indices = [self.free_slots.pop() for _ in range(count)]
        self.token_indices[slot].extend(new_indices)
        return new_indices

    def free_slot(self, slot: int):
        self.free_slots.extend(self.token_indices[slot])
        self.token_indices[slot].clear()

    def seq_len(self, slot: int) -> int:
        return len(self.token_indices[slot])

    def free_count(self) -> int:
        return len(self.free_slots)

    def is_active(self) -> bool:
        return self.max_total_tokens > 0

    def total_memory_bytes(self) -> int:
        return self.max_total_tokens * self.bytes_per_token + self.fixed_cost


def make_pool(dtype="bf16", budget_mb=100, num_layers=36, num_kv_heads=8, head_dim=128, num_slots=4):
    return TokenKVPoolMeta(
        num_layers,
        num_kv_heads,
        head_dim,
        num_slots,
        int(budget_mb * 1e6),
        dtype,
    )


class TestTokenKVPoolINT8:
    def test_int8_has_more_tokens_than_bf16(self):
        bf16 = make_pool(dtype="bf16", budget_mb=100)
        int8 = make_pool(dtype="int8", budget_mb=100)
        assert int8.max_total_tokens > bf16.max_total_tokens * 1.5

    def test_int8_memory_savings(self):
        bf16 = make_pool(dtype="bf16", budget_mb=100)
        int8 = make_pool(dtype="int8", budget_mb=100)
        bf16_total = bf16.total_memory_bytes()
        int8_total = int8.total_memory_bytes()
        assert abs(bf16_total - int8_total) / bf16_total < 0.1

    def test_alloc_free_int8(self):
        pool = make_pool(dtype="int8")
        initial_free = pool.free_count()
        pool.alloc_tokens(0, 100)
        assert pool.seq_len(0) == 100
        assert pool.free_count() == initial_free - 100
        pool.free_slot(0)
        assert pool.seq_len(0) == 0
        assert pool.free_count() == initial_free

    def test_int8_pool_is_active(self):
        pool = make_pool(dtype="int8")
        assert pool.is_active()

    def test_int8_budget_accounts_for_working_buffer(self):
        pool = make_pool(dtype="int8", budget_mb=1, num_layers=2, num_kv_heads=4, head_dim=64)
        kv_dim = 4 * 64
        fixed = kv_dim * 2 * 2
        assert pool.fixed_cost == fixed
        assert pool.max_total_tokens >= pool.num_slots

    def test_multi_slot_int8(self):
        pool = make_pool(dtype="int8", num_slots=4)
        pool.alloc_tokens(0, 50)
        pool.alloc_tokens(1, 100)
        pool.alloc_tokens(2, 75)
        assert pool.seq_len(0) == 50
        assert pool.seq_len(1) == 100
        assert pool.seq_len(2) == 75
        assert pool.free_count() == pool.max_total_tokens - 225

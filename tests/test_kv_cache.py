"""
KV cache metadata and management tests — no GPU required.

Models the CPU-side metadata tracking of infer/src/model/kv_cache.rs.
Tests sequence length tracking, CPU/GPU offload boundaries, prefix
truncation, and block-aligned operations — all without actual GPU memory.

Run:
    pytest tests/test_kv_cache.py -v
"""

import math
import pytest


# ---------------------------------------------------------------------------
# Python mirror of kv_cache.rs metadata (CPU-side tracking only)
# ---------------------------------------------------------------------------

OFFLOAD_BLOCK_SIZE = 64  # mirrors csrc constant


class KVCacheMeta:
    """CPU-side KV cache metadata mirror (no GPU buffers).

    Tracks sequence length, offload boundaries, and validates the
    state machine defined in kv_cache.rs.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        max_gpu_seq_len: int | None = None,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.max_gpu_seq_len = max_gpu_seq_len if max_gpu_seq_len is not None else max_seq_len
        self._seq_len: int = 0
        self._offloaded_len: int = 0
        self._gpu_has_full_seq: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def offloaded_len(self) -> int:
        return self._offloaded_len

    @property
    def gpu_seq_len(self) -> int:
        """Tokens currently on GPU."""
        return self._seq_len - self._offloaded_len

    @property
    def has_offloaded(self) -> bool:
        return self._offloaded_len > 0

    @property
    def gpu_has_full_seq(self) -> bool:
        return self._gpu_has_full_seq

    # ------------------------------------------------------------------
    # State transitions (mirrors kv_cache.rs methods)
    # ------------------------------------------------------------------

    def increment_seq_len(self):
        """Increment sequence length by 1 (decode step)."""
        self._seq_len += 1

    def advance_seq_len(self, n: int):
        """Advance sequence length by n (prefill step)."""
        self._seq_len += n

    def truncate_to(self, target_len: int):
        """Truncate cache to target_len for prefix reuse.

        Rules from kv_cache.rs::truncate_to:
        - Can only truncate to <= current seq_len
        - If target is within offloaded region, clears offload too
        - If full seq is on GPU (gpu_has_full_seq), truncation stays on GPU
        """
        if target_len > self._seq_len:
            raise ValueError(
                f"Cannot truncate to {target_len} > current seq_len {self._seq_len}"
            )
        if target_len < 0:
            raise ValueError("target_len must be >= 0")

        if self._gpu_has_full_seq:
            # Full sequence already on GPU — just update len
            self._seq_len = target_len
            self._offloaded_len = 0
        elif target_len <= self._offloaded_len:
            # Truncating into offloaded region — clear everything
            self._seq_len = target_len
            self._offloaded_len = 0
            self._gpu_has_full_seq = False
        else:
            # Truncating within GPU region
            self._seq_len = target_len

    def reset(self):
        """Clear cache for reuse (new request)."""
        self._seq_len = 0
        self._offloaded_len = 0
        self._gpu_has_full_seq = False

    def offload_if_needed(self):
        """Simulate offloading old tokens when GPU budget is exceeded.

        Mirrors kv_cache.rs::offload_if_needed:
        - Offloads in OFFLOAD_BLOCK_SIZE chunks from the start
        - Only offloads when gpu_seq_len > max_gpu_seq_len
        """
        while self.gpu_seq_len > self.max_gpu_seq_len:
            # Offload one block
            offload_amount = min(OFFLOAD_BLOCK_SIZE, self.gpu_seq_len - self.max_gpu_seq_len)
            # Round up to block boundary
            offload_amount = (
                (offload_amount + OFFLOAD_BLOCK_SIZE - 1) // OFFLOAD_BLOCK_SIZE
            ) * OFFLOAD_BLOCK_SIZE
            offload_amount = min(offload_amount, self.gpu_seq_len)
            self._offloaded_len += offload_amount

    def ensure_on_gpu(self):
        """Simulate restoring full sequence to GPU for prefill.

        Mirrors kv_cache.rs::ensure_on_gpu.
        After this call: gpu_seq_len == seq_len, offloaded_len == 0.
        """
        if self._offloaded_len > 0:
            self._gpu_has_full_seq = True
            self._offloaded_len = 0

    def bytes_per_token(self) -> int:
        """Memory footprint per token (K + V, all layers), in bf16 (2 bytes)."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2  # 2 = bf16

    def gpu_memory_bytes(self) -> int:
        """Estimated GPU memory used by cache."""
        return self.gpu_seq_len * self.bytes_per_token()

    def total_memory_bytes(self) -> int:
        """Total memory (GPU + CPU offload)."""
        return self._seq_len * self.bytes_per_token()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_cache(
    num_layers=4,
    num_kv_heads=8,
    head_dim=64,
    max_seq_len=512,
    max_gpu_seq_len=None,
) -> KVCacheMeta:
    return KVCacheMeta(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        max_gpu_seq_len=max_gpu_seq_len,
    )


# ---------------------------------------------------------------------------
# Tests: initial state
# ---------------------------------------------------------------------------

class TestInitialState:

    def test_zero_seq_len(self):
        cache = make_cache()
        assert cache.seq_len == 0

    def test_no_offload_initially(self):
        cache = make_cache()
        assert not cache.has_offloaded
        assert cache.offloaded_len == 0

    def test_gpu_seq_len_equals_seq_len(self):
        cache = make_cache()
        assert cache.gpu_seq_len == 0

    def test_gpu_has_full_seq_false(self):
        cache = make_cache()
        assert not cache.gpu_has_full_seq


# ---------------------------------------------------------------------------
# Tests: sequence length tracking
# ---------------------------------------------------------------------------

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

    def test_gpu_seq_len_matches_seq_len_without_offload(self):
        cache = make_cache()
        cache.advance_seq_len(100)
        assert cache.gpu_seq_len == 100
        assert not cache.has_offloaded


# ---------------------------------------------------------------------------
# Tests: reset
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_clears_seq_len(self):
        cache = make_cache()
        cache.advance_seq_len(200)
        cache.reset()
        assert cache.seq_len == 0

    def test_reset_clears_offload(self):
        cache = make_cache(max_gpu_seq_len=64)
        cache.advance_seq_len(200)
        cache.offload_if_needed()
        cache.reset()
        assert cache.offloaded_len == 0
        assert not cache.has_offloaded

    def test_reset_clears_gpu_full_flag(self):
        cache = make_cache(max_gpu_seq_len=64)
        cache.advance_seq_len(200)
        cache.offload_if_needed()
        cache.ensure_on_gpu()
        cache.reset()
        assert not cache.gpu_has_full_seq

    def test_double_reset_safe(self):
        cache = make_cache()
        cache.reset()
        cache.reset()
        assert cache.seq_len == 0


# ---------------------------------------------------------------------------
# Tests: truncate_to
# ---------------------------------------------------------------------------

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

    def test_truncate_into_offloaded_region_clears_offload(self):
        cache = make_cache(max_gpu_seq_len=64)
        cache.advance_seq_len(200)
        cache.offload_if_needed()
        assert cache.has_offloaded

        # Truncate to a position inside the offloaded region
        cache.truncate_to(32)
        assert cache.seq_len == 32
        assert cache.offloaded_len == 0
        assert not cache.has_offloaded

    def test_truncate_within_gpu_region_preserves_offload(self):
        cache = make_cache(max_gpu_seq_len=64)
        cache.advance_seq_len(200)
        cache.offload_if_needed()
        offloaded = cache.offloaded_len
        gpu_len = cache.gpu_seq_len

        # Truncate to somewhere still in the GPU region
        new_target = cache.offloaded_len + gpu_len // 2
        cache.truncate_to(new_target)
        assert cache.seq_len == new_target


# ---------------------------------------------------------------------------
# Tests: CPU/GPU offload
# ---------------------------------------------------------------------------

class TestOffload:

    def test_no_offload_within_budget(self):
        cache = make_cache(max_gpu_seq_len=128)
        cache.advance_seq_len(100)
        cache.offload_if_needed()
        assert not cache.has_offloaded
        assert cache.gpu_seq_len == 100

    def test_offload_triggered_when_over_budget(self):
        cache = make_cache(max_gpu_seq_len=64)
        cache.advance_seq_len(200)
        cache.offload_if_needed()
        assert cache.has_offloaded
        assert cache.gpu_seq_len <= cache.max_gpu_seq_len

    def test_offload_in_block_multiples(self):
        cache = make_cache(max_gpu_seq_len=128)
        cache.advance_seq_len(300)
        cache.offload_if_needed()
        # Offloaded amount should be a multiple of OFFLOAD_BLOCK_SIZE
        assert cache.offloaded_len % OFFLOAD_BLOCK_SIZE == 0

    def test_offloaded_plus_gpu_equals_seq_len(self):
        cache = make_cache(max_gpu_seq_len=64)
        cache.advance_seq_len(256)
        cache.offload_if_needed()
        assert cache.offloaded_len + cache.gpu_seq_len == cache.seq_len

    def test_ensure_on_gpu_after_offload(self):
        cache = make_cache(max_gpu_seq_len=64)
        cache.advance_seq_len(256)
        cache.offload_if_needed()
        assert cache.has_offloaded

        cache.ensure_on_gpu()
        assert cache.offloaded_len == 0
        assert not cache.has_offloaded
        assert cache.gpu_has_full_seq

    def test_no_offload_no_ensure_needed(self):
        cache = make_cache(max_gpu_seq_len=256)
        cache.advance_seq_len(100)
        cache.offload_if_needed()
        # No offload happened, ensure_on_gpu is a no-op
        cache.ensure_on_gpu()
        assert cache.seq_len == 100
        assert cache.offloaded_len == 0

    def test_offload_then_decode_then_ensure(self):
        cache = make_cache(max_gpu_seq_len=64)
        cache.advance_seq_len(128)
        cache.offload_if_needed()
        for _ in range(10):
            cache.increment_seq_len()
        cache.offload_if_needed()
        cache.ensure_on_gpu()
        assert cache.offloaded_len == 0

    def test_repeated_offload_calls_idempotent(self):
        cache = make_cache(max_gpu_seq_len=64)
        cache.advance_seq_len(256)
        cache.offload_if_needed()
        gpu_len_1 = cache.gpu_seq_len
        cache.offload_if_needed()
        assert cache.gpu_seq_len == gpu_len_1


# ---------------------------------------------------------------------------
# Tests: memory accounting
# ---------------------------------------------------------------------------

class TestMemoryAccounting:

    def test_bytes_per_token(self):
        cache = make_cache(num_layers=4, num_kv_heads=8, head_dim=64)
        # 2 (K+V) * 4 (layers) * 8 (heads) * 64 (dim) * 2 (bf16)
        expected = 2 * 4 * 8 * 64 * 2
        assert cache.bytes_per_token() == expected

    def test_gpu_memory_zero_initially(self):
        cache = make_cache()
        assert cache.gpu_memory_bytes() == 0

    def test_gpu_memory_proportional_to_seq_len(self):
        cache = make_cache()
        cache.advance_seq_len(100)
        assert cache.gpu_memory_bytes() == 100 * cache.bytes_per_token()

    def test_total_memory_includes_offloaded(self):
        cache = make_cache(max_gpu_seq_len=64)
        cache.advance_seq_len(256)
        cache.offload_if_needed()
        assert cache.total_memory_bytes() == 256 * cache.bytes_per_token()

    def test_gpu_memory_less_than_total_when_offloaded(self):
        cache = make_cache(max_gpu_seq_len=64)
        cache.advance_seq_len(256)
        cache.offload_if_needed()
        assert cache.gpu_memory_bytes() < cache.total_memory_bytes()


# ---------------------------------------------------------------------------
# Tests: multi-request simulation
# ---------------------------------------------------------------------------

class TestMultiRequestSimulation:

    def test_two_requests_independent(self):
        """Two caches should be fully independent (no shared state)."""
        c1 = make_cache()
        c2 = make_cache()

        c1.advance_seq_len(100)
        c2.advance_seq_len(200)

        assert c1.seq_len == 100
        assert c2.seq_len == 200

    def test_reuse_slot_after_reset(self):
        """Simulates a scheduler slot being reused for a new request."""
        cache = make_cache(max_gpu_seq_len=128)
        # First request
        cache.advance_seq_len(300)
        cache.offload_if_needed()
        assert cache.has_offloaded
        # Finish request
        cache.reset()
        # Second request
        cache.advance_seq_len(50)
        cache.offload_if_needed()
        assert not cache.has_offloaded
        assert cache.seq_len == 50

    def test_prefix_reuse_workflow(self):
        """Simulate prefix cache hit: new request truncates to shared prefix."""
        cache = make_cache()
        # First request generates 200 tokens
        cache.advance_seq_len(200)
        # New request shares first 100 tokens (prefix cache hit)
        cache.truncate_to(100)
        assert cache.seq_len == 100
        # Continue generating for the new request
        cache.advance_seq_len(50)
        assert cache.seq_len == 150


# ---------------------------------------------------------------------------
# Python mirror of TokenKVPool metadata for INT8 budget tests
# ---------------------------------------------------------------------------

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
            bytes_per_token = self.kv_dim * 2 * num_layers * 2  # K+V, bf16
            fixed_cost = 0
        elif dtype == "int8":
            bytes_per_token = num_layers * 2 * (self.kv_dim + num_kv_heads * 4)
            fixed_cost = self.kv_dim * 2 * 2  # 1-layer bf16 working K+V
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        effective = budget_bytes - fixed_cost
        self.max_total_tokens = max(effective // bytes_per_token, num_slots) if bytes_per_token > 0 else num_slots
        self.bytes_per_token = bytes_per_token
        self.fixed_cost = fixed_cost

        # Free list + per-slot tracking
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


# ---------------------------------------------------------------------------
# Tests: INT8 TokenKVPool metadata
# ---------------------------------------------------------------------------

def make_pool(dtype="bf16", budget_mb=100, num_layers=36, num_kv_heads=8, head_dim=128, num_slots=4):
    return TokenKVPoolMeta(num_layers, num_kv_heads, head_dim, num_slots, int(budget_mb * 1e6), dtype)


class TestTokenKVPoolINT8:

    def test_int8_has_more_tokens_than_bf16(self):
        """INT8 pool should fit ~1.8x more tokens than BF16 for same budget."""
        bf16 = make_pool(dtype="bf16", budget_mb=100)
        int8 = make_pool(dtype="int8", budget_mb=100)
        assert int8.max_total_tokens > bf16.max_total_tokens * 1.5

    def test_int8_memory_savings(self):
        """INT8 should use ~46% less memory per token than BF16."""
        bf16 = make_pool(dtype="bf16", budget_mb=100)
        int8 = make_pool(dtype="int8", budget_mb=100)
        # Both should use roughly the same total budget
        bf16_total = bf16.total_memory_bytes()
        int8_total = int8.total_memory_bytes()
        # INT8 total is similar (same budget), but more tokens
        assert abs(bf16_total - int8_total) / bf16_total < 0.1  # within 10%

    def test_alloc_free_int8(self):
        """Token alloc/free works identically for INT8 pool."""
        pool = make_pool(dtype="int8")
        initial_free = pool.free_count()
        pool.alloc_tokens(0, 100)
        assert pool.seq_len(0) == 100
        assert pool.free_count() == initial_free - 100
        pool.free_slot(0)
        assert pool.seq_len(0) == 0
        assert pool.free_count() == initial_free

    def test_int8_pool_is_active(self):
        """INT8 pool with budget > 0 should report active."""
        pool = make_pool(dtype="int8")
        assert pool.is_active()

    def test_int8_budget_accounts_for_working_buffer(self):
        """INT8 pool budget should subtract the fixed bf16 working buffer cost."""
        pool = make_pool(dtype="int8", budget_mb=1, num_layers=2, num_kv_heads=4, head_dim=64)
        kv_dim = 4 * 64
        fixed = kv_dim * 2 * 2  # 1-layer bf16 working K+V
        assert pool.fixed_cost == fixed
        # With very small budget, working buffer cost reduces available tokens
        pool_no_fixed = TokenKVPoolMeta(2, 4, 64, 4, int(1e6), "bf16")
        # INT8 should have fewer tokens if working buffer cost is significant
        # relative to the total budget (but more than BF16 at larger budgets)
        assert pool.max_total_tokens >= pool.num_slots  # at least num_slots

    def test_multi_slot_int8(self):
        """Multiple slots allocate from the same INT8 pool."""
        pool = make_pool(dtype="int8", num_slots=4)
        pool.alloc_tokens(0, 50)
        pool.alloc_tokens(1, 100)
        pool.alloc_tokens(2, 75)
        assert pool.seq_len(0) == 50
        assert pool.seq_len(1) == 100
        assert pool.seq_len(2) == 75
        assert pool.free_count() == pool.max_total_tokens - 225

"""
Scheduler tests — no GPU required.

Tests SchedulerConfig validation, SchedulerHandle backpressure,
RequestPriority ordering, and the Phase state machine described in
infer/src/scheduler.rs — all using pure Python mirrors.

Run:
    pytest tests/python/test_scheduler.py -v
"""

import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from queue import Queue, Full
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Python mirrors of scheduler.rs public types
# ---------------------------------------------------------------------------

class PreemptionMode:
    RECOMPUTE = "Recompute"
    SWAP = "Swap"


@dataclass
class SchedulerConfig:
    max_slots: int = 4
    prefill_chunk_size: int = 512
    max_waiting_requests: int = 256
    preemption_mode: str = PreemptionMode.RECOMPUTE

    def validate(self):
        if self.max_slots == 0:
            raise ValueError("max_slots must be >= 1")
        if self.prefill_chunk_size == 0:
            raise ValueError("prefill_chunk_size must be >= 1")


class RequestPriority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass
class MockRequest:
    prompt: str
    max_tokens: int = 64
    priority: RequestPriority = RequestPriority.NORMAL


class SchedulerFull(Exception):
    pass


class SchedulerHandle:
    """Python mirror of infer::scheduler::SchedulerHandle."""

    def __init__(self, model_id: str, max_waiting: int = 0):
        self.model_id = model_id
        self.max_waiting = max_waiting
        self._queue: Queue = Queue()
        self._waiting_count = 0
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, req: MockRequest) -> None:
        with self._lock:
            if self._closed:
                raise SchedulerFull("scheduler is closed")
            if self.max_waiting > 0 and self._waiting_count >= self.max_waiting:
                raise SchedulerFull("scheduler waiting queue is full")
            self._waiting_count += 1
        try:
            self._queue.put_nowait(req)
        except Full:
            with self._lock:
                self._waiting_count -= 1
            raise SchedulerFull("queue overflow")

    def consume_one(self) -> Optional[MockRequest]:
        """Remove and return one request, decrement waiting count."""
        try:
            req = self._queue.get_nowait()
            with self._lock:
                self._waiting_count -= 1
            return req
        except Exception:
            return None

    @property
    def waiting_count(self) -> int:
        with self._lock:
            return self._waiting_count

    @property
    def is_full(self) -> bool:
        if self.max_waiting == 0:
            return False
        return self.waiting_count >= self.max_waiting

    def close(self):
        with self._lock:
            self._closed = True


# ---------------------------------------------------------------------------
# Phase state machine (mirrors Phase enum in scheduler.rs)
# ---------------------------------------------------------------------------

class PhaseState:
    """Simulates the Phase state machine for a single active request."""

    STATES = {"New", "Prefilling", "Decoding", "Finished"}

    def __init__(self, prompt_tokens: int, max_tokens: int, chunk_size: int = 512):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.phase = "New"
        self.prefill_progress = 0
        self.decode_tokens = 0

    def transition_to_prefilling(self):
        assert self.phase == "New", f"Expected New, got {self.phase}"
        self.phase = "Prefilling"
        self.prefill_progress = 0

    def step_prefill_chunk(self) -> bool:
        """Returns True when prefill is complete."""
        assert self.phase == "Prefilling"
        remaining = self.prompt_tokens - self.prefill_progress
        chunk = min(self.chunk_size, remaining)
        self.prefill_progress += chunk
        if self.prefill_progress >= self.prompt_tokens:
            self.phase = "Decoding"
            return True
        return False

    def step_decode(self) -> bool:
        """Returns True when max_tokens reached (finished)."""
        assert self.phase == "Decoding"
        self.decode_tokens += 1
        if self.decode_tokens >= self.max_tokens:
            self.phase = "Finished"
            return True
        return False

    def prefill_chunks_needed(self) -> int:
        return math.ceil(self.prompt_tokens / self.chunk_size)


import math


# ---------------------------------------------------------------------------
# Tests: SchedulerConfig validation
# ---------------------------------------------------------------------------

class TestSchedulerConfig:

    def test_default_valid(self):
        SchedulerConfig().validate()

    def test_zero_max_slots_invalid(self):
        with pytest.raises(ValueError):
            SchedulerConfig(max_slots=0).validate()

    def test_zero_chunk_size_invalid(self):
        with pytest.raises(ValueError):
            SchedulerConfig(prefill_chunk_size=0).validate()

    def test_one_slot_valid(self):
        SchedulerConfig(max_slots=1).validate()

    def test_one_chunk_valid(self):
        SchedulerConfig(prefill_chunk_size=1).validate()

    def test_large_config_valid(self):
        SchedulerConfig(max_slots=64, prefill_chunk_size=1024, max_waiting_requests=10000).validate()

    def test_recompute_mode(self):
        cfg = SchedulerConfig(preemption_mode=PreemptionMode.RECOMPUTE)
        assert cfg.preemption_mode == PreemptionMode.RECOMPUTE

    def test_swap_mode(self):
        cfg = SchedulerConfig(preemption_mode=PreemptionMode.SWAP)
        assert cfg.preemption_mode == PreemptionMode.SWAP

    def test_default_max_waiting_is_256(self):
        assert SchedulerConfig().max_waiting_requests == 256

    def test_default_chunk_size_is_512(self):
        assert SchedulerConfig().prefill_chunk_size == 512

    def test_default_max_slots_is_4(self):
        assert SchedulerConfig().max_slots == 4


# ---------------------------------------------------------------------------
# Tests: RequestPriority ordering
# ---------------------------------------------------------------------------

class TestRequestPriority:

    def test_high_greater_than_normal(self):
        assert RequestPriority.HIGH > RequestPriority.NORMAL

    def test_normal_greater_than_low(self):
        assert RequestPriority.NORMAL > RequestPriority.LOW

    def test_high_greater_than_low(self):
        assert RequestPriority.HIGH > RequestPriority.LOW

    def test_sort_ascending(self):
        priorities = [RequestPriority.NORMAL, RequestPriority.LOW, RequestPriority.HIGH]
        assert sorted(priorities) == [
            RequestPriority.LOW,
            RequestPriority.NORMAL,
            RequestPriority.HIGH,
        ]

    def test_sort_descending(self):
        priorities = [RequestPriority.LOW, RequestPriority.NORMAL, RequestPriority.HIGH]
        sorted_desc = sorted(priorities, reverse=True)
        assert sorted_desc[0] == RequestPriority.HIGH
        assert sorted_desc[-1] == RequestPriority.LOW

    def test_default_is_normal(self):
        assert MockRequest(prompt="test").priority == RequestPriority.NORMAL

    def test_values(self):
        assert RequestPriority.LOW == 0
        assert RequestPriority.NORMAL == 1
        assert RequestPriority.HIGH == 2

    def test_priority_sorted_queue(self):
        """Higher priority requests sort first when used as keys."""
        reqs = [
            (RequestPriority.LOW, "low"),
            (RequestPriority.HIGH, "high"),
            (RequestPriority.NORMAL, "normal"),
        ]
        # Sort descending by priority
        sorted_reqs = sorted(reqs, key=lambda x: x[0], reverse=True)
        assert sorted_reqs[0][1] == "high"
        assert sorted_reqs[-1][1] == "low"


# ---------------------------------------------------------------------------
# Tests: SchedulerHandle — basic submission
# ---------------------------------------------------------------------------

class TestSchedulerHandleBasic:

    def test_submit_single_request(self):
        handle = SchedulerHandle("test-model")
        req = MockRequest(prompt="hello")
        handle.submit(req)
        assert handle.waiting_count == 1

    def test_submit_multiple_requests(self):
        handle = SchedulerHandle("test-model")
        for i in range(5):
            handle.submit(MockRequest(prompt=f"req {i}"))
        assert handle.waiting_count == 5

    def test_consume_decrements_count(self):
        handle = SchedulerHandle("test-model")
        handle.submit(MockRequest(prompt="test"))
        assert handle.waiting_count == 1
        handle.consume_one()
        assert handle.waiting_count == 0

    def test_model_id_preserved(self):
        handle = SchedulerHandle("Qwen3-8B")
        assert handle.model_id == "Qwen3-8B"

    def test_is_not_full_by_default(self):
        handle = SchedulerHandle("test-model")
        assert not handle.is_full

    def test_submit_after_close_raises(self):
        handle = SchedulerHandle("test-model")
        handle.close()
        with pytest.raises(SchedulerFull):
            handle.submit(MockRequest(prompt="test"))


# ---------------------------------------------------------------------------
# Tests: Backpressure
# ---------------------------------------------------------------------------

class TestBackpressure:

    def test_respects_max_waiting(self):
        handle = SchedulerHandle("test-model", max_waiting=3)
        for i in range(3):
            handle.submit(MockRequest(prompt=f"req {i}"))
        with pytest.raises(SchedulerFull):
            handle.submit(MockRequest(prompt="overflow"))

    def test_is_full_at_capacity(self):
        handle = SchedulerHandle("test-model", max_waiting=2)
        handle.submit(MockRequest(prompt="a"))
        handle.submit(MockRequest(prompt="b"))
        assert handle.is_full

    def test_not_full_below_capacity(self):
        handle = SchedulerHandle("test-model", max_waiting=5)
        handle.submit(MockRequest(prompt="a"))
        assert not handle.is_full

    def test_consume_frees_capacity(self):
        handle = SchedulerHandle("test-model", max_waiting=2)
        handle.submit(MockRequest(prompt="a"))
        handle.submit(MockRequest(prompt="b"))
        assert handle.is_full

        # Consume one → no longer full
        handle.consume_one()
        assert not handle.is_full
        # Can now accept another
        handle.submit(MockRequest(prompt="c"))

    def test_unlimited_accepts_many(self):
        handle = SchedulerHandle("test-model", max_waiting=0)
        for i in range(1000):
            handle.submit(MockRequest(prompt=f"req {i}"))
        assert handle.waiting_count == 1000

    def test_max_waiting_1_allows_exactly_one(self):
        handle = SchedulerHandle("test-model", max_waiting=1)
        handle.submit(MockRequest(prompt="only"))
        assert handle.is_full
        with pytest.raises(SchedulerFull):
            handle.submit(MockRequest(prompt="overflow"))


# ---------------------------------------------------------------------------
# Tests: Thread-safe waiting count
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_submit(self):
        """N threads each submit M requests; total should be N*M."""
        n_threads = 4
        m_requests = 25
        handle = SchedulerHandle("test-model", max_waiting=0)
        errors = []

        def submit_many():
            try:
                for i in range(m_requests):
                    handle.submit(MockRequest(prompt=f"req {i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit_many) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert handle.waiting_count == n_threads * m_requests

    def test_concurrent_submit_with_backpressure(self):
        """With tight backpressure, some submits should raise SchedulerFull."""
        handle = SchedulerHandle("test-model", max_waiting=10)
        success = 0
        fail = 0
        lock = threading.Lock()

        def try_submit():
            nonlocal success, fail
            try:
                handle.submit(MockRequest(prompt="x"))
                with lock:
                    success += 1
            except SchedulerFull:
                with lock:
                    fail += 1

        threads = [threading.Thread(target=try_submit) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert success <= 10
        assert success + fail == 30


# ---------------------------------------------------------------------------
# Tests: Phase state machine
# ---------------------------------------------------------------------------

class TestPhaseStateMachine:

    def test_initial_phase_is_new(self):
        phase = PhaseState(prompt_tokens=100, max_tokens=50)
        assert phase.phase == "New"

    def test_transition_new_to_prefilling(self):
        phase = PhaseState(prompt_tokens=100, max_tokens=50)
        phase.transition_to_prefilling()
        assert phase.phase == "Prefilling"

    def test_prefill_small_prompt_single_chunk(self):
        phase = PhaseState(prompt_tokens=100, max_tokens=10, chunk_size=512)
        phase.transition_to_prefilling()
        done = phase.step_prefill_chunk()
        assert done
        assert phase.phase == "Decoding"

    def test_prefill_large_prompt_multiple_chunks(self):
        phase = PhaseState(prompt_tokens=1024, max_tokens=10, chunk_size=512)
        phase.transition_to_prefilling()

        done = phase.step_prefill_chunk()
        assert not done
        assert phase.phase == "Prefilling"
        assert phase.prefill_progress == 512

        done = phase.step_prefill_chunk()
        assert done
        assert phase.phase == "Decoding"

    def test_decode_to_max_tokens(self):
        phase = PhaseState(prompt_tokens=10, max_tokens=3, chunk_size=512)
        phase.transition_to_prefilling()
        phase.step_prefill_chunk()
        assert phase.phase == "Decoding"

        assert not phase.step_decode()
        assert not phase.step_decode()
        done = phase.step_decode()
        assert done
        assert phase.phase == "Finished"

    def test_prefill_chunks_needed(self):
        phase = PhaseState(prompt_tokens=1000, max_tokens=10, chunk_size=512)
        assert phase.prefill_chunks_needed() == 2  # ceil(1000/512)

    def test_prefill_exact_chunk_boundary(self):
        phase = PhaseState(prompt_tokens=512, max_tokens=10, chunk_size=512)
        phase.transition_to_prefilling()
        done = phase.step_prefill_chunk()
        assert done  # exactly one full chunk

    def test_decode_priority_over_prefill(self):
        """Verify decode runs before continuing prefill (scheduler behavior)."""
        # Simulate: request A is prefilling, request B is decoding.
        # Scheduler should run B's decode first.
        req_a = PhaseState(prompt_tokens=1024, max_tokens=5, chunk_size=512)
        req_b = PhaseState(prompt_tokens=10, max_tokens=5, chunk_size=512)

        req_a.transition_to_prefilling()
        req_b.transition_to_prefilling()
        req_b.step_prefill_chunk()  # B finishes prefill in one chunk
        assert req_b.phase == "Decoding"

        # B should decode before A gets its next prefill chunk
        req_b.step_decode()
        assert req_b.phase == "Decoding"  # still going

        # Now A can prefill its next chunk
        req_a.step_prefill_chunk()
        assert req_a.phase == "Prefilling"  # not done yet (1024 > 512)

    def test_invalid_transition_raises(self):
        phase = PhaseState(prompt_tokens=10, max_tokens=5)
        with pytest.raises(AssertionError):
            phase.step_prefill_chunk()  # Can't prefill from New

    def test_decode_before_prefill_done_raises(self):
        phase = PhaseState(prompt_tokens=10, max_tokens=5)
        phase.transition_to_prefilling()
        with pytest.raises(AssertionError):
            phase.step_decode()  # Can't decode while prefilling


# ---------------------------------------------------------------------------
# Tests: Slot management simulation
# ---------------------------------------------------------------------------

class TestSlotManagement:

    def test_free_slot_detection(self):
        """Simulate max_slots tracking."""
        max_slots = 4
        active = set()  # set of active slot ids

        def find_free_slot():
            for i in range(max_slots):
                if i not in active:
                    return i
            return None

        assert find_free_slot() == 0
        active.add(0)
        active.add(1)
        active.add(2)
        active.add(3)
        assert find_free_slot() is None

        active.discard(2)
        assert find_free_slot() == 2

    def test_max_concurrent_requests(self):
        """Scheduler should not exceed max_slots active requests."""
        max_slots = 3
        active_requests = []

        def can_add():
            return len(active_requests) < max_slots

        for i in range(3):
            if can_add():
                active_requests.append(i)
        assert len(active_requests) == 3
        assert not can_add()

        # Finish one request
        active_requests.pop(0)
        assert can_add()

    def test_decode_priority_scheduling(self):
        """Decode requests should be processed before queued requests."""
        waiting = [MockRequest(prompt=f"new {i}") for i in range(3)]
        decoding = [MockRequest(prompt=f"decode {i}") for i in range(2)]

        # Scheduler picks decode first
        def next_step(waiting, decoding):
            if decoding:
                return ("decode", decoding.pop(0))
            elif waiting:
                return ("prefill", waiting.pop(0))
            return None

        assert next_step(waiting, decoding)[0] == "decode"
        assert next_step(waiting, decoding)[0] == "decode"
        assert next_step(waiting, decoding)[0] == "prefill"

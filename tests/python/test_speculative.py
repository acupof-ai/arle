"""
Speculative decoding tests — no GPU required.

Mirrors and extends infer/src/speculative.rs in pure Python.
Tests the acceptance/rejection sampling algorithm, AcceptanceTracker,
and expected_speedup formula.

Run:
    pytest tests/python/test_speculative.py -v
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Python mirrors of speculative.rs types
# ---------------------------------------------------------------------------

@dataclass
class SpecConfig:
    num_speculative_tokens: int = 5
    vocab_size: int = 32000
    min_acceptance_rate: float = 0.0

    def validate(self):
        if self.num_speculative_tokens == 0:
            raise ValueError("num_speculative_tokens must be >= 1")
        if self.vocab_size == 0:
            raise ValueError("vocab_size must be >= 1")
        if not (0.0 <= self.min_acceptance_rate <= 1.0):
            raise ValueError("min_acceptance_rate must be in [0, 1]")


@dataclass
class TokenProposal:
    tokens: list[int]
    draft_probs: list[float]
    target_probs: list[float]
    target_bonus_dist: list[float] = field(default_factory=list)

    def __len__(self):
        return len(self.tokens)

    def is_empty(self):
        return len(self.tokens) == 0

    def validate(self):
        k = len(self.tokens)
        if len(self.draft_probs) != k:
            raise ValueError(f"draft_probs length {len(self.draft_probs)} != tokens length {k}")
        if len(self.target_probs) != k:
            raise ValueError(f"target_probs length {len(self.target_probs)} != tokens length {k}")
        for i, (p, q) in enumerate(zip(self.target_probs, self.draft_probs)):
            if not (0.0 <= p <= 1.0 + 1e-5):
                raise ValueError(f"target_probs[{i}] = {p} out of [0, 1]")
            if not (0.0 <= q <= 1.0 + 1e-5):
                raise ValueError(f"draft_probs[{i}] = {q} out of [0, 1]")


@dataclass
class VerificationResult:
    accepted: list[int]
    bonus_token: Optional[int]
    num_accepted: int
    rejection_index: int

    def total_tokens(self) -> int:
        return self.num_accepted + (1 if self.bonus_token is not None else 0)

    def acceptance_rate(self, k: int) -> float:
        if k == 0:
            return 1.0
        return self.num_accepted / k


def sample_from_dist(dist: list[float], rng: random.Random) -> Optional[int]:
    if not dist:
        return None
    total = sum(dist)
    if total <= 0.0:
        return None
    threshold = rng.random() * total
    cumulative = 0.0
    for i, p in enumerate(dist):
        cumulative += p
        if cumulative >= threshold:
            return i
    return len(dist) - 1


def verify_tokens(proposal: TokenProposal, rng: random.Random) -> VerificationResult:
    """Pure Python port of speculative.rs::verify_tokens."""
    k = len(proposal.tokens)
    accepted = []
    rejection_index = k  # default: all accepted

    for i in range(k):
        p = max(proposal.target_probs[i], 0.0)
        q = max(proposal.draft_probs[i], 1e-45)  # float MIN_POSITIVE approx
        accept_prob = min(p / q, 1.0)

        if rng.random() < accept_prob:
            accepted.append(proposal.tokens[i])
        else:
            rejection_index = i
            break

    # Bonus token
    if rejection_index == k:
        bonus_token = sample_from_dist(proposal.target_bonus_dist, rng) \
            if proposal.target_bonus_dist else None
    else:
        bonus_token = None

    return VerificationResult(
        accepted=accepted,
        bonus_token=bonus_token,
        num_accepted=len(accepted),
        rejection_index=rejection_index,
    )


class AcceptanceTracker:
    """Rolling window acceptance rate tracker."""

    def __init__(self, window_size: int):
        self.window_size = max(window_size, 1)
        self.history: deque[float] = deque()

    def record(self, rate: float):
        if len(self.history) >= self.window_size:
            self.history.popleft()
        self.history.append(rate)

    def mean(self) -> float:
        if not self.history:
            return 1.0  # optimistic start
        return sum(self.history) / len(self.history)

    def should_disable(self, min_rate: float) -> bool:
        return len(self.history) >= self.window_size and self.mean() < min_rate


def expected_speedup(k: int, alpha: float) -> float:
    """Theoretical throughput multiplier from Chen et al. 2023."""
    if alpha <= 0.0:
        return 1.0
    if alpha >= 1.0:
        return float(k + 1)
    alpha_k = alpha ** k
    return (1.0 - alpha_k) / ((1.0 - alpha) * k) * k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seeded_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def all_accept_proposal(k: int) -> TokenProposal:
    """target_prob == draft_prob → accept_prob == 1 → always accepted."""
    return TokenProposal(
        tokens=list(range(k)),
        draft_probs=[0.5] * k,
        target_probs=[0.5] * k,
    )


def all_reject_proposal(k: int) -> TokenProposal:
    """target_prob == 0 → accept_prob == 0 → always rejected."""
    return TokenProposal(
        tokens=[0] * k,
        draft_probs=[0.5] * k,
        target_probs=[0.0] * k,
    )


# ---------------------------------------------------------------------------
# Tests: verify_tokens — basic acceptance/rejection
# ---------------------------------------------------------------------------

class TestVerifyTokens:

    def test_all_accepted_when_target_eq_draft(self):
        proposal = all_accept_proposal(5)
        rng = seeded_rng()
        result = verify_tokens(proposal, rng)
        assert result.num_accepted == 5
        assert result.rejection_index == 5
        assert result.accepted == [0, 1, 2, 3, 4]

    def test_none_accepted_when_target_zero(self):
        proposal = all_reject_proposal(5)
        rng = seeded_rng()
        result = verify_tokens(proposal, rng)
        assert result.num_accepted == 0
        assert result.rejection_index == 0
        assert result.accepted == []

    def test_partial_acceptance(self):
        # Tokens 0,1,2 always accepted (p==q), token 3 always rejected (p==0).
        proposal = TokenProposal(
            tokens=[10, 11, 12, 13],
            draft_probs=[0.5, 0.5, 0.5, 0.5],
            target_probs=[0.5, 0.5, 0.5, 0.0],
        )
        rng = seeded_rng()
        result = verify_tokens(proposal, rng)
        assert result.num_accepted == 3
        assert result.rejection_index == 3
        assert result.accepted == [10, 11, 12]

    def test_bonus_token_sampled_when_all_accepted(self):
        proposal = all_accept_proposal(3)
        proposal.target_bonus_dist = [0.0, 0.0, 1.0, 0.0]  # always token 2
        rng = seeded_rng()
        result = verify_tokens(proposal, rng)
        assert result.num_accepted == 3
        assert result.bonus_token == 2
        assert result.total_tokens() == 4

    def test_no_bonus_when_rejected(self):
        proposal = all_reject_proposal(3)
        proposal.target_bonus_dist = [0.0, 1.0, 0.0]
        rng = seeded_rng()
        result = verify_tokens(proposal, rng)
        assert result.bonus_token is None

    def test_no_bonus_dist_means_none(self):
        proposal = all_accept_proposal(3)
        # target_bonus_dist is empty by default
        rng = seeded_rng()
        result = verify_tokens(proposal, rng)
        assert result.bonus_token is None

    def test_total_tokens_with_bonus(self):
        result = VerificationResult(accepted=[1, 2], bonus_token=5, num_accepted=2, rejection_index=2)
        assert result.total_tokens() == 3

    def test_total_tokens_without_bonus(self):
        result = VerificationResult(accepted=[1, 2], bonus_token=None, num_accepted=2, rejection_index=2)
        assert result.total_tokens() == 2

    def test_empty_proposal(self):
        proposal = TokenProposal(tokens=[], draft_probs=[], target_probs=[])
        rng = seeded_rng()
        result = verify_tokens(proposal, rng)
        assert result.num_accepted == 0
        assert result.accepted == []
        assert result.rejection_index == 0

    def test_k_equals_1_accept(self):
        proposal = TokenProposal(tokens=[99], draft_probs=[0.3], target_probs=[0.3])
        rng = seeded_rng()
        result = verify_tokens(proposal, rng)
        assert result.num_accepted == 1

    def test_k_equals_1_reject(self):
        proposal = TokenProposal(tokens=[99], draft_probs=[0.5], target_probs=[0.0])
        rng = seeded_rng()
        result = verify_tokens(proposal, rng)
        assert result.num_accepted == 0
        assert result.rejection_index == 0


# ---------------------------------------------------------------------------
# Tests: verify_tokens — probabilistic properties
# ---------------------------------------------------------------------------

class TestVerifyTokensProbabilistic:

    def test_high_target_high_acceptance(self):
        """When p >> q, acceptance should be near 100%."""
        proposal = TokenProposal(
            tokens=[0] * 5,
            draft_probs=[0.1] * 5,
            target_probs=[1.0] * 5,
        )
        rng = seeded_rng(0)
        total_accepted = 0
        trials = 200
        for _ in range(trials):
            r = verify_tokens(proposal, rng)
            total_accepted += r.num_accepted
        # Accept prob = min(1.0/0.1, 1) = 1.0 → all 5 always accepted
        assert total_accepted == trials * 5

    def test_low_target_low_acceptance(self):
        """When p << q, acceptance should be low."""
        # accept_prob = min(0.01/0.5, 1) = 0.02
        proposal = TokenProposal(
            tokens=[0] * 5,
            draft_probs=[0.5] * 5,
            target_probs=[0.01] * 5,
        )
        rng = seeded_rng(123)
        total_accepted = 0
        trials = 500
        for _ in range(trials):
            r = verify_tokens(proposal, rng)
            total_accepted += r.num_accepted
        avg = total_accepted / trials
        # Expected: first token accepted ~2% of the time, so avg << 1
        assert avg < 0.5, f"Expected low acceptance, got avg={avg}"

    def test_equal_probs_all_accepted(self):
        """p == q → accept_prob == 1 → deterministically accept all."""
        proposal = TokenProposal(
            tokens=[7, 8, 9],
            draft_probs=[0.3, 0.6, 0.9],
            target_probs=[0.3, 0.6, 0.9],
        )
        rng = seeded_rng()
        for _ in range(50):
            result = verify_tokens(proposal, rng)
            assert result.num_accepted == 3

    def test_bonus_token_distribution(self):
        """Bonus dist [0.25, 0.25, 0.25, 0.25] → roughly uniform sampling."""
        dist = [0.25, 0.25, 0.25, 0.25]
        rng = seeded_rng(42)
        counts = [0, 0, 0, 0]
        for _ in range(1000):
            t = sample_from_dist(dist, rng)
            counts[t] += 1
        for c in counts:
            assert 150 < c < 350, f"count {c} outside expected range"


# ---------------------------------------------------------------------------
# Tests: VerificationResult.acceptance_rate()
# ---------------------------------------------------------------------------

class TestAcceptanceRate:

    def test_partial_acceptance(self):
        result = VerificationResult(accepted=[1, 2, 3], bonus_token=None, num_accepted=3, rejection_index=3)
        assert abs(result.acceptance_rate(5) - 0.6) < 1e-5

    def test_full_acceptance(self):
        result = VerificationResult(accepted=[1, 2, 3], bonus_token=None, num_accepted=3, rejection_index=3)
        assert abs(result.acceptance_rate(3) - 1.0) < 1e-5

    def test_k_zero_returns_one(self):
        result = VerificationResult(accepted=[], bonus_token=None, num_accepted=0, rejection_index=0)
        assert abs(result.acceptance_rate(0) - 1.0) < 1e-5

    def test_zero_acceptance(self):
        result = VerificationResult(accepted=[], bonus_token=None, num_accepted=0, rejection_index=0)
        assert abs(result.acceptance_rate(5) - 0.0) < 1e-5


# ---------------------------------------------------------------------------
# Tests: TokenProposal.validate()
# ---------------------------------------------------------------------------

class TestTokenProposalValidate:

    def test_valid_proposal(self):
        all_accept_proposal(3).validate()  # should not raise

    def test_draft_probs_length_mismatch(self):
        p = all_accept_proposal(3)
        p.draft_probs.append(0.5)  # length 4 ≠ 3
        with pytest.raises(ValueError):
            p.validate()

    def test_target_probs_length_mismatch(self):
        p = all_accept_proposal(3)
        p.target_probs.pop()  # length 2 ≠ 3
        with pytest.raises(ValueError):
            p.validate()

    def test_target_prob_out_of_range(self):
        p = TokenProposal(tokens=[0], draft_probs=[0.5], target_probs=[1.5])
        with pytest.raises(ValueError):
            p.validate()

    def test_draft_prob_negative(self):
        p = TokenProposal(tokens=[0], draft_probs=[-0.1], target_probs=[0.5])
        with pytest.raises(ValueError):
            p.validate()

    def test_boundary_probs_valid(self):
        p = TokenProposal(tokens=[0, 1], draft_probs=[0.0, 1.0], target_probs=[1.0, 0.0])
        p.validate()  # should not raise

    def test_is_empty(self):
        p = TokenProposal(tokens=[], draft_probs=[], target_probs=[])
        assert p.is_empty()

    def test_not_empty(self):
        assert not all_accept_proposal(1).is_empty()


# ---------------------------------------------------------------------------
# Tests: SpecConfig.validate()
# ---------------------------------------------------------------------------

class TestSpecConfig:

    def test_default_valid(self):
        SpecConfig().validate()

    def test_zero_speculative_tokens(self):
        with pytest.raises(ValueError):
            SpecConfig(num_speculative_tokens=0).validate()

    def test_zero_vocab_size(self):
        with pytest.raises(ValueError):
            SpecConfig(vocab_size=0).validate()

    def test_negative_acceptance_rate(self):
        with pytest.raises(ValueError):
            SpecConfig(min_acceptance_rate=-0.1).validate()

    def test_acceptance_rate_above_1(self):
        with pytest.raises(ValueError):
            SpecConfig(min_acceptance_rate=1.5).validate()

    def test_acceptance_rate_boundaries_valid(self):
        SpecConfig(min_acceptance_rate=0.0).validate()
        SpecConfig(min_acceptance_rate=1.0).validate()

    def test_large_k_valid(self):
        SpecConfig(num_speculative_tokens=100).validate()


# ---------------------------------------------------------------------------
# Tests: AcceptanceTracker
# ---------------------------------------------------------------------------

class TestAcceptanceTracker:

    def test_starts_optimistic(self):
        tracker = AcceptanceTracker(10)
        assert abs(tracker.mean() - 1.0) < 1e-5
        assert not tracker.should_disable(0.5)

    def test_mean_correct(self):
        tracker = AcceptanceTracker(4)
        for rate in [1.0, 0.5, 0.5, 0.0]:
            tracker.record(rate)
        assert abs(tracker.mean() - 0.5) < 1e-5

    def test_window_evicts_old(self):
        tracker = AcceptanceTracker(2)
        tracker.record(0.0)
        tracker.record(0.0)
        tracker.record(1.0)  # evicts first 0.0
        tracker.record(1.0)  # evicts second 0.0
        assert abs(tracker.mean() - 1.0) < 1e-5

    def test_disable_when_below_threshold(self):
        tracker = AcceptanceTracker(3)
        for _ in range(3):
            tracker.record(0.1)
        assert tracker.should_disable(0.5)   # 0.1 < 0.5
        assert not tracker.should_disable(0.0)  # 0.1 >= 0.0

    def test_not_disabled_before_window_full(self):
        tracker = AcceptanceTracker(3)
        tracker.record(0.0)
        tracker.record(0.0)
        # Only 2 records, window requires 3 → not disabled yet
        assert not tracker.should_disable(0.5)

    def test_window_size_1(self):
        tracker = AcceptanceTracker(1)
        tracker.record(0.3)
        assert abs(tracker.mean() - 0.3) < 1e-5
        assert tracker.should_disable(0.5)

    def test_record_many_values(self):
        tracker = AcceptanceTracker(5)
        rates = [1.0, 0.8, 0.6, 0.4, 0.2]
        for r in rates:
            tracker.record(r)
        expected = sum(rates) / len(rates)
        assert abs(tracker.mean() - expected) < 1e-5

    def test_window_maintains_size(self):
        tracker = AcceptanceTracker(3)
        for i in range(10):
            tracker.record(float(i) / 10)
        # Only last 3 values should be in window: 0.7, 0.8, 0.9
        assert abs(tracker.mean() - 0.8) < 1e-5


# ---------------------------------------------------------------------------
# Tests: expected_speedup()
# ---------------------------------------------------------------------------

class TestExpectedSpeedup:

    def test_alpha_zero_returns_one(self):
        assert abs(expected_speedup(5, 0.0) - 1.0) < 1e-5

    def test_alpha_one_returns_k_plus_one(self):
        k = 5
        s = expected_speedup(k, 1.0)
        assert abs(s - (k + 1)) < 0.1

    def test_speedup_increases_with_alpha(self):
        k = 5
        s_low = expected_speedup(k, 0.3)
        s_mid = expected_speedup(k, 0.6)
        s_high = expected_speedup(k, 0.9)
        assert s_low < s_mid < s_high

    def test_speedup_increases_with_k(self):
        alpha = 0.8
        s_small = expected_speedup(3, alpha)
        s_large = expected_speedup(10, alpha)
        assert s_small < s_large

    def test_negative_alpha_returns_one(self):
        assert abs(expected_speedup(5, -0.1) - 1.0) < 1e-5

    def test_k_equals_1(self):
        # k=1, alpha=0.5: (1-0.5)/(0.5*1)*1 = 1.0
        s = expected_speedup(1, 0.5)
        assert s > 0.0

    def test_result_always_positive(self):
        for k in [1, 3, 5, 10]:
            for alpha in [0.0, 0.1, 0.5, 0.9, 1.0]:
                assert expected_speedup(k, alpha) > 0.0


# ---------------------------------------------------------------------------
# Tests: sample_from_dist()
# ---------------------------------------------------------------------------

class TestSampleFromDist:

    def test_empty_dist_returns_none(self):
        rng = seeded_rng()
        assert sample_from_dist([], rng) is None

    def test_all_zero_returns_none(self):
        rng = seeded_rng()
        assert sample_from_dist([0.0, 0.0, 0.0], rng) is None

    def test_peaked_dist_always_same_token(self):
        dist = [0.0, 0.0, 1.0, 0.0]  # always token 2
        rng = seeded_rng()
        for _ in range(20):
            assert sample_from_dist(dist, rng) == 2

    def test_uniform_dist_all_tokens_sampled(self):
        dist = [1.0] * 4
        rng = seeded_rng(42)
        seen = set()
        for _ in range(200):
            t = sample_from_dist(dist, rng)
            seen.add(t)
        assert seen == {0, 1, 2, 3}

    def test_single_token_dist(self):
        dist = [0.5]  # only token 0
        rng = seeded_rng()
        for _ in range(10):
            assert sample_from_dist(dist, rng) == 0

    def test_unnormalized_dist_works(self):
        # Unnormalized: [10, 0, 0] → always token 0
        dist = [10.0, 0.0, 0.0]
        rng = seeded_rng()
        for _ in range(10):
            assert sample_from_dist(dist, rng) == 0

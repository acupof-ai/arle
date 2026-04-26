"""
Sampling parameter tests — no GPU required.

Tests the sampling math and parameter validation logic that mirrors
infer/src/sampler.rs. All tests are pure Python.

Run:
    pytest tests/python/test_sampling.py -v
"""

import math
import pytest


# ---------------------------------------------------------------------------
# Python mirror of SamplingParams (mirrors sampler.rs logic exactly)
# ---------------------------------------------------------------------------

class SamplingParams:
    """CPU-side mirror of infer::sampler::SamplingParams."""

    def __init__(
        self,
        temperature: float = 0.0,
        top_k: int = -1,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        ignore_eos: bool = False,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.ignore_eos = ignore_eos
        self.stop_token_ids = stop_token_ids or []
        self.seed = seed

    def is_greedy(self) -> bool:
        """True when deterministic greedy decoding should be used.

        Matches sampler.rs: temperature <= 0 OR top_k == 1,
        AND top_p >= 1, AND min_p <= 0.
        """
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p >= 1.0 and self.min_p <= 0.0

    def has_penalties(self) -> bool:
        return (
            self.repetition_penalty != 1.0
            or self.frequency_penalty != 0.0
            or self.presence_penalty != 0.0
        )

    def apply_penalties(self, logits: list[float], token_counts: list[int]) -> list[float]:
        """Apply repetition, frequency, and presence penalties in-place.

        Mirrors sampler.rs::SamplingParams::apply_penalties exactly.
        """
        if not self.has_penalties():
            return logits

        result = list(logits)
        rep = self.repetition_penalty
        freq = self.frequency_penalty
        pres = self.presence_penalty

        for token_id, count in enumerate(token_counts):
            if count == 0:
                continue
            if token_id >= len(result):
                break

            logit = result[token_id]

            # Repetition penalty: divide if positive, multiply if negative.
            if rep != 1.0:
                penalized = logit / rep if logit >= 0.0 else logit * rep
            else:
                penalized = logit

            # Frequency + presence (additive).
            penalized -= freq * count
            if count > 0:
                penalized -= pres

            result[token_id] = penalized

        return result


def sampling_params_from_request(
    temperature=None,
    top_p=None,
    top_k=None,
    min_p=None,
    repetition_penalty=None,
    frequency_penalty=None,
    presence_penalty=None,
    ignore_eos=None,
    seed=None,
    stop_token_ids=None,
) -> SamplingParams:
    """Mirror of sampler.rs::sampling_params_from_request."""
    return SamplingParams(
        temperature=temperature if temperature is not None else 0.0,
        top_k=top_k if top_k is not None else -1,
        top_p=top_p if top_p is not None else 1.0,
        min_p=min_p if min_p is not None else 0.0,
        repetition_penalty=repetition_penalty if repetition_penalty is not None else 1.0,
        frequency_penalty=frequency_penalty if frequency_penalty is not None else 0.0,
        presence_penalty=presence_penalty if presence_penalty is not None else 0.0,
        ignore_eos=ignore_eos if ignore_eos is not None else False,
        seed=seed,
        stop_token_ids=stop_token_ids or [],
    )


# ---------------------------------------------------------------------------
# Tests: is_greedy()
# ---------------------------------------------------------------------------

class TestIsGreedy:

    def test_defaults_are_greedy(self):
        assert SamplingParams().is_greedy()

    def test_zero_temperature_greedy(self):
        p = SamplingParams(temperature=0.0)
        assert p.is_greedy()

    def test_negative_temperature_greedy(self):
        # Negative temperature is treated as ≤ 0 → greedy.
        p = SamplingParams(temperature=-1.0)
        assert p.is_greedy()

    def test_top_k_1_overrides_temperature(self):
        p = SamplingParams(temperature=0.7, top_k=1)
        assert p.is_greedy()

    def test_positive_temperature_not_greedy(self):
        p = SamplingParams(temperature=0.7, top_k=-1)
        assert not p.is_greedy()

    def test_min_p_disables_greedy(self):
        p = SamplingParams(temperature=0.0, min_p=0.05)
        assert not p.is_greedy()

    def test_top_p_lt_1_disables_greedy(self):
        p = SamplingParams(temperature=0.0, top_p=0.9)
        assert not p.is_greedy()

    def test_top_p_exactly_1_keeps_greedy(self):
        p = SamplingParams(temperature=0.0, top_p=1.0)
        assert p.is_greedy()


# ---------------------------------------------------------------------------
# Tests: has_penalties()
# ---------------------------------------------------------------------------

class TestHasPenalties:

    def test_no_penalty_by_default(self):
        assert not SamplingParams().has_penalties()

    def test_repetition_penalty_detected(self):
        assert SamplingParams(repetition_penalty=1.1).has_penalties()

    def test_repetition_penalty_below_1_detected(self):
        assert SamplingParams(repetition_penalty=0.9).has_penalties()

    def test_repetition_penalty_exactly_1_no_penalty(self):
        assert not SamplingParams(repetition_penalty=1.0).has_penalties()

    def test_frequency_penalty_detected(self):
        assert SamplingParams(frequency_penalty=0.5).has_penalties()

    def test_presence_penalty_detected(self):
        assert SamplingParams(presence_penalty=0.3).has_penalties()

    def test_all_penalties_combined(self):
        p = SamplingParams(repetition_penalty=1.2, frequency_penalty=0.5, presence_penalty=0.3)
        assert p.has_penalties()


# ---------------------------------------------------------------------------
# Tests: apply_penalties() — repetition penalty
# ---------------------------------------------------------------------------

class TestRepetitionPenalty:

    def test_positive_logit_divided(self):
        logits = [1.0, 2.0, 3.0]
        counts = [1, 0, 1]
        p = SamplingParams(repetition_penalty=2.0)
        result = p.apply_penalties(logits, counts)
        assert abs(result[0] - 0.5) < 1e-6   # 1.0 / 2.0
        assert abs(result[1] - 2.0) < 1e-6   # unchanged (count=0)
        assert abs(result[2] - 1.5) < 1e-6   # 3.0 / 2.0

    def test_negative_logit_multiplied(self):
        logits = [-2.0]
        counts = [1]
        p = SamplingParams(repetition_penalty=2.0)
        result = p.apply_penalties(logits, counts)
        assert abs(result[0] - (-4.0)) < 1e-6  # -2.0 * 2.0

    def test_zero_logit_unchanged(self):
        logits = [0.0]
        counts = [1]
        p = SamplingParams(repetition_penalty=1.5)
        result = p.apply_penalties(logits, counts)
        # 0.0 >= 0.0 → divide: 0.0 / 1.5 = 0.0
        assert abs(result[0]) < 1e-6

    def test_penalty_1_no_change(self):
        logits = [2.0, -1.5]
        counts = [3, 2]
        p = SamplingParams(repetition_penalty=1.0)
        result = p.apply_penalties(logits, counts)
        assert result == logits

    def test_count_zero_skips_token(self):
        logits = [3.0]
        counts = [0]
        p = SamplingParams(repetition_penalty=1.5)
        result = p.apply_penalties(logits, counts)
        assert abs(result[0] - 3.0) < 1e-6  # count=0 → no penalty

    def test_token_id_beyond_logits_safe(self):
        logits = [1.0, 2.0]
        counts = [1, 1, 1, 1]  # more entries than logits
        p = SamplingParams(repetition_penalty=2.0)
        # Should not raise — silently ignores out-of-bounds token ids.
        result = p.apply_penalties(logits, counts)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: apply_penalties() — frequency penalty
# ---------------------------------------------------------------------------

class TestFrequencyPenalty:

    def test_scales_with_count(self):
        logits = [5.0, 5.0, 5.0]
        counts = [3, 1, 0]
        p = SamplingParams(frequency_penalty=1.0)
        result = p.apply_penalties(logits, counts)
        assert abs(result[0] - 2.0) < 1e-6  # 5 - 1*3
        assert abs(result[1] - 4.0) < 1e-6  # 5 - 1*1
        assert abs(result[2] - 5.0) < 1e-6  # unchanged

    def test_negative_penalty_rewards_repetition(self):
        logits = [3.0]
        counts = [2]
        p = SamplingParams(frequency_penalty=-0.5)
        result = p.apply_penalties(logits, counts)
        # 3.0 - (-0.5)*2 = 3.0 + 1.0 = 4.0
        assert abs(result[0] - 4.0) < 1e-6

    def test_high_count_can_go_negative(self):
        logits = [2.0]
        counts = [10]
        p = SamplingParams(frequency_penalty=1.0)
        result = p.apply_penalties(logits, counts)
        # 2.0 - 1.0*10 = -8.0
        assert abs(result[0] - (-8.0)) < 1e-6


# ---------------------------------------------------------------------------
# Tests: apply_penalties() — presence penalty
# ---------------------------------------------------------------------------

class TestPresencePenalty:

    def test_flat_penalty_for_seen_tokens(self):
        logits = [5.0, 5.0, 5.0]
        counts = [2, 1, 0]
        p = SamplingParams(presence_penalty=0.5)
        result = p.apply_penalties(logits, counts)
        assert abs(result[0] - 4.5) < 1e-6  # 5.0 - 0.5
        assert abs(result[1] - 4.5) < 1e-6  # 5.0 - 0.5
        assert abs(result[2] - 5.0) < 1e-6  # untouched

    def test_presence_independent_of_count(self):
        # presence_penalty should be the same regardless of count (1 vs 10)
        logits = [5.0, 5.0]
        counts = [1, 10]
        p = SamplingParams(presence_penalty=1.0)
        result = p.apply_penalties(logits, counts)
        assert abs(result[0] - 4.0) < 1e-6
        assert abs(result[1] - 4.0) < 1e-6


# ---------------------------------------------------------------------------
# Tests: apply_penalties() — combined
# ---------------------------------------------------------------------------

class TestCombinedPenalties:

    def test_all_penalties_combined(self):
        # token 0: count=2, logit=4.0
        # rep=1.5 → 4.0/1.5 ≈ 2.667
        # freq=0.5 → -0.5*2 = -1.0  → 1.667
        # pres=0.2 → -0.2           → 1.467
        logits = [4.0]
        counts = [2]
        p = SamplingParams(repetition_penalty=1.5, frequency_penalty=0.5, presence_penalty=0.2)
        result = p.apply_penalties(logits, counts)
        expected = (4.0 / 1.5) - (0.5 * 2) - 0.2
        assert abs(result[0] - expected) < 1e-5

    def test_no_penalty_skips_all(self):
        logits = [1.0, 2.0, 3.0]
        counts = [5, 5, 5]
        p = SamplingParams()  # all default → no penalties
        result = p.apply_penalties(logits, counts)
        assert result == logits


# ---------------------------------------------------------------------------
# Tests: sampling_params_from_request()
# ---------------------------------------------------------------------------

class TestSamplingParamsFromRequest:

    def test_all_none_uses_defaults(self):
        p = sampling_params_from_request()
        assert p.temperature == 0.0
        assert p.top_k == -1
        assert p.top_p == 1.0
        assert p.min_p == 0.0
        assert p.repetition_penalty == 1.0
        assert p.frequency_penalty == 0.0
        assert p.presence_penalty == 0.0
        assert not p.ignore_eos
        assert p.seed is None
        assert p.stop_token_ids == []

    def test_all_explicit(self):
        p = sampling_params_from_request(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            min_p=0.05,
            repetition_penalty=1.1,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            ignore_eos=True,
            seed=42,
            stop_token_ids=[1, 2, 3],
        )
        assert abs(p.temperature - 0.7) < 1e-6
        assert abs(p.top_p - 0.9) < 1e-6
        assert p.top_k == 50
        assert abs(p.min_p - 0.05) < 1e-6
        assert abs(p.repetition_penalty - 1.1) < 1e-6
        assert abs(p.frequency_penalty - 0.2) < 1e-6
        assert abs(p.presence_penalty - 0.1) < 1e-6
        assert p.ignore_eos is True
        assert p.seed == 42
        assert p.stop_token_ids == [1, 2, 3]

    def test_top_k_none_becomes_minus_one(self):
        p = sampling_params_from_request(top_k=None)
        assert p.top_k == -1

    def test_seed_preserved_as_none(self):
        p = sampling_params_from_request(seed=None)
        assert p.seed is None

    def test_seed_zero_preserved(self):
        p = sampling_params_from_request(seed=0)
        assert p.seed == 0

    def test_stop_token_ids_none_becomes_empty(self):
        p = sampling_params_from_request(stop_token_ids=None)
        assert p.stop_token_ids == []

    def test_stop_token_ids_empty_list(self):
        p = sampling_params_from_request(stop_token_ids=[])
        assert p.stop_token_ids == []


# ---------------------------------------------------------------------------
# Tests: top_p / top_k / min_p filter semantics
# ---------------------------------------------------------------------------

class TestFilterSemantics:
    """Test that filter params work together correctly with is_greedy()."""

    def test_top_p_09_makes_stochastic(self):
        p = SamplingParams(temperature=0.0, top_p=0.9)
        assert not p.is_greedy()

    def test_min_p_001_makes_stochastic(self):
        p = SamplingParams(temperature=0.0, min_p=0.01)
        assert not p.is_greedy()

    def test_both_top_p_and_min_p_stochastic(self):
        p = SamplingParams(temperature=0.0, top_p=0.9, min_p=0.05)
        assert not p.is_greedy()

    def test_temperature_1_top_k_1_greedy(self):
        # Even high temperature is greedy if top_k=1.
        p = SamplingParams(temperature=1.0, top_k=1, top_p=1.0, min_p=0.0)
        assert p.is_greedy()

    def test_high_temperature_not_greedy(self):
        p = SamplingParams(temperature=2.0)
        assert not p.is_greedy()


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_logits(self):
        p = SamplingParams(repetition_penalty=2.0)
        result = p.apply_penalties([], [1, 2, 3])
        assert result == []

    def test_empty_counts(self):
        p = SamplingParams(repetition_penalty=2.0)
        result = p.apply_penalties([1.0, 2.0], [])
        assert result == [1.0, 2.0]

    def test_penalty_does_not_mutate_input(self):
        logits = [3.0, 4.0]
        counts = [1, 0]
        p = SamplingParams(repetition_penalty=2.0)
        result = p.apply_penalties(logits, counts)
        # Input unchanged.
        assert logits == [3.0, 4.0]
        # Output changed.
        assert result[0] != logits[0]

    def test_very_large_repetition_penalty(self):
        logits = [100.0]
        counts = [1]
        p = SamplingParams(repetition_penalty=1000.0)
        result = p.apply_penalties(logits, counts)
        assert abs(result[0] - 0.1) < 1e-4  # 100.0 / 1000.0

    def test_inf_logit_with_penalty(self):
        logits = [math.inf]
        counts = [1]
        p = SamplingParams(repetition_penalty=2.0)
        result = p.apply_penalties(logits, counts)
        assert math.isinf(result[0])  # inf / 2 is still inf

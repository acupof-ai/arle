"""Offline unit tests for scripts/arle_capability_eval.py.

These tests do NOT require an `arle serve` running. They cover the parsing
helpers (MMLU letter extraction, GSM8K gold/predicted answer extraction)
that determine correctness of the eval and would silently mis-score
otherwise.

Run:
    python -m pytest scripts/tests/test_arle_capability_eval.py -v
or:
    python -m unittest scripts.tests.test_arle_capability_eval
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow `import arle_capability_eval` when running from repo root.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import arle_capability_eval as eval_mod  # noqa: E402


class MMLULetterExtraction(unittest.TestCase):
    def test_plain_letter(self):
        self.assertEqual(eval_mod._mmlu_extract_letter("A"), "A")
        self.assertEqual(eval_mod._mmlu_extract_letter("D"), "D")

    def test_letter_with_paren(self):
        self.assertEqual(eval_mod._mmlu_extract_letter("B)"), "B")
        self.assertEqual(eval_mod._mmlu_extract_letter("C) because reasons"), "C")

    def test_letter_with_dot(self):
        self.assertEqual(eval_mod._mmlu_extract_letter("A. obvious"), "A")

    def test_lowercase_normalized(self):
        self.assertEqual(eval_mod._mmlu_extract_letter("b"), "B")
        self.assertEqual(eval_mod._mmlu_extract_letter("c) yes"), "C")

    def test_leading_whitespace_ok(self):
        self.assertEqual(eval_mod._mmlu_extract_letter("  A"), "A")
        self.assertEqual(eval_mod._mmlu_extract_letter("\n\nD)"), "D")

    def test_no_letter_returns_none(self):
        self.assertIsNone(eval_mod._mmlu_extract_letter(""))
        self.assertIsNone(eval_mod._mmlu_extract_letter("E"))  # outside A-D

    def test_full_word_answer_rejected(self):
        # "answer is something" contains no A-D letter at all in the
        # first 60 chars, so all layers (1-4) miss → None.
        self.assertIsNone(eval_mod._mmlu_extract_letter("answer is something"))

    def test_layer2_parenthesized(self):
        # Base models commonly output "(A)" — layer 2 catches this.
        self.assertEqual(eval_mod._mmlu_extract_letter("(A)"), "A")
        self.assertEqual(eval_mod._mmlu_extract_letter("(B) some reasoning"), "B")

    def test_layer3_answer_is_pattern(self):
        # "The answer is C." → layer 3 phrase match.
        self.assertEqual(eval_mod._mmlu_extract_letter("The answer is C."), "C")
        self.assertEqual(eval_mod._mmlu_extract_letter("Answer: D, because..."), "D")
        # "correct option B" → layer 3 phrase match.
        self.assertEqual(eval_mod._mmlu_extract_letter("The correct option is B"), "B")

    def test_layer4_first60_char_fallback(self):
        # No leading letter, no "answer is" phrase, but a free-standing
        # A-D in the first 60 chars → layer 4 fallback.
        self.assertEqual(eval_mod._mmlu_extract_letter("After reasoning, A"), "A")

    def test_avocado_not_extracted_as_a(self):
        # "Avocado" should NOT match — the A is part of a longer word.
        # All layers should miss because no standalone A-D appears.
        # Layer 1 fails because "A" is followed by "v" (not punctuation
        # or whitespace). Layers 2-4 also miss because there's no
        # standalone A.
        self.assertIsNone(eval_mod._mmlu_extract_letter("Avocado"))


class GSM8KGoldAnswer(unittest.TestCase):
    def test_simple_int(self):
        self.assertEqual(eval_mod._gsm8k_gold_answer("reasoning. #### 42"), "42")

    def test_negative(self):
        self.assertEqual(eval_mod._gsm8k_gold_answer("...#### -7"), "-7")

    def test_decimal(self):
        self.assertEqual(eval_mod._gsm8k_gold_answer("....\n#### 3.14"), "3.14")

    def test_with_commas_strips(self):
        self.assertEqual(eval_mod._gsm8k_gold_answer("#### 1,234,567"), "1234567")

    def test_no_marker_empty(self):
        self.assertEqual(eval_mod._gsm8k_gold_answer("no marker here"), "")


class GSM8KPredExtraction(unittest.TestCase):
    def test_marker_preferred(self):
        # Marker present → use the marker number, not the last number in text.
        text = "First we get 5 apples, then 3 more. #### 8\nbonus reasoning 99"
        self.assertEqual(eval_mod._gsm8k_extract_answer(text), "8")

    def test_marker_with_commas(self):
        self.assertEqual(eval_mod._gsm8k_extract_answer("#### 1,000"), "1000")

    def test_fallback_to_last_number(self):
        # No #### marker → last number in body.
        text = "5 plus 3 equals 8."
        self.assertEqual(eval_mod._gsm8k_extract_answer(text), "8")

    def test_fallback_negative(self):
        text = "answer is -42 ultimately"
        self.assertEqual(eval_mod._gsm8k_extract_answer(text), "-42")

    def test_fallback_decimal(self):
        text = "the ratio comes out to 1.5"
        self.assertEqual(eval_mod._gsm8k_extract_answer(text), "1.5")

    def test_empty_returns_none(self):
        self.assertIsNone(eval_mod._gsm8k_extract_answer(""))
        self.assertIsNone(eval_mod._gsm8k_extract_answer("no numbers here"))


class MMLUShotFormatting(unittest.TestCase):
    def test_shot_layout(self):
        ex = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,  # index of "4"
        }
        shot = eval_mod._mmlu_format_shot(ex)
        self.assertIn("What is 2+2?", shot)
        self.assertIn("A) 3", shot)
        self.assertIn("B) 4", shot)
        self.assertIn("C) 5", shot)
        self.assertIn("D) 6", shot)
        self.assertTrue(shot.rstrip().endswith("Answer: B"))


class TaskRunnerRegistry(unittest.TestCase):
    def test_supported_tasks_present(self):
        self.assertIn("mmlu", eval_mod.TASK_RUNNERS)
        self.assertIn("gsm8k", eval_mod.TASK_RUNNERS)

    def test_callable(self):
        for task, fn in eval_mod.TASK_RUNNERS.items():
            self.assertTrue(callable(fn), f"runner {task!r} not callable")


if __name__ == "__main__":
    unittest.main()

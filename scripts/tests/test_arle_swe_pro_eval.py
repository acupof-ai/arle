"""Offline tests for scripts/arle_swe_pro_eval.py.

No network, Docker, Modal, or ARLE server is required. These tests pin the
result-responsibility boundary: model-visible prompts exclude gold fields, patch
extraction is mechanical, and official evaluation is invoked as an external
deterministic scorer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import arle_swe_pro_eval as swe_mod  # noqa: E402


def sample_row() -> dict:
    return {
        "repo": "example/repo",
        "instance_id": "example.repo-abc123",
        "base_commit": "0123456789abcdef0123456789abcdef01234567",
        "problem_statement": "Fix the frobnicator when input is empty.",
        "requirements": "Python 3.11",
        "interface": "frobnicate(value: str) -> str",
        "repo_language": "Python",
        "patch": "diff --git a/gold b/gold",
        "test_patch": "diff --git a/tests b/tests",
        "fail_to_pass": "['test_empty']",
        "pass_to_pass": "['test_existing']",
    }


class PromptBoundary(unittest.TestCase):
    def test_sanitize_instance_excludes_gold_fields(self):
        sanitized = swe_mod.sanitize_instance(sample_row())
        for forbidden in swe_mod.FORBIDDEN_MODEL_FIELDS:
            self.assertNotIn(forbidden, sanitized)
        self.assertEqual(sanitized["instance_id"], "example.repo-abc123")
        self.assertIn("problem_statement", sanitized)

    def test_patch_prompt_does_not_leak_gold_fields(self):
        messages = swe_mod.build_patch_prompt(sample_row())
        text = "\n".join(message["content"] for message in messages)
        self.assertIn("Fix the frobnicator", text)
        self.assertNotIn("diff --git a/gold", text)
        self.assertNotIn("test_empty", text)
        self.assertNotIn("test_existing", text)


class PatchExtraction(unittest.TestCase):
    def test_extracts_fenced_diff_mechanically(self):
        raw = "Here is the patch:\n```diff\ndiff --git a/a.py b/a.py\n+ok\n```\n"
        self.assertEqual(swe_mod.extract_unified_diff(raw), "diff --git a/a.py b/a.py\n+ok")

    def test_extracts_raw_diff_from_preface(self):
        raw = "Patch follows:\n\ndiff --git a/a.py b/a.py\n+ok\n"
        self.assertEqual(swe_mod.extract_unified_diff(raw), "diff --git a/a.py b/a.py\n+ok")

    def test_non_diff_text_is_preserved_not_repaired(self):
        self.assertEqual(swe_mod.extract_unified_diff("I cannot solve this."), "I cannot solve this.")


class Selection(unittest.TestCase):
    def test_select_rows_by_instance_id_reports_missing(self):
        rows = [sample_row()]
        with self.assertRaisesRegex(ValueError, "unknown instance ids"):
            swe_mod.select_rows(rows, instance_ids=["missing"], limit=None, repo=None, seed=None)

    def test_select_rows_seed_is_deterministic(self):
        rows = [{**sample_row(), "instance_id": f"id-{i}"} for i in range(10)]
        first = swe_mod.select_rows(rows, limit=4, instance_ids=None, repo=None, seed=7)
        second = swe_mod.select_rows(rows, limit=4, instance_ids=None, repo=None, seed=7)
        self.assertEqual([r["instance_id"] for r in first], [r["instance_id"] for r in second])


class OfficialEvaluatorCommand(unittest.TestCase):
    def test_safe_prefix_removes_path_separators(self):
        self.assertEqual(swe_mod._safe_prefix("org/model name"), "org_model_name")
        self.assertEqual(swe_mod._safe_prefix("///"), "arle")

    def test_command_uses_official_paths_and_flags(self):
        cmd = swe_mod.build_official_eval_command(
            python="python3",
            eval_repo=Path("/repo/SWE-bench_Pro-os"),
            output=Path("/tmp/out"),
            num_workers=8,
            dockerhub_username="jefzda",
            use_local_docker=True,
            docker_platform="linux/amd64",
            block_network=True,
            redo=True,
        )
        joined = " ".join(cmd)
        self.assertIn("/repo/SWE-bench_Pro-os/swe_bench_pro_eval.py", joined)
        self.assertIn("--raw_sample_path=/tmp/out/raw_samples.csv", cmd)
        self.assertIn("--patch_path=/tmp/out/patches.json", cmd)
        self.assertIn("--scripts_dir=/repo/SWE-bench_Pro-os/run_scripts", cmd)
        self.assertIn("--use_local_docker", cmd)
        self.assertIn("--docker_platform=linux/amd64", cmd)
        self.assertIn("--block_network", cmd)
        self.assertIn("--redo", cmd)

    def test_generation_contract_pins_official_scorer(self):
        self.assertEqual(swe_mod.GENERATION_CONTRACT["candidate_source"], "ARLE inference output")
        self.assertFalse(swe_mod.GENERATION_CONTRACT["harness_may_repair_patch"])
        self.assertEqual(swe_mod.GENERATION_CONTRACT["scoring_policy"], "official SWE-bench Pro Docker evaluator")


if __name__ == "__main__":
    unittest.main()

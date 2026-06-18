"""Synthetic end-to-end checks for OPD MATH500 capability collection.

No model, serve, CUDA, or datasets dependency is used. The test writes fake
`arle_capability_eval.py`-shaped MATH500 outputs, then runs the real curve and
multi-seed collection scripts over them.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from statistics import mean, stdev

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import arle_capability_eval as eval_mod  # noqa: E402


N = 500
COUNTS = {
    "baseline": {0: 300, 1: 310, 2: 305},
    "reverse-kl": {0: 320, 1: 325, 2: 315},
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def write_fake_math500(out_dir: Path, *, seed: int, n_correct: int, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(N):
        correct = i < n_correct
        records.append(
            {
                "i": i,
                "seed": seed,
                "gold": f"q{i}",
                "predicted": f"q{i}" if correct else "wrong",
                "correct": correct,
                "status": "scored",
            }
        )
    ci95 = list(eval_mod._wilson_ci(n_correct, N))
    report = {
        "task": "math500",
        "status": "ok",
        "schema_version": eval_mod.SCRIPT_SCHEMA_VERSION,
        "n_samples": N,
        "n_scored": N,
        "n_invalid": 0,
        "n_correct": n_correct,
        "accuracy": n_correct / N,
        "ci95": ci95,
        "seed": seed,
    }
    summary = {
        "schema_version": eval_mod.SCRIPT_SCHEMA_VERSION,
        "backend": "synthetic",
        "model_id": label,
        "seed": seed,
        "tasks": {"math500": report},
    }
    (out_dir / "math500.json").write_text(json.dumps(report, indent=2))
    (out_dir / "math500_perquestion.json").write_text(json.dumps(records, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


class SyntheticOPDCapabilityCurve(unittest.TestCase):
    def test_math500_scoring_delta_and_curve_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            curve_out = root / "curve"
            analysis = root / "analysis"
            manifest = []

            for arm, by_seed in COUNTS.items():
                for seed, correct in by_seed.items():
                    label = f"{arm}-s{seed}"
                    capability_dir = curve_out / label / "capability"
                    write_fake_math500(capability_dir, seed=seed, n_correct=correct, label=label)
                    write_fake_math500(
                        analysis / arm / f"seed_{seed}",
                        seed=seed,
                        n_correct=correct,
                        label=label,
                    )
                    entry = {
                        "label": label,
                        "model_path": f"/synthetic/{label}",
                        "arm": arm,
                        "training_seed": seed,
                    }
                    if arm != "baseline":
                        entry["delta_vs_arm"] = "baseline"
                    manifest.append(entry)

            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))

            curve_run = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "opd_capability_curve.py"),
                    "--manifest",
                    str(manifest_path),
                    "--baseline-label",
                    "baseline-s0",
                    "--tasks",
                    "math500",
                    "--n-samples",
                    str(N),
                    "--output",
                    str(curve_out),
                    "--skip-existing",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("across training seeds", curve_run.stdout)
            self.assertIn("reverse-kl", curve_run.stdout)

            curve = json.loads((curve_out / "curve.json").read_text())
            self.assertEqual(curve["baseline_label"], "baseline-s0")
            self.assertTrue((curve_out / "curve.svg").exists())
            self.assertIn("<svg", (curve_out / "curve.svg").read_text())

            by_arm = {"baseline": [], "reverse-kl": []}
            for point in curve["points"]:
                by_arm[point["arm"]].append(point["capability"]["tasks"]["math500"]["accuracy"])
            self.assertAlmostEqual(mean(by_arm["baseline"]), 0.61)
            self.assertAlmostEqual(mean(by_arm["reverse-kl"]), 0.64)
            deltas = [
                COUNTS["reverse-kl"][seed] / N - COUNTS["baseline"][seed] / N
                for seed in sorted(COUNTS["baseline"])
            ]
            self.assertAlmostEqual(mean(deltas), 0.03)
            self.assertAlmostEqual(stdev(deltas), 0.01)

            report = json.loads((curve_out / "baseline-s0" / "capability" / "math500.json").read_text())
            self.assertEqual(report["ci95"], list(wilson(300, N)))

            analyze_run = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "analyze_multi_seed.py"),
                    str(analysis / "reverse-kl"),
                    "--paired-vs",
                    str(analysis / "baseline"),
                    "--task",
                    "math500",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("Across seeds: mean=0.6400 sigma=0.0100", analyze_run.stdout)
            self.assertIn("Paired mean delta: +3.00pp", analyze_run.stdout)
            self.assertIn("TOTAL        1500", analyze_run.stdout)


if __name__ == "__main__":
    unittest.main()

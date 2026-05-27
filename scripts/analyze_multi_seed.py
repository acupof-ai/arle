#!/usr/bin/env python3
"""Analyze multi-seed OPD capability eval results.

Reads summary.json from <out_base>/seed_<N>/ and optionally one or more
baseline summary.json paths, then reports per-seed accuracy + binomial
95% CI, across-seed mean/sample-σ, and a verdict against the kill
criterion in docs/research/2026-05-28-opd-effect-axis-next.md.

Usage:
    python scripts/analyze_multi_seed.py <out_base> [--baseline path ...]
                                          [--task mmlu|gsm8k]
                                          [--threshold-mean 0.505]
                                          [--threshold-sigma 0.015]

Examples:
    # Default — both tasks, kill criterion from research doc
    python scripts/analyze_multi_seed.py \\
        runs/2026-05-26-rollout128-v4-diverse1k-train-60/capability_seeds \\
        --baseline runs/2026-05-26-rollout128-v4-diverse1k-train-60/capability/step_000020

    # GSM8K-only with custom threshold
    python scripts/analyze_multi_seed.py <out_base> --task gsm8k --threshold-mean 0.32

The kill criterion is per-task: pass if mean(seeds) >= threshold_mean
AND sample-σ <= threshold_sigma. Default thresholds are MMLU-tuned.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — better than normal approx at small n + extreme p."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfwidth = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - halfwidth), min(1.0, center + halfwidth))


def load_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def per_task_stats(seeds_data: list[dict], baseline_data: list[dict], task: str) -> dict:
    seed_accs = []
    seed_rows = []
    for d in seeds_data:
        t = d["tasks"].get(task)
        if not t or t.get("status") != "ok":
            continue
        acc = t["accuracy"]
        k = t["n_correct"]
        n = t["n_scored"]
        lo, hi = wilson_ci(k, n)
        seed_accs.append(acc)
        seed_rows.append({
            "seed": d.get("seed"),
            "acc": acc,
            "k": k,
            "n_scored": n,
            "n_invalid": t.get("n_invalid", 0),
            "ci95": (lo, hi),
        })

    baseline_rows = []
    for d in baseline_data:
        t = d["tasks"].get(task)
        if not t or t.get("status") != "ok":
            continue
        acc = t["accuracy"]
        k = t["n_correct"]
        n = t["n_scored"]
        lo, hi = wilson_ci(k, n)
        baseline_rows.append({
            "label": d.get("_label", "baseline"),
            "acc": acc,
            "k": k,
            "n_scored": n,
            "n_invalid": t.get("n_invalid", 0),
            "ci95": (lo, hi),
        })

    out = {
        "task": task,
        "seeds": seed_rows,
        "baselines": baseline_rows,
        "n_seeds": len(seed_accs),
    }
    if seed_accs:
        out["mean"] = mean(seed_accs)
        out["sigma"] = stdev(seed_accs) if len(seed_accs) > 1 else 0.0
        out["min"] = min(seed_accs)
        out["max"] = max(seed_accs)
    return out


def print_report(stats: dict, threshold_mean: float, threshold_sigma: float) -> None:
    task = stats["task"]
    print(f"\n══════════ {task.upper()} ══════════")

    if stats["baselines"]:
        print("\nBaselines:")
        for b in stats["baselines"]:
            lo, hi = b["ci95"]
            print(f"  {b['label']:>20s}: acc={b['acc']:.4f} "
                  f"({b['k']}/{b['n_scored']}, invalid={b['n_invalid']}) "
                  f"CI95=[{lo:.4f}, {hi:.4f}]")

    if not stats["seeds"]:
        print("\nNo seed data yet.")
        return

    print(f"\nPer-seed (n_seeds={stats['n_seeds']}):")
    for s in stats["seeds"]:
        lo, hi = s["ci95"]
        print(f"  seed={s['seed']:>3}: acc={s['acc']:.4f} "
              f"({s['k']}/{s['n_scored']}, invalid={s['n_invalid']}) "
              f"CI95=[{lo:.4f}, {hi:.4f}]")

    print(f"\nAcross seeds: mean={stats['mean']:.4f} "
          f"sigma={stats['sigma']:.4f} "
          f"min={stats['min']:.4f} max={stats['max']:.4f}")

    if stats["n_seeds"] >= 2:
        sem = stats["sigma"] / math.sqrt(stats["n_seeds"])
        ci_lo = stats["mean"] - 1.96 * sem
        ci_hi = stats["mean"] + 1.96 * sem
        print(f"Mean 95% CI (across seeds): [{ci_lo:.4f}, {ci_hi:.4f}] (SEM={sem:.4f})")

    # Kill-criterion gate
    pass_mean = stats["mean"] >= threshold_mean
    pass_sigma = stats["sigma"] <= threshold_sigma
    verdict = "PASS" if (pass_mean and pass_sigma) else "KILL"
    print(f"\nKill criterion: mean>={threshold_mean:.4f} AND sigma<={threshold_sigma:.4f}")
    print(f"  mean check : {pass_mean}  (got {stats['mean']:.4f})")
    print(f"  sigma check: {pass_sigma} (got {stats['sigma']:.4f})")
    print(f"  verdict    : {verdict}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_base", type=Path, help="dir containing seed_<N>/ subdirs")
    p.add_argument("--baseline", type=Path, action="append", default=[],
                   help="path to an additional summary.json dir to include as baseline (repeatable)")
    p.add_argument("--task", choices=["mmlu", "gsm8k", "both"], default="both")
    p.add_argument("--threshold-mean", type=float, default=0.505,
                   help="kill threshold for mean (default 0.505 = MMLU cross-base gate)")
    p.add_argument("--threshold-sigma", type=float, default=0.015,
                   help="kill threshold for sample-σ (default 0.015)")
    args = p.parse_args()

    seeds_data = []
    for d in sorted(args.out_base.glob("seed_*")):
        sp = d / "summary.json"
        if sp.exists():
            data = json.loads(sp.read_text())
            data["_label"] = d.name
            seeds_data.append(data)

    baseline_data = []
    for bp in args.baseline:
        sp = bp / "summary.json"
        if not sp.exists():
            print(f"warn: {sp} not found, skipping", flush=True)
            continue
        data = json.loads(sp.read_text())
        data["_label"] = bp.name
        baseline_data.append(data)

    tasks = ["mmlu", "gsm8k"] if args.task == "both" else [args.task]
    for task in tasks:
        stats = per_task_stats(seeds_data, baseline_data, task)
        print_report(stats, args.threshold_mean, args.threshold_sigma)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

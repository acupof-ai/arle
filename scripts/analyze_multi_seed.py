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


def mcnemar_question_paired(treated_out: Path, control_out: Path, task: str) -> dict:
    """Question-level McNemar paired test, aggregated across matched seeds.

    Reads `<task>_perquestion.json` from <treated>/seed_<N>/ and
    <control>/seed_<N>/ at the same seed values. The eval uses a
    deterministic seed → shuffle mapping (random.Random(f"<task>-{seed}-
    {subj}") for MMLU, random.Random(f"gsm8k-{seed}") for GSM8K), so the
    question pool at matching seed is identical between runs and pairing
    by index `i` is safe.

    Aggregates `b` (treated-correct + control-wrong) and `c` (treated-
    wrong + control-correct) across all (seed, question) pairs where
    BOTH sides have a valid extraction. Returns paired mean Δ =
    (b - c) / n_paired with binomial standard error.

    Far tighter than 5-seed paired when discordance is low (both models
    answer most questions the same way) — which is exactly the regime
    for small-delta OPD comparisons.
    """
    treated_seeds = {p.name: p for p in treated_out.glob("seed_*")}
    control_seeds = {p.name: p for p in control_out.glob("seed_*")}
    common = sorted(set(treated_seeds) & set(control_seeds))

    total_b = 0  # treated correct, control wrong
    total_c = 0  # treated wrong, control correct
    total_a = 0  # both correct (concordant)
    total_d = 0  # both wrong (concordant)
    total_n = 0  # paired questions with valid both-sides
    per_seed_breakdown = []

    for name in common:
        tp = treated_seeds[name] / f"{task}_perquestion.json"
        cp = control_seeds[name] / f"{task}_perquestion.json"
        if not (tp.exists() and cp.exists()):
            continue
        t_recs = json.loads(tp.read_text())
        c_recs = json.loads(cp.read_text())
        if len(t_recs) != len(c_recs):
            continue
        a = b = c = d = paired_n = 0
        for tr, cr in zip(t_recs, c_recs):
            # Require both sides successfully extracted (status == 'scored').
            if tr.get("status") != "scored" or cr.get("status") != "scored":
                continue
            if tr.get("gold") != cr.get("gold"):
                continue
            paired_n += 1
            tc = bool(tr.get("correct"))
            cc = bool(cr.get("correct"))
            if tc and cc:
                a += 1
            elif tc and not cc:
                b += 1
            elif not tc and cc:
                c += 1
            else:
                d += 1
        per_seed_breakdown.append({
            "seed": tr.get("seed") if t_recs else name,  # best effort
            "name": name,
            "n_paired": paired_n,
            "a_both_correct": a,
            "b_treated_only": b,
            "c_control_only": c,
            "d_both_wrong": d,
        })
        total_a += a
        total_b += b
        total_c += c
        total_d += d
        total_n += paired_n

    if total_n == 0:
        return {"task": task, "n_paired": 0, "per_seed": per_seed_breakdown}

    delta = (total_b - total_c) / total_n
    # Var((b-c)/n) ≈ (b + c - (b-c)²/n) / n² under multinomial
    var_delta_num = (total_b + total_c) - (total_b - total_c) ** 2 / total_n
    var_delta = var_delta_num / (total_n ** 2)
    se = math.sqrt(max(0.0, var_delta))
    ci_lo = delta - 1.96 * se
    ci_hi = delta + 1.96 * se
    # McNemar χ² (continuity-corrected)
    discordant = total_b + total_c
    chi2 = ((abs(total_b - total_c) - 1) ** 2) / discordant if discordant > 0 else 0.0
    return {
        "task": task,
        "n_paired": total_n,
        "n_seeds_paired": len(per_seed_breakdown),
        "a_both_correct": total_a,
        "b_treated_only": total_b,
        "c_control_only": total_c,
        "d_both_wrong": total_d,
        "delta": delta,
        "se": se,
        "ci95": (ci_lo, ci_hi),
        "mcnemar_chi2": chi2,
        "per_seed": per_seed_breakdown,
    }


def print_mcnemar_report(stats: dict) -> None:
    task = stats["task"]
    print(f"\n══════════ McNemar QUESTION-LEVEL PAIRED {task.upper()} ══════════")
    if stats["n_paired"] == 0:
        print("No paired question data (per-question JSON missing on one side?)")
        return
    print(f"{'seed_dir':<10} {'n':>6} {'both ✓':>7} {'T only':>7} {'C only':>7} {'both ✗':>7}")
    for r in stats["per_seed"]:
        print(f"{r['name']:<10} {r['n_paired']:>6} "
              f"{r['a_both_correct']:>7} {r['b_treated_only']:>7} "
              f"{r['c_control_only']:>7} {r['d_both_wrong']:>7}")
    print(f"{'TOTAL':<10} {stats['n_paired']:>6} "
          f"{stats['a_both_correct']:>7} {stats['b_treated_only']:>7} "
          f"{stats['c_control_only']:>7} {stats['d_both_wrong']:>7}")

    delta_pp = 100 * stats["delta"]
    se_pp = 100 * stats["se"]
    lo_pp = 100 * stats["ci95"][0]
    hi_pp = 100 * stats["ci95"][1]
    print(f"\nPaired delta (T − C) = {delta_pp:+.2f}pp  SE = {se_pp:.2f}pp  "
          f"95% CI = [{lo_pp:+.2f}pp, {hi_pp:+.2f}pp]")
    print(f"McNemar χ² (continuity-corrected) = {stats['mcnemar_chi2']:.3f}  "
          f"({'reject H0' if stats['mcnemar_chi2'] > 3.841 else 'fail to reject H0'} at 95%)")


def paired_delta(treated_out: Path, control_out: Path, task: str) -> dict:
    """Per-seed paired delta (treated - control) at matched seeds.

    The question subset is determined by the seed alone, so paired
    differences cancel the question-subset variance and give a much
    tighter estimate of the true model effect.
    """
    seeds = []
    for td in sorted(treated_out.glob("seed_*")):
        cd = control_out / td.name
        ts = td / "summary.json"
        cs = cd / "summary.json"
        if not (ts.exists() and cs.exists()):
            continue
        t = json.loads(ts.read_text())
        c = json.loads(cs.read_text())
        tt = t["tasks"].get(task)
        cc = c["tasks"].get(task)
        if not (tt and cc and tt.get("status") == "ok" and cc.get("status") == "ok"):
            continue
        seeds.append({
            "seed": t.get("seed"),
            "treated": tt["accuracy"],
            "control": cc["accuracy"],
            "delta": tt["accuracy"] - cc["accuracy"],
        })
    if not seeds:
        return {"task": task, "n_seeds": 0, "seeds": seeds}
    deltas = [s["delta"] for s in seeds]
    out = {
        "task": task,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "mean_delta": mean(deltas),
        "sigma_delta": stdev(deltas) if len(deltas) > 1 else 0.0,
    }
    return out


def print_paired_report(stats: dict) -> None:
    task = stats["task"]
    print(f"\n══════════ PAIRED {task.upper()} (treated − control) ══════════")
    if stats["n_seeds"] == 0:
        print("No matched seed pairs found.")
        return
    print(f"{'seed':>5} {'treated':>10} {'control':>10} {'delta(pp)':>11}")
    for s in stats["seeds"]:
        print(f"{str(s['seed']):>5} {s['treated']:>10.4f} {s['control']:>10.4f} {100*s['delta']:>+11.2f}")

    md = stats["mean_delta"]
    sd = stats["sigma_delta"]
    n = stats["n_seeds"]
    sem = sd / math.sqrt(n) if n > 1 else float("inf")
    ci_lo = md - 1.96 * sem
    ci_hi = md + 1.96 * sem
    # t-statistic for H0: mean_delta = 0
    t_stat = (md / sem) if sem > 0 else float("inf")
    print(f"\nPaired mean delta: {100*md:+.2f}pp  sample-σ: {100*sd:.2f}pp  n={n}")
    print(f"95% CI of mean delta: [{100*ci_lo:+.2f}pp, {100*ci_hi:+.2f}pp]  (SEM={100*sem:.2f}pp)")
    if math.isfinite(t_stat):
        print(f"t (H0 delta=0)     : {t_stat:+.2f}  ({'reject H0' if abs(t_stat) > 2 else 'fail to reject H0'} at ~95% with n={n})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_base", type=Path, help="dir containing seed_<N>/ subdirs")
    p.add_argument("--baseline", type=Path, action="append", default=[],
                   help="path to an additional summary.json dir to include as baseline (repeatable)")
    p.add_argument("--paired-vs", type=Path, default=None,
                   help="path to another out_base; runs a per-seed paired (treated-control) "
                        "analysis where treated=out_base, control=this arg. Tightens the "
                        "estimate by canceling question-subset variance.")
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

    if args.paired_vs is not None:
        for task in tasks:
            stats = paired_delta(args.out_base, args.paired_vs, task)
            print_paired_report(stats)
            mc = mcnemar_question_paired(args.out_base, args.paired_vs, task)
            print_mcnemar_report(mc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

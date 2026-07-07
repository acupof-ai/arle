#!/usr/bin/env python3
"""Agentic-OPD capability curve: held-out pass-rate + train accept-rate/loss.

Reads what `arle train agent-opd` already emits — no Rust-side changes:
  - <eval-dir>/eval_round_base.jsonl, eval_round_<N>.jsonl (aggregate line
    per file; label N = evaluated after N training rounds)
  - the training stderr log (round lines:
    "[arle train agent-opd] round R: tasks=.. rollouts=.. passed=.. ...
     trained_pairs=.. mean_loss=..")
  - optional extra baseline eval dirs (same-config repeats) -> the MoE
    non-determinism envelope the curve must clear.

Writes curve.json next to the PNG. Plan:
docs/plans/2026-07-03-agentic-opd-27b-capability-curve.md
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# House palette (matches scripts/plot_dsv4_perf_journey.py)
GREY = "#868e96"
MAGENTA = "#d6336c"
GREEN = "#2f9e44"
INK = "#212529"

ROUND_RE = re.compile(
    r"\[arle train agent-opd\] round (\d+): tasks=(\d+) rollouts=(\d+) "
    r"passed=(\d+) distinct=(\d+) no_token_record=(\d+) trained_pairs=(\d+) "
    r"mean_loss=([\d.eE+-]+)"
)


def read_eval_dir(eval_dir):
    """{round: aggregate-dict} from eval_round_*.jsonl ('base' -> 0)."""
    points = {}
    for path in sorted(Path(eval_dir).glob("eval_round_*.jsonl")):
        label = path.stem.removeprefix("eval_round_")
        rnd = 0 if label == "base" else int(label)
        agg = None
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("aggregate"):
                agg = row
        if agg is not None:
            points[rnd] = agg
    return points


def read_metrics_jsonl(path):
    """Per-round rows from the structured metrics.jsonl sink (one JSON/round).

    Preferred over regex-scraping the stderr log: the Rust loop emits the full
    AgentRoundReport + phase timers here. Returns [] if the file is absent.
    """
    path = Path(path)
    if not path.exists():
        return []
    rounds = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rollouts = int(row.get("rollouts", 0))
        passed = int(row.get("passed", 0))
        rounds.append(
            {
                "round": int(row["round"]),
                "tasks": int(row.get("tasks", 0)),
                "rollouts": rollouts,
                "passed": passed,
                "distinct": int(row.get("distinct_passed", 0)),
                "trained_pairs": int(row.get("trained_pairs", 0)),
                "accept_rate": passed / rollouts if rollouts else 0.0,
                "mean_loss": float(row.get("mean_train_loss", 0.0)),
            }
        )
    return rounds


def read_train_log(log_path):
    rounds = []
    for m in ROUND_RE.finditer(Path(log_path).read_text(errors="replace")):
        rnd, tasks, rollouts, passed, distinct, _, pairs, loss = m.groups()
        rollouts = int(rollouts)
        rounds.append(
            {
                "round": int(rnd),
                "tasks": int(tasks),
                "rollouts": rollouts,
                "passed": int(passed),
                "distinct": int(distinct),
                "trained_pairs": int(pairs),
                "accept_rate": int(passed) / rollouts if rollouts else 0.0,
                "mean_loss": float(loss),
            }
        )
    return rounds


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--train-log", type=Path)
    ap.add_argument("--metrics", type=Path,
                    help="structured metrics.jsonl; defaults to "
                         "<eval-dir>/metrics.jsonl. Preferred over --train-log.")
    ap.add_argument("--baseline-extra", type=Path, nargs="*", default=[],
                    help="eval dirs of same-config baseline repeats (envelope)")
    ap.add_argument("--out", type=Path, default=Path("agent_opd_curve.png"))
    ap.add_argument("--title", default="Agentic OPD — 27B held-out pass-rate")
    args = ap.parse_args()

    evals = read_eval_dir(args.eval_dir)
    if not evals:
        sys.exit(f"no eval_round_*.jsonl under {args.eval_dir}")
    # Prefer the structured metrics.jsonl sink; fall back to regex-scraping the
    # stderr train log only when the sink is absent.
    metrics_path = args.metrics or (args.eval_dir / "metrics.jsonl")
    train = read_metrics_jsonl(metrics_path)
    if not train and args.train_log:
        train = read_train_log(args.train_log)

    baseline = evals.get(0, {}).get("pass_rate")
    envelope = [baseline] if baseline is not None else []
    for extra in args.baseline_extra:
        for agg in read_eval_dir(extra).values():
            envelope.append(agg["pass_rate"])

    curve = {
        "baseline_pass_rate": baseline,
        "baseline_envelope": envelope,
        "evals": [
            {"round": r, **{k: evals[r][k] for k in
                            ("pass_rate", "passed", "tasks")}}
            for r in sorted(evals)
        ],
        "train": train,
    }
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(curve, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#495057",
        "axes.linewidth": 0.9,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)

    xs = sorted(evals)
    ys = [evals[r]["pass_rate"] for r in xs]
    ax.plot(xs, ys, color=MAGENTA, marker="o", lw=2.2, zorder=5,
            label="held-out pass-rate (greedy)")
    if envelope:
        lo, hi = min(envelope), max(envelope)
        ax.axhspan(lo, hi, color=GREY, alpha=0.18, zorder=1,
                   label=f"baseline envelope (n={len(envelope)})")
        ax.axhline(hi, color=GREY, lw=0.8, ls=":")
    if train:
        tx = [r["round"] + 1 for r in train]
        ax.plot(tx, [r["accept_rate"] for r in train], color=GREY, ls="--",
                lw=1.4, marker=".", zorder=3,
                label="train accept-rate (temp 1.0)")

        ax2 = ax.twinx()
        ax2.plot(tx, [r["mean_loss"] for r in train], color=GREEN, lw=1.2,
                 alpha=0.85, zorder=2)
        ax2.set_ylabel("train mean CE loss", color=GREEN, fontsize=9.5)
        ax2.tick_params(axis="y", labelcolor=GREEN, labelsize=8.5)
        ax2.set_ylim(bottom=0)

    for r, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):
        n = evals[r]
        ax.annotate(f"{y:.0%} ({n['passed']}/{n['tasks']})",
                    (r, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=INK, fontweight="bold")

    ax.set_xlabel("training round")
    ax.set_ylabel("pass-rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=-0.3)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.set_title(args.title)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out} and {json_path}")
    if baseline is not None and xs[-1] != 0:
        final = ys[-1]
        print(f"baseline={baseline:.4f} final={final:.4f} "
              f"delta={final - baseline:+.4f} "
              f"envelope_max={max(envelope):.4f}")


if __name__ == "__main__":
    main()

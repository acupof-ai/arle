#!/usr/bin/env python3
"""Compare two benchmark snapshots and flag regressions.

Usage:
  python scripts/bench_compare.py benchmarks/baseline.json benchmarks/current.json
  python scripts/bench_compare.py benchmarks/baseline.json benchmarks/current.json --threshold 5
"""

import argparse
import json
import sys


def load_snapshot(path):
    with open(path) as f:
        return json.load(f)


def config_key(c):
    return (c["input_len"], c["output_len"], c["concurrency"])


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark snapshots")
    parser.add_argument("baseline", help="Baseline snapshot JSON")
    parser.add_argument("current", help="Current snapshot JSON")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Regression threshold %% (default: 5)")
    parser.add_argument("--metric", default="throughput",
                        choices=["throughput", "itl_p50", "ttft_p50"],
                        help="Primary metric to compare")
    args = parser.parse_args()

    base = load_snapshot(args.baseline)
    curr = load_snapshot(args.current)

    print(f"Baseline: {base['label']} ({base.get('gpu', '?')}, {base['timestamp'][:10]})")
    print(f"Current:  {curr['label']} ({curr.get('gpu', '?')}, {curr['timestamp'][:10]})")
    print(f"Metric:   {args.metric}, threshold: {args.threshold}%")
    print()

    base_map = {config_key(c): c for c in base["configs"]}
    curr_map = {config_key(c): c for c in curr["configs"]}

    common_keys = sorted(set(base_map) & set(curr_map))
    if not common_keys:
        print("No matching configs found!")
        return 1

    higher_is_better = args.metric == "throughput"
    regressions = []

    hdr = f"{'In':>5} | {'Out':>5} | {'C':>2} | {'Baseline':>10} | {'Current':>10} | {'Delta':>8} | {'Status'}"
    print(hdr)
    print("-" * len(hdr))

    for key in common_keys:
        b = base_map[key]
        c = curr_map[key]
        bv = b[args.metric]
        cv = c[args.metric]

        if bv == 0 and cv == 0:
            delta_pct = 0
        elif bv == 0:
            delta_pct = 100
        else:
            delta_pct = ((cv - bv) / abs(bv)) * 100

        if higher_is_better:
            is_regression = delta_pct < -args.threshold
        else:
            is_regression = delta_pct > args.threshold

        status = "REGRESS" if is_regression else ("IMPROVE" if abs(delta_pct) > args.threshold else "OK")
        marker = " <<<" if is_regression else ""

        unit = "t/s" if args.metric == "throughput" else "ms"
        print(f"{key[0]:5d} | {key[1]:5d} | {key[2]:2d} | "
              f"{bv:8.1f}{unit:>2} | {cv:8.1f}{unit:>2} | "
              f"{delta_pct:+6.1f}% | {status}{marker}")

        if is_regression:
            regressions.append((key, bv, cv, delta_pct))

    print()
    if regressions:
        print(f"REGRESSIONS DETECTED: {len(regressions)} configs exceed {args.threshold}% threshold")
        for key, bv, cv, d in regressions:
            print(f"  in={key[0]} out={key[1]} C={key[2]}: {bv:.1f} → {cv:.1f} ({d:+.1f}%)")
        return 1
    else:
        print(f"ALL CLEAR: no regressions above {args.threshold}% threshold")
        return 0


if __name__ == "__main__":
    sys.exit(main())

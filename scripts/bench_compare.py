#!/usr/bin/env python3
"""Compare two bench_throughput.py v1 snapshots and flag regressions.

Usage:
  python scripts/bench_compare.py bench-output/a/result.json bench-output/b/result.json
  python scripts/bench_compare.py a.json b.json --threshold 5 --metric ttft_p50
"""

import argparse
import json
import sys
import time


def load_snapshot(path):
    with open(path) as f:
        return json.load(f)


def point_key(p):
    return p["summary"]["concurrency"]


def metrics(summary):
    itl = summary.get("itl") or {}
    ttft = summary.get("ttft") or {}
    itl_mean = itl.get("mean_ms")
    return {
        "decode": (1000.0 / itl_mean) if itl_mean else None,
        "ttft_p50": ttft.get("p50_ms"),
        "itl_p50": itl.get("p50_ms"),
        "rps": summary.get("requests_per_s"),
        "complete": summary.get("complete", 0),
    }


HIGHER_IS_BETTER = {"decode", "rps"}
UNIT = {"decode": "t/s", "ttft_p50": "ms", "itl_p50": "ms", "rps": "r/s"}


def main():
    parser = argparse.ArgumentParser(description="Compare bench_throughput v1 snapshots")
    parser.add_argument("baseline", help="Baseline snapshot JSON")
    parser.add_argument("current", help="Current snapshot JSON")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Regression threshold %% (default: 5)")
    parser.add_argument("--metric", default="decode",
                        choices=sorted(UNIT),
                        help="Primary metric to compare (default: decode tok/s)")
    args = parser.parse_args()

    base = load_snapshot(args.baseline)
    curr = load_snapshot(args.current)

    for name, snap in (("baseline", base), ("current", curr)):
        if snap.get("schema") != "arle.bench_throughput.v1":
            print(f"error: {name} snapshot is not arle.bench_throughput.v1", file=sys.stderr)
            return 2

    # Refuse cross-workload comparisons: different prompt sets or output lengths
    # make every delta meaningless.
    base_fp = base.get("fingerprint") or {}
    curr_fp = curr.get("fingerprint") or {}
    if base_fp.get("dataset_sha256") != curr_fp.get("dataset_sha256"):
        print("error: dataset_sha256 mismatch — snapshots are from different prompt sets", file=sys.stderr)
        return 2
    if base["config"].get("max_tokens") != curr["config"].get("max_tokens"):
        print("error: max_tokens mismatch — snapshots used different output lengths", file=sys.stderr)
        return 2

    date_fmt = lambda s: time.strftime("%Y-%m-%d", time.localtime(s))
    print(f"Baseline: {base['config'].get('model', '?')} ({date_fmt(base['started_unix_s'])})")
    print(f"Current:  {curr['config'].get('model', '?')} ({date_fmt(curr['started_unix_s'])})")
    print(f"Metric:   {args.metric}, threshold: {args.threshold}%")
    print()

    base_map = {point_key(p): metrics(p["summary"]) for p in base["points"]}
    curr_map = {point_key(p): metrics(p["summary"]) for p in curr["points"]}

    common = sorted(set(base_map) & set(curr_map))
    if not common:
        print("No matching concurrency points found!")
        return 1

    higher = args.metric in HIGHER_IS_BETTER
    unit = UNIT[args.metric]
    regressions = []

    hdr = f"{'C':>4} | {'Baseline':>10} | {'Current':>10} | {'Delta':>8} | {'Status'}"
    print(hdr)
    print("-" * len(hdr))

    for c in common:
        bm = base_map[c]
        cm = curr_map[c]
        bv = bm.get(args.metric)
        cv = cm.get(args.metric)
        # Zero completed requests is a collapse, not a skip.
        if bm["complete"] == 0 or cm["complete"] == 0:
            print(f"{c:4d} | {'n/a':>10} | {'n/a':>10} | {'n/a':>8} | COLLAPSE")
            regressions.append((c, bv or 0, cv or 0, 0.0))
            continue
        if bv is None or cv is None:
            print(f"{c:4d} | {'n/a':>10} | {'n/a':>10} | {'n/a':>8} | SKIP")
            continue

        if bv == 0:
            delta_pct = 100.0 if cv != 0 else 0.0
        else:
            delta_pct = ((cv - bv) / abs(bv)) * 100

        is_regression = (delta_pct < -args.threshold) if higher else (delta_pct > args.threshold)
        status = "REGRESS" if is_regression else (
            "IMPROVE" if abs(delta_pct) > args.threshold else "OK")
        marker = " <<<" if is_regression else ""

        print(f"{c:4d} | {bv:8.1f}{unit:>2} | {cv:8.1f}{unit:>2} | "
              f"{delta_pct:+6.1f}% | {status}{marker}")

        if is_regression:
            regressions.append((c, bv, cv, delta_pct))

    print()
    if regressions:
        print(f"REGRESSIONS DETECTED: {len(regressions)} points exceed {args.threshold}% threshold")
        for c, bv, cv, d in regressions:
            print(f"  c={c}: {bv:.1f} → {cv:.1f} ({d:+.1f}%)")
        return 1
    print(f"ALL CLEAR: no regressions above {args.threshold}% threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

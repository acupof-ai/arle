#!/usr/bin/env python3
"""Symlink a pool's sweet-spot tasks (0 < pass < attempts) into an out dir — the
difficulty-calibrated substrate. Difficulty = the model's own pass rate (Tmax),
the only ground truth. Prints the per-family band (all-100% too easy / all-0%
too hard). Source-agnostic (real TB or generated tasks).

    filter_inband.py <results.json> <pool_dir> <out_dir> [attempts]
"""
import collections
import json
import os
import re
import sys


def main() -> int:
    results, pool, out = sys.argv[1], sys.argv[2], sys.argv[3]
    attempts = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 0 = infer from data

    rows = json.load(open(results)).get("results", [])
    passes = collections.Counter()  # task_id -> #resolved
    seen = collections.Counter()    # task_id -> #trials
    for x in rows:
        tid = x.get("task_id")
        seen[tid] += 1
        passes[tid] += 1 if x.get("is_resolved") else 0
    if not seen:
        sys.exit("no results rows")
    k = attempts or max(seen.values())

    inband = sorted(t for t in seen if 0 < passes[t] < seen[t])
    always = sorted(t for t in seen if passes[t] == seen[t])
    never = sorted(t for t in seen if passes[t] == 0)

    os.makedirs(out, exist_ok=True)
    for t in inband:
        src = os.path.realpath(os.path.join(pool, t))
        dst = os.path.join(out, t)
        if os.path.lexists(dst):
            os.remove(dst)
        if os.path.isdir(src):
            os.symlink(src, dst)

    # per-family band (strip trailing -NNN instance suffix)
    fam = collections.defaultdict(lambda: [0, 0])
    for t in seen:
        f = re.sub(r"-\d+$", "", t)
        fam[f][0] += passes[t]
        fam[f][1] += seen[t]
    print(f"pool={pool}  attempts={k}  tasks={len(seen)}")
    print(f"  in-band (0<pass<{k}): {len(inband)}   always-pass: {len(always)}   never: {len(never)}")
    print("per-family pass rate (want some strictly between 0 and 1):")
    for f, (p, n) in sorted(fam.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        band = "  <-- in-band" if 0 < p < n else ("  (too easy)" if p == n else "  (too hard)")
        print(f"  {p/n:4.0%}  {p:2d}/{n:2d}  {f}{band}")
    print(f"linked {len(inband)} in-band tasks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

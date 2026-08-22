#!/usr/bin/env python3
"""Keep only localized single-point mutations from a swe-smith task corpus.

A swe-smith corpus injects synthetic bugs. `combine_file` / `combine_module`
stack several mutations into one task, which produces a failure surface no real
bug has: in `staged-sweetspot3` the combined tasks sit at a median of 24-74
failing tests and reach 286. Training on those teaches "rewrite a wide area",
not "locate and repair".

The two caps are read off the corpus, not chosen: after dropping the combined
mutations, `fail_to_pass` runs p90=81 then jumps to p95=252, and `gold_patch`
runs p90=77 then p95=109 with a 668-line maximum. Both cliffs separate
localized bugs from sprawling ones.

This does NOT replace `opd_security_filter.py`; run that too.

    python3 scripts/filter_swesmith_localized.py IN.jsonl OUT.jsonl
    python3 scripts/filter_swesmith_localized.py IN.jsonl --report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

MAX_FAIL_TO_PASS = 80
MAX_GOLD_PATCH_LINES = 110
DROPPED_MUTATIONS = ("combine_file", "combine_module")


def mutation_kind(instance_id: str) -> str:
    return instance_id.split(".")[-1].split("__")[0]


def reject_reason(rec: dict) -> str | None:
    """None = keep."""
    kind = mutation_kind(rec.get("instance_id", ""))
    if kind in DROPPED_MUTATIONS:
        return "combined-mutation"
    n_fail = len(rec.get("fail_to_pass") or ())
    if n_fail > MAX_FAIL_TO_PASS:
        return "shotgun-failure-surface"
    if not n_fail:
        return "no-failing-test"
    if not rec.get("gold_patch"):
        return "no-gold-patch"
    if rec["gold_patch"].count("\n") > MAX_GOLD_PATCH_LINES:
        return "sprawling-gold-patch"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--report", action="store_true", help="counts only, write nothing")
    args = ap.parse_args()
    if not args.report and not args.dst:
        ap.error("dst is required unless --report")

    kept, dropped = [], Counter()
    with open(args.src) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            reason = reject_reason(rec)
            if reason:
                dropped[reason] += 1
            else:
                kept.append(rec)

    total = len(kept) + sum(dropped.values())
    print(f"{args.src}: {total} in, {len(kept)} kept, {sum(dropped.values())} dropped")
    for reason, n in dropped.most_common():
        print(f"  dropped {n:5d}  {reason}")
    repos = Counter(r.get("repo", "?") for r in kept)
    print(f"  kept across {len(repos)} repos: {dict(repos)}")

    if args.report:
        return 0
    with open(args.dst, "w") as fh:
        for rec in kept:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

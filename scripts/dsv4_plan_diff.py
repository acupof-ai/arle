#!/usr/bin/env python3
"""Align [dsv4-plan] fingerprint streams across ranks; print first divergence."""
import re
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/data01/build/arle_serve_allreduce.log"
pat = re.compile(r"\[dsv4-plan\] rank=(\d+) tick=(\d+) (.+)$")
ticks = defaultdict(dict)  # tick -> rank -> fingerprint
for line in open(path, errors="replace"):
    m = pat.search(line)
    if m:
        ticks[int(m.group(2))][int(m.group(1))] = m.group(3).strip()

if not ticks:
    print("no [dsv4-plan] lines found")
    sys.exit(1)

ranks_seen = sorted({r for t in ticks.values() for r in t})
print(f"ticks={len(ticks)} ranks={ranks_seen}")
diverged = False
for t in sorted(ticks):
    fps = ticks[t]
    uniq = set(fps.values())
    missing = [r for r in ranks_seen if r not in fps]
    if len(uniq) > 1 or missing:
        print(f"\n=== FIRST DIVERGENCE at tick {t} ===")
        if missing:
            print(f"missing ranks (no fingerprint logged): {missing}")
        for r in sorted(fps):
            print(f"  rank {r}: {fps[r]}")
        # context: previous 2 ticks
        for ctx in (t - 2, t - 1):
            if ctx in ticks:
                u = set(ticks[ctx].values())
                print(f"  [ctx tick {ctx}] uniform={len(u) == 1} "
                      f"ranks={len(ticks[ctx])} fp={next(iter(u))[:90]}")
        diverged = True
        break
if not diverged:
    last = max(ticks)
    print(f"all {len(ticks)} ticks uniform across ranks; last tick {last}: "
          f"{next(iter(ticks[last].values()))[:90]}")

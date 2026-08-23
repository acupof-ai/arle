#!/usr/bin/env python3
"""Attribute GPU kernel time to the engine's NVTX op ranges in an nsys SQLite.

Joins CUPTI_ACTIVITY_KIND_KERNEL to the innermost overlapping NVTX push-pop
range (ARLE_NVTX=1) and aggregates per (op, kernel-family). Use it to see
exactly which op each decode millisecond goes to, on any slice or full-model
trace.

Usage: python3 nsys_op_attrib.py <trace.sqlite> [--marks "sample seq=32"] [--limit N]

With --marks, the window is the first..last NVTX mark of that text and the
per-step column divides by the mark count (e.g. 128 marks = 128 decode steps).
Without it, the whole trace is one window.
"""

import argparse
import re
import sqlite3


KERNEL_FAMILIES = [
    ("marlin", "marlin"),
    ("nvjet", "nvjet"),
    ("deep_gemm", "deep_gemm"),
    ("gdr_decode", "gdr"),
    ("paged_attention", "paged_attn"),
    ("FlashAttn", "paged_attn"),
    ("ncclDev", "nccl"),
    ("rms_norm", "rms_norm"),
    ("argmax", "argmax"),
    ("silu_mul", "silu_mul"),
    ("add_native", "add"),
    ("conv1d", "conv1d"),
    ("quantize", "quantize"),
]


def kernel_family(name: str) -> str:
    for needle, family in KERNEL_FAMILIES:
        if needle in name:
            return family
    return name.split("(")[0][:40]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite_path")
    ap.add_argument("--marks", default="sample seq=32")
    ap.add_argument("--max-marks", type=int, default=0, help="use only the first N marks (single-wave window)")
    ap.add_argument("--limit", type=int, default=30)
    a = ap.parse_args()

    db = sqlite3.connect(a.sqlite_path)
    c = db.cursor()
    if a.marks:
        marks = c.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text = ? ORDER BY start",
            (a.marks,),
        ).fetchall()
        if not marks:
            raise SystemExit(f"no marks: {a.marks}")
        if a.max_marks:
            marks = marks[: a.max_marks]
        t0, t1 = marks[0][0], max(r[1] for r in marks)
        nsteps = len(marks)
    else:
        t0, t1 = 0, 1 << 62
        nsteps = 1

    ranges = c.execute(
        "SELECT start, end, text FROM NVTX_EVENTS "
        "WHERE eventType = 59 AND text IS NOT NULL AND start < ? AND end > ?",
        (t1, t0),
    ).fetchall()
    kernels = c.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName = s.id "
        "WHERE k.start < ? AND k.end > ?",
        (t1, t0),
    ).fetchall()

    agg: dict[tuple[str, str], list] = {}
    starts = sorted(r[0] for r in ranges)
    for ks, ke, kname in kernels:
        # Kernels launch async: the GPU start can fall after the CPU range
        # popped. Attribute to the innermost range active at the kernel start;
        # if the CPU is between ops, to the last range popped before it.
        active = [r for r in ranges if r[0] <= ks <= r[1]]
        if active:
            best = min(active, key=lambda r: r[1] - r[0])
        else:
            past = [r for r in ranges if r[1] <= ks]
            if not past:
                continue
            best = max(past, key=lambda r: r[1])
        op = re.sub(r"layer\d+", "layerN", best[2])
        key = (op, kernel_family(kname))
        n, busy = agg.get(key, (0, 0))
        agg[key] = (n + 1, busy + (min(ke, t1) - max(ks, t0)))

    win = t1 - t0
    print(f"window={win / 1e6:.1f}ms steps={nsteps} -> {win / nsteps / 1e3:.3f}ms/step")
    print(f"{'us/step':>9} {'cnt':>6} {'share':>6}  op @ kernel")
    for (op, kf), (n, busy) in sorted(agg.items(), key=lambda x: -x[1][1])[: a.limit]:
        print(f"{busy / nsteps / 1e3:9.1f} {n:6d} {busy / win * 100:5.1f}%  {op[:42]:42s} @ {kf}")


if __name__ == "__main__":
    main()

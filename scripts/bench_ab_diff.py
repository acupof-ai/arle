#!/usr/bin/env python3
"""Cross-label A/B diff for bench_ab.sh.

Reads each arm's bench_throughput.json and writes a markdown diff. Refuses
(nonzero exit) when either arm has no data rows — a single-arm comparison is
not an A/B, and bench_ab.sh turns this into a fail-loud exit.
"""
import sys
import json
import pathlib


def load(arm_dir):
    p = pathlib.Path(arm_dir) / "bench_throughput.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    rows = {}
    for point in j.get("points", []):
        m = point.get("summary", {})
        key = f"conc{m.get('concurrency', '?')}"
        rows[key] = {
            "ttft_p50": (m.get("ttft") or {}).get("p50_ms"),
            "itl_p50": (m.get("itl") or {}).get("p50_ms"),
            "tok_s": (1000.0 / im if (im := (m.get("itl") or {}).get("mean_ms")) else None),
        }
    return rows


def pct(x, y):
    if x is None or y is None or x == 0:
        return "n/a"
    return f"{((y - x) / x) * 100:+.1f}%"


def fmt(x, d=1):
    if x is None:
        return "n/a"
    return f"{x:.{d}f}"


def main():
    a_dir, b_dir, label_a, label_b, out_path = sys.argv[1:6]
    a = load(a_dir) or {}
    b = load(b_dir) or {}
    if not a or not b:
        print(
            f"error: refusing to diff without both arms: {label_a}={len(a)} rows, "
            f"{label_b}={len(b)} rows",
            file=sys.stderr,
        )
        return 1
    keys = sorted(set(a) | set(b), key=lambda k: (
        0 if k == "sync" else 1 if k.startswith("conc") else 2, k
    ))

    lines = [
        f"# A/B diff — {label_a} vs {label_b}",
        "",
        f"- A: {a_dir}",
        f"- B: {b_dir}",
        "",
        "| rate | A decode tok/s | B decode tok/s | Δ decode | A TTFT p50 | B TTFT p50 | Δ TTFT |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in keys:
        av, bv = a.get(k, {}), b.get(k, {})
        lines.append(
            f"| {k} | {fmt(av.get('tok_s'), 2)} | {fmt(bv.get('tok_s'), 2)} "
            f"| {pct(av.get('tok_s'), bv.get('tok_s'))} "
            f"| {fmt(av.get('ttft_p50'), 1)} | {fmt(bv.get('ttft_p50'), 1)} "
            f"| {pct(av.get('ttft_p50'), bv.get('ttft_p50'))} |"
        )
    lines += [
        "",
        "Δ is (B - A) / A. Negative TTFT Δ is faster; positive tok/s Δ is faster.",
        "",
        "Reminder: effects <=10% in a single session are thermal noise; rerun "
        "or extend the cell duration before trusting small deltas.",
        "",
    ]

    pathlib.Path(out_path).write_text("\n".join(lines) + "\n")
    print("".join(f"{l}\n" for l in lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

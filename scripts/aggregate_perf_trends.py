#!/usr/bin/env python3
"""Aggregate performance trends from docs/experience/wins/*.md entries.

Extracts decode (ITL/TPOT/latency), prefill (TTFT), and throughput (tok/s)
metrics, groups them by date, and prints a markdown table plus a monthly
summary to stdout.

Usage: python3 scripts/aggregate_perf_trends.py
"""

import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

WINS_DIR = Path(__file__).resolve().parent.parent / "docs" / "experience" / "wins"

DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")

# Metric patterns. Each entry: (compiled_regex, category, take_group)
# Arrow pairs ("39.57 -> 24.90 ms") capture the post-value in group 2.
ARROW_RE = re.compile(
    r"(ITL|TPOT|TTFT|decode|prefill|lat(?:ency)?)\s*(?:p\d{2})?"
    r"[^.\n]{0,40}?(?<![p\d])([\d][\d,]*\.?\d*)\s*(?:→|->)\s*([\d][\d,]*\.?\d*)\s*ms",
    re.IGNORECASE,
)
SINGLE_RE = re.compile(
    r"(ITL|TPOT|TTFT|decode|prefill|lat(?:ency)?)\s*(?:p\d{2})?"
    r"[^.\n]{0,40}?(?<![p\d])([\d][\d,]*\.?\d*)\s*ms",
    re.IGNORECASE,
)
TOKPS_RE = re.compile(r"([\d][\d,]*\.?\d*)\s*(?:tok/s|tps)\b", re.IGNORECASE)

DECODE_KEYWORDS = {"itl", "tpot", "decode", "lat", "latency"}
PREFILL_KEYWORDS = {"ttft", "prefill"}

# Model name patterns, in priority order. Maps normalized name -> regex.
MODEL_PATTERNS = [
    ("DSv4", re.compile(r"\b(?:dsv4|deepseek-?v4)\b", re.IGNORECASE)),
    ("GLM-5.2", re.compile(r"\bglm-?5\.2\b|\bglm52\b", re.IGNORECASE)),
    ("ThinkingCap-27B", re.compile(r"\bthinkingcap\b|\btc-27b\b", re.IGNORECASE)),
    (
        "Qwen3.6-35B",
        re.compile(r"\bqwen3\.6-35b\b|\bqwen36-35b\b", re.IGNORECASE),
    ),
    (
        "Qwen3.6-27B",
        re.compile(r"\bqwen3\.6-27b\b|\bqwen36-27b\b", re.IGNORECASE),
    ),
    (
        "Qwen3.8-27B",
        re.compile(r"\bqwen3\.8-27b\b|\bqwen38-27b\b", re.IGNORECASE),
    ),
    (
        "Qwen3.5-122B",
        re.compile(r"\bqwen3\.5-122b\b|\bqwen35-122b\b", re.IGNORECASE),
    ),
    ("Qwen3.5-9B", re.compile(r"\bqwen3\.5-9b\b|\bqwen35-09b\b", re.IGNORECASE)),
    ("Qwen3.5-4B", re.compile(r"\bqwen3\.5-4b\b|\bqwen35-4b\b", re.IGNORECASE)),
    (
        "Qwen3.5-0.8B",
        re.compile(r"\bqwen3\.5-0\.8b\b|\bqwen35-08b\b", re.IGNORECASE),
    ),
    ("Qwen3-0.6B", re.compile(r"\bqwen3-0\.6b\b", re.IGNORECASE)),
    ("Qwen3-1.7B", re.compile(r"\bqwen3-1\.7b\b", re.IGNORECASE)),
    ("Qwen3-4B", re.compile(r"\bqwen3-4b\b", re.IGNORECASE)),
    ("Qwen3-30B", re.compile(r"\bqwen3-30b\b", re.IGNORECASE)),
    ("Qwen3-235B", re.compile(r"\bqwen3-235b\b", re.IGNORECASE)),
    ("Llama", re.compile(r"\bllama[0-9.]*\b", re.IGNORECASE)),
    ("MiniCPM", re.compile(r"\bminicpm\b", re.IGNORECASE)),
    ("InternVL", re.compile(r"\binternvl\b", re.IGNORECASE)),
]


def to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def classify(keyword: str) -> str | None:
    kw = keyword.lower()
    if kw in DECODE_KEYWORDS:
        return "decode"
    if kw in PREFILL_KEYWORDS:
        return "prefill"
    return None


def extract_metrics(text: str) -> dict:
    """Return {'decode': ms|None, 'prefill': ms|None, 'throughput': tps|None}.

    First document-order match wins per category; arrow pairs ("a -> b ms")
    take the post-value, since entries report baseline -> result.
    """
    metrics: dict = {"decode": None, "prefill": None, "throughput": None}

    # Arrow pairs first so SINGLE_RE does not swallow the pre-arrow number.
    consumed_spans: list[tuple[int, int]] = []
    for m in ARROW_RE.finditer(text):
        consumed_spans.append(m.span())
        category = classify(m.group(1))
        if category and metrics[category] is None:
            metrics[category] = to_float(m.group(3))

    def consumed(span: tuple[int, int]) -> bool:
        return any(
            span[0] >= c[0] and span[1] <= c[1] for c in consumed_spans
        )

    for m in SINGLE_RE.finditer(text):
        if consumed(m.span()):
            continue
        category = classify(m.group(1))
        if category and metrics[category] is None:
            metrics[category] = to_float(m.group(2))

    tps_values = [
        to_float(m.group(1))
        for m in TOKPS_RE.finditer(text)
        if not consumed(m.span())
    ]
    if tps_values:
        # Entries often cite baseline and result; the median is the stable pick.
        metrics["throughput"] = statistics.median(tps_values)

    return metrics


def extract_model(text: str) -> str:
    """Most frequently mentioned known model; ties broken by first occurrence."""
    counts: Counter = Counter()
    first_pos: dict = {}
    for name, pattern in MODEL_PATTERNS:
        for m in pattern.finditer(text):
            counts[name] += 1
            first_pos.setdefault(name, m.start())
    if not counts:
        return "-"
    # Highest count, then earliest first occurrence.
    return sorted(counts.items(), key=lambda kv: (-kv[1], first_pos[kv[0]]))[0][0]


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def main() -> int:
    if not WINS_DIR.is_dir():
        print(f"error: {WINS_DIR} not found", file=sys.stderr)
        return 1

    files = sorted(WINS_DIR.glob("*.md"))
    rows = []
    skipped = 0
    for path in files:
        m = DATE_RE.match(path.name)
        if not m:
            skipped += 1
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        metrics = extract_metrics(text)
        if all(v is None for v in metrics.values()):
            skipped += 1
            continue
        rows.append(
            {
                "date": m.group(1),
                "model": extract_model(text),
                "decode_ms": metrics["decode"],
                "prefill_ms": metrics["prefill"],
                "throughput": metrics["throughput"],
                "source": path.name,
            }
        )

    rows.sort(key=lambda r: r["date"])

    print("# Performance trends from wins entries\n")
    print(f"Parsed {len(files)} entries; {len(rows)} with metrics, {skipped} without.\n")
    print("| date | model | decode_ms | prefill_ms | throughput_tok_s | source_file |")
    print("|------|-------|-----------|------------|------------------|-------------|")
    for r in rows:
        print(
            f"| {r['date']} | {r['model']} | {fmt(r['decode_ms'])} "
            f"| {fmt(r['prefill_ms'])} | {fmt(r['throughput'])} | {r['source']} |"
        )

    # Monthly summary over all dated files (with or without metrics).
    per_month: dict = defaultdict(lambda: {"total": 0, "decode": [], "prefill": []})
    for path in files:
        m = DATE_RE.match(path.name)
        if not m:
            continue
        month = m.group(1)[:7]
        per_month[month]["total"] += 1
    for r in rows:
        month = r["date"][:7]
        if r["decode_ms"] is not None:
            per_month[month]["decode"].append(r["decode_ms"])
        if r["prefill_ms"] is not None:
            per_month[month]["prefill"].append(r["prefill_ms"])

    print("\n## Monthly summary\n")
    print("| month | entries | decode_median_ms | prefill_median_ms |")
    print("|-------|---------|------------------|-------------------|")
    for month in sorted(per_month):
        s = per_month[month]
        dec = statistics.median(s["decode"]) if s["decode"] else None
        pre = statistics.median(s["prefill"]) if s["prefill"] else None
        print(f"| {month} | {s['total']} | {fmt(dec)} | {fmt(pre)} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Parse `needle_gate.py` SUMMARY lines — the one parser both gates import."""

from __future__ import annotations

import re

SUMMARY = re.compile(
    r"^SUMMARY len=(?P<len>\d+) depth=(?P<depth>[\d.]+) exact=(?P<exact>\d+) "
    r"partial=(?P<partial>\d+) miss=(?P<miss>\d+) (?P<det>DET|NONDET)(?: kv=.*)?$"
)


def parse_summaries(text: str, runs: int) -> dict[int, dict]:
    """`{length: {"exact", "partial", "miss", "det"}}`.

    Raises ValueError on a request ERROR line, a malformed or duplicate
    SUMMARY, or a summary whose counts do not add up to `runs`.
    """
    counts: dict[int, dict] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if " ERROR " in line:
            raise ValueError(f"request error: {line}")
        if not line.startswith("SUMMARY "):
            continue
        match = SUMMARY.match(line)
        if match is None:
            raise ValueError(f"malformed summary: {line}")
        length = int(match["len"])
        row = {k: int(match[k]) for k in ("exact", "partial", "miss")}
        row["det"] = match["det"]
        if row["exact"] + row["partial"] + row["miss"] != runs:
            raise ValueError(f"incomplete summary: {row['exact']}+{row['partial']}+{row['miss']} != runs={runs}")
        if length in counts:
            raise ValueError(f"duplicate summary length {length}")
        counts[length] = row
    return counts


def all_exact_deterministic(counts: dict[int, dict], runs: int) -> bool:
    return bool(counts) and all(r["exact"] == runs and r["det"] == "DET" for r in counts.values())

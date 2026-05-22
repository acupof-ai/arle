#!/usr/bin/env python3
"""Side-by-side capability eval comparison report.

Loads summary.json files from multiple eval runs (each produced by
scripts/arle_capability_eval.py) and prints a markdown comparison
table. The comparison is purely client-side; each run is assumed to
have been independently driven (different model / different backend /
different checkpoint).

Use cases:

  1. **Train before/after** — eval the HF base, train OPD with checkpoint
     save, eval the saved adapter, then:
       python scripts/arle_capability_compare.py \\
         --label "base" bench-output/<base-eval>/summary.json \\
         --label "distill@2k"  bench-output/<2k-eval>/summary.json \\
         --label "distill@10k" bench-output/<10k-eval>/summary.json \\
         --label "teacher" bench-output/<teacher-eval>/summary.json

  2. **ARLE vs PyTorch ecosystem** — eval the same checkpoint via ARLE
     serve AND via transformers in-process:
       python scripts/arle_capability_eval.py --backend arle ... \\
         --output bench-output/arle-eval/
       python scripts/arle_capability_eval.py --backend hf ... \\
         --output bench-output/hf-eval/
       python scripts/arle_capability_compare.py \\
         --label "ARLE serve" bench-output/arle-eval/summary.json \\
         --label "HF transformers" bench-output/hf-eval/summary.json

  3. **Cross-eval validation** — when ARLE and HF agree to within ~1pp on
     the same checkpoint, the harness + serve path is validated. When
     they disagree by >2pp, either a serve bug (like 2026-05-22
     long-prompt) or a tokenizer/format mismatch is in play.

Output: a markdown table written to stdout; also `--output-md` writes
to a file for direct inclusion in wins/errors entries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"summary not found: {path}")
    return json.loads(path.read_text())


def format_acc(task_report: dict[str, Any]) -> str:
    if task_report.get("status") != "ok":
        return f"({task_report.get('status', 'missing')})"
    correct = task_report.get("n_correct", 0)
    scored = task_report.get("n_scored", 0)
    invalid = task_report.get("n_invalid", 0)
    acc = task_report.get("accuracy", 0.0)
    return f"{acc * 100:.1f}% ({correct}/{scored}, inv {invalid})"


def build_table(labels: list[str], summaries: list[dict[str, Any]]) -> str:
    tasks = sorted({t for s in summaries for t in s.get("tasks", {})})
    header = ["Label", "Backend", "Model"] + tasks
    rows = [header, ["---"] * len(header)]

    for label, summary in zip(labels, summaries):
        backend = summary.get("backend", "?")
        model_id = summary.get("model_id", "?")
        row = [label, backend, model_id]
        for task in tasks:
            row.append(format_acc(summary.get("tasks", {}).get(task, {})))
        rows.append(row)

    if len(summaries) >= 2:
        # Add delta rows: each non-first row's delta vs the first
        base = summaries[0]
        for label, summary in zip(labels[1:], summaries[1:]):
            row = [f"Δ {label} − {labels[0]}", "", ""]
            for task in tasks:
                b_acc = base.get("tasks", {}).get(task, {}).get("accuracy")
                s_acc = summary.get("tasks", {}).get(task, {}).get("accuracy")
                if b_acc is None or s_acc is None:
                    row.append("—")
                else:
                    delta = (s_acc - b_acc) * 100
                    sign = "+" if delta >= 0 else ""
                    row.append(f"{sign}{delta:.2f}pp")
            rows.append(row)

    widths = [max(len(str(r[i])) for r in rows) for i in range(len(header))]
    out_lines = []
    for r in rows:
        out_lines.append("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(header))) + " |")
    return "\n".join(out_lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", action="append", required=True, metavar="LABEL=PATH",
                        help="repeated: LABEL=path/to/summary.json (use --pair once per run to compare)")
    parser.add_argument("--output-md", type=Path, default=None,
                        help="also write the table to this markdown file")
    args = parser.parse_args(argv)

    labels: list[str] = []
    paths: list[Path] = []
    for pair in args.pair:
        if "=" not in pair:
            print(f"bad --pair {pair!r}: must be LABEL=path/to/summary.json", file=sys.stderr)
            return 2
        label, _, path_str = pair.partition("=")
        labels.append(label)
        paths.append(Path(path_str))

    summaries = [load_summary(p) for p in paths]
    table = build_table(labels, summaries)

    print(table)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(table + "\n")
        print(f"\nwrote {args.output_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Score rubric-OPD in-process eval dumps (eval_round*.jsonl with {problem, gold, answer}).

Reuses the validated MATH-500 \\boxed extraction + normalization (copied verbatim from
scripts/arle_capability_eval.py so numbers are comparable). Prints accuracy per round —
the rubric-OPD capability curve (base vs each round).

Usage: python3 scripts/score_rubric_eval.py <eval_dir>
"""
import glob
import json
import os
import re
import sys


def _extract_last_braced(text, marker):
    last = None
    start = 0
    while True:
        pos = text.find(marker, start)
        if pos < 0:
            return last
        i = pos + len(marker)
        depth = 1
        out = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = "".join(out).strip()
                    if cand:
                        last = cand
                    break
            out.append(ch)
            i += 1
        start = pos + len(marker)


def _norm(answer):
    s = answer.strip()
    boxed = _extract_last_braced(s, "\\boxed{")
    if boxed is not None:
        s = boxed
    s = s.replace("\\$", "").strip("$")
    for old, new in (
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\:", ""),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
    ):
        s = s.replace(old, new)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s.lower()


def score_file(path):
    n = correct = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        gold = _norm(str(d.get("gold", "")))
        pred = _norm(str(d.get("answer", "")))
        n += 1
        if gold and pred == gold:
            correct += 1
    return correct, n


def _order_key(path):
    base = os.path.basename(path)
    if "base" in base:
        return -1
    m = re.search(r"round(\d+)", base)
    return int(m.group(1)) if m else 999


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(d, "eval_round*.jsonl")), key=_order_key)
    if not files:
        print(f"no eval_round*.jsonl in {d}", file=sys.stderr)
        sys.exit(1)
    print(f"{'round':<8} {'acc':>7} {'n':>6}")
    for path in files:
        correct, n = score_file(path)
        label = os.path.basename(path).replace("eval_round", "").replace(".jsonl", "")
        acc = correct / n if n else 0.0
        print(f"{label:<8} {acc:>7.3f} {n:>6}  ({correct}/{n})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the math-reasoning train/eval corpora for GSPO length-compression RL.

Inputs in --raw-dir:
  numina-train-*.parquet          AI-MO/NuminaMath-CoT shards
                                  (columns: source, problem, solution)
  aime2025-*.jsonl                opencompass/AIME2025 ({question, answer})
  aimo-validation-aime.parquet    AI-MO/aimo-validation-aime
                                  (id, problem, solution, answer, url)

Outputs in --out-dir:
  train.jsonl   ~2000 problems, stratified easy (gsm8k, synthetic_math) +
                hard (math, olympiads, amc_aime) mix
  eval.jsonl    ~200 problems: AIME 2024 + AIME 2025 gold sets, held out
                amc_aime + olympiads; zero problem overlap with train

Schema per line: {"text": <problem statement>, "answer": <canonical gold>}

Answer extraction takes the LAST complete \\boxed{...} in the solution (nested
braces matched). Canonicalization strips LaTeX spacing commands (\\! \\, \\; \\:
\\quad \\qquad), $ and \\$, whitespace, thousands-commas on grouped numbers,
and leading zeros on integer answers. AIME golds additionally reduce to their
leading integer (sources variously write "033" or "336^\\circ"). Problems with
no boxed answer are dropped; problems are deduped across both files by
lowercased, punctuation-stripped problem text.

Deterministic (fixed seed) and idempotent: reruns overwrite the outputs.
Stdlib + pandas/pyarrow only.
"""

import argparse
import glob
import json
import os
import re
import string
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED = 42

EASY_SOURCES = {"gsm8k", "synthetic_math"}
HARD_SOURCES = {"amc_aime", "olympiads", "math"}
WANTED_SOURCES = EASY_SOURCES | HARD_SOURCES

TRAIN_QUOTA = {
    "gsm8k": 500,
    "synthetic_math": 500,
    "math": 400,
    "olympiads": 400,
    "amc_aime": 200,
}
EVAL_NUMINA_QUOTA = {"amc_aime": 70, "olympiads": 70}

_BOXED_RE = re.compile(r"\\boxed\s*\{")
_SPACING_CMDS = ("\\quad", "\\qquad", "\\!", "\\,", "\\;", "\\:", "\\ ")
_NUMERIC_RE = re.compile(r"-?\d+(\.\d+)?(/\d+)?")
_INT_RE = re.compile(r"-?\d+")
_GROUPED_RE = re.compile(r"-?\d{1,3}(,\d{3})+(\.\d+)?")
_LEADING_INT_RE = re.compile(r"\d+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _match_braces(text, i):
    """Return text inside the brace group starting at text[i] == '{', or None."""
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def extract_boxed(text):
    """Content of the last complete \\boxed{...} (nested-brace aware), or None."""
    if not isinstance(text, str):
        return None
    for m in reversed(list(_BOXED_RE.finditer(text))):
        content = _match_braces(text, m.end() - 1)
        if content is not None:
            return content
    return None


def canonicalize_answer(raw):
    """Normalize a gold answer; return None if it reduces to empty."""
    s = raw
    for cmd in _SPACING_CMDS:
        s = s.replace(cmd, "")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\s+", "", s)
    if _GROUPED_RE.fullmatch(s):
        s = s.replace(",", "")
    if _INT_RE.fullmatch(s):
        s = str(int(s))  # drop leading zeros on integer answers
    return s or None


def canonicalize_aime_answer(raw):
    """AIME golds are integers 0-999; some sources append units like 336^\\circ."""
    s = canonicalize_answer(raw)
    if not s:
        return None
    m = _LEADING_INT_RE.match(s)
    return str(int(m.group(0))) if m else None


def is_numeric(answer):
    return bool(_NUMERIC_RE.fullmatch(answer))


def normalize_problem(text):
    """Lowercase, drop ASCII punctuation and whitespace, for dedup keys."""
    return re.sub(r"\s+", "", text.lower().translate(_PUNCT_TABLE))


def load_numina(raw_dir):
    frames = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "numina-train-*.parquet"))):
        frames.append(pq.read_table(path, columns=["source", "problem", "solution"]).to_pandas())
    if not frames:
        sys.exit(f"no numina-train-*.parquet under {raw_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["source"].isin(WANTED_SOURCES)].copy()
    # pandas 3.0 arrow backend turns a None map result into float nan, so guard
    # on isinstance rather than `is not None`.
    df["answer"] = df["solution"].map(extract_boxed).map(
        lambda x: canonicalize_answer(x) if isinstance(x, str) else None
    )
    df = df[df["answer"].notna()].copy()
    df["norm"] = df["problem"].map(normalize_problem)
    df = df[df["norm"] != ""]
    return df.drop_duplicates(subset="norm", keep="first").reset_index(drop=True)


def load_aime2025(raw_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "aime2025-*.jsonl"))):
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                ans = canonicalize_aime_answer(str(obj["answer"]))
                if ans:
                    rows.append(
                        {"problem": obj["question"], "answer": ans, "source": "aime_2025"}
                    )
    return pd.DataFrame(rows)


def load_aime2024(raw_dir):
    path = os.path.join(raw_dir, "aimo-validation-aime.parquet")
    df = pq.read_table(path).to_pandas()
    df = df[df["url"].str.contains("2024_AIME", regex=False)]
    return pd.DataFrame(
        {
            "problem": df["problem"].to_numpy(),
            "answer": [canonicalize_aime_answer(str(a)) for a in df["answer"]],
            "source": "aime_2024",
        }
    )


def write_jsonl(df, path):
    with open(path, "w") as f:
        for row in df.itertuples(index=False):
            f.write(
                json.dumps(
                    {"text": row.problem, "answer": row.answer},
                    ensure_ascii=False,
                )
                + "\n"
            )


def report(name, df):
    print(f"== {name}: {len(df)} rows ==")
    print("per-source:")
    for source, count in df["source"].value_counts().sort_index().items():
        print(f"  {source}: {count}")
    numeric = df["answer"].map(is_numeric)
    print(f"answer types: numeric={int(numeric.sum())} symbolic={int((~numeric).sum())}")
    pl = df["problem"].str.len()
    print(
        f"problem chars: mean={pl.mean():.0f} p50={pl.median():.0f} "
        f"p95={pl.quantile(0.95):.0f} max={pl.max()}"
    )
    if "solution" in df:
        sl = df["solution"].dropna().str.len()
        if len(sl):
            print(
                f"solution chars: n={len(sl)} mean={sl.mean():.0f} p50={sl.median():.0f} "
                f"p95={sl.quantile(0.95):.0f} max={sl.max()}"
            )
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/host/opd-corpora/math-reasoning/raw")
    ap.add_argument("--out-dir", default="/host/opd-corpora/math-reasoning")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    numina = load_numina(args.raw_dir)
    print("numina pool after extraction+dedup:")
    for source, count in numina["source"].value_counts().sort_index().items():
        print(f"  {source}: {count}")
    print()

    aime25 = load_aime2025(args.raw_dir)
    aime24 = load_aime2024(args.raw_dir)
    for df in (aime25, aime24):
        df["norm"] = df["problem"].map(normalize_problem)

    # Eval: AIME gold sets first, then hard NuminaMath sources not already used.
    eval_frames = [aime25, aime24]
    used = set(aime25["norm"]) | set(aime24["norm"])
    rng = np.random.default_rng(SEED)
    for source, quota in EVAL_NUMINA_QUOTA.items():
        pool = numina[(numina["source"] == source) & (~numina["norm"].isin(used))]
        picked = pool.sample(n=min(quota, len(pool)), random_state=rng)
        eval_frames.append(picked)
        used.update(picked["norm"])
    eval_df = pd.concat(eval_frames, ignore_index=True).drop_duplicates(
        subset="norm", keep="first"
    )

    # Train: exclude every eval problem, then sample per-source quotas.
    train_pool = numina[~numina["norm"].isin(used)]
    train_frames = []
    for source, quota in TRAIN_QUOTA.items():
        pool = train_pool[train_pool["source"] == source]
        if len(pool) < quota:
            print(f"WARN: {source} pool {len(pool)} < quota {quota}", file=sys.stderr)
        train_frames.append(pool.sample(n=min(quota, len(pool)), random_state=rng))
    train_df = pd.concat(train_frames, ignore_index=True)

    write_jsonl(train_df, os.path.join(args.out_dir, "train.jsonl"))
    write_jsonl(eval_df, os.path.join(args.out_dir, "eval.jsonl"))

    report("train", train_df)
    report("eval", eval_df)

    overlap = set(train_df["norm"]) & set(eval_df["norm"])
    print(f"train/eval normalized-problem overlap: {len(overlap)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fetch hard, verifiable task corpora from HuggingFace for the OPD rollout lane.

Runs fully locally; honors HF_ENDPOINT (use https://hf-mirror.com). Streams
parquet shards via hf:// so only the rows consumed are downloaded — no full
snapshot pulls. Idempotent: an existing non-empty output file is skipped
(--force re-pulls). Cleaning lives in scripts/clean_opd_corpora.py.

Survey verdict (2026-07 — license must permit training use, python-tractable
without Docker preferred; see scripts/stage_swe_pro.py for why the pod is
Docker-free):
  KEEP  SWE-Gym/SWE-Gym        MIT        2,438 rows  repo+tests, SweTask lane
  KEEP  SWE-bench/SWE-smith    MIT       59,136 rows  synthetic repo tasks, capped
  KEEP  BAAI/TACO              Apache-2.0 ~25k rows   problem + I/O unit tests
  KEEP  deepmind/code_contests CC-BY-4.0  ~13k rows   problem + I/O unit tests
  DROP  KodCode/KodCode-V1     CC-BY-NC-4.0 (training-restricted)
  DROP  R2E-Gym/R2E-Gym-V1     no license
  DROP  livecodebench/*        `cc` (ambiguous) + live eval set — keep for eval only
  DROP  Multi-SWE-bench        mostly non-python; python lane already covered
  DROP  open-thoughts/*        reasoning traces, not verifiable tasks

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  python3 scripts/fetch_opd_corpora.py [--only NAME] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RAW_DIR = Path("data/opd-corpora/raw")

# name -> (hf glob, row cap, columns to keep; None = all).
# Row caps keep each pull well under the 2 GB budget (code_contests rows are
# huge: solutions + generated_tests are dropped below, but the shard bytes
# still stream through — the cap bounds that).
SOURCES = {
    "swe_gym": {
        "data_files": "hf://datasets/SWE-Gym/SWE-Gym/data/train-*.parquet",
        "cap": 3000,
        "columns": [
            "instance_id", "problem_statement", "repo", "base_commit",
            "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "version",
        ],
    },
    "swe_smith": {
        "data_files": "hf://datasets/SWE-bench/SWE-smith/data/train-*.parquet",
        "cap": 20000,
        "columns": [
            "instance_id", "problem_statement", "repo", "base_commit",
            "patch", "FAIL_TO_PASS", "PASS_TO_PASS", "image_name",
        ],
    },
    "taco": {
        "data_files": "hf://datasets/BAAI/TACO/ALL/train-*.parquet",
        "cap": 15000,
        "columns": [
            "question", "input_output", "difficulty", "raw_tags", "source",
            "starter_code", "time_limit", "memory_limit",
        ],
    },
    "code_contests": {
        "data_files": "hf://datasets/deepmind/code_contests/data/train-*.parquet",
        "cap": 6000,
        "columns": [
            "name", "description", "public_tests", "private_tests",
            "difficulty", "cf_rating", "source", "time_limit", "memory_limit_bytes",
        ],
    },
}


def _jsonable(value):
    """Arrow scalars arrive as numpy types / bytes; make them JSON-safe."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def fetch(name: str, spec: dict, force: bool) -> tuple[int, int]:
    from datasets import load_dataset

    out = RAW_DIR / f"{name}.jsonl"
    if out.exists() and out.stat().st_size > 0 and not force:
        rows = sum(1 for _ in out.open())
        print(f"[{name}] SKIP (exists: {rows} rows, {out.stat().st_size:,} bytes)")
        return rows, out.stat().st_size

    print(f"[{name}] streaming {spec['data_files']} (cap {spec['cap']} rows)")
    ds = load_dataset(
        "parquet", data_files={"train": spec["data_files"]},
        split="train", streaming=True,
    )
    cols = spec["columns"]
    tmp = out.with_suffix(".jsonl.tmp")
    rows = 0
    with tmp.open("w") as f:
        for row in ds:
            rec = {c: _jsonable(row.get(c)) for c in cols}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows += 1
            if rows % 2000 == 0:
                print(f"[{name}]   {rows} rows, {tmp.stat().st_size:,} bytes")
            if rows >= spec["cap"]:
                break
    tmp.rename(out)
    print(f"[{name}] DONE {rows} rows, {out.stat().st_size:,} bytes -> {out}")
    return rows, out.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=sorted(SOURCES), help="fetch a single source")
    ap.add_argument("--force", action="store_true", help="re-pull existing outputs")
    args = ap.parse_args()

    if "hf-mirror" not in os.environ.get("HF_ENDPOINT", ""):
        print("WARN: HF_ENDPOINT is not hf-mirror.com — set "
              "`export HF_ENDPOINT=https://hf-mirror.com` for mirror access",
              file=sys.stderr)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    names = [args.only] if args.only else sorted(SOURCES)
    total_rows = total_bytes = 0
    for name in names:
        rows, nbytes = fetch(name, SOURCES[name], args.force)
        total_rows += rows
        total_bytes += nbytes
    print(f"\nTOTAL {total_rows} rows, {total_bytes:,} bytes across {len(names)} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())

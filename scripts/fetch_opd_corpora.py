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
import contextlib
import fnmatch
import json
import os
import re
import shutil
import sys
from pathlib import Path

RAW_DIR = Path("data/opd-corpora/raw")

# name -> (hf glob, row cap, columns to keep). Reads go through pyarrow with
# column projection over HfFileSystem range requests, so dropped columns
# (code_contests solutions/generated_tests = the bulk of its 2.2 GB parquet)
# are never downloaded; row caps bound the rest well under the 2 GB budget.
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
        # No PASS_TO_PASS: ~69 KB/row of test names (2.3 GB over the cap'd
        # pull) that the SweTask lane never reads. No base_commit column
        # exists — tasks are branches of swesmith/<repo>.<shortsha> mirrors.
        "columns": [
            "instance_id", "problem_statement", "repo",
            "patch", "FAIL_TO_PASS", "image_name",
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
    # Whole-shard GETs (~760 KB/s each on the mirror, 3 in flight) beat fsspec
    # streaming (~130 KB/s) and projected range reads (~9 KB/s) by 6-80x;
    # column projection then happens locally and shards are deleted right after.
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    import requests
    from huggingface_hub import list_repo_files

    out = RAW_DIR / f"{name}.jsonl"
    if out.exists() and out.stat().st_size > 0 and not force:
        rows = sum(1 for _ in out.open())
        print(f"[{name}] SKIP (exists: {rows} rows, {out.stat().st_size:,} bytes)")
        return rows, out.stat().st_size

    org, repo = spec["data_files"].removeprefix("hf://datasets/").split("/")[:2]
    repo_id = f"{org}/{repo}"
    prefix = spec["data_files"].split(repo_id + "/")[1]
    shard_re = re.compile(fnmatch.translate(prefix))
    shards = sorted(
        f for f in list_repo_files(repo_id, repo_type="dataset") if shard_re.match(f)
    )
    print(f"[{name}] {len(shards)} shards in {repo_id}, cap {spec['cap']} rows",
          flush=True)

    cache = RAW_DIR / ".hf-cache"
    cache.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

    def pull(shard: str) -> Path:
        # Plain resolve-URL GET with Range resume: hf_hub_download's metadata
        # HEAD 401s on the mirror, and mirror connections drop mid-stream.
        local = cache / shard.replace("/", "__")
        url = f"{endpoint}/datasets/{repo_id}/resolve/main/{shard}"
        local.write_bytes(b"")
        for attempt in range(12):
            headers = {"Range": f"bytes={local.stat().st_size}-"}
            try:
                with requests.get(url, stream=True, allow_redirects=True,
                                  timeout=60, headers=headers) as r:
                    if r.status_code == 416:  # already complete
                        return local
                    r.raise_for_status()
                    with local.open("ab") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                return local
            except requests.RequestException as e:
                print(f"[{name}]   retry {attempt + 1} {shard}: {e}", flush=True)
        raise RuntimeError(f"download failed after retries: {url}")

    cols = spec["columns"]
    tmp = out.with_suffix(".jsonl.tmp")
    rows = dl_bytes = 0
    with tmp.open("w") as f, ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(pull, s) for s in shards[:3]]
        next_shard = 3
        while pending and rows < spec["cap"]:
            local = pending.pop(0).result()
            dl_bytes += local.stat().st_size
            for batch in pq.ParquetFile(local).iter_batches(batch_size=512, columns=cols):
                for rec in batch.to_pylist():
                    f.write(json.dumps(_jsonable(rec), ensure_ascii=False) + "\n")
                    rows += 1
                    if rows >= spec["cap"]:
                        break
                if rows >= spec["cap"]:
                    break
            local.unlink()
            print(f"[{name}]   {rows} rows, downloaded {dl_bytes:,} bytes",
                  flush=True)
            if next_shard < len(shards) and rows < spec["cap"]:
                pending.append(pool.submit(pull, shards[next_shard]))
                next_shard += 1
        for fut in pending:  # cap reached: drain prefetched shards
            with contextlib.suppress(Exception):
                fut.result().unlink()
    shutil.rmtree(cache, ignore_errors=True)
    tmp.rename(out)
    print(f"[{name}] DONE {rows} rows, {out.stat().st_size:,} bytes jsonl "
          f"({dl_bytes:,} bytes downloaded) -> {out}")
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

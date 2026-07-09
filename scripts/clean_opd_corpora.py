#!/usr/bin/env python3
"""Multi-round cleaner for the OPD rollout corpora fetched by fetch_opd_corpora.py.

Three rounds, each printing kept/dropped tallies per rule — no silent caps:
  1 QUALITY       parse + required fields, size bounds, exact dedup (sha256),
                  near-dup (minhash LSH + shingle-jaccard > 0.8)
  2 SECURITY      scripts/opd_security_filter.is_security_flagged over EVERY
                  text field; any match drops the row (err on dropping)
  3 VERIFIABILITY tests runnable in principle + non-trivial; difficulty bucket;
                  10 random survivors spot-printed per dataset

Output lanes (data/opd-corpora/cleaned/<name>.jsonl):
  swe_gym, swe_smith — SweTask schema (crates/train/src/swe_dataset.rs):
    instance_id, problem_statement, repo, base_commit, test_patch,
    fail_to_pass, selected_test_files_to_run, before_repo_set_cmd,
    requirements; plus gold_patch (scoring gate, stage_swe_pro.py precedent),
    pass_to_pass, difficulty. swe_smith has no test_patch — its FAIL_TO_PASS
    tests already exist in the repo at base_commit (test_patch = "").
  taco, code_contests — function-level lane (stdin/stdout or fn-call problems
    verified by a python sandbox runner, no repo checkout):
    {task_id, statement, tests: {inputs: [..], outputs: [..], fn_name?},
     starter_code, difficulty, source, lane: "function"}

Usage:
  python3 scripts/clean_opd_corpora.py [--only NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from opd_security_filter import is_security_flagged  # noqa: E402

RAW_DIR = Path("data/opd-corpora/raw")
CLEAN_DIR = Path("data/opd-corpora/cleaned")

MAX_STATEMENT_CHARS = 128_000  # ~32K tokens at ~4 chars/token
MIN_STATEMENT_CHARS = 40
NEAR_DUP_JACCARD = 0.8
MINHASH_PERMS = 64
LSH_BANDS = 8  # 8 bands x 8 rows
_MERSENNE = (1 << 61) - 1


# ------------------------------------------------------------- near-dup ----

def _shingles(text: str) -> set[int]:
    words = re.sub(r"\s+", " ", text.lower()).split()
    return {
        zlib.crc32(" ".join(words[i : i + 5]).encode())
        for i in range(max(1, len(words) - 4))
    }


class NearDupIndex:
    """Minhash-LSH; candidate pairs verified by true shingle jaccard."""

    def __init__(self) -> None:
        rng = random.Random(0)
        self._perms = [
            (rng.randrange(1, _MERSENNE), rng.randrange(_MERSENNE))
            for _ in range(MINHASH_PERMS)
        ]
        self._bands: list[dict[tuple, list[int]]] = [{} for _ in range(LSH_BANDS)]
        self._shingle_sets: list[set[int]] = []

    def is_dup(self, text: str) -> bool:
        """True if a previously-added doc has jaccard > threshold; else adds."""
        sh = _shingles(text)
        sig = [
            min((a * h + b) % _MERSENNE for h in sh) if sh else 0
            for a, b in self._perms
        ]
        rows = MINHASH_PERMS // LSH_BANDS
        keys = [tuple(sig[i * rows : (i + 1) * rows]) for i in range(LSH_BANDS)]
        candidates = {i for k, band in zip(keys, self._bands) for i in band.get(k, ())}
        for i in candidates:
            other = self._shingle_sets[i]
            inter = len(sh & other)
            union = len(sh) + len(other) - inter
            if union and inter / union > NEAR_DUP_JACCARD:
                return True
        idx = len(self._shingle_sets)
        self._shingle_sets.append(sh)
        for k, band in zip(keys, self._bands):
            band.setdefault(k, []).append(idx)
        return False


# ------------------------------------------------------ per-dataset shape ----

def _swe_task(row: dict, name: str) -> dict:
    return {
        "instance_id": row["instance_id"],
        "problem_statement": row["problem_statement"],
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "test_patch": row.get("test_patch") or "",
        "fail_to_pass": row.get("FAIL_TO_PASS") or [],
        "selected_test_files_to_run": sorted(
            {t.split("::")[0] for t in (row.get("FAIL_TO_PASS") or [])}
        ),
        "before_repo_set_cmd": None,
        "requirements": None,
        "gold_patch": row.get("patch") or "",
        "pass_to_pass": row.get("PASS_TO_PASS") or [],
        "source": name,
    }


def _parse_tests(row: dict, name: str) -> dict | None:
    """Normalize to {inputs, outputs, fn_name?}; None if absent/unparseable."""
    if name == "taco":
        raw = row.get("input_output")
        if not raw:
            return None
        try:
            io = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(io, dict) or not io.get("inputs"):
            return None
        tests = {"inputs": io["inputs"], "outputs": io.get("outputs") or []}
        if io.get("fn_name"):
            tests["fn_name"] = io["fn_name"]
        return tests
    pub, priv = row.get("public_tests") or {}, row.get("private_tests") or {}
    inputs = list(pub.get("input") or []) + list(priv.get("input") or [])
    outputs = list(pub.get("output") or []) + list(priv.get("output") or [])
    return {"inputs": inputs, "outputs": outputs} if inputs else None


def _function_task(row: dict, name: str, idx: int) -> dict:
    statement = row.get("question") if name == "taco" else row.get("description")
    return {
        "task_id": f"{name}__{idx}",
        "statement": statement or "",
        "tests": _parse_tests(row, name),
        "starter_code": row.get("starter_code") or "",
        "difficulty": row.get("difficulty"),
        "source": row.get("source"),
        "lane": "function",
    }


def _statement(rec: dict) -> str:
    return rec.get("problem_statement") or rec.get("statement") or ""


def _all_text(rec: dict) -> str:
    """Every string field, recursively — security round scans all of it."""
    parts: list[str] = []

    def walk(v):
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(rec)
    return "\n".join(parts)


# ----------------------------------------------------------------- rounds ----

def round1_quality(records: list[dict], name: str) -> list[dict]:
    tallies: Counter = Counter()
    seen_hashes: set[str] = set()
    near_dup = NearDupIndex()
    kept = []
    for rec in records:
        stmt = _statement(rec)
        if not stmt or (name in ("swe_gym", "swe_smith") and not (
            rec["instance_id"] and rec["repo"] and rec["base_commit"]
        )):
            tallies["missing-required-field"] += 1
            continue
        if len(stmt) < MIN_STATEMENT_CHARS:
            tallies["statement-too-short"] += 1
            continue
        if len(stmt) > MAX_STATEMENT_CHARS:
            tallies["statement-oversized"] += 1
            continue
        digest = hashlib.sha256(re.sub(r"\s+", " ", stmt.lower()).encode()).hexdigest()
        if digest in seen_hashes:
            tallies["exact-dup"] += 1
            continue
        if near_dup.is_dup(stmt):
            tallies["near-dup"] += 1
            continue
        seen_hashes.add(digest)
        tallies["kept"] += 1
        kept.append(rec)
    _print_tallies(name, "1 QUALITY", tallies)
    return kept


def round2_security(records: list[dict], name: str) -> list[dict]:
    tallies: Counter = Counter()
    kept = []
    for rec in records:
        rule = is_security_flagged(_all_text(rec))
        if rule:
            tallies[f"flagged:{rule}"] += 1
            continue
        tallies["kept"] += 1
        kept.append(rec)
    _print_tallies(name, "2 SECURITY", tallies)
    return kept


def _difficulty_bucket(rec: dict, name: str) -> str:
    if name in ("swe_gym", "swe_smith"):
        score = (
            (len(rec["fail_to_pass"]) > 3)
            + (rec["gold_patch"].count("\n") > 80)
            + (len(rec["problem_statement"]) > 4000)
        )
    else:
        n_tests = len(rec["tests"]["inputs"])
        rating = rec.get("difficulty")
        hard_rating = (
            rating in ("HARD", "VERY_HARD")
            or (isinstance(rating, int) and rating >= 3)
        )
        score = (n_tests > 20) + hard_rating + (len(rec["statement"]) > 3000)
    return ("easy", "medium", "hard")[min(score, 2)]


def round3_verifiability(records: list[dict], name: str) -> list[dict]:
    tallies: Counter = Counter()
    kept = []
    for rec in records:
        if name in ("swe_gym", "swe_smith"):
            if not rec["fail_to_pass"]:
                tallies["no-fail-to-pass-tests"] += 1
                continue
            if name == "swe_gym" and len(rec["test_patch"]) < 50:
                tallies["test-patch-trivial"] += 1
                continue
            if len(rec["gold_patch"]) < 20:
                tallies["gold-patch-trivial"] += 1
                continue
        else:
            tests = rec["tests"]
            if not tests:
                tallies["no-tests"] += 1
                continue
            n = min(len(tests["inputs"]), len(tests["outputs"]))
            if n < 3:
                tallies["fewer-than-3-test-cases"] += 1
                continue
            if sum(len(str(x)) for x in tests["inputs"][:3]) < 3:
                tallies["trivial-test-inputs"] += 1
                continue
        rec["difficulty_bucket"] = _difficulty_bucket(rec, name)
        tallies["kept"] += 1
        kept.append(rec)
    _print_tallies(name, "3 VERIFIABILITY", tallies)
    buckets = Counter(rec["difficulty_bucket"] for rec in kept)
    print(f"  [{name}] difficulty buckets: " + ", ".join(
        f"{b}={buckets[b]}" for b in ("easy", "medium", "hard")))

    print(f"  [{name}] spot-check (10 random survivors):")
    for rec in random.Random(0).sample(kept, min(10, len(kept))):
        rid = rec.get("instance_id") or rec.get("task_id")
        stmt = re.sub(r"\s+", " ", _statement(rec))[:160]
        extra = (
            f"f2p={len(rec['fail_to_pass'])}"
            if "fail_to_pass" in rec
            else f"tests={len(rec['tests']['inputs'])}"
        )
        print(f"    {rid} [{rec['difficulty_bucket']}] {extra} | {stmt}")
    return kept


def _print_tallies(name: str, round_name: str, tallies: Counter) -> None:
    total = sum(tallies.values())
    print(f"  [{name}] round {round_name}: {tallies['kept']}/{total} kept")
    for rule, n in sorted(tallies.items()):
        if rule != "kept":
            print(f"    dropped {rule}: {n}")


# ------------------------------------------------------------------- main ----

def clean(name: str) -> tuple[int, int]:
    raw = RAW_DIR / f"{name}.jsonl"
    print(f"\n=== {name} ({raw})")
    records = []
    for i, line in enumerate(raw.open()):
        row = json.loads(line)
        records.append(
            _swe_task(row, name)
            if name in ("swe_gym", "swe_smith")
            else _function_task(row, name, i)
        )
    n_raw = len(records)
    records = round1_quality(records, name)
    records = round2_security(records, name)
    records = round3_verifiability(records, name)

    out = CLEAN_DIR / f"{name}.jsonl"
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [{name}] FINAL {len(records)}/{n_raw} rows -> {out}")
    return n_raw, len(records)


def main() -> int:
    names = sorted(p.stem for p in RAW_DIR.glob("*.jsonl"))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=names, help="clean a single dataset")
    args = ap.parse_args()
    if args.only:
        names = [args.only]
    if not names:
        print("no raw corpora found — run scripts/fetch_opd_corpora.py first",
              file=sys.stderr)
        return 1

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    results = [(n, *clean(n)) for n in names]
    print("\n=== SUMMARY")
    for name, n_raw, n_clean in results:
        print(f"  {name}: {n_clean}/{n_raw} rows survived")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""SWE-bench_Pro corpus staging for the agentic-OPD phase-2 curve.

Two subcommands running on different machines:

  select  (local, HF access)  — pick python-language candidates from
          ScaleAI/SWE-bench_Pro and emit a SweTask-schema JSONL
          (crates/train/src/swe_dataset.rs field names + gold_patch for the
          pod-side scoring gate).
  stage   (pod, GitHub access) — clone each candidate at base_commit, run the
          self-check gate (test_patch alone must FAIL, gold_patch on top must
          PASS — mirrors scripts/cc_swe_baseline.py score()), and land accepted
          trees at <staged-root>/<instance_id>/ plus train/eval JSONL rows.

Usage:
  python3 scripts/stage_swe_pro.py select --out candidates.jsonl --limit 60
  python3 scripts/stage_swe_pro.py stage --candidates candidates.jsonl \
      --staged-root /host/staged --train-out train.jsonl --eval-out eval.jsonl \
      --train-n 12 --eval-n 12

Plan: docs/plans/2026-07-03-agentic-opd-27b-capability-curve.md (phase 2).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PYTEST_TIMEOUT = 300
SETUP_TIMEOUT = 600
# Repos known to need pre-3.12 interpreters on the pod (python3.12-only box).
EXCLUDE_REPO_RE = re.compile(r"ansible", re.IGNORECASE)


# ---------------------------------------------------------------- select ----

# SweTask schema (crates/train/src/swe_dataset.rs) + repo/base_commit for the
# stager + gold_patch (dataset column `patch`) for the pod-side scoring gate.
CANDIDATE_FIELDS = [
    "instance_id", "problem_statement", "repo", "base_commit", "test_patch",
    "fail_to_pass", "selected_test_files_to_run", "before_repo_set_cmd",
    "requirements",
]


def _as_list(value):
    """Dataset list columns are inconsistently encoded: JSON array string
    (NodeBB-style `["a"]`) or Python-repr (openlibrary-style `['a']`).
    Normalize to a real list so json.loads / the Rust normalizer never see
    single-quoted pseudo-JSON."""
    if isinstance(value, list):
        return value
    s = (value or "").strip()
    if not s:
        return []
    for parse in (json.loads, __import__("ast").literal_eval):
        try:
            out = parse(s)
            if isinstance(out, list):
                return [str(x) for x in out]
        except (ValueError, SyntaxError):
            pass
    return [s]


def _as_text(value):
    """Some text columns are double-JSON-encoded (`"\\"**Title..\\""`)."""
    s = value or ""
    if s.startswith('"'):
        try:
            out = json.loads(s)
            if isinstance(out, str):
                return out
        except ValueError:
            pass
    return s


def cmd_select(args):
    try:
        from datasets import load_dataset
        rows = list(load_dataset("ScaleAI/SWE-bench_Pro", split="test"))
    except ImportError:
        rows = _load_via_hub()

    python_rows = [r for r in rows if r.get("repo_language") == "python"]
    kept = [r for r in python_rows if not EXCLUDE_REPO_RE.search(r["repo"])]
    print(f"[select] dataset rows={len(rows)} python={len(python_rows)} "
          f"after-exclude={len(kept)}", flush=True)

    # Round-robin across repos so the candidate list is repo-diverse; the pod
    # gate over-rejects, so over-select (default 60 for a 12+12 target).
    by_repo = {}
    for r in kept:
        by_repo.setdefault(r["repo"], []).append(r)
    picked, i = [], 0
    while len(picked) < args.limit and any(by_repo.values()):
        for repo in sorted(by_repo):
            if by_repo[repo] and len(picked) < args.limit:
                picked.append(by_repo[repo].pop(0))
        i += 1

    with open(args.out, "w") as f:
        for r in picked:
            row = {k: r.get(k) for k in CANDIDATE_FIELDS}
            row["fail_to_pass"] = _as_list(row["fail_to_pass"])
            row["selected_test_files_to_run"] = \
                _as_list(row["selected_test_files_to_run"])
            row["problem_statement"] = _as_text(row["problem_statement"])
            row["requirements"] = _as_text(row["requirements"])
            row["gold_patch"] = r.get("patch", "")
            f.write(json.dumps(row) + "\n")
    print(f"[select] wrote {len(picked)} candidates -> {args.out}", flush=True)
    if picked:
        first = {k: str(v)[:100] for k, v in picked[0].items()
                 if k in CANDIDATE_FIELDS or k == "patch"}
        print("[select] first candidate fields:", json.dumps(first, indent=2),
              flush=True)


def _load_via_hub():
    """Fallback without the `datasets` lib: download the parquet data files
    via huggingface_hub and read them with pyarrow/pandas."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = [f for f in api.list_repo_files("ScaleAI/SWE-bench_Pro",
                                            repo_type="dataset")
             if f.endswith(".parquet")]
    if not files:
        sys.exit("[select] no parquet data files in ScaleAI/SWE-bench_Pro; "
                 "install the `datasets` lib instead")
    rows = []
    import pandas as pd
    for f in files:
        path = hf_hub_download("ScaleAI/SWE-bench_Pro", f, repo_type="dataset")
        rows.extend(pd.read_parquet(path).to_dict("records"))
    return rows


# ----------------------------------------------------------------- stage ----

def run(cmd, cwd, timeout=None, env=None):
    e = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, check=False, env=e)


def clone_at_commit(repo, sha, dest):
    """Shallow-fetch `sha` (may not be a branch head) into `dest`. Every git
    command runs with cwd inside `dest` — never the launch cwd (the pod's
    default cwd has a poisoned .git/config auth extraheader)."""
    dest.mkdir(parents=True)
    url = f"https://github.com/{repo}.git"
    for cmd in (["git", "init", "-q"],
                ["git", "remote", "add", "origin", url],
                ["git", "fetch", "--depth", "1", "origin", sha],
                ["git", "checkout", "-q", "FETCH_HEAD"]):
        r = run(cmd, dest, timeout=SETUP_TIMEOUT)
        if r.returncode != 0:
            return f"{' '.join(cmd[:3])} rc={r.returncode}: {r.stderr.strip()[:200]}"
    return None


def apply_patch(workdir, patch, name):
    (workdir / name).write_text(patch)
    r = run(["git", "apply", name], workdir)
    (workdir / name).unlink(missing_ok=True)
    if r.returncode != 0:
        return f"git apply {name} rc={r.returncode}: {r.stderr.strip()[:200]}"
    return None


def fail_to_pass_list(task):
    f2p = task.get("fail_to_pass", [])
    if isinstance(f2p, str):
        f2p = json.loads(f2p) if f2p.strip().startswith("[") else [f2p]
    return f2p


def run_pytest(workdir, f2p, python):
    """Returns (rc, summary_tail). rc None on timeout."""
    try:
        r = run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", *f2p],
                workdir, timeout=PYTEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, "pytest timeout"
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
    return r.returncode, tail[0][:200]


def self_check(repo_dir, task, scratch, python):
    """Mirror cc_swe_baseline.py score() semantics in a scratch copy:
    test_patch alone must FAIL with real test failures (pytest rc 1, no
    collection errors), gold_patch on top must PASS (rc 0)."""
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(repo_dir, scratch, symlinks=True)
    f2p = fail_to_pass_list(task)
    if not f2p:
        return False, "empty fail_to_pass"

    err = apply_patch(scratch, task.get("test_patch", ""), ".t.diff")
    if err:
        return False, err
    rc, tail = run_pytest(scratch, f2p, python)
    if rc is None:
        return False, f"pre-fix {tail}"
    if rc == 0:
        return False, f"pre-fix already passes: {tail}"
    # rc 2/3/4/5 = interrupted / internal / usage / nothing collected; rc 1
    # with an error summary = collection error dressed as a failure.
    if rc != 1 or "error" in tail.lower():
        return False, f"broken env (rc={rc}): {tail}"

    err = apply_patch(scratch, task.get("gold_patch", ""), ".g.diff")
    if err:
        return False, err
    rc, tail = run_pytest(scratch, f2p, python)
    if rc != 0:
        return False, f"gold does not pass (rc={rc}): {tail}"
    return True, tail


def load_accepted(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def cmd_stage(args):
    candidates = [json.loads(l) for l in
                  args.candidates.read_text().splitlines() if l.strip()]
    args.staged_root.mkdir(parents=True, exist_ok=True)

    # Resumable: instance_ids already in an out file count toward its quota.
    train_rows = load_accepted(args.train_out)
    eval_rows = load_accepted(args.eval_out)
    done = {r["instance_id"] for r in train_rows + eval_rows}
    accepted = rejected = 0

    for task in candidates:
        if len(train_rows) >= args.train_n and len(eval_rows) >= args.eval_n:
            break
        iid = task["instance_id"]
        if iid in done:
            print(f"[stage] {iid}: SKIP already staged", flush=True)
            continue
        staged = args.staged_root / iid
        if staged.exists():  # partial from a previous run — redo
            shutil.rmtree(staged)

        tmp = Path(tempfile.mkdtemp(prefix=f"{iid}.", dir=args.staged_root))
        repo_dir, scratch = tmp / "repo", tmp / "scratch"
        try:
            err = clone_at_commit(task["repo"], task["base_commit"], repo_dir)
            if err:
                rejected += 1
                print(f"[stage] {iid}: REJECT clone: {err}", flush=True)
                continue

            setup = (task.get("before_repo_set_cmd") or "").strip()
            if setup:
                try:
                    r = run(["bash", "-lc", setup], repo_dir,
                            timeout=SETUP_TIMEOUT)
                    if r.returncode != 0:
                        print(f"[stage] {iid}: WARN before_repo_set_cmd "
                              f"rc={r.returncode}: {r.stderr.strip()[:200]}",
                              flush=True)
                except subprocess.TimeoutExpired:
                    print(f"[stage] {iid}: WARN before_repo_set_cmd timeout",
                          flush=True)

            ok, note = self_check(repo_dir, task, scratch, args.python)
            if not ok:
                rejected += 1
                print(f"[stage] {iid}: REJECT gate: {note}", flush=True)
                continue

            # Land the clean base tree without .git — the runner
            # (cc_swe_baseline.py boot_workdir / sandbox.rs) git-inits it.
            shutil.rmtree(repo_dir / ".git")
            repo_dir.rename(staged)

            row = dict(task)
            if len(train_rows) < args.train_n:
                train_rows.append(row)
                out, split = args.train_out, "train"
            else:
                eval_rows.append(row)
                out, split = args.eval_out, "eval"
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            done.add(iid)
            accepted += 1
            print(f"[stage] {iid}: ACCEPT -> {split} ({note})", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"[stage] summary: accepted={accepted} rejected={rejected} "
          f"train={len(train_rows)}/{args.train_n} "
          f"eval={len(eval_rows)}/{args.eval_n}", flush=True)
    if len(train_rows) < args.train_n or len(eval_rows) < args.eval_n:
        sys.exit("[stage] quotas NOT filled — feed more candidates")


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sel = sub.add_parser("select", help="pick candidates from HF (local)")
    sel.add_argument("--out", type=Path, default=Path("candidates.jsonl"))
    sel.add_argument("--limit", type=int, default=60)
    sel.set_defaults(fn=cmd_select)

    stg = sub.add_parser("stage", help="clone + gate + stage (pod)")
    stg.add_argument("--candidates", type=Path, required=True)
    stg.add_argument("--staged-root", type=Path, required=True)
    stg.add_argument("--train-out", type=Path, required=True)
    stg.add_argument("--eval-out", type=Path, required=True)
    stg.add_argument("--train-n", type=int, default=12)
    stg.add_argument("--eval-n", type=int, default=12)
    stg.add_argument("--python", default="python3",
                     help="gate interpreter (pod default python3 = 3.12)")
    stg.set_defaults(fn=cmd_stage)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

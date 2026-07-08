#!/usr/bin/env python3
"""Claude-Code-as-harness SWE baseline: pass@1 over staged SWE-Pro instances.

For each task in a SWE-Pro JSONL (same schema as `crates/train/src/
swe_dataset.rs`), boots a sandbox copy of the staged tree, lets the `claude`
CLI (pointed at an ARLE serve via ANTHROPIC_BASE_URL) attempt the fix, then
scores exactly like `sandbox.rs::score_workdir`: non-empty `git diff` AND
`git apply test_patch` + `python3 -m pytest <fail_to_pass>` exit-0.

Usage:
  ANTHROPIC_BASE_URL=http://127.0.0.1:8000 python3 scripts/cc_swe_baseline.py \
      --dataset /host/agent_opd_eval.jsonl --staged-root /host/eval_staged \
      --work-root /tmp/cc_baseline --max-turns 30 --out results.jsonl

The serve must run with --max-running-requests >= 2 (Claude Code fires
concurrent requests). Plan:
docs/plans/2026-07-03-agentic-opd-27b-capability-curve.md (phase 2, cc-as-harness).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(cmd, cwd, timeout=None, env=None):
    e = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, check=False, env=e)


def preflight(base):
    """Fail fast if the serve isn't up — else every task eats CC's ~38s backoff."""
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
            if r.status != 200:
                sys.exit(f"serve at {base} returned {r.status}, not 200")
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"serve not reachable at {base}/v1/models: {e}")


def cc_version():
    try:
        return subprocess.run(["claude", "--version"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def boot_workdir(staged, workdir, setup_cmd=None):
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(staged, workdir, symlinks=True)
    if setup_cmd:
        r = run(["bash", "-lc", setup_cmd], workdir, timeout=600)
        if r.returncode != 0:
            print(f"[cc-baseline] WARN before_repo_set_cmd rc={r.returncode}: "
                  f"{r.stderr.strip()[:200]}", flush=True)
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.email=a@b.c", "-c", "user.name=arle",
                 "commit", "-qm", "base"]):
        r = run(cmd, workdir)
        if r.returncode != 0:
            raise RuntimeError(f"git setup failed: {r.stderr.strip()}")


def score(workdir, task, test_timeout, pythonpath=None, python="python3"):
    diff = run(["git", "diff"], workdir).stdout
    if not diff.strip():
        return False, False, "no edits"
    patch = task.get("test_patch", "")
    if patch.strip():
        # Reset test files the agent may have dirtied (mirrors score_workdir).
        prev_minus = False
        for line in patch.splitlines():
            if prev_minus and line.startswith("+++ b/"):
                p = line[6:].strip()
                if p and p != "/dev/null":
                    run(["git", "checkout", "--", p], workdir)
            prev_minus = line.startswith("--- ")
        (workdir / ".t.diff").write_text(patch)
        r = run(["git", "apply", ".t.diff"], workdir)
        if r.returncode != 0:
            return False, True, f"test_patch apply failed: {r.stderr.strip()[:200]}"
    f2p = task.get("fail_to_pass", [])
    if isinstance(f2p, str):
        f2p = json.loads(f2p) if f2p.strip().startswith("[") else [f2p]
    env = {"PYTHONPATH": pythonpath} if pythonpath else None
    try:
        r = run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", *f2p],
                workdir, timeout=test_timeout, env=env)
    except subprocess.TimeoutExpired:
        return False, True, "pytest timeout"
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
    return r.returncode == 0, True, tail[0][:200]


def cc_attempt(workdir, task, args):
    prompt = (
        f"Fix a bug in this repository (cwd = repo root).\n\n"
        f"Problem statement:\n{task['problem_statement'][:3000]}\n\n"
        "Work in this order and be decisive:\n"
        "1. Briefly locate the buggy code (a few reads/greps — do NOT read the "
        "whole codebase).\n"
        "2. As soon as you have identified the fix, EDIT the source file. Do "
        "not keep exploring after you understand the bug — make the edit.\n"
        "3. Make the SMALLEST correct change that resolves the issue.\n\n"
        "You MUST edit at least one source file before you finish; an answer "
        "with no edit is a failure. Do not write or run the hidden tests (they "
        "are applied at scoring time). Do not commit."
    )
    # Allowlist keeps CC off WebFetch/WebSearch/Task — the sandbox is offline and
    # one web call stalls on CC's ~38s retry-backoff. No turn-cap flag exists in
    # CC 2.1.204; --cc-timeout is the runaway bound.
    cmd = ["claude", "-p", "--model", args.model,
           "--allowedTools", "Bash Read Write Edit Grep Glob",
           "--output-format", "json", "--dangerously-skip-permissions", prompt]
    # IS_SANDBOX=1 is mandatory: this container is root (uid 0) and
    # --dangerously-skip-permissions refuses under root without it.
    env = {"ANTHROPIC_API_KEY": "dummy-local", "IS_SANDBOX": "1"}
    try:
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                           timeout=args.cc_timeout,
                           env={**os.environ, **env}, check=False)
    except subprocess.TimeoutExpired:
        return {"error": "cc timeout"}
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"error": f"non-json cc output rc={r.returncode}: "
                         f"{(r.stderr or r.stdout)[-300:]}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--staged-root", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, default=Path("/tmp/cc_baseline"))
    ap.add_argument("--model", default="default")
    ap.add_argument("--max-turns", type=int, default=30,
                    help="no-op: CC 2.1.204 has no turn cap; bound via --cc-timeout")
    ap.add_argument("--cc-timeout", type=int, default=1800)
    ap.add_argument("--test-timeout", type=int, default=300)
    ap.add_argument("--task-limit", type=int)
    ap.add_argument("--pythonpath", default=None,
                    help="PYTHONPATH for scoring, e.g. lib:test (ansible)")
    ap.add_argument("--python", default="python3",
                    help="scoring interpreter (old trees may need <=3.11)")
    ap.add_argument("--samples", type=int, default=1,
                    help="attempts per task; workdir re-booted per attempt")
    ap.add_argument("--windows-out", type=Path, default=None,
                    help="JSONL of `arle train cc-convert --windows` rows for "
                         "PASSING attempts")
    ap.add_argument("--out", type=Path, default=Path("cc_baseline_results.jsonl"))
    args = ap.parse_args()

    base = os.environ.get("ANTHROPIC_BASE_URL")
    if not base:
        sys.exit("set ANTHROPIC_BASE_URL to the ARLE serve")
    preflight(base)  # fail fast instead of eating CC's ~38s backoff per task

    tasks = [json.loads(l) for l in args.dataset.read_text().splitlines()
             if l.strip()][: args.task_limit]
    # Config envelope — first line of the run, for reproducibility.
    print("[cc-baseline] " + json.dumps({
        "envelope": True, "base_url": base, "model": args.model,
        "cc_version": cc_version(), "cc_timeout": args.cc_timeout,
        "allowed_tools": "Bash Read Write Edit Grep Glob",
        "dataset": str(args.dataset), "tasks": len(tasks),
        "samples": args.samples,
    }), flush=True)
    results, windows = [], []
    for task in tasks:
        iid = task["instance_id"]
        staged = args.staged_root / iid
        workdir = args.work_root / iid
        for sample in range(args.samples):
            print(f"[cc-baseline] {iid}#{sample}: boot", flush=True)
            boot_workdir(staged, workdir, task.get("before_repo_set_cmd"))
            t_start_ms = int(time.time() * 1000)
            cc = cc_attempt(workdir, task, args)
            t_end_ms = int(time.time() * 1000)
            passed, edited, note = score(workdir, task, args.test_timeout,
                                         args.pythonpath, args.python)
            row = {
                "instance_id": iid, "sample": sample, "passed": passed,
                "edited": edited, "note": note,
                "t_start_ms": t_start_ms, "t_end_ms": t_end_ms,
                "wall_s": round((t_end_ms - t_start_ms) / 1000, 1),
                "cc_error": cc.get("error") or (cc.get("is_error") and cc.get("result")) or None,
                "cc_turns": cc.get("num_turns"),
                "cc_output_tokens": (cc.get("usage") or {}).get("output_tokens"),
            }
            results.append(row)
            if passed:
                windows.append({"label": f"{iid}#{sample}",
                                "t_start_ms": t_start_ms,
                                "t_end_ms": t_end_ms})
            print(f"[cc-baseline] {iid}#{sample}: passed={passed} "
                  f"edited={edited} turns={row['cc_turns']} "
                  f"wall={row['wall_s']}s :: {note}", flush=True)

    task_ids = list(dict.fromkeys(r["instance_id"] for r in results))
    n = len(task_ids)
    p1 = sum(1 for t in task_ids if any(
        r["passed"] for r in results
        if r["instance_id"] == t and r["sample"] == 0))
    p_any = sum(1 for t in task_ids if any(
        r["passed"] for r in results if r["instance_id"] == t))
    per_sample = [sum(1 for r in results
                      if r["sample"] == s and r["passed"])
                  for s in range(args.samples)]
    aggregate = {
        "aggregate": True, "tasks": n, "samples": args.samples,
        "passed": p1, "pass_rate": p1 / n if n else 0.0,  # legacy keys
        "pass_at_1": p1 / n if n else 0.0,
        "pass_at_any": p_any / n if n else 0.0,
        "per_sample_passed": per_sample,
    }
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(aggregate) + "\n")
    if args.windows_out:
        with open(args.windows_out, "w") as f:
            for w in windows:
                f.write(json.dumps(w) + "\n")
        print(f"[cc-baseline] {len(windows)} passing window(s) -> "
              f"{args.windows_out}", flush=True)
    print(f"[cc-baseline] pass@1={p1}/{n} pass@any={p_any}/{n} "
          f"per-sample={per_sample} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

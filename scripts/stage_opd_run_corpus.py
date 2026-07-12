#!/usr/bin/env python3
"""Stage the OPD run-1 task corpus (swe_gym + swe_smith + synthetic).

Thin orchestrator over stage_swe_pro.py primitives. Runs locally (network =
github clones + PyPI wheels for the gate venvs); the pod receives only the
staged-root tarball. Pipeline:

  1. Repo blocklist  — drop rows whose repo slug is security/auth-adjacent
     (rollouts on those repos inevitably discuss tokens/secrets), even though
     their statements passed the text filter.
  2. Slice selection — 50 swe_gym (tractable-install repos only, see
     GYM_TRACTABLE) + 100 swe_smith (easy/med/hard ~30/50/20), round-robin
     across repos, over-selected because the gate rejects.
  3. Stage + gate    — clone at ref (FETCH_HEAD handles swe_smith instance
     branches), strip CI/meta files (see META_DIRS/META_FILES — dead weight
     that carries `secrets.*`/SECURITY.md wording), security tree pre-scan
     over everything that remains, per-instance uv venv + editable
     install, then the self-check gate IN PLACE (test_patch alone FAILS,
     gold on top PASSES; both pytest runs share a 300s budget). Accepted
     trees are reverted to base (`git apply -R` + `git clean -fdx`), .git
     stripped, landed at <root>/staged/<instance_id>. Process pool, 8 wide.
     If a bucket's candidates exhaust below target, ship what passed — the
     gate is never relaxed.
  4. Split           — train/eval ~70/30 by REPO (no repo in both splits).
  5. Synthetic       — gen_agent_opd_tasks.py --difficulty medium
     (easy measured saturated: untrained 27B fixed 24/24) into the same root.
  6. Final sweep     — opd_security_filter over every staged file + emitted
     JSONL line; any hit drops that instance and re-emits; the CLI --scan is
     the last word and manifest.json records scan="clean". Rule-level drop
     detail goes to a droplog OUTSIDE the scanned root (rule names would
     re-flag the tree).

Usage:
  python3 scripts/stage_opd_run_corpus.py \
      --out-root data/opd-corpora/staged-run1 --workers 8

Plan: docs/plans/2026-07-03-agentic-opd-27b-capability-curve.md (phase 2).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_swe_pro as swe  # noqa: E402
from gen_agent_opd_tasks import new_file_diff  # noqa: E402
from opd_security_filter import is_security_flagged, scan_skips  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent

# Extra defense layer: repos whose SLUG is security/auth-adjacent — working in
# them inevitably surfaces tokens/secrets/credentials in rollouts even when
# the problem statement itself passed the text filter.
REPO_BLOCKLIST = [
    "oauthlib", "pyjwt", "jwt", "cryptography", "paramiko", "passlib",
    "itsdangerous", "keyring", "secret", "auth", "crypto", "ssl", "tls",
    "bandit", "safety",
]

# Tractable-install heuristic (swe_gym): pure-python `uv pip install -e .`
# with no build toolchain. Excluded families: pandas (cython build), MONAI
# (torch), modin (pandas/ray engines), bokeh (bundled JS assets).
GYM_TRACTABLE = {
    "getmoto/moto", "python/mypy", "iterative/dvc",
    "conan-io/conan", "pydantic/pydantic", "dask/dask",
}

# Best-effort test-dep manifests installed on top of `-e . pytest`.
REQ_FILES = [
    "requirements-dev.txt", "requirements-test.txt", "requirements-tests.txt",
    "test-requirements.txt", "requirements/test.txt", "requirements/dev.txt",
    "requirements/testing.txt",
]

GATE_BUDGET = 300      # s — both pytest runs combined, per instance
INSTALL_TIMEOUT = 600  # s — venv + installs, per instance
SMITH_MIX = {"medium": 0.6, "hard": 0.4}


# CI/meta files stripped from every staged tree BEFORE the security scan:
# rollouts never read them (the runner git-inits a fresh tree; CI never runs)
# and they are where `secrets.GITHUB_TOKEN` / SECURITY.md / bandit configs
# live — dropping the files, not relaxing the scan. Everything left is still
# scanned and any hit still drops the instance.
META_DIRS = {".github", ".circleci", ".azure-pipelines"}
META_FILES = {
    ".gitignore", ".gitattributes", ".travis.yml", ".gitlab-ci.yml",
    "appveyor.yml", ".appveyor.yml", "azure-pipelines.yml",
    ".pre-commit-config.yaml", ".bandit", ".bandit.yml", ".bandit.yaml",
    "codecov.yml", ".codecov.yml", "SECURITY.md",
}
# Binary media (demo GIFs/images, PDFs, video) bloats the staged tree ~20×
# (127 alive-progress instances each carried ~50 MB of img/ demos) and is never
# code or a test the model touches. Strip it OUTSIDE test paths — a test fixture
# that reads an image stays.
MEDIA_SUFFIXES = {
    ".gif", ".png", ".jpg", ".jpeg", ".bmp", ".ico", ".webp",
    ".pdf", ".mp4", ".mov", ".webm",
}


def blocklisted(repo: str) -> str | None:
    slug = repo.lower()
    return next((b for b in REPO_BLOCKLIST if b in slug), None)


def strip_meta(tree: Path):
    """Remove CI/meta files, then rebase the git base state onto the stripped
    tree so the post-gate `git status` restore check stays meaningful."""
    for p in tree.rglob("*"):
        if ".git" in p.parts[:-1] or not p.exists():
            continue
        if p.is_dir() and p.name in META_DIRS:
            shutil.rmtree(p)
        elif p.is_file() and p.name in META_FILES:
            p.unlink()
        elif (p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES
              and not any("test" in part.lower() for part in p.relative_to(tree).parts)):
            p.unlink()
    swe.run(["git", "add", "-A"], tree)
    swe.run(["git", "-c", "user.email=a@b.c", "-c", "user.name=arle",
             "commit", "-qm", "strip-meta", "--allow-empty"], tree)


def scan_tree(root: Path):
    """(rule, relpath) of the first security-flagged file, else (None, None)."""
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if ".git" in path.parts or scan_skips(path):
            continue
        try:
            rule = is_security_flagged(path.read_text(errors="replace"))
        except OSError:
            continue
        if rule:
            return rule, str(path.relative_to(root))
    return None, None


# ------------------------------------------------------------ pool worker ---

def stage_one(task: dict, root_s: str) -> dict:
    """Clone → tree pre-scan → venv → in-place gate → land base tree."""
    root = Path(root_s)
    iid = task["instance_id"]
    res = {"instance_id": iid, "repo": task["repo"], "source": task["source"],
           "difficulty": task["difficulty_bucket"], "ok": False, "reason": ""}
    tmp = Path(tempfile.mkdtemp(prefix=f"{iid}.", dir=root / "tmp"))
    tree, env = tmp / "repo", tmp / "venv"
    try:
        err = swe.clone_at_commit(task["repo"], task["base_commit"], tree)
        if err:
            res["reason"] = f"clone: {err}"
            return res

        strip_meta(tree)
        rule, rel = scan_tree(tree)
        if rule:
            res["reason"] = f"security-tree [{rule}] {rel}"
            res["security_rule"] = rule
            return res

        task, synthesized, err = synthesize_test_patch(tree, task)
        if err:
            res["reason"] = f"tests: {err}"
            return res
        if synthesized:
            rule = is_security_flagged(task["test_patch"])
            if rule:  # per-instance, not a dirty-repo signal
                res["reason"] = f"security-testpatch [{rule}]"
                return res
            res["synth_test_patch"] = task["test_patch"]

        py, err = make_env(tree, env)
        if err:
            res["reason"] = f"env: {err}"
            return res

        ok, note, fix = gate_in_place(tree, task, py)
        res["note"] = note
        if not ok:
            res["reason"] = f"gate: {note}"
            return res
        res["fix_gold_patch"] = fix

        err = restore_base(tree)
        if err:
            res["reason"] = f"restore: {err}"
            return res
        for attempt in range(3):  # background `git maintenance` races rmtree
            try:
                shutil.rmtree(tree / ".git")
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)
        staged = root / "staged" / iid
        if staged.exists():
            shutil.rmtree(staged)
        tree.rename(staged)
        res["ok"] = True
        return res
    except Exception as e:  # never kill the pool on one instance
        res["reason"] = f"exception: {type(e).__name__}: {e}"
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def synthesize_test_patch(tree: Path, task: dict):
    """SWE-smith instance branches DELETE the fail-to-pass test files (the
    agent must not see them); eval re-adds them from the parent state. Fetch
    the missing files from the fork's default branch and emit them as a
    new-file test_patch so the pod runner reproduces the same tree.
    Returns (task, synthesized, err)."""
    paths = {t.split("::")[0] for t in swe.fail_to_pass_list(task)}
    paths |= {p for p in task.get("selected_test_files_to_run") or [] if p}
    missing = sorted(p for p in paths if p and not (tree / p).exists())
    if not missing:
        return task, False, None
    r = swe.run(["git", "fetch", "-q", "--depth", "1", "origin", "HEAD"],
                tree, timeout=swe.SETUP_TIMEOUT)
    if r.returncode != 0:
        return task, False, f"fetch default branch rc={r.returncode}"
    patches = []
    for p in missing:
        r = swe.run(["git", "show", f"FETCH_HEAD:{p}"], tree)
        if r.returncode != 0 or not r.stdout.strip():
            return task, False, f"{p} absent on default branch"
        patches.append(new_file_diff(p, r.stdout))
    return dict(task, test_patch="".join(patches)), True, None


def make_env(tree: Path, env: Path):
    """Per-instance uv venv with the tree editable-installed (the gate patches
    the tree in place, so imports must resolve into it)."""
    deadline = time.monotonic() + INSTALL_TIMEOUT

    def uv(*args, required):
        left = deadline - time.monotonic()
        if left <= 0:
            return "install timeout"
        try:
            r = swe.run(["uv", *args], tree, timeout=left)
        except subprocess.TimeoutExpired:
            return "install timeout"
        if r.returncode != 0 and required:
            return f"uv {args[0]} rc={r.returncode}: {r.stderr.strip()[-200:]}"
        return None

    err = uv("venv", str(env), required=True)
    py = str(env / "bin" / "python")
    err = err or uv("pip", "install", "--python", py, "-e", ".", "pytest",
                    required=True)
    if err:
        return None, err
    for req in REQ_FILES:
        if (tree / req).is_file():
            uv("pip", "install", "--python", py, "-r", req, required=False)
    return py, None


def gate_in_place(tree: Path, task: dict, py: str):
    """stage_swe_pro.self_check semantics, applied in place (editable install
    pins imports to `tree`, so a scratch copy would test the wrong code).
    `-o addopts=` neutralizes repo addopts (e.g. --cov without the plugin)."""
    f2p = swe.fail_to_pass_list(task)
    if not f2p:
        return False, "empty fail_to_pass", None
    args = [*f2p, "-o", "addopts="]
    deadline = time.monotonic() + GATE_BUDGET

    err = swe.apply_patch(tree, task.get("test_patch", ""), ".t.diff")
    if err:
        return False, err, None
    rc, tail = swe.run_pytest(tree, args, py,
                              timeout=deadline - time.monotonic())
    if rc is None:
        return False, f"pre-fix {tail}", None
    if rc == 0:
        return False, f"pre-fix already passes: {tail}", None
    if rc != 1 or "error" in tail.lower():
        return False, f"broken env (rc={rc}): {tail}", None

    # SWE-smith `gold_patch` is the BUG-INJECTION patch (the instance branch
    # already contains the bug) — the fix is its reverse. Verify with -R and
    # emit the normalized forward fix (`git diff`) so downstream scoring can
    # apply gold_patch forward. swe_gym/SWE-Pro golds are forward fixes.
    gold = task.get("gold_patch", "")
    if not gold.strip():
        return False, "empty gold_patch", None
    if task.get("source") == "swe_smith":
        (tree / ".g.diff").write_text(gold)
        r = swe.run(["git", "apply", "-R", ".g.diff"], tree)
        (tree / ".g.diff").unlink(missing_ok=True)
        if r.returncode != 0:
            return False, (f"git apply -R gold rc={r.returncode}: "
                           f"{r.stderr.strip()[:200]}"), None
        # Synthesized test files are untracked, so they can't appear here —
        # but the scorer applies test_patch before gold_patch, so an overlap
        # would double-apply. Assert the invariant instead of trusting it.
        tpaths = {m.group(1) for m in
                  re.finditer(r"^\+\+\+ b/(\S+)", task.get("test_patch", ""),
                              re.M)}
        fix = swe.run(["git", "diff"], tree).stdout
        gpaths = {m.group(1) for m in
                  re.finditer(r"^(?:\+\+\+|---) [ab]/(\S+)", fix, re.M)}
        if tpaths & gpaths:
            return False, f"fix diff touches test files {sorted(tpaths & gpaths)[:3]}", None
    else:
        err = swe.apply_patch(tree, gold, ".g.diff")
        if err:
            return False, err, None
        fix = gold
    left = deadline - time.monotonic()
    if left <= 0:
        return False, "gate budget exhausted before gold run", None
    rc, tail = swe.run_pytest(tree, args, py, timeout=left)
    if rc != 0:
        return False, f"gold does not pass (rc={rc}): {tail}", None
    return True, tail, fix


def restore_base(tree: Path):
    """Reset to the (post-strip) base commit and scrub everything untracked;
    the landed tree must be byte-identical to the base checkout."""
    swe.run(["git", "checkout", "-q", "--", "."], tree)
    swe.run(["git", "clean", "-qfdx"], tree)
    r = swe.run(["git", "status", "--porcelain"], tree)
    if r.stdout.strip():
        return f"tree not restored: {r.stdout.strip()[:200]}"
    return None


# -------------------------------------------------------------- selection ---

def load_rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def apply_blocklist(rows, log):
    kept, dropped = [], Counter()
    for r in rows:
        hit = blocklisted(r["repo"])
        if hit:
            dropped[r["repo"]] += 1
        else:
            kept.append(r)
    for repo, n in sorted(dropped.items()):
        log(f"[blocklist] dropped {n} rows from {repo}")
    return kept, sum(dropped.values())


def round_robin(rows, cap):
    """Interleave difficulty buckets, and repos within each bucket, so the
    accepted slice stays repo- and difficulty-diverse."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        buckets[r["difficulty_bucket"]][r["repo"]].append(r)
    per_bucket = {}
    for diff, by_repo in buckets.items():
        queues = deque(deque(v) for _, v in sorted(by_repo.items()))
        out = []
        while queues:
            q = queues.popleft()
            out.append(q.popleft())
            if q:
                queues.append(q)
        per_bucket[diff] = deque(out)
    order = deque(d for d in ("easy", "medium", "hard") if d in per_bucket)
    out = []
    while order and len(out) < cap:
        d = order.popleft()
        out.append(per_bucket[d].popleft())
        if per_bucket[d]:
            order.append(d)
    return out


def select_candidates(args, log):
    stats = {"blocklist": 0}
    gym = load_rows(args.gym)
    gym, n = apply_blocklist(gym, log)
    stats["blocklist"] += n
    gym = [r for r in gym if r["repo"] in GYM_TRACTABLE]
    gym_cands = round_robin(gym, args.gym_n * args.over_select)

    smith = load_rows(args.smith)
    smith, n = apply_blocklist(smith, log)
    stats["blocklist"] += n
    smith = [r for r in smith
             if any(".py" in t for t in swe.fail_to_pass_list(r))]  # python only
    smith_by_diff = defaultdict(list)
    for r in smith:
        smith_by_diff[r["difficulty_bucket"]].append(r)
    smith_cands, smith_quota = {}, {}
    for diff, frac in SMITH_MIX.items():
        quota = round(args.smith_n * frac)
        smith_quota[diff] = quota
        smith_cands[diff] = deque(round_robin(
            smith_by_diff[diff], quota * args.over_select))
    return gym_cands, smith_cands, smith_quota, stats


# -------------------------------------------------------------- dispatch ---

def key_of(task):
    if task["source"] == "swe_gym":
        return ("swe_gym", "all")
    return ("swe_smith", task["difficulty_bucket"])


def run_pool(args, gym_cands, smith_cands, smith_quota, log):
    root = args.out_root
    (root / "staged").mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(parents=True, exist_ok=True)

    queues = {("swe_gym", "all"): deque(gym_cands)}
    need = {("swe_gym", "all"): args.gym_n}
    for diff, q in smith_cands.items():
        queues[("swe_smith", diff)] = q
        need[("swe_smith", diff)] = smith_quota[diff]

    accepted, reject_tally = [], Counter()
    reject_rules = Counter()
    dirty_repos = set()
    inflight, inflight_per_key = {}, Counter()

    def next_task():
        for key, q in queues.items():
            while q:
                if need[key] - inflight_per_key[key] <= 0:
                    break
                t = q.popleft()
                if t["repo"] in dirty_repos:
                    reject_tally["skipped-dirty-repo"] += 1
                    continue
                return key, t
        return None, None

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        def fill():
            while len(inflight) < args.workers:
                key, t = next_task()
                if t is None:
                    return
                fut = ex.submit(stage_one, t, str(root))
                inflight[fut] = (key, t)
                inflight_per_key[key] += 1

        fill()
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                key, t = inflight.pop(fut)
                inflight_per_key[key] -= 1
                res = fut.result()
                if res["ok"] and need[key] > 0:
                    need[key] -= 1
                    row = dict(t, before_repo_set_cmd="",
                               note=res.get("note", ""))
                    if "synth_test_patch" in res:
                        row["test_patch"] = res["synth_test_patch"]
                    if res.get("fix_gold_patch"):
                        row["gold_patch"] = res["fix_gold_patch"]
                    accepted.append(row)
                    log(f"[stage] {res['instance_id']}: ACCEPT "
                        f"({res.get('note', '')})")
                else:
                    reason = res["reason"] or "quota already filled"
                    reject_tally[reason.split(":")[0].split(" ")[0]] += 1
                    if "security_rule" in res:
                        reject_rules[res["security_rule"]] += 1
                        dirty_repos.add(res["repo"])
                    log(f"[stage] {res['instance_id']}: REJECT {reason}")
            fill()

    for key, n in need.items():
        if n > 0:
            log(f"[stage] PLATEAU {key}: {n} short of target "
                f"(candidates exhausted; gate not relaxed)")
    return accepted, reject_tally, reject_rules, need


# ------------------------------------------------------- split + emission ---

def split_by_repo(accepted, eval_frac=0.3):
    by_repo = defaultdict(list)
    for r in accepted:
        by_repo[r["repo"]].append(r)
    eval_target = round(eval_frac * len(accepted))
    eval_repos, n = set(), 0
    for repo, rows in sorted(by_repo.items(), key=lambda kv: len(kv[1])):
        if len(by_repo) - len(eval_repos) <= 1:
            break
        if abs(n + len(rows) - eval_target) > abs(n - eval_target):
            break  # adding this repo would overshoot further than stopping
        eval_repos.add(repo)
        n += len(rows)
    train = [r for r in accepted if r["repo"] not in eval_repos]
    ev = [r for r in accepted if r["repo"] in eval_repos]
    return train, ev


def write_jsonl(path: Path, rows):
    drop = ("note",)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items()
                                if k not in drop}) + "\n")


# ------------------------------------------------------------ final sweep ---

def final_sweep(root: Path, jsonl_names, log):
    """Drop any instance with a flagged staged file or JSONL row, re-emit,
    then let the filter CLI have the last word."""
    dropped, rules = {}, Counter()
    staged = root / "staged"
    for inst in sorted(p for p in staged.iterdir() if p.is_dir()):
        rule, rel = scan_tree(inst)
        if rule:
            dropped[inst.name] = rule
            rules[rule] += 1
            log(f"[sweep] drop {inst.name}: [{rule}] {rel}")
            shutil.rmtree(inst)
    for name in jsonl_names:
        path = root / name
        if not path.exists():
            continue
        kept = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            iid = json.loads(line)["instance_id"]
            rule = dropped.get(iid) or is_security_flagged(line)
            if rule:
                rules[rule] += 1
                if iid not in dropped:
                    dropped[iid] = rule
                    inst = staged / iid
                    if inst.exists():
                        shutil.rmtree(inst)
                    log(f"[sweep] drop {iid}: [{rule}] jsonl row")
            else:
                kept.append(line)
        path.write_text("".join(l + "\n" for l in kept))

    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "opd_security_filter.py"),
         "--scan", str(root)],
        capture_output=True, text=True, check=False)
    clean = r.returncode == 0
    log(f"[sweep] CLI --scan: {'clean' if clean else 'FLAGGED'} "
        f"({r.stdout.strip().splitlines()[-1] if r.stdout else ''})")
    return dropped, rules, clean


# ------------------------------------------------------------------ main ---

def count_rows(root, name, by=("source", "difficulty_bucket")):
    path = root / name
    if not path.exists():
        return {}
    tally = Counter()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            tally["/".join(str(r.get(k, "synthetic")) for k in by)] += 1
    return dict(sorted(tally.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gym", type=Path,
                    default=Path("data/opd-corpora/cleaned/swe_gym.jsonl"))
    ap.add_argument("--smith", type=Path,
                    default=Path("data/opd-corpora/cleaned/swe_smith.jsonl"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("data/opd-corpora/staged-run1"))
    ap.add_argument("--gym-n", type=int, default=50)
    ap.add_argument("--smith-n", type=int, default=100)
    ap.add_argument("--over-select", type=int, default=3,
                    help="candidate multiplier per target slot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()

    root = args.out_root
    droplog = open(root.parent / f"{root.name}.droplog.txt", "a")

    def log(msg):
        print(msg, flush=True)
        droplog.write(msg + "\n")
        droplog.flush()

    gym_cands, smith_cands, smith_quota, stats = select_candidates(args, log)
    log(f"[select] gym candidates={len(gym_cands)} smith="
        f"{ {d: len(q) for d, q in smith_cands.items()} } "
        f"quota={smith_quota} blocklist-dropped={stats['blocklist']}")

    accepted, rejects, reject_rules, shortfall = run_pool(
        args, gym_cands, smith_cands, smith_quota, log)
    train, ev = split_by_repo(accepted)
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "eval.jsonl", ev)
    log(f"[split] train={len(train)} eval={len(ev)} "
        f"(repos disjoint: {not set(r['repo'] for r in train) & set(r['repo'] for r in ev)})")

    if not args.skip_synthetic:
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "gen_agent_opd_tasks.py"),
             "--out", str(root), "--difficulty", "medium", "--self-check"],
            text=True, check=False)
        if r.returncode != 0:
            sys.exit("[synthetic] gen_agent_opd_tasks.py self-check FAILED")

    shutil.rmtree(root / "tmp", ignore_errors=True)
    jsonl_names = ["train.jsonl", "eval.jsonl",
                   "tasks_train.jsonl", "tasks_eval.jsonl"]
    swept, sweep_rules, clean = final_sweep(root, jsonl_names, log)

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "targets": {"swe_gym": args.gym_n, "swe_smith": args.smith_n,
                    "synthetic": 60},
        "counts": {name: count_rows(root, name) for name in jsonl_names},
        "staged_instances": sum(1 for p in (root / "staged").iterdir()
                                if p.is_dir()),
        "dropped": {
            "repo_blocklist": stats["blocklist"],
            "stage_rejects": sum(rejects.values()),
            "final_sweep": len(swept),
        },
        "shortfall": {f"{k[0]}/{k[1]}": v for k, v in shortfall.items() if v},
        "scan": "clean" if clean else "FLAGGED",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    tarball = root.parent / f"{root.name}.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(root, arcname=root.name)
    log(f"[done] manifest={root / 'manifest.json'} tarball={tarball} "
        f"({tarball.stat().st_size / 1e6:.1f} MB)")
    log(f"[done] stage-reject tally: {dict(rejects)}")
    log(f"[done] security-tree rules: {dict(reject_rules)} "
        f"sweep rules: {dict(sweep_rules)}")
    if not clean:
        sys.exit("[done] final scan NOT clean — do not ship")


if __name__ == "__main__":
    main()

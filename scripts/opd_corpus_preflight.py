#!/usr/bin/env python3
"""Pre-flight: which corpus repos cannot collect their own tests on this box.

The f2p test files ship in test_patch, not in the staged tree, so the patch has
to land before collection means anything. A collection error scores as
reward=0.0 and is indistinguishable from a model failure in the metrics, so a
box missing one dependency silently deflates every run on it.

    python3 scripts/opd_corpus_preflight.py <corpus-root>

<corpus-root> holds train_localized.jsonl / eval_localized.jsonl and staged/.
Exit 1 if any repo cannot collect.
"""
import json, os, subprocess, sys, shutil, tempfile, pathlib

C = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
tasks = {}
for name in ("train_localized.jsonl", "eval_localized.jsonl"):
    for line in (C / name).open():
        d = json.loads(line)
        tasks.setdefault(d["instance_id"].split(".")[0], (name, d))

def apply_tp(wd, tp):
    # A creating hunk whose target the repo already ships: patch(1) skips it and
    # git apply refuses, so drop the target first (mirrors sandbox.rs).
    creates = False
    for line in tp.splitlines():
        if line.startswith("+++ b/"):
            p = line[6:].strip()
            if creates and p and p != "/dev/null":
                (wd / p).unlink(missing_ok=True)
        creates = line.startswith("--- /dev/null")
    return subprocess.run(["patch", "-p1", "--batch", "--forward"], cwd=wd,
                          input=tp, capture_output=True, text=True)

bad = {}
for repo, (split, d) in sorted(tasks.items()):
    src = C / "staged" / d["instance_id"]
    f2p = d.get("fail_to_pass") or []
    files = sorted({t.split("::")[0] for t in f2p})
    if not src.is_dir() or not files:
        bad[repo] = (split, "no staged dir or no fail_to_pass"); continue
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        wd = pathlib.Path(tmp) / "r"
        shutil.copytree(src, wd, symlinks=True)
        pr = apply_tp(wd, d["test_patch"])
        if pr.returncode != 0:
            bad[repo] = (split, "test_patch rc=%d: %s" % (
                pr.returncode, (pr.stdout + pr.stderr).strip().splitlines()[-1][:120]))
            continue
        env = dict(os.environ, PYTHONNOUSERSITE="1",
                   PYTHONPATH=os.pathsep.join([str(wd), str(wd / "src")]))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "--collect-only", *files], cwd=wd, env=env,
                           capture_output=True, text=True, timeout=180)
        if r.returncode in (0, 5):
            continue
        missing = sorted({l.split("No module named")[1].strip().strip("'\" ")
                          for l in r.stdout.splitlines() if "No module named" in l})
        bad[repo] = (split, "missing: " + ", ".join(missing) if missing else
                     "collect rc=%d: %s" % (r.returncode,
                     (r.stdout.strip().splitlines() or [""])[-1][:160]))

print("repos=%d  broken=%d" % (len(tasks), len(bad)))
for repo, (split, why) in bad.items():
    print("  %-45s %-22s %s" % (repo, split, why))
mods = sorted({m for _, why in bad.values() if why.startswith("missing: ")
               for m in why[len("missing: "):].split(", ")})
print("MISSING_MODULES:", " ".join(mods) or "(none)")
sys.exit(1 if bad else 0)

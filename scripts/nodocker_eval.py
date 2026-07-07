#!/usr/bin/env python3
"""Docker-free eval for generated tasks (gen_terminal_tasks.py format).

Our generated tasks are pure file-I/O — no packages, no services, no network.
They need a clean workdir + the agent's shell commands + pytest, NOT a container.
This runs the full agent loop against an OpenAI endpoint in a tmpdir per task,
skipping every image build / container start — seconds per task vs docker minutes.
Tasks run in parallel (model I/O bound). pass@N by re-running each task N times.

    nodocker_eval.py --pool DIR --base http://127.0.0.1:PORT/v1 --model M
                     [--attempts 3] [--workers 8] [--max-steps 12]

# ponytail: agent shell runs in a host tmpdir via `bash -c`, isolated by cwd +
# timeout only (our own benign tasks). For untrusted tasks wrap in `bwrap
# --unshare-all --bind $tmp /app`; the harness would gain a --sandbox flag.
"""
import argparse, concurrent.futures as cf, importlib.util, json, os, re
import shutil, subprocess, sys, tempfile, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = {  # matches terminus CommandBatchResponse (subset)
    "type": "object",
    "properties": {
        "state_analysis": {"type": "string"},
        "is_task_complete": {"type": "boolean"},
        "commands": {"type": "array", "items": {"type": "object", "properties": {
            "keystrokes": {"type": "string"}}, "required": ["keystrokes"]}},
    },
    "required": ["state_analysis", "is_task_complete", "commands"],
}
PROMPT = ("You solve a command-line task by emitting JSON batches of shell commands.\n"
          "Task:\n{instr}\n\nRespond ONLY as JSON: {{\"state_analysis\":str,"
          "\"is_task_complete\":bool,\"commands\":[{{\"keystrokes\":\"<shell>\"}}]}}.\n"
          "Commands run in /app via bash. Set is_task_complete=true when done.")


def chat(base, model, messages):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.2,
                       "max_tokens": 2048, "response_format":
                       {"type": "json_schema", "json_schema": {"name": "r", "schema": SCHEMA}}}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=180))
    return r["choices"][0]["message"].get("content") or ""


def run_tests(test_file, app):  # same contract as gen_terminal_tasks self_check
    os.environ["APP_DIR"] = str(app)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(f"t_{app.name}_{os.getpid()}", test_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for n in dir(mod):
        if n.startswith("test_"):
            getattr(mod, n)()


def attempt(task_dir, base, model, max_steps):
    instr = ""
    for ln in open(task_dir / "task.yaml"):
        if ln.startswith("instruction:"):
            instr = task_dir.joinpath("task.yaml").read_text().split("instruction:", 1)[1]
            instr = instr.split("author_name:")[0].replace("|-", "").strip()
            break
    with tempfile.TemporaryDirectory(prefix="nd-") as tmp:
        app = Path(tmp) / "app"
        shutil.copytree(task_dir / "fixture", app)
        msgs = [{"role": "user", "content": PROMPT.format(instr=instr)}]
        for _ in range(max_steps):
            try:
                resp = json.loads(chat(base, model, msgs))
            except Exception:
                return False
            out = []
            for c in resp.get("commands", []):
                k = c.get("keystrokes", "")
                try:
                    p = subprocess.run(["bash", "-c", k], cwd=app, capture_output=True,
                                       text=True, timeout=30)
                    out.append(f"$ {k}\n{p.stdout}{p.stderr}"[:2000])
                except subprocess.TimeoutExpired:
                    out.append(f"$ {k}\n[timeout]")
            msgs.append({"role": "assistant", "content": json.dumps(resp)})
            msgs.append({"role": "user", "content": "\n".join(out) or "(no output)"})
            if resp.get("is_task_complete"):
                break
        try:
            run_tests(task_dir / "tests" / "test_outputs.py", app)
            return True
        except Exception:
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=12)
    a = ap.parse_args()

    tasks = sorted(p for p in a.pool.iterdir() if (p / "task.yaml").exists())
    jobs = [(t, i) for t in tasks for i in range(a.attempts)]
    res = defaultdict(list)
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(attempt, t, a.base, a.model, a.max_steps): t.name for t, _ in jobs}
        for f in cf.as_completed(futs):
            res[futs[f]].append(f.result())

    fam = defaultdict(lambda: [0, 0])
    p1 = 0
    for name, oks in sorted(res.items()):
        fam[re.sub(r"-\d+$", "", name)][0] += sum(oks)
        fam[re.sub(r"-\d+$", "", name)][1] += len(oks)
        p1 += 1 if oks and oks[0] else 0
    print(f"tasks={len(res)} attempts={a.attempts} pass@1={p1}/{len(res)}")
    for f, (p, n) in sorted(fam.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        band = "<-BAND" if 0 < p < n else ("easy" if p == n else "hard")
        print(f"  {p/n:4.0%} {p:2d}/{n:2d} {f} {band}")


if __name__ == "__main__":
    main()

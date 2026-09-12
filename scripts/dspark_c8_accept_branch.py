#!/usr/bin/env python3
"""Decide which mechanism explains DSpark's c=8 0% acceptance from counters.

The acceptance rate is accepted/drafted; 0.0000 alone cannot distinguish
"no draft row was verified" from "every draft missed". The three branches
(see docs/experience/errors/2026-09-12-dspark-c8-confidence-budget-admits-zero.md):

  NOT-SEEDING          chains=0 over the decode window
  BUDGET-ZERO-KEEPS    chains>0, drafted=0 (confidence budget truncated every
                       chain to its anchor — the scheduling explanation)
  PROPOSED-AND-REJECTED chains>0, drafted>0, accepted=0 (scheduling exonerated;
                       verify-side cross-slot bug survives)

Runs the SAME greedy load at c=1 and c=8 against a running `arle serve
--spec-type dspark ...`, snapshots /v1/stats spec_decode counters around each
window, and prints the branch. Pass --server-log to capture the engine's
[dspark-seed] reason lines emitted during the windows.

The c=1 point is a positive control: the known symptom is ~13% there, so
chains/drafted must be nonzero. A c=8 verdict taken without a live c=1 point
in the same run is refused.

Usage:
  python3 scripts/dspark_c8_accept_branch.py --port 8000 \\
      --server-log /tmp/arle-serve.log
  # self-contained (spawns serve itself):
  python3 scripts/dspark_c8_accept_branch.py \\
      --serve-bin ./target/release/arle \\
      --model-path <models>/Qwen3.8-27B-FP8 \\
      --draft-model <models>/Qwen3.8-27B-DSpark --gpus 0

Negative control: --selftest feeds synthetic counter tuples through the
classifier, including an accepted>drafted plumbing inconsistency that MUST
refuse a verdict; run it before trusting a green output.

A prereg row is opened before the c=1 window and closed with the branch
(hypothesis fixed before the counters are read); skip with
ARLE_C8_SKIP_PREREG=1.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arle_stats import spec_decode  # noqa: E402

NOT_SEEDING = "NOT-SEEDING"
BUDGET_ZERO_KEEPS = "BUDGET-ZERO-KEEPS"
PROPOSED_AND_REJECTED = "PROPOSED-AND-REJECTED"
ACCEPTING = "ACCEPTING"

PROMPT = "Explain how speculative decoding works in a transformer, step by step."


def http_get(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_ready(port, deadline_s):
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            http_get(f"http://127.0.0.1:{port}/v1/stats", timeout=2.0)
            return
        except Exception:
            time.sleep(2.0)
    sys.exit(f"server did not become ready on port {port} within {deadline_s}s")


def counters(port):
    sd = spec_decode(http_get(f"http://127.0.0.1:{port}/v1/stats"))
    need = ("chains", "drafted", "accepted")
    if not all(k in sd for k in need):
        sys.exit(f"stats missing spec_decode counters: got {sd}")
    return {k: int(sd[k]) for k in need}


def classify(delta):
    """Map a counters delta to one branch. Garbage is a hard error, never a branch.

    Negative deltas and accepted>drafted mean the snapshots were misread
    (wrong window, wrong server, swapped fields); printing a branch then would
    pin the wrong mechanism.
    """
    chains, drafted, accepted = delta["chains"], delta["drafted"], delta["accepted"]
    if chains < 0 or drafted < 0 or accepted < 0:
        return None, f"negative counter delta {delta} — snapshots taken out of order"
    if accepted > drafted:
        return None, f"accepted {accepted} > drafted {drafted} — counter plumbing misread"
    if chains == 0:
        return NOT_SEEDING, "no verify chain in the window (mixed-step starvation or seed mismatch)"
    if drafted == 0:
        return BUDGET_ZERO_KEEPS, "every chain truncated to its anchor by the confidence budget"
    if accepted == 0:
        return PROPOSED_AND_REJECTED, "drafts proposed and all rejected — scheduling exonerated"
    return ACCEPTING, "drafts accepted — collapse not reproduced this run"


def drive_once(port, concurrency, max_tokens):
    """Send `concurrency` identical greedy requests simultaneously."""
    barrier = threading.Barrier(concurrency)
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()

    def one(errors):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        barrier.wait()
        try:
            with urllib.request.urlopen(req, timeout=600.0) as r:
                json.loads(r.read().decode())
        except Exception as e:  # one failed request is reported, not fatal to counters
            errors.append(str(e))

    errors = []
    threads = [threading.Thread(target=one, args=(errors,)) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def window(port, concurrency, max_tokens, settle_s):
    before = counters(port)
    errors = drive_once(port, concurrency, max_tokens)
    time.sleep(settle_s)
    after = counters(port)
    delta = {k: after[k] - before[k] for k in before}
    return delta, errors, after


def prereg(name, verb, **fields):
    if os.environ.get("ARLE_C8_SKIP_PREREG") == "1":
        return
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts/prereg.py"), verb, "--name", name]
    for k, v in fields.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    subprocess.run(cmd, cwd=root, check=False)


def new_seed_lines(log_path, start_pos):
    if not log_path:
        return []
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            f.seek(start_pos)
            chunk = f.read()
    except OSError:
        return []
    return [ln for ln in chunk.splitlines() if "[dspark-seed]" in ln]


def run_measure(args):
    wait_ready(args.port, args.ready_timeout_s)
    log_pos = Path(args.server_log).stat().st_size if args.server_log else 0

    name = "dspark-c8-accept-branch-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    prereg(
        name, "start",
        cmd=f"scripts/dspark_c8_accept_branch.py --port {args.port} --concurrency {args.concurrency}",
        hypothesis=("c=8 acceptance 0 lands in exactly one branch: chains=0 NOT-SEEDING; "
                    "chains>0 drafted=0 BUDGET-ZERO-KEEPS (confidence budget bar 0.0025->0.0197); "
                    "chains>0 drafted>0 accepted=0 PROPOSED-AND-REJECTED (scheduling exonerated). "
                    "c=1 positive control must show chains>0 and drafted>0."),
    )

    points = []
    for c in (1, args.concurrency):
        delta, errors, _ = window(args.port, c, args.max_tokens, args.settle_s)
        branch, why = classify(delta)
        seeds = new_seed_lines(args.server_log, log_pos) if c == args.concurrency else []
        points.append((c, delta, branch, why, errors, seeds))

    print("\n== DSpark c=1 vs c=%d spec_decode counters ==" % args.concurrency)
    for c, delta, branch, why, errors, seeds in points:
        rate = (delta["accepted"] / delta["drafted"]) if delta["drafted"] else 0.0
        print(f"c={c}: chains={delta['chains']} drafted={delta['drafted']} "
              f"accepted={delta['accepted']} accept_rate={rate:.4f}")
        if branch is None:
            print(f"  VERDICT: COUNTER ERROR — {why}")
        else:
            print(f"  VERDICT: {branch} — {why}")
        if errors:
            print(f"  request errors: {len(errors)} (first: {errors[0][:160]})")
        if seeds:
            print(f"  [dspark-seed] lines during c={c} window ({len(seeds)}):")
            for ln in seeds[:20]:
                print("    " + ln.strip())

    c1, c8 = points
    c1_live = c1[1]["chains"] > 0 and c1[1]["drafted"] > 0
    counter_error = any(p[2] is None for p in points)

    result = f"c1 {c1[1]}; c{args.concurrency} {c8[1]} -> {c8[2]}"
    finding = {
        NOT_SEEDING: "rows never seeded in the window; inspect [dspark-seed] reasons and mixed-step scheduling",
        BUDGET_ZERO_KEEPS: "confidence budget truncated every chain to its anchor; scheduling/calibration explains the 0.0000 rate",
        PROPOSED_AND_REJECTED: "drafts proposed but all rejected; scheduling exonerated, verify-side hypothesis survives",
        ACCEPTING: "nonzero acceptance at c=8; collapse not reproduced under this load",
    }.get(c8[2], "counter plumbing error; no branch assigned")
    decision = {
        NOT_SEEDING: "next: seed-condition / mixed-step probe, not a kernel fix",
        BUDGET_ZERO_KEEPS: "next: confidence-distribution probe or SPS bar sweep, not a kernel fix",
        PROPOSED_AND_REJECTED: "next: run the whole-drafter-step gate (#374) GPU batch",
        ACCEPTING: "record load shape; the collapse needs the original workload to reproduce",
    }.get(c8[2], "fix stats collection before re-running")
    if not c1_live:
        status, decision = "killed", "drafter not engaged at c=1; fix serve/config before reading c=8"
    elif counter_error:
        status = "killed"
    else:
        status = "ok"
    prereg(name, "done", status=status, result=result, finding=finding, decision=decision,
           entry="docs/experience/errors/2026-09-12-dspark-c8-confidence-budget-admits-zero.md")

    if not c1_live:
        print("\nREFUSED: c=1 positive control is dead — no chains/drafted at c=1; "
              "the c=8 verdict cannot be trusted (drafter not engaged at all).")
        return 2
    if counter_error:
        return 3
    print(f"\nBRANCH: {c8[2]}")
    return 0


def selftest():
    # (delta, expected branch or None=must-error)
    cases = [
        ({"chains": 0, "drafted": 0, "accepted": 0}, NOT_SEEDING),
        ({"chains": 120, "drafted": 0, "accepted": 0}, BUDGET_ZERO_KEEPS),
        ({"chains": 120, "drafted": 300, "accepted": 0}, PROPOSED_AND_REJECTED),
        ({"chains": 120, "drafted": 300, "accepted": 39}, ACCEPTING),
        # Negative controls: misread plumbing must NOT produce a branch.
        ({"chains": 10, "drafted": 3, "accepted": 5}, None),  # accepted>drafted
        ({"chains": -4, "drafted": 0, "accepted": 0}, None),  # snapshots reversed
    ]
    bad = 0
    for delta, expect in cases:
        branch, _ = classify(delta)
        ok = branch == expect
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {delta} -> {branch} (want {expect})")
    if bad:
        print(f"SELFTEST FAILED: {bad} case(s)")
        return 1
    print("SELFTEST OK — every synthetic branch classified and both plumbing traps refused")
    return 0


def spawn_serve(args):
    cmd = [
        args.serve_bin, "serve",
        "--model-path", args.model_path,
        "--spec-type", "dspark", "--mtp-draft-model", args.draft_model,
        "--port", str(args.port),
    ]
    env = dict(os.environ)
    if args.gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    log = open(args.server_log or "/tmp/dspark_c8_serve.log", "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    try:
        wait_ready(args.port, args.ready_timeout_s)
    except SystemExit:
        proc.send_signal(signal.SIGTERM)
        raise
    return proc, log


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--ready-timeout-s", type=float, default=1200.0)
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--serve-bin", default=None, help="spawn this arle binary")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--gpus", type=int, default=None, help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--selftest", action="store_true", help="negative control, no server")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    proc = None
    try:
        if args.serve_bin:
            if not (args.model_path and args.draft_model):
                sys.exit("--serve requires --model and --draft-model")
            if not args.server_log:
                args.server_log = "/tmp/dspark_c8_serve.log"
            proc, _ = spawn_serve(args)
        rc = run_measure(args)
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGTERM)
    sys.exit(rc)


if __name__ == "__main__":
    main()

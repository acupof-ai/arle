#!/usr/bin/env python3
"""Time-series + alerting for an agent-OPD run.

Metrics only answer questions while the process is alive; this appends them to
a file so a finished run can still be asked what happened at hour three.

The alerts are the three failures that actually happened on 2026-08-23, each
of which produced a plausible-looking result that was not about the model:

- every rollout returning `edited=false` for N consecutive rounds — the
  harness was broken three separate ways (missing binary, aborted stream,
  silent trajectory skip) and each looked like a weak model;
- a poisoned import path, where `import <pkg>` resolves outside the task tree
  and scoring measures an unrelated checkout;
- decode throughput drifting, which turns a capability comparison into a
  timeout comparison.

    python3 scripts/opd_monitor.py --port 8074 --out /tmp/opd-metrics.jsonl
    python3 scripts/opd_monitor.py --check-imports flake8 sqlparse
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Consecutive zero-edit rounds before the run is called broken. Two is below
# the noise floor for a weak model on hard tasks; five is not.
EDIT_DROUGHT_ROUNDS = 5
# Fractional TPOT rise over the run's own opening window that counts as drift.
TPOT_DRIFT = 0.5


def scrape(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/stats", timeout=10) as fh:
            return json.load(fh)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def sample(port: int) -> dict | None:
    raw = scrape(port)
    if not raw:
        return None
    t, s = raw.get("throughput", {}), raw.get("scheduler", {})
    n = max(t.get("ttft_count", 0), 1)
    decode_steps = max(t.get("decode_forward_steps", 0), 1)
    return {
        "requests": t.get("requests_completed", 0),
        "failed": t.get("requests_failed", 0),
        "gen_tokens": t.get("generated_tokens", 0),
        "prefill_tokens": t.get("prefill_tokens", 0),
        "ttft_s": t.get("ttft_micros_total", 0) / n / 1e6,
        "tpot_ms": t.get("tpot_micros_total", 0) / n / 1e3,
        "decode_ms_per_step": t.get("decode_forward_busy_micros", 0) / decode_steps / 1e3,
        "active": s.get("active_requests", 0),
        "queued": s.get("queue_depth", 0),
        "kv_free_pages": s.get("kv_free_pages", 0),
    }


def check_imports(packages: list[str], workdir: Path | None) -> list[str]:
    """A package resolving outside the task tree means scoring measures the
    wrong checkout. This is the exact shape of the 2026-08-23 poisoning."""
    alerts = []
    for pkg in packages:
        probe = subprocess.run(
            [sys.executable, "-c", f"import {pkg}, sys; print({pkg}.__file__ or '')"],
            capture_output=True,
            text=True,
            cwd=workdir,
        )
        if probe.returncode != 0:
            continue  # not installed is fine; wrongly installed is not
        path = probe.stdout.strip()
        if workdir and path and not path.startswith(str(workdir)):
            alerts.append(f"import {pkg} resolves to {path}, outside {workdir}")
    return alerts


def rollout_alerts(rows: list[dict]) -> list[str]:
    """`rows` are per-rollout records carrying `edited`."""
    alerts = []
    tail = rows[-EDIT_DROUGHT_ROUNDS:]
    if len(tail) == EDIT_DROUGHT_ROUNDS and not any(r.get("edited") for r in tail):
        alerts.append(
            f"{EDIT_DROUGHT_ROUNDS} consecutive rollouts made no edit — "
            "check the harness before reading this as capability"
        )
    return alerts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8074)
    ap.add_argument("--out", default="/tmp/opd-metrics.jsonl")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--check-imports", nargs="*", default=[])
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    if args.check_imports:
        wd = Path(args.workdir).resolve() if args.workdir else None
        found = check_imports(args.check_imports, wd)
        for a in found:
            print(f"ALERT {a}")
        return 1 if found else 0

    out = Path(args.out)
    opening: float | None = None
    while True:
        row = sample(args.port)
        if row is None:
            print("ALERT serve unreachable", flush=True)
        else:
            row["t"] = int(time.time())
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            if opening is None and row["requests"] >= 20:
                opening = row["tpot_ms"]
            if opening and row["tpot_ms"] > opening * (1 + TPOT_DRIFT):
                print(
                    f"ALERT decode drift: {row['tpot_ms']:.1f} ms/tok "
                    f"against {opening:.1f} at the run's start",
                    flush=True,
                )
            if row["failed"]:
                print(f"ALERT {row['failed']} requests failed", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())

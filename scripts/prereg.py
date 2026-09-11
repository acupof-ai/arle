#!/usr/bin/env python3
"""Pre-registration ledger: the hypothesis is written before the run starts.

A wins/errors entry is written after the fact, which cannot constrain what the
run was for. `start` records the question the run answers while the answer is
still unknown; `done` closes the row with three separate fields — `result` is
the number, `finding` is what the number means, `decision` is what changes
because of it. Collapsing them is how a measurement becomes a conclusion
without anyone deciding it should.

    scripts/prereg.py start --name c1-mtp-batch \
        --cmd "scripts/bench_multiturn_ttft.py --turns 12" \
        --hypothesis "batched MTP verify moves per-turn TTFT at 12 turns"
    scripts/prereg.py done --name c1-mtp-batch --status ok \
        --result "6.20 -> 4.31 ms" --finding "..." --decision "..."

`check_repo_hygiene.py` fails on a row left running past 24h: a job that died
without closing its row leaves no other trace.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "docs/experience/prereg.jsonl"
STALE_AFTER = timedelta(hours=24)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_rows() -> list[dict]:
    if not LEDGER.is_file():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def write_rows(rows: list[dict]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def open_row(rows: list[dict], name: str) -> dict | None:
    return next((row for row in rows if row["name"] == name and row["status"] == "running"), None)


def cmd_start(args: argparse.Namespace) -> int:
    rows = read_rows()
    if open_row(rows, args.name):
        print(f"prereg: {args.name} is already running; close it with `done` first", file=sys.stderr)
        return 2
    rows.append({
        "name": args.name,
        "started": now(),
        "cmd": args.cmd,
        "hypothesis": args.hypothesis,
        "status": "running",
    })
    write_rows(rows)
    print(f"prereg: {args.name} running — {args.hypothesis}")
    return 0


def cmd_done(args: argparse.Namespace) -> int:
    rows = read_rows()
    row = open_row(rows, args.name)
    if row is None:
        print(f"prereg: no running row named {args.name}", file=sys.stderr)
        return 2
    row.update({
        "ended": now(),
        "result": args.result,
        "finding": args.finding,
        "decision": args.decision,
        "status": args.status,
        "entry": args.entry,
    })
    write_rows(rows)
    print(f"prereg: {args.name} {args.status} — {args.result}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for row in read_rows():
        if args.running and row["status"] != "running":
            continue
        tail = row.get("result") or row["hypothesis"]
        print(f'{row["started"]}  {row["status"]:<8} {row["name"]:<32} {tail}')
    return 0


def stale_running(rows: list[dict] | None = None, cutoff: datetime | None = None) -> list[str]:
    """Rows still `running` past STALE_AFTER — the job died without closing."""
    cutoff = cutoff or datetime.now(timezone.utc)
    stale = []
    for row in read_rows() if rows is None else rows:
        if row.get("status") != "running":
            continue
        started = datetime.strptime(row["started"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if cutoff - started > STALE_AFTER:
            stale.append(f'{row["name"]}: running since {row["started"]}')
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start", help="open a row before the run starts")
    start.add_argument("--name", required=True)
    start.add_argument("--cmd", required=True, help="the command, verbatim")
    start.add_argument("--hypothesis", required=True, help="the question this run answers")
    start.set_defaults(func=cmd_start)

    done = sub.add_parser("done", help="close a row after the run ends")
    done.add_argument("--name", required=True)
    done.add_argument("--result", required=True, help="the number")
    done.add_argument("--finding", required=True, help="what the number means, not the number")
    done.add_argument("--decision", required=True, help="what changes because of it")
    done.add_argument("--status", default="ok", choices=["ok", "rejected", "killed", "void"])
    done.add_argument("--entry", default="", help="path to the wins/errors entry this produced")
    done.set_defaults(func=cmd_done)

    listing = sub.add_parser("list", help="print the ledger")
    listing.add_argument("--running", action="store_true")
    listing.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

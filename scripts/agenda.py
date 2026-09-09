#!/usr/bin/env python3
"""Agenda ledger: goals, the gate each exits on, and who is executing what.

Three layers, one join key each:

    goal  --.
            `-- task (owner, exit gate) --.
                                          `-- run  (scripts/prereg.py)

A goal states the outcome and the one metric that measures it. A task is what
one agent is executing against that goal, and it carries the **exit** — the
named, measurable event that closes it, never a date. A run is one measurement,
pre-registered by `scripts/prereg.py`; a task lists the run names it spawned.

`report` renders the three layers as one page and is the only thing a human
reads. It is derived: nothing is written to it by hand.

    scripts/agenda.py goal add --id 27b-sota --statement "..." --north-star "..."
    scripts/agenda.py task add --id mtp-batch --goal 27b-sota --owner agent-infer-e2 \
        --exit "ms/committed-token at c=16 beats the no-spec control on the same binary"
    scripts/agenda.py task update --id mtp-batch --note "op_timing says ..." --run c16-mtp
    scripts/agenda.py task close --id mtp-batch --status done \
        --verdict "accepted, -32% at c=16" --entry docs/experience/wins/2026-09-10-...md
    scripts/agenda.py report --html /tmp/agenda.html
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/agenda.jsonl"
PREREG = ROOT / "docs/experience/prereg.jsonl"
STALE_AFTER = timedelta(hours=72)
OPEN_STATES = ("open", "active")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_rows(path: Path = LEDGER) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(rows: list[dict]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def find(rows: list[dict], kind: str, ident: str) -> dict | None:
    return next((row for row in rows if row["kind"] == kind and row["id"] == ident), None)


def cmd_goal_add(args: argparse.Namespace) -> int:
    rows = read_rows()
    if find(rows, "goal", args.id):
        print(f"agenda: goal {args.id} already exists", file=sys.stderr)
        return 2
    rows.append({
        "kind": "goal", "id": args.id, "statement": args.statement,
        "north_star": args.north_star, "status": "open", "opened": now(),
    })
    write_rows(rows)
    print(f"agenda: goal {args.id} — {args.north_star}")
    return 0


def cmd_task_add(args: argparse.Namespace) -> int:
    rows = read_rows()
    if find(rows, "task", args.id):
        print(f"agenda: task {args.id} already exists", file=sys.stderr)
        return 2
    if not find(rows, "goal", args.goal):
        print(f"agenda: no goal named {args.goal}", file=sys.stderr)
        return 2
    rows.append({
        "kind": "task", "id": args.id, "goal": args.goal, "owner": args.owner,
        "exit": args.exit, "status": "active" if args.owner else "open",
        "opened": now(), "updated": now(), "notes": [], "runs": [],
        "verdict": "", "entry": "",
    })
    write_rows(rows)
    print(f"agenda: task {args.id} -> {args.goal} ({args.owner or 'unassigned'})")
    return 0


def cmd_task_update(args: argparse.Namespace) -> int:
    rows = read_rows()
    task = find(rows, "task", args.id)
    if task is None:
        print(f"agenda: no task named {args.id}", file=sys.stderr)
        return 2
    if args.note:
        task["notes"].append({"at": now(), "text": args.note})
    if args.run and args.run not in task["runs"]:
        task["runs"].append(args.run)
    if args.owner:
        task["owner"] = args.owner
    if args.status:
        task["status"] = args.status
    task["updated"] = now()
    write_rows(rows)
    print(f'agenda: task {args.id} {task["status"]} ({len(task["notes"])} notes, {len(task["runs"])} runs)')
    return 0


def cmd_task_close(args: argparse.Namespace) -> int:
    rows = read_rows()
    task = find(rows, "task", args.id)
    if task is None:
        print(f"agenda: no task named {args.id}", file=sys.stderr)
        return 2
    task.update({"status": args.status, "verdict": args.verdict,
                 "entry": args.entry, "updated": now(), "closed": now()})
    write_rows(rows)
    print(f"agenda: task {args.id} {args.status} — {args.verdict}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for row in read_rows():
        if row["kind"] != "task":
            continue
        if args.goal and row["goal"] != args.goal:
            continue
        if args.owner and row["owner"] != args.owner:
            continue
        if args.status and row["status"] != args.status:
            continue
        print(f'{row["status"]:<7} {row["id"]:<24} {row["goal"]:<14} {row["owner"] or "-":<18} {row["exit"]}')
    return 0


def runs_for(task: dict, prereg: list[dict]) -> list[dict]:
    return [row for row in prereg if row["name"] in task["runs"]]


def defects() -> list[str]:
    """What the ledger itself says is wrong. Read by check_repo_hygiene.py."""
    rows = read_rows()
    goals = {row["id"] for row in rows if row["kind"] == "goal"}
    out = []
    for task in (row for row in rows if row["kind"] == "task"):
        if not task.get("exit"):
            out.append(f'task {task["id"]}: no exit gate — a task closes on a named measurable event')
        if task["goal"] not in goals:
            out.append(f'task {task["id"]}: goal {task["goal"]} does not exist')
        if task["status"] == "done" and not task.get("entry"):
            out.append(f'task {task["id"]}: done with no wins/errors entry')
        if task["status"] in OPEN_STATES:
            updated = datetime.strptime(task["updated"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - updated > STALE_AFTER:
                out.append(f'task {task["id"]}: {task["status"]} and untouched since {task["updated"]}')
    return out


def cmd_report(args: argparse.Namespace) -> int:
    rows = read_rows()
    prereg = read_rows(PREREG)
    lines = ["# Agenda", "", f"Rendered {now()} — derived from `docs/agenda.jsonl`; do not edit by hand.", ""]
    for goal in (row for row in rows if row["kind"] == "goal"):
        tasks = [row for row in rows if row["kind"] == "task" and row["goal"] == goal["id"]]
        done = sum(1 for task in tasks if task["status"] in ("done", "killed"))
        lines += [f'## {goal["id"]} — {goal["statement"]}', "",
                  f'**Measured by:** {goal["north_star"]}  ', f"**Tasks:** {done}/{len(tasks)} closed", ""]
        if tasks:
            lines += ["| Task | Owner | Status | Exits on | Verdict |", "|---|---|---|---|---|"]
            for task in tasks:
                verdict = task["verdict"] or "—"
                if task["entry"]:
                    verdict = f'{verdict} ([entry]({task["entry"]}))'
                lines.append(f'| `{task["id"]}` | {task["owner"] or "—"} | {task["status"]} | {task["exit"]} | {verdict} |')
            lines.append("")
        for task in tasks:
            runs = runs_for(task, prereg)
            if not task["notes"] and not runs:
                continue
            lines.append(f'### {task["id"]}')
            for run in runs:
                got = run.get("result") or f'running — {run["hypothesis"]}'
                lines.append(f'- run `{run["name"]}` ({run["status"]}): {got}')
            for note in task["notes"][-5:]:
                lines.append(f'- {note["at"]}: {note["text"]}')
            lines.append("")
    bad = defects()
    if bad:
        lines += ["## Ledger defects", ""] + [f"- {item}" for item in bad] + [""]
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"agenda: wrote {args.out}")
    else:
        print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    goal = sub.add_parser("goal").add_subparsers(dest="op", required=True)
    add = goal.add_parser("add")
    add.add_argument("--id", required=True)
    add.add_argument("--statement", required=True, help="the outcome, one sentence")
    add.add_argument("--north-star", required=True, dest="north_star", help="the one metric that measures it")
    add.set_defaults(func=cmd_goal_add)

    task = sub.add_parser("task").add_subparsers(dest="op", required=True)
    add = task.add_parser("add")
    add.add_argument("--id", required=True)
    add.add_argument("--goal", required=True)
    add.add_argument("--exit", required=True, help="the named measurable event that closes it, never a date")
    add.add_argument("--owner", default="")
    add.set_defaults(func=cmd_task_add)

    upd = task.add_parser("update")
    upd.add_argument("--id", required=True)
    upd.add_argument("--note", default="")
    upd.add_argument("--run", default="", help="a scripts/prereg.py run name this task spawned")
    upd.add_argument("--owner", default="")
    upd.add_argument("--status", default="", choices=["", "open", "active", "blocked"])
    upd.set_defaults(func=cmd_task_update)

    close = task.add_parser("close")
    close.add_argument("--id", required=True)
    close.add_argument("--verdict", required=True)
    close.add_argument("--entry", default="", help="the wins/errors entry; required when status is done")
    close.add_argument("--status", default="done", choices=["done", "killed"])
    close.set_defaults(func=cmd_task_close)

    listing = sub.add_parser("list")
    listing.add_argument("--goal", default="")
    listing.add_argument("--owner", default="")
    listing.add_argument("--status", default="")
    listing.set_defaults(func=cmd_list)

    report = sub.add_parser("report")
    report.add_argument("--out", default="", help="write the rendered page here instead of stdout")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

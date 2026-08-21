#!/usr/bin/env python3
"""Turn a needle_gate.py log into an `arle.operator-e2e/v1` artifact.

The artifact is the e2e half of operator qualification
(`scripts/run_fp8_probe.sh` → `scripts/reduce_operator_evidence.py`). Identity
is the kernel bundle (`arle --kernel-build-id`), which the serve binary and the
probe binary share; the serve binary's own digest is recorded alongside.

    python3 scripts/operator_e2e_artifact.py \
        --needle-log needle.log --runs 3 \
        --serve-binary target/release/arle \
        --model-revision Qwen/Qwen3.6-27B-FP8@<commit> \
        --output e2e.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "arle.operator-e2e/v1"
SUMMARY = re.compile(
    r"^SUMMARY len=(?P<len>\d+) depth=(?P<depth>[\d.]+) exact=(?P<exact>\d+) "
    r"partial=(?P<partial>\d+) miss=(?P<miss>\d+) (?P<det>DET|NONDET)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_needle(log: str, runs: int) -> tuple[bool, list[dict]]:
    """Pass = every SUMMARY line has exact == runs and is DET; no ERROR lines."""
    rows = []
    for line in log.splitlines():
        match = SUMMARY.match(line.strip())
        if match:
            rows.append({k: (v if k == "det" else float(v) if k == "depth" else int(v)) for k, v in match.groupdict().items()})
    errored = any(" ERROR " in line for line in log.splitlines())
    passed = bool(rows) and not errored and all(r["exact"] == runs and r["det"] == "DET" for r in rows)
    return passed, rows


def build(args: argparse.Namespace, log: str, bundle_id: str) -> dict:
    passed, rows = parse_needle(log, args.runs)
    return {
        "schema_version": SCHEMA,
        "gate": "needle_gate",
        "passed": passed,
        "bundle_id": bundle_id,
        "binary_id": "sha256:" + sha256_file(args.serve_binary),
        "model_revision": {"kind": "actual", "id": args.model_revision},
        "runs_per_length": args.runs,
        "summary": rows,
        "log_sha256": hashlib.sha256(log.encode()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--needle-log", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--serve-binary", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--bundle-id", help="override; default asks the serve binary with --kernel-build-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        good = "SUMMARY len=512 depth=0.00 exact=3 partial=0 miss=0 DET kv=\nSUMMARY len=4096 depth=0.00 exact=3 partial=0 miss=0 DET kv=\n"
        assert parse_needle(good, 3)[0] is True
        assert parse_needle(good.replace("exact=3 partial=0", "exact=2 partial=1", 1), 3)[0] is False
        assert parse_needle(good.replace("DET", "NONDET", 1), 3)[0] is False
        assert parse_needle(good + "len=16384 depth=0.00 run=0 ERROR 'x'\n", 3)[0] is False
        assert parse_needle("", 3)[0] is False
        print("[operator-e2e] self-test OK")
        return 0

    for name in ("needle_log", "serve_binary", "model_revision", "output"):
        if getattr(args, name) in (None, ""):
            parser.error(f"--{name.replace('_', '-')} is required")
    bundle_id = args.bundle_id or subprocess.run(
        [str(args.serve_binary), "--kernel-build-id"], text=True, stdout=subprocess.PIPE, check=True
    ).stdout.splitlines()[0].strip()
    if not re.fullmatch(r"bundle:[0-9a-f]{64}", bundle_id):
        print(f"[operator-e2e] serve binary reports no verified kernel bundle: {bundle_id!r}", file=sys.stderr)
        return 1
    artifact = build(args, args.needle_log.read_text(encoding="utf-8"), bundle_id)
    args.output.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[operator-e2e] passed={artifact['passed']} bundle={bundle_id} -> {args.output}")
    return 0 if artifact["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())

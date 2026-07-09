"""Engine correctness harness — unified runner.

Usage:
  python3 -m eval_harness                    # run all gates
  python3 -m eval_harness prefix_reuse       # run one gate
  python3 -m eval_harness prefix_reuse needle_ladder  # multiple

Env: PORT, MODEL, TEMPLATE, NEEDLE
Exit 0 = all pass, 1 = any fail.
"""
from __future__ import annotations

import json
import sys

from . import GateRunner
from .needle_ladder import NeedleLadderGate
from .prefix_reuse import PrefixReuseGate

GATES = {
    "needle_ladder": lambda: NeedleLadderGate(),
    "prefix_reuse": lambda: PrefixReuseGate(),
}


def main():
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(GATES.keys())
    unknown = [g for g in requested if g not in GATES]
    if unknown:
        print(f"Unknown gates: {unknown}. Available: {list(GATES.keys())}", file=sys.stderr)
        sys.exit(2)

    gates = [GATES[name]() for name in requested]
    runner = GateRunner(gates)
    verdicts = runner.run_all()
    report = runner.report(verdicts)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"VERDICT: {'PASS' if report['pass'] else 'FAIL'} ({report['gates_passed']})",
          file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    print(json.dumps(report, indent=2))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

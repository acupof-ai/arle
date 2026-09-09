"""Engine correctness harness — unified runner.

Usage:
  python3 -m eval_harness                    # run all gates
  python3 -m eval_harness prefix_reuse       # run one gate
  python3 -m eval_harness prefix_reuse token_reuse  # multiple

Env: PORT, MODEL, TEMPLATE; token_reuse/resident_reuse: PROMPT_TOKENS, GEN_TOKENS, PAGE;
multiturn_concurrent: CONCURRENCY, TURNS, PROMPT_TOKENS, GEN_TOKENS, PAGE
Exit 0 = all pass, 1 = any fail.
"""
from __future__ import annotations

import json
import sys

import os

from . import GateRunner
from .multiturn_concurrent import MultiTurnConcurrentGate
from .prefix_reuse import PrefixReuseGate
from .resident_reuse import ResidentReuseGate
from .token_reuse import TokenReuseGate

GATES = {
    "prefix_reuse": lambda: PrefixReuseGate(),
    "token_reuse": lambda: TokenReuseGate(
        prompt_tokens=int(os.environ.get("PROMPT_TOKENS", "500")),
        gen_tokens=int(os.environ.get("GEN_TOKENS", "128")),
        page=int(os.environ.get("PAGE", "16")),
    ),
    "resident_reuse": lambda: ResidentReuseGate(
        prompt_tokens=int(os.environ.get("PROMPT_TOKENS", "500")),
        gen_tokens=int(os.environ.get("GEN_TOKENS", "128")),
    ),
    "multiturn_concurrent": lambda: MultiTurnConcurrentGate(
        concurrency=[
            int(x) for x in os.environ.get("CONCURRENCY", "1,4,8,16").split(",") if x.strip()
        ],
        turns=int(os.environ.get("TURNS", "4")),
        prompt_tokens=int(os.environ.get("PROMPT_TOKENS", "500")),
        gen_tokens=int(os.environ.get("GEN_TOKENS", "128")),
        page=int(os.environ.get("PAGE", "16")),
    ),
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

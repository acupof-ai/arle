"""Standalone entry point for prefix_reuse_gate (backward compat).

Delegates to eval_harness.prefix_reuse.PrefixReuseGate.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from eval_harness.prefix_reuse import PrefixReuseGate

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    gate = PrefixReuseGate(target_tokens=target, runs=runs)
    try:
        gate.check_server()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)

    verdict = gate.run()
    print(json.dumps(verdict.to_dict(), indent=2))
    sys.exit(0 if verdict.passed else 1)

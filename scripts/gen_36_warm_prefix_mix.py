#!/usr/bin/env python3
"""Generate a JSONL workload with controlled warm/cold prefix mix.

Purpose: provide a deterministic dataset that exercises ARLE's
PrefixAwareAdmission gate. Synthetic guidellm random data has no prefix
overlap → all requests look "cold" → PrefixAware gate fires but win
mechanism (warm session reuse) is not measurable.

This generator emits N requests where:
  - W% share one of K shared prefixes ("warm sessions")
  - (100-W)% have unique random suffixes (cold)

Output is JSONL compatible with `guidellm benchmark run --data <path>`.

Usage:
  ./scripts/gen_36_warm_prefix_mix.py \\
      --tokenizer infer/models/Qwen3-4B/tokenizer.json \\
      --out bench-output/36-warm-mix.jsonl \\
      --num-requests 256 \\
      --warm-fraction 0.6 \\
      --num-sessions 4 \\
      --shared-prefix-tokens 1024 \\
      --tail-tokens 256 \\
      --output-tokens 128

The --warm-fraction 0.6 + --num-sessions 4 setup creates ~38 requests per
session sharing the same 1024-token prefix, plus ~102 unique cold requests.
PrefixAwareAdmission should reject ~102 cold under cold_soft_cap pressure
and admit the warm-session continuations directly.

Companion to docs/research/2026-05-10-36-prefix-aware-admission-substrate-
complete-bench-pending.md "Open question — does this workload actually
exercise the gate".

Per kernel-optimization skill v1.9.0 anti-pattern #6 ("license on capture
exists not capture reused"): #26 added "smoke-test small-shape ≠ production-
shape capture-key cardinality". Same family: bench claims need workload
that actually exercises the mechanism, not just a synthetic random hit.
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Path to Qwen3 tokenizer.json")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSONL path")
    parser.add_argument("--num-requests", type=int, default=256,
                        help="Total requests to generate (default 256)")
    parser.add_argument("--warm-fraction", type=float, default=0.6,
                        help="Fraction of requests that share a session prefix "
                             "(default 0.6)")
    parser.add_argument("--num-sessions", type=int, default=4,
                        help="Number of distinct warm session prefixes "
                             "(default 4 → ~38 requests per session at 256 total)")
    parser.add_argument("--shared-prefix-tokens", type=int, default=1024,
                        help="Tokens in the shared prefix per session "
                             "(default 1024 = above 256 typical block size to "
                             "guarantee multi-block reuse)")
    parser.add_argument("--tail-tokens", type=int, default=256,
                        help="Unique tail tokens per request (default 256, "
                             "low enough to keep full prompt < 4k)")
    parser.add_argument("--output-tokens", type=int, default=128,
                        help="Output tokens per request (default 128)")
    parser.add_argument("--seed", type=int, default=0xA12E,
                        help="RNG seed (default 0xA12E, deterministic)")
    args = parser.parse_args()

    if not args.tokenizer.exists():
        print(f"error: tokenizer not found: {args.tokenizer}", file=sys.stderr)
        return 2

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("error: install `tokenizers`: pip install tokenizers", file=sys.stderr)
        return 2

    tok = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tok.get_vocab_size()
    rng = random.Random(args.seed)

    def random_tokens(n: int) -> list[int]:
        # Avoid special tokens (commonly id < 256 are reserved). Sample from
        # a generous range that's still well within vocab.
        return [rng.randrange(256, vocab_size - 1) for _ in range(n)]

    # Pre-generate session prefixes (deterministic per seed).
    sessions = [random_tokens(args.shared_prefix_tokens)
                for _ in range(args.num_sessions)]

    num_warm = int(args.num_requests * args.warm_fraction)
    num_cold = args.num_requests - num_warm

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w") as f:
        # Warm: each request = session_prefix + unique tail
        for i in range(num_warm):
            session_idx = i % args.num_sessions
            prompt_ids = sessions[session_idx] + random_tokens(args.tail_tokens)
            prompt_text = tok.decode(prompt_ids, skip_special_tokens=False)
            row = {
                "prompt": prompt_text,
                "output_tokens": args.output_tokens,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

        # Cold: each request = unique random prefix + unique random tail
        cold_full_len = args.shared_prefix_tokens + args.tail_tokens
        for _ in range(num_cold):
            prompt_ids = random_tokens(cold_full_len)
            prompt_text = tok.decode(prompt_ids, skip_special_tokens=False)
            row = {
                "prompt": prompt_text,
                "output_tokens": args.output_tokens,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} rows to {args.out}", file=sys.stderr)
    print(f"  warm:    {num_warm} ({args.num_sessions} sessions × "
          f"~{num_warm // args.num_sessions} reqs each, "
          f"{args.shared_prefix_tokens}-tok shared prefix)", file=sys.stderr)
    print(f"  cold:    {num_cold} (unique random {cold_full_len}-tok prompts)",
          file=sys.stderr)
    print(f"  output:  {args.output_tokens} tokens per request", file=sys.stderr)
    print(f"  seed:    {args.seed} (deterministic)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

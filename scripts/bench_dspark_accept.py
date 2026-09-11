#!/usr/bin/env python3
"""DSpark acceptance-rate A/B benchmark: baseline (no train) vs trained.

Measures the mean DSpark draft acceptance rate by polling the existing
`GET /v1/stats` endpoint (`spec_decode.accepted` / `spec_decode.drafted`)
before and after a measurement window. Connects to an already-running
server like `bench_throughput.py` — no serve spawning or log parsing.

The /v1/stats counters are server-global, so the before/after snapshot MUST
bracket the whole client group: `--concurrency N` runs N clients in a thread
pool inside this one process with a single snapshot pair. Running N separate
processes would sum N overlapping global deltas and overcount the drafts.

Usage:
  # Start serve (separate terminal), then:
  python3 scripts/bench_dspark_accept.py --port 8000 --measure-requests 50
  python3 scripts/bench_dspark_accept.py --port 8000 --concurrency 8
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("Install httpx: pip install httpx")

from arle_stats import spec_decode

PROMPTS = [
    "Explain how a transformer model works in one paragraph.",
    "Write a Python function that computes the Fibonacci sequence.",
    "What is the difference between TCP and UDP? Give examples.",
    "Describe the CAP theorem and its implications for distributed systems.",
    "How does gradient descent work in machine learning?",
    "Write a Rust function that reverses a linked list.",
    "Explain bloom filters and a common use case.",
    "How does consistent hashing work in distributed caches?",
    "What are the differences between SQL and NoSQL databases?",
    "Explain the concept of context window in LLMs.",
    "Describe how speculative decoding works.",
    "What is the purpose of a tokenizer in NLP?",
    "Explain the difference between prefill and decode phases.",
    "How does KV caching improve inference efficiency?",
    "What is the role of the attention mechanism in transformers?",
]


def get_stats(client: httpx.Client) -> dict:
    """Fetch /v1/stats and return the spec_decode counters."""
    r = client.get("/v1/stats")
    r.raise_for_status()
    return spec_decode(r.json())


def send_requests(base_url: str, n: int, max_tokens: int) -> None:
    """Send n chat completion requests (one client thread) to fill the window."""
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for i in range(n):
            prompt = PROMPTS[i % len(PROMPTS)]
            try:
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "default",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                    },
                    timeout=120.0,
                )
            except Exception as e:
                print(f"  request {i} failed: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSpark acceptance-rate benchmark")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--measure-requests", type=int, default=50, help="requests per client"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="concurrent clients in one process, one stats pair",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()
    if args.concurrency < 1:
        sys.exit("--concurrency must be >= 1")

    base_url = f"http://{args.host}:{args.port}"
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        # Wait for server ready.
        for _ in range(120):
            try:
                get_stats(client)
                break
            except Exception:
                time.sleep(1.0)
        else:
            sys.exit("Server not reachable")

        # ONE baseline counter snapshot before the whole client group.
        before = get_stats(client)
        if not before.get("available", False):
            sys.exit("spec_decode stats not available — is --spec-type dspark enabled?")

        # Measurement window: N concurrent clients in-process.
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(
                pool.map(
                    lambda _: send_requests(
                        base_url, args.measure_requests, args.max_tokens
                    ),
                    range(args.concurrency),
                )
            )
        time.sleep(2.0)  # let in-flight requests settle

        # ONE post counter snapshot for the whole group.
        after = get_stats(client)

    drafted = after["drafted"] - before["drafted"]
    accepted = after["accepted"] - before["accepted"]
    rate = accepted / drafted if drafted > 0 else 0.0

    result = {
        "drafted": drafted,
        "accepted": accepted,
        "accept_rate": rate,
        "measure_requests": args.measure_requests,
        "concurrency": args.concurrency,
        "before": before,
        "after": after,
    }
    print(f"Acceptance rate (c={args.concurrency}): {rate:.4f} ({accepted}/{drafted})")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()

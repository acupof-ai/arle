#!/usr/bin/env python3
"""DSpark acceptance-rate A/B benchmark: baseline (no train) vs trained.

Measures the mean DSpark draft acceptance rate by polling the existing
`GET /v1/stats` endpoint (`spec_decode.accepted` / `spec_decode.drafted`)
before and after a measurement window. Connects to an already-running
server like `bench_throughput.py` — no serve spawning or log parsing.

Usage:
  # Start serve (separate terminal), then:
  python3 scripts/bench_dspark_accept.py --port 8000 --measure-requests 50
"""

import argparse
import json
import sys
import time
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


def send_requests(client: httpx.Client, n: int, max_tokens: int) -> None:
    """Send n chat completion requests to populate the measurement window."""
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
    parser.add_argument("--measure-requests", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()

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

        # Baseline counters (before measurement window).
        before = get_stats(client)
        if not before.get("available", False):
            sys.exit("spec_decode stats not available — is --spec-type dspark enabled?")

        # Measurement window.
        send_requests(client, args.measure_requests, args.max_tokens)
        time.sleep(2.0)  # let in-flight requests settle

        # Post counters.
        after = get_stats(client)

    drafted = after["drafted"] - before["drafted"]
    accepted = after["accepted"] - before["accepted"]
    rate = accepted / drafted if drafted > 0 else 0.0

    result = {
        "drafted": drafted,
        "accepted": accepted,
        "accept_rate": rate,
        "measure_requests": args.measure_requests,
        "before": before,
        "after": after,
    }
    print(f"Acceptance rate: {rate:.4f} ({accepted}/{drafted})")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()

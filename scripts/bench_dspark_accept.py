#!/usr/bin/env python3
"""DSpark acceptance-rate A/B benchmark: baseline (no train) vs trained.

Measures the mean DSpark draft acceptance rate with and without the
`--dspark-train` sidecar, on the same prompt set. The serve binary must
be built with the `dspark_accept:` log line (see
`crates/infer-cuda/src/executor/dspark_train.rs::record_accept`).

Usage:
  # Baseline (no training)
  python3 scripts/bench_dspark_accept.py --mode baseline --serve-cmd "arle serve ..."

  # Trained (sidecar runs for --train-seconds before measurement)
  python3 scripts/bench_dspark_accept.py --mode trained --train-seconds 300 --serve-cmd "arle serve ..."
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Install openai: pip install openai")

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

ACCEPT_RE = re.compile(r"dspark_accept: avg_rate=([0-9.]+)")


def wait_for_ready(client: OpenAI, timeout: float = 120.0) -> None:
    """Poll the server until it responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"Serve did not become ready within {timeout}s")


def send_requests(client: OpenAI, n: int, max_tokens: int = 64) -> int:
    """Send n chat completion requests; return total completion tokens."""
    total = 0
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        try:
            resp = client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            total += resp.usage.completion_tokens
        except Exception as e:
            print(f"  request {i} failed: {e}", file=sys.stderr)
    return total


def parse_accept_rates(log_text: str) -> list[float]:
    """Extract all avg_rate values from serve log output."""
    return [float(m.group(1)) for m in ACCEPT_RE.finditer(log_text)]


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run one benchmark mode and return results dict."""
    print(f"=== Mode: {args.mode} ===")
    print(f"Serve cmd: {args.serve_cmd}")

    # Start serve, capture stdout+stderr to a log file.
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w")
    serve_proc = subprocess.Popen(
        args.serve_cmd,
        shell=True,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    print(f"Serve PID: {serve_proc.pid}, log: {log_path}")

    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="x")
        print("Waiting for serve ready...")
        wait_for_ready(client, timeout=args.ready_timeout)
        print("Serve ready.")

        if args.mode == "trained":
            # Phase 1: feed experiences so the trainer has data.
            print(f"Feeding {args.feed_requests} requests to populate experience buffer...")
            send_requests(client, args.feed_requests, max_tokens=args.max_tokens)
            # Phase 2: let the trainer run.
            print(f"Training for {args.train_seconds}s...")
            time.sleep(args.train_seconds)

        # Phase 3: measurement.
        print(f"Sending {args.measure_requests} measurement requests...")
        send_requests(client, args.measure_requests, max_tokens=args.max_tokens)
        # Let the log flush.
        time.sleep(3.0)
    finally:
        print("Stopping serve...")
        serve_proc.terminate()
        try:
            serve_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            serve_proc.kill()
            serve_proc.wait()
        log_f.close()

    # Parse acceptance rates from the log.
    log_text = log_path.read_text()
    rates = parse_accept_rates(log_text)
    result = {
        "mode": args.mode,
        "serve_cmd": args.serve_cmd,
        "log_file": str(log_path),
        "num_accept_samples": len(rates),
        "accept_rates": rates,
        "mean_accept_rate": sum(rates) / len(rates) if rates else None,
    }
    if rates:
        print(f"Acceptance rate samples: {len(rates)}")
        print(f"Mean acceptance rate: {result['mean_accept_rate']:.4f}")
        print(f"  min={min(rates):.4f} max={max(rates):.4f}")
    else:
        print("WARNING: no dspark_accept lines found in log!", file=sys.stderr)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DSpark acceptance-rate A/B benchmark")
    parser.add_argument("--mode", choices=["baseline", "trained"], required=True)
    parser.add_argument("--serve-cmd", required=True, help="Full serve command line")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--feed-requests", type=int, default=50,
                        help="Requests to feed the trainer (trained mode only)")
    parser.add_argument("--train-seconds", type=int, default=300,
                        help="Seconds to let the trainer run (trained mode only)")
    parser.add_argument("--measure-requests", type=int, default=50,
                        help="Requests used for acceptance-rate measurement")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument("--log-file", default="/tmp/dspark_accept_bench.log")
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()

    result = run_benchmark(args)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()

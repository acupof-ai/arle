#!/usr/bin/env python3
"""Throughput sweep benchmark: vary input/output lengths and concurrency.

Measures throughput (output tok/s), TTFT, and ITL across different
sequence length configurations. Results are comparable with sglang/vllm benchmarks.

Usage:
  # Against infer server
  python3 scripts/bench_throughput_sweep.py --url http://localhost:8000

  # Against sglang server
  python3 scripts/bench_throughput_sweep.py --url http://localhost:30000 --label sglang

  # Quick mode (fewer configs)
  python3 scripts/bench_throughput_sweep.py --url http://localhost:8000 --quick
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")


# Sweep configurations: (input_len, output_len, concurrency)
SWEEP_CONFIGS = [
    # Short context, vary output
    (128,   64,  1),
    (128,  128,  1),
    (128,  256,  1),
    (128,  512,  1),
    # Medium context
    (512,  128,  1),
    (512,  256,  1),
    (512,  512,  1),
    # Long context
    (1024, 128,  1),
    (1024, 256,  1),
    (1024, 512,  1),
    (2048, 256,  1),
    # Concurrency sweep (medium context)
    (512,  256,  2),
    (512,  256,  4),
    # Concurrency sweep (short context)
    (128,  128,  2),
    (128,  128,  4),
    # High concurrency sweep
    (128,  256,  8),
    (512,  256,  8),
    (128,  256, 16),
    (512,  256, 16),
    (128,  256, 32),
    (512,  256, 32),
    (128,  256, 64),
]

QUICK_CONFIGS = [
    (128,  128,  1),
    (128,  512,  1),
    (512,  256,  1),
    (1024, 256,  1),
    (2048, 256,  1),
    (512,  256,  4),
]

# Generate synthetic prompt of target token length
# ~4 chars per token for English text
def make_prompt(target_tokens: int) -> str:
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "in", "a", "world", "of", "data", "and", "models", "we", "find",
        "that", "performance", "matters", "for", "every", "single", "token",
        "generated", "by", "our", "system", "running", "on", "gpu",
    ]
    target_chars = target_tokens * 4
    parts = []
    while len(" ".join(parts)) < target_chars:
        parts.append(random.choice(words))
    text = " ".join(parts)[:target_chars]
    return f"Repeat the following text back to me exactly, then continue writing similar text:\n\n{text}"


@dataclass
class RunResult:
    input_len: int
    output_len: int
    concurrency: int
    num_requests: int
    total_output_tokens: int
    wall_time: float
    throughput: float  # output tok/s
    ttft_p50: float
    ttft_p99: float
    itl_p50: float
    itl_p99: float
    errors: int


async def single_request(client, url, prompt, max_tokens):
    """Send a single streaming request, return (ttft_ms, itl_ms, output_tokens, elapsed)."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    ttft = None
    token_times = []
    output_tokens = 0
    t0 = time.time()

    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=payload, timeout=120) as r:
            if r.status_code != 200:
                await r.aread()
                return None, None, 0, time.time() - t0, True

            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                d = line[6:].strip()
                if d == "[DONE]":
                    break
                try:
                    chunk = json.loads(d)
                except:
                    continue
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        now = time.time()
                        if ttft is None:
                            ttft = (now - t0) * 1000
                        token_times.append(now)
                        output_tokens += 1  # approximate: 1 SSE chunk ≈ 1 token

    except Exception:
        return None, None, 0, time.time() - t0, True

    elapsed = time.time() - t0

    # Compute ITL
    itl = None
    if len(token_times) > 1:
        deltas = [(token_times[i] - token_times[i-1]) * 1000 for i in range(1, len(token_times))]
        itl = statistics.median(deltas)

    return ttft, itl, output_tokens, elapsed, False


async def run_config(client, url, input_len, output_len, concurrency, num_requests=8) -> RunResult:
    """Run one sweep configuration."""
    prompt = make_prompt(input_len)

    all_ttft = []
    all_itl = []
    total_tokens = 0
    errors = 0

    t0 = time.time()

    # Run in batches of `concurrency`
    for batch_start in range(0, num_requests, concurrency):
        batch_size = min(concurrency, num_requests - batch_start)
        tasks = [single_request(client, url, prompt, output_len) for _ in range(batch_size)]
        results = await asyncio.gather(*tasks)

        for ttft, itl, n_tok, elapsed, is_err in results:
            if is_err:
                errors += 1
                continue
            total_tokens += n_tok
            if ttft is not None:
                all_ttft.append(ttft)
            if itl is not None:
                all_itl.append(itl)

    wall_time = time.time() - t0
    throughput = total_tokens / wall_time if wall_time > 0 else 0

    def pctl(arr, p):
        if not arr:
            return 0
        arr_sorted = sorted(arr)
        idx = int(len(arr_sorted) * p / 100)
        return arr_sorted[min(idx, len(arr_sorted) - 1)]

    return RunResult(
        input_len=input_len,
        output_len=output_len,
        concurrency=concurrency,
        num_requests=num_requests,
        total_output_tokens=total_tokens,
        wall_time=wall_time,
        throughput=throughput,
        ttft_p50=pctl(all_ttft, 50),
        ttft_p99=pctl(all_ttft, 99),
        itl_p50=pctl(all_itl, 50),
        itl_p99=pctl(all_itl, 99),
        errors=errors,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--label", default="infer", help="Label for this server")
    parser.add_argument("--quick", action="store_true", help="Run fewer configs")
    parser.add_argument("--requests", type=int, default=8, help="Requests per config")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON snapshot file")
    args = parser.parse_args()

    configs = QUICK_CONFIGS if args.quick else SWEEP_CONFIGS

    print("=" * 100)
    print(f"Throughput Sweep Benchmark — {args.label}")
    print(f"  Server:   {args.url}")
    print(f"  Configs:  {len(configs)}")
    print(f"  Requests: {args.requests} per config")
    print("=" * 100)

    async with httpx.AsyncClient() as client:
        # Verify server
        try:
            r = await client.get(f"{args.url}/v1/stats", timeout=5)
            if r.status_code != 200:
                # Try models endpoint (sglang compat)
                r = await client.get(f"{args.url}/v1/models", timeout=5)
            print(f"  Server:   OK")
        except Exception as e:
            print(f"  Server:   UNREACHABLE ({e})")
            return

        print()
        hdr = f"{'In':>5} | {'Out':>5} | {'C':>2} | {'Throughput':>10} | {'TTFT p50':>9} | {'TTFT p99':>9} | {'ITL p50':>8} | {'ITL p99':>8} | {'Err':>3} | {'Wall':>6}"
        print(hdr)
        print("-" * len(hdr))

        results = []
        for input_len, output_len, conc in configs:
            num_req = max(args.requests, conc * 2)
            r = await run_config(client, args.url, input_len, output_len, conc, num_req)
            results.append(r)

            print(
                f"{r.input_len:5d} | {r.output_len:5d} | {r.concurrency:2d} | "
                f"{r.throughput:8.1f} t/s | "
                f"{r.ttft_p50:7.0f}ms | {r.ttft_p99:7.0f}ms | "
                f"{r.itl_p50:6.1f}ms | {r.itl_p99:6.1f}ms | "
                f"{r.errors:3d} | {r.wall_time:5.1f}s"
            )

        # Summary
        print()
        print("=" * 100)
        print(f"SUMMARY — {args.label}")
        print("=" * 100)

        ok_results = [r for r in results if r.errors < r.num_requests]
        if ok_results:
            best = max(ok_results, key=lambda r: r.throughput)
            print(f"  Peak throughput:    {best.throughput:.1f} tok/s (in={best.input_len}, out={best.output_len}, C={best.concurrency})")

            c1 = [r for r in ok_results if r.concurrency == 1]
            if c1:
                best_c1 = max(c1, key=lambda r: r.throughput)
                print(f"  Peak (C=1):         {best_c1.throughput:.1f} tok/s (in={best_c1.input_len}, out={best_c1.output_len})")

            all_itl = [r.itl_p50 for r in ok_results if r.itl_p50 > 0]
            if all_itl:
                print(f"  ITL p50 range:      {min(all_itl):.1f}ms — {max(all_itl):.1f}ms")

        total_tok = sum(r.total_output_tokens for r in results)
        total_wall = sum(r.wall_time for r in results)
        print(f"  Total tokens:       {total_tok:,}")
        print(f"  Total wall time:    {total_wall:.1f}s")
        print("=" * 100)

        # Save JSON snapshot for regression tracking
        if args.save:
            import dataclasses, datetime, platform, subprocess
            gpu_name = "unknown"
            try:
                gpu_name = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    timeout=5
                ).decode().strip()
            except Exception:
                pass
            snapshot = {
                "label": args.label,
                "timestamp": datetime.datetime.now().isoformat(),
                "gpu": gpu_name,
                "platform": platform.platform(),
                "url": args.url,
                "configs": [dataclasses.asdict(r) for r in results],
            }
            import pathlib
            out_path = pathlib.Path(args.save)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(snapshot, f, indent=2)
            print(f"\nSnapshot saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())

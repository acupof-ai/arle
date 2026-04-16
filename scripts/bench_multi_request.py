#!/usr/bin/env python3
"""Multi-request benchmark for infer scheduler.

Usage:
  python3 scripts/bench_multi_request.py --url http://localhost:8000
"""

import argparse
import asyncio
import time
import statistics
import httpx
import json
import sys

_args = None


def _base_url():
    return f"{_args.url}/v1/completions"


def _model():
    return _args.model

# Shared long system prompt (~1500 tokens) to test prefix cache
SYSTEM_PREFIX = """<|im_start|>system
You are a helpful AI coding assistant with access to tools including web_search, read_file, write_file, execute_command, and code_search. Follow best practices for clean code, error handling, and security. Always think step by step.

Available tools:
- web_search(query, num_results=10): Search the web
- read_file(file_path, offset=0, limit=2000): Read file contents
- write_file(file_path, content): Write to file
- execute_command(command, timeout=120): Run shell command
- code_search(pattern, path=".", file_glob=None): Search code

Response format: be concise, show code snippets, suggest follow-ups.
<|im_end|>
<|im_start|>user
"""

QUERIES = [
    "What is 2+2?",
    "Name 3 programming languages.",
    "What is the capital of France?",
    "Explain HTTP status 404.",
    "What does 'git rebase' do?",
    "List 5 prime numbers.",
    "What is Big O notation?",
    "Define 'recursion' briefly.",
]

LONG_QUERIES = [
    "Write a detailed explanation of how hash tables work, including collision handling, load factors, and amortized complexity analysis.",
    "Explain the complete lifecycle of an HTTP request from the browser to the server and back, including DNS, TCP, TLS, and HTTP/2 multiplexing.",
]


async def send_request(client, prompt, max_tokens=32, stream=False):
    """Send one completion request, return (text, elapsed_ms, prompt_tokens, completion_tokens)."""
    t0 = time.perf_counter()
    if stream:
        tokens_times = []
        text = ""
        async with client.stream("POST", _base_url(), json={
            "model": _model(), "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "stream": True,
        }, timeout=120) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    t = chunk.get("choices", [{}])[0].get("text", "")
                    if t:
                        tokens_times.append(time.perf_counter())
                        text += t
        elapsed = (time.perf_counter() - t0) * 1000
        return {"text": text, "elapsed_ms": elapsed, "token_times": tokens_times, "t0": t0}
    else:
        resp = await client.post(_base_url(), json={
            "model": _model(), "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0,
        }, timeout=120)
        elapsed = (time.perf_counter() - t0) * 1000
        data = resp.json()
        return {
            "text": data["choices"][0]["text"],
            "elapsed_ms": elapsed,
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
        }


async def bench_sequential(client, n=4, max_tokens=32):
    """Baseline: send requests one at a time."""
    times = []
    for i, q in enumerate(QUERIES[:n]):
        prompt = SYSTEM_PREFIX + q + "<|im_end|>\n<|im_start|>assistant\n"
        r = await send_request(client, prompt, max_tokens)
        times.append(r["elapsed_ms"])
        print(f"  Seq {i+1}: {r['elapsed_ms']:7.1f}ms  ({r.get('completion_tokens','?')} tok) {r['text'][:50]}...")
    return times


async def bench_concurrent(client, n=4, max_tokens=32):
    """Send N requests concurrently."""
    prompts = []
    for q in QUERIES[:n]:
        prompts.append(SYSTEM_PREFIX + q + "<|im_end|>\n<|im_start|>assistant\n")

    tasks = [send_request(client, p, max_tokens) for p in prompts]
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_time = (time.perf_counter() - t0) * 1000

    times = []
    for i, r in enumerate(results):
        times.append(r["elapsed_ms"])
        print(f"  Con {i+1}: {r['elapsed_ms']:7.1f}ms  ({r.get('completion_tokens','?')} tok) {r['text'][:50]}...")

    return times, wall_time


async def bench_streaming(client):
    """Streaming request: measure TTFT and ITL."""
    prompt = SYSTEM_PREFIX + "Write a haiku about coding." + "<|im_end|>\n<|im_start|>assistant\n"
    r = await send_request(client, prompt, max_tokens=64, stream=True)

    if r["token_times"]:
        ttft = (r["token_times"][0] - r["t0"]) * 1000
        if len(r["token_times"]) > 1:
            itls = [(r["token_times"][i] - r["token_times"][i-1]) * 1000
                    for i in range(1, len(r["token_times"]))]
            avg_itl = statistics.mean(itls)
        else:
            avg_itl = 0
        print(f"  TTFT: {ttft:.1f}ms")
        print(f"  Avg ITL: {avg_itl:.1f}ms ({len(r['token_times'])} chunks)")
        print(f"  Total: {r['elapsed_ms']:.1f}ms")
        print(f"  Text: {r['text'][:80]}...")
    return r


async def bench_interleave(client):
    """Send a long request, then a short one. Short should not wait for long."""
    long_prompt = SYSTEM_PREFIX + LONG_QUERIES[0] + "<|im_end|>\n<|im_start|>assistant\n"
    short_prompt = SYSTEM_PREFIX + "What is 1+1?" + "<|im_end|>\n<|im_start|>assistant\n"

    t0 = time.perf_counter()
    long_task = asyncio.create_task(send_request(client, long_prompt, max_tokens=128))
    await asyncio.sleep(0.1)  # slight delay so long starts first
    short_task = asyncio.create_task(send_request(client, short_prompt, max_tokens=16))

    short_result = await short_task
    long_result = await long_task

    print(f"  Long:  {long_result['elapsed_ms']:7.1f}ms ({long_result.get('completion_tokens','?')} tok)")
    print(f"  Short: {short_result['elapsed_ms']:7.1f}ms ({short_result.get('completion_tokens','?')} tok)")

    return long_result, short_result


async def bench_stress(client, n=8, max_tokens=32):
    """Stress test: N concurrent requests."""
    prompts = []
    for i in range(n):
        q = QUERIES[i % len(QUERIES)]
        prompts.append(SYSTEM_PREFIX + q + "<|im_end|>\n<|im_start|>assistant\n")

    t0 = time.perf_counter()
    tasks = [send_request(client, p, max_tokens) for p in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_time = (time.perf_counter() - t0) * 1000

    successes = [r for r in results if isinstance(r, dict)]
    failures = [r for r in results if isinstance(r, Exception)]

    times = [r["elapsed_ms"] for r in successes]
    if times:
        print(f"  Success: {len(successes)}/{n}")
        print(f"  Failures: {len(failures)}")
        print(f"  Wall time: {wall_time:.0f}ms")
        print(f"  Per-request: min={min(times):.0f}ms  avg={statistics.mean(times):.0f}ms  max={max(times):.0f}ms")
        total_tokens = sum(r.get("completion_tokens", 0) for r in successes)
        print(f"  Total tokens: {total_tokens}  ({total_tokens / (wall_time/1000):.1f} tok/s throughput)")
    if failures:
        for f in failures:
            print(f"  Error: {f}")

    return successes, failures, wall_time


async def fetch_stats(url: str) -> dict | None:
    async with httpx.AsyncClient() as c:
        try:
            resp = await c.get(f"{url}/v1/stats", timeout=5)
            if resp.status_code != 200:
                return None
            return {k: v for tok in resp.text.split() if "=" in tok for k, v in [tok.split("=", 1)]}
        except Exception:
            return None


async def main():
    print("=" * 70)
    print("PEGAINFER MULTI-REQUEST BENCHMARK")
    print(f"Model: {_model()}  |  Server: {_args.url}")
    print("=" * 70)

    stats_before = await fetch_stats(_args.url)

    async with httpx.AsyncClient() as client:
        # Warmup
        print("\n[Warmup]")
        prompt = SYSTEM_PREFIX + "Say hi.<|im_end|>\n<|im_start|>assistant\n"
        r = await send_request(client, prompt, max_tokens=8)
        print(f"  Warmup: {r['elapsed_ms']:.1f}ms")

        # 1. Sequential baseline
        print("\n[1] Sequential (1 request at a time, N=4)")
        seq_times = await bench_sequential(client, n=4, max_tokens=32)
        seq_total = sum(seq_times)
        seq_avg = statistics.mean(seq_times)

        # 2. Concurrent (same N)
        print(f"\n[2] Concurrent (4 requests simultaneously)")
        con_times, con_wall = await bench_concurrent(client, n=4, max_tokens=32)
        con_avg = statistics.mean(con_times)

        # 3. Streaming
        print(f"\n[3] Streaming")
        await bench_streaming(client)

        # 4. Interleave test
        print(f"\n[4] Interleave (long=128tok + short=16tok)")
        long_r, short_r = await bench_interleave(client)

        # 5. Stress N=8
        print(f"\n[5] Stress test (N=8, max_tokens=32)")
        successes_8, _, wall_8 = await bench_stress(client, n=8, max_tokens=32)

        # 6. Stress N=16
        print(f"\n[6] Stress test (N=16, max_tokens=32)")
        successes_16, _, wall_16 = await bench_stress(client, n=16, max_tokens=32)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Metric':<40} {'Value':>12}")
        print("-" * 55)
        print(f"{'Sequential avg (N=4, 32tok)':<40} {seq_avg:>10.1f}ms")
        print(f"{'Sequential total (N=4)':<40} {seq_total:>10.1f}ms")
        print(f"{'Concurrent avg (N=4, 32tok)':<40} {con_avg:>10.1f}ms")
        print(f"{'Concurrent wall time (N=4)':<40} {con_wall:>10.1f}ms")
        print(f"{'Concurrent speedup vs sequential':<40} {seq_total/con_wall:>10.2f}x")
        print(f"{'Interleave: short request latency':<40} {short_r['elapsed_ms']:>10.1f}ms")
        print(f"{'Interleave: long request latency':<40} {long_r['elapsed_ms']:>10.1f}ms")
        if successes_8:
            t8 = sum(r.get("completion_tokens", 0) for r in successes_8)
            print(f"{'Stress N=8 wall time':<40} {wall_8:>10.0f}ms")
            print(f"{'Stress N=8 throughput':<40} {t8/(wall_8/1000):>9.1f} tok/s")
        if successes_16:
            t16 = sum(r.get("completion_tokens", 0) for r in successes_16)
            print(f"{'Stress N=16 wall time':<40} {wall_16:>10.0f}ms")
            print(f"{'Stress N=16 throughput':<40} {t16/(wall_16/1000):>9.1f} tok/s")
        print("=" * 70)

    stats_after = await fetch_stats(_args.url)
    if stats_after:
        print(f"\n/v1/stats: prefix_hit_rate={stats_after.get('prefix_hit_rate','?')} "
              f"kv_util={stats_after.get('kv_util','?')} "
              f"ttft_p50={stats_after.get('ttft_p50','?')} tpot_p50={stats_after.get('tpot_p50','?')}")


def cli():
    global _args
    p = argparse.ArgumentParser(description="Multi-request scheduler benchmark")
    p.add_argument("--url", default="http://localhost:8000", help="Server URL")
    p.add_argument("--model", default="default", help="Model name in request body")
    _args = p.parse_args()
    asyncio.run(main())


if __name__ == "__main__":
    cli()

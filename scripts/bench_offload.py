#!/usr/bin/env python3
"""Benchmark KV prefix cache reuse vs cold re-prefill.

Two modes, each running against a fresh server:
1. EVICTION: interleave unrelated prompts to evict cached prefix
2. KV CACHE: send related prompts sequentially to reuse prefix

Measures: per-query latency, throughput, prefix hit/miss from both
server logs and /v1/stats (now wired for CUDA backend).
"""

import argparse
import json
import os
import re
import subprocess
import statistics
import sys
import time


SYSTEM_PREFIX = (
    "You are a helpful AI assistant with extensive capabilities. "
    "You have access to tools including python execution, shell commands, "
    "file operations, web search, and database queries. "
    "Always think step by step. Be precise and thorough. "
    "When using tools, validate inputs and handle errors gracefully. "
) * 5  # ~500 tokens of system context

QUERIES = [
    "What is 2+2?",
    "What is the capital of France?",
    "Calculate 100 factorial.",
    "List the first 10 prime numbers.",
    "What is the speed of light?",
    "Name three oceans.",
    "What year was Python created?",
    "How many days in a leap year?",
]


def fetch_stats(base_url: str) -> dict | None:
    import httpx
    try:
        resp = httpx.get(f"{base_url}/v1/stats", timeout=5)
        if resp.status_code != 200:
            return None
        fields = {}
        for tok in resp.text.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                fields[k] = v
        return fields
    except Exception:
        return None


def run_queries(args, label: str, evict_between: bool) -> dict:
    """Run all queries against a fresh server, return results."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia:/usr/local/cuda/lib64"

    subprocess.run(["killall", "-9", "infer"], capture_output=True)
    time.sleep(1)

    server = subprocess.Popen(
        ["./target/release/infer",
         "--model-path", args.model_path,
         "--port", str(args.port),
         "--cuda-graph=false"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(7)

    import httpx
    base_url = f"http://localhost:{args.port}"

    # Warmup request
    httpx.post(f"{base_url}/v1/completions", json={
        "prompt": "warmup", "max_tokens": 1, "temperature": 0,
    }, timeout=30)

    stats_before = fetch_stats(base_url)
    results = []

    for i, query in enumerate(QUERIES):
        prompt = f"{SYSTEM_PREFIX}\n\nQuestion: {query}\nAnswer:"

        if evict_between and i > 0:
            httpx.post(f"{base_url}/v1/completions", json={
                "prompt": "EVICT" * 50, "max_tokens": 1, "temperature": 0,
            }, timeout=30)

        start = time.perf_counter()
        resp = httpx.post(f"{base_url}/v1/completions", json={
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stop": ["\n\n"],
        }, timeout=60)
        elapsed = (time.perf_counter() - start) * 1000
        data = resp.json()

        prompt_toks = data["usage"]["prompt_tokens"]
        compl_toks = data["usage"]["completion_tokens"]
        tok_s = compl_toks / (elapsed / 1000) if elapsed > 0 else 0

        results.append({
            "query": query[:40],
            "time_ms": round(elapsed, 1),
            "prompt_tokens": prompt_toks,
            "completion_tokens": compl_toks,
            "tok_s": round(tok_s, 1),
        })

    stats_after = fetch_stats(base_url)

    # Parse server logs
    server.terminate()
    server.wait()
    log = server.stdout.read().decode(errors="replace")
    kv_hits = len(re.findall(r"prefix (HIT|PARTIAL)", log))
    kv_misses = len(re.findall(r"prefix MISS", log))

    times = [r["time_ms"] for r in results]
    total_compl = sum(r["completion_tokens"] for r in results)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_time_s = sum(times) / 1000

    return {
        "label": label,
        "results": results,
        "avg_time_ms": round(statistics.mean(times), 1),
        "p50_time_ms": round(statistics.median(times), 1),
        "total_time_ms": round(sum(times)),
        "total_tokens": total_compl,
        "total_prompt_tokens": total_prompt,
        "throughput_tok_s": round(total_compl / total_time_s, 1) if total_time_s > 0 else 0,
        "kv_hits": kv_hits,
        "kv_misses": kv_misses,
        "stats_before": stats_before,
        "stats_after": stats_after,
    }


def main():
    p = argparse.ArgumentParser(description="KV prefix cache benchmark: eviction vs reuse")
    p.add_argument("--model-path", default="models/Qwen3-4B")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--out", default=None, help="JSON output path")
    args = p.parse_args()

    print("=" * 70)
    print("KV Prefix Cache: Eviction vs Reuse Benchmark")
    print("=" * 70)
    print(f"Model: {args.model_path}  Port: {args.port}  Max tokens: {args.max_tokens}")
    print(f"Queries: {len(QUERIES)}  Shared prefix: ~{len(SYSTEM_PREFIX.split())} words")
    print()

    print("[1/2] Running EVICTION mode ...")
    evict = run_queries(args, "EVICTION (recompute)", evict_between=True)
    time.sleep(2)
    print("[2/2] Running KV CACHE mode ...")
    cache = run_queries(args, "KV CACHE (prefix reused)", evict_between=False)

    # Report
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for r in [evict, cache]:
        print(f"\n  {r['label']}:")
        print(f"  {'Query':<42} {'Time':>8} {'Prompt':>7} {'Compl':>6} {'tok/s':>7}")
        print(f"  {'-'*72}")
        for q in r["results"]:
            print(f"  {q['query']:<42} {q['time_ms']:>7.1f}ms {q['prompt_tokens']:>6} "
                  f"{q['completion_tokens']:>5} {q['tok_s']:>6.1f}")
        print(f"  {'-'*72}")
        print(f"  Avg: {r['avg_time_ms']:.1f}ms  P50: {r['p50_time_ms']:.1f}ms  "
              f"Throughput: {r['throughput_tok_s']:.1f} tok/s  "
              f"Hits: {r['kv_hits']}  Misses: {r['kv_misses']}")
        if r["stats_after"]:
            sa = r["stats_after"]
            print(f"  /v1/stats: prefix_hit_rate={sa.get('prefix_hit_rate','?')} "
                  f"kv_util={sa.get('kv_util','?')} "
                  f"ttft_p50={sa.get('ttft_p50','?')} tpot_p50={sa.get('tpot_p50','?')}")

    speedup = evict["avg_time_ms"] / cache["avg_time_ms"] if cache["avg_time_ms"] > 0 else 0
    saved_ms = evict["total_time_ms"] - cache["total_time_ms"]
    pct = saved_ms / evict["total_time_ms"] * 100 if evict["total_time_ms"] > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  KV cache speedup: {speedup:.2f}x  |  Time saved: {saved_ms:.0f}ms ({pct:.1f}%)")
    print(f"{'=' * 70}")

    if args.out:
        import datetime
        snapshot = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": args.model_path,
            "max_tokens": args.max_tokens,
            "eviction": evict,
            "cache": cache,
            "speedup": round(speedup, 3),
        }
        out = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

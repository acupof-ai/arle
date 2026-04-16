"""Benchmark KV offload vs eviction under memory pressure.

Simulates high concurrency: multiple "requests" compete for limited GPU KV space.
Each request shares a long system prompt (prefix), simulating agent conversations.

Two modes:
1. EVICTION: KV cache resets between requests (no offload, recompute prefix each time)
2. OFFLOAD: KV prefix offloaded to CPU, fetched back instead of recomputed

Measures: time per request, total throughput, prefix recomputation savings.
"""

import subprocess
import time
import re
import sys
import os

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-4B"

# Shared long system prompt (simulates agent tool definitions + context)
SYSTEM_PREFIX = (
    "You are a helpful AI assistant with extensive capabilities. "
    "You have access to tools including python execution, shell commands, "
    "file operations, web search, and database queries. "
    "Always think step by step. Be precise and thorough. "
    "When using tools, validate inputs and handle errors gracefully. "
) * 5  # ~500 tokens of system context

# Simulate 8 sequential "concurrent" requests sharing the same prefix
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


def run_queries(label: str, max_gpu_kv: int | None, evict_between: bool) -> dict:
    """Run all queries sequentially, measuring time per query."""
    # Start infer HTTP server
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia:/usr/local/cuda/lib64"

    # Kill any existing server
    subprocess.run(["pkill", "-9", "-f", "target/release/infer"], capture_output=True)
    time.sleep(1)

    server = subprocess.Popen(
        ["./target/release/infer",
         "--model-path", MODEL_PATH,
         "--port", "8100",
         "--cuda-graph=false"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(6)

    import httpx

    results = []
    total_offloads = 0
    total_prefetches = 0

    for i, query in enumerate(QUERIES):
        prompt = f"{SYSTEM_PREFIX}\n\nQuestion: {query}\nAnswer:"

        if evict_between and i > 0:
            # Simulate eviction: send a totally different prompt to evict cached KV
            httpx.post("http://localhost:8100/v1/completions", json={
                "prompt": "EVICT" * 50,
                "max_tokens": 1,
                "temperature": 0,
            }, timeout=30)

        start = time.perf_counter()
        resp = httpx.post("http://localhost:8100/v1/completions", json={
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
            "stop": ["\n\n"],
        }, timeout=60)
        elapsed = (time.perf_counter() - start) * 1000
        data = resp.json()

        results.append({
            "query": query[:40],
            "time_ms": elapsed,
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
        })

    # Read server logs for prefix-cache hits/misses
    server.terminate()
    server.wait()
    log = server.stdout.read().decode(errors="replace")
    # Current tracing format: "Request N → slot M (prompt=P tokens, radix_hit=H, reusable_prefix=R, ...)"
    # A radix_hit > 0 with reusable_prefix > 0 counts as a prefix hit.
    kv_hits = len(re.findall(r"reusable_prefix=[1-9]\d*", log))
    # "prefix HIT" appears in the prefill path for exact full prefix matches
    kv_hits += len(re.findall(r"prefix HIT", log))
    # Misses: radix_hit=0 or "not reusable"
    kv_misses = len(re.findall(r"radix_hit=0", log))
    kv_misses += len(re.findall(r"not reusable", log))
    total_offloads = len(re.findall(r"offload:", log))
    total_prefetches = len(re.findall(r"prefetch:", log))

    avg_time = sum(r["time_ms"] for r in results) / len(results)
    total_time = sum(r["time_ms"] for r in results)

    return {
        "label": label,
        "results": results,
        "avg_time_ms": avg_time,
        "total_time_ms": total_time,
        "kv_hits": kv_hits,
        "kv_misses": kv_misses,
        "offloads": total_offloads,
        "prefetches": total_prefetches,
    }


def main():
    print("=" * 70)
    print("KV Offload vs Eviction Benchmark")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Queries: {len(QUERIES)} sequential (simulated concurrency)")
    print(f"Shared prefix: ~{len(SYSTEM_PREFIX.split())} words")
    print()

    # Mode 1: Eviction (send different prompt between requests to evict KV)
    print("[1/2] Running EVICTION mode (no KV reuse, recompute prefix each time)...")
    evict_result = run_queries("EVICTION (recompute)", max_gpu_kv=None, evict_between=True)

    time.sleep(2)

    # Mode 2: KV prefix cache (requests share prefix, KV reused)
    print("[2/2] Running KV CACHE mode (prefix reused across requests)...")
    cache_result = run_queries("KV CACHE (prefix reused)", max_gpu_kv=None, evict_between=False)

    # Report
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for r in [evict_result, cache_result]:
        print(f"\n  {r['label']}:")
        print(f"  {'Query':<42} {'Time':>8} {'Prompt':>7} {'Compl':>6}")
        print(f"  {'-'*65}")
        for q in r["results"]:
            print(f"  {q['query']:<42} {q['time_ms']:>7.1f}ms {q['prompt_tokens']:>6} {q['completion_tokens']:>5}")
        print(f"  {'-'*65}")
        print(f"  Avg: {r['avg_time_ms']:.1f}ms | Total: {r['total_time_ms']:.0f}ms | "
              f"KV hits: {r['kv_hits']} | Misses: {r['kv_misses']}")

    speedup = evict_result["avg_time_ms"] / cache_result["avg_time_ms"]
    saved_ms = evict_result["total_time_ms"] - cache_result["total_time_ms"]

    print(f"\n{'=' * 70}")
    print(f"  KV cache speedup: {speedup:.2f}x")
    print(f"  Total time saved: {saved_ms:.0f}ms ({saved_ms/evict_result['total_time_ms']*100:.1f}%)")
    print(f"  KV cache = inference acceleration, not a requirement")
    print(f"  Without KV: inference works fine, just slower (recomputes prefix)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

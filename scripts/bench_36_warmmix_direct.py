#!/usr/bin/env python3
"""Direct HTTP benchmark for #36 PrefixAware warm-mix workload.

Replaces GuideLLM JSONL path which is broken for this finite workload
(per docs/experience/errors/2026-05-10-36-warmmix-guidellm-jsonl-invalid.md
8150bfe — drains 256 rows in 1.3s, reports TTFT p50=0.0, B-arm reports
0 successful requests despite /v1/stats showing 257 served).

Derived from scripts/bench_multitenant_burst.py (M_ibp Phase 0 runner)
extended with:
  - Read JSONL workload (matches scripts/gen_36_warm_prefix_mix.py output)
  - Track per-row warm vs cold label (first WARM_COUNT warm, rest cold)
  - Warm-vs-cold p50/p95/p99 TTFT split
  - /v1/stats capture before + after
  - License/kill threshold report inline

Usage:
  python3 scripts/bench_36_warmmix_direct.py \\
    http://localhost:8765 Qwen3-4B-W4-hybrid-zpfix \\
    bench-output/36-warm-mix.jsonl \\
    --concurrency 8 \\
    --warm-count 153

Companion to:
  - scripts/gen_36_warm_prefix_mix.py (workload generator)
  - 8150bfe codex errors entry "Next Step" recommendation
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import List, Tuple

import aiohttp


async def fire_completion(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    request_id: int,
    is_warm: bool,
) -> Tuple[int, bool, float, float, int]:
    """Fire one /v1/completions request, return
    (request_id, is_warm, TTFT_ms, total_latency_ms, output_tokens_received)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    start = time.perf_counter()
    ttft = None
    output_count = 0
    async with session.post(url, json=payload, timeout=300) as resp:
        async for line in resp.content:
            if not line.startswith(b"data: "):
                continue
            chunk = line[6:].strip()
            if chunk == b"[DONE]":
                break
            try:
                data = json.loads(chunk)
                choice = data.get("choices", [{}])[0]
                text = choice.get("text") or choice.get("delta", {}).get("content")
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    output_count += 1
            except json.JSONDecodeError:
                continue
    total = time.perf_counter() - start
    return request_id, is_warm, (ttft or 0) * 1000, total * 1000, output_count


async def capture_stats(session: aiohttp.ClientSession, target: str) -> dict:
    """Snapshot /v1/stats."""
    try:
        async with session.get(f"{target}/v1/stats", timeout=10) as resp:
            return await resp.json()
    except Exception as exc:
        return {"error": str(exc)}


async def run_bench(args: argparse.Namespace) -> int:
    target = args.target.rstrip("/")
    url = f"{target}/v1/completions"

    # Load workload
    rows = []
    with open(args.workload) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)

    n = len(rows)
    if args.warm_count < 0 or args.warm_count > n:
        print(f"error: --warm-count {args.warm_count} out of range [0, {n}]",
              file=sys.stderr)
        return 2

    print(f"Target: {url}")
    print(f"Model: {args.model}")
    print(f"Workload: {args.workload} ({n} rows; first {args.warm_count} warm, "
          f"rest {n - args.warm_count} cold)")
    print(f"Concurrency: {args.concurrency}")
    print()

    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Warmup probe
        print("--- Warmup probe ---")
        warmup_prompt = rows[0].get("prompt", "Hello")
        warmup = await fire_completion(
            session, url, args.model, warmup_prompt, 16, -1, True)
        print(f"  warmup TTFT={warmup[2]:.0f}ms total={warmup[3]:.0f}ms "
              f"out={warmup[4]} tokens")
        await asyncio.sleep(1)

        # Snapshot stats before
        stats_before = await capture_stats(session, target)
        print(f"--- /v1/stats BEFORE ---")
        print(json.dumps(stats_before.get("agent_cache", stats_before),
                         indent=2)[:600])
        print()

        # Fire all rows at fixed concurrency
        print(f"--- Firing {n} requests at concurrency {args.concurrency} ---")
        sem = asyncio.Semaphore(args.concurrency)

        async def fire_with_sem(idx: int, row: dict) -> Tuple:
            async with sem:
                return await fire_completion(
                    session, url, args.model,
                    row.get("prompt", ""),
                    row.get("output_tokens", 128),
                    idx,
                    is_warm=(idx < args.warm_count),
                )

        bench_start = time.perf_counter()
        tasks = [fire_with_sem(i, row) for i, row in enumerate(rows)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        bench_wall = (time.perf_counter() - bench_start) * 1000

        # Snapshot stats after
        stats_after = await capture_stats(session, target)
        print(f"--- /v1/stats AFTER ---")
        print(json.dumps(stats_after.get("agent_cache", stats_after),
                         indent=2)[:1200])
        print()

    # Process results
    warm_ttfts = []
    cold_ttfts = []
    warm_totals = []
    cold_totals = []
    output_tokens_total = 0
    completed = 0
    failed = 0
    for r in results:
        if isinstance(r, Exception):
            failed += 1
            continue
        rid, is_warm, ttft_ms, total_ms, out_tokens = r
        completed += 1
        output_tokens_total += out_tokens
        if is_warm:
            warm_ttfts.append(ttft_ms)
            warm_totals.append(total_ms)
        else:
            cold_ttfts.append(ttft_ms)
            cold_totals.append(total_ms)

    def pct(vals: List[float], q: float) -> float:
        if not vals:
            return 0.0
        return statistics.quantiles(sorted(vals), n=100)[max(0, int(q) - 1)] \
            if len(vals) >= 100 else sorted(vals)[max(0, int(len(vals) * q / 100) - 1)]

    print("--- Bench results ---")
    print(f"  Total requests:       {n} ({completed} completed, {failed} failed)")
    print(f"  Wall time:            {bench_wall:.0f} ms ({bench_wall/1000:.1f} s)")
    print(f"  Aggregate output toks: {output_tokens_total}")
    print(f"  Throughput:           {output_tokens_total / (bench_wall / 1000):.1f} tok/s")
    print()
    print(f"  Warm requests ({len(warm_ttfts)}):")
    if warm_ttfts:
        print(f"    TTFT p50:           {pct(warm_ttfts, 50):.0f} ms")
        print(f"    TTFT p95:           {pct(warm_ttfts, 95):.0f} ms")
        print(f"    TTFT p99:           {pct(warm_ttfts, 99):.0f} ms")
        print(f"    Total p50:          {pct(warm_totals, 50):.0f} ms")
    print(f"  Cold requests ({len(cold_ttfts)}):")
    if cold_ttfts:
        print(f"    TTFT p50:           {pct(cold_ttfts, 50):.0f} ms")
        print(f"    TTFT p95:           {pct(cold_ttfts, 95):.0f} ms")
        print(f"    TTFT p99:           {pct(cold_ttfts, 99):.0f} ms")
        print(f"    Total p50:          {pct(cold_totals, 50):.0f} ms")
    print()

    # Counter delta
    def get_counter(stats: dict, key: str) -> int:
        ac = stats.get("agent_cache", {})
        return int(ac.get(key, 0))

    deferrals_delta = (get_counter(stats_after, "prefix_aware_admit_deferrals") -
                       get_counter(stats_before, "prefix_aware_admit_deferrals"))
    print(f"  prefix_aware_admit_deferrals delta: {deferrals_delta}")

    # License preview (per 5453ee4 spec §"Gate-license matrix")
    print()
    print("--- License preview (vs canonical thresholds, n=1 not σ-tight) ---")
    if warm_ttfts and cold_ttfts:
        warm_p50 = pct(warm_ttfts, 50)
        cold_p95 = pct(cold_ttfts, 95)
        warm_p95 = pct(warm_ttfts, 95)
        starv_ratio = cold_p95 / warm_p95 if warm_p95 > 0 else 0
        print(f"  Warm p50 TTFT:       {warm_p50:.0f} ms")
        print(f"  Cold p95 / Warm p95: {starv_ratio:.2f}x  (≤3x = no starvation)")
        if starv_ratio > 3:
            print(f"  ⚠️  STARVATION FLAG: cold p95 > 3× warm p95")

    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="ARLE server URL e.g. http://localhost:8765")
    parser.add_argument("model", help="Model identifier")
    parser.add_argument("workload", help="JSONL workload path "
                        "(e.g. bench-output/36-warm-mix.jsonl)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warm-count", type=int, default=153,
                        help="First N rows treated as warm (matches "
                             "gen_36_warm_prefix_mix.py default 4 sessions × 38 ≈ 153)")
    args = parser.parse_args()
    return asyncio.run(run_bench(args))


if __name__ == "__main__":
    sys.exit(main())

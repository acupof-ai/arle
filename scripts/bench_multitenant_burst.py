#!/usr/bin/env python3
"""Multi-tenant shared-prefix burst bench (3-engine: ARLE/vLLM/SGLang).

Originally M_ibp Phase 0 license-or-kill runner (2026-05-07).
Promoted from /tmp/ to scripts/ on 2026-05-08 to support
M_world1 P0.2 multi-tenant baseline (task #23) — close ARLE
4-shape verdict full table (only SGLang multi-tenant missing).

Workload (matches existing ARLE 318ms / vLLM 573ms baselines):
- 4 concurrent requests
- Shared system prompt ~6k tokens (structured English * 70 reps)
- Different short user queries ~100 tokens (unique per request)
- max_tokens=64, temperature=0.0, stream=true
- Pre-warmed by 1 request to populate cache
- Measures TTFT min/p50/max + burst total wall

Usage:
  ARLE:   python3 scripts/bench_multitenant_burst.py http://localhost:8000 default
  vLLM:   python3 scripts/bench_multitenant_burst.py http://localhost:8000 \\
            /home/ckl/projects/arle/infer/models/Qwen3-4B
  SGLang: python3 scripts/bench_multitenant_burst.py http://localhost:8001 \\
            /home/ckl/projects/arle/infer/models/Qwen3-4B

Reference data (M_ibp Phase 0 wins entry):
- ARLE post-F4-Small/M_b.1/M_pf-P0: TTFT mdn 318 ms
- vLLM 0.20.1 s8: TTFT mdn 573 ms
- SGLang: pending - task #23
"""

import asyncio
import json
import time
from typing import Tuple

import aiohttp

# A 6000-token shared system prompt. Construction: a large block
# of repeated structured English text. Tokenizes to roughly 6k
# Qwen3 tokens.
SYSTEM_PROMPT_BASE = (
    "You are an extensively detailed technical assistant with deep expertise "
    "in software engineering, distributed systems, machine learning, and "
    "GPU programming. When you respond, you provide thorough, well-structured "
    "answers that include relevant context, edge cases, alternative approaches, "
    "and concrete code examples where applicable. You prefer precision over "
    "brevity. Your tone is professional but friendly. You do not hallucinate "
    "information; if uncertain, you state your confidence level explicitly. "
    "You reference industry-standard sources where relevant. "
)
# Repeat to inflate token count
SYSTEM_PROMPT = SYSTEM_PROMPT_BASE * 70  # ~5k-6k tokens

USER_QUERIES = [
    "Explain the role of CUDA streams in async kernel scheduling.",
    "Describe how speculative decoding interacts with KV cache management.",
    "Walk me through the trade-offs of FP8 vs INT4 quantization for KV.",
    "What are the key considerations when implementing chunked prefill?",
]


async def fire_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    user_query: str,
    request_id: int,
) -> Tuple[int, float, float, str]:
    """Fire one request, return (request_id, TTFT_ms, total_latency_ms, first_text)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": True,
    }
    start = time.perf_counter()
    ttft = None
    first_text = ""
    async with session.post(url, json=payload) as resp:
        async for line in resp.content:
            if not line.startswith(b"data: "):
                continue
            chunk = line[6:].strip()
            if chunk == b"[DONE]":
                break
            try:
                data = json.loads(chunk)
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                        first_text = delta
                    # accumulate but don't track further
            except json.JSONDecodeError:
                continue
    total = time.perf_counter() - start
    return request_id, (ttft or 0) * 1000, total * 1000, first_text


async def main():
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-4B"
    url = f"{target}/v1/chat/completions"

    print(f"Target: {url}")
    print(f"Model: {model}")
    print(f"System prompt length (chars): {len(SYSTEM_PROMPT)}")

    # Warmup with 1 request to populate cache (separately for fairness)
    print("--- WARMUP single request (populate any caches) ---")
    async with aiohttp.ClientSession() as session:
        warmup = await fire_request(session, url, model, "Hello", 99)
        print(f"  warmup TTFT={warmup[1]:.0f}ms total={warmup[2]:.0f}ms")

    # Brief pause to let cache settle
    await asyncio.sleep(2)

    # Now fire 4 concurrent requests with shared system prompt
    print("--- 4 CONCURRENT requests with shared system prompt ---")
    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        tasks = [
            fire_request(session, url, model, q, i)
            for i, q in enumerate(USER_QUERIES)
        ]
        results = await asyncio.gather(*tasks)
        burst_total = (time.perf_counter() - t0) * 1000

    print()
    print("Results:")
    print(f"{'req_id':<8} {'TTFT (ms)':<12} {'total (ms)':<12} {'first_text':<40}")
    ttfts = []
    for r in sorted(results):
        rid, ttft, total, first = r
        ttfts.append(ttft)
        print(f"{rid:<8} {ttft:<12.0f} {total:<12.0f} {first[:40]!r}")

    if ttfts:
        ttft_min = min(ttfts)
        ttft_max = max(ttfts)
        ttft_mdn = sorted(ttfts)[len(ttfts) // 2]
        print()
        print(f"Burst total wall: {burst_total:.0f} ms")
        print(f"TTFT p50/min/max: {ttft_mdn:.0f} / {ttft_min:.0f} / {ttft_max:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())

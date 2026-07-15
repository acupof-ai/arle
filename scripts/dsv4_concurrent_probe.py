#!/usr/bin/env python3
"""DSv4 8-slot concurrent throughput + correctness probe (pod-local, aiohttp).

This focused probe predates the canonical native runner. It does two things the
serial
needle gate cannot:

  1. THROUGHPUT under real N-way concurrency (aggregate tok/s, per-request p50/p99).
  2. CORRECTNESS stress of concurrent slots: N needle requests in flight at once,
     verifying each retrieves its code — exercises the per-row decode loop's
     serial shared-buffer reuse (#60) under genuine concurrency, which a serial
     gate leaves untested.

Usage: dsv4_concurrent_probe.py [PORT] [CONCURRENCY] [REQS_PER_CLIENT] [OUT_TOKENS]
"""
import asyncio
import sys
import time

import aiohttp

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18188
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
REQS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
OUT_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "DeepSeek-V4-Flash"

# ~512-token filler prompt so the throughput run reflects a realistic decode load.
FILLER = ("The quick brown fox jumps over the lazy dog. " * 64).strip()
THROUGHPUT_PROMPT = FILLER + "\n\nContinue the story:"


async def one_completion(session, prompt, max_tokens):
    t0 = time.perf_counter()
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    async with session.post(f"{BASE}/v1/completions", json=payload) as resp:
        data = await resp.json()
    dt = time.perf_counter() - t0
    text = data["choices"][0]["text"]
    out_tok = data.get("usage", {}).get("completion_tokens", 0)
    return dt, out_tok, text


async def throughput_run():
    print(f"=== THROUGHPUT: conc={CONC} reqs/client={REQS} out_tokens={OUT_TOKENS} ===")
    conn = aiohttp.TCPConnector(limit=CONC + 4)
    timeout = aiohttp.ClientTimeout(total=600)
    lat = []
    total_out = 0
    wall0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        async def client():
            nonlocal total_out
            for _ in range(REQS):
                dt, out_tok, _ = await one_completion(session, THROUGHPUT_PROMPT, OUT_TOKENS)
                lat.append(dt)
                total_out += out_tok
        await asyncio.gather(*[client() for _ in range(CONC)])
    wall = time.perf_counter() - wall0
    lat.sort()
    n = len(lat)
    p50 = lat[n // 2]
    p99 = lat[min(n - 1, int(n * 0.99))]
    print(f"requests      : {n}")
    print(f"wall_seconds  : {wall:.2f}")
    print(f"total_out_tok : {total_out}")
    print(f"aggregate_tok/s: {total_out / wall:.1f}")
    print(f"req_latency_s : mean={sum(lat)/n:.2f} p50={p50:.2f} p99={p99:.2f}")
    print(f"per_req_tok/s : {OUT_TOKENS / (sum(lat)/n):.1f} (single-stream equiv)")


def build_needle(target_len_tokens, code="738291"):
    # Mirror needle_gate's shape: filler notes + a buried secret code.
    topics = ["the river", "a old key", "the market", "her letter", "the storm"]
    n_notes = max(1, target_len_tokens // 8)
    notes = [f"Note {i+1}: {topics[i % len(topics)]}." for i in range(n_notes)]
    mid = len(notes) // 2
    notes.insert(mid, f"IMPORTANT the secret access code is: {code}.")
    body = " ".join(notes)
    prompt = body + "\n\nQuestion: what is the secret access code? Answer with the code"
    return prompt, code


async def concurrent_correctness():
    print(f"\n=== CONCURRENT CORRECTNESS: {CONC} needles in flight (shared-buffer stress) ===")
    prompt, code = build_needle(2000)
    conn = aiohttp.TCPConnector(limit=CONC + 4)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        results = await asyncio.gather(
            *[one_completion(session, prompt, 24) for _ in range(CONC)]
        )
    exact = 0
    for i, (dt, _out, text) in enumerate(results):
        hit = code in text
        exact += hit
        cls = "exact" if hit else "MISS"
        print(f"  req{i}: {cls} wall={dt:.2f}s out={text[:60]!r}")
    print(f"CONCURRENT SUMMARY: {exact}/{CONC} retrieved the code")
    return exact, CONC


async def main():
    await throughput_run()
    exact, total = await concurrent_correctness()
    print(f"\nVERDICT throughput=ok concurrent_correctness={exact}/{total}")


if __name__ == "__main__":
    asyncio.run(main())

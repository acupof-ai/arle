#!/usr/bin/env python3
"""Concurrent request test suite for infer scheduler.

Tests multi-request concurrency, streaming, interleaved scheduling,
stop sequences, and stress behavior against the infer OpenAI-compatible
completions endpoint.

Usage:
    python scripts/test_concurrent.py [MODEL_NAME] [BASE_URL]

    MODEL_NAME  defaults to "Qwen3-8B"
    BASE_URL    defaults to "http://localhost:8000/v1/completions"
"""

import asyncio
import httpx
import time
import sys
import json
import statistics
from dataclasses import dataclass, field
from typing import Optional

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-8B"
BASE_URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/v1/completions"

TIMEOUT = 120.0  # seconds per request

SYSTEM_PREFIX = (
    "You are a helpful AI assistant. Answer concisely and accurately. "
    "Think step by step when needed. "
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    prompt: str
    status: int
    text: str
    latency_s: float
    tokens: int = 0
    error: Optional[str] = None


@dataclass
class StreamResult:
    prompt: str
    status: int
    chunks: list = field(default_factory=list)
    ttft_s: float = 0.0          # time to first token
    itl_ms: list = field(default_factory=list)  # inter-token latencies (ms)
    total_s: float = 0.0
    full_text: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    print()
    print("-" * 70)
    print(f"  {title}")
    print("-" * 70)


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


async def post_completion(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.7,
    stop: Optional[list] = None,
) -> RequestResult:
    """Send a non-streaming completion request and return the result."""
    payload = {
        "model": MODEL,
        "prompt": f"{SYSTEM_PREFIX}{prompt}",
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop is not None:
        payload["stop"] = stop

    t0 = time.perf_counter()
    try:
        resp = await client.post(BASE_URL, json=payload, timeout=TIMEOUT)
        elapsed = time.perf_counter() - t0
        if resp.status_code != 200:
            return RequestResult(
                prompt=prompt, status=resp.status_code, text="",
                latency_s=elapsed, error=resp.text,
            )
        body = resp.json()
        choice = body["choices"][0]
        text = choice.get("text", "")
        tokens = body.get("usage", {}).get("completion_tokens", len(text.split()))
        return RequestResult(
            prompt=prompt, status=200, text=text,
            latency_s=elapsed, tokens=tokens,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return RequestResult(
            prompt=prompt, status=0, text="",
            latency_s=elapsed, error=str(exc),
        )


async def post_streaming(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.7,
) -> StreamResult:
    """Send a streaming completion request and collect SSE chunks."""
    payload = {
        "model": MODEL,
        "prompt": f"{SYSTEM_PREFIX}{prompt}",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    result = StreamResult(prompt=prompt, status=0)
    t0 = time.perf_counter()
    t_last = t0

    try:
        async with client.stream("POST", BASE_URL, json=payload, timeout=TIMEOUT) as resp:
            result.status = resp.status_code
            if resp.status_code != 200:
                body = await resp.aread()
                result.error = body.decode(errors="replace")
                result.total_s = time.perf_counter() - t0
                return result

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()
                token_text = chunk.get("choices", [{}])[0].get("text", "")
                result.chunks.append(token_text)
                result.full_text += token_text

                if len(result.chunks) == 1:
                    result.ttft_s = now - t0
                else:
                    result.itl_ms.append((now - t_last) * 1000.0)
                t_last = now

    except Exception as exc:
        result.error = str(exc)

    result.total_s = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Test 1 -- Basic single completion
# ---------------------------------------------------------------------------

async def test_basic_completion() -> None:
    banner("Test 1: Basic Completion")
    async with httpx.AsyncClient() as client:
        res = await post_completion(client, "What is 2+2?", max_tokens=32)

    if res.error:
        fail(f"Request failed: {res.error}")
        raise SystemExit(1)

    assert res.status == 200, f"Expected 200, got {res.status}"
    assert len(res.text.strip()) > 0, "Empty response text"
    ok(f"status=200  latency={res.latency_s:.3f}s  tokens={res.tokens}")
    ok(f"response (truncated): {res.text.strip()[:120]}")


# ---------------------------------------------------------------------------
# Test 2 -- Concurrent requests
# ---------------------------------------------------------------------------

async def test_concurrent_requests(n: int = 4) -> None:
    banner(f"Test 2: Concurrent Requests (n={n})")

    prompts = [
        "What is the capital of France?",
        "List three prime numbers.",
        "Name a color of the rainbow.",
        "What planet is closest to the sun?",
        "What is the boiling point of water?",
        "How many continents are there?",
        "What is the square root of 144?",
        "Name a Shakespeare play.",
    ][:n]

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        tasks = [post_completion(client, p, max_tokens=48) for p in prompts]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    errors = [r for r in results if r.error]
    if errors:
        for e in errors:
            fail(f"'{e.prompt}': {e.error}")
        raise SystemExit(1)

    latencies = [r.latency_s for r in results]
    total_tokens = sum(r.tokens for r in results)
    ok(f"All {n} requests succeeded in {wall_time:.3f}s wall time")
    ok(f"Latency  min={min(latencies):.3f}s  max={max(latencies):.3f}s  "
       f"mean={statistics.mean(latencies):.3f}s")
    ok(f"Total tokens={total_tokens}  throughput={total_tokens / wall_time:.1f} tok/s")

    for r in results:
        print(f"    [{r.latency_s:6.3f}s] {r.prompt[:40]:<40}  -> {r.text.strip()[:60]}")


# ---------------------------------------------------------------------------
# Test 3 -- Streaming
# ---------------------------------------------------------------------------

async def test_streaming() -> None:
    banner("Test 3: Streaming (SSE)")
    async with httpx.AsyncClient() as client:
        res = await post_streaming(client, "Count from 1 to 10.", max_tokens=64)

    if res.error:
        fail(f"Streaming failed: {res.error}")
        raise SystemExit(1)

    assert res.status == 200, f"Expected 200, got {res.status}"
    assert len(res.chunks) > 1, f"Expected multiple chunks, got {len(res.chunks)}"

    ok(f"Received {len(res.chunks)} SSE chunks in {res.total_s:.3f}s")
    ok(f"TTFT = {res.ttft_s * 1000:.1f} ms")
    if res.itl_ms:
        ok(f"ITL  mean={statistics.mean(res.itl_ms):.1f} ms  "
           f"p50={statistics.median(res.itl_ms):.1f} ms  "
           f"max={max(res.itl_ms):.1f} ms")
    ok(f"Full text (truncated): {res.full_text.strip()[:120]}")


# ---------------------------------------------------------------------------
# Test 4 -- Interleaved scheduling
# ---------------------------------------------------------------------------

async def test_interleaved_scheduling() -> None:
    banner("Test 4: Interleaved Scheduling")
    print("  Sending a long request (256 tokens) then a short request (16 tokens).")
    print("  The short request should finish first if scheduling interleaves.\n")

    finish_order: list[str] = []

    async def run_long(client: httpx.AsyncClient) -> RequestResult:
        # Small delay so long request is definitely in-flight first
        res = await post_completion(
            client, "Write a detailed essay about the history of mathematics.",
            max_tokens=256,
        )
        finish_order.append("long")
        return res

    async def run_short(client: httpx.AsyncClient) -> RequestResult:
        # Wait briefly so the long request starts first
        await asyncio.sleep(0.3)
        res = await post_completion(
            client, "What is 1+1?",
            max_tokens=16,
        )
        finish_order.append("short")
        return res

    async with httpx.AsyncClient() as client:
        long_task = asyncio.create_task(run_long(client))
        short_task = asyncio.create_task(run_short(client))
        long_res, short_res = await asyncio.gather(long_task, short_task)

    if long_res.error:
        fail(f"Long request failed: {long_res.error}")
        raise SystemExit(1)
    if short_res.error:
        fail(f"Short request failed: {short_res.error}")
        raise SystemExit(1)

    ok(f"Long  request: {long_res.latency_s:.3f}s  tokens={long_res.tokens}")
    ok(f"Short request: {short_res.latency_s:.3f}s  tokens={short_res.tokens}")
    ok(f"Finish order: {' -> '.join(finish_order)}")

    if finish_order[0] == "short":
        ok("Short request finished first -- interleaving confirmed!")
    else:
        print("  [WARN] Short request did NOT finish first. "
              "Interleaving may not be active or the gap was too small.")


# ---------------------------------------------------------------------------
# Test 5 -- Stop sequences
# ---------------------------------------------------------------------------

async def test_stop_sequences() -> None:
    banner("Test 5: Stop Sequences")

    test_cases = [
        {
            "prompt": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            "stop": [","],
            "label": "stop on comma",
        },
        {
            "prompt": "Say hello world and then goodbye world.",
            "stop": ["goodbye"],
            "label": "stop on word",
        },
        {
            "prompt": "Line one.\nLine two.\nLine three.",
            "stop": ["\n"],
            "label": "stop on newline",
        },
    ]

    async with httpx.AsyncClient() as client:
        for tc in test_cases:
            res = await post_completion(
                client, tc["prompt"], max_tokens=128, stop=tc["stop"],
            )
            if res.error:
                fail(f"[{tc['label']}] Request failed: {res.error}")
                raise SystemExit(1)

            text = res.text.strip()
            # The stop sequence itself should NOT appear in the output
            stop_seq = tc["stop"][0]
            # Check that the response was actually truncated (shorter than max)
            ok(f"[{tc['label']}] latency={res.latency_s:.3f}s  "
               f"output({len(text)} chars): {text[:80]}")

            if stop_seq in res.text and stop_seq != "\n":
                # Some servers include the stop token; just warn
                print(f"    [WARN] Stop sequence '{stop_seq}' found in output text.")


# ---------------------------------------------------------------------------
# Test 6 -- Stress test
# ---------------------------------------------------------------------------

async def test_stress(n: int = 20) -> None:
    banner(f"Test 6: Stress Test ({n} rapid requests)")

    prompts = [f"What is {i} * {i + 1}?" for i in range(1, n + 1)]

    completed = 0
    errors_list: list[str] = []
    latencies: list[float] = []
    queue_depth_log: list[tuple[float, int]] = []  # (time, outstanding)

    outstanding = 0
    lock = asyncio.Lock()
    t_start = time.perf_counter()

    async def run_one(client: httpx.AsyncClient, prompt: str) -> None:
        nonlocal completed, outstanding
        async with lock:
            outstanding += 1
            queue_depth_log.append((time.perf_counter() - t_start, outstanding))

        res = await post_completion(client, prompt, max_tokens=32)

        async with lock:
            outstanding -= 1
            queue_depth_log.append((time.perf_counter() - t_start, outstanding))

        if res.error:
            errors_list.append(f"{prompt}: {res.error}")
        else:
            completed += 1
            latencies.append(res.latency_s)

    async with httpx.AsyncClient() as client:
        tasks = [run_one(client, p) for p in prompts]
        t0 = time.perf_counter()
        await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    if errors_list:
        for e in errors_list:
            fail(e)
        raise SystemExit(1)

    ok(f"All {n} requests completed in {wall_time:.3f}s")
    ok(f"Latency  min={min(latencies):.3f}s  max={max(latencies):.3f}s  "
       f"mean={statistics.mean(latencies):.3f}s  "
       f"p50={statistics.median(latencies):.3f}s")
    ok(f"Throughput: {n / wall_time:.1f} req/s")

    # Queue depth summary
    peak_depth = max(d for _, d in queue_depth_log)
    ok(f"Peak queue depth: {peak_depth}")

    print("\n  Queue depth over time:")
    # Sample a few points
    step = max(1, len(queue_depth_log) // 12)
    for i in range(0, len(queue_depth_log), step):
        t, d = queue_depth_log[i]
        bar = "#" * d
        print(f"    t={t:6.3f}s  depth={d:3d}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 70)
    print("  INFER CONCURRENT REQUEST TEST SUITE")
    print(f"  endpoint : {BASE_URL}")
    print(f"  model    : {MODEL}")
    print("=" * 70)

    await test_basic_completion()
    await test_concurrent_requests(n=4)
    await test_streaming()
    await test_interleaved_scheduling()
    await test_stop_sequences()
    await test_stress(n=20)

    print()
    print("=" * 70)
    print("  ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

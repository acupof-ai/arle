#!/usr/bin/env python3
"""
infer throughput & latency benchmark.

Measures:
  - TTFT  (time-to-first-token)
  - ITL   (inter-token latency / TBT / time-between-tokens)
  - E2E   (end-to-end request latency)
  - Throughput (output tokens / second, aggregate over all requests)

Dataset modes:
  --dataset synthetic   built-in synthetic prompts (default, no GPU / no data needed)
  --dataset sharegpt    load ShareGPT JSON from --dataset-path

Usage (GPU server):
  python3 scripts/bench_throughput.py --url http://localhost:8000 \\
      --dataset sharegpt --dataset-path data/ShareGPT_V3_unfiltered_cleaned_split.json \\
      --num-prompts 500 --concurrency 32 --max-tokens 256

Usage (local mock server, no GPU):
  python3 scripts/bench_throughput.py --mock --num-prompts 20 --concurrency 4
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import httpx
except ImportError:
    sys.exit("Install httpx:  pip install httpx")

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float           # seconds to first token
    itl_s: list             # per-inter-token latency in seconds
    e2e_s: float            # total request time in seconds
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def mean_itl_s(self) -> float:
        return statistics.mean(self.itl_s) if self.itl_s else 0.0

    @property
    def output_tokens_per_s(self) -> float:
        if self.e2e_s <= 0:
            return 0.0
        return self.completion_tokens / self.e2e_s


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

SYNTHETIC_PROMPTS = [
    "Explain how a transformer model works.",
    "Write a Python quicksort implementation.",
    "What is the difference between TCP and UDP?",
    "Describe the CAP theorem in distributed systems.",
    "How does gradient descent work in machine learning?",
    "Write a Rust function that reverses a linked list.",
    "What are the SOLID principles of OOP?",
    "Explain the difference between mutex and semaphore.",
    "How does HTTPS protect data in transit?",
    "Describe the attention mechanism in neural networks.",
    "Write a SQL query to find the top 5 customers by revenue.",
    "Explain bloom filters and their use cases.",
    "What is the difference between BFS and DFS?",
    "How does consistent hashing work?",
    "Describe memory management in Rust.",
    "Explain the difference between process and thread.",
    "What is MapReduce and when would you use it?",
    "How does LRU cache eviction work?",
    "Describe the two-phase commit protocol.",
    "Explain virtual memory and page tables.",
]


def load_synthetic(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    prompts = []
    pool = SYNTHETIC_PROMPTS * ((n // len(SYNTHETIC_PROMPTS)) + 1)
    rng.shuffle(pool)
    return pool[:n]


def load_sharegpt(path: str, n: int, seed: int = 42) -> list[str]:
    """Load up to `n` user-first-turn prompts from a ShareGPT JSON file."""
    with open(path) as f:
        data = json.load(f)

    prompts = []
    for conv in data:
        turns = conv.get("conversations") or conv.get("messages") or []
        for turn in turns:
            role = turn.get("from") or turn.get("role") or ""
            if role in ("human", "user"):
                content = turn.get("value") or turn.get("content") or ""
                if content.strip():
                    prompts.append(content.strip())
                break  # only first user turn per conversation

    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n]


# ---------------------------------------------------------------------------
# HTTP client helpers
# ---------------------------------------------------------------------------

async def send_streaming(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    t_start = time.perf_counter()
    t_first = None
    t_prev = t_start
    itl = []
    completion_tokens = 0

    try:
        async with client.stream(
            "POST",
            f"{url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return RequestResult(
                    prompt_tokens=0, completion_tokens=0,
                    ttft_s=0, itl_s=[], e2e_s=0,
                    error=f"HTTP {resp.status_code}: {body[:200]}",
                )

            prompt_tokens = 0
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Usage chunk (last chunk with include_usage).
                if "usage" in chunk and chunk.get("choices") is None:
                    usage = chunk["usage"]
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                text = choices[0].get("text") or (
                    (choices[0].get("delta") or {}).get("content") or ""
                )
                if text:
                    t_now = time.perf_counter()
                    if t_first is None:
                        t_first = t_now
                    else:
                        itl.append(t_now - t_prev)
                    t_prev = t_now
                    completion_tokens += 1  # rough estimate if usage chunk absent

                # Grab usage from within-choice if present.
                usage_inline = chunk.get("usage")
                if usage_inline:
                    prompt_tokens = usage_inline.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage_inline.get("completion_tokens", completion_tokens)

    except Exception as exc:
        return RequestResult(
            prompt_tokens=0, completion_tokens=0,
            ttft_s=0, itl_s=[], e2e_s=0,
            error=str(exc),
        )

    t_end = time.perf_counter()
    ttft = (t_first - t_start) if t_first else (t_end - t_start)
    return RequestResult(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_s=ttft,
        itl_s=itl,
        e2e_s=t_end - t_start,
    )


# ---------------------------------------------------------------------------
# Mock server
# ---------------------------------------------------------------------------

async def _mock_server(host: str, port: int, tokens_per_request: int, delay_ms: float):
    """Tiny asyncio HTTP server that returns fake SSE token streams."""
    try:
        from aiohttp import web  # optional dep
    except ImportError:
        # Minimal fallback using raw asyncio streams.
        await _mock_server_raw(host, port, tokens_per_request, delay_ms)
        return

    async def handle(request):
        body = await request.json()
        max_tok = min(body.get("max_tokens", tokens_per_request), tokens_per_request)
        is_stream = body.get("stream", False)

        if not is_stream:
            await asyncio.sleep(delay_ms * max_tok / 1000)
            resp = {
                "id": "cmpl-mock",
                "object": "text_completion",
                "created": int(time.time()),
                "model": "mock",
                "choices": [{"text": "mock " * max_tok, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": max_tok, "total_tokens": 10 + max_tok},
            }
            return web.Response(content_type="application/json", text=json.dumps(resp))

        async def gen():
            for i in range(max_tok):
                await asyncio.sleep(delay_ms / 1000)
                chunk = {"id": "cmpl-mock", "object": "text_completion", "created": int(time.time()),
                         "model": "mock", "choices": [{"text": "tok ", "index": 0, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            final = {"id": "cmpl-mock", "object": "text_completion", "created": int(time.time()),
                     "model": "mock", "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": max_tok, "total_tokens": 10 + max_tok}}
            yield f"data: {json.dumps(final)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return web.Response(
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
            body=b"".join([chunk async for chunk in gen()]),
        )

    app = web.Application()
    app.router.add_post("/v1/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    # Server runs until the event loop is stopped.
    await asyncio.Event().wait()


async def _mock_server_raw(host: str, port: int, tokens_per_request: int, delay_ms: float):
    """Pure-asyncio fallback mock server (no aiohttp)."""

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request_line = await reader.readline()
            headers = {}
            while True:
                line = (await reader.readline()).decode()
                if line in ("\r\n", "\n", ""):
                    break
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            content_length = int(headers.get("content-length", 0))
            body_bytes = await reader.read(content_length) if content_length else b"{}"
            try:
                body = json.loads(body_bytes)
            except Exception:
                body = {}

            max_tok = min(body.get("max_tokens", tokens_per_request), tokens_per_request)
            is_stream = body.get("stream", False)

            if is_stream:
                chunks = []
                for i in range(max_tok):
                    await asyncio.sleep(delay_ms / 1000)
                    chunk = {"id": "cmpl-mock", "object": "text_completion",
                             "choices": [{"text": "tok ", "finish_reason": None}]}
                    chunks.append(f"data: {json.dumps(chunk)}\r\n\r\n")
                final = {"id": "cmpl-mock", "object": "text_completion",
                         "choices": [{"text": "", "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": max_tok, "total_tokens": 10 + max_tok}}
                chunks.append(f"data: {json.dumps(final)}\r\n\r\n")
                chunks.append("data: [DONE]\r\n\r\n")
                body_str = "".join(chunks)
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    f"Content-Length: {len(body_str.encode())}\r\n"
                    "Connection: close\r\n\r\n"
                ) + body_str
            else:
                await asyncio.sleep(delay_ms * max_tok / 1000)
                resp_obj = {
                    "id": "cmpl-mock", "object": "text_completion",
                    "choices": [{"text": "mock " * max_tok, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": max_tok, "total_tokens": 10 + max_tok},
                }
                body_str = json.dumps(resp_obj)
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body_str.encode())}\r\n"
                    "Connection: close\r\n\r\n"
                ) + body_str

            writer.write(response.encode())
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle_client, host, port)
    async with server:
        await server.serve_forever()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    temperature: float,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(prompts)

    async with httpx.AsyncClient(timeout=None) as client:

        async def worker(idx: int, prompt: str):
            async with semaphore:
                results[idx] = await send_streaming(
                    client, url, model, prompt, max_tokens, temperature
                )

        tasks = [asyncio.create_task(worker(i, p)) for i, p in enumerate(prompts)]
        completed = 0
        last_print = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            completed += 1
            if completed - last_print >= max(1, len(prompts) // 20):
                print(f"  {completed}/{len(prompts)} requests done ...", flush=True)
                last_print = completed

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def print_latency_table(name: str, values_s: list[float]):
    if not values_s:
        print(f"  {name}: no data")
        return
    vals_ms = [v * 1000 for v in values_s]
    print(
        f"  {name:<8} "
        f"mean={statistics.mean(vals_ms):7.1f}ms  "
        f"p50={percentile(vals_ms, 50):7.1f}ms  "
        f"p90={percentile(vals_ms, 90):7.1f}ms  "
        f"p99={percentile(vals_ms, 99):7.1f}ms"
    )


def print_report(results: list[RequestResult], wall_time_s: float):
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print(f"\n{'='*68}")
    print("RESULTS")
    print(f"{'='*68}")
    print(f"  Total requests:     {len(results)}")
    print(f"  Succeeded:          {len(ok)}")
    print(f"  Failed:             {len(failed)}")
    if failed:
        for r in failed[:3]:
            print(f"    Error: {r.error}")

    if not ok:
        return

    total_prompt  = sum(r.prompt_tokens for r in ok)
    total_output  = sum(r.completion_tokens for r in ok)
    throughput    = total_output / wall_time_s if wall_time_s > 0 else 0
    req_per_s     = len(ok) / wall_time_s if wall_time_s > 0 else 0

    print(f"\n  Wall time:          {wall_time_s:.2f}s")
    print(f"  Throughput:         {throughput:.1f} output tok/s")
    print(f"  Request rate:       {req_per_s:.2f} req/s")
    print(f"  Prompt tokens:      {total_prompt}")
    print(f"  Output tokens:      {total_output}")

    print("\nLatency:")
    print_latency_table("TTFT",  [r.ttft_s for r in ok])
    print_latency_table("ITL",   [itl for r in ok for itl in r.itl_s])
    print_latency_table("E2E",   [r.e2e_s for r in ok])
    print(f"{'='*68}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print("DEPRECATED: use scripts/bench_guidellm.sh instead", file=sys.stderr)
    p = argparse.ArgumentParser(description="infer throughput & latency benchmark")
    p.add_argument("--url", default="http://localhost:8000",
                   help="Server base URL (default: http://localhost:8000)")
    p.add_argument("--model", default="",
                   help="Model name sent in request (informational only)")
    p.add_argument("--dataset", choices=["synthetic", "sharegpt"], default="synthetic",
                   help="Prompt dataset (default: synthetic)")
    p.add_argument("--dataset-path", default="",
                   help="Path to ShareGPT JSON file (required with --dataset sharegpt)")
    p.add_argument("--num-prompts", type=int, default=50,
                   help="Number of prompts to benchmark (default: 50)")
    p.add_argument("--concurrency", type=int, default=16,
                   help="Max concurrent requests (default: 16)")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Max output tokens per request (default: 128)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0 = greedy)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for dataset shuffling (default: 42)")
    p.add_argument("--mock", action="store_true",
                   help="Start a local mock server for testing without GPU")
    p.add_argument("--mock-port", type=int, default=18000,
                   help="Port for mock server (default: 18000)")
    p.add_argument("--mock-delay-ms", type=float, default=5.0,
                   help="Per-token delay in mock server (default: 5ms)")
    p.add_argument("--mock-tokens", type=int, default=32,
                   help="Tokens per mock response (default: 32)")
    return p.parse_args()


async def async_main():
    args = parse_args()

    server_task = None

    # ------------------------------------------------------------------ mock
    if args.mock:
        host = "127.0.0.1"
        port = args.mock_port
        url = f"http://{host}:{port}"
        print(f"Starting mock server on {url} (delay={args.mock_delay_ms}ms/tok, "
              f"tokens={args.mock_tokens})...")
        server_task = asyncio.create_task(
            _mock_server_raw(host, port, args.mock_tokens, args.mock_delay_ms)
        )
        await asyncio.sleep(0.2)  # let server start

    else:
        url = args.url

    # --------------------------------------------------------------- dataset
    print(f"\nLoading dataset ({args.dataset}, n={args.num_prompts}) ...")
    if args.dataset == "sharegpt":
        if not args.dataset_path:
            sys.exit("--dataset-path is required for --dataset sharegpt")
        prompts = load_sharegpt(args.dataset_path, args.num_prompts, args.seed)
    else:
        prompts = load_synthetic(args.num_prompts, args.seed)

    print(f"Loaded {len(prompts)} prompts.")

    # -------------------------------------------------------------- warmup
    print("\n[Warmup] Sending single request ...")
    async with httpx.AsyncClient(timeout=60) as client:
        wr = await send_streaming(
            client, url, args.model, prompts[0], min(16, args.max_tokens), 0.0
        )
    if not wr.ok:
        sys.exit(f"Warmup failed: {wr.error}")
    print(f"  Warmup TTFT: {wr.ttft_s*1000:.1f}ms  E2E: {wr.e2e_s*1000:.1f}ms")

    # ----------------------------------------------------------- benchmark
    print(f"\n[Benchmark] concurrency={args.concurrency}  max_tokens={args.max_tokens}  "
          f"temperature={args.temperature}  n={len(prompts)}")

    t_wall_start = time.perf_counter()
    results = await run_benchmark(
        url=url,
        model=args.model,
        prompts=prompts,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    wall_time = time.perf_counter() - t_wall_start

    print_report(results, wall_time)

    if server_task:
        server_task.cancel()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

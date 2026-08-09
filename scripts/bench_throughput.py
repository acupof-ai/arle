#!/usr/bin/env python3
"""Native OpenAI-compatible streaming throughput benchmark."""

import argparse
import asyncio
import csv
import json
import os
import random
import re
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    sys.exit("Install httpx: pip install httpx")


DEFAULT_CONCURRENCY_GRID = (1, 4, 8, 16)
SYNTHETIC_PROMPTS = (
    "Explain how a transformer model works.",
    "Write a Python quicksort implementation.",
    "What is the difference between TCP and UDP?",
    "Describe the CAP theorem in distributed systems.",
    "How does gradient descent work in machine learning?",
    "Write a Rust function that reverses a linked list.",
    "Explain bloom filters and their use cases.",
    "How does consistent hashing work?",
)


@dataclass
class RequestResult:
    request_id: int
    prompt: str
    status: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_s: float | None
    itl_s: list[float]
    e2e_s: float
    finish_reason: str | None
    output: str
    output_events: int
    usage_received: bool
    saw_done: bool
    correctness_error: str | None = None
    error: str | None = None

    @property
    def gate_pass(self) -> bool:
        return self.status == "complete" and self.correctness_error is None


def extract_prompt(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if not isinstance(value, dict):
        return None
    for key in ("prompt", "text", "input"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    turns = value.get("messages") or value.get("conversations") or []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from")
        text = turn.get("content") or turn.get("value")
        if role in ("user", "human") and isinstance(text, str) and text.strip():
            return text.strip()
    return None


def load_jsonl_prompts(path: Path) -> list[str]:
    prompts = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            prompt = extract_prompt(value)
            if prompt is None:
                raise ValueError(f"{path}:{line_no}: no prompt or first user message")
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"{path}: no prompts")
    return prompts


def synthetic_prompts(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(SYNTHETIC_PROMPTS) for _ in range(count)]


def repetition_error(text: str) -> str | None:
    words = re.findall(r"\w+|[^\w\s]", text.casefold())
    if len(words) < 8:
        return None
    if len(set(words)) / len(words) <= 0.12:
        return "low token diversity"
    for period in range(1, min(16, len(words) // 4) + 1):
        matches = sum(word == words[index % period] for index, word in enumerate(words))
        if matches / len(words) >= 0.9:
            return f"repeating period {period}"
    return None


def correctness_error(text: str) -> str | None:
    if not text.strip():
        return "empty output"
    return repetition_error(text)


def error_result(request_id: int, prompt: str, started: float, error: str) -> RequestResult:
    return RequestResult(
        request_id=request_id,
        prompt=prompt,
        status="error",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        ttft_s=None,
        itl_s=[],
        e2e_s=time.perf_counter() - started,
        finish_reason=None,
        output="",
        output_events=0,
        usage_received=False,
        saw_done=False,
        error=error,
    )


async def send_streaming(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_id: int,
) -> RequestResult:
    started = time.perf_counter()
    first_event = None
    previous_event = None
    itl = []
    output_parts = []
    output_events = 0
    prompt_tokens = completion_tokens = total_tokens = 0
    usage_received = saw_done = False
    finish_reason = None

    try:
        async with client.stream(
            "POST",
            f"{url.rstrip('/')}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                return error_result(
                    request_id, prompt, started, f"HTTP {response.status_code}: {body[:500]}"
                )
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].lstrip()
                if payload == "[DONE]":
                    saw_done = True
                    break
                if not payload:
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as exc:
                    return error_result(request_id, prompt, started, f"invalid SSE JSON: {exc}")
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                    completion_tokens = int(usage.get("completion_tokens", completion_tokens))
                    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
                    usage_received = True
                for choice in chunk.get("choices") or []:
                    reason = choice.get("finish_reason")
                    if reason is not None:
                        finish_reason = str(reason)
                    text = choice.get("text")
                    if text is None:
                        delta = choice.get("delta") or {}
                        # A thinking model streams the block as reasoning_content;
                        # those are output tokens and usage counts them, so not
                        # reading them reports zero events on a short cap.
                        text = delta.get("content") or delta.get("reasoning_content")
                    if not text:
                        continue
                    now = time.perf_counter()
                    if first_event is None:
                        first_event = now
                    elif previous_event is not None:
                        itl.append(now - previous_event)
                    previous_event = now
                    output_parts.append(str(text))
                    output_events += 1
    except Exception as exc:
        return error_result(request_id, prompt, started, f"{type(exc).__name__}: {exc}")

    ended = time.perf_counter()
    missing = []
    if not saw_done:
        missing.append("missing [DONE]")
    if not usage_received:
        missing.append("missing usage")
    if first_event is None:
        missing.append("missing output event")
    if finish_reason is None:
        missing.append("missing finish_reason")
    # MTP/speculative decode sends multiple accepted tokens per SSE event, so
    # output_events may be < completion_tokens. The usage-based token count is
    # the ground truth for tok/s; ITL uses inter-event intervals (approximate).
    output = "".join(output_parts)
    return RequestResult(
        request_id=request_id,
        prompt=prompt,
        status="incomplete" if missing else "complete",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        ttft_s=None if first_event is None else first_event - started,
        itl_s=itl,
        e2e_s=ended - started,
        finish_reason=finish_reason,
        output=output,
        output_events=output_events,
        usage_received=usage_received,
        saw_done=saw_done,
        correctness_error=correctness_error(output),
        error=", ".join(missing) or None,
    )


async def fetch_stats(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    try:
        response = await client.get(f"{url.rstrip('/')}/v1/stats")
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = response.text[:2000]
        return {"status_code": response.status_code, "body": body}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def run_point(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
    requests: int | None,
    duration_s: float | None,
    max_tokens: int,
    temperature: float,
) -> tuple[list[RequestResult], float]:
    results = []
    next_id = 0
    started = time.perf_counter()
    deadline = None if duration_s is None else started + duration_s

    async def worker() -> None:
        nonlocal next_id
        while True:
            if requests is not None and next_id >= requests:
                return
            if deadline is not None and time.perf_counter() >= deadline:
                return
            request_id = next_id
            next_id += 1
            result = await send_streaming(
                client,
                url,
                model,
                prompts[request_id % len(prompts)],
                max_tokens,
                temperature,
                request_id,
            )
            results.append(result)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    results.sort(key=lambda result: result.request_id)
    return results, time.perf_counter() - started


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = percentile_value / 100 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean_ms": None if not values else statistics.mean(values) * 1000,
        "p50_ms": None if not values else percentile(values, 50) * 1000,
        "p90_ms": None if not values else percentile(values, 90) * 1000,
        "p99_ms": None if not values else percentile(values, 99) * 1000,
    }


def summarize(concurrency: int, results: list[RequestResult], wall_time_s: float) -> dict[str, Any]:
    complete = [result for result in results if result.status == "complete"]
    prompt_tokens = sum(result.prompt_tokens for result in complete)
    output_tokens = sum(result.completion_tokens for result in complete)
    total_tokens = sum(result.total_tokens for result in complete)
    return {
        "concurrency": concurrency,
        "wall_time_s": wall_time_s,
        "requests": len(results),
        "complete": len(complete),
        "incomplete": sum(result.status == "incomplete" for result in results),
        "error": sum(result.status == "error" for result in results),
        "correctness_failed": sum(not result.gate_pass for result in complete),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "output_tokens_per_s": output_tokens / wall_time_s if wall_time_s else 0.0,
        "total_tokens_per_s": total_tokens / wall_time_s if wall_time_s else 0.0,
        "requests_per_s": len(complete) / wall_time_s if wall_time_s else 0.0,
        "ttft": latency_summary([result.ttft_s for result in complete if result.ttft_s is not None]),
        "itl": latency_summary([value for result in complete for value in result.itl_s]),
        "e2e": latency_summary([result.e2e_s for result in complete]),
    }


def atomic_write(path: Path, render) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
            render(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_outputs(output_prefix: Path, report: dict[str, Any]) -> None:
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    atomic_write(json_path, lambda output: json.dump(report, output, indent=2, sort_keys=True))

    columns = (
        "concurrency", "wall_time_s", "requests", "complete", "incomplete", "error",
        "correctness_failed", "prompt_tokens", "output_tokens", "total_tokens",
        "output_tokens_per_s", "total_tokens_per_s", "requests_per_s",
        "ttft_mean_ms", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
        "itl_mean_ms", "itl_p50_ms", "itl_p90_ms", "itl_p99_ms",
        "e2e_mean_ms", "e2e_p50_ms", "e2e_p90_ms", "e2e_p99_ms",
    )

    def render_csv(output) -> None:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for point in report["points"]:
            summary = dict(point["summary"])
            row = {key: summary.get(key) for key in columns}
            for metric in ("ttft", "itl", "e2e"):
                latency = summary[metric]
                for stat in ("mean_ms", "p50_ms", "p90_ms", "p99_ms"):
                    row[f"{metric}_{stat}"] = latency[stat]
            writer.writerow(row)

    atomic_write(csv_path, render_csv)


def parse_grid(value: str) -> tuple[int, ...]:
    try:
        grid = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("concurrency grid must be comma-separated integers") from exc
    if not grid or any(value <= 0 for value in grid) or len(set(grid)) != len(grid):
        raise argparse.ArgumentTypeError("concurrency grid must contain unique positive integers")
    return grid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="")
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--synthetic-prompts", type=int, default=64)
    parser.add_argument("--concurrency-grid", type=parse_grid, default=DEFAULT_CONCURRENCY_GRID)
    measure = parser.add_mutually_exclusive_group()
    measure.add_argument("--requests-per-concurrency", type=int)
    measure.add_argument("--seconds-per-concurrency", type=float)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("bench_throughput"))
    args = parser.parse_args(argv)
    if args.requests_per_concurrency is None and args.seconds_per_concurrency is None:
        args.requests_per_concurrency = 64
    if args.requests_per_concurrency is not None and args.requests_per_concurrency <= 0:
        parser.error("--requests-per-concurrency must be positive")
    if args.seconds_per_concurrency is not None and args.seconds_per_concurrency <= 0:
        parser.error("--seconds-per-concurrency must be positive")
    if args.max_tokens <= 0 or args.timeout_seconds <= 0 or args.synthetic_prompts <= 0:
        parser.error("token, timeout, and prompt counts must be positive")
    return args


async def async_main(args: argparse.Namespace) -> int:
    prompts = (
        load_jsonl_prompts(args.prompts_jsonl)
        if args.prompts_jsonl
        else synthetic_prompts(args.synthetic_prompts, args.seed)
    )
    report: dict[str, Any] = {
        "schema": "arle.bench_throughput.v1",
        "started_unix_s": time.time(),
        "config": {
            "url": args.url,
            "model": args.model,
            "prompt_count": len(prompts),
            "concurrency_grid": list(args.concurrency_grid),
            "requests_per_concurrency": args.requests_per_concurrency,
            "seconds_per_concurrency": args.seconds_per_concurrency,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "ignore_eos": True,
        },
        "points": [],
    }
    limits = httpx.Limits(max_connections=max(args.concurrency_grid))
    timeout = httpx.Timeout(args.timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        warmup = await send_streaming(client, args.url, args.model, prompts[0], min(16, args.max_tokens), 0.0, -1)
        if not warmup.gate_pass:
            print(f"warmup failed: {warmup.error or warmup.correctness_error}", file=sys.stderr)
            return 2
        for concurrency in args.concurrency_grid:
            stats_before = await fetch_stats(client, args.url)
            results, wall_time_s = await run_point(
                client,
                args.url,
                args.model,
                prompts,
                concurrency,
                args.requests_per_concurrency,
                args.seconds_per_concurrency,
                args.max_tokens,
                args.temperature,
            )
            stats_after = await fetch_stats(client, args.url)
            summary = summarize(concurrency, results, wall_time_s)
            report["points"].append({
                "summary": summary,
                "stats_before": stats_before,
                "stats_after": stats_after,
                "results": [asdict(result) for result in results],
            })
            write_outputs(args.output, report)
            print(
                f"c={concurrency}: complete={summary['complete']}/{summary['requests']} "
                f"out={summary['output_tokens_per_s']:.1f} tok/s "
                f"total={summary['total_tokens_per_s']:.1f} tok/s "
                f"req={summary['requests_per_s']:.2f}/s"
            )
    failed = any(
        summary["requests"] == 0
        or summary["complete"] != summary["requests"]
        or summary["correctness_failed"]
        for summary in (point["summary"] for point in report["points"])
    )
    return int(failed)


def main() -> None:
    sys.exit(asyncio.run(async_main(parse_args())))


if __name__ == "__main__":
    main()

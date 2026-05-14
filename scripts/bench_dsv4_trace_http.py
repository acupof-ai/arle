#!/usr/bin/env python3
"""Run DSv4 HTTP benchmarks and collect request_trace summaries.

The server emits one structured log line per completed request:

    request_trace {"request_id": "...", ...}

When --trace-log points at that server log, this script captures only the new
lines produced during the run and includes them in the final JSON summary.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Case:
    label: str
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


def repeated_prompt(words: int) -> str:
    return ("one " * words) + "\nQuestion: What is one plus one? Answer briefly."


def decode_prompt() -> str:
    return (
        "You are benchmarking decoding speed. Continue with short plain English "
        "words separated by spaces. Keep going until the token limit. Seed words: "
        "alpha beta gamma delta."
    )


def default_cases(include_long: bool) -> list[Case]:
    cases = [
        Case("decode64", decode_prompt(), 64, True),
        Case("prefill1k", repeated_prompt(1000), 1),
        Case("prefill4k", repeated_prompt(4000), 1),
        Case("math", "Calculate 17 * 23 + 19. Return only the final integer.", 32),
        Case(
            "write_zh",
            "用三句话写一段中文发布说明，主题是长上下文推理性能优化，需要提到吞吐、流式输出和 trace 可观测性。",
            96,
        ),
    ]
    if include_long:
        cases.insert(3, Case("prefill20k", repeated_prompt(20000), 1))
    return cases


def case_summary(case: Case) -> dict[str, Any]:
    return {
        "label": case.label,
        "prompt_bytes": len(case.prompt.encode("utf-8")),
        "max_tokens": case.max_tokens,
        "ignore_eos": case.ignore_eos,
    }


def request_stream(
    host: str,
    port: int,
    model: str,
    case: Case,
    timeout: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": case.prompt}],
            "max_tokens": case.max_tokens,
            "temperature": 0,
            "ignore_eos": case.ignore_eos,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        ensure_ascii=False,
    ).encode()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    started_at = time.time()
    first_content_at = None
    output: list[str] = []
    usage = None
    keepalives = 0
    status = None
    error = None
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        status = resp.status
        while True:
            line = resp.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            if text.startswith(":"):
                keepalives += 1
                continue
            if not text.startswith("data: "):
                continue
            data = text[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    if first_content_at is None:
                        first_content_at = time.time()
                    output.append(content)
    except Exception as exc:  # pragma: no cover - used as bench tool.
        error = repr(exc)
    finally:
        conn.close()

    total_s = time.time() - started_at
    ttft_s = None if first_content_at is None else first_content_at - started_at
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens")
    total_tokens = (usage or {}).get("total_tokens")
    decode_window_s = None if ttft_s is None else max(total_s - ttft_s, 0.0)
    return {
        "label": case.label,
        "status": status,
        "error": error,
        "max_tokens": case.max_tokens,
        "ignore_eos": case.ignore_eos,
        "ttft_s": None if ttft_s is None else round(ttft_s, 4),
        "total_s": round(total_s, 4),
        "usage": usage,
        "prompt_tok_s_at_ttft": (
            round(prompt_tokens / ttft_s, 2)
            if prompt_tokens and ttft_s and ttft_s > 0
            else None
        ),
        "post_first_decode_tok_s": (
            round((completion_tokens - 1) / decode_window_s, 2)
            if completion_tokens
            and completion_tokens > 1
            and decode_window_s
            and decode_window_s > 0
            else None
        ),
        "e2e_total_tok_s": (
            round(total_tokens / total_s, 2) if total_tokens and total_s > 0 else None
        ),
        "keepalives": keepalives,
        "output": "".join(output)[:500],
    }


def parse_new_request_traces(path: str, start_offset: int) -> list[dict[str, Any]]:
    traces = []
    if not path or not os.path.exists(path):
        return traces
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        handle.seek(start_offset)
        for line in handle:
            marker = "request_trace "
            idx = line.find(marker)
            if idx < 0:
                continue
            payload = line[idx + len(marker) :].strip()
            try:
                traces.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    return traces


def run_fanout(args: argparse.Namespace) -> dict[str, Any]:
    case = Case("fanout_decode16", decode_prompt(), 16, True)
    started_at = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.fanout) as pool:
        results = list(
            pool.map(
                lambda _: request_stream(
                    args.host, args.port, args.model, case, args.timeout
                ),
                range(args.fanout),
            )
        )
    total_s = time.time() - started_at
    completion_tokens = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in results)
    total_tokens = sum((r.get("usage") or {}).get("total_tokens", 0) for r in results)
    return {
        "label": "fanout_decode16",
        "fanout": args.fanout,
        "wall_s": round(total_s, 4),
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "aggregate_completion_tok_s": (
            round(completion_tokens / total_s, 2) if total_s > 0 else None
        ),
        "aggregate_total_tok_s": round(total_tokens / total_s, 2) if total_s > 0 else None,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--include-long", action="store_true")
    parser.add_argument(
        "--fanout",
        type=int,
        default=0,
        help="Optional concurrent decode requests; default 0 keeps trace smoke single-request.",
    )
    parser.add_argument("--trace-log", help="Server log containing request_trace lines")
    parser.add_argument("--output-json", help="Write full summary JSON to this path")
    args = parser.parse_args()

    start_offset = 0
    if args.trace_log and os.path.exists(args.trace_log):
        start_offset = os.path.getsize(args.trace_log)

    results = []
    for case in default_cases(args.include_long):
        result = request_stream(args.host, args.port, args.model, case, args.timeout)
        results.append(result)
        print("RESULT " + json.dumps(result, ensure_ascii=False), flush=True)

    fanout = run_fanout(args) if args.fanout > 0 else None
    if fanout is not None:
        print("FANOUT " + json.dumps(fanout, ensure_ascii=False), flush=True)

    traces = parse_new_request_traces(args.trace_log, start_offset) if args.trace_log else []
    summary = {
        "server": {"host": args.host, "port": args.port, "model": args.model},
        "cases": [case_summary(case) for case in default_cases(args.include_long)],
        "results": results,
        "fanout": fanout,
        "request_traces": traces,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()

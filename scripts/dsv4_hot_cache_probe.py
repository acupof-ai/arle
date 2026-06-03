#!/usr/bin/env python3
"""Probe DSv4 hot-prefix performance with an explicit warm request.

Runs against an already-started OpenAI-compatible server. The first request
publishes the prefix into the server cache. The second request streams the same
prompt and reports the target metrics: TTFT, TPOT, E2E, and output throughput.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RequestResult:
    label: str
    status: int | None
    error: str | None
    ttft_s: float | None
    total_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    tpot_ms: float | None
    output_tok_s: float | None
    decode_tok_s: float | None
    keepalives: int
    output_prefix: str


def make_prompt(words: int, word: str) -> str:
    prefix = "You are measuring long-context decode throughput. "
    suffix = "\nContinue with short plain English words until the token limit."
    return prefix + ((word + " ") * words) + suffix


def post_stream(
    host: str,
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    label: str,
) -> RequestResult:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        ensure_ascii=False,
    ).encode()

    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    start = time.perf_counter()
    first_content = None
    status = None
    error = None
    usage: dict[str, Any] | None = None
    output_parts: list[str] = []
    keepalives = 0
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
                    if first_content is None:
                        first_content = time.perf_counter()
                    output_parts.append(content)
    except Exception as exc:  # pragma: no cover - bench helper.
        error = repr(exc)
    finally:
        conn.close()

    end = time.perf_counter()
    total_s = end - start
    ttft_s = None if first_content is None else first_content - start
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens")
    total_tokens = (usage or {}).get("total_tokens")
    decode_window_s = None if ttft_s is None else max(total_s - ttft_s, 0.0)
    tpot_ms = None
    decode_tok_s = None
    if completion_tokens and completion_tokens > 1 and decode_window_s and decode_window_s > 0:
        tpot_ms = 1000.0 * decode_window_s / (completion_tokens - 1)
        decode_tok_s = (completion_tokens - 1) / decode_window_s
    output_tok_s = completion_tokens / total_s if completion_tokens and total_s > 0 else None
    return RequestResult(
        label=label,
        status=status,
        error=error,
        ttft_s=None if ttft_s is None else round(ttft_s, 6),
        total_s=round(total_s, 6),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tpot_ms=None if tpot_ms is None else round(tpot_ms, 6),
        output_tok_s=None if output_tok_s is None else round(output_tok_s, 6),
        decode_tok_s=None if decode_tok_s is None else round(decode_tok_s, 6),
        keepalives=keepalives,
        output_prefix="".join(output_parts)[:240],
    )


def parse_new_request_traces(path: str | None, start_offset: int) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
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


def hot_prefix_status(trace: dict[str, Any] | None) -> dict[str, Any]:
    prefix = (trace or {}).get("prefix") or {}
    return {
        "direct_gpu_attach": bool(prefix.get("direct_gpu_attach")),
        "lookup_reusable_tokens": prefix.get("lookup_reusable_tokens"),
        "resume_prefill_tokens": prefix.get("resume_prefill_tokens"),
        "matched_tokens": prefix.get("matched_tokens"),
        "ready_on_gpu": prefix.get("ready_on_gpu"),
        "staged": prefix.get("staged"),
        "recompute": prefix.get("recompute"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18300)
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--prompt-words", type=int, default=262144)
    parser.add_argument("--prompt-word", default="one")
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--warmup-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--trace-log", help="Server log with request_trace lines")
    parser.add_argument(
        "--require-hot-prefix",
        action="store_true",
        help="Exit non-zero unless the measured request used direct GPU prefix attach.",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    trace_start_offset = 0
    if args.trace_log and os.path.exists(args.trace_log):
        trace_start_offset = os.path.getsize(args.trace_log)

    prompt = make_prompt(args.prompt_words, args.prompt_word)
    prompt_bytes = len(prompt.encode("utf-8"))
    print(
        "CONFIG "
        + json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "model": args.model,
                "prompt_words": args.prompt_words,
                "prompt_bytes": prompt_bytes,
                "max_tokens": args.max_tokens,
                "warmup_tokens": args.warmup_tokens,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    warmup = post_stream(
        args.host,
        args.port,
        args.model,
        prompt,
        args.warmup_tokens,
        args.timeout,
        "warmup",
    )
    print("WARMUP " + json.dumps(asdict(warmup), sort_keys=True), flush=True)

    measured = post_stream(
        args.host,
        args.port,
        args.model,
        prompt,
        args.max_tokens,
        args.timeout,
        "measured",
    )
    print("MEASURED " + json.dumps(asdict(measured), sort_keys=True), flush=True)

    request_traces = parse_new_request_traces(args.trace_log, trace_start_offset)
    warmup_trace = request_traces[-2] if len(request_traces) >= 2 else None
    measured_trace = request_traces[-1] if request_traces else None
    hot_prefix = hot_prefix_status(measured_trace)
    if measured_trace is not None:
        print("MEASURED_TRACE " + json.dumps(measured_trace, sort_keys=True), flush=True)
    print("HOT_PREFIX " + json.dumps(hot_prefix, sort_keys=True), flush=True)

    summary = {
        "config": {
            "host": args.host,
            "port": args.port,
            "model": args.model,
            "prompt_words": args.prompt_words,
            "prompt_bytes": prompt_bytes,
            "max_tokens": args.max_tokens,
            "warmup_tokens": args.warmup_tokens,
        },
        "warmup": asdict(warmup),
        "measured": asdict(measured),
        "warmup_trace": warmup_trace,
        "measured_trace": measured_trace,
        "hot_prefix": hot_prefix,
        "request_traces": request_traces,
        "target": {
            "ttft_s": 0.44,
            "tpot_ms": 4.85,
            "e2e_s": 7.7,
            "output_tok_s": 196.0,
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if args.require_hot_prefix and not hot_prefix["direct_gpu_attach"]:
        print("HOT_PREFIX_FAIL measured request did not use direct GPU prefix attach", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()

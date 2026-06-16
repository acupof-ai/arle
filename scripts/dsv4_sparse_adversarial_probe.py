#!/usr/bin/env python3
"""DSv4 sparse-attention adversarial probe with strict answer extraction."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


FINAL_RE = re.compile(r"FINAL:\s*([^\n\r<]+)", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")


@dataclass
class ProbeCase:
    label: str
    prompt: str
    expected: str


def post_completion(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "ignore_eos": False,
        },
        ensure_ascii=False,
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            status = resp.status
    except urllib.error.HTTPError as exc:
        return {
            "status": exc.code,
            "error": exc.read().decode("utf-8", "replace")[:500],
            "wall_s": time.perf_counter() - started,
            "text": "",
            "usage": None,
        }
    except Exception as exc:  # pragma: no cover - bench tool path.
        return {
            "status": None,
            "error": repr(exc),
            "wall_s": time.perf_counter() - started,
            "text": "",
            "usage": None,
        }
    text = ""
    try:
        text = data["choices"][0].get("text", "")
    except Exception:
        pass
    return {
        "status": status,
        "error": None,
        "wall_s": time.perf_counter() - started,
        "text": text,
        "usage": data.get("usage"),
    }


def extract_answer(text: str) -> str:
    match = FINAL_RE.search(text)
    if match:
        return normalize_answer(match.group(1))
    ints = INT_RE.findall(text)
    if ints:
        return normalize_answer(",".join(ints))
    return normalize_answer(text)


def normalize_answer(value: str) -> str:
    ints = INT_RE.findall(value)
    if ints:
        return ",".join(ints)
    return re.sub(r"\s+", " ", value.strip().lower())


def near_duplicate_blocks(source_count: int, target_idx: int, rep: int) -> ProbeCase:
    blocks = []
    expected = str(700000 + target_idx * 37 + rep)
    for idx in range(source_count):
        tag = f"TAG-{idx:04d}"
        value = expected if idx == target_idx else str(700000 + idx * 37 + rep + 11)
        blocks.append(
            f"[{tag}] The audit record is intentionally near-identical. "
            f"Project=delta lane=blue checksum={value}. Keep this exact checksum."
        )
    prompt = "\n".join(blocks)
    prompt += (
        f"\n\nQuestion: Find the checksum for TAG-{target_idx:04d}. "
        "Answer exactly one line in this format: FINAL: <integer>"
    )
    return ProbeCase(f"c1_near_blocks_{source_count}", prompt, normalize_answer(expected))


def multisource_case(source_count: int, rep: int) -> ProbeCase:
    src_a = 3 + rep % max(1, source_count)
    src_b = max(0, source_count - 5 - rep % max(1, source_count // 4))
    shared = [9000 + rep, 9017 + rep]
    exclude = 9033 + rep
    blocks = []
    for idx in range(source_count):
        values = [1000 + idx, 2000 + idx]
        if idx in (src_a, src_b):
            values.extend(shared)
        if idx == src_a:
            values.append(exclude)
        if idx == max(0, src_a - 1):
            values.append(exclude)
        blocks.append(f"[SRC-{idx:04d}] values: {', '.join(str(v) for v in sorted(values))}.")
    expected = ",".join(str(v) for v in shared)
    prompt = "\n".join(blocks)
    prompt += (
        f"\n\nQuestion: Return values that appear in BOTH SRC-{src_a:04d} and SRC-{src_b:04d}, "
        f"but NOT in SRC-{max(0, src_a - 1):04d}. "
        "Answer sorted ascending, exactly: FINAL: <comma-separated integers>"
    )
    return ProbeCase(f"c2_multisource_{source_count}", prompt, normalize_answer(expected))


def run_case(args: argparse.Namespace, case: ProbeCase, rep: int) -> dict[str, Any]:
    response = post_completion(args.base_url, args.model, case.prompt, args.max_tokens, args.timeout)
    extracted = extract_answer(response["text"])
    ok = response["error"] is None and extracted == case.expected
    return {
        "label": case.label,
        "rep": rep,
        "prompt_bytes": len(case.prompt.encode("utf-8")),
        "status": response["status"],
        "error": response["error"],
        "wall_s": round(response["wall_s"], 4),
        "usage": response["usage"],
        "expected": case.expected,
        "extracted": extracted,
        "ok": ok,
        "text_head": response["text"][:300],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18188")
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--source-counts", default="64,128,240")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    source_counts = [int(v) for v in args.source_counts.split(",") if v.strip()]
    results = []
    for count in source_counts:
        for rep in range(args.reps):
            c1_target = (count * 3 // 5 + rep * 7) % count
            for case in (
                near_duplicate_blocks(count, c1_target, rep),
                multisource_case(count, rep),
            ):
                row = run_case(args, case, rep)
                results.append(row)
                print("RESULT " + json.dumps(row, ensure_ascii=False), flush=True)

    summary = []
    labels = sorted({r["label"] for r in results})
    for label in labels:
        rows = [r for r in results if r["label"] == label]
        ok_count = sum(1 for r in rows if r["ok"])
        walls = [r["wall_s"] for r in rows if r["error"] is None]
        item = {
            "label": label,
            "ok": ok_count,
            "total": len(rows),
            "accuracy": ok_count / len(rows) if rows else 0.0,
            "median_wall_s": statistics.median(walls) if walls else None,
        }
        summary.append(item)
        print("SUMMARY_ROW " + json.dumps(item, ensure_ascii=False), flush=True)

    payload = {"config": vars(args), "summary": summary, "results": results}
    print("JSON_SUMMARY=" + json.dumps(payload, ensure_ascii=False), flush=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

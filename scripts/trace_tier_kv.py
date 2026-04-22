#!/usr/bin/env python3
"""Trace staged KV recall across T1/T2/T3 with a fixed three-phase workload.

Phase A warms one long canonical prefix repeatedly so it is published into the
prefix cache and accumulates hit count.
Phase B churns unrelated long prompts to force demotion / spill.
Phase C replays the canonical prefix and reads the service stats surface to see
where the staged prefix was recalled from.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/"
    "1cfa9a7208912126459214e8b04321603b3df60c"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warm-prefixes", type=int, default=1)
    parser.add_argument("--warm-repeats", type=int, default=4)
    parser.add_argument("--churn-requests", type=int, default=24)
    parser.add_argument("--replay-prefixes", type=int, default=1)
    parser.add_argument("--replay-repeats", type=int, default=2)
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=4608)
    parser.add_argument("--mem-fraction-static", type=float, default=0.94)
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--max-prefill-tokens", type=int, default=16384)
    parser.add_argument("--t1-host-pinned-high-water", type=float, default=0.35)
    parser.add_argument("--t1-host-pinned-low-water", type=float, default=0.20)
    parser.add_argument("--t1-host-pinned-keepalive-ticks", type=int, default=512)
    parser.add_argument("--disk-store-root", required=True)
    parser.add_argument("--cluster-shared-root")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def wait_for_server(base_url: str, timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            return http_json(f"{base_url}/v1/models")
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(1.0)
    raise RuntimeError(f"server did not become ready at {base_url}: {last_err}")


def http_json(url: str, payload: dict | None = None, timeout: float = 300.0) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_stats(base_url: str) -> dict[str, str]:
    with urllib.request.urlopen(f"{base_url}/v1/stats", timeout=30) as response:
        raw = response.read().decode("utf-8")
    fields: dict[str, str] = {}
    for token in raw.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def build_prompt(tokenizer, token_count: int, label: str) -> str:
    seed_text = (
        f"{label} system context. "
        "The prompt intentionally repeats structured factual clauses so tokenization "
        "stays deterministic across runs. "
    )
    words: list[str] = []
    while True:
        words.append(seed_text)
        text = "".join(words)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) >= token_count:
            return tokenizer.decode(token_ids[:token_count], clean_up_tokenization_spaces=False)


def issue_completion(base_url: str, model_id: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
    }
    started = time.perf_counter()
    response = http_json(f"{base_url}/v1/completions", payload, timeout=600.0)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    usage = response.get("usage", {})
    choices = response.get("choices", [])
    text = choices[0].get("text", "") if choices else ""
    return {
        "elapsed_ms": round(elapsed_ms, 1),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "text_len": len(text),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
    }


def run_phase(
    base_url: str,
    model_id: str,
    prompts: list[str],
    max_tokens: int,
) -> list[dict]:
    results = []
    for prompt in prompts:
        results.append(issue_completion(base_url, model_id, prompt, max_tokens))
    return results


def summarize_phase(name: str, stats: dict[str, str], results: list[dict]) -> dict:
    return {
        "name": name,
        "stats": stats,
        "requests": len(results),
        "elapsed_ms": [result["elapsed_ms"] for result in results],
        "prompt_tokens": [result["prompt_tokens"] for result in results],
        "completion_tokens": [result["completion_tokens"] for result in results],
        "text_len": [result["text_len"] for result in results],
        "finish_reason": [result["finish_reason"] for result in results],
    }


def main() -> int:
    args = parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as err:  # pragma: no cover - tool dependency
        raise SystemExit(f"transformers is required for trace_tier_kv.py: {err}") from err

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.disk_store_root).mkdir(parents=True, exist_ok=True)
    if args.cluster_shared_root:
        Path(args.cluster_shared_root).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    warm_prompts = [
        build_prompt(tokenizer, args.prompt_tokens, f"warm-prefix-{index:03d}")
        for index in range(args.warm_prefixes)
    ]
    churn_prompts = [
        build_prompt(tokenizer, args.prompt_tokens, f"churn-{index:03d}")
        for index in range(args.churn_requests)
    ]

    server_cmd = [
        "./target/release/infer",
        "--model-path",
        args.model_path,
        "--port",
        str(args.port),
        "--num-slots",
        str(args.num_slots),
        "--max-seq-len",
        str(args.max_seq_len),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-prefill-tokens",
        str(args.max_prefill_tokens),
        "--t1-host-pinned-high-water",
        str(args.t1_host_pinned_high_water),
        "--t1-host-pinned-low-water",
        str(args.t1_host_pinned_low_water),
        "--t1-host-pinned-keepalive-ticks",
        str(args.t1_host_pinned_keepalive_ticks),
        "--disk-store-root",
        args.disk_store_root,
        "--trace-output-path",
        str(trace_dir),
    ]
    if args.cluster_shared_root:
        server_cmd.extend(["--cluster-shared-root", args.cluster_shared_root])

    env = os.environ.copy()
    env.setdefault("RUST_LOG", "info")

    with log_path.open("wb") as log_file:
        server = subprocess.Popen(
            server_cmd,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            base_url = f"http://127.0.0.1:{args.port}"
            models = wait_for_server(base_url, timeout_s=180.0)
            model_id = models["data"][0]["id"]

            before = fetch_stats(base_url)
            warm_results = run_phase(
                base_url,
                model_id,
                [
                    prompt
                    for prompt in warm_prompts
                    for _ in range(args.warm_repeats)
                ],
                args.max_tokens,
            )
            after_warm = fetch_stats(base_url)

            churn_results = run_phase(
                base_url,
                model_id,
                churn_prompts,
                args.max_tokens,
            )
            time.sleep(2.0)
            after_churn = fetch_stats(base_url)

            replay_results = run_phase(
                base_url,
                model_id,
                [
                    prompt
                    for prompt in warm_prompts[: args.replay_prefixes]
                    for _ in range(args.replay_repeats)
                ],
                args.max_tokens,
            )
            after_replay = fetch_stats(base_url)
        finally:
            server.send_signal(signal.SIGINT)
            server.wait(timeout=120.0)

    summary = {
        "command": server_cmd,
        "base_url": f"http://127.0.0.1:{args.port}",
        "before": before,
        "after_warm": after_warm,
        "after_churn": after_churn,
        "after_replay": after_replay,
        "warm": summarize_phase("warm", after_warm, warm_results),
        "churn": summarize_phase("churn", after_churn, churn_results),
        "replay": summarize_phase("replay", after_replay, replay_results),
        "artefacts": {
            "log_path": str(log_path),
            "trace_dir": str(trace_dir),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "after_churn": {
                "prefix_hit_rate": after_churn.get("prefix_hit_rate"),
                "prefix_skip_rate": after_churn.get("prefix_skip_rate"),
                "kv_store_q": after_churn.get("kv_store_q"),
                "kv_fetch_q": after_churn.get("kv_fetch_q"),
            },
            "after_replay": {
                "prefix_hit_rate": after_replay.get("prefix_hit_rate"),
                "prefix_skip_rate": after_replay.get("prefix_skip_rate"),
                "tier_recall": after_replay.get("tier_recall"),
                "tier_src": after_replay.get("tier_src"),
                "tier_promoted": after_replay.get("tier_promoted"),
                "tier_fallback": after_replay.get("tier_fallback"),
            },
            "out": str(out_path),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""PPL evaluation for KV quantization using HuggingFace datasets.

Computes pseudo-PPL by collecting per-token logprobs from greedy streaming
decode. Runs each KV format (bf16/fp8/int8) sequentially on the same GPU.

Usage:
  python3 scripts/eval_ppl.py [--datasets wikitext,humaneval] [--max-tokens 200]
"""

import argparse
import httpx
import json
import math
import os
import subprocess
import sys
import time

URL = "http://localhost:8090"
BIN = "target/release/infer"
MODEL = "models/Qwen3-4B"


def load_dataset_texts(name, max_samples=20):
    """Load text samples from HuggingFace datasets."""
    from datasets import load_dataset

    texts = []
    if name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        for row in ds:
            t = row["text"].strip()
            if len(t) > 100:  # skip short/empty
                texts.append(t)
                if len(texts) >= max_samples:
                    break

    elif name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        for row in ds:
            # Use the prompt (function signature + docstring) as context
            t = row["prompt"].strip()
            if len(t) > 50:
                texts.append(t)
                if len(texts) >= max_samples:
                    break

    elif name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for row in ds:
            t = row["question"].strip()
            if len(t) > 50:
                texts.append(t)
                if len(texts) >= max_samples:
                    break

    elif name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        for row in ds:
            t = row["prompt"].strip()
            if len(t) > 30:
                texts.append(t)
                if len(texts) >= max_samples:
                    break

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return texts


def start_server(kv_dtype=None):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia:/usr/local/cuda/lib64:" + env.get(
        "LD_LIBRARY_PATH", ""
    )
    cmd = [BIN, "--model-path", MODEL, "--port", "8090", "--num-slots", "1"]
    if kv_dtype:
        cmd += ["--kv-cache-dtype", kv_dtype]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
    )
    for _ in range(30):
        try:
            r = httpx.post(
                f"{URL}/v1/completions",
                json={"model": "q", "prompt": "Hi", "max_tokens": 1, "temperature": 0},
                timeout=5,
            )
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(2)
    proc.kill()
    raise RuntimeError(f"Server failed to start (dtype={kv_dtype})")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(1)


def collect_logprobs(prompt, max_tokens=50):
    """Stream completion and collect per-token logprobs."""
    logprobs = []
    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{URL}/v1/completions",
            json={
                "model": "q",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
            },
        ) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                d = line[6:]
                if d.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(d)
                    lp = obj.get("choices", [{}])[0].get("logprobs")
                    if lp and "token_logprobs" in lp:
                        logprobs.extend(lp["token_logprobs"])
                except json.JSONDecodeError:
                    pass
    return logprobs


def compute_ppl(logprobs):
    if not logprobs:
        return float("inf")
    avg = sum(logprobs) / len(logprobs)
    return math.exp(-avg)


def eval_format(label, dtype, dataset_texts, max_tokens):
    print(f"\n  Starting {label} server...")
    proc = start_server(dtype)
    all_lps = []
    for i, text in enumerate(dataset_texts):
        lps = collect_logprobs(text, max_tokens)
        all_lps.extend(lps)
        if (i + 1) % 5 == 0 or i == len(dataset_texts) - 1:
            print(
                f"    [{i+1}/{len(dataset_texts)}] {len(all_lps)} tokens so far, "
                f"running PPL={compute_ppl(all_lps):.4f}"
            )
    stop_server(proc)
    ppl = compute_ppl(all_lps)
    return ppl, len(all_lps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="wikitext,humaneval",
        help="Comma-separated dataset names: wikitext,humaneval,gsm8k,mbpp",
    )
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=15)
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    formats = [("BF16", None), ("FP8", "fp8"), ("INT8", "int8")]

    # Download datasets first
    print("Loading datasets...")
    datasets = {}
    for name in dataset_names:
        texts = load_dataset_texts(name, max_samples=args.max_samples)
        datasets[name] = texts
        print(f"  {name}: {len(texts)} samples")

    all_results = {}  # {dataset: {format: (ppl, n_tokens)}}

    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({len(datasets[ds_name])} samples, {args.max_tokens} tok/sample)")
        print(f"{'='*60}")

        all_results[ds_name] = {}
        for label, dtype in formats:
            ppl, n = eval_format(label, dtype, datasets[ds_name], args.max_tokens)
            all_results[ds_name][label] = (ppl, n)
            print(f"  {label}: PPL={ppl:.4f} ({n} tokens)")

    # Summary
    print(f"\n{'='*70}")
    print(f"PPL Summary (lower = better)")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'BF16 PPL':>10} {'FP8 PPL':>10} {'FP8 Δ':>8} {'INT8 PPL':>10} {'INT8 Δ':>8}")
    print("-" * 70)
    for ds_name in dataset_names:
        r = all_results[ds_name]
        bp = r["BF16"][0]
        fp = r["FP8"][0]
        ip = r["INT8"][0]
        fd = ((fp / bp) - 1) * 100
        id_ = ((ip / bp) - 1) * 100
        print(f"{ds_name:<12} {bp:>10.4f} {fp:>10.4f} {fd:>+7.2f}% {ip:>10.4f} {id_:>+7.2f}%")

    # Aggregated
    all_bf16_lps, all_fp8_lps, all_int8_lps = 0, 0, 0
    all_bf16_n, all_fp8_n, all_int8_n = 0, 0, 0
    for ds_name in dataset_names:
        all_bf16_n += all_results[ds_name]["BF16"][1]
        all_fp8_n += all_results[ds_name]["FP8"][1]
        all_int8_n += all_results[ds_name]["INT8"][1]
    print("-" * 70)
    total_bf16 = sum(r["BF16"][0] for r in all_results.values()) / len(dataset_names)
    total_fp8 = sum(r["FP8"][0] for r in all_results.values()) / len(dataset_names)
    total_int8 = sum(r["INT8"][0] for r in all_results.values()) / len(dataset_names)
    fd = ((total_fp8 / total_bf16) - 1) * 100
    id_ = ((total_int8 / total_bf16) - 1) * 100
    print(f"{'avg':<12} {total_bf16:>10.4f} {total_fp8:>10.4f} {fd:>+7.2f}% {total_int8:>10.4f} {id_:>+7.2f}%")


if __name__ == "__main__":
    main()

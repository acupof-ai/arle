#!/usr/bin/env python3
"""PF8.3 PPL gate — A/B prefill activation FP8 vs INT8 baseline.

Adapted from scripts/eval_ppl.py (KV-format axis) per
docs/research/2026-05-10-pf83-ppl-gate-methodology.md (aebd4a5) §2
Option A: env-var axis instead of --kv-cache-dtype.

Computes pseudo-PPL via per-token logprobs from greedy streaming
decode, A/B with INFER_MARLIN_W4_FP8_PREFILL=0 (baseline) vs =1
(treatment).

License gate per a66d99a §2 + aebd4a5 §3:
  PPL Δ% ≤ +1.0%  → license PF8.3 (paired with greedy_consistency PASS)
  PPL Δ% > +5%    → KILL PF8.3 (FP8 prefill quant breaks accuracy)

Usage:
  # default 15 wikitext samples × 200 max-tokens
  python3 scripts/eval_ppl_pf83.py

  # explicit
  python3 scripts/eval_ppl_pf83.py \\
      --model infer/models/Qwen3-4B-W4-hybrid-zpfix \\
      --datasets wikitext,humaneval --max-samples 15 --max-tokens 200

NOTE: default model is the HYBRID checkpoint (quant_type=marlin_w4_hybrid)
because PF8 dispatch only activates on hybrid weights per linear.rs:86
hybrid_w4_fp8_aligned() guard. W4A8-only checkpoints will silently
skip the new branch (anti-pattern #29 risk per b551bea + 473081d).
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
DEFAULT_MODEL = "infer/models/Qwen3-4B-W4-hybrid-zpfix"


def load_dataset_texts(name, max_samples=15):
    from datasets import load_dataset

    texts = []
    if name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        for row in ds:
            t = row["text"].strip()
            if len(t) > 100:
                texts.append(t)
                if len(texts) >= max_samples:
                    break
    elif name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        for row in ds:
            t = row["prompt"].strip()
            if len(t) > 50:
                texts.append(t)
                if len(texts) >= max_samples:
                    break
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return texts


def start_server(model_path, env_overrides):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia:/usr/local/cuda/lib64:" + env.get(
        "LD_LIBRARY_PATH", ""
    )
    env.update(env_overrides)
    cmd = [BIN, "--model-path", model_path, "--port", "8090", "--num-slots", "1"]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
    )
    for _ in range(60):
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
    raise RuntimeError(f"Server failed to start (env={env_overrides})")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(1)


def collect_logprobs(prompt, max_tokens=50):
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


def eval_treatment(label, env_overrides, model_path, dataset_texts, max_tokens):
    print(f"\n  Starting {label} server (env={env_overrides})...")
    proc = start_server(model_path, env_overrides)
    all_lps = []
    for i, text in enumerate(dataset_texts):
        lps = collect_logprobs(text, max_tokens)
        all_lps.extend(lps)
        if (i + 1) % 5 == 0 or i == len(dataset_texts) - 1:
            print(
                f"    [{i+1}/{len(dataset_texts)}] {len(all_lps)} tokens, "
                f"running PPL={compute_ppl(all_lps):.4f}"
            )
    stop_server(proc)
    ppl = compute_ppl(all_lps)
    return ppl, len(all_lps)


def license_verdict(delta_pct):
    if delta_pct <= 1.0:
        return "LICENSE  (Δ% ≤ +1.0%)"
    if delta_pct > 5.0:
        return "KILL     (Δ% > +5.0% — FP8 prefill quant breaks accuracy)"
    return "REVIEW   (between +1.0% and +5.0% — need n=3 reproducibility)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--datasets", default="wikitext")
    parser.add_argument("--max-samples", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    treatments = [
        ("baseline_W4_INT8", {"INFER_MARLIN_W4_FP8_PREFILL": "0", "INFER_HYBRID_W4A8_PREFILL": "1"}),
        ("treatment_W4_FP8", {"INFER_MARLIN_W4_FP8_PREFILL": "1", "INFER_HYBRID_W4A8_PREFILL": "1"}),
    ]

    print("Loading datasets...")
    datasets = {}
    for name in dataset_names:
        texts = load_dataset_texts(name, max_samples=args.max_samples)
        datasets[name] = texts
        print(f"  {name}: {len(texts)} samples")

    all_results = {}
    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({len(datasets[ds_name])} samples, {args.max_tokens} tok/sample)")
        print(f"{'='*60}")
        all_results[ds_name] = {}
        for label, env in treatments:
            ppl, n = eval_treatment(label, env, args.model, datasets[ds_name], args.max_tokens)
            all_results[ds_name][label] = (ppl, n)
            print(f"  {label}: PPL={ppl:.4f} ({n} tokens)")

    print(f"\n{'='*70}")
    print(f"PF8.3 PPL Gate Summary (lower PPL = better)")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Baseline INT8':>14} {'Treatment FP8':>14} {'Δ%':>8} {'Verdict':<55}")
    print("-" * 100)
    overall_max_delta = -float("inf")
    for ds_name in dataset_names:
        r = all_results[ds_name]
        bp = r["baseline_W4_INT8"][0]
        fp = r["treatment_W4_FP8"][0]
        delta_pct = ((fp / bp) - 1) * 100
        overall_max_delta = max(overall_max_delta, delta_pct)
        print(
            f"{ds_name:<12} {bp:>14.4f} {fp:>14.4f} {delta_pct:>+7.2f}% "
            f"{license_verdict(delta_pct):<55}"
        )
    print("-" * 100)
    print(f"\nOverall max Δ% across datasets: {overall_max_delta:+.2f}%")
    print(f"Final verdict: {license_verdict(overall_max_delta)}")

    sys.exit(0 if overall_max_delta <= 5.0 else 1)


if __name__ == "__main__":
    main()

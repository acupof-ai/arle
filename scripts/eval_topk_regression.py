#!/usr/bin/env python3
"""
Tier-1 deterministic regression for INFER_MOE_TOP_K (M_e.8).

Boots `metal_serve` once per top_k value, runs N fixed prompts with
temperature=0 max_tokens=256 against /v1/chat/completions, and
diffs the outputs token-for-token. Prints exact-match rate and
mean first-divergence-token-index across all prompt pairs.

Per docs/plans/M_e8-moe-quality-eval-scaffold.md:
  Pass criterion: match_rate >= 0.90 AND mean_first_divergence >= 32

Usage:
  scripts/eval_topk_regression.py \
      --model mlx-community/Qwen3.6-35B-A3B-4bit \
      --top-ks 8,6,4 \
      --max-tokens 256

The greedy-fingerprint pattern is the same idea as
infer/tests/spec_decode_correctness.rs:162-191; see Apple's
Recurrent Drafter §3.2 for the literature precedent
(https://arxiv.org/abs/2403.09919). We're applying the
exact-match metric as a *fingerprint* — top_k=6 is NOT
function-equivalent to top_k=8, so divergence is expected;
the question is the *amount*.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

# ─── prompts ───────────────────────────────────────────────────────────
# 50 deterministic prompts spanning chat/code/reasoning/instruction.
# Edit-as-needed; matched-A/B is over the SAME set across runs.
DEFAULT_PROMPTS = [
    # 0..9 — chat/conversational
    "Explain why the sky appears blue, in one paragraph.",
    "Write a haiku about a thunderstorm.",
    "What's the difference between ML and statistics?",
    "Summarize the plot of Hamlet in three sentences.",
    "Draft a polite reply declining a meeting invitation.",
    "List 5 benefits of regular exercise.",
    "What are the main causes of inflation? Explain briefly.",
    "Recommend a book for someone who liked The Great Gatsby.",
    "How do I check disk usage on Linux?",
    "Define 'machine learning' for a high school student.",
    # 10..19 — code generation
    "Write a Python function to compute the nth Fibonacci number iteratively.",
    "Implement quicksort in Rust.",
    "Write a SQL query to find the top 3 customers by order total.",
    "Convert this JSON to YAML: {\"name\": \"Alice\", \"age\": 30}",
    "Write a regex matching valid email addresses.",
    "Show a Python decorator that times a function call.",
    "Implement a Bash script that lists files larger than 100 MB.",
    "Write a JavaScript function that flattens a nested array.",
    "Show a one-liner to count lines in all .py files recursively.",
    "Write Python code reading a CSV file with pandas, printing first 5 rows.",
    # 20..29 — reasoning / multi-step
    "If a train leaves Boston at 9am going 60 mph and another leaves NYC at 10am going 75 mph, when do they meet?",
    "John has 3 apples. He gives 1 to Mary and buys 5 more. How many does he have?",
    "Sort these numbers ascending: 17, 3, 42, 8, 25, 1, 99, 14",
    "What's 25% of 80? Show your work.",
    "If today is Tuesday, what day will it be in 100 days?",
    "A rectangle has area 24 and perimeter 20. What are its dimensions?",
    "Three friends share $90 in ratio 2:3:5. How much does each get?",
    "If P implies Q and Q implies R, does P imply R? Explain.",
    "Two dice are rolled. What's the probability they sum to 7?",
    "List the prime numbers between 50 and 70.",
    # 30..39 — instruction-following / formatting
    "Output exactly the words: hello world",
    "Reply with ONLY the digit 5.",
    "Translate 'good morning' to Spanish, French, and German.",
    "Output a markdown table with columns: Name, Age, City. Three example rows.",
    "Write a one-sentence definition of recursion.",
    "Continue the sequence: 2, 4, 8, 16, ...",
    "Capitalize each word: the quick brown fox jumps over the lazy dog",
    "Reverse this string: programming",
    "Remove duplicates from this list: [1, 2, 2, 3, 4, 4, 5]",
    "Output the first 10 even numbers, comma-separated.",
    # 40..49 — domain mixing
    "Explain monads to a Python developer in 3 sentences.",
    "What's the time complexity of binary search?",
    "How does a transformer attention head work?",
    "Compare gradient descent and Adam optimizer briefly.",
    "Why does ReLU help with vanishing gradients?",
    "What's a context-free grammar? Give an example.",
    "Describe the OSI model in one sentence per layer.",
    "What is referential transparency?",
    "Explain BFS vs DFS for graph traversal.",
    "What is amortized time complexity? Give an example.",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = str(REPO_ROOT / "target/release/metal_serve")
URL = "http://127.0.0.1:8765"
MODEL_DEFAULT = "mlx-community/Qwen3.6-35B-A3B-4bit"


def start_server(model_path: str, bin_path: str, top_k: Optional[int]) -> subprocess.Popen:
    env = os.environ.copy()
    if top_k is not None:
        env["INFER_MOE_TOP_K"] = str(top_k)
    elif "INFER_MOE_TOP_K" in env:
        del env["INFER_MOE_TOP_K"]
    cmd = [
        bin_path,
        "--model-path",
        model_path,
        "--port",
        "8765",
        "--max-running-requests",
        "16",
    ]
    log_label = f"top_k={top_k}" if top_k is not None else "top_k=DEFAULT"
    print(f"  starting metal_serve ({log_label})…", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    deadline = time.time() + 600  # 10 min for cold MoE load
    while time.time() < deadline:
        try:
            r = httpx.get(f"{URL}/v1/models", timeout=2)
            if r.status_code == 200:
                print(f"  ready ({log_label})", flush=True)
                return proc
        except Exception:
            pass
        time.sleep(2)
    proc.kill()
    raise RuntimeError(f"metal_serve failed to come up for {log_label}")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(1)


def model_id_from_path(path: str) -> str:
    """Mirror metal_serve's id derivation: basename of the model path."""
    return Path(path).name


def run_prompt(model_id: str, prompt: str, max_tokens: int) -> str:
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = httpx.post(f"{URL}/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    payload = r.json()
    return payload["choices"][0]["message"]["content"]


def collect_outputs(model_id: str, prompts: list[str], max_tokens: int) -> list[str]:
    outputs = []
    for i, p in enumerate(prompts):
        out = run_prompt(model_id, p, max_tokens)
        outputs.append(out)
        if (i + 1) % 10 == 0 or i == len(prompts) - 1:
            print(f"    [{i+1}/{len(prompts)}] collected", flush=True)
    return outputs


def first_divergence_chars(a: str, b: str) -> int:
    """Return the index where strings first differ; len(a) if a is prefix of b."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n  # one is a strict prefix of the other; divergence at end of shorter


def main() -> int:
    p = argparse.ArgumentParser(description="Tier-1 deterministic regression for INFER_MOE_TOP_K")
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--bin", default=DEFAULT_BIN)
    p.add_argument(
        "--top-ks",
        default="8,6,4",
        help="Comma-separated top_k values to test (first is the baseline)",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Override the default 50-prompt set (one prompt per line, '#' comments)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/eval_topk_regression.json"),
    )
    args = p.parse_args()

    if args.prompts_file is not None:
        prompts = [
            line.strip()
            for line in args.prompts_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        prompts = DEFAULT_PROMPTS

    top_ks: list[Optional[int]] = []
    for tok in args.top_ks.split(","):
        tok = tok.strip()
        if tok in ("", "default", "DEFAULT"):
            top_ks.append(None)
        else:
            top_ks.append(int(tok))
    if not top_ks:
        print("no top-ks specified", file=sys.stderr)
        return 2

    model_id = model_id_from_path(args.model)

    if not Path(args.bin).exists():
        print(f"missing binary: {args.bin}", file=sys.stderr)
        return 2

    print(f"model: {args.model} (id={model_id})")
    print(f"prompts: {len(prompts)}")
    print(f"top-ks:  {top_ks}")
    print(f"max-tokens: {args.max_tokens}")

    runs: dict[str, list[str]] = {}
    for top_k in top_ks:
        proc = start_server(args.model, args.bin, top_k)
        try:
            outs = collect_outputs(model_id, prompts, args.max_tokens)
        finally:
            stop_server(proc)
        key = "default" if top_k is None else str(top_k)
        runs[key] = outs
        print(f"  done: top_k={key}, {len(outs)} outputs", flush=True)

    # Diff: first key is baseline.
    baseline_key = "default" if top_ks[0] is None else str(top_ks[0])
    baseline = runs[baseline_key]

    print()
    print("=" * 78)
    print(f"Baseline: top_k={baseline_key}")
    print("=" * 78)

    summary = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "n_prompts": len(prompts),
        "baseline_top_k": baseline_key,
        "comparisons": {},
    }

    for k, outs in runs.items():
        if k == baseline_key:
            continue
        n = len(prompts)
        exact_matches = sum(1 for i in range(n) if outs[i] == baseline[i])
        match_rate = exact_matches / n
        # First-divergence-character index per prompt; longer is better.
        divergences = [
            first_divergence_chars(baseline[i], outs[i]) for i in range(n)
        ]
        mean_div = sum(divergences) / len(divergences)
        median_div = sorted(divergences)[len(divergences) // 2]
        # The pass criterion is character-based since we don't tokenize here.
        # For first-token-divergence-≥-32 from the plan, characters ≥ 32 is
        # a conservative proxy (most tokens are 1-4 chars).
        passed = match_rate >= 0.90 and mean_div >= 32 * 4
        summary["comparisons"][k] = {
            "match_rate": match_rate,
            "exact_matches": exact_matches,
            "mean_divergence_chars": mean_div,
            "median_divergence_chars": median_div,
            "passed_proxy": passed,
        }
        print(f"\nvs top_k={k}:")
        print(f"  exact match rate  : {match_rate:.2%} ({exact_matches}/{n})")
        print(f"  mean divergence   : {mean_div:.1f} chars")
        print(f"  median divergence : {median_div} chars")
        print(f"  Tier-1 pass-proxy : {'PASS' if passed else 'FAIL'} "
              "(match_rate≥0.90 AND mean_div≥128 chars; chars≈4× tokens)")

    # Save raw outputs for follow-up Tier-2 (HumanEval/GSM8K accuracy).
    outpath = args.out
    outpath.parent.mkdir(parents=True, exist_ok=True)
    full = {
        **summary,
        "prompts": prompts,
        "runs": runs,
    }
    outpath.write_text(json.dumps(full, indent=2))
    print(f"\nfull results written to {outpath}")

    # Exit non-zero on Tier-1 fail of any non-baseline arm, so this can run
    # in CI as a quick check.
    any_fail = any(not c["passed_proxy"] for c in summary["comparisons"].values())
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())

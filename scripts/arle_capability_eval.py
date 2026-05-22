#!/usr/bin/env python3
"""ARLE capability eval — minimal MMLU + GSM8K harness against an ARLE serve.

Talks to ARLE's OpenAI v1 surface (`/v1/chat/completions` or `/v1/completions`)
exposed by `arle serve` / `infer`. Designed for the capability-eval plan
defined in `docs/plans/2026-05-22-arle-opd-capability-eval-plan.md` P0 phase:

    arle serve --backend cuda --model-path <path> --port 8123 &
    ARLE_BASE_URL=http://localhost:8123 \
      python scripts/arle_capability_eval.py \
        --tasks mmlu,gsm8k \
        --n-samples 200 \
        --output bench-output/<dated-dir>/

Why not lm-evaluation-harness / simple-evals: this harness has zero install
weight (stdlib + `datasets`), prints PASS/FAIL per task in one screen, and
writes a flat JSON the wins/errors entry can quote directly. lm-evaluation-
harness's 50k LOC + heavy deps are overkill for the first capability data
point on a freshly-distilled student. Once we have the baseline triplet,
we can graduate to lm-eval if we need broader task coverage.

Tasks supported (P0 cut):
  - mmlu     — MMLU 5-shot exact-match on the answer letter (A/B/C/D)
  - gsm8k   — GSM8K exact-match on the final numeric answer after `####`

Tasks deferred:
  - ifeval  — requires rule-based verifier set; ship in a later patch
  - hellaswag — needs `echo` logprob support in ARLE HTTP (not yet shipped)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


# ───────────────────────── HTTP client ──────────────────────────


class ArleClient:
    """Minimal OpenAI-v1 client. Uses stdlib `urllib` to avoid a `requests` dep."""

    def __init__(self, base_url: str, model_id: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout

    def chat(self, messages: list[dict], max_tokens: int = 64, temperature: float = 0.0) -> str:
        body = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["message"]["content"]

    def completion(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        body = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["text"]


# ───────────────────────── MMLU ─────────────────────────────────


MMLU_FEW_SHOT_TEMPLATE = (
    "The following are multiple choice questions (with answers) about {subject}.\n\n"
    "{shots}\n"
    "{question}\n"
    "A) {a}\nB) {b}\nC) {c}\nD) {d}\n"
    "Answer:"
)


def _mmlu_format_shot(example: dict) -> str:
    return (
        f"{example['question']}\n"
        f"A) {example['choices'][0]}\n"
        f"B) {example['choices'][1]}\n"
        f"C) {example['choices'][2]}\n"
        f"D) {example['choices'][3]}\n"
        f"Answer: {chr(ord('A') + example['answer'])}\n"
    )


def _mmlu_extract_letter(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    # Match a leading letter A/B/C/D, possibly followed by ) or .
    m = re.match(r"\s*([A-D])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def run_mmlu(client: ArleClient, n_samples: int, output_dir: Path) -> dict:
    try:
        from datasets import load_dataset
    except ImportError:
        return {
            "task": "mmlu",
            "status": "skipped",
            "reason": "datasets package not installed; pip install datasets",
        }

    # MMLU has a `dev` split (5 shots per subject) and `test` split (eval).
    print("[mmlu] loading dataset...", flush=True)
    ds_test = load_dataset("cais/mmlu", "all", split="test")
    ds_dev = load_dataset("cais/mmlu", "all", split="dev")

    # Build per-subject dev pools for the 5-shot prompt.
    dev_by_subject: dict[str, list[dict]] = {}
    for ex in ds_dev:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    # Sample n_samples evenly across subjects for speed.
    subjects = sorted({ex["subject"] for ex in ds_test})
    n_per_subject = max(1, n_samples // len(subjects))
    pool: list[dict] = []
    for subj in subjects:
        subj_pool = [ex for ex in ds_test if ex["subject"] == subj][:n_per_subject]
        pool.extend(subj_pool)
    pool = pool[:n_samples]
    print(f"[mmlu] sampling {len(pool)} questions across {len(subjects)} subjects", flush=True)

    correct = 0
    invalid = 0
    per_subject: dict[str, dict] = {}
    t0 = time.time()
    for i, ex in enumerate(pool):
        subj = ex["subject"]
        shots = "\n".join(_mmlu_format_shot(s) for s in dev_by_subject.get(subj, [])[:5])
        prompt = MMLU_FEW_SHOT_TEMPLATE.format(
            subject=subj.replace("_", " "),
            shots=shots,
            question=ex["question"],
            a=ex["choices"][0],
            b=ex["choices"][1],
            c=ex["choices"][2],
            d=ex["choices"][3],
        )
        try:
            resp = client.completion(prompt, max_tokens=4, temperature=0.0)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"[mmlu] sample {i} request error: {exc}", flush=True)
            invalid += 1
            continue
        letter = _mmlu_extract_letter(resp)
        gold = chr(ord("A") + ex["answer"])
        sub_stat = per_subject.setdefault(subj, {"correct": 0, "total": 0})
        sub_stat["total"] += 1
        if letter is None:
            invalid += 1
        elif letter == gold:
            correct += 1
            sub_stat["correct"] += 1
        if (i + 1) % 50 == 0:
            print(f"[mmlu] {i + 1}/{len(pool)} acc={correct / max(1, i + 1 - invalid):.3f}", flush=True)

    elapsed = time.time() - t0
    scored = len(pool) - invalid
    accuracy = correct / scored if scored else 0.0
    report = {
        "task": "mmlu",
        "status": "ok",
        "n_samples": len(pool),
        "n_scored": scored,
        "n_invalid": invalid,
        "n_correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "per_subject": per_subject,
    }
    (output_dir / "mmlu.json").write_text(json.dumps(report, indent=2))
    print(f"[mmlu] accuracy={accuracy:.3f} ({correct}/{scored}, invalid={invalid}, {elapsed:.1f}s)", flush=True)
    return report


# ───────────────────────── GSM8K ────────────────────────────────


GSM8K_FEW_SHOT = """\
Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast and uses four to bake muffins for her friends. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
A: She has 16 - 3 - 4 = 9 eggs left to sell. She makes 9 * 2 = $18 per day. #### 18

Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: Half of 2 bolts is 1 bolt of white fiber. Total bolts = 2 + 1 = 3. #### 3

Q: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
A: The house was bought for 80000 and increased in value by 150%, so the new value is 80000 + 80000*1.5 = 200000. After repair costs of 50000, his profit was 200000 - 80000 - 50000 = 70000. #### 70000

"""


_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
_GSM8K_LAST_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _gsm8k_gold_answer(answer: str) -> str:
    m = _GSM8K_ANSWER_RE.search(answer)
    if not m:
        return ""
    return m.group(1).replace(",", "")


def _gsm8k_extract_answer(text: str) -> str | None:
    # Prefer the `#### N` marker if the model used it.
    m = _GSM8K_ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    # Fall back to the last number in the response.
    numbers = _GSM8K_LAST_NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def run_gsm8k(client: ArleClient, n_samples: int, output_dir: Path) -> dict:
    try:
        from datasets import load_dataset
    except ImportError:
        return {
            "task": "gsm8k",
            "status": "skipped",
            "reason": "datasets package not installed; pip install datasets",
        }

    print("[gsm8k] loading dataset...", flush=True)
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    pool = list(ds_test.select(range(min(n_samples, len(ds_test)))))
    print(f"[gsm8k] running {len(pool)} problems", flush=True)

    correct = 0
    invalid = 0
    t0 = time.time()
    for i, ex in enumerate(pool):
        prompt = GSM8K_FEW_SHOT + f"Q: {ex['question']}\nA:"
        try:
            resp = client.completion(prompt, max_tokens=256, temperature=0.0)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"[gsm8k] sample {i} request error: {exc}", flush=True)
            invalid += 1
            continue
        gold = _gsm8k_gold_answer(ex["answer"])
        pred = _gsm8k_extract_answer(resp)
        if pred is None:
            invalid += 1
        elif pred == gold:
            correct += 1
        if (i + 1) % 25 == 0:
            print(
                f"[gsm8k] {i + 1}/{len(pool)} acc={correct / max(1, i + 1 - invalid):.3f}",
                flush=True,
            )

    elapsed = time.time() - t0
    scored = len(pool) - invalid
    accuracy = correct / scored if scored else 0.0
    report = {
        "task": "gsm8k",
        "status": "ok",
        "n_samples": len(pool),
        "n_scored": scored,
        "n_invalid": invalid,
        "n_correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "gsm8k.json").write_text(json.dumps(report, indent=2))
    print(f"[gsm8k] accuracy={accuracy:.3f} ({correct}/{scored}, invalid={invalid}, {elapsed:.1f}s)", flush=True)
    return report


# ───────────────────────── CLI ──────────────────────────────────


TASK_RUNNERS = {
    "mmlu": run_mmlu,
    "gsm8k": run_gsm8k,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("ARLE_BASE_URL", "http://localhost:8123"))
    parser.add_argument("--model-id", required=True, help="model id as loaded by arle serve (e.g. Qwen3___5-0___8B-Base)")
    parser.add_argument("--tasks", default="mmlu,gsm8k", help="comma-separated subset of: " + ", ".join(TASK_RUNNERS))
    parser.add_argument("--n-samples", type=int, default=200, help="samples per task")
    parser.add_argument("--output", type=Path, required=True, help="output directory for per-task reports")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    client = ArleClient(args.base_url, args.model_id)

    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TASK_RUNNERS]
    if unknown:
        print(f"unknown tasks: {unknown}. supported: {list(TASK_RUNNERS)}", file=sys.stderr)
        return 2

    summary = {
        "base_url": args.base_url,
        "model_id": args.model_id,
        "tasks": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for task in requested:
        print(f"\n========== {task} ==========", flush=True)
        report = TASK_RUNNERS[task](client, args.n_samples, args.output)
        summary["tasks"][task] = report

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n========== summary ==========", flush=True)
    for task, report in summary["tasks"].items():
        if report["status"] == "ok":
            print(f"  {task}: {report['accuracy']:.3f} ({report['n_correct']}/{report['n_scored']})")
        else:
            print(f"  {task}: {report['status']} — {report.get('reason', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

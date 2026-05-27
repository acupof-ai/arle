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
  - gsm8k   — GSM8K 8-shot exact-match on the final numeric answer after `####`

Tasks deferred:
  - ifeval  — requires rule-based verifier set; ship in a later patch
  - hellaswag — needs `echo` logprob support in ARLE HTTP (not yet shipped)
"""

from __future__ import annotations

import argparse
import json
import os
import random
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


class HfTransformersClient:
    """PyTorch / HuggingFace transformers client.

    Loads a HF model directory directly and runs `model.generate()`. Same
    interface as `ArleClient` so the eval task runners don't care which
    backend they're scoring. Lets us cross-validate ARLE's serve numbers
    against the PyTorch-ecosystem reference (transformers + Qwen Auto*
    classes) for the same model checkpoint, on identical prompts.

    Cost: loads the full model into VRAM on first call. Use for the
    baseline triplet eval; don't mix with an ARLE serve process on the
    same GPU unless there's free headroom.
    """

    def __init__(self, model_path: str, model_id: str | None = None, dtype: str = "bfloat16"):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HfTransformersClient requires `torch` + `transformers`. "
                "Install with `pip install torch transformers accelerate`."
            ) from exc
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.model_path = model_path
        self.model_id = model_id or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
        self.model.eval()
        self._torch = torch
        self._device = next(self.model.parameters()).device

    def completion(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        torch = self._torch
        ids = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, ids["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def chat(self, messages: list[dict], max_tokens: int = 64, temperature: float = 0.0) -> str:
        # Apply the tokenizer's chat template if available; fall back to
        # naive concatenation for base models without one.
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except (ValueError, KeyError):
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return self.completion(prompt, max_tokens=max_tokens, temperature=temperature)


def build_client(backend: str, *, base_url: str | None, model_id: str | None, model_path: str | None, dtype: str = "bfloat16"):
    """Factory shared by CLI and tests so the eval-driver code stays backend-agnostic."""
    if backend == "arle":
        if not (base_url and model_id):
            raise ValueError("--backend arle requires --base-url and --model-id")
        return ArleClient(base_url=base_url, model_id=model_id)
    if backend == "hf":
        if not model_path:
            raise ValueError("--backend hf requires --model-path")
        return HfTransformersClient(model_path=model_path, model_id=model_id, dtype=dtype)
    raise ValueError(f"unknown backend {backend!r}; pick arle | hf")


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
    """Extract the model's MMLU answer letter A/B/C/D.

    Base (non-instruct) models output a variety of shapes after `Answer:`:
        " A"                    → easy
        " A)"                   → leading letter
        " (A)"                  → parenthesized
        " The answer is A."     → embedded
        " A. <reasoning>"       → leading + reasoning
        " <empty>"              → none

    The strategy is layered: try the cleanest leading-letter match first,
    then progressively-noisier patterns. Returns None only when the
    response truly contains no decipherable letter in the first chunk.
    """
    if not text:
        return None
    text = text.strip()
    if not text:
        return None

    # Layer 1: leading letter, possibly with trailing punctuation.
    #   "A", "A)", "A.", "A:"
    m = re.match(r"([A-D])(?:[\)\.\:,;]|$|\s)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Layer 2: leading parenthesized letter "(A)".
    m = re.match(r"\(([A-D])\)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Layer 3: "answer is X" / "answer: X" / "is X" embedded in early text.
    early = text[:200]
    m = re.search(
        r"\b(?:answer\s*(?:is|:)|correct\s*(?:is|:)|option)\s*\(?\s*([A-D])\b",
        early,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Layer 4: any standalone letter A-D in the first 60 chars as a last
    # resort. Only fires when no clearer signal was found above.
    short = text[:60]
    m = re.search(r"\b([A-D])\b", short, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def run_mmlu(
    client: ArleClient,
    n_samples: int,
    output_dir: Path,
    debug_samples: int = 5,
    seed: int | None = None,
) -> dict:
    """Run MMLU 5-shot eval. Saves the first `debug_samples` raw responses
    to <output_dir>/mmlu_debug.json so future extractor fixes can target
    real model output instead of guessed prompt-shape.

    When `seed` is set, each per-subject pool is shuffled before subject-
    balanced subsampling so multiple runs at the same `n_samples` produce
    independent draws — needed for binomial-noise variance estimates at
    the small-n eval shapes used today."""
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
        subj_pool = [ex for ex in ds_test if ex["subject"] == subj]
        if seed is not None:
            random.Random(f"mmlu-{seed}-{subj}").shuffle(subj_pool)
        pool.extend(subj_pool[:n_per_subject])
    pool = pool[:n_samples]
    seed_tag = f" seed={seed}" if seed is not None else ""
    print(f"[mmlu] sampling {len(pool)} questions across {len(subjects)} subjects{seed_tag}", flush=True)

    correct = 0
    invalid = 0
    per_subject: dict[str, dict] = {}
    debug_records: list[dict] = []
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
        # Base models often need a few tokens to commit to a letter when
        # leading whitespace / paren / "The answer is" pattern lands.
        # 32 tokens covers all those shapes and is still fast.
        try:
            resp = client.completion(prompt, max_tokens=32, temperature=0.0)
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
        if i < debug_samples:
            debug_records.append(
                {
                    "i": i,
                    "subject": subj,
                    "gold": gold,
                    "extracted": letter,
                    "response_first_200": resp[:200],
                }
            )
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
        "seed": seed,
        "per_subject": per_subject,
    }
    (output_dir / "mmlu.json").write_text(json.dumps(report, indent=2))
    if debug_records:
        (output_dir / "mmlu_debug.json").write_text(json.dumps(debug_records, indent=2))
    print(f"[mmlu] accuracy={accuracy:.3f} ({correct}/{scored}, invalid={invalid}, {elapsed:.1f}s)", flush=True)
    return report


# ───────────────────────── GSM8K ────────────────────────────────


_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
_GSM8K_LAST_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _gsm8k_format_shot(example: dict) -> str:
    return f"Q: {example['question']}\nA: {example['answer']}\n"


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


def run_gsm8k(
    client: ArleClient,
    n_samples: int,
    output_dir: Path,
    debug_samples: int = 5,
    n_shots: int = 8,
    seed: int | None = None,
) -> dict:
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
    ds_train = load_dataset("openai/gsm8k", "main", split="train") if n_shots > 0 else []
    shot_examples = list(ds_train.select(range(min(n_shots, len(ds_train))))) if n_shots > 0 else []
    few_shot = "\n".join(_gsm8k_format_shot(ex) for ex in shot_examples)
    if few_shot:
        few_shot += "\n"
    indices = list(range(len(ds_test)))
    if seed is not None:
        random.Random(f"gsm8k-{seed}").shuffle(indices)
    pool = [ds_test[i] for i in indices[: min(n_samples, len(ds_test))]]
    seed_tag = f" seed={seed}" if seed is not None else ""
    print(f"[gsm8k] running {len(pool)} problems with {len(shot_examples)} shots{seed_tag}", flush=True)

    correct = 0
    invalid = 0
    debug_records: list[dict] = []
    t0 = time.time()
    for i, ex in enumerate(pool):
        prompt = few_shot + f"Q: {ex['question']}\nA:"
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
        if i < debug_samples:
            debug_records.append(
                {
                    "i": i,
                    "gold": gold,
                    "extracted": pred,
                    "response_first_300": resp[:300],
                }
            )
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
        "n_shots": len(shot_examples),
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "seed": seed,
    }
    (output_dir / "gsm8k.json").write_text(json.dumps(report, indent=2))
    if debug_records:
        (output_dir / "gsm8k_debug.json").write_text(json.dumps(debug_records, indent=2))
    print(f"[gsm8k] accuracy={accuracy:.3f} ({correct}/{scored}, invalid={invalid}, {elapsed:.1f}s)", flush=True)
    return report


# ───────────────────────── CLI ──────────────────────────────────


TASK_RUNNERS = {
    "mmlu": run_mmlu,
    "gsm8k": run_gsm8k,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["arle", "hf"], default="arle",
                        help="arle = OpenAI-v1 HTTP to arle serve; hf = transformers in-process")
    parser.add_argument("--base-url", default=os.environ.get("ARLE_BASE_URL", "http://localhost:8123"),
                        help="--backend arle only")
    parser.add_argument("--model-id", required=True,
                        help="logical name; for arle = served id (e.g. Qwen3___5-0___8B-Base); "
                             "for hf = same id (used in report labelling)")
    parser.add_argument("--model-path", default=None,
                        help="--backend hf only: HF model directory (or HF repo id)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="--backend hf only")
    parser.add_argument("--tasks", default="mmlu,gsm8k", help="comma-separated subset of: " + ", ".join(TASK_RUNNERS))
    parser.add_argument("--n-samples", type=int, default=200, help="samples per task")
    parser.add_argument("--gsm8k-shots", type=int, default=8, help="few-shot examples for GSM8K")
    parser.add_argument("--seed", type=int, default=None,
                        help="if set, shuffle per-subject MMLU pool and GSM8K test pool before "
                             "sampling. Distinct seeds give independent draws for variance estimation. "
                             "Default unset = original deterministic ordering, reproduces older runs.")
    parser.add_argument("--output", type=Path, required=True, help="output directory for per-task reports")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    client = build_client(args.backend, base_url=args.base_url, model_id=args.model_id,
                          model_path=args.model_path, dtype=args.dtype)

    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TASK_RUNNERS]
    if unknown:
        print(f"unknown tasks: {unknown}. supported: {list(TASK_RUNNERS)}", file=sys.stderr)
        return 2

    summary = {
        "backend": args.backend,
        "base_url": args.base_url if args.backend == "arle" else None,
        "model_path": args.model_path if args.backend == "hf" else None,
        "model_id": args.model_id,
        "gsm8k_shots": args.gsm8k_shots,
        "seed": args.seed,
        "tasks": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for task in requested:
        print(f"\n========== {task} ==========", flush=True)
        if task == "gsm8k":
            report = run_gsm8k(client, args.n_samples, args.output, n_shots=args.gsm8k_shots, seed=args.seed)
        else:
            report = TASK_RUNNERS[task](client, args.n_samples, args.output, seed=args.seed)
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

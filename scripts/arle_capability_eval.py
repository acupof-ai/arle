#!/usr/bin/env python3
"""ARLE capability eval — minimal capability harness against an ARLE serve.

Talks to ARLE's OpenAI v1 surface (`/v1/chat/completions` or `/v1/completions`)
exposed by `arle serve` / `infer`. Designed for the capability-eval P0 phase:

    arle serve --backend cuda --model-path <path> --port 8123 &
    ARLE_BASE_URL=http://localhost:8123 \
      python scripts/arle_capability_eval.py \
        --tasks mmlu,gsm8k,math500 \
        --n-samples 200 \
        --output bench-output/<dated-dir>/

Why not lm-evaluation-harness / simple-evals: this harness has zero install
weight (stdlib + `datasets`), prints PASS/FAIL per task in one screen, and
writes a flat JSON the wins/errors entry can quote directly. lm-evaluation-
harness's 50k LOC + heavy deps are overkill for the first capability data
point on a freshly-distilled student. Once we have the baseline triplet,
we can graduate to lm-eval if we need broader task coverage.

Tasks supported:
  - mmlu     — MMLU 5-shot exact-match on the answer letter (A/B/C/D)
  - gsm8k   — GSM8K 8-shot exact-match on the final numeric answer after `####`
  - math500  — MATH-500 greedy exact-match on the final boxed answer

Tasks deferred:
  - ifeval  — requires rule-based verifier set; ship in a later patch
  - hellaswag — needs `echo` logprob support in ARLE HTTP (not yet shipped)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _issue_many(call, items, concurrency):
    """Run `call(item)` over `items`, `concurrency` at a time, in input order.

    The serve is a continuous-batching engine, so one-request-at-a-time leaves
    every slot but one idle: a 500-problem GSM8K seed took 80 min at 1-way.
    Exceptions are returned in place so each caller keeps its own per-item
    error handling. Paired comparisons must use the SAME concurrency on both
    arms — batch composition perturbs MoE numerics.
    """
    def guarded(item):
        try:
            return call(item)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            return exc

    if concurrency <= 1:
        return [guarded(item) for item in items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(guarded, items))


SCRIPT_SCHEMA_VERSION = "arle-capability-eval-v2"
DEFAULT_DATASET_REVISION = "main"


FINAL_RESULT_CONTRACT = {
    "candidate_source": "ARLE inference output",
    "scoring_policy": "deterministic extractor + exact-match grader",
    "no_model_judge": True,
    "grader_may_not_rewrite_candidate": True,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _fingerprint_records(records: list[dict]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(records)).hexdigest()


def _parse_seed_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    seeds: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates")
    return seeds


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ───────────────────────── HTTP client ──────────────────────────


class ArleClient:
    """Minimal OpenAI-v1 client. Uses stdlib `urllib` to avoid a `requests` dep."""

    def __init__(self, base_url: str, model_id: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout

    def _v1_url(self, suffix: str) -> str:
        suffix = suffix.lstrip("/")
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/{suffix}"
        return f"{self.base_url}/v1/{suffix}"

    def chat(self, messages: list[dict], max_tokens: int = 64, temperature: float = 0.0) -> str:
        body = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            self._v1_url("chat/completions"),
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
            self._v1_url("completions"),
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


def build_client(
    backend: str,
    *,
    base_url: str | None,
    model_id: str | None,
    model_path: str | None,
    dtype: str = "bfloat16",
    request_timeout: float = 120.0,
):
    """Factory shared by CLI and tests so the eval-driver code stays backend-agnostic."""
    if backend == "arle":
        if not (base_url and model_id):
            raise ValueError("--backend arle requires --base-url and --model-id")
        return ArleClient(base_url=base_url, model_id=model_id, timeout=request_timeout)
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
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    max_tokens: int = 32,
    concurrency: int = 1,
    local_dir: str | None = None,
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
    if local_dir is not None:
        ds_test = load_dataset("parquet", data_files=f"{local_dir}/all/test-*.parquet", split="train")
        ds_dev = load_dataset("parquet", data_files=f"{local_dir}/all/dev-*.parquet", split="train")
    else:
        ds_test = load_dataset("cais/mmlu", "all", split="test", revision=dataset_revision)
        ds_dev = load_dataset("cais/mmlu", "all", split="dev", revision=dataset_revision)

    # Build per-subject dev pools for the 5-shot prompt.
    dev_by_subject: dict[str, list[dict]] = {}
    for ex in ds_dev:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    # Sample n_samples evenly across subjects for speed. Keep the actual pool
    # at the requested size when possible: floor-only allocation under-ran the
    # default 200 as 57 subjects * 3 = 171.
    subjects = sorted({ex["subject"] for ex in ds_test})
    test_by_subject: dict[str, list[dict]] = {subj: [] for subj in subjects}
    for ex in ds_test:
        test_by_subject[ex["subject"]].append(ex)
    subject_order = subjects.copy()
    if seed is not None:
        random.Random(f"mmlu-{seed}-subjects").shuffle(subject_order)
    base_quota, extra = divmod(max(n_samples, 0), len(subject_order) or 1)
    pool: list[dict] = []
    used_by_subject: dict[str, int] = {}
    for index, subj in enumerate(subject_order):
        quota = base_quota + (1 if index < extra else 0)
        if quota == 0:
            used_by_subject[subj] = 0
            continue
        subj_pool = list(test_by_subject[subj])
        if seed is not None:
            random.Random(f"mmlu-{seed}-{subj}").shuffle(subj_pool)
        take = min(quota, len(subj_pool))
        used_by_subject[subj] = take
        pool.extend(subj_pool[:take])
    if len(pool) < n_samples:
        for subj in subject_order:
            subj_pool = list(test_by_subject[subj])
            if seed is not None:
                random.Random(f"mmlu-{seed}-{subj}").shuffle(subj_pool)
            start = used_by_subject.get(subj, 0)
            need = n_samples - len(pool)
            if need <= 0:
                break
            pool.extend(subj_pool[start : start + need])
    pool = pool[: max(n_samples, 0)]
    sample_fingerprint = _fingerprint_records([
        {
            "subject": ex["subject"],
            "question": ex["question"],
            "choices": list(ex["choices"]),
            "answer": int(ex["answer"]),
        }
        for ex in pool
    ])
    seed_tag = f" seed={seed}" if seed is not None else ""
    print(f"[mmlu] sampling {len(pool)} questions across {len(subjects)} subjects{seed_tag}", flush=True)

    correct = 0
    invalid = 0
    per_subject: dict[str, dict] = {}
    debug_records: list[dict] = []
    invalid_records: list[dict] = []
    # Per-question outcomes for paired McNemar-style analysis at the
    # question level (n=~145 paired per seed vs n=5 paired across seeds —
    # much tighter SE when comparing two models on the same seed).
    per_question: list[dict] = []
    t0 = time.time()
    prompts = [
        MMLU_FEW_SHOT_TEMPLATE.format(
            subject=ex["subject"].replace("_", " "),
            shots="\n".join(
                _mmlu_format_shot(s) for s in dev_by_subject.get(ex["subject"], [])[:5]
            ),
            question=ex["question"],
            a=ex["choices"][0],
            b=ex["choices"][1],
            c=ex["choices"][2],
            d=ex["choices"][3],
        )
        for ex in pool
    ]
    # Base models often need a few tokens to commit to a letter when leading
    # whitespace / paren / "The answer is" pattern lands. 32 tokens covers all
    # those shapes and is still fast (per-task budget, see --mmlu-max-tokens).
    responses = _issue_many(
        lambda prompt: client.completion(prompt, max_tokens=max_tokens, temperature=0.0),
        prompts,
        concurrency,
    )
    for i, (ex, resp) in enumerate(zip(pool, responses)):
        subj = ex["subject"]
        if isinstance(resp, Exception):
            print(f"[mmlu] sample {i} request error: {resp}", flush=True)
            invalid += 1
            invalid_records.append({"i": i, "subject": subj, "kind": "request_error", "reason": str(resp)})
            per_question.append({"i": i, "subject": subj, "gold": None, "predicted": None,
                                 "status": "request_error"})
            continue
        letter = _mmlu_extract_letter(resp)
        gold = chr(ord("A") + ex["answer"])
        sub_stat = per_subject.setdefault(subj, {"correct": 0, "total": 0})
        sub_stat["total"] += 1
        status = "scored"
        if letter is None:
            invalid += 1
            status = "extract_fail"
            # Save every extractor-fail response so future extractor patches
            # can target empirical failure modes (not guessed shapes).
            invalid_records.append({
                "i": i,
                "subject": subj,
                "gold": gold,
                "kind": "extract_fail",
                "response": resp[:300],
            })
        elif letter == gold:
            correct += 1
            sub_stat["correct"] += 1
        per_question.append({
            "i": i,
            "subject": subj,
            "gold": gold,
            "predicted": letter,
            "correct": letter == gold,
            "status": status,
        })
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
    ci95 = _wilson_ci(correct, scored)
    report = {
        "task": "mmlu",
        "status": "ok",
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "final_result_contract": FINAL_RESULT_CONTRACT,
        "dataset": {
            "name": "cais/mmlu",
            "config": "all",
            "split": "test",
            "revision": dataset_revision,
            "sample_fingerprint": sample_fingerprint,
        },
        "prompting": {
            "shots": 5,
            "answer_format": "A/B/C/D letter extracted from ARLE completion text",
        },
        "n_samples": len(pool),
        "n_scored": scored,
        "n_invalid": invalid,
        "n_correct": correct,
        "accuracy": accuracy,
        "ci95": list(ci95),
        "elapsed_seconds": elapsed,
        "seed": seed,
        "per_subject": per_subject,
    }
    (output_dir / "mmlu.json").write_text(json.dumps(report, indent=2))
    if debug_records:
        (output_dir / "mmlu_debug.json").write_text(json.dumps(debug_records, indent=2))
    if invalid_records:
        (output_dir / "mmlu_invalid.json").write_text(json.dumps(invalid_records, indent=2))
    (output_dir / "mmlu_perquestion.json").write_text(json.dumps(per_question, indent=2))
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


def _gsm8k_difficulty(answer: str) -> int:
    """Arithmetic steps in the gold solution — GSM8K's own `<<expr=val>>`
    calculator annotations. Deterministic, needs no external labels, and is the
    difficulty axis the sampler stratifies on."""
    return answer.count("<<")


def _gsm8k_extract_answer(text: str) -> str | None:
    # Thinking models emit the whole chain-of-thought in a <think>...</think>
    # block, then the final answer after it. The block is full of arithmetic
    # (48/2 = 24, 48+24 = 72, ...), so scoring must look only *after* the
    # closing tag — otherwise the last-number fallback grabs a scratch value.
    close = text.rfind("</think>")
    if close != -1:
        text = text[close + len("</think>"):]
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
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    max_tokens: int = 2048,
    concurrency: int = 1,
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
    ds_test = load_dataset("openai/gsm8k", "main", split="test", revision=dataset_revision)
    ds_train = load_dataset("openai/gsm8k", "main", split="train", revision=dataset_revision) if n_shots > 0 else []
    shot_examples = list(ds_train.select(range(min(n_shots, len(ds_train))))) if n_shots > 0 else []
    few_shot = "\n".join(_gsm8k_format_shot(ex) for ex in shot_examples)
    if few_shot:
        few_shot += "\n"
    # Stratify on difficulty (gold-solution step count) instead of sampling the
    # test set uniformly: same estimand as the full 1319 but a tighter one at the
    # same n, and it makes the per-difficulty breakdown below free. Proportional,
    # NOT equal-per-bucket — equal quotas would over-weight the rare deep items
    # and the aggregate would stop being comparable to earlier runs.
    want = min(max(n_samples, 0), len(ds_test))
    by_steps: dict[int, list[int]] = {}
    for i, ex in enumerate(ds_test):
        by_steps.setdefault(_gsm8k_difficulty(ex["answer"]), []).append(i)
    indices: list[int] = []
    # Largest-remainder allocation, so the pool lands exactly on `want`.
    quotas = {
        steps: want * len(bucket) // len(ds_test) for steps, bucket in by_steps.items()
    }
    remainders = sorted(
        by_steps,
        key=lambda s: (-((want * len(by_steps[s])) % len(ds_test)), s),
    )
    for steps in remainders[: want - sum(quotas.values())]:
        quotas[steps] += 1
    for steps in sorted(by_steps):
        bucket = list(by_steps[steps])
        if seed is not None:
            random.Random(f"gsm8k-{seed}-{steps}").shuffle(bucket)
        indices.extend(bucket[: quotas[steps]])
    if seed is not None:
        random.Random(f"gsm8k-{seed}").shuffle(indices)
    pool = [ds_test[i] for i in indices]
    sample_fingerprint = _fingerprint_records([
        {
            "question": ex["question"],
            "answer": ex["answer"],
        }
        for ex in pool
    ])
    seed_tag = f" seed={seed}" if seed is not None else ""
    print(f"[gsm8k] running {len(pool)} problems with {len(shot_examples)} shots{seed_tag}", flush=True)

    correct = 0
    invalid = 0
    debug_records: list[dict] = []
    per_question: list[dict] = []
    t0 = time.time()
    # Thinking models collapse on completion-endpoint few-shot (they emit 1
    # token and stop when they see prior `#### N` exemplars). The chat endpoint
    # drives their trained reason-then-answer mode; a zero-shot instruction to
    # end with `#### <number>` gives a clean extractable tail.
    instructions = [
        f"{ex['question']}\n\n"
        "Solve step by step. End your response with the final numeric "
        "answer on its own, preceded by ####."
        for ex in pool
    ]
    responses = _issue_many(
        lambda instruction: client.chat(
            [{"role": "user", "content": instruction}],
            max_tokens=max_tokens, temperature=0.0,
        ),
        instructions,
        concurrency,
    )
    for i, (ex, resp) in enumerate(zip(pool, responses)):
        if isinstance(resp, Exception):
            print(f"[gsm8k] sample {i} request error: {resp}", flush=True)
            invalid += 1
            per_question.append({"i": i, "gold": None, "predicted": None, "status": "request_error"})
            continue
        gold = _gsm8k_gold_answer(ex["answer"])
        pred = _gsm8k_extract_answer(resp)
        status = "scored"
        if pred is None:
            invalid += 1
            status = "extract_fail"
        elif pred == gold:
            correct += 1
        per_question.append({
            "i": i,
            "steps": _gsm8k_difficulty(ex["answer"]),
            "gold": gold,
            "predicted": pred,
            "correct": pred == gold if pred is not None else False,
            "status": status,
        })
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
    ci95 = _wilson_ci(correct, scored)
    # Per-difficulty accuracy: an OPD delta shows up in the deep-step buckets
    # first, and the aggregate hides that.
    by_difficulty: dict[str, dict] = {}
    for rec in per_question:
        bucket = by_difficulty.setdefault(
            str(rec.get("steps", -1)), {"n": 0, "scored": 0, "correct": 0}
        )
        bucket["n"] += 1
        if rec["status"] == "scored":
            bucket["scored"] += 1
            bucket["correct"] += int(rec["correct"])
    for bucket in by_difficulty.values():
        bucket["accuracy"] = (
            bucket["correct"] / bucket["scored"] if bucket["scored"] else 0.0
        )
    report = {
        "task": "gsm8k",
        "status": "ok",
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "final_result_contract": FINAL_RESULT_CONTRACT,
        "dataset": {
            "name": "openai/gsm8k",
            "config": "main",
            "split": "test",
            "revision": dataset_revision,
            "sample_fingerprint": sample_fingerprint,
        },
        "prompting": {
            "shots": len(shot_examples),
            "answer_format": "numeric answer extracted from ARLE completion text",
        },
        "n_samples": len(pool),
        "n_scored": scored,
        "n_invalid": invalid,
        "n_correct": correct,
        "n_shots": len(shot_examples),
        "accuracy": accuracy,
        "ci95": list(ci95),
        "sampling": "difficulty-stratified (proportional, gold-solution step count)",
        "by_difficulty": by_difficulty,
        "elapsed_seconds": elapsed,
        "seed": seed,
    }
    (output_dir / "gsm8k.json").write_text(json.dumps(report, indent=2))
    if debug_records:
        (output_dir / "gsm8k_debug.json").write_text(json.dumps(debug_records, indent=2))
    (output_dir / "gsm8k_perquestion.json").write_text(json.dumps(per_question, indent=2))
    print(f"[gsm8k] accuracy={accuracy:.3f} ({correct}/{scored}, invalid={invalid}, {elapsed:.1f}s)", flush=True)
    return report


# ───────────────────────── MATH-500 ──────────────────────────────


MATH500_PROMPT_TEMPLATE = (
    "Problem:\n{problem}\n\n"
    "Solve the problem. Put the final answer in \\boxed{{}}. "
    "Do not write anything after the boxed answer.\n"
    "Solution:\n"
)


def _extract_last_braced(text: str, marker: str) -> str | None:
    last: str | None = None
    start = 0
    while True:
        pos = text.find(marker, start)
        if pos < 0:
            return last
        i = pos + len(marker)
        depth = 1
        out: list[str] = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = "".join(out).strip()
                    if candidate:
                        last = candidate
                    break
            out.append(ch)
            i += 1
        start = pos + len(marker)


def _math_normalize_answer(answer: str) -> str:
    s = answer.strip()
    boxed = _extract_last_braced(s, "\\boxed{")
    if boxed is not None:
        s = boxed
    s = s.replace("\\$", "").strip("$")
    for old, new in (
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\:", ""),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
    ):
        s = s.replace(old, new)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s.lower()


def _math_gold_answer(example: dict) -> str:
    raw = str(example.get("answer") or "")
    if raw:
        return _math_normalize_answer(raw)
    boxed = _extract_last_braced(str(example.get("solution") or ""), "\\boxed{")
    return _math_normalize_answer(boxed or "")


def _math_extract_answer(text: str) -> str | None:
    boxed = _extract_last_braced(text, "\\boxed{")
    if boxed:
        return _math_normalize_answer(boxed)
    m = re.search(r"####\s*(.+)", text)
    if m:
        return _math_normalize_answer(m.group(1).splitlines()[0])
    for pattern in (
        r"final answer is\s*:?\s*(.+)",
        r"answer is\s*:?\s*(.+)",
        r"therefore\s*,?\s*(.+)",
    ):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return _math_normalize_answer(matches[-1].splitlines()[0])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return _math_normalize_answer(lines[-1])


def run_math500(
    client: ArleClient,
    n_samples: int,
    output_dir: Path,
    debug_samples: int = 5,
    seed: int | None = None,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    max_tokens: int = 4096,
    local_jsonl: Path | None = None,
) -> dict:
    print("[math500] loading dataset...", flush=True)
    if local_jsonl is not None:
        ds_test = [
            json.loads(line)
            for line in local_jsonl.read_text().splitlines()
            if line.strip()
        ]
        dataset_name = str(local_jsonl)
        split = "jsonl"
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            return {
                "task": "math500",
                "status": "skipped",
                "reason": "datasets package not installed; pip install datasets",
            }
        ds_test = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=dataset_revision)
        dataset_name = "HuggingFaceH4/MATH-500"
        split = "test"
    indices = list(range(len(ds_test)))
    if seed is not None:
        random.Random(f"math500-{seed}").shuffle(indices)
    pool = [ds_test[i] for i in indices[: min(n_samples, len(ds_test))]]
    sample_fingerprint = _fingerprint_records([
        {
            "problem": ex.get("problem"),
            "answer": ex.get("answer"),
            "solution": ex.get("solution"),
        }
        for ex in pool
    ])
    seed_tag = f" seed={seed}" if seed is not None else ""
    print(f"[math500] running {len(pool)} problems{seed_tag}", flush=True)

    correct = 0
    invalid = 0
    debug_records: list[dict] = []
    per_question: list[dict] = []
    t0 = time.time()
    for i, ex in enumerate(pool):
        prompt = MATH500_PROMPT_TEMPLATE.format(problem=ex["problem"])
        try:
            resp = client.completion(prompt, max_tokens=max_tokens, temperature=0.0)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"[math500] sample {i} request error: {exc}", flush=True)
            invalid += 1
            per_question.append({"i": i, "gold": None, "predicted": None, "status": "request_error"})
            continue
        gold = _math_gold_answer(ex)
        pred = _math_extract_answer(resp)
        status = "scored"
        if pred is None or not gold:
            invalid += 1
            status = "extract_fail"
        elif pred == gold:
            correct += 1
        per_question.append({
            "i": i,
            "gold": gold,
            "predicted": pred,
            "correct": pred == gold if pred is not None else False,
            "status": status,
            "response": resp,
        })
        if i < debug_samples:
            debug_records.append(
                {
                    "i": i,
                    "gold": gold,
                    "extracted": pred,
                    "response_first_500": resp[:500],
                }
            )
        if (i + 1) % 25 == 0:
            print(
                f"[math500] {i + 1}/{len(pool)} acc={correct / max(1, i + 1 - invalid):.3f}",
                flush=True,
            )

    elapsed = time.time() - t0
    scored = len(pool) - invalid
    accuracy = correct / scored if scored else 0.0
    ci95 = _wilson_ci(correct, scored)
    report = {
        "task": "math500",
        "status": "ok",
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "final_result_contract": FINAL_RESULT_CONTRACT,
        "dataset": {
            "name": dataset_name,
            "split": split,
            "revision": dataset_revision,
            "sample_fingerprint": sample_fingerprint,
        },
        "prompting": {
            "shots": 0,
            "answer_format": "final answer normalized from \\boxed{} or fallback final-answer text",
        },
        "n_samples": len(pool),
        "n_scored": scored,
        "n_invalid": invalid,
        "n_correct": correct,
        "accuracy": accuracy,
        "ci95": list(ci95),
        "elapsed_seconds": elapsed,
        "seed": seed,
    }
    (output_dir / "math500.json").write_text(json.dumps(report, indent=2))
    if debug_records:
        (output_dir / "math500_debug.json").write_text(json.dumps(debug_records, indent=2))
    (output_dir / "math500_perquestion.json").write_text(json.dumps(per_question, indent=2))
    print(f"[math500] accuracy={accuracy:.3f} ({correct}/{scored}, invalid={invalid}, {elapsed:.1f}s)", flush=True)
    return report


# ───────────────────────── CLI ──────────────────────────────────


TASK_RUNNERS = {
    "mmlu": run_mmlu,
    "gsm8k": run_gsm8k,
    "math500": run_math500,
}


def _run_suite_once(args: argparse.Namespace, client, requested: list[str], output_dir: Path, seed: int | None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "final_result_contract": FINAL_RESULT_CONTRACT,
        "backend": args.backend,
        "base_url": args.base_url if args.backend == "arle" else None,
        "model_path": args.model_path if args.backend == "hf" else None,
        "model_id": args.model_id,
        "gsm8k_shots": args.gsm8k_shots,
        "seed": seed,
        "dataset_revisions": {
            "mmlu": args.mmlu_revision,
            "gsm8k": args.gsm8k_revision,
            "math500": args.math500_revision,
        },
        "tasks": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for task in requested:
        print(f"\n========== {task} ==========", flush=True)
        if task == "gsm8k":
            report = run_gsm8k(
                client,
                args.n_samples,
                output_dir,
                n_shots=args.gsm8k_shots,
                seed=seed,
                dataset_revision=args.gsm8k_revision,
                max_tokens=args.gsm8k_max_tokens,
                concurrency=args.concurrency,
            )
        elif task == "mmlu":
            report = run_mmlu(
                client,
                args.n_samples,
                output_dir,
                seed=seed,
                dataset_revision=args.mmlu_revision,
                max_tokens=args.mmlu_max_tokens,
                concurrency=args.concurrency,
                local_dir=args.mmlu_local_dir,
            )
        elif task == "math500":
            report = run_math500(
                client,
                args.n_samples,
                output_dir,
                seed=seed,
                dataset_revision=args.math500_revision,
                max_tokens=args.math_max_tokens,
                local_jsonl=args.math500_jsonl,
            )
        else:
            report = TASK_RUNNERS[task](client, args.n_samples, output_dir, seed=seed)
        summary["tasks"][task] = report

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _print_summary(summary: dict) -> None:
    print("\n========== summary ==========", flush=True)
    for task, report in summary["tasks"].items():
        if report["status"] == "ok":
            ci = report.get("ci95")
            suffix = f" CI95=[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            print(f"  {task}: {report['accuracy']:.3f} ({report['n_correct']}/{report['n_scored']}){suffix}")
        else:
            print(f"  {task}: {report['status']} - {report.get('reason', '')}")


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
    parser.add_argument("--request-timeout", type=float, default=1800.0,
                        help="HTTP request timeout in seconds for --backend arle")
    parser.add_argument("--tasks", default="mmlu,gsm8k", help="comma-separated subset of: " + ", ".join(TASK_RUNNERS))
    parser.add_argument("--n-samples", type=int, default=200, help="samples per task")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="in-flight requests against the serve (default 1 = serial). "
                             "The serve batches continuously, so 1 leaves every slot but one "
                             "idle; 8 turns an 80-min GSM8K seed into ~10 min. Paired "
                             "comparisons must use the SAME value on both arms.")
    parser.add_argument("--gsm8k-shots", type=int, default=8, help="few-shot examples for GSM8K")
    parser.add_argument("--gsm8k-max-tokens", type=int, default=2048,
                        help="generation budget for GSM8K (reasoning). 2048 covers base-model CoT; "
                             "raise to >=10K for thinking-mode models with long reasoning traces. "
                             "The old hardcoded 256 truncated longer chains before the #### answer.")
    parser.add_argument("--math-max-tokens", type=int, default=4096,
                        help="generation budget for MATH-500 greedy reasoning; OPD A/B uses >=4096.")
    parser.add_argument("--mmlu-max-tokens", type=int, default=32,
                        help="generation budget for MMLU (single-letter answer); 32 is ample.")
    parser.add_argument("--seed", type=int, default=None,
                        help="if set, shuffle per-subject MMLU pool and GSM8K test pool before "
                             "sampling. Distinct seeds give independent draws for variance estimation. "
                             "Default unset = original deterministic ordering, reproduces older runs.")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seed list. When set, writes seed_<N>/ subdirectories "
                             "compatible with scripts/analyze_multi_seed.py. Do not use for full "
                             "deterministic MATH-500; use training-seed checkpoints instead.")
    parser.add_argument("--mmlu-revision", default=os.environ.get("ARLE_MMLU_REVISION", DEFAULT_DATASET_REVISION),
                        help="Hugging Face revision for cais/mmlu")
    parser.add_argument("--mmlu-local-dir", default=os.environ.get("ARLE_MMLU_LOCAL_DIR"),
                        help="local MMLU snapshot dir with all/{dev,test}-*.parquet; skips Hub download")
    parser.add_argument("--gsm8k-revision", default=os.environ.get("ARLE_GSM8K_REVISION", DEFAULT_DATASET_REVISION),
                        help="Hugging Face revision for openai/gsm8k")
    parser.add_argument("--math500-revision", default=os.environ.get("ARLE_MATH500_REVISION", DEFAULT_DATASET_REVISION),
                        help="Hugging Face revision for HuggingFaceH4/MATH-500")
    parser.add_argument("--math500-jsonl", type=Path, default=os.environ.get("ARLE_MATH500_JSONL"),
                        help="local MATH-500 jsonl; avoids Hugging Face network access")
    parser.add_argument("--output", type=Path, required=True, help="output directory for per-task reports")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    client = build_client(
        args.backend,
        base_url=args.base_url,
        model_id=args.model_id,
        model_path=args.model_path,
        dtype=args.dtype,
        request_timeout=args.request_timeout,
    )

    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TASK_RUNNERS]
    if unknown:
        print(f"unknown tasks: {unknown}. supported: {list(TASK_RUNNERS)}", file=sys.stderr)
        return 2

    try:
        seeds = _parse_seed_list(args.seeds)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if seeds is not None and args.seed is not None:
        print("--seed and --seeds are mutually exclusive", file=sys.stderr)
        return 2
    if seeds is not None and requested == ["math500"] and args.n_samples >= 500:
        print(
            "full deterministic MATH-500 uses one greedy eval and Wilson CI; "
            "use separate training-seed checkpoints for variance",
            file=sys.stderr,
        )
        return 2

    if seeds is None:
        summary = _run_suite_once(args, client, requested, args.output, args.seed)
        _print_summary(summary)
        return 0

    aggregate = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "final_result_contract": FINAL_RESULT_CONTRACT,
        "mode": "multi_seed",
        "seeds": seeds,
        "backend": args.backend,
        "base_url": args.base_url if args.backend == "arle" else None,
        "model_path": args.model_path if args.backend == "hf" else None,
        "model_id": args.model_id,
        "dataset_revisions": {
            "mmlu": args.mmlu_revision,
            "gsm8k": args.gsm8k_revision,
            "math500": args.math500_revision,
        },
        "tasks": requested,
        "per_seed": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for seed in seeds:
        print(f"\n========== seed {seed} ==========", flush=True)
        seed_dir = args.output / f"seed_{seed}"
        summary = _run_suite_once(args, client, requested, seed_dir, seed)
        aggregate["per_seed"][str(seed)] = {
            task: {
                "status": report["status"],
                "accuracy": report.get("accuracy"),
                "n_correct": report.get("n_correct"),
                "n_scored": report.get("n_scored"),
                "n_invalid": report.get("n_invalid"),
                "ci95": report.get("ci95"),
            }
            for task, report in summary["tasks"].items()
        }
        _print_summary(summary)
    aggregate["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

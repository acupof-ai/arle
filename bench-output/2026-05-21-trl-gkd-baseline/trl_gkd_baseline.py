#!/usr/bin/env python3
"""TRL GKD baseline for ARLE OPD Qwen3-0.6B head-to-head.

This script intentionally mirrors the ARLE real-checkpoint OPD harness:

* Qwen3-0.6B teacher is frozen.
* Qwen3-0.6B student starts from the same checkpoint plus deterministic
  uniform perturbation with amplitude 1e-3.
* 32 train prompts + 4 held-out prompts use the same token-id arrays as
  crates/train/examples/opd_step_cuda_realckpt_train.rs.
* Greedy on-policy generation uses max_new_tokens=8.
* Distillation loss is configured as beta=0, which makes TRL's generalized JSD
  helper compute forward KL(teacher || student).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.experimental.gkd import GKDConfig, GKDTrainer


MODEL_DIR = Path("/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B")
OUTPUT_DIR = Path("bench-output/2026-05-21-trl-gkd-baseline")
PERTURB_SEED = 0x0F0D_CAFE_2026_0521
PERTURB_SCALE = 1.0e-3
DEFAULT_LR = 1.0e-7
DEFAULT_STEPS = 500
DEFAULT_ROLLOUT_LEN = 8
DECODE_LEN = 16
EVAL_STEPS = (0, 50, 100, 250, 500)

TRAIN_PROMPTS_32 = [
    [1, 872, 198, 3456],
    [1, 198, 1512, 429],
    [1, 770, 3186, 25, 220],
    [1, 644, 374, 279, 1887],
    [1, 3838, 374, 264, 2077, 13],
    [1, 785, 594, 287, 374, 1690],
    [1, 3347, 11, 358, 1052, 429],
    [1, 2610, 527, 1139, 304, 279, 1670],
    [1, 888, 536, 4697, 972],
    [1, 374, 11, 279, 1372, 315],
    [1, 2874, 369, 279, 31559],
    [1, 7521, 481, 362, 5714],
    [1, 43059, 21938, 315, 7148],
    [1, 358, 646, 944, 1490, 432],
    [1, 477, 11, 323, 279, 62],
    [1, 576, 1102, 315, 264, 729],
    [1, 291, 504, 279, 1467, 11],
    [1, 702, 1012, 1483, 311, 7512],
    [1, 264, 11245, 2168, 429, 702],
    [1, 3555, 374, 264, 5714, 30],
    [1, 19257, 311, 279, 1251, 315],
    [1, 1156, 3019, 304, 279, 1882],
    [1, 2701, 1467, 25, 4710, 785],
    [1, 315, 279, 3364, 13, 576],
    [1, 279, 897, 5927, 553, 279],
    [1, 2055, 11, 369, 279, 1140],
    [1, 28469, 9363, 525, 279],
    [1, 1012, 13570, 14975, 304, 279],
    [1, 1887, 2242, 1294, 2827, 8],
    [1, 62, 716, 477, 11, 323],
    [1, 1512, 429, 374, 11, 279],
    [1, 74595, 11, 714, 279, 1467],
]

HELDOUT_PROMPTS = [
    [1, 4438, 374, 279, 2768],
    [1, 1516, 374, 264, 1296, 4339],
    [1, 785, 1401, 315, 279, 1967],
    [1, 3198, 279, 1296, 25, 220],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--rollout-len", type=int, default=DEFAULT_ROLLOUT_LEN)
    parser.add_argument("--eval-steps", default=",".join(str(x) for x in EVAL_STEPS))
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help=(
            "Optional JSONL prompt file. Rows may contain {'tokens':[...]} or "
            "{'text':'...'}. The final --heldout-count rows become held-out."
        ),
    )
    parser.add_argument("--heldout-count", type=int, default=4)
    parser.add_argument("--prompt-max-tokens", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Load/evaluate only; useful for smoke tests.")
    return parser.parse_args()


def parse_eval_steps(raw: str, max_step: int) -> list[int]:
    steps = sorted({int(part) for part in raw.split(",") if part.strip()})
    if 0 not in steps:
        steps.insert(0, 0)
    if max_step not in steps:
        steps.append(max_step)
    return steps


def load_prompt_file(path: Path, tokenizer: Any, heldout_count: int, max_tokens: int) -> tuple[list[list[int]], list[list[int]]]:
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "tokens" in payload:
                tokens = [int(x) for x in payload["tokens"]]
            elif "text" in payload:
                cap = int(payload.get("max_tokens", max_tokens))
                tokens = tokenizer.encode(payload["text"], add_special_tokens=False)[:cap]
            else:
                raise ValueError(f"{path}:{line_no}: expected 'tokens' or 'text'")
            if not tokens:
                raise ValueError(f"{path}:{line_no}: prompt tokenized to empty")
            rows.append(tokens[:max_tokens])
    if len(rows) <= heldout_count:
        raise ValueError(f"{path}: need more than {heldout_count} prompts")
    return rows[:-heldout_count], rows[-heldout_count:]


def perturb_student(model: torch.nn.Module, scale: float, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed & 0x7FFF_FFFF_FFFF_FFFF)
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            noise = torch.rand(param.shape, dtype=param.dtype, generator=generator)
            param.add_(noise.to(device=param.device) * scale)


def make_dataset(prompts: list[list[int]]) -> Dataset:
    return Dataset.from_list([{"prompt_ids": prompt, "prompt_index": idx} for idx, prompt in enumerate(prompts)])


def make_collator(pad_token_id: int):
    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["prompt_ids"]) for feature in features)
        prompts = []
        masks = []
        for feature in features:
            ids = list(feature["prompt_ids"])
            pad = max_len - len(ids)
            prompts.append(ids + [pad_token_id] * pad)
            masks.append([1] * len(ids) + [0] * pad)
        prompt_tensor = torch.tensor(prompts, dtype=torch.long)
        mask_tensor = torch.tensor(masks, dtype=torch.long)
        return {
            "prompts": prompt_tensor,
            "prompt_attention_mask": mask_tensor,
            "input_ids": prompt_tensor.clone(),
            "attention_mask": mask_tensor.clone(),
            "labels": prompt_tensor.clone(),
        }

    return collate


def exact_overlap_pct(lhs: list[int], rhs: list[int]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError("overlap inputs must have the same length")
    return 100.0 * sum(1 for a, b in zip(lhs, rhs) if a == b) / len(lhs)


@torch.no_grad()
def greedy_suffix(
    model: torch.nn.Module,
    prompt: list[int],
    decode_len: int,
    device: torch.device,
    pad_token_id: int,
) -> list[int]:
    input_ids = torch.tensor([prompt], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=decode_len,
        pad_token_id=pad_token_id,
        use_cache=True,
    )
    return [int(x) for x in generated[0, len(prompt) : len(prompt) + decode_len].tolist()]


@torch.no_grad()
def teacher_forced_metrics(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    prompt: list[int],
    teacher_suffix: list[int],
    device: torch.device,
) -> dict[str, float]:
    sequence = prompt + teacher_suffix
    ids = torch.tensor([sequence], device=device, dtype=torch.long)
    mask = torch.ones_like(ids)
    teacher_logits = teacher(input_ids=ids, attention_mask=mask).logits
    student_logits = student(input_ids=ids, attention_mask=mask).logits
    start = len(prompt) - 1
    end = start + len(teacher_suffix)
    t_logits = teacher_logits[:, start:end, :]
    s_logits = student_logits[:, start:end, :]
    t_logp = torch.log_softmax(t_logits, dim=-1)
    s_logp = torch.log_softmax(s_logits, dim=-1)
    t_probs = torch.exp(t_logp)
    kl = torch.sum(t_probs * (t_logp - s_logp), dim=-1).mean()
    targets = torch.tensor(teacher_suffix, device=device, dtype=torch.long).view(1, -1, 1)
    teacher_nll = -torch.gather(s_logp, dim=-1, index=targets).mean()
    top3 = torch.topk(s_logits, k=3, dim=-1).indices
    top3_hits = (top3 == targets).any(dim=-1).float().mean() * 100.0
    return {
        "kl": float(kl.item()),
        "teacher_nll": float(teacher_nll.item()),
        "top3_overlap_pct": float(top3_hits.item()),
    }


@torch.no_grad()
def evaluate_split(
    split: str,
    step: int,
    prompts: list[list[int]],
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    device: torch.device,
    pad_token_id: int,
    output_jsonl: Path,
) -> dict[str, float]:
    teacher.eval()
    student.eval()
    rows = []
    for prompt_index, prompt in enumerate(prompts):
        teacher_tokens = greedy_suffix(teacher, prompt, DECODE_LEN, device, pad_token_id)
        student_tokens = greedy_suffix(student, prompt, DECODE_LEN, device, pad_token_id)
        overlap = exact_overlap_pct(student_tokens, teacher_tokens)
        metrics = teacher_forced_metrics(teacher, student, prompt, teacher_tokens, device)
        row = {
            "kind": "eval_detail",
            "step": step,
            "split": split,
            "prompt_index": prompt_index,
            "prompt": prompt,
            "teacher_suffix": teacher_tokens,
            "student_suffix": student_tokens,
            "overlap_pct": overlap,
            **metrics,
        }
        rows.append(row)
        print(
            "eval_detail "
            f"step={step} split={split} prompt_index={prompt_index} prompt={prompt} "
            f"overlap_pct={overlap:.6f} kl={metrics['kl']:.12e} "
            f"teacher_nll={metrics['teacher_nll']:.12e} "
            f"top3_overlap_pct={metrics['top3_overlap_pct']:.6f}"
        )
    with output_jsonl.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    student.train()
    return {
        f"{split}_overlap_pct": statistics.fmean(row["overlap_pct"] for row in rows),
        f"{split}_kl": statistics.fmean(row["kl"] for row in rows),
        f"{split}_teacher_nll": statistics.fmean(row["teacher_nll"] for row in rows),
        f"{split}_top3_overlap_pct": statistics.fmean(row["top3_overlap_pct"] for row in rows),
    }


def evaluate_snapshot(
    step: int,
    train_prompts: list[list[int]],
    heldout_prompts: list[list[int]],
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    device: torch.device,
    pad_token_id: int,
    output_jsonl: Path,
) -> dict[str, float]:
    started = time.perf_counter()
    train = evaluate_split("train", step, train_prompts, teacher, student, device, pad_token_id, output_jsonl)
    heldout = evaluate_split("heldout", step, heldout_prompts, teacher, student, device, pad_token_id, output_jsonl)
    row = {
        "kind": "eval_summary",
        "step": step,
        **train,
        **heldout,
        "eval_seconds": time.perf_counter() - started,
    }
    with output_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        "eval_summary "
        f"step={step} train_overlap_pct={row['train_overlap_pct']:.6f} "
        f"heldout_overlap_pct={row['heldout_overlap_pct']:.6f} "
        f"train_kl={row['train_kl']:.12e} heldout_kl={row['heldout_kl']:.12e} "
        f"train_teacher_nll={row['train_teacher_nll']:.12e} "
        f"heldout_teacher_nll={row['heldout_teacher_nll']:.12e} "
        f"eval_seconds={row['eval_seconds']:.6f}"
    )
    return row


class TimingAndEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_steps: set[int],
        train_prompts: list[list[int]],
        heldout_prompts: list[list[int]],
        output_jsonl: Path,
        pad_token_id: int,
    ) -> None:
        self.eval_steps = eval_steps
        self.train_prompts = train_prompts
        self.heldout_prompts = heldout_prompts
        self.output_jsonl = output_jsonl
        self.pad_token_id = pad_token_id
        self.step_seconds: list[float] = []
        self.eval_rows: list[dict[str, float]] = []
        self._started = 0.0
        self.trainer: GKDTrainer | None = None

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._started = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._started
        self.step_seconds.append(elapsed)
        print(f"train_step step={state.global_step} step_seconds={elapsed:.6f}")
        with self.output_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "train_step",
                        "step": int(state.global_step),
                        "step_seconds": elapsed,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if int(state.global_step) in self.eval_steps:
            if self.trainer is None:
                raise RuntimeError("callback.trainer was not set")
            model = self.trainer.accelerator.unwrap_model(kwargs["model"])
            teacher = self.trainer.accelerator.unwrap_model(self.trainer.teacher_model)
            row = evaluate_snapshot(
                int(state.global_step),
                self.train_prompts,
                self.heldout_prompts,
                teacher,
                model,
                torch.device(args.device),
                self.pad_token_id,
                self.output_jsonl,
            )
            self.eval_rows.append(row)


def summarize(step_seconds: list[float], eval_rows: list[dict[str, float]]) -> dict[str, Any]:
    if step_seconds:
        mean = statistics.fmean(step_seconds)
        median = statistics.median(step_seconds)
        sigma = statistics.pstdev(step_seconds)
        sigma_pct = 100.0 * sigma / mean if mean else math.nan
    else:
        mean = median = sigma = sigma_pct = math.nan
    return {
        "kind": "training_summary",
        "steps": len(step_seconds),
        "mean_step_seconds": mean,
        "median_step_seconds": median,
        "sigma_step_seconds": sigma,
        "sigma_pct": sigma_pct,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0,
        "eval_rows": eval_rows,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this TRL GKD baseline")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    summary_path = args.output_dir / "summary.json"
    metrics_path.write_text("", encoding="utf-8")

    eval_steps = parse_eval_steps(args.eval_steps, args.steps)
    print(
        "config "
        f"model_dir={args.model_dir} steps={args.steps} lr={args.lr:.9e} "
        f"rollout_len={args.rollout_len} perturb_scale={PERTURB_SCALE:.9e} "
        f"perturb_seed=0x{PERTURB_SEED:016x} eval_steps={eval_steps} "
        f"gradient_checkpointing={args.gradient_checkpointing}"
    )
    print(f"cuda device={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.prompts_file is None:
        train_prompts = [list(prompt) for prompt in TRAIN_PROMPTS_32]
        heldout_prompts = [list(prompt) for prompt in HELDOUT_PROMPTS]
        prompt_source = "builtin_32"
    else:
        train_prompts, heldout_prompts = load_prompt_file(
            args.prompts_file, tokenizer, args.heldout_count, args.prompt_max_tokens
        )
        prompt_source = str(args.prompts_file)
    print(
        f"prompts source={prompt_source} train_count={len(train_prompts)} "
        f"heldout_count={len(heldout_prompts)}"
    )

    print("loading teacher")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float32,
        device_map=None,
    )
    teacher.requires_grad_(False)
    teacher.eval()

    print("loading student")
    student = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float32,
        device_map=None,
    )
    student.train()
    print("perturbing student")
    perturb_student(student, PERTURB_SCALE, PERTURB_SEED)

    trainable_params = sum(param.numel() for param in student.parameters() if param.requires_grad)
    print(f"trainable_params={trainable_params}")

    training_args = GKDConfig(
        output_dir=str(args.output_dir / "trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=args.steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1.0e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=10,
        logging_strategy="steps",
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        seed=20260521,
        data_seed=20260521,
        shuffle_dataset=False,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=False,
        fp16=False,
        max_new_tokens=args.rollout_len,
        temperature=1.0,
        lmbda=1.0,
        beta=0.0,
        seq_kd=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    callback = TimingAndEvalCallback(
        set(eval_steps),
        train_prompts,
        heldout_prompts,
        metrics_path,
        int(tokenizer.pad_token_id),
    )
    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        train_dataset=make_dataset(train_prompts),
        processing_class=tokenizer,
        data_collator=make_collator(int(tokenizer.pad_token_id)),
        callbacks=[callback],
    )
    callback.trainer = trainer

    trainer.generation_config.do_sample = False
    trainer.generation_config.temperature = 1.0
    trainer.generation_config.top_k = 0
    trainer.generation_config.max_new_tokens = args.rollout_len
    trainer.generation_kwargs.update(
        {
            "do_sample": False,
            "temperature": 1.0,
            "top_k": 0,
            "max_new_tokens": args.rollout_len,
            "pad_token_id": int(tokenizer.pad_token_id),
        }
    )

    device = torch.device(trainer.args.device)
    row0 = evaluate_snapshot(
        0,
        train_prompts,
        heldout_prompts,
        teacher.to(device),
        student.to(device),
        device,
        int(tokenizer.pad_token_id),
        metrics_path,
    )
    callback.eval_rows.append(row0)

    if not args.skip_train:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trainer.train()

    summary = summarize(callback.step_seconds, callback.eval_rows)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(
        "training_summary "
        f"steps={summary['steps']} mean_step_seconds={summary['mean_step_seconds']:.6f} "
        f"median_step_seconds={summary['median_step_seconds']:.6f} "
        f"sigma_pct={summary['sigma_pct']:.6f} "
        f"peak_allocated_mib={summary['peak_allocated_mib']:.1f} "
        f"peak_reserved_mib={summary['peak_reserved_mib']:.1f}"
    )
    for row in callback.eval_rows:
        print(
            "summary_eval_row "
            f"step={int(row['step'])} train_overlap_pct={row['train_overlap_pct']:.6f} "
            f"heldout_overlap_pct={row['heldout_overlap_pct']:.6f} "
            f"train_kl={row['train_kl']:.12e} heldout_kl={row['heldout_kl']:.12e} "
            f"train_teacher_nll={row['train_teacher_nll']:.12e} "
            f"heldout_teacher_nll={row['heldout_teacher_nll']:.12e}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare ARLE TurboQuant logits against a PyTorch BF16 baseline.

The gate for this bench axis is the dominant-logit top-64 relative error for a
fixed prompt token. Full-vocab max relative error is intentionally not used
because near-zero logits make it unstable and not representative.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bf16-model",
        default="/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B",
    )
    parser.add_argument("--arle-logits", required=True)
    parser.add_argument("--input-ids", default="9419")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--top64", required=True)
    parser.add_argument("--pytorch-logits", required=True)
    return parser.parse_args()


def parse_ids(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def load_arle_logits(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    logits = np.asarray(payload["logits"], dtype=np.float32)
    expected = int(payload["vocab_size"]) * int(payload["seq_len"])
    if logits.size != expected:
        raise ValueError(f"ARLE logits size {logits.size} != expected {expected}")
    return logits.reshape(int(payload["seq_len"]), int(payload["vocab_size"]))[-1]


def main() -> None:
    args = parse_args()
    input_ids = parse_ids(args.input_ids)
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    arle_logits = load_arle_logits(Path(args.arle_logits))

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.bf16_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
    ).eval()
    load_seconds = time.perf_counter() - load_started

    forward_started = time.perf_counter()
    tokens = torch.tensor([input_ids], dtype=torch.long)
    output = model(input_ids=tokens, use_cache=False)
    ref_logits = output.logits[0, -1].float().cpu().numpy().astype(np.float32)
    forward_seconds = time.perf_counter() - forward_started

    if ref_logits.shape != arle_logits.shape:
        raise ValueError(f"shape mismatch: torch={ref_logits.shape} arle={arle_logits.shape}")
    if not np.all(np.isfinite(ref_logits)):
        raise ValueError("PyTorch logits contain non-finite values")
    if not np.all(np.isfinite(arle_logits)):
        raise ValueError("ARLE logits contain non-finite values")

    dominant = np.argsort(np.abs(ref_logits))[-64:][::-1]
    abs_err = np.abs(arle_logits[dominant] - ref_logits[dominant])
    rel_err = abs_err / np.maximum(np.abs(ref_logits[dominant]), 1.0e-6)
    full_abs = np.abs(arle_logits - ref_logits)
    summary = {
        "bf16_model": args.bf16_model,
        "arle_logits": args.arle_logits,
        "input_ids": input_ids,
        "vocab_size": int(ref_logits.shape[0]),
        "torch_load_seconds": load_seconds,
        "torch_forward_seconds": forward_seconds,
        "top64_max_abs": float(abs_err.max()),
        "top64_mean_abs": float(abs_err.mean()),
        "top64_max_rel": float(rel_err.max()),
        "top64_mean_rel": float(rel_err.mean()),
        "top64_ref_mean_abs": float(np.mean(np.abs(ref_logits[dominant]))),
        "full_vocab_max_abs": float(full_abs.max()),
        "full_vocab_mean_abs": float(full_abs.mean()),
        "full_vocab_rmse": float(math.sqrt(float(np.mean(full_abs * full_abs)))),
        "gate_top64_max_rel_threshold": 5.0e-2,
        "gate_pass": bool(float(rel_err.max()) <= 5.0e-2),
    }

    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    Path(args.pytorch_logits).write_text(
        json.dumps({"input_ids": input_ids, "logits": ref_logits.tolist()}) + "\n",
        encoding="utf-8",
    )
    with Path(args.top64).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "token_id", "torch_bf16", "arle_tq4", "abs_err", "rel_err"])
        for rank, idx in enumerate(dominant):
            writer.writerow(
                [
                    rank,
                    int(idx),
                    float(ref_logits[idx]),
                    float(arle_logits[idx]),
                    float(abs(arle_logits[idx] - ref_logits[idx])),
                    float(abs(arle_logits[idx] - ref_logits[idx]) / max(abs(ref_logits[idx]), 1.0e-6)),
                ]
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

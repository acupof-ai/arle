#!/usr/bin/env python3
"""Tensor-local parity for Qwen3.5-9B TurboQuant weights.

This script compares three tensors for one small slice:

1. Original BF16 source weight.
2. Faithful Python dequant of the `.tq_packed/.tq_scales/.tq_signs` tensors,
   matching `scripts/turboquant_weights.py`.
3. ARLE CUDA dequant output dumped by `infer/examples/turboquant_weight_dequant_dump.rs`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--tq-model", required=True)
    parser.add_argument("--tensor-base", required=True)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-count", type=int, default=8)
    parser.add_argument("--cuda-json", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--top-errors", required=True)
    return parser.parse_args()


def shard_for(model_dir: Path, key: str) -> Path:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][key]
    return model_dir / shard


def load_tensor(model_dir: Path, key: str):
    with safe_open(shard_for(model_dir, key), framework="pt") as handle:
        return handle.get_tensor(key)


def fwht_inplace(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = x[..., i : i + h].copy()
            b = x[..., i + h : i + 2 * h].copy()
            x[..., i : i + h] = a + b
            x[..., i + h : i + 2 * h] = a - b
        h *= 2
    x /= np.sqrt(n)
    return x


def faithful_python_dequant(
    packed: np.ndarray,
    scales: np.ndarray,
    signs: np.ndarray,
    centroids: np.ndarray,
    bits: int,
    group_size: int,
) -> np.ndarray:
    rows = packed.shape[0]
    num_groups = scales.shape[1]
    k_dim = num_groups * group_size
    effective_bits = 4 if bits == 3 else bits
    indices_per_byte = 8 // effective_bits
    mask = (1 << effective_bits) - 1

    indices = np.zeros((rows, k_dim), dtype=np.int64)
    for k in range(k_dim):
        byte_idx = k // indices_per_byte
        sub_idx = k % indices_per_byte
        indices[:, k] = (packed[:, byte_idx] >> (sub_idx * effective_bits)) & mask

    rotated = centroids[indices].reshape(rows, num_groups, group_size)
    rotated = rotated * scales.astype(np.float32)[:, :, None]
    out = fwht_inplace(rotated.copy()).reshape(rows, k_dim)
    out *= signs.astype(np.float32)[None, :]
    return out.astype(np.float32)


def cuda_fwht_sign_bug_dequant(
    packed: np.ndarray,
    scales: np.ndarray,
    signs: np.ndarray,
    centroids: np.ndarray,
    bits: int,
    group_size: int,
) -> np.ndarray:
    """Emulate the current CUDA weight FWHT sign convention.

    `turboquant_weight_gemv.cu::fwht_warp_optimized` currently computes the
    upper butterfly lane as `upper - lower`. The quantizer's Python FWHT uses
    `lower - upper`. This control tells us whether the measured CUDA output is
    the intended transform or this sign convention.
    """
    rows = packed.shape[0]
    num_groups = scales.shape[1]
    k_dim = num_groups * group_size
    effective_bits = 4 if bits == 3 else bits
    indices_per_byte = 8 // effective_bits
    mask = (1 << effective_bits) - 1

    indices = np.zeros((rows, k_dim), dtype=np.int64)
    for k in range(k_dim):
        byte_idx = k // indices_per_byte
        sub_idx = k % indices_per_byte
        indices[:, k] = (packed[:, byte_idx] >> (sub_idx * effective_bits)) & mask

    x = centroids[indices].reshape(rows, num_groups, group_size)
    x = x * scales.astype(np.float32)[:, :, None]
    stride = 1
    while stride < group_size:
        y = x.copy()
        for tid in range(group_size):
            pair = tid ^ stride
            other = x[..., pair]
            val = x[..., tid]
            y[..., tid] = val - other if (tid & stride) else val + other
        x = y
        stride <<= 1
    x /= np.sqrt(group_size)
    out = x.reshape(rows, k_dim) * signs.astype(np.float32)[None, :]
    return out.astype(np.float32)


def metrics(lhs: np.ndarray, rhs: np.ndarray, lhs_name: str, rhs_name: str) -> dict:
    diff = lhs - rhs
    abs_err = np.abs(diff)
    rel_err = abs_err / np.maximum(np.abs(rhs), 1.0e-6)
    rhs_rms = float(np.sqrt(np.mean(rhs * rhs)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    top1_threshold = float(np.quantile(rel_err, 0.99))
    top1 = rel_err[rel_err >= top1_threshold]
    return {
        "lhs": lhs_name,
        "rhs": rhs_name,
        "elements": int(lhs.size),
        "max_abs": float(abs_err.max()),
        "mean_abs": float(abs_err.mean()),
        "rmse": rmse,
        "rhs_rms": rhs_rms,
        "rmse_over_rhs_rms": float(rmse / max(rhs_rms, 1.0e-12)),
        "max_rel": float(rel_err.max()),
        "mean_rel": float(rel_err.mean()),
        "rel_p50": float(np.quantile(rel_err, 0.50)),
        "rel_p90": float(np.quantile(rel_err, 0.90)),
        "rel_p95": float(np.quantile(rel_err, 0.95)),
        "rel_p99": top1_threshold,
        "rel_p999": float(np.quantile(rel_err, 0.999)),
        "top1pct_rel_mean": float(top1.mean()),
        "top1pct_rel_max": float(top1.max()),
    }


def main() -> None:
    args = parse_args()
    source_model = Path(args.source_model)
    tq_model = Path(args.tq_model)
    row_end = args.row_start + args.row_count
    source_key = args.tensor_base + ".weight"
    packed_key = args.tensor_base + ".tq_packed"
    scales_key = args.tensor_base + ".tq_scales"
    signs_key = args.tensor_base + ".tq_signs"

    config = json.loads((tq_model / "turboquant_config.json").read_text())
    bits = int(config["bits"])
    group_size = int(config["group_size"])
    config_centroids = np.asarray(config["centroids"], dtype=np.float32)

    source = load_tensor(source_model, source_key)[args.row_start:row_end].float().numpy()
    packed = load_tensor(tq_model, packed_key)[args.row_start:row_end].numpy()
    scales = load_tensor(tq_model, scales_key)[args.row_start:row_end].numpy().astype(np.float32)
    signs = load_tensor(tq_model, signs_key).numpy().astype(np.float32)
    python_dequant = faithful_python_dequant(packed, scales, signs, config_centroids, bits, group_size)
    cuda_bug_dequant = cuda_fwht_sign_bug_dequant(
        packed, scales, signs, config_centroids, bits, group_size
    )
    cuda_bug_dequant_bf16 = (
        torch.tensor(cuda_bug_dequant, dtype=torch.float32).to(dtype=torch.bfloat16).float().numpy()
    )

    cuda_payload = json.loads(Path(args.cuda_json).read_text())
    cuda_dequant = np.asarray(cuda_payload["values"], dtype=np.float32).reshape(
        cuda_payload["shape"]
    )
    cuda_centroids = np.asarray(cuda_payload["centroids"], dtype=np.float32)

    if source.shape != python_dequant.shape or source.shape != cuda_dequant.shape:
        raise ValueError(
            f"shape mismatch: source={source.shape} python={python_dequant.shape} cuda={cuda_dequant.shape}"
        )

    reports = [
        metrics(python_dequant, source, "python_faithful_dequant", "bf16_source"),
        metrics(cuda_dequant, source, "arle_cuda_dequant", "bf16_source"),
        metrics(cuda_dequant, python_dequant, "arle_cuda_dequant", "python_faithful_dequant"),
    ]
    cent_abs = np.abs(cuda_centroids - config_centroids)
    summary = {
        "source_model": str(source_model),
        "tq_model": str(tq_model),
        "tensor_base": args.tensor_base,
        "row_start": args.row_start,
        "row_count": args.row_count,
        "shape": list(source.shape),
        "bits": bits,
        "group_size": group_size,
        "centroids_config_vs_cuda": {
            "max_abs": float(cent_abs.max()),
            "mean_abs": float(cent_abs.mean()),
            "config": config_centroids.tolist(),
            "cuda": cuda_centroids.tolist(),
        },
        "metrics": reports,
        "fwht_bug_control": {
            "cuda_vs_good_python_max_abs": metrics(
                cuda_dequant, python_dequant, "arle_cuda_dequant", "python_faithful_dequant"
            )["max_abs"],
            "cuda_vs_good_python_rmse": metrics(
                cuda_dequant, python_dequant, "arle_cuda_dequant", "python_faithful_dequant"
            )["rmse"],
            "cuda_vs_bug_emulation_max_abs": metrics(
                cuda_dequant, cuda_bug_dequant, "arle_cuda_dequant", "cuda_bug_emulation"
            )["max_abs"],
            "cuda_vs_bug_emulation_rmse": metrics(
                cuda_dequant, cuda_bug_dequant, "arle_cuda_dequant", "cuda_bug_emulation"
            )["rmse"],
            "cuda_vs_bug_emulation_bf16_max_abs": metrics(
                cuda_dequant,
                cuda_bug_dequant_bf16,
                "arle_cuda_dequant",
                "cuda_bug_emulation_bf16",
            )["max_abs"],
            "cuda_vs_bug_emulation_bf16_rmse": metrics(
                cuda_dequant,
                cuda_bug_dequant_bf16,
                "arle_cuda_dequant",
                "cuda_bug_emulation_bf16",
            )["rmse"],
            "interpretation": (
                "Compare ARLE CUDA against the faithful Python dequant and the pre-fix FWHT "
                "sign-bug emulation. A fixed kernel should match faithful Python after BF16 "
                "rounding and differ from the bug emulation."
            ),
        },
        "decision": (
            "cuda_dequant_matches_python_with_bf16_rounding"
            if reports[2]["rmse_over_rhs_rms"] <= 0.005
            else "cuda_dequant_differs_from_python"
        ),
    }

    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    rel_cuda_python = np.abs(cuda_dequant - python_dequant) / np.maximum(
        np.abs(python_dequant), 1.0e-6
    )
    flat_order = np.argsort(rel_cuda_python.reshape(-1))[-128:][::-1]
    with Path(args.top_errors).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "rank",
                "row",
                "col",
                "bf16_source",
                "python_faithful",
                "arle_cuda",
                "cuda_vs_python_abs",
                "cuda_vs_python_rel",
            ]
        )
        cols = source.shape[1]
        for rank, idx in enumerate(flat_order):
            row = int(idx // cols)
            col = int(idx % cols)
            writer.writerow(
                [
                    rank,
                    row + args.row_start,
                    col,
                    float(source[row, col]),
                    float(python_dequant[row, col]),
                    float(cuda_dequant[row, col]),
                    float(abs(cuda_dequant[row, col] - python_dequant[row, col])),
                    float(rel_cuda_python[row, col]),
                ]
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

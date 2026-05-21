#!/usr/bin/env python3
"""Qwen3.5 TQ4 dense tensor and dense module parity diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, logging


TQ_SUFFIXES = (".tq_packed", ".tq_scales", ".tq_signs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-model",
        default="/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B",
        type=Path,
    )
    parser.add_argument(
        "--tq-model",
        default="/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4",
        type=Path,
    )
    parser.add_argument("--token-id", default=9419, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arle-json", type=Path)
    parser.add_argument("--skip-arle-run", action="store_true")
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--module-gate", default=0.01, type=float)
    return parser.parse_args()


def safetensor_index(model_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in index:
                    raise RuntimeError(f"duplicate tensor key {key}")
                index[key] = path
    return index


def tensor_bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.uint8)


def dense_tensor_compare(source_model: Path, tq_model: Path) -> tuple[list[dict[str, Any]], bool]:
    source_index = safetensor_index(source_model)
    tq_index = safetensor_index(tq_model)
    dense_keys = sorted(key for key in tq_index if not key.endswith(TQ_SUFFIXES))
    rows: list[dict[str, Any]] = []
    all_pass = True
    for key in dense_keys:
        source_path = source_index.get(key)
        if source_path is None:
            rows.append(
                {
                    "tensor": key,
                    "source_file": None,
                    "tq_file": str(tq_index[key]),
                    "shape": None,
                    "dtype": None,
                    "bit_identical": False,
                    "reason": "missing_source_tensor",
                }
            )
            all_pass = False
            continue
        with safe_open(source_path, framework="pt", device="cpu") as source_handle:
            source = source_handle.get_tensor(key)
        with safe_open(tq_index[key], framework="pt", device="cpu") as tq_handle:
            tq = tq_handle.get_tensor(key)
        same_shape = tuple(source.shape) == tuple(tq.shape)
        same_dtype = source.dtype == tq.dtype
        bit_identical = same_shape and same_dtype and torch.equal(tensor_bits(source), tensor_bits(tq))
        row: dict[str, Any] = {
            "tensor": key,
            "source_file": str(source_path),
            "tq_file": str(tq_index[key]),
            "shape": list(tq.shape),
            "dtype": str(tq.dtype),
            "numel": tq.numel(),
            "bit_identical": bool(bit_identical),
        }
        if not bit_identical:
            all_pass = False
            row["same_shape"] = same_shape
            row["same_dtype"] = same_dtype
            if same_shape and source.is_floating_point() and tq.is_floating_point():
                diff = (source.float() - tq.float()).abs()
                row["max_abs"] = float(diff.max().item())
                row["mean_abs"] = float(diff.mean().item())
        rows.append(row)
        del source, tq
    return rows, all_pass


def deterministic_bf16(length: int, salt: int) -> torch.Tensor:
    values = [(((idx * 37 + salt * 17) % 257) - 128) / 64.0 for idx in range(length)]
    return torch.tensor(values, dtype=torch.bfloat16)


def run_arle_dump(args: argparse.Namespace, arle_json: Path) -> None:
    if args.skip_arle_run:
        return
    env = os.environ.copy()
    env.setdefault("NVCC_CCBIN", "/usr/bin/g++-14")
    env.setdefault("INFER_TILELANG_PYTHON", str(Path.cwd() / ".venv/bin/python"))
    env.setdefault("CUDARC_CUDA_VERSION", "13010")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env.setdefault("CARGO_BUILD_JOBS", "1")
    cmd = [
        args.cargo,
        "run",
        "-p",
        "infer",
        "--example",
        "qwen35_dense_module_dump",
        "--release",
        "--features",
        "cuda",
        "--",
        "--model-path",
        str(args.tq_model),
        "--token-id",
        str(args.token_id),
        "--output",
        str(arle_json),
    ]
    subprocess.run(cmd, check=True, env=env)


def compare_vectors(name: str, arle: list[float], reference: torch.Tensor, gate: float) -> dict[str, Any]:
    arle_tensor = torch.tensor(arle, dtype=torch.float32)
    ref = reference.reshape(-1).float().cpu()
    if arle_tensor.numel() != ref.numel():
        raise RuntimeError(f"{name} length mismatch: arle={arle_tensor.numel()} ref={ref.numel()}")
    diff = (arle_tensor - ref).abs()
    rmse = math.sqrt(float((diff * diff).mean().item()))
    ref_rms = math.sqrt(float((ref * ref).mean().item()))
    rel = diff / ref.abs().clamp_min(1.0e-6)
    first8 = []
    for idx in range(min(8, arle_tensor.numel())):
        first8.append(
            {
                "index": idx,
                "arle": float(arle_tensor[idx].item()),
                "pytorch": float(ref[idx].item()),
                "abs_err": float(diff[idx].item()),
                "rel_err": float(rel[idx].item()),
            }
        )
    ratio = rmse / max(ref_rms, 1.0e-12)
    return {
        "module": name,
        "len": arle_tensor.numel(),
        "max_abs": float(diff.max().item()),
        "max_rel": float(rel.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": rmse,
        "ref_rms": ref_rms,
        "rmse_over_ref_rms": ratio,
        "gate": gate,
        "gate_pass": ratio <= gate,
        "first8": first8,
    }


def module_compare(source_model: Path, arle_json: Path, token_id: int, gate: float) -> tuple[list[dict[str, Any]], bool]:
    logging.set_verbosity_error()
    model = AutoModelForCausalLM.from_pretrained(
        source_model,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )
    model.eval()
    with arle_json.open("r", encoding="utf-8") as handle:
        arle = json.load(handle)
    hidden = model.config.hidden_size
    with torch.no_grad():
        input_ids = torch.tensor([[token_id]], dtype=torch.long)
        embedding = model.model.embed_tokens(input_ids).reshape(-1)
        norm_input = deterministic_bf16(hidden, 17).reshape(1, 1, hidden)
        final_norm = model.model.norm(norm_input).reshape(-1)
        lm_head_input = deterministic_bf16(hidden, 29).reshape(1, hidden)
        lm_head = model.lm_head(lm_head_input).reshape(-1)
    rows = [
        compare_vectors("embedding", arle["embedding"], embedding, gate),
        compare_vectors("final_rmsnorm", arle["final_rmsnorm"], final_norm, gate),
        compare_vectors("lm_head", arle["lm_head"], lm_head, gate),
    ]
    return rows, all(row["gate_pass"] for row in rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arle_json = args.arle_json or (args.output_dir / "arle-dense-modules.json")

    dense_rows, dense_pass = dense_tensor_compare(args.source_model, args.tq_model)
    dense_summary = {
        "source_model": str(args.source_model),
        "tq_model": str(args.tq_model),
        "dense_tensor_count": len(dense_rows),
        "gate": "all dense tensors bit-identical",
        "gate_pass": dense_pass,
        "rows": dense_rows,
    }
    (args.output_dir / "dense-tensor-bitcompare.json").write_text(
        json.dumps(dense_summary, indent=2), encoding="utf-8"
    )
    print(f"dense_tensor_count={len(dense_rows)} gate_pass={dense_pass}")
    if not dense_pass:
        print("dense tensor bit-compare failed; skipping module scan")
        (args.output_dir / "summary.json").write_text(
            json.dumps({"dense": dense_summary, "modules": None}, indent=2), encoding="utf-8"
        )
        raise SystemExit(1)

    run_arle_dump(args, arle_json)
    module_rows, module_pass = module_compare(
        args.source_model, arle_json, args.token_id, args.module_gate
    )
    module_summary = {
        "source_model": str(args.source_model),
        "tq_model": str(args.tq_model),
        "token_id": args.token_id,
        "arle_json": str(arle_json),
        "gate": f"rmse/ref_rms <= {args.module_gate}",
        "gate_pass": module_pass,
        "rows": module_rows,
    }
    (args.output_dir / "dense-module-parity.json").write_text(
        json.dumps(module_summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps({"dense": dense_summary, "modules": module_summary}, indent=2),
        encoding="utf-8",
    )
    for row in module_rows:
        print(
            f"module={row['module']} rmse/ref_rms={row['rmse_over_ref_rms']:.8e} "
            f"max_abs={row['max_abs']:.8e} gate_pass={row['gate_pass']}"
        )
    if not module_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

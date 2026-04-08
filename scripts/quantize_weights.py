#!/usr/bin/env python3
"""Offline weight quantization: BF16 → INT8/INT4 per-group symmetric.

Quantizes all linear layer weights in a safetensors model and saves a new
model directory with quantized weights + scales.

Usage:
  python3 scripts/quantize_weights.py models/Qwen3-4B --bits 8 --output models/Qwen3-4B-W8
  python3 scripts/quantize_weights.py models/Qwen3-4B --bits 4 --output models/Qwen3-4B-W4
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def quantize_symmetric(weight: torch.Tensor, bits: int, group_size: int = 128):
    """Quantize weight tensor with per-group symmetric quantization.

    Args:
        weight: [N, K] float tensor
        bits: 4 or 8
        group_size: elements per group (default 128)

    Returns:
        qweight: [N, K] int8 (for bits=8) or [N, K//2] uint8 packed (for bits=4)
        scales: [N, K//group_size] bfloat16
    """
    N, K = weight.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    num_groups = K // group_size

    # Reshape to [N, num_groups, group_size]
    w = weight.float().reshape(N, num_groups, group_size)

    # Per-group absmax
    absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)  # [N, num_groups, 1]

    if bits == 8:
        max_val = 127.0
        scales = (absmax / max_val).squeeze(-1)  # [N, num_groups]
        # Quantize
        w_q = torch.clamp(torch.round(w / (absmax / max_val)), -max_val, max_val).to(torch.int8)
        qweight = w_q.reshape(N, K)
        scales = scales.to(torch.bfloat16)
        return qweight, scales

    elif bits == 4:
        max_val = 7.0
        scales = (absmax / max_val).squeeze(-1)
        w_q = torch.clamp(torch.round(w / (absmax / max_val)), -max_val, max_val).to(torch.int8)
        w_q = w_q.reshape(N, K)
        # Pack: two int4 values per byte, low nibble first
        # Shift signed [-8,7] to unsigned [0,15] by adding 8
        w_unsigned = (w_q + 8).to(torch.uint8)
        # Pack pairs: byte = low | (high << 4)
        assert K % 2 == 0
        low = w_unsigned[:, 0::2]   # even indices
        high = w_unsigned[:, 1::2]  # odd indices
        packed = low | (high << 4)
        scales = scales.to(torch.bfloat16)
        return packed, scales

    elif bits == 2:
        max_val = 1.0  # range [-2, 1] → 4 levels
        scales = (absmax / 2.0).squeeze(-1)  # scale so max maps to ±2
        w_q = torch.clamp(torch.round(w / (absmax / 2.0)), -2, 1).to(torch.int8)
        w_q = w_q.reshape(N, K)
        # Pack: 4 int2 values per byte (2 bits each)
        # Shift signed [-2,1] to unsigned [0,3] by adding 2
        w_unsigned = (w_q + 2).to(torch.uint8)
        assert K % 4 == 0
        v0 = w_unsigned[:, 0::4]
        v1 = w_unsigned[:, 1::4]
        v2 = w_unsigned[:, 2::4]
        v3 = w_unsigned[:, 3::4]
        packed = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
        scales = scales.to(torch.bfloat16)
        return packed, scales

    else:
        raise ValueError(f"Unsupported bits={bits}")


def dequantize_check(qweight, scales, bits, group_size, original):
    """Verify quantization by dequantizing and computing error."""
    N, K = original.shape
    num_groups = K // group_size

    if bits == 8:
        w_q = qweight.float()
        s = scales.float().unsqueeze(-1).expand(-1, -1, group_size).reshape(N, K)
        recon = w_q * s
    elif bits == 4:
        # Unpack
        low = (qweight & 0x0F).to(torch.int8) - 8
        high = ((qweight >> 4) & 0x0F).to(torch.int8) - 8
        # Interleave back
        w_q = torch.zeros(N, K, dtype=torch.float32)
        w_q[:, 0::2] = low.float()
        w_q[:, 1::2] = high.float()
        s = scales.float().unsqueeze(-1).expand(-1, -1, group_size).reshape(N, K)
        recon = w_q * s
    else:
        raise ValueError

    err = (original.float() - recon).abs()
    return err.mean().item(), err.max().item()


def main():
    parser = argparse.ArgumentParser(description="Quantize model weights")
    parser.add_argument("model_path", help="Path to bf16 model directory")
    parser.add_argument("--bits", type=int, default=8, choices=[2, 4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--output", help="Output directory (default: <model>-W<bits>)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output) if args.output else model_path.parent / f"{model_path.name}-W{args.bits}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Quantizing {model_path} → {output_path}")
    print(f"  bits={args.bits}, group_size={args.group_size}")

    # Copy non-weight files
    for f in model_path.iterdir():
        if f.suffix in (".json", ".txt", ".model", ".tiktoken"):
            shutil.copy2(f, output_path / f.name)

    # Find safetensor files
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files in {model_path}")

    total_params = 0
    quantized_params = 0
    total_mean_err = 0.0
    total_max_err = 0.0
    n_quantized = 0

    # Linear layer weight patterns to quantize
    quant_patterns = [
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
        "qkv_proj.weight", "gate_up_proj.weight",  # merged variants
    ]

    for st_file in st_files:
        print(f"\n  Processing {st_file.name}...")
        tensors = {}
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        output_tensors = {}
        for key, tensor in tensors.items():
            total_params += tensor.numel()

            should_quantize = any(p in key for p in quant_patterns)
            if should_quantize and tensor.dim() == 2:
                N, K = tensor.shape
                if K % args.group_size != 0:
                    print(f"    SKIP {key} [{N}×{K}]: K not divisible by group_size")
                    output_tensors[key] = tensor
                    continue

                qweight, scales = quantize_symmetric(tensor, args.bits, args.group_size)
                mean_err, max_err = dequantize_check(qweight, scales, args.bits, args.group_size, tensor)

                quantized_params += tensor.numel()
                total_mean_err += mean_err
                total_max_err = max(total_max_err, max_err)
                n_quantized += 1

                # Store with naming convention
                output_tensors[key.replace(".weight", ".qweight")] = qweight
                output_tensors[key.replace(".weight", ".scales")] = scales

                size_orig = tensor.numel() * 2  # bf16
                size_quant = qweight.numel() * qweight.element_size() + scales.numel() * 2
                ratio = size_orig / size_quant
                print(f"    {key}: [{N}×{K}] → {args.bits}bit "
                      f"({size_orig/1e6:.1f}→{size_quant/1e6:.1f} MB, {ratio:.1f}x) "
                      f"err: mean={mean_err:.6f} max={max_err:.4f}")
            else:
                output_tensors[key] = tensor

        out_file = output_path / st_file.name
        save_file(output_tensors, str(out_file))
        print(f"    Saved {out_file.name}")

    # Write quantization config
    quant_config = {
        "bits": args.bits,
        "group_size": args.group_size,
        "quant_method": f"symmetric_w{args.bits}a16",
        "quantized_params": quantized_params,
        "total_params": total_params,
    }
    with open(output_path / "quantize_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)

    # Regenerate safetensors index with new tensor names
    index_path = output_path / "model.safetensors.index.json"
    if index_path.exists():
        new_weight_map = {}
        for st_file in sorted(output_path.glob("*.safetensors")):
            with safe_open(str(st_file), framework="pt") as f:
                for key in f.keys():
                    new_weight_map[key] = st_file.name
        index = {"metadata": {"total_size": sum(v.numel() * v.element_size() for v in [])}, "weight_map": new_weight_map}
        # Compute total size properly
        total_size = 0
        for st_file in sorted(output_path.glob("*.safetensors")):
            with safe_open(str(st_file), framework="pt") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    total_size += t.numel() * t.element_size()
        index["metadata"]["total_size"] = total_size
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"  Updated index: {len(new_weight_map)} tensors")

    print(f"\n{'='*60}")
    print(f"Quantization complete: {output_path}")
    print(f"  Quantized: {quantized_params/1e6:.1f}M / {total_params/1e6:.1f}M params "
          f"({quantized_params/total_params*100:.1f}%)")
    if n_quantized > 0:
        print(f"  Mean error: {total_mean_err/n_quantized:.6f}")
        print(f"  Max error:  {total_max_err:.4f}")
    print(f"  Config: {output_path}/quantize_config.json")


if __name__ == "__main__":
    main()

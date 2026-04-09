#!/usr/bin/env python3
"""Repack W4A16 quantized weights to Marlin tile layout for fast prefill.

Converts our symmetric INT4 [N, K/2] packed weights + [N, K/gs] bf16 scales
to Marlin format: tiled int32 weights + permuted fp16 scales.

Based on IST-DASLab/marlin Layer.pack() and vLLM's marlin_utils.py.

Usage:
  python3 scripts/marlin_repack.py models/Qwen3-4B-W4-g128 --output models/Qwen3-4B-W4-g128-marlin
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


def get_perms():
    """Build Marlin permutation tables for MMA-friendly tile layout."""
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)

    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = get_perms()


def repack_weight(qweight_packed: torch.Tensor, scales: torch.Tensor,
                  N: int, K: int, group_size: int):
    """Repack our [N, K/2] uint8 packed weights to Marlin tile layout.

    Args:
        qweight_packed: [N, K/2] uint8, our format (lo=even, hi=odd nibble)
        scales: [N, K/group_size] bfloat16
        N: output dimension (rows)
        K: input dimension (cols)
        group_size: quantization group size

    Returns:
        marlin_packed: [K/16, N*16/8] int32 (Marlin tiled layout)
        marlin_scales: [K/group_size, N] float16 (permuted)
    """
    tile = 16

    # Skip if not Marlin-compatible
    if K % 128 != 0 or N % 64 != 0:
        return None, None

    # Step 1: Unpack our format to [N, K] unsigned int4 values [0, 15]
    lo = (qweight_packed & 0x0F).to(torch.int32)
    hi = ((qweight_packed >> 4) & 0x0F).to(torch.int32)
    w = torch.zeros(N, K, dtype=torch.int32)
    w[:, 0::2] = lo
    w[:, 1::2] = hi  # w[n, k] = unsigned int4 value for weight[n, k]

    # Step 2: Transpose to [K, N] (Marlin convention: weight as [input, output])
    w = w.t().contiguous()  # [K, N]

    # Step 3: Tile reshape + permutation (from original Marlin pack)
    w = w.reshape((K // tile, tile, N // tile, tile))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((K // tile, N * tile))

    # Step 4: Apply MMA permutation
    res = w.reshape((-1, _perm.numel()))[:, _perm].reshape(w.shape)

    # Step 5: Pack 8 x 4-bit values into uint32 with interleave
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res_np = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        q |= res_np[:, i::8] << 4 * i
    marlin_packed = torch.from_numpy(q.astype(np.int32))

    # Step 6: Permute scales
    num_groups = K // group_size
    s = scales.to(torch.float16).t().contiguous()  # [K/gs, N]
    if group_size < K:
        s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
    else:
        s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
    marlin_scales = s.reshape((-1, N)).contiguous()

    return marlin_packed, marlin_scales


def main():
    parser = argparse.ArgumentParser(description="Repack W4 weights to Marlin format")
    parser.add_argument("model_path", help="Path to W4 quantized model")
    parser.add_argument("--output", help="Output directory (default: <model>-marlin)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output) if args.output else model_path.parent / f"{model_path.name}-marlin"
    output_path.mkdir(parents=True, exist_ok=True)

    # Load quantization config
    qconfig_path = model_path / "quantize_config.json"
    if not qconfig_path.exists():
        raise FileNotFoundError(f"No quantize_config.json in {model_path}")
    with open(qconfig_path) as f:
        qconfig = json.load(f)
    bits = qconfig["bits"]
    group_size = qconfig["group_size"]
    assert bits == 4, f"Marlin only supports 4-bit, got {bits}"
    print(f"Repacking {model_path} → {output_path} (bits={bits}, group_size={group_size})")

    # Copy non-weight files
    for f in model_path.iterdir():
        if f.suffix in (".json", ".txt", ".model", ".tiktoken"):
            shutil.copy2(f, output_path / f.name)

    # Process safetensor files
    quant_patterns = [
        "q_proj.qweight", "k_proj.qweight", "v_proj.qweight", "o_proj.qweight",
        "gate_proj.qweight", "up_proj.qweight", "down_proj.qweight",
    ]

    st_files = sorted(model_path.glob("*.safetensors"))
    total_repacked = 0

    for st_file in st_files:
        print(f"  Processing {st_file.name}...")
        tensors = {}
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        output_tensors = {}
        for key, tensor in tensors.items():
            output_tensors[key] = tensor

            if not any(p in key for p in quant_patterns):
                continue

            # Find matching scales
            scales_key = key.replace(".qweight", ".scales")
            if scales_key not in tensors:
                print(f"    SKIP {key}: no matching scales")
                continue

            qw = tensor  # [N, K/2] uint8
            scales = tensors[scales_key]  # [N, K/gs] bf16
            N = qw.shape[0]
            K = qw.shape[1] * 2  # unpacked K

            marlin_packed, marlin_scales = repack_weight(qw, scales, N, K, group_size)
            if marlin_packed is None:
                print(f"    SKIP {key}: [{N}x{K}] not Marlin-aligned")
                continue

            # Store Marlin tensors alongside originals
            marlin_key = key.replace(".qweight", ".marlin_qweight")
            marlin_scales_key = key.replace(".qweight", ".marlin_scales")
            output_tensors[marlin_key] = marlin_packed
            output_tensors[marlin_scales_key] = marlin_scales
            total_repacked += 1

            print(f"    {key}: [{N}x{K}] → Marlin {marlin_packed.shape} + scales {marlin_scales.shape}")

        out_file = output_path / st_file.name
        save_file(output_tensors, str(out_file))

    # Update quantize_config
    qconfig["marlin_repacked"] = True
    with open(output_path / "quantize_config.json", "w") as f:
        json.dump(qconfig, f, indent=2)

    # Regenerate index
    index_path = output_path / "model.safetensors.index.json"
    if index_path.exists():
        new_weight_map = {}
        for sf in sorted(output_path.glob("*.safetensors")):
            with safe_open(str(sf), framework="pt") as f:
                for key in f.keys():
                    new_weight_map[key] = sf.name
        total_size = 0
        for sf in sorted(output_path.glob("*.safetensors")):
            with safe_open(str(sf), framework="pt") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    total_size += t.numel() * t.element_size()
        index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    print(f"\nDone: {total_repacked} weights repacked to Marlin format")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

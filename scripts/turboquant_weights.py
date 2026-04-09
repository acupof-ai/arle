#!/usr/bin/env python3
"""TurboQuant weight quantization: Hadamard rotation + Lloyd-Max 3-bit.

Offline conversion tool: reads BF16/FP16 safetensors weights, applies
randomized Hadamard rotation per output group, then Lloyd-Max quantizes
to 2-4 bits. Saves packed weights + scales in safetensors format.

Usage:
    python scripts/turboquant_weights.py \
        --model-path models/Qwen3-4B \
        --output-path models/Qwen3-4B-TQ3 \
        --bits 3 \
        --group-size 128

Format:
    For each linear weight tensor `model.layers.*.{q,k,v,o,gate,up,down}_proj.weight`:
    - `.tq_packed` : uint8 packed indices, shape [N, packed_K] where packed_K = ceil(K * eff_bits / 8)
    - `.tq_scales` : f16 per-group norms, shape [N, K / group_size]
    - `.tq_signs`  : int8 Hadamard signs, shape [K] (shared per layer group)

    Plus `turboquant_config.json` with format metadata.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def generate_signs(dim: int, seed: int) -> np.ndarray:
    """Generate deterministic random signs {-1, +1} for Hadamard rotation."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=dim).astype(np.float32)


def fwht_numpy(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform along last axis. In-place, unnormalized."""
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"FWHT requires power-of-2 dim, got {n}"
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[..., j].copy()
                b = x[..., j + h].copy()
                x[..., j] = a + b
                x[..., j + h] = a - b
        h *= 2
    x /= math.sqrt(n)
    return x


def lloyd_max_codebook(dim: int, bits: int, max_iters: int = 200) -> tuple:
    """Compute Lloyd-Max optimal codebook for Beta((d-1)/2, (d-1)/2) on [-1,1].

    Returns (centroids, boundaries) as numpy arrays.
    """
    from scipy.special import betainc, beta as betafn
    from scipy.integrate import quad

    alpha = (dim - 1) / 2.0
    num_levels = 1 << bits

    # Beta PDF on [-1, 1]
    def pdf(x):
        if abs(x) >= 1.0:
            return 0.0
        return (1 - x**2) ** (alpha - 1)

    # Initialize centroids at quantile midpoints
    centroids = np.array([-1.0 + (2.0 * (i + 0.5)) / num_levels for i in range(num_levels)])
    boundaries = np.zeros(num_levels + 1)
    boundaries[0] = -1.0
    boundaries[num_levels] = 1.0

    for _ in range(max_iters):
        # Update boundaries
        for i in range(1, num_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        # Update centroids
        for i in range(num_levels):
            a, b = boundaries[i], boundaries[i + 1]
            if b - a < 1e-15:
                continue
            m0, _ = quad(pdf, a, b)
            m1, _ = quad(lambda x: x * pdf(x), a, b)
            if m0 > 1e-30:
                centroids[i] = m1 / m0

    return centroids.astype(np.float32), boundaries.astype(np.float32)


def quantize_tensor(
    weight: np.ndarray,
    bits: int,
    group_size: int,
    centroids: np.ndarray,
    boundaries: np.ndarray,
    seed: int,
) -> tuple:
    """Quantize a weight tensor [N, K] using TurboQuant.

    Returns (packed_indices, scales, signs).
    """
    N, K = weight.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    num_groups = K // group_size

    # Generate signs for this layer's K dimension
    signs = generate_signs(K, seed)

    # Apply sign flip
    rotated = weight * signs[None, :]

    # Apply FWHT per group
    rotated_groups = rotated.reshape(N, num_groups, group_size)
    for g in range(num_groups):
        rotated_groups[:, g, :] = fwht_numpy(rotated_groups[:, g, :].copy())
    rotated = rotated_groups.reshape(N, K)

    # Per-group norm + normalize
    norms = np.linalg.norm(rotated.reshape(N, num_groups, group_size), axis=2)  # [N, num_groups]
    norms = np.maximum(norms, 1e-10)

    # Normalize each group
    rotated_normed = rotated.reshape(N, num_groups, group_size) / norms[:, :, None]
    rotated_normed = rotated_normed.reshape(N, K)

    # Searchsorted quantization
    # boundaries[1:-1] are the interior decision boundaries
    interior = boundaries[1:-1]  # [num_levels - 1]
    indices = np.searchsorted(interior, rotated_normed.ravel()).reshape(N, K)
    indices = np.clip(indices, 0, len(centroids) - 1).astype(np.uint8)

    # Bitpack
    effective_bits = 4 if bits == 3 else bits
    indices_per_byte = 8 // effective_bits
    packed_K = math.ceil(K * effective_bits / 8)

    packed = np.zeros((N, packed_K), dtype=np.uint8)
    for i in range(K):
        byte_idx = i // indices_per_byte
        sub_idx = i % indices_per_byte
        mask = int((1 << effective_bits) - 1)
        packed[:, byte_idx] |= (indices[:, i].astype(np.uint8) & mask) << (sub_idx * effective_bits)

    scales = norms.astype(np.float16)  # [N, num_groups]
    signs_i8 = signs.astype(np.int8)

    return packed, scales, signs_i8


def main():
    parser = argparse.ArgumentParser(description="TurboQuant weight quantization")
    parser.add_argument("--model-path", required=True, help="Path to BF16/FP16 model directory")
    parser.add_argument("--output-path", required=True, help="Output directory for quantized model")
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4], help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=128, help="Per-group quantization size")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"TurboQuant weight quantization: {args.bits}-bit, group_size={args.group_size}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")

    # Copy non-weight files
    for f in model_path.iterdir():
        if f.suffix in (".json", ".txt", ".model", ".tiktoken"):
            import shutil
            shutil.copy2(f, output_path / f.name)
            print(f"  Copied {f.name}")

    # Compute codebook (same for all layers with same group_size)
    print(f"Computing Lloyd-Max codebook for D={args.group_size}, bits={args.bits}...")
    centroids, boundaries = lloyd_max_codebook(args.group_size, args.bits)
    print(f"  Centroids: {centroids}")

    # Process safetensors files
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        print("ERROR: No .safetensors files found")
        return

    linear_suffixes = (".q_proj.weight", ".k_proj.weight", ".v_proj.weight",
                       ".o_proj.weight", ".gate_proj.weight", ".up_proj.weight",
                       ".down_proj.weight")

    total_original = 0
    total_compressed = 0

    for st_file in st_files:
        print(f"\nProcessing {st_file.name}...")
        output_tensors = {}

        with safe_open(st_file, framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if any(key.endswith(s) for s in linear_suffixes):
                    # Quantize this linear weight
                    w = tensor.astype(np.float32)
                    N, K = w.shape

                    if K % args.group_size != 0:
                        print(f"  SKIP {key}: K={K} not divisible by group_size={args.group_size}")
                        output_tensors[key] = torch.from_numpy(tensor)
                        continue

                    # Deterministic seed from tensor name
                    seed = hash(key) & 0xFFFFFFFF

                    packed, scales, signs = quantize_tensor(
                        w, args.bits, args.group_size, centroids, boundaries, seed
                    )

                    original_bytes = N * K * 2  # bf16
                    compressed_bytes = packed.nbytes + scales.nbytes + signs.nbytes
                    ratio = original_bytes / compressed_bytes

                    total_original += original_bytes
                    total_compressed += compressed_bytes

                    print(f"  {key}: [{N}, {K}] → packed={packed.shape} scales={scales.shape} "
                          f"({ratio:.1f}x compression)")

                    base_key = key.replace(".weight", "")
                    output_tensors[f"{base_key}.tq_packed"] = torch.from_numpy(packed)
                    output_tensors[f"{base_key}.tq_scales"] = torch.from_numpy(scales)
                    output_tensors[f"{base_key}.tq_signs"] = torch.from_numpy(signs)
                else:
                    # Keep non-linear tensors as-is
                    output_tensors[key] = torch.from_numpy(tensor)

        # Save quantized safetensors
        out_file = output_path / st_file.name
        save_file(output_tensors, str(out_file))
        print(f"  Saved {out_file.name}")

    # Write config
    config = {
        "quant_type": "turboquant",
        "bits": args.bits,
        "group_size": args.group_size,
        "rotation": "hadamard",
        "centroids": centroids.tolist(),
        "boundaries": boundaries.tolist(),
    }
    config_path = output_path / "turboquant_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if total_original > 0:
        print(f"\nTotal: {total_original / 1e9:.2f} GB → {total_compressed / 1e9:.2f} GB "
              f"({total_original / total_compressed:.1f}x compression)")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()

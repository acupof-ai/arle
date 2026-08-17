#!/usr/bin/env python3
"""Quantize DeepSeek-V4-Flash routed MoE experts to W4A16 (ARLE native format).

Reads the mixed-precision checkpoint (INT8/FP8 expert weights + E8M0 block
scales), dequantizes routed expert weights to float32, RTN-quantizes to 4-bit,
and writes ARLE-native W4A16 tensors (U8 packed weight + BF16 per-group scales).

Shared expert, attention, norms, and embeddings are copied unchanged.

Output format (per expert weight):
  {name}.weight       U8  [N, K//2]  — 2 int4/byte, low nibble = even K
  {name}.weight_scale BF16 [N, K//group_size]

Dequant convention on device: (uint4 - 8) * scale  (zero-point 8 baked in).
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

GROUP_SIZE = 128
SCALE_BLOCK_COLS = 16  # E8M0 scale covers 1 row x 16 cols

# Matches: model.layers.N.mlp.experts.M.w{1,2,3}.weight
EXPERT_WEIGHT_RE = re.compile(r"\.experts\.\d+\.w[123]\.weight$")


def decode_e8m0_scale(scale_bytes: bytes, rows: int, cols: int) -> np.ndarray:
    """Decode E8M0 scale bytes to float32, broadcast to [rows, cols].

    Scale layout: [rows, cols // 16], one scale per 1x16 block.
    E8M0 value = 2^(byte - 127).
    """
    scale_u8 = np.frombuffer(scale_bytes, dtype=np.uint8).reshape(
        rows, cols // SCALE_BLOCK_COLS
    )
    scale_f32 = np.exp2(scale_u8.astype(np.float32) - 127.0)
    return np.repeat(scale_f32, SCALE_BLOCK_COLS, axis=1)


def decode_fp8_e4m3(bytes_arr: np.ndarray) -> np.ndarray:
    """Decode FP8 E4M3 bytes to float32."""
    b = bytes_arr.astype(np.uint8)
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    mant = b & 0x7

    result = np.zeros_like(b, dtype=np.float32)

    normal = (exp > 0) & (exp < 0xF)
    result[normal] = (1.0 + mant[normal] / 8.0) * np.power(
        2.0, exp[normal].astype(np.int32) - 7
    )

    subnormal = (exp == 0) & (mant > 0)
    result[subnormal] = (mant[subnormal] / 8.0) * np.power(2.0, -6)

    # E4M3 NaN (exp=15, mant=7) — clamp to max
    result[(exp == 0xF) & (mant == 0x7)] = 448.0

    result[sign == 1] = -result[sign == 1]
    return result


def decode_weight_to_f32(tensor: torch.Tensor) -> np.ndarray:
    """Decode I8 or F8_E4M3 weight tensor to float32 numpy."""
    if tensor.dtype == torch.int8:
        return tensor.float().numpy()
    if tensor.dtype == torch.uint8:
        return decode_fp8_e4m3(tensor.numpy())
    # torch float8_e4m3fn (if available)
    if hasattr(torch, "float8_e4m3fn") and tensor.dtype == torch.float8_e4m3fn:
        return tensor.float().numpy()
    # Fallback: try direct float conversion
    return tensor.float().numpy()


def read_scale_bytes(f: safe_open, scale_key: str, weight_shape: tuple) -> bytes:
    """Read scale tensor as raw bytes, handling F8_E8M0 and F32."""
    tensor = f.get_tensor(scale_key)
    rows, cols = weight_shape
    if tensor.dtype in (torch.uint8, torch.int8):
        return tensor.numpy().tobytes()
    if tensor.dtype == torch.float32:
        return tensor.numpy().tobytes()
    # F8_E8M0 might come through as a special dtype — try raw bytes
    try:
        return tensor.numpy().tobytes()
    except Exception:
        # Last resort: read the safetensors metadata for raw byte offset
        raise RuntimeError(
            f"Cannot read scale {scale_key} with dtype {tensor.dtype}; "
            "need raw byte access"
        )


def decode_scale(scale_bytes: bytes, scale_dtype: str, rows: int, cols: int) -> np.ndarray:
    """Decode scale bytes to float32, broadcast to [rows, cols].

    Handles F8_E8M0 (1x16 blocks) and F32 (128x128 blocks, power-of-two).
    """
    if scale_dtype == "F8_E8M0":
        return decode_e8m0_scale(scale_bytes, rows, cols)
    if scale_dtype == "F32":
        scale_f32 = np.frombuffer(scale_bytes, dtype=np.float32).reshape(
            rows // 128, cols // 128
        )
        return np.repeat(np.repeat(scale_f32, 128, axis=0), 128, axis=1)
    raise ValueError(f"Unsupported scale dtype: {scale_dtype}")


def quantize_w4a16(weight_f32: np.ndarray, group_size: int = GROUP_SIZE):
    """RTN quantize float32 [N, K] to ARLE W4A16.

    Returns:
        packed: U8 [N, K//2], low nibble = even K
        scales: BF16 [N, K//group_size]
    """
    rows, cols = weight_f32.shape
    assert cols % group_size == 0, f"cols {cols} not divisible by group_size {group_size}"
    num_groups = cols // group_size

    w_grouped = weight_f32.reshape(rows, num_groups, group_size)
    amax = np.max(np.abs(w_grouped), axis=2, keepdims=True)
    amax = np.maximum(amax, 1e-12)

    scales = amax / 7.0
    q = np.round(w_grouped / scales).astype(np.int32)
    q = np.clip(q, -7, 7)
    u = (q + 8).astype(np.uint8)

    u_flat = u.reshape(rows, cols)
    packed = np.zeros((rows, cols // 2), dtype=np.uint8)
    packed[:, :] = u_flat[:, 0::2] | (u_flat[:, 1::2] << 4)

    scales_out = scales.squeeze(axis=2)
    return packed, scales_out


def process_file(st_file: Path, output_path: Path) -> int:
    """Process one safetensors shard. Returns expert count quantized."""
    tensors = {}
    expert_count = 0

    with safe_open(str(st_file), framework="pt") as f:
        keys = set(f.keys())
        for key in sorted(keys):
            tensor = f.get_tensor(key)

            if not EXPERT_WEIGHT_RE.search(key):
                tensors[key] = tensor
                continue

            # Routed expert weight — quantize to W4A16
            scale_key = key[: -len(".weight")] + ".scale"
            if scale_key not in keys:
                print(f"  WARNING: {key} missing {scale_key}, copying unchanged")
                tensors[key] = tensor
                continue

            rows, cols = tensor.shape
            scale_meta = f.get_slice(scale_key)
            scale_dtype = scale_meta.get_dtype()

            scale_bytes = read_scale_bytes(f, scale_key, (rows, cols))
            scale_f32 = decode_scale(scale_bytes, scale_dtype, rows, cols)

            weight_f32 = decode_weight_to_f32(tensor)
            dequant = weight_f32 * scale_f32

            packed, scales_out = quantize_w4a16(dequant)

            base = key[: -len(".weight")]
            tensors[key] = torch.from_numpy(packed)
            tensors[f"{base}.weight_scale"] = torch.from_numpy(scales_out).to(
                torch.bfloat16
            )
            expert_count += 1

    out_file = output_path / st_file.name
    save_file(tensors, str(out_file), metadata={"format": "pt"})
    return expert_count


def main():
    parser = argparse.ArgumentParser(
        description="Quantize DSv4 routed MoE experts to ARLE W4A16"
    )
    parser.add_argument("input_dir", help="Input checkpoint directory")
    parser.add_argument("output_dir", help="Output checkpoint directory")
    parser.add_argument(
        "--group-size", type=int, default=GROUP_SIZE, help="W4A16 group size"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    st_files = sorted(input_path.glob("*.safetensors"))
    if not st_files:
        print(f"No safetensors files in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(st_files)} shards from {input_path}")
    total_experts = 0
    for st_file in st_files:
        print(f"  {st_file.name}...", end=" ", flush=True)
        n = process_file(st_file, output_path)
        total_experts += n
        print(f"{n} experts quantized")

    # Update config.json
    config_src = input_path / "config.json"
    if config_src.exists():
        with open(config_src) as f:
            config = json.load(f)
        config["quantization_config"] = {
            "quant_method": "w4a16",
            "bits": 4,
            "group_size": args.group_size,
            "sym": True,
            "desc_act": False,
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("Updated config.json with quantization_config")

    # Copy non-safetensors files (index.json, tokenizer, etc.)
    for f in input_path.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "config.json":
            shutil.copy2(f, output_path / f.name)

    print(f"\nDone: {total_experts} expert weights quantized -> {output_path}")


if __name__ == "__main__":
    main()

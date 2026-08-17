#!/usr/bin/env python3
"""Quantize DeepSeek-V4-Flash routed MoE experts to W4AFP8 (SGLang CUTLASS format).

Reads the mixed-precision checkpoint (INT8/FP8 expert weights + E8M0 block
scales), dequantizes routed expert weights to float32, RTN-quantizes to
signed INT4 (two's complement), and writes SGLang-compatible tensors:

  {name}.weight       int8 [N, K//2]  — 2 signed int4/byte, low nibble = even K
  {name}.weight_scale BF16 [K//512, N*4] — interleaved per 512-K chunk

The interleaved scale layout matches the SGLang CUTLASS W4A8 MoE kernel
(github.com/sgl-project/sglang PR #7772, Apache-2.0):
  [N, K//128] → reshape [N, K//512, 4] → permute [K//512, N, 4] → [K//512, N*4]

Shared expert, attention, norms, and embeddings are copied unchanged.
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

EXPERT_WEIGHT_RE = re.compile(r"\.experts\.\d+\.w[123]\.weight$")
EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def decode_e8m0_scale(scale_bytes: bytes, rows: int, cols: int) -> np.ndarray:
    scale_u8 = np.frombuffer(scale_bytes, dtype=np.uint8).reshape(
        rows, cols // SCALE_BLOCK_COLS
    )
    scale_f32 = np.exp2(scale_u8.astype(np.float32) - 127.0)
    return np.repeat(scale_f32, SCALE_BLOCK_COLS, axis=1)


def decode_fp8_e4m3(bytes_arr: np.ndarray) -> np.ndarray:
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
    result[(exp == 0xF) & (mant == 0x7)] = 448.0
    result[sign == 1] = -result[sign == 1]
    return result


def decode_weight_to_f32(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.int8:
        return tensor.float().numpy()
    if tensor.dtype == torch.uint8:
        return decode_fp8_e4m3(tensor.numpy())
    if hasattr(torch, "float8_e4m3fn") and tensor.dtype == torch.float8_e4m3fn:
        return tensor.float().numpy()
    return tensor.float().numpy()


def read_scale_bytes(f: safe_open, scale_key: str) -> bytes:
    tensor = f.get_tensor(scale_key)
    if tensor.dtype in (torch.uint8, torch.int8, torch.float32):
        return tensor.numpy().tobytes()
    if tensor.dtype == torch.float8_e8m0fnu:
        return tensor.view(torch.uint8).numpy().tobytes()
    raise RuntimeError(f"Cannot read scale {scale_key} with dtype {tensor.dtype}")


def decode_scale(scale_bytes: bytes, scale_dtype: str, rows: int, cols: int) -> np.ndarray:
    if scale_dtype == "F8_E8M0":
        return decode_e8m0_scale(scale_bytes, rows, cols)
    if scale_dtype == "F32":
        scale_f32 = np.frombuffer(scale_bytes, dtype=np.float32).reshape(
            rows // 128, cols // 128
        )
        return np.repeat(np.repeat(scale_f32, 128, axis=0), 128, axis=1)
    raise ValueError(f"Unsupported scale dtype: {scale_dtype}")


def quantize_w4afp8(weight_f32: np.ndarray, group_size: int = GROUP_SIZE):
    """RTN quantize float32 [N, K] to SGLang W4AFP8.

    Returns:
        packed: int8 [N, K//2], signed int4 two's complement, low nibble = even K
        scales_interleaved: BF16 [K//512, N*4]
    """
    rows, cols = weight_f32.shape
    assert cols % group_size == 0
    num_groups = cols // group_size

    w_grouped = weight_f32.reshape(rows, num_groups, group_size)
    amax = np.max(np.abs(w_grouped), axis=2, keepdims=True)
    amax = np.maximum(amax, 1e-12)

    scales = amax / 8.0  # signed int4 range [-8, 7]
    q = np.round(w_grouped / scales).astype(np.int32)
    q = np.clip(q, -8, 7)

    # Pack signed int4: low nibble = even K, high nibble = odd K
    q_flat = q.reshape(rows, cols).astype(np.int8)
    packed = (q_flat[:, 0::2].astype(np.uint8) & 0x0F) | (
        (q_flat[:, 1::2].astype(np.uint8) & 0x0F) << 4
    )

    # Interleave scales: [N, K//128] → [K//512, N*4]
    scales_squeezed = scales.squeeze(axis=2)  # [N, K//128]
    assert num_groups % 4 == 0, f"num_groups {num_groups} not divisible by 4"
    scales_reshaped = scales_squeezed.reshape(rows, num_groups // 4, 4)
    scales_permuted = scales_reshaped.transpose(1, 0, 2)  # [K//512, N, 4]
    scales_interleaved = scales_permuted.reshape(
        num_groups // 4, rows * 4
    )  # [K//512, N*4]

    return packed, scales_interleaved


def process_file(st_file: Path, output_path: Path) -> int:
    tensors = {}
    expert_count = 0

    with safe_open(str(st_file), framework="pt") as f:
        keys = set(f.keys())
        for key in sorted(keys):
            if EXPERT_SCALE_RE.search(key):
                continue

            tensor = f.get_tensor(key)

            if not EXPERT_WEIGHT_RE.search(key):
                tensors[key] = tensor
                continue

            scale_key = key[: -len(".weight")] + ".scale"
            if scale_key not in keys:
                print(f"  WARNING: {key} missing {scale_key}, copying unchanged")
                tensors[key] = tensor
                continue

            rows, cols = tensor.shape
            scale_meta = f.get_slice(scale_key)
            scale_dtype = scale_meta.get_dtype()

            scale_bytes = read_scale_bytes(f, scale_key)
            scale_f32 = decode_scale(scale_bytes, scale_dtype, rows, cols)

            weight_f32 = decode_weight_to_f32(tensor)
            dequant = weight_f32 * scale_f32

            packed, scales_out = quantize_w4afp8(dequant)

            base = key[: -len(".weight")]
            # Signed INT4 two's complement: store as I8 so the loader's W4AFP8
            # detection (I8+BF16 scale) fires, not the W4A16 (U8) branch.
            tensors[key] = torch.from_numpy(packed).view(torch.int8)
            tensors[f"{base}.weight_scale"] = torch.from_numpy(scales_out).to(
                torch.bfloat16
            )
            expert_count += 1

    out_file = output_path / st_file.name
    save_file(tensors, str(out_file), metadata={"format": "pt"})
    return expert_count


def main():
    parser = argparse.ArgumentParser(
        description="Quantize DSv4 routed MoE experts to W4AFP8 (SGLang CUTLASS format)"
    )
    parser.add_argument("input_dir", help="Input checkpoint directory")
    parser.add_argument("output_dir", help="Output checkpoint directory")
    parser.add_argument(
        "--group-size", type=int, default=GROUP_SIZE, help="W4AFP8 group size"
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

    config_src = input_path / "config.json"
    if config_src.exists():
        with open(config_src) as f:
            config = json.load(f)
        config["quantization_config"] = {
            "quant_method": "w4afp8",
            "bits": 4,
            "group_size": args.group_size,
            "sym": True,
            "desc_act": False,
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("Updated config.json with quantization_config")

    for f in input_path.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "config.json":
            shutil.copy2(f, output_path / f.name)

    print(f"\nDone: {total_experts} expert weights quantized -> {output_path}")


if __name__ == "__main__":
    main()

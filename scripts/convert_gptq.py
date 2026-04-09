#!/usr/bin/env python3
"""Convert GPTQ-Int4 model to our internal quantized format.

GPTQ stores weights as [K/8, N] packed int32 (column-major pack, 8 int4 per int32).
Our format: [N, K/2] packed uint8 (row-major pack, 2 int4 per byte) + [N, K/group_size] bf16 scales.

Usage:
  python3 scripts/convert_gptq.py models/Qwen3-4B-GPTQ-Int4 --output models/Qwen3-4B-GPTQ-converted
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def unpack_gptq_weight(qweight, qzeros, scales, bits=4, group_size=128):
    """Unpack GPTQ format to our internal format.

    GPTQ layout:
      qweight: [K // (32//bits), N] int32  — 8 int4 values packed per int32
      qzeros:  [num_groups, N // (32//bits)] int32 — packed zero-points
      scales:  [num_groups, N] float16

    Output (our format):
      packed_weight: [N, K//2] uint8  — 2 int4 per byte, row-major
      scales_out:    [N, num_groups] bfloat16
    """
    vals_per_int32 = 32 // bits  # 8 for 4-bit
    K = qweight.shape[0] * vals_per_int32
    N = qweight.shape[1]
    num_groups = scales.shape[0]

    # Unpack qweight: [K//8, N] int32 → [K, N] int4 values
    weight_unpacked = torch.zeros(K, N, dtype=torch.int32)
    for i in range(vals_per_int32):
        weight_unpacked[i::vals_per_int32] = (qweight >> (bits * i)) & ((1 << bits) - 1)

    # Unpack qzeros: [num_groups, N//8] int32 → [num_groups, N] int4 values
    zeros_unpacked = torch.zeros(num_groups, N, dtype=torch.int32)
    for i in range(vals_per_int32):
        zeros_unpacked[:, i::vals_per_int32] = (qzeros >> (bits * i)) & ((1 << bits) - 1)

    # Dequantize to verify: w_float = (w_int - zero) * scale
    # For symmetric GPTQ: zero = 2^(bits-1) = 8 for 4-bit
    # w_float[k, n] = (weight_unpacked[k, n] - zeros_unpacked[g, n]) * scales[g, n]
    # where g = k // group_size

    # Repack to our format: [N, K//2] uint8, signed symmetric [-8, 7]
    # Our format: (val + 8) stored as uint8, low nibble first
    # GPTQ unsigned [0, 15] → our signed: val_signed = val_gptq - zero_point

    weight_signed = torch.zeros(N, K, dtype=torch.int8)
    scales_out = torch.zeros(N, num_groups, dtype=torch.bfloat16)

    for g in range(num_groups):
        k_start = g * group_size
        k_end = min(k_start + group_size, K)
        for n in range(N):
            zp = zeros_unpacked[g, n].item()
            sc = scales[g, n].item()
            scales_out[n, g] = sc  # transpose: [num_groups, N] → [N, num_groups]
            for k in range(k_start, k_end):
                val = weight_unpacked[k, n].item()
                # GPTQ: w_float = (val - zp) * scale
                # Our kernel: w_float = w_signed * scale (symmetric)
                # So: w_signed = val - zp, clamped to [-8, 7]
                w_s = max(-8, min(7, val - zp))
                weight_signed[n, k] = w_s

    # Pack to uint8: 2 int4 per byte (low nibble = even index, high nibble = odd)
    weight_unsigned = (weight_signed + 8).to(torch.uint8)  # [0, 15]
    packed = torch.zeros(N, K // 2, dtype=torch.uint8)
    packed = weight_unsigned[:, 0::2] | (weight_unsigned[:, 1::2] << 4)

    return packed, scales_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to GPTQ model directory")
    parser.add_argument("--output", help="Output directory")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output) if args.output else model_path.parent / f"{model_path.name}-converted"
    output_path.mkdir(parents=True, exist_ok=True)

    # Read GPTQ config
    with open(model_path / "quantize_config.json") as f:
        gptq_config = json.load(f)
    bits = gptq_config["bits"]
    group_size = gptq_config["group_size"]
    sym = gptq_config.get("sym", True)
    print(f"GPTQ config: bits={bits}, group_size={group_size}, sym={sym}")

    # Copy non-weight files
    for f in model_path.iterdir():
        if f.suffix in (".json", ".txt", ".model", ".tiktoken", ".jinja"):
            shutil.copy2(f, output_path / f.name)

    # Process safetensors
    st_files = sorted(model_path.glob("*.safetensors"))
    quant_patterns = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    for st_file in st_files:
        print(f"\nProcessing {st_file.name}...")
        tensors = {}
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        output_tensors = {}
        processed_layers = set()

        for key, tensor in tensors.items():
            # Check if this is a GPTQ layer (has .qweight suffix)
            if key.endswith(".qweight"):
                layer_prefix = key.replace(".qweight", "")
                if layer_prefix in processed_layers:
                    continue
                processed_layers.add(layer_prefix)

                # Check if it's a quantized linear layer
                is_quant_linear = any(p in layer_prefix for p in quant_patterns)
                if not is_quant_linear:
                    # Keep as-is
                    for suffix in [".qweight", ".qzeros", ".scales", ".g_idx"]:
                        k = layer_prefix + suffix
                        if k in tensors:
                            output_tensors[k] = tensors[k]
                    continue

                qweight = tensors[layer_prefix + ".qweight"]
                qzeros = tensors[layer_prefix + ".qzeros"]
                scales_t = tensors[layer_prefix + ".scales"]

                print(f"  Converting {layer_prefix}: qweight={qweight.shape}, scales={scales_t.shape}")
                packed, scales_out = unpack_gptq_weight(qweight, qzeros, scales_t, bits, group_size)

                # Store in our naming convention
                weight_name = layer_prefix + ".qweight"
                scales_name = layer_prefix + ".scales"
                output_tensors[weight_name] = packed
                output_tensors[scales_name] = scales_out

                vals_per_int32 = 32 // bits
                K = qweight.shape[0] * vals_per_int32
                N = qweight.shape[1]
                print(f"    → [{N}x{K}] INT4, packed [{N}x{K//2}], scales [{N}x{scales_out.shape[1]}]")

            elif not any(key.endswith(s) for s in [".qzeros", ".scales", ".g_idx"]):
                # Non-quantized tensor (layernorm, embed, etc.)
                output_tensors[key] = tensor

        out_file = output_path / st_file.name
        save_file(output_tensors, str(out_file))
        print(f"  Saved {out_file.name} ({len(output_tensors)} tensors)")

    # Write our quantize_config
    our_config = {
        "bits": bits,
        "group_size": group_size,
        "quant_method": f"gptq_w{bits}a16",
        "source": "GPTQ converted",
    }
    with open(output_path / "quantize_config.json", "w") as f:
        json.dump(our_config, f, indent=2)

    # Update index
    new_weight_map = {}
    for st_file in sorted(output_path.glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt") as f:
            for key in f.keys():
                new_weight_map[key] = st_file.name
    total_size = 0
    for st_file in sorted(output_path.glob("*.safetensors")):
        total_size += st_file.stat().st_size
    index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nConversion complete: {output_path}")
    print(f"  {len(new_weight_map)} tensors, {total_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert AutoGPTQ W4A16 (qweight/scales/qzeros) to ARLE W4A16 (weight/weight_scale).

GPTQ packing: each uint32 holds 8 uint4 values, bits [0:4]=val0, [4:8]=val1, ...
ARLE packing: each uint8 holds 2 uint4 values, low nibble=even, high nibble=odd.

For symmetric quantization (sym=True, qzeros=8), dequant = (uint4 - 8) * scale
in both formats — only the packing layout differs.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.tensor import save_file


def gptq_unpack_qweight(qweight: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Unpack GPTQ qweight [rows, cols//8] uint32 -> [rows, cols] uint4 (0-15)."""
    # qweight shape: [rows, cols // 8], dtype uint32
    flat = qweight.reshape(-1)
    out = np.zeros(rows * cols, dtype=np.uint8)
    for i in range(8):
        shift = i * 4
        vals = ((flat >> shift) & 0xF).astype(np.uint8)
        out[i::8] = vals
    return out.reshape(rows, cols)


def arle_pack_weight(uint4: np.ndarray) -> np.ndarray:
    """Pack [rows, cols] uint4 -> [rows, cols//2] uint8 (low nibble first)."""
    rows, cols = uint4.shape
    assert cols % 2 == 0
    flat = uint4.reshape(-1)
    lo = flat[0::2].astype(np.uint8) & 0x0F
    hi = flat[1::2].astype(np.uint8) & 0x0F
    packed = lo | (hi << 4)
    return packed.reshape(rows, cols // 2)


def convert_model(src_dir: Path, dst_dir: Path):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    with open(src_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Collect all shard files
    shards = sorted(set(weight_map.values()))

    # First pass: figure out which tensors are W4A16 (have qweight)
    w4a16_bases = set()
    all_keys = set(weight_map.keys())
    for k in all_keys:
        if k.endswith(".qweight"):
            base = k[: -len(".qweight")]
            w4a16_bases.add(base)

    print(f"Found {len(w4a16_bases)} W4A16 tensors")

    # Convert each shard
    for shard in shards:
        print(f"Processing {shard}...")
        out_tensors = {}
        with safe_open(src_dir / shard, framework="numpy") as f:
            keys = list(f.keys())
            for k in keys:
                if k.endswith(".qweight"):
                    base = k[: -len(".qweight")]
                    qweight = f.get_tensor(k)  # uint32 [rows, cols//8]
                    scales = f.get_tensor(f"{base}.scales")  # bf16 [rows, cols//group_size]
                    # qzeros may not be present for sym=True; if present, check it's 8
                    qzeros_key = f"{base}.qzeros"
                    if qzeros_key in keys:
                        qzeros = f.get_tensor(qzeros_key)
                        # For sym quantization, all zeros should be 8
                        if not np.all((qzeros & 0xF) == 8):
                            print(f"  WARNING: {base} has non-8 qzeros (asymmetric), "
                                  f"ARLE W4A16 assumes sym (zero=8)")
                    rows, packed_cols = qweight.shape
                    cols = packed_cols * 8
                    # Unpack GPTQ -> uint4
                    uint4 = gptq_unpack_qweight(qweight, rows, cols)
                    # Repack to ARLE format
                    arle_weight = arle_pack_weight(uint4)
                    out_tensors[f"{base}.weight"] = arle_weight
                    # scales: ARLE expects BF16, same shape
                    out_tensors[f"{base}.weight_scale"] = scales
                    print(f"  {base}: {rows}x{cols}, group_size={cols // scales.shape[1]}")
                elif k.endswith(".scales") or k.endswith(".qzeros"):
                    # Skip — handled with qweight
                    continue
                else:
                    out_tensors[k] = f.get_tensor(k)

        save_file(out_tensors, dst_dir / shard)
        print(f"  saved {dst_dir / shard} ({len(out_tensors)} tensors)")

    # Copy non-safetensors files
    for f in src_dir.iterdir():
        if f.suffix not in (".safetensors",) and f.is_file():
            dst = dst_dir / f.name
            if not dst.exists():
                dst.write_bytes(f.read_bytes())
                print(f"  copied {f.name}")

    # Update index: remove .qweight/.scales/.qzeros, add .weight/.weight_scale
    new_weight_map = {}
    for k, v in weight_map.items():
        if k.endswith(".qweight"):
            base = k[: -len(".qweight")]
            new_weight_map[f"{base}.weight"] = v
        elif k.endswith(".scales"):
            base = k[: -len(".scales")]
            new_weight_map[f"{base}.weight_scale"] = v
        elif k.endswith(".qzeros"):
            continue
        else:
            new_weight_map[k] = v

    index["weight_map"] = new_weight_map
    with open(dst_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Updated index: {len(new_weight_map)} tensors")

    # Remove quantization_config.json if present (ARLE detects format from tensor names)
    qcfg = dst_dir / "quantization_config.json"
    if qcfg.exists():
        qcfg.unlink()
        print("Removed quantization_config.json")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <src_dir> <dst_dir>")
        sys.exit(1)
    convert_model(sys.argv[1], sys.argv[2])

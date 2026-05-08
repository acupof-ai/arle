#!/usr/bin/env python3
"""
Qwen3 W4A8 Marlin packer for ARLE.

Converts local BF16 safetensors into ARLE's W4A8 side-tensor convention:

  <name>.marlin_w4a8_qweight   int32 Marlin-packed INT4, raw-loaded as bytes
  <name>.marlin_w4a8_s_channel float32 [1, out_features]
  <name>.marlin_w4a8_s_group   float16 [in_features / 128, out_features]

Status (2026-05-08 EOD+27):
  Investigation of W4A8 100%-token-diff bug landed multiple iterations on
  this packing logic:
    - H3  (25391f3): row stride 4-consecutive → 2-skip-8. Later
                     diagnostics found this compared against PR #31's
                     plain Layer path, not W4A8Layer; W4A8Layer uses
                     4-consecutive rows.
    - H3b (3479a87): apply scale_perm_single to s_channel (was deleted)
    - H3c (4dea952): defer scale_perm_single until after division. This
                     was temporarily reverted, then reinstated after the
                     pack round-trip diagnostic isolated the remaining
                     scale-chain asymmetry.
    - H4  (592779a): redundant `s_pack = s.t()` mis-aligns broadcast during
                     quant division; current canonical = remove transpose.

Kernel (crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu) and Rust FFI
wiring audited 0-diff vs PR #31 reference (per 01ace86).

Usage:
  python scripts/quantize_qwen3_w4a8.py \
    --src infer/models/Qwen3-4B \
    --dst infer/models/Qwen3-4B-W4A8-marlin
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


GROUP_SIZE = 128


def get_perms(groupsize: int, k: int):
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            # PR #31 W4A8Layer._get_perms uses the 4-consecutive row pattern.
            # Do not use the top-level Layer._get_perms skip-8 pattern here;
            # that path is for the non-W4A8 Marlin layer.
            for row in [
                4 * (i % 4),
                4 * (i % 4) + 1,
                4 * (i % 4) + 2,
                4 * (i % 4) + 3,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    if groupsize == k:
        interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
    else:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return torch.from_numpy(perm), scale_perm, scale_perm_single


def is_quantized_linear(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2 or not name.endswith(".weight"):
        return False
    if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
        return False
    out_features, in_features = tensor.shape
    return in_features % 128 == 0 and out_features % 256 == 0


def pack_w4a8(weight: torch.Tensor, groupsize: int = GROUP_SIZE,
              gptq_scales: torch.Tensor | None = None):
    """Pack BF16/FP16 weight to ARLE W4A8 Marlin format.

    Args:
      weight: tensor of shape (n, k) = (out_features, in_features)
      groupsize: per-group quant size (default 128)
      gptq_scales: optional upstream GPTQ scales of shape (n, k/groupsize) BF16/FP16.
        When provided, pack uses these instead of re-deriving max-scale from data,
        preserving GPTQ calibration through re-pack. Set to None for plain naive
        max-scale quant of FP weights (default behavior).
    """
    weight = weight.to(dtype=torch.float16, device="cpu").contiguous()
    n, k = weight.shape
    if k % 128 != 0 or n % 256 != 0 or k % groupsize != 0:
        raise ValueError(f"unsupported W4A8 shape [{n}, {k}] groupsize={groupsize}")

    perm, scale_perm, scale_perm_single = get_perms(groupsize, k)

    ref = weight.t().contiguous()
    s_channel = ref.t().abs().amax(dim=-1, keepdim=True).div(127.0).to(torch.float32)
    s_channel = torch.where(s_channel == 0, torch.ones_like(s_channel), s_channel)
    s_channel = s_channel.reshape(1, n)

    if gptq_scales is not None:
        # GPTQ-aware path: use upstream calibrated scales directly.
        # GPTQ stores (n, k/groupsize); transpose to (k/groupsize, n) to match
        # our internal s layout (k/gs first, n second matches w's flat order).
        s = gptq_scales.t().to(torch.float16).contiguous()
        if s.shape != (k // groupsize, n):
            raise ValueError(
                f"gptq_scales shape after transpose {tuple(s.shape)} "
                f"!= expected ({k // groupsize}, {n})"
            )
        # Fix A per b255828 (revised): kernel dequant_per_group uses
        # MAGIC_NUM=0x6480 (FP16 1152) IEEE-754 fast-path. Result range
        # required: [1024, 1280) → (q-8)*s_group_stored ∈ [-128, 128).
        # For q=0 (t0=-8): s_group_stored ≤ 16 (TIGHTER than 127/7).
        # For q=15 (t0=7): s_group_stored < 18.286.
        # Effective bound: s_group_stored ≤ 16 (binding constraint = q=0 case).
        # Equivalent: s ≤ 16 * s_channel per element.
        max_s = 16.0 * s_channel.to(torch.float16).reshape(1, n)
        s = torch.minimum(s, max_s)
    else:
        reshaped = ref.reshape(k // groupsize, groupsize, n)
        s = reshaped.abs().amax(dim=1).clamp_min(1e-6).div(7.0).to(torch.float16)
    # H4: do NOT transpose s here. s has shape (k/gs, n) which matches w's
    # flat-index order (i_kgs * n + i_n) after permute+reshape. Adding .t()
    # would rotate to (n, k/gs) and reshape((1,-1)) would flatten with
    # wrong order (i_n*(k/gs)+i_kgs), mis-aligning broadcast division.

    w = ref.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))
    s_work = s.reshape((1, -1))
    w = torch.round(w / s_work).to(torch.int32)
    w += 8
    w = torch.clamp(w, 0, 15)

    s_group = (s_work.reshape(-1, n) / s_channel).to(torch.float16)
    w = w.reshape((groupsize, -1, n)).permute(1, 0, 2).reshape((k, n)).contiguous()
    s_group = s_group.reshape((-1, len(scale_perm)))[:, scale_perm]
    s_group = s_group.reshape((-1, n)).contiguous()
    # PR #31 W4A8Layer.pack divides group scales by raw per-channel scales,
    # then stores the per-channel scales in the kernel's permuted layout.
    s_channel = s_channel.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s_channel = s_channel.reshape((-1, n)).contiguous()

    tile = 16
    w = w.reshape((k // tile, tile, n // tile, tile))
    w = w.permute((0, 2, 1, 3)).reshape((k // tile, n * tile))
    res = w.reshape((-1, perm.numel()))[:, perm].reshape(w.shape)
    res_np = res.cpu().numpy().astype(np.uint32)
    q = np.zeros((res_np.shape[0], res_np.shape[1] // 8), dtype=np.uint32)
    for i in range(8):
        q |= res_np[:, i::8] << (4 * i)
    qweight = torch.from_numpy(q.astype(np.int32))
    return qweight, s_channel.contiguous(), s_group.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    args = parser.parse_args()

    src = args.src
    dst = args.dst
    dst.mkdir(parents=True, exist_ok=True)

    for pattern in ["*.json", "*.model", "*.txt", "tokenizer*"]:
        for path in src.glob(pattern):
            if path.name == "model.safetensors.index.json":
                continue
            target = dst / path.name
            if path.is_file() and not target.exists():
                shutil.copy2(path, target)

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = sorted(p.name for p in src.glob("*.safetensors"))

    out_tensors = {}
    converted = 0
    for shard in shard_names:
        tensors = load_file(src / shard, device="cpu")
        for name, tensor in tensors.items():
            if is_quantized_linear(name, tensor):
                qweight, s_channel, s_group = pack_w4a8(tensor)
                prefix = name[: -len(".weight")]
                out_tensors[f"{prefix}.marlin_w4a8_qweight"] = qweight
                out_tensors[f"{prefix}.marlin_w4a8_s_channel"] = s_channel
                out_tensors[f"{prefix}.marlin_w4a8_s_group"] = s_group
                converted += 1
            else:
                out_tensors[name] = tensor

    save_file(out_tensors, dst / "model.safetensors")
    stale_index = dst / "model.safetensors.index.json"
    if stale_index.exists():
        stale_index.unlink()

    config_path = dst / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    config["quantization_config"] = {
        "quant_type": "marlin_w4a8",
        "group_size": GROUP_SIZE,
    }
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"converted {converted} linear tensors to {dst}")


if __name__ == "__main__":
    main()

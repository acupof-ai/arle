#!/usr/bin/env python3
"""Convert GPTQ-W4A16-Marlin checkpoint → ARLE W4A8-Marlin format.

Per codex `8bb57ea` correction to da19d71 Phase 0:re-pack from the
ORIGINAL `*.qweight` [N, K/2] U8(pre-Marlin GPTQ-calibrated weights),
NOT from `*.marlin_qweight`(W4A16-perm bytes,wrong layout for W4A8)。

Decoded GPTQ weights pass through ARLE's `pack_w4a8`(scripts/quantize_qwen3_w4a8.py)
which uses W4A8 4-consecutive perms。Calibration preserved because pack_w4a8's
naive max-scale recovers the same integer levels when applied to weights
already at GPTQ-quantized values(integer multiples of GPTQ scale)。

Usage:
  python scripts/convert_gptq_w4a16_to_w4a8_marlin.py \\
    --src infer/models/Qwen3-4B-GPTQ-Int4-marlin \\
    --dst infer/models/Qwen3-4B-GPTQ-W4A8-marlin

Codex KILL criteria(see `8bb57ea`):
  - re-quant noise > 5% on diag → fall back to AutoGPTQ-direct
  - kernel still token-diff with re-packed weights → bug in scale split
  - no `*.qweight` in source → no shortcut,re-quantize via AutoGPTQ
"""

from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path

import torch

from convert import load_all_tensors, save_checkpoint, copy_config_files


def load_pack_w4a8():
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "qpack", repo_root / "scripts" / "quantize_qwen3_w4a8.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.pack_w4a8


def repack_w4a16_to_w4a8(qweight_u8, scales_bf16, groupsize: int, pack_w4a8):
    """Decode GPTQ U8 qweight → BF16 weights → re-pack as W4A8."""
    n, k_half = qweight_u8.shape
    k = k_half * 2

    lo = (qweight_u8 & 0x0F).to(torch.int32)
    hi = ((qweight_u8 >> 4) & 0x0F).to(torch.int32)
    w_int = torch.zeros(n, k, dtype=torch.int32)
    w_int[:, 0::2] = lo
    w_int[:, 1::2] = hi

    scales_per_element = scales_bf16.repeat_interleave(groupsize, dim=1)
    w_real = (w_int - 8).float() * scales_per_element.float()
    # Pass GPTQ scales through to pack_w4a8 to preserve calibration. Without
    # this, pack re-derives s_pack = max/7 from data which drifts ~4% on
    # boundary groups (per b7176d3 empirical). Pass-through gives near-zero
    # drift since w_real lives at integer multiples of scales_bf16 already.
    return pack_w4a8(w_real.to(torch.bfloat16), groupsize=groupsize, gptq_scales=scales_bf16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--groupsize", type=int, default=128)
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"src not found: {args.src}")

    pack_w4a8 = load_pack_w4a8()
    tensors = load_all_tensors(args.src)

    new_state: dict[str, torch.Tensor] = {}
    n_repacked = 0
    n_passthrough = 0

    for k, t in tensors.items():
        if k.endswith(".qweight"):
            base = k[:-len(".qweight")]
            scales_key = f"{base}.scales"
            if scales_key not in tensors:
                print(f"  skip {base}: missing {scales_key}")
                continue
            qweight, s_channel, s_group = repack_w4a16_to_w4a8(
                t, tensors[scales_key], args.groupsize, pack_w4a8
            )
            new_state[f"{base}.marlin_w4a8_qweight"] = qweight
            new_state[f"{base}.marlin_w4a8_s_channel"] = s_channel
            new_state[f"{base}.marlin_w4a8_s_group"] = s_group
            n_repacked += 1
            if n_repacked == 1:
                print(f"  first re-pack: {base} → qweight={list(qweight.shape)} "
                      f"s_channel={list(s_channel.shape)} s_group={list(s_group.shape)}")
        elif k.endswith((".scales", ".marlin_qweight", ".marlin_scales", ".g_idx", ".qzeros")):
            continue  # consumed or W4A16-only intermediate
        else:
            new_state[k] = t
            n_passthrough += 1

    print(f"\n{n_repacked} layers re-packed, {n_passthrough} tensors passthrough")

    copy_config_files(args.src, args.dst)
    save_checkpoint(
        new_state, args.dst,
        quant_config={"quant_type": "marlin_w4a8", "group_size": args.groupsize},
    )
    print(f"patched config.json with quant_type=marlin_w4a8")

    # Do NOT write a separate quantize_config.json — loader's
    # load_quant_meta order is GGUF > TurboQuant > GPTQ-via-quantize_config.json
    # > AWQ > config.json fallback. A quantize_config.json forces GPTQ branch
    # which sets marlin_w4a8: false. Use config.json inline quantization_config
    # only (patched above with quant_type=marlin_w4a8).


if __name__ == "__main__":
    main()

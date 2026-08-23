#!/usr/bin/env python3
"""Merge W4A16 + W4A8 converted checkpoints into one hybrid checkpoint.

Per codex `128fe32` plan + Claude `1959a21` Phase 0 reconnaissance:
hybrid prefill-decode dispatch needs BOTH `marlin_qweight + marlin_scales`
(W4A16 decode path) AND `marlin_w4a8_qweight + marlin_w4a8_s_channel
+ marlin_w4a8_s_group`(W4A8 prefill path)co-resident per Linear。

This script merges two ALREADY-converted source checkpoints into a single
hybrid checkpoint(no new pack logic — just safetensors merge + config
patch)。

Inputs:
  src_w4a16:e.g. `Qwen3-4B-GPTQ-W4A16-marlin-zpfix`(post `marlin_repack.py`)
  src_w4a8 :e.g. `Qwen3-4B-GPTQ-W4A8-zpfix`(post `convert_gptq_w4a16_to_w4a8_marlin.py`)

Output:
  dst:single safetensors with all `marlin_*` and `marlin_w4a8_*` tensors
       per Linear,passthrough embed/norm/lm_head from src_w4a16

Loader contract(per codex Phase 1):
  config.json `quantization_config: {quant_type: "marlin_w4_hybrid", group_size: 128}`
  Loader detects and reads BOTH side-tensor sets per Linear。

Usage:
  python scripts/merge_w4_hybrid_checkpoint.py \\
    --w4a16 models/Qwen3-4B-GPTQ-W4A16-marlin-zpfix \\
    --w4a8  models/Qwen3-4B-GPTQ-W4A8-zpfix \\
    --dst   models/Qwen3-4B-W4-hybrid-zpfix
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

from convert import load_all_tensors, save_checkpoint, copy_config_files


W4A8_SUFFIXES = (".marlin_w4a8_qweight", ".marlin_w4a8_s_channel", ".marlin_w4a8_s_group")
W4A16_SUFFIXES = (".marlin_qweight", ".marlin_scales")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w4a16", type=Path, required=True)
    ap.add_argument("--w4a8", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--groupsize", type=int, default=128)
    args = ap.parse_args()

    if not args.w4a16.exists():
        sys.exit(f"src w4a16 not found: {args.w4a16}")
    if not args.w4a8.exists():
        sys.exit(f"src w4a8 not found: {args.w4a8}")

    print(f"Loading W4A16 source: {args.w4a16}")
    a = load_all_tensors(args.w4a16)
    print(f"  {len(a)} tensors")
    print(f"Loading W4A8 source: {args.w4a8}")
    b = load_all_tensors(args.w4a8)
    print(f"  {len(b)} tensors")

    merged: dict[str, torch.Tensor] = {}
    n_w4a16_kept = 0
    n_w4a8_kept = 0
    n_passthrough = 0

    for k, t in a.items():
        if k.endswith(W4A16_SUFFIXES):
            merged[k] = t
            n_w4a16_kept += 1
        elif k.endswith((".qweight", ".scales", ".g_idx", ".qzeros")):
            continue  # source-format tensors not needed in hybrid
        else:
            merged[k] = t
            n_passthrough += 1

    for k, t in b.items():
        if k.endswith(W4A8_SUFFIXES):
            merged[k] = t
            n_w4a8_kept += 1

    print(f"\nMerge: w4a16={n_w4a16_kept}, w4a8={n_w4a8_kept}, passthrough={n_passthrough}")
    print(f"Total tensors in hybrid: {len(merged)}")

    copy_config_files(args.w4a16, args.dst)
    save_checkpoint(
        merged, args.dst,
        quant_config={"quant_type": "marlin_w4_hybrid", "group_size": args.groupsize},
    )
    print(f"patched config.json with quant_type=marlin_w4_hybrid")


if __name__ == "__main__":
    main()

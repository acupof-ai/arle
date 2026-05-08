#!/usr/bin/env python3
"""Diagnose how many GPTQ groups exceed the W4A8 kernel MAGIC_NUM bound.

Per codex `b255828` finding:kernel `dequant_per_group` uses MAGIC_NUM
0x6480 IEEE-754 trick that imposes hard constraint
`(q-8) * s_group_stored ∈ [-128, 127)`,i.e. `|s_group_stored| ≤ 127/7 ≈ 18.14`。

Naive max-scale guarantees `s_group_stored = (max/7) / (max/127) = 127/7
= 18.143` exactly。GPTQ scales often EXCEED this(empirical 21.25 max,
17% overshoot per `492513c`)。

This diag scans all Linear weights in a GPTQ checkpoint and counts how
many groups would overshoot — calibrates Fix A's expected calibration
loss before codex applies the clamp。

Usage:
  python scripts/diag_gptq_w4a8_magic_num_bound.py \\
    --src infer/models/Qwen3-4B-GPTQ-Int4-marlin
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import safetensors.torch as st
import torch


# Codex 163c8ee corrected derivation: MAGIC_NUM 0x6480 = FP16 1152.0 (not 1536).
# Constraint: (q-8) * s_group_stored + 1152 ∈ [1024, 1280) → product ∈ [-128, 128).
# q=0 case is BINDING: -8 * s ≥ -128 → s ≤ 16. (Tighter than q=15: s < 18.286.)
KERNEL_BOUND = 16.0


def compute_s_group_stored(qweight_u8, scales_bf16, groupsize=128):
    """Mirror pack_w4a8 GPTQ-aware path:return s_group_stored = s_gptq / s_channel.

    s_channel = max(|w_real|) / 127 per output channel(same in both modes)。
    s_group_stored = s_gptq / s_channel(GPTQ-aware path,12a54da)。
    """
    n, k_half = qweight_u8.shape
    k = k_half * 2

    lo = (qweight_u8 & 0x0F).to(torch.int32)
    hi = ((qweight_u8 >> 4) & 0x0F).to(torch.int32)
    w_int = torch.zeros(n, k, dtype=torch.int32)
    w_int[:, 0::2] = lo
    w_int[:, 1::2] = hi

    scales_per_element = scales_bf16.repeat_interleave(groupsize, dim=1)
    w_real = (w_int - 8).float() * scales_per_element.float()

    # s_channel = max(|w_real|) / 127 per output channel(matching pack_w4a8 line 113)
    s_channel = w_real.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 127.0  # (n, 1)

    # s_gptq is shape (n, k/groupsize) = (n, num_groups)
    s_group_stored = scales_bf16.float() / s_channel  # (n, num_groups)
    return s_group_stored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="infer/models/Qwen3-4B-GPTQ-Int4-marlin")
    ap.add_argument("--groupsize", type=int, default=128)
    args = ap.parse_args()

    src = Path(args.src)
    idx_path = src / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    wm = idx["weight_map"]

    # Find all `*.qweight` keys
    qweight_keys = [k for k in wm if k.endswith(".qweight")]
    print(f"Found {len(qweight_keys)} GPTQ Linear layers")
    print(f"Kernel MAGIC_NUM bound: |s_group_stored| ≤ {KERNEL_BOUND:.6f}\n")

    file_handles = {}
    def get_tensor(key):
        f = wm[key]
        if f not in file_handles:
            file_handles[f] = st.safe_open(src / f, framework="pt")
        return file_handles[f].get_tensor(key)

    total_groups = 0
    over_groups = 0
    max_overshoot = 0.0
    layer_stats = []

    for qkey in qweight_keys:
        base = qkey[:-len(".qweight")]
        skey = f"{base}.scales"
        if skey not in wm:
            continue
        qw = get_tensor(qkey)
        sc = get_tensor(skey)
        s_group_stored = compute_s_group_stored(qw, sc, args.groupsize)

        n_groups = s_group_stored.numel()
        n_over = int((s_group_stored.abs() > KERNEL_BOUND).sum())
        max_v = float(s_group_stored.abs().max())
        overshoot_pct = 100.0 * n_over / n_groups
        over_pct = 100.0 * (max_v - KERNEL_BOUND) / KERNEL_BOUND if max_v > KERNEL_BOUND else 0.0

        total_groups += n_groups
        over_groups += n_over
        max_overshoot = max(max_overshoot, max_v)
        layer_stats.append((base, n_over, n_groups, max_v, overshoot_pct, over_pct))

    # Print top 5 worst layers + bottom 5
    layer_stats.sort(key=lambda x: -x[4])  # sort by overshoot_pct desc
    print(f"Top 5 layers by overshoot %:")
    for base, n_over, n_groups, max_v, overshoot_pct, over_pct in layer_stats[:5]:
        print(f"  {base}: {n_over}/{n_groups} ({overshoot_pct:.2f}%) max={max_v:.3f} (+{over_pct:.1f}%)")
    print()
    print(f"Bottom 5 layers (least overshoot):")
    for base, n_over, n_groups, max_v, overshoot_pct, over_pct in layer_stats[-5:]:
        print(f"  {base}: {n_over}/{n_groups} ({overshoot_pct:.2f}%) max={max_v:.3f} (+{over_pct:.1f}%)")
    print()
    print(f"=" * 70)
    print(f"Total: {over_groups:,} / {total_groups:,} groups exceed bound ({100*over_groups/total_groups:.4f}%)")
    print(f"Worst overshoot: max s_group_stored = {max_overshoot:.3f} (bound {KERNEL_BOUND:.3f}, +{100*(max_overshoot-KERNEL_BOUND)/KERNEL_BOUND:.1f}%)")

    # Histogram-like binning of overshoot magnitude
    print()
    bins = [(KERNEL_BOUND, KERNEL_BOUND * 1.01),
            (KERNEL_BOUND * 1.01, KERNEL_BOUND * 1.05),
            (KERNEL_BOUND * 1.05, KERNEL_BOUND * 1.10),
            (KERNEL_BOUND * 1.10, KERNEL_BOUND * 1.20),
            (KERNEL_BOUND * 1.20, KERNEL_BOUND * 1.50)]
    print("Overshoot magnitude distribution(over total groups):")
    for lo, hi in bins:
        cnt = 0
        for qkey in qweight_keys:
            base = qkey[:-len(".qweight")]
            skey = f"{base}.scales"
            if skey not in wm:
                continue
            qw = get_tensor(qkey)
            sc = get_tensor(skey)
            sgs = compute_s_group_stored(qw, sc, args.groupsize).abs()
            cnt += int(((sgs >= lo) & (sgs < hi)).sum())
        pct = 100 * cnt / total_groups
        print(f"  [{lo:.2f}, {hi:.2f}): {cnt:,} ({pct:.4f}%)")


if __name__ == "__main__":
    main()

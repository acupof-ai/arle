#!/usr/bin/env python3
"""Direct comparison of pack_w4a8(naive) vs pack_w4a8(gptq_scales=) outputs.

Runs both pack modes on the SAME decoded weight tensor (from the existing
GPTQ-Int4-marlin checkpoint) and compares qweight + s_channel + s_group
element-wise. Identifies which axis (if any) diverges between naive and
gptq_scales modes.

Drives the codex e2e-fail investigation (`592b80c`): kernel sees garbage
despite 0.02% Python round-trip drift. If naive vs gptq_scales paths
produce non-identical output tensors at SAME input, divergence in pack
output is the bug surface.

Usage:
  python scripts/diff_pack_w4a8_naive_vs_gptq.py [--layer 0] [--proj self_attn.q_proj]
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import safetensors.torch as st
import torch


def load_pack():
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "qpack", repo_root / "scripts" / "quantize_qwen3_w4a8.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_gptq_w4a16(qweight_u8, scales_bf16, groupsize=128):
    n, k_half = qweight_u8.shape
    k = k_half * 2
    lo = (qweight_u8 & 0x0F).to(torch.int32)
    hi = ((qweight_u8 >> 4) & 0x0F).to(torch.int32)
    w_int = torch.zeros(n, k, dtype=torch.int32)
    w_int[:, 0::2] = lo
    w_int[:, 1::2] = hi
    scales_per_element = scales_bf16.repeat_interleave(groupsize, dim=1)
    w_real = (w_int - 8).float() * scales_per_element.float()
    return w_real, scales_bf16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="infer/models/Qwen3-4B-GPTQ-Int4-marlin")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--proj", default="self_attn.q_proj")
    ap.add_argument("--groupsize", type=int, default=128)
    args = ap.parse_args()

    base = f"model.layers.{args.layer}.{args.proj}"
    src = Path(args.src)
    idx = json.load(open(src / "model.safetensors.index.json"))
    wm = idx["weight_map"]

    qpath = src / wm[f"{base}.qweight"]
    spath = src / wm[f"{base}.scales"]
    with st.safe_open(qpath, framework="pt") as h:
        qweight_u8 = h.get_tensor(f"{base}.qweight")
    with st.safe_open(spath, framework="pt") as h:
        scales_bf16 = h.get_tensor(f"{base}.scales")

    print(f"Layer = {base}")
    print(f"qweight: {list(qweight_u8.shape)} {qweight_u8.dtype}")
    print(f"scales:  {list(scales_bf16.shape)} {scales_bf16.dtype}\n")

    w_real, gptq_scales = decode_gptq_w4a16(qweight_u8, scales_bf16, args.groupsize)
    w_bf16 = w_real.to(torch.bfloat16)
    n, k = w_bf16.shape
    print(f"decoded w_real: {list(w_bf16.shape)} mean_abs={w_bf16.abs().mean().item():.4e}\n")

    qpack = load_pack()

    # Pack twice: naive vs gptq_scales
    qw_a, sc_a, sg_a = qpack.pack_w4a8(w_bf16, groupsize=args.groupsize)
    qw_b, sc_b, sg_b = qpack.pack_w4a8(w_bf16, groupsize=args.groupsize, gptq_scales=gptq_scales)

    print(f"NAIVE  qweight  {list(qw_a.shape)} {qw_a.dtype}  mean_abs={qw_a.abs().float().mean():.4e}")
    print(f"GPTQ   qweight  {list(qw_b.shape)} {qw_b.dtype}  mean_abs={qw_b.abs().float().mean():.4e}")
    print(f"NAIVE  s_chan   {list(sc_a.shape)} {sc_a.dtype}  mean={sc_a.mean():.4e}  max={sc_a.max():.4e}")
    print(f"GPTQ   s_chan   {list(sc_b.shape)} {sc_b.dtype}  mean={sc_b.mean():.4e}  max={sc_b.max():.4e}")
    print(f"NAIVE  s_group  {list(sg_a.shape)} {sg_a.dtype}  mean={sg_a.float().mean():.4e}  max={sg_a.float().max():.4e}")
    print(f"GPTQ   s_group  {list(sg_b.shape)} {sg_b.dtype}  mean={sg_b.float().mean():.4e}  max={sg_b.float().max():.4e}")
    print()

    # Element-wise diff
    qw_diff = (qw_a.long() - qw_b.long()).abs().float()
    sc_diff = (sc_a - sc_b).abs()
    sg_diff = (sg_a.float() - sg_b.float()).abs()

    print(f"qweight   diff: max={qw_diff.max():.0f}  mean={qw_diff.mean():.4f}  nz={int((qw_diff>0).sum())}/{qw_diff.numel()}")
    print(f"s_channel diff: max={sc_diff.max():.4e}  mean={sc_diff.mean():.4e}  rel_max={sc_diff.max()/sc_a.abs().max():.4f}")
    print(f"s_group   diff: max={sg_diff.max():.4e}  mean={sg_diff.mean():.4e}  rel_max={sg_diff.max()/sg_a.float().abs().max():.4f}")
    print()

    # Top-5 locations where qweight diverges
    if qw_diff.max() > 0:
        flat_idx = qw_diff.flatten().argsort(descending=True)[:5]
        print("Top-5 qweight divergences:")
        for i in flat_idx:
            r, c = int(i) // qw_a.shape[1], int(i) % qw_a.shape[1]
            print(f"  [{r},{c}]: naive={int(qw_a[r,c])} gptq={int(qw_b[r,c])} diff={int(qw_diff[r,c])}")


if __name__ == "__main__":
    main()

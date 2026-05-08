#!/usr/bin/env python3
"""Verify GPTQ→W4A8 re-pack quality by comparing decoded weights.

Per `8bb57ea` codex correction risk: re-pack via naive max-scale pack_w4a8
may add noise on top of GPTQ-calibrated weights. Test single layer:
1. Decode source GPTQ qweight + scales → BF16 weight tensor (calibrated)
2. Decode dst marlin_w4a8 qweight + s_channel + s_group → BF16 weight tensor
3. Compare element-wise — should be near-identical if calibration preserved

Usage:
  python scripts/verify_gptq_w4a8_repack_quality.py \\
    --src infer/models/Qwen3-4B-GPTQ-Int4-marlin \\
    --dst infer/models/Qwen3-4B-GPTQ-W4A8-marlin \\
    [--layer 0] [--proj down_proj]
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import safetensors.torch as st
import torch


def load_diag_module():
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "diag", repo_root / "scripts" / "diag_w4a8_pack_roundtrip.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_pack():
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "qpack", repo_root / "scripts" / "quantize_qwen3_w4a8.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_gptq_w4a16(qweight_u8: torch.Tensor, scales_bf16: torch.Tensor, groupsize: int = 128):
    """Decode GPTQ qweight [N, K/2] U8 + scales [N, num_groups] BF16 → [N, K] BF16 weight."""
    n, k_half = qweight_u8.shape
    k = k_half * 2
    lo = (qweight_u8 & 0x0F).to(torch.int32)
    hi = ((qweight_u8 >> 4) & 0x0F).to(torch.int32)
    w_int = torch.zeros(n, k, dtype=torch.int32)
    w_int[:, 0::2] = lo
    w_int[:, 1::2] = hi
    scales_per_element = scales_bf16.repeat_interleave(groupsize, dim=1)
    w_real = (w_int - 8).float() * scales_per_element.float()
    return w_real


def find_tensor(weight_map: dict, base: str, fallback_dir: Path | None = None):
    """Find file containing tensor `base` and return loaded tensor."""
    fname = weight_map.get(base)
    if not fname:
        return None
    if fallback_dir:
        path = fallback_dir / fname
    else:
        return None
    with st.safe_open(path, framework="pt") as h:
        return h.get_tensor(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--proj", default="self_attn.q_proj")
    ap.add_argument("--groupsize", type=int, default=128)
    args = ap.parse_args()

    base = f"model.layers.{args.layer}.{args.proj}"
    print(f"Layer = {base}\n")

    # Load src GPTQ
    idx = json.load(open(args.src / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    src_qweight = find_tensor(wm, f"{base}.qweight", args.src)
    src_scales = find_tensor(wm, f"{base}.scales", args.src)
    if src_qweight is None or src_scales is None:
        sys.exit(f"src missing {base}.qweight or .scales")
    print(f"src qweight: shape={list(src_qweight.shape)} dtype={src_qweight.dtype}")
    print(f"src scales:  shape={list(src_scales.shape)} dtype={src_scales.dtype}")

    w_src = decode_gptq_w4a16(src_qweight, src_scales, args.groupsize)
    print(f"decoded src: shape={list(w_src.shape)} mean_abs={w_src.abs().mean().item():.4e}\n")

    # Load dst W4A8
    dst_path = args.dst / "model.safetensors"
    with st.safe_open(dst_path, framework="pt") as h:
        dst_qweight = h.get_tensor(f"{base}.marlin_w4a8_qweight")
        dst_s_channel = h.get_tensor(f"{base}.marlin_w4a8_s_channel")
        dst_s_group = h.get_tensor(f"{base}.marlin_w4a8_s_group")
    print(f"dst qweight:  shape={list(dst_qweight.shape)} dtype={dst_qweight.dtype}")
    print(f"dst s_channel: shape={list(dst_s_channel.shape)} dtype={dst_s_channel.dtype}")
    print(f"dst s_group:   shape={list(dst_s_group.shape)} dtype={dst_s_group.dtype}\n")

    diag = load_diag_module()
    qpack = load_pack()
    n, k = w_src.shape
    perm, scale_perm, scale_perm_single = qpack.get_perms(args.groupsize, k)
    w_dst_recovered = diag.manual_unpack_w4a8(
        dst_qweight, dst_s_channel, dst_s_group, perm, scale_perm, scale_perm_single,
        n, k, args.groupsize,
    )

    # manual_unpack returns transposed; expect [k, n] vs source [n, k]
    if w_dst_recovered.shape == w_src.shape:
        w_dst = w_dst_recovered
    elif w_dst_recovered.shape == (k, n):
        w_dst = w_dst_recovered.t()
    else:
        sys.exit(f"unexpected w_dst shape {list(w_dst_recovered.shape)} vs src {list(w_src.shape)}")
    print(f"recovered dst: shape={list(w_dst.shape)} mean_abs={w_dst.abs().mean().item():.4e}\n")

    # Compare
    diff = (w_dst - w_src).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    flat = diff.flatten()
    if flat.numel() > 16_000_000:
        idx = torch.randperm(flat.numel())[:16_000_000]
        flat = flat[idx]
    p99 = torch.quantile(flat, 0.99).item()

    src_max = w_src.abs().max().item()
    src_flat = w_src.abs().flatten()
    if src_flat.numel() > 16_000_000:
        idx2 = torch.randperm(src_flat.numel())[:16_000_000]
        src_flat = src_flat[idx2]
    src_p99 = torch.quantile(src_flat, 0.99).item()

    print("Re-pack quality diagnostic:")
    print(f"  max abs diff   = {max_abs:.6e}")
    print(f"  mean abs diff  = {mean_abs:.6e}")
    print(f"  p99 abs diff   = {p99:.6e}")
    print(f"  src max abs    = {src_max:.6e}")
    print(f"  src p99 abs    = {src_p99:.6e}")
    print(f"  rel max diff   = {max_abs / src_max if src_max > 0 else float('nan'):.4f}")
    print(f"  rel mean diff  = {mean_abs / w_src.abs().mean().item():.4f}\n")

    threshold_rel = 0.01  # 1% relative
    if max_abs / src_max < threshold_rel:
        print(f"✅ PASS: re-pack preserves GPTQ calibration (rel max < 1%)")
        sys.exit(0)
    else:
        print(f"❌ FAIL: re-pack added noise > 1% relative — calibration drifted")
        print(f"   → fall back to AutoGPTQ-direct path or modify pack_w4a8 to accept GPTQ scales")
        sys.exit(1)


if __name__ == "__main__":
    main()

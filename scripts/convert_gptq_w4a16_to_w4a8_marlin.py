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
import json
import shutil
import sys
from pathlib import Path

import safetensors.torch as st
import torch


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
    return pack_w4a8(w_real.to(torch.bfloat16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--groupsize", type=int, default=128)
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"src not found: {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)

    pack_w4a8 = load_pack_w4a8()

    idx_path = args.src / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        weight_map = idx["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        files = [f.name for f in args.src.glob("*.safetensors")]
        weight_map = None

    new_state: dict[str, torch.Tensor] = {}
    n_repacked = 0
    n_passthrough = 0

    for fname in files:
        fpath = args.src / fname
        with st.safe_open(fpath, framework="pt") as h:
            keys = list(h.keys())
            tensors = {k: h.get_tensor(k) for k in keys}

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
    out_path = args.dst / "model.safetensors"
    st.save_file(new_state, str(out_path))
    print(f"saved → {out_path}")

    for cfg in ["config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                "added_tokens.json", "merges.txt", "vocab.json"]:
        src_cfg = args.src / cfg
        if src_cfg.exists():
            shutil.copy2(src_cfg, args.dst / cfg)

    quant_cfg = {
        "bits": 4,
        "group_size": args.groupsize,
        "quant_method": "gptq_w4a8",
        "source": "GPTQ W4A16 re-packed via convert_gptq_w4a16_to_w4a8_marlin.py",
        "marlin_repacked": True,
    }
    (args.dst / "quantize_config.json").write_text(json.dumps(quant_cfg, indent=2))
    print(f"wrote quantize_config.json with quant_method=gptq_w4a8")


if __name__ == "__main__":
    main()

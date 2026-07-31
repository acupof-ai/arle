"""BF16 HF checkpoint -> W8A16 (per-group signed INT8 weights, BF16 scales).

Sibling of scripts/fp8_block_cast.py. INT8 weights are ~2x more accurate than
FP8 block-cast at the same 8 bits (uniform grid beats E4M3's exponential grid
once 128x128 blocking has already isolated outliers — see the FP8-vs-INT8 probe:
1.46% vs 2.65% rel-L2 on real Qwen weights). This produces the checkpoint the
CUDA `w8a16_gemv` kernel consumes.

Format the loader's detect_quant_format + load_w8a16_view expect:
  {base}.weight        : INT8, shape [rows, cols]   (NOT packed — 1 byte/elem)
  {base}.weight_scale  : BF16, shape [rows, cols/group_size]  (per-row, per
                         column-group). Dequant: w ~= int8 * scale.
  group_size divides cols; the kernel indexes scales[row*num_groups + k/gs].

Symmetric quant, no zero-point: scale = group_absmax / 127, q = round(w/scale)
clamped to [-127, 127]. Which tensors get quantized comes from --ref, whose
index.json lists the linear .weight tensors (mirrors fp8_block_cast's scope
rule); lm_head/embed/norm stay BF16.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_GROUP = 128
INT8_MAX = 127.0


def per_group_int8(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row, per-column-group symmetric INT8. Returns (int8 [rows,cols],
    scale bf16 [rows, cols/group_size]). cols must be group-divisible."""
    rows, cols = w.shape
    assert cols % group_size == 0, f"cols {cols} not divisible by group {group_size}"
    ng = cols // group_size
    view = w.float().view(rows, ng, group_size)
    amax = view.abs().amax(dim=2, keepdim=True).clamp_(1e-8)
    scale = amax / INT8_MAX
    q = torch.round(view / scale).clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q.view(rows, cols).contiguous(), scale.view(rows, ng).to(torch.bfloat16).contiguous()


def quant_weight_names(ref_dir: Path) -> set[str]:
    """Linear .weight base names to quantize: those the reference checkpoint
    stores quantized (carry a .weight_scale or .weight_scale_inv sidecar)."""
    idx = json.load(open(ref_dir / "model.safetensors.index.json"))["weight_map"]
    out = set()
    for k in idx:
        for suf in (".weight_scale_inv", ".weight_scale"):
            if k.endswith(suf):
                out.add(k[: -len(suf)])
    return out


def run(bf16_dir: Path, ref_dir: Path, out_dir: Path, group_size: int) -> None:
    quant_set = quant_weight_names(ref_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(bf16_dir):
        src = bf16_dir / f
        if src.is_file() and not f.endswith(".safetensors"):
            shutil.copy(src, out_dir / f)

    weight_map = json.load(open(bf16_dir / "model.safetensors.index.json"))["weight_map"]
    shards: dict[str, list[str]] = {}
    for name, fname in weight_map.items():
        shards.setdefault(fname, []).append(name)

    new_map: dict[str, str] = {}
    n_quant = 0
    for fname in sorted(shards):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(bf16_dir / fname), framework="pt") as f:
            for name in sorted(shards[fname]):
                w = f.get_tensor(name)
                if name in quant_set and w.dim() == 2 and w.shape[1] % group_size == 0:
                    q, scale = per_group_int8(w, group_size)
                    tensors[name] = q
                    tensors[name + "_scale"] = scale
                    new_map[name] = fname
                    new_map[name + "_scale"] = fname
                    n_quant += 1
                else:
                    tensors[name] = w.to(torch.bfloat16) if w.is_floating_point() else w
                    new_map[name] = fname
        save_file(tensors, str(out_dir / fname))
        print(f"wrote {fname} ({len(tensors)} tensors)", flush=True)

    # Record the quant config so the loader/consumers know group_size.
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))
        cfg["quantization_config"] = {
            "quant_method": "w8a16",
            "bits": 8,
            "group_size": group_size,
        }
        json.dump(cfg, open(cfg_path, "w"), indent=2)

    total = sum(
        os.path.getsize(out_dir / f)
        for f in os.listdir(out_dir)
        if f.endswith(".safetensors")
    )
    json.dump(
        {"metadata": {"total_size": total}, "weight_map": new_map},
        open(out_dir / "model.safetensors.index.json", "w"),
        indent=2,
    )
    print(f"done: {n_quant} tensors W8A16-quantized (group={group_size}) -> {out_dir}", flush=True)


def _selfcheck() -> None:
    # Round-trip pins dequant direction (multiply) and layout; must beat FP8's
    # 2.65% on the same data (uniform INT8 grid vs E4M3 within a group).
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fp8_block_cast import per_block_cast_to_fp8
    from fp8_cast_loss_probe import dequant as fp8_dequant

    torch.manual_seed(0)
    w = torch.randn(256, 512, dtype=torch.bfloat16)
    w += 0.3 * torch.randn(256, 1) * torch.randn(1, 512)  # heavy-tail like real weights
    q, scale = per_group_int8(w, DEFAULT_GROUP)
    assert q.shape == (256, 512) and q.dtype == torch.int8, (q.shape, q.dtype)
    assert scale.shape == (256, 512 // DEFAULT_GROUP), scale.shape
    deq = (q.float().view(256, 4, DEFAULT_GROUP) * scale.float().view(256, 4, 1)).view(256, 512)
    int8_rel = ((deq - w.float()).norm() / w.float().norm()).item()

    fp8, sf = per_block_cast_to_fp8(w)
    fp8_rel = ((fp8_dequant(fp8, sf) - w.float()).norm() / w.float().norm()).item()

    assert int8_rel < fp8_rel, f"INT8 {int8_rel:.4f} should beat FP8 {fp8_rel:.4f}"
    assert int8_rel < 0.02, f"INT8 rel err {int8_rel:.4f} too high"
    print(f"selfcheck ok: int8 rel-L2 {int8_rel:.4f} < fp8 {fp8_rel:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bf16", help="source BF16 HF checkpoint dir")
    ap.add_argument("--ref", help="reference quantized checkpoint (defines quant scope)")
    ap.add_argument("--out", help="output W8A16 checkpoint dir")
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    if not (args.bf16 and args.ref and args.out):
        ap.error("--bf16, --ref, --out required (or --selfcheck)")
    run(Path(args.bf16), Path(args.ref), Path(args.out), args.group_size)


if __name__ == "__main__":
    main()

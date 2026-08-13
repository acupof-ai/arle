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


def read_weight_map(model_dir: Path) -> dict[str, str]:
    """tensor name -> shard filename, for both sharded (index.json) and
    single-file (model.safetensors) checkpoints."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        return json.load(open(idx))["weight_map"]
    single = "model.safetensors"
    with safe_open(str(model_dir / single), framework="pt") as f:
        return {k: single for k in f.keys()}


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
    idx = read_weight_map(ref_dir)
    out = set()
    for k in idx:
        for suf in (".weight_scale_inv", ".weight_scale"):
            if k.endswith(suf):
                out.add(k[: -len(suf)])
    return out


# When no quantized reference exists, quantize all 2D linear .weight tensors
# except these. Source of truth is the CUDA loader: qwen35.rs:3296-3298 loads
# in_proj_a/in_proj_b (per-v-head scalar gates) via BF16-only load_matrix and
# conv1d via load_conv1d_vec — NOT load_matrix_quant_aware. Quant scope MUST NOT
# exceed the loader's quant-aware coverage or serve reads I8 through the BF16
# path and crashes (fixed 195ba2e5d). embed/lm_head set logits/inputs directly,
# norms are 1D. If the loader flips any of these to quant-aware, update here —
# or pass --ref to derive scope from a checkpoint instead of this list.
# Matched as exact ".<name>.weight" / ".<name>_weight" endings so a broad token
# like "norm" can't swallow an unrelated tensor in a differently-named export.
ALL_LINEAR_SKIP_ENDINGS = (
    "embed_tokens.weight", "lm_head.weight", "in_proj_a.weight", "in_proj_b.weight",
    "conv1d.weight", "gate.weight",
)


def all_linear_names(weight_map: dict[str, str]) -> set[str]:
    return {
        k for k in weight_map
        if k.endswith(".weight")
        and "norm" not in k.rsplit(".", 2)[-2]  # LayerNorm scales are 1D, not linear
        and not any(k.endswith(e) for e in ALL_LINEAR_SKIP_ENDINGS)
    }


def run(bf16_dir: Path, ref_dir: Path | None, out_dir: Path, group_size: int) -> None:
    weight_map = read_weight_map(bf16_dir)
    quant_set = all_linear_names(weight_map) if ref_dir is None else quant_weight_names(ref_dir)
    print(f"quant scope: {len(quant_set)} tensors "
          f"({'all-linear' if ref_dir is None else 'from ref'})", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(bf16_dir):
        src = bf16_dir / f
        if src.is_file() and not f.endswith(".safetensors"):
            shutil.copy(src, out_dir / f)

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
    from fp8_block_cast import per_block_cast_to_fp8, dequant as fp8_dequant

    torch.manual_seed(0)
    rows, cols = 256, 512
    ng = cols // DEFAULT_GROUP
    w = torch.randn(rows, cols, dtype=torch.bfloat16)
    w += 0.3 * torch.randn(rows, 1) * torch.randn(1, cols)  # heavy-tail like real weights
    q, scale = per_group_int8(w, DEFAULT_GROUP)
    assert q.shape == (rows, cols) and q.dtype == torch.int8, (q.shape, q.dtype)
    assert scale.shape == (rows, ng), scale.shape
    deq = (q.float().view(rows, ng, DEFAULT_GROUP) * scale.float().unsqueeze(-1)).view(rows, cols)
    int8_rel = ((deq - w.float()).norm() / w.float().norm()).item()

    fp8, sf = per_block_cast_to_fp8(w)
    fp8_rel = ((fp8_dequant(fp8, sf) - w.float()).norm() / w.float().norm()).item()

    assert int8_rel < fp8_rel, f"INT8 {int8_rel:.4f} should beat FP8 {fp8_rel:.4f}"
    assert int8_rel < 0.02, f"INT8 rel err {int8_rel:.4f} too high"
    print(f"selfcheck ok: int8 rel-L2 {int8_rel:.4f} < fp8 {fp8_rel:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bf16", help="source BF16 HF checkpoint dir")
    ap.add_argument("--ref", help="reference quantized checkpoint defining quant scope; "
                                  "omit to quantize all linear weights (--all-linear scope)")
    ap.add_argument("--out", help="output W8A16 checkpoint dir")
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    if not (args.bf16 and args.out):
        ap.error("--bf16 and --out required (or --selfcheck)")
    run(Path(args.bf16), Path(args.ref) if args.ref else None, Path(args.out), args.group_size)


if __name__ == "__main__":
    main()

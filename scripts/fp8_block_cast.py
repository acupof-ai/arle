"""BF16 HF checkpoint -> DeepSeek-style FP8 block-scaled checkpoint.

Post-merge quantizer for ISO-Merger output (see scripts/iso_merger.py). The
merge runs in BF16/FP32 by design — SVD in the Stiefel tangent space needs full
precision, so we never quantize *inside* the merge. This is the separate
"cast to storage format" step: merged BF16 dir -> servable FP8 dir.

Algorithm is DeepGEMM's `per_block_cast_to_fp8` (the vendored reference at
crates/cuda-kernels/vendor/deepgemm/deep_gemm/utils/math.py), which is also what
the CUDA loader's `detect_quant_format` decodes:

  128x128 block, sf = blockwise_amax / 448.0 (clamp 1e-4), w_fp8 = w / sf.
  Stored: {base}.weight (E4M3) + {base}.weight_scale_inv = sf  (BF16,
  shape [ceil(m/128), ceil(n/128)]). Dequant multiplies: w ~= w_fp8 * sf.
  (The name says "inv" but the stored value is the dequant multiplier sf, not
  1/sf — DeepSeek's convention. The self-check below pins this direction.)

Quantization *scope* is not guessed: --ref-fp8 points at the official FP8
checkpoint whose index.json says which tensors are F8_E4M3. Same-named tensors
in the BF16 dir get cast; everything else is copied verbatim. This keeps the
output byte-format-compatible with the reference the loader was validated on.
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

BLOCK = 128
E4M3_MAX = 448.0


def per_block_cast_to_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepGEMM per_block_cast: 128x128 blocks, sf = amax/448. Returns
    (fp8 weight [m,n], scale sf [ceil(m/128), ceil(n/128)] as bf16)."""
    m, n = w.shape
    pm, pn = ((m + BLOCK - 1) // BLOCK) * BLOCK, ((n + BLOCK - 1) // BLOCK) * BLOCK
    padded = torch.zeros(pm, pn, dtype=torch.float32)
    padded[:m, :n] = w.float()
    view = padded.view(pm // BLOCK, BLOCK, pn // BLOCK, BLOCK)
    amax = view.abs().amax(dim=(1, 3), keepdim=True).clamp_(1e-4)
    sf = amax / E4M3_MAX
    fp8 = (view / sf).to(torch.float8_e4m3fn).view(pm, pn)[:m, :n].contiguous()
    return fp8, sf.view(pm // BLOCK, pn // BLOCK).to(torch.bfloat16).contiguous()


def dequant(fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Inverse of per_block_cast_to_fp8: pad, multiply by block scale, slice.
    Mirrors the loader's dequant (w ~= fp8 * sf)."""
    m, n = fp8.shape
    pm, pn = ((m + BLOCK - 1) // BLOCK) * BLOCK, ((n + BLOCK - 1) // BLOCK) * BLOCK
    padded = torch.zeros(pm, pn, dtype=torch.float32)
    padded[:m, :n] = fp8.float()
    view = padded.view(pm // BLOCK, BLOCK, pn // BLOCK, BLOCK)
    deq = (view * sf.float().view(pm // BLOCK, 1, pn // BLOCK, 1)).reshape(pm, pn)
    return deq[:m, :n]


def fp8_weight_names(ref_dir: Path) -> set[str]:
    """Base names ({...}.weight) that the reference FP8 checkpoint stores as
    block-scaled FP8 (i.e. carry a .weight_scale_inv sidecar)."""
    idx = json.load(open(ref_dir / "model.safetensors.index.json"))["weight_map"]
    return {k[: -len("_scale_inv")] for k in idx if k.endswith(".weight_scale_inv")}


def run(bf16_dir: Path, ref_dir: Path, out_dir: Path) -> None:
    fp8_set = fp8_weight_names(ref_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # config/tokenizer/etc. come from the *reference* FP8 dir so the merged
    # output carries quantization_config; the BF16 merge dir has none.
    for f in os.listdir(ref_dir):
        src = ref_dir / f
        if src.is_file() and not f.endswith(".safetensors"):
            shutil.copy(src, out_dir / f)

    weight_map = json.load(
        open(bf16_dir / "model.safetensors.index.json")
    )["weight_map"]
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
                if name in fp8_set and w.dim() == 2:
                    fp8, sf = per_block_cast_to_fp8(w)
                    tensors[name] = fp8
                    tensors[name + "_scale_inv"] = sf
                    new_map[name] = fname
                    new_map[name + "_scale_inv"] = fname
                    n_quant += 1
                else:
                    tensors[name] = w.to(torch.bfloat16) if w.is_floating_point() else w
                    new_map[name] = fname
        save_file(tensors, str(out_dir / fname))
        print(f"wrote {fname} ({len(tensors)} tensors)", flush=True)

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
    print(f"done: {n_quant} tensors FP8-block-cast -> {out_dir}", flush=True)


def _selfcheck() -> None:
    # Round-trip pins the dequant direction (multiply by sf) and the block
    # layout. E4M3 3-bit mantissa over a 448-normalized block: rel err < 8%.
    torch.manual_seed(0)
    w = torch.randn(300, 500, dtype=torch.bfloat16)  # non-128-aligned on purpose
    fp8, sf = per_block_cast_to_fp8(w)
    assert fp8.shape == (300, 500), fp8.shape
    assert sf.shape == (3, 4), sf.shape  # ceil(300/128)=3, ceil(500/128)=4
    # dequant on padded grid (fp8 is unpadded), then slice — mirrors the loader.
    padded = torch.zeros(384, 512)
    padded[:300, :500] = fp8.float()
    deq = (padded.view(3, 128, 4, 128) * sf.float().view(3, 1, 4, 1)).reshape(384, 512)[:300, :500]
    rel = (deq - w.float()).norm() / w.float().norm()
    assert rel < 0.08, f"round-trip rel err {rel:.4f} too high — wrong direction?"
    print(f"selfcheck ok (round-trip rel err {rel:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bf16", help="merged BF16 HF checkpoint dir")
    ap.add_argument("--ref-fp8", help="official FP8 checkpoint (defines quant scope + config)")
    ap.add_argument("--out", help="output FP8 checkpoint dir")
    ap.add_argument("--selfcheck", action="store_true", help="run round-trip self-check and exit")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    if not (args.bf16 and args.ref_fp8 and args.out):
        ap.error("--bf16, --ref-fp8, --out required (or --selfcheck)")
    run(Path(args.bf16), Path(args.ref_fp8), Path(args.out))


if __name__ == "__main__":
    main()

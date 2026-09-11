"""Unified weight quantization framework.

Each quantizer implements `quantize(weight) -> {suffix: tensor}`,
`scope_names(weight_map, ref_dir)`, `quant_config()`, and `selfcheck()`.
`CheckpointIO` handles shard traversal, index.json, and config.json uniformly.

Usage:
  python scripts/quantize.py --format fp8 --bf16 <dir> --ref <fp8-ref> --out <dir>
  python scripts/quantize.py --format w8a16 --bf16 <dir> [--ref <w8a16-ref>] --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Checkpoint IO — shared by all quantizers
# ---------------------------------------------------------------------------

def read_weight_map(model_dir: Path) -> dict[str, str]:
    """tensor name -> shard filename (sharded or single-file)."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        return json.load(open(idx))["weight_map"]
    single = "model.safetensors"
    with safe_open(str(model_dir / single), framework="pt") as f:
        return {k: single for k in f.keys()}


def copy_non_safetensors(src: Path, dst: Path) -> None:
    """Copy config/tokenizer/etc. (everything except .safetensors)."""
    for f in os.listdir(src):
        p = src / f
        if p.is_file() and not f.endswith(".safetensors"):
            shutil.copy(p, dst / f)


def write_index(out_dir: Path, weight_map: dict[str, str]) -> None:
    total = sum(
        os.path.getsize(out_dir / f)
        for f in os.listdir(out_dir)
        if f.endswith(".safetensors")
    )
    json.dump(
        {"metadata": {"total_size": total}, "weight_map": weight_map},
        open(out_dir / "model.safetensors.index.json", "w"),
        indent=2,
    )


def patch_quant_config(out_dir: Path, quant_config: dict) -> None:
    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.load(open(cfg_path))
    cfg["quantization_config"] = quant_config
    json.dump(cfg, open(cfg_path, "w"), indent=2)


class CheckpointIO:
    """Drives a quantizer over a checkpoint's shards.

    The quantizer decides which tensors to quantize (`scope_names`) and how
    (`quantize` returns a dict of name-suffix -> tensor). Everything else —
    shard iteration, non-quantized tensor passthrough, index.json, config.json —
    is handled here so each quantizer stays algorithm-only.
    """

    def __init__(self, quantizer: "Quantizer"):
        self.q = quantizer

    def run(
        self,
        src_dir: Path,
        out_dir: Path,
        ref_dir: Path | None = None,
        *,
        config_src: Path | None = None,
    ) -> None:
        """Quantize `src_dir` -> `out_dir`.

        `config_src` is the directory to copy non-safetensors files from
        (defaults to src_dir). For FP8 the reference FP8 dir supplies config.
        """
        weight_map = read_weight_map(src_dir)
        scope = self.q.scope_names(weight_map, ref_dir)
        print(
            f"[{self.q.name}] quant scope: {len(scope)} tensors "
            f"({'from ref' if ref_dir else 'auto'})",
            flush=True,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        copy_non_safetensors(config_src or src_dir, out_dir)

        shards: dict[str, list[str]] = {}
        for name, fname in weight_map.items():
            shards.setdefault(fname, []).append(name)

        new_map: dict[str, str] = {}
        n_quant = 0
        for fname in sorted(shards):
            tensors: dict[str, torch.Tensor] = {}
            with safe_open(str(src_dir / fname), framework="pt") as f:
                for name in sorted(shards[fname]):
                    w = f.get_tensor(name)
                    if name in scope and self.q.can_quantize(w):
                        out = self.q.quantize(w)
                        for suffix, t in out.items():
                            tensors[name + suffix] = t
                            new_map[name + suffix] = fname
                        n_quant += 1
                    else:
                        tensors[name] = (
                            w.to(torch.bfloat16) if w.is_floating_point() else w
                        )
                        new_map[name] = fname
            save_file(tensors, str(out_dir / fname))
            print(f"wrote {fname} ({len(tensors)} tensors)", flush=True)

        write_index(out_dir, new_map)
        patch_quant_config(out_dir, self.q.quant_config())
        print(f"done: {n_quant} tensors {self.q.name}-quantized -> {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Quantizer base
# ---------------------------------------------------------------------------

class Quantizer(ABC):
    name: str = ""

    @abstractmethod
    def quantize(self, weight: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantize one 2D weight. Returns {suffix: tensor} — the original
        tensor name gets each suffix appended (suffix '' overwrites it)."""

    def can_quantize(self, weight: torch.Tensor) -> bool:
        """Default: any 2D tensor. Override for shape/alignment constraints."""
        return weight.dim() == 2

    @abstractmethod
    def scope_names(
        self, weight_map: dict[str, str], ref_dir: Path | None
    ) -> set[str]:
        """Base names ({...}.weight) to quantize."""

    @abstractmethod
    def quant_config(self) -> dict:
        """quantization_config written to config.json."""

    def selfcheck(self) -> None:
        """Optional round-trip / sanity check. No-op by default."""


# ---------------------------------------------------------------------------
# FP8 block-scaled (DeepGEMM per_block_cast_to_fp8)
# ---------------------------------------------------------------------------

FP8_BLOCK = 128
FP8_E4M3_MAX = 448.0


def per_block_cast_to_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """128x128 block, sf = amax/448. Returns (fp8 [m,n], sf [ceil(m/128), ceil(n/128)] bf16)."""
    m, n = w.shape
    pm = ((m + FP8_BLOCK - 1) // FP8_BLOCK) * FP8_BLOCK
    pn = ((n + FP8_BLOCK - 1) // FP8_BLOCK) * FP8_BLOCK
    padded = torch.zeros(pm, pn, dtype=torch.float32)
    padded[:m, :n] = w.float()
    view = padded.view(pm // FP8_BLOCK, FP8_BLOCK, pn // FP8_BLOCK, FP8_BLOCK)
    amax = view.abs().amax(dim=(1, 3), keepdim=True).clamp_(1e-4)
    sf = amax / FP8_E4M3_MAX
    fp8 = (view / sf).to(torch.float8_e4m3fn).view(pm, pn)[:m, :n].contiguous()
    return fp8, sf.view(pm // FP8_BLOCK, pn // FP8_BLOCK).to(torch.bfloat16).contiguous()


def fp8_dequant(fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Inverse of per_block_cast_to_fp8."""
    m, n = fp8.shape
    pm = ((m + FP8_BLOCK - 1) // FP8_BLOCK) * FP8_BLOCK
    pn = ((n + FP8_BLOCK - 1) // FP8_BLOCK) * FP8_BLOCK
    padded = torch.zeros(pm, pn, dtype=torch.float32)
    padded[:m, :n] = fp8.float()
    view = padded.view(pm // FP8_BLOCK, FP8_BLOCK, pn // FP8_BLOCK, FP8_BLOCK)
    deq = (view * sf.float().view(pm // FP8_BLOCK, 1, pn // FP8_BLOCK, 1)).reshape(pm, pn)
    return deq[:m, :n]


class FP8BlockCastQuantizer(Quantizer):
    name = "fp8"

    def quantize(self, weight: torch.Tensor) -> dict[str, torch.Tensor]:
        fp8, sf = per_block_cast_to_fp8(weight)
        return {"": fp8, "_scale_inv": sf}

    def scope_names(
        self, weight_map: dict[str, str], ref_dir: Path | None
    ) -> set[str]:
        assert ref_dir is not None, "fp8 requires --ref (official FP8 checkpoint)"
        idx = read_weight_map(ref_dir)
        return {
            k[: -len("_scale_inv")]
            for k in idx
            if k.endswith(".weight_scale_inv")
        }

    def quant_config(self) -> dict:
        return {"quant_method": "fp8", "bits": 8, "block_size": FP8_BLOCK}

    def selfcheck(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(300, 500, dtype=torch.bfloat16)
        fp8, sf = per_block_cast_to_fp8(w)
        assert fp8.shape == (300, 500)
        assert sf.shape == (3, 4)
        deq = fp8_dequant(fp8, sf)
        rel = (deq - w.float()).norm() / w.float().norm()
        assert rel < 0.08, f"fp8 round-trip rel err {rel:.4f}"
        print(f"fp8 selfcheck ok (round-trip rel err {rel:.4f})")


# ---------------------------------------------------------------------------
# W8A16 per-group signed INT8
# ---------------------------------------------------------------------------

W8A16_GROUP = 128
INT8_MAX = 127.0

# Tensors the loader reads BF16-only — must NOT be quantized (else serve reads
# I8 through the BF16 path and crashes). Source: qwen35.rs load_matrix coverage.
W8A16_SKIP_ENDINGS = (
    "embed_tokens.weight", "lm_head.weight", "in_proj_a.weight", "in_proj_b.weight",
    "conv1d.weight", "gate.weight",
)


def per_group_int8(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row, per-column-group symmetric INT8. (int8 [rows,cols], scale bf16 [rows, cols/gs])."""
    rows, cols = w.shape
    assert cols % group_size == 0
    ng = cols // group_size
    view = w.float().view(rows, ng, group_size)
    amax = view.abs().amax(dim=2, keepdim=True).clamp_(1e-8)
    scale = amax / INT8_MAX
    q = torch.round(view / scale).clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q.view(rows, cols).contiguous(), scale.view(rows, ng).to(torch.bfloat16).contiguous()


class W8A16Quantizer(Quantizer):
    name = "w8a16"

    def __init__(self, group_size: int = W8A16_GROUP):
        self.group_size = group_size

    def can_quantize(self, weight: torch.Tensor) -> bool:
        return weight.dim() == 2 and weight.shape[1] % self.group_size == 0

    def quantize(self, weight: torch.Tensor) -> dict[str, torch.Tensor]:
        q, scale = per_group_int8(weight, self.group_size)
        return {"": q, "_scale": scale}

    def scope_names(
        self, weight_map: dict[str, str], ref_dir: Path | None
    ) -> set[str]:
        if ref_dir is not None:
            idx = read_weight_map(ref_dir)
            out = set()
            for k in idx:
                for suf in (".weight_scale_inv", ".weight_scale"):
                    if k.endswith(suf):
                        out.add(k[: -len(suf)])
            return out
        # all-linear fallback
        return {
            k for k in weight_map
            if k.endswith(".weight")
            and "norm" not in k.rsplit(".", 2)[-2]
            and not any(k.endswith(e) for e in W8A16_SKIP_ENDINGS)
        }

    def quant_config(self) -> dict:
        return {"quant_method": "w8a16", "bits": 8, "group_size": self.group_size}

    def selfcheck(self) -> None:
        torch.manual_seed(0)
        rows, cols = 256, 512
        w = torch.randn(rows, cols, dtype=torch.bfloat16)
        w += 0.3 * torch.randn(rows, 1) * torch.randn(1, cols)
        q, scale = per_group_int8(w, W8A16_GROUP)
        assert q.shape == (rows, cols) and q.dtype == torch.int8
        assert scale.shape == (rows, cols // W8A16_GROUP)
        deq = (
            q.float().view(rows, cols // W8A16_GROUP, W8A16_GROUP)
            * scale.float().unsqueeze(-1)
        ).view(rows, cols)
        rel = ((deq - w.float()).norm() / w.float().norm()).item()
        # INT8 uniform grid must beat FP8 block-cast on the same data.
        fp8, sf = per_block_cast_to_fp8(w)
        fp8_rel = ((fp8_dequant(fp8, sf) - w.float()).norm() / w.float().norm()).item()
        assert rel < fp8_rel, f"w8a16 {rel:.4f} should beat fp8 {fp8_rel:.4f}"
        assert rel < 0.02, f"w8a16 rel err {rel:.4f} too high"
        print(f"w8a16 selfcheck ok: rel-L2 {rel:.4f} < fp8 {fp8_rel:.4f}")


# ---------------------------------------------------------------------------
# Registry + CLI
# ---------------------------------------------------------------------------

QUANTIZERS: dict[str, type[Quantizer]] = {
    "fp8": FP8BlockCastQuantizer,
    "w8a16": W8A16Quantizer,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", required=True, choices=list(QUANTIZERS),
                    help="quantization format")
    ap.add_argument("--selfcheck", action="store_true")
    # Generic checkpoint args
    ap.add_argument("--bf16", help="source BF16 checkpoint dir (fp8, w8a16)")
    ap.add_argument("--ref", help="reference quantized checkpoint (fp8, w8a16 scope)")
    ap.add_argument("--out", help="output dir (fp8, w8a16)")
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    qcls = QUANTIZERS[args.format]
    if args.format in ("w8a16",):
        quantizer = qcls(group_size=args.group_size)
    else:
        quantizer = qcls()

    if args.selfcheck:
        quantizer.selfcheck()
        return

    src = args.bf16
    out = args.out
    if not (src and out):
        ap.error(f"--format {args.format} requires source + output args")

    ref = Path(args.ref) if args.ref else None
    io = CheckpointIO(quantizer)
    io.run(Path(src), Path(out), ref_dir=ref)


if __name__ == "__main__":
    main()

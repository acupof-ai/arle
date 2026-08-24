"""Unified weight quantization framework.

Each quantizer implements `quantize(weight) -> {suffix: tensor}`,
`scope_names(weight_map, ref_dir)`, `quant_config()`, and `selfcheck()`.
`CheckpointIO` handles shard traversal, index.json, and config.json uniformly.

Usage:
  python scripts/quantize.py --format fp8 --bf16 <dir> --ref <fp8-ref> --out <dir>
  python scripts/quantize.py --format w8a16 --bf16 <dir> [--ref <w8a16-ref>] --out <dir>
  python scripts/quantize.py --format w4a8-marlin --src <dir> --dst <dir>
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
# W4A8 Marlin (pack_w4a8) — canonical implementation from quantize_qwen3_w4a8.py
# ---------------------------------------------------------------------------

W4A8_GROUP = 128


def _get_perms(groupsize: int, k: int):
    import numpy as np

    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [4 * (i % 4) + j for j in range(4)]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
    perm = np.array(perm)
    interleave = (
        np.array([4, 0, 5, 1, 6, 2, 7, 3])
        if groupsize == k
        else np.array([0, 2, 4, 6, 1, 3, 5, 7])
    )
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return torch.from_numpy(perm), scale_perm, scale_perm_single


def pack_w4a8(
    weight: torch.Tensor,
    groupsize: int = W4A8_GROUP,
    gptq_scales: torch.Tensor | None = None,
):
    """Pack BF16/FP16 weight to ARLE W4A8 Marlin format.

    Returns (qweight int32, s_channel float32 [1, out], s_group float16 [in/gs, out]).
    """
    import numpy as np

    weight = weight.to(dtype=torch.float16, device="cpu").contiguous()
    n, k = weight.shape
    if k % 128 != 0 or n % 256 != 0 or k % groupsize != 0:
        raise ValueError(f"unsupported W4A8 shape [{n}, {k}] groupsize={groupsize}")

    perm, scale_perm, scale_perm_single = _get_perms(groupsize, k)

    ref = weight.t().contiguous()
    s_channel = ref.t().abs().amax(dim=-1, keepdim=True).div(127.0).to(torch.float32)
    s_channel = torch.where(s_channel == 0, torch.ones_like(s_channel), s_channel)
    s_channel = s_channel.reshape(1, n)

    if gptq_scales is not None:
        s = gptq_scales.t().to(torch.float16).contiguous()
        if s.shape != (k // groupsize, n):
            raise ValueError(
                f"gptq_scales shape after transpose {tuple(s.shape)} "
                f"!= expected ({k // groupsize}, {n})"
            )
        max_s = 16.0 * s_channel.to(torch.float16).reshape(1, n)
        s = torch.minimum(s, max_s)
    else:
        reshaped = ref.reshape(k // groupsize, groupsize, n)
        s = reshaped.abs().amax(dim=1).clamp_min(1e-6).div(7.0).to(torch.float16)

    w = ref.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))
    s_work = s.reshape((1, -1))
    w = torch.round(w / s_work).to(torch.int32)
    w += 8
    w = torch.clamp(w, 0, 15)

    s_group = (s_work.reshape(-1, n) / s_channel).to(torch.float16)
    w = w.reshape((groupsize, -1, n)).permute(1, 0, 2).reshape((k, n)).contiguous()
    s_group = s_group.reshape((-1, len(scale_perm)))[:, scale_perm]
    s_group = s_group.reshape((-1, n)).contiguous()
    s_channel = s_channel.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s_channel = s_channel.reshape((-1, n)).contiguous()

    tile = 16
    w = w.reshape((k // tile, tile, n // tile, tile))
    w = w.permute((0, 2, 1, 3)).reshape((k // tile, n * tile))
    res = w.reshape((-1, perm.numel()))[:, perm].reshape(w.shape)
    res_np = res.cpu().numpy().astype(np.uint32)
    q = np.zeros((res_np.shape[0], res_np.shape[1] // 8), dtype=np.uint32)
    for i in range(8):
        q |= res_np[:, i::8] << (4 * i)
    qweight = torch.from_numpy(q.astype(np.int32))
    return qweight, s_channel.contiguous(), s_group.contiguous()


def is_w4a8_quantizable(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2 or not name.endswith(".weight"):
        return False
    if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
        return False
    out_features, in_features = tensor.shape
    return in_features % 128 == 0 and out_features % 256 == 0


class W4A8MarlinQuantizer(Quantizer):
    name = "w4a8-marlin"

    def can_quantize(self, weight: torch.Tensor) -> bool:
        return weight.dim() == 2 and weight.shape[0] % 256 == 0 and weight.shape[1] % 128 == 0

    def quantize(self, weight: torch.Tensor) -> dict[str, torch.Tensor]:
        qweight, s_channel, s_group = pack_w4a8(weight)
        return {
            ".marlin_w4a8_qweight": qweight,
            ".marlin_w4a8_s_channel": s_channel,
            ".marlin_w4a8_s_group": s_group,
        }

    def scope_names(
        self, weight_map: dict[str, str], ref_dir: Path | None
    ) -> set[str]:
        return {
            k for k in weight_map
            if k.endswith(".weight")
            and not k.endswith("embed_tokens.weight")
            and not k.endswith("lm_head.weight")
        }

    def quant_config(self) -> dict:
        return {"quant_type": "marlin_w4a8", "group_size": W4A8_GROUP}


# ---------------------------------------------------------------------------
# Registry + CLI
# ---------------------------------------------------------------------------

QUANTIZERS: dict[str, type[Quantizer]] = {
    "fp8": FP8BlockCastQuantizer,
    "w8a16": W8A16Quantizer,
    "w4a8-marlin": W4A8MarlinQuantizer,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", required=True, choices=list(QUANTIZERS),
                    help="quantization format")
    ap.add_argument("--selfcheck", action="store_true")
    # Generic checkpoint args
    ap.add_argument("--bf16", help="source BF16 checkpoint dir (fp8, w8a16)")
    ap.add_argument("--src", help="source checkpoint dir (w4a8-marlin)")
    ap.add_argument("--ref", help="reference quantized checkpoint (fp8, w8a16 scope)")
    ap.add_argument("--out", help="output dir (fp8, w8a16)")
    ap.add_argument("--dst", help="output dir (w4a8-marlin)")
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

    src = args.bf16 or args.src
    out = args.out or args.dst
    if not (src and out):
        ap.error(f"--format {args.format} requires source + output args")

    ref = Path(args.ref) if args.ref else None
    io = CheckpointIO(quantizer)
    io.run(Path(src), Path(out), ref_dir=ref)


if __name__ == "__main__":
    main()

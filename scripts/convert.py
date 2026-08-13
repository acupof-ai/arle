"""Shared checkpoint conversion IO.

All model-format conversions (GPTQ→W4A16, W4A16→Marlin, GPTQ→W4A8-Marlin,
W4A16+W4A8→hybrid) share the same IO skeleton: load tensors, transform a
subset, write back, patch quantization_config. This module provides those
shared helpers so each conversion script stays algorithm-only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CONFIG_FILES = [
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    "added_tokens.json", "merges.txt", "vocab.json",
]


def load_all_tensors(src: Path) -> dict[str, torch.Tensor]:
    """Load every tensor from a sharded or single-file safetensors checkpoint."""
    out: dict[str, torch.Tensor] = {}
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        files = sorted({v for v in json.loads(idx.read_text())["weight_map"].values()})
    else:
        files = [f.name for f in src.glob("*.safetensors")]
    for fname in files:
        with safe_open(src / fname, framework="pt") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k)
    return out


def save_checkpoint(
    tensors: dict[str, torch.Tensor],
    dst: Path,
    quant_config: dict | None = None,
    *,
    single_file: bool = True,
) -> None:
    """Write tensors to dst and patch config.json's quantization_config.

    When single_file=True all tensors go into model.safetensors and any
    existing model.safetensors.index.json is removed (the merge/repack
    outputs are small enough).
    """
    dst.mkdir(parents=True, exist_ok=True)
    if single_file:
        save_file(tensors, str(dst / "model.safetensors"))
        stale = dst / "model.safetensors.index.json"
        if stale.exists():
            stale.unlink()
    else:
        # Sharded write — caller manages sharding.
        raise NotImplementedError("sharded save not yet needed by conversions")

    if quant_config is not None:
        cfg_path = dst / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            cfg["quantization_config"] = quant_config
            cfg_path.write_text(json.dumps(cfg, indent=2))


def copy_config_files(src: Path, dst: Path) -> None:
    """Copy config/tokenizer files from src to dst (idempotent)."""
    for name in CONFIG_FILES:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)

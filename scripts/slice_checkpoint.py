#!/usr/bin/env python3
"""Slice an HF checkpoint to its first N language-model layers.

Profile / kernel-A/B vehicle: the output is a self-contained checkpoint in the
same format (config.json with num_hidden_layers=N and layer_types truncated,
all non-layer weights kept — embedding, lm_head, norms, visual, mtp), so the
engine loads it unmodified. A 2-layer slice of the 27B NVFP4 model is ~1 GB
and loads in seconds versus minutes for the full checkpoint, while every
decode kernel path (packed weights, router, attention) is the real one.

Usage: python3 slice_checkpoint.py <src> <dst> [--layers N]
"""

import argparse
import json
import os
import re
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", type=int, default=2)
    a = ap.parse_args()

    os.makedirs(a.dst, exist_ok=True)

    cfg = json.load(open(f"{a.src}/config.json"))
    tc = cfg["text_config"]
    tc["num_hidden_layers"] = a.layers
    tc["layer_types"] = tc["layer_types"][:a.layers]
    json.dump(cfg, open(f"{a.dst}/config.json", "w"), indent=2)

    idx = json.load(open(f"{a.src}/model.safetensors.index.json"))
    keep, dropped = [], 0
    for k in idx["weight_map"]:
        m = LAYER_RE.match(k)
        if m and int(m.group(1)) >= a.layers:
            dropped += 1
        else:
            keep.append(k)

    by_shard: dict[str, list[str]] = {}
    for k in keep:
        by_shard.setdefault(idx["weight_map"][k], []).append(k)

    tensors = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(f"{a.src}/{shard}", framework="pt") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k)
    save_file(tensors, f"{a.dst}/model.safetensors")
    json.dump(
        {"metadata": {}, "weight_map": {k: "model.safetensors" for k in keep}},
        open(f"{a.dst}/model.safetensors.index.json", "w"),
    )

    for name in os.listdir(a.src):
        if name.endswith(".safetensors") or name.endswith(".bak"):
            continue
        if name in ("config.json", "model.safetensors.index.json"):
            continue
        p = f"{a.src}/{name}"
        if os.path.isfile(p) and os.path.getsize(p) < 100_000_000:
            shutil.copy2(p, f"{a.dst}/{name}")

    print(f"kept {len(keep)} keys, dropped {dropped} layer keys -> {a.dst}")


if __name__ == "__main__":
    main()

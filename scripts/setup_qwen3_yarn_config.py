#!/usr/bin/env python3
"""Patch Qwen3 model config.json with YARN/Linear/NtkAware rope_scaling.

Creates a sibling copy of the model dir (preserves original) with
rope_scaling injected. Output dir name encodes the scaling for
reproducibility.

Usage:
  ./scripts/setup_qwen3_yarn_config.py \\
      --src infer/models/Qwen3-4B \\
      --type yarn --factor 2.0 --orig-max-pos 40960
  # → creates infer/models/Qwen3-4B-yarn-f2.0/

  # --in-place to mutate the source config.json (no copy)
  ./scripts/setup_qwen3_yarn_config.py --src infer/models/Qwen3-4B \\
      --type linear --factor 2.0 --in-place

Per docs/plans/2026-05-10-rope-yarn-phase3-cuda-bench-plan.md Phase 3a/b/c.
Tests M_rope-yarn-scaling Phase 1+2 wire (qwen3-spec config parse + inv_freq
compute + weight_loader.rs precompute_rope_with_scaling).
"""

from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path


def build_rope_scaling(scaling_type: str, factor: float, orig_max_pos: int | None) -> dict:
    if scaling_type == "yarn":
        if orig_max_pos is None:
            raise SystemExit("yarn requires --orig-max-pos")
        return {
            "type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": orig_max_pos,
        }
    elif scaling_type == "linear":
        return {"type": "linear", "factor": factor}
    elif scaling_type == "ntk_aware":
        return {"type": "ntk_aware", "factor": factor}
    else:
        raise SystemExit(f"unknown scaling type: {scaling_type}")


def patch_config(config_path: Path, rope_scaling: dict, max_pos_override: int | None) -> dict:
    """Load config.json, inject rope_scaling, optionally bump
    max_position_embeddings to fit the extended context."""
    cfg = json.loads(config_path.read_text())
    cfg["rope_scaling"] = rope_scaling
    if max_pos_override:
        cfg["max_position_embeddings"] = max_pos_override
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="source model dir (with config.json)")
    p.add_argument("--type", choices=["yarn", "linear", "ntk_aware"], required=True)
    p.add_argument("--factor", type=float, required=True, help="scaling factor")
    p.add_argument("--orig-max-pos", type=int, default=None, help="YARN original_max_position_embeddings (required for yarn)")
    p.add_argument("--max-pos", type=int, default=None, help="extended max_position_embeddings (default: factor * orig-max-pos for yarn, factor * src config max for linear/ntk)")
    p.add_argument("--in-place", action="store_true", help="modify src config.json in place (no copy)")
    p.add_argument("--out", type=Path, default=None, help="output dir (default: <src>-<type>-f<factor>/)")

    args = p.parse_args()
    src_config = args.src / "config.json"
    if not src_config.exists():
        raise SystemExit(f"no config.json in {args.src}")

    rope_scaling = build_rope_scaling(args.type, args.factor, args.orig_max_pos)

    src_cfg = json.loads(src_config.read_text())
    src_max = src_cfg.get("max_position_embeddings", 0)

    if args.max_pos:
        new_max = args.max_pos
    elif args.type == "yarn":
        new_max = int(args.factor * args.orig_max_pos)
    else:
        new_max = int(args.factor * src_max)

    new_cfg = patch_config(src_config, rope_scaling, new_max)

    if args.in_place:
        target = args.src
        target_cfg = src_config
        print(f"[in-place] patching {target_cfg}")
    else:
        if args.out:
            target = args.out
        else:
            target = args.src.parent / f"{args.src.name}-{args.type}-f{args.factor}"
        if target.exists():
            raise SystemExit(f"output dir exists: {target} (rm or pass --out)")
        print(f"[copy] {args.src} → {target}")
        shutil.copytree(args.src, target, symlinks=True)
        target_cfg = target / "config.json"

    target_cfg.write_text(json.dumps(new_cfg, indent=2) + "\n")
    print(f"[done] rope_scaling={rope_scaling}")
    print(f"[done] max_position_embeddings={new_max} (was {src_max})")
    print(f"[done] config: {target_cfg}")
    print()
    print("Run bench with:")
    print(f"  --model-path {target}")


if __name__ == "__main__":
    main()

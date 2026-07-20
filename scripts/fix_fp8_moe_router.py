#!/usr/bin/env python3
"""Restore the MoE router (gate) tensors of an FP8 MoE checkpoint to bf16.

A broken FP8 export (observed: `ThinkingCap-Qwen3.6-27B-FP8`) quantized the MoE
router weights `*.mlp.gate.weight` and `*.mlp.shared_expert_gate.weight` to FP8.
Routing is discrete: a ~1-2% FP8 weight perturbation flips top-k expert
selection, so greedy (top-1) survives but temp>0 samples the scrambled tail into
multilingual salad. A correctly-exported sibling (`Qwen3.6-27B-FP8`) keeps every
router in bf16 and is coherent at temp=1.0.

This surgically swaps ONLY the router tensors back to bf16 (copied from the
original bf16 checkpoint) and drops their FP8 `weight_scale_inv` sidecars. Every
FP8 expert / attention / MLP weight is copied byte-for-byte — experts stay FP8,
the memory win is preserved. It is NOT a re-quantization.

Router tensors are matched by EXACT suffix so `.mlp.gate.weight` (the router) is
never confused with `.mlp.gate_proj.weight` (a dense FFN projection) or
`.mlp.shared_expert.gate_proj.weight` (the shared expert's FFN gate).

The output's `quantization_config.modules_to_not_convert` is taken from a known-
good reference config (`--ref-config`, e.g. the base FP8 model) so the runtime
loader recognizes the bf16 routers as DenseBf16 (see QuantManifest::ignored — a
bf16 tensor in an FP8 manifest must be `ignored()` or it fails to resolve). The
script then self-validates that every swapped router IS matched by that list.

Usage:
  fix_fp8_moe_router.py --fp8-dir BROKEN_FP8 --bf16-dir ORIG_BF16 \
      --out-dir FIXED_FP8 --ref-config GOOD_FP8/config.json
  fix_fp8_moe_router.py --dry-run ...        # header-only, lists the swaps
  fix_fp8_moe_router.py --self-test          # unit-test the pure logic, no I/O
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Exact suffixes of the MoE routing gates (bf16 in a correct export).
# `.mlp.gate.weight` != `.mlp.gate_proj.weight`; `.mlp.shared_expert_gate.weight`
# != `.mlp.shared_expert.gate_proj.weight` — suffix match keeps them distinct.
ROUTER_SUFFIXES = (".mlp.gate.weight", ".mlp.shared_expert_gate.weight")


def is_router(name: str) -> bool:
    return name.endswith(ROUTER_SUFFIXES)


def scale_name(weight_name: str) -> str:
    """The FP8 block-scale sidecar our loader pairs with a weight."""
    return weight_name[: -len(".weight")] + ".weight_scale_inv"


def ignored_by(name: str, not_convert: list[str]) -> bool:
    """Mirror QuantManifest::ignored — starts_with over the not_convert prefixes."""
    return any(name.startswith(prefix) for prefix in not_convert)


def load_not_convert(ref_config: Path) -> list[str]:
    cfg = json.loads(ref_config.read_text())
    qc = cfg.get("quantization_config", {})
    # QuantManifest::ignored chains BOTH keys; mirror it or a ref that lists
    # routers under `ignore` would look empty here.
    nc = list(qc.get("modules_to_not_convert", [])) + list(qc.get("ignore", []))
    if not nc:
        raise SystemExit(f"{ref_config}: quantization_config has no modules_to_not_convert/ignore")
    return nc


def plan_swaps(weight_map: dict[str, str]) -> tuple[list[str], list[str]]:
    """Return (router weights to swap, scale sidecars to drop)."""
    routers = sorted(n for n in weight_map if is_router(n))
    scales = [scale_name(n) for n in routers if scale_name(n) in weight_map]
    return routers, scales


def validate_coverage(routers: list[str], not_convert: list[str]) -> None:
    """Every swapped-to-bf16 router MUST be ignored() by the output manifest,
    else the loader can't resolve it (QuantManifest::ignored)."""
    missing = [n for n in routers if not ignored_by(n, not_convert)]
    if missing:
        raise SystemExit(
            "modules_to_not_convert does not cover these routers (loader would "
            f"reject them):\n  " + "\n  ".join(missing[:10])
            + (f"\n  ... (+{len(missing) - 10} more)" if len(missing) > 10 else "")
        )


def read_weight_map(model_dir: Path) -> dict[str, str]:
    """tensor name -> shard filename, for single- or multi-shard checkpoints."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    singles = list(model_dir.glob("*.safetensors"))
    if len(singles) != 1:
        raise SystemExit(f"{model_dir}: no index.json and not exactly one .safetensors")
    from safetensors import safe_open

    with safe_open(singles[0], framework="numpy") as f:
        return {k: singles[0].name for k in f.keys()}


def run(args: argparse.Namespace) -> None:
    fp8_dir, bf16_dir, out_dir = Path(args.fp8_dir), Path(args.bf16_dir), Path(args.out_dir)
    not_convert = load_not_convert(Path(args.ref_config))

    fp8_map = read_weight_map(fp8_dir)
    bf16_map = read_weight_map(bf16_dir)
    routers, scales = plan_swaps(fp8_map)

    if not routers:
        raise SystemExit(f"{fp8_dir}: no `.mlp.gate.weight`/`.shared_expert_gate.weight` found")
    missing_src = [n for n in routers if n not in bf16_map]
    if missing_src:
        raise SystemExit(f"bf16 source lacks {len(missing_src)} routers, e.g. {missing_src[:3]}")
    validate_coverage(routers, not_convert)

    already_bf16 = sum(1 for n in routers if scale_name(n) not in fp8_map)
    print(f"routers found : {len(routers)} ({len(routers) - already_bf16} FP8, {already_bf16} already bf16)")
    print(f"scales to drop: {len(scales)}")
    print(f"not_convert   : {len(not_convert)} prefixes from {args.ref_config}")
    if args.dry_run:
        for n in routers[:6]:
            tag = "DROP-scale+swap" if scale_name(n) in fp8_map else "already-bf16"
            print(f"  {tag:16} {n}")
        print("  ... (dry-run, no files written)")
        return

    from safetensors import safe_open
    from safetensors.numpy import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(set(fp8_map.values()))
    # Which bf16 shard holds each router we must pull in.
    router_set = set(routers)
    scale_set = set(scales)
    new_weight_map: dict[str, str] = {}

    for shard in shards:
        names = [n for n, s in fp8_map.items() if s == shard]
        swap_here = [n for n in names if n in router_set]
        drop_here = [n for n in names if n in scale_set]
        keep = [n for n in names if n not in scale_set]  # weights + non-router stay; scales dropped

        if not swap_here and not drop_here:
            shutil.copyfile(fp8_dir / shard, out_dir / shard)  # untouched shard: byte copy
            for n in names:
                new_weight_map[n] = shard
            print(f"copy  {shard} ({len(names)} tensors)")
            continue

        tensors = {}
        with safe_open(fp8_dir / shard, framework="numpy") as f:
            for n in keep:
                tensors[n] = f.get_tensor(n)
        for n in swap_here:  # overwrite router with the bf16 version
            src_shard = bf16_map[n]
            with safe_open(bf16_dir / src_shard, framework="numpy") as f:
                tensors[n] = f.get_tensor(n)
        save_file(tensors, str(out_dir / shard))
        for n in keep:
            new_weight_map[n] = shard
        print(f"write {shard} (+{len(swap_here)} bf16 routers, -{len(drop_here)} scales)")

    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": new_weight_map}, indent=2)
    )

    # config.json: copy fp8-dir's, override modules_to_not_convert with the ref list.
    cfg = json.loads((fp8_dir / "config.json").read_text())
    cfg.setdefault("quantization_config", {})["modules_to_not_convert"] = not_convert
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Carry the loose companions verbatim (tokenizer, generation_config, ...).
    for f in fp8_dir.iterdir():
        if f.suffix == ".safetensors" or f.name in ("config.json", "model.safetensors.index.json"):
            continue
        if f.is_file():
            shutil.copyfile(f, out_dir / f.name)

    print(f"\ndone -> {out_dir}")
    print("VERIFY temp=1.0 top_k20 top_p0.95 coherent + PPL ~= bf16 before trusting it.")


def self_test() -> None:
    # suffix discrimination: router vs dense FFN gate
    assert is_router("model.language_model.layers.3.mlp.gate.weight")
    assert is_router("model.language_model.layers.3.mlp.shared_expert_gate.weight")
    assert not is_router("model.language_model.layers.3.mlp.gate_proj.weight")
    assert not is_router("model.language_model.layers.3.mlp.shared_expert.gate_proj.weight")
    assert not is_router("model.language_model.layers.3.mlp.experts.7.gate_proj.weight")
    # scale sidecar name
    assert scale_name("x.mlp.gate.weight") == "x.mlp.gate.weight_scale_inv"
    # ignored() semantics: starts_with over full names
    r = "model.language_model.layers.3.mlp.gate.weight"
    assert ignored_by(r, ["model.language_model.layers.3.mlp.gate"])
    assert not ignored_by(r, ["mlp.gate"])  # bare suffix does NOT match a full name
    # planner + coverage
    wm = {
        "a.mlp.gate.weight": "s0", "a.mlp.gate.weight_scale_inv": "s0",
        "a.mlp.gate_proj.weight": "s0", "a.mlp.gate_proj.weight_scale_inv": "s0",
        "a.mlp.shared_expert_gate.weight": "s1",
    }
    routers, scales = plan_swaps(wm)
    assert routers == ["a.mlp.gate.weight", "a.mlp.shared_expert_gate.weight"], routers
    assert scales == ["a.mlp.gate.weight_scale_inv"], scales  # shared_expert_gate had no scale here
    validate_coverage(routers, ["a.mlp.gate", "a.mlp.shared_expert_gate"])
    try:
        validate_coverage(routers, ["a.mlp.gate"])  # misses shared_expert_gate
    except SystemExit:
        pass
    else:
        raise AssertionError("expected coverage failure")
    print("self-test OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="unit-test the pure logic, no I/O")
    p.add_argument("--dry-run", action="store_true", help="header-only; list the swaps, write nothing")
    p.add_argument("--fp8-dir", help="the broken FP8 checkpoint (router quantized)")
    p.add_argument("--bf16-dir", help="the original bf16 checkpoint (router source)")
    p.add_argument("--out-dir", help="output dir for the fixed FP8 checkpoint")
    p.add_argument("--ref-config", help="known-good FP8 config.json to copy modules_to_not_convert from")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    for req in ("fp8_dir", "bf16_dir", "out_dir", "ref_config"):
        if not getattr(args, req):
            p.error(f"--{req.replace('_', '-')} is required (or use --self-test)")
    run(args)


if __name__ == "__main__":
    main()

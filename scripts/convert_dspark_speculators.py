#!/usr/bin/env python3
"""Convert a speculators-format DSpark/DFlash draft checkpoint to the
deepspec-native layout `qwen35_spec::DsparkConfig::from_dir` parses.

Tensor names already match the deepspec contract (verified against
pablohassan/Qwen3.6-27B-DSpark-FR); only config.json needs rewriting, so the
safetensors file is hardlinked (copy fallback) after a header sanity check.

Usage: convert_dspark_speculators.py <src_dir> <dst_dir>
"""

import json
import os
import shutil
import struct
import sys
from pathlib import Path

BACKBONE = ["fc.weight", "hidden_norm.weight", "norm.weight",
            "layers.0.self_attn.q_proj.weight"]


def safetensor_names(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return set(header) - {"__metadata__"}


def convert_config(src):
    t = src["transformer_layer_config"]
    proposal = src["speculators_config"]["proposal_methods"][0]
    block_size = src["block_size"]
    cfg = {
        "architectures": ["DSparkDraftModel"],
        "num_hidden_layers": t["num_hidden_layers"],
        "hidden_size": t["hidden_size"],
        "intermediate_size": t["intermediate_size"],
        "num_attention_heads": t["num_attention_heads"],
        "num_key_value_heads": t["num_key_value_heads"],
        "head_dim": t["head_dim"],
        "rms_norm_eps": t["rms_norm_eps"],
        "layer_types": t["layer_types"],
        "block_size": block_size,
        "markov_rank": src.get("markov_rank", 0),
        "markov_head_type": src.get("markov_head_type"),
        "enable_confidence_head": src.get("enable_confidence_head", False),
        "confidence_head_with_markov": src.get("confidence_head_with_markov", False),
    }
    for k in ("rope_theta", "sliding_window"):
        if t.get(k) is not None:
            cfg[k] = t[k]
    # speculators aux ids index hidden_states (embedding = 0, layer i = i+1);
    # deepspec target_layer_ids are layer indices — shift by -1.
    ids = [i - 1 for i in src["aux_hidden_state_layer_ids"]]
    # speculative_tokens == block_size-1 marks the same-position (DFlash) row
    # convention; DsparkConfig keys next_token_heads off dflash_config nesting.
    if proposal["speculative_tokens"] == block_size - 1:
        cfg["dflash_config"] = {"mask_token_id": src["mask_token_id"],
                                "target_layer_ids": ids}
    else:
        cfg["mask_token_id"] = src["mask_token_id"]
        cfg["target_layer_ids"] = ids
    return cfg


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[-1])
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
    src = json.loads((src_dir / "config.json").read_text())
    if src.get("speculators_model_type") not in ("dspark", "dflash"):
        sys.exit(f"not a speculators dspark/dflash config: {src_dir}")

    weights = src_dir / "model.safetensors"
    names = safetensor_names(weights)
    missing = [n for n in BACKBONE if n not in names]
    if missing:
        sys.exit(f"backbone tensors missing from {weights}: {missing}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    cfg = convert_config(src)
    (dst_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    dst = dst_dir / "model.safetensors"
    if not dst.exists():
        try:
            os.link(weights, dst)
        except OSError:
            shutil.copyfile(weights, dst)
    conv = "same-position" if "dflash_config" in cfg else "next-token"
    print(f"converted -> {dst_dir} ({conv}, block={cfg['block_size']}, "
          f"markov={cfg['markov_rank']}, conf={cfg['enable_confidence_head']})")


if __name__ == "__main__":
    main()

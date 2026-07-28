#!/usr/bin/env python3
"""Generic SVD analysis for MoE models. Auto-detects tensor naming.

Works with DSv4 (layers.X.ffn.experts.E.w1.weight) and
GLM-5.2 (whatever naming it uses). Loads all experts per layer per
projection type, runs truncated SVD, computes spectral/frame similarity.
"""

from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

MODEL = Path(os.environ.get("SVD_MODEL", "/host/glm52"))
TOPK = 32
SAMPLE = int(os.environ.get("SVD_SAMPLE", "256"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def discover():
    """Auto-detect layer count, expert IDs, and projection names."""
    idx_path = MODEL / "model.safetensors.index.json"
    if not idx_path.exists():
        # try single safetensors
        sts = sorted(MODEL.glob("*.safetensors"))
        if not sts:
            raise RuntimeError(f"No model index or safetensors in {MODEL}")
        # build a fake index from the first file
        with safe_open(sts[0], framework="pt") as f:
            keys = list(f.keys())
        wm = {k: sts[0].name for k in keys}
    else:
        idx = json.loads(idx_path.read_text())
        wm = idx["weight_map"]

    # find expert weight tensors
    expert_keys = [k for k in wm if "experts" in k and k.endswith(".weight")]
    if not expert_keys:
        raise RuntimeError(f"No expert weight tensors found in {MODEL}")

    # detect layer numbers
    layer_nums = set()
    for k in expert_keys:
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            layer_nums.add(int(m.group(1)))
    layers = sorted(layer_nums)

    # detect expert IDs
    expert_ids = set()
    for k in expert_keys:
        m = re.search(r"experts\.(\d+)\.", k)
        if m:
            expert_ids.add(int(m.group(1)))
    experts = sorted(expert_ids)

    # detect projection names (the part after experts.N.)
    proj_names = set()
    for k in expert_keys:
        m = re.search(r"experts\.\d+\.(.+)\.weight", k)
        if m:
            proj_names.add(m.group(1))
    # filter to likely projections (w1/w2/w3 or gate/down/up)
    proj_names = sorted(proj_names)

    return layers, experts, proj_names, wm


def sample_experts(experts, n):
    if len(experts) <= n:
        return experts
    idx = torch.linspace(0, len(experts) - 1, n).long().tolist()
    return [experts[i] for i in idx]


def load_all(wm, layer, experts, proj):
    """Load all expert weights for one layer+projection, handle FP8 scales."""
    names = []
    for e in experts:
        # try common patterns
        candidates = [
            f"layers.{layer}.ffn.experts.{e}.{proj}.weight",
            f"model.layers.{layer}.ffn.experts.{e}.{proj}.weight",
            f"transformer.layers.{layer}.ffn.experts.{e}.{proj}.weight",
        ]
        for c in candidates:
            if c in wm:
                names.append(c)
                break
        else:
            # fallback: search wm for matching key
            for k in wm:
                if f"layers.{layer}." in k and f"experts.{e}." in k and k.endswith(f".{proj}.weight"):
                    names.append(k)
                    break
    if not names:
        return None

    by_shard = {}
    for i, n in enumerate(names):
        by_shard.setdefault(wm[n], []).append((i, n))
    ws = [None] * len(names)
    has_scale = True
    ss = [None] * len(names)
    for shard, items in by_shard.items():
        with safe_open(MODEL / shard, framework="torch") as f:
            for i, n in items:
                ws[i] = f.get_tensor(n)
                sc = n.replace(".weight", ".scale")
                if sc in f.keys():
                    ss[i] = f.get_tensor(sc)
                else:
                    has_scale = False
    w = torch.stack(ws).to(torch.float32)
    if has_scale and all(s is not None for s in ss):
        s = torch.stack(ss)
        reps = [1, w.shape[1] // s.shape[1], w.shape[2] // s.shape[2]]
        s = s.repeat(*reps)
        w = w * s
    return w


def analyze(w):
    w = w.to(DEV)
    B = w.shape[0]
    U, S, V = torch.svd_lowrank(w, q=TOPK, niter=2)
    Sn = S / S.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    cos = Sn @ Sn.T
    mask = ~torch.eye(B, dtype=torch.bool, device=cos.device)
    spec_cos_mean = float(cos[mask].mean())
    Uk = U.reshape(B * TOPK, -1)
    gu = (Uk @ Uk.T).reshape(B, TOPK, B, TOPK).permute(0, 2, 1, 3)
    Vk = V.reshape(B * TOPK, -1)
    gv = (Vk @ Vk.T).reshape(B, TOPK, B, TOPK).permute(0, 2, 1, 3)
    su = torch.linalg.svdvals(gu.reshape(-1, TOPK, TOPK)).reshape(B, B, TOPK)
    sv = torch.linalg.svdvals(gv.reshape(-1, TOPK, TOPK)).reshape(B, B, TOPK)
    frame_u = float(su[..., 0][mask].mean())
    frame_v = float(sv[..., 0][mask].mean())
    s_max = float(S[:, 0].mean())
    # effective rank
    eff_rank = float((S.sum(dim=-1) / S[:, 0].clamp(min=1e-12)).mean())
    return {
        "spec_cos_mean": spec_cos_mean,
        "frame_sim_u": frame_u,
        "frame_sim_v": frame_v,
        "s_max_mean": s_max,
        "eff_rank_mean": eff_rank,
    }


# map projection names to standard labels
PROJ_LABEL = {
    "w1": "gate_proj", "w2": "down_proj", "w3": "up_proj",
    "gate_proj": "gate_proj", "down_proj": "down_proj", "up_proj": "up_proj",
    "gate": "gate_proj", "down": "down_proj", "up": "up_proj",
}


def main():
    print(f"Model: {MODEL}", file=sys.stderr)
    layers, experts, proj_names, wm = discover()
    print(f"Layers: {len(layers)}, Experts: {len(experts)}, Projections: {proj_names}", file=sys.stderr)

    sampled = sample_experts(experts, SAMPLE)
    print(f"Sampled {len(sampled)} experts", file=sys.stderr)

    results = {"per_layer": {}, "meta": {
        "num_layers": len(layers),
        "num_experts": len(experts),
        "sampled_experts": len(sampled),
        "projections": [PROJ_LABEL.get(p, p) for p in proj_names],
    }}

    t0 = time.time()
    for li, layer in enumerate(layers):
        layer_res = {}
        for proj in proj_names:
            label = PROJ_LABEL.get(proj, proj)
            w = load_all(wm, layer, sampled, proj)
            if w is None:
                print(f"  layer {layer} {proj}: not found", file=sys.stderr)
                continue
            r = analyze(w)
            layer_res[label] = r
            del w
            torch.cuda.empty_cache() if DEV == "cuda" else None
        results["per_layer"][str(layer)] = layer_res
        if li % 5 == 0:
            print(f"  layer {layer}/{layers[-1]} done ({time.time()-t0:.0f}s)", file=sys.stderr)

    print(json.dumps(results))


if __name__ == "__main__":
    main()

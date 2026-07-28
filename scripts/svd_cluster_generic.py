#!/usr/bin/env python3
"""Generic clustering analysis for MoE models. Auto-detects tensor naming.

Outputs: heatmap of cluster ratios (k=2/4/8/16) per layer, and 2D MDS
scatter for key layers. Works with DSv4 and GLM-5.2.
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
SCATTER_LAYERS = [0, 10, 21, 30, 42]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

PROJ_LABEL = {
    "w1": "gate_proj", "w2": "down_proj", "w3": "up_proj",
    "gate_proj": "gate_proj", "down_proj": "down_proj", "up_proj": "up_proj",
}


def discover():
    idx_path = MODEL / "model.safetensors.index.json"
    if not idx_path.exists():
        sts = sorted(MODEL.glob("*.safetensors"))
        with safe_open(sts[0], framework="pt") as f:
            keys = list(f.keys())
        wm = {k: sts[0].name for k in keys}
    else:
        idx = json.loads(idx_path.read_text())
        wm = idx["weight_map"]

    expert_keys = [k for k in wm if "experts" in k and k.endswith(".weight")]
    layer_nums = set()
    for k in expert_keys:
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            layer_nums.add(int(m.group(1)))
    layers = sorted(layer_nums)

    expert_ids = set()
    for k in expert_keys:
        m = re.search(r"experts\.(\d+)\.", k)
        if m:
            expert_ids.add(int(m.group(1)))
    experts = sorted(expert_ids)

    proj_names = set()
    for k in expert_keys:
        m = re.search(r"experts\.\d+\.(.+)\.weight", k)
        if m:
            proj_names.add(m.group(1))
    # prefer gate_proj / w1 for clustering
    proj_names = sorted(proj_names)
    gate_proj = None
    for p in proj_names:
        if PROJ_LABEL.get(p) == "gate_proj":
            gate_proj = p
            break
    if gate_proj is None:
        gate_proj = proj_names[0]

    return layers, experts, gate_proj, wm


def sample_experts(experts, n):
    if len(experts) <= n:
        return experts
    idx = torch.linspace(0, len(experts) - 1, n).long().tolist()
    return [experts[i] for i in idx]


def load_all(wm, layer, experts, proj):
    names = []
    for e in experts:
        for c in [
            f"layers.{layer}.ffn.experts.{e}.{proj}.weight",
            f"model.layers.{layer}.ffn.experts.{e}.{proj}.weight",
            f"transformer.layers.{layer}.ffn.experts.{e}.{proj}.weight",
        ]:
            if c in wm:
                names.append(c)
                break
        else:
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
    ss = [None] * len(names)
    has_scale = True
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


def spectral_cosine(w):
    B = w.shape[0]
    U, S, V = torch.svd_lowrank(w, q=TOPK, niter=2)
    Sn = S / S.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return Sn @ Sn.T  # [B, B]


def cluster_ratio(sim, k):
    """k-means clustering, return intra/inter mean ratio."""
    B = sim.shape[0]
    if B < k:
        return None
    # use spectral embedding for clustering
    # eigendecompose similarity
    w, v = torch.linalg.eigh(sim)
    emb = v[:, -k:]  # top-k eigenvectors
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    # k-means
    centroids = emb[torch.randperm(B)[:k]]
    for _ in range(20):
        dists = ((emb[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        labels = dists.argmin(-1)
        new_c = torch.stack([emb[labels == i].mean(0) if (labels == i).any() else centroids[i] for i in range(k)])
        if torch.allclose(new_c, centroids):
            break
        centroids = new_c
    # intra / inter ratio
    intra = []
    inter = []
    for i in range(B):
        for j in range(i + 1, B):
            if labels[i] == labels[j]:
                intra.append(sim[i, j].item())
            else:
                inter.append(sim[i, j].item())
    if not intra or not inter:
        return None
    return sum(intra) / len(intra) / (sum(inter) / len(inter))


def mds_2d(sim):
    """Classical MDS to 2D."""
    B = sim.shape[0]
    D = (1 - sim).clamp(min=0)
    D2 = D ** 2
    H = torch.eye(B, device=sim.device) - torch.ones(B, B, device=sim.device) / B
    S = -0.5 * H @ D2 @ H
    w, v = torch.linalg.eigh(S)
    idx = torch.argsort(w, descending=True)[:2]
    return (v[:, idx] * w[idx].sqrt().clamp(min=0)).cpu().tolist()


def main():
    print(f"Model: {MODEL}", file=sys.stderr)
    layers, experts, gate_proj, wm = discover()
    print(f"Layers: {len(layers)}, Experts: {len(experts)}, gate_proj key: {gate_proj}", file=sys.stderr)

    sampled = sample_experts(experts, SAMPLE)
    print(f"Sampled {len(sampled)} experts", file=sys.stderr)

    # pick scatter layers from available layers
    scatter_layers = [l for l in SCATTER_LAYERS if l in layers]
    if not scatter_layers:
        scatter_layers = [layers[0], layers[len(layers)//4], layers[len(layers)//2],
                         layers[3*len(layers)//4], layers[-1]]

    heatmap = {str(k): [] for k in [2, 4, 8, 16]}
    scatter = {}

    t0 = time.time()
    for li, layer in enumerate(layers):
        w = load_all(wm, layer, sampled, gate_proj)
        if w is None:
            print(f"  layer {layer}: not found", file=sys.stderr)
            continue
        sim = spectral_cosine(w)
        for k in [2, 4, 8, 16]:
            r = cluster_ratio(sim, k)
            heatmap[str(k)].append(r if r else 1.0)
        if layer in scatter_layers:
            coords = mds_2d(sim)
            scatter[str(layer)] = {"coords": coords}
        del w, sim
        torch.cuda.empty_cache() if DEV == "cuda" else None
        if li % 5 == 0:
            print(f"  layer {layer}/{layers[-1]} done ({time.time()-t0:.0f}s)", file=sys.stderr)

    result = {
        "heatmap": {"layers": layers, "ks": [2, 4, 8, 16], "gate_proj": heatmap},
        "scatter": scatter,
        "meta": {"num_layers": len(layers), "num_experts": len(experts),
                 "sampled": len(sampled), "gate_proj_key": gate_proj},
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()

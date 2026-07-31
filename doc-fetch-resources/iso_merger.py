"""ISO-Merger: data-free composition of shared-base RL experts in
fixed-spectrum Stiefel coordinates (frozen NIPS-submission configuration).

Given a base model W_0 = U_0 Sigma_0 V_0^T and K RL experts
W_i = U_i Sigma_i V_i^T fine-tuned from that same base, this script builds
a single merged model

    W_* = U_* Sigma_0 V_*^T

that reuses the base spectrum Sigma_0 (never modified) and composes the
experts' singular-frame changes in the Stiefel tangent spaces at
(U_0, V_0).

Algorithm (per 2D weight matrix; all computation in float32):

  Step 1.  Thin SVD of the base:  W_0 = U_0 Sigma_0 V_0^T.

  Step 2.  For each expert i, thin SVD W_i = U_i Sigma_i V_i^T and
           sign-canonicalize each singular pair against the base:
               s_k = sign(<u_{0,k}, u_{i,k}>)   (sign(0) := +1)
               u_{i,k} <- s_k u_{i,k},   v_{i,k} <- s_k v_{i,k}
           (the SVD determines each pair only up to a joint sign; without
           alignment the frame displacement is gauge-ambiguous).

  Step 3.  Stiefel-tangent projection of the frame displacements:
               xi_{u,i} = Pi_{U_0}(U_i - U_0)
               xi_{v,i} = Pi_{V_0}(V_i - V_0)
           where  Pi_X(Y) = Y - X sym(X^T Y)  is the orthogonal projection
           onto the tangent space at X (matrices satisfying
           X^T xi + xi^T X = 0).

  Step 4.  Top-k singular-mode masking (keep ratio RHO_KEEP = 0.9):
           zero out tangent columns r >= round(RHO_KEEP * q). Masking is a
           coordinate-selection operation — the masked xi need not satisfy
           the tangent constraint; feasibility is restored only after
           aggregating the experts (Step 7).

  Step 5.  Per-expert first-order weight-effect proxy:
               g_i = xi_{u,i} Sigma_0 V_0^T + U_0 Sigma_0 xi_{v,i}^T

  Step 6.  Retention coefficients with a unit-retention target:
               Gamma_{ij} = <g_i, g_j>_F
               (Gamma + RIDGE * I) c = diag(Gamma)
               c* = clip(c, [C_CLIP_MIN, C_CLIP_MAX])
           c solves ret_i(c) = (Gamma c)_i / Gamma_{ii} = 1 for every i,
           so each expert's own first-order weight effect survives at unit
           strength in the merged proxy. The ridge stabilizes an
           ill-conditioned Gram matrix and the clip suppresses extreme
           coefficients (sign reversal / over-amplification).

  Step 7.  Aggregate, project, retract, reconstruct:
               xi_{u,*} = Pi_{U_0}(sum_i c*_i xi_{u,i})
               xi_{v,*} = Pi_{V_0}(sum_i c*_i xi_{v,i})
               U_* = polar(U_0 + xi_{u,*})
               V_* = polar(V_0 + xi_{v,*})
               W_* = U_* Sigma_0 V_*^T
           where polar(X) = P Q^T for the thin SVD X = P S Q^T. Because
           U_* and V_* have orthonormal columns, W_* shares the base
           singular values Sigma_0 up to floating-point error.

Parameter scope:
  - Per-layer 2D projection matrices (attention + linear-attn + MLP) are
    merged with the fixed-spectrum Stiefel construction above.
  - The embedding / unembedding matrices (embed_tokens, lm_head) and all 1D
    parameters (LayerNorm scales, attention biases) are composed with a
    uniform task-vector average:  w_* = w_0 + (1/K) sum_i (w_i - w_0). RL
    barely rotates the embed/lm_head frames (measured c*~0), so Stiefel
    there is a costly no-op.
  - Any other tensor is copied from the base unchanged.

ISO-Merger requires no post-merge data, rollouts, gradient updates, or
distillation. This file freezes the exact configuration behind the
paper's ISO-Merger rows; run with only --base/--experts/--out to
reproduce the submission models.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Frozen algorithm hyperparameters (paper Appendix E).
# ---------------------------------------------------------------------------
RHO_KEEP: float = 0.9      # fraction of base singular modes kept in tangent
RIDGE: float = 1e-12       # ridge stabilizer on the Gram system
C_CLIP_MIN: float = 0.0    # lower clip on retention coefficients
C_CLIP_MAX: float = 1.5    # upper clip on retention coefficients


# 2D projection matrices per block. Qwen3.6 is hybrid-attention: self_attn
# layers carry q/k/v/o, linear_attn (gated-delta) layers carry in_proj_*/
# out_proj. Both are 2D frames the RL fine-tune rotates, so both merge.
PER_LAYER_2D_PROJS: tuple[str, ...] = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

# embed_tokens / lm_head are large vocab×hidden matrices. RL barely rotates
# their frames (measured c*≈0 under Stiefel — the merge returns base), and the
# reference impl treats the CLIP text_projection analog as plain averaging, so
# these go to task-mean: no SVD, no 250s float CPU cost per tensor.
GLOBAL_2D_EMBEDS: tuple[str, ...] = (
    "model.language_model.embed_tokens.weight",
    "lm_head.weight",
)


def is_per_layer_2d(name: str) -> bool:
    # Qwen3.6 nests decoder layers under model.language_model.layers.N;
    # the vision tower (model.visual.*) is left to copy-base.
    if not name.startswith("model.language_model.layers."):
        return False
    return any(name.endswith(p) for p in PER_LAYER_2D_PROJS)


def is_taskmean(name: str, W: torch.Tensor, exp_idxs: list[dict]) -> bool:
    # embed/lm_head (see above) and every 1D param (LayerNorm scales,
    # biases) present in all experts get the uniform task-vector average.
    if name in GLOBAL_2D_EMBEDS:
        return all(name in idx for idx in exp_idxs)
    return W.dim() == 1 and all(name in idx for idx in exp_idxs)


def thin_svd(W: torch.Tensor):
    """Thin SVD; transpose first when it's cheaper."""
    m, n = W.shape
    if m >= n:
        return torch.linalg.svd(W, full_matrices=False)
    Ut, S, Vht = torch.linalg.svd(
        W.transpose(0, 1).contiguous(), full_matrices=False
    )
    return Vht.transpose(0, 1).contiguous(), S, Ut.transpose(0, 1).contiguous()


def polar_columns(X: torch.Tensor) -> torch.Tensor:
    """Closest matrix with orthonormal columns (polar retraction)."""
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return U @ Vh


def project_tangent(X0: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection Pi_{X0}(Y) = Y - X0 sym(X0^T Y) onto the
    Stiefel tangent space at X0."""
    sym = (X0.transpose(0, 1) @ Y + Y.transpose(0, 1) @ X0) / 2
    return Y - X0 @ sym


def merge_one_tensor(
    W0: torch.Tensor,
    Ws: list[torch.Tensor],
) -> tuple[torch.Tensor, np.ndarray]:
    """ISO-Merger reduction of one (m, n) weight matrix.

    See module docstring for the algorithm. Inputs may be in any float
    dtype; computation is performed in float32 (the reference impl's SVD
    precision — bf16/fp16 SVD is unstable, float64 is needlessly slow on
    consumer/datacenter GPUs) and cast back to the expert dtype on return.
    Returns (W_merged, c_star).
    """
    out_dtype = Ws[0].dtype
    W0 = W0.to(torch.float32)
    Ws64 = [w.to(torch.float32) for w in Ws]
    n_exp = len(Ws64)

    # Step 1: thin SVD of the base.
    U0, s0, Vh0 = thin_svd(W0)
    V0 = Vh0.transpose(0, 1).contiguous()
    k_full = s0.shape[0]
    k_keep = max(1, int(round(RHO_KEEP * k_full)))

    # Steps 2-4: per-expert sign-canonical SVD, tangent projection, top-k
    # masking (mask only — feasibility restored after aggregation).
    xi_us, xi_vs = [], []
    for We in Ws64:
        Ue, _, Vhe = thin_svd(We)
        diag_dot = (U0 * Ue).sum(dim=0)
        signs = torch.where(
            diag_dot >= 0,
            torch.ones_like(diag_dot),
            -torch.ones_like(diag_dot),
        )
        Ue = Ue * signs.unsqueeze(0)
        Vhe = Vhe * signs.unsqueeze(1)
        Ve = Vhe.transpose(0, 1).contiguous()

        xi_u = project_tangent(U0, Ue - U0)
        xi_v = project_tangent(V0, Ve - V0)

        if k_keep < k_full:
            xi_u = xi_u.clone()
            xi_v = xi_v.clone()
            xi_u[:, k_keep:] = 0.0
            xi_v[:, k_keep:] = 0.0

        xi_us.append(xi_u)
        xi_vs.append(xi_v)

    # Step 5: first-order weight-effect proxies
    #   g_i = xi_u_i Sigma_0 V_0^T + U_0 Sigma_0 xi_v_i^T.
    g_list = []
    for xi_u, xi_v in zip(xi_us, xi_vs):
        g_left = (xi_u * s0.unsqueeze(0)) @ V0.transpose(0, 1)
        g_right = (U0 * s0.unsqueeze(0)) @ xi_v.transpose(0, 1)
        g_list.append(g_left + g_right)

    # Step 6: retention coefficients (Gamma + ridge I) c = diag(Gamma).
    G = torch.zeros(n_exp, n_exp, dtype=torch.float64)
    for i in range(n_exp):
        for j in range(i, n_exp):
            v = (g_list[i] * g_list[j]).sum().item()
            G[i, j] = v
            G[j, i] = v
    G_np = G.cpu().numpy()
    diag_G = np.diag(G_np)
    try:
        c_star = np.linalg.solve(G_np + RIDGE * np.eye(n_exp), diag_G)
    except np.linalg.LinAlgError:
        # Exactly singular Gram (experts collinear in tangent space):
        # keep every expert at unit strength.
        c_star = np.ones(n_exp)
    c_star = np.clip(c_star, C_CLIP_MIN, C_CLIP_MAX)

    # Step 7: aggregate, re-project onto the tangent space (masking broke
    # the skew constraint X0^T xi + xi^T X0 = 0), polar retract,
    # reconstruct W_* = U_* Sigma_0 V_*^T.
    c_t = torch.tensor(c_star, dtype=torch.float32, device=U0.device)
    xi_u_star = sum(c_t[i] * xi_us[i] for i in range(n_exp))
    xi_v_star = sum(c_t[i] * xi_vs[i] for i in range(n_exp))
    xi_u_star = project_tangent(U0, xi_u_star)
    xi_v_star = project_tangent(V0, xi_v_star)

    U_star = polar_columns(U0 + xi_u_star)
    V_star = polar_columns(V0 + xi_v_star)
    W_merged = (U_star * s0.unsqueeze(0)) @ V_star.transpose(0, 1)

    return W_merged.to(out_dtype), c_star


def merge_1d_taskmean(
    W0: torch.Tensor,
    Ws: list[torch.Tensor],
) -> torch.Tensor:
    """Uniform task-vector average  w_* = w_0 + (1/K) sum_i (w_i - w_0),
    computed as a Welford rolling mean in float32."""
    W0_f = W0.float()
    delta = (Ws[0].float() - W0_f).clone()
    for i, We in enumerate(Ws[1:], 1):
        d_i = We.float() - W0_f
        delta.add_((d_i - delta) / (i + 1))
    return W0_f + delta


def main():
    ap = argparse.ArgumentParser(
        description="ISO-Merger: data-free fixed-spectrum merge of "
                    "shared-base RL experts.",
    )
    ap.add_argument("--base", required=True,
                    help="path to the shared base model directory "
                         "(HuggingFace format with safetensors shards)")
    ap.add_argument("--experts", required=True, nargs="+",
                    help="paths to expert model directories")
    ap.add_argument("--out", required=True,
                    help="output directory for the merged model")
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu")
    ap.add_argument("--out-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--shard-mod", type=int, default=1,
                    help="total number of parallel workers (data-parallel "
                         "over safetensors shards; run one process per GPU)")
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="this worker's index in [0, shard-mod)")
    args = ap.parse_args()

    base_dir = Path(args.base)
    exp_dirs = [Path(e) for e in args.experts]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    out_dtype = out_dtype_map[args.out_dtype]

    # Copy non-weight files (tokenizer, config, generation_config, ...) from
    # the base so the merged directory is a self-contained HF checkpoint.
    # Only worker 0 writes them (all workers share --out).
    if args.shard_idx == 0:
        for f in os.listdir(base_dir):
            src = base_dir / f
            if src.is_file() and not f.endswith(".safetensors"):
                shutil.copy(src, out_dir / f)

    base_idx = json.load(
        open(base_dir / "model.safetensors.index.json")
    )["weight_map"]
    exp_idxs = [
        json.load(open(d / "model.safetensors.index.json"))["weight_map"]
        for d in exp_dirs
    ]

    shards: dict[str, list[str]] = {}
    for name, fname in base_idx.items():
        shards.setdefault(fname, []).append(name)

    # Data-parallel over shards: this worker owns shard i where
    # i % shard_mod == shard_idx. Shards are independent (each output file
    # is self-contained), so no cross-worker merge step is needed.
    all_shards = sorted(shards.keys())
    my_shards = [s for i, s in enumerate(all_shards)
                 if i % args.shard_mod == args.shard_idx]

    n_stiefel = sum(
        1 for f in my_shards for n in shards[f] if is_per_layer_2d(n)
    )
    print(
        f"[iso-merger w{args.shard_idx}/{args.shard_mod}] {len(exp_dirs)} "
        f"experts, {len(my_shards)}/{len(all_shards)} shards, "
        f"stiefel={n_stiefel} tensors, rho_keep={RHO_KEEP}, "
        f"c-clip=[{C_CLIP_MIN}, {C_CLIP_MAX}]",
        flush=True,
    )

    c_log = []  # (name, c_star) per Stiefel-merged tensor
    t_global = time.time()
    seen = 0
    for fname in my_shards:
        out_tensors: dict[str, torch.Tensor] = {}
        names = sorted(shards[fname])

        with safe_open(str(base_dir / fname), framework="pt") as base_f:
            for name in names:
                seen += 1
                t0 = time.time()
                W0_raw = base_f.get_tensor(name)

                if is_per_layer_2d(name):
                    # Fixed-spectrum Stiefel merge. Per-layer projections are
                    # all small (<= ~5k^2), so they fit on GPU in fp32.
                    W0 = W0_raw.to(args.device)
                    Ws = []
                    for d, idx in zip(exp_dirs, exp_idxs):
                        with safe_open(str(d / idx[name]), framework="pt") as f:
                            Ws.append(f.get_tensor(name).to(args.device))
                    W_merged, c_star = merge_one_tensor(W0, Ws)
                    c_log.append((name, c_star.tolist()))
                    out_tensors[name] = (
                        W_merged.to(out_dtype).cpu().contiguous()
                    )
                    del W0, Ws, W_merged
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                    if seen % 20 == 0:
                        c_str = "/".join(f"{c:.3f}" for c in c_star)
                        print(
                            f"[iso-merger {seen}/{len(base_idx)}] {name}  "
                            f"c*={c_str}  {(time.time() - t0):.1f}s",
                            flush=True,
                        )
                elif is_taskmean(name, W0_raw, exp_idxs):
                    # embed/lm_head and 1D params (LayerNorm, biases):
                    # uniform task-vector average.
                    Ws = []
                    for d, idx in zip(exp_dirs, exp_idxs):
                        with safe_open(str(d / idx[name]), framework="pt") as f:
                            Ws.append(f.get_tensor(name))
                    merged = merge_1d_taskmean(W0_raw, Ws)
                    out_tensors[name] = merged.to(out_dtype).contiguous()
                else:
                    # Any other tensor: copy from base unchanged.
                    out_tensors[name] = W0_raw.to(out_dtype).contiguous()

        save_file(out_tensors, str(out_dir / fname))
        print(
            f"[iso-merger] wrote shard {fname} ({len(out_tensors)} tensors)",
            flush=True,
        )
        del out_tensors

    # Save retention coefficients for reproducibility diagnostics
    # (per-worker file; concatenate across workers for the full picture).
    ret_name = (
        "retention_coefficients.json" if args.shard_mod == 1
        else f"retention_coefficients.w{args.shard_idx}.json"
    )
    with open(out_dir / ret_name, "w") as f:
        json.dump({"tensors": c_log}, f, indent=2)

    print(
        f"[iso-merger w{args.shard_idx}] DONE "
        f"total={(time.time() - t_global) / 60:.1f}min -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

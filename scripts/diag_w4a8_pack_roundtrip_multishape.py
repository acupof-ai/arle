#!/usr/bin/env python3
"""Multi-shape pack/unpack round-trip sweep for W4A8.

Companion to `scripts/diag_w4a8_pack_roundtrip.py` (codex `ab43959`).
Sweeps (N, K, groupsize) over Marlin-compatible shapes to characterize
whether pack_w4a8 forward/inverse asymmetry is shape-dependent.

Use this AFTER scale-chain instrumentation finds a candidate fix — re-run
to confirm the fix holds across all shapes (regression gate). Pre-fix it
documents the shape distribution of the bug.

Usage:
  python scripts/diag_w4a8_pack_roundtrip_multishape.py [--seed 0]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def load_diag_module():
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "diag_w4a8_pack_roundtrip.py"
    spec = importlib.util.spec_from_file_location("diag_rt", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Marlin-compat shapes: N % 64 == 0, K % groupsize == 0.
SHAPES = [
    # (out=N, in=K, groupsize)
    (128, 128, 128),
    (256, 128, 128),
    (256, 256, 128),
    (512, 128, 128),
    (512, 512, 128),
    (1024, 256, 128),
    (1024, 1024, 128),
    (2048, 512, 128),
]


def run_one(diag, n: int, k: int, groupsize: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    qpack = diag.load_pack_module()

    w_bf16 = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
    qweight, s_channel, s_group = qpack.pack_w4a8(w_bf16, groupsize=groupsize)
    perm, scale_perm, scale_perm_single = qpack.get_perms(groupsize, k)

    w_recovered = diag.manual_unpack_w4a8(
        qweight, s_channel, s_group, perm, scale_perm, scale_perm_single,
        n, k, groupsize,
    )

    w_orig = w_bf16.float()
    diff = (w_recovered - w_orig).abs()
    s_group_real = s_group.float() * s_channel.float()
    expected = s_group_real.median().item() / 2

    max_abs = diff.max().item()
    p99_abs = torch.quantile(diff.flatten(), 0.99).item()

    # Top mismatch ratio (recovered / orig) at the worst position.
    flat_idx = diff.flatten().argmax().item()
    row, col = flat_idx // k, flat_idx % k
    orig_v = w_orig[row, col].item()
    rec_v = w_recovered[row, col].item()
    ratio = (rec_v / orig_v) if abs(orig_v) > 1e-6 else float("nan")

    threshold = expected * 5
    return {
        "shape": (n, k, groupsize),
        "max_abs": max_abs,
        "p99_abs": p99_abs,
        "expected": expected,
        "threshold": threshold,
        "pass": max_abs < threshold,
        "worst_row": row,
        "worst_col": col,
        "worst_ratio": ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    diag = load_diag_module()

    results = []
    for n, k, gs in SHAPES:
        try:
            r = run_one(diag, n, k, gs, args.seed)
            results.append(r)
        except Exception as e:
            print(f"  ({n},{k},gs={gs}) ERROR: {e}")
            results.append({"shape": (n, k, gs), "error": str(e)})

    print("\n" + "=" * 96)
    print(f"{'shape (N,K,gs)':<22}{'max_abs':>14}{'expected':>14}{'×over':>8}"
          f"{'worst_row':>11}{'worst_ratio':>14}{'verdict':>10}")
    print("-" * 96)
    for r in results:
        if "error" in r:
            print(f"{str(r['shape']):<22}{r['error']}")
            continue
        n, k, gs = r["shape"]
        x_over = r["max_abs"] / r["expected"] if r["expected"] > 0 else float("inf")
        verdict = "PASS" if r["pass"] else "FAIL"
        print(f"{str(r['shape']):<22}{r['max_abs']:>14.4e}{r['expected']:>14.4e}"
              f"{x_over:>8.1f}{r['worst_row']:>11}{r['worst_ratio']:>14.3f}{verdict:>10}")

    print("=" * 96)
    n_fail = sum(1 for r in results if "error" not in r and not r["pass"])
    print(f"\n{n_fail}/{len(results)} shapes FAIL pack/unpack round-trip\n")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

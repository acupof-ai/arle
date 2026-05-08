#!/usr/bin/env python3
"""
W4A8 pack/unpack round-trip diagnostic.

Per docs/research/2026-05-08-w4a8-kernel-and-wiring-audit-clean.md (`01ace86`)
Option 1 audit recommendation: isolate pack_w4a8 correctness from kernel
and Rust FFI by doing manual unpack + dequant in pure Python and
comparing against the BF16 reference.

Usage:
  python scripts/diag_w4a8_pack_roundtrip.py [--seed 0] [--shape 256 128]

Test methodology:
  1. Generate random BF16 weight tensor W (out, in)
  2. Pack via scripts/quantize_qwen3_w4a8.py::pack_w4a8 → (qweight, s_channel, s_group)
  3. Manually unpack qweight 4-bit values from int32 packing (inverse of bit-packing)
  4. Manually un-permute via inverse of perm/scale_perm/scale_perm_single
  5. Dequantize: w_recovered = (w_int4 - 8) * s_group * s_channel
  6. Compare w_recovered against W:
     - element-wise max abs diff
     - element-wise relative error
     - histogram of differences

Exit codes:
  0  — pack/unpack round-trip within expected quant noise (max abs < ~scale/2)
  1  — pack/unpack round-trip OFF (proves bug in pack_w4a8)
  2  — script error

If exit=0 with passing round-trip → bug is NOT in pack_w4a8; investigate
kernel/loader more deeply. If exit=1 → either (a) pack is broken, or
(b) the manual unpack inverse logic in this script is wrong. Both should
be verified by running with a tiny shape (k=128, n=128, groupsize=128)
and tracing intermediate values before drawing conclusions about pack.

KNOWN STATE (2026-05-08 EOD+28): initial run on (256, 128) shows ~+35%
systematic bias in recovered values vs original. This is consistent with
either:
  (i)  pack scaling bug (e.g., max_per_group vs max_per_channel mismatch)
  (ii) manual unpack scale_perm/perm inverse logic bug in this script
  (iii) both interacting

Codex action: run with tiny shape, instrument intermediate sg_unpermuted /
sc_unpermuted / w_q values to isolate which is broken.

Note: this diagnostic does NOT exercise the kernel; it only verifies
that the FORWARD math of pack_w4a8 round-trips correctly using ITS OWN
interpretation of the storage format. A passing round-trip alone does
not guarantee kernel agrees on storage format — that requires the
end-to-end greedy_consistency test.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def load_pack_module():
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "quantize_qwen3_w4a8.py"
    spec = importlib.util.spec_from_file_location("qpack", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_w4a8_get_perms(groupsize: int, k: int):
    """PR #31 W4A8Layer._get_perms, copied for verbose comparison."""
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                4 * (i % 4),
                4 * (i % 4) + 1,
                4 * (i % 4) + 2,
                4 * (i % 4) + 3,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    if groupsize == k:
        interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
    else:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return torch.from_numpy(perm), scale_perm, scale_perm_single


def pack_w4a8_with_intermediates(
    weight: torch.Tensor,
    groupsize: int,
    perm: torch.Tensor,
    scale_perm: list,
    scale_perm_single: list,
):
    """Pack using the W4A8Layer flow and retain scale-chain intermediates."""
    weight = weight.to(dtype=torch.float16, device="cpu").contiguous()
    n, k = weight.shape
    tile = 16

    ref = weight.t().contiguous()
    s_channel_raw = ref.t().abs().amax(dim=-1, keepdim=True).div(127.0).to(torch.float32)
    s_channel_raw = torch.where(s_channel_raw == 0, torch.ones_like(s_channel_raw), s_channel_raw)
    s_channel_raw = s_channel_raw.reshape(1, n)

    reshaped = ref.reshape(k // groupsize, groupsize, n)
    s_raw = reshaped.abs().amax(dim=1).clamp_min(1e-6).div(7.0).to(torch.float16)

    w = ref.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))
    s_work = s_raw.reshape((1, -1))
    w = torch.round(w / s_work).to(torch.int32)
    w += 8
    w = torch.clamp(w, 0, 15)

    s_group_before_perm = (s_work.reshape(-1, n) / s_channel_raw).to(torch.float16)
    w_int_layout = w.reshape((groupsize, -1, n)).permute(1, 0, 2).reshape((k, n)).contiguous()

    s_group = s_group_before_perm.reshape((-1, len(scale_perm)))[:, scale_perm]
    s_group = s_group.reshape((-1, n)).contiguous()
    s_channel = s_channel_raw.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s_channel = s_channel.reshape((-1, n)).contiguous()

    w_tiled = w_int_layout.reshape((k // tile, tile, n // tile, tile))
    w_tiled = w_tiled.permute((0, 2, 1, 3)).reshape((k // tile, n * tile))
    res = w_tiled.reshape((-1, perm.numel()))[:, perm].reshape(w_tiled.shape)
    res_np = res.cpu().numpy().astype(np.uint32)
    q = np.zeros((res_np.shape[0], res_np.shape[1] // 8), dtype=np.uint32)
    for i in range(8):
        q |= res_np[:, i::8] << (4 * i)
    qweight = torch.from_numpy(q.astype(np.int32))

    debug = {
        "s_channel_raw": s_channel_raw,
        "s_channel_stored": s_channel,
        "s_raw": s_raw,
        "s_work": s_work,
        "s_group_before_perm": s_group_before_perm,
        "s_group_stored": s_group,
        "w_int4_layout": w_int_layout,
    }
    return qweight, s_channel.contiguous(), s_group.contiguous(), debug


def manual_unpack_w4a8(
    qweight: torch.Tensor,
    s_channel: torch.Tensor,
    s_group: torch.Tensor,
    perm: torch.Tensor,
    scale_perm: list,
    scale_perm_single: list,
    n: int,
    k: int,
    groupsize: int,
):
    """Inverse of pack_w4a8 storage steps to recover (k, n) BF16-equivalent weight.

    Mirrors the pack steps in reverse:
      1. unpack 8-element-per-uint32 bit pack → res shape (k//tile, n*tile)
      2. inverse perm permutation
      3. inverse tile permute (k//tile, tile, n//tile, tile) ← (k//tile, tile, n//tile, tile)
      4. permute back (k//tile, n//tile, tile, tile) → (k, n)
      5. apply (q-8) integer offset, multiply by s_group * s_channel
         (with inverse scale_perm + scale_perm_single)
    """
    tile = 16

    # Step 1: bit-unpack (i::8 stride)
    qw_np = qweight.cpu().numpy().astype(np.uint32)
    res = np.zeros((qw_np.shape[0], qw_np.shape[1] * 8), dtype=np.uint32)
    for i in range(8):
        res[:, i::8] = (qw_np >> (4 * i)) & 0xF

    # Step 2: inverse perm permutation
    # forward: res = w.reshape((-1, perm.numel()))[:, perm].reshape(w.shape)
    # inverse: w = res.reshape((-1, perm.numel()))[:, inverse_perm].reshape(w.shape)
    perm_np = perm.cpu().numpy()
    inv_perm = np.argsort(perm_np)
    res_flat = res.reshape((-1, perm.numel()))
    w_unpermuted = res_flat[:, inv_perm].reshape(res.shape)

    # Step 3: inverse tile permute
    # forward steps:
    #   w (k, n) → (k//tile, tile, n//tile, tile)
    #   w.permute((0, 2, 1, 3))
    #   w.reshape((k // tile, n * tile))
    # inverse:
    #   w_unpermuted shape (k//tile, n*tile) → (k//tile, n//tile, tile, tile)
    #   permute (0, 2, 1, 3) → (k//tile, tile, n//tile, tile)
    #   reshape (k, n)
    w_unpermuted_t = torch.from_numpy(w_unpermuted.astype(np.int32))
    w_int = w_unpermuted_t.reshape((k // tile, n // tile, tile, tile))
    w_int = w_int.permute((0, 2, 1, 3)).reshape((k, n)).contiguous()

    # Step 4: invert (groupsize, -1, n).permute(1, 0, 2).reshape(k, n) operation
    # Forward: w (gs, k*n/gs after step 3 NO wait — pack does this BEFORE tile permute)
    # Actually in pack_w4a8:
    #   line 105: w = ref.reshape((-1, gs, n)).permute(1, 0, 2).reshape((gs, -1)) — pre-quant
    #   line 112: w = w.reshape((gs, -1, n)).permute(1, 0, 2).reshape((k, n)).contiguous() — POST-quant integer
    # So at this point in pack, w has shape (k, n) integer with values 0..15.
    # Then tile permute (lines 116-119) reshapes to (k//tile, n*tile) integer.
    # Our step 3 inverse already brought us back to (k, n) integer.

    w_q = w_int.float()  # (k, n) integer 0..15

    # Step 5: dequantize. Inverse the (q - 8) offset, multiply scales.
    # s_group is per (k/gs, n) post scale_perm permutation
    # s_channel is per (1, n) post scale_perm_single permutation
    # Need to inverse-permute to recover original (k/gs, n) and (1, n) layout.

    inv_scale_perm = np.argsort(np.array(scale_perm))
    inv_scale_perm_single = np.argsort(np.array(scale_perm_single))

    sg = s_group.cpu().float()
    sg_unpermuted = sg.reshape((-1, len(scale_perm)))[:, inv_scale_perm].reshape((-1, n))

    sc = s_channel.cpu().float()
    sc_unpermuted = sc.reshape((-1, len(scale_perm_single)))[:, inv_scale_perm_single].reshape((-1, n))

    # Reconstruct: w_recovered[i_kgs*gs+i_gs, i_n] = (w_q[...] - 8) * sg_orig[i_kgs, i_n] * sc_orig[0, i_n]
    # With sg shape (k/gs, n), sc shape (1, n)
    # Broadcast: expand sg per group
    sg_expanded = sg_unpermuted.repeat_interleave(groupsize, dim=0)  # (k, n)
    sc_expanded = sc_unpermuted  # (1, n) broadcasts with (k, n)

    # In pack, s_group_stored = s_group_real / s_channel.
    # During dequant: w_real = (q - 8) * s_group_stored * s_channel
    #               = (q - 8) * (s_group_real / s_channel) * s_channel
    #               = (q - 8) * s_group_real
    w_recovered = (w_q - 8.0) * sg_expanded * sc_expanded

    # transposed back since pack used ref = weight.t(). Caller expects (n, k) input → recovered should be (n, k).
    return w_recovered.t()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shape", type=int, nargs=2, default=[256, 128],
                        help="(out_features, in_features) — default 256 128")
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--verbose", action="store_true",
                        help="print scale-chain and PR #31 W4A8Layer comparison details")
    parser.add_argument("--verbose-row", type=int, default=112,
                        help="output-channel row to inspect in verbose mode")
    parser.add_argument("--verbose-cols", type=int, default=16,
                        help="number of input columns / scale columns to print in verbose mode")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n, k = args.shape  # n = out_features, k = in_features
    print(f"Shape: out={n} in={k} groupsize={args.groupsize}")

    qpack = load_pack_module()

    # Original BF16 weight
    w_bf16 = torch.randn(n, k, dtype=torch.bfloat16) * 0.1

    qweight, s_channel, s_group = qpack.pack_w4a8(w_bf16, groupsize=args.groupsize)
    perm, scale_perm, scale_perm_single = qpack.get_perms(args.groupsize, k)

    print(f"Pack output shapes: qweight={list(qweight.shape)} dtype={qweight.dtype}")
    print(f"                    s_channel={list(s_channel.shape)} dtype={s_channel.dtype}")
    print(f"                    s_group={list(s_group.shape)} dtype={s_group.dtype}")

    cur_dbg = None
    if args.verbose:
        dbg_q, dbg_sc, dbg_sg, cur_dbg = pack_w4a8_with_intermediates(
            w_bf16,
            args.groupsize,
            perm,
            scale_perm,
            scale_perm_single,
        )
        ref_perm, ref_scale_perm, ref_scale_perm_single = reference_w4a8_get_perms(args.groupsize, k)
        ref_q, ref_sc, ref_sg, ref_dbg = pack_w4a8_with_intermediates(
            w_bf16,
            args.groupsize,
            ref_perm,
            ref_scale_perm,
            ref_scale_perm_single,
        )
        print("\nVerbose scale-chain diagnostic:")
        print(f"  helper matches pack_w4a8 qweight: {torch.equal(dbg_q, qweight)}")
        print(f"  helper s_channel max abs diff: {(dbg_sc - s_channel).abs().max().item():.6e}")
        print(f"  helper s_group max abs diff: {(dbg_sg - s_group).abs().max().item():.6e}")
        print(f"  PR #31 W4A8Layer perm equal current: {torch.equal(ref_perm, perm)}")
        print(f"  PR #31 qweight equal current: {torch.equal(ref_q, qweight)}")
        print(f"  PR #31 s_channel max abs diff: {(ref_sc - s_channel).abs().max().item():.6e}")
        print(f"  PR #31 s_group max abs diff: {(ref_sg - s_group).abs().max().item():.6e}")

        row = min(args.verbose_row, n - 1)
        cols = min(args.verbose_cols, k)
        scale_cols = min(args.verbose_cols, n)
        print(f"  s_work shape={list(cur_dbg['s_work'].shape)} "
              f"sample={cur_dbg['s_work'][0, :scale_cols].float().tolist()}")
        print(f"  s_raw shape={list(cur_dbg['s_raw'].shape)} "
              f"sample={cur_dbg['s_raw'][0, :scale_cols].float().tolist()}")
        print(f"  s_group before scale_perm shape={list(cur_dbg['s_group_before_perm'].shape)} "
              f"sample={cur_dbg['s_group_before_perm'][0, :scale_cols].float().tolist()}")
        print(f"  s_group after scale_perm shape={list(cur_dbg['s_group_stored'].shape)} "
              f"sample={cur_dbg['s_group_stored'][0, :scale_cols].float().tolist()}")
        print(f"  s_channel raw sample={cur_dbg['s_channel_raw'][0, :scale_cols].float().tolist()}")
        print(f"  s_channel stored sample={cur_dbg['s_channel_stored'][0, :scale_cols].float().tolist()}")
        print(f"  w_int4 output-row {row} sample={cur_dbg['w_int4_layout'].t()[row, :cols].tolist()}")

    w_recovered = manual_unpack_w4a8(
        qweight, s_channel, s_group, perm, scale_perm, scale_perm_single,
        n, k, args.groupsize,
    )

    # Compare
    w_orig = w_bf16.float()
    diff = (w_recovered - w_orig).abs()
    rel = diff / (w_orig.abs() + 1e-6)

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    p99_abs = torch.quantile(diff.flatten(), 0.99).item()
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()

    # Expected quant noise: per-element max < (s_group_real / 2). For random
    # weight with scale ~max/7 per group and channel scale ~max_per_channel/127,
    # round-trip noise should be ~scale_group_real/2 ~ |w|/14.
    s_group_real = s_group.float() * s_channel.float()
    sg_med = s_group_real.median().item()
    expected_noise = sg_med / 2

    if args.verbose and cur_dbg is not None:
        row = min(args.verbose_row, n - 1)
        cols = min(args.verbose_cols, k)
        scale_cols = min(args.verbose_cols, n)
        inv_scale_perm = np.argsort(np.array(scale_perm))
        inv_scale_perm_single = np.argsort(np.array(scale_perm_single))
        sg_unpermuted = (
            s_group.cpu().float()
            .reshape((-1, len(scale_perm)))[:, inv_scale_perm]
            .reshape((-1, n))
        )
        sc_unpermuted = (
            s_channel.cpu().float()
            .reshape((-1, len(scale_perm_single)))[:, inv_scale_perm_single]
            .reshape((-1, n))
        )
        sg_real_unpermuted = sg_unpermuted * sc_unpermuted
        scale_ratio = sg_real_unpermuted / cur_dbg["s_raw"].float()
        print(f"  reconstructed s_group_real max abs diff vs s_raw: "
              f"{(sg_real_unpermuted - cur_dbg['s_raw'].float()).abs().max().item():.6e}")
        print(f"  reconstructed/raw scale ratio sample={scale_ratio[0, :scale_cols].tolist()}")
        print(f"  original output-row {row} sample={w_orig[row, :cols].tolist()}")
        print(f"  recovered output-row {row} sample={w_recovered[row, :cols].tolist()}")

    print(f"\nRound-trip diagnostic:")
    print(f"  max abs diff   = {max_abs:.6e}  (expected ~{expected_noise:.4e})")
    print(f"  mean abs diff  = {mean_abs:.6e}")
    print(f"  p99 abs diff   = {p99_abs:.6e}")
    print(f"  max rel diff   = {max_rel:.4f}")
    print(f"  mean rel diff  = {mean_rel:.4f}")

    pass_threshold = expected_noise * 5  # 5× headroom for FP16 conversion noise
    if max_abs < pass_threshold:
        print(f"\n✅ PASS: pack/unpack round-trip within quant noise band ({pass_threshold:.4e})")
        print("   → pack_w4a8 storage math is internally consistent.")
        print("   → Bug must be in kernel storage interpretation OR kernel-loader handshake.")
        sys.exit(0)
    else:
        print(f"\n❌ FAIL: pack/unpack round-trip OUT OF noise band ({pass_threshold:.4e})")
        print("   → pack_w4a8 has a forward/inverse asymmetry; pack is broken.")
        # Show first few biggest mismatches for debugging
        flat_diff = diff.flatten()
        topk = torch.topk(flat_diff, k=10).indices
        print("\n   Top-10 mismatch positions:")
        for idx in topk:
            i = idx.item()
            row, col = i // k, i % k
            print(f"     [{row},{col}]: orig={w_orig[row, col].item():+.4f} "
                  f"recovered={w_recovered[row, col].item():+.4f} "
                  f"diff={diff[row, col].item():+.4f}")
        sys.exit(1)


if __name__ == "__main__":
    main()

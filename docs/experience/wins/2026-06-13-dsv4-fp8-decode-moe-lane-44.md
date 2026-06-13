# DSv4 FP8 decode-band MoE lane — 36.9 → 44.0 tok/s B=1, era-class WITH correctness

## Context

End of the DSv4 B=1 regression campaign
([record](../../projects/2026-06-13-dsv4-b1-regression-campaign.md)). The
64-align fix had recovered 33.5 → 36.9; the residual to era (43-44) was the
MoE decode padding tax — the contiguous DeepGEMM lane pads each expert's ~1
real decode row to a 64-row tile, then grinds the pad rows through
`pack_quantize`/`swiglu`/`scatter` every layer. The reframe: **era's 44.04
(`d7be8c9b`) ran a subtly-wrong compact MoE** (per-tile contract violation
when a rank held ≥2 of the token's 6 routes), so "recovering 44" meant
matching era's speed *with* correctness, which the contiguous kernel can't
(compact ⊕ correct are mutually exclusive there).

## What Worked

Two from-scratch FP8 decode kernels (`csrc/gemm/dsv4_fp8_decode_moe.cu`),
the FP8 analog of the in-tree BF16 `*_decode` kernels:

- `dsv4_fp8_grouped_swiglu_decode` — one warp per intermediate row reads the
  gate and up FP8 matrices once (16-byte `uint4` loads), accumulates the ≤8
  routed rows in fp32, and writes `act = clamped_swiglu(gate, up)` directly
  (the separate swiglu launch is gone).
- `dsv4_fp8_grouped_down_decode` — w2 GEMM, 4 rows/warp.

w8a16: BF16 activations, FP8 e4m3 weights, **f32** 128×128 block scales (the
MoE expert encoding — NOT UE8M0; that was last round's kill). **Per-route
correct without padding**: a warp owns one output row, so there is no
per-tile single-group contract to violate — compact packing is legal here.
No pad rows, no activation quantize, no separate swiglu pass. The DSv4 clamp
(`min(gate,limit)`, `clamp(up,±limit)`) is applied exactly in the fused
epilogue, matching `dsv4_swiglu_clamped_one`/`dg_swiglu`.

This is the bandwidth-root fix for last round's scalar-GEMV kill: `uint4`
weight loads ≈ 80% HBM vs the GEMV's 1-byte loads at 25%.

Default ON in the decode band (`total_routes ≤ 8`);
`ARLE_DSV4_MOE_DECODE_FP8=0` reverts to the contiguous lane.

## Results (same binary, same session)

| arm | B=1 tok/s | TPOT |
|---|---|---|
| pre-campaign (regressed) | 33.5 | 29.9ms |
| 64-align fix (`a1e15307`) | 36.9 | 27.1ms |
| **FP8 decode lane** | **44.11 / 43.76 / 44.18** (3 boots) | **~22.4ms** |
| era control `d7be8c9b` (incorrect MoE) | 42.9 / 43.5 / 44.04(matched A/B) | 23.3ms |

**+19.5% over the 64-align baseline; matches/edges era — with correct MoE.**

- Needle ×3 same-config: 512 exact-DET, 2048 partial-stable, 6000 exact-DET
  — identical to the locked correct envelope, 0 garbage, 0 table-build warns.
- step-profile: tail backlog 6.5 → ~4.5ms, layers-launch host 19.9 → 16.8ms.
  The padding tax is gone; the residual backlog now overlaps cleanly.

## Rule

- **A dequant decode kernel is licensed by achieved bandwidth, not row
  count** — and the cure for a bandwidth-bound scalar kernel is *vectorize
  the weight load* (uint4), not abandon the compact structure. Last round's
  GEMV had the right shape (compact, correct) and the wrong load width; the
  shape was reusable, only the load needed widening.
- **"Restore the old number" can be a correctness trap.** era's 44 was
  fast partly because it was occasionally wrong; the honest target was
  era-speed-with-correctness, and naming that explicitly is what kept the
  fix from chasing an unreachable (incorrect) ceiling.
- **The in-tree BF16 `*_decode` kernels were the right template** — porting
  their structure (warp-per-row, vector loads, exactly-once weight reads) to
  FP8 + f32 block scales was a contained change, not a new kernel design.

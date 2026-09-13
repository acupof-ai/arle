# DSv4 oproj parity gate failed at relL2 exactly 1.000: the f64 oracle used the raw fp8 byte as a number and omitted the activation scale — example, 2026-09-13

## Context

`dsv4_tp_oproj_parity` (DeepGEMM SM90 FP8 path) reported the wo_a latent at
relL2 `1.000e0` on every token/rank while the scalar GEMV lane passed exactly
and the head output-slice copy was bit-exact. The DeepGEMM kernel computed
nonzero, populated, sane-shaped output (all 1024 elements written, max abs
16.5), so this was not an unwritten buffer or a zeroed scale region.

A sequence of structural hypotheses was each measured and rejected before the
cause was found: m=1 layout selection, the conversion-cache slice, an unwritten
output buffer, a permuted sfb (`use_uniform_sfb=2`), and sfb element layout.
The tuner reports `block_n=16|128` so `uniform_sfb=1`, and the generated kernel
instantiates `kMajorSFB=Major::K`, i.e. row-major `[n/128][k/128]`, matching
both cache builders element for element.

## Root Cause

The f64 reference modeled neither the fp8 decode nor the activation scale.
`quantize_blocks` returned

```rust
out[i] = f64::from(e4m3_encode((v / scale) as f32));
```

which is the raw e4m3 byte reinterpreted as an integer (magnitude up to 126;
negative values map to 128–255), and `wo_a_group` multiplied weight*scale by
that integer while never applying `sfa`. DeepGEMM reconstructs the activation as
decoded fp8 (magnitude up to 448) times the per-128-k-block `sfa`, then times
weight times `sfb`. Two errors followed: every k-block was reweighted by a
different nonlinear factor, and the magnitude was off by about 614x.

The decisive measurement was the norm ratio and cosine, not another structural
read:

```
device |g| = 4.34e2, reference |w| = 2.67e5, g/f = 1.625e-3, cos = -0.281
```

`rel = 1.000` here means a tiny output measured against a 614x-too-large
reference. relL2 = 1 is the signature of a magnitude collapse, not of
orthogonality — orthogonal equal-energy vectors give sqrt(2). Magnitude and
direction were separated in one run by printing `|g|/|w|` and `g·w/(|g||w|)`;
the value of `g/f` (~`sfa`) pointed straight at the missing activation scale,
and only then at the reference that omitted it.

Three oracle accumulation models — f64 sequential, f32 sequential, and
per-128-block "accumulate raw fp8 products, apply sfa*sfb once per block" —
were byte-identical against the device. That result is what makes the residuals
below findings rather than suspicions of the reference, and it is the part
worth not redoing.

## Fix

All changes are in the gate's f64 oracle and harness, in
`crates/infer-cuda/examples/dsv4_tp_oproj_parity.rs`. No kernel or bound
constant changed.

- Reconstruct the quantized activation as DeepGEMM does:
  `decode(encode(v/scale)) * scale`.
- Rebuild each stage's reference from the latent the device itself produced
  (the construction used in #434), so a comparison measures one operator rather
  than upstream error amplified through a repack. The wo_b reference is applied
  to the device wo_a latent: repack-to-fp8 for the DeepGEMM lane, bf16 direct
  for the scalar GEMV lane; dot products accumulate in f32 and rank partials
  round to bf16 before the host cross-rank sum, matching the device all-reduce
  the gate header describes.

On H20 the single-token direct-cache arm goes from `1.000e0` to passing:

- wo_a latent relL2 `6.3e-4`, cosine `0.9999998`.
- decomposed wo_b TP-sum relL2 `6.652e-4`.

The decomposed reference also removes a false set of 36 wo_b outliers: before
decomposition the wo_b comparison was fed the oracle latent while the device
consumed its own, so the 36 were upstream latent error under near-cancelling
rank partials, not a wo_b fault. At tp=2 s=32 the scalar GEMV output passes
with zero per-element violations (relL2 `4.19e-5`).

## Residuals that stay open (no bound moved to hide them)

The full positive sweep (12 shapes: tp in {1,2,4,8} times s in {1,8,32},
12/12 DeepGEMM families run) exits 0, but "the gate passes" does not mean the
path is covered. The actual coverage:

| lane | shapes | wo_a latent | wo_b / TP-sum output |
|---|---|---|---|
| scalar GEMV | tp ≤ 2, all s (6 shapes) | gated | gated |
| scalar GEMV | tp > 2, all s (6 shapes) | gated | NOT GATED (warp-tree order) |
| DeepGEMM direct-cache | tp=1, s=1 (1 shape) | gated (6.3e-4) | gated (6.65e-4) |
| DeepGEMM direct-cache | s>1, any tp (8 shapes) | NOT GATED (token-6) | NOT GATED |
| DeepGEMM direct-cache | s=1, tp>1 (3 shapes) | gated | NOT GATED (wgmma order) |
| DeepGEMM re-quant (production) | all 12 shapes | NOT GATED (no requant model) | NOT GATED |

So the production re-quant path is bounded on no shape; the direct-cache path
is fully bounded on one shape and latent-bounded on three more; the scalar path
is latent-bounded everywhere and output-bounded on tp ≤ 2. The honest statement
"latent is gated everywhere it can be" is a real result but is not the same as
the oproj path being covered. Each ungated arm prints its reason. An arm that
cannot be bounded and says nothing is the failure this batch has been about.

- **Re-quantization arm (the production path), every shape.** Production
  re-quantizes e4m3+e8m0 weights through
  `dsv4_block_scaled_to_fp8_deepgemm`; the oracle still uses the original e8m0
  scales and does not model that step, so neither latent nor output has a tight
  bound. It runs (a crash or NaN still fails) but never contributes a pass.
  Modeling it means reproducing the conversion kernel's per-block amax/448
  re-scale and fp4 decode in the oracle; estimated half a day.
- **DeepGEMM direct-cache, m>1 — the token-6 finding.** At m=8, token row 6 of
  the wo_a latent has two wrong elements, one a sign flip on order-1e-1 values:
  `g=+1.167e-1 vs w=-1.543e-1` (diff 0.271) and `g=-2.490e-1 vs w=-1.855e-1`.
  The other seven token rows are exact. There is no cancellation story at that
  magnitude and the three accumulation orders agree, so this is a suspected
  real m>1 device defect, tracked on agenda row `dsv4-oproj-token6-m8`.
  Discriminators written on the row: sweep m in {2,4,16,32} and read which row
  is wrong (fixed slot vs tail mask vs tile boundary), then vary the seed at
  m=8 (a moving row means a race or uninitialized read and changes priority).
- **DeepGEMM direct-cache, tp>1.** The wo_b rank partials come from wgmma tile
  reduction, whose f32 order can land a large (~20–55) partial on an adjacent
  bf16 grid point versus the sequential reference; where two partials nearly
  cancel, that one-ulp gap becomes an absolute error on a small output (example
  parts 23.375/-23.25, output 0.125 vs 0.0). Needs a wgmma-order oracle.
- **Scalar GEMV, tp>2 output.** With four large bf16 partials nearly cancelling,
  the kernel's warp-tree f32 reduction can differ from sequential f32 by one
  partial ulp: tp=4 s=32 has 1/131072 outputs off by one ulp (relL2 3.03e-5).
  tp<=2 cancels at most two partials and is exact. Needs a warp-order oracle;
  that is a large modelling effort for a single element, so it stays honestly
  ungated rather than modelled.

The two reduction-order residuals (wgmma and warp-tree) are the same numerical
class: a relative rounding error on a large intermediate converts to an
absolute error on a small output under cancellation. They are oracle-modeling
gaps with a specific missing piece, not tolerances to widen.

## Rule

An fp8/fp4 parity oracle must model the device's value reconstruction, not its
storage encoding: the quantized operand is decoded_fp8 times the per-block
scale, never the byte as an integer. When a gate sits at relL2 exactly 1.000,
print the norm ratio and the cosine before theorizing about layout — a ratio
near zero with nonzero cosine is a scale/magnitude fault, cosine near zero is a
permutation or direction fault, and the two need different investigations.

When a trusted reference path fails the same way as the path under test (here
the scalar GEMV and DeepGEMM shared sparse cancellation residuals), suspect
what the harness makes them share — the rank-partial sum and the reference
beside it — rather than opening two kernel bugs.

The tp<=2 residual is an instance of the #434 class — a reference computed
from an idealized input rather than from the values the kernel actually
consumed. Here the wo_b reference was fed the oracle wo_a latent while device
wo_b consumed its own in-bound, non-bit-exact latent; near-cancelling rank
partials amplified that sub-ulp input difference into a one-grid-point output
error. Cancellation was the amplifier; the input mismatch was the cause. This
is the third wrong verdict this week from that class, caught each time by a
different person who did not connect it: rebuild every downstream reference
from the device's own output, and when a residual appears under cancellation,
check what input the two sides were fed before blaming the summation precision
or the kernel. Distinguish "reference lacks a reduction order" (the tp>2 / wgmma
residuals above) from "reference used the wrong input" (this one) by making the
accumulation-order change byte-identical before claiming a device defect.

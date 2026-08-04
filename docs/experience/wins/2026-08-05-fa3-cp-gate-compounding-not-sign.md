# FA3 as the unconditional CP ring: the gate that worked was "does it compound", not "which arm is closer"

**Date:** 2026-08-05 · **Commit:** `15caff0d0` (flip), gate run at `4846f8046` · **Pod:** 8×H20, `qwen35-08b-clean`, seq 2048, LoRA attention-qv · **Verdict: flip stands**

## Context

`ARLE_CP_RING_FA3` was deleted, making FA3 the only CP ring path at head_dim 256
on sm_90. FA3 is worth 3.54× per step at seq 81920. The gate had to answer
whether the ARLE glue around the vendored kernel — the zigzag (q_run, k_run) pair
decomposition, the (o, lse) merge into the flash-2 accumulators, and the
torch-free backward instantiation — produces the right gradient.

## What the first four gates said

- `cp_hidden_parity`, head_dim 256, world 2 (FA3 genuinely engages; the config
  was verified from the log): `cp_vs_cpu_f32` 3.16e-2 against `single_vs_cpu_f32`
  3.49e-2 — CP tracks the CPU f32 anchor better than single-card.
- `nd_parallel_parity`, depth 8, hybrid 4: CP's signed deviation from f32 is
  smaller than single-card's and in the same direction, at cp=2 and cp=4 alike.
- Real 27B, cp=2, seq 32768: loss 10.871086 (exact to the reference), grad_norm
  2.263385 vs 2.264733 (0.06%).
- 0.8B dense, cp=1 vs cp=2, 3 reps each: within-arm spreads 2.44e-4 and 7.25e-5
  relative, non-overlapping ranges, gap 1.34e-3. Real, not noise.

The last one looked like a problem. It is on the model whose CP was certified by
finite differences, and the gap was larger than the published 8.56e-4 and had the
opposite sign.

## The two traps, both walked into

**A stale comparator.** The "8.5e-5 CE floor" cited as the bound for
`cp_hidden_parity` was measured at commits that predate both the FA3 route and
the switch of that example to head_dim 256 — i.e. at hd128, where FA3 never
engages. A gate was nearly failed against a baseline that had never run the code
under test.

**An unanchored yardstick.** The 0.8B result compared cp=2 against cp=1 and read
the gap as FA3's error. cp=1 is not ground truth — it runs a different attention
backward (fused SDPA recompute), so it is a third numerical path. Against it,
FA3-off deviates 1.09e-3 and FA3-on 1.42e-3; both ~0.1%, and "30% worse against
an uncertified reference" is not evidence of a defect. Reverting on that would
have repeated the earlier error of attributing #85 to FA3.

## The gate that discriminated

A 2×2×3 on ONE binary at `4846f8046`, which still carries the env var, so the only
variable is the flag. Then a third world size.

**Control:** at cp=1, FA3 is provably inert — identical loss (8.963640) and
grad_norm within the within-cell spread. The ring does not exist at cp=1, and the
measurement confirms it. Without this the rest would not be interpretable.

**Deviation from cp=1, relative:**

| | cp=2 | cp=4 |
|---|---|---|
| FA3 off | +1.085e-3 | **+1.655e-3** |
| FA3 on | −1.419e-3 | **−1.80e-4** |

Within-cell spreads across all six cells: 5.2e-5 to 2.1e-4 relative.

**FA3's deviation does not compound with ring-step count — it collapses to the
noise floor at cp=4. The scalar path's grows.** A wrong block pairing, a
mis-merged LSE or a dropped partial accumulates with the number of ring steps;
nothing here does. The sign flip between the two settings is real but is what a
different reduction order looks like, not what a defect looks like.

Flip stands.

## Rule

**When no arm is ground truth, stop asking which is closer and ask whether the
error compounds.** Comparing two uncertified paths yields a number with no
verdict attached — cp=1 vs cp=2 gave a clean, reproducible, non-overlapping
1.4e-3 that meant nothing on its own. Adding a world size converted the same
measurement into a decision, because "grows with rank count" is a property of the
mechanism under test and needs no reference value.

Two corollaries this run paid for:

- Check that a cited baseline was measured on the code under test. A number from
  a commit that predates the feature is not a bound.
- Build the null cell in. cp=1 with the flag toggled proved the knob was doing
  nothing where it should do nothing; every other number depended on that.

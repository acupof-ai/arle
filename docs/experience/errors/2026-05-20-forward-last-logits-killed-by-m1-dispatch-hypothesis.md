# `forward_last_logits` killed at Qwen3-0.6B — M-aware dispatch is the lurking axis

## Context

Local commit `7aa11d7` (now reverted as `0a1f945`) shipped
`forward_last_logits`: slice the post-`final_norm` hidden to the last
position before applying `lm_head`, saving `(seq_len - 1) × hidden × vocab`
FMAs during OPD rollout. FLOP math projected ~5 % wall-clock saving at
Qwen3-0.6B (vocab = 151 936). Correctness gates were green
(determinism + grad-check bit-identical). Moderate-shape (vocab = 32 768)
A/B showed 0.995 × — within noise — and was attributed to the small
`lm_head` share at moderate shape.

The wins entry shipped as `pending-bench` with an explicit kill
criterion: *"mean ≤ 1.0 × at sigma_pct ≤ 2 % → revert."* Codex ran the
production-vocab A/B with proper memory budget (after revving the
harness with `retain_ids` between rollout iterations to fit memory),
produced **0.997 × at σ ≈ 0.5 %**, and executed the KILL. The revert
preserves the public OPD API surface.

## Root cause — M-aware dispatch in the mixed-dispatch sgemm

The mixed-dispatch routing
(`crates/autograd/src/backend.rs::sgemm_row_major`, `15fa6cf`) only
keys on `N`:

```rust
const SAXPY_N_THRESHOLD: usize = 32_768;
if n < SAXPY_N_THRESHOLD { /* saxpy */ } else { /* matrixmultiply */ }
```

At `vocab = 151 936` both A/B variants take the `matrixmultiply` path —
but with very different `M`:

| Variant | M (rows) | FMAs | matrixmultiply M-regime |
|---|---:|---:|---|
| `forward` (full lm_head) | 3-4 | 1.09 × 10⁹ | M ≥ 3, good packing |
| `forward_last_logits`    | 1     | 3.11 × 10⁸ | M = 1, poor packing |

`matrixmultiply::sgemm` is a packed-tile kernel optimised for
medium-to-large M. At M = 1 the inner loop degenerates: each tile pack
amortises over one output row only, the SIMD reduction is wasted on a
single result vector, and the realised throughput drops well below the
M = 4 case. The result: the candidate variant does ~70 % fewer FMAs but
at ~3-4 × worse per-FMA throughput, and **wall-clock comes out
equivalent.**

This is the matched-control framing of the
[`2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md`](../wins/2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md)
finding that *"matrixmultiply regresses 3 × on thin-tall M = 4 forward
but wins 1.92 × on wide-tall lm_head."* The decision tree there
considered the thin-tall case `M = 4` and chose `matrixmultiply` for the
lm_head shape because saxpy was worse at `M = 4, N = vocab`. **It did
not consider `M = 1` — the rollout-last-row case is a regime that
existing dispatch never saw at production vocab.**

## What the SOLID gate caught — and what it didn't

The license-or-kill cycle worked exactly as designed:

1. Code shipped with an explicit, falsifiable acceptance criterion
   written into the wins entry hand-off ticket.
2. Codex measured at the production shape, in matched-control conditions
   (same model, same prompt, identical rollout tokens — equivalence
   asserted).
3. The kill criterion triggered at σ ≈ 0.5 %, well below the 2 % noise
   floor.
4. Codex executed the revert + documented the evidence in the commit
   message.

What it didn't catch upfront:

- **Hypothesis-chain blind spot.** The FLOP-count projection only counted
  *work saved*. It did not account for *throughput regime change*
  (M = 3-4 → M = 1 in `matrixmultiply`). Both numbers are real; the
  hypothesis only ever modelled one of them.
- **Moderate-shape framing trap.** The moderate-shape 0.995 × result
  was attributed to "vocab too small to show the win." The right
  framing was *"both shapes are M=1-vs-M=3-4 — the dispatch behaviour
  at M=1 is the load-bearing question, not the vocab size."* §0 SOLID
  says framing-cross-check must be wall-clock ground truth, and the
  moderate-shape wall-clock *was* already wall-clock ground truth — it
  said the win wasn't there, and that signal was misread.

## Fix

Reverted in `0a1f945`. No code change to ship.

## Rule

**Project rule (CPU sgemm dispatch).** Any rewrite that changes the
`M` dimension of a matmul without explicit M-aware dispatch is a
hypothesis, not an optimisation — even if it strictly reduces FMAs.
Before licensing such a change, either:

1. Verify that the existing dispatch handles the new M regime
   competitively (M-A/B benchmark at the new shape), or
2. Add M-aware dispatch as part of the same tranche, or
3. Confirm the FMA reduction is large enough that the worst-case
   per-FMA regression is still net positive.

For `lm_head` specifically: M = 1 with N = 151 936 is a NEW regime
post-15fa6cf — there is no benched evidence that either saxpy or
`matrixmultiply` is good there. A dedicated M = 1 path may be needed
(potentially a hand-rolled dot-product-vs-bulk-broadcast loop).

**Process rule (license-or-kill).** Always write the kill criterion
into the wins stub at the same time as the code. The 2026-05-20 kill
cycle worked because the criterion was machine-checkable; if it had been
"verify it's a win" without numbers, codex would have had to invent the
threshold. The pending-bench wins entry → explicit kill ticket → codex
runs measurement → commit + revert: this is the canonical SOLID
license-or-kill loop, document it as a project pattern.

## Lessons preserved

- The `lm_head` transpose-copy hypothesis (Axis A of the killed
  research doc) is *not* refuted by this kill. The hypothesis was about
  eliminating a 623 MB physical transpose per call; the
  forward_last_logits A/B did not control for transpose-copy cost at all
  (both variants do it identically). The next investigation pass for
  the `lm_head` perf surface needs a different variable.
- The retain-ids leak finding in
  [`../research/2026-05-20-opd-production-step-retain-ids-leak.md`](../../research/2026-05-20-opd-production-step-retain-ids-leak.md)
  (and codex's in-flight fix on `crates/train/src/opd.rs` adding
  `cleanup_after_backward`) is independent of this kill and remains
  load-bearing for any multi-step bench.

## Cross-links

- Killed commit (reverted): `7aa11d7` (`perf(train): forward_last_logits rollout path — pending Qwen3-0.6B verification`)
- Kill commit: `0a1f945` (`perf(train): kill unlicensed forward_last_logits rollout path`)
- Codex's measurement: `bench-output/2026-05-20-rollout-last-logits-ab/rollout_last_logits_ab.txt`
- Substrate that this builds on: [`../wins/2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md`](../wins/2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md)
- Dispatch policy under test: `crates/autograd/src/backend.rs::sgemm_row_major` (15fa6cf)

# OPD backward: kill the O(n²) per-chunk prefix re-forward

**Date**: 2026-05-29
**Plan**: [`docs/plans/2026-05-29-opd-student-rollout-via-infer.md`](../../plans/2026-05-29-opd-student-rollout-via-infer.md) (next-axis after the infer-rollout 5× win)

## Context

After routing student rollout through infer (step 250s → 50s, 5×), the
backward became 78% of the step (~39–63s depending on warmth). Attribution
(profiled run, `ARLE_OPD_BACKWARD_PROFILE=1`) showed `tape.backward` dominated
by `MatmulBT` (66%, of which 99.6% is frozen base-weight `dX=dY·Wᵀ`) +
`LinearAttention` (29%); LoRA adapters were **0.4%**. Per-chunk backward time
grew **1.5s → 14.5s** across 8 chunks — the signature of O(n²).

## Root cause

`backward_chunked_kl_rollout` (`crates/train/src/opd.rs`) re-ran
`student.forward(prefix)` **and** `teacher.forward(prefix)` over the **full
growing prefix `rollout[..seq_end]`** on **every chunk**: chunk 8 forwarded
`[0..128]`, chunk 7 `[0..112]`, … `[0..16]`. Total forward work =
16+32+…+128 = 16·Σ₁⁸ = 576 token-forwards vs the minimal 128 → **~4.5×
redundant** dense base-model forward+backward.

The chunking was meant to bound the vocab-sized softmax/log-softmax
intermediates, but it chunked the **forward** (expensive) instead of just the
**loss graph**.

## Fix

Forward teacher + student over the scored prefix **exactly once**, then chunk
**only** the KL/softmax via the already-existing `kl_distill_loss_chunked`
(the same primitive the non-rollout `kl_distill_loss_for_config` path already
uses), then a single backward.

Correctness is exact, not approximate: causal attention makes position `p`'s
logits independent of tokens after `p`, so a single `[0..seq_end]` forward
yields the **same per-position logits** the old per-chunk re-forward produced.
The summed weighted-chunk loss is the identical mean-over-(positions×vocab)
scalar, so the gradient is identical too. This aligns the rollout backward
with the forward-once/chunk-the-loss pattern already used elsewhere.

~10 lines net: replaced the per-chunk forward loop with one forward + one
`kl_distill_loss_chunked` + one backward.

## Verification (per user direction: solve + verify once, no confirmation sweep)

- **Math**: provably identical loss + gradient (above). The change removes
  redundant recompute only.
- **CPU `cargo test -p train --features no-cuda --test test_opd_step`: 14/14**,
  including `opd_step_runs_end_to_end` and
  `opd_step_updates_student_without_mutating_teacher` at `kl_chunk_size=Some(2)`
  on a multi-token rollout (exercises the new single-forward + >1-chunk-loss
  path; asserts finite loss, correct student update, unchanged teacher, clean
  tape).
- **Compiles** clean under `--features cuda` and `cuda,no-cuda`.

## Perf — confirmed end-to-end on the real config (2026-05-29 probe)

A 3-step pre-flight of the **real capability config** (4B teacher + 0.8B
student LoRA + the now-default in-process infer student engine, rollout-64,
`opd-diverse-1k`) on the RTX 4070 Ti SUPER measured **~12.8s/step warm**
(steps 2–3): backward ~8–9s, student_rollout ~2.6s, teacher_forward ~0.03s.
The old rollout-64 path's `student_rollout` term alone was ~60s (perf fit), so
the infer-rollout + this backward fix together deliver **~8× at the step level
on the real 4B-teacher config** — confirmed by a real run, not a synthetic
sweep (closing the earlier "projected" caveat per user direction). Forward work
on the scored prefix dropped ~4.5× → 1× as designed; backward is now the
~8–9s term and the natural next axis only if a bigger GPU lifts the memory cap.

**VRAM finding (hard cap on this 16GB card):** the same probe OOM'd at
**rollout-128** — `slice_bwd` `alloc_zeros` failed on the first backward, peak
~14.5GB. rollout-64 fits but with only ~720MB headroom; the two infer engines
are tiny (`mem_fraction_static=0.05`), so VRAM is dominated by the 4B teacher
weights + the autograd backward tape. **The "longer rollout = more on-policy
signal" capability hypothesis is hardware-blocked at rollout-128 on 16GB**;
testing it needs a 24GB+ GPU or a `slice_bwd`/activation-memory reduction.

## Rule

- **Chunk the loss, never the forward.** When a memory-bounding chunk loop
  re-runs an expensive shared forward per chunk, it converts an O(n) cost into
  O(n²). Chunk only the cheap per-chunk reduction (softmax/logits) over a
  single shared forward.
- **When the root cause is already proven with evidence, implement the fix and
  verify once (math + existing unit test), don't run a confirmation sweep.**

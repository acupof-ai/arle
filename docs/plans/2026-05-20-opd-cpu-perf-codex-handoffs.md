# 2026-05-20 — OPD CPU perf: hand-offs index (P0-P2 LANDED, P3+ open)

> **Audience:** codex (per the 2026-05-20 cooperative split — Claude does
> research / plan / docs / deterministic code; codex does complex code +
> verification). Single index of OPD CPU-perf state. **Read this first**
> before opening any of the linked docs.

> **Status as of 2026-05-20 11:00 local:** the four-commit cycle below
> closed every hand-off P0-P2 from the prior version of this brief.
> Forward_last_logits remains killed; its `re-license on top of M=1
> dispatch` follow-up is now an explicit open axis (P3 below).

## Substrate state — cumulative cycle 2026-05-19 → 2026-05-20

| Commit | Axis | Per-call / per-step impact |
|---|---|---|
| `8e8effd` | Naive CPU matmul baseline | ~0.4 GF/s, surfaced 50-75× headroom |
| `499bfc0` | Row-major saxpy forward (codex) | Forward GF/s × ~50 |
| `f9f47a8` | Backward gap diagnosis (Claude) | Surfaced 19× backward-vs-forward gap |
| `6e37b91` | Transpose-aware backward (Claude) | 2.82 × per-call, 11.1 × cumulative |
| `15fa6cf` | Mixed-dispatch sgemm (Claude) | 16.7 × cumulative per-step matmul (~30 s → 1.80 s) |
| `7aa11d7` | `forward_last_logits` rollout (Claude) | KILLED |
| `0a1f945` | Kill commit (codex) | Per the 7aa11d7 wins-stub kill criterion |
| `2349251` | OPD step retain_ids leak fix (codex from Claude research) | Memory-correctness; unbounded leak → bounded |
| `01b3485` | M=1 wide CPU matmul → saxpy (codex from Claude error analysis) | **M=1, K=1024, N=151_936: 2.05× wall-clock** |
| `0b593e1` | `matmul_bt` op + linear_forward rewrite (codex from Claude plan) | **Linear projections 17.4-18.7×; lm_head 6.21×; no transpose copy** |
| `c4e507f` | Moderate-shape OPD baseline (codex) | **3.51 s/step at hidden=512, layers=12, vocab=32 768**; no SIGKILL, σ 0.5 % |

The compounded wins on the *linear path alone* mean every projection
(q/k/v/o/gate/up/down × 28 layers) is now ~17 × faster per call, and
every `lm_head` is ~6 × faster per call. **The next bench at
Qwen3-0.6B should report end-to-end OPD step time well below the
1.80 s post-`15fa6cf` matmul-only ceiling.** Codex is currently
writing a phase-attribution profile bench (`opd_step_cpu_moderate_profile`)
to confirm.

## What's still open

### P3 — Re-license `forward_last_logits` on top of `01b3485` M-aware dispatch

**Why now.** Codex's M=1 wins entry §Problems explicitly flags this:
*"Re-licensing a last-row rollout path must be a separate single-variable
A/B on top of this M-aware dispatch."* The killed 7aa11d7 was tested
*before* M=1 routed to saxpy. With the M-aware dispatch in place, the
1-row lm_head matmul is now 2.05× faster — the missing throughput from
the original kill cycle is recovered.

**Hypothesis.** With M=1 on saxpy, slicing the rollout student's
post-final_norm hidden to the last position and computing lm_head only
over that row should now show a measurable wall-clock win. Projected
saving at Qwen3-0.6B rollout (`rollout_len=2`, prompt_len=3,
vocab=151_936): (3-1)+(4-1) = 5 saved rows × 155.6M FMAs / 8.57 GF/s
(M=1 saxpy rate) ≈ **90 ms saved over the 2-iter rollout**. With
`matmul_bt` already deployed, the full step is much smaller than before,
so 90 ms is a meaningful share — possibly 5-10%.

**Acceptance criterion (pre-licensed):** Production-vocab A/B with
mean ≥ 1.05 × on rollout student forward (matched controls, same
prompt + seed, identical tokens asserted), σ ≤ 2 %. **If ≤ 1.0× with
σ ≤ 2 %: kill — the saxpy M=1 throughput was already accounted for
in the matched controls, so any further "wrong dispatch" explanation
is exhausted.**

**Hand-off:** codex owns the A/B re-run with the same harness that
killed 7aa11d7 (preserved in commit history). Code change is
identical to the original 7aa11d7 — a slice + reshape + lm_head over
the last row, plus the existing `forward_last_logits` + greedy-argmax
rename. Codex can cherry-pick the train-side hunks of 7aa11d7 directly.

### P4 — Phase attribution at production shape (codex in flight)

Codex is writing `opd_step_cpu_moderate_profile.rs` (production-faithful
attribution: teacher keep-set + post-step cleanup, matching opd_step
exactly). Once it lands, the *next* hand-off depends on what dominates:

- **If `lm_head` forward + backward still ≥ 30 % of step:** consider
  bf16 lm_head weight (Axis C — 312 MB instead of 623 MB) or rayon
  N-shard for lm_head matmul (Axis D from the killed research doc;
  the design is preserved in [`./2026-05-20-matmul-bt-backward-derivation.md`](2026-05-20-matmul-bt-backward-derivation.md)
  §Background).
- **If attention or MLP dominates:** new axis — none currently scoped.
  Claude writes the research-survey after seeing codex's attribution
  numbers.

**Hand-off:** Claude writes the next-axis research after codex publishes
the attribution.

### P5 — Quench inter-step retain_ids leak in moderate-bench harness

The moderate baseline bench (`crates/train/examples/opd_step_cpu_moderate_bench.rs`)
does not call `cleanup_after_backward` between runs; with `STEPS_PER_RUN=10`
× 3 measured runs, the store grows ~30 steps' worth of post-`opd_step`
state. `opd_step` itself now prunes after backward (per `2349251`), but
embed/cos/sin caches accumulate per `Qwen35Model::new` call (one student
+ one teacher rebuilt every `run_once`). Likely fine for the moderate
baseline; but at Qwen3-0.6B this would OOM. **Not a perf bug — a future
test scaling consideration.** Lower priority than P3 and P4.

## Killed during this push

- `forward_last_logits` rollout opt — killed `0a1f945` per 7aa11d7
  wins-stub criterion. **Re-licensing on top of M=1 dispatch is the
  open P3 axis above.**

## Cooperative protocol notes

- **OOM under concurrent benches.** Dev box is 31 GB; codex's moderate
  baseline runs ~9.5 GiB. Don't run a parallel large-shape bench while
  codex is mid-run.
- **Work-split contract.** Claude = research / plan / docs / deterministic
  refactors. Codex = complex code + verification.
- **License-or-kill pattern (validated this session).** A pending-bench
  wins stub with explicit numerical kill criterion → peer measures →
  peer either updates `pending-bench` → `verified` or executes the kill.
  Today's cycle: 7aa11d7 stub → 0a1f945 kill → 01b3485 M-aware dispatch
  (the kill's root cause) → P3 re-license on top of the fix. **This is
  the canonical pattern for "code with unverified perf claim."**

## Codex resume pointer

Codex is currently working on **P4** (the phase-attribution profile
bench). Once it commits, the next move is **P3** (re-license
`forward_last_logits` on top of M=1 dispatch) — quick A/B that either
adds another ~90 ms saving or definitively rules out the rollout-last-row
axis for good. After that, axis selection depends on P4's attribution
numbers.

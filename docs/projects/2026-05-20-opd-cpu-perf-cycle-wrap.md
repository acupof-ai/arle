# 2026-05-20 — OPD CPU perf cycle: session wrap

> **Status:** active session, last updated 2026-05-20 14:50 local. Captures
> the day's cooperative-cycle (Claude research + codex implementation +
> verification) state for any future agent picking up the OPD CPU perf
> track. Replaces the rolling
> [`../plans/2026-05-20-opd-cpu-perf-codex-handoffs.md`](../plans/2026-05-20-opd-cpu-perf-codex-handoffs.md)
> as the single read-first entry point once this cycle closes.

## Headline

End-to-end OPD step at moderate shape (hidden=512, intermediate=1536,
layers=12, vocab=32 768, rollout_len=2):

- **Naive baseline (pre-cycle):** ~30 s / step (`8e8effd`)
- **Post-cycle:** **0.83 s / step** (post-`506f02b` AdamW host-zip-loop)
- **Cumulative speedup:** ~35× since naive substrate; ~4.2× since the
  2026-05-20 morning baseline `c4e507f`

The cycle also added comprehensive OPD-substrate hardening (input
validation, vocab compatibility checks, loader half-state avoidance) so
the perf wins are now defended by actionable error paths.

## Commit chronicle (16 commits this session)

### Perf cycle (8 commits)

| Commit | Axis | Wall-clock impact |
|---|---|---|
| `8e8effd` | Naive CPU matmul diagnosis | Surfaced 50-75× headroom |
| `499bfc0` | Row-major saxpy forward (codex) | Forward GF/s × ~50 |
| `f9f47a8` | Backward gap diagnosis (Claude) | Surfaced 19× backward gap |
| `6e37b91` | Transpose-aware backward (Claude) | 2.82× per-call |
| `15fa6cf` | Mixed-dispatch sgemm (Claude) | 16.7× cumulative substrate |
| `01b3485` | M=1 wide CPU matmul → saxpy (codex) | 2.05× wall-clock at M=1 |
| `0b593e1` | `matmul_bt` op + linear_forward (codex) | Linear projs 17-19×, lm_head 6.21× per call |
| `e0bfbb0` | LoRA matmul_bt extension (codex) | **3.06× end-to-end** (3.51 s → 1.17 s) |
| `506f02b` | AdamW host-zip-loop (codex) | **3.01× isolated, 1.40× end-to-end** (1.17 → 0.83 s) |

### Robustness cycle (7 commits)

| Commit | Hardening |
|---|---|
| `2349251` | OPD step retain_ids leak fix (from Claude research) |
| `1a1d4c9` | `OpdError::InvalidInput` for empty prompt / shape overflow + regression test |
| `c903e8b` | `kl_distill_loss` shape validation |
| `344fb15` | Out-of-vocab prompt rejection (early) |
| `c97f47f` | Teacher/student vocab mismatch rejection |
| `1522cc6` | `greedy_next_token` exact-shape requirement (no silent prefix sampling) |
| `2196fbd` | Invalid OPD student params rejection |
| `b65d9a7` | OPD loader half-state avoidance on missing shards |

### Diagnostic + revert (2 commits)

| Commit | Purpose |
|---|---|
| `5a92878` | OPD backward op attribution instrumentation (`backward_profiled`) |
| `3492ec3` | Revert `e53654a` merge-grad sharing per wall-clock kill |

### Killed axes (2)

| Axis | Where killed | Why |
|---|---|---|
| `forward_last_logits` rollout opt | `0a1f945` (codex) | Production-vocab A/B 0.997× ± 0.5%, below 1.05× threshold from the original wins stub |
| `merge_grad` sharing / clone elim | `3492ec3` (codex from Claude follow-up) | Larger-sample A/B flipped sign: +2.6% step regression despite -31% on isolated `merge_grad` metric |

Both kills validated the SOLID license-or-kill pattern.

## Phase attribution at session end

Post-`506f02b` (15 step samples, σ 0.5%):

| Phase | Seconds | % of step |
|---|---:|---:|
| `backward` | 3.83 s | **29.5 %** |
| `optimizer_step` | 3.02 s | 23.3 % |
| `rollout_student_forward` | 2.60 s | 20.1 % |
| `teacher_forward` | 1.20 s | 9.2 % |
| `student_forward` | 1.14 s | 8.8 % |
| `grad_clip` + minor | ~1.3 s | ~10 % |

Within `backward` (4.23 s over 5 steps):

| Sub-phase | % of `backward` | % of step |
|---|---:|---:|
| `MatmulBT` op kernel | 56.1 % | 16.3 % |
| `merge_grad` host accumulation | 39.1 % | 11.4 % |
| All other 16 ops combined | < 5 % | < 1.4 % |

## Open perf axes (Axis F killed; per-layer profile + bf16 remain)

~~**Axis F** — rayon N-shard for `MatmulBT` backward (lm_head bwd
specifically). **KILLED 2026-05-20** after codex implemented + benched
the snippet from
[`../plans/2026-05-20-axis-f-matmul-bt-backward-rayon-shard.md`](../plans/2026-05-20-axis-f-matmul-bt-backward-rayon-shard.md).
Microbench at lm_head [4×1024 @ 1024×151_936]: baseline 0.344 s vs
parallel 0.348 s = **+1.1 % regression**. Same regression pattern
across all 8 OPD backward shapes (+0-1.6 %). Root cause:
`cpu_matmul_backward` already runs at 7.2 GF/s on lm_head, which is
at single-channel DDR4 bandwidth ceiling — kernel is memory-bound,
parallelism cannot help. Captured as memory entry
`feedback_cpu_parallel_kernel_bandwidth_bound.md`. **The next axes for
this kernel surface must reduce bandwidth, not add threads.**~~
**Axis C — bf16 `lm_head` weight storage**. Halves 622 MB weight to
311 MB. With Axis F killed by memory ceiling, **this is now the
top-ranked perf axis** — bf16 directly attacks the binding constraint
(bandwidth) where Axis F's threads failed. Complex (new dtype path in
autograd, kernel variant, precision tolerance experiment).
Acceptance criterion (pre-licensed): step median ≥ 1.10×, max relerr
in `test_opd_grad_check` ≤ 1 % (i.e. ≥ 4× current).
3. **Per-layer profile**. The 12-layer transformer body could have
   non-uniform per-layer cost. A per-layer instrument inside
   `opd_step_cpu_moderate_profile` would surface it. Cheap to add —
   but only worth doing if #1 saturates the M=1+matmul_bt path.

The merge_grad-host axis is now considered exhausted: both clone
elimination and shared-first short-circuit failed at the step-level
wall-clock gate. Further work on this surface requires a
fundamentally different framing (e.g. fused matmul-bwd-accumulate that
writes directly into the parameter grad without an intermediate buffer).

## Cooperative-cycle lessons (memory entries written this session)

| Lesson | Memory file |
|---|---|
| Research-doc → impl in <30 min when prescriptions are SOLID | `feedback_cooperative_cycle_research_to_impl_under_30min.md` |
| Plans get same kill criterion as code (revaluate ROI when substrate moves) | `feedback_plans_get_same_kill_criterion_as_code.md` |
| Clone-elimination ROI ≈ half of naive projection (f32 vector wrapper) | `feedback_clone_elimination_smaller_than_projected.md` |
| 3-sample A/B too noisy for ≤10% effects | `feedback_3sample_too_noisy_for_10pct_effects.md` |
| License-or-kill needs explicit numerical threshold in wins stub | `feedback_license_or_kill_with_explicit_threshold.md` |
| SOLID framing: moderate-shape null is already ground truth | `feedback_solid_framing_moderate_shape_misread.md` |

## Cooperative-cycle protocol observations

- **Work split (validated this session):** Claude = research / plan /
  docs / deterministic refactors. Codex = complex code + verification
  (bench A/B, multi-process runs, gates). The split lets each agent
  operate at peak without stepping on the other's commits.
- **Bench scheduling must be serialised.** Dev box is 31 GB; both
  agents + a large bench can SIGKILL the smaller process (observed
  once at the start of the cycle). Always peek `tmux` + `free -h`
  before any large run.
- **`git rebase` refuses dirty working tree** even on unrelated
  files. When the peer has WIP, either wait or coordinate explicitly.
  Never `git stash` the peer's changes.
- **Codex's gates discipline.** Every commit goes through
  `cargo fmt + test + clippy + check + build`, run with `timeout` and
  `-j 1` to bound memory. Two `codex review` rounds per non-trivial
  diff. This is what makes a 16-commit session land cleanly without
  any rollback noise.
- **Plan docs need:** FLOP/bandwidth math, backward derivation,
  acceptance threshold, file:line cross-refs. With those, codex
  implements + verifies in <30 min. Without one or more, the cycle
  stalls.

## Artifacts (linked)

Wins entries (new this session):
- `docs/experience/wins/2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md`
- `docs/experience/wins/2026-05-20-cpu-matmul-m1-dispatch.md`
- `docs/experience/wins/2026-05-20-cpu-linear-matmul-bt.md`
- `docs/experience/wins/2026-05-20-opd-linearwithlora-matmul-bt.md`
- `docs/experience/wins/2026-05-20-opd-step-cpu-moderate-post-matmul-bt.md`
- `docs/experience/wins/2026-05-20-opd-step-cpu-moderate-profile.md`
- `docs/experience/wins/2026-05-20-adamw-host-zip-loop.md`

Errors entries (new this session):
- `docs/experience/errors/2026-05-20-forward-last-logits-killed-by-m1-dispatch-hypothesis.md`
- `docs/experience/errors/2026-05-20-opd-merge-grad-shared-first-revert.md`

Research notes (new this session):
- `docs/research/2026-05-20-opd-production-step-retain-ids-leak.md`
- `docs/research/2026-05-20-opd-backward-sub-phase-attribution.md`

Plan docs (new this session):
- `docs/plans/2026-05-20-matmul-bt-backward-derivation.md`
- `docs/plans/2026-05-20-opd-cpu-perf-codex-handoffs.md` (rolling hand-off index)

## Next session pickup pointer

Read `docs/plans/2026-05-20-opd-cpu-perf-codex-handoffs.md` for the
ranked open hand-offs. The current state is "perf cycle paused, all
attempted axes either landed or killed cleanly; codex completed a
robustness sprint on top." Future work either:

1. Picks up Axis F (rayon N-shard) — needs the M=1 dispatch and
   matmul_bt substrate that this cycle delivered, both in place
2. Pivots to per-layer profiling to find sub-axes inside the
   transformer body
3. Investigates the `kl_distill_loss` numerical normalization decision
   that codex's `c903e8b` flagged but didn't change

All four candidate axes' acceptance criteria are documented in the
hand-offs brief.

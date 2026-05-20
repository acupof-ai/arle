# 2026-05-20 — OPD CPU perf: hand-offs index (post-LoRA-matmul-bt update)

> **Audience:** codex (per the 2026-05-20 cooperative split — Claude does
> research / plan / docs / deterministic code; codex does complex code +
> verification). Single index of OPD CPU-perf state. **Read this first**
> before opening any of the linked docs.

> **Status as of 2026-05-20 11:30 local:** the cycle below closed every
> hand-off P0-P4 from the prior version of this brief. End-to-end
> moderate-shape OPD step is now **3.06× faster** (3.51 s → 1.17 s)
> after the LoRA matmul_bt extension landed (`e0bfbb0`). `optimizer_step`
> is now the dominant coarse phase (~46 % of step) — codex is mid-A/B on
> an AdamW host-zip-loop rewrite (3× microbench observed; end-to-end
> conversion pending). The original P3 (re-license `forward_last_logits`)
> is **superseded** — see §"P3 supersession" below for the SOLID math.

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
| `67a4d63` | Production-faithful phase attribution (codex) | rollout_student_forward 30.4 %, backward 21.6 %, optimizer_step 15.1 % |
| `e0bfbb0` | LoRA matmul_bt extension (codex) | **3.06× end-to-end** (3.51 s → 1.17 s/step); rollout_student_forward 6.37 ×, teacher 6.88 ×, student 7.26 ×, backward 3.07 × |
| `506f02b` | AdamW host-zip-loop rewrite (codex) | **3.01× isolated** (65 ms → 22 ms/step), **1.40× end-to-end** (1.17 → 0.83 s/step); optimizer_step share 45.5 % → 23.3 % |
| `5a92878` | OPD backward op attribution (codex) | Diagnostic only; no perf claim |
| `e53654a` | Share-first accumulated grad tensor (codex) | KILLED at step level (+2.6 % regression on larger-sample A/B despite -31 % merge_grad and -7 % backward isolated wins). Reverted in `3492ec3`. |
| `1a1d4c9` | OPD error path actionability (codex) | Robustness; not a perf claim. `OpdError::InvalidInput` for empty prompt / vocab=0 / shape overflow + regression test. |
| `3492ec3` | Revert merge-grad sharing (codex) | License-or-kill cycle close: wall-clock ground truth wins over targeted metric. Errors entry preserved. |

**Cumulative: ~35× over naive 8e8effd baseline** at moderate shape, ~4.2×
since the 2026-05-20 morning baseline (`c4e507f`). End-to-end OPD step:
30 s (naive) → 1.80 s (substrate) → 0.83 s (post-AdamW). 9 commits this
session, every perf claim license-or-kill validated. Two axes killed
(forward_last_logits, merge_grad sharing) confirming the SOLID gate works.

## Phase attribution rolling table

| Phase | Pre-LoRA-bt (`67a4d63`) | Post-LoRA-bt (`e0bfbb0`) | Post-AdamW (`506f02b`) | Post-AdamW share |
|---|---:|---:|---:|---:|
| `backward` | 11.56 s | 3.76 s | 3.76 s | **29.0 %** |
| `optimizer_step` | 8.09 s | 8.18 s | **3.02 s** | 23.3 % |
| `rollout_student_forward` | 16.29 s | 2.56 s | 2.56 s | 19.7 % |
| `teacher_forward` | 8.16 s | 1.19 s | 1.19 s | 9.2 % |
| `student_forward` | 8.15 s | 1.12 s | 1.12 s | 8.6 % |
| `grad_clip` + minor | ~1.3 s | ~1.2 s | ~1.3 s | ~10 % |

(15 measured steps; AdamW affects only `optimizer_step`; other phases
unchanged between `e0bfbb0` and `506f02b`. Total step time
$\approx 17.98 / 15 \rightarrow 12.97 / 15 = 1.20 \rightarrow 0.83$ s.)

**`backward` is now the dominant phase at 29 %.** It's already on the
transpose-aware (`6e37b91`) + matmul_bt (`0b593e1`) substrate, so the
remaining cost is concentrated in non-matmul backward kernels —
embedding scatter-add, rmsnorm bwd, rope bwd, sdpa bwd. **The next
hand-off is sub-phase profiling within `backward`** — see P4 below.

## What's still open

### P3 supersession — `forward_last_logits` re-license should NOT proceed

**Math after `01b3485` + `e0bfbb0`.** The original P3 was sized assuming
rollout_student_forward was 30 % of step. After `e0bfbb0`, rollout is
**14 %** of step (`2.56 s / 17.98 s × 100`). Re-license ROI in
production-vocab terms:

- Per rollout iter at Qwen3-0.6B (vocab=151_936, K=1024):
  - Full lm_head at M=3-4 (matrixmultiply): 0.075 s
  - Last-row at M=1 (saxpy, via `01b3485`): 0.036 s
  - Per-iter saving: 0.039 s (only when M ≥ 3 — at M=2 the M=1 saxpy
    path is slower than the M=2 matrixmultiply path)
- Per OPD step (rollout_len=2, prompt_len=3): seq=3 then seq=4. At
  seq=3 (M=3) the matrixmultiply path (~10-12 GF/s for M=3) may already
  be on par with M=1 saxpy (8.6 GF/s), so the per-call saving at seq=3
  is plausibly **negative** (M=3 mm wins). Saving only realises at the
  seq=4 iter: ~0.039 s.
- Per-step saving: ~0.039 s; per-step total post-LoRA: ~1.17 s. So **3 %
  of step**. Below the 1.05× kill criterion.

**Conclusion.** P3 was strongly justified pre-`e0bfbb0` when rollout was
30 % of step. After LoRA matmul_bt landed and rollout shrank to 14 %,
the same arithmetic that licenses P3 also kills it. **Do not re-license
this axis** without a different framing. The supersession is itself a
license-or-kill outcome on the *plan* — the kill criterion the plan
itself defined ($\geq 1.05\times$ step) is no longer reachable.

### P3' — AdamW host-zip-loop rewrite (codex in flight, NEW dominant phase)

**Why.** Per the post-`e0bfbb0` attribution, `optimizer_step` is now
**45.5 %** of step. Codex's tmux shows a 3× microbench on the AdamW
inner loop using a "host-zip-loop" approach (vs the current per-tensor
accessor pattern). If the 3× microbench converts to end-to-end:

- Optimizer drops from 0.546 s → 0.182 s per step
- Step total: 1.17 s → ~0.81 s (~30 % step saving)

Codex is currently producing `bench-output/2026-05-20-adamw-host-zip-loop-ab/opd_profile_after.txt`
to measure the end-to-end conversion.

**Acceptance criterion (codex-defined per the in-flight A/B):** Step
median speedup ≥ 1.20× at σ ≤ 5 %.

**Hand-off:** codex owns this entirely. Claude's role is post-result:
write the next-axis research after the AdamW result lands.

### P4 — Backward (~29 % of post-AdamW step) — SUB-PHASE DATA IN

Backward is the new dominant phase after `506f02b`. Codex's existing
`backward_op_summary` instrumentation already produced sub-phase data
(`bench-output/2026-05-20-opd-backward-op-profile/run.txt`); full
analysis in
[`../research/2026-05-20-opd-backward-sub-phase-attribution.md`](../research/2026-05-20-opd-backward-sub-phase-attribution.md).

| Sub-phase | % of `backward` | % of step |
|---|---:|---:|
| `MatmulBT` backward kernel | 56.1 % | 16.3 % |
| `merge_grad` host accumulation | 39.1 % | 11.4 % |
| All other 16 ops combined | < 5 % | < 1.4 % |

**Axis E (`merge_grad` shared-first / clone elimination) — KILLED
2026-05-20.** Codex tried two variants:

- *shared-first short-circuit* (committed as `e53654a`): initial 3-sample
  A/B claimed -8.92 % step; follow-up larger-sample A/B showed +2.6 %
  step regression. Wall-clock ground truth kills the axis. Revert
  pending.
- *`add_into_host` clone elimination* (my originally-proposed Axis E,
  prototyped in working tree): same regression pattern at step level
  even though `merge_grad` isolated improved -31 % and `backward`
  total -7 %. Per §0 SOLID framing-cross-check: wall-clock framing
  beats targeted-metric framing — the targeted win does not survive
  end-to-end integration. Reverted in working tree before commit.

The lesson is captured in
[`../experience/errors/2026-05-20-forward-last-logits-killed-by-m1-dispatch-hypothesis.md`](../experience/errors/2026-05-20-forward-last-logits-killed-by-m1-dispatch-hypothesis.md)
and reinforced by memory entries on 3-sample noise + clone-elimination
ROI projection error. **Do not re-license merge_grad host-side
optimisations without ≥ 5 samples per side AND a step-level cross-check.**

**Axis F — `MatmulBT` backward kernel parallelism** is now codex's
active scope (2026-05-20 EOD tmux: exploring `rayon`, `std::thread`,
`available_parallelism`). The largest single backward call is
`lm_head` bwd at `[N=151_936, K=hidden]`; N-axis sharding with explicit
per-thread `sgemm` is the natural shape. `matrixmultiply::threading`
already regressed at OPD M=4 shapes, so the manual sharding path is
the correct choice.

**Hand-off:** codex owns Axis F implementation + verification. Claude
writes the next-axis research after Axis F result lands (PASS or KILL).

### P5 — `rollout_student_forward` re-investigation (only after P4)

Currently 14 % of step. Cheaper than backward in absolute terms. After
P3' and P4 land the next dominant phase may shift again — defer until
then.

### P6 — Quench inter-step retain_ids leak in moderate-bench harness

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
  wins-stub criterion (forward A/B).
- **P3 re-license plan** — killed-by-math after `e0bfbb0` shrank
  rollout_student_forward to 14 % of step; the ~3 % step saving
  projection falls below the original 1.05× kill criterion. The plan
  itself was license-or-kill'd on its own arithmetic; same SOLID rule
  applies to plans, not just code.

## Cooperative protocol notes

- **OOM under concurrent benches.** Dev box is 31 GB; codex's moderate
  baseline runs ~9.5 GiB. Don't run a parallel large-shape bench while
  codex is mid-run.
- **Work-split contract.** Claude = research / plan / docs / deterministic
  refactors. Codex = complex code + verification.
- **License-or-kill pattern (validated this session, twice).**
  Cycle 1: 7aa11d7 stub → 0a1f945 kill → 01b3485 M-aware dispatch
  (root cause of the kill) → P3 *plan* re-license. Cycle 2: P3 plan
  re-license → killed-by-math when LoRA matmul_bt landed and rollout
  share dropped below the threshold. **Plans get the same kill criterion
  as code.**

## Codex resume pointer

Codex is currently mid-A/B on **P3'** (AdamW host-zip-loop —
`bench-output/2026-05-20-adamw-host-zip-loop-ab/`). When that lands,
the next move is **P4** (backward sub-phase profiling). After that, axis
selection depends on what's then-dominant — likely backward sub-ops
or attention/MLP if backward shrinks.

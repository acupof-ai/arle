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

**Cumulative: ~25× over naive 8e8effd baseline** at moderate shape. Per-call
linear projection ops are ~17-19 × faster; lm_head per call is ~6 × faster;
M=1 wide matmul is ~2 × faster. End-to-end matters more: 30 s → 1.17 s.

## Post-LoRA-matmul-bt phase attribution

After `e0bfbb0`, `optimizer_step` becomes the dominant phase because every
other coarse phase shrank around it:

| Phase | Before LoRA-bt | After LoRA-bt | Share after |
|---|---:|---:|---:|
| `optimizer_step` | 8.09 s | 8.18 s | **45.5 %** |
| `backward` | 11.56 s | 3.76 s | 20.9 % |
| `rollout_student_forward` | 16.29 s | 2.56 s | 14.2 % |
| `teacher_forward` | 8.16 s | 1.19 s | 6.6 % |
| `student_forward` | 8.15 s | 1.12 s | 6.2 % |
| `grad_clip` + minor | ~1.3 s | ~1.2 s | 6.6 % |

(15 measured steps total; totals match codex's `e0bfbb0` profile table.)

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

### P4 — Backward (~21 % of post-LoRA step)

`backward` is now the second-largest phase (3.76 s for 15 steps = 0.251 s
per step = 21.4 %). Already on the transpose-aware (`6e37b91`) and
matmul_bt-backward (`0b593e1`) substrate. Likely sub-axes:

- **Per-op breakdown.** Profile within `backward` to see whether matmul
  backward, rmsnorm backward, rope backward, or sdpa backward dominates.
  This is a profiling task, not a perf change — codex can extend
  `opd_step_cpu_moderate_profile` to record sub-phases when backward
  is the next licensed axis.
- **Backward kernel reuse.** Some autograd `BackwardOp` variants may
  still re-compute saved activations rather than reuse forward-saved
  tensors. Worth surveying after sub-phase profiling.

**Hand-off:** deferred until AdamW (P3') lands. Claude writes the
backward sub-phase research after seeing AdamW's end-to-end.

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

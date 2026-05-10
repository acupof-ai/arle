---
title: Codex pickup queue — 2026-05-10 (post-cooperative-loop tail)
date: 2026-05-10
type: plan
status: open (supersedes 2026-05-09 queue; pickup-ready as Task #35 completes)
---

# Codex pickup queue — 2026-05-10

> **Purpose**: single-source-of-truth queue for codex's next task pickup
> when Task #35 cap=8 prefill warmup commits. Supersedes
> `codex-pickup-queue-2026-05-09.md` which is 1+ day stale.
>
> Each entry has: trigger condition, scope estimate, scaffold artifact,
> cross-references. Codex picks based on the prevailing trigger
> condition (which often depends on user actions like running bench v11).

## §1 Pickup matrix (priority order)

| Priority | Task | Trigger | Scope | Scaffold artifact |
|---|---|---|---|---|
| **P0** | #43 W4A16 fragmentation hypothesis test | always available, no user gate | ~30 min single bash | `scripts/task43_hypothesis_test.sh` (commit 458394c) |
| **P1 (LICENSE branch)** | #47 PF8.3 H1' static-scratch refactor | bench v11 LICENSES PF8 at conc=1 (USER must run `bash scripts/pf85_bench_v11_user.sh` per ead46dc) | ~70 LOC + tests + bench, 3-4 hours | `docs/plans/M_pf83_h1prime_static_scratch.md` (05e2135) + REVISION `docs/research/2026-05-10-h1prime-design-revision-marlinscratch-already-exists.md` (2cc608a) |
| **P1 (KILL branch)** | #28 Medusa Phase 1.A via Alpaca | bench v11 KILLS PF8 at conc=1 | ~80 LOC Python + 2 hrs data prep + ~1 week training | `docs/research/2026-05-10-medusa-pivot-ready-alpaca-unblocks-hf-auth.md` (63769be) |
| **P2** | #48 W4A8-vs-BF16 84.4% accuracy regression bisect | always (NOT blocking #35) | ~1 hr bisect + fix scoping | `docs/research/2026-05-10-w4a8-vs-bf16-accuracy-regression-observed.md` (e3e1ab5) |
| **P3** | #30 Hybrid W4A16/W4A8 dispatch Phase 1-3 substrate | always (older pending task) | unknown (likely substantial) | (no recent scaffold; defer to discovery) |
| **P4** | #44 PF8 chain (PF8.5 license sequence completion) | bench v11 LICENSES + Task #47 lands first | depends on H1' outcome | `docs/plans/M_pf83_h1prime_static_scratch.md` §10 estimates |

## §2 Recommended dispatch logic

```
codex_finishes_task_35():
    # Independent of bench v11 outcome, P0 is cheap + valuable
    if not has_run_task_43_hypothesis():
        run_task_43_hypothesis_test()  # ~30 min
        # Result determines whether to fix Task #43 in same PR as #47

    # Branch on bench v11 result (user-runnable)
    if user_has_run_bench_v11():
        if pf8_licensed:
            pickup_task_47_h1_refactor()  # 3-4 hours
        else:  # pf8_killed
            pickup_task_28_medusa_via_alpaca()  # 2 hrs setup + 1 wk train
    else:
        # User hasn't run bench v11 yet
        if want_to_make_progress_blind:
            pickup_task_48_w4a8_bisect()  # ~1 hr, no user gate
        else:
            await user_decision()
```

## §3 Per-task pickup briefs

### §3.1 P0: Task #43 W4A16 fragmentation hypothesis test (~30 min)

**Cheapest, fully scaffolded, codex-pickup-ready.**

Run:
```bash
bash scripts/task43_hypothesis_test.sh
```

Outputs verdict (HEALTHY / SUBSTRATE-KILL / TOOL-QUIRK / NO-OUTPUT) +
hypothesis result (CONFIRMED / DISPROVEN / AMBIGUOUS).

If CONFIRMED: Task #43 root cause = env-gated scratch fallback per
1ba06f0 §3 (W4A16 dispatch falls back to per-call alloc when
`marlin_scratch=None` at `linear.rs:2064-2095`, gated on
`INFER_PREFILL_GRAPH=1` per `qwen3/forward.rs:312-313`).

If DISPROVEN: pivot to other Task #43 investigation paths (H8 was
DISPROVEN for PF8.3 too — different cause).

**Output**: wins entry under `docs/experience/wins/` confirming or
killing the hypothesis. If confirmed, link to fix PR direction.

### §3.2 P1 LICENSE: Task #47 PF8.3 H1' static-scratch refactor (3-4 hrs)

**Trigger**: bench v11 reports LICENSE (TTFT mdn ≤ 49.3ms) at conc=1
per `bash scripts/pf85_bench_v11_user.sh` (ead46dc).

**Plan**: read REVISION `2cc608a` FIRST, then original `05e2135`.

Original plan (110 LOC): create new `PF8Scratch` struct from scratch.
**REVISION (~70 LOC)**: extend existing `MarlinScratch` (linear.rs:317-323)
+ add `run_marlin_w4_fp8_prefill_with_scratch` mirror of
`run_marlin_w4a8_linear_with_scratch` (linear.rs:1484).

10-step pickup checklist in `M_pf83_h1prime_static_scratch.md` §9.

**Bonus**: same PR can fix Task #43 (per §3.1 if hypothesis confirmed)
by routing non-`_with_scratch` W4A16 callers through scratch variant
= **two-tasks-one-PR** opportunity (per 2cc608a §2.2).

**Verification gates** (per SKILL v1.12.0 #33 + #34):
1. cargo build --release + clippy -D warnings
2. greedy_consistency at conc=1 (PASS gate)
3. Sustained-load bench at conc=1+2+4 (PASS gate per #34)
4. `codex review --uncommitted` (mandatory per #33 — caught 3 bugs in
   PF8.3 substrate `ace3cbe`, 3 bugs in Task #35)
5. Wins entry with both regression-guard + acceptance-target shapes
   per SKILL candidate #37 multi-shape discipline

### §3.3 P1 KILL: Task #28 Medusa via Alpaca (~2 hrs setup + 1 wk train)

**Trigger**: bench v11 reports KILL (TTFT mdn > 55.2ms regression) at
conc=1.

**Plan**: per `63769be` cross-link, use `tatsu-lab/alpaca` (52k samples,
public, no HF auth) — sufficient for first-pass Medusa-1 per
Cai et al. 2024 §4.1.

§4 codex pickup recipe:
1. Verify `crates/train/src/hub_dataset.rs` exists
2. Write `scripts/medusa_training_data.py` (~50-80 LOC Python loader)
3. Wire to existing trainer-side adapter
4. Smoke-test: load 100 samples, verify safetensors output shape
5. Full data prep: 52k samples → ~10M tokens
6. Hand off to Medusa-1 training scaffold per `afdddec` plan

Bypasses 2026-05-10 ad14636 HF auth blocker on `lmsys-chat-1m` (gated).

### §3.4 P2: Task #48 W4A8 84.4% accuracy regression bisect (~1 hr)

**Always available** (independent of bench v11). NOT blocking Task #35
(codex correctly used targeted test in #35 verification per e3e1ab5).

**Steps**:
1. Re-run `test_w4a8_vs_bf16_token_diff` to confirm 84.4% diff
2. Bisect entry candidates (per e3e1ab5 §3):
   - `35fc3cf` #24 W4A8 prefill graph capture hoist
   - `c44788f` #40 Path B.2 bucketing fix
   - `09ae5a5` Path B Phase 1 marlin_dequant.cuh 651 LOC
3. Reconcile docstring (says <1% rule per skill v1.3.0) vs actual
   25% lenient assert at greedy_consistency.rs:365
4. Either fix root cause OR tighten threshold + add canary
5. SKILL candidate #35 (per e3e1ab5): "Tasks closed `root cause TBD`
   need a regression test or canary assertion; otherwise workaround
   quietly decays past lenient gates."

## §4 What's deferred / not on pickup queue

- #44 PF8 chain (depends on Task #47 outcome)
- #46 PF8.3 H8 (CLOSED, DISPROVEN per 57c37b5)
- #39 M_rope-yarn-scaling (in_progress, not codex-blocked)
- Bench v11 license run (USER ONLY per Claude session sleep limits)

## §5 SKILL candidates to watch for n+1 evidence

When picking up tasks, watch for these patterns to graduate from
single-evidence candidate to canonical SKILL anti-pattern:

- **#35** (root-cause-TBD canary, e3e1ab5): if Task #48 fix workflow
  surfaces additional cases of "old workaround decayed past lenient
  gate" → graduate
- **#36** (grep variants before designing, 2cc608a): when Task #47
  H1' lands, check if codex naturally greps for variants before
  creating new struct → likely n=2 just from this implementation
- **#37** (multi-shape bench discipline, 2d00de3): when Task #47
  H1' wins entry has both regression-guard AND acceptance-target
  shapes → n=2
- **#38** (warmup shape ≠ effective workload budget, b4a3c38): if
  Task #47 H1' PF8Scratch sizing needs similar clamp guard → n=2
- **#39** (post-fix bench data stale, 3beab7f): if any task pickup
  involves prior bench data being invalidated → n=2

If any of these reach n=2 in the next codex session, sediment into
SKILL.md v1.13.0+.

## §6 Cross-references

- `docs/index.md` (canonical truth surface, will need pickup-queue
  link update)
- `docs/research/2026-05-10-next-session-pickup-state.md` (pickup
  state with full session-end context)
- `2cc608a` H1' design REVISION (Task #47 scope reduction)
- `1ba06f0` Task #43 dispatch audit (hypothesis evidence)
- `63769be` Medusa Alpaca cross-link (Task #28 unblock)
- `458394c` Task #43 hypothesis test script
- `ead46dc` PF8.5 bench v11 user script
- `868e147` `pf83_bench_health.sh` (verdict tool used by §3.1 + §3.2)
- `0be7220` SKILL kernel-optimization v1.12.0
- All commits since `40a9184` (EOD+580 baseline) listed in
  `docs/research/2026-05-10-next-session-pickup-state.md` §3
  POST-COOPERATIVE-LOOP block (commit 20adfb3 + addendum)

## §7 Status

**Pickup-queue-ready** for codex's next dispatch decision. P0 (Task
#43 hypothesis test) is cheapest + valuable + always available.
P1 branches on bench v11 outcome (user gate). P2 always available.

Replaces stale `codex-pickup-queue-2026-05-09.md`. Update
`docs/index.md` reference next tick if needed.

## §8 Dispatch log

- **2026-05-10 ~10:08 KST**: P0 Task #43 hypothesis test DISPATCHED to
  codex via tmux nudge after Task #35 landing (a2ad788) + 25min codex
  idle gap. Codex Working as of ~10:09. Expected ~30 min wall-clock
  to verdict.
- **2026-05-10 ~10:13 KST INTERIM**: Arm A (`INFER_PREFILL_GRAPH=1`)
  completed — NO substrate KILL (server log clean, no kernel failure
  lines). BUT guidellm reports 0 output tokens / TTFT 0 →
  `pf83_bench_health.sh` classifies as **TOOL-QUIRK** (exit code 2).
  Codex investigating /v1/stats + server log + considering direct
  curl bench fallback. Validates SKILL #34b discrimination (bench
  tool quirk vs substrate KILL distinguishable). Arm B starting now;
  if Arm B also TOOL-QUIRK → AMBIGUOUS verdict per verdict-implications
  doc (cb86836) §4.
- **2026-05-10 ~10:16 KST**: codex applied max-seq fix to script
  (NUM_SLOTS=8 + MAX_SEQ_LEN=5120 — root cause of Arm A's TOOL-QUIRK
  was insufficient max-seq for 4k bench). Arm A RE-RAN successfully
  with real request metrics this time. But TWO new issues surfaced:
  1. `pf83_bench_health.sh` JSON parser bug — reads success=0 even
     when guidellm produced real request metrics. Likely results.json
     schema mismatch (parser expects `requests_successful_total` but
     guidellm 0.6.0 may use different key). Script refinement needed.
  2. Server log has **36 failure-pattern matches** — codex notes
     these may be REAL kernel faults OR benign Pass 3 warmup
     OOM/backoff patterns (per SKILL #38 graceful fallback semantics).
     The grep pattern `failed with code|gemm.*failed|prefill batch
     failed|cudaError` is **too broad** — matches both KILL signals
     and benign-fallback signals. Script discrimination needs
     refinement (e.g. exclude `Marlin scratch OOM, falling back to
     1024 tokens/row` patterns).
  Codex waiting for Arm B + manual raw log inspection before
  finalizing verdict.

  **SKILL implication**: candidate v1.13.0+ #40 (single evidence
  point so far): "bench-health discriminator must distinguish KILL
  signals from graceful-fallback signals — same log pattern can mean
  both. Refinement: enumerate fallback patterns + exclude from
  failure count. Companion to #38 (warmup graceful fallback) — #38
  introduces the fallback patterns; #40 ensures discrimination tools
  don't conflate them with real failures."
- **2026-05-10 ~10:18 KST**: Both arms COMPLETED. GPU idle (1293 MiB /
  0%) confirms. Codex bypassing buggy `pf83_bench_health.sh` JSON
  parser — running ad-hoc Python script to iterate over both arms'
  raw `results.json`. Visible percentile output: `'p90': 415.91,
  'p95': 515.74, 'p99': 872.90, 'p999': 2419.91` (likely TTFT or
  request-total-latency in ms). Verdict imminent.

  **SKILL operational pattern observed**: when wrapper tool has a
  parsing bug (#34b's discriminator), fall back to raw inspection
  via lightweight script. Codex's discipline matches #34b's
  underlying principle — "trust raw data, scripts are conveniences
  that may fail". Reinforces #34b without adding new candidate.
- **2026-05-10 ~10:21 KST**: Task #43 codex DISPROVEN commit `83fc5d0`
  + Claude INVERSE analysis `e8b6b31` + SKILL v1.14.0 graduation
  `d2c987f` (#36 grep-+-A/B both required). Codex idle ~2 min after
  Task #43 wrap. Per cb86836 §6 dispatch logic: Task #48 W4A8
  regression bisect = safest non-bench-v11-gated pickup.
- **2026-05-10 ~10:23 KST**: P2 Task #48 W4A8-vs-BF16 84.4%
  regression bisect DISPATCHED to codex via tmux nudge. Codex
  Working as of ~10:23. Expected ~1 hour for: re-confirm 84.4% diff
  + bisect 3 candidate commits (35fc3cf #24 / c44788f #40 /
  09ae5a5 Path B Phase 1) + reconcile docstring (<1% rule per skill
  v1.3.0) vs assert (25% lenient gate). Then either fix root cause
  + tighten threshold OR add canary that breaks if >25% (per skill
  candidate #35 root-cause-TBD canary watch).
- **2026-05-10 ~10:25 KST**: codex Task #48 started step 1 (re-run
  targeted test_w4a8_vs_bf16_token_diff, loading 2 models for
  BF16-vs-W4A8 comparison). Mid-run mechanical state. First action
  matches brief verbatim — cooperative dispatch pattern healthy.
- **2026-05-10 ~10:28 KST**: Task #48 step 1 COMPLETED — test
  FAILED (`error: test failed, to rerun pass...`). Regression
  confirmed exists; exact diff number pending codex's analysis of
  346-line test output (ctrl+t transcript). Codex now likely moving
  to step 2 (bisect 35fc3cf / c44788f / 09ae5a5 candidates).
  GPU idle — test was brief 32-token greedy decode comparison.
